#!/usr/bin/env python3
"""
Видеоэкстензометр для разрывной машины (Shimadzu AG-X plus и др.).

Камера не синхронизирована с машиной: видео пишется отдельно, обрабатывается
после испытания, а затем (по желанию) совмещается с данными машины по времени.

Команды:
  record   запись видео с камеры + точные метки времени каждого кадра
  analyze  трекинг двух меток на образце -> L(t), ΔL(t), ширина шейки,
           автоматический поиск разрыва и L2 (последний кадр перед разрывом)
  merge    совмещение результата с CSV машины (сила–время) -> кривая F–ΔL

Примеры:
  python extensometer.py record --camera 0 --out test1.avi
  python extensometer.py analyze test1.avi --l0-mm 50 --neck
  python extensometer.py merge test1_results/test1_ext.csv machine.csv \
         --time-col 0 --force-col 1
"""
import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

MAX_VIEW = (1600, 900)          # максимальный размер окна предпросмотра
FONT = cv2.FONT_HERSHEY_SIMPLEX


# --------------------------------------------------------------------- утилиты

def fit_scale(img, max_wh=MAX_VIEW):
    h, w = img.shape[:2]
    return min(1.0, max_wh[0] / w, max_wh[1] / h)


def scaled(img):
    s = fit_scale(img)
    return (cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img.copy()), s


def to_gray(frame):
    return frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def select_roi(img, title):
    view, s = scaled(img)
    x, y, w, h = cv2.selectROI(title, view, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(title)
    if w == 0 or h == 0:
        sys.exit("Выбор отменён.")
    return tuple(int(round(v / s)) for v in (x, y, w, h))


def pick_points(img, n, title):
    """Клик мышью по n точкам. Backspace — отменить точку, Enter — готово."""
    view, s = scaled(img)
    pts = []

    def on_mouse(ev, x, y, flags, param):
        if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < n:
            pts.append((x / s, y / s))

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, on_mouse)
    while True:
        disp = view.copy()
        for i, p in enumerate(pts):
            q = (int(p[0] * s), int(p[1] * s))
            cv2.drawMarker(disp, q, (0, 0, 255), cv2.MARKER_CROSS, 25, 2)
            cv2.putText(disp, str(i + 1), (q[0] + 12, q[1] - 12), FONT, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        hint = f"points: {len(pts)}/{n}" + ("   ENTER = ok" if len(pts) == n else "") + "   BACKSPACE = undo"
        cv2.putText(disp, hint, (10, 25), FONT, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(disp, hint, (10, 25), FONT, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(title, disp)
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10, 32) and len(pts) == n:
            break
        if k == 8 and pts:
            pts.pop()
        if k == 27:
            sys.exit("Отменено (Esc).")
        if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
            sys.exit("Окно закрыто — отменено.")
    cv2.destroyWindow(title)
    return np.array(pts)


def roi_from_click(gray, pt, dark=True):
    """По клику на метку находит её контрастное пятно и возвращает рамку с запасом (x, y, w, h)."""
    H, W = gray.shape
    r = max(30, int(0.04 * min(H, W)))
    x0, y0 = max(0, int(pt[0]) - r), max(0, int(pt[1]) - r)
    win = cv2.GaussianBlur(gray[y0:int(pt[1]) + r + 1, x0:int(pt[0]) + r + 1], (3, 3), 0)
    mode = (cv2.THRESH_BINARY_INV if dark else cv2.THRESH_BINARY) + cv2.THRESH_OTSU
    _, mask = cv2.threshold(win, 0, 255, mode)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    local = np.array([pt[0] - x0, pt[1] - y0])
    best, best_d = None, np.inf
    for i in range(1, n):
        bx, by, bw, bh, area = stats[i]
        if area < 4 or bx == 0 or by == 0 or bx + bw >= win.shape[1] or by + bh >= win.shape[0]:
            continue
        d = np.hypot(*(cents[i] - local))
        if d < best_d:
            best, best_d = i, d
    if best is None or best_d > r * 0.6:
        print(f"  Возле точки ({pt[0]:.0f}, {pt[1]:.0f}) не найдено контрастного пятна — беру рамку 40x40.")
        return int(pt[0]) - 20, int(pt[1]) - 20, 40, 40
    bx, by, bw, bh = (int(v) for v in stats[best][:4])
    pad = max(6, int(0.5 * max(bw, bh)))
    return x0 + bx - pad, y0 + by - pad, bw + 2 * pad, bh + 2 * pad


def clamp_roi(roi, shape, min_size=6):
    """Обрезает рамку метки по границам кадра."""
    H, W = shape[:2]
    x, y, w, h = (int(v) for v in roi)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 - x0 < min_size or y1 - y0 < min_size:
        raise ValueError("Метка слишком близко к краю кадра — выберите точку подальше от края.")
    return x0, y0, x1 - x0, y1 - y0


def parse_roi(text):
    v = [int(float(t)) for t in text.split(",")]
    if len(v) != 4:
        raise argparse.ArgumentTypeError("ROI задаётся как x,y,w,h")
    return tuple(v)


def parabola_peak(l, c, r):
    """Субпиксельное смещение вершины параболы по трём точкам."""
    d = l - 2 * c + r
    return 0.0 if d == 0 else 0.5 * (l - r) / d


def timestamps_path(video):
    return Path(video).with_suffix(".ts.csv")


def read_timestamps(video):
    p = timestamps_path(video)
    if not p.exists():
        return None
    ts = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or line.startswith("frame"):
                continue
            parts = line.strip().split(",")
            if len(parts) >= 2:
                ts.append(float(parts[1]))
    return np.array(ts)


def draw_overlay(frame, p1, p2, lines, ok=True):
    img = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    fs = max(0.5, img.shape[1] / 1600)
    color = (0, 255, 0) if ok else (0, 0, 255)
    a = tuple(int(round(v)) for v in p1)
    b = tuple(int(round(v)) for v in p2)
    cv2.line(img, a, b, color, 1, cv2.LINE_AA)
    for p in (a, b):
        cv2.drawMarker(img, p, color, cv2.MARKER_CROSS, int(30 * fs), max(1, int(2 * fs)))
    for i, t in enumerate(lines):
        org = (int(15 * fs), int((35 + 32 * i) * fs))
        cv2.putText(img, t, org, FONT, 0.8 * fs, (0, 0, 0), int(4 * fs) + 1, cv2.LINE_AA)
        cv2.putText(img, t, org, FONT, 0.8 * fs, (255, 255, 255), max(1, int(2 * fs)), cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------- record

def cmd_record(a):
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(a.camera, backend)
    if not cap.isOpened():
        sys.exit(f"Не удалось открыть камеру {a.camera}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    cap.set(cv2.CAP_PROP_FPS, a.fps)
    if a.focus is not None:          # фиксированный фокус: автофокус «плавает» и портит масштаб
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_FOCUS, a.focus)
    if a.exposure is not None:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_EXPOSURE, a.exposure)

    ok, frame = cap.read()
    if not ok:
        sys.exit("Камера не отдаёт кадры.")
    h, w = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS) or a.fps
    out = Path(a.out)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*a.codec), fps, (w, h))
    if not writer.isOpened():
        sys.exit(f"Не удалось создать файл {out}")
    print(f"Камера: {w}x{h} @ {fps:.1f} fps. Запись в {out}. Остановка: Q или Esc.")

    tsf = open(timestamps_path(out), "w", encoding="utf-8", newline="")
    tsf.write(f"# start_wall={datetime.now().isoformat(timespec='milliseconds')}\n")
    tw = csv.writer(tsf)
    tw.writerow(["frame", "t_s"])
    t0 = time.perf_counter()
    i = 0
    try:
        while True:
            t = time.perf_counter() - t0
            writer.write(frame)
            tw.writerow([i, f"{t:.6f}"])
            view, _ = scaled(frame)
            cv2.putText(view, f"REC  {t:7.2f} s  frame {i}", (15, 30), FONT, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow("record", view)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
            ok, frame = cap.read()
            if not ok:
                print("Камера перестала отдавать кадры.")
                break
            i += 1
    finally:
        tsf.close()
        writer.release()
        cap.release()
        cv2.destroyAllWindows()
    real_fps = i / max(time.perf_counter() - t0, 1e-9)
    print(f"Записано {i + 1} кадров, фактически {real_fps:.1f} fps. Метки времени: {timestamps_path(out)}")


# --------------------------------------------------------------------- tracking

class Marker:
    """Трекер одной метки: шаблонное сопоставление или центроид пятна, субпиксельно."""

    def __init__(self, gray, roi, method, search, dark, min_score, update_thr, log=print):
        x, y, w, h = clamp_roi(roi, gray.shape)
        self.method, self.search, self.dark = method, search, dark
        self.min_score, self.update_thr = min_score, update_thr
        self.size = (w, h)
        self.pos = np.array([x + (w - 1) / 2, y + (h - 1) / 2], float)
        self.vel = np.zeros(2)
        self.tpl = gray[y:y + h, x:x + w].copy()
        self.area0 = None
        self.score = 1.0
        if method == "blob":
            p = self._blob(gray, self.pos, init=True)
            if p is None:
                log("  Внимание: в области метки не найдено отдельного контрастного пятна — "
                      "для неё используется метод template.")
                self.method = "template"
            else:
                self.pos = p

    def _window(self, gray, center, half):
        H, W = gray.shape
        x0, y0 = max(0, int(center[0] - half[0])), max(0, int(center[1] - half[1]))
        x1, y1 = min(W, int(center[0] + half[0]) + 2), min(H, int(center[1] + half[1]) + 2)
        return gray[y0:y1, x0:x1], x0, y0

    def _template(self, gray, pred):
        th, tw = self.tpl.shape
        win, x0, y0 = self._window(gray, pred, ((tw - 1) / 2 + self.search, (th - 1) / 2 + self.search))
        if win.shape[0] < th or win.shape[1] < tw:
            return None, 0.0
        res = cv2.matchTemplate(win, self.tpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, (mx, my) = cv2.minMaxLoc(res)
        dx = parabola_peak(res[my, mx - 1], res[my, mx], res[my, mx + 1]) if 0 < mx < res.shape[1] - 1 else 0.0
        dy = parabola_peak(res[my - 1, mx], res[my, mx], res[my + 1, mx]) if 0 < my < res.shape[0] - 1 else 0.0
        return np.array([x0 + mx + dx + (tw - 1) / 2, y0 + my + dy + (th - 1) / 2]), score

    def _blob(self, gray, pred, init=False):
        w, h = self.size
        extra = 0 if init else self.search
        win, x0, y0 = self._window(gray, pred, (w / 2 + extra, h / 2 + extra))
        if win.shape[0] < 3 or win.shape[1] < 3:          # метка ушла за край кадра
            return None
        blur = cv2.GaussianBlur(win, (3, 3), 0)
        mode = (cv2.THRESH_BINARY_INV if self.dark else cv2.THRESH_BINARY) + cv2.THRESH_OTSU
        T, mask = cv2.threshold(blur, 0, 255, mode)
        n, lab, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        local = pred - (x0, y0)
        best, best_d = None, np.inf
        for i in range(1, n):
            bx, by, bw, bh, area = stats[i]
            if area < 4:
                continue
            if bx == 0 or by == 0 or bx + bw >= win.shape[1] or by + bh >= win.shape[0]:
                continue                         # касается края окна — это фон/кромка, не метка
            if self.area0 and not (0.25 * self.area0 <= area <= 20 * self.area0):
                continue
            d = np.hypot(*(cents[i] - local))
            if d < best_d:
                best, best_d = i, d
        if best is None:
            return None
        ys, xs = np.nonzero(lab == best)
        vals = blur[ys, xs].astype(float)
        wgt = np.clip((T - vals) if self.dark else (vals - T), 0, None) + 1.0   # центроид, взвешенный по контрасту
        if init:
            self.area0 = stats[best, cv2.CC_STAT_AREA]
        return np.array([(xs * wgt).sum() / wgt.sum() + x0, (ys * wgt).sum() / wgt.sum() + y0])

    def update(self, gray):
        pred = self.pos + self.vel
        if self.method == "blob":
            pos = self._blob(gray, pred)
            score = 1.0 if pos is not None else 0.0
        else:
            pos, score = self._template(gray, pred)
            if pos is not None and score < self.min_score:
                pos = None
        self.score = score
        if pos is None:
            self.vel[:] = 0
            return None
        self.vel = pos - self.pos
        self.pos = pos
        if self.method == "template" and score < self.update_thr:
            # метка деформируется вместе с образцом — обновляем шаблон (субпиксельно, без дрейфа на округлении)
            self.tpl = cv2.getRectSubPix(gray, self.size, (float(pos[0]), float(pos[1])))
        return pos


class Neck:
    """Ширина образца между метками (поиск шейки / «мостика» и момента разрыва)."""

    def __init__(self, gray, p1, p2, halfw, margin_px):
        self.halfw, self.margin = int(halfw), margin_px
        strip = self._strip(gray, p1, p2)
        if strip.shape[0] < 3:
            raise ValueError("метки слишком близко друг к другу для анализа шейки")
        self.T, _ = cv2.threshold(cv2.GaussianBlur(strip, (3, 3), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.bright = (strip[:, self.halfw] > self.T).mean() >= 0.5     # образец светлее фона?
        self.w0 = self.measure(gray, p1, p2)[0]

    def _strip(self, gray, p1, p2):
        """Развёртка полосы вдоль оси p1->p2: строки — вдоль оси, столбцы — поперёк."""
        v = p2 - p1
        L = np.hypot(*v)
        u = v / L
        n = np.array([-u[1], u[0]])
        s = np.arange(self.margin, L - self.margin)
        t = np.arange(-self.halfw, self.halfw + 1)
        S, Tt = np.meshgrid(s, t, indexing="ij")
        mx = (p1[0] + S * u[0] + Tt * n[0]).astype(np.float32)
        my = (p1[1] + S * u[1] + Tt * n[1]).astype(np.float32)
        if mx.size == 0:
            return np.zeros((0, 2 * self.halfw + 1), np.uint8)
        return cv2.remap(gray, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def measure(self, gray, p1, p2):
        """-> (минимальная ширина, px; положение шейки 0..1 от метки 1 к метке 2)."""
        strip = self._strip(gray, p1, p2)
        if strip.shape[0] < 3:
            return np.nan, np.nan
        L = np.hypot(*(p2 - p1))
        c = self.halfw
        # щель после разрыва бывает 1-2 px — ищем её по несглаженным пикселям: строка, где у оси нет материала
        raw = strip > self.T if self.bright else strip <= self.T
        gap = np.nonzero(~raw[:, c - 2:c + 3].any(1))[0]
        if len(gap):
            return 0.0, float((self.margin + gap[len(gap) // 2]) / L)
        strip = cv2.GaussianBlur(strip, (3, 3), 0)
        fg = strip > self.T if self.bright else strip <= self.T
        left, right = fg[:, c::-1], fg[:, c:]
        lw = np.where(left.all(1), left.shape[1], np.argmin(left, 1))
        rw = np.where(right.all(1), right.shape[1], np.argmin(right, 1))
        width = np.maximum(lw + rw - 1, 0).astype(float)
        if len(width) >= 3:                          # медиана по 3 строкам гасит шум отдельных пикселей
            width = np.median(np.stack([width[:-2], width[1:-1], width[2:]]), axis=0)
        i = int(np.argmin(width))
        return float(width[i]), float((self.margin + i + 1) / L)


# ---------------------------------------------------------------------- analyze

def detect_start(L, n0, k=5.0, min_px=0.5, run=3):
    base = L[:n0]
    L0 = np.nanmedian(base)
    noise = np.nanstd(base) if np.isfinite(base).sum() > 2 else 0.0
    thr = max(k * noise, min_px)
    above = np.abs(L - L0) > thr
    for i in range(len(L) - run):
        if above[i:i + run].all():
            return i
    return None


def detect_break(L, ok, neck, jump_k, jump_min_px, neck_break_px, start):
    """Первый кадр, который уже «после разрыва», и причина."""
    n = len(L)
    cands = []
    first = start or 0
    lost = np.nonzero(~ok[first:])[0]
    if len(lost):
        cands.append((first + lost[0], "потеря метки"))
    if neck is not None:
        low = (neck <= neck_break_px) & np.isfinite(neck)
        for i in range(first, n - 1):
            if low[i] and low[i + 1]:
                cands.append((i, "разрыв перемычки (ширина=0)"))
                break
    dL = np.diff(L)
    good = np.isfinite(dL)
    if good.sum() > 10:
        med = np.median(dL[good])
        sigma = 1.4826 * np.median(np.abs(dL[good] - med))
        thr = max(jump_k * sigma, jump_min_px)
        jumps = np.nonzero(good & (np.abs(dL - med) > thr))[0]
        jumps = jumps[jumps >= first]
        if len(jumps):
            cands.append((jumps[0] + 1, f"скачок L ({dL[jumps[0]]:+.2f} px за кадр)"))
    if not cands:
        return None, "разрыв не найден"
    return min(cands, key=lambda c: c[0])


def review(cap, rows_pos, L, ok, idx0, l2, scale, L0):
    """Ручная проверка кадра L2. A/D (←/→) ±1 кадр, Z/C ±10, Enter — принять, Esc — оставить авто."""
    n = len(L)
    i = l2
    title = "L2 check: A/D or arrows = +-1, Z/C = +-10, ENTER = this is the last frame before break, ESC = keep auto"
    print("\nПроверка кадра L2: выберите последний кадр, где образец ещё цел (перемычка есть).")
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx0 + i)
        okf, frame = cap.read()
        if not okf:
            i = max(0, i - 1)
            continue
        p1, p2 = rows_pos[i]
        if np.isfinite(L[i]):
            d = L[i] - L0
            dl = f"dL = {d * scale:.3f} mm" if scale else f"dL = {d:.2f} px"
        else:
            dl = "dL = n/a (marker lost)"
        lines = [f"frame {idx0 + i}  ({i - l2:+d} from auto L2)", dl]
        view, _ = scaled(draw_overlay(frame, p1, p2, lines, ok[i]))
        cv2.imshow(title, view)
        k = -1
        while k == -1:                       # ждём клавишу, но замечаем закрытие окна крестиком
            k = cv2.waitKeyEx(50)
            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                return l2
        if k in (ord("a"), 2424832):
            i = max(0, i - 1)
        elif k in (ord("d"), 2555904):
            i = min(n - 1, i + 1)
        elif k == ord("z"):
            i = max(0, i - 10)
        elif k == ord("c"):
            i = min(n - 1, i + 10)
        elif k in (13, 10):
            cv2.destroyWindow(title)
            return i
        elif k == 27:
            cv2.destroyWindow(title)
            return l2


@dataclass
class Options:
    method: str = "blob"            # blob | template
    bright_marks: bool = False
    search: int = 40
    min_score: float = 0.5
    update_thr: float = 0.9
    neck: bool = True
    neck_halfwidth: int = None
    neck_margin: float = None
    neck_break_px: float = 0
    jump_k: float = 10
    jump_min_px: float = 2.0
    l0_frames: int = 10
    stop_after_lost: int = 30
    l0_mm: float = None
    mm_per_px: float = None
    calib_mm: float = None
    calib_px: float = None


@dataclass
class Track:
    video: str
    rows: list
    fps: float
    total: int
    start_frame: int
    roi1: tuple
    roi2: tuple
    neck_w0: float = None
    stopped: bool = False

    def row_of_frame(self, frame_idx):
        i = frame_idx - self.start_frame
        return i if 0 <= i < len(self.rows) else None


def track_video(video, roi1, roi2, opt, start_frame=0, end_frame=None,
                on_frame=None, should_stop=None, log=print):
    """Трекинг меток по видео. on_frame(row, frame) вызывается на каждом кадре."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ts = read_timestamps(video)
    if ts is None:
        log(f"Файл меток времени не найден — время берётся из видеофайла (номинально {fps:.2f} fps).")
    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    okf, frame = cap.read()
    if not okf:
        raise RuntimeError("Не удалось прочитать кадр.")
    gray = to_gray(frame)
    mk = dict(method=opt.method, search=opt.search, dark=not opt.bright_marks,
              min_score=opt.min_score, update_thr=opt.update_thr)
    m1, m2 = Marker(gray, roi1, log=log, **mk), Marker(gray, roi2, log=log, **mk)

    neck = None
    if opt.neck:
        L_init = np.hypot(*(m2.pos - m1.pos))
        halfw = opt.neck_halfwidth or max(20, int(0.5 * L_init))
        margin = opt.neck_margin if opt.neck_margin is not None else 0.6 * max(max(roi1[2:]), max(roi2[2:]))
        try:
            neck = Neck(gray, m1.pos, m2.pos, halfw, margin)
        except ValueError as e:
            log(f"Анализ шейки отключён: {e} (расстояние {L_init:.0f} px). Разрыв ищется по скачку L.")
    if neck is not None:
        log(f"Шейка: полоса ±{halfw}px, порог {neck.T:.0f}, образец {'светлее' if neck.bright else 'темнее'} "
            f"фона, начальная ширина {neck.w0:.1f}px")
        if neck.w0 >= 2 * halfw:
            log("ВНИМАНИЕ: образец шире полосы анализа шейки — ширина шейки будет неверной.")

    tr = Track(str(video), [], fps, total, start_frame, tuple(roi1), tuple(roi2),
               neck.w0 if neck else None)
    idx = start_frame
    lost_run = 0
    while True:
        if tr.rows:
            okf, frame = cap.read()
            if not okf or (end_frame and idx > end_frame):
                break
            gray = to_gray(frame)
            r1, r2 = m1.update(gray), m2.update(gray)
            both = r1 is not None and r2 is not None
        else:
            both = True
        p1, p2 = m1.pos.copy(), m2.pos.copy()
        L = float(np.hypot(*(p2 - p1))) if both else np.nan
        nw, npos = neck.measure(gray, p1, p2) if (neck and both) else (np.nan, np.nan)
        if ts is not None and idx < len(ts):
            t = ts[idx]
        else:
            # видео с телефона часто с переменной частотой кадров — берём время кадра из самого файла
            ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            t = ms / 1000 if ms > 0 or idx == 0 else idx / fps
        row = dict(frame=idx, t=t, p1=p1, p2=p2, L=L, neck=nw, neck_pos=npos, ok=both, s1=m1.score, s2=m2.score)
        tr.rows.append(row)
        if on_frame:
            on_frame(row, frame)
        lost_run = 0 if both else lost_run + 1
        if lost_run > opt.stop_after_lost:
            log(f"Метки потеряны {lost_run} кадров подряд — обработка остановлена на кадре {idx}.")
            break
        if should_stop and should_stop():
            tr.stopped = True
            log(f"Обработка остановлена пользователем на кадре {idx}.")
            break
        idx += 1
    cap.release()
    return tr


def summarize(tr, opt, l2=None, log=print):
    """Расчёт L0, разрыва, L2. l2 — индекс строки, если задан вручную."""
    rows = tr.rows
    n = len(rows)
    L = np.array([r["L"] for r in rows])
    ok = np.array([r["ok"] for r in rows])
    T = np.array([r["t"] for r in rows])
    NW = np.array([r["neck"] for r in rows]) if tr.neck_w0 is not None else None
    n0 = max(1, min(opt.l0_frames, n))
    L0 = float(np.nanmedian(L[:n0]))

    if opt.mm_per_px:
        scale = opt.mm_per_px
    elif opt.l0_mm:
        scale = opt.l0_mm / L0
    elif opt.calib_mm and opt.calib_px:
        scale = opt.calib_mm / opt.calib_px
    else:
        scale = None
    k, unit = (scale, "mm") if scale else (1.0, "px")

    start = detect_start(L, n0)
    brk, reason = detect_break(L, ok, NW, opt.jump_k, opt.jump_min_px, opt.neck_break_px, start)
    if brk is None:
        valid = np.nonzero(np.isfinite(L))[0]
        auto_l2 = int(valid[-1]) if len(valid) else 0
    else:
        valid = np.nonzero(np.isfinite(L[:brk]))[0]
        auto_l2 = int(valid[-1]) if len(valid) else 0
    manual = l2 is not None
    if not manual:
        l2 = auto_l2
    L2 = float(L[l2])

    summary = {
        "video": tr.video,
        "unit": unit,
        "mm_per_px": scale,
        "L0_px": L0,
        f"L0_{unit}": L0 * k,
        f"L2_{unit}": L2 * k,
        f"dL_break_{unit}": (L2 - L0) * k,
        "strain_at_break_pct": (L2 - L0) / L0 * 100,
        "L2_frame": rows[l2]["frame"],
        "t_L2_s": rows[l2]["t"],
        "L2_source": "вручную" if manual else "авто",
        "break_reason": reason,
        "auto_L2_frame": rows[auto_l2]["frame"],
        "motion_start_frame": rows[start]["frame"] if start is not None else None,
        "t_motion_start_s": rows[start]["t"] if start is not None else None,
        "noise_L0_px_std": float(np.nanstd(L[:n0])),
        "roi1": list(tr.roi1), "roi2": list(tr.roi2), "method": opt.method,
    }
    if NW is not None:
        summary[f"neck_w0_{unit}"] = tr.neck_w0 * k
        summary[f"neck_w_at_L2_{unit}"] = rows[l2]["neck"] * k
        summary["neck_pos_at_L2"] = rows[l2]["neck_pos"]
    return dict(L=L, ok=ok, T=T, NW=NW, L0=L0, scale=scale, k=k, unit=unit, start=start,
                brk=brk, reason=reason, l2=l2, auto_l2=auto_l2, L2=L2, summary=summary,
                dL=(L - L0) * k, strain=(L - L0) / L0 * 100)


def save_results(tr, res, out_dir=None, frame_l2=None):
    video = Path(tr.video)
    out = Path(out_dir) if out_dir else video.parent / f"{video.stem}_results"
    out.mkdir(parents=True, exist_ok=True)
    k, unit, L0, l2 = res["k"], res["unit"], res["L0"], res["l2"]
    with open(out / f"{video.stem}_ext.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t_s", "x1", "y1", "x2", "y2", "L_px", f"L_{unit}", f"dL_{unit}", "strain_pct",
                    f"neck_w_{unit}", "neck_pos", "score1", "score2", "valid", "after_break"])
        for i, r in enumerate(tr.rows):
            Li = r["L"]
            w.writerow([r["frame"], f"{r['t']:.6f}", f"{r['p1'][0]:.3f}", f"{r['p1'][1]:.3f}",
                        f"{r['p2'][0]:.3f}", f"{r['p2'][1]:.3f}", f"{Li:.4f}", f"{Li * k:.5f}",
                        f"{(Li - L0) * k:.5f}", f"{(Li - L0) / L0 * 100:.4f}",
                        f"{r['neck'] * k:.4f}" if np.isfinite(r["neck"]) else "",
                        f"{r['neck_pos']:.3f}" if np.isfinite(r["neck_pos"]) else "",
                        f"{r['s1']:.3f}", f"{r['s2']:.3f}", int(r["ok"]), int(i > l2)])
    (out / f"{video.stem}_summary.json").write_text(
        json.dumps(res["summary"], ensure_ascii=False, indent=2), encoding="utf-8")

    if frame_l2 is None:
        cap = cv2.VideoCapture(str(video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, tr.rows[l2]["frame"])
        okf, frame_l2 = cap.read()
        cap.release()
        frame_l2 = frame_l2 if okf else None
    if frame_l2 is not None:
        r = tr.rows[l2]
        img = draw_overlay(frame_l2, r["p1"], r["p2"],
                           [f"L2 frame {r['frame']}  t={r['t']:.3f}s",
                            f"L2={res['L2'] * k:.3f} {unit}  dL={(res['L2'] - L0) * k:.3f} {unit}"])
        cv2.imwrite(str(out / f"{video.stem}_L2_frame.png"), img)

    from matplotlib.figure import Figure
    fig = Figure(figsize=(10, 7 if res["NW"] is not None else 3.5))
    plot_results(fig, res)
    fig.tight_layout()
    fig.savefig(out / f"{video.stem}_plot.png", dpi=120)
    return out


def plot_results(fig, res, cursor_t=None):
    """Рисует ΔL(t) и ширину шейки(t) на matplotlib Figure."""
    fig.clear()
    T, unit, l2, start = res["T"], res["unit"], res["l2"], res["start"]
    nplots = 2 if res["NW"] is not None else 1
    axes = [fig.add_subplot(nplots, 1, 1)]
    if nplots == 2:
        axes.append(fig.add_subplot(2, 1, 2, sharex=axes[0]))
    axes[0].plot(T, res["dL"], lw=1)
    axes[0].set_ylabel(f"ΔL, {unit}")
    axes[0].plot([T[l2]], [res["dL"][l2]], "o", color="red", ms=5)
    if nplots == 2:
        axes[1].plot(T, res["NW"] * res["k"], lw=1, color="tab:orange")
        axes[1].set_ylabel(f"шейка, {unit}")
    for a in axes:
        a.axvline(T[l2], color="red", ls="--", lw=1, label="L2")
        if start is not None:
            a.axvline(T[start], color="gray", ls=":", lw=1, label="старт")
        if cursor_t is not None:
            a.axvline(cursor_t, color="tab:green", lw=0.8)
        a.grid(alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=8)
    axes[-1].set_xlabel("время видео, с")
    return axes


# ------------------------------------------------------------- данные машины

def _num(s):
    s = s.strip().strip('"').replace(" ", "").replace(" ", "").replace(",", ".")
    return float(s)


def _is_num(s):
    try:
        _num(s)
        return True
    except ValueError:
        return False


def load_machine_table(path):
    """Читает CSV/TXT экспорт машины. Сам находит разделитель, заголовок и строку единиц.
    -> (имена столбцов, массив данных [n, ncols] с NaN для нечисловых)."""
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "cp1251", "utf-16"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    lines = [ln for ln in text.splitlines() if ln.strip()]
    try:
        delim = csv.Sniffer().sniff("\n".join(lines[:50]), delimiters=",;\t").delimiter
    except csv.Error:
        delim = "\t" if "\t" in lines[0] else (";" if ";" in lines[0] else ",")
    rows = list(csv.reader(lines, delimiter=delim))
    numeric = [sum(_is_num(c) for c in r) >= 2 for r in rows]
    first = next((i for i in range(len(rows) - 2) if numeric[i] and numeric[i + 1] and numeric[i + 2]), None)
    if first is None:
        raise ValueError("Не найдено числовых данных в файле.")
    ncols = max(len(r) for r in rows[first:first + 50])
    head = []
    j = first - 1
    while j >= 0 and not numeric[j] and len(head) < 2:
        head.insert(0, rows[j])
        j -= 1
    names = []
    for c in range(ncols):
        parts = [h[c].strip().strip('"') for h in head if c < len(h) and h[c].strip().strip('"')]
        names.append(" ".join(parts[:1]) + (f" ({parts[1]})" if len(parts) > 1 else "") if parts else f"столбец {c + 1}")
    data = np.full((len(rows) - first, ncols), np.nan)
    for i, r in enumerate(rows[first:]):
        for c in range(min(ncols, len(r))):
            try:
                data[i, c] = _num(r[c])
            except ValueError:
                pass
    data = data[np.isfinite(data).sum(1) >= 2]
    return names, data


def guess_columns(names):
    low = [n.lower() for n in names]
    ti = next((i for i, n in enumerate(low) if any(k in n for k in ("time", "врем", "sec", "сек"))), 0)
    fi = next((i for i, n in enumerate(low) if i != ti and any(k in n for k in ("force", "сил", "load", "нагр", "(n)", "(kn)", "(н)", "(кн)"))),
              1 if ti != 1 else 0)
    return ti, fi


def align_merge(vt, vdl, vstrain, vvalid, tv_start, tv_break, mt, mf, mode="break", offset=0.0, start_frac=0.02):
    """Совмещает время видео с временем машины. -> dict с параметрами и совмещёнными массивами."""
    ok = np.isfinite(mt) & np.isfinite(mf)
    mt, mf = mt[ok], mf[ok]
    ipk = int(np.argmax(mf))
    i_brk = ipk + int(np.argmin(np.diff(mf[ipk:]))) if ipk < len(mf) - 1 else ipk
    i_start = int(np.argmax(mf > start_frac * mf[ipk]))
    tm_break, tm_start = mt[i_brk], mt[i_start]
    if mode == "offset":
        ka, kb = 1.0, offset
    elif mode == "start":
        ka, kb = 1.0, tm_start - tv_start
    elif mode == "both":
        ka = (tm_break - tm_start) / (tv_break - tv_start)
        kb = tm_start - ka * tv_start
    else:
        ka, kb = 1.0, tm_break - tv_break
    vt_m = ka * vt + kb
    lo, hi = vt_m[vvalid].min(), vt_m[vvalid].max()
    m = (mt >= lo) & (mt <= hi)
    t = mt[m]
    return dict(ka=ka, kb=kb, tm_start=tm_start, tm_break=tm_break, Fmax=mf[ipk], t_Fmax=mt[ipk],
                t=t, F=mf[m], dL=np.interp(t, vt_m[vvalid], vdl[vvalid]),
                strain=np.interp(t, vt_m[vvalid], vstrain[vvalid]))


def save_merged(path, mg, unit, force_name="force"):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_machine_s", force_name, f"dL_{unit}", "strain_pct"])
        for row in zip(mg["t"], mg["F"], mg["dL"], mg["strain"]):
            w.writerow([f"{row[0]:.5f}", f"{row[1]:.6g}", f"{row[2]:.5f}", f"{row[3]:.4f}"])


# ------------------------------------------------------------------ CLI

def opts_from_args(a):
    return Options(method=a.method, bright_marks=a.bright_marks, search=a.search, min_score=a.min_score,
                   update_thr=a.update_thr, neck=a.neck, neck_halfwidth=a.neck_halfwidth,
                   neck_margin=a.neck_margin, neck_break_px=a.neck_break_px, jump_k=a.jump_k,
                   jump_min_px=a.jump_min_px, l0_frames=a.l0_frames, stop_after_lost=a.stop_after_lost,
                   l0_mm=a.l0_mm, mm_per_px=a.mm_per_px, calib_mm=a.calib_mm)


def cmd_analyze(a):
    video = Path(a.video)
    opt = opts_from_args(a)
    cap = cv2.VideoCapture(str(video))
    if a.start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, a.start_frame)
    okf, frame = cap.read()
    cap.release()
    if not okf:
        sys.exit(f"Не удалось прочитать {video}")
    gray = to_gray(frame)
    if a.roi1 and a.roi2:
        roi1, roi2 = a.roi1, a.roi2
    elif a.box:
        roi1 = select_roi(frame, "Marker 1: drag a box around the mark, ENTER")
        roi2 = select_roi(frame, "Marker 2: drag a box around the mark, ENTER")
    else:
        print("Кликните по метке 1, затем по метке 2 и нажмите Enter (Backspace — отменить клик, Esc — выход).")
        pts = pick_points(frame, 2, "Click mark 1, then mark 2, then ENTER")
        roi1, roi2 = (roi_from_click(gray, p, not a.bright_marks) for p in pts)
    print(f"ROI метки 1: {','.join(map(str, roi1))}   ROI метки 2: {','.join(map(str, roi2))}"
          "  (можно передать через --roi1/--roi2 для повторного запуска)")
    if a.calib_mm:
        pts = pick_points(frame, 2, f"Calibration: click 2 points {a.calib_mm} mm apart, ENTER")
        opt.calib_px = float(np.hypot(*(pts[1] - pts[0])))

    def on_frame(row, fr):
        if a.show and row["frame"] % a.show_every == 0:
            info = [f"frame {row['frame']}  t={row['t']:.2f}s", f"L={row['L']:.2f}px"]
            view, _ = scaled(draw_overlay(fr, row["p1"], row["p2"], info, row["ok"]))
            cv2.imshow("analyze", view)
            cv2.waitKey(1)

    t_wall = time.perf_counter()
    tr = track_video(video, roi1, roi2, opt, a.start_frame, a.end_frame, on_frame)
    cv2.destroyAllWindows()
    print(f"Обработано {len(tr.rows)} кадров за {time.perf_counter() - t_wall:.1f} с.")
    res = summarize(tr, opt)
    if res["scale"] is None:
        print("Масштаб не задан (--l0-mm / --mm-per-px / --calib-mm) — результаты в пикселях.")
    l2 = None
    if a.break_frame is not None:
        valid = np.nonzero(np.isfinite(res["L"][:a.break_frame - a.start_frame]))[0]
        l2 = int(valid[-1])
    elif res["brk"] is not None:
        print(f"Разрыв: кадр {tr.rows[min(res['brk'], len(tr.rows) - 1)]['frame']} ({res['reason']}). "
              f"Последний целый кадр: {tr.rows[res['l2']]['frame']}")
    else:
        print("Разрыв автоматически не найден; за L2 взят последний кадр с валидными метками.")
    if not a.no_review:
        cap = cv2.VideoCapture(str(video))
        l2 = review(cap, [(r["p1"], r["p2"]) for r in tr.rows], res["L"], res["ok"], a.start_frame or 0,
                    res["l2"] if l2 is None else l2, res["scale"], res["L0"])
        cap.release()
    if l2 is not None and l2 != res["l2"]:
        res = summarize(tr, opt, l2)
    out = save_results(tr, res, a.out_dir)

    k, unit, L0, L2 = res["k"], res["unit"], res["L0"], res["L2"]
    r2 = tr.rows[res["l2"]]
    print("\n================ РЕЗУЛЬТАТ ================")
    print(f"L0  = {L0 * k:.4f} {unit}   (шум L0: {np.nanstd(res['L'][:opt.l0_frames]) * k:.4f} {unit})")
    print(f"L2  = {L2 * k:.4f} {unit}   кадр {r2['frame']}, t = {r2['t']:.3f} с")
    print(f"ΔL  = {(L2 - L0) * k:.4f} {unit}   ε = {(L2 - L0) / L0 * 100:.2f} %")
    if tr.neck_w0 is not None:
        print(f"Ширина шейки в L2: {r2['neck'] * k:.3f} {unit} (исходная {tr.neck_w0 * k:.3f})")
    print(f"Файлы: {out}")


def cmd_merge(a):
    with open(a.ext_csv, encoding="utf-8-sig") as f:
        vr = list(csv.DictReader(f))
    unit = "mm" if "dL_mm" in vr[0] else "px"
    vt = np.array([float(r["t_s"]) for r in vr])
    vdl = np.array([float(r[f"dL_{unit}"]) for r in vr])
    vstrain = np.array([float(r["strain_pct"]) for r in vr])
    valid = np.array([r["valid"] == "1" and r["after_break"] == "0" for r in vr])
    summ_path = Path(a.ext_csv).with_name(Path(a.ext_csv).name.replace("_ext.csv", "_summary.json"))
    summ = json.loads(summ_path.read_text(encoding="utf-8")) if summ_path.exists() else {}

    names, data = load_machine_table(a.machine_csv)
    gti, gfi = guess_columns(names)

    def pick(spec, default):
        if spec is None:
            return default
        if spec.isdigit():
            return int(spec)
        for i, n in enumerate(names):
            if n.lower().startswith(spec.lower()):
                return i
        sys.exit(f"Столбец '{spec}' не найден. Есть: {names}")

    ti, fi = pick(a.time_col, gti), pick(a.force_col, gfi)
    print(f"Столбцы машины: время = '{names[ti]}', сила = '{names[fi]}'")
    mg = align_merge(vt, vdl, vstrain, valid, summ.get("t_motion_start_s"), summ.get("t_L2_s"),
                     data[:, ti], data[:, fi], a.align, a.offset, a.start_frac)
    print(f"Машина: старт {mg['tm_start']:.3f} с, Fmax {mg['Fmax']:.4g} @ {mg['t_Fmax']:.3f} с, разрыв {mg['tm_break']:.3f} с")
    print(f"Совмещение '{a.align}': t_машины = {mg['ka']:.6f} * t_видео + {mg['kb']:.4f}")
    out = Path(a.out) if a.out else Path(a.ext_csv).with_name(Path(a.ext_csv).stem.replace("_ext", "") + "_merged.csv")
    save_merged(out, mg, unit, names[fi])
    from matplotlib.figure import Figure
    fig = Figure(figsize=(9, 5.5))
    ax = fig.add_subplot()
    ax.plot(mg["dL"], mg["F"], lw=1)
    ax.set_xlabel(f"ΔL (видеоэкстензометр), {unit}")
    ax.set_ylabel(names[fi])
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=120)
    print(f"Совмещённые данные: {out}")


def main():
    import faulthandler
    faulthandler.enable()                        # при аварийном падении OpenCV напечатать, где именно
    for s in (sys.stdout, sys.stderr):           # консоль cp1251 не знает «Δ», «ε» — не падаем на выводе
        try:
            s.reconfigure(errors="replace")
        except AttributeError:
            pass
    p = argparse.ArgumentParser(description="Видеоэкстензометр: ΔL по видео обычной камеры.")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("record", help="запись видео с камеры с метками времени")
    r.add_argument("--camera", type=int, default=0)
    r.add_argument("--out", default=f"test_{datetime.now():%Y%m%d_%H%M%S}.avi")
    r.add_argument("--width", type=int, default=1920)
    r.add_argument("--height", type=int, default=1080)
    r.add_argument("--fps", type=float, default=30)
    r.add_argument("--codec", default="MJPG", help="FOURCC: MJPG (по умолчанию), FFV1 — без потерь, но тяжелее")
    r.add_argument("--focus", type=float, help="фиксированный фокус (отключает автофокус)")
    r.add_argument("--exposure", type=float, help="ручная экспозиция (отключает автоэкспозицию)")
    r.set_defaults(func=cmd_record)

    an = sub.add_parser("analyze", help="обработка видео")
    an.add_argument("video")
    g = an.add_argument_group("масштаб (одно из)")
    g.add_argument("--l0-mm", type=float, help="расстояние между метками в начале испытания, мм")
    g.add_argument("--mm-per-px", type=float, help="готовый масштаб, мм/пиксель")
    g.add_argument("--calib-mm", type=float, help="кликнуть 2 точки на линейке на таком расстоянии, мм")
    an.add_argument("--roi1", type=parse_roi, help="x,y,w,h метки 1 (без выделения мышью)")
    an.add_argument("--roi2", type=parse_roi, help="x,y,w,h метки 2")
    an.add_argument("--box", action="store_true", help="выделять метки рамкой вместо клика")
    an.add_argument("--method", choices=["blob", "template"], default="blob",
                    help="blob — центроид контрастной точки (лучше для нарисованных меток); "
                         "template — сопоставление шаблона (любая текстура, наклейки)")
    an.add_argument("--bright-marks", action="store_true", help="метки светлее образца")
    an.add_argument("--search", type=int, default=40, help="радиус поиска метки между кадрами, px")
    an.add_argument("--min-score", type=float, default=0.5, help="template: ниже — метка потеряна")
    an.add_argument("--update-thr", type=float, default=0.9, help="template: ниже — обновить шаблон")
    an.add_argument("--neck", action="store_true", help="измерять ширину шейки и ловить разрыв перемычки")
    an.add_argument("--neck-halfwidth", type=int, help="полуширина полосы анализа, px (должна быть больше полуширины образца)")
    an.add_argument("--neck-margin", type=float, help="отступ от меток при поиске шейки, px")
    an.add_argument("--neck-break-px", type=float, default=0, help="ширина перемычки, при которой считается разрыв, px")
    an.add_argument("--jump-k", type=float, default=10, help="порог скачка L, в сигмах шума")
    an.add_argument("--jump-min-px", type=float, default=2.0, help="минимальный скачок L, px")
    an.add_argument("--break-frame", type=int, help="задать кадр разрыва вручную (первый кадр уже после разрыва)")
    an.add_argument("--l0-frames", type=int, default=10, help="сколько первых кадров усреднять для L0")
    an.add_argument("--start-frame", type=int, default=0)
    an.add_argument("--end-frame", type=int)
    an.add_argument("--stop-after-lost", type=int, default=30)
    an.add_argument("--show", action="store_true", help="показывать трекинг в процессе")
    an.add_argument("--show-every", type=int, default=1)
    an.add_argument("--no-review", action="store_true", help="не открывать ручную проверку кадра L2")
    an.add_argument("--out-dir")
    an.set_defaults(func=cmd_analyze)

    m = sub.add_parser("merge", help="совмещение с данными машины (CSV)")
    m.add_argument("ext_csv", help="*_ext.csv из analyze")
    m.add_argument("machine_csv", help="экспорт машины (время, сила)")
    m.add_argument("--time-col", help="номер (с 0) или начало имени столбца времени (по умолчанию — угадать)")
    m.add_argument("--force-col", help="номер (с 0) или начало имени столбца силы (по умолчанию — угадать)")
    m.add_argument("--align", choices=["break", "start", "both", "offset"], default="break",
                   help="break — по моменту разрыва (по умолчанию), start — по старту нагружения, "
                        "both — по обоим (корректирует и сдвиг, и масштаб времени), offset — вручную")
    m.add_argument("--offset", type=float, default=0.0, help="для --align offset: t_машины = t_видео + offset")
    m.add_argument("--start-frac", type=float, default=0.02, help="старт нагружения: F > доля от Fmax")
    m.add_argument("--out")
    m.set_defaults(func=cmd_merge)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
