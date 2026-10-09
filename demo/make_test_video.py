"""
Синтетическое испытание для проверки extensometer.py: светлый образец на тёмном фоне,
две тёмные метки, равномерное удлинение -> локализация шейки -> разрыв перемычки.
Пишет demo.avi, demo.ts.csv, demo_truth.json и фиктивный экспорт машины demo_machine.csv.
"""
import csv
import json
from pathlib import Path

import cv2
import numpy as np

W, H, FPS = 1280, 720, 30
CX = 640                       # ось образца
Y_GRIP = 680                   # нижний захват (неподвижен)
Y1, Y2, Y_NECK = 200, 520, 380 # метки и место будущей шейки (исходные координаты)
W0 = 120                       # ширина образца, px
MM_PER_PX = 50 / (Y2 - Y1)     # L0 = 50 мм

N_STILL, N_LOAD, N_AFTER = 15, 200, 25
BREAK = N_STILL + N_LOAD       # первый кадр после разрыва
rng = np.random.default_rng(1)

# «материал» в исходных координатах: текстура + метки
mat = np.full((H, W0 + 1), 205, np.float32) + rng.normal(0, 6, (H, W0 + 1)).astype(np.float32)
mat = cv2.GaussianBlur(mat, (3, 3), 0)
for y in (Y1, Y2):
    cv2.circle(mat, (W0 // 2, y), 7, 35, -1, cv2.LINE_AA)

Yg = np.arange(0, H, 0.25)     # сетка исходных координат для обращения отображения


def state(f):
    k = np.clip((f - N_STILL) / N_LOAD, 0, 1)
    eps = 0.20 * k                                   # равномерная деформация
    neck = np.clip((k - 0.7) / 0.3, 0, 1)            # шейка в последние 30% нагружения
    return eps, 45 * neck ** 2, 0.97 * neck ** 1.5    # (ε, доп. удлинение в шейке px, глубина шейки)


def forward(Y, eps, extra, broken):
    """исходная координата -> текущая (верх тянется вверх, низ у захвата неподвижен)."""
    y = Y_GRIP - (Y_GRIP - Y) * (1 + eps) - extra / (1 + np.exp((Y - Y_NECK) / 6))
    if broken:
        y = y - 12 / (1 + np.exp((Y - Y_NECK) / 0.5))   # верхняя часть отскакивает при разрыве
    return y


def render(f):
    eps, extra, depth = state(f if f < BREAK else BREAK - 1)
    broken = f >= BREAK
    yg = forward(Yg, eps, extra, broken)
    ys = np.arange(H, dtype=np.float64)
    Ysrc = np.interp(ys, yg, Yg, left=-1e9, right=1e9)   # обратное отображение (yg монотонна)
    half = W0 / 2 / np.sqrt(1 + eps) * (1 - depth * np.exp(-((Ysrc - Y_NECK) / 14) ** 2))
    if broken:
        half[np.abs(Ysrc - Y_NECK) < 0.6] = 0             # перемычка разорвана
    xs = np.arange(W, dtype=np.float64)
    u = (xs[None, :] - CX) / np.maximum(half[:, None], 1e-6)
    mapx = (W0 / 2 + u * W0 / 2).astype(np.float32)
    mapy = np.repeat(Ysrc[:, None], W, 1).astype(np.float32)
    img = cv2.remap(mat, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=40)
    inside = (np.abs(u) <= 1) & (half[:, None] > 0) & (Ysrc[:, None] > 40) & (Ysrc[:, None] < H)
    img = np.where(inside, img, 40).astype(np.float32)
    img += rng.normal(0, 2.5, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), forward(np.array([Y1, Y2]), eps, extra, broken)


def main(out_dir="."):
    out = Path(out_dir)
    vw = cv2.VideoWriter(str(out / "demo.avi"), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    truth_L = []
    with open(out / "demo.ts.csv", "w", newline="") as f:
        f.write("# synthetic\n")
        w = csv.writer(f)
        w.writerow(["frame", "t_s"])
        for i in range(BREAK + N_AFTER):
            img, (y1, y2) = render(i)
            vw.write(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
            w.writerow([i, f"{i / FPS:.6f}"])
            truth_L.append(y2 - y1)
    vw.release()
    L0 = truth_L[0]
    L2 = truth_L[BREAK - 1]
    truth = {"mm_per_px": MM_PER_PX, "L0_px": L0, "L2_px": L2, "L2_frame": BREAK - 1,
             "dL_break_mm": (L2 - L0) * MM_PER_PX, "strain_at_break_pct": (L2 - L0) / L0 * 100,
             "L_px": truth_L}
    (out / "demo_truth.json").write_text(json.dumps(truth, indent=1))

    # фиктивный экспорт машины: свои часы (сдвиг +7.3 с), 100 Гц, сила растёт и падает в разрыв
    with open(out / "demo_machine.csv", "w", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["Time", "Force"])
        w.writerow(["sec", "N"])
        t_break = (BREAK - 1) / FPS + 7.3
        for t in np.arange(0, t_break + 1.0, 0.01):
            tv = t - 7.3
            k = np.clip((tv - N_STILL / FPS) / (N_LOAD / FPS), 0, 1)
            F = 0.0 if t > t_break + 0.005 else 12000 * (1 - np.exp(-k * 12)) * (1 - 0.25 * max(0, k - 0.7) / 0.3)
            w.writerow([f"{t:.2f}".replace(".", ","), f"{F + rng.normal(0, 5):.1f}".replace(".", ",")])
    print(f"Готово. Истинные значения: L2 кадр {BREAK - 1}, ΔL = {truth['dL_break_mm']:.4f} мм, "
          f"ε = {truth['strain_at_break_pct']:.3f} %")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
