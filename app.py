"""
Видеоэкстензометр — графическое приложение.
Слева — видео (камера или файл), справа — параметры, результаты и графики.
"""
import csv
import os
import queue
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import extensometer as ex

APP_TITLE = "Видеоэкстензометр"
METHODS = {"Точки (центр пятна)": "blob", "Шаблон (любая текстура)": "template"}
ALIGN = {"По моменту разрыва": "break", "По началу нагружения": "start",
         "По началу и разрыву": "both", "Сдвиг вручную, с": "offset"}
RESOLUTIONS = ["1920x1080", "1280x720", "2560x1440", "3840x2160", "640x480"]
VIDEO_TYPES = [("Видео", "*.avi *.mp4 *.mov *.mkv *.wmv *.m4v"), ("Все файлы", "*.*")]


def list_cameras():
    """Названия камер в порядке номеров DirectShow: ['0: Integrated Camera', '1: Camo', …]."""
    try:
        from pygrabber.dshow_graph import FilterGraph
        names = FilterGraph().get_input_devices()
        if names:
            return [f"{i}: {n}" for i, n in enumerate(names)]
    except Exception:
        pass
    return [str(i) for i in range(6)]


def app_dir():
    return Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).parent


# ------------------------------------------------------------------ источники

class VideoFile:
    def __init__(self, path):
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Не удалось открыть видео:\n{path}")
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.ts = ex.read_timestamps(self.path)
        self.next_idx = 0
        if self.n <= 0:                        # у некоторых файлов число кадров неизвестно — посчитаем
            n = 0
            while self.cap.grab():
                n += 1
            self.n = n
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def time_of(self, idx):
        return self.ts[idx] if self.ts is not None and idx < len(self.ts) else idx / self.fps

    def read(self, idx):
        if idx != self.next_idx:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        self.next_idx = idx + 1
        return frame if ok else None

    def close(self):
        self.cap.release()


class Camera:
    """Захват камеры в отдельном потоке; запись в AVI (MJPG) + файл меток времени."""

    NO_PICTURE = ("Камера {i} подключена, но не передаёт изображение (кадры чёрные).\n\n"
                  "Проверьте:\n"
                  "• шторку на камере ноутбука (ползунок над экраном);\n"
                  "• клавишу отключения камеры (значок перечёркнутой камеры, часто F9 / Fn+F9) "
                  "или режим «Camera privacy» в Lenovo Vantage;\n"
                  "• Параметры Windows → Конфиденциальность → Камера → доступ для классических приложений;\n"
                  "• не открыта ли камера в другой программе (Zoom, Teams, браузер);\n"
                  "• другой номер камеры в списке «Камера №».")

    def __init__(self, source, width, height, fps=30):
        """source — номер камеры (0, 1, …) или адрес видеопотока (http://…, rtsp://…), напр. с телефона."""
        source = str(source).strip()
        is_url = not source.isdigit()
        index = source if is_url else int(source)
        if is_url:
            backends = [cv2.CAP_FFMPEG]
        elif sys.platform == "win32":
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF]       # DirectShow, затем Media Foundation
        else:
            backends = [cv2.CAP_ANY]
        self.cap = frame = None
        opened = False
        for be in backends:
            cap = cv2.VideoCapture(index, be)
            if not cap.isOpened():
                continue
            opened = True
            if not is_url:                                 # у потока с телефона размер задаётся в самом телефоне
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_FPS, fps)
            f = self._probe(cap, 8.0 if is_url else 4.0)
            if f is not None:
                self.cap, frame = cap, f
                break
            cap.release()
        if self.cap is None:
            if is_url:
                raise RuntimeError(f"Не удалось получить видео по адресу\n{source}\n\n"
                                   "Проверьте, что приложение-камера на телефоне запущено, телефон и компьютер "
                                   "в одной сети Wi‑Fi, и адрес введён полностью (как показывает приложение).")
            if not opened:
                raise RuntimeError(f"Камера {index} не найдена. Выберите другой номер в списке «Камера №».")
            raise RuntimeError(self.NO_PICTURE.format(i=index))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not self.fps or self.fps <= 0:
            self.fps = fps
        self.frame = frame
        self.lock = threading.Lock()
        self.running = True
        self.error = None
        self.writer = self.tsf = self.tsw = None
        self.t0 = 0.0
        self.nrec = 0
        self.rec_path = None
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    @staticmethod
    def _probe(cap, seconds=4.0):
        """Ждёт первый «живой» кадр (камере нужно время на автоэкспозицию). None — изображения нет."""
        t0 = time.perf_counter()
        last = None
        while time.perf_counter() - t0 < seconds:
            ok, f = cap.read()
            if not ok:
                continue
            last = f
            if f.mean() > 2 or f.std() > 2:
                return f
        return None if last is None or (last.mean() <= 2 and last.std() <= 2) else last

    @property
    def size(self):
        h, w = self.frame.shape[:2]
        return w, h

    @property
    def recording(self):
        return self.writer is not None

    def rec_time(self):
        return time.perf_counter() - self.t0 if self.recording else 0.0

    def _loop(self):
        while self.running:
            ok, f = self.cap.read()
            if not ok:
                self.error = "Камера перестала отдавать кадры."
                break
            with self.lock:
                if self.writer is not None:
                    t = time.perf_counter() - self.t0
                    self.writer.write(f)
                    self.tsw.writerow([self.nrec, f"{t:.6f}"])
                    self.nrec += 1
                self.frame = f
        self.cap.release()
        self.stop_record()

    def start_record(self, path):
        with self.lock:
            w, h = self.size
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), self.fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError(f"Не удалось создать файл {path}")
            self.tsf = open(ex.timestamps_path(path), "w", encoding="utf-8", newline="")
            self.tsf.write(f"# start_wall={datetime.now().isoformat(timespec='milliseconds')}\n")
            self.tsw = csv.writer(self.tsf)
            self.tsw.writerow(["frame", "t_s"])
            self.nrec = 0
            self.rec_path = str(path)
            self.t0 = time.perf_counter()
            self.writer = writer

    def stop_record(self):
        with self.lock:
            n, dur, path = self.nrec, self.rec_time(), self.rec_path
            if self.writer is not None:
                self.writer.release()
                self.tsf.close()
            self.writer = self.tsf = self.tsw = None
        return path, n, dur

    def close(self):
        self.running = False
        self.thread.join(timeout=3)


# ---------------------------------------------------------------------- окно

class App:
    def __init__(self, root):
        self.root = root
        root.title(APP_TITLE)
        root.geometry("1500x900")
        try:
            root.state("zoomed")
        except tk.TclError:
            pass
        root.report_callback_exception = self._on_tk_error
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.q = queue.Queue()
        self.video = None
        self.camera = None
        self.cur_idx = 0
        self.cur_frame = None
        self.picking = False
        self.pick_pts = []
        self.pick_frame = None
        self.rois = None
        self.tr = self.res = self.opt = None
        self.l2_manual = None
        self.out_dir = None
        self.analysis_thread = None
        self.stop_flag = False
        self.live = None
        self.view = (1.0, 0, 0)
        self._photo = None
        self._slider_lock = False
        self.machine = None
        self.merged = None

        self._build_ui()
        self.root.after(30, self._poll)
        self.set_status("Откройте видео или включите камеру.")

    # ---------------------------------------------------------- интерфейс
    def _build_ui(self):
        style = ttk.Style()
        style.configure("Big.TButton", font=("Segoe UI", 10, "bold"), padding=6)
        style.configure("Val.TLabel", font=("Segoe UI", 11, "bold"))
        style.configure("Head.TLabel", font=("Segoe UI", 10))

        paned = ttk.PanedWindow(self.root, orient="horizontal")
        paned.pack(fill="both", expand=True)
        left = ttk.Frame(paned)
        right = ttk.Frame(paned, width=520)
        paned.add(left, weight=3)
        paned.add(right, weight=1)

        # --- панель источника
        bar = ttk.Frame(left, padding=(6, 6, 6, 2))
        bar.pack(fill="x")
        ttk.Button(bar, text="📂 Открыть видео…", command=self.open_video_dialog).pack(side="left")
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Label(bar, text="Камера или адрес:").pack(side="left")
        self.var_cam = tk.StringVar(value="0")
        # камера из списка («1: Camo») или адрес потока с телефона (http://192.168.x.x:port/video)
        self.cb_cam = ttk.Combobox(bar, textvariable=self.var_cam, width=30)
        self.cb_cam.pack(side="left", padx=(2, 0))
        ttk.Button(bar, text="⟳", width=3, command=self.refresh_cameras).pack(side="left", padx=(2, 6))
        self.refresh_cameras()
        self.var_res = tk.StringVar(value=RESOLUTIONS[0])
        ttk.Combobox(bar, textvariable=self.var_res, values=RESOLUTIONS, width=10,
                     state="readonly").pack(side="left", padx=(0, 6))
        self.btn_cam = ttk.Button(bar, text="🎥 Включить камеру", command=self.toggle_camera)
        self.btn_cam.pack(side="left")
        self.btn_rec = ttk.Button(bar, text="⏺ Начать запись", command=self.toggle_record, state="disabled")
        self.btn_rec.pack(side="left", padx=6)

        # --- видео
        self.canvas = tk.Canvas(left, bg="#1e1e1e", highlightthickness=0, cursor="arrow")
        self.canvas.pack(fill="both", expand=True, padx=6)
        self.canvas.bind("<Configure>", lambda e: self.render())
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        nav = ttk.Frame(left, padding=(6, 4))
        nav.pack(fill="x")
        for txt, d in (("⏮", -10), ("◀", -1), ("▶", 1), ("⏭", 10)):
            ttk.Button(nav, text=txt, width=3, command=lambda d=d: self.step(d)).pack(side="left")
        self.slider = ttk.Scale(nav, from_=0, to=1, orient="horizontal", command=self.on_slider)
        self.slider.pack(side="left", fill="x", expand=True, padx=8)
        self.lbl_frame = ttk.Label(nav, text="—")
        self.lbl_frame.pack(side="left")

        self.lbl_status = ttk.Label(left, text="", padding=(8, 2, 8, 6), foreground="#1a5fb4")
        self.lbl_status.pack(fill="x")

        # --- правая панель
        rp = ttk.Frame(right, padding=6)
        rp.pack(fill="both", expand=True)

        p1 = ttk.LabelFrame(rp, text=" 1. Параметры ", padding=6)
        p1.pack(fill="x")
        ttk.Label(p1, text="Расстояние между метками L0, мм:").grid(row=0, column=0, sticky="w")
        self.var_l0 = tk.StringVar(value="50")
        e = ttk.Entry(p1, textvariable=self.var_l0, width=10)
        e.grid(row=0, column=1, sticky="w", padx=4)
        e.bind("<Return>", lambda ev: self.recalc())
        e.bind("<FocusOut>", lambda ev: self.recalc())
        ttk.Label(p1, text="Метод трекинга:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.var_method = tk.StringVar(value=list(METHODS)[0])
        ttk.Combobox(p1, textvariable=self.var_method, values=list(METHODS), state="readonly",
                     width=24).grid(row=1, column=1, sticky="w", padx=4, pady=(4, 0))
        self.var_bright = tk.BooleanVar(value=False)
        ttk.Checkbutton(p1, text="Метки светлее образца", variable=self.var_bright).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))
        self.var_neck = tk.BooleanVar(value=True)
        ttk.Checkbutton(p1, text="Следить за шейкой и разрывом перемычки", variable=self.var_neck).grid(
            row=3, column=0, columnspan=2, sticky="w")

        p2 = ttk.LabelFrame(rp, text=" 2. Метки ", padding=6)
        p2.pack(fill="x", pady=(6, 0))
        self.btn_pick = ttk.Button(p2, text="✚ Указать метки на видео", command=self.start_picking)
        self.btn_pick.pack(side="left")
        self.lbl_pick = ttk.Label(p2, text="не указаны", foreground="#a51d2d")
        self.lbl_pick.pack(side="left", padx=8)

        p3 = ttk.LabelFrame(rp, text=" 3. Анализ ", padding=6)
        p3.pack(fill="x", pady=(6, 0))
        self.btn_run = ttk.Button(p3, text="▶ Запустить анализ", style="Big.TButton", command=self.start_analysis)
        self.btn_run.pack(side="left")
        self.btn_stop = ttk.Button(p3, text="■ Стоп", command=self.stop_analysis, state="disabled")
        self.btn_stop.pack(side="left", padx=6)
        self.prog = ttk.Progressbar(p3, mode="determinate")
        self.prog.pack(side="left", fill="x", expand=True, padx=(6, 0))

        nb = ttk.Notebook(rp)
        nb.pack(fill="both", expand=True, pady=(8, 0))
        self.nb = nb
        self._build_results_tab(nb)
        self._build_merge_tab(nb)
        self._build_log_tab(nb)

        for key, d in (("<Left>", -1), ("<Right>", 1), ("<Prior>", -10), ("<Next>", 10)):
            self.root.bind(key, lambda e, d=d: self._key_step(e, d))
        self.root.bind("<Escape>", lambda e: self.cancel_picking())

    def _build_results_tab(self, nb):
        tab = ttk.Frame(nb, padding=6)
        nb.add(tab, text="Результаты")
        grid = ttk.Frame(tab)
        grid.pack(fill="x")
        self.vals = {}
        items = [("L0", "L0 (начальная база)"), ("L2", "L2 (последний кадр до разрыва)"),
                 ("dL", "ΔL при разрыве"), ("eps", "Удлинение при разрыве ε"),
                 ("frame", "Кадр L2 / время"), ("neck", "Ширина шейки в L2"), ("reason", "Как найден разрыв")]
        for r, (k, title) in enumerate(items):
            ttk.Label(grid, text=title + ":", style="Head.TLabel").grid(row=r, column=0, sticky="w", pady=1)
            v = tk.StringVar(value="—")
            ttk.Label(grid, textvariable=v, style="Val.TLabel").grid(row=r, column=1, sticky="w", padx=8)
            self.vals[k] = v

        l2bar = ttk.Frame(tab)
        l2bar.pack(fill="x", pady=(6, 2))
        self.btn_setl2 = ttk.Button(l2bar, text="Назначить L2 = текущий кадр", command=self.set_l2_here, state="disabled")
        self.btn_setl2.pack(side="left")
        self.btn_autol2 = ttk.Button(l2bar, text="Авто L2", command=self.auto_l2, state="disabled")
        self.btn_autol2.pack(side="left", padx=4)
        self.btn_gol2 = ttk.Button(l2bar, text="Перейти к L2", command=self.goto_l2, state="disabled")
        self.btn_gol2.pack(side="left")
        self.btn_folder = ttk.Button(l2bar, text="📁 Папка результатов", command=self.open_folder, state="disabled")
        self.btn_folder.pack(side="right")
        ttk.Label(tab, text="Подсказка: листайте кадры ←/→ и проверьте, что в кадре L2 перемычка ещё цела. "
                            "Клик по графику — переход к этому моменту.", wraplength=480,
                  foreground="#5e5c64").pack(fill="x")

        self.fig = Figure(figsize=(5, 4), dpi=90)
        self.fig_canvas = FigureCanvasTkAgg(self.fig, master=tab)
        self.fig_canvas.get_tk_widget().pack(fill="both", expand=True, pady=(4, 0))
        self.fig_canvas.mpl_connect("button_press_event", self.on_plot_click)
        self.cursor_lines = []

    def _build_merge_tab(self, nb):
        tab = ttk.Frame(nb, padding=6)
        nb.add(tab, text="Кривая F–ΔL")
        top = ttk.Frame(tab)
        top.pack(fill="x")
        ttk.Button(top, text="📄 Загрузить данные машины (CSV/TXT)…", command=self.load_machine).grid(
            row=0, column=0, columnspan=4, sticky="w")
        self.lbl_mfile = ttk.Label(top, text="файл не загружен", foreground="#5e5c64")
        self.lbl_mfile.grid(row=1, column=0, columnspan=4, sticky="w", pady=(2, 4))
        ttk.Label(top, text="Время:").grid(row=2, column=0, sticky="w")
        self.var_tcol = tk.StringVar()
        self.cb_tcol = ttk.Combobox(top, textvariable=self.var_tcol, state="readonly", width=20)
        self.cb_tcol.grid(row=2, column=1, sticky="w", padx=4)
        ttk.Label(top, text="Сила:").grid(row=2, column=2, sticky="w")
        self.var_fcol = tk.StringVar()
        self.cb_fcol = ttk.Combobox(top, textvariable=self.var_fcol, state="readonly", width=20)
        self.cb_fcol.grid(row=2, column=3, sticky="w", padx=4)
        ttk.Label(top, text="Совмещение:").grid(row=3, column=0, sticky="w", pady=(4, 0))
        self.var_align = tk.StringVar(value=list(ALIGN)[0])
        ttk.Combobox(top, textvariable=self.var_align, values=list(ALIGN), state="readonly", width=20).grid(
            row=3, column=1, sticky="w", padx=4, pady=(4, 0))
        ttk.Label(top, text="Сдвиг, с:").grid(row=3, column=2, sticky="w", pady=(4, 0))
        self.var_offset = tk.StringVar(value="0")
        ttk.Entry(top, textvariable=self.var_offset, width=8).grid(row=3, column=3, sticky="w", padx=4, pady=(4, 0))
        ttk.Button(top, text="Построить кривую", style="Big.TButton", command=self.build_merge).grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.lbl_merge = ttk.Label(tab, text="", wraplength=480, foreground="#5e5c64")
        self.lbl_merge.pack(fill="x", pady=(4, 0))
        self.fig2 = Figure(figsize=(5, 4), dpi=90)
        self.fig2_canvas = FigureCanvasTkAgg(self.fig2, master=tab)
        self.fig2_canvas.get_tk_widget().pack(fill="both", expand=True, pady=(4, 0))

    def _build_log_tab(self, nb):
        tab = ttk.Frame(nb, padding=4)
        nb.add(tab, text="Журнал")
        self.txt_log = tk.Text(tab, height=10, wrap="word", font=("Consolas", 9))
        sb = ttk.Scrollbar(tab, command=self.txt_log.yview)
        self.txt_log.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.txt_log.pack(fill="both", expand=True)

    # ------------------------------------------------------------ сервис
    def log(self, msg):
        self.txt_log.insert("end", f"[{datetime.now():%H:%M:%S}] {msg}\n")
        self.txt_log.see("end")

    def set_status(self, msg):
        self.lbl_status.configure(text=msg)

    def _on_tk_error(self, exc, val, tb):
        text = "".join(traceback.format_exception(exc, val, tb))
        self.log(text)
        messagebox.showerror(APP_TITLE, f"Ошибка: {val}\n\nПодробности — во вкладке «Журнал».")

    def _key_step(self, e, d):
        if isinstance(e.widget, (tk.Entry, ttk.Entry, ttk.Combobox, tk.Text)):
            return
        self.step(d)

    def opts(self):
        txt = self.var_l0.get().strip().replace(",", ".")
        l0 = None
        if txt:
            try:
                l0 = float(txt)
                if l0 <= 0:
                    raise ValueError
            except ValueError:
                raise ValueError("L0 должно быть положительным числом в мм (или пустым — тогда результат в пикселях).")
        return ex.Options(method=METHODS[self.var_method.get()], bright_marks=self.var_bright.get(),
                          neck=self.var_neck.get(), l0_mm=l0)

    # ------------------------------------------------------------ видео
    def open_video_dialog(self):
        path = filedialog.askopenfilename(title="Открыть видео испытания", filetypes=VIDEO_TYPES)
        if path:
            self.open_video(path)

    def open_video(self, path):
        if self.analysis_running():
            return
        if self.camera:
            self.toggle_camera()
        try:
            v = VideoFile(path)
        except RuntimeError as e:
            messagebox.showerror(APP_TITLE, str(e))
            return
        if self.video:
            self.video.close()
        self.video = v
        self.reset_analysis()
        self.slider.configure(to=max(1, v.n - 1))
        self.log(f"Открыто видео: {path} ({v.n} кадров, {v.fps:.1f} fps"
                 f"{', с метками времени' if v.ts is not None else ''})")
        self.root.title(f"{APP_TITLE} — {Path(path).name}")
        self.seek(0)
        self.set_status("Видео открыто. Перейдите к кадру до начала растяжения и нажмите «Указать метки на видео».")

    def reset_analysis(self):
        self.rois = None
        self.pick_pts = []
        self.pick_frame = None
        self.tr = self.res = None
        self.l2_manual = None
        self.out_dir = None
        self.merged = None
        self.lbl_pick.configure(text="не указаны", foreground="#a51d2d")
        for v in self.vals.values():
            v.set("—")
        for b in (self.btn_setl2, self.btn_autol2, self.btn_gol2, self.btn_folder):
            b.configure(state="disabled")
        self.fig.clear()
        self.fig_canvas.draw_idle()
        self.prog.configure(value=0)

    def seek(self, idx):
        if not self.video or self.analysis_running():
            return
        idx = int(max(0, min(self.video.n - 1, idx)))
        frame = self.video.read(idx)
        if frame is None:
            return
        self.cur_idx, self.cur_frame = idx, frame
        self._slider_lock = True
        self.slider.set(idx)
        self._slider_lock = False
        self.update_frame_label()
        self.render()
        self.update_cursor()

    def step(self, d):
        self.seek(self.cur_idx + d)

    def on_slider(self, val):
        if self._slider_lock or not self.video:
            return
        idx = int(round(float(val)))
        if idx != self.cur_idx:
            self.seek(idx)

    def update_frame_label(self):
        if not self.video:
            return
        t = self.video.time_of(self.cur_idx)
        txt = f"кадр {self.cur_idx} / {self.video.n - 1}   t = {t:.3f} с"
        if self.tr:
            i = self.tr.row_of_frame(self.cur_idx)
            if i is not None and np.isfinite(self.res["dL"][i]):
                txt += f"   ΔL = {self.res['dL'][i]:.3f} {self.res['unit']}"
        self.lbl_frame.configure(text=txt)

    # ------------------------------------------------------------ отрисовка
    def render(self, frame=None, row=None, banner=None):
        if frame is None:
            frame = self.cur_frame
        if frame is None:
            return
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 20 or ch < 20:
            return
        h, w = frame.shape[:2]
        s = min(cw / w, ch / h)
        dw, dh = max(1, int(w * s)), max(1, int(h * s))
        img = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        ox, oy = (cw - dw) // 2, (ch - dh) // 2
        self.view = (s, ox, oy)

        def P(p):
            return int(round(p[0] * s)), int(round(p[1] * s))

        if row is None and self.tr is not None and frame is self.cur_frame:
            i = self.tr.row_of_frame(self.cur_idx)
            row = self.tr.rows[i] if i is not None else None
            is_l2 = i is not None and i == self.res["l2"]
        else:
            is_l2 = False
        if row is not None:
            col = (0, 220, 0) if row["ok"] else (0, 0, 255)
            if is_l2:
                col = (0, 140, 255)
            cv2.line(img, P(row["p1"]), P(row["p2"]), col, 1, cv2.LINE_AA)
            for p in (row["p1"], row["p2"]):
                cv2.drawMarker(img, P(p), col, cv2.MARKER_CROSS, 26, 2, cv2.LINE_AA)
            if is_l2:
                cv2.putText(img, "L2", (P(row["p1"])[0] + 18, (P(row["p1"])[1] + P(row["p2"])[1]) // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2, cv2.LINE_AA)
        if (self.picking or self.rois) and self.pick_frame == self.cur_idx and frame is self.cur_frame and self.tr is None:
            for k, p in enumerate(self.pick_pts):
                cv2.drawMarker(img, P(p), (0, 255, 255), cv2.MARKER_CROSS, 26, 2, cv2.LINE_AA)
                cv2.putText(img, str(k + 1), (P(p)[0] + 12, P(p)[1] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 255), 2, cv2.LINE_AA)
            for x, y, rw, rh in (self.rois or []):
                cv2.rectangle(img, P((x, y)), P((x + rw, y + rh)), (0, 255, 255), 1)
        if banner:
            cv2.putText(img, banner, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(img, banner, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
        self._photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        self.canvas.delete("all")
        self.canvas.create_image(ox, oy, anchor="nw", image=self._photo)

    # ------------------------------------------------------------ камера
    def refresh_cameras(self):
        cams = list_cameras()
        self.cb_cam.configure(values=cams)
        cur = self.var_cam.get().strip()
        if not cur or cur.split(":", 1)[0].strip().isdigit():          # адрес потока не трогаем
            num = cur.split(":", 1)[0].strip() if cur else "0"
            self.var_cam.set(next((c for c in cams if c.split(":", 1)[0] == num), cams[0]))
        if hasattr(self, "txt_log"):
            self.log("Камеры: " + "; ".join(cams))

    def toggle_camera(self):
        if self.camera:
            if self.camera.recording:
                self.toggle_record()
            self.camera.close()
            self.camera = None
            self.btn_cam.configure(text="🎥 Включить камеру")
            self.btn_rec.configure(state="disabled")
            self.set_status("Камера выключена.")
            self.render()
            return
        if self.analysis_running():
            return
        try:
            w, h = (int(v) for v in self.var_res.get().split("x"))
            self.root.configure(cursor="watch")
            self.set_status("Подключение к камере…")
            self.root.update()
            src = self.var_cam.get().strip()
            if ":" in src and src.split(":", 1)[0].strip().isdigit():     # «1: Camo» -> 1
                src = src.split(":", 1)[0].strip()
            self.camera = Camera(src, w, h)
        except RuntimeError as e:
            messagebox.showerror(APP_TITLE, str(e))
            return
        finally:
            self.root.configure(cursor="")
        cw, chh = self.camera.size
        self.btn_cam.configure(text="⏹ Выключить камеру")
        self.btn_rec.configure(state="normal")
        self.log(f"Камера {self.var_cam.get()}: {cw}x{chh} @ {self.camera.fps:.0f} fps")
        self.set_status("Камера включена. Выставьте кадр и фокус, затем «Начать запись» — до старта машины.")

    def toggle_record(self):
        cam = self.camera
        if not cam:
            return
        if not cam.recording:
            rec_dir = app_dir() / "records"
            rec_dir.mkdir(exist_ok=True)
            path = filedialog.asksaveasfilename(
                title="Куда сохранить запись", initialdir=str(rec_dir),
                initialfile=f"test_{datetime.now():%Y%m%d_%H%M%S}.avi",
                defaultextension=".avi", filetypes=[("AVI", "*.avi")])
            if not path:
                return
            try:
                cam.start_record(path)
            except RuntimeError as e:
                messagebox.showerror(APP_TITLE, str(e))
                return
            self.btn_rec.configure(text="⏹ Остановить запись")
            self.log(f"Запись: {path}")
            self.set_status("Идёт запись. Запустите машину; остановите запись после разрыва образца.")
        else:
            path, n, dur = cam.stop_record()
            self.btn_rec.configure(text="⏺ Начать запись")
            self.log(f"Запись остановлена: {n} кадров за {dur:.1f} с ({n / max(dur, 1e-9):.1f} fps)")
            if path and n > 0:
                self.toggle_camera()
                self.open_video(path)

    # ------------------------------------------------------------ метки
    def start_picking(self):
        if self.camera:
            messagebox.showinfo(APP_TITLE, "Метки указываются на записанном видео. "
                                           "Остановите запись или откройте видеофайл.")
            return
        if not self.video:
            messagebox.showinfo(APP_TITLE, "Сначала откройте видео или сделайте запись.")
            return
        if self.analysis_running():
            return
        if self.tr is not None:
            if not messagebox.askyesno(APP_TITLE, "Текущие результаты будут сброшены. Указать метки заново?"):
                return
            self.reset_analysis()
        self.picking = True
        self.pick_pts = []
        self.rois = None
        self.pick_frame = self.cur_idx
        self.canvas.configure(cursor="crosshair")
        self.lbl_pick.configure(text="кликните по метке 1…", foreground="#c64600")
        self.set_status("Кликните по ВЕРХНЕЙ метке, затем по НИЖНЕЙ. Esc — отмена. "
                        "Анализ начнётся с этого кадра.")
        self.render()

    def cancel_picking(self):
        if self.picking:
            self.picking = False
            self.pick_pts = []
            self.canvas.configure(cursor="arrow")
            self.lbl_pick.configure(text="не указаны", foreground="#a51d2d")
            self.render()

    def on_canvas_click(self, e):
        if not self.picking or self.cur_frame is None:
            return
        if self.cur_idx != self.pick_frame:            # кадр сменился во время выбора — начинаем на новом
            self.pick_pts = []
            self.pick_frame = self.cur_idx
        s, ox, oy = self.view
        x, y = (e.x - ox) / s, (e.y - oy) / s
        h, w = self.cur_frame.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return
        self.pick_pts.append((x, y))
        if len(self.pick_pts) == 1:
            self.lbl_pick.configure(text="кликните по метке 2…")
        elif np.hypot(x - self.pick_pts[0][0], y - self.pick_pts[0][1]) < 15:
            self.pick_pts.pop()
            messagebox.showwarning(APP_TITLE, "Вторая метка почти совпадает с первой. "
                                              "Кликните по второй метке подальше от первой.")
        else:
            self.picking = False
            self.canvas.configure(cursor="arrow")
            gray = ex.to_gray(self.cur_frame)
            self.rois = [ex.roi_from_click(gray, p, not self.var_bright.get()) for p in self.pick_pts]
            self.lbl_pick.configure(text=f"указаны на кадре {self.pick_frame}", foreground="#26a269")
            self.log(f"Метки: ROI1={self.rois[0]}, ROI2={self.rois[1]} (кадр {self.pick_frame})")
            self.set_status("Метки указаны (жёлтые рамки должны охватывать точки). Нажмите «Запустить анализ».")
        self.render()

    # ------------------------------------------------------------ анализ
    def analysis_running(self):
        return self.analysis_thread is not None and self.analysis_thread.is_alive()

    def start_analysis(self):
        if self.analysis_running():
            return
        if not self.video:
            messagebox.showinfo(APP_TITLE, "Сначала откройте видео или сделайте запись.")
            return
        if not self.rois:
            messagebox.showinfo(APP_TITLE, "Сначала укажите метки: кнопка «Указать метки на видео», "
                                           "затем клик по каждой из двух меток.")
            return
        try:
            opt = self.opts()
        except ValueError as e:
            messagebox.showerror(APP_TITLE, str(e))
            return
        self.opt = opt
        self.stop_flag = False
        self.live = None
        self.tr = self.res = None
        self.l2_manual = None
        self.prog.configure(maximum=max(1, self.video.n - self.pick_frame), value=0)
        for b in (self.btn_run, self.btn_pick, self.btn_cam):
            b.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.slider.state(["disabled"])
        self.set_status("Идёт анализ…")
        self.log("Анализ запущен.")
        path, rois, start = self.video.path, list(self.rois), self.pick_frame

        def on_frame(row, frame):
            self.live = (row, frame)

        def work():
            try:
                t0 = time.perf_counter()
                tr = ex.track_video(path, rois[0], rois[1], opt, start_frame=start, on_frame=on_frame,
                                    should_stop=lambda: self.stop_flag, log=lambda m: self.q.put(("log", m)))
                self.q.put(("log", f"Обработано {len(tr.rows)} кадров за {time.perf_counter() - t0:.1f} с."))
                res = ex.summarize(tr, opt)
                out = ex.save_results(tr, res)
                self.q.put(("done", tr, res, out))
            except Exception as e:
                self.q.put(("error", e, traceback.format_exc()))

        self.analysis_thread = threading.Thread(target=work, daemon=True)
        self.analysis_thread.start()

    def stop_analysis(self):
        self.stop_flag = True

    def _analysis_finished(self):
        for b in (self.btn_run, self.btn_pick, self.btn_cam):
            b.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.slider.state(["!disabled"])
        self.live = None

    def on_done(self, tr, res, out):
        self.analysis_thread = None
        self._analysis_finished()
        self.tr, self.res, self.out_dir = tr, res, out
        for b in (self.btn_setl2, self.btn_autol2, self.btn_gol2, self.btn_folder):
            b.configure(state="normal")
        self.prog.configure(value=self.prog.cget("maximum"))
        r = tr.rows[res["l2"]]
        if res["brk"] is not None:
            self.log(f"Разрыв: {res['reason']}. Последний целый кадр: {r['frame']}.")
        else:
            self.log("Разрыв автоматически не найден — L2 = последний кадр с метками. Проверьте вручную.")
        self.log(f"Результаты сохранены: {out}")
        self.show_results()
        self.nb.select(0)
        self.goto_l2()
        self.set_status("Готово. Проверьте кадр L2 (←/→); при необходимости «Назначить L2 = текущий кадр».")

    def show_results(self):
        res, tr = self.res, self.tr
        k, u = res["k"], res["unit"]
        r = tr.rows[res["l2"]]
        self.vals["L0"].set(f"{res['L0'] * k:.3f} {u}")
        self.vals["L2"].set(f"{res['L2'] * k:.3f} {u}")
        self.vals["dL"].set(f"{(res['L2'] - res['L0']) * k:.3f} {u}")
        self.vals["eps"].set(f"{(res['L2'] - res['L0']) / res['L0'] * 100:.2f} %")
        self.vals["frame"].set(f"{r['frame']}  /  {r['t']:.3f} с")
        self.vals["neck"].set(f"{r['neck'] * k:.3f} {u}  (исх. {tr.neck_w0 * k:.3f})"
                              if tr.neck_w0 is not None and np.isfinite(r["neck"]) else "—")
        src = "задан вручную" if self.l2_manual is not None else res["reason"]
        self.vals["reason"].set(src)
        axes = ex.plot_results(self.fig, res)
        t = self.video.time_of(self.cur_idx) if self.video else None
        self.cursor_lines = [a.axvline(t if t is not None else 0, color="tab:green", lw=0.8) for a in axes]
        self.fig.tight_layout()
        self.fig_canvas.draw_idle()
        self.update_frame_label()

    def update_cursor(self):
        if self.cursor_lines and self.video:
            t = self.video.time_of(self.cur_idx)
            for ln in self.cursor_lines:
                ln.set_xdata([t, t])
            self.fig_canvas.draw_idle()

    def on_plot_click(self, e):
        if e.xdata is None or self.tr is None:
            return
        T = self.res["T"]
        i = int(np.argmin(np.abs(T - e.xdata)))
        self.seek(self.tr.rows[i]["frame"])

    def recalc(self, l2=None, save=True):
        if self.tr is None or self.analysis_running():
            return
        try:
            opt = self.opts()
        except ValueError as e:
            messagebox.showerror(APP_TITLE, str(e))
            return
        self.opt = opt
        self.res = ex.summarize(self.tr, opt, self.l2_manual)
        if save:
            self.out_dir = ex.save_results(self.tr, self.res)
        self.show_results()
        self.render()

    def set_l2_here(self):
        if self.tr is None:
            return
        i = self.tr.row_of_frame(self.cur_idx)
        if i is None or not np.isfinite(self.res["L"][i]):
            messagebox.showwarning(APP_TITLE, "На этом кадре нет данных трекинга (метки не найдены).")
            return
        self.l2_manual = i
        self.recalc()
        self.log(f"L2 назначен вручную: кадр {self.cur_idx}.")

    def auto_l2(self):
        self.l2_manual = None
        self.recalc()
        self.goto_l2()

    def goto_l2(self):
        if self.tr is not None:
            self.seek(self.tr.rows[self.res["l2"]]["frame"])

    def open_folder(self):
        if self.out_dir and Path(self.out_dir).exists():
            os.startfile(self.out_dir)

    # ------------------------------------------------------------ данные машины
    def load_machine(self):
        path = filedialog.askopenfilename(title="Данные машины (экспорт Trapezium)",
                                          filetypes=[("Таблицы", "*.csv *.txt *.tsv *.dat"), ("Все файлы", "*.*")])
        if not path:
            return
        try:
            names, data = ex.load_machine_table(path)
        except Exception as e:
            messagebox.showerror(APP_TITLE, f"Не удалось прочитать файл:\n{e}")
            return
        self.machine = (path, names, data)
        ti, fi = ex.guess_columns(names)
        self.cb_tcol.configure(values=names)
        self.cb_fcol.configure(values=names)
        self.var_tcol.set(names[ti])
        self.var_fcol.set(names[fi])
        self.lbl_mfile.configure(text=f"{Path(path).name}: {len(data)} строк, {len(names)} столбцов")
        self.log(f"Данные машины: {path}; столбцы: {names}")

    def build_merge(self):
        if self.res is None:
            messagebox.showinfo(APP_TITLE, "Сначала выполните анализ видео.")
            return
        if self.machine is None:
            messagebox.showinfo(APP_TITLE, "Загрузите файл с данными машины.")
            return
        path, names, data = self.machine
        ti, fi = names.index(self.var_tcol.get()), names.index(self.var_fcol.get())
        mode = ALIGN[self.var_align.get()]
        try:
            offset = float(self.var_offset.get().replace(",", "."))
        except ValueError:
            offset = 0.0
        s = self.res["summary"]
        if mode in ("start", "both") and s["t_motion_start_s"] is None:
            messagebox.showerror(APP_TITLE, "Начало движения на видео не определено — выберите совмещение по разрыву.")
            return
        valid = self.res["ok"] & (np.arange(len(self.res["L"])) <= self.res["l2"])
        try:
            mg = ex.align_merge(self.res["T"], self.res["dL"], self.res["strain"], valid,
                                s["t_motion_start_s"], s["t_L2_s"], data[:, ti], data[:, fi], mode, offset)
        except Exception as e:
            messagebox.showerror(APP_TITLE, f"Не удалось совместить данные:\n{e}")
            return
        if len(mg["t"]) < 2:
            messagebox.showerror(APP_TITLE, "После совмещения данные не перекрываются по времени. "
                                            "Попробуйте другой способ совмещения или сдвиг вручную.")
            return
        self.merged = mg
        unit = self.res["unit"]
        out = Path(self.out_dir) / f"{Path(self.tr.video).stem}_merged.csv"
        ex.save_merged(out, mg, unit, names[fi])
        self.fig2.clear()
        ax = self.fig2.add_subplot()
        ax.plot(mg["dL"], mg["F"], lw=1)
        ax.plot([mg["dL"][-1]], [mg["F"][-1]], "o", color="red", ms=5, label="L2")
        ax.set_xlabel(f"ΔL, {unit}")
        ax.set_ylabel(names[fi])
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
        self.fig2.tight_layout()
        self.fig2_canvas.draw_idle()
        self.fig2.savefig(out.with_suffix(".png"), dpi=120)
        self.lbl_merge.configure(text=f"t_машины = {mg['ka']:.5f}·t_видео {mg['kb']:+.3f} с.   "
                                      f"Fmax = {mg['Fmax']:.4g}.   Сохранено: {out.name}")
        self.log(f"Кривая F–ΔL: совмещение '{mode}', сдвиг {mg['kb']:+.3f} с; файл {out}")

    # ------------------------------------------------------------ цикл
    def _poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                if msg[0] == "log":
                    self.log(msg[1])
                elif msg[0] == "done":
                    self.on_done(*msg[1:])
                elif msg[0] == "error":
                    self.analysis_thread = None
                    self._analysis_finished()
                    self.log(msg[2])
                    self.set_status("Ошибка анализа — см. «Журнал».")
                    messagebox.showerror(APP_TITLE, f"Ошибка анализа: {msg[1]}")
        except queue.Empty:
            pass

        if self.camera:
            if self.camera.error:
                err = self.camera.error
                self.toggle_camera()
                messagebox.showerror(APP_TITLE, err)
            else:
                banner = f"REC {self.camera.rec_time():6.1f} s  ({self.camera.nrec} fr)" if self.camera.recording else "LIVE"
                self.render(self.camera.frame, banner=banner)
        elif self.live is not None:
            row, frame = self.live
            self.live = None
            self.render(frame, row=row, banner=f"ANALYSIS  frame {row['frame']}")
            self.prog.configure(value=row["frame"] - self.pick_frame)
            self.lbl_frame.configure(text=f"анализ: кадр {row['frame']} / {self.video.n - 1}")
        self.root.after(30, self._poll)

    def on_close(self):
        self.stop_flag = True
        if self.camera:
            if self.camera.recording:
                if not messagebox.askyesno(APP_TITLE, "Идёт запись. Остановить её и выйти?"):
                    return
            self.camera.stop_record()
            self.camera.close()
        if self.video:
            self.video.close()
        self.root.destroy()


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--list-cameras":     # диагностика: список камер в файл
        Path(sys.argv[2]).write_text("\n".join(list_cameras()), encoding="utf-8")
        return
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    root = tk.Tk()
    icon = Path(getattr(sys, "_MEIPASS", app_dir())) / "icon.ico"
    if icon.exists():
        try:
            root.iconbitmap(default=str(icon))
        except tk.TclError:
            pass
    app = App(root)
    if len(sys.argv) > 1 and Path(sys.argv[1]).exists():      # открыть видео, перетащенное на exe
        root.after(300, lambda: app.open_video(sys.argv[1]))
    root.mainloop()


if __name__ == "__main__":
    main()
