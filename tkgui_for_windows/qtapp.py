#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 Spiking Neural Network Simulator — Tkinter Edition
ПОЛНАЯ ВЕРСИЯ С ГОРИЗОНТАЛЬНЫМ СКРОЛЛОМ
• Рабочее редактирование ячеек
• Вертикальный + горизонтальный скролл во всех вкладках
• Адаптивный интерфейс
"""

import sys
import numpy as np
import json
import threading
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
from datetime import datetime

import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
matplotlib.use('TkAgg')
import matplotlib.patches as mpatches
import networkx as nx

from SpikingNN.Izh_net import Izhikevich_Network, types2params


# ============================================================
# 📊 НАДЁЖНАЯ РЕДАКТИРУЕМАЯ МАТРИЦА
# ============================================================
class EditableMatrix:
    def __init__(self, parent, rows=10, cols=10, on_change=None,
                 value_format="%.2f", inactive_bg='#f0f0f0', compact=False):
        self.parent = parent
        self.rows = rows
        self.cols = cols
        self.on_change = on_change
        self.value_format = value_format
        self.inactive_bg = inactive_bg
        self.compact = compact

        self.cell_width = 5 if compact else 7
        self.font_size = 8 if compact else 9

        self.data = np.zeros((rows, cols))
        self.mask = np.zeros((rows, cols), dtype=bool)
        self.cells = {}

        self.frame = ttk.Frame(parent)
        self._build()

    def _build(self):
        """Построить интерфейс матрицы со скроллами"""
        # Header
        header = ttk.Frame(self.frame)
        header.pack(fill=tk.X)
        ttk.Label(header, text="", width=3 if self.compact else 4).pack(side=tk.LEFT)
        for c in range(self.cols):
            ttk.Label(header, text=str(c), width=self.cell_width, anchor=tk.CENTER,
                     font=('Arial', self.font_size, 'bold')).pack(side=tk.LEFT)

        # Canvas + scrollbars
        canvas_frame = ttk.Frame(self.frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_frame, highlightthickness=0)
        self.v_scroll = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.h_scroll = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.v_scroll.set, xscrollcommand=self.h_scroll.set)

        self.content = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.content, anchor=tk.NW)

        self.content.bind("<Configure>", self._on_content_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel, add="+")

        self.v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._create_cells()

    def _on_content_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self.canvas.find_withtag("all")[0] if self.canvas.find_withtag("all") else None, 
                              width=max(event.width, self.content.winfo_reqwidth()))

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _create_cells(self):
        """✅ ИСПРАВЛЕНО: правильное замыкание через default args"""
        for widget in self.content.winfo_children():
            widget.destroy()
        self.cells = {}

        for r in range(self.rows):
            ttk.Label(self.content, text=str(r), width=3 if self.compact else 4,
                     anchor=tk.E, font=('Arial', self.font_size, 'bold')
                    ).grid(row=r, column=0, sticky=tk.E, padx=2, pady=1)

            for c in range(self.cols):
                lbl = tk.Label(self.content, text="", width=self.cell_width, anchor=tk.CENTER,
                              relief=tk.SOLID, borderwidth=1, bg=self.inactive_bg,
                              font=('Arial', self.font_size), cursor="hand2")
                lbl.grid(row=r, column=c + 1, padx=1, pady=1, sticky=tk.NSEW)
                
                # ✅ FIX: используем default args в lambda для захвата текущих row/col
                lbl.bind('<Button-1>', lambda e, row=r, col=c: self._on_click(row, col))
                lbl.bind('<Enter>', lambda e, row=r, col=c: self._on_hover(row, col, True))
                lbl.bind('<Leave>', lambda e, row=r, col=c: self._on_hover(row, col, False))
                self.cells[(r, c)] = lbl

    def _on_click(self, row, col):
        """Обработка клика — открыть диалог редактирования"""
        if not self.mask[row, col]:
            return

        dialog = tk.Toplevel(self.parent)
        dialog.title(f"✏️ [{col}→{row}]")
        dialog.geometry("220x120")
        dialog.transient(self.parent)
        dialog.grab_set()
        dialog.resizable(False, False)

        try:
            x = self.parent.winfo_x() + self.parent.winfo_width() // 2 - 110
            y = self.parent.winfo_y() + self.parent.winfo_height() // 2 - 60
            dialog.geometry(f"+{x}+{y}")
        except:
            pass

        ttk.Label(dialog, text="Новое значение:").pack(pady=(15, 5))

        entry = ttk.Entry(dialog, width=20, justify=tk.CENTER)
        entry.pack(pady=5)
        entry.insert(0, self.value_format % self.data[row, col])
        entry.select_range(0, tk.END)
        entry.focus_set()

        def on_ok():
            try:
                val = float(entry.get())
                self.data[row, col] = val
                self._update_cell(row, col)
                if self.on_change:
                    self.on_change(row, col, val)
                dialog.destroy()
            except ValueError:
                messagebox.showerror("Ошибка", "Введите число", parent=dialog)
                entry.focus_set()

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="✓ OK", command=on_ok).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="✕ Отмена", command=dialog.destroy).pack(side=tk.LEFT, padx=10)

        entry.bind('<Return>', lambda e: on_ok())
        entry.bind('<Escape>', lambda e: dialog.destroy())
        dialog.wait_window(dialog)

    def _on_hover(self, row, col, enter):
        if not self.mask[row, col]:
            return
        lbl = self.cells.get((row, col))
        if lbl:
            lbl.config(bg='#e3f2fd' if enter else '#ffffff')

    def _update_cell(self, row, col):
        lbl = self.cells.get((row, col))
        if lbl:
            if self.mask[row, col]:
                lbl.config(text=self.value_format % self.data[row, col], bg='#ffffff', fg='black')
            else:
                lbl.config(text="", bg=self.inactive_bg, fg='gray')

    def set_data(self, data, mask=None):
        self.data = np.array(data, dtype=float)
        if mask is not None:
            self.mask = np.array(mask, dtype=bool)
        else:
            self.mask = np.ones_like(self.data, dtype=bool)
        for (r, c), lbl in self.cells.items():
            self._update_cell(r, c)

    def get_data(self):
        return self.data.copy()

    def update_size(self, rows, cols):
        self.rows, self.cols = rows, cols
        self.data = np.zeros((rows, cols))
        self.mask = np.zeros((rows, cols), dtype=bool)
        self._create_cells()

    def pack(self, **kw):
        self.frame.pack(**kw)

    def grid(self, **kw):
        self.frame.grid(**kw)


# ============================================================
# 🧵 WORKER ДЛЯ СИМУЛЯЦИИ
# ============================================================
class SimulationWorker(threading.Thread):
    def __init__(self, network, sim_time, dt, input_vector, progress_q, result_q):
        super().__init__(daemon=True)
        self.network = network
        self.sim_time = sim_time
        self.dt = dt
        self.input_vector = input_vector
        self.progress_q = progress_q
        self.result_q = result_q
        self._stop_flag = False
        self._is_running = True

    def stop(self):
        self._stop_flag = True

    def is_running(self):
        return self._is_running

    def run(self):
        try:
            n_steps = int(self.sim_time / self.dt)
            V_hist, t_hist, spike_hist = [], [], []

            for step in range(n_steps):
                if self._stop_flag:
                    break
                self.network.step(dt=self.dt, Iapp=self.input_vector)
                t_hist.append(step * self.dt)
                V_hist.append(self.network.V_prev.copy())
                spike_hist.append(self.network.output.copy())
                self.progress_q.put(('progress', (step + 1) / n_steps * 100))

            if not self._stop_flag:
                self.result_q.put(('done', {
                    'time': np.array(t_hist, dtype=float),
                    'voltage': np.array(V_hist, dtype=float),
                    'spikes': np.array(spike_hist, dtype=float)
                }))
            else:
                self.result_q.put(('stopped', None))
        except Exception as e:
            self.result_q.put(('error', str(e)))
        finally:
            self._is_running = False


# ============================================================
# 🎲 DIALOG HELPER
# ============================================================
class _RandDialog:
    def __init__(self, parent, title, minv, maxv, unit):
        self.result = None
        dlg = tk.Toplevel(parent)
        dlg.title(f"🎲 {title}")
        dlg.geometry("320x200")
        dlg.resizable(False, False)
        dlg.transient(parent)
        dlg.grab_set()

        f = ttk.Frame(dlg, padding=15)
        f.pack(fill=tk.BOTH, expand=True)

        ttk.Label(f, text=f"Мин ({unit}):").grid(row=0, column=0, sticky=tk.W, pady=5)
        mn = ttk.Spinbox(f, from_=minv, to=maxv, width=12)
        mn.set(minv)
        mn.grid(row=0, column=1, padx=10)

        ttk.Label(f, text=f"Макс ({unit}):").grid(row=1, column=0, sticky=tk.W, pady=5)
        mx = ttk.Spinbox(f, from_=minv, to=maxv, width=12)
        mx.set(maxv)
        mx.grid(row=1, column=1, padx=10)

        ttk.Label(f, text="Распределение:").grid(row=2, column=0, sticky=tk.W, pady=5)
        dv = tk.StringVar(value='uniform')
        ttk.Combobox(f, textvariable=dv, values=['uniform', 'normal'], state='readonly', width=10).grid(row=2, column=1, padx=10)

        ttk.Label(f, text="Стд. откл.:").grid(row=3, column=0, sticky=tk.W, pady=5)
        sd = ttk.Spinbox(f, from_=0.1, to=100, width=12)
        sd.set(1.0)
        sd.grid(row=3, column=1, padx=10)

        def on_gen():
            try:
                mi, ma, s = float(mn.get()), float(mx.get()), float(sd.get())
                if mi >= ma:
                    raise ValueError()
                self.result = {'min': mi, 'max': ma, 'std': s, 'dist': dv.get()}
                dlg.destroy()
            except:
                messagebox.showerror("Ошибка", "Проверьте значения", parent=dlg)

        bf = ttk.Frame(dlg, padding=10)
        bf.pack(fill=tk.X)
        ttk.Button(bf, text="🎲 Сгенерировать", command=on_gen).pack(side=tk.LEFT, padx=5)
        ttk.Button(bf, text="✕ Отмена", command=dlg.destroy).pack(side=tk.LEFT, padx=5)

        dlg.wait_window(dlg)


# ============================================================
# 🎨 MAIN APPLICATION
# ============================================================
class SpikingNNApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🧠 Spiking Neural Network Simulator")
        self.root.geometry("1400x800")
        self.root.minsize(1000, 600)

        self.network = None
        self.sim_data = None
        self.simulation_thread = None
        self.input_vec = None
        self.log_messages = []
        self.progress_q = queue.Queue()
        self.result_q = queue.Queue()

        try:
            style = ttk.Style()
            style.theme_use('clam')
        except:
            pass

        self._setup_ui()
        self.log_msg("🚀 Готов к работе. Создайте сеть.")
        self.root.after(100, self._process_queues)

    def _setup_ui(self):
        self._create_menu()
        self._create_toolbar()
        self._create_paned()
        self._create_statusbar()

    def _create_menu(self):
        mb = tk.Menu(self.root)
        self.root.config(menu=mb)

        fm = tk.Menu(mb, tearoff=0)
        mb.add_cascade(label="📁 Файл", menu=fm)
        fm.add_command(label="💾 Сохранить сеть", command=self.save_network)
        fm.add_command(label="📂 Загрузить сеть", command=self.load_network)
        fm.add_separator()
        fm.add_command(label="📤 Экспорт", command=self.export)
        fm.add_separator()
        fm.add_command(label="❌ Выход", command=self.root.quit)

        sm = tk.Menu(mb, tearoff=0)
        mb.add_cascade(label="▶️ Симуляция", menu=sm)
        sm.add_command(label="▶️ Старт", command=self.start_sim)
        sm.add_command(label="⏹ Стоп", command=self.stop_sim)
        sm.add_command(label="🔄 Сброс", command=self.reset_sim)

        hm = tk.Menu(mb, tearoff=0)
        mb.add_cascade(label="❓ Справка", menu=hm)
        hm.add_command(label="ℹ️ О программе", command=self.show_about)

    def _create_toolbar(self):
        tb = ttk.Frame(self.root, padding=3)
        tb.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(tb, text="▶️", command=self.start_sim, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Button(tb, text="⏹", command=self.stop_sim, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Button(tb, text="🔄", command=self.reset_sim, width=5).pack(side=tk.LEFT, padx=2)

        ttk.Separator(tb, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill=tk.Y)
        self.status_lbl = ttk.Label(tb, text="Статус: Ожидание")
        self.status_lbl.pack(side=tk.LEFT)

    def _create_paned(self):
        self.paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(self.paned, padding=8)
        self.paned.add(left, weight=1)
        self._build_left_panel(left)

        right = ttk.Frame(self.paned, padding=8)
        self.paned.add(right, weight=3)
        self._build_right_panel(right)

    def _build_left_panel(self, parent):
        canvas = tk.Canvas(parent, highlightthickness=0)
        vscroll = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        content = tk.Frame(canvas)

        content.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=content, anchor=tk.NW)
        canvas.configure(yscrollcommand=vscroll.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-e.delta / 120), "units"), add="+")

        # Network config
        f = ttk.LabelFrame(content, text="🔗 Сеть", padding=8)
        f.pack(fill=tk.X, pady=4)

        ttk.Label(f, text="Нейронов:").grid(row=0, column=0, sticky=tk.W)
        self.n_spin = ttk.Spinbox(f, from_=1, to=100, width=8)
        self.n_spin.set(10)
        self.n_spin.grid(row=0, column=1, padx=5)

        ttk.Label(f, text="Тип:").grid(row=1, column=0, sticky=tk.W)
        self.ntype_var = tk.StringVar(value='RS')
        self.ntype_cb = ttk.Combobox(f, textvariable=self.ntype_var,
                                     values=['RS', 'IB', 'CH', 'FS', 'TC', 'RZ', 'LTS', 'Custom'],
                                     state='readonly', width=10)
        self.ntype_cb.grid(row=1, column=1, padx=5)
        self.ntype_cb.bind('<<ComboboxSelected>>', self._on_ntype_change)

        self.param_f = ttk.LabelFrame(f, text="Параметры", padding=5)
        self.param_f.grid(row=2, column=0, columnspan=2, sticky=tk.EW, pady=5)
        self.param_f.grid_remove()

        for i, (nm, var, fr, to, inc, df) in enumerate([
            ('a:', 'pa', 0, 1, 0.01, 0.02), ('b:', 'pb', 0, 1, 0.01, 0.2),
            ('c:', 'pc', -100, 0, 1, -65), ('d:', 'pd', 0, 20, 0.5, 8)
        ]):
            ttk.Label(self.param_f, text=nm).grid(row=i, column=0, sticky=tk.W)
            sp = ttk.Spinbox(self.param_f, from_=fr, to=to, increment=inc, width=8)
            sp.set(df)
            sp.grid(row=i, column=1, padx=5)
            setattr(self, var, sp)

        ttk.Button(f, text="🔨 Создать", command=self.create_network).grid(
            row=3, column=0, columnspan=2, pady=8, sticky=tk.EW)

        # Simulation
        f = ttk.LabelFrame(content, text="▶️ Симуляция", padding=8)
        f.pack(fill=tk.X, pady=4)
        ttk.Label(f, text="Время (мс):").grid(row=0, column=0, sticky=tk.W)
        self.t_spin = ttk.Spinbox(f, from_=10, to=10000, increment=10, width=8)
        self.t_spin.set(1000)
        self.t_spin.grid(row=0, column=1, padx=5)
        ttk.Label(f, text="Шаг:").grid(row=1, column=0, sticky=tk.W)
        self.dt_spin = ttk.Spinbox(f, from_=0.01, to=1.0, increment=0.01, width=8)
        self.dt_spin.set(0.1)
        self.dt_spin.grid(row=1, column=1, padx=5)

        # Connections
        f = ttk.LabelFrame(content, text="🔌 Связи", padding=8)
        f.pack(fill=tk.X, pady=4)
        ttk.Label(f, text="i/j/w:").grid(row=0, column=0, sticky=tk.W)
        self.ci = ttk.Spinbox(f, from_=0, to=99, width=4)
        self.ci.set(0)
        self.ci.grid(row=0, column=1, padx=2)
        self.cj = ttk.Spinbox(f, from_=0, to=99, width=4)
        self.cj.set(1)
        self.cj.grid(row=0, column=2, padx=2)
        self.cw = ttk.Spinbox(f, from_=-10, to=10, increment=0.1, width=6)
        self.cw.set(1.0)
        self.cw.grid(row=0, column=3, padx=2)
        ttk.Button(f, text="➕", command=self.add_conn).grid(row=1, column=0, pady=3)
        ttk.Button(f, text="🗑️", command=self.del_conn).grid(row=1, column=1, pady=3)

        # Stats
        f = ttk.LabelFrame(content, text="📊 Статистика", padding=8)
        f.pack(fill=tk.X, pady=4)
        self.stats_lbl = ttk.Label(f, text="Нейронов: 0\nСвязей: 0", justify=tk.LEFT)
        self.stats_lbl.pack()

    def _build_right_panel(self, parent):
        self.tabs = ttk.Notebook(parent)
        self.tabs.pack(fill=tk.BOTH, expand=True)

        self._tab_viz = ttk.Frame(self.tabs)
        self.tabs.add(self._tab_viz, text="📈 Визуализация")
        self._init_viz_tab()

        self._tab_graph = ttk.Frame(self.tabs)
        self.tabs.add(self._tab_graph, text="🕸️ Граф")
        self._init_graph_tab()

        self._tab_W = ttk.Frame(self.tabs)
        self.tabs.add(self._tab_W, text="📊 Веса")
        self._init_W_tab()

        self._tab_tau = ttk.Frame(self.tabs)
        self.tabs.add(self._tab_tau, text="⏱ τ")
        self._init_tau_tab()

        self._tab_log = ttk.Frame(self.tabs)
        self.tabs.add(self._tab_log, text="📝 Лог")
        self._init_log_tab()

        self._tab_params = ttk.Frame(self.tabs)
        self.tabs.add(self._tab_params, text="⚙️ Параметры")
        self._init_params_tab()

        self._tab_input = ttk.Frame(self.tabs)
        self.tabs.add(self._tab_input, text="⚡ Токи")
        self._init_input_tab()

        self.prog_f = ttk.Frame(parent)
        self.prog = ttk.Progressbar(self.prog_f, mode='determinate')
        self.prog.pack(fill=tk.X, padx=5)
        self.prog_f.pack(fill=tk.X, pady=(5, 0))
        self.prog_f.pack_forget()

    # ============================================================
    # 📜 УНИВЕРСАЛЬНАЯ ФУНКЦИЯ ДЛЯ СКРОЛЛИРУЕМЫХ ВКЛАДОК
    # ============================================================
    def _create_scrollable_frame(self, parent):
        """
        Создаёт канвас с вертикальным и горизонтальным скроллом.
        Возвращает: (canvas, content_frame)
        """
        canvas = tk.Canvas(parent, highlightthickness=0)
        
        v_scroll = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        h_scroll = ttk.Scrollbar(parent, orient=tk.HORIZONTAL, command=canvas.xview)
        
        canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        
        # ✅ ВАЖНО: используем tk.Frame, а не ttk.Frame!
        content = tk.Frame(canvas)
        
        def _on_frame_configure(event=None):
            """Обновляем область прокрутки при изменении размера контента"""
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        def _on_canvas_configure(event):
            """Подгоняем ширину контента под канвас"""
            canvas.itemconfig(canvas_window, width=max(event.width, content.winfo_reqwidth()))
        
        canvas.bind("<Configure>", _on_canvas_configure)
        content.bind("<Configure>", _on_frame_configure)
        
        canvas_window = canvas.create_window((0, 0), window=content, anchor=tk.NW)
        
        # Скролл мышью
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
        canvas.bind_all("<MouseWheel>", _on_mousewheel, add="+")
        canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"), add="+")
        canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"), add="+")
        
        # Упаковка
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        return canvas, content

    # ========== TAB: VISUALIZATION ==========
    def _init_viz_tab(self):
        canvas, content = self._create_scrollable_frame(self._tab_viz)

        self.fig_v = Figure(figsize=(10, 5), dpi=100)
        self.cv = FigureCanvasTkAgg(self.fig_v, content)
        self.cv.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.cv, content).pack(side=tk.TOP, fill=tk.X)

        bf = ttk.Frame(content)
        bf.pack(fill=tk.X, pady=5)
        ttk.Button(bf, text="💾 NPZ", command=self._export_npz).pack(side=tk.LEFT, padx=5)
        ttk.Button(bf, text="💾 CSV", command=self._export_csv).pack(side=tk.LEFT, padx=5)

    def _plot_voltage(self):
        if self.sim_data is None:
            return

        self.fig_v.clear()

        t = self.sim_data['time']
        V = np.array(self.sim_data['voltage'])
        sp = np.array(self.sim_data['spikes'])

        if V.ndim != 2:
            messagebox.showerror("Ошибка", f"Неверная форма данных: V.shape={V.shape}")
            return

        n_neurons = V.shape[1]
    
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        from mpl_toolkits.axes_grid1 import make_axes_locatable
    
        norm = mcolors.Normalize(vmin=0, vmax=n_neurons - 1)
        cmap_name = getattr(self, 'viz_cmap_var', None)
        cmap_name = cmap_name.get() if cmap_name else 'plasma'
        cmap = cm.get_cmap(cmap_name)
    
        # === ПОТЕНЦИАЛЫ ===
        ax1 = self.fig_v.add_subplot(211)
        lw = 0.3 if n_neurons > 100 else 0.5
        alpha = 0.5 if n_neurons > 100 else 0.8
    
        for i in range(n_neurons):
            ax1.plot(t, V[:, i], color=cmap(norm(i)), lw=lw, alpha=alpha)
    
        ax1.set_title(f'Потенциалы ({n_neurons} нейронов)', fontsize=10)
        ax1.set_ylabel('мВ', fontsize=9)
        ax1.grid(alpha=0.3)
        ax1.tick_params(labelbottom=False)
    
        # === СПАЙКИ ===
        ax2 = self.fig_v.add_subplot(212, sharex=ax1)
    
        for i in range(n_neurons):
            spike_mask = sp[:, i] > 0
            if np.any(spike_mask):
                spike_times = t[spike_mask]
                ax2.vlines(spike_times, i - 0.4, i + 0.4, 
                          color=cmap(norm(i)), lw=0.5, alpha=alpha)
    
        ax2.set_title('Спайковая активность', fontsize=10)
        ax2.set_xlabel('Время (мс)', fontsize=9)
        ax2.set_ylabel('Нейрон', fontsize=9)
        ax2.set_ylim(-0.5, n_neurons - 0.5)
        ax2.grid(alpha=0.3, axis='x')
    
        # === COLORBAR: делаем его справа от ax2, но растягиваем на оба графика ===
        if n_neurons > 5:
            # Создаём axes для colorbar справа от ax2
            divider = make_axes_locatable(ax2)
            cbar_ax = divider.append_axes("right", size="5%", pad=0.15)
        
            cbar = self.fig_v.colorbar(
            cm.ScalarMappable(norm=norm, cmap=cmap), 
                cax=cbar_ax, label='Нейрон'
            )
            cbar.ax.tick_params(labelsize=8)
            cbar.set_label('Нейрон', fontsize=9, rotation=270, labelpad=15)
        
            # Растягиваем colorbar вверх до середины ax1
            # Получаем позиции в нормализованных координатах фигуры
            pos2 = ax2.get_position()
            pos1 = ax1.get_position()
            cbar_pos = cbar_ax.get_position()
        
            # Новая высота: от низа ax2 до верха ax1
            new_height = pos1.y1 - pos2.y0
            cbar_ax.set_position([cbar_pos.x0, pos2.y0, cbar_pos.width, new_height])
    
        self.fig_v.tight_layout()
        self.cv.draw()    # ========== TAB: GRAPH ==========
    
    def _init_graph_tab(self):
        canvas, content = self._create_scrollable_frame(self._tab_graph)

        lf = ttk.Frame(content)
        lf.pack(fill=tk.X, pady=5)
        ttk.Label(lf, text="Layout:").pack(side=tk.LEFT, padx=5)
        self.layout_var = tk.StringVar(value='spring')
        ttk.Combobox(lf, textvariable=self.layout_var, values=['circular', 'spring', 'kamada_kawai', 'random'],
                    state='readonly', width=12).pack(side=tk.LEFT, padx=5)
        ttk.Button(lf, text="🔄 Обновить", command=self._update_graph).pack(side=tk.RIGHT, padx=5)

        self.fig_g = Figure(figsize=(10, 7), dpi=100)
        self.cg = FigureCanvasTkAgg(self.fig_g, content)
        self.cg.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.cg, content).pack(side=tk.TOP, fill=tk.X)

    def _update_graph(self):
        if not self.network:
            return
        self.fig_g.clear()
        try:
            G = nx.DiGraph()
            G.add_nodes_from(range(self.network.N))
            for i in range(self.network.N):
                for j in range(self.network.N):
                    if self.network.M[j, i] != 0:
                        G.add_edge(i, j, w=self.network.W[j, i],
                                col='#2E86AB' if self.network.W[j, i] >= 0 else '#E74C3C')

            layouts = {
                'circular': nx.circular_layout,
                'spring': lambda g: nx.spring_layout(g, seed=42),
                'kamada_kawai': nx.kamada_kawai_layout,
                'random': lambda g: nx.random_layout(g, seed=42)
            }
            pos = layouts.get(self.layout_var.get(), nx.spring_layout)(G)

            ax = self.fig_g.add_subplot(111)
            for i, j in G.edges():
                ax.annotate('', xy=pos[j], xytext=pos[i],
                          arrowprops=dict(arrowstyle='->', color=G.edges[i, j]['col'], lw=1.5))
                if G.edges[i, j]['w'] != 0:
                    mx, my = (pos[i][0] + pos[j][0]) / 2, (pos[i][1] + pos[j][1]) / 2
                    ax.text(mx, my, f"{G.edges[i, j]['w']:.1f}", fontsize=7,
                           bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))

            nx.draw_networkx_nodes(G, pos, ax=ax, node_color='#2E86AB', node_size=400, alpha=0.9)
            nx.draw_networkx_labels(G, pos, ax=ax, font_color='white', font_size=8, font_weight='bold')

            ax.set_title(f"🕸️ Сеть: {self.network.N} нейронов, {G.number_of_edges()} связей")
            ax.axis('off')
            ax.legend(handles=[mpatches.Patch(color='#2E86AB', label='🔵 Возб.'),
                            mpatches.Patch(color='#E74C3C', label='🔴 Торм.')], loc='upper right')
            self.fig_g.tight_layout()
            self.cg.draw()
        except Exception as e:
            self.log_msg(f"❌ Граф: {e}")

    # ========== TAB: WEIGHTS ==========
    def _init_W_tab(self):
        canvas, content = self._create_scrollable_frame(self._tab_W)

        # Controls
        cf = ttk.Frame(content)
        cf.pack(fill=tk.X, pady=2)
        ttk.Label(cf, text="Cmap:").pack(side=tk.LEFT, padx=5)
        self.cmap_var = tk.StringVar(value='RdBu_r')
        ttk.Combobox(cf, textvariable=self.cmap_var,
                    values=['RdBu_r', 'viridis', 'plasma', 'coolwarm', 'seismic'],
                    state='readonly', width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(cf, text="🔄", command=self._update_W_matrix).pack(side=tk.RIGHT, padx=5)

        # Тепловая карта
        viz_frame = ttk.Frame(content)
        viz_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.fig_W = Figure(figsize=(12, 7), dpi=90)
        self.cW = FigureCanvasTkAgg(self.fig_W, viz_frame)
        self.cW.get_tk_widget().pack(fill=tk.BOTH, expand=True, side=tk.TOP)

        # Таблица
        table_frame = ttk.LabelFrame(content, text="📋 Редактирование", padding=5)
        table_frame.pack(fill=tk.X, pady=(5, 0))

        self.mat_W = EditableMatrix(table_frame, on_change=self._on_W_change,
                                   value_format="%.2f", compact=True)
        self.mat_W.pack(fill=tk.X, expand=False)

        # Buttons
        bf = ttk.Frame(content)
        bf.pack(fill=tk.X, pady=2)
        ttk.Button(bf, text="✅ Применить", command=self._apply_W).pack(side=tk.LEFT, padx=3)
        ttk.Button(bf, text="🗑️ Сбросить", command=self._reset_W).pack(side=tk.LEFT, padx=3)
        ttk.Button(bf, text="🎲 Рандом", command=self._rand_W).pack(side=tk.LEFT, padx=3)

        # Quick edit
        qf = ttk.LabelFrame(content, text="🟡 Быстро", padding=3)
        qf.pack(fill=tk.X, pady=2)
        ttk.Label(qf, text="i:").grid(row=0, column=0)
        self.qi = ttk.Spinbox(qf, from_=0, to=99, width=4)
        self.qi.set(0)
        self.qi.grid(row=0, column=1, padx=2)
        ttk.Label(qf, text="j:").grid(row=0, column=2)
        self.qj = ttk.Spinbox(qf, from_=0, to=99, width=4)
        self.qj.set(0)
        self.qj.grid(row=0, column=3, padx=2)
        ttk.Label(qf, text="w:").grid(row=0, column=4)
        self.qw = ttk.Spinbox(qf, from_=-10, to=10, increment=0.1, width=6)
        self.qw.set(1.0)
        self.qw.grid(row=0, column=5, padx=2)
        ttk.Button(qf, text="✏️", command=self._quick_W).grid(row=0, column=6, padx=5)

    def _update_W_matrix(self):
        if not self.network:
            return
        N = self.network.N
        self.mat_W.update_size(N, N)
        self.mat_W.set_data(self.network.W, mask=(self.network.M != 0))
        self._plot_W_viz()

    def _on_W_change(self, r, c, v):
        if self.network and self.network.M[r, c] != 0:
            self.network.W[r, c] = v
            self._plot_W_viz()
            self.log_msg(f"✏️ W[{c}→{r}]={v:.2f}")

    def _plot_W_viz(self):
        if not self.network:
            return
        self.fig_W.clear()
        W = self.network.W.copy()
        W[self.network.M == 0] = np.nan
        ax = self.fig_W.add_subplot(111)
        im = ax.imshow(W, cmap=self.cmap_var.get(), aspect='auto', origin='lower', interpolation='nearest')
        ax.set_xlabel('i (пресин.)', fontsize=9)
        ax.set_ylabel('j (постсин.)', fontsize=9)
        ax.set_title('Матрица весов', fontsize=11, fontweight='bold')
        cbar = self.fig_W.colorbar(im, ax=ax, label='Вес', fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        ax.set_xticks(np.arange(-0.5, self.network.N, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.network.N, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle=':', linewidth=0.3, alpha=0.3)
        self.fig_W.tight_layout(pad=1.5)
        self.cW.draw()

    def _apply_W(self):
        if not self.network:
            return
        data = self.mat_W.get_data()
        for i in range(self.network.N):
            for j in range(self.network.N):
                if self.network.M[j, i] != 0:
                    self.network.W[j, i] = data[j, i]
        self.log_msg("✅ Веса применены")
        self._plot_W_viz()

    def _reset_W(self):
        if not self.network:
            return
        for i in range(self.network.N):
            for j in range(self.network.N):
                if self.network.M[j, i] != 0:
                    self.network.W[j, i] = np.sign(self.network.M[j, i])
        self._update_W_matrix()
        self.log_msg("🔄 Веса сброшены")

    def _rand_W(self):
        if not self.network:
            return
        dlg = _RandDialog(self.root, "Веса", -10, 10, "мкА")
        res = dlg.result
        if res:
            for i in range(self.network.N):
                for j in range(self.network.N):
                    if self.network.M[j, i] != 0:
                        if res['dist'] == 'uniform':
                            self.network.W[j, i] = np.random.uniform(res['min'], res['max'])
                        else:
                            self.network.W[j, i] = np.clip(np.random.normal((res['min'] + res['max']) / 2, res['std']), res['min'], res['max'])
            self._update_W_matrix()
            self.log_msg(f"🎲 Веса: [{res['min']},{res['max']}]")

    def _quick_W(self):
        if not self.network:
            return
        i, j, w = int(self.qi.get()), int(self.qj.get()), float(self.qw.get())
        if self.network.M[j, i] != 0:
            self.network.W[j, i] = w
            self._update_W_matrix()
            self.log_msg(f"✏️ Быстро: [{i}→{j}]={w}")

    # ========== TAB: TAU ==========
    def _init_tau_tab(self):
        canvas, content = self._create_scrollable_frame(self._tab_tau)

        cf = ttk.Frame(content)
        cf.pack(fill=tk.X, pady=2)
        ttk.Button(cf, text="🔄", command=self._update_tau_matrix).pack(side=tk.RIGHT, padx=5)

        viz_frame = ttk.Frame(content)
        viz_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.fig_tau = Figure(figsize=(12, 7), dpi=90)
        self.ctau = FigureCanvasTkAgg(self.fig_tau, viz_frame)
        self.ctau.get_tk_widget().pack(fill=tk.BOTH, expand=True, side=tk.TOP)

        table_frame = ttk.LabelFrame(content, text="📋 Редактирование τ", padding=5)
        table_frame.pack(fill=tk.X, pady=(5, 0))

        self.mat_tau = EditableMatrix(table_frame, on_change=self._on_tau_change,
                                     value_format="%.1f", compact=True)
        self.mat_tau.pack(fill=tk.X, expand=False)

        bf = ttk.Frame(content)
        bf.pack(fill=tk.X, pady=2)
        ttk.Button(bf, text="✅ Применить τ", command=self._apply_tau).pack(side=tk.LEFT, padx=3)
        ttk.Button(bf, text="🗑️ Сбросить τ", command=self._reset_tau).pack(side=tk.LEFT, padx=3)
        ttk.Button(bf, text="🎲 Рандом τ", command=self._rand_tau).pack(side=tk.LEFT, padx=3)

        qf = ttk.LabelFrame(content, text="🟡 Быстро τ", padding=3)
        qf.pack(fill=tk.X, pady=2)
        ttk.Label(qf, text="i:").grid(row=0, column=0)
        self.qti = ttk.Spinbox(qf, from_=0, to=99, width=4)
        self.qti.set(0)
        self.qti.grid(row=0, column=1, padx=2)
        ttk.Label(qf, text="j:").grid(row=0, column=2)
        self.qtj = ttk.Spinbox(qf, from_=0, to=99, width=4)
        self.qtj.set(0)
        self.qtj.grid(row=0, column=3, padx=2)
        ttk.Label(qf, text="τ:").grid(row=0, column=4)
        self.qtau = ttk.Spinbox(qf, from_=0.1, to=100, increment=0.5, width=6)
        self.qtau.set(10.0)
        self.qtau.grid(row=0, column=5, padx=2)
        ttk.Button(qf, text="✏️", command=self._quick_tau).grid(row=0, column=6, padx=5)

    def _update_tau_matrix(self):
        if not self.network:
            return
        N = self.network.N
        self.mat_tau.update_size(N, N)
        data = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if self.network.M[j, i] != 0:
                    data[j, i] = 1.0 / self.network.tau_syn[j, i]
        self.mat_tau.set_data(data, mask=(self.network.M != 0))
        self._plot_tau_viz()

    def _on_tau_change(self, r, c, v):
        if self.network and self.network.M[r, c] != 0 and v > 0:
            self.network.tau_syn[r, c] = 1.0 / v
            self._plot_tau_viz()
            self.log_msg(f"✏️ τ[{c}→{r}]={v:.1f} мс")

    def _plot_tau_viz(self):
        if not self.network:
            return
        self.fig_tau.clear()
        TAU = np.zeros_like(self.network.W)
        for i in range(self.network.N):
            for j in range(self.network.N):
                if self.network.M[j, i] != 0:
                    TAU[j, i] = 1.0 / self.network.tau_syn[j, i]
        ax = self.fig_tau.add_subplot(111)
        im = ax.imshow(TAU, cmap='Greens', aspect='auto', origin='lower', interpolation='nearest')
        ax.set_xlabel('i', fontsize=9)
        ax.set_ylabel('j', fontsize=9)
        ax.set_title('τ (мс)', fontsize=11, fontweight='bold')
        cbar = self.fig_tau.colorbar(im, ax=ax, label='τ (мс)', fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        self.fig_tau.tight_layout(pad=1.5)
        self.ctau.draw()

    def _apply_tau(self):
        if not self.network:
            return
        data = self.mat_tau.get_data()
        for i in range(self.network.N):
            for j in range(self.network.N):
                if self.network.M[j, i] != 0 and data[j, i] > 0:
                    self.network.tau_syn[j, i] = 1.0 / data[j, i]
        self.log_msg("✅ τ применены")
        self._plot_tau_viz()

    def _reset_tau(self):
        if not self.network:
            return
        for i in range(self.network.N):
            for j in range(self.network.N):
                if self.network.M[j, i] != 0:
                    self.network.tau_syn[j, i] = 1.0 / 10.0
        self._update_tau_matrix()
        self.log_msg("🔄 τ сброшены к 10 мс")

    def _rand_tau(self):
        if not self.network:
            return
        dlg = _RandDialog(self.root, "τ", 1, 50, "мс")
        res = dlg.result
        if res:
            for i in range(self.network.N):
                for j in range(self.network.N):
                    if self.network.M[j, i] != 0:
                        tau = np.random.uniform(res['min'], res['max']) if res['dist'] == 'uniform' else np.clip(np.random.normal((res['min'] + res['max']) / 2, res['std']), res['min'], res['max'])
                        self.network.tau_syn[j, i] = 1.0 / tau
            self._update_tau_matrix()
            self.log_msg(f"🎲 τ: [{res['min']},{res['max']}] мс")

    def _quick_tau(self):
        if not self.network:
            return
        i, j, tau = int(self.qti.get()), int(self.qtj.get()), float(self.qtau.get())
        if self.network.M[j, i] != 0 and tau > 0:
            self.network.tau_syn[j, i] = 1.0 / tau
            self._update_tau_matrix()
            self.log_msg(f"✏️ τ быстро: [{i}→{j}]={tau} мс")

    # ========== TAB: INPUT ==========
    def _init_input_tab(self):
        canvas, content = self._create_scrollable_frame(self._tab_input)

        self.fig_in = Figure(figsize=(10, 3), dpi=100)
        self.cin = FigureCanvasTkAgg(self.fig_in, content)
        self.cin.get_tk_widget().pack(fill=tk.X, pady=5)

        self.mat_in = EditableMatrix(content, rows=10, cols=1, on_change=self._on_in_change,
                                    value_format="%.2f", inactive_bg='#e8f5e9', compact=False)
        self.mat_in.pack(fill=tk.BOTH, expand=True, pady=5)

        bf = ttk.Frame(content)
        bf.pack(fill=tk.X, pady=5)
        ttk.Button(bf, text="✅ Применить", command=self._apply_inputs).pack(side=tk.LEFT, padx=5)
        ttk.Button(bf, text="🗑️ Сбросить", command=self._reset_inputs).pack(side=tk.LEFT, padx=5)
        ttk.Button(bf, text="🎲 Рандом", command=self._rand_inputs).pack(side=tk.LEFT, padx=5)

        self.in_stats = ttk.Label(content, text="Средний: 0 | Мин: 0 | Макс: 0 | Активных: 0")
        self.in_stats.pack(pady=5)

    def _update_input_matrix(self):
        if not self.network or self.input_vec is None:
            return
        N = self.network.N
        self.mat_in.update_size(N, 1)
        self.mat_in.set_data(self.input_vec.reshape(-1, 1), mask=np.ones((N, 1), dtype=bool))
        self._plot_inputs()
        self._update_in_stats()

    def _on_in_change(self, r, c, v):
        if self.input_vec is not None:
            self.input_vec[r] = v
            self._plot_inputs()
            self._update_in_stats()

    def _plot_inputs(self):
        if self.input_vec is None:
            return
        self.fig_in.clear()
        ax = self.fig_in.add_subplot(111)
        N = len(self.input_vec)
        cols = ['#2E86AB' if v > 0 else '#E74C3C' if v < 0 else '#888888' for v in self.input_vec]
        ax.bar(range(N), self.input_vec, color=cols)
        ax.set_xlabel('Нейрон')
        ax.set_ylabel('Ток (мкА)')
        ax.set_title('Входные токи')
        ax.grid(axis='y', alpha=0.3)
        self.fig_in.tight_layout()
        self.cin.draw()

    def _update_in_stats(self):
        if self.input_vec is None:
            return
        v = self.input_vec
        act = np.sum(np.abs(v) > 0.01)
        self.in_stats.config(text=f"Средний: {np.mean(v):.2f} | Мин: {np.min(v):.2f} | Макс: {np.max(v):.2f} | Активных: {int(act)}")

    def _apply_inputs(self):
        if not self.network:
            return
        self.input_vec = self.mat_in.get_data().flatten()
        self.log_msg("✅ Токи применены")
        self._plot_inputs()

    def _reset_inputs(self):
        if self.network:
            self.input_vec = np.zeros(self.network.N)
            self._update_input_matrix()
            self.log_msg("🔄 Токи сброшены")

    def _rand_inputs(self):
        if not self.network:
            return
        dlg = _RandDialog(self.root, "Токи", -100, 100, "мкА")
        res = dlg.result
        if res:
            if res['dist'] == 'uniform':
                self.input_vec = np.random.uniform(res['min'], res['max'], self.network.N)
            else:
                self.input_vec = np.clip(np.random.normal((res['min'] + res['max']) / 2, res['std'], self.network.N), res['min'], res['max'])
            self._update_input_matrix()
            self.log_msg(f"🎲 Токи: [{res['min']},{res['max']}] мкА")

    # ========== TAB: LOG ==========
    def _init_log_tab(self):
        self.log_txt = ScrolledText(self._tab_log, state=tk.DISABLED, font=('Consolas', 9))
        self.log_txt.pack(fill=tk.BOTH, expand=True, pady=5)
        ttk.Button(self._tab_log, text="🗑️ Очистить", command=self._clear_log).pack(pady=5)

    def _clear_log(self):
        self.log_txt.config(state=tk.NORMAL)
        self.log_txt.delete(1.0, tk.END)
        self.log_txt.config(state=tk.DISABLED)
        self.log_messages = []

    # ========== TAB: PARAMETERS ==========
    def _init_params_tab(self):
        canvas, content = self._create_scrollable_frame(self._tab_params)

        f = ttk.LabelFrame(content, text="Параметры Izhikevich", padding=8)
        f.pack(fill=tk.X, pady=5)
        self.pstats = {}
        for i, nm in enumerate(['a', 'b', 'c', 'd']):
            ttk.Label(f, text=f"{nm} (среднее):").grid(row=i, column=0, sticky=tk.W)
            lbl = ttk.Label(f, text="0.000")
            lbl.grid(row=i, column=1, sticky=tk.W, padx=10)
            self.pstats[nm] = lbl

        cf = ttk.LabelFrame(content, text="Список связей", padding=8)
        cf.pack(fill=tk.BOTH, expand=True, pady=5)
        self.conn_txt = ScrolledText(cf, state=tk.DISABLED, height=10)
        self.conn_txt.pack(fill=tk.BOTH, expand=True)

    # ========== CORE METHODS ==========
    def _process_queues(self):
        try:
            while True:
                t, d = self.progress_q.get_nowait()
                if t == 'progress':
                    self.prog['value'] = d
                    self.root.update_idletasks()
        except queue.Empty:
            pass

        try:
            while True:
                t, d = self.result_q.get_nowait()
                if t == 'done':
                    self._on_sim_done(d)
                elif t == 'error':
                    messagebox.showerror("Ошибка", d)
                    self.log_msg(f"❌ {d}")
                elif t == 'stopped':
                    self.simulation_thread = None
                    self.prog_f.pack_forget()
                    self.log_msg("⏹ Остановлено")
        except queue.Empty:
            pass

        self.root.after(100, self._process_queues)

    def _on_ntype_change(self, e=None):
        if self.ntype_var.get() == 'Custom':
            self.param_f.grid()
        else:
            self.param_f.grid_remove()

    def create_network(self):
        try:
            N = int(self.n_spin.get())
            nt = self.ntype_var.get()
            if nt == 'Custom':
                a = np.ones(N) * float(self.pa.get())
                b = np.ones(N) * float(self.pb.get())
                c = np.ones(N) * float(self.pc.get())
                d = np.ones(N) * float(self.pd.get())
                self.network = Izhikevich_Network(N=N, a=a, b=b, c=c, d=d)
            else:
                a, b, c, d = types2params([nt] * N)
                self.network = Izhikevich_Network(N=N, a=a, b=b, c=c, d=d)

            self.network.set_init_conditions(v_noise=np.random.normal(size=N, scale=0.5))
            self.input_vec = np.zeros(N)

            self.ci.config(to=N - 1)
            self.cj.config(to=N - 1)
            self.qi.config(to=N - 1)
            self.qj.config(to=N - 1)
            self.qti.config(to=N - 1)
            self.qtj.config(to=N - 1)

            self._update_W_matrix()
            self._update_tau_matrix()
            self._update_input_matrix()
            self._update_stats()
            self._update_graph()
            self.log_msg(f"✅ Сеть: {N} нейронов, {nt}")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def add_conn(self):
        if not self.network:
            return
        try:
            i, j, w = int(self.ci.get()), int(self.cj.get()), float(self.cw.get())
            coef = 1 if w >= 0 else -1
            self.network.connect(i, j, coef=coef, w=abs(w))
            self._update_W_matrix()
            self._update_tau_matrix()
            self._update_stats()
            self.log_msg(f"🔗 {i}→{j} w={w}")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def del_conn(self):
        if not self.network:
            return
        try:
            i, j = int(self.ci.get()), int(self.cj.get())
            self.network.M[j, i] = self.network.W[j, i] = self.network.tau_syn[j, i] = 0
            self._update_W_matrix()
            self._update_tau_matrix()
            self._update_stats()
            self.log_msg(f"🗑️ Удалено: {i}→{j}")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def start_sim(self):
        if not self.network:
            messagebox.showwarning("Предупреждение", "Создайте сеть!")
            return

        if self.simulation_thread is not None and self.simulation_thread.is_running():
            return

        try:
            T, dt = float(self.t_spin.get()), float(self.dt_spin.get())
            self.prog_f.pack(fill=tk.X, pady=(5, 0))
            self.prog['value'] = 0

            self.simulation_thread = SimulationWorker(
                self.network, T, dt, self.input_vec, self.progress_q, self.result_q
            )
            self.simulation_thread.start()
            self.log_msg("▶️ Старт")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def stop_sim(self):
        if self.simulation_thread is not None:
            self.simulation_thread.stop()
            self.log_msg("⏹ Стоп")

    def reset_sim(self):
        self.stop_sim()
        if self.network:
            self.network.set_init_conditions(v_noise=np.random.normal(size=self.network.N, scale=0.5))
        self.sim_data = None
        self.prog_f.pack_forget()
        self._plot_voltage()
        self.log_msg("🔄 Сброс")

    def _on_sim_done(self, data):
        self.sim_data = data
        self.prog_f.pack_forget()
        self.simulation_thread = None
        self._plot_voltage()
        self._update_stats()
        self.log_msg(f"✅ Завершено: {len(data['time'])} шагов")

    def _update_stats(self):
        if not self.network:
            return
        nc = np.count_nonzero(self.network.M)
        ns = int(np.sum(self.sim_data['spikes'] > 0)) if self.sim_data is not None else 0
        self.stats_lbl.config(text=f"Нейронов: {self.network.N}\nСвязей: {nc}\nСпайков: {ns}")

        for nm, arr in [('a', self.network.a), ('b', self.network.b), ('c', self.network.c), ('d', self.network.d)]:
            self.pstats[nm].config(text=f"{np.mean(arr):.3f}" if nm in ['a', 'b'] else f"{np.mean(arr):.1f}")

        conns = []
        for i in range(self.network.N):
            for j in range(self.network.N):
                if self.network.M[j, i] != 0:
                    t = 'возб.' if self.network.W[j, i] > 0 else 'торм.'
                    conns.append(f"{i}→{j} (w:{self.network.W[j, i]:.2f}, {t})")

        self.conn_txt.config(state=tk.NORMAL)
        self.conn_txt.delete(1.0, tk.END)
        self.conn_txt.insert(tk.END, "\n".join(conns) if conns else "Нет связей")
        self.conn_txt.config(state=tk.DISABLED)

    def log_msg(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        entry = f"[{ts}] {msg}\n"
        self.log_messages.append(entry)
        if len(self.log_messages) > 200:
            self.log_messages.pop(0)
        self.log_txt.config(state=tk.NORMAL)
        self.log_txt.insert(tk.END, entry)
        self.log_txt.see(tk.END)
        self.log_txt.config(state=tk.DISABLED)

    def _status(self, msg):
        self.status_lbl.config(text=f"Статус: {msg}")

    # ========== FILE OPS ==========
    def save_network(self):
        if not self.network:
            return
        fp = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if fp:
            try:
                cfg = {'N': self.network.N, 'a': self.network.a.tolist(), 'b': self.network.b.tolist(),
                       'c': self.network.c.tolist(), 'd': self.network.d.tolist(),
                       'M': self.network.M.tolist(), 'W': self.network.W.tolist(),
                       'tau_syn': self.network.tau_syn.tolist()}
                with open(fp, 'w', encoding='utf-8') as f:
                    json.dump(cfg, f, indent=2, ensure_ascii=False)
                self.log_msg(f"💾 Сохранено: {fp}")
            except Exception as e:
                messagebox.showerror("Ошибка", str(e))

    def load_network(self):
        fp = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if fp:
            try:
                with open(fp, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
                self.network = Izhikevich_Network(N=cfg['N'])
                for k in ['a', 'b', 'c', 'd', 'M', 'W']:
                    setattr(self.network, k, np.array(cfg[k]))
                self.network.tau_syn = np.array(cfg.get('tau_syn', np.ones((cfg['N'], cfg['N'])) / 10.0))
                self.network.set_init_conditions(v_noise=np.random.normal(size=self.network.N, scale=0.5))
                self.input_vec = np.zeros(self.network.N)
                self.sim_data = None
                self.n_spin.set(cfg['N'])
                self._update_W_matrix()
                self._update_tau_matrix()
                self._update_input_matrix()
                self._update_stats()
                self._update_graph()
                self.log_msg(f"📂 Загружено: {fp}")
            except Exception as e:
                messagebox.showerror("Ошибка", str(e))

    def export(self):
        if self.sim_data is None:
            return
        fp = filedialog.asksaveasfilename(defaultextension=".npz", filetypes=[("NPZ", "*.npz"), ("CSV", "*.csv")])
        if fp:
            try:
                d = self.sim_data
                if fp.endswith('.npz'):
                    np.savez(fp, time=d['time'], voltage=d['voltage'], spikes=d['spikes'])
                else:
                    csv = np.column_stack([d['time'], d['voltage']])
                    hdr = "time," + ",".join(f"V_{i}" for i in range(d['voltage'].shape[1]))
                    np.savetxt(fp, csv, delimiter=',', header=hdr, comments='')
                self.log_msg(f"📤 Экспорт: {fp}")
            except Exception as e:
                messagebox.showerror("Ошибка", str(e))

    def _export_npz(self):
        self.export()

    def _export_csv(self):
        self.export()

    def show_about(self):
        messagebox.showinfo("О программе", "🧠 Spiking Neural Network Simulator\n\n"
            "Модель Ижикевича с интерактивной визуализацией (Tkinter)\n\n"
            "Типы: RS, IB, CH, FS, TC, RZ, LTS, Custom\n\n"
            "© 2026")

    def _create_statusbar(self):
        self.statusbar = ttk.Label(self.root, text="Готов", relief=tk.SUNKEN, anchor=tk.W)
        self.statusbar.pack(side=tk.BOTTOM, fill=tk.X)


# ============================================================
# 🚀 MAIN
# ============================================================
def main():
    root = tk.Tk()
    app = SpikingNNApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()