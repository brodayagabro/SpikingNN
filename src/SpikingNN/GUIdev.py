#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spiking Neural Network Simulator - Qt5 GUI
Модель Izhikevich с интерактивной визуализацией
"""

import sys
import numpy as np
import json
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QGroupBox, QLabel, QPushButton, QSpinBox, QDoubleSpinBox,
    QComboBox, QTextEdit, QFileDialog, QMessageBox, QProgressBar,
    QStatusBar, QMenu, QAction, QMenuBar, QSplitter, QScrollArea,
    QTableWidget, QTableWidgetItem, QHeaderView, QDialog, QFormLayout,
    QDialogButtonBox, QCheckBox, QToolBar, QSystemTrayIcon
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt5.QtGui import QFont, QIcon, QColor, QPalette

# Matplotlib integration
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# Import from Izh_net
from Izh_net import (
    Izhikevich_Network,
    Izhikevich_IO_Network,
    Network,
    NameNetwork,
    types2params,
    izhikevich_neuron
)


# ============================================================
# WORKER THREAD FOR SIMULATION
# ============================================================
class SimulationWorker(QThread):
    """Worker thread for running simulation without blocking GUI"""
    progress_signal = pyqtSignal(float)
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)
    
    def __init__(self, network, sim_time, dt, input_vector):
        super().__init__()
        self.network = network
        self.sim_time = sim_time
        self.dt = dt
        self.input_vector = input_vector
        self._stop_flag = False
    
    def stop(self):
        self._stop_flag = True
    
    def run(self):
        try:
            n_steps = int(self.sim_time / self.dt)
            V_history = []
            time_history = []
            spike_history = []
            
            for step in range(n_steps):
                if self._stop_flag:
                    break
                
                self.network.step(dt=self.dt, Iapp=self.input_vector)
                time_history.append(step * self.dt)
                V_history.append(self.network.V_prev.copy())
                spike_history.append(self.network.output.copy())
                
                # Update progress
                progress = (step + 1) / n_steps * 100
                self.progress_signal.emit(progress)
            
            if not self._stop_flag:
                self.finished_signal.emit({
                    'time': np.array(time_history),
                    'voltage': np.array(V_history),
                    'spikes': np.array(spike_history)
                })
        except Exception as e:
            self.error_signal.emit(str(e))


# ============================================================
# MAIN WINDOW
# ============================================================
class SpikingNNMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initialize state
        self.network = None
        self.simulation_data = None
        self.simulation_worker = None
        self.input_current_vector = None
        self.log_messages = []
        
        # Setup UI
        self.setWindowTitle("🧠 Spiking Neural Network Simulator")
        self.setGeometry(100, 100, 1600, 1000)
        
        self.create_menu_bar()
        self.create_toolbar()
        self.create_main_layout()
        self.create_status_bar()
        
        self.log_message("Готов к работе. Создайте сеть для начала.")
    
    def create_menu_bar(self):
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("📁 Файл")
        
        save_action = QAction("💾 Сохранить сеть", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_network)
        file_menu.addAction(save_action)
        
        load_action = QAction("📂 Загрузить сеть", self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self.load_network)
        file_menu.addAction(load_action)
        
        file_menu.addSeparator()
        
        export_action = QAction("📤 Экспорт результатов", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self.export_results)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("❌ Выход", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Simulation menu
        sim_menu = menubar.addMenu("▶️ Симуляция")
        
        start_action = QAction("▶️ Старт", self)
        start_action.setShortcut("F5")
        start_action.triggered.connect(self.start_simulation)
        sim_menu.addAction(start_action)
        
        stop_action = QAction("⏹ Стоп", self)
        stop_action.setShortcut("F6")
        stop_action.triggered.connect(self.stop_simulation)
        sim_menu.addAction(stop_action)
        
        reset_action = QAction("🔄 Сброс", self)
        reset_action.setShortcut("F7")
        reset_action.triggered.connect(self.reset_simulation)
        sim_menu.addAction(reset_action)
        
        # Help menu
        help_menu = menubar.addMenu("❓ Справка")
        
        about_action = QAction("ℹ️ О программе", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def create_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        # Add actions
        start_btn = QPushButton("▶️ Старт")
        start_btn.clicked.connect(self.start_simulation)
        toolbar.addWidget(start_btn)
        
        stop_btn = QPushButton("⏹ Стоп")
        stop_btn.clicked.connect(self.stop_simulation)
        toolbar.addWidget(stop_btn)
        
        reset_btn = QPushButton("🔄 Сброс")
        reset_btn.clicked.connect(self.reset_simulation)
        toolbar.addWidget(reset_btn)
        
        toolbar.addSeparator()
        
        self.status_label = QLabel("Статус: Ожидание")
        toolbar.addWidget(self.status_label)
    
    def create_main_layout(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel (controls)
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # Center panel (visualization)
        center_panel = self.create_center_panel()
        splitter.addWidget(center_panel)
        
        # Set splitter sizes (30% left, 70% center)
        splitter.setSizes([400, 1200])
        
        main_layout.addWidget(splitter)
    
    def create_left_panel(self):
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(10)
        
        # Scroll area for left panel
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        
        # Network configuration
        network_group = QGroupBox("🔗 Конфигурация сети")
        network_layout = QFormLayout()
        
        self.n_neurons_spin = QSpinBox()
        self.n_neurons_spin.setRange(1, 100)
        self.n_neurons_spin.setValue(10)
        network_layout.addRow("Количество нейронов:", self.n_neurons_spin)
        
        self.neuron_type_combo = QComboBox()
        self.neuron_type_combo.addItems(['RS', 'IB', 'CH', 'FS', 'TC', 'RZ', 'LTS', 'Custom'])
        network_layout.addRow("Тип нейронов:", self.neuron_type_combo)
        self.neuron_type_combo.currentTextChanged.connect(self.on_neuron_type_changed)
        
        # Izhikevich parameters (hidden by default)
        self.params_group = QGroupBox("Параметры Izhikevich")
        self.params_group.setVisible(False)
        params_layout = QFormLayout()
        
        self.param_a = QDoubleSpinBox()
        self.param_a.setRange(0, 1)
        self.param_a.setValue(0.02)
        self.param_a.setSingleStep(0.01)
        params_layout.addRow("a:", self.param_a)
        
        self.param_b = QDoubleSpinBox()
        self.param_b.setRange(0, 1)
        self.param_b.setValue(0.2)
        self.param_b.setSingleStep(0.01)
        params_layout.addRow("b:", self.param_b)
        
        self.param_c = QDoubleSpinBox()
        self.param_c.setRange(-100, 0)
        self.param_c.setValue(-65)
        self.param_c.setSingleStep(1)
        params_layout.addRow("c:", self.param_c)
        
        self.param_d = QDoubleSpinBox()
        self.param_d.setRange(0, 20)
        self.param_d.setValue(8)
        self.param_d.setSingleStep(0.5)
        params_layout.addRow("d:", self.param_d)
        
        self.params_group.setLayout(params_layout)
        network_layout.addRow(self.params_group)
        
        create_btn = QPushButton("🔨 Создать сеть")
        create_btn.clicked.connect(self.create_network)
        network_layout.addRow(create_btn)
        
        network_group.setLayout(network_layout)
        scroll_layout.addWidget(network_group)
        
        # Simulation parameters
        sim_group = QGroupBox("▶️ Параметры симуляции")
        sim_layout = QFormLayout()
        
        self.sim_time_spin = QDoubleSpinBox()
        self.sim_time_spin.setRange(10, 10000)
        self.sim_time_spin.setValue(1000)
        self.sim_time_spin.setSuffix(" мс")
        sim_layout.addRow("Время:", self.sim_time_spin)
        
        self.dt_spin = QDoubleSpinBox()
        self.dt_spin.setRange(0.01, 1.0)
        self.dt_spin.setValue(0.1)
        self.dt_spin.setSuffix(" мс")
        sim_layout.addRow("Шаг dt:", self.dt_spin)
        
        sim_group.setLayout(sim_layout)
        scroll_layout.addWidget(sim_group)
        
        # Connections
        conn_group = QGroupBox("🔌 Связи")
        conn_layout = QFormLayout()
        
        self.conn_i_spin = QSpinBox()
        self.conn_i_spin.setRange(0, 99)
        self.conn_i_spin.setValue(0)
        conn_layout.addRow("От (i):", self.conn_i_spin)
        
        self.conn_j_spin = QSpinBox()
        self.conn_j_spin.setRange(0, 99)
        self.conn_j_spin.setValue(1)
        conn_layout.addRow("До (j):", self.conn_j_spin)
        
        self.conn_w_spin = QDoubleSpinBox()
        self.conn_w_spin.setRange(-10, 10)
        self.conn_w_spin.setValue(1.0)
        self.conn_w_spin.setSingleStep(0.1)
        conn_layout.addRow("Вес:", self.conn_w_spin)
        
        add_conn_btn = QPushButton("➕ Добавить связь")
        add_conn_btn.clicked.connect(self.add_connection)
        conn_layout.addRow(add_conn_btn)
        
        del_conn_btn = QPushButton("🗑️ Удалить связь")
        del_conn_btn.clicked.connect(self.delete_connection)
        conn_layout.addRow(del_conn_btn)
        
        conn_group.setLayout(conn_layout)
        scroll_layout.addWidget(conn_group)
        
        # Input currents
        input_group = QGroupBox("⚡ Входные токи")
        input_layout = QVBoxLayout()
        
        self.input_table = QTableWidget()
        self.input_table.setColumnCount(1)
        self.input_table.setHorizontalHeaderLabels(["Ток (мкА)"])
        self.input_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        input_layout.addWidget(self.input_table)
        
        apply_input_btn = QPushButton("✅ Применить токи")
        apply_input_btn.clicked.connect(self.apply_input_currents)
        input_layout.addWidget(apply_input_btn)
        
        input_group.setLayout(input_layout)
        scroll_layout.addWidget(input_group)
        
        # Statistics
        stats_group = QGroupBox("📊 Статистика")
        stats_layout = QVBoxLayout()
        
        self.stats_label = QLabel("Нейронов: 0\nСвязей: 0\nСпайков: 0")
        self.stats_label.setWordWrap(True)
        stats_layout.addWidget(self.stats_label)
        
        stats_group.setLayout(stats_layout)
        scroll_layout.addWidget(stats_group)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        left_layout.addWidget(scroll)
        
        return left_widget
    
    def create_center_panel(self):
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        
        # Create tab widget
        self.tabs = QTabWidget()
        
        # Tab 1: Visualization
        self.tab_visualization = self.create_visualization_tab()
        self.tabs.addTab(self.tab_visualization, "📈 Визуализация")
        
        # Tab 2: Network Graph
        self.tab_graph = self.create_network_graph_tab()
        self.tabs.addTab(self.tab_graph, "🕸️ Граф сети")
        
        # Tab 3: Weight Matrix
        self.tab_matrix = self.create_weight_matrix_tab()
        self.tabs.addTab(self.tab_matrix, "📊 Матрица весов")
        
        # Tab 4: Log
        self.tab_log = self.create_log_tab()
        self.tabs.addTab(self.tab_log, "📝 Лог событий")
        
        # Tab 5: Parameters
        self.tab_params = self.create_parameters_tab()
        self.tabs.addTab(self.tab_params, "⚙️ Параметры")
        
        # Tab 6: Input Currents
        self.tab_inputs = self.create_inputs_tab()
        self.tabs.addTab(self.tab_inputs, "⚡ Входные токи")
        
        center_layout.addWidget(self.tabs)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        center_layout.addWidget(self.progress_bar)
        
        return center_widget
    
    def create_visualization_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Matplotlib figure
        self.fig_voltage = Figure(figsize=(10, 6))
        self.canvas_voltage = FigureCanvas(self.fig_voltage)
        self.toolbar_voltage = NavigationToolbar(self.canvas_voltage, self)
        
        layout.addWidget(self.toolbar_voltage)
        layout.addWidget(self.canvas_voltage)
        
        # Export buttons
        btn_layout = QHBoxLayout()
        
        export_npz_btn = QPushButton("💾 NPZ")
        export_npz_btn.clicked.connect(self.export_npz)
        btn_layout.addWidget(export_npz_btn)
        
        export_csv_btn = QPushButton("💾 CSV")
        export_csv_btn.clicked.connect(self.export_csv)
        btn_layout.addWidget(export_csv_btn)
        
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        return widget
    
    def create_network_graph_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Matplotlib figure for network graph
        self.fig_network = Figure(figsize=(10, 8))
        self.canvas_network = FigureCanvas(self.fig_network)
        self.toolbar_network = NavigationToolbar(self.canvas_network, self)
        
        layout.addWidget(self.toolbar_network)
        layout.addWidget(self.canvas_network)
        
        # Layout selector
        layout_selector = QHBoxLayout()
        layout_selector.addWidget(QLabel("Расположение:"))
        
        self.layout_combo = QComboBox()
        self.layout_combo.addItems(['circular', 'spring', 'kamada_kawai', 'random'])
        self.layout_combo.currentTextChanged.connect(self.update_network_graph)
        layout_selector.addWidget(self.layout_combo)
        
        layout_selector.addStretch()
        layout.addLayout(layout_selector)
        
        return widget
    
    def create_weight_matrix_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Weight matrix table
        self.weight_table = QTableWidget()
        self.weight_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.weight_table)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        apply_weights_btn = QPushButton("✅ Применить веса")
        apply_weights_btn.clicked.connect(self.apply_weights)
        btn_layout.addWidget(apply_weights_btn)
        
        reset_weights_btn = QPushButton("🗑️ Сбросить веса")
        reset_weights_btn.clicked.connect(self.reset_weights)
        btn_layout.addWidget(reset_weights_btn)
        
        random_weights_btn = QPushButton("🎲 Рандомные веса")
        random_weights_btn.clicked.connect(self.randomize_weights)
        btn_layout.addWidget(random_weights_btn)
        
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        # Tau matrix
        tau_label = QLabel("Матрица констант релаксации (τ):")
        layout.addWidget(tau_label)
        
        self.tau_table = QTableWidget()
        self.tau_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.tau_table)
        
        return widget
    
    def create_log_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
        
        clear_log_btn = QPushButton("🗑️ Очистить лог")
        clear_log_btn.clicked.connect(self.clear_log)
        layout.addWidget(clear_log_btn)
        
        return widget
    
    def create_parameters_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Izhikevich parameters
        params_group = QGroupBox("Параметры Izhikevich")
        params_layout = QFormLayout()
        
        self.param_a_avg = QLabel("0.000")
        params_layout.addRow("a (среднее):", self.param_a_avg)
        
        self.param_b_avg = QLabel("0.000")
        params_layout.addRow("b (среднее):", self.param_b_avg)
        
        self.param_c_avg = QLabel("0.0")
        params_layout.addRow("c (среднее):", self.param_c_avg)
        
        self.param_d_avg = QLabel("0.0")
        params_layout.addRow("d (среднее):", self.param_d_avg)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Connection list
        conn_group = QGroupBox("Список связей")
        conn_layout = QVBoxLayout()
        
        self.conn_list = QTextEdit()
        self.conn_list.setReadOnly(True)
        conn_layout.addWidget(self.conn_list)
        
        conn_group.setLayout(conn_layout)
        layout.addWidget(conn_group)
        
        return widget
    
    def create_inputs_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Input current table
        self.input_current_table = QTableWidget()
        self.input_current_table.setColumnCount(1)
        self.input_current_table.setHorizontalHeaderLabels(["Ток (мкА)"])
        self.input_current_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.input_current_table)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        apply_input_btn = QPushButton("✅ Применить токи")
        apply_input_btn.clicked.connect(self.apply_input_currents)
        btn_layout.addWidget(apply_input_btn)
        
        reset_input_btn = QPushButton("🗑️ Сбросить токи")
        reset_input_btn.clicked.connect(self.reset_input_currents)
        btn_layout.addWidget(reset_input_btn)
        
        random_input_btn = QPushButton("🎲 Рандомные токи")
        random_input_btn.clicked.connect(self.randomize_inputs)
        btn_layout.addWidget(random_input_btn)
        
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        return widget
    
    def create_status_bar(self):
        self.statusBar().showMessage("Готов")
    
    # ============================================================
    # SLOT METHODS
    # ============================================================
    
    def on_neuron_type_changed(self, neuron_type):
        self.params_group.setVisible(neuron_type == 'Custom')
    
    def create_network(self):
        try:
            N = self.n_neurons_spin.value()
            neuron_type = self.neuron_type_combo.currentText()
            
            if neuron_type == 'Custom':
                a = np.ones(N) * self.param_a.value()
                b = np.ones(N) * self.param_b.value()
                c = np.ones(N) * self.param_c.value()
                d = np.ones(N) * self.param_d.value()
                self.network = Izhikevich_Network(N=N, a=a, b=b, c=c, d=d)
            else:
                types = [neuron_type] * N
                a, b, c, d = types2params(types)
                self.network = Izhikevich_Network(N=N, a=a, b=b, c=c, d=d)
            
            self.network.set_init_conditions(v_noise=np.random.normal(size=N, scale=0.5))
            
            # Initialize input vector
            self.input_current_vector = np.zeros(N)
            self.update_input_table()
            
            # Update UI
            self.conn_i_spin.setMaximum(N - 1)
            self.conn_j_spin.setMaximum(N - 1)
            
            self.log_message(f"✅ Сеть создана: {N} нейронов, тип: {neuron_type}")
            self.update_stats()
            self.update_status("Сеть создана")
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось создать сеть: {str(e)}")
            self.log_message(f"❌ Ошибка создания сети: {str(e)}")
    
    def add_connection(self):
        if self.network is None:
            QMessageBox.warning(self, "Предупреждение", "Сначала создайте сеть!")
            return
        
        try:
            i = self.conn_i_spin.value()
            j = self.conn_j_spin.value()
            w = self.conn_w_spin.value()
            
            coef = 1 if w >= 0 else -1
            self.network.connect(i, j, coef=coef, w=abs(w))
            
            self.log_message(f"🔗 Добавлена связь: {i} -> {j}, вес={w}")
            self.update_stats()
            self.update_weight_matrix()
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось добавить связь: {str(e)}")
    
    def delete_connection(self):
        if self.network is None:
            QMessageBox.warning(self, "Предупреждение", "Сначала создайте сеть!")
            return
        
        try:
            i = self.conn_i_spin.value()
            j = self.conn_j_spin.value()
            
            self.network.M[j, i] = 0
            self.network.W[j, i] = 0
            self.network.tau_syn[j, i] = 0
            
            self.log_message(f"🗑️ Удалена связь: {i} -> {j}")
            self.update_stats()
            self.update_weight_matrix()
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось удалить связь: {str(e)}")
    
    def start_simulation(self):
        if self.network is None:
            QMessageBox.warning(self, "Предупреждение", "Сначала создайте сеть!")
            return
        
        if self.simulation_worker is not None and self.simulation_worker.isRunning():
            return
        
        try:
            sim_time = self.sim_time_spin.value()
            dt = self.dt_spin.value()
            
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            self.simulation_worker = SimulationWorker(
                self.network, sim_time, dt, self.input_current_vector
            )
            self.simulation_worker.progress_signal.connect(self.progress_bar.setValue)
            self.simulation_worker.finished_signal.connect(self.on_simulation_finished)
            self.simulation_worker.error_signal.connect(self.on_simulation_error)
            
            self.simulation_worker.start()
            
            self.update_status("Симуляция запущена")
            self.log_message("▶️ Симуляция запущена")
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось запустить симуляцию: {str(e)}")
    
    def stop_simulation(self):
        if self.simulation_worker is not None:
            self.simulation_worker.stop()
            self.update_status("Симуляция остановлена")
            self.log_message("⏹ Симуляция остановлена")
    
    def reset_simulation(self):
        self.stop_simulation()
        
        if self.network is not None:
            self.network.set_init_conditions(v_noise=np.random.normal(size=self.network.N, scale=0.5))
        
        self.simulation_data = None
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        
        self.clear_voltage_plot()
        self.update_status("Симуляция сброшена")
        self.log_message("🔄 Симуляция сброшена")
    
    def on_simulation_finished(self, data):
        self.simulation_data = data
        self.progress_bar.setVisible(False)
        self.update_status("Симуляция завершена")
        self.log_message(f"✅ Симуляция завершена: {len(data['time'])} шагов")
        
        self.plot_voltage()
        self.update_stats()
    
    def on_simulation_error(self, error_msg):
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Ошибка симуляции", error_msg)
        self.log_message(f"❌ Ошибка симуляции: {error_msg}")
    
    def plot_voltage(self):
        if self.simulation_data is None:
            return
        
        self.fig_voltage.clear()
        
        time_data = self.simulation_data['time']
        voltage_data = self.simulation_data['voltage']
        spike_data = self.simulation_data['spikes']
        
        n_show = min(5, voltage_data.shape[1])
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
        
        # Voltage plot
        ax1 = self.fig_voltage.add_subplot(211)
        for i in range(n_show):
            ax1.plot(time_data, voltage_data[:, i], label=f'Neuron {i}', color=colors[i], linewidth=2)
        ax1.set_title('Мембранные потенциалы')
        ax1.set_xlabel('Время (ms)')
        ax1.set_ylabel('Потенциал (mV)')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Spike plot
        ax2 = self.fig_voltage.add_subplot(212)
        for i in range(n_show):
            spike_times = time_data[spike_data[:, i] > 0]
            ax2.scatter(spike_times, [i] * len(spike_times), s=10, color=colors[i])
        ax2.set_title('Спайковая активность')
        ax2.set_xlabel('Время (ms)')
        ax2.set_ylabel('Нейрон')
        ax2.set_ylim(-0.5, n_show - 0.5)
        ax2.grid(True, alpha=0.3)
        
        self.fig_voltage.tight_layout()
        self.canvas_voltage.draw()
    
    def clear_voltage_plot(self):
        self.fig_voltage.clear()
        self.canvas_voltage.draw()
    
    def update_network_graph(self):
        if self.network is None:
            return
        
        self.fig_network.clear()
        
        try:
            import networkx as nx
            
            G = nx.DiGraph()
            for i in range(self.network.N):
                G.add_node(i)
            
            for i in range(self.network.N):
                for j in range(self.network.N):
                    if self.network.M[j, i] != 0:
                        G.add_edge(i, j, weight=self.network.W[j, i])
            
            layout_type = self.layout_combo.currentText()
            if layout_type == 'circular':
                pos = nx.circular_layout(G)
            elif layout_type == 'spring':
                pos = nx.spring_layout(G, seed=42)
            elif layout_type == 'kamada_kawai':
                pos = nx.kamada_kawai_layout(G)
            else:
                pos = nx.random_layout(G, seed=42)
            
            ax = self.fig_network.add_subplot(111)
            nx.draw(G, pos, ax=ax, with_labels=True, node_color='#2E86AB', 
                   node_size=500, font_size=8, font_color='white',
                   edge_color='#888888', arrows=True, arrowsize=20)
            
            self.fig_network.tight_layout()
            self.canvas_network.draw()
            
        except Exception as e:
            self.log_message(f"❌ Ошибка визуализации графа: {str(e)}")
    
    def update_weight_matrix(self):
        if self.network is None:
            return
        
        N = self.network.N
        self.weight_table.setRowCount(N)
        self.weight_table.setColumnCount(N)
        
        for i in range(N):
            for j in range(N):
                item = QTableWidgetItem(f"{self.network.W[j, i]:.2f}")
                if self.network.M[j, i] == 0:
                    item.setBackground(QColor('#EEEEEE'))
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.weight_table.setItem(j, i, item)
        
        # Tau matrix
        self.tau_table.setRowCount(N)
        self.tau_table.setColumnCount(N)
        
        for i in range(N):
            for j in range(N):
                tau = 1.0 / self.network.tau_syn[j, i] if self.network.M[j, i] != 0 else 0
                item = QTableWidgetItem(f"{tau:.1f}")
                if self.network.M[j, i] == 0:
                    item.setBackground(QColor('#EEEEEE'))
                self.tau_table.setItem(j, i, item)
    
    def apply_weights(self):
        if self.network is None:
            return
        
        N = self.network.N
        for i in range(N):
            for j in range(N):
                if self.network.M[j, i] != 0:
                    item = self.weight_table.item(j, i)
                    if item:
                        try:
                            self.network.W[j, i] = float(item.text())
                        except:
                            pass
        
        self.log_message("✅ Веса обновлены")
    
    def reset_weights(self):
        if self.network is None:
            return
        
        for i in range(self.network.N):
            for j in range(self.network.N):
                if self.network.M[j, i] != 0:
                    coef = np.sign(self.network.M[j, i])
                    self.network.W[j, i] = coef * 1.0
        
        self.update_weight_matrix()
        self.log_message("🔄 Веса сброшены")
    
    def randomize_weights(self):
        if self.network is None:
            return
        
        dialog = RandomWeightDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            min_w, max_w = dialog.get_values()
            for i in range(self.network.N):
                for j in range(self.network.N):
                    if self.network.M[j, i] != 0:
                        self.network.W[j, i] = np.random.uniform(min_w, max_w)
            
            self.update_weight_matrix()
            self.log_message(f"🎲 Веса рандомизированы: [{min_w}, {max_w}]")
    
    def update_input_table(self):
        if self.network is None or self.input_current_vector is None:
            return
        
        N = self.network.N
        self.input_table.setRowCount(N)
        self.input_table.setVerticalHeaderLabels([f"Нейрон {i}" for i in range(N)])
        
        for i in range(N):
            item = QTableWidgetItem(f"{self.input_current_vector[i]:.2f}")
            self.input_table.setItem(i, 0, item)
        
        # Also update input current tab
        self.input_current_table.setRowCount(N)
        self.input_current_table.setVerticalHeaderLabels([f"Нейрон {i}" for i in range(N)])
        
        for i in range(N):
            item = QTableWidgetItem(f"{self.input_current_vector[i]:.2f}")
            self.input_current_table.setItem(i, 0, item)
    
    def apply_input_currents(self):
        if self.network is None:
            return
        
        N = self.network.N
        self.input_current_vector = np.zeros(N)
        
        for i in range(N):
            item = self.input_current_table.item(i, 0)
            if item:
                try:
                    self.input_current_vector[i] = float(item.text())
                except:
                    pass
        
        self.log_message("⚡ Вектор токов обновлён")
    
    def reset_input_currents(self):
        if self.network is None:
            return
        
        self.input_current_vector = np.zeros(self.network.N)
        self.update_input_table()
        self.log_message("🔄 Токи сброшены")
    
    def randomize_inputs(self):
        if self.network is None:
            return
        
        dialog = RandomInputDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            min_i, max_i = dialog.get_values()
            self.input_current_vector = np.random.uniform(min_i, max_i, self.network.N)
            self.update_input_table()
            self.log_message(f"🎲 Токи рандомизированы: [{min_i}, {max_i}]")
    
    def update_stats(self):
        if self.network is None:
            return
        
        n_connections = np.count_nonzero(self.network.M)
        n_spikes = 0
        
        if self.simulation_data is not None:
            n_spikes = int(np.sum(self.simulation_data['spikes'] > 0))
        
        self.stats_label.setText(f"Нейронов: {self.network.N}\nСвязей: {n_connections}\nСпайков: {n_spikes}")
        
        # Update parameters tab
        self.param_a_avg.setText(f"{np.mean(self.network.a):.3f}")
        self.param_b_avg.setText(f"{np.mean(self.network.b):.3f}")
        self.param_c_avg.setText(f"{np.mean(self.network.c):.1f}")
        self.param_d_avg.setText(f"{np.mean(self.network.d):.1f}")
        
        # Update connection list
        connections = []
        for i in range(self.network.N):
            for j in range(self.network.N):
                if self.network.M[j, i] != 0:
                    conn_type = 'возбуждающий' if self.network.M[j, i] > 0 else 'тормозной'
                    connections.append(f"{i} → {j} (вес: {self.network.W[j, i]:.2f}, {conn_type})")
        
        self.conn_list.setText("\n".join(connections) if connections else "Связей не найдено")
    
    def update_status(self, status):
        self.status_label.setText(f"Статус: {status}")
        self.statusBar().showMessage(status)
    
    def log_message(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        self.log_messages.append(log_entry)
        self.log_text.append(log_entry)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
    
    def clear_log(self):
        self.log_text.clear()
        self.log_messages = []
    
    def save_network(self):
        if self.network is None:
            QMessageBox.warning(self, "Предупреждение", "Нет сети для сохранения!")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить сеть", "", "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            try:
                config = {
                    'N': self.network.N,
                    'a': self.network.a.tolist(),
                    'b': self.network.b.tolist(),
                    'c': self.network.c.tolist(),
                    'd': self.network.d.tolist(),
                    'M': self.network.M.tolist(),
                    'W': self.network.W.tolist(),
                    'tau_syn': self.network.tau_syn.tolist(),
                    'names': self.network.names if hasattr(self.network, 'names') else []
                }
                
                with open(file_path, 'w') as f:
                    json.dump(config, f, indent=2)
                
                self.log_message(f"💾 Сеть сохранена: {file_path}")
                QMessageBox.information(self, "Успех", "Сеть успешно сохранена!")
                
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить сеть: {str(e)}")
    
    def load_network(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Загрузить сеть", "", "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    config = json.load(f)
                
                self.network = Izhikevich_Network(N=config['N'])
                self.network.a = np.array(config['a'])
                self.network.b = np.array(config['b'])
                self.network.c = np.array(config['c'])
                self.network.d = np.array(config['d'])
                self.network.M = np.array(config['M'])
                self.network.W = np.array(config['W'])
                
                if 'tau_syn' in config:
                    self.network.tau_syn = np.array(config['tau_syn'])
                else:
                    self.network.tau_syn = np.ones((config['N'], config['N'])) / 10.0
                
                if config.get('names', []):
                    self.network.names = config['names']
                
                self.network.set_init_conditions(v_noise=np.random.normal(size=self.network.N, scale=0.5))
                
                self.input_current_vector = np.zeros(self.network.N)
                self.simulation_data = None
                
                # Update UI
                self.n_neurons_spin.setValue(config['N'])
                self.update_input_table()
                self.update_weight_matrix()
                self.update_stats()
                
                self.log_message(f"📂 Сеть загружена: {file_path}")
                QMessageBox.information(self, "Успех", "Сеть успешно загружена!")
                
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить сеть: {str(e)}")
    
    def export_results(self):
        if self.simulation_data is None:
            QMessageBox.warning(self, "Предупреждение", "Нет данных для экспорта!")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Экспорт результатов", "", "NPZ Files (*.npz);;CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            try:
                if file_path.endswith('.npz'):
                    np.savez(file_path,
                            time=self.simulation_data['time'],
                            voltage=self.simulation_data['voltage'],
                            spikes=self.simulation_data['spikes'])
                elif file_path.endswith('.csv'):
                    csv_data = np.column_stack([self.simulation_data['time'], self.simulation_data['voltage']])
                    header = "time," + ",".join([f"V_{i}" for i in range(self.simulation_data['voltage'].shape[1])])
                    np.savetxt(file_path, csv_data, delimiter=',', header=header, comments='')
                
                self.log_message(f"📤 Результаты экспортированы: {file_path}")
                QMessageBox.information(self, "Успех", "Результаты успешно экспортированы!")
                
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось экспортировать: {str(e)}")
    
    def export_npz(self):
        if self.simulation_data is None:
            QMessageBox.warning(self, "Предупреждение", "Нет данных для экспорта!")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить NPZ", "", "NPZ Files (*.npz)"
        )
        
        if file_path:
            try:
                np.savez(file_path,
                        time=self.simulation_data['time'],
                        voltage=self.simulation_data['voltage'],
                        spikes=self.simulation_data['spikes'])
                self.log_message(f"💾 NPZ сохранён: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", str(e))
    
    def export_csv(self):
        if self.simulation_data is None:
            QMessageBox.warning(self, "Предупреждение", "Нет данных для экспорта!")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить CSV", "", "CSV Files (*.csv)"
        )
        
        if file_path:
            try:
                csv_data = np.column_stack([self.simulation_data['time'], self.simulation_data['voltage']])
                header = "time," + ",".join([f"V_{i}" for i in range(self.simulation_data['voltage'].shape[1])])
                np.savetxt(file_path, csv_data, delimiter=',', header=header, comments='')
                self.log_message(f"💾 CSV сохранён: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", str(e))
    
    def show_about(self):
        QMessageBox.about(
            self,
            "О программе",
            "🧠 Spiking Neural Network Simulator\n\n"
            "Модель Izhikevich с интерактивной визуализацией\n\n"
            "Версия: 1.0\n"
            "Разработано для моделирования нейронных сетей\n\n"
            "Поддерживаемые типы нейронов:\n"
            "- RS (Regular Spiking)\n"
            "- IB (Intrinsically Bursting)\n"
            "- CH (Chattering)\n"
            "- FS (Fast Spiking)\n"
            "- TC (Thalamo-Cortical)\n"
            "- RZ (Resonator)\n"
            "- LTS (Low-Threshold Spiking)"
        )


# ============================================================
# DIALOG CLASSES
# ============================================================
class RandomWeightDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Генерация случайных весов")
        
        layout = QFormLayout(self)
        
        self.min_spin = QDoubleSpinBox()
        self.min_spin.setRange(-10, 10)
        self.min_spin.setValue(-5)
        layout.addRow("Мин:", self.min_spin)
        
        self.max_spin = QDoubleSpinBox()
        self.max_spin.setRange(-10, 10)
        self.max_spin.setValue(5)
        layout.addRow("Макс:", self.max_spin)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
    
    def get_values(self):
        return self.min_spin.value(), self.max_spin.value()


class RandomInputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Генерация случайных токов")
        
        layout = QFormLayout(self)
        
        self.min_spin = QDoubleSpinBox()
        self.min_spin.setRange(-100, 100)
        self.min_spin.setValue(-10)
        layout.addRow("Мин (мкА):", self.min_spin)
        
        self.max_spin = QDoubleSpinBox()
        self.max_spin.setRange(-100, 100)
        self.max_spin.setValue(10)
        layout.addRow("Макс (мкА):", self.max_spin)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
    
    def get_values(self):
        return self.min_spin.value(), self.max_spin.value()


# ============================================================
# MAIN ENTRY POINT
# ============================================================
def main():
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Set font
    font = QFont("Arial", 10)
    app.setFont(font)
    
    window = SpikingNNMainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()