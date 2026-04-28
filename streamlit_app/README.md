# SpikingNN Streamlit GUI

Interactive web interface for spiking neural network simulation using the Izhikevich model.

## Quick Start

```bash
# 1. Install the core package (if not already)
pip install SpikingNN

# 2. Install GUI dependencies
cd streamlit_app
pip install -r requirements.txt

# 3. Run the app
./run.sh
# or
streamlit run app.py
```

# Folders
```text
streamlit_app/
│
├── requirements.txt              # Зависимости для запуска GUI
├── run.sh                        # Скрипт запуска: streamlit run app.py
├── README.md                     # Краткая инструкция
│
├── config.py                     # Настройки: язык, шрифты, темы
├── state.py                      # Управление session_state
├── utils.py                      # Вспомогательные функции (numba, предвыделение)
│
├── components/                   # Переиспользуемые UI-компоненты
│   ├── __init__.py
│   ├── network_graph.py          # Визуализация графа сети
│   ├── matrix_editor.py          # Универсальный редактор матриц с маской
│   ├── muscle_viz.py             # Визуализация мышц и маятника
│   ├── afferent_panel.py         # Панель афферентов
│   └── simulation_controls.py    # Кнопки старт/стоп, прогресс
│
├── tabs/                         # Вкладки приложения
│   ├── __init__.py
│   ├── tab_visualization.py      # 📈 Графики потенциалов, спайков
│   ├── tab_network.py            # 🕸️ Граф + маска соединений
│   ├── tab_matrices.py           # 📊 W, tau_syn с валидацией по маске
│   ├── tab_muscles.py            # 💪 Мышцы + маятник (фикс. параметры)
│   ├── tab_afferents.py          # 🔌 Матрица Q_aff
│   ├── tab_io.py                 # ⚙️ Матрицы IO_Network: Q_app, Q_aff, P
│   ├── tab_params.py             # ⚙️ Параметры нейронов (types2params)
│   └── tab_docs.py               # 📚 Документация EN/RU
│
└── app.py                        # Точка входа: сборка вкладок, логика
```