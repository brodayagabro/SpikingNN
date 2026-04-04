import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import io
from datetime import datetime
import networkx as nx
import pandas as pd

# Импорт из вашего файла
from Izh_net import (
    Izhikevich_Network, 
    Izhikevich_IO_Network, 
    Network,
    NameNetwork,
    types2params,
    izhikevich_neuron
)

# ============================================================
# КОНФИГУРАЦИЯ СТРАНИЦЫ
# ============================================================
st.set_page_config(
    page_title="Спайковая Нейронная Сеть",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Стили CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 0.5rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Заголовок
st.markdown('<h1 class="main-header">🔬 Izhikevich Neural Network Simulator</h1>', unsafe_allow_html=True)
st.markdown("### Численное моделирование динамики спайковых нейронных сетей")

# ============================================================
# ОПИСАНИЕ ПРИЛОЖЕНИЯ И СЦЕНАРИИ РАБОТЫ
# ============================================================
st.markdown("""
<div class="info-box">
    <h4>📖 О приложении</h4>
    <p>
        Это интерактивная веб-платформа для моделирования, визуализации и исследования 
        <b>спайковых нейронных сетей</b> на основе модели Ижикевича. 
        Приложение позволяет создавать сети различной топологии, настраивать параметры нейронов 
        и синапсов, задавать входные стимулы и анализировать динамику активности в реальном времени.
    </p>
</div>

<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin: 1rem 0;">
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 0.5rem; color: white;">
        <h5 style="margin: 0 0 0.5rem 0;">🔬 Научные исследования</h5>
        <p style="margin: 0; font-size: 0.9rem;">Изучение паттернов спайковой активности, синхронизации и распространения волн возбуждения</p>
    </div>
    <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 1rem; border-radius: 0.5rem; color: white;">
        <h5 style="margin: 0 0 0.5rem 0;">🎓 Образовательный инструмент</h5>
        <p style="margin: 0; font-size: 0.9rem;">Визуальное изучение принципов работы нейронных сетей для студентов и преподавателей</p>
    </div>
    <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 1rem; border-radius: 0.5rem; color: white;">
        <h5 style="margin: 0 0 0.5rem 0;">⚙️ Прототипирование</h5>
        <p style="margin: 0; font-size: 0.9rem;">Быстрое тестирование архитектур сетей перед реализацией в нейроморфных системах</p>
    </div>
</div>

<h4>🎯 Возможные сценарии работы</h4>

<table style="width: 100%; border-collapse: collapse; margin: 1rem 0;">
    <thead>
        <tr style="background-color: #2E86AB; color: white;">
            <th style="padding: 0.75rem; text-align: left; border-radius: 0.5rem 0 0 0;">Сценарий</th>
            <th style="padding: 0.75rem; text-align: left;">Описание</th>
            <th style="padding: 0.75rem; text-align: left; border-radius: 0 0.5rem 0 0;">Вкладки</th>
        </tr>
    </thead>
    <tbody>
        <tr style="background-color: #f8f9fa;">
            <td style="padding: 0.75rem; border-left: 3px solid #2E86AB;"><b>1. Базовое моделирование</b></td>
            <td style="padding: 0.75rem;">Создание сети из нейронов одного типа, запуск симуляции с постоянным входным током, анализ спайковой активности</td>
            <td style="padding: 0.75rem;">1, 4, 6</td>
        </tr>
        <tr>
            <td style="padding: 0.75rem; border-left: 3px solid #A23B72;"><b>2. Исследование типов нейронов</b></td>
            <td style="padding: 0.75rem;">Сравнение динамики RS, FS, IB, CH нейронов при одинаковых входных стимулах</td>
            <td style="padding: 0.75rem;">1, 5, 6</td>
        </tr>
        <tr style="background-color: #f8f9fa;">
            <td style="padding: 0.75rem; border-left: 3px solid #F18F01;"><b>3. Сетевая топология</b></td>
            <td style="padding: 0.75rem;">Построение сетей с различной связностью: полносвязные, разреженные, малый мир, масштабно-инвариантные</td>
            <td style="padding: 0.75rem;">2, 3, 4</td>
        </tr>
        <tr>
            <td style="padding: 0.75rem; border-left: 3px solid #C73E1D;"><b>4. Синаптическая пластичность</b></td>
            <td style="padding: 0.75rem;">Ручное изменение весов связей, исследование влияния силы синапсов на синхронизацию</td>
            <td style="padding: 0.75rem;">3, 1</td>
        </tr>
        <tr style="background-color: #f8f9fa;">
            <td style="padding: 0.75rem; border-left: 3px solid #6A994E;"><b>5. Входные паттерны</b></td>
            <td style="padding: 0.75rem;">Задание пространственно-распределённых входных токов, стимуляция отдельных групп нейронов</td>
            <td style="padding: 0.75rem;">6, 1</td>
        </tr>
        <tr>
            <td style="padding: 0.75rem; border-left: 3px solid #BC4749;"><b>6. Генерация ритмов</b></td>
            <td style="padding: 0.75rem;">Создание центральных генераторов паттернов (CPG) для моделирования локомоторной активности</td>
            <td style="padding: 0.75rem;">2, 3, 6, 1</td>
        </tr>
        <tr style="background-color: #f8f9fa;">
            <td style="padding: 0.75rem; border-left: 3px solid #577590;"><b>7. Возбуждение/Торможение</b></td>
            <td style="padding: 0.75rem;">Баланс возбуждающих и тормозных связей, исследование E/I баланса в сети</td>
            <td style="padding: 0.75rem;">2, 3, 1</td>
        </tr>
        <tr>
            <td style="padding: 0.75rem; border-left: 3px solid #8D5524;"><b>8. Сохранение и загрузка</b></td>
            <td style="padding: 0.75rem;">Экспорт конфигураций сетей для воспроизведения экспериментов и совместной работы</td>
            <td style="padding: 0.75rem;">Sidebar</td>
        </tr>
    </tbody>
</table>

<h4>🚀 Быстрый старт</h4>
<ol>
    <li><b>Создайте сеть:</b> В боковой панели укажите количество нейронов и выберите тип (например, RS)</li>
    <li><b>Добавьте связи:</b> Используйте секцию "🔌 Связи" или редактируйте матрицу во вкладке 3</li>
    <li><b>Настройте токи:</b> Перейдите во вкладку 6 "⚡ Входные токи" и задайте стимуляцию</li>
    <li><b>Запустите симуляцию:</b> Нажмите "▶️ Старт" и наблюдайте за динамикой во вкладке 1</li>
    <li><b>Анализируйте:</b> Изучайте графики потенциалов, спайков и статистику активности</li>
</ol>

<h4>💡 Советы по использованию</h4>
<ul>
    <li>Для больших сетей (>50 нейронов) используйте <b>spring</b> или <b>kamada_kawai</b> layout для наглядной визуализации</li>
    <li>Начинайте с малых временных шагов (<b>dt = 0.1 мс</b>) для точности численного интегрирования</li>
    <li>Используйте <b>пресеты токов</b> во вкладке 6 для быстрого тестирования различных режимов стимуляции</li>
    <li>Сохраняйте успешные конфигурации в <b>JSON</b> для последующего воспроизведения</li>
    <li>Для исследования синхронизации создавайте сети с <b>однородными весами</b> и варьируйте силу связей</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.divider()

# ============================================================
# ИНИЦИАЛИЗАЦИЯ SESSION STATE
# ============================================================
if 'network' not in st.session_state:
    st.session_state.network = None
if 'simulation_data' not in st.session_state:
    st.session_state.simulation_data = None
if 'log_messages' not in st.session_state:
    st.session_state.log_messages = []
if 'input_current_vector' not in st.session_state:
    st.session_state.input_current_vector = None

def log_message(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.log_messages.append(f"[{timestamp}] {message}")
    if len(st.session_state.log_messages) > 100:
        st.session_state.log_messages.pop(0)

# ============================================================
# ФУНКЦИЯ ВИЗУАЛИЗАЦИИ СЕТИ (ГРАФ)
# ============================================================
def create_network_graph(network, layout_type='circular', show_weights=True):
    """Создаёт интерактивный граф нейронной сети с направленными связями и весами"""
    if network is None or network.N == 0:
        return None
    
    G = nx.DiGraph()
    
    for i in range(network.N):
        G.add_node(i, label=f"Neuron {i}")
    
    for i in range(network.N):
        for j in range(network.N):
            if network.M[j, i] != 0:
                weight = network.W[j, i]
                edge_type = 'excitatory' if network.W[j, i] >= 0 else 'inhibitory'
                G.add_edge(i, j, weight=weight, type=edge_type)
    
    if layout_type == 'circular':
        pos = nx.circular_layout(G)
    elif layout_type == 'spring':
        pos = nx.spring_layout(G, seed=42, k=2/np.sqrt(max(1, network.N)))
    elif layout_type == 'kamada_kawai':
        try:
            pos = nx.kamada_kawai_layout(G)
        except:
            pos = nx.circular_layout(G)
    else:
        pos = nx.random_layout(G, seed=42)
    
    node_x = [pos[i][0] for i in range(network.N)]
    node_y = [pos[i][1] for i in range(network.N)]
    
    fig = go.Figure()
    
    # Рёбра (линии)
    edge_x, edge_y = [], []
    for edge in G.edges():
        i, j = edge
        x0, y0 = pos[i]
        x1, y1 = pos[j]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888888'),
        hoverinfo='none',
        mode='lines',
        showlegend=False
    ))
    
    # Стрелки и веса через annotations
    annotations = []
    for edge in G.edges():
        i, j = edge
        x0, y0 = pos[i]
        x1, y1 = pos[j]
        
        dx = x1 - x0
        dy = y1 - y0
        length = np.sqrt(dx**2 + dy**2)
        
        if length > 0:
            dx /= length
            dy /= length
            
            edge_data = G.edges[edge[0], edge[1]]
            arrow_color = '#2E86AB' if edge_data['type'] == 'excitatory' else '#E74C3C'
            
            # Стрелка
            annotations.append(dict(
                x=x1, y=y1, ax=x0, ay=y0,
                xref='x', yref='y', axref='x', ayref='y',
                showarrow=True, arrowhead=2, arrowsize=1.5,
                arrowwidth=2, arrowcolor=arrow_color, opacity=0.8
            ))
            
            # Вес связи
            if show_weights:
                mid_x = (x0 + x1) / 2
                mid_y = (y0 + y1) / 2
                offset = 0.05
                text_x = mid_x - offset * dy
                text_y = mid_y + offset * dx
                
                weight = edge_data['weight']
                weight_str = f"{weight:.1f}" if abs(weight) >= 1 else f"{weight:.2f}"
                
                annotations.append(dict(
                    x=text_x, y=text_y, xref='x', yref='y',
                    text=weight_str, showarrow=False,
                    font=dict(size=9, color=arrow_color, family="Arial Black"),
                    bgcolor='rgba(255, 255, 255, 0.8)',
                    borderpad=2, bordercolor=arrow_color, opacity=0.9
                ))
    
    # Узлы (нейроны)
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(size=25, color='#2E86AB', line=dict(width=2, color='white')),
        text=list(range(network.N)),
        textposition="middle center",
        textfont=dict(size=14, color='white', family="Arial Black"),
        hovertemplate='<b>Нейрон %{text}</b><br>Потенциал: %{customdata[0]:.1f} мВ<br>Входящих: %{customdata[1]}<br>Исходящих: %{customdata[2]}<extra></extra>',
        customdata=[[float(network.V_prev[i]) if hasattr(network, 'V_prev') else 0, G.in_degree(i), G.out_degree(i)] for i in range(network.N)],
        name='Нейроны'
    ))
    
    # Layout
    fig.update_layout(
        title={'text': f"🕸️ Структура сети ({network.N} нейронов, {G.number_of_edges()} синапсов)", 'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
        showlegend=True, hovermode='closest',
        margin=dict(b=20, l=20, r=20, t=80),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.2, 1.2]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.2, 1.2]),
        plot_bgcolor='white', paper_bgcolor='white',
        height=600, template='plotly_white',
        annotations=annotations
    )
    
    # Легенда
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=12, color='#2E86AB'), name='🔵 Возбуждающий', hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=12, color='#E74C3C'), name='🔴 Тормозной', hoverinfo='skip'))
    
    return fig

# ============================================================
# ФУНКЦИЯ ВИЗУАЛИЗАЦИИ МАТРИЦЫ ВЕСОВ
# ============================================================
def create_weight_matrix(network, show_annotations=True, colorscale='RdBu'):
    """Создаёт тепловую карту матрицы весов"""
    if network is None or network.N == 0:
        return None
    
    N = network.N
    W = network.W.copy()
    M = network.M.copy()
    
    fig = go.Figure()
    
    max_abs_w = np.max(np.abs(W)) if np.max(np.abs(W)) > 0 else 1
    
    fig.add_trace(go.Heatmap(
        z=W, x=list(range(N)), y=list(range(N)),
        colorscale=colorscale,
        zmin=-max_abs_w * 1.1,
        zmax=max_abs_w * 1.1,
        showscale=True,
        colorbar=dict(title='Вес'),
        hovertemplate='<b>Связь: %{x} → %{y}</b><br>Вес: %{z:.2f}<br>Тип: %{customdata}<br><extra></extra>',
        customdata=[['Возбуждающая' if M[j, i] > 0 else 'Тормозная' if M[j, i] < 0 else 'Нет связи' for i in range(N)] for j in range(N)],
        name='Веса'
    ))
    
    # Аннотации с значениями
    if show_annotations:
        annotations = []
        for i in range(N):
            for j in range(N):
                if M[j, i] != 0:
                    annotations.append(dict(
                        x=i, y=j, xref='x', yref='y',
                        text=f'{W[j, i]:.1f}', showarrow=False,
                        font=dict(size=10, color='white' if abs(W[j, i]) > max_abs_w * 0.5 else 'black')
                    ))
        fig.update_layout(annotations=annotations)
    
    fig.update_layout(
        title={'text': f"📊 Матрица весов ({N}×{N})", 'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
        xaxis=dict(title='Пресинаптический нейрон (i)', tickmode='linear', dtick=1),
        yaxis=dict(title='Постсинаптический нейрон (j)', tickmode='linear', dtick=1, autorange='reversed'),
        height=500, template='plotly_white'
    )
    
    return fig

# ============================================================
# ФУНКЦИЯ ВИЗУАЛИЗАЦИИ ВЕКТОРА ТОКОВ
# ============================================================
def create_input_current_plot(input_vector, N):
    """Создаёт визуализацию вектора входных токов"""
    if input_vector is None or len(input_vector) == 0:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=list(range(N)),
        y=input_vector,
        marker_color=['#2E86AB' if v > 0 else '#E74C3C' if v < 0 else '#888888' for v in input_vector],
        hovertemplate='<b>Нейрон %{x}</b><br>Ток: %{y:.2f} мкА<br><extra></extra>',
        name='Входной ток'
    ))
    
    fig.update_layout(
        title={'text': f"⚡ Вектор входных токов ({N} нейронов)", 'y':0.95, 'x':0.5},
        xaxis=dict(title='Нейрон', tickmode='linear', dtick=1),
        yaxis=dict(title='Ток (мкА)'),
        height=300,
        template='plotly_white',
        showlegend=False
    )
    
    return fig

# ============================================================
# БОКОВАЯ ПАНЕЛЬ (БЕЗ ТОКА!)
# ============================================================
with st.sidebar:
    st.header("⚙️ Параметры")
    
    # Конфигурация сети
    st.subheader("🔗 Конфигурация сети")
    n_neurons = st.number_input("Количество нейронов", min_value=1, max_value=100, value=10)
    
    neuron_types = ['RS', 'IB', 'CH', 'FS', 'TC', 'RZ', 'LTS', 'Custom']
    neuron_type = st.selectbox("Тип нейронов", neuron_types, index=0)
    
    if neuron_type == 'Custom':
        st.subheader("Параметры Izhikevich")
        param_a = st.number_input("a", value=0.02, format="%.3f")
        param_b = st.number_input("b", value=0.2, format="%.3f")
        param_c = st.number_input("c", value=-65.0, format="%.1f")
        param_d = st.number_input("d", value=8.0, format="%.1f")
    
    if st.button("🔨 Создать сеть", use_container_width=True):
        try:
            if neuron_type == 'Custom':
                st.session_state.network = Izhikevich_Network(
                    N=n_neurons, 
                    a=np.ones(n_neurons) * param_a,
                    b=np.ones(n_neurons) * param_b,
                    c=np.ones(n_neurons) * param_c,
                    d=np.ones(n_neurons) * param_d
                )
            else:
                types = [neuron_type] * n_neurons
                a, b, c, d = types2params(types)
                st.session_state.network = Izhikevich_Network(N=n_neurons, a=a, b=b, c=c, d=d)
            
            st.session_state.network.set_init_conditions(v_noise=np.random.normal(size=n_neurons, scale=0.5))
            
            # Инициализация вектора токов
            st.session_state.input_current_vector = np.zeros(n_neurons)
            
            log_message(f"✅ Сеть создана: {n_neurons} нейронов, тип: {neuron_type}")
            st.success("Сеть успешно создана!")
        except Exception as e:
            st.error(f"❌ Ошибка: {str(e)}")
            log_message(f"❌ Ошибка создания сети: {str(e)}")
    
    st.divider()
    
    # Симуляция (БЕЗ Iapp!)
    st.subheader("▶️ Симуляция")
    sim_time = st.number_input("Время (ms)", min_value=10, max_value=10000, value=1000)
    dt = st.number_input("Шаг dt (ms)", min_value=0.01, max_value=1.0, value=0.1, format="%.2f")
    # ❌ УБРАНО: iapp = st.number_input(...)
    
    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("▶️ Старт", use_container_width=True, disabled=st.session_state.network is None)
    with col2:
        reset_btn = st.button("🔄 Сброс", use_container_width=True)
    
    st.divider()
    
    # Связи
    st.subheader("🔌 Связи")
    col_i, col_j, col_w = st.columns(3)
    with col_i:
        conn_i = st.number_input("От (i)", min_value=0, max_value=max(1, n_neurons-1), value=0, key="conn_i")
    with col_j:
        conn_j = st.number_input("До (j)", min_value=0, max_value=max(1, n_neurons-1), value=1, key="conn_j")
    with col_w:
        conn_w = st.number_input("Вес", min_value=-10.0, max_value=10.0, value=1.0, format="%.1f", key="conn_w")
    
    if st.button("➕ Добавить", use_container_width=True):
        if st.session_state.network is not None:
            try:
                st.session_state.network.connect(int(conn_i), int(conn_j), coef=1 if conn_w > 0 else -1, w=abs(conn_w))
                log_message(f"🔗 Добавлена связь: {conn_i} -> {conn_j}, вес={conn_w}")
                st.success("Связь добавлена!")
            except Exception as e:
                st.error(f"❌ Ошибка: {str(e)}")
    
    if st.button("🗑️ Удалить", use_container_width=True):
        if st.session_state.network is not None:
            try:
                st.session_state.network.M[int(conn_j), int(conn_i)] = 0
                st.session_state.network.W[int(conn_j), int(conn_i)] = 0
                log_message(f"🗑️ Удалена связь: {conn_i} -> {conn_j}")
                st.success("Связь удалена!")
            except Exception as e:
                st.error(f"❌ Ошибка: {str(e)}")
    
    # ============================================================
    # БОКОВАЯ ПАНЕЛЬ - ИСПРАВЛЕННАЯ СЕКЦИЯ ФАЙЛОВ
    # ============================================================
    st.divider()

    # Файлы
    st.subheader("💾 Файлы")

    # ✅ СОХРАНЕНИЕ (работает)
    if st.button("📥 Сохранить сеть", use_container_width=True):
        if st.session_state.network is not None:
            config = {
                'N': st.session_state.network.N,
                'a': st.session_state.network.a.tolist(),
                'b': st.session_state.network.b.tolist(),
                'c': st.session_state.network.c.tolist(),
                'd': st.session_state.network.d.tolist(),
                'M': st.session_state.network.M.tolist(),
                'W': st.session_state.network.W.tolist(),
                'tau_syn': st.session_state.network.tau_syn.tolist(),
                'names': st.session_state.network.names if hasattr(st.session_state.network, 'names') else []
            }
            json_str = json.dumps(config, indent=2)
            st.download_button(
                label="⬇️ Скачать JSON",
                data=json_str,
                file_name=f"network_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

    # ✅ ЗАГРУЗКА (ИСПРАВЛЕНО!)
    st.markdown("**Загрузить сеть:**")
    uploaded_file = st.file_uploader(
        "Выберите JSON файл",
        type=['json'],
        key="network_upload",
        help="Выберите файл .json для загрузки конфигурации сети"
    )

    if uploaded_file is not None:
        col_load1, col_load2 = st.columns(2)
        with col_load1:
            if st.button("📤 Загрузить", use_container_width=True, key="load_network_btn"):
                try:
                    config = json.load(uploaded_file)
                
                    # Создаём сеть
                    st.session_state.network = Izhikevich_Network(N=config['N'])
                    st.session_state.network.a = np.array(config['a'])
                    st.session_state.network.b = np.array(config['b'])
                    st.session_state.network.c = np.array(config['c'])
                    st.session_state.network.d = np.array(config['d'])
                    st.session_state.network.M = np.array(config['M'])
                    st.session_state.network.W = np.array(config['W'])
                
                    # ✅ Загружаем tau_syn если есть в файле
                    if 'tau_syn' in config:
                        st.session_state.network.tau_syn = np.array(config['tau_syn'])
                    else:
                        # По умолчанию 10 мс
                        st.session_state.network.tau_syn = np.ones((config['N'], config['N'])) / 10.0
                
                    if config.get('names', []):
                        st.session_state.network.names = config['names']
                
                    # Инициализируем состояния
                    st.session_state.network.set_init_conditions(
                        v_noise=np.random.normal(size=st.session_state.network.N, scale=0.5)
                    )
                
                    # Сбрасываем данные симуляции
                    st.session_state.simulation_data = None
                
                    # Инициализируем вектор токов
                    st.session_state.input_current_vector = np.zeros(st.session_state.network.N)
                
                    log_message(f"✅ Сеть загружена из файла: {uploaded_file.name}")
                    st.success(f"Сеть успешно загружена! ({config['N']} нейронов)")
                    st.rerun()
                
                except Exception as e:
                    st.error(f"❌ Ошибка загрузки: {str(e)}")
                    log_message(f"❌ Ошибка загрузки сети: {str(e)}")
    
        with col_load2:
            if st.button("🗑️ Очистить файл", use_container_width=True, key="clear_upload"):
                st.session_state.network_upload = None
                st.rerun()
    else:
        st.info("📁 Выберите файл для загрузки")

    
    st.divider()
    
    # Статистика
    st.subheader("📊 Статистика")
    if st.session_state.network is not None:
        n_connections = np.count_nonzero(st.session_state.network.M)
        n_spikes = int(np.sum(st.session_state.simulation_data['spikes'] > 0)) if st.session_state.simulation_data is not None else 0
        st.metric("Нейронов", st.session_state.network.N)
        st.metric("Связей", n_connections)
        st.metric("Спайков", n_spikes)

# ============================================================
# ОСНОВНАЯ ОБЛАСТЬ
# ============================================================
if st.session_state.network is None:
    st.info("👈 Создайте сеть в боковой панели для начала работы")
else:
    N = st.session_state.network.N
    
    # Инициализация вектора токов если нет
    if st.session_state.input_current_vector is None or len(st.session_state.input_current_vector) != N:
        st.session_state.input_current_vector = np.zeros(N)
    
    # ============================================================
    # СОЗДАНИЕ ВКЛАДОК (6 ВКЛАДОК!)
    # ============================================================
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Визуализация", 
        "🕸️ Граф сети", 
        "📊 Матрица весов", 
        "📝 Лог событий", 
        "⚙️ Параметры",
        "⚡ Входные токи"  # ← НОВАЯ ВКЛАДКА!
    ])
    
    # ========== ВКЛАДКА 1: ВИЗУАЛИЗАЦИЯ ==========
    with tab1:
        if start_btn:
            with st.spinner("🔄 Запуск симуляции..."):
                try:
                    n_steps = int(sim_time / dt)
                    V_history, time_history, spike_history = [], [], []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for step in range(n_steps):
                        # ✅ Используем вектор токов вместо одиночного Iapp
                        st.session_state.network.step(dt=dt, Iapp=st.session_state.input_current_vector)
                        time_history.append(step * dt)
                        V_history.append(st.session_state.network.V_prev.copy())
                        spike_history.append(st.session_state.network.output.copy())
                        progress_bar.progress((step + 1) / n_steps)
                        status_text.text(f"Прогресс: {(step + 1) / n_steps * 100:.1f}%")
                    
                    st.session_state.simulation_data = {
                        'time': np.array(time_history),
                        'voltage': np.array(V_history),
                        'spikes': np.array(spike_history)
                    }
                    log_message(f"✅ Симуляция завершена: {len(time_history)} шагов")
                    st.success("Симуляция завершена!")
                except Exception as e:
                    st.error(f"❌ Ошибка симуляции: {str(e)}")
                    log_message(f"❌ Ошибка симуляции: {str(e)}")
        
        if reset_btn:
            st.session_state.network.set_init_conditions(v_noise=np.random.normal(size=st.session_state.network.N, scale=0.5))
            st.session_state.simulation_data = None
            log_message("🔄 Симуляция сброшена")
            st.rerun()
        
        if st.session_state.simulation_data is not None:
            data = st.session_state.simulation_data
            time_data, voltage_data, spike_data = data['time'], data['voltage'], data['spikes']
            
            fig = make_subplots(rows=2, cols=1, subplot_titles=('Мембранные потенциалы', 'Спайковая активность'), vertical_spacing=0.1)
            n_show = min(5, voltage_data.shape[1])
            colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
            
            for i in range(n_show):
                fig.add_trace(go.Scatter(x=time_data, y=voltage_data[:, i], mode='lines', name=f'Neuron {i}', line=dict(color=colors[i % len(colors)], width=2)), row=1, col=1)
                spike_times = time_data[spike_data[:, i] > 0]
                if len(spike_times) > 0:
                    fig.add_trace(go.Scatter(x=spike_times, y=[i] * len(spike_times), mode='markers', marker=dict(size=8, color=colors[i % len(colors)]), showlegend=False), row=2, col=1)
            
            fig.update_layout(height=600, showlegend=True, hovermode='x unified', template='plotly_white')
            fig.update_xaxes(title_text="Время (ms)", row=2, col=1)
            fig.update_yaxes(title_text="Потенциал (mV)", row=1, col=1)
            fig.update_yaxes(title_text="Нейрон", row=2, col=1)
            st.plotly_chart(fig, use_container_width=True)
            
            # Экспорт
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("💾 NPZ"):
                    buffer = io.BytesIO()
                    np.savez(buffer, time=time_data, voltage=voltage_data, spikes=spike_data)
                    st.download_button(label="⬇️ Скачать", data=buffer.getvalue(), file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz", mime="application/octet-stream")
            with col2:
                if st.button("💾 CSV"):
                    csv_data = np.column_stack([time_data, voltage_data])
                    header = "time," + ",".join([f"V_{i}" for i in range(voltage_data.shape[1])])
                    csv_str = header + "\n" + "\n".join([",".join(map(str, row)) for row in csv_data])
                    st.download_button(label="⬇️ Скачать", data=csv_str, file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")
            with col3:
                if st.button("📊 Статистика"):
                    spike_counts = np.sum(spike_data > 0, axis=0)
                    st.write(f"**Всего спайков:** {np.sum(spike_counts)}")
                    st.write(f"**Средняя частота:** {np.mean(spike_counts)/sim_time*1000:.2f} Гц")
    
    # ========== ВКЛАДКА 2: ГРАФ СЕТИ ==========
    with tab2:
        st.subheader("🕸️ Визуализация структуры сети")
        col1, col2 = st.columns([3, 1])
        with col1:
            layout_type = st.selectbox("Расположение", ['circular', 'spring', 'kamada_kawai', 'random'], format_func=lambda x: {'circular': '🔵 Круговое', 'spring': '🌐 Пружинное', 'kamada_kawai': '📐 Камада-Каваи', 'random': '🎲 Случайное'}[x])
        with col2:
            show_weights_graph = st.checkbox("Веса", value=True)
        
        network_fig = create_network_graph(st.session_state.network, layout_type, show_weights_graph)
        if network_fig:
            st.plotly_chart(network_fig, use_container_width=True)
            st.markdown("""
            <div class="info-box">
                <h4>📖 Условные обозначения:</h4>
                <ul>
                    <li>🔵 <b>Синие узлы</b> — нейроны</li>
                    <li>🔵 <b>Синие стрелки</b> — возбуждающие синапсы</li>
                    <li>🔴 <b>Красные стрелки</b> — тормозные синапсы</li>
                    <li>📏 <b>Число</b> — вес связи</li>
                    <li>🖱️ <b>Наведите на нейрон</b> — информация</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # ========== ВКЛАДКА 3: МАТРИЦА ВЕСОВ ==========
    with tab3:
        st.subheader("📊 Интерактивная матрица весов и синаптических констант")
        st.markdown("Редактируйте веса связей и константы релаксации синапсов.")
        
        # Настройки
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            show_annotations_matrix = st.checkbox("Показывать значения", value=True, key="show_ann")
        with col2:
            color_scale = st.selectbox("Цветовая схема", ['RdBu', 'Viridis', 'Plasma', 'Coolwarm'])
        with col3:
            if st.button("🔄 Обновить", use_container_width=True):
                st.rerun()
        
        st.divider()
        
        # Матрица весов
        st.markdown("### 🔵 Матрица весов (W)")
        
        weight_df = pd.DataFrame(
            st.session_state.network.W,
            index=[f"Нейрон {j}" for j in range(N)],
            columns=[f"Нейрон {i}" for i in range(N)]
        )
        
        edited_df = st.data_editor(
            weight_df,
            use_container_width=True,
            height=300,
            key="weight_editor",
            num_rows="fixed",
            column_config={
                col: st.column_config.NumberColumn(col, min_value=-10.0, max_value=10.0, step=0.1, format="%.2f")
                for col in weight_df.columns
            }
        )
        
        col_apply1, col_apply2, col_apply3 = st.columns(3)
        with col_apply1:
            if st.button("✅ Применить веса", use_container_width=True, key="apply_weights"):
                try:
                    changes_applied = False
                    for i in range(N):
                        for j in range(N):
                            if st.session_state.network.M[j, i] != 0:
                                old_weight = st.session_state.network.W[j, i]
                                new_weight = edited_df.iloc[j, i]
                                if abs(old_weight - new_weight) > 0.001:
                                    coef = np.sign(st.session_state.network.M[j, i])
                                    st.session_state.network.W[j, i] = abs(new_weight) * coef
                                    log_message(f"✏️ Вес [{i}→{j}]: {old_weight:.2f} → {st.session_state.network.W[j, i]:.2f}")
                                    changes_applied = True
                    if changes_applied:
                        st.success("✅ Веса обновлены!")
                    else:
                        st.info("ℹ️ Изменений нет")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Ошибка: {str(e)}")
        
        with col_apply2:
            if st.button("🗑️ Сбросить веса", use_container_width=True, key="reset_weights"):
                try:
                    for i in range(N):
                        for j in range(N):
                            if st.session_state.network.M[j, i] != 0:
                                coef = np.sign(st.session_state.network.M[j, i])
                                st.session_state.network.W[j, i] = coef * 1.0
                    log_message("🔄 Веса сброшены к 1.0")
                    st.success("Веса сброшены!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Ошибка: {str(e)}")
        
        with col_apply3:
            if st.button("🎲 Рандомные веса", use_container_width=True, key="random_weights"):
                try:
                    st.session_state.show_random_dialog = True
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Ошибка: {str(e)}")
        
        # Диалог генерации случайных весов
        if st.session_state.get('show_random_dialog', False):
            st.info("🎲 **Генерация случайных весов**")
            col_rw1, col_rw2, col_rw3, col_rw4 = st.columns(4)
            with col_rw1:
                rw_min = st.number_input("Мин", value=-5.0, format="%.2f", key="rw_min")
            with col_rw2:
                rw_max = st.number_input("Макс", value=5.0, format="%.2f", key="rw_max")
            with col_rw3:
                rw_dist = st.selectbox("Распределение", ['uniform', 'normal'], key="rw_dist")
            with col_rw4:
                rw_std = st.number_input("Стд. откл.", value=1.0, format="%.2f", key="rw_std")
            
            col_rw_btn1, col_rw_btn2 = st.columns(2)
            with col_rw_btn1:
                if st.button("🎲 Сгенерировать", use_container_width=True, key="gen_random_w"):
                    try:
                        for i in range(N):
                            for j in range(N):
                                if st.session_state.network.M[j, i] != 0:
                                    if rw_dist == 'uniform':
                                        new_w = np.random.uniform(rw_min, rw_max)
                                    else:
                                        new_w = np.random.normal((rw_min+rw_max)/2, rw_std)
                                    coef = np.sign(st.session_state.network.M[j, i])
                                    st.session_state.network.W[j, i] = abs(new_w) * coef
                        log_message(f"🎲 Веса сгенерированы: [{rw_min}, {rw_max}], {rw_dist}")
                        st.success("Веса сгенерированы!")
                        st.session_state.show_random_dialog = False
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Ошибка: {str(e)}")
            with col_rw_btn2:
                if st.button("❌ Отмена", use_container_width=True, key="cancel_random_w"):
                    st.session_state.show_random_dialog = False
                    st.rerun()
        
        st.divider()
        
        # Матрица tau_syn
        st.markdown("### 🟢 Матрица констант релаксации (τ)")
        st.markdown("Время релаксации синаптического тока в мс (по умолчанию 10 мс)")
        
        tau_display = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if st.session_state.network.M[j, i] != 0:
                    tau_display[j, i] = 1.0 / st.session_state.network.tau_syn[j, i]
                else:
                    tau_display[j, i] = 0
        
        tau_df = pd.DataFrame(
            tau_display,
            index=[f"Нейрон {j}" for j in range(N)],
            columns=[f"Нейрон {i}" for i in range(N)]
        )
        
        edited_tau_df = st.data_editor(
            tau_df,
            use_container_width=True,
            height=300,
            key="tau_editor",
            num_rows="fixed",
            column_config={
                col: st.column_config.NumberColumn(col, min_value=0.1, max_value=100.0, step=0.5, format="%.1f")
                for col in tau_df.columns
            }
        )
        
        col_tau1, col_tau2, col_tau3 = st.columns(3)
        with col_tau1:
            if st.button("✅ Применить τ", use_container_width=True, key="apply_tau"):
                try:
                    changes_applied = False
                    for i in range(N):
                        for j in range(N):
                            if st.session_state.network.M[j, i] != 0:
                                old_tau = 1.0 / st.session_state.network.tau_syn[j, i]
                                new_tau = edited_tau_df.iloc[j, i]
                                if abs(old_tau - new_tau) > 0.01 and new_tau > 0:
                                    st.session_state.network.tau_syn[j, i] = 1.0 / new_tau
                                    log_message(f"✏️ τ [{i}→{j}]: {old_tau:.1f} мс → {new_tau:.1f} мс")
                                    changes_applied = True
                    if changes_applied:
                        st.success("✅ Константы релаксации обновлены!")
                    else:
                        st.info("ℹ️ Изменений нет")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Ошибка: {str(e)}")
        
        with col_tau2:
            if st.button("🗑️ Сбросить τ", use_container_width=True, key="reset_tau"):
                try:
                    for i in range(N):
                        for j in range(N):
                            if st.session_state.network.M[j, i] != 0:
                                st.session_state.network.tau_syn[j, i] = 1.0 / 10.0
                    log_message("🔄 τ сброшены к 10 мс")
                    st.success("Константы сброшены!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Ошибка: {str(e)}")
        
        with col_tau3:
            if st.button("🎲 Рандомные τ", use_container_width=True, key="random_tau"):
                try:
                    st.session_state.show_tau_random_dialog = True
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Ошибка: {str(e)}")
        
        # Диалог генерации случайных tau
        if st.session_state.get('show_tau_random_dialog', False):
            st.info("🎲 **Генерация случайных констант релаксации**")
            col_rt1, col_rt2, col_rt3 = st.columns(3)
            with col_rt1:
                rt_min = st.number_input("Мин (мс)", value=5.0, format="%.1f", key="rt_min")
            with col_rt2:
                rt_max = st.number_input("Макс (мс)", value=20.0, format="%.1f", key="rt_max")
            with col_rt3:
                rt_dist = st.selectbox("Распределение", ['uniform', 'normal'], key="rt_dist")
            
            col_rt_btn1, col_rt_btn2 = st.columns(2)
            with col_rt_btn1:
                if st.button("🎲 Сгенерировать", use_container_width=True, key="gen_random_tau"):
                    try:
                        for i in range(N):
                            for j in range(N):
                                if st.session_state.network.M[j, i] != 0:
                                    if rt_dist == 'uniform':
                                        new_tau = np.random.uniform(rt_min, rt_max)
                                    else:
                                        new_tau = np.random.normal((rt_min+rt_max)/2, (rt_max-rt_min)/4)
                                        new_tau = max(rt_min, min(rt_max, new_tau))
                                    st.session_state.network.tau_syn[j, i] = 1.0 / new_tau
                        log_message(f"🎲 τ сгенерированы: [{rt_min}, {rt_max}] мс")
                        st.success("Константы сгенерированы!")
                        st.session_state.show_tau_random_dialog = False
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Ошибка: {str(e)}")
            with col_rt_btn2:
                if st.button("❌ Отмена", use_container_width=True, key="cancel_random_tau"):
                    st.session_state.show_tau_random_dialog = False
                    st.rerun()
        
        st.divider()
        
        # Быстрое редактирование
        st.markdown("### 🟡 Быстрое редактирование")
        
        col_sel1, col_sel2, col_sel3, col_sel4 = st.columns(4)
        with col_sel1:
            select_i = st.number_input("От (i)", min_value=0, max_value=N-1, value=0, key="edit_i")
        with col_sel2:
            select_j = st.number_input("До (j)", min_value=0, max_value=N-1, value=0, key="edit_j")
        with col_sel3:
            current_weight = st.session_state.network.W[int(select_j), int(select_i)] if st.session_state.network.M[int(select_j), int(select_i)] != 0 else 0
            st.metric("Вес", f"{current_weight:.2f}")
        with col_sel4:
            current_tau = 1.0 / st.session_state.network.tau_syn[int(select_j), int(select_i)] if st.session_state.network.M[int(select_j), int(select_i)] != 0 else 0
            st.metric("τ (мс)", f"{current_tau:.1f}")
        
        col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
        with col_btn1:
            new_weight = st.number_input("Новый вес", min_value=-10.0, max_value=10.0, value=float(current_weight), format="%.2f", key="new_w")
            if st.button("✏️ Вес", use_container_width=True, key="change_weight"):
                try:
                    i, j = int(select_i), int(select_j)
                    if st.session_state.network.M[j, i] != 0:
                        old_w = st.session_state.network.W[j, i]
                        coef = np.sign(st.session_state.network.M[j, i])
                        st.session_state.network.W[j, i] = abs(new_weight) * coef
                        log_message(f"✏️ Вес [{i}→{j}]: {old_w:.2f} → {st.session_state.network.W[j, i]:.2f}")
                        st.success(f"Вес обновлён: {i} → {j}")
                        st.rerun()
                    else:
                        st.warning("⚠️ Нет связи!")
                except Exception as e:
                    st.error(f"❌ Ошибка: {str(e)}")
        
        with col_btn2:
            new_tau = st.number_input("Новая τ (мс)", min_value=0.1, max_value=100.0, value=float(current_tau), format="%.1f", key="new_tau")
            if st.button("⏱ τ", use_container_width=True, key="change_tau"):
                try:
                    i, j = int(select_i), int(select_j)
                    if st.session_state.network.M[j, i] != 0 and new_tau > 0:
                        old_tau = 1.0 / st.session_state.network.tau_syn[j, i]
                        st.session_state.network.tau_syn[j, i] = 1.0 / new_tau
                        log_message(f"✏️ τ [{i}→{j}]: {old_tau:.1f} мс → {new_tau:.1f} мс")
                        st.success(f"τ обновлена: {i} → {j}")
                        st.rerun()
                    else:
                        st.warning("⚠️ Нет связи или τ <= 0!")
                except Exception as e:
                    st.error(f"❌ Ошибка: {str(e)}")
        
        with col_btn3:
            if st.button("🔌 Добавить", use_container_width=True, key="add_conn_matrix"):
                try:
                    i, j = int(select_i), int(select_j)
                    w = abs(new_weight)
                    coef = 1 if new_weight >= 0 else -1
                    st.session_state.network.connect(i, j, coef=coef, w=w, tau=10.0)
                    log_message(f"🔗 Добавлена связь: {i} → {j}")
                    st.success(f"Связь добавлена: {i} → {j}")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Ошибка: {str(e)}")
        
        with col_btn4:
            if st.button("🗑️ Удалить", use_container_width=True, key="del_conn_matrix"):
                try:
                    i, j = int(select_i), int(select_j)
                    st.session_state.network.M[j, i] = 0
                    st.session_state.network.W[j, i] = 0
                    st.session_state.network.tau_syn[j, i] = 0
                    log_message(f"🗑️ Удалена связь: {i} → {j}")
                    st.success(f"Связь удалена: {i} → {j}")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Ошибка: {str(e)}")
        
        st.divider()
        
       # ============================================================
    # СЕКЦИЯ 5: ВИЗУАЛИЗАЦИЯ
    # ============================================================

        st.markdown("### 📈 Визуализация")
    
        viz_tab1, viz_tab2 = st.tabs(["Матрица весов", "Матрица τ"])
        
        with viz_tab1:
            network_matrix_fig = create_weight_matrix(st.session_state.network, show_annotations_matrix, color_scale)
            if network_matrix_fig:
                st.plotly_chart(network_matrix_fig, use_container_width=True)
    
        with viz_tab2:
            # Визуализация tau матрицы
            tau_viz = np.zeros((N, N))
            for i in range(N):
                for j in range(N):
                    if st.session_state.network.M[j, i] != 0:
                        tau_viz[j, i] = 1.0 / st.session_state.network.tau_syn[j, i]
        
            fig_tau = go.Figure(data=go.Heatmap(
                z=tau_viz,
                x=list(range(N)),
                y=list(range(N)),
                colorscale='Greens',
                showscale=True,
                colorbar=dict(title='τ (мс)'),
                hovertemplate='<b>Связь: %{x} → %{y}</b><br>τ: %{z:.1f} мс<br><extra></extra>'
            ))
        
            fig_tau.update_layout(
                title={'text': f"⏱ Матрица констант релаксации ({N}×{N})", 'y':0.95, 'x':0.5},
                xaxis=dict(title='Пресинаптический нейрон (i)', tickmode='linear', dtick=1),
                yaxis=dict(title='Постсинаптический нейрон (j)', tickmode='linear', dtick=1, autorange='reversed'),
                height=400,
                template='plotly_white'
        )
            st.plotly_chart(fig_tau, use_container_width=True)
            
    # ============================================================
    # СЕКЦИЯ 6: СТАТИСТИКА
    # ============================================================
        st.divider()
        st.markdown("### 📊 Статистика")
        
        W = st.session_state.network.W
        st.session_state.network.M = np.sign(st.session_state.network.W)
        M = st.session_state.network.M
        TAU = np.zeros_like(W)
        for i in range(N):
            for j in range(N):
                if M[j, i] != 0:
                    TAU[j, i] = 1.0 / st.session_state.network.tau_syn[j, i]
        n_conn = np.count_nonzero(M)
        n_exc = np.sum(M > 0)
        n_inh = np.sum(M < 0)
        w_mean = np.mean(W[M != 0]) if n_conn > 0 else 0
        w_std = np.std(W[M != 0]) if n_conn > 0 else 0
        w_min = np.min(W[M != 0]) if n_conn > 0 else 0
        w_max = np.max(W[M != 0]) if n_conn > 0 else 0
        tau_mean = np.mean(TAU[M != 0]) if n_conn > 0 else 0
        tau_std = np.std(TAU[M != 0]) if n_conn > 0 else 0
        
        col_stat1, col_stat2, col_stat3, col_stat4, col_stat5 = st.columns(5)
        with col_stat1:
            st.metric("Всего связей", n_conn)
        with col_stat2:
            st.metric("Возбуждающих", n_exc)
        with col_stat3:
            st.metric("Тормозных", n_inh)
        with col_stat4:
            st.metric("Средний вес", f"{w_mean:.2f}")
        with col_stat5:
            st.metric("Средняя τ (мс)", f"{tau_mean:.1f}")
    
    
    # ========== ВКЛАДКА 4: ЛОГ ==========
    with tab4:
        st.subheader("📝 Лог событий")
        if st.session_state.log_messages:
            for msg in reversed(st.session_state.log_messages[-50:]):
                st.text(msg)
        else:
            st.info("Лог пуст")
        if st.button("🗑️ Очистить лог"):
            st.session_state.log_messages = []
            st.rerun()
    
    # ========== ВКЛАДКА 5: ПАРАМЕТРЫ ==========
    with tab5:
        st.subheader("⚙️ Параметры сети")
        
        st.write("**Параметры Izhikevich:**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("a (среднее)", f"{np.mean(st.session_state.network.a):.3f}")
        with col2:
            st.metric("b (среднее)", f"{np.mean(st.session_state.network.b):.3f}")
        with col3:
            st.metric("c (среднее)", f"{np.mean(st.session_state.network.c):.1f}")
        with col4:
            st.metric("d (среднее)", f"{np.mean(st.session_state.network.d):.1f}")
        
        # Список связей
        st.write("**Список связей:**")
        connections = []
        for i in range(N):
            for j in range(N):
                if st.session_state.network.M[j, i] != 0:
                    connections.append(f"{i} → {j} (вес: {st.session_state.network.W[j, i]:.2f}, тип: {'возбуждающий' if st.session_state.network.M[j, i] > 0 else 'тормозной'})")
        if connections:
            for conn in connections:
                st.write(f"- {conn}")
        else:
            st.info("Связей не найдено")
    
    # ========== ВКЛАДКА 6: ВХОДНЫЕ ТОКИ (НОВАЯ!) ==========
    with tab6:
        st.subheader("⚡ Вектор входных токов")
        st.markdown("Настройте индивидуальные входные токи для каждого нейрона сети.")
        
        # Отображение текущего вектора
        st.markdown("### 📊 Текущий вектор токов")
        input_fig = create_input_current_plot(st.session_state.input_current_vector, N)
        if input_fig:
            st.plotly_chart(input_fig, use_container_width=True)
        
        st.divider()
        
        # Редактирование вектора токов
        st.markdown("### 🔵 Редактирование вектора")
        
        input_df = pd.DataFrame(
            st.session_state.input_current_vector.reshape(-1, 1),
            index=[f"Нейрон {i}" for i in range(N)],
            columns=["Ток (мкА)"]
        )
        
        edited_input_df = st.data_editor(
            input_df,
            use_container_width=True,
            height=300,
            key="input_editor",
            num_rows="fixed",
            column_config={
                "Ток (мкА)": st.column_config.NumberColumn("Ток (мкА)", min_value=-100.0, max_value=100.0, step=0.5, format="%.2f")
            }
        )
        
        col_inp1, col_inp2, col_inp3 = st.columns(3)
        with col_inp1:
            if st.button("✅ Применить токи", use_container_width=True, key="apply_input"):
                try:
                    st.session_state.input_current_vector = edited_input_df["Ток (мкА)"].values
                    log_message(f"⚡ Вектор токов обновлён")
                    st.success("Вектор токов применён!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Ошибка: {str(e)}")
        
        with col_inp2:
            if st.button("🗑️ Сбросить токи", use_container_width=True, key="reset_input"):
                try:
                    st.session_state.input_current_vector = np.zeros(N)
                    log_message("🔄 Токи сброшены к 0")
                    st.success("Токи сброшены!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Ошибка: {str(e)}")
        
        with col_inp3:
            if st.button("🎲 Рандомные токи", use_container_width=True, key="random_input"):
                try:
                    st.session_state.show_input_random_dialog = True
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Ошибка: {str(e)}")
        
        # Диалог генерации случайных токов
        if st.session_state.get('show_input_random_dialog', False):
            st.info("🎲 **Генерация случайных входных токов**")
            col_ri1, col_ri2, col_ri3, col_ri4 = st.columns(4)
            with col_ri1:
                ri_min = st.number_input("Мин (мкА)", value=-10.0, format="%.1f", key="ri_min")
            with col_ri2:
                ri_max = st.number_input("Макс (мкА)", value=10.0, format="%.1f", key="ri_max")
            with col_ri3:
                ri_dist = st.selectbox("Распределение", ['uniform', 'normal'], key="ri_dist")
            with col_ri4:
                ri_std = st.number_input("Стд. откл.", value=2.0, format="%.1f", key="ri_std")
            
            col_ri_btn1, col_ri_btn2 = st.columns(2)
            with col_ri_btn1:
                if st.button("🎲 Сгенерировать", use_container_width=True, key="gen_random_input"):
                    try:
                        if ri_dist == 'uniform':
                            st.session_state.input_current_vector = np.random.uniform(ri_min, ri_max, N)
                        else:
                            st.session_state.input_current_vector = np.random.normal((ri_min+ri_max)/2, ri_std, N)
                        log_message(f"🎲 Токи сгенерированы: [{ri_min}, {ri_max}], {ri_dist}")
                        st.success("Токи сгенерированы!")
                        st.session_state.show_input_random_dialog = False
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Ошибка: {str(e)}")
            with col_ri_btn2:
                if st.button("❌ Отмена", use_container_width=True, key="cancel_random_input"):
                    st.session_state.show_input_random_dialog = False
                    st.rerun()
        
        st.divider()
        
        # Быстрое редактирование отдельного нейрона
        st.markdown("### 🟢 Быстрое редактирование")
        
        col_q1, col_q2, col_q3 = st.columns(3)
        with col_q1:
            select_neuron = st.number_input("Нейрон", min_value=0, max_value=N-1, value=0, key="input_neuron")
        with col_q2:
            current_input = st.session_state.input_current_vector[int(select_neuron)]
            st.metric("Текущий ток", f"{current_input:.2f} мкА")
        with col_q3:
            new_input = st.number_input("Новый ток (мкА)", min_value=-100.0, max_value=100.0, value=float(current_input), format="%.2f", key="new_input")
        
        col_q_btn1, col_q_btn2 = st.columns(2)
        with col_q_btn1:
            if st.button("✏️ Установить ток", use_container_width=True, key="set_input"):
                try:
                    st.session_state.input_current_vector[int(select_neuron)] = new_input
                    log_message(f"⚡ Ток нейрона {select_neuron}: {current_input:.2f} → {new_input:.2f} мкА")
                    st.success(f"Ток нейрона {select_neuron} установлен!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Ошибка: {str(e)}")
        
        with col_q_btn2:
            if st.button("🔁 Копировать на все", use_container_width=True, key="copy_input_all"):
                try:
                    st.session_state.input_current_vector = np.ones(N) * new_input
                    log_message(f"⚡ Ток {new_input:.2f} мкА применён ко всем нейронам")
                    st.success("Ток применён ко всем нейронам!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Ошибка: {str(e)}")
        
        st.divider()
        
        # Пресеты токов
        st.markdown("### 🟡 Пресеты входных токов")
        
        col_preset1, col_preset2, col_preset3, col_preset4 = st.columns(4)
        with col_preset1:
            if st.button("📊 Все нули", use_container_width=True, key="preset_zeros"):
                st.session_state.input_current_vector = np.zeros(N)
                log_message("📊 Токи установлены в 0")
                st.rerun()
        with col_preset2:
            if st.button("📈 Все положительные", use_container_width=True, key="preset_positive"):
                st.session_state.input_current_vector = np.ones(N) * 5.0
                log_message("📈 Токи установлены в 5.0 мкА")
                st.rerun()
        with col_preset3:
            if st.button("📉 Шахматный порядок", use_container_width=True, key="preset_checker"):
                st.session_state.input_current_vector = np.array([5.0 if i % 2 == 0 else 0.0 for i in range(N)])
                log_message("📉 Шахматный порядок токов")
                st.rerun()
        with col_preset4:
            if st.button("🎯 Только первый", use_container_width=True, key="preset_first"):
                st.session_state.input_current_vector = np.zeros(N)
                st.session_state.input_current_vector[0] = 10.0
                log_message("🎯 Ток только на первом нейроне")
                st.rerun()
        
        st.divider()
        
        # Статистика токов
        st.markdown("### 📊 Статистика вектора токов")
        
        input_mean = np.mean(st.session_state.input_current_vector)
        input_std = np.std(st.session_state.input_current_vector)
        input_min = np.min(st.session_state.input_current_vector)
        input_max = np.max(st.session_state.input_current_vector)
        n_active = np.sum(np.abs(st.session_state.input_current_vector) > 0.01)
        
        col_stat1, col_stat2, col_stat3, col_stat4, col_stat5 = st.columns(5)
        with col_stat1:
            st.metric("Средний ток", f"{input_mean:.2f} мкА")
        with col_stat2:
            st.metric("Стд. откл.", f"{input_std:.2f} мкА")
        with col_stat3:
            st.metric("Мин", f"{input_min:.2f} мкА")
        with col_stat4:
            st.metric("Макс", f"{input_max:.2f} мкА")
        with col_stat5:
            st.metric("Активных нейронов", n_active)

# Футер
st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Модель спайковой нейронной сети на модели Ижикевича</p>
    <p>ФТЛ им. П.Л. Капицы</p>
    <p>2026 г</p>
</div>
""", unsafe_allow_html=True)