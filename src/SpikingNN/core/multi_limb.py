"""
Модуль для моделирования нейромеханических систем с несколькими конечностями.
Module for modeling neuromechanical systems with multiple limbs.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from SpikingNN.core.Izh_net import (
    Izhikevich_IO_Network, 
    Afferented_Limb, 
    SimpleAdaptedMuscle, 
    OneDOFLimb,
    types2params
)


class MultiLimbSystem:
    """
    Нейромеханическая система с несколькими конечностями и единой сетью.
    
    Neuromechanical system with multiple limbs and a single network.
    
    Attributes:
        net (Izhikevich_IO_Network): Центральная сеть нейронов.
        limbs (List[Afferented_Limb]): Список объектов конечностей.
        limb_names (List[str]): Имена конечностей.
        
    Args:
        network (Izhikevich_IO_Network): Готовая нейронная сеть.
        limbs (List[Afferented_Limb]): Список готовых объектов конечностей.
        names (Optional[List[str]]): Опциональный список имен для конечностей.
    """
    
    def __init__(self, network: Izhikevich_IO_Network, limbs: List[Afferented_Limb], names: Optional[List[str]] = None):
        """
        Инициализация системы.
        
        Initialization of the system.
        
        Args:
            network: Экземпляр сети Ижикевича.
            limbs: Список экземпляров Afferented_Limb.
            names: Опциональный список имен для конечностей.
        """
        self.net = network
        self.limbs = limbs
        
        if names is None:
            self.limb_names = [f"limb_{i}" for i in range(len(limbs))]
        else:
            self.limb_names = names
            
        # Проверка соответствия размеров
        expected_aff_size = len(self.limbs) * 6
        if self.net.afferent_size != expected_aff_size:
            raise ValueError(
                f"Network afferent_size ({self.net.afferent_size}) mismatch. "
                f"Expected {expected_aff_size} for {len(self.limbs)} limbs."
            )
            
        expected_out_size = len(self.limbs) * 2
        if self.net.output_size != expected_out_size:
            raise ValueError(
                f"Network output_size ({self.net.output_size}) mismatch. "
                f"Expected {expected_out_size} for {len(self.limbs)} limbs."
            )

    def get_combined_afferents(self) -> np.ndarray:
        """
        Объединяет афферентные сигналы всех конечностей в один вектор.
        
        Combines afferent signals from all limbs into a single vector.
        
        Returns:
            np.ndarray: Вектор афферентов размером (total_afferent_size,).
        """
        if not self.limbs:
            return np.zeros(self.net.afferent_size)
            
        afferents = []
        for limb in self.limbs:
            afferents.extend(limb.output.tolist())
        return np.array(afferents)
        
    def step(self, dt: float = 0.1, Iapp: np.ndarray = None):
        """
        Один шаг симуляции всей системы.
        
        One simulation step for the entire system.
        
        Args:
            dt: Шаг по времени в мс.
            Iapp: Внешний входной ток размером (input_size,).
        """
        if Iapp is None:
            Iapp = np.zeros(self.net.input_size)
            
        # 1. Сбор афферентов от ВСЕХ конечностей
        Iaff_total = self.get_combined_afferents()
        
        # 2. Шаг нейронной сети
        self.net.step(dt=dt, Iapp=Iapp, Iaff=Iaff_total)
        
        # 3. Распределение выходов сети по конечностям
        V_out = self.net.V_out
        
        for i, limb in enumerate(self.limbs):
            uf = V_out[i * 2]      # Выход на сгибатель i-й конечности
            ue = V_out[i * 2 + 1]  # Выход на разгибатель i-й конечности
            limb.step(dt=dt, uf=uf, ue=ue)
            
    def get_state(self) -> Dict[str, Any]:
        """
        Возвращает полное состояние системы.
        
        Returns full system state.
        
        Returns:
            Dict: Словарь с состоянием нейронов и механикой всех конечностей.
        """
        state = {
            "neurons_V": self.net.V_prev.tolist(),
            "neurons_U": self.net.U_prev.tolist(),
            "limbs": {}
        }
        
        for name, limb in zip(self.limb_names, self.limbs):
            state["limbs"][name] = {
                "angle_q": float(limb.q),
                "angular_velocity_w": float(limb.w),
                "force_flex": float(limb.F_flex),
                "force_ext": float(limb.F_ext),
                "afferents": limb.output.tolist()
            }
            
        return state


"""
Ручная сборка и тест системы из 4 конечностей.
Manual assembly and test of a 4-limb system.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm as pbar
from SpikingNN.core.multi_limb import MultiLimbSystem
from SpikingNN.core.Izh_net import (
    Izhikevich_IO_Network,
    Afferented_Limb,
    SimpleAdaptedMuscle,
    OneDOFLimb,
    types2params
)

def create_limb(name: str) -> Afferented_Limb:
    """
    Helper function to create a standard limb.
    """
    flexor = SimpleAdaptedMuscle(w=0.5, N=5, tau_c=39, tau_1=21)
    extensor = SimpleAdaptedMuscle(w=0.5, N=5, tau_c=39, tau_1=21)
    # Начальный угол pi/2 (вертикально вниз)
    limb_mech = OneDOFLimb(m=0.3, ls=0.3, b=0.01, q0=np.pi/2, a1=0.4, a2=0.05)
    return Afferented_Limb(Limb=limb_mech, Flexor=flexor, Extensor=extensor)

def run_4_limbs_simulation():
    """
    Запуск симуляции для 4 конечностей.
    Run simulation for 4 limbs.
    """
    print("🚀 Initializing 4-limb system...")
    
    num_limbs = 4
    neurons_per_limb = 4  # CPG_F, MN_F, CPG_E, MN_E
    N = num_limbs * neurons_per_limb
    input_size = num_limbs  # Ток на каждый CPG_F
    output_size = num_limbs * 2  # Flex и Ext для каждой ноги
    afferent_size = num_limbs * 6
    
    # 1. Создаем имена и типы нейронов
    names = []
    types = []
    for i in range(num_limbs):
        names.extend([f"CPG{i}_F", f"MN{i}_F", f"CPG{i}_E", f"MN{i}_E"])
        types.extend(['CH', 'FS', 'CH', 'FS'])
        
    A, B, C, D = types2params(types)
    
    # 2. Создаем матрицы соединений
    Q_app = np.zeros((N, input_size))
    for i in range(num_limbs):
        Q_app[i * neurons_per_limb, i] = 1.0  # Ток на CPG_F каждой ноги
        
    Q_aff = np.zeros((N, afferent_size))  # Пока без обратной связи для стабильности
    
    P = np.zeros((output_size, N))
    for i in range(num_limbs):
        P[2*i, 4*i + 1] = 1.0      # MN_F -> Flex
        P[2*i + 1, 4*i + 3] = 1.0  # MN_E -> Ext
        
    W = np.zeros((N, N))
    for i in range(num_limbs):
        base = i * neurons_per_limb
        # CPG_F -> MN_F (excitation)
        W[base + 1, base] = 1.5
        # CPG_E -> MN_E (excitation)
        W[base + 3, base + 2] = 1.5
        # CPG_F <-> CPG_E (mutual inhibition)
        W[base + 2, base] = -1.5
        W[base, base + 2] = -1.5
        
    tau_syn = 20 * np.ones((N, N))
    
    # 3. Создаем сеть
    net = Izhikevich_IO_Network(
        N=N, input_size=input_size, output_size=output_size,
        afferent_size=afferent_size, names=names,
        Q_app=Q_app, Q_aff=Q_aff, P=P
    )
    net.set_params(a=A, b=B, c=C, d=D)
    net.M = np.ones((N, N))
    net.set_weights(W)
    net.set_synaptic_relax_constant(tau_syn)
    
    # 4. Создаем 4 конечности вручную
    limbs = []
    limb_names = ["Front_Left", "Front_Right", "Hind_Left", "Hind_Right"]
    for name in limb_names:
        limbs.append(create_limb(name))
        
    # 5. Собираем систему
    system = MultiLimbSystem(network=net, limbs=limbs, names=limb_names)
    
    # 6. Параметры симуляции
    T_sim = 2000  # ms
    dt = 0.1      # ms
    steps = int(T_sim / dt)
    
    # Входной ток: стимулируем все CPG_F одинаково
    Iapp = np.ones(input_size) * 5.0 + np.random.randint(5, size=input_size) 
    
    # Буферы для сохранения данных
    time_vec = np.arange(steps) * dt
    V_hist = np.zeros((steps, N))
    
    # Для мышц и углов создаем словари или массивы
    F_flex_hist = {name: np.zeros(steps) for name in limb_names}
    F_ext_hist = {name: np.zeros(steps) for name in limb_names}
    q_hist = {name: np.zeros(steps) for name in limb_names}
    
    print("✅ System initialized. Starting simulation loop...")
    
    for i in pbar(range(steps)):
        # Сохраняем состояние
        V_hist[i] = system.net.V_prev
        
        for j, name in enumerate(limb_names):
            F_flex_hist[name][i] = system.limbs[j].F_flex
            F_ext_hist[name][i] = system.limbs[j].F_ext
            q_hist[name][i] = system.limbs[j].q
            
        # Шаг системы
        system.step(dt=dt, Iapp=Iapp)
        
        if i % 500 == 0 and np.any(np.isnan(system.net.V_prev)):
            print(f"⚠️ NaN detected at step {i}! Stopping.")
            break
            
    print("✅ Simulation finished.")
    
    # 7. Визуализация
    fig = plt.figure(figsize=(16, 12))
    
    # --- График 1: Растр активности нейронов (Spike Raster) ---
    ax1 = plt.subplot(3, 1, 1)
    spikes_found = False
    for neuron_idx in range(N):
        # Находим моменты спайков (V > 20 mV)
        spike_times = time_vec[V_hist[:, neuron_idx] > 20]
        if len(spike_times) > 0:
            spikes_found = True
            # Рисуем вертикальные линии для каждого спайка
            ax1.vlines(spike_times, neuron_idx - 0.4, neuron_idx + 0.4, color='black', linewidth=0.5)
            
    if spikes_found:
        ax1.set_yticks(range(N))
        ax1.set_yticklabels(names, fontsize=8)
        ax1.set_title("Neuron Spike Raster Plot")
        ax1.set_ylabel("Neuron Index")
    else:
        ax1.text(0.5, 0.5, 'No spikes detected', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
        
    ax1.grid(True, axis='x')
    ax1.set_ylim(-1, N)
    
    # --- График 2: Силы всех мышц ---
    ax2 = plt.subplot(3, 1, 2)
    colors = plt.cm.tab10(np.linspace(0, 1, num_limbs * 2))
    color_idx = 0
    for name in limb_names:
        ax2.plot(time_vec, F_flex_hist[name], label=f"{name}_Flex", color=colors[color_idx], linestyle='-')
        ax2.plot(time_vec, F_ext_hist[name], label=f"{name}_Ext", color=colors[color_idx+1], linestyle='--')
        color_idx += 2
        
    ax2.set_title("Muscle Forces for All Limbs")
    ax2.set_ylabel("Force (N)")
    ax2.legend(ncol=4, fontsize=8, loc='upper right')
    ax2.grid(True)
    
    # --- График 3: Углы всех конечностей ---
    ax3 = plt.subplot(3, 1, 3)
    color_idx = 0
    for name in limb_names:
        ax3.plot(time_vec, q_hist[name], label=name, color=colors[color_idx], linewidth=1.5)
        color_idx += 2 # Используем те же цвета, что и для флексоров
        
    ax3.set_title("Limb Angles for All Limbs")
    ax3.set_ylabel("Angle (rad)")
    ax3.set_xlabel("Time (ms)")
    ax3.legend(ncol=4, fontsize=8, loc='upper right')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig("4_limbs_manual_test.png", dpi=150)
    print("📊 Plot saved to 4_limbs_manual_test.png")
    plt.show()

if __name__ == "__main__":
    run_4_limbs_simulation()
