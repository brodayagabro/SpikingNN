#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple example: Run Rybak2002 CPG network with visualization.

Demonstrates usage of run_simulation() function with Rybak_2002_network
and plots membrane potentials, muscle forces, and limb kinematics.

Простой пример: Запуск сети ЦПГ Rybak2002 с визуализацией.

Демонстрирует использование функции run_simulation() с классом 
Rybak_2002_network и строит графики мембранных потенциалов, 
активности мышц и кинематики конечности.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

# Import from SpikingNN package
from SpikingNN.Networks.Rybak2002 import Rybak_2002_network
from SpikingNN.Izh_net import *
from SpikingNN.Var_Limb import *
from dynamics import run_simulation


def create_constant_drive(
    flexor: float = 5.0, 
    extensor: float = 5.0
) -> Callable[[float], np.ndarray]:
    """
    Create constant external drive input function.
    
    Создает функцию постоянного внешнего управляющего сигнала.
    
    Parameters
    ----------
    flexor : float, optional
        Drive amplitude for flexor CPG neuron (nA).
        Амплитуда сигнала для нейрона ЦПГ сгибателя (нА).
    extensor : float, optional
        Drive amplitude for extensor CPG neuron (nA).
        Амплитуда сигнала для нейрона ЦПГ разгибателя (нА).
    
    Returns
    -------
    callable
        Function Iapp(t) returning array [flex_drive, ext_drive].
        Функция Iapp(t), возвращающая массив [сигнал_сгиб, сигнал_разгиб].
    """
    def Iapp(t: float) -> np.ndarray:
        return np.array([flexor, extensor])
    return Iapp


def plot_rybak_results(
    results: dict, 
    neuron_names: list,
    title: str = "Rybak 2002 CPG Simulation",
    save_path: str = None
) -> plt.Figure:
    """
    Plot simulation results: membrane potentials, muscle forces, limb dynamics.
    
    Визуализация результатов симуляции: мембранные потенциалы, 
    силы мышц, динамика конечности.
    
    Parameters
    ----------
    results : dict
        Dictionary from run_simulation with keys: T, V, F_flex, F_ext, q, w.
        Словарь из run_simulation с ключами: T, V, F_flex, F_ext, q, w.
    neuron_names : list
        List of neuron names for legend.
        Список имён нейронов для легенды.
    title : str, optional
        Figure title. Заголовок фигуры.
    save_path : str, optional
        If provided, save figure to this path.
        Если указан, сохранить фигуру по этому пути.
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure object with 3 subplots.
        Объект фигуры с 3 подграфиками.
    """
    T = results['T']
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    # 1. Membrane potentials (all neurons on one plot)
    # Мембранные потенциалы (все нейроны на одном графике)
    ax1 = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(neuron_names)))
    for i, name in enumerate(neuron_names):
        ax1.plot(T, results['V'][:, i], label=name, color=colors[i], linewidth=1)
    ax1.axhline(30, color='gray', linestyle='--', alpha=0.3, label='Spike threshold')
    ax1.set_ylabel('Membrane Potential (mV)')
    ax1.set_title('A) Neural Activity')
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(alpha=0.3)
    
    # 2. Muscle forces
    # Силы мышц
    ax2 = axes[1]
    ax2.plot(T, results['F_flex'], label='Flexor', color='tab:blue', linewidth=1.5)
    ax2.plot(T, results['F_ext'], label='Extensor', color='tab:red', linewidth=1.5)
    ax2.set_ylabel('Force (N)')
    ax2.set_title('B) Muscle Activity')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. Limb dynamics (angle and angular velocity)
    # Динамика конечности (угол и угловая скорость)
    ax3 = axes[2]
    ax3.plot(T, results['q'], label='Angle (rad)', color='tab:green', linewidth=1.5)
    ax3_twin = ax3.twinx()
    ax3_twin.plot(T, results['w'], label='Velocity (rad/s)', color='tab:orange', 
                  linewidth=1.5, alpha=0.8)
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Angle (rad)', color='tab:green')
    ax3_twin.set_ylabel('Velocity (rad/s)', color='tab:orange')
    ax3.set_title('C) Limb Kinematics')
    
    # Combine legends
    # Объединение легенд
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax3.grid(alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def main():
    """
    Run Rybak CPG simulation and visualize results.
    
    Запуск симуляции ЦПГ Рыбака и визуализация результатов.
    """
    # === 1. Initialize network ===
    # === 1. Инициализация сети ===
    print("Initializing Rybak 2002 network...")
    Flexor = SimpleAdaptedMuscle(w=0.5, N=1)
    Extensor = SimpleAdaptedMuscle(w=0.5, N=1)
    Flexor.tau_1 = 1/21
    Flexor.tau_c = 1/39
    Extensor.tau_1 = 1/21
    Extensor.tau_c = 1/39
    Pendulum = OneDOFLimb(
        q0=np.pi/2,
        w0=0.,
        b=0.01,
        a1=0.4,
        a2=0.05,
        m=0.3,
        l=0.3
    )
    Limb = Simple_Afferented_Limb(
        Limb=Pendulum,
        Flexor=Flexor,
        Extensor=Extensor
    )
    Qapp = np.zeros((12, 2))
    Qapp[1, 0] = Qapp[7, 1] = 1
    Net = Rybak_2002_network(
        input_size=2, 
        output_size=2, 
        afferent_size=6, 
        Qapp=Qapp, 
        exitatory_w=0.5, 
        inhibitory_w=-0.5
    )
    system = Var_Limb(Network=Net, Limb=Limb)
    system.set_afferents_by_names('Ia_Flex', 'Ia_IN_Flex', 10)
    # === 2. Setup simulation parameters ===
    # === 2. Настройка параметров симуляции ===
    duration = 5000  # ms
    dt = 0.1  # ms
    T = np.arange(0, duration, dt)
    
    # External drive: constant input to both CPG neurons
    # Внешний сигнал: постоянный вход на оба нейрона ЦПГ
    Iapp = create_constant_drive(flexor=5.0, extensor=1.0)
    
    # === 3. Run simulation ===
    # === 3. Запуск симуляции ===
    print(f"Running simulation: {len(T)} steps, dt={dt} ms...")
    results = run_simulation(
        system=system,
        T=T,
        Iapp=Iapp,
        record_vars=["V", "q", "w", "F_flex", "F_ext"]
    )
    
    # === 4. Visualize results ===
    # === 4. Визуализация результатов ===
    print("Generating plots...")
    neuron_names = system.names['neurons']  # ['CPG_Flex', 'MN_Flex', 'CPG_Ext', 'MN_Ext']
    
    fig = plot_rybak_results(
        results=results,
        neuron_names=neuron_names,
        title="Rybak 2002: Half-Center CPG with Limb Feedback",
        save_path="rybak_simple_demo.png"
    )
    
    plt.show()
    
    # === 5. Print summary ===
    # === 5. Печать сводки ===
    print("\n=== Simulation Summary ===")
    print(f"Duration: {T[-1]:.1f} ms")
    print(f"Mean flexor force: {np.mean(results['F_flex']):.4f} N")
    print(f"Mean extensor force: {np.mean(results['F_ext']):.4f} N")
    print(f"Angle range: [{results['q'].min():.3f}, {results['q'].max():.3f}] rad")
    print(f"Velocity range: [{results['w'].min():.3f}, {results['w'].max():.3f}] rad/s")
    
    # Count spikes per neuron
    # Подсчёт спайков для каждого нейрона
    print("\nSpike counts (threshold = 30 mV):")
    for i, name in enumerate(neuron_names):
        V_trace = results['V'][:, i]
        spikes = np.sum(np.diff(np.signbit(30 - V_trace)) > 0)
        print(f"  {name}: {spikes}")


if __name__ == "__main__":
    main()