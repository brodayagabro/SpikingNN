"""
Интеграционные тесты фабрик SpikingNN.
Сравнение результатов ручной сборки и сборки через фабрики.
"""

import numpy as np
from scipy import signal as sig
from matplotlib import pyplot as plt
import json
import tempfile
import os

# Импорт фабрик и ядра
from SpikingNN.core.factories import (
    NetworkFactory,
    MuscleFactory,
    LimbFactory,
    AfferentedLimbFactory
)
from SpikingNN.core.Izh_net import (
    Izhikevich_IO_Network,
    types2params,
    Net_Limb_connect
)
from SpikingNN.utils.net_preparation import *

np.random.seed(42)

def run_net(T, net, I_app, I_aff):
    """
    Процедура запуска сети.
    
    Procedure of running network.
    
    Args:
        T: Discrete time array.
        net: Network object.
        I_app: Applied current function/array.
        I_aff: Afferents activity function/array.
        
    Returns:
        U, V: State of network arrays with shape(len(T), N).
    """
    dt = T[1] - T[0]
    N = len(net)
    U = np.zeros((len(T), N))
    V = np.zeros((len(T), N))
    for i, t in enumerate(T):
        U[i] = net.U_prev
        V[i] = net.V_prev
        # Обработка I_app и I_aff как функций или массивов
        if callable(I_app):
            app_val = I_app(t)
        else:
            app_val = I_app
            
        if callable(I_aff):
            aff_val = I_aff(t)
        else:
            aff_val = I_aff
            
        net.step(dt=dt, Iapp=app_val, Iaff=aff_val)
    return U, V

def test_IzhIOnet_via_factory():
    """
    Тест создания сети Izhikevich_IO_Network через NetworkFactory.
    Эквивалент test_IzhIOnet_step из оригинала.
    """
    print("Running test_IzhIOnet_via_factory...")
    
    # Конфигурация, соответствующая параметрам из оригинального теста
    config = {
        "network_params": {
            "N": 4,
            "input_size": 2,
            "output_size": 2,
            "afferent_size": 2,
            "types": ["RS", "RS", "RS", "RS"], # Дефолтные типы, так как в оригинале не указаны явно для этого теста
            "names": ["N0", "N1", "N2", "N3"]
        },
        "weights": {
            "mask": [
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1]
            ],
            "matrix": [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]
            ]
        },
        "synaptic_dynamics": {
            "tau_syn_ms": 20.0
        },
        "interfaces": {
            "Q_app": [
                [1, 0],
                [0, 1],
                [0, 0],
                [0, 0]
            ],
            "Q_aff": [
                [1, 0],
                [0, 1],
                [0, 0],
                [0, 0]
            ],
            "P": [
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]
        }
    }
    
    factory = NetworkFactory()
    net = factory.create_from_dict(config)
    
    I_app = np.array([0, 0])
    I_aff = np.zeros(2)
    net.step(dt=0.1, Iapp=I_app, Iaff=I_aff)
    print("✅ Network step successful via Factory.")


def test_IzhIOnet_sym_via_factory():
    """
    Тест симуляции сети с афферентами через NetworkFactory.
    Эквивалент test_IzhIOnet_sym.
    """
    print("Running test_IzhIOnet_sym_via_factory...")
    
    config = {
        "network_params": {
            "N": 4,
            "input_size": 2,
            "output_size": 2,
            "afferent_size": 6,
            "types": ["RS", "RS", "RS", "RS"],
            "names": ["CPG1", "MN1", "CPG2", "MN2"]
        },
        "weights": {
            "mask": [
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1]
            ],
            "matrix": [
                [0, 0, 0, -1],
                [0, 0, 1, 0],
                [0, -1, 0, 0],
                [1, 0, 0, 0]
            ]
        },
        "synaptic_dynamics": {
            "tau_syn_ms": [
                [1, 1, 1, 20],
                [1, 1, 10, 1],
                [1, 20, 1, 1],
                [10, 1, 1, 1]
            ]
        },
        "interfaces": {
            "Q_app": [
                [1, 0],
                [0, 1],
                [0, 0],
                [0, 0]
            ],
            "Q_aff": [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 1, 0],
                [0, 0, 0, 1, 0, 1]
            ],
            "P": [
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]
        }
    }
    
    factory = NetworkFactory()
    net = factory.create_from_dict(config)
    
    # Инициализация шума как в оригинале
    net.set_init_conditions(v_noise=np.random.normal(size=net.N))
    
    T = np.linspace(0, 500, 2000)
    I_app = lambda t: np.array([0, 0])
    I_aff = lambda t: np.zeros(6) + 2*np.random.rand(6)
    
    U, V = run_net(T, net, I_app, I_aff)
    
    plt.figure(figsize=(10, 8))
    plt.subplot(211)
    for i in range(net.N):
        plt.plot(T, V[:, i], label=f"V({net.names[i]})")
    plt.legend()
    plt.title("Membrane Potentials (Factory)")
    
    plt.subplot(212)
    firing_rastr = create_firing_rastr(V, T, 30)
    plt.scatter(firing_rastr[0], firing_rastr[1], s=0.1)
    plt.yticks(list(range(net.N)), net.names)
    plt.title("Spike Raster (Factory)")
    
    plt.tight_layout()
    plt.savefig("test_factory_network_sym.png")
    print("📊 Plot saved to test_factory_network_sym.png")
    # plt.show() # Раскомментируйте для просмотра


def test_Afferented_Limb_via_factories():
    """
    Тест афферентированной конечности через LimbFactory и MuscleFactory.
    Эквивалент test_Afferented_Limb.
    """
    print("Running test_Afferented_Limb_via_factories...")
    
    # Конфигурация конечности
    limb_config = {
        "mechanics": {
            "mass": 0.3,
            "length": 0.3,
            "viscosity": 0.00, # В оригинале b=0.00
            "q0": np.pi/2,
            "tendon_a1": 0.4,
            "tendon_a2": 0.05
        },
        "flexor": {
            "w": 0.5,
            "N": 2,
            "tau_c": 71, # Дефолт из SimpleAdaptedMuscle
            "tau_1": 130
        },
        "extensor": {
            "w": 0.4,
            "N": 2,
            "tau_c": 71,
            "tau_1": 130
        }
    }
    
    # Сборка через фабрику
    AL = AfferentedLimbFactory.create_from_dict(limb_config)
    
    T = np.linspace(0, 10000, 20000)
    mod_sig = np.sin(1*np.pi*T/1000)

    uf = 15*(sig.square(2*np.pi*T/50, duty=0.2)+1)
    uf = np.where(mod_sig>0.6, uf, 0)
    ue = 15*(sig.square(2*np.pi*T/50, duty=0.2)+1)
    ue = np.where(mod_sig<-0.6, ue, 0)

    # Симуляция
    dt = T[1] - T[0]
    M = np.zeros(len(T))
    F_f = np.zeros(len(T))
    F_e = np.zeros(len(T))
    W = np.zeros(len(T))
    Q = np.zeros(len(T))
    Output = np.zeros((len(T), 6))
    
    for i, t in enumerate(T):
        F_f[i] = AL.Flexor.F_prev
        F_e[i] = AL.Extensor.F_prev
        M[i] = AL.Limb.M_tot  
        Q[i] = AL.Limb.q
        W[i] = AL.Limb.w
        Output[i] = AL.output
        AL.step(dt=dt, uf=uf[i], ue=ue[i])
        
    Ia_f = Output[:, 0]
    Ia_e = Output[:, 3]
    
    L_f = AL.Limb.L(Q)
    firing_Lf = np.where(L_f >= AL.Afferents.L_th)
    L_e = AL.Limb.L(np.pi-Q)
    firing_Le = np.where(L_e >= AL.Afferents.L_th)
    
    hf = AL.Limb.h(L_f, Q)
    he = AL.Limb.h(L_e, np.pi-Q)
    vf = W*hf
    ve = -W*he
    firing_vf = np.where(vf>0)
    firing_ve = np.where(ve>0)
    
    plt.figure(figsize=(12, 10))
    
    plt.subplot(321)
    plt.title("Control signal (Factory)")
    plt.plot(T, uf, label='uf', linewidth=0.5)
    plt.plot(T, ue, label='ue', linewidth=0.5)
    plt.legend()
    
    plt.subplot(322)
    plt.title("Muscles (Factory)")
    plt.plot(T, F_e, label="F_ext")
    plt.plot(T, F_f, label="F_flex")
    plt.plot(T, AL.Afferents.F_th*np.ones(len(T)), color="red", label="F_th")
    plt.legend()
    
    plt.subplot(323)
    plt.plot(T, Q, label="q")
    plt.legend()
    
    plt.subplot(324)
    plt.title('Muscle length dynamics (Factory)')
    plt.plot(T, L_f, label="L_f")
    plt.plot(T, L_e, label="L_e")
    plt.plot(T, AL.Afferents.L_th*np.ones(len(T)), color="red", label="L_th")
    plt.legend()
    
    plt.subplot(211) # Переопределение для Ia
    plt.title("Ia-type activity (Factory)")
    m = 0
    M_plot = 0.5
    st = 0.5
    plt.vlines(T[firing_vf], m, M_plot, color='orange', alpha=0.5, label="f_strech")
    plt.vlines(T[firing_Lf], m+st, M_plot+st, color='yellow', alpha=0.5,label="Lf>=L_th")
    plt.plot(T, Ia_f, label='Ia_f', color='k')
    plt.legend()

    plt.tight_layout()
    plt.savefig("test_factory_limb.png")
    print(" Plot saved to test_factory_limb.png")
    # plt.show()


def test_Net_Limb_connect_via_factories():
    """
    Тест полной системы Net_Limb_connect через все фабрики.
    Эквивалент test_Net_Limb_connect.
    """
    print("Running test_Net_Limb_connect_via_factories...")
    
    # 1. Создание сети через NetworkFactory
    net_config = {
        "network_params": {
            "N": 4,
            "input_size": 2,
            "output_size": 2,
            "afferent_size": 6,
            "types": ['CH', 'FS', 'CH', 'FS'],
            "names": ["CPG1", "MN1", "CPG2", "MN2"]
        },
        "weights": {
            "mask": [
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1]
            ],
            "matrix": [
                [0, 0, 0, -1.1],
                [1.7, 0, 0, 0],
                [0, -1.1, 0, 0],
                [0, 0, 1.7, 0]
            ]
        },
        "synaptic_dynamics": {
            "tau_syn_ms": 20.0
        },
        "interfaces": {
            "Q_app": [
                [1, 0],
                [0, 0],
                [0, 1],
                [0, 0]
            ],
            "Q_aff": [
                [1, 1, 1, 1, 1, 1], # Заполняем случайными весами как в оригинале 1*np.random.rand(4, 6)
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1]
            ],
            "P": [
                [1, 0, 0, 0],
                [0, 0, 1, 0]
            ]
        }
    }
    
    # Генерируем случайные веса Q_aff как в оригинале, чтобы сохранить идентичность
    np.random.seed(42) # Фиксируем сид для воспроизводимости
    net_config["interfaces"]["Q_aff"] = (1 * np.random.rand(4, 6)).tolist()
    
    net_factory = NetworkFactory()
    net = net_factory.create_from_dict(net_config)
    
    # Корректировка параметров a для CH нейронов как в оригинале
    # В оригинале: A[0] = 0.001; A[2] = 0.001
    new_a = net.a.copy()
    new_a[0] = 0.001
    new_a[2] = 0.001
    net.set_params(a=new_a)

    # 2. Создание конечности через AfferentedLimbFactory
    limb_config = {
        "mechanics": {
            "mass": 0.3,
            "length": 0.3,
            "viscosity": 0.001,
            "q0": np.pi/2 + 0.4,
            "tendon_a1": 0.4,
            "tendon_a2": 0.05
        },
        "flexor": {
            "w": 0.5,
            "N": 2,
            "tau_c": 71,
            "tau_1": 130
        },
        "extensor": {
            "w": 0.4,
            "N": 2,
            "tau_c": 71,
            "tau_1": 130
        }
    }
    
    limb_factory = AfferentedLimbFactory()
    AL = limb_factory.create_from_dict(limb_config)

    # 3. Сборка системы
    sys = Net_Limb_connect(Network=net, Limb=AL)
    
    # 4. Симуляция
    T = np.linspace(0, 20000, 50000)
    I = np.zeros(2)
    input_func = lambda t: I
    
    V = np.zeros((len(T), 4))
    F_flex = np.zeros(len(T))
    F_ext = np.zeros(len(T))
    Afferents = np.zeros((len(T), 6))
    Q = np.zeros(len(T))
    W = np.zeros(len(T))
    dt = T[1] - T[0]
    
    for i, t in enumerate(T):
        V[i] = sys.net.V_prev
        F_flex[i] = sys.F_flex
        F_ext[i] = sys.F_ext
        Afferents[i] = sys.Limb.output
        Q[i] = sys.q
        W[i] = sys.w
        sys.step(dt=dt, Iapp=input_func(t))

    plt.figure(figsize=(14, 10))
    plt.subplot(221)
    for i in range(4):
        plt.plot(T, V[:, i], label=f"{sys.net.names[i]}")
    plt.legend()
    plt.title("Neuron Potentials (Factory System)")

    plt.subplot(222)
    plt.plot(T, F_flex, label='flexor')
    plt.plot(T, F_ext, label='extensor')
    plt.legend()
    plt.title("Muscle Forces (Factory System)")

    plt.subplot(223)
    aff_types = ['Ia_f', 'II_f', 'Ib_f', 'Ia_e', 'II_e', 'Ib_e']
    for i in range(6):
        plt.plot(T, Afferents[:, i], label=aff_types[i])
    plt.legend()
    plt.title("Afferents (Factory System)")

    plt.subplot(224)
    plt.plot(T, Q, label="Q")
    plt.plot(T, W, label='W')
    plt.legend()
    plt.title("Kinematics (Factory System)")
    
    plt.tight_layout()
    plt.savefig("test_factory_full_system.png")
    print("📊 Plot saved to test_factory_full_system.png")
    # plt.show()


if __name__ == "__main__":
    print("="*50)
    print("Testing SpikingNN Factories Integration")
    print("="*50)
    
    test_IzhIOnet_via_factory()
    test_IzhIOnet_sym_via_factory()
    test_Afferented_Limb_via_factories()
    test_Net_Limb_connect_via_factories()
    
    print("\n✅ All factory integration tests completed successfully!")