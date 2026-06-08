import numpy as np
import matplotlib.pyplot as plt
import json
import tempfile
import os
from SpikingNN.core.multy_limb import (
    MultiLimbSystem, 
    create_network_from_config, 
    create_multi_limb_system_from_json
)

def test_multi_limb_creation():
    """Тест создания и симуляции системы."""
    config = {
        "network": {
            "N": 8,
            "input_size": 4,
            "output_size": 4,
            "afferent_size": 12,
            "types": ["CH", "FS", "CH", "FS", "CH", "FS", "CH", "FS"],
            "names": ["CPG1", "MN1_F", "CPG2", "MN1_E", 
                      "CPG3", "MN2_F", "CPG4", "MN2_E"],
            "Q_app": [[1,0,0,0], [0,0,0,0], [0,1,0,0], [0,0,0,0],
                      [0,0,1,0], [0,0,0,0], [0,0,0,1], [0,0,0,0]],
            "Q_aff": (0.5 * np.random.rand(8, 12)).tolist(),
            "P": [[1,0,0,0,0,0,0,0],
                  [0,1,0,0,0,0,0,0],
                  [0,0,0,0,1,0,0,0],
                  [0,0,0,0,0,1,0,0]],
            "W": [
                [0, 0, 0, -1.5, 0, 0, 0, 0],
                [1.5, 0, 0, 0, 0, 0, 0, 0],
                [0, -1.5, 0, 0, 0, 0, 0, 0],
                [0, 0, 1.5, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, -1.5],
                [0, 0, 0, 0, 1.5, 0, 0, 0],
                [0, 0, 0, 0, 0, -1.5, 0, 0],
                [0, 0, 0, 0, 0, 0, 1.5, 0]
            ],
            "tau_syn": (20 * np.ones((8,8))).tolist()
        },
        "limbs": [
            {"name": "left_leg", "params": {"mass": 0.3, "length": 0.3}},
            {"name": "right_leg", "params": {"mass": 0.3, "length": 0.3}}
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f)
        tmp_path = f.name
        
    try:
        system = create_multi_limb_system_from_json(tmp_path)
        print("✅ System created successfully")
        
        T_sim = 2000
        dt = 0.1
        steps = int(T_sim / dt)
        Iapp_const = np.array([5.0, 0.0, 5.0, 0.0])
        
        states = []
        times = []
        
        for i in range(steps):
            t = i * dt
            system.step(dt=dt, Iapp=Iapp_const)
            if i % 10 == 0:
                states.append(system.get_state())
                times.append(t)
                
        # Визуализация
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        V_data = np.array([s["neurons_V"] for s in states])
        for i, name in enumerate(system.net.names):
            axes[0].plot(times, V_data[:, i], label=name, alpha=0.8, linewidth=0.8)
        axes[0].set_title("Neuron Membrane Potentials (mV)")
        axes[0].legend(fontsize=8, ncol=2)
        axes[0].grid(True)
        
        for name in system.limb_names:
            q_data = [s["limbs"][name]["angle_q"] for s in states]
            axes[1].plot(times, q_data, label=name, linewidth=1.5)
        axes[1].set_title("Limb Angles (rad)")
        axes[1].legend()
        axes[1].grid(True)
        
        for name in system.limb_names:
            ff = [s["limbs"][name]["force_flex"] for s in states]
            fe = [s["limbs"][name]["force_ext"] for s in states]
            axes[2].plot(times, ff, label=f"{name}_Flex", linestyle='--')
            axes[2].plot(times, fe, label=f"{name}_Ext")
        axes[2].set_title("Muscle Forces (N)")
        axes[2].legend()
        axes[2].grid(True)
        axes[2].set_xlabel("Time (ms)")
        
        plt.tight_layout()
        plt.savefig("multi_limb_fixed_test.png", dpi=150)
        print("📊 Visualization saved to multi_limb_fixed_test.png")
        plt.show()
        
    finally:
        os.unlink(tmp_path)

if __name__ == "__main__":
    test_multi_limb_creation()
