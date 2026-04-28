"""
Streamlit GUI for Neuromechanical System Configuration & Simulation.
Features:
- Per-neuron presets & Izhikevich parameters
- Per-neuron input current configuration
- JSON import/export
- Fixed simulation runner & robust plotting
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

from SpikingNN.Izh_net import (
    Izhikevich_IO_Network, Afferented_Limb, OneDOFLimb, 
    SimpleAdaptedMuscle, Afferents
)
from SpikingNN.Var_Limb import Var_Limb
from src.SpikingNN.Networks.Rybak2002 import Rybak_2002_network
from core.dynamics import run_simulation
from core.io_json import save_system_json, load_system_json, dict_to_system, system_to_dict

# Izhikevich preset lookup: [a, b, c, d]
NEURON_PRESETS = {
    'RS': [0.02, 0.20, -65.0, 8.00], 'IB': [0.02, 0.20, -55.0, 4.00],
    'CH': [0.02, 0.20, -50.0, 2.00], 'FS': [0.10, 0.20, -65.0, 0.05],
    'TC': [0.02, 0.25, -65.0, 0.05], 'RZ': [0.10, 0.26, -65.0, 8.00],
    'LTS': [0.02, 0.25, -65.0, 2.00]
}

st.set_page_config(page_title="NeuroMech CPG Configurator", layout="wide")

# ---------------- Session State Initialization ----------------
if 'config' not in st.session_state:
    default_neurons = [
        {'name': f'N{i}', 'type': 'RS', 'a': 0.02, 'b': 0.2, 'c': -65, 'd': 2, 'Iapp': 0.0}
        for i in range(12)
    ]
    st.session_state.config = {
        'N': 12,
        'neurons': default_neurons,
        'limb': {'m': 0.3, 'ls': 0.3, 'b': 0.002, 'a1': 0.4, 'a2': 0.05, 'q0': np.pi/2, 'w0': 0.0, 'g': 9.81},
        'muscles': {
            'flexor': {'w': 0.5, 'A': 0.0074, 'N': 10, 'tau_c': 71, 'tau_1': 130, 'm': 2.5, 'k': 0.75},
            'extensor': {'w': 0.4, 'A': 0.0074, 'N': 10, 'tau_c': 71, 'tau_1': 130, 'm': 2.5, 'k': 0.75}
        },
        'afferents': {'p_v': 0.6, 'k_v': 6.2, 'k_dI': 2.0, 'k_dII': 1.5, 'k_nI': 0.06, 
                      'k_nII': 0.06, 'k_f': 1.0, 'L_th': 0.059, 'F_th': 3.38, 'const_I': 0, 'const_II': 0},
        'sim': {'T_max': 5000, 'steps': 5000, 'Iapp_type': 'constant'}
    }

cfg = st.session_state.config

# ---------------- Helper Functions ----------------
def apply_neuron_preset(idx, preset_name):
    """Update a,b,c,d for a specific neuron based on preset selection."""
    if preset_name in NEURON_PRESETS:
        a, b, c, d = NEURON_PRESETS[preset_name]
        cfg['neurons'][idx]['a'] = a
        cfg['neurons'][idx]['b'] = b
        cfg['neurons'][idx]['c'] = c
        cfg['neurons'][idx]['d'] = d

def load_rybak_preset():
    """Apply Rybak2002 CPG configuration to current session state."""
    cfg['N'] = 12
    rybak_names = ['CPG_IN_Flex', 'CPG_N_Flex', 'Ib_IN_Flex', 'Ia_IN_Flex', 'MN_Flex', 'R_Flex',
                   'CPG_IN_Ext', 'CPG_N_Ext', 'Ib_IN_Ext', 'Ia_IN_Ext', 'MN_Ext', 'R_Ext']
    rybak_types = ['CH', 'CH', 'FS', 'FS', 'RS', 'RS', 'CH', 'CH', 'FS', 'FS', 'RS', 'RS']
    cfg['neurons'] = []
    for i, (name, t) in enumerate(zip(rybak_names, rybak_types)):
        a,b,c,d = NEURON_PRESETS[t]
        cfg['neurons'].append({
            'name': name, 'type': t, 'a': a, 'b': b, 'c': c, 'd': d, 'Iapp': 5.0 if 'CPG' in name else 0.0
        })
    cfg['sim']['T_max'] = 5000
    st.success("✅ Rybak2002 preset applied")

# ---------------- UI Layout ----------------
with st.sidebar:
    st.header("⚙️ Global Controls")
    if st.button("🔄 Apply Rybak2002 CPG", type="primary"):
        load_rybak_preset()
        st.rerun()
        
    st.divider()
    uploaded_file = st.file_uploader("📂 Import Config (JSON)", type=["json"])
    if uploaded_file is not None:
        try:
            data = json.load(uploaded_file)
            if 'neurons' in data:
                st.session_state.config = data
                st.success("✅ Configuration loaded successfully!")
                st.rerun()
            else:
                st.warning("⚠️ File does not contain 'neurons' config structure.")
        except Exception as e:
            st.error(f"❌ Load failed: {str(e)}")
            
    if st.button("💾 Export Current Config"):
        filepath = "exported_config.json"
        with open(filepath, 'w') as f:
            json.dump(cfg, f, indent=2)
        st.success(f"✅ Saved to `{filepath}`")

# Main Tabs
tab_network, tab_limb, tab_muscles, tab_afferents, tab_sim = st.tabs([
    "🧠 Network & Neurons", "🦵 Biomechanics", "💪 Muscles", "📡 Afferents", "📈 Simulation"
])

with tab_network:
    st.header("Neuron-by-Neuron Configuration")
    st.info("Each neuron receives independent external current. Select type to auto-fill Izhikevich parameters.")
    
    # Dynamic table-like layout
    for i in range(cfg['N']):
        with st.expander(f"Neuron {i}: {cfg['neurons'][i]['name']}"):
            col1, col2, col3 = st.columns([3, 2, 3])
            with col1:
                cfg['neurons'][i]['name'] = st.text_input(
                    "Name", value=cfg['neurons'][i]['name'], key=f"name_{i}"
                )
            with col2:
                # Render selectbox and capture NEW value
                selected_type = st.selectbox(
                    "Preset Type", 
                    list(NEURON_PRESETS.keys()), 
                    index=list(NEURON_PRESETS.keys()).index(cfg['neurons'][i]['type']), 
                    key=f"type_{i}"
                )
                # ✅ FIX: Update params AFTER widget is rendered, using returned value
                if selected_type != cfg['neurons'][i]['type']:
                    cfg['neurons'][i]['type'] = selected_type
                    a, b, c, d = NEURON_PRESETS[selected_type]
                    cfg['neurons'][i]['a'] = a
                    cfg['neurons'][i]['b'] = b
                    cfg['neurons'][i]['c'] = c
                    cfg['neurons'][i]['d'] = d
                    
            with col3:
                cfg['neurons'][i]['Iapp'] = st.number_input(
                    "Input Current (nA)", 
                    value=float(cfg['neurons'][i]['Iapp']), 
                    format="%.2f", 
                    key=f"iapp_{i}"
                )
                
            cols = st.columns(4)
            cfg['neurons'][i]['a'] = cols[0].number_input(
                "a", value=float(cfg['neurons'][i]['a']), format="%.3f", key=f"a_{i}"
            )
            cfg['neurons'][i]['b'] = cols[1].number_input(
                "b", value=float(cfg['neurons'][i]['b']), format="%.3f", key=f"b_{i}"
            )
            cfg['neurons'][i]['c'] = cols[2].number_input(
                "c", value=float(cfg['neurons'][i]['c']), format="%.1f", key=f"c_{i}"
            )
            cfg['neurons'][i]['d'] = cols[3].number_input(
                "d", value=float(cfg['neurons'][i]['d']), format="%.2f", key=f"d_{i}"
            )
with tab_limb:
    st.header("Biomechanical Limb")
    cols = st.columns(4)
    cfg['limb']['m'] = cols[0].number_input("Mass (kg)", value=cfg['limb']['m'])
    cfg['limb']['ls'] = cols[1].number_input("Length (m)", value=cfg['limb']['ls'])
    cfg['limb']['b'] = cols[2].number_input("Damping", value=cfg['limb']['b'], format="%.4f")
    cfg['limb']['g'] = cols[3].number_input("Gravity", value=cfg['limb']['g'])
    cols2 = st.columns(3)
    cfg['limb']['a1'] = cols2[0].number_input("Tendon a1", value=cfg['limb']['a1'], format="%.3f")
    cfg['limb']['a2'] = cols2[1].number_input("Tendon a2", value=cfg['limb']['a2'], format="%.4f")
    cfg['limb']['q0'] = cols2[2].number_input("Init Angle (rad)", value=cfg['limb']['q0'], format="%.3f")

with tab_muscles:
    st.header("Muscle Parameters")
    col_f, col_e = st.columns(2)
    with col_f:
        st.subheader("Flexor")
        for k in cfg['muscles']['flexor']:
            cfg['muscles']['flexor'][k] = st.number_input(f"flex_{k}", value=float(cfg['muscles']['flexor'][k]), format="%.4f", key=f"flex_{k}")
    with col_e:
        st.subheader("Extensor")
        for k in cfg['muscles']['extensor']:
            cfg['muscles']['extensor'][k] = st.number_input(f"ext_{k}", value=float(cfg['muscles']['extensor'][k]), format="%.4f", key=f"ext_{k}")

with tab_afferents:
    st.header("Afferent Gains & Thresholds")
    cols = st.columns(3)
    for i, k in enumerate(cfg['afferents']):
        cfg['afferents'][k] = cols[i % 3].number_input(f"aff_{k}", value=float(cfg['afferents'][k]), format="%.3f", key=f"aff_{k}")

with tab_sim:
    st.header("Simulation Control")
    col1, col2 = st.columns(2)
    cfg['sim']['T_max'] = col1.number_input("Max Time (ms)", value=cfg['sim']['T_max'])
    cfg['sim']['steps'] = col2.number_input("Steps", value=cfg['sim']['steps'])
    
    if st.button("▶️ Run Simulation", type="primary"):
        try:
            # Extract per-neuron arrays
            names = [n['name'] for n in cfg['neurons']]
            a = np.array([n['a'] for n in cfg['neurons']])
            b = np.array([n['b'] for n in cfg['neurons']])
            c = np.array([n['c'] for n in cfg['neurons']])
            d = np.array([n['d'] for n in cfg['neurons']])
            
            N = cfg['N']
            # Identity Q_app ensures each neuron gets its own independent Iapp
            Q_app = np.eye(N)
            Q_aff = np.ones((N, 6)) * 0.1
            P = np.array([[1 if 'MN' in names[i] or 'R' in names[i] else 0 for i in range(N)]])
            
            net = Izhikevich_IO_Network(
                N=N, a=a, b=b, c=c, d=d, names=names,
                input_size=N, output_size=P.shape[0], afferent_size=6,
                Q_app=Q_app, Q_aff=Q_aff, P=P,
                M=np.zeros((N,N)), W=np.eye(N)*0.1, tau_syn=np.ones((N,N))*10
            )
            
            flex = SimpleAdaptedMuscle(**cfg['muscles']['flexor'])
            ext = SimpleAdaptedMuscle(**cfg['muscles']['extensor'])
            limb = OneDOFLimb(**cfg['limb'])
            al = Afferented_Limb(Limb=limb, Flexor=flex, Extensor=ext)
            for k, v in cfg['afferents'].items():
                setattr(al.Afferents, k, v)
                
            system = Var_Limb(Network=net, Limb=al)
            system.set_init_conditions(v_noise=np.random.normal(size=N, scale=0.5))
            
            T = np.linspace(0, cfg['sim']['T_max'], cfg['sim']['steps'])
            # Per-neuron input array
            Iapp_vals = np.array([n['Iapp'] for n in cfg['neurons']])
            def Iapp(t): return Iapp_vals.copy()
            
            results = run_simulation(system, T, Iapp)
            
            # Plotting
            fig, axes = plt.subplots(2, 2, figsize=(14, 9))
            axes[0,0].plot(T, results['V'])
            axes[0,0].set_title("Membrane Potentials (V)"); axes[0,0].set_ylabel("mV")
            
            axes[0,1].plot(T, results['F_flex'], label='Flexor')
            axes[0,1].plot(T, results['F_ext'], label='Extensor')
            axes[0,1].set_title("Muscle Forces"); axes[0,1].set_ylabel("N"); axes[0,1].legend()
            
            axes[1,0].plot(T, results['q'])
            axes[1,0].set_title("Joint Angle (q)"); axes[1,0].set_ylabel("rad")
            
            if 'output' in results:
                axes[1,1].plot(T, results['output'][:,0], label='Out_0')
                if results['output'].shape[1] > 1: axes[1,1].plot(T, results['output'][:,1], label='Out_1')
                axes[1,1].set_title("Network Output"); axes[1,1].set_ylabel("Current"); axes[1,1].legend()
                
            plt.tight_layout()
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"❌ Simulation failed: {str(e)}")