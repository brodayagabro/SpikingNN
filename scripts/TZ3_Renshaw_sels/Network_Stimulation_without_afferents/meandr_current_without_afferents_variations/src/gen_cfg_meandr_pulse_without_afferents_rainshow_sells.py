import json
import numpy as np
import pandas as pd
from itertools import product
import sys
import os

# ==========================================================
# 1. LOAD CONFIGURATION
# ==========================================================
CONFIG_FILE = sys.argv[1] if len(sys.argv) > 1 else "config.json"
if not os.path.exists(CONFIG_FILE):
    raise FileNotFoundError(f"Config file '{CONFIG_FILE}' not found.")

with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
    CFG = json.load(f)

# Безопасные генераторы диапазонов
def safe_arange(cfg):
    return np.arange(cfg["min"], cfg["max"] + cfg["step"] * 0.51, cfg["step"])

def safe_linspace(cfg):
    return np.linspace(cfg["min"], cfg["max"], int(cfg["num"]))

# Парсим конфиг
BASE_WEIGHT = CFG.get("base_weight", 0.5)
RAINSHOW_WEIGHT_RANGE = safe_arange(CFG["renshaw_weights"])

mp = CFG["meander"]
PERIOD_RANGE = safe_linspace(mp["period"])
DURATION_FLEX_RANGE = safe_linspace(mp["duration_flex"])
DURATION_EXT_RANGE = safe_linspace(mp["duration_ext"])
AMPLITUDE_CPG_IN_RANGE = safe_linspace(mp["amplitude_cpg_in"])
AMPLITUDE_CPG_N_RANGE = safe_linspace(mp["amplitude_cpg_n"])
PHASE_FLEX_RANGE = safe_linspace(mp["phase_flex"])
PHASE_EXT_RANGE = safe_linspace(mp["phase_ext"])

BASE_CURRENTS = CFG.get("base_currents", [0, 0, 0, 0])
NOISE_PERCENT = CFG.get("noise_percent", 0.05)
OUTPUT_FILE = CFG.get("output_filename", "cfg.csv")

# ==========================================================
# 2. LOAD NETWORK STRUCTURE
# ==========================================================
from Rybak2002 import Rybak2002_Mask, Rybak_2002_names_types

M = Rybak2002_Mask()
NEURON_NAMES, _ = Rybak_2002_names_types()

ALL_CONNECTIONS = []
RAINSHOW_CONNECTIONS = []
RAINSHOW_INDICES = [5, 11]

for target_idx in range(12):
    for source_idx in range(12):
        if M[target_idx, source_idx] != 0:
            conn_str = f"{NEURON_NAMES[source_idx]}->{NEURON_NAMES[target_idx]}"
            ALL_CONNECTIONS.append(conn_str)
            if source_idx in RAINSHOW_INDICES or target_idx in RAINSHOW_INDICES:
                RAINSHOW_CONNECTIONS.append(conn_str)

print(f"[INFO] Total connections in Rybak mask: {len(ALL_CONNECTIONS)}")
print(f"[INFO] Renshaw-related connections to vary: {len(RAINSHOW_CONNECTIONS)}")
print(f"[INFO] Weight values to test: {RAINSHOW_WEIGHT_RANGE}")

# ==========================================================
# 3. GENERATE CONFIGURATION COMBINATIONS
# ==========================================================
BASE_WEIGHTS = {conn: BASE_WEIGHT for conn in ALL_CONNECTIONS}
weight_combinations = list(product(RAINSHOW_WEIGHT_RANGE, repeat=len(RAINSHOW_CONNECTIONS)))

# Оценка количества комбинаций
n_w = len(RAINSHOW_WEIGHT_RANGE) ** len(RAINSHOW_CONNECTIONS)
n_p = len(PERIOD_RANGE) ** 2
n_df = len(DURATION_FLEX_RANGE)
n_de = len(DURATION_EXT_RANGE)
n_af = len(AMPLITUDE_CPG_IN_RANGE)
n_an = len(AMPLITUDE_CPG_N_RANGE)
n_pf = len(PHASE_FLEX_RANGE)
n_pe = len(PHASE_EXT_RANGE)
est_total = n_w * n_p * n_df * n_de * n_af * n_an * n_pf * n_pe

print(f"[INFO] Estimated combinations: ~{est_total:,}")

if est_total > 5_000_000:
    print("WARNING: More than 5M combinations. This may take time and disk space.")
    confirm = input("Continue? (y/N): ").strip().lower()
    if confirm != 'y':
        sys.exit("Aborted by user.")

data = []
comb_id = 0

for weights in weight_combinations:
    current_weights = dict(BASE_WEIGHTS)
    for conn, w in zip(RAINSHOW_CONNECTIONS, weights):
        current_weights[conn] = w

    for p0, p1 in product(PERIOD_RANGE, repeat=2):
        for d_flex, d_ext in product(DURATION_FLEX_RANGE, DURATION_EXT_RANGE):
            # Проверка: длительность не может превышать период
            if d_flex > p0 or d_ext > p1:
                continue
            
            for amp_in, amp_n in product(AMPLITUDE_CPG_IN_RANGE, AMPLITUDE_CPG_N_RANGE):
                for ph_flex, ph_ext in product(PHASE_FLEX_RANGE, PHASE_EXT_RANGE):
                    record = {
                        "combination_id": comb_id,
                        "target_module": "Renshaw_Variation",
                        
                        # Группа 0,2 (CPG_IN channels)
                        "pulse_period_ch0": p0, 
                        "pulse_period_ch2": p0,
                        "pulse_duration_ch0": d_flex, 
                        "pulse_duration_ch2": d_flex,
                        "amplitude_ch0": amp_in, 
                        "amplitude_ch2": amp_in,
                        "phase_ch0": ph_flex, 
                        "phase_ch2": ph_flex,
                        
                        # Группа 1,3 (CPG_N channels)
                        "pulse_period_ch1": p1, 
                        "pulse_period_ch3": p1,
                        "pulse_duration_ch1": d_ext, 
                        "pulse_duration_ch3": d_ext,
                        "amplitude_ch1": amp_n, 
                        "amplitude_ch3": amp_n,
                        "phase_ch1": ph_ext, 
                        "phase_ch3": ph_ext,
                        
                        # Базовые токи и шум
                        "base_current_I1": BASE_CURRENTS[0],
                        "base_current_I2": BASE_CURRENTS[1],
                        "base_current_I3": BASE_CURRENTS[2],
                        "base_current_I4": BASE_CURRENTS[3],
                        "noise_percent": NOISE_PERCENT,
                    }
                    record.update(current_weights)
                    data.append(record)
                    comb_id += 1

# ==========================================================
# 4. CREATE & SAVE DATAFRAME
# ==========================================================
df = pd.DataFrame(data)

id_cols = ['combination_id', 'target_module']
pulse_cols = [
    'pulse_period_ch0', 'pulse_duration_ch0', 'amplitude_ch0', 'phase_ch0',
    'pulse_period_ch1', 'pulse_duration_ch1', 'amplitude_ch1', 'phase_ch1',
    'pulse_period_ch2', 'pulse_duration_ch2', 'amplitude_ch2', 'phase_ch2',
    'pulse_period_ch3', 'pulse_duration_ch3', 'amplitude_ch3', 'phase_ch3'
]
curr_noise_cols = ['base_current_I1', 'base_current_I2', 'base_current_I3', 'base_current_I4', 'noise_percent']
rainshow_cols = RAINSHOW_CONNECTIONS
other_conn_cols = [c for c in ALL_CONNECTIONS if c not in RAINSHOW_CONNECTIONS]

df = df[id_cols + pulse_cols + curr_noise_cols + rainshow_cols + other_conn_cols]

df.to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ SUCCESS: Generated {len(df)} total combinations.")
print(f"💾 Saved to: {OUTPUT_FILE}")
print(f"\n📊 Parameter summary:")
print(f"   Period: {PERIOD_RANGE}")
print(f"   Duration Flex: {DURATION_FLEX_RANGE}")
print(f"   Duration Ext: {DURATION_EXT_RANGE}")
print(f"   Amplitude CPG_IN: {AMPLITUDE_CPG_IN_RANGE}")
print(f"   Amplitude CPG_N: {AMPLITUDE_CPG_N_RANGE}")
print(f"   Phase Flex: {PHASE_FLEX_RANGE}")
print(f"   Phase Ext: {PHASE_EXT_RANGE}")