import numpy as np
import pandas as pd
from itertools import product

# Ненулевые связи с положительными базовыми весами
connections_weights = {
    'Ia_IN_Ext->CPG_IN_Flex': 0.5,
    'CPG_IN_Flex->CPG_N_Flex': 0.5,
    'CPG_N_Ext->CPG_N_Flex': 0.5,
    'Ib_IN_Ext->Ib_IN_Flex': 0.5,
    'CPG_N_Flex->Ia_IN_Flex': 0.5,
    'R_Flex->Ia_IN_Flex': 0.5,
    'Ia_IN_Ext->Ia_IN_Flex': 0.5,
    'CPG_N_Flex->MN_Flex': 0.5,
    'Ib_IN_Flex->MN_Flex': 0.5,
    'R_Flex->MN_Flex': 0.5,
    'Ia_IN_Ext->MN_Flex': 0.5,
    'MN_Flex->R_Flex': 0.5,
    'R_Ext->R_Flex': 0.5,
    'Ia_IN_Flex->CPG_IN_Ext': 0.5,
    'CPG_N_Flex->CPG_N_Ext': 0.5,
    'CPG_IN_Ext->CPG_N_Ext': 0.5,
    'Ib_IN_Flex->Ib_IN_Ext': 0.5,
    'Ia_IN_Flex->Ia_IN_Ext': 0.5,
    'CPG_N_Ext->Ia_IN_Ext': 0.5,
    'R_Ext->Ia_IN_Ext': 0.5,
    'Ia_IN_Flex->MN_Ext': 0.5,
    'CPG_N_Ext->MN_Ext': 0.5,
    'Ib_IN_Ext->MN_Ext': 0.5,
    'R_Ext->MN_Ext': 0.5,
    'R_Flex->R_Ext': 0.5,
    'MN_Ext->R_Ext': 0.5
}

# Пары связей, в которых будет варьироваться вес
neuron_pairs_connections = [
    ['CPG_N_Flex->CPG_N_Ext', 'CPG_N_Ext->CPG_N_Flex'],
    ['Ib_IN_Flex->Ib_IN_Ext', 'Ib_IN_Ext->Ib_IN_Flex'],
    ['Ia_IN_Flex->Ia_IN_Ext', 'Ia_IN_Ext->Ia_IN_Flex'],
    ['R_Flex->R_Ext', 'R_Ext->R_Flex']
]

pair_names = [
    'CPG_N_Flex-CPG_N_Ext',
    'Ib_IN_Flex-Ib_IN_Ext', 
    'Ia_IN_Flex-Ia_IN_Ext',
    'R_Flex-R_Ext'
]

# Параметры варьирования
WEIGHT_VARIATION = [0.001, 0.5, 1]
PERIOD_RANGE = np.linspace(1, 1000, 1)
DURATION_RANGE = np.linspace(0, 400, 1)
PHASE_RANGE = np.linspace(0, 1, 1)
BASE_CURRENTS = [0, 0, 0, 0]
NOISE_PERCENT = 0.05

# Генерация комбинаций весов для каждой пары
weight_combinations = [
    list(product(WEIGHT_VARIATION, repeat=len(pair)))
    for pair in neuron_pairs_connections
]

# Сбор всех записей конфигурации
data = []
comb_id = 0

for pair_idx, (pair_name, pair_connections) in enumerate(zip(pair_names, neuron_pairs_connections)):
    for weight_comb in weight_combinations[pair_idx]:
        # Базовая конфигурация весов
        base_config = dict(connections_weights)
        # Обновляем веса для текущей пары
        for conn, weight in zip(pair_connections, weight_comb):
            base_config[conn] = weight
        
        # Параметры импульсов
        for period0, period1 in product(PERIOD_RANGE, repeat=2):
            for dur0, dur1 in product(DURATION_RANGE, repeat=2):
                # Проверка ограничения длительности
                if dur0 > period0 or dur1 > period1:
                    continue
                    
                for phase0, phase1 in product(PHASE_RANGE, repeat=2):
                    record = {
                        "combination_id": comb_id,
                        "target_pair": pair_name,
                        
                        # Группа 0 (каналы 0 и 2)
                        "pulse_period_ch0": period0,
                        "pulse_period_ch2": period0,
                        "pulse_duration_ch0": dur0,
                        "pulse_duration_ch2": dur0,
                        "amplitude_ch0": 5,
                        "amplitude_ch2": 5,
                        "phase_ch0": phase0,
                        "phase_ch2": phase0,
                        
                        # Группа 1 (каналы 1 и 3)
                        "pulse_period_ch1": period1,
                        "pulse_period_ch3": period1,
                        "pulse_duration_ch1": dur1,
                        "pulse_duration_ch3": dur1,
                        "amplitude_ch1": 4,
                        "amplitude_ch3": 4,
                        "phase_ch1": phase1,
                        "phase_ch3": phase1,
                        
                        # Базовые токи и шум
                        "base_current_I1": BASE_CURRENTS[0],
                        "base_current_I2": BASE_CURRENTS[1],
                        "base_current_I3": BASE_CURRENTS[2],
                        "base_current_I4": BASE_CURRENTS[3],
                        "noise_percent": NOISE_PERCENT
                    }
                    
                    # Добавляем веса связей из пары
                    for conn, weight in zip(pair_connections, weight_comb):
                        record[conn] = weight
                    # Добавляем все остальные веса
                    for conn, weight in base_config.items():
                        record[conn] = weight
                    
                    data.append(record)
                    comb_id += 1

# Создание DataFrame
df = pd.DataFrame(data)

# Упорядочивание столбцов (основные -> веса пар -> остальные веса)
base_columns = ['combination_id', 'target_pair']
pulse_columns = [
    'pulse_period_ch0', 'pulse_period_ch2', 'pulse_duration_ch0', 'pulse_duration_ch2',
    'amplitude_ch0', 'amplitude_ch2', 'phase_ch0', 'phase_ch2',
    'pulse_period_ch1', 'pulse_period_ch3', 'pulse_duration_ch1', 'pulse_duration_ch3',
    'amplitude_ch1', 'amplitude_ch3', 'phase_ch1', 'phase_ch3',
    'base_current_I1', 'base_current_I2', 'base_current_I3', 'base_current_I4',
    'noise_percent'
]
weight_columns = [conn for pair in neuron_pairs_connections for conn in pair]
other_columns = sorted([col for col in df.columns if col not in base_columns + pulse_columns + weight_columns])

df = df[base_columns + pulse_columns + weight_columns + other_columns]

# Сохранение
config_file = "cfg.csv"
df.to_csv(config_file, index=False)

# Вывод статистики
total_combinations = len(df)
pairs_count = len(neuron_pairs_connections)
weight_combinations_per_pair = [len(c) for c in weight_combinations]
pulse_combinations = len(list(product(PERIOD_RANGE, repeat=2))) * len(list(product(DURATION_RANGE, repeat=2))) * len(list(product(PHASE_RANGE, repeat=2)))

print(f"Сгенерировано {total_combinations} комбинаций параметров")
print(f"  - Целевых пар: {pairs_count}")
print(f"  - Комбинаций весов на пару: {weight_combinations_per_pair}")
print(f"  - Вариаций импульсов: {pulse_combinations}")
print(f"Конфигурация сохранена в: {config_file}")