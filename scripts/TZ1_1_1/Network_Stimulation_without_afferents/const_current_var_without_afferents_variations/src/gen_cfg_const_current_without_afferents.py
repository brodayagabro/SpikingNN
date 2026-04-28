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
VARIATION_RANGE = np.arange(0, 20, 10)  # [0, 10]
WEIGHT_VARIATION = [0.001, 0.5, 5]
NOISE_PERCENT = 0.05

# Генерация комбинаций весов для каждой пары
weight_combinations = [
    list(product(WEIGHT_VARIATION, repeat=len(pair)))
    for pair in neuron_pairs_connections
]

print("Комбинации весов для каждой пары:")
for i, (pair_name, combinations) in enumerate(zip(pair_names, weight_combinations)):
    print(f"{pair_name}: {len(combinations)} комбинаций")
    for j, comb in enumerate(combinations[:3]):
        print(f"  {j+1}. {dict(zip(neuron_pairs_connections[i], comb))}")

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
        
        # Вариации входных токов
        for current_comb0, current_comb1 in product(VARIATION_RANGE, repeat=2):
            record = {
                "combination_id": comb_id,
                "target_pair": pair_name,
                "variation_I1": current_comb0,
                "variation_I2": current_comb1,
                "variation_I3": current_comb0,
                "variation_I4": current_comb1,
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

# Упорядочивание столбцов
base_columns = ['combination_id', 'target_pair', 'variation_I1', 'variation_I2', 
                'variation_I3', 'variation_I4', 'noise_percent']

weight_columns = [conn for pair in neuron_pairs_connections for conn in pair]
other_columns = sorted([col for col in df.columns if col not in base_columns + weight_columns])

df = df[base_columns + weight_columns + other_columns]

# Сохранение
config_file = "cfg.csv"
df.to_csv(config_file, index=False)

# Вывод статистики
total_combinations = len(df)
pairs_count = len(neuron_pairs_connections)
weight_combinations_per_pair = [len(c) for c in weight_combinations]
current_combinations = len(list(product(VARIATION_RANGE, repeat=2)))

print(f"\nСгенерировано конфигураций:")
print(f"  - Целевых пар: {pairs_count}")
print(f"  - Комбинаций весов на пару: {weight_combinations_per_pair}")
print(f"  - Вариаций токов: {current_combinations}")
print(f"  - Всего комбинаций: {total_combinations}")
print(f"Конфигурация сохранена в: {config_file}")

print(f"\nСтруктура таблицы:")
print("Основные колонки:", base_columns)
print("Колонки весов целевых пар:", weight_columns[:4])