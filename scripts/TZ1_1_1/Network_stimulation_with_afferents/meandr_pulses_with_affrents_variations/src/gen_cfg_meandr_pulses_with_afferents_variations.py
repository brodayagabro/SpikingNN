import numpy as np
import os
import pandas as pd
from itertools import product, chain
from Rybak2002 import Rybak2002Afferents as Aff

DEBAG = True

def flat_product(*iterables):
    """Декартово произведение с плоским результатом"""
    for comb in product(*iterables):
        yield list(chain.from_iterable(comb))

afferent_pairs = {
        'Ia': ['Ia_Flex', 'Ia_Ext'],
        'II': ['II_Flex', 'II_Ext'],
        'Ib': ['Ib_Flex', 'Ib_Ext']
        }

if DEBAG:
    print('afferent pairs:')
    print(afferent_pairs)

afferent_types = [key for key in afferent_pairs.keys()]

if DEBAG:
    print('afferent types:')
    print(afferent_types)


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



def gen_meandrs_pulses_cfg():
    # variation params
    WEIGHT_VARIATION = [0.001, 0.5, 1]
    AMPLITUDE_RANGE = np.array([0])  # np.linspace(1, 40, 5)
    PERIOD_RANGE = np.linspace(1, 1000, 1)
    DURATION_RANGE = np.linspace(0, 400, 1)
    PHASE_RANGE = np.linspace(0, np.pi, 1)
    # Фиксированные параметры
    NOISE_PERCENT = 0.05
    BASE_CURRENTS = [0, 0, 0, 0]  # Базовые токи для каждого канала


    # create config table
    data = []
    comb_id = 0
    for Ia, II, Ib in product(WEIGHT_VARIATION, repeat=len(afferent_types)): 
        for period0, period1 in product(PERIOD_RANGE, repeat=2):
            for dur0, dur1 in product(DURATION_RANGE, repeat=2):
                # Проверка ограничения длительности для каждой группы
                if dur0 > period0 or dur1 > period1:
                    continue             
                for phase0, phase1 in product(PHASE_RANGE, repeat=2):
                    record = {
                        "combination_id": comb_id,
                        # афферентные веса
                        'Ia': Ia,
                        'II': II,
                        'Ib': Ib,
                        
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
                    
                    
                    # appending neuron weights
                    for conn, weight in connections_weights.items():
                        record[conn] = weight
                    
                    data.append(record)
                    comb_id += 1

    df = pd.DataFrame(data)
    config_file = 'cfg.csv'

    cfg_dir = '.'
    os.makedirs(cfg_dir, exist_ok=True)
    path = os.path.join(cfg_dir, config_file) 

    df.to_csv(path)
    print(df.head(10))


    # Статистика
    total_combinations = len(df)

    print(f"\nСгенерировано конфигураций:")
    print(f"  - Всего комбинаций: {total_combinations}")
    print(f"Конфигурация сохранена в: {path}")

if __name__=='__main__':
    gen_meandrs_pulses_cfg()
