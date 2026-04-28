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


def gen_constant_current_cfg():
    # general arrays
    # variation of current value 
    VARIATION_RANGE = np.arange(0, 20, 5)
    # variation of weight
    WEIGHT_VARIATION = np.array([0.001, 0.5, 1])
    # noise
    NOISE_PERCENT = 0.05

    # gen curent cominations
    current_combinations = list(product(VARIATION_RANGE, repeat=2))
    
    if DEBAG:
        print('current combinations:')
        print(np.shape(current_combinations))
        print(current_combinations[4])

    # gen weight variation
    weight_combinations = list(product(WEIGHT_VARIATION, repeat=len(afferent_types)))

    if DEBAG:
        print('weight combinations')
        print(np.shape(weight_combinations))
        print(weight_combinations[4])

    all_combinations = list(flat_product(
            weight_combinations,
            current_combinations
        ))

    if DEBAG :
        print('all combinations')
        print(np.shape(all_combinations))
        print(all_combinations[4])
    

    # create config table
    data = []

    for comb_id, values in enumerate(all_combinations):
        weights = values[:3]
        currents = values[3:]
        current_comb0 = currents[0]
        current_comb1 = currents[1]
        record = {
                'combination_id': comb_id, 
                'Ia': weights[0],
                'II': weights[1],
                'Ib': weights[2], 
                "variation_I1": current_comb0,
                "variation_I2": current_comb1,
                "variation_I3": current_comb0,
                "variation_I4": current_comb1,
                "noise_percent": NOISE_PERCENT
        }

        # appending neuron weights
        for conn, weight in connections_weights.items():
            record[conn] = weight

        data.append(record)

    df = pd.DataFrame(data)
    config_file = './cfg.csv'


    df.to_csv(config_file)
    print(df.head(10))


    # Статистика
    total_combinations = len(df)

    print(f"\nСгенерировано конфигураций:")
    print(f"  - Всего комбинаций: {total_combinations}")
    print(f"Конфигурация сохранена в: {config_file}")

if __name__=='__main__':
    gen_constant_current_cfg()
