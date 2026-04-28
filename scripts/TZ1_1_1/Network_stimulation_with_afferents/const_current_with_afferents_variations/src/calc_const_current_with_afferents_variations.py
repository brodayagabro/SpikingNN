import numpy as np
from matplotlib import pyplot as plt
from SpikingNN.Izh_net import *
from SpikingNN.Var_Limb import *
from Rybak2002 import *
from tqdm import tqdm as pbar
import os
import pandas as pd
from itertools import product
from tqdm import tqdm
from joblib import Parallel, delayed
import logging
from pathlib import Path
from feature_extractor import *  # Импорт функции метрик
from uuid import uuid4

DEBAG = False
TEST = False

def initialize_model():
    Flexor = SimpleAdaptedMuscle(w=0.5, N=50)
    Extensor = SimpleAdaptedMuscle(w=0.5, N=50)
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
    Qapp = np.zeros((12, 4))
    Qapp[0, 0] = Qapp[1, 1] = Qapp[6, 2] = Qapp[7, 3] = 1
    Rybak2002Net = Rybak_2002_network(
        input_size=4, 
        output_size=2, 
        afferent_size=6, 
        Qapp=Qapp, 
        exitatory_w=0.5, 
        inhibitory_w=-0.5
    )
    Rybak2002FullSystem = Var_Limb(Network=Rybak2002Net, Limb=Limb)
    return Rybak2002FullSystem


output_dir = "../data"
os.makedirs(output_dir, exist_ok=True)



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
def make_pulse_generator(params):
    current_variation = [
        params["variation_I1"],  # было variation_I0
        params["variation_I2"],  # было variation_I1
        params["variation_I3"],  # было variation_I2
        params["variation_I4"]   # было variation_I3
    ]
    noise_percent = params["noise_percent"]

    def Iapp(t):
        I = np.zeros(4)
        for i in range(4):
            noise = np.random.normal(0, current_variation[i] * noise_percent)
            I[i] = current_variation[i] + noise
        return I
    return Iapp

afferent_pairs = {
        'Ia': ['Ia_Flex', 'Ia_Ext'],
        'II': ['II_Flex', 'II_Ext'],
        'Ib': ['Ib_Flex', 'Ib_Ext']
        }

AffMask = np.where(Rybak2002Afferents()!=0, 1, 0)

if DEBAG:
    print(AffMask)


def set_afferent_weights_to_model(model, params):
    afferent_names = model.names['afferents']
    neuron_names = model.names['neurons']
    for afferent in afferent_pairs.keys():
        if DEBAG:
            print(afferent)
            print(afferent_pairs[afferent])
        for name in afferent_pairs[afferent]:
            idx = afferent_names.index(name)
            if DEBAG:
                print(idx)
            weight=params[afferent]
            target_weights = np.where(AffMask[:, idx]==1, weight, 0)
            for neuron, weight in zip(neuron_names, target_weights):
                model.set_afferents_by_names(name, neuron, weight)
    return model        

        



def prepare_model(model, params):
    return set_afferent_weights_to_model(model, params)
    

    
# Функция для выполнения в одном потоке    
def run_simulation_task(params):
    try:
        logger.info(f"Запуск симуляции для комбинации {params['combination_id']}")

        # Создаём базовый результат — копия входных параметров (всё из cfg.csv)
        result = dict(params)
        # подготовка модели к симуляции
        # Иннициализация модели
        model = initialize_model()

        model = prepare_model(model, params)

        Iapp = make_pulse_generator(params)
        scale = 5
        Tmax = 10000
        T = np.linspace(0, Tmax, scale*Tmax)
        dt = T[1] - T[0]
        N = 12
        
        V_curr = np.zeros((len(T), N))
        U_curr = np.zeros((len(T), N))
        F_flex_curr = np.zeros(len(T))
        F_ext_curr = np.zeros(len(T))
        Afferents_curr = np.zeros((len(T), 6))
        Q_curr = np.zeros(len(T))
        W_curr = np.zeros(len(T))
        
        for i, t in enumerate(T):
            V_curr[i] = model.V
            U_curr[i] = model.U
            F_flex_curr[i] = model.F_flex
            F_ext_curr[i] = model.F_ext
            Afferents_curr[i] = model.Limb.output
            Q_curr[i] = model.q
            W_curr[i] = model.w
            model.step(dt=dt, Iapp=Iapp(t))
        
        start_index = int(len(T) * 0.5)
        T_trimmed = T[start_index:]
        V_trimmed = V_curr[start_index:]
        U_trimmed = U_curr[start_index:]
        F_flex_trimmed = F_flex_curr[start_index:]
        F_ext_trimmed = F_ext_curr[start_index:]
        Afferents_trimmed = Afferents_curr[start_index:]
        Q_trimmed = Q_curr[start_index:]
        W_trimmed = W_curr[start_index:]

        # === СОХРАНЕНИЕ ВРЕМЕННЫХ РЯДОВ МЫШЦ (CSV) ===
        output_dir = Path("../data")
        output_dir.mkdir(exist_ok=True)
        muscle_csv_dir = output_dir
        
        current_variation = [
            params["variation_I1"],  # было variation_I0
            params["variation_I2"],  # было variation_I1
            params["variation_I3"],  # было variation_I2
            params["variation_I4"] 
        ]
        
        var_str_current = "_".join(map(str, current_variation))
        muscle_file_id = uuid4()
        muscle_csv_filename = f"constant_current_with_afferents_{muscle_file_id}_muscles.csv"
        muscle_csv_path = muscle_csv_dir / muscle_csv_filename

        muscle_df = pd.DataFrame({
            't': T_trimmed,
            'F_flex': F_flex_trimmed,
            'F_ext': F_ext_trimmed
        })
        muscle_df.to_csv(muscle_csv_path, index=False)

        # === ВЫЧИСЛЕНИЕ ВСЕХ МЕТРИК ===
        features = compute_all_features(
            F_flex_trimmed, F_ext_trimmed,
            V_trimmed, U_trimmed, Afferents_trimmed,
            Q_trimmed, W_trimmed, T_trimmed
        )

        # === ВЫЧИСЛЕНИЕ РЕЖИМА РАБОТЫ ===
        mode = classify_operation_mode(features)

        # === ОБНОВЛЕНИЕ РЕЗУЛЬТАТА ===
        result.update(features)  # добавляем все фичи
        result['mode'] = mode     # добавляем режим работы
        
        result.update({
            "muscle_csv_file": str(muscle_csv_filename),
            "status": "success"
        })
        logger.info(f"Успешно завершена комбинация {params['combination_id']}")
        return result

    except Exception as e:
        logger.error(f"Ошибка в комбинации {params['combination_id']}: {str(e)}")
        # В случае ошибки тоже возвращаем все исходные поля + статус
        result = dict(params)
        result.update({
            "muscle_csv_file": None,
            "mode": "Error processing",
            "status": "error"
        })
        return result

def test():

    config_file = "./cfg.csv"

    df = pd.read_csv(config_file)
    print(f"Загружено {len(df)} конфигураций")
    selected_params_list = [row.to_dict() for _, row in df.iterrows()]

    run_simulation_task(selected_params_list[0])

def main():
    print("="*50)
    print("Программа моделирования импульсных воздействий")
    print("="*50)
    
    # Загрузка конфигурационного файла
    config_file = "./cfg.csv"
    if not os.path.exists(config_file):
        print(f"Файл конфигурации не найден: {config_file}")
        print("Сначала запустите config_generator_current.py")
        return
        
        
    
    df = pd.read_csv(config_file)
    print(f"Загружено {len(df)} конфигураций")

    selected_params_list = [row.to_dict() for _, row in df.iterrows()]
    
    n_jobs = -1
    
    # Настройка параллельного выполнения
    results = Parallel(n_jobs=n_jobs, verbose=4)(
        delayed(run_simulation_task)(params) 
        for params in tqdm(selected_params_list, desc="Подготовка задач")
    )
    
    # Сохранение результатов
    if results:
        results_df = pd.DataFrame(results)
        file_name = "../experiment_rezults_const_current_with_afferents.csv"
        results_df.to_csv(file_name, index=False)
        print(f"Результаты сохранены в {file_name}")
        
        # Статистика выполнения
        success_count = results_df[results_df['status'] == 'success'].shape[0]
        error_count = results_df[results_df['status'] == 'error'].shape[0]
        print(f"\n Статистика выполнения:")
        print(f"  Успешно: {success_count}")
        print(f"  С ошибками: {error_count}")
        
        if error_count > 0:
            print("\n Ошибочные комбинации:")
            print(results_df[results_df['status'] == 'error'][['combination_id', 'error']])

if __name__ == "__main__":
    try:
        if TEST:
            test()
        else:
            main()
    except KeyboardInterrupt:
        print("\n Программа прервана пользователем")
    except Exception as e:
        print(f"\n Критическая ошибка: {str(e)}")
        import traceback
        traceback.print_exc()
