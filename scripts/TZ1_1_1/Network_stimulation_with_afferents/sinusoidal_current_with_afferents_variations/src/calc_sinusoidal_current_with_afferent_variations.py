import numpy as np
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
from feature_extractor import *
from uuid import uuid4

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
    Rybak2002FullSystem.set_afferents_by_names('Ia_Flex', 'Ia_IN_Flex', 10)
    Rybak2002FullSystem.set_afferents_by_names('Ia_Ext', 'Ia_IN_Ext', 10)
    
    return Rybak2002FullSystem

class SourceTargetParser:
    """Класс для парсинга строк вида 'source-target' или 'source->target'"""
    
    @staticmethod
    def parse(text, delimiter=None):
        """
        Парсит строку с source и target
        
        Args:
            text (str): входная строка
            delimiter (str, optional): конкретный разделитель. Если None - определяется автоматически
        
        Returns:
            tuple: (source, target)
        """
        if delimiter:
            if delimiter not in text:
                raise ValueError(f"Разделитель '{delimiter}' не найден в строке")
            source, target = text.split(delimiter, 1)
        else:
            if '->' in text:
                source, target = text.split('->', 1)
            elif '-' in text:
                source, target = text.split('-', 1)
            else:
                raise ValueError("Не удалось определить разделитель")
        
        return source.strip(), target.strip()
    
    @staticmethod
    def parse_dash(text):
        """Алиас для разделителя '-'"""
        return SourceTargetParser.parse(text, '-')
    
    @staticmethod
    def parse_arrow(text):
        """Алиас для разделителя '->'"""
        return SourceTargetParser.parse(text, '->')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Генератор синусоидального тока
def make_sine_generator(params):
    periods = [
        params["sine_period_ch0"],
        params["sine_period_ch1"], 
        params["sine_period_ch2"],
        params["sine_period_ch3"]
    ]
    amplitudes = [
        params["amplitude_ch0"],
        params["amplitude_ch1"],
        params["amplitude_ch2"],
        params["amplitude_ch3"]
    ]
    phases = [
        params["phase_ch0"],
        params["phase_ch1"],
        params["phase_ch2"],
        params["phase_ch3"]
    ]
    base_currents = [
        params["base_current_I1"],
        params["base_current_I2"],
        params["base_current_I3"],
        params["base_current_I4"]
    ]
    noise_percent = params["noise_percent"]
    
    def Iapp(t):
        I = np.zeros(4)
        for i in range(4):
            # Синусоидальный ток: I = base + amplitude * sin(2π*(t/period - phase))
            if periods[i] > 0:
                sine_value = amplitudes[i] * np.sin(2 * np.pi * (t / periods[i] - phases[i]))
            else:
                sine_value = 0
            noise = np.random.normal(0, base_currents[i] * noise_percent)
            I[i] = base_currents[i] + sine_value + noise
        return I
    
    return Iapp

def run_simulation_task(params):
    try:
        logger.info(f"Запуск симуляции для комбинации {params['combination_id']}")

        # Создаём базовый результат — копия входных параметров (всё из cfg.csv)
        result = dict(params)

        # Подготовка модели
        model = initialize_model()

        # Нахождение целевой пары
        target_pair = params['target_pair']
        parser = SourceTargetParser()
        source_neuron_name, target_neuron_name = parser.parse(target_pair)

        # Установка весов
        forward_key = f"{source_neuron_name}->{target_neuron_name}"
        backward_key = f"{target_neuron_name}->{source_neuron_name}"

        model.set_weights_by_names(source_neuron_name, target_neuron_name, params[forward_key])
        model.set_weights_by_names(target_neuron_name, source_neuron_name, params[backward_key])

        # Генерация входного тока
        Iapp = make_sine_generator(params)

        # Параметры интегрирования
        scale = 5
        Tmax = 10000
        T = np.linspace(0, Tmax, scale * Tmax)
        dt = T[1] - T[0]
        N = 12

        # Инициализация массивов
        V_curr = np.zeros((len(T), N))
        U_curr = np.zeros((len(T), N))
        F_flex_curr = np.zeros(len(T))
        F_ext_curr = np.zeros(len(T))
        Afferents_curr = np.zeros((len(T), 6))
        Q_curr = np.zeros(len(T))
        W_curr = np.zeros(len(T))

        # Интегрирование
        for i, t in enumerate(T):
            V_curr[i] = model.V
            U_curr[i] = model.U
            F_flex_curr[i] = model.F_flex
            F_ext_curr[i] = model.F_ext
            Afferents_curr[i] = model.Limb.output
            Q_curr[i] = model.q
            W_curr[i] = model.w
            model.step(dt=dt, Iapp=Iapp(t))

        # Обрезка первых 50% данных
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

        # Имя файла на основе ID
        muscle_file_id = uuid4()
        muscle_csv_filename = f"sinusoudal_current_without_afferents_{muscle_file_id}_muscles.csv"
        muscle_csv_path = output_dir / muscle_csv_filename

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

def main():
    print("="*50)
    print("Программа моделирования синусоидальных воздействий")
    print("="*50)
    
    # Загрузка конфигурационного файла
    config_file = "cfg.csv"
    if not os.path.exists(config_file):
        print(f"Файл конфигурации не найден: {config_file}")
        return
        
    df = pd.read_csv(config_file)
    print(f"Загружено {len(df)} конфигураций")

    selected_params_list = [row.to_dict() for _, row in df.iterrows()]
    
    n_jobs = -1
    
    # Настройка параллельного выполнения
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(run_simulation_task)(params) 
        for params in tqdm(selected_params_list, desc="Calculating")
    )
    
    # Сохранение результатов
    if results:
        results_df = pd.DataFrame(results)
        file_name = "../experiment_results_sinusoidal_currents_with_afferents.csv"
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
            print(results_df[results_df['status'] == 'error'][['combination_id']])

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n Программа прервана пользователем")
    except Exception as e:
        print(f"\n Критическая ошибка: {str(e)}")
        import traceback
        traceback.print_exc()
