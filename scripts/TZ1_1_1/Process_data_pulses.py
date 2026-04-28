import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
from pathlib import Path
from scipy.signal import find_peaks, welch
from joblib import Parallel, delayed
import seaborn as sns
from tqdm.auto import tqdm
import time
from pandas.plotting import parallel_coordinates
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay

class TimeSeriesAnalyzer:
    """
    Класс для анализа временных рядов данных стимуляции
    """
    
    def __init__(self, data_dir=None, plots_dir=None, analysis_dir=None):
        """
        Инициализация анализатора
        
        Args:
            data_dir: директория с исходными данными
            plots_dir: директория для графиков временных рядов
            analysis_dir: директория для результатов анализа
        """
        self.data_dir = Path(data_dir) if data_dir else Path("experiment_data")
        self.plots_dir = Path(plots_dir) if plots_dir else Path("time_series_plots")
        self.analysis_dir = Path(analysis_dir) if analysis_dir else Path("analysis_results")
        
        # Создаем директории если они не существуют
        self.plots_dir.mkdir(exist_ok=True)
        self.analysis_dir.mkdir(exist_ok=True)
        
        # Настройка стиля
        plt.style.use('default')
        sns.set_palette("tab10")
        
        # Атрибуты для хранения результатов
        self.results_df = None
        self.npz_files = []
        self.valid_results = []

    @staticmethod
    def extract_params_from_filename(stem):
        """Извлечение параметров стимуляции из имени файла"""
        params = {}
        
        # Извлекаем значение comb
        comb_match = re.search(r'comb_([\d\.]+)', stem)
        if comb_match:
            params['comb'] = float(comb_match.group(1))
        
        # Функция для извлечения параметров
        def extract_param(pattern, count=4):
            match = re.search(pattern, stem)
            if match:
                return list(map(float, match.group(1).split('_')))
            return [0.0] * count
        
        # Извлекаем параметры
        params['per'] = extract_param(r'per_([\d\._]+)_dur')
        params['dur'] = extract_param(r'dur_([\d\._]+)_amp')
        params['amp'] = extract_param(r'amp_([\d\._]+)_ph')
        
        # Извлекаем фазы
        ph_match = re.search(r'ph([\d\._]+)(\.npz)?$', stem)
        if ph_match:
            ph_str = ph_match.group(1)
            if '_' in ph_str:
                params['ph'] = list(map(float, ph_str.split('_')))
            else:
                params['ph'] = [float(ph_str[i:i+4]) for i in range(0, len(ph_str), 4)]
        else:
            params['ph'] = [0.0] * 4
        
        return params

    @staticmethod
    def analyze_time_series(data, time):
        """Анализ временных рядов"""
        results = {}
        
        #1. Анализ нейронной активности (V)
        if 'V' in data:
            V_data = data['V']
            for i in range(V_data.shape[1]):
                V = V_data[:, i]
                results[f'neuron_{i}_mean'] = np.mean(V)
                results[f'neuron_{i}_std'] = np.std(V)
                
                # Нахождение спайков
                spikes = V > -20
                spike_times = time[spikes]
                
                if len(spike_times) > 1:
                    isi = np.diff(spike_times)
                    mean_isi = np.mean(isi)
                    results[f'neuron_{i}_freq'] = 1 / mean_isi
                    results[f'neuron_{i}_cv'] = np.std(isi) / mean_isi if mean_isi > 0 else 0
                else:
                    results[f'neuron_{i}_freq'] = 0
                    results[f'neuron_{i}_cv'] = 0
        
        # 2. Анализ мышечных сил
        for muscle in ['F_flex', 'F_ext']:
            if muscle in data:
                F = data[muscle]
                results[f'{muscle}_mean'] = np.mean(F)
                results[f'{muscle}_max'] = np.max(F)
                results[f'{muscle}_min'] = np.min(F)
                results[f'{muscle}_std'] = np.std(F)
                
                if len(F) > 10:
                    try:
                        peaks, _ = find_peaks(F, prominence=np.std(F)/2)
                        if len(peaks) > 1:
                            peak_times = time[peaks]
                            isi = np.diff(peak_times)
                            mean_isi = np.mean(isi)
                            results[f'{muscle}_freq'] = 1 / mean_isi if mean_isi > 0 else 0
                            results[f'{muscle}_cv'] = np.std(isi) / mean_isi if mean_isi > 0 else 0
                        else:
                            results[f'{muscle}_freq'] = 0
                            results[f'{muscle}_cv'] = 0
                    except:
                        results[f'{muscle}_freq'] = 0
                        results[f'{muscle}_cv'] = 0
                    
                    # Спектральный анализ
                    fs = 1 / (time[1] - time[0])
                    if len(F) >= 1024:
                        f, Pxx = welch(F, fs, nperseg=1024)
                        results[f'{muscle}_dom_freq'] = f[np.argmax(Pxx)]
                    else:
                        results[f'{muscle}_dom_freq'] = 0
        
        # 3. Соотношение сил
        if 'F_flex' in data and 'F_ext' in data:
            ratio = data['F_flex'] / (data['F_ext'] + 1e-6)
            results['flex_ext_ratio_mean'] = np.mean(ratio)
            results['flex_ext_ratio_std'] = np.std(ratio)
        
        return results

    @staticmethod
    def classify_operation_mode(results):
        """Классификация режима работы"""
        neuron_freqs = [results.get(f'neuron_{i}_freq', 0) for i in range(4)]
        freq_std = np.std([f for f in neuron_freqs if f > 0])
        
        flex_cv = results.get('F_flex_cv', 1)
        ext_cv = results.get('F_ext_cv', 1)
        mean_cv = (flex_cv + ext_cv) / 2 if flex_cv > 0 and ext_cv > 0 else 1
        
        if 'F_flex_mean' not in results or 'F_ext_mean' not in results:
            if freq_std < 0.1: return "Стабильный нейронный"
            if freq_std < 0.3: return "Ритмический нейронный"
            return "Хаотический нейронный"
        
        if freq_std < 0.1 and mean_cv < 0.1:
            return "Стабильный синхронный"
        if freq_std < 0.3 and mean_cv < 0.3:
            return "Ритмический"
        if results.get('F_flex_dom_freq', 0) > 5 and results.get('F_ext_dom_freq', 0) > 5:
            return "Кластерный"
        return "Хаотический"

    def process_single_file(self, npz_path):
        """Обработка одного файла - ядро параллельной обработки"""
        try:
            # Загрузка данных с немедленным закрытием файла
            with np.load(npz_path) as data:
                # Создаем копию данных в памяти
                data_dict = {key: data[key].copy() for key in data.files}
                
            # Извлечение параметров из имени файла
            stem = npz_path.stem
            params = self.extract_params_from_filename(stem)
            
            # Подготовка временной оси
            time = data_dict['T'] if 'T' in data_dict else np.arange(len(next(iter(data_dict.values()))))
            
            # Анализ временных рядов
            results = self.analyze_time_series(data_dict, time)
            
            # Классификация режима
            results['mode'] = self.classify_operation_mode(results)
            
            # Добавляем метаданные
            results['filename'] = stem
            results['comb'] = params.get('comb', 0)
            
            for i in range(4):
                results[f'per_{i}'] = params['per'][i] if i < len(params['per']) else 0
                results[f'dur_{i}'] = params['dur'][i] if i < len(params['dur']) else 0
                results[f'amp_{i}'] = params['amp'][i] if i < len(params['amp']) else 0
                results[f'ph_{i}'] = params['ph'][i] if i < len(params['ph']) else 0
            
            return results
        
        except Exception as e:
            print(f"Ошибка обработки {npz_path.name}: {str(e)}")
            return {'filename': npz_path.stem, 'mode': f'Ошибка обработки: {str(e)}'}

    def load_data(self):
        """Загрузка файлов данных"""
        print(f"📁 Поиск файлов данных в {self.data_dir}...")
        self.npz_files = list(self.data_dir.glob('*.npz'))
        print(f"✅ Найдено {len(self.npz_files)} файлов данных")
        return self.npz_files

    def process_files_parallel(self, max_files=None, n_jobs=-1):
        """Параллельная обработка файлов"""
        if not self.npz_files:
            raise ValueError("Файлы данных не загружены. Вызовите load_data() сначала.")
        
        # Ограничиваем количество файлов если нужно
        files_to_process = self.npz_files
        if max_files:
            files_to_process = self.npz_files[:max_files]
        
        print(f"🔄 Обработка {len(files_to_process)} файлов данных...")
        
        with tqdm(total=len(files_to_process), desc="Обработка файлов") as pbar:
            processed_results = Parallel(n_jobs=n_jobs, verbose=0)(
                delayed(self.process_single_file)(npz_path) 
                for npz_path in files_to_process
            )
            pbar.update(len(files_to_process))
        
        # Фильтрация успешных результатов
        self.valid_results = [res for res in processed_results if res is not None and 'mode' in res]
        print(f"✅ Успешно обработано {len(self.valid_results)}/{len(files_to_process)} файлов")
        
        return self.valid_results

    def create_dataframe(self):
        """Создание DataFrame из результатов"""
        if not self.valid_results:
            raise ValueError("Нет обработанных результатов. Вызовите process_files_parallel() сначала.")
        
        self.results_df = pd.DataFrame(self.valid_results)
        
        # Гарантируем, что столбец 'mode' всегда есть
        if 'mode' not in self.results_df.columns:
            self.results_df['mode'] = 'Неизвестный'
        
        # Кодируем режимы для анализа
        le = LabelEncoder()
        self.results_df['mode_encoded'] = le.fit_transform(self.results_df['mode'])
        
        return self.results_df

    def save_results(self, filename="time_series_analysis.csv"):
        """Сохранение результатов анализа"""
        if self.results_df is None:
            raise ValueError("DataFrame не создан. Вызовите create_dataframe() сначала.")
        
        filepath = self.analysis_dir / filename
        self.results_df.to_csv(filepath, index=False)
        print(f"✅ Результаты анализа сохранены в {filepath}")
        
        # Также сохраняем обогащенные данные
        enriched_filepath = self.analysis_dir / "enriched_time_series_analysis.csv"
        self.results_df.to_csv(enriched_filepath, index=False)
        print(f"✅ Обогащенные данные сохранены в {enriched_filepath}")
        
        return filepath, enriched_filepath

    def get_mode_statistics(self):
        """Получение статистики по режимам работы"""
        if self.results_df is None:
            raise ValueError("DataFrame не создан. Вызовите create_dataframe() сначала.")
        
        mode_stats = self.results_df['mode'].value_counts()
        stats_dict = {}
        for mode, count in mode_stats.items():
            percentage = (count / len(self.results_df)) * 100
            stats_dict[mode] = {'count': count, 'percentage': percentage}
        
        return stats_dict

    def print_mode_statistics(self):
        """Вывод статистики по режимам работы"""
        stats = self.get_mode_statistics()
        print("\n📊 Статистика по режимам работы:")
        for mode, info in stats.items():
            print(f"  {mode}: {info['count']} файлов ({info['percentage']:.1f}%)")

    def get_results(self):
        """Получение результатов анализа"""
        return self.results_df

    def get_valid_results(self):
        """Получение валидных результатов"""
        return self.valid_results

    def get_processed_files_count(self):
        """Получение количества обработанных файлов"""
        return len(self.valid_results)

    def get_total_files_count(self):
        """Получение общего количества файлов"""
        return len(self.npz_files)