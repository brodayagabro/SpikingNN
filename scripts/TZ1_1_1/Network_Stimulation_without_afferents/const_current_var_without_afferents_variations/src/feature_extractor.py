import numpy as np
from scipy.signal import find_peaks, welch
from scipy.fft import fft, fftfreq
import pandas as pd

def compute_all_features(F_flex, F_ext, V, U, Afferents, Q, W, T):
    """
    Вычисляет все фичи, аналогично методу process_single_file из TimeSeriesAnalyzer.
    """
    features = {}

    # --- Метрики нейронов (V) ---
    if V.size > 0:
        for i in range(V.shape[1]):
            V_signal = V[:, i]
            features[f'neuron_{i}_V_mean'] = np.mean(V_signal)
            features[f'neuron_{i}_V_std'] = np.std(V_signal)

            # Нахождение спайков
            spikes = V_signal > -20
            spike_times = T[spikes]

            if len(spike_times) > 1:
                isi = np.diff(spike_times)
                mean_isi = np.mean(isi)
                freq = 1 / mean_isi if mean_isi > 0 else 0
                cv = np.std(isi) / mean_isi if mean_isi > 0 else 0
            else:
                freq = 0
                cv = 0
            features[f'neuron_{i}_freq'] = freq
            features[f'neuron_{i}_cv'] = cv

    # --- Метрики мышц (F_flex, F_ext) ---
    for muscle_signal, muscle_name in [(F_flex, 'F_flex'), (F_ext, 'F_ext')]:
        if muscle_signal.size > 0:
            features[f'{muscle_name}_mean'] = np.mean(muscle_signal)
            features[f'{muscle_name}_max'] = np.max(muscle_signal)
            features[f'{muscle_name}_min'] = np.min(muscle_signal)
            features[f'{muscle_name}_std'] = np.std(muscle_signal)

            if len(muscle_signal) > 10:
                try:
                    peaks, _ = find_peaks(muscle_signal, prominence=np.std(muscle_signal)/2)
                    if len(peaks) > 1:
                        peak_times = T[peaks]
                        isi = np.diff(peak_times)
                        mean_isi = np.mean(isi)
                        freq = 1 / mean_isi if mean_isi > 0 else 0
                        cv = np.std(isi) / mean_isi if mean_isi > 0 else 0
                    else:
                        freq = 0
                        cv = 0
                except:
                    freq = 0
                    cv = 0
                features[f'{muscle_name}_freq'] = freq
                features[f'{muscle_name}_cv'] = cv

                # Спектральный анализ
                fs = 1 / (T[1] - T[0]) if len(T) > 1 else 1
                if len(muscle_signal) >= 1024:
                    f, Pxx = welch(muscle_signal, fs, nperseg=1024)
                    features[f'{muscle_name}_dom_freq'] = f[np.argmax(Pxx)] if len(f) > 0 else 0
                else:
                    features[f'{muscle_name}_dom_freq'] = 0
            else:
                features[f'{muscle_name}_freq'] = 0
                features[f'{muscle_name}_cv'] = 0
                features[f'{muscle_name}_dom_freq'] = 0

    # --- Соотношение сил ---
    if F_flex.size > 0 and F_ext.size > 0:
        ratio = F_flex / (F_ext + 1e-6) # Добавляем малую величину, чтобы избежать деления на 0
        features['flex_ext_ratio_mean'] = np.mean(ratio)
        features['flex_ext_ratio_std'] = np.std(ratio)

    # --- Метрики U ---
    if U.size > 0:
        for i in range(U.shape[1]):
            features[f'neuron_{i}_U_mean'] = np.mean(U[:, i])
            features[f'neuron_{i}_U_std'] = np.std(U[:, i])

    # --- Метрики афферентов ---
    if Afferents.size > 0:
        for i in range(Afferents.shape[1]):
            aff_signal = Afferents[:, i]
            features[f'aff_{i}_mean'] = np.mean(aff_signal)
            features[f'aff_{i}_std'] = np.std(aff_signal)
            features[f'aff_{i}_max'] = np.max(aff_signal)
            features[f'aff_{i}_min'] = np.min(aff_signal)

    # --- Метрики сустава (Q, W) ---
    if Q.size > 0:
        features['q_mean'] = np.mean(Q)
        features['q_std'] = np.std(Q)
        features['q_max'] = np.max(Q)
        features['q_min'] = np.min(Q)
    if W.size > 0:
        features['w_mean'] = np.mean(W)
        features['w_std'] = np.std(W)
        features['w_max'] = np.max(W)
        features['w_min'] = np.min(W)

    # --- Корреляции ---
    if F_flex.size > 0 and F_ext.size > 0:
        corr_val = np.corrcoef(F_flex, F_ext)[0,1] if len(F_flex) > 1 else 0
        features['F_flex_F_ext_corr'] = corr_val if not np.isnan(corr_val) else 0
    if Q.size > 0 and W.size > 0:
        corr_val = np.corrcoef(Q, W)[0,1] if len(Q) > 1 else 0
        features['q_w_corr'] = corr_val if not np.isnan(corr_val) else 0

    # --- Средние и std для V и U по всем нейронам ---
    if V.size > 0:
        features['V_mean_overall'] = np.mean(V)
        features['V_std_overall'] = np.std(V)
    if U.size > 0:
        features['U_mean_overall'] = np.mean(U)
        features['U_std_overall'] = np.std(U)

    return features

# --- Добавляем функцию классификации ---
def classify_operation_mode(results):
    """Классификация режима работы (копия из Process_data_pulses1.txt)"""
    neuron_freqs = [results.get(f'neuron_{i}_freq', 0) for i in range(4)]
    # Ограничиваемся первыми 4 нейронами, как в оригинале
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
