"""Вынесено для multiprocessing (spawn на Windows)."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, hilbert, sosfiltfilt

SAMPLING_RATE_DEFAULT = 1000.0
FREQ_BAND = (1.0, 10.0)
FLEX_COL = "F_flex"
EXT_COL = "F_ext"
TIME_COL_PREFERRED = "T"
N_BOOTSTRAP = 1000
CONFIDENCE_LEVEL = 0.95
BUTTER_ORDER = 4


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _phase_antiphase_score_deg(mean_phase_deg: float) -> float:
    """
    Score in [0,1]: 1 near 180°, 0 near 90° or 0°.
    Uses distance d = | |phi| - 180 | (phi in [-180,180]).
    """
    if not np.isfinite(mean_phase_deg):
        return float("nan")
    d = abs(abs(float(mean_phase_deg)) - 180.0)  # 0 at 180°, 180 at 0°
    return _clip01(1.0 - d / 90.0)


def _amplitude_balance_score(sig1: np.ndarray, sig2: np.ndarray) -> float:
    """
    Score in [0,1] based on RMS amplitude ratio symmetry:
      r = rms1/rms2 ; score = min(r, 1/r)
    """
    s1 = np.asarray(sig1, dtype=np.float64)
    s2 = np.asarray(sig2, dtype=np.float64)
    rms1 = float(np.sqrt(np.mean(s1 * s1))) if len(s1) else float("nan")
    rms2 = float(np.sqrt(np.mean(s2 * s2))) if len(s2) else float("nan")
    if not np.isfinite(rms1) or not np.isfinite(rms2) or rms1 <= 0 or rms2 <= 0:
        return float("nan")
    r = rms1 / rms2
    return _clip01(min(r, 1.0 / r))


def _envelope_peak_count(sig_filt: np.ndarray, fs: float) -> int:
    """
    Count peaks of amplitude envelope (Hilbert abs), after heavy smoothing.
    Intended to detect burst-like amplitude structure (few peaks).
    """
    x = np.asarray(sig_filt, dtype=np.float64)
    if len(x) < 10 or not np.isfinite(x).all():
        return 0
    env = np.abs(hilbert(x))
    # Smooth ~0.5 s moving average (clipped to reasonable sizes)
    win = int(max(5, min(len(env) // 10, round(fs * 0.5))))
    kernel = np.ones(win, dtype=np.float64) / float(win)
    env_s = np.convolve(env, kernel, mode="same")
    prom = float(np.std(env_s) * 0.5)
    dist = int(max(1, round(fs * 0.5)))
    peaks, _ = find_peaks(env_s, prominence=prom, distance=dist)
    return int(len(peaks))


def _soft_gate_from_env_peaks(env_peaks_flex: int, env_peaks_ext: int, alpha: float) -> float:
    """
    Soft gate:
      g = exp(-alpha * max(0, peaks - 2))
    where peaks = max(peaks_flex, peaks_ext).
    """
    p = float(max(int(env_peaks_flex), int(env_peaks_ext)))
    a = float(alpha)
    if not np.isfinite(a) or a < 0:
        a = 0.0
    return float(np.exp(-a * max(0.0, p - 2.0)))

def infer_sampling_rate_from_t(t: np.ndarray, freq_high: float = FREQ_BAND[1]) -> float:
    t = np.asarray(t, dtype=float)
    if len(t) < 4:
        return SAMPLING_RATE_DEFAULT
    dt = float(np.median(np.diff(t)))
    if not math.isfinite(dt) or dt <= 0:
        return SAMPLING_RATE_DEFAULT
    fs = 1.0 / dt
    need = 2.5 * freq_high
    if fs < need:
        fs_ms = 1.0 / (dt * 1e-3)
        if fs_ms >= need:
            return fs_ms
    return fs


def bandpass_filter(
    signal: np.ndarray,
    sampling_rate: float,
    low_freq: float = 1.0,
    high_freq: float = 10.0,
    order: int = BUTTER_ORDER,
) -> np.ndarray:
    nyquist = sampling_rate / 2.0
    low = max(low_freq / nyquist, 0.001)
    high = min(high_freq / nyquist, 0.999)
    if low >= high:
        return signal
    # SOS + sosfiltfilt: численно устойчивее filtfilt при высокой Fs (как у рядов в data_muscles)
    sos = butter(order, [low, high], btype="band", output="sos")
    pad = min(3 * order, len(signal) // 3)
    sig = np.asarray(signal, dtype=np.float64)
    return sosfiltfilt(sos, sig, padlen=pad)


def extract_phase(
    signal: np.ndarray, sampling_rate: float, freq_band: Tuple[float, float]
) -> np.ndarray:
    signal = signal - np.mean(signal)
    filtered = bandpass_filter(signal, sampling_rate, freq_band[0], freq_band[1])
    if np.std(filtered) < 1e-10:
        return np.zeros_like(filtered)
    analytic = hilbert(filtered)
    return np.angle(analytic)


def calculate_plv_with_bootstrap(
    phase_1: np.ndarray,
    phase_2: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    confidence_level: float = CONFIDENCE_LEVEL,
) -> Dict:
    if len(phase_1) != len(phase_2):
        min_len = min(len(phase_1), len(phase_2))
        phase_1, phase_2 = phase_1[:min_len], phase_2[:min_len]

    phase_diff = np.angle(np.exp(1j * (phase_1 - phase_2)))
    plv_base = np.abs(np.mean(np.exp(1j * phase_diff)))
    mean_phase_lag_rad = float(np.angle(np.mean(np.exp(1j * phase_diff))))
    mean_phase_lag_deg = float(np.degrees(mean_phase_lag_rad))

    plv_boot = np.zeros(n_bootstrap)
    np.random.seed(42)
    n = len(phase_diff)
    for i in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        plv_boot[i] = np.abs(np.mean(np.exp(1j * phase_diff[idx])))

    alpha = 1 - confidence_level
    return {
        "plv": plv_base,
        "mean_phase_lag_rad": mean_phase_lag_rad,
        "mean_phase_lag_deg": mean_phase_lag_deg,
        "plv_std": np.std(plv_boot, ddof=1),
        "plv_ci_lower": np.percentile(plv_boot, 100 * alpha / 2),
        "plv_ci_upper": np.percentile(plv_boot, 100 * (1 - alpha / 2)),
    }


def calculate_phase_lag_peaks(
    sig1: np.ndarray,
    sig2: np.ndarray,
    T: np.ndarray,
    sr: float,
    freq_band: Tuple[float, float],
) -> Dict:
    s1 = bandpass_filter(sig1 - np.mean(sig1), sr, *freq_band)
    s2 = bandpass_filter(sig2 - np.mean(sig2), sr, *freq_band)

    prom1, prom2 = np.std(s1) * 0.5, np.std(s2) * 0.5
    min_dist = int(sr * 0.1)

    p1, _ = find_peaks(s1, prominence=prom1, distance=min_dist)
    p2, _ = find_peaks(s2, prominence=prom2, distance=min_dist)

    if len(p1) < 2 or len(p2) < 2:
        return {"phase_lag_deg": np.nan, "n_peaks_1": len(p1), "n_peaks_2": len(p2)}

    t1 = T[p1] if len(T) >= len(sig1) else p1 / sr
    t2 = T[p2] if len(T) >= len(sig2) else p2 / sr

    offsets = np.array([t2[np.argmin(np.abs(t2 - tt))] for tt in t1])
    period = np.mean(np.diff(t1)) if len(t1) > 1 else 1.0
    phase_deg = np.mod((offsets / period) * 360 + 180, 360) - 180

    return {
        "phase_lag_deg": float(np.median(phase_deg)),
        "phase_lag_std": float(np.std(phase_deg)),
        "n_peaks_1": len(p1),
        "n_peaks_2": len(p2),
    }


def compute_plv_for_muscle_csv(path: Path) -> Dict[str, float]:
    out: Dict[str, float] = {
        "plv_hilbert": np.nan,
        "plv_hilbert_std": np.nan,
        "plv_hilbert_ci_lower": np.nan,
        "plv_hilbert_ci_upper": np.nan,
        "phase_shift_hilbert_deg": np.nan,
        "plv_composite_anti": np.nan,  # legacy name; now weighted scheme
        "plv_comp_amp_score": np.nan,
        "plv_comp_phase_score": np.nan,
        "plv_comp_env_peaks_flex": np.nan,
        "plv_comp_env_peaks_ext": np.nan,
        "plv_comp_zeroed_by_peaks": np.nan,
        "plv_comp_gate_alpha": np.nan,
        "plv_comp_gate_g": np.nan,
        "plv_comp_w_plv": np.nan,
        "plv_comp_w_amp": np.nan,
        "plv_comp_w_phase": np.nan,
        "plv_phase_lag_deg": np.nan,
        "plv_phase_lag_std": np.nan,
        "plv_filtered_corr": np.nan,
        "plv_n_peaks_flex": np.nan,
        "plv_n_peaks_ext": np.nan,
        "plv_sampling_hz": np.nan,
        "plv_error": np.nan,
    }
    try:
        df = pd.read_csv(path)
    except Exception:
        out["plv_error"] = 1.0
        return out

    time_col = TIME_COL_PREFERRED if TIME_COL_PREFERRED in df.columns else "t"
    if time_col not in df.columns:
        out["plv_error"] = 2.0
        return out
    if FLEX_COL not in df.columns or EXT_COL not in df.columns:
        out["plv_error"] = 3.0
        return out

    T = df[time_col].values.astype(float)
    F_flex = df[FLEX_COL].values.astype(float)
    F_ext = df[EXT_COL].values.astype(float)

    min_len = min(len(F_flex), len(F_ext), len(T))
    F_flex, F_ext, T = F_flex[:min_len], F_ext[:min_len], T[:min_len]

    sampling_rate = infer_sampling_rate_from_t(T)
    out["plv_sampling_hz"] = float(sampling_rate)

    F_flex_filt = bandpass_filter(F_flex - np.mean(F_flex), sampling_rate, *FREQ_BAND)
    F_ext_filt = bandpass_filter(F_ext - np.mean(F_ext), sampling_rate, *FREQ_BAND)

    phase_flex = extract_phase(F_flex, sampling_rate, FREQ_BAND)
    phase_ext = extract_phase(F_ext, sampling_rate, FREQ_BAND)

    plv_results = calculate_plv_with_bootstrap(
        phase_flex, phase_ext, N_BOOTSTRAP, CONFIDENCE_LEVEL
    )
    peak_results = calculate_phase_lag_peaks(F_flex, F_ext, T, sampling_rate, FREQ_BAND)

    if min_len > 2:
        s1 = np.asarray(F_flex_filt, dtype=np.float64)
        s2 = np.asarray(F_ext_filt, dtype=np.float64)
        if np.std(s1) > 1e-12 and np.std(s2) > 1e-12:
            corr_val = float(np.corrcoef(s1, s2)[0, 1])
        else:
            corr_val = np.nan
    else:
        corr_val = np.nan

    out["plv_hilbert"] = float(plv_results["plv"])
    out["plv_hilbert_std"] = float(plv_results["plv_std"])
    out["plv_hilbert_ci_lower"] = float(plv_results["plv_ci_lower"])
    out["plv_hilbert_ci_upper"] = float(plv_results["plv_ci_upper"])
    out["phase_shift_hilbert_deg"] = float(plv_results["mean_phase_lag_deg"])
    out["plv_phase_lag_deg"] = float(peak_results.get("phase_lag_deg", np.nan))
    out["plv_phase_lag_std"] = float(peak_results.get("phase_lag_std", np.nan))
    out["plv_n_peaks_flex"] = float(peak_results.get("n_peaks_1", np.nan))
    out["plv_n_peaks_ext"] = float(peak_results.get("n_peaks_2", np.nan))
    out["plv_filtered_corr"] = float(corr_val) if np.isfinite(corr_val) else np.nan

    # --- Composite PLV-based anti-phase metric (weighted scheme) ---
    amp_score = _amplitude_balance_score(F_flex_filt, F_ext_filt)
    phase_score = _phase_antiphase_score_deg(out["phase_shift_hilbert_deg"])
    env_peaks_flex = _envelope_peak_count(F_flex_filt, sampling_rate)
    env_peaks_ext = _envelope_peak_count(F_ext_filt, sampling_rate)

    out["plv_comp_amp_score"] = float(amp_score) if np.isfinite(amp_score) else np.nan
    out["plv_comp_phase_score"] = float(phase_score) if np.isfinite(phase_score) else np.nan
    out["plv_comp_env_peaks_flex"] = float(env_peaks_flex)
    out["plv_comp_env_peaks_ext"] = float(env_peaks_ext)

    # Default weights (can be re-fit later on merged dataset)
    w_plv, w_amp, w_phase = 0.50, 0.20, 0.30
    out["plv_comp_w_plv"] = float(w_plv)
    out["plv_comp_w_amp"] = float(w_amp)
    out["plv_comp_w_phase"] = float(w_phase)

    base = float(out["plv_hilbert"])
    if np.isfinite(base) and np.isfinite(amp_score) and np.isfinite(phase_score):
        w_sum = w_plv + w_amp + w_phase
        comp = (w_plv * base + w_amp * float(amp_score) + w_phase * float(phase_score)) / w_sum
    else:
        comp = float("nan")

    # Soft gate (replaces hard zeroing). Keep the old indicator for analysis.
    alpha_gate = 1.0
    g = _soft_gate_from_env_peaks(env_peaks_flex, env_peaks_ext, alpha_gate)
    out["plv_comp_gate_alpha"] = float(alpha_gate)
    out["plv_comp_gate_g"] = float(g)

    out["plv_comp_zeroed_by_peaks"] = 1.0 if ((env_peaks_flex > 2) or (env_peaks_ext > 2)) else 0.0
    out["plv_composite_anti"] = float(g * comp) if np.isfinite(comp) else np.nan

    return out


def worker_compute_plv(path_str: str) -> Tuple[str, Dict[str, float]]:
    return path_str, compute_plv_for_muscle_csv(Path(path_str))
