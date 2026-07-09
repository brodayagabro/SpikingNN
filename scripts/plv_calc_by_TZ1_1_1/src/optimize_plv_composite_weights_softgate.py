r"""
Optimize weights + alpha for soft peak-gate:
  g = exp(-alpha * max(0, peaks-2))

Goal: maximize AUC for anti_phase vs not_anti_phase (exclude auto_inactive).

Writes:
  - merged CSV column `plv_composite_soft_opt`
  - JSON with best params into Obsidian Статья
  - appends summary into Report_PLV_vs_proxy.md
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(
    r"c:\Users\vvguba\OneDrive\Рабочий стол\лаборатория нейробиоморфных технологий\Статья_на_май"
)
MERGED_PATH = PROJECT_ROOT / "data" / "merged_only_data_muscles_auto_inactive.csv"
OUT_DIR = Path(r"C:\Users\vvguba\OneDrive\Документы\Obsidian Vault\Статья")
REPORT_PATH = OUT_DIR / "Report_PLV_vs_proxy.md"
JSON_PATH = OUT_DIR / "best_plv_composite_softgate.json"


def clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def score_phase180_deg(phi_deg: np.ndarray) -> np.ndarray:
    d = np.abs(np.abs(phi_deg) - 180.0)
    return clip01(1.0 - d / 90.0)


def score_corr_antiphase(r: np.ndarray) -> np.ndarray:
    return clip01((1.0 - r) / 2.0)


def score_peakcount_similarity(n1: np.ndarray, n2: np.ndarray) -> np.ndarray:
    n1 = np.maximum(n1, 0.0)
    n2 = np.maximum(n2, 0.0)
    ok = (n1 >= 3) & (n2 >= 3)
    sim = 1.0 - (np.abs(n1 - n2) / np.maximum(np.maximum(n1, n2), 1.0))
    sim = clip01(sim)
    return np.where(ok, sim, 0.0)


def auc_rank(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(scores, dtype=float)
    m = np.isfinite(s)
    y = y[m]
    s = s[m]
    n1 = int(y.sum())
    n0 = int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s)  # asc
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(s), dtype=float) + 1.0
    vals = s[order]
    start = 0
    while start < len(vals):
        end = start + 1
        while end < len(vals) and vals[end] == vals[start]:
            end += 1
        if end - start > 1:
            avg = (start + 1 + end) / 2.0
            ranks[order[start:end]] = avg
        start = end
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def soft_gate(peaks: np.ndarray, alpha: float) -> np.ndarray:
    a = float(max(0.0, alpha))
    return np.exp(-a * np.maximum(0.0, peaks - 2.0))


@dataclass
class Best:
    auc: float
    w: np.ndarray
    alpha: float


def random_simplex(rng: np.random.Generator, n: int, k: int) -> np.ndarray:
    x = rng.random((n, k))
    x = x / x.sum(axis=1, keepdims=True)
    return x


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(MERGED_PATH)

    mask = df["mark"].isin(["anti_phase", "not_anti_phase"]).values
    d = df.loc[mask].copy()
    y = (d["mark"].astype(str) == "anti_phase").values

    plv = pd.to_numeric(d["plv_hilbert"], errors="coerce").values
    amp = pd.to_numeric(d["plv_comp_amp_score"], errors="coerce").values
    phase_h = pd.to_numeric(d["plv_comp_phase_score"], errors="coerce").values
    lag = pd.to_numeric(d["plv_phase_lag_deg"], errors="coerce").values
    corr = pd.to_numeric(d["plv_filtered_corr"], errors="coerce").values
    n1 = pd.to_numeric(d["plv_n_peaks_flex"], errors="coerce").fillna(0).values
    n2 = pd.to_numeric(d["plv_n_peaks_ext"], errors="coerce").fillna(0).values

    env1 = pd.to_numeric(d["plv_comp_env_peaks_flex"], errors="coerce").fillna(999).values
    env2 = pd.to_numeric(d["plv_comp_env_peaks_ext"], errors="coerce").fillna(999).values
    peaks = np.maximum(env1, env2)

    comps: List[Tuple[str, np.ndarray]] = [
        ("plv", clip01(plv)),
        ("amp", clip01(amp)),
        ("phase180_hilbert", clip01(phase_h)),
        ("phase180_peaks", score_phase180_deg(lag)),
        ("corr_antiphase", score_corr_antiphase(corr)),
        ("peakcount_similarity", score_peakcount_similarity(n1, n2)),
    ]
    names = [n for n, _ in comps]
    X = np.vstack([v for _, v in comps]).T
    X = np.where(np.isfinite(X), X, 0.0)

    rng = np.random.default_rng(42)
    k = X.shape[1]

    # Search space
    W = random_simplex(rng, n=120000, k=k)
    alphas = rng.uniform(0.0, 3.0, size=len(W))

    best = Best(auc=-1.0, w=np.ones(k) / k, alpha=0.0)
    for w, a in zip(W, alphas):
        s = X @ w
        g = soft_gate(peaks, float(a))
        s = s * g
        auc = auc_rank(y, s)
        if np.isfinite(auc) and auc > best.auc:
            best = Best(auc=float(auc), w=w.copy(), alpha=float(a))

    # Apply to full dataset
    plv_all = pd.to_numeric(df["plv_hilbert"], errors="coerce").values
    amp_all = pd.to_numeric(df["plv_comp_amp_score"], errors="coerce").values
    phase_all = pd.to_numeric(df["plv_comp_phase_score"], errors="coerce").values
    lag_all = pd.to_numeric(df["plv_phase_lag_deg"], errors="coerce").values
    corr_all = pd.to_numeric(df["plv_filtered_corr"], errors="coerce").values
    n1_all = pd.to_numeric(df["plv_n_peaks_flex"], errors="coerce").fillna(0).values
    n2_all = pd.to_numeric(df["plv_n_peaks_ext"], errors="coerce").fillna(0).values
    env1_all = pd.to_numeric(df["plv_comp_env_peaks_flex"], errors="coerce").fillna(999).values
    env2_all = pd.to_numeric(df["plv_comp_env_peaks_ext"], errors="coerce").fillna(999).values
    peaks_all = np.maximum(env1_all, env2_all)

    X_all = np.vstack(
        [
            clip01(plv_all),
            clip01(amp_all),
            clip01(phase_all),
            score_phase180_deg(lag_all),
            score_corr_antiphase(corr_all),
            score_peakcount_similarity(n1_all, n2_all),
        ]
    ).T
    X_all = np.where(np.isfinite(X_all), X_all, 0.0)
    s_all = (X_all @ best.w) * soft_gate(peaks_all, best.alpha)

    df["plv_composite_soft_opt"] = s_all
    df.to_csv(MERGED_PATH, index=False, encoding="utf-8-sig")

    payload = {
        "setting": "anti_phase vs not_anti_phase",
        "auc": best.auc,
        "alpha": best.alpha,
        "components": names,
        "weights": {names[i]: float(best.w[i]) for i in range(len(names))},
        "gate": "g = exp(-alpha * max(0, peaks-2)), peaks=max(env_peaks_flex, env_peaks_ext)",
        "search": {"weights_samples": int(len(W)), "alpha_range": [0.0, 3.0], "seed": 42},
    }
    JSON_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if REPORT_PATH.exists():
        appendix = (
            "\n\n## Soft-gate оптимизация (веса + α)\n\n"
            "Использован мягкий множитель:\n\n"
            "\\[\n"
            "g=\\exp\\left(-\\alpha\\max(0,\\mathrm{peaks}-2)\\right),\\quad \\mathrm{peaks}=\\max(peaks_{flex},peaks_{ext})\n"
            "\\]\n\n"
            f"- Лучший AUC (anti_phase vs not_anti_phase): **{payload['auc']:.4f}**\n"
            f"- Лучший α: **{payload['alpha']:.4f}**\n"
            f"- JSON: `{JSON_PATH}`\n"
        )
        REPORT_PATH.write_text(REPORT_PATH.read_text(encoding="utf-8") + appendix, encoding="utf-8")


if __name__ == "__main__":
    main()

