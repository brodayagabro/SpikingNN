r"""
По строкам CSV (как в merged: есть `muscle_csv_file` и прочие метаданные):
  1) находит CSV временного ряда мышц в заданной папке (по имени файла из `muscle_csv_file`);
  2) считает PLV-признаки (`_muscle_plv_worker.compute_plv_for_muscle_csv`);
  3) записывает их в те же строки;
  4) считает `plv_composite_soft_opt` по сохранённым весам + α (`data/best_plv_composite_softgate.json`);
  5) считает скор полного HistGB (`data/histgb_anti_phase_vs_not_anti_phase.joblib`) → колонка `histgb_full_predict_p_anti_phase`.

Запуск из корня проекта:
  python infer_histgb_softgate_from_muscle_csv.py --input in.csv --muscle-dir "path/to/csv" --output out.csv

Параллельный расчёт PLV по строкам: `joblib.Parallel` (по умолчанию все ядра, см. `--jobs`).

Требуются: pandas, numpy, scipy, scikit-learn, joblib, tqdm (как для PLV-пайплайна).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Hashable

import joblib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from _muscle_plv_worker import compute_plv_for_muscle_csv
from gb_compare_muscle_features import build_feature_matrix
from optimize_plv_composite_weights_softgate import (
    clip01,
    score_corr_antiphase,
    score_peakcount_similarity,
    score_phase180_deg,
    soft_gate,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_HISTGB = PROJECT_ROOT / "data" / "histgb_anti_phase_vs_not_anti_phase.joblib"
DEFAULT_SOFTGATE_JSON = PROJECT_ROOT / "data" / "best_plv_composite_softgate.json"

PLV_COLUMNS: list[str] = [
    "plv_hilbert",
    "plv_hilbert_std",
    "plv_hilbert_ci_lower",
    "plv_hilbert_ci_upper",
    "phase_shift_hilbert_deg",
    "plv_comp_amp_score",
    "plv_comp_phase_score",
    "plv_comp_env_peaks_flex",
    "plv_comp_env_peaks_ext",
    "plv_comp_zeroed_by_peaks",
    "plv_comp_gate_alpha",
    "plv_comp_gate_g",
    "plv_comp_w_plv",
    "plv_comp_w_amp",
    "plv_comp_w_phase",
    "plv_phase_lag_deg",
    "plv_phase_lag_std",
    "plv_filtered_corr",
    "plv_n_peaks_flex",
    "plv_n_peaks_ext",
    "plv_sampling_hz",
    "plv_error",
    "plv_composite_anti",
]


def resolve_muscle_csv(muscle_cell: str, muscle_dir: Path) -> Path | None:
    raw = str(muscle_cell).strip().strip('"')
    if not raw or raw.lower() == "nan":
        return None
    p = Path(raw)
    name = p.name
    candidates = [
        muscle_dir / name,
        muscle_dir / p,
        PROJECT_ROOT / p,
        PROJECT_ROOT / "data_muscles" / name,
        PROJECT_ROOT / "data" / name,
    ]
    for c in candidates:
        try:
            if c.is_file():
                return c.resolve()
        except OSError:
            continue
    return None


def _plv_row_result(idx: Hashable, muscle_cell: str, muscle_dir_str: str) -> tuple[Hashable, dict[str, float]]:
    """
    Одна строка манифеста: путь к CSV + compute_plv_for_muscle_csv.
    Вынесено на уровень модуля для joblib (pickle воркеров на Windows).
    """
    muscle_dir = Path(muscle_dir_str)
    path = resolve_muscle_csv(muscle_cell, muscle_dir)
    if path is None:
        feats = {k: np.nan for k in PLV_COLUMNS}
        feats["plv_error"] = 4.0
    else:
        raw = compute_plv_for_muscle_csv(path)
        feats = {k: float(raw.get(k, np.nan)) if k in raw else np.nan for k in PLV_COLUMNS}
    return idx, feats


def load_softgate(path: Path) -> tuple[np.ndarray, list[str], float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    names: list[str] = list(payload["components"])
    w = np.array([float(payload["weights"][k]) for k in names], dtype=np.float64)
    alpha = float(payload["alpha"])
    return w, names, alpha


def vector_plv_composite_soft_opt(df: pd.DataFrame, w: np.ndarray, alpha: float) -> np.ndarray:
    """Та же формула, что в optimize_plv_composite_weights_softgate.main (блок Apply to full dataset)."""
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
    ).T.astype(np.float64)
    X_all = np.where(np.isfinite(X_all), X_all, 0.0)
    s_all = (X_all @ w) * soft_gate(peaks_all, alpha)
    return s_all.astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser(description="PLV features + HistGB + plv_composite_soft_opt on a CSV manifest.")
    ap.add_argument("--input", "-i", type=Path, required=True, help="Входной CSV (должна быть колонка muscle_csv_file).")
    ap.add_argument(
        "--muscle-dir",
        "-m",
        type=Path,
        required=True,
        help="Папка с CSV рядов (ищется файл с именем как в muscle_csv_file).",
    )
    ap.add_argument("--output", "-o", type=Path, required=True, help="Куда записать CSV с новыми колонками.")
    ap.add_argument("--histgb-model", type=Path, default=DEFAULT_HISTGB, help="joblib от gb_compare_muscle_features.py")
    ap.add_argument("--softgate-json", type=Path, default=None, help="best_plv_composite_softgate.json")
    ap.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=-1,
        help="Число процессов joblib для PLV по строкам (-1 = все ядра, 1 = без параллелизма).",
    )
    args = ap.parse_args()

    muscle_dir: Path = args.muscle_dir.resolve()
    if not muscle_dir.is_dir():
        raise SystemExit(f"Не папка: {muscle_dir}")

    sg_path = args.softgate_json
    if sg_path is None:
        cand = DEFAULT_SOFTGATE_JSON
        if cand.is_file():
            sg_path = cand
        if sg_path is None:
            raise SystemExit(
                "Не найден best_plv_composite_softgate.json — укажите --softgate-json "
                f"или положите файл в {DEFAULT_SOFTGATE_JSON}"
            )
    sg_path = sg_path.resolve()

    if not args.histgb_model.is_file():
        raise SystemExit(f"Нет модели бустинга: {args.histgb_model}. Сначала запустите gb_compare_muscle_features.py")

    df = pd.read_csv(args.input, encoding="utf-8-sig")
    if "muscle_csv_file" not in df.columns:
        raise SystemExit("В CSV нет колонки muscle_csv_file")

    for c in PLV_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan

    w_soft, _, alpha_soft = load_softgate(sg_path)

    muscle_dir_str = str(muscle_dir.resolve())
    tasks = [(idx, str(df.at[idx, "muscle_csv_file"])) for idx in df.index]

    if args.jobs == 1:
        for idx, rel in tqdm(tasks, desc="PLV по строкам", unit="row"):
            _, feats = _plv_row_result(idx, rel, muscle_dir_str)
            for k, v in feats.items():
                df.at[idx, k] = v
    else:
        gen = joblib.Parallel(n_jobs=args.jobs, return_as="generator", backend="loky")(
            joblib.delayed(_plv_row_result)(idx, rel, muscle_dir_str) for idx, rel in tasks
        )
        for idx, feats in tqdm(gen, total=len(tasks), desc="PLV по строкам", unit="row"):
            for k, v in feats.items():
                df.at[idx, k] = v

    df["plv_composite_soft_opt"] = vector_plv_composite_soft_opt(df, w_soft, alpha_soft)

    bundle: dict[str, Any] = joblib.load(args.histgb_model)
    clf = bundle["estimator"]
    Xb, _feat_names = build_feature_matrix(df)
    # строки без конечных признаков: predict_proba всё равно вызовем; sklearn обычно справится если нет NaN после impute
    p = clf.predict_proba(Xb)[:, 1].astype(float)
    df["histgb_full_predict_p_anti_phase"] = p

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print("Wrote", args.output.resolve())


if __name__ == "__main__":
    main()
