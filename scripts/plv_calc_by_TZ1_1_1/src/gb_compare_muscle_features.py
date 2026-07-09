r"""
Gradient boosting vs hand-crafted scores: same engineered features as extended PLV analysis.

- Features (8): 6 components like extended-opt + env envelope peak counts (used in gating).
- Target: anti_phase (1) vs not_anti_phase (0), and anti_phase vs rest (all rows).
- Evaluation: stratified 5-fold CV — for each fold, AUC on held-out test only.
  Hand-crafted scores are evaluated the same way (no refit; score is fixed per row).

Outputs:
  - JSON metrics next to figures
  - PNGs in Obsidian Vault .../Статья/assets_plv_reports/
  - Full-fit HistGB (anti_phase vs not_anti_phase) saved under data/ as histgb_anti_phase_vs_not_anti_phase.joblib
  - Appends a section to Report_PLV_vs_proxy.md and updates Report.md table if markers found.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

PROJECT_ROOT = Path(
    r"c:\Users\vvguba\OneDrive\Рабочий стол\лаборатория нейробиоморфных технологий\Статья_на_май"
)
MERGED_PATH = PROJECT_ROOT / "data" / "merged_only_data_muscles_auto_inactive.csv"
HISTGB_MODEL_PATH = PROJECT_ROOT / "data" / "histgb_anti_phase_vs_not_anti_phase.joblib"
OUT_DIR = Path(r"C:\Users\vvguba\OneDrive\Документы\Obsidian Vault\Статья")
ASSETS = OUT_DIR / "assets_plv_reports"
JSON_OUT = OUT_DIR / "gb_muscle_features_cv_auc.json"
REPORT_PROXY = OUT_DIR / "Report_PLV_vs_proxy.md"
REPORT_SUMMARY = OUT_DIR / "Report.md"

N_SPLITS = 5
RNG = 42


def clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def score_phase180_deg(phi_deg: np.ndarray) -> np.ndarray:
    d = np.abs(np.abs(phi_deg) - 180.0)
    return clip01(1.0 - d / 90.0)


def score_corr_antiphase(r: np.ndarray) -> np.ndarray:
    return clip01((1.0 - r) / 2.0)


def score_peakcount_reasonable(n1: np.ndarray, n2: np.ndarray) -> np.ndarray:
    n1 = np.maximum(n1, 0.0)
    n2 = np.maximum(n2, 0.0)
    ok = (n1 >= 3) & (n2 >= 3)
    sim = 1.0 - (np.abs(n1 - n2) / np.maximum(np.maximum(n1, n2), 1.0))
    sim = clip01(sim)
    return np.where(ok, sim, 0.0)


def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    plv = pd.to_numeric(df["plv_hilbert"], errors="coerce").values
    amp = pd.to_numeric(df["plv_comp_amp_score"], errors="coerce").values
    ph = pd.to_numeric(df["plv_comp_phase_score"], errors="coerce").values
    lag = pd.to_numeric(df["plv_phase_lag_deg"], errors="coerce").values
    corr = pd.to_numeric(df["plv_filtered_corr"], errors="coerce").values
    n1 = pd.to_numeric(df["plv_n_peaks_flex"], errors="coerce").fillna(0).values
    n2 = pd.to_numeric(df["plv_n_peaks_ext"], errors="coerce").fillna(0).values
    e1 = pd.to_numeric(df["plv_comp_env_peaks_flex"], errors="coerce").values
    e2 = pd.to_numeric(df["plv_comp_env_peaks_ext"], errors="coerce").values

    pp = score_phase180_deg(lag)
    ca = score_corr_antiphase(corr)
    pk = score_peakcount_reasonable(n1, n2)

    X = np.column_stack(
        [
            clip01(plv),
            clip01(amp),
            clip01(ph),
            pp,
            ca,
            pk,
            e1,
            e2,
        ]
    ).astype(np.float64)
    col_med = np.nanmedian(X, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0)
    inds = np.where(~np.isfinite(X))
    if inds[0].size:
        X[inds] = np.take(col_med, inds[1])
    names = [
        "plv_clip",
        "amp",
        "phase_hilbert",
        "phase_peaks180",
        "corr_anti",
        "peakcount_sim",
        "env_peaks_flex",
        "env_peaks_ext",
    ]
    return X, names


def cv_auc_sklearn(
    y: np.ndarray,
    score: np.ndarray,
    splitter,
    groups: np.ndarray | None = None,
) -> tuple[float, float, list[float]]:
    aucs: list[float] = []
    if groups is None:
        splits = splitter.split(np.zeros(len(y)), y)
    else:
        splits = splitter.split(np.zeros(len(y)), y, groups)
    for tr, te in splits:
        yt = y[te]
        st = score[te]
        m = np.isfinite(st)
        if m.sum() < 5 or len(np.unique(yt[m])) < 2:
            continue
        aucs.append(float(roc_auc_score(yt[m], st[m])))
    if not aucs:
        return float("nan"), float("nan"), []
    a = np.asarray(aucs, dtype=float)
    return float(a.mean()), float(a.std(ddof=1) if len(a) > 1 else 0.0), aucs


def _hist_gb() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_depth=5,
        max_iter=200,
        learning_rate=0.06,
        min_samples_leaf=20,
        l2_regularization=1e-3,
        random_state=RNG,
    )


def oof_histgb_predict_proba_positive(
    X: np.ndarray,
    y: np.ndarray,
    splitter,
    groups: np.ndarray | None = None,
) -> np.ndarray:
    """
    Out-of-fold P(y=1) from HistGradientBoosting (column index 1 of predict_proba).
    Rows not participating in any test fold remain NaN (should not happen with standard CV).
    """
    proba = np.full(len(y), np.nan, dtype=float)
    if groups is None:
        splits = splitter.split(X, y)
    else:
        splits = splitter.split(X, y, groups)
    for tr, te in splits:
        clf = _hist_gb()
        clf.fit(X[tr], y[tr])
        proba[te] = clf.predict_proba(X[te])[:, 1].astype(float)
    return proba


def cv_auc_hgb(
    X: np.ndarray,
    y: np.ndarray,
    splitter,
    groups: np.ndarray | None = None,
) -> tuple[float, float, list[float]]:
    aucs: list[float] = []
    if groups is None:
        splits = splitter.split(X, y)
    else:
        splits = splitter.split(X, y, groups)
    for tr, te in splits:
        clf = _hist_gb()
        clf.fit(X[tr], y[tr])
        proba = clf.predict_proba(X[te])[:, 1]
        yt = y[te]
        m = np.isfinite(proba)
        if m.sum() < 5 or len(np.unique(yt[m])) < 2:
            continue
        aucs.append(float(roc_auc_score(yt[m], proba[m])))
    a = np.asarray(aucs, dtype=float)
    return float(a.mean()), float(a.std(ddof=1) if len(a) > 1 else 0.0), aucs


def plot_bar_comparison(rows: list[dict], title: str, out_path: Path) -> None:
    names = [r["name"] for r in rows]
    means = [r["mean"] for r in rows]
    stds = [r["std"] for r in rows]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(x, means, yerr=stds, capsize=3, color="steelblue", edgecolor="black", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("AUC (mean ± SD, 5-fold CV, test fold)")
    ax.set_ylim(0.45, 1.0)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_fold_lines(rows: list[dict], title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for r in rows:
        aucs = r.get("folds") or []
        if not aucs:
            continue
        ax.plot(range(1, len(aucs) + 1), aucs, marker="o", label=r["name"])
    ax.set_xlabel("Fold")
    ax.set_ylabel("Test AUC")
    ax.set_xticks(range(1, 6))
    ax.set_ylim(0.45, 1.0)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(MERGED_PATH)

    required = [
        "mark",
        "plv_hilbert",
        "plv_comp_amp_score",
        "plv_comp_phase_score",
        "plv_phase_lag_deg",
        "plv_filtered_corr",
        "plv_n_peaks_flex",
        "plv_n_peaks_ext",
        "plv_comp_env_peaks_flex",
        "plv_comp_env_peaks_ext",
        "anti_proxy_S",
        "plv_composite_anti",
        "plv_composite_anti_opt",
        "plv_composite_anti_ext_opt",
        "plv_composite_soft_opt",
    ]
    for c in required:
        if c not in df.columns:
            raise RuntimeError(f"Missing column: {c}")

    # Группировка: предпочтительно по файлу мышц (один CSV ≈ одна пара рядов/запись эксперимента),
    # иначе по combination_id. Это сильнее ограничивает утечку, чем случайное разбиение строк.
    group_col: str | None = None
    if "muscle_csv_file" in df.columns:
        group_col = "muscle_csv_file"
    elif "combination_id" in df.columns:
        group_col = "combination_id"

    if group_col is not None:
        splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RNG)
        if group_col == "combination_id":
            groups_all = pd.to_numeric(df["combination_id"], errors="coerce").fillna(-1).astype(int).values
        else:
            groups_all = df[group_col].astype(str).values
    else:
        splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RNG)
        groups_all = None

    results: dict = {
        "n_splits": N_SPLITS,
        "group_col": group_col,
        "cv_type": (
            f"StratifiedGroupKFold({group_col})" if group_col is not None else "StratifiedKFold"
        ),
        "settings": {},
    }

    # --- Binary ---
    mask_bin = df["mark"].isin(["anti_phase", "not_anti_phase"]).values
    # Сохраняем исходные индексы строк (для записи OOF-вероятностей в merged CSV)
    dfb = df.loc[mask_bin].copy()
    yb = (dfb["mark"].astype(str) == "anti_phase").values.astype(int)
    Xb, feat_names = build_feature_matrix(dfb)
    if group_col is None:
        groups_bin = None
    elif group_col == "combination_id":
        groups_bin = pd.to_numeric(dfb["combination_id"], errors="coerce").fillna(-1).astype(int).values
    else:
        groups_bin = dfb[group_col].astype(str).values

    score_cols = [
        ("anti_proxy_S", "anti_proxy_S"),
        ("plv_hilbert", "plv_hilbert"),
        ("plv_composite_anti", "plv_composite_anti"),
        ("plv_composite_anti_opt", "plv_composite_anti_opt"),
        ("plv_composite_anti_ext_opt", "plv_composite_anti_ext_opt"),
        ("plv_composite_soft_opt", "plv_composite_soft_opt"),
    ]

    rows_bin: list[dict] = []
    gb_mean, gb_std, gb_folds = cv_auc_hgb(Xb, yb, splitter, groups_bin)
    rows_bin.append(
        {"name": "HistGB (8 признаков)", "mean": gb_mean, "std": gb_std, "folds": gb_folds}
    )

    oof_p = oof_histgb_predict_proba_positive(Xb, yb, splitter, groups_bin)
    col_prob = "histgb_oof_p_anti_phase"
    df[col_prob] = np.nan
    df.loc[dfb.index, col_prob] = oof_p

    for name, col in score_cols:
        s = pd.to_numeric(dfb[col], errors="coerce").values.astype(float)
        m, sd, folds = cv_auc_sklearn(yb, s, splitter, groups_bin)
        rows_bin.append({"name": name, "mean": m, "std": sd, "folds": folds})

    results["settings"]["anti_phase_vs_not_anti_phase"] = {
        "n": int(len(dfb)),
        "positives": int(yb.sum()),
        "features": feat_names,
        "rows": rows_bin,
        "histgb_probability_column": col_prob,
        "histgb_probability_note": (
            "Значения predict_proba[:,1] (оценка P(anti_phase)), посчитанные out-of-fold "
            "тем же StratifiedGroupKFold; вне бинарной подвыборки — NaN."
        ),
    }

    if len(np.unique(yb)) >= 2:
        HISTGB_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        clf_full = _hist_gb()
        clf_full.fit(Xb, yb)
        joblib.dump(
            {
                "estimator": clf_full,
                "feature_names": feat_names,
                "positive_label": "anti_phase",
                "negative_label": "not_anti_phase",
            },
            HISTGB_MODEL_PATH,
            compress=3,
        )
        results["settings"]["anti_phase_vs_not_anti_phase"]["histgb_full_fit_joblib"] = str(
            HISTGB_MODEL_PATH.resolve()
        )

    gtag = group_col or "shuffle"
    plot_bar_comparison(
        rows_bin,
        f"Сравнение AUC (5-fold Group CV по {gtag}, test): anti_phase vs not_anti_phase",
        ASSETS / "auc_cv_compare_binary_bar.png",
    )
    plot_fold_lines(
        rows_bin,
        f"AUC по фолдам (Group CV, {gtag}): anti_phase vs not_anti_phase",
        ASSETS / "auc_cv_compare_binary_folds.png",
    )

    # --- All marks: anti_phase vs rest ---
    y_all = (df["mark"].astype(str) == "anti_phase").values.astype(int)
    X_all, _ = build_feature_matrix(df)

    rows_all: list[dict] = []
    gbm, gbs, gbf = cv_auc_hgb(X_all, y_all, splitter, groups_all)
    rows_all.append({"name": "HistGB (8 признаков)", "mean": gbm, "std": gbs, "folds": gbf})

    dfa = df.reset_index(drop=True)
    for name, col in score_cols:
        s = pd.to_numeric(dfa[col], errors="coerce").values.astype(float)
        m, sd, folds = cv_auc_sklearn(y_all, s, splitter, groups_all)
        rows_all.append({"name": name, "mean": m, "std": sd, "folds": folds})

    results["settings"]["anti_phase_vs_rest"] = {
        "n": int(len(df)),
        "positives": int(y_all.sum()),
        "features": feat_names,
        "rows": rows_all,
    }

    plot_bar_comparison(
        rows_all,
        f"Сравнение AUC (5-fold Group CV по {gtag}, test): anti_phase vs все остальные метки",
        ASSETS / "auc_cv_compare_all_bar.png",
    )
    plot_fold_lines(
        rows_all,
        f"AUC по фолдам (Group CV, {gtag}): anti_phase vs остальные",
        ASSETS / "auc_cv_compare_all_folds.png",
    )

    JSON_OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    df.to_csv(MERGED_PATH, index=False, encoding="utf-8-sig")

    # --- Patch main "E0–E6" table: add/replace E7 HistGB (Group CV) ---
    if REPORT_PROXY.exists():
        rmain = REPORT_PROXY.read_text(encoding="utf-8")
        gcl = group_col or "—"
        e7a = (
            f"| E7 | `HistGB` (8 признаков, HistGB, Group CV по `{gcl}`) | "
            f"`anti_phase` vs остальные | **{gbm:.4f}** ± {gbs:.4f} |\n"
        )
        e7b = (
            f"| E7 | `HistGB` (8 признаков, HistGB, Group CV по `{gcl}`) | "
            f"`anti_phase` vs `not_anti_phase` | **{gb_mean:.4f}** ± {gb_std:.4f} |\n"
        )
        note_e7 = (
            "**Примечание по E7:** AUC для HistGB — **среднее ± SD по 5 тестовым фолдам** "
            f"(`StratifiedGroupKFold` по `{gcl}`, sklearn `roc_auc_score`). "
            "В отличие от строк **E0–E6**, где AUC посчитан на **полной выборке** ранговой формулой. "
            "Полное сравнение эвристик и HistGB **на одном CV-протоколе** — в разделе "
            "«Градиентный бустинг…» ниже.\n\n"
        )
        pat = re.compile(
            r"(\| E6 \| `plv_composite_soft_opt` \(soft-gate\) \| `anti_phase` vs `not_anti_phase` \| \*\*0\.7659\*\* \|)\n\n"
            r"(?:\| E7 \|[^\n]+\n\| E7 \|[^\n]+\n\n\*\*Примечание по E7:\*\*[\s\S]*?\n\n)?"
            r"(\*\*Замечание по `plv_hilbert`\*\*:)",
        )
        rmain2, n = pat.subn(r"\1\n\n" + e7a + e7b + "\n" + note_e7 + r"\g<2>", rmain, count=1)
        if n:
            REPORT_PROXY.write_text(rmain2, encoding="utf-8")

    # --- Markdown appendix ---
    def fmt_row(r: dict) -> str:
        return f"| {r['name']} | {r['mean']:.4f} | ± {r['std']:.4f} |"

    md_bin = "\n".join(fmt_row(r) for r in rows_bin)
    md_all = "\n".join(fmt_row(r) for r in rows_all)

    section = f"""### Градиентный бустинг (HistGradientBoosting) на признаках анализа

**Признаки (8 шт.)** — те же преобразования, что в расширенной весовой схеме (`optimize_plv_composite_weights_extended.py`): `clip01(plv_hilbert)`, `plv_comp_amp_score`, `plv_comp_phase_score`, близость фазы по пикам к \\(180^\\circ\\), антикорреляция отфильтрованных сил, сходство чисел пиков + **два счётчика** `plv_comp_env_peaks_flex/ext` (как в правиле gate).

**Оценка качества:** **{N_SPLITS}-кратная** кросс-валидация (`random_state={RNG}`) — **`StratifiedGroupKFold`** по колонке **`{group_col or "—"}`** (предпочтительно `muscle_csv_file`, иначе `combination_id`), чтобы **все строки, относящиеся к одному и тому же файлу/группе**, целиком оказывались либо в train, либо в test (строковый `StratifiedKFold` здесь сильно завышает AUC из-за повторяющихся конфигураций). На каждом фолде AUC считается **только на отложенной тестовой части** (`sklearn.metrics.roc_auc_score`). Для фиксированных скоров (`plv_composite_*`, `anti_proxy_S`, …) модель не переобучается — берутся готовые значения строки; для HistGB на train-группах обучается классификатор, на test-группах — `predict_proba`.

**Параметры HistGB:** `max_depth=5`, `max_iter=200`, `learning_rate=0.06`, `min_samples_leaf=20`, `l2_regularization=1e-3`.

#### Скор «вероятности» в merged CSV

В `merged_only_data_muscles_auto_inactive.csv` записана колонка **`histgb_oof_p_anti_phase`**: для строк с метками `anti_phase` и `not_anti_phase` это **out-of-fold** `predict_proba[:, 1]` того же HistGB и того же `StratifiedGroupKFold` (модель на каждой строке обучалась **без** её группы в train). Для остальных меток — `NaN`. Это **скор классификатора** в \\([0,1]\\); если нужна калибровка под частоты в данных, имеет смысл отдельно применить, например, Platt или isotonic на OOF-скоре.

Файл **`data/histgb_anti_phase_vs_not_anti_phase.joblib`** — один и тот же HistGB, дообученный на **всех** строках с метками `anti_phase` / `not_anti_phase` (для инференса вне OOF). Загрузка: `payload = joblib.load(...); proba = payload["estimator"].predict_proba(X)[:, 1]`; порядок признаков — `payload["feature_names"]` (совпадает с `build_feature_matrix`).

#### Таблица: mean ± SD AUC по фолдам (бинарная постановка)

| Метод | mean AUC | SD |
|---|---:|---:|
{md_bin}

#### Таблица: mean ± SD AUC по фолдам (anti_phase vs остальные)

| Метод | mean AUC | SD |
|---|---:|---:|
{md_all}

![Сравнение AUC (CV, бинарная)](assets_plv_reports/auc_cv_compare_binary_bar.png)

![AUC по фолдам (бинарная)](assets_plv_reports/auc_cv_compare_binary_folds.png)

![Сравнение AUC (CV, все метки)](assets_plv_reports/auc_cv_compare_all_bar.png)

![AUC по фолдам (все метки)](assets_plv_reports/auc_cv_compare_all_folds.png)

JSON с числами: `{JSON_OUT}`.
"""

    heading = "## Градиентный бустинг vs эвристики (AUC, Group CV)"
    if REPORT_PROXY.exists():
        cur = REPORT_PROXY.read_text(encoding="utf-8")
        sec_heading_re = re.compile(
            r"^## Градиентный бустинг vs эвристики[\s\S]*?(?=^## [^\n#]|\Z)",
            flags=re.MULTILINE,
        )
        new_block = heading + "\n\n" + section.strip() + "\n"
        if sec_heading_re.search(cur):
            # Callable repl: avoid re interpreting backslashes in `section` (LaTeX `\(` etc.)
            cur = sec_heading_re.sub(lambda _m: new_block, cur, count=1)
        else:
            cur = cur.rstrip() + "\n\n" + new_block
        REPORT_PROXY.write_text(cur, encoding="utf-8")

    # Update Report.md: inject GB row into markdown table if present
    if REPORT_SUMMARY.exists():
        ssum = REPORT_SUMMARY.read_text(encoding="utf-8")
        gb_line_bin = next((r for r in rows_bin if r["name"].startswith("HistGB")), None)
        gb_line_all = next((r for r in rows_all if r["name"].startswith("HistGB")), None)
        if gb_line_bin and gb_line_all:
            gcl2 = group_col or "?"
            new_row = (
                f"| `HistGB` (8 признаков, Group CV по `{gcl2}`) | "
                f"**{gb_line_all['mean']:.4f}** ± {gb_line_all['std']:.4f} | "
                f"**{gb_line_bin['mean']:.4f}** ± {gb_line_bin['std']:.4f} |\n"
            )
            row_pat = re.compile(r"^\| `HistGB`[^\n]*\n", flags=re.MULTILINE)
            if row_pat.search(ssum):
                ssum = row_pat.sub(new_row, ssum, count=1)
            else:
                m = re.search(r"^\| `plv_composite_soft_opt`[^\n]*\n", ssum, flags=re.MULTILINE)
                if m:
                    ssum = ssum[: m.end()] + new_row + ssum[m.end() :]
            gcl3 = group_col or "?"
            note = (
                f"\n*Примечание:* `HistGB` — **Group CV** по `{gcl3}` (без утечки одного файла между фолдами); "
                "остальные строки таблицы — **полная выборка**, rank AUC как ранее. "
                "Сопоставление всех методов на одном протоколе — в `Report_PLV_vs_proxy.md`, раздел про бустинг.\n"
            )
            if "без утечки одного файла между фолдами" not in ssum:
                ssum = ssum.rstrip() + note + "\n"
        REPORT_SUMMARY.write_text(ssum, encoding="utf-8")

    print("Wrote", JSON_OUT)
    print("Figures in", ASSETS)


if __name__ == "__main__":
    main()
