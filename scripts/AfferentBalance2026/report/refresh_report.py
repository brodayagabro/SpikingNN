"""Rebuild the current report from saved results, without importing the simulator.

The template contains scientific explanations; every result table and copied
figure is backed by a saved artifact. Run before build_pdf.py after template edits.
"""
import ast
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile
import xml.etree.ElementTree as ET

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
EXPERIMENTS = HERE.parent / "experiments"
SERIES = "sweep_population_mutation_20261008"
WIDE = "wide_replay_20261009"
POLYNOMIAL_STUDY = "polynomial_five_targets_20261015"
EXPERIMENT_NAMES = ("equilibrium", "small_nm_weights")
FIGURES = (
    ("equilibrium", SERIES, "convergence", "01_equilibrium_convergence"),
    ("equilibrium", SERIES, "validation_fitness_detail", "02_equilibrium_validation"),
    ("equilibrium", SERIES, "test_late_rmse_deg", "03_equilibrium_error"),
    ("small_nm_weights", SERIES, "convergence", "04_small_convergence"),
    ("small_nm_weights", SERIES, "validation_fitness_detail", "05_small_validation"),
    ("small_nm_weights", SERIES, "test_late_rmse_deg", "06_small_error"),
    ("small_nm_weights", SERIES, "comparison_delta_validation_fitness", "07_comparison"),
    ("equilibrium", SERIES, "genome_rmse_final", "08_equilibrium_genomes"),
    ("small_nm_weights", SERIES, "genome_rmse_final", "09_small_genomes"),
    ("equilibrium", WIDE, "angle_responses", "10_equilibrium_wide"),
    ("small_nm_weights", WIDE, "angle_responses", "11_small_wide"),
    ("equilibrium", WIDE, "settling_s", "12_equilibrium_settling"),
    ("small_nm_weights", WIDE, "settling_s", "13_small_settling"),
    ("equilibrium", WIDE, "case_094_nonlinear", "14_equilibrium_nonlinear"),
    ("small_nm_weights", WIDE, "case_094_nonlinear_0_1s", "15_small_nonlinear"),
    ("angle_setpoint", POLYNOMIAL_STUDY, "convergence", "16_setpoint_convergence"),
    ("angle_setpoint", POLYNOMIAL_STUDY, "current_laws", "17_setpoint_current_laws"),
    ("angle_setpoint", POLYNOMIAL_STUDY, "angle_responses", "18_setpoint_trajectories"),
    ("angle_setpoint", POLYNOMIAL_STUDY, "phase_portrait", "19_setpoint_phase"),
    ("angle_setpoint", POLYNOMIAL_STUDY, "test_success", "20_setpoint_success"),
)


def table(headers, rows):
    """Format a report table as Markdown."""
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "|" + "---|" * len(headers),
                      *("| " + " | ".join(map(str, row)) + " |" for row in rows)])


def main():
    # Current scientific report follows the factor-1 recomputation.
    """Rebuild the active unscaled report, or a restored historical report when explicitly available."""
    # The current article is derived from the corrected-mechanics retraining.
    if any((HERE / 'results' / study / 'manifest.json').exists()
           for study in ('muscle_tau50_20261026', 'tensile_retrain_20261026')):
        import runpy
        runpy.run_path(str(HERE / 'build_retraining_article.py'), run_name='__main__')
        return
    pointer = HERE / 'results/current_unscaled.json'
    if pointer.exists():
        import runpy
        import sys
        sys.path.insert(0, str(REPO))
        from scripts.AfferentBalance2026.tools.runtime_support import current_results
        module = runpy.run_path(str(HERE.parent / 'tools/analyze_unscaled.py'))
        module['analyze'](current_results(pointer))
        return
    inputs = {}

    def track(path):
        """Register an input artifact SHA256 before including it in the report."""
        inputs[str(path.relative_to(REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    def read(path):
        """Read JSON while recording the input hash."""
        return json.loads(track(path).read_text())

    def rows(path):
        """Read CSV rows while recording the input hash."""
        return list(csv.DictReader(track(path).open()))

    best, genomes, wide, final_rows, distances, audits, drift = {}, {}, {}, {}, {}, {}, {}
    for exp in EXPERIMENT_NAMES:
        root = EXPERIMENTS / exp / "report/results" / SERIES
        analysis = read(root / "analysis.json")
        assert analysis["completed"] == analysis["expected"] == 60
        best[exp] = analysis["best_by_validation"]
        b = best[exp]
        assert (b["model"], b["dt_ms"], b["seed"]) == ("SpikingNN.Afferents/v2", .1, 42)
        genomes[exp] = read(root / b["directory"] / "best_genome.json")
        assert genomes[exp]["parameters"]["bias_nA"] == [0., 0.]
        all_rows = rows(root / "summary.csv")
        assert len(all_rows) == 60
        final_rows[exp] = [r for r in all_rows if int(r["generations"]) == 200]
        assert len(final_rows[exp]) == 12
        assert all(float(r["test_success"]) == 1. for r in final_rows[exp])
        assert (b["population"], b["generations"], b["mutation_probability"]) == (50,200,.3)
        with np.load(track(root / "genome_comparison.npz"), allow_pickle=False) as data:
            selected = np.array(["E=200," in s for s in data["labels"]])
            assert selected.sum() == 12
            matrix = data["rmse"][np.ix_(selected, selected)]
            d = matrix[np.triu_indices(12, 1)]
            distances[exp] = [float(d.min()), float(np.median(d)), float(d.max())]
        audit = read(root / "verification.json")
        assert audit["status"] == "passed"
        wr = EXPERIMENTS / exp / "report/results" / WIDE
        wide[exp] = read(wr / "summary.json")
        run = read(wr / "run.json")
        assert (run["model"], run["dt_ms"], run["mode"], run["cases"]) == ("SpikingNN.Afferents/v2", .1, 0, 171)
        assert run["hyperparameters"]["muscle_torque_factor"] == 10.
        assert run["pd_used"] is False
        metrics = rows(wr / "metrics.csv")
        assert len(metrics) == wide[exp]["cases"] == 171
        assert sum(float(r["success"]) for r in metrics) == wide[exp]["successes"]
        for filename in ("verification.json", "saved_files_verification.json"):
            assert read(wr / filename)["status"] == "passed"
        suites = ET.parse(track(wr / "tests.xml")).getroot().iter("testsuite")
        audits[exp] = [{k: int(s.get(k, 0)) for k in ("tests", "failures", "errors")} for s in suites]
        assert sum(s["tests"] for s in audits[exp]) == 72
        assert all(s["failures"] == s["errors"] == 0 for s in audits[exp])
        # Verify the actual archived code, not only the model label in metadata.
        with zipfile.ZipFile(track(wr / "source_snapshot.zip")) as archive:
            for suffix in ("experiments/core/afferents.py", "src/SpikingNN/core/Izh_net.py"):
                member = next(n for n in archive.namelist() if n.endswith(suffix))
                key = next(k for k in run["source_sha256"] if k.endswith(suffix))
                content = archive.read(member)
                assert hashlib.sha256(content).hexdigest() == run["source_sha256"][key]
                if suffix.endswith("afferents.py"):
                    assert b"from SpikingNN.core.Izh_net import Afferents" in content
                    assert b"v**p_v" in content and b"k_nI * activation" in content
                else:
                    tree = ast.parse(content)
                    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Afferents")
                    receptor = {n.targets[0].id: ast.literal_eval(n.value) for n in cls.body
                                if isinstance(n, ast.Assign)}
                    current = REPO / "src/SpikingNN/core/Izh_net.py"
                    drift[exp] = hashlib.sha256(current.read_bytes()).hexdigest() != hashlib.sha256(content).hexdigest()

    baseline = read(EXPERIMENTS / "small_nm_weights/report/results" / SERIES / "initial_controller_baseline.json")
    smoke = read(EXPERIMENTS / "angle_setpoint/report/results/smoke_full_afferents/result.json")
    assert smoke["model"] == "SpikingNN.Afferents/v2" and smoke["dt_ms"] == .1
    assert smoke["test_success"] == 3/76
    polynomial_root = EXPERIMENTS / "angle_setpoint/report/results" / POLYNOMIAL_STUDY
    polynomial = read(polynomial_root / "summary.json")
    polynomial_audit = read(polynomial_root / "verification.json")
    assert polynomial_audit["status"] == "passed" and polynomial_audit["full_traces_verified"] == 36
    assert polynomial["protocol"]["dt_ms"] == .1 and polynomial["protocol"]["pd_used"] is False
    assert len(polynomial["runs"]) == 4
    for path, expected in polynomial_audit["input_sha256"].items():
        actual = track(REPO / path)
        assert hashlib.sha256(actual.read_bytes()).hexdigest() == expected, path
    track(Path(__file__).resolve())
    values = {}
    values["POLYNOMIAL_SECTION"] = track(polynomial_root / "report_section.md").read_text()
    values["POLYNOMIAL_CONCLUSION"] = track(polynomial_root / "conclusion.txt").read_text().strip()
    values["POLYNOMIAL_ABSTRACT"] = values["POLYNOMIAL_CONCLUSION"]
    values["BEST_TABLE"] = table(["Опыт", "Validation fitness", "Test fitness", "Успех test", "Поздняя RMSE, °"],
        [[i+1, f'{best[e]["validation_fitness"]:.9f}', f'{best[e]["test_fitness"]:.9f}',
          "48/48", f'{best[e]["test_late_rmse_deg"]:.9f}'] for i,e in enumerate(EXPERIMENT_NAMES)])
    values["WIDE_TABLE"] = table(["Опыт", "Успех", "Fitness", "Макс. время, с", "Средняя поздняя RMSE, °", "Макс. поздняя RMSE, °"],
        [[i+1, f'{wide[e]["successes"]}/{wide[e]["cases"]}', f'{wide[e]["fitness"]:.9f}',
          f'{wide[e]["max_settling_s"]:.4f}', f'{wide[e]["mean_late_rmse_deg"]:.9f}',
          f'{wide[e]["max_late_rmse_deg"]:.9f}'] for i,e in enumerate(EXPERIMENT_NAMES)])
    values["BASELINE_TABLE"] = table(["Опыт", "Исходные w", "Test fitness", "Успех test", "Поздняя RMSE, °"],
        [[i+1, ", ".join(f'{w:.2f}' for w in baseline["controllers"][e]["weights"]),
          f'{baseline["controllers"][e]["test"]["fitness"]:.9f}',
          f'{round(baseline["controllers"][e]["test"]["success"]*48)}/48',
          f'{baseline["controllers"][e]["test"]["late_rmse_deg"]:.6f}'] for i,e in enumerate(EXPERIMENT_NAMES)])
    for exp, label in zip(EXPERIMENT_NAMES, ("EQ", "SMALL")):
        values[label+"_FINAL_TABLE"] = table(["Популяция", "p", "Validation fitness", "Test fitness", "Поздняя RMSE, °"],
            [[r["population"], r["mutation_probability"], f'{float(r["validation_fitness"]):.9f}',
              f'{float(r["test_fitness"]):.9f}', f'{float(r["test_late_rmse_deg"]):.6f}'] for r in final_rows[exp]])
    values["DISTANCE_TABLE"] = table(["Опыт", "Минимум", "Медиана", "Максимум"],
        [[i+1, *(f'{v:.6f}' for v in distances[e])] for i,e in enumerate(EXPERIMENT_NAMES)])
    values["MUSCLE_TABLE"] = table(["Опыт / мышца", "w", "N мышцы", "A, Н/мс", "τ силы, мс", "ANτ, Н"],
        [[f'{i+1} / {"сгибатель" if side=="flexor" else "разгибатель"}',
          *(f'{p[k]:.6f}' for k in ("w","N","A","tau_1")), f'{p["A"]*p["N"]*p["tau_1"]:.6f}']
         for i,e in enumerate(EXPERIMENT_NAMES) for side,p in genomes[e]["parameters"]["muscles"].items()])
    values["SMOKE_TABLE"] = table(["Популяция", "Эпохи", "Validation fitness", "Test fitness", "Успех test"],
        [[3, 2, f'{smoke["validation_fitness"]:.9f}', f'{smoke["test_fitness"]:.9f}', "3/76"]])
    values["RECEPTOR_TABLE"] = table(["Параметр класса Afferents", "Значение"],
        [[f'`{k}`', receptor[k]] for k in ("p_v","k_v","k_dI","k_dII","k_nI","k_nII","k_f","const_I","const_II")])

    images = HERE / "figures_full_afferents"
    images.mkdir(exist_ok=True)
    for exp, series, stem, output in FIGURES:
        for ext in ("png", "pdf"):
            src = EXPERIMENTS / exp / "img" / series / f"{stem}.{ext}"
            if src.exists():
                shutil.copy2(track(src), images / f"{output}.{ext}")
    template = track(HERE / "REPORT.template.md").read_text()
    for key, value in values.items():
        assert "{{"+key+"}}" in template, key
        template = template.replace("{{"+key+"}}", value)
    assert not re.search(r"\{\{[A-Z_]+\}\}", template)
    (HERE / "REPORT.md").write_text(template)

    genes = genomes["equilibrium"]["genes"]
    other = genomes["small_nm_weights"]["genes"]
    appendix = "# Приложение. Параметры геномов полных Afferents {#appendix-parameters}\n\n"
    appendix += "Победители по validation: N популяции = 50, p = 0.3, эпоха 200, seed = 42. "
    appendix += "52 физических поля, 50 свободных генов; токи обоих нейронов фиксированы в нуле. "
    appendix += "Границы в таблице канонические; в опыте 2 дополнительно 0.02 ≤ w < 0.1. "
    appendix += "Все числа извлечены из сохранённых best_genome.json.\n\n"
    appendix += table(["Индекс", "Параметр", "Границы", "Шкала", "Опыт 1", "Опыт 2"],
        [[g["index"], f'`{g["name"]}`', f'{g["low"]:.6g} … {g["high"]:.6g}',
          "log" if g["log"] else "linear", f'{g["value"]:.9g}', f'{h["value"]:.9g}']
         for g,h in zip(genes, other)])
    appendix += "\n\n## Смысл параметров и соответствие системе\n\n"
    appendix += table(["Индекс / поле", "Смысл", "Единицы"],
        [[f'{g["index"]}: `{g["name"]}`', g["description"], g["unit"]] for g in genes])
    appendix += "\n\n## Фиксированные гиперпараметры\n\n"
    appendix += table(["Параметр", "Значение"], list(genomes["equilibrium"]["hyperparameters"]["mechanics"].items()) +
                      [["dt_ms", .1], ["afferent_model", "SpikingNN.Afferents/v2"]]) + "\n"
    (HERE / "APPENDIX_PARAMETERS.md").write_text(appendix)
    manifest = {"status": "passed", "model": "SpikingNN.Afferents/v2", "dt_ms": .1,
                "training_trajectories": 24, "checkpoints": 120, "wide_cases": 342,
                "figure_count": len(FIGURES), "saved_tests": audits,
                "angle_setpoint": {"study": POLYNOMIAL_STUDY, "runs": 4,
                    "targets": polynomial["protocol"]["targets_deg"], "full_traces": 36,
                    "selected_by_validation": polynomial["selected_overall"],
                    "saved_tests": polynomial_audit["test_suites"]},
                "working_library_differs_from_saved_runs": drift,
                "input_sha256": inputs,
                "output_sha256": {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                                  for name in ("REPORT.md", "APPENDIX_PARAMETERS.md")}}
    out = HERE / "data_full_afferents"
    out.mkdir(exist_ok=True)
    (out / "report_verification.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({k:v for k,v in manifest.items() if not k.endswith("sha256")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
