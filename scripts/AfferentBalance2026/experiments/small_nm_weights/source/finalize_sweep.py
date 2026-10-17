"""Verify the completed second sweep and assemble its numerical findings."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
import numpy as np
from scripts.SteppingReflexGA.experiments.core.sweep_verification import audit
from scripts.SteppingReflexGA.experiments.small_nm_weights.source.compare_with_equilibrium import compare, read_rows


def finalize(root, reference):
    verification = audit(root)
    subprocess.run([sys.executable, "-m", "scripts.SteppingReflexGA.experiments.core.sweep_detail",
                    "--output", str(root)], check=True)
    compare(reference, root)
    analysis = json.loads((root / "analysis.json").read_text())
    winner = analysis["best_by_validation"]
    genome = json.loads((root / winner["directory"] / "best_genome.json").read_text())
    w_flex = genome["parameters"]["muscles"]["flexor"]["w"]
    w_ext = genome["parameters"]["muscles"]["extensor"]["w"]
    settings = json.loads((root / "sweep.json").read_text())
    rows = read_rows(root)
    final_indices = [i for i, row in enumerate(rows) if row["generations"] == max(settings["epochs"])]
    with np.load(root / "genome_comparison.npz", allow_pickle=False) as data:
        assert list(data["labels"]) == [r["label"] for r in rows]
        distances = data["rmse"][np.ix_(final_indices, final_indices)]
        values = distances[np.triu_indices(len(final_indices), 1)]
    final = [rows[i] for i in final_indices]
    successful = sum(row["test_success"] == 1. for row in final)
    baseline = json.loads((root / "initial_controller_baseline.json").read_text())["controllers"]["small_nm_weights"]
    tests = ET.parse(root / "tests.xml").getroot().find("testsuite")
    assert tests is not None and tests.attrib["failures"] == tests.attrib["errors"] == "0"
    progress = json.loads((root / "progress.json").read_text())
    report = ["# Итоги серии малых нейромышечных весов\n",
              f"Завершены {len(rows)} контрольных результатов, {len(final)} траекторий; "
              f"время серии — {progress['elapsed_s']:.0f} с ({progress['elapsed_s']/60:.1f} мин).\n",
              "Полные Afferents, шаг 0.1 мс, эпизод 10 с, Iapp=[0,0]. "
              "Управление только спайковой сетью через мышцы (`mode=0`); ПД-режим не используется.\n",
              f"Лучший по validation: **N={winner['population']}, p={winner['mutation_probability']:g}, "
              f"эпоха {winner['generations']}**, validation fitness={winner['validation_fitness']:.9f}, "
              f"test fitness={winner['test_fitness']:.9f}, успех {winner['test_success']:.1%}. "
              f"Средняя поздняя угловая RMSE на test — **{winner['test_late_rmse_deg']:.6f}°**. "
              f"Нейромышечные веса: сгибатель **{w_flex:.8f}**, разгибатель **{w_ext:.8f}**.\n",
              f"[Выбранный геном с пояснениями]({winner['directory']}/best_genome.json).\n",
              f"После 200 эпох {successful}/{len(final)} контроллеров проходят все 48 тестов. "
              f"Попарный RMSE их геномов: **{values.min():.4f}–{values.max():.4f}**, "
              f"медиана **{np.median(values):.4f}** по 50 каноническим нормированным координатам.\n",
              "## Результаты после 200 эпох\n",
              "| N | p | Validation fitness | Test fitness | Успех test | Поздняя RMSE, ° |",
              "|---:|---:|---:|---:|---:|---:|"]
    for row in final:
        report.append(f"| {row['population']} | {row['mutation_probability']:g} | {row['validation_fitness']:.6f} | "
                      f"{row['test_fitness']:.6f} | {row['test_success']:.1%} | {row['test_late_rmse_deg']:.6f} |")
    report += ["", "## Исходное состояние и интерпретация\n",
               f"Заданный исходный контроллер с w=0.08 до обучения имеет validation fitness="
               f"{baseline['validation']['fitness']:.6f}, test fitness={baseline['test']['fitness']:.6f}, "
               f"успех {round(baseline['test']['success']*48)}/48 и позднюю RMSE {baseline['test']['late_rmse_deg']:.6f}°. "
               "Это одна исходная особь, а не победитель случайной начальной популяции. "
               "Качество первого поколения и его дальнейшее улучшение отражены в history.csv каждого запуска.\n",
               "[Оценка исходной особи](initial_controller_baseline.json), "
               "[исходные метрики по испытаниям](initial_controller_baseline.npz).\n",
               "Один seed=42; эпохи внутри линии — зависимые контрольные точки. "
               "Выбор победителей выполняется только по validation. Высокое качество на заданном банке "
               "не устанавливает устойчивость при иных условиях или длительности.\n",
               "## Проверки\n",
               f"Пройдены **{tests.attrib['tests']} тестов** и аудит всех {verification['completed']} популяций. "
               f"Во всех сохранённых популяциях веса лежат от {verification['neuromuscular_weight_min']:.10f} "
               f"до {verification['neuromuscular_weight_max']:.10f}; строгое w < 0.1 подтверждено. "
               "Проверены нулевые токи, механические гиперпараметры, отбор по validation, продолжение эпох, "
               "fitness, RMSE и снимок исходников.\n",
               "[Аудит](verification.json), [тесты](tests.xml), "
               "[сравнение с первым опытом](COMPARISON.md), [основные карты и данные](REPORT.md).\n"]
    (root / "FINDINGS.md").write_text("\n".join(report))
    with (root / "REPORT.md").open("a") as stream:
        stream.write("\n## Численные итоги и проверки\n\n"
                     "[Итоги, таблица конечных результатов и проверки](FINDINGS.md). "
                     "[Сопоставление с первым опытом и карты разностей](COMPARISON.md).\n")
    print(json.dumps(dict(best=winner, weights=[w_flex, w_ext], final_successful=successful,
                         final_genome_rmse_min=float(values.min()), final_genome_rmse_max=float(values.max()),
                         final_genome_rmse_median=float(np.median(values))), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    while not json.loads((args.output / "progress.json").read_text()).get("finished"):
        if not args.wait:
            parser.error("The training sweep has not finished")
        time.sleep(5)
    finalize(args.output.resolve(), args.reference.resolve())
