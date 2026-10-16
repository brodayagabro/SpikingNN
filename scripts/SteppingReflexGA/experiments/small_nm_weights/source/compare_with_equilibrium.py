"""Compare completed balance sweeps on identical saved trial banks."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scripts.SteppingReflexGA.layout import image_path
from scripts.SteppingReflexGA.experiments.core.run import write_csv, write_json
from scripts.SteppingReflexGA.experiments.core.sweep_analysis import grid_values


def read_rows(root):
    with (root / "summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in ("population", "generations", "seed"):
            row[key] = int(row[key])
        for key in ("mutation_probability", "validation_fitness", "test_fitness",
                    "test_success", "test_late_rmse_deg"):
            row[key] = float(row[key])
    return rows


def key(row):
    return tuple(row[k] for k in ("population", "generations", "mutation_probability", "seed"))


def compare(reference, root):
    settings = json.loads((root / "sweep.json").read_text())
    original = json.loads((reference / "sweep.json").read_text())
    assert settings["experiment"] == "small_nm_weights"
    assert original["experiment"] == "equilibrium"
    for field in ("populations", "epochs", "probabilities", "seeds", "dt_ms", "model",
                  "duration_s", "sigma_schedule", "probability_policy", "epoch_policy"):
        assert settings[field] == original[field], field
    numerical_sources = ("config.py", "genome.py", "afferents.py", "dynamics.py", "fitness.py",
                         "population.py", "evolution.py", "cases.py", "commands.py", "run.py", "Izh_net.py")
    for filename in numerical_sources:
        left = [v for k, v in original["source_sha256"].items() if Path(k).name == filename]
        right = [v for k, v in settings["source_sha256"].items() if Path(k).name == filename]
        assert len(left) == len(right) == 1 and left == right, filename
    before = {key(row): row for row in read_rows(reference)}
    after = {key(row): row for row in read_rows(root)}
    expected = np.prod([len(settings[name]) for name in ("populations", "epochs", "probabilities", "seeds")])
    assert len(before) == len(after) == expected and before.keys() == after.keys()
    vectors = []
    for directory in (reference, root):
        with np.load(directory / "genome_comparison.npz", allow_pickle=False) as data:
            vectors.append(dict(zip(data["labels"], data["normalized"])))
    rows = []
    for combination, second in after.items():
        first = before[combination]
        p1, p2 = reference / first["directory"], root / second["directory"]
        for bank in ("training", "validation", "test"):
            assert (p1 / f"{bank}_cases.csv").read_bytes() == (p2 / f"{bank}_cases.csv").read_bytes()
        assert json.loads((p1 / "run.json").read_text())["hyperparameters"] == json.loads((p2 / "run.json").read_text())["hyperparameters"]
        row = {name: second[name] for name in ("population", "generations", "mutation_probability", "seed")}
        for metric in ("validation_fitness", "test_fitness", "test_success", "test_late_rmse_deg"):
            row[f"{metric}_equilibrium"] = first[metric]
            row[f"{metric}_small_weights"] = second[metric]
            row[f"delta_{metric}"] = second[metric] - first[metric]
        row["paired_genome_rmse"] = float(np.sqrt(np.mean(
            (vectors[1][second["label"]] - vectors[0][first["label"]]) ** 2)))
        rows.append(row)
    write_csv(root / "comparison_with_equilibrium.csv", rows)
    figures = []
    for metric, title in (("validation_fitness", "Validation fitness: small weights minus equilibrium"),
                          ("test_success", "Test success fraction: small weights minus equilibrium"),
                          ("test_late_rmse_deg", "Late test angle RMSE difference (degrees)")):
        field = f"delta_{metric}"
        grids = [grid_values(rows, settings["populations"], settings["epochs"], p, field)
                 for p in settings["probabilities"]]
        bound = max(1e-12, max(float(np.max(np.abs(g))) for g in grids))
        fig, axes = plt.subplots(1, len(grids), figsize=(14.7, 4.5), squeeze=False, layout="constrained")
        for ax, p, grid in zip(axes[0], settings["probabilities"], grids):
            shown = ax.imshow(grid, cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
            ax.set(xticks=range(len(settings["epochs"])), xticklabels=settings["epochs"],
                   yticks=range(len(settings["populations"])), yticklabels=settings["populations"],
                   xlabel="Epochs completed", ylabel="Population size", title=f"Mutation p = {p:g}")
            for i, j in np.ndindex(grid.shape):
                value = grid[i, j]
                label = f"{value:+.1%}" if metric == "test_success" else f"{value:+.4f}"
                ax.text(j, i, label, ha="center", va="center", fontsize=8,
                        color="white" if abs(value) > .65 * bound else "black")
        fig.colorbar(shown, ax=axes[0].tolist(), shrink=.8, label="Experiment 2 minus experiment 1")
        fig.suptitle(title)
        destination = image_path(root / f"comparison_{field}.png")
        fig.savefig(destination, dpi=180, bbox_inches="tight")
        fig.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        figures.append((title, os.path.relpath(destination, root)))
    winners = [max(records.values(), key=lambda r: r["validation_fitness"]) for records in (before, after)]
    report = ["# Сопоставление опытов равновесия\n",
              "Проверены совпадение численного ядра, механики, сетки параметров и всех сохранённых банков испытаний. "
              "Разности в таблицах и на картах: **опыт 2 минус опыт 1**. Для fitness и успешности положительная разность "
              "означает большее значение; для RMSE угла положительная разность означает большую ошибку.\n",
              "Ограничение области двух весов меняет начальную популяцию в физических координатах. "
              "Исходные веса контроллера в центре популяции — 0.15 в опыте 1 и 0.08 в опыте 2. "
              "Это сравнение двух протоколов поиска при одном seed, а не изолированная оценка причинного эффекта ограничения.\n",
              "## Лучшие контроллеры по validation\n",
              "| Опыт | N | p | Эпоха | Validation fitness | Test fitness | Test успех | Поздняя RMSE, ° |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, winner in zip(("Без ограничения малых весов", "0.02 ≤ w < 0.1"), winners):
        report.append(f"| {name} | {winner['population']} | {winner['mutation_probability']:g} | {winner['generations']} | "
                      f"{winner['validation_fitness']:.6f} | {winner['test_fitness']:.6f} | {winner['test_success']:.1%} | "
                      f"{winner['test_late_rmse_deg']:.5f} |")
    report += ["", "[Все 60 пар сравнений, включая RMSE геномов](comparison_with_equilibrium.csv).\n"]
    report += [f"![{title}]({path})\n" for title, path in figures]
    (root / "COMPARISON.md").write_text("\n".join(report))
    write_json(root / "comparison_verification.json", dict(
        status="passed", pairs=len(rows), numerical_sources_equal=True, banks_equal=True,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reference=str(reference), winners=winners))
    print(f"Compared {len(rows)} pairs; numerical sources and case banks match.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compare(args.reference.resolve(), args.output.resolve())
