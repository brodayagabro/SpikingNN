"""Heatmaps and coordinatewise RMSE comparisons of saved sweep genomes."""
import csv
import json
import os
from pathlib import Path

import numpy as np
from ...layout import image_path
from . import genome
from .run import write_csv, write_json


def pairwise_rmse(vectors):
    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim != 2 or not len(vectors) or not vectors.shape[1] or not np.isfinite(vectors).all():
        raise ValueError("Expected a nonempty finite matrix of genome coordinates")
    return np.sqrt(np.mean((vectors[:, None, :] - vectors[None, :, :])**2, axis=2))


def grid_values(records, populations, epochs, probability, field):
    grid = np.full((len(populations), len(epochs)), np.nan)
    for i, size in enumerate(populations):
        for j, budget in enumerate(epochs):
            values = [r[field] for r in records if r["population"] == size and
                      r["generations"] == budget and r["mutation_probability"] == probability]
            if values:
                grid[i, j] = np.mean(values)
    return grid


def analyze(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(root).resolve()
    settings = json.loads((root / "sweep.json").read_text())
    records, vectors, physical = [], [], []
    files = sorted(root.glob("mutation_*/population_*/seed_*/epochs_*/result.json"))
    if not files:
        raise ValueError("No completed training results found")
    for path in files:
        result = json.loads(path.read_text())
        directory = path.parent
        saved_settings = json.loads((directory / "run.json").read_text())
        if saved_settings["source_sha256"] != settings["source_sha256"]:
            raise ValueError(f"Mixed calculation sources: {directory}")
        p = genome.decode(genome.load_genome(directory / "best_genome.json"))
        with np.load(directory / "final_population.npz", allow_pickle=False) as final:
            winner = int(final["selected_index"])
            if winner != int(np.argmax(final["validation_scores"])):
                raise ValueError("Saved winner does not match validation ranking")
            np.testing.assert_allclose(p, final["physical_controllers"][winner], atol=1e-13, rtol=1e-13)
            population = final["population"].copy()
            if result["experiment"] == "angle_setpoint":
                vector = population[winner]
            else:
                # Common canonical gene scales, not mixed physical units.
                vector = genome.encode(p)[genome.FREE_INDICES]
            result["within_population_rmse"] = float(pairwise_rmse(population)[np.triu_indices(len(population), 1)].mean())
        label = f"p={result['mutation_probability']:g}, N={result['population']}, E={result['generations']}, s={result['seed']}"
        records.append(dict(result, directory=str(directory.relative_to(root)), label=label))
        vectors.append(vector)
        physical.append(p)
    vectors, physical = np.array(vectors), np.array(physical)
    distances = pairwise_rmse(vectors)
    best = int(np.argmax([r["validation_fitness"] for r in records]))
    for i, record in enumerate(records):
        record["genome_rmse_to_best"] = float(distances[i, best])
    labels = [r["label"] for r in records]
    write_csv(root / "summary.csv", records)
    write_csv(root / "genome_rmse.csv", [dict(genome=label, **dict(zip(labels, map(float, distances[i]))))
                                         for i, label in enumerate(labels)])
    names = ([f"command.neuron_{i}.power_{j}" for i in range(2) for j in range(1, 4)]
             if settings["experiment"] == "angle_setpoint" else [genome.GENES[i].name for i in genome.FREE_INDICES])
    np.savez_compressed(root / "genome_comparison.npz", labels=np.array(labels),
                        normalized=vectors, physical_controllers=physical, rmse=distances,
                        differences_to_best=vectors-vectors[best], gene_names=np.array(names),
                        best_index=best)
    write_csv(root / "genome_differences_to_best.csv", [dict(gene=name,
              **{label: float(vectors[i, j]-vectors[best, j]) for i, label in enumerate(labels)})
              for j, name in enumerate(names)])

    figures = []

    def save(fig, stem):
        destination = image_path(root / f"{stem}.png")
        fig.savefig(destination, dpi=180, bbox_inches="tight")
        fig.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        figures.append((stem, os.path.relpath(destination, root)))

    populations, epochs, probabilities = settings["populations"], settings["epochs"], settings["probabilities"]
    metrics = [
        ("training_fitness", "Training fitness", "viridis", 0., 1.),
        ("validation_fitness", "Validation fitness (used for selection)", "viridis", 0., 1.),
        ("test_fitness", "Held-out test fitness", "viridis", 0., 1.),
        ("test_success", "Held-out test success fraction", "viridis", 0., 1.),
        ("test_late_rmse_deg", "Late angle RMSE on test (degrees; lower is better)", "magma_r", 0., None),
        ("genome_rmse_to_best", "Genome RMSE to best validation result (normalized genes)", "magma_r", 0., None),
        ("within_population_rmse", "Mean pairwise RMSE inside the final population", "magma", 0., None),
    ]
    for field, title, cmap, low, high in metrics:
        grids = [grid_values(records, populations, epochs, p, field) for p in probabilities]
        if high is None:
            high = max(1e-12, max(float(np.nanmax(g)) for g in grids if np.isfinite(g).any()))
        fig, axes = plt.subplots(1, len(probabilities), figsize=(4.6*len(probabilities), 4.4),
                                 squeeze=False, layout="constrained")
        color = plt.get_cmap(cmap).with_extremes(bad="#eeeeee")
        for ax, probability, grid in zip(axes[0], probabilities, grids):
            displayed = ax.imshow(grid, vmin=low, vmax=high, cmap=color, aspect="auto")
            ax.set(xticks=range(len(epochs)), xticklabels=epochs,
                   yticks=range(len(populations)), yticklabels=populations,
                   xlabel="Epochs completed", ylabel="Population size", title=f"Mutation p = {probability:g}")
            for i in range(len(populations)):
                for j in range(len(epochs)):
                    value = grid[i, j]
                    if not np.isfinite(value):
                        ax.text(j, i, "pending", ha="center", va="center", fontsize=8)
                        continue
                    rgba = color((value-low)/(high-low))
                    brightness = .2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
                    text = f"{value:.1%}" if field == "test_success" else f"{value:.3f}"
                    ax.text(j, i, text, ha="center", va="center", fontsize=9,
                            color="black" if brightness > .55 else "white")
        fig.colorbar(displayed, ax=axes[0].tolist(), shrink=.82)
        fig.suptitle(title)
        save(fig, field)

    def distance_plot(indices, stem, title):
        matrix = distances[np.ix_(indices, indices)]
        ticklabels = [labels[i] for i in indices]
        size = 14 if len(indices) > 20 else 10
        fig, ax = plt.subplots(figsize=(size, size), layout="constrained")
        image = ax.imshow(matrix, cmap="magma_r", vmin=0., vmax=max(1e-12, float(distances.max())))
        ax.set(xticks=range(len(indices)), xticklabels=ticklabels,
               yticks=range(len(indices)), yticklabels=ticklabels, title=title)
        ax.tick_params(axis="both", labelsize=5 if len(indices) > 20 else 8)
        plt.setp(ax.get_xticklabels(), rotation=90)
        if len(indices) <= 16:
            for i in range(len(indices)):
                for j in range(len(indices)):
                    ax.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center", fontsize=7,
                            color="black" if matrix[i,j] <= .5*max(1e-12, distances.max()) else "white")
        fig.colorbar(image, ax=ax, shrink=.7, label="RMSE in normalized gene coordinates")
        save(fig, stem)

    distance_plot(list(range(len(records))), "genome_rmse_all", "Pairwise distance between validation-selected genomes")
    final_indices = [i for i, r in enumerate(records) if r["generations"] == max(epochs)]
    if final_indices:
        distance_plot(final_indices, "genome_rmse_final", f"Final genomes after {max(epochs)} epochs")

    fig, axes = plt.subplots(1, len(probabilities), figsize=(4.6*len(probabilities), 4.),
                             squeeze=False, layout="constrained")
    for ax, probability in zip(axes[0], probabilities):
        for size in populations:
            for seed in settings["seeds"]:
                candidates = [r for r in records if r["population"] == size and
                              r["mutation_probability"] == probability and r["seed"] == seed]
                if not candidates:
                    continue
                longest = max(candidates, key=lambda r: r["generations"])
                with (root / longest["directory"] / "history.csv").open() as stream:
                    history = list(csv.DictReader(stream))
                ax.plot([int(r["generation"]) for r in history], [float(r["best_fitness"]) for r in history],
                        label=f"N={size}, seed={seed}")
        ax.set(xlabel="Epoch", ylabel="Best training fitness", title=f"Mutation p = {probability:g}", ylim=(0., 1.))
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    save(fig, "convergence")

    # All winners are chosen by validation; test results never enter the ranking.
    winner = records[best]
    final_winner = records[max(final_indices, key=lambda i: records[i]["validation_fitness"])] if final_indices else None
    expected = len(populations)*len(epochs)*len(probabilities)*len(settings["seeds"])
    summary = dict(completed=len(records), expected=expected,
                   independent_trajectories=len(populations)*len(probabilities)*len(settings["seeds"]),
                   best_by_validation=winner, best_at_final_epoch=final_winner,
                   rmse_definition="sqrt(mean((z_i-z_j)^2)), canonical normalized free controller genes (or 6 command genes)",
                   seeds=settings["seeds"], conditional_epoch_snapshots=True)
    write_json(root / "analysis.json", summary)
    titles = {"equilibrium": "Обучение равновесию без входных токов",
              "small_nm_weights": "Равновесие при малых нейромышечных весах (0.02 ≤ w < 0.1)",
              "angle_setpoint": "Удержание заданного угла: обучение закона входных токов"}
    conditions = "Полные Afferents; шаг 0.1 мс; эпизод 10 с; масса и геометрия фиксированы. "
    conditions += ("Внешние токи равны нулю. " if settings["experiment"] != "angle_setpoint"
                   else "Параметры выбранного контроллера фиксированы; оптимизируется закон токов. ")
    if settings["experiment"] == "small_nm_weights":
        conditions += "Оба нейромышечных веса ограничены диапазоном **0.02 ≤ w < 0.1**. "
    report = [f"# {titles[settings['experiment']]}: серия параметров\n",
              f"Завершено {len(records)}/{expected} контрольных точек; "
              f"отдельных траекторий обучения: {summary['independent_trajectories']}.\n",
              f"Размеры популяции: {populations}; эпохи: {epochs}; вероятности мутации: {probabilities}; seed: {settings['seeds']}.\n",
              conditions + "\n",
              "Вероятность задаётся **на каждый ген**, постоянна в пределах линии. σ уменьшается от 0.20 до 0.015 "
              f"по единому косинусному расписанию на {max(epochs)} эпох. Результаты меньших бюджетов — контрольные точки той же линии, "
              "а не независимые повторные обучения. Продолжение сохраняет RNG и исходную популяцию.\n",
              "Train/validation/test содержат 12/24/48 независимых испытаний. В каждой контрольной точке оценивается вся популяция "
              "на validation; test применяется к её победителю. Между настройками лучший результат также выбирается по validation.\n",
              "## Лучший результат по validation\n",
              f"N={winner['population']}, p={winner['mutation_probability']:g}, эпох={winner['generations']}: "
              f"validation fitness={winner['validation_fitness']:.6f}, test fitness={winner['test_fitness']:.6f}, "
              f"успех на test={winner['test_success']:.1%}.\n",
              f"[Геном с пояснениями]({winner['directory']}/best_genome.json), "
              f"[метрики отдельных тестов]({winner['directory']}/test_metrics.csv).\n",
              "## Тепловые карты и сходимость\n"]
    report.extend(f"![{stem}]({relative})\n" for stem, relative in figures)
    report.extend(["## Расстояния геномов\n",
                   "RMSE вычисляется по нормированным свободным генам: sqrt(mean((z_i−z_j)²)). "
                   "У равновесия 50 координат; два фиксированных нулевых тока исключены. "
                   "Параметры с логарифмической шкалой сравниваются в координатах своего кодирования. "
                   "Сырые физические величины с разными единицами в один RMSE не смешиваются. "
                   "Это расстояние параметров, а не доказательство функционального различия контроллеров.\n",
                   "[Все метрики](summary.csv), [матрица RMSE](genome_rmse.csv), "
                   "[покомпонентные разности относительно лучшего](genome_differences_to_best.csv), "
                   "[численные массивы](genome_comparison.npz). Снимок использованных исходников: [source_snapshot.zip](source_snapshot.zip).\n",
                   "## Ограничения\n",
                   "При одном seed эта серия показывает результаты конкретных траекторий; она не оценивает разброс между независимыми запусками. "
                   "Строки разных эпох внутри одной линии зависимы. Малое расстояние геномов и высокий fitness сами по себе не доказывают устойчивость "
                   "вне заданного банка и десятисекундного горизонта.\n"])
    (root / "REPORT.md").write_text("\n".join(report))
    return summary
