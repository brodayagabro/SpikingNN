"""Resolve near-unity fitness values without changing the saved training results."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from ...layout import image_path
from .sweep_analysis import grid_values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    settings = json.loads((root / "sweep.json").read_text())
    with (root / "summary.csv").open() as stream:
        records = list(csv.DictReader(stream))
    for row in records:
        for key in ("population", "generations"):
            row[key] = int(row[key])
        for key in ("mutation_probability", "training_fitness", "validation_fitness", "test_fitness"):
            row[key] = float(row[key])
    populations, epochs, probabilities = settings["populations"], settings["epochs"], settings["probabilities"]
    links = []
    for field in ("validation_fitness", "test_fitness"):
        values = [grid_values(records, populations, epochs, p, field) for p in probabilities]
        deficits = [1.-v for v in values]
        positive = np.concatenate([d[np.isfinite(d) & (d > 0.)] for d in deficits])
        low = max(1e-12, float(positive.min())) if len(positive) else 1e-12
        high = max(low*10., float(positive.max())) if len(positive) else 1.
        norm = LogNorm(vmin=low, vmax=high)
        color = plt.get_cmap("viridis_r").with_extremes(bad="#eeeeee")
        fig, axes = plt.subplots(1, len(probabilities), figsize=(4.9*len(probabilities), 4.4),
                                 squeeze=False, layout="constrained")
        for ax, probability, grid, deficit in zip(axes[0], probabilities, values, deficits):
            shown = ax.imshow(np.maximum(deficit, 1e-12), cmap=color, norm=norm, aspect="auto")
            ax.set(xticks=range(len(epochs)), xticklabels=epochs, yticks=range(len(populations)),
                   yticklabels=populations, xlabel="Epochs completed", ylabel="Population size",
                   title=f"Mutation p = {probability:g}")
            for i, j in np.ndindex(grid.shape):
                if not np.isfinite(grid[i, j]):
                    ax.text(j, i, "pending", ha="center", va="center", fontsize=8)
                    continue
                rgba = color(norm(max(1e-12, deficit[i, j])))
                light = .2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
                ax.text(j, i, f"{grid[i,j]:.5f}", ha="center", va="center", fontsize=9,
                        color="black" if light > .55 else "white")
        fig.colorbar(shown, ax=axes[0].tolist(), shrink=.82, label="1 - fitness (logarithmic color scale)")
        fig.suptitle(f"{field.replace('_', ' ').capitalize()}: near-unity differences")
        destination = image_path(root / f"{field}_detail.png")
        fig.savefig(destination, dpi=180, bbox_inches="tight")
        fig.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        links.append((field, os.path.relpath(destination, root)))
    report = root / "REPORT.md"
    text = report.read_text()
    marker = "\n## Различия fitness около единицы\n"
    if marker in text:
        text = text.split(marker)[0]
    text += marker + "\nЧисла в клетках — fitness с пятью знаками; цвет показывает 1−fitness "
    text += "на логарифмической шкале. Эти карты детализируют уже сохранённые оценки и не меняют отбор.\n\n"
    text += "\n\n".join(f"![{field}]({link})" for field, link in links) + "\n"
    report.write_text(text)
    (root / "detail_plot_source.json").write_text(json.dumps(dict(
        source=str(Path(__file__).resolve()), sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2)+"\n")


if __name__ == "__main__":
    main()
