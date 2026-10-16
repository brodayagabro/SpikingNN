"""Small, common review plots. Numerical results always live in report/results."""
import os
import numpy as np
from ...layout import image_path
from .commands import input_currents


def plot_run(output, history, traces, cases, controller, coefficients):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    axes[0].plot([r["generation"] for r in history], [r["best_fitness"] for r in history], label="Best")
    axes[0].plot([r["generation"] for r in history], [r["median_fitness"] for r in history], label="Median")
    axes[0].set(xlabel="Generation", ylabel="Training fitness")
    for name, trace in traces.items():
        index = int(name.split("_")[1])
        target = np.rad2deg(cases[index, 0])
        axes[1].plot(trace[:, 0], np.rad2deg(trace[:, 1]), label=f"Target {target:g}°")
    axes[1].set(xlabel="Time (s)", ylabel="Angle from upright (deg)")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    destination = image_path(output / "overview.png")
    fig.savefig(destination, dpi=140)
    plt.close(fig)
    with (output / "REPORT.md").open("a") as stream:
        stream.write(f"\n![Обучение и динамика]({os.path.relpath(destination, output)})\n")
    if coefficients is not None:
        angles = np.linspace(-45., 45., 181)
        currents = np.array([input_currents(controller, coefficients, np.deg2rad(a)) for a in angles])
        fig, ax = plt.subplots(layout="constrained")
        ax.plot(angles, currents[:, 0], label="Flexor neuron")
        ax.plot(angles, currents[:, 1], label="Extensor neuron")
        ax.set(xlabel="Command angle (deg)", ylabel="Applied current")
        ax.grid(alpha=.2)
        ax.legend()
        fig.savefig(image_path(output / "input_currents.png"), dpi=140)
        plt.close(fig)
