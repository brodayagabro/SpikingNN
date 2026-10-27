"""Notebook utilities for saved genomes, state traces, phase grids and GA sweeps."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from .core.genome import GenomeConfig, MIRROR_INDICES, NAMES, SCHEMA, load_genome
from .simulation import case_grid, simulate


def configure_fonts(size=13):
    """Use Times New Roman when installed and a readable serif fallback otherwise."""
    candidates = Path.home() / ".wine/drive_c/windows/Fonts"
    for path in candidates.glob("times*.ttf"):
        font_manager.fontManager.addfont(str(path))
    available = {font.name for font in font_manager.fontManager.ttflist}
    family = "Times New Roman" if "Times New Roman" in available else "DejaVu Serif"
    plt.rcParams.update({"font.family": family, "font.size": size,
                         "axes.titlesize": size+1, "pdf.fonttype": 42, "svg.fonttype": "none"})


def load_solution(path, rank=1):
    """Load a JSON genome or a ranked candidate from a saved generation/population.

    Final populations use validation ranks. Intermediate generations use training
    ranks, which are explicitly returned in the metadata. Geometry is restored
    from the saved configuration and cannot be silently replaced by live defaults.
    """
    path = Path(path).resolve()
    if path.suffix == ".json":
        config, physical = load_genome(path)
        return config, physical, {"selection": "saved JSON genome", "file": str(path)}
    with np.load(path, allow_pickle=False) as saved:
        if str(saved["schema"]) != SCHEMA or tuple(saved["names"]) != NAMES:
            raise ValueError("Incompatible population schema; historical arrays require explicit conversion")
        config_path = next((parent / "genome_cfg.json" for parent in path.parents
                            if (parent / "genome_cfg.json").is_file()), None)
        if config_path is None:
            raise FileNotFoundError("No saved genome_cfg.json found for this population")
        config = GenomeConfig(config_path)
        if str(saved["configuration_sha256"]) != config.signature:
            raise ValueError("Population configuration checksum mismatch")
        key = "validation_scores" if "validation_scores" in saved else "training_scores"
        if not isinstance(rank, int) or not 1 <= rank <= len(saved[key]):
            raise ValueError("Requested rank is outside the population")
        index = int(np.argsort(-saved[key], kind="stable")[rank-1])
        physical = saved["physical"][index].copy()
        config.encode(physical)
        return config, physical, dict(selection=key, rank=rank, population_index=index,
                                      generation=int(saved["generation"]), file=str(path))


def replay_grid(config, physical, angles_deg=None, speeds=None, *, target_deg=0.,
                currents=(0., 0.), duration_s=None, stride=10):
    """Integrate an initial-condition grid at 0.1 ms with optional trace decimation.

    Default recording stride 10 gives 1 ms plot samples; fitness still uses every
    0.1 ms integration step. Set stride=1 when full arrays are required.
    """
    duration_s = config.data["simulation"]["duration_s"] if duration_s is None else duration_s
    result = []
    for case in case_grid(angles_deg, speeds, target_deg):
        metrics, trace = simulate(physical, case, config.mechanics, currents=currents,
                                  duration_s=duration_s, stride=stride)
        result.append(dict(case=case, metrics=metrics, trace=trace,
                           currents=np.asarray(currents), stride=stride))
    return result


def parameter_scan(config, physical, name, values, *, theta0_deg=30., omega0=0.,
                   target_deg=0., currents=(0., 0.), duration_s=None):
    """Replay one case while varying a mutable parameter and its tied partner.

    Fixed geometry cannot be swept here; change a1/a2 in a separate configuration
    to compare explicitly different mechanical systems. A mirrored domain changes
    both physical members of the selected pair, including a nonrepresentative one.
    """
    if name not in NAMES:
        raise ValueError(f"Unknown genome parameter: {name}")
    index = NAMES.index(name)
    if index not in config.mutable:
        raise ValueError("Fixed geometry/hyperparameters require a separate configuration")
    results = []
    for value in values:
        candidate = np.array(physical, copy=True)
        candidate[index] = value
        if config.mirror_symmetry:
            candidate[MIRROR_INDICES[index]] = value
        config.encode(candidate)
        record = replay_grid(config, candidate, [theta0_deg], [omega0], target_deg=target_deg,
                             currents=currents, duration_s=duration_s, stride=10)[0]
        record.update(parameter=name, value=float(value))
        results.append(record)
    return results


def plot_phase(records):
    """Plot real trajectories, initial states, target positions and early fall markers."""
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for record in records:
        trace, case = record["trace"], record["case"]
        color = plt.cm.coolwarm((np.rad2deg(case[1])+90)/180)
        ax.plot(np.rad2deg(trace[:, 1]), trace[:, 2], color=color, lw=.8, alpha=.8)
        ax.scatter(np.rad2deg(case[1]), case[2], s=12, c=[color])
        if abs(trace[-1, 1]) > np.pi/2:
            ax.plot(np.rad2deg(trace[-1, 1]), trace[-1, 2], "kx", ms=5)
    for target in sorted({float(record["case"][0]) for record in records}):
        ax.plot(np.rad2deg(target), 0., "k*", ms=12)
    ax.set(xlabel="Angle from upright (deg)", ylabel="Angular velocity (rad/s)",
           title=f"Phase trajectories: {len(records)} initial conditions")
    ax.grid(alpha=.2)
    return fig


def plot_angles(records):
    """Plot angle histories without suppressing early failures."""
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for record in records:
        case, trace = record["case"], record["trace"]
        label = (f"{record['parameter']}={record['value']:.4g}" if "parameter" in record
                 else f"{np.rad2deg(case[1]):.1f} deg; {case[2]:.2g} rad/s")
        ax.plot(trace[:, 0], np.rad2deg(trace[:, 1]), label=label)
    for target in sorted({float(record["case"][0]) for record in records}):
        ax.axhline(np.rad2deg(target), color="black", ls="--", lw=1)
    ax.set(xlabel="Time (s)", ylabel="Angle from upright (deg)")
    ax.grid(alpha=.2)
    if len(records) <= 12:
        ax.legend(fontsize=9, ncol=2)
    return fig


def plot_states(record):
    """Return limb, neuron, muscle and afferent figures for exactly one trajectory."""
    trace = record["trace"]
    t = trace[:, 0]
    figures = {}
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, constrained_layout=True)
    axes[0].plot(t, np.rad2deg(trace[:, 1]))
    axes[0].axhline(np.rad2deg(record["case"][0]), color="black", ls="--")
    axes[1].plot(t, trace[:, 2])
    axes[0].set_ylabel("Angle (deg)")
    axes[1].set(xlabel="Time (s)", ylabel="Angular velocity (rad/s)")
    figures["limb"] = fig
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, constrained_layout=True)
    for i, color in enumerate(("tab:blue", "tab:orange")):
        axes[0].plot(t, trace[:, 5+i], color=color, label=f"Neuron {i}")
        axes[1].plot(t, trace[:, 7+i], color=color)
        spikes = t[trace[:, 19+i] > 0]
        axes[2].plot(spikes, np.full(len(spikes), i), "|", color=color, ms=12)
    axes[0].set_ylabel("Membrane V (mV)")
    axes[1].set_ylabel("Recovery U")
    axes[2].set(xlabel="Time (s)", ylabel="Spike events", yticks=[0, 1])
    axes[0].legend()
    figures["neurons"] = fig
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, constrained_layout=True)
    for i, label in enumerate(("Flexor", "Extensor")):
        for ax, offset in zip(axes, (17, 3, 9)):
            ax.plot(t, trace[:, offset+i], label=label)
    for ax, label in zip(axes, ("Activation x", "Force (N)", "Calcium C")):
        ax.set_ylabel(label)
    axes[0].legend()
    axes[-1].set_xlabel("Time (s)")
    figures["muscles"] = fig
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, constrained_layout=True)
    for j, kind in enumerate(("Ia", "II", "Ib")):
        for i, label in enumerate(("Flexor", "Extensor")):
            axes[j].plot(t, trace[:, 11+3*i+j], label=label)
        axes[j].set_ylabel(f"{kind} (model units)")
    axes[0].legend()
    axes[-1].set_xlabel("Time (s)")
    figures["afferents"] = fig
    for figure in figures.values():
        for ax in figure.axes:
            ax.grid(alpha=.2)
    return figures


def plot_sweep(directory, rows):
    """Save fitness heatmaps for each mutation/seed combination and genome RMSE."""
    directory = Path(directory)
    configure_fonts()
    sizes = sorted({r["population"] for r in rows})
    epochs = sorted({r["generation"] for r in rows})
    for seed, probability in sorted({(r["seed"], r["mutation_probability"]) for r in rows}):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for ax, field, title in zip(axes, ("validation_fitness", "test_success"),
                                    ("Validation fitness", "Test success")):
            grid = np.full((len(sizes), len(epochs)), np.nan)
            for row in rows:
                if row["seed"] == seed and row["mutation_probability"] == probability:
                    grid[sizes.index(row["population"]), epochs.index(row["generation"])] = row[field]
            image = ax.imshow(grid, origin="lower", aspect="auto", vmin=0, vmax=1)
            ax.set(xticks=range(len(epochs)), xticklabels=epochs,
                   yticks=range(len(sizes)), yticklabels=sizes,
                   xlabel="Generation", ylabel="Population", title=title)
            fig.colorbar(image, ax=ax)
        fig.suptitle(f"Mutation probability={probability}; seed={seed}")
        for extension in ("png", "pdf"):
            fig.savefig(directory / f"heatmap_seed_{seed}_mutation_{probability:.8g}.{extension}", dpi=180)
        plt.close(fig)
    with np.load(directory / "genome_rmse.npz", allow_pickle=False) as saved:
        distances = saved["rmse"]
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    image = ax.imshow(distances, origin="lower")
    ax.set(xlabel="Checkpoint index (summary.csv order)", ylabel="Checkpoint index",
           title="Genome RMSE: normalized mutable parameters")
    fig.colorbar(image, ax=ax)
    for extension in ("png", "pdf"):
        fig.savefig(directory / f"genome_rmse.{extension}", dpi=180)
    plt.close(fig)


def plot_condition_banks(directory, banks):
    """Show the actual fixed angle/speed samples used by a balance study."""
    configure_fonts()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, name in zip(axes, ('training', 'validation', 'test')):
        cases = banks[name]
        ax.scatter(np.rad2deg(cases[:, 1]), cases[:, 2], s=24)
        ax.axvline(0, color='grey', linewidth=.5)
        ax.axhline(0, color='grey', linewidth=.5)
        ax.set(xlabel='Initial angle (deg)', ylabel='Initial speed (rad/s)',
               title=f'{name.capitalize()}: {len(cases)} cases')
        ax.grid(alpha=.2)
    for extension in ('png', 'pdf'):
        fig.savefig(Path(directory) / f'initial_conditions.{extension}', dpi=180)
    plt.close(fig)
