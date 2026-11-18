"""Тактильная обратная связь. Demonstrate additive afferents in a CPG loop.

Run after installing SpikingNN with ``python -m pip install -e .``::

    python examples/example_additional_afferents.py
    python examples/example_additional_afferents.py --no-show

Two simulations share parameters, seed and initial state. Scheduled contact
pulses add to Ib_Flex and, at half amplitude, Ib_Ext. They represent a sensory
input, not a physical collision or a ground-reaction force. The network reads
the previous step's afferents, so a pulse affects neurons one dt later.

Intrinsic and total afferents are recorded at the SAME post-step state.
The separate no-contact run measures the resulting change in CPG activity and
limb motion. Existing mechanical equations are used without modification.
"""

import argparse
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, Optional, Union

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np

from SpikingNN.core.Izh_net import (
    Afferented_Limb,
    Izhikevich_IO_Network,
    OneDOFLimb,
    SimpleAdaptedMuscle,
    types2params,
)
from SpikingNN.core.Var_Limb import Var_Limb


NAMES = ("CPG_Flex", "MN_Flex", "CPG_Ext", "MN_Ext")
# Okabe-Ito palette: distinguishable with common color-vision deficiencies.
COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")
Trace = Dict[str, np.ndarray]


def create_tactile_signal(
    T: np.ndarray,
    contact_times: Iterable[float],
    amplitude: float = 2.0,
    duration: float = 100.0,
) -> np.ndarray:
    """Создать контактные импульсы. Build six-channel tactile input pulses.

    Parameters
    ----------
    T : np.ndarray
        One-dimensional finite sample times, in milliseconds.
    contact_times : iterable of float
        Contact onset times in milliseconds.
    amplitude : float, optional
        Nonnegative Ib_Flex amplitude in afferent units (a.u.).
    duration : float, optional
        Positive pulse duration in milliseconds.

    Returns
    -------
    np.ndarray
        Shape (len(T), 6); Ib_Ext receives half the Ib_Flex amplitude.

    Notes
    -----
    Pulses occupy [onset, onset + duration). Overlapping contacts saturate
    at amplitude; they do not sum. All other channels remain zero.
    """
    times = np.asarray(T, dtype=float)
    contacts = np.asarray(list(contact_times), dtype=float)
    if times.ndim != 1 or not np.isfinite(times).all():
        raise ValueError("T must be a finite one-dimensional array")
    if contacts.ndim != 1 or not np.isfinite(contacts).all():
        raise ValueError("contact_times must contain finite onset times")
    if not np.isfinite(amplitude) or amplitude < 0:
        raise ValueError("amplitude must be finite and nonnegative")
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be finite and positive")
    signal = np.zeros((len(times), 6))
    for onset in contacts:
        mask = (times >= onset) & (times < onset + duration)
        signal[mask, 2] = amplitude
        signal[mask, 5] = amplitude * 0.5
    return signal


def create_system(seed: int = 42) -> Var_Limb:
    """Собрать ЦГП и конечность. Build a reproducible neuromechanical loop.

    Parameters
    ----------
    seed : int, optional
        Seed for small initial voltage perturbations.

    Returns
    -------
    Var_Limb
        Four neurons, antagonistic muscles and a limb under gravity.

    Notes
    -----
    Afferent weights are illustrative, fixed, and identical in both runs.
    Motor neurons drive the muscles through P. Ib_Flex inhibits the flexor
    CPG and excites the extensor CPG, making contact effects observable.
    """
    a, b, c, d = types2params(["CH", "FS", "CH", "FS"])
    a[[0, 2]] = 0.001
    net = Izhikevich_IO_Network(
        N=4, names=list(NAMES), input_size=2, output_size=2,
        afferent_size=6,
        Q_app=np.array([[1., 0.], [0., 0.], [0., 1.], [0., 0.]]),
        Q_aff=np.array([
            [0.05, 0.02, -2.0, 0., 0., 0.],
            [0.10, 0., -0.5, 0., 0., 0.],
            [0., 0., 0.75, 0.05, 0.02, -2.0],
            [0., 0., 0., 0.10, 0., -0.5],
        ]),
        P=np.array([[0., 1., 0., 0.], [0., 0., 0., 1.]]),
    )
    net.set_params(a=a, b=b, c=c, d=d)
    net.set_weights(np.array([
        [0., 0., 0., -1.1],
        [1.7, 0., 0., 0.],
        [0., -1.1, 0., 0.],
        [0., 0., 1.7, 0.],
    ]))
    net.set_synaptic_relax_constant(np.full((4, 4), 20.0))
    limb = Afferented_Limb(
        Limb=OneDOFLimb(
            q0=np.pi / 2 + 0.4, b=0.01, a1=0.06, a2=0.007,
            m=0.3, ls=0.3,
        ),
        Flexor=SimpleAdaptedMuscle(w=0.5, N=2),
        Extensor=SimpleAdaptedMuscle(w=0.4, N=2),
    )
    # Старый конструктор использует глобальный RNG; сохраняем его состояние.
    rng_state = np.random.get_state()
    try:
        system = Var_Limb(Network=net, Limb=limb)
    finally:
        np.random.set_state(rng_state)
    noise = np.random.default_rng(seed).normal(0., 0.5, size=4)
    system.set_init_conditions(v_noise=noise)
    system.Limb.calc_afferents()
    return system


def simulate(
    T: np.ndarray,
    dt: float,
    additional_afferents: Optional[np.ndarray] = None,
    seed: int = 42,
) -> Trace:
    """Выполнить один опыт. Record a deterministic closed-loop trajectory.

    Parameters
    ----------
    T : np.ndarray
        Uniform post-step sample times in milliseconds, starting at dt.
    dt : float
        Integration step in milliseconds.
    additional_afferents : np.ndarray or None, optional
        Finite input history of shape (len(T), 6); None means no contact.
    seed : int, optional
        Initial voltage seed, reused for the comparison run.

    Returns
    -------
    dict of str to np.ndarray
        Time, V, U, forces, angle, velocity, intrinsic afferents, total
        afferents and additional input, all sampled after each step.

    Notes
    -----
    The intrinsic component is total minus the applied additive input at
    the same state. It is distinct from the independent no-contact run.
    """
    times = np.asarray(T, dtype=float)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all():
        raise ValueError("T must be a nonempty finite one-dimensional array")
    if not np.allclose(times, dt * np.arange(1, len(times) + 1)):
        raise ValueError("T must contain uniform times starting at dt")
    additions = (np.zeros((len(times), 6)) if additional_afferents is None
                 else np.asarray(additional_afferents, dtype=float))
    if additions.shape != (len(times), 6) or not np.isfinite(additions).all():
        raise ValueError("Expected finite additional_afferents, (len(T), 6)")
    system = create_system(seed)
    trace = {
        "time": times.copy(), "V": np.empty((len(times), 4)),
        "U": np.empty((len(times), 4)), "forces": np.empty((len(times), 2)),
        "q": np.empty(len(times)), "w": np.empty(len(times)),
        "intrinsic": np.empty((len(times), 6)),
        "afferents": np.empty((len(times), 6)),
        "additional": additions.copy(),
    }
    drive = np.array([5., 5.])
    for index, addition in enumerate(additions):
        # Один порядок шага и одинаковые начальные условия в обоих опытах.
        system.step(dt=dt, Iapp=drive,
                    additional_afferents=addition if addition.any() else None)
        trace["V"][index] = system.V
        trace["U"][index] = system.U
        trace["forces"][index] = system.F_flex, system.F_ext
        trace["q"][index], trace["w"][index] = system.q, system.w
        trace["afferents"][index] = system.Limb.output
        trace["intrinsic"][index] = system.Limb.output - addition
    if not all(np.isfinite(values).all() for values in trace.values()):
        raise FloatingPointError("Nonfinite simulation trajectory")
    return trace


def run_simulation_with_tactile_feedback(
    duration_ms: float = 10000.0, dt: float = 0.1, seed: int = 42,
) -> Dict[str, Trace]:
    """Сравнить два опыта. Run matched simulations with and without contact.

    Parameters
    ----------
    duration_ms : float, optional
        Positive simulation duration, an integer multiple of dt.
    dt : float, optional
        Integration step in milliseconds; default 0.1 ms.
    seed : int, optional
        Shared seed for initial conditions.

    Returns
    -------
    dict of str to dict
        Baseline and tactile trajectories, under those two keys.
    """
    if not np.isfinite(duration_ms) or duration_ms <= 0:
        raise ValueError("duration_ms must be finite and positive")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    steps = int(round(duration_ms / dt))
    if steps < 1 or not np.isclose(steps * dt, duration_ms):
        raise ValueError("duration_ms must be an integer multiple of dt")
    times = dt * np.arange(1, steps + 1)
    tactile = create_tactile_signal(times, [2000., 4000., 6000.])
    return {
        "baseline": simulate(times, dt, seed=seed),
        "tactile": simulate(times, dt, tactile, seed=seed),
    }


def plot_results(
    results: Dict[str, Trace],
    output: Union[str, Path] = "additional_afferents_example.png",
    show: bool = True,
) -> Figure:
    """Построить девять графиков. Save a nine-panel comparison as PNG.

    Parameters
    ----------
    results : dict
        Baseline and tactile traces returned by the simulation runner.
    output : str or pathlib.Path, optional
        PNG destination, saved at 150 DPI.
    show : bool, optional
        Whether to open the interactive plot window after saving.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with nine data panels, a twin velocity axis and a colorbar.
    """
    base, tactile = results["baseline"], results["tactile"]
    times = tactile["time"]
    blue, orange, green, purple = COLORS
    fig, grid = plt.subplots(3, 3, figsize=(20, 12))
    axes = grid.ravel()

    for index, (name, color) in enumerate(zip(NAMES, COLORS)):
        axes[0].plot(times, base["V"][:, index], color=color,
                     linestyle=":", linewidth=0.5, alpha=0.4)
        axes[0].plot(times, tactile["V"][:, index], color=color,
                     linewidth=0.5, label=name)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([], [], color="0.4", linestyle=":"))
    axes[0].legend(handles, labels + ["No tactile: dotted"], fontsize=8)
    axes[0].set(title="CPG and motor neuron activity", ylabel="Voltage (mV)")

    for index, (name, color) in enumerate(zip(("Flexor", "Extensor"), COLORS)):
        axes[1].plot(times, tactile["forces"][:, index], color=color,
                     linewidth=0.8, label=name + " + tactile")
        axes[1].plot(times, base["forces"][:, index], color=color,
                     linestyle="--", linewidth=0.8, label=name + " baseline")
    axes[1].set(title="Antagonistic muscle forces", ylabel="Force (N)")

    twin = axes[2].twinx()
    for trace, suffix, style in ((tactile, " + tactile", "-"),
                                 (base, " baseline", "--")):
        axes[2].plot(times, trace["q"], color=blue, linestyle=style,
                     linewidth=0.8, label="Angle" + suffix)
        twin.plot(times, trace["w"], color=orange, linestyle=style,
                  linewidth=0.8, label="Velocity" + suffix)
    axes[2].set(title="Limb kinematics", ylabel="Angle (rad)")
    twin.set_ylabel("Angular velocity (rad/s)")
    twin.grid(True, alpha=0.3)
    handles, labels = axes[2].get_legend_handles_labels()
    extra_handles, extra_labels = twin.get_legend_handles_labels()
    axes[2].legend(handles + extra_handles, labels + extra_labels, fontsize=8)

    for axis, offset, muscle in (
        (axes[3], 0, "Flexor"), (axes[4], 3, "Extensor"),
    ):
        for index, (kind, color) in enumerate(zip(("Ia", "II", "Ib"), COLORS)):
            axis.plot(times, tactile["intrinsic"][:, offset + index],
                      color=color, linewidth=0.8, label=kind + "_" + muscle)
        axis.set(title=muscle + " intrinsic afferents (tactile run)",
                 ylabel="Afferent activity (a.u.)")

    for index, name, color in ((2, "Ib_Flex", purple), (5, "Ib_Ext", green)):
        axes[5].plot(times, tactile["additional"][:, index],
                     color=color, linewidth=0.8, label=name)
        difference = (
            tactile["afferents"][:, index] - tactile["intrinsic"][:, index]
        )
        axes[7].plot(times, difference, color=color, linewidth=0.8,
                     label="Delta " + name)
    axes[5].set(title="Scheduled tactile contact pulses",
                ylabel="Additional input (a.u.)")
    axes[7].set(title="Added component at the same state",
                ylabel="Total - intrinsic (a.u.)")

    axes[6].plot(times, base["afferents"][:, 2], color="0.4",
                 linestyle="--", linewidth=0.8,
                 label="Independent no-contact run")
    axes[6].plot(times, tactile["intrinsic"][:, 2], color=blue,
                 linewidth=0.8, label="Intrinsic, tactile run")
    axes[6].plot(times, tactile["afferents"][:, 2], color=orange,
                 linewidth=0.8, label="Total, tactile run")
    axes[6].set(title="Ib_Flex: intrinsic and total feedback",
                ylabel="Afferent activity (a.u.)")

    axes[8].plot(base["q"], base["w"], color="0.5", linewidth=0.8,
                 linestyle="--", label="No tactile")
    scatter = axes[8].scatter(tactile["q"], tactile["w"], c=times,
                              cmap="viridis", s=1, alpha=0.6,
                              label="With tactile")
    fig.colorbar(scatter, ax=axes[8], label="Time (ms)")
    axes[8].set(title="Limb phase portrait", xlabel="Angle (rad)",
                ylabel="Angular velocity (rad/s)")

    for index, axis in enumerate(axes):
        axis.grid(True, alpha=0.3)
        if index < 8:
            axis.set_xlabel("Time (ms)")
        if index not in (0, 2):
            axis.legend(fontsize=8)
    fig.suptitle("Tactile feedback: matched states, one-step feedback delay")
    fig.tight_layout()
    fig.savefig(output, dpi=150, format="png")
    if show:
        plt.show()
    return fig


def main() -> None:
    """Запустить демонстрацию. Run the example and print comparison metrics.

    Returns
    -------
    None
        Writes the PNG and optionally displays the figure.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true",
                        help="Save the PNG without opening a plot window")
    parser.add_argument("--output", type=Path,
                        default=Path("additional_afferents_example.png"))
    args = parser.parse_args()
    start = perf_counter()
    results = run_simulation_with_tactile_feedback()
    elapsed = perf_counter() - start
    base, tactile = results["baseline"], results["tactile"]
    print(f"Две симуляции по 10 с завершены за {elapsed:.2f} с.")
    delta_v = np.max(np.abs(tactile["V"] - base["V"]))
    delta_q = np.max(np.abs(tactile["q"] - base["q"]))
    print(f"Максимальное |Delta V|: {delta_v:.3f} mV")
    print(f"Максимальное |Delta q|: {delta_q:.6f} rad")
    spikes = (base["V"] >= 30).sum(axis=0), (tactile["V"] >= 30).sum(axis=0)
    print(f"Спайки без контакта / с контактом: {spikes[0]} / {spikes[1]}")
    figure = plot_results(results, output=args.output, show=not args.no_show)
    plt.close(figure)
    print(f"Графики сохранены: {args.output}")


if __name__ == "__main__":
    main()
