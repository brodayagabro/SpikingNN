"""Plots and nonlinear signal inspection from full-resolution replay files."""
import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ...layout import image_path
from .wide_replay import nonlinear_signals

TERM_NAMES = ("Ia_velocity", "Ia_length", "Ia_activation", "Ia_constant",
              "II_length", "II_activation", "II_constant", "Ib_force")


def read_rows(root):
    with (Path(root)/"metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for field in ("trial", "samples"):
            row[field] = int(row[field])
        for field in ("theta0_deg", "omega0_rad_s", "kick_time_s", "kick_delta_omega",
                      "score", "success", "settling_s", "late_rmse_deg", "survival_s"):
            row[field] = float(row[field])
    return rows


def save(fig, root, stem):
    path = image_path(Path(root)/f"{stem}.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return os.path.relpath(path, root)


def inspect_case(root, row, start=0., stop=None):
    root = Path(root)
    with np.load(root/row["trace_file"], allow_pickle=False) as data:
        trace, p = data["trace"].copy(), data["physical_controller"].copy()
        spikes = [data[f"spike_times_{side}_s"].copy() for side in ("flex", "ext")]
    diagnostics = nonlinear_signals(trace, p)
    destination = root/"diagnostics"/f"case_{row['trial']:03d}.npz"
    destination.parent.mkdir(exist_ok=True)
    np.savez_compressed(destination, time_s=trace[:, 0], **diagnostics,
                        afferent_term_names=np.array(TERM_NAMES),
                        muscle_names=np.array(["flexor", "extensor"]))
    stop = trace[-1, 0] if stop is None else min(stop, trace[-1, 0])
    if start < 0. or stop <= start:
        raise ValueError("Choose a nonempty time window inside the saved trace")
    selected = np.flatnonzero((trace[:, 0] >= start) & (trace[:, 0] <= stop))
    shown = selected[::max(1, len(selected)//6000)]
    t = trace[shown, 0]
    fig, ax = plt.subplots(4, 2, figsize=(13, 13), layout="constrained")
    ax[0, 0].plot(t, np.rad2deg(trace[shown, 1]))
    ax[0, 0].axhspan(-3., 3., alpha=.15, color="green", label="success angle band")
    ax[0, 0].set(xlabel="Time (s)", ylabel="Angle from upright (degrees)")
    ax[0, 0].legend(fontsize=8)
    for i, side in enumerate(("flexor", "extensor")):
        ax[0, 1].plot(t, trace[shown, 3+i], label=side)
        calcium = trace[shown, 9+i]
        ax[1, 1].plot(calcium, diagnostics["activation"][shown, i], '.', ms=1.5, label=side)
        for j, (kind, style) in enumerate(zip(("Ia", "II", "Ib"), ("-", "--", ":"))):
            ax[2, 0].plot(t, trace[shown, 11+3*i+j], style, lw=.8, label=f"{kind} {side}")
        for j, kind in enumerate(("velocity", "length", "activation")):
            ax[2, 1].plot(t, diagnostics["afferent_terms"][shown, i, j],
                          "-" if i == 0 else "--", lw=.8, label=f"Ia {kind}, {side}")
        events = spikes[i][(spikes[i] >= start) & (spikes[i] <= stop)]
        ax[3, 0].scatter(events, np.full(len(events), i), marker="|", s=8)
    ax[0, 1].set(xlabel="Time (s)", ylabel="Muscle force (N)")
    ax[0, 1].legend()
    ax[1, 0].plot(np.rad2deg(trace[shown, 1]), trace[shown, 2], lw=.8)
    ax[1, 0].set(xlabel="Angle from upright (degrees)", ylabel="Angular velocity (rad/s)")
    ax[1, 1].set(xlabel="Calcium Cn", ylabel="Nonlinear muscle activation x", ylim=(-.02, 1.02))
    ax[1, 1].legend()
    ax[2, 0].set(xlabel="Time (s)", ylabel="Recorded afferent response")
    ax[2, 0].legend(fontsize=7, ncol=2)
    ax[2, 1].set(xlabel="Time (s)", ylabel="Nonlinear Ia components")
    ax[2, 1].legend(fontsize=7, ncol=2)
    ax[3, 0].set(xlabel="Time (s)", yticks=[0, 1], yticklabels=["flexor", "extensor"],
                 ylim=(-.5, 1.5), title="Exact spike times (no temporal decimation)")
    zoom = selected[trace[selected, 0] <= min(start+.25, stop)]
    for i, side in enumerate(("flexor", "extensor")):
        ax[3, 1].plot(trace[zoom, 0], trace[zoom, 5+i], lw=.7, label=side)
    ax[3, 1].set(xlabel="Time (s), first 250 ms of selected window", ylabel="Membrane V (mV)")
    ax[3, 1].legend()
    for axis in ax.flat:
        axis.grid(alpha=.2)
    fig.suptitle(f"Case {row['trial']}: initial angle {row['theta0_deg']:g}°, "
                 f"velocity {row['omega0_rad_s']:g} rad/s, kick {row['kick_delta_omega']:g} rad/s; "
                 f"success={int(row['success'])}")
    suffix = "" if start == 0. and stop == trace[-1, 0] else f"_{start:g}_{stop:g}s"
    return save(fig, root, f"case_{row['trial']:03d}_nonlinear{suffix}")


def analyze(root):
    root = Path(root).resolve()
    settings = json.loads((root/"run.json").read_text())
    summary = json.loads((root/"summary.json").read_text())
    rows = read_rows(root)
    angles, velocities, kicks = (settings[k] for k in ("angles_deg", "velocities_rad_s", "kicks_rad_s"))
    figures = []
    for field, title, cmap in (("success", "Success on wide initial conditions", "viridis"),
                               ("settling_s", "Last exit from the angle / speed band (s)", "magma_r"),
                               ("late_rmse_deg", "Late angular RMSE (degrees)", "magma_r")):
        fig, axes = plt.subplots(1, len(kicks), figsize=(6.5*len(kicks), 3.7),
                                 squeeze=False, layout="constrained")
        upper = 1. if field == "success" else max(1e-12, max(r[field] for r in rows))
        for axis, kick in zip(axes[0], kicks):
            grid = np.full((len(velocities), len(angles)), np.nan)
            for row in rows:
                if row["kick_delta_omega"] == kick:
                    i = velocities.index(row["omega0_rad_s"])
                    j = int(np.argmin(np.abs(np.array(angles)-row["theta0_deg"])))
                    grid[i, j] = row[field]
            shown = axis.imshow(grid, vmin=0., vmax=upper, aspect="auto", cmap=cmap)
            for i, j in np.ndindex(grid.shape):
                label = f"{grid[i,j]:.0f}" if field == "success" else f"{grid[i,j]:.2f}"
                rgb = plt.get_cmap(cmap)(grid[i,j]/upper)
                light = .2126*rgb[0]+.7152*rgb[1]+.0722*rgb[2]
                axis.text(j, i, label, ha="center", va="center", fontsize=7,
                          color="black" if light > .55 else "white")
            axis.set(xticks=range(len(angles)), xticklabels=[f"{a:g}" for a in angles],
                     yticks=range(len(velocities)), yticklabels=velocities,
                     xlabel="Initial angle from upright (degrees)", ylabel="Initial velocity (rad/s)",
                     title=f"Kick Δω={kick:g} rad/s" + (f" at {settings['kick_time_s']:g} s" if kick else ""))
            axis.tick_params(labelsize=8)
        fig.colorbar(shown, ax=axes[0].tolist(), shrink=.8)
        fig.suptitle(title)
        figures.append((title, save(fig, root, field)))

    neutral_kick = min(kicks, key=abs)
    neutral_speed = min(velocities, key=abs)
    neutral = [r for r in rows if r["omega0_rad_s"] == neutral_speed and r["kick_delta_omega"] == neutral_kick]
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), layout="constrained")
    color = plt.get_cmap("coolwarm")
    for row in neutral:
        with np.load(root/row["trace_file"], allow_pickle=False) as data:
            tr = data["trace"][::20].copy()
        c = color((row["theta0_deg"]-min(angles))/max(1e-12, max(angles)-min(angles)))
        axes[0].plot(tr[:, 0], np.rad2deg(tr[:, 1]), color=c, label=f"{row['theta0_deg']:g}°", lw=.8)
        axes[1].plot(np.rad2deg(tr[:, 1]), tr[:, 2], color=c, lw=.8)
        late = tr[:, 0] >= max(0., settings["duration_s"]-2.)
        axes[2].plot(tr[late, 0], np.rad2deg(tr[late, 1]), color=c, lw=.7)
    axes[0].axhspan(-3., 3., color="green", alpha=.12)
    axes[0].set(xlabel="Time (s)", ylabel="Angle from upright (degrees)")
    axes[0].legend(ncol=7, fontsize=8, loc="upper right")
    axes[1].set(xlabel="Angle from upright (degrees)", ylabel="Angular velocity (rad/s)")
    axes[2].set(xlabel="Time (s)", ylabel="Late angle (degrees)")
    for axis in axes:
        axis.grid(alpha=.2)
    fig.suptitle(f"{settings['experiment']}: recovery from initial angles; "
                 f"ω₀={neutral_speed:g}, kick={neutral_kick:g} rad/s")
    figures.append(("Угол, фазовые траектории и поздняя ошибка", save(fig, root, "angle_responses")))
    for angle in sorted(set((min(angles), min(angles, key=abs), max(angles)))):
        row = min(neutral, key=lambda r: abs(r["theta0_deg"]-angle))
        figures.append((f"Нелинейные сигналы из {angle:g}°", inspect_case(root, row)))

    report = [f"# Проверка больших начальных отклонений: {settings['experiment']}\n",
              "Проверяется возвращение к **верхней вертикали θ*=0** из разных начальных углов. "
              "Контроллер зафиксирован по validation прежней серии; новые испытания не участвуют в выборе или обучении.\n",
              f"Начальные углы: {angles}; скорости: {velocities} рад/с; толчки: {kicks} рад/с "
              f"в {settings['kick_time_s']:g} с. Эпизод {settings['duration_s']:g} с, шаг **0.1 мс**, "
              "полные Afferents, нейронное управление `mode=0`, Iapp=[0,0], ПД не используется.\n",
              f"**Успех {summary['successes']}/{summary['cases']}**, падений {summary['falls']}, "
              f"численных отказов {summary['numerical_failures']}. "
              f"Максимальное время последнего выхода из допустимой полосы: {summary['max_settling_s']:.4f} с. "
              f"Средняя / максимальная поздняя угловая RMSE: {summary['mean_late_rmse_deg']:.6f}° / "
              f"{summary['max_late_rmse_deg']:.6f}°.\n",
              "Успех требует |θ|≤3° и |ω|≤0.35 рад/с с момента не позже 3 с до конца, включая толчок. "
              "Выход за ±90° завершает испытание падением. Это конечная сетка испытаний на заданном горизонте, "
              "а не доказательство устойчивости для всех состояний между узлами или вне диапазона.\n",
              "## Данные и воспроизводимость\n",
              "[Все условия](cases.csv), [метрики и SHA-256 траекторий](metrics.csv), [численная сводка](summary.json), "
              "[выбранный геном](best_genome.json), [точные физические параметры](physical_controller.npy), "
              "[настройки и происхождение](run.json), [аудит](verification.json), "
              "[точное повторение прежних train/validation](original_replay_check.json), "
              "[снимок исходников](source_snapshot.zip).\n",
              f"В [traces](traces) сохранены **{summary['total_samples']:,} отсчётов**, "
              f"объём NPZ {summary['trace_bytes']/1024**3:.2f} GiB. Каждый `case_XXX.npz` содержит "
              "все вычислительные отсчёты (без прореживания) и отдельную строку t=0. "
              "Имена 17 каналов находятся в `columns`, времена спайков — в `spike_times_flex_s/ext_s`. "
              "До начала интегрирования афференты, силы и кальций сброшены в ноль.\n",
              "[diagnostics](diagnostics) содержит полные ряды активации, длины/скорости мышц, "
              "слагаемых рецепторного отклика и афферентного тока для показанных ниже отдельных испытаний. "
              "Их можно вычислить для любой сохранённой траектории функцией `nonlinear_signals`.\n",
              "На обзорных графиках линии прорежены только для отображения. Времена спайков и окно "
              "мембранных потенциалов показаны без временного прореживания.\n",
              "## Графики\n"]
    report.extend(f"![{title}]({link})\n" for title, link in figures)
    (root/"REPORT.md").write_text("\n".join(report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trial", type=int)
    parser.add_argument("--start", type=float, default=0.)
    parser.add_argument("--stop", type=float)
    args = parser.parse_args()
    if args.trial is None:
        analyze(args.output)
    else:
        rows = read_rows(args.output)
        matches = [r for r in rows if r["trial"] == args.trial]
        if len(matches) != 1:
            parser.error("Trial not found in metrics.csv")
        print(inspect_case(args.output.resolve(), matches[0], args.start, args.stop))


if __name__ == "__main__":
    main()
