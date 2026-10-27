"""Compare frozen C winners on a shared test bank and identical example starts.

This is descriptive postprocessing, never a controller-selection step. The old
and new studies may have different GA seeds and search budgets; their difference
must not be interpreted as an isolated causal effect of a relaxation-time bound.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.AfferentBalance2026.experiments.core.genome import file_hash, write_json
from scripts.AfferentBalance2026.experiments.core.run import PopulationEvaluator, write_csv
from scripts.AfferentBalance2026.experiments.C_setpoint_balance.joint_study import (
    evaluate_joint, load_joint_genome,
)
from scripts.AfferentBalance2026.experiments.visualization import configure_fonts

TIME_INDICES = (19, 20, 26, 27)


def compare(previous, current, *, workers=1):
    """Save common-bank performance, late force variation and unsmoothed plots.

    Parameters
    ----------
    previous, current : pathlib.Path
        Completed sweep directories with audited per-method diagnostics.
    workers : int
        Worker count for replaying the previous winners on the new test bank.

    Returns
    -------
    dict
        Result location, time bounds and per-method performance and force rows.
    """
    import matplotlib.pyplot as plt

    previous, current = Path(previous).resolve(), Path(current).resolve()
    for folder in (previous, current):
        if json.loads((folder/'audit.json').read_text())['status'] != 'passed':
            raise ValueError('Both studies must pass their audits before comparison')
    output = current/'relaxation_comparison'
    output.mkdir(exist_ok=False)
    performance, forces, times, traces, payloads = [], [], [], {}, {}
    test_cases = None
    for method in ('joint', 'staged'):
        current_folder = current/'method_winners'/method
        with np.load(current_folder/'test_metrics.npz') as saved:
            cases, new_metrics = saved['cases'], saved['metrics']
        if test_cases is not None:
            np.testing.assert_array_equal(cases, test_cases)
        test_cases = cases
        reference_case = None
        reference_mechanics = None
        reference_duration = None
        for label, study in (('previous', previous), ('current', current)):
            folder = study/'method_winners'/method
            genome_path = folder/'best_genome.json'
            genome, unit = load_joint_genome(genome_path)
            physical, command = genome.decode(unit)
            cfg = genome.controller
            duration = cfg.data['simulation']['duration_s']
            if reference_mechanics is not None:
                np.testing.assert_array_equal(cfg.mechanics, reference_mechanics)
                assert duration == reference_duration
            reference_mechanics, reference_duration = cfg.mechanics, duration
            if label == 'previous':
                with PopulationEvaluator(cfg.mechanics, workers) as evaluator:
                    _, metrics = evaluate_joint(evaluator, genome, unit[None], cases, duration)
                metrics = metrics[0]
            else:
                metrics = new_metrics
            payloads[f'{label}_{method}'] = metrics
            performance.append(dict(study=label, method=method,
                successes=int(metrics[:, 5].sum()), cases=len(cases),
                falls=int(np.count_nonzero(metrics[:, 1] < duration - 1e-8)),
                mean_late_rmse_deg=float(np.rad2deg(metrics[:, 7]).mean()),
                genome_sha256=file_hash(genome_path)))
            for index in TIME_INDICES:
                field = cfg.data['parameters'][index]
                times.append(dict(study=label, method=method, index=index,
                    parameter=field['name'], value_ms=float(physical[index]),
                    minimum_ms=field['minimum'], maximum_ms=field['maximum']))
            with np.load(folder/'diagnostics/example/case_000.npz') as saved:
                trace, case = saved['trace'], saved['case']
                np.testing.assert_array_equal(saved['physical'], physical)
                np.testing.assert_array_equal(saved['currents'], genome.currents(command, case[None])[0])
            if reference_case is not None:
                np.testing.assert_array_equal(case, reference_case)
            reference_case = case
            if duration < 2:
                raise ValueError('An episode of at least 2 s is needed for late-window comparison')
            complete = bool(trace[-1, 0] >= duration - 1e-8)
            late = trace[trace[:, 0] >= duration - 2]
            for muscle, col in (('flexor', 3), ('extensor', 4)):
                values = late[:, col]
                mean = float(values.mean()) if complete else None
                std = float(values.std()) if complete else None
                forces.append(dict(study=label, method=method, muscle=muscle,
                    window_start_s=duration-2, window_end_s=duration,
                    complete_episode=complete, example_survival_s=float(trace[-1, 0]),
                    mean_N=mean, std_N=std,
                    peak_to_peak_N=float(np.ptp(values)) if complete else None,
                    coefficient_of_variation=std/abs(mean) if mean else None,
                    angle_std_deg=float(np.rad2deg(late[:, 1]).std()) if complete else None,
                    samples=len(late) if complete else 0))
            traces[label, method] = trace
    write_csv(output/'common_test.csv', performance)
    write_csv(output/'force_variation.csv', forces)
    write_csv(output/'time_constants.csv', times)
    np.savez_compressed(output/'common_test_metrics.npz', cases=test_cases, **payloads)
    configure_fonts(15)
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
    for column, method in enumerate(('joint', 'staged')):
        for label, color in (('previous', '#777777'), ('current', '#1565b0')):
            trace = traces[label, method]
            axes[0, column].plot(trace[::10, 0], np.rad2deg(trace[::10, 1]),
                                label=label, color=color, lw=1.3)
            complete = trace[-1, 0] >= reference_duration - 1e-8
            zoom = trace[trace[:, 0] >= reference_duration - .2] if complete else trace[:0]
            force_label = label if complete else f'{label}: fell at {trace[-1, 0]:.2f} s'
            # Keep every 0.1 ms sample: no smoothing or decimation in force panels.
            for row, channel in ((1, 3), (2, 4)):
                axes[row, column].plot(zoom[:, 0], zoom[:, channel],
                                       color=color, lw=1.1, label=force_label)
        axes[0, column].axhline(np.rad2deg(reference_case[0]), color='black', ls='--', lw=.8)
        axes[0, column].set(title=f'{method}: target +22.5°, start −30°, speed 0', ylabel='Angle (deg)')
        axes[1, column].set(ylabel='Flexor force (N)')
        axes[2, column].set(ylabel='Extensor force (N)')
        for ax in axes[:, column]:
            ax.set_xlabel('Time (s)'); ax.grid(alpha=.2); ax.legend()
    for extension in ('png', 'pdf'):
        fig.savefig(output/f'comparison.{extension}', dpi=180)
    plt.close(fig)
    result = dict(previous=str(previous), current=str(current),
                  selection='Previously frozen validation winners; no reselection',
                  test_bank_seed=cfg.data['simulation']['bank_seeds']['test'],
                  performance=performance, force_variation=forces, time_constants=times,
                  note='Bounds, GA seed and search budgets differ. This is not an isolated causal comparison.')
    write_json(output/'comparison.json', result)
    for suffix, ru in (('md', True), ('en.md', False)):
        text = ('# Сравнение после повторного обучения C\n\n[English](README.en.md)\n\n' if ru else
                '# C retraining comparison\n\n[Русский](README.md)\n\n')
        text += ('Четыре времени релаксации ограничены диапазоном 5,1–100 мс вместо 5–100 мс. '
                 'Предыдущие лучшие геномы уже удовлетворяли условию >5 мс. '
                 'Новый seed ГА и меньший набор запусков также изменены: результаты не выделяют эффект границы отдельно. '
                 'Старые и новые заранее выбранные по валидации решения проверены на одном новом тестовом банке; '
                 'повторного выбора по тесту нет.\n\n' if ru else
                 'All four relaxation times now range from 5.1 to 100 ms, previously 5 to 100 ms. '
                 'Previous winners already met the >5 ms condition. '
                 'The GA seed and search budget also differ; the bound effect is not isolated. '
                 'Previously frozen validation winners are evaluated on the same new test bank, with no test-based reselection.\n\n')
        text += '| Study | Method | Common test | Falls | Mean late RMSE (deg) |\n|---|---|---:|---:|---:|\n'
        for row in performance:
            text += f"| {row['study']} | {row['method']} | {row['successes']}/{row['cases']} | {row['falls']} | {row['mean_late_rmse_deg']:.6f} |\n"
        text += ('\n[Осциллограммы](comparison.png) · [PDF](comparison.pdf) · [Константы](time_constants.csv) · '
                 '[Разброс сил](force_variation.csv) · [Общий тест](common_test.csv)\n\n' if ru else
                 '\n[Oscillograms](comparison.png) · [PDF](comparison.pdf) · [Time constants](time_constants.csv) · '
                 '[Force variation](force_variation.csv) · [Common test](common_test.csv)\n\n')
        text += ('Силы показаны без сглаживания за последние 0,2 с. Статистика вычислена по всем отсчётам '
                 'последних 2 с: стандартное отклонение, размах и отношение стандартного отклонения к средней силе. '
                 'Пример одинаковый: цель +22,5°, старт −30°, скорость 0. '
                 'При падении пример не заменяется: поздние статистики отмечаются как отсутствующие. '
                 'В модели dF/dt=A·N·x−F/τ₁, поэтому τ₁ влияет также на установившуюся силу, а не только на её сглаживание.\n' if ru else
                 'Force panels retain every sample of the last 0.2 s. Statistics use every sample in the final 2 s: '
                 'standard deviation, peak-to-peak range and standard deviation divided by mean force. '
                 'Both examples target +22.5° from −30° with zero initial speed. '
                 'A fallen example is never replaced: its late statistics are marked as missing. '
                 'The model uses dF/dt=A·N·x−F/τ₁, so τ₁ also changes steady force, not only smoothing.\n')
        (output/f'README.{suffix}').write_text(text)
    return result


def main():
    """Compare two completed C sweeps without modifying their selected genomes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('previous', type=Path)
    parser.add_argument('current', type=Path)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    result = compare(args.previous, args.current, workers=args.workers)
    print(json.dumps(result['performance'], indent=2))


if __name__ == '__main__':
    main()
