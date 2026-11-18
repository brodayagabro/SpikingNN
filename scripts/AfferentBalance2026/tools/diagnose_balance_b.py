"""Compare historical and current balance controllers without retraining.

Historical sources are verified and run in a separate process. The main process
uses the active full-Afferents model. Weight clipping and joint weight/threshold
rescaling are separate interventions; neither replaces a saved sweep winner.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parents[1]
LEGACY_LINE = Path('small_nm_weights/sweep/mutation_0.100/population_100/seed_42/epochs_200')


def rescale_muscle_drive(physical, config):
    """Reduce w to the configured cap while preserving w/k for each muscle.

    With zero initial calcium, scaling w and k by the same positive factor
    scales calcium but preserves Hill activation and the force equations.
    All remaining coordinates are unchanged. Reject a transformation that
    violates any target-domain bound; never silently clip k or other genes.
    """
    candidate = np.asarray(physical, dtype=float).copy()
    for weight, threshold in ((16, 22), (23, 29)):
        cap = config.data['parameters'][weight]['maximum']
        factor = min(1., cap / candidate[weight])
        candidate[threshold] *= factor
        candidate[weight] = min(candidate[weight], cap)
    config.encode(candidate)
    return candidate


def frozen_export(root, output):
    """Verify frozen sources and export old banks, metrics and one dense trace."""
    runtime = root / 'factor_1/runtime'
    provenance = json.loads((runtime.parent/'runtime_provenance.json').read_text())
    for name, expected in provenance['runtime_sha256'].items():
        assert hashlib.sha256((runtime/name).read_bytes()).hexdigest() == expected, name
    assert provenance['torque_factor'] == 1. and provenance['dt_ms'] == .1
    sys.path.insert(0, str(runtime))
    package = 'scripts.SteppingReflexGA.experiments.core'
    cases_module = importlib.import_module(package+'.cases')
    dynamics = importlib.import_module(package+'.dynamics')
    genome = importlib.import_module(package+'.genome')
    assert Path(dynamics.__file__).resolve().is_relative_to(runtime)
    payload = json.loads((root/LEGACY_LINE/'best_genome.json').read_text())
    physical = np.array([gene['value'] for gene in payload['genes']])
    banks = cases_module.banks('small_nm_weights')
    with np.load(output/'phase_cases.npz') as data:
        phase = data['cases']
    metrics = {}
    for name, cases in [('old_test', banks['test']), ('phase_grid', phase)]:
        metrics[name] = np.array([dynamics.rollout(physical, case[1:], target_rad=case[0],
                                 duration_s=10., mode=0, stride=0, torque_factor=1.)[0]
                                 for case in cases])
    case = np.array([0., np.deg2rad(-30.), .7, -1., 0.])
    _, trace = dynamics.rollout(physical, case[1:], duration_s=.5, mode=0, stride=1, torque_factor=1.)
    np.savez_compressed(output/'frozen_reference.npz', **metrics, trace=trace,
                        trace_case=case, **{f'{name}_cases': cases for name, cases in banks.items()})
    domain = genome.ControllerDomain(small_weights=True)
    np.savez_compressed(output/'historical_domain.npz', lower=domain.decode(np.zeros(50)),
                        upper=domain.decode(np.ones(50)), initial=domain.decode(domain.initial()))
    print('Verified frozen sources:', len(provenance['runtime_sha256']), flush=True)


def run(root, output, notebook_genome, workers):
    """Measure paired interventions on fixed old and current condition banks."""
    sys.path.insert(0, str(REPO))
    from scripts.AfferentBalance2026.experiments.core.genome import (
        GenomeConfig, NAMES, file_hash, import_legacy, load_genome, write_json,
    )
    from scripts.AfferentBalance2026.experiments.core.run import PopulationEvaluator, save_replays, write_csv
    from scripts.AfferentBalance2026.experiments.simulation import (
        METRIC_COLUMNS, PARAMETERS, balance_banks, case_grid, simulate,
    )
    output.mkdir(parents=True, exist_ok=False)
    experiments = BASE/'experiments'
    a_config, a = load_genome(experiments/'A_balance/results/best_genome.json')
    config, b = load_genome(experiments/'B_balance/results/best_genome.json')
    assert np.array_equal(config.mechanics, a_config.mechanics)
    old_path = root/LEGACY_LINE/'best_genome.json'
    old = import_legacy(old_path, a_config)
    old_metadata = json.loads(old_path.read_text())['hyperparameters']
    assert old_metadata['dt_ms'] == .1
    old_mechanics = old_metadata['mechanics']
    for key, value in config.data['mechanics'].items():
        assert old_mechanics[key] == value
    assert old_mechanics['muscle_torque_factor'] == 1.
    assert [old_mechanics['a1_m'], old_mechanics['a2_m']] == old[46:48].tolist()
    old_run = json.loads((root/LEGACY_LINE/'run.json').read_text())
    assert list(old_run['afferent_parameters'].values()) == list(PARAMETERS)
    candidates = {'B_sweep': b, 'historical_B': old, 'A_original': a}
    if notebook_genome:
        notebook_config, candidates['B_notebook'] = load_genome(notebook_genome)
        assert notebook_config.signature == config.signature
    for name, source in [('historical_B', old), ('A', a)]:
        clipped = source.copy()
        clipped[[16, 23]] = np.minimum(clipped[[16, 23]], .02)
        config.encode(clipped)
        candidates[name+'_w_only'] = clipped
        candidates[name+'_rescaled'] = rescale_muscle_drive(source, config)
    labels = list(candidates)
    phase = case_grid()
    np.savez_compressed(output/'phase_cases.npz', cases=phase)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', NUMBA_CACHE_DIR=str(output/'jit_cache'),
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    subprocess.run([sys.executable, str(Path(__file__).resolve()), '--historical-root', str(root),
                    '--output', str(output), '--frozen-export'], env=env, check=True)
    banks = balance_banks(config.data['simulation'])
    with np.load(output/'frozen_reference.npz') as frozen:
        banks['old_test'] = frozen['test_cases']
    banks['phase_grid'] = phase
    rows, saved_metrics = [], {}
    with PopulationEvaluator(config.mechanics, workers=workers) as evaluator:
        for bank, cases in banks.items():
            scores, metrics = evaluator.evaluate(np.array(list(candidates.values())), cases)
            saved_metrics[bank] = metrics
            np.savez_compressed(output/f'{bank}_metrics.npz', labels=labels, cases=cases,
                                physical=np.array(list(candidates.values())), metrics=metrics,
                                metric_columns=METRIC_COLUMNS, fitness=scores)
            for label, score, values in zip(labels, scores, metrics):
                survived = values[:, 1] >= 10.
                row = dict(controller=label, bank=bank, successes=int(values[:, 5].sum()),
                           count=len(cases), falls=int((~survived).sum()), fitness=float(score),
                           late_rmse_deg=float(np.rad2deg(values[:, 7].mean())),
                           survivors_late_rmse_deg=(float(np.rad2deg(values[survived, 7].mean()))
                                                    if survived.any() else None))
                rows.append(row)
                print(label, bank, f"{row['successes']}/{len(cases)}", flush=True)
    write_csv(output/'comparison.csv', rows)
    with np.load(output/'frozen_reference.npz') as frozen:
        _, trace = simulate(old, frozen['trace_case'], config.mechanics, duration_s=.5)
        assert trace[1:, :17].shape == frozen['trace'].shape
        parity = dict(trace_max_abs=float(np.max(abs(trace[1:, :17]-frozen['trace']))))
        for bank in ('old_test', 'phase_grid'):
            live = saved_metrics[bank][labels.index('historical_B')]
            parity[bank+'_metrics_max_abs'] = float(np.max(abs(live-frozen[bank])))
            assert np.array_equal(live[:, 5], frozen[bank][:, 5])
        np.testing.assert_allclose(trace[1:, :17], frozen['trace'], rtol=1e-10, atol=1e-10)
    write_json(output/'model_parity.json', parity)
    parameters = [dict(index=i, parameter=name, **{label:float(p[i]) for label,p in candidates.items()})
                  for i,name in enumerate(NAMES)]
    write_csv(output/'parameters.csv', parameters)
    derived = []
    for label, p in candidates.items():
        for side, i in [('flexor',16),('extensor',23)]:
            derived.append(dict(controller=label, muscle=side, w=float(p[i]), k=float(p[i+6]),
                                w_over_k=float(p[i]/p[i+6]),
                                saturated_force_N=float(p[i+1]*p[i+2]*p[i+4]),
                                tau_c_ms=float(p[i+3]), tau_force_ms=float(p[i+4])))
    write_csv(output/'muscle_parameters.csv', derived)
    for label in ('historical_B_rescaled','A_rescaled'):
        config.save(output/(label+'_genome.json'), candidates[label], training_performed=False,
                    transformation='w and k scaled together; all other coordinates preserved',
                    source=('historical small_nm_weights' if label.startswith('historical') else
                            str(experiments/'A_balance/results/best_genome.json')))
    example_cases = case_grid([-50., 50.], [0.])
    for label in ('B_sweep', 'A_rescaled'):
        save_replays(output/label, config, candidates[label], example_cases, duration_s=10.)
    plot_comparison(output, labels, saved_metrics['phase_grid'])
    source_paths = [Path(__file__), experiments/'simulation.py', experiments/'core/genome.py',
                    experiments/'core/fitness.py', REPO/'src/SpikingNN/core/Izh_net.py',
                    old_path, root/LEGACY_LINE/'run.json',
                    experiments/'A_balance/results/best_genome.json',
                    experiments/'B_balance/results/best_genome.json']
    if notebook_genome:
        source_paths.append(notebook_genome)
    write_json(output/'provenance.json', dict(training_performed=False, historical_root=str(root),
               dt_ms=.1, duration_s=10., currents=[0.,0.], pd_used=False, muscle_torque_factor=1.,
               source_sha256={str(p):file_hash(p) for p in source_paths}))
    write_json(output/'summary.json', dict(comparison=rows, model_parity=parity,
               transformed_genomes=['historical_B_rescaled_genome.json','A_rescaled_genome.json'],
               original_results_replaced=False, training_performed=False))
    shutil.copy2(__file__, output/'diagnosis_source.py')
    print('Completed:', output, flush=True)


def plot_comparison(output, labels, metrics):
    """Plot binary success on the same 10-by-10 initial-condition grid."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scripts.AfferentBalance2026.experiments.visualization import configure_fonts
    configure_fonts(13)
    selected = [name for name in ('historical_B', 'B_sweep', 'B_notebook', 'A_rescaled') if name in labels]
    titles = {'historical_B':'Historical B (w < 0.1)', 'B_sweep':'Current B: sweep',
              'B_notebook':'Current B: notebook', 'A_rescaled':'A mapped into B (w ≤ 0.02)'}
    fig, axes = plt.subplots(1,len(selected),figsize=(5*len(selected),4.5),constrained_layout=True)
    for ax, label in zip(np.atleast_1d(axes), selected):
        success = metrics[labels.index(label), :, 5].reshape(10, 10)
        ax.imshow(success, origin='lower', cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        ticks = [0,3,6,9]
        ax.set(xticks=ticks, xticklabels=[f'{np.linspace(-50,50,10)[i]:.1f}' for i in ticks],
               yticks=ticks, yticklabels=[f'{np.linspace(-1,1,10)[i]:.2f}' for i in ticks],
               xlabel='Initial angle (deg)', ylabel='Initial speed (rad/s)',
               title=f"{titles[label]}\n{int(success.sum())}/100 successful")
    fig.suptitle('Green: settled by 3 s and stable to 10 s; red: failure')
    for extension in ('png','pdf'):
        fig.savefig(output/f'success_basins.{extension}', dpi=160)
    plt.close(fig)


def main():
    """Parse explicit data locations and run an isolated historical comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--historical-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--notebook-genome', type=Path)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--frozen-export', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.frozen_export:
        frozen_export(args.historical_root.resolve(), args.output.resolve())
    else:
        run(args.historical_root.resolve(), args.output.resolve(),
            args.notebook_genome.resolve() if args.notebook_genome else None, args.workers)


if __name__ == '__main__':
    main()
