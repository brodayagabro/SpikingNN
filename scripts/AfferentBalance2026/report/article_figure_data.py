"""Prepare and verify saved data for article_figures.ipynb.

All dynamics come from the selected study's immutable runtime. This module
chooses no new genome and performs no training. Plotting lives in the notebook.
Run ``--help`` for the optional compute/verify CLI.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from functools import lru_cache
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[3]
# Spawned workers inherit the legacy snapshot's regular ``scripts`` package.
# Import the live path resolver directly, without relying on that namespace.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from runtime_support import (
    BASE, activate_runtime, core_module, current_results, runtime_directory,
)

CASE_COLUMNS = ('target_rad', 'theta0_rad', 'omega0_rad_s', 'kick_time_s', 'kick_delta_omega')
TARGETS = (-45., -22.5, 0., 22.5, 45.)
SIGNAL_INDEX = 2


def read_json(path):
    """Read a UTF-8 protocol or selection record."""
    return json.loads(Path(path).read_text())


def write_json(path, value):
    """Atomically write finite JSON without replacing unrelated study files."""
    path = Path(path)
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def sha256(path):
    """Hash source or trajectory bytes without loading the whole file at once."""
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def case_banks(target_deg=0.):
    """Return a 100-case phase grid and ten zero-velocity angle traces.

    Angles are absolute deviations from upright, not target-relative errors.
    The even-sized velocity grid excludes zero, so the ten oscillograms are
    additional simulations. All episodes have no external velocity impulse.
    Columns follow CASE_COLUMNS; angles are radians and velocities rad/s.
    """
    if not np.isfinite(target_deg) or abs(target_deg) >= 90.:
        raise ValueError('Target must be finite and strictly inside ±90 degrees')
    angles = np.linspace(-50., 50., 10)
    velocities = np.linspace(-1., 1., 10)
    phase = np.array([[np.deg2rad(target_deg), np.deg2rad(angle), speed, -1., 0.]
                      for speed in velocities for angle in angles])
    zero = np.array([[np.deg2rad(target_deg), np.deg2rad(angle), 0., -1., 0.]
                     for angle in angles])
    return phase, zero


def winner_from_run(run, runtime):
    """Load the saved validation choice and verify its generation/source contract.

    Return physical parameters, optional polynomial coefficients and provenance.
    The selected index must maximize the original final validation scores.
    Replaying the new grid never changes this selection.
    """
    activate_runtime(runtime)
    final_path = run / 'final_population.npz'
    with np.load(final_path, allow_pickle=False) as final:
        index = int(final['selected_index'])
        assert index == int(np.argmax(final['validation_scores']))
        generation = int(final['generation'])
        snapshot = run / f'generations/generation_{generation:04d}.npz'
        saved = core_module('training_history').load_generation(snapshot)
        kind = str(saved['command_kind'])
        if kind == 'controller':
            np.testing.assert_array_equal(saved['physical_controllers'], final['physical_controllers'])
            coefficients = None
        elif kind == 'polynomial':
            np.testing.assert_array_equal(saved['command_coefficients'], final['command_coefficients'])
            np.testing.assert_array_equal(saved['physical_controllers'][index], final['fixed_controller'])
            coefficients = final['command_coefficients'][index].copy()
        else:
            raise ValueError(f'Unexpected command type: {kind}')
        controller = saved['physical_controllers'][index].copy()
        assert np.all(controller[46:48] == 0.)
        provenance = dict(run=str(run.relative_to(REPO)), selected_index=index,
                          generation=generation, command_kind=kind,
                          validation_fitness=float(final['validation_scores'][index]),
                          final_population_sha256=sha256(final_path),
                          generation_file=str(snapshot.relative_to(REPO)),
                          generation_sha256=sha256(snapshot))
    return controller, coefficients, provenance


def scenarios(root, runtime):
    """Collect three balance winners and the already selected five-target cubic."""
    result = []
    for experiment, title in [('equilibrium', 'Balance: ordinary weights'),
                              ('small_nm_weights', 'Balance: small neuromuscular weights')]:
        sweep = root / experiment / 'sweep'
        choice = read_json(sweep / 'analysis.json')['best_by_validation']
        controller, coefficients, provenance = winner_from_run(sweep / choice['directory'], runtime)
        result.append(dict(name=experiment, title=title, target_deg=0., controller=controller.tolist(),
                           coefficients=None, provenance=provenance))
    seed = read_json(root / 'A1/diagnostics/summary.json')['selected_seed']
    controller, coefficients, provenance = winner_from_run(root / f'A1/seed_{seed}', runtime)
    result.append(dict(name='balance_A1', title='Balance: A1 large-angle training', target_deg=0.,
                       controller=controller.tolist(), coefficients=None, provenance=provenance))
    study = root / 'angle_setpoint/polynomial'
    selected = read_json(study / 'selection.json')['overall']
    controller, coefficients, provenance = winner_from_run(study / selected, runtime)
    law = read_json(study / selected / 'current_law.json')
    np.testing.assert_array_equal(coefficients, law['coefficients'])
    np.testing.assert_array_equal(controller, law['physical_controller'])
    for target in TARGETS:
        slug = f'{target:+05.1f}'.replace('+', 'plus_').replace('-', 'minus_').replace('.', 'p')
        result.append(dict(name=f'setpoint_{slug}', title=f'Angle holding: target {target:+g}°',
                           target_deg=target, controller=controller.tolist(),
                           coefficients=coefficients.tolist(), provenance=provenance))
    return result


def initialize_worker(runtime):
    """Verify the original numerical runtime in each spawned worker process."""
    activate_runtime(Path(runtime))
    assert core_module('config').DT_MS == .1
    assert core_module('config').MECHANICS.muscle_torque_factor == 1.


def simulate_case(task):
    """Integrate and save one full 0.1 ms trajectory, checking all stored metrics.

    Each NPZ includes t=0, states, applied controller, original controller,
    command coefficients, metrics and exact initial conditions. No temporal
    decimation is applied to these scientific data.
    """
    directory, scenario, bank, trial, case = task
    case = np.asarray(case, dtype=float)
    controller = np.asarray(scenario['controller'])
    dynamics = core_module('dynamics')
    if scenario['coefficients'] is None:
        metrics, trace = core_module('wide_replay').full_replay(controller, case)
        applied = controller
        coefficients = np.empty((2, 0))
    else:
        coefficients = np.asarray(scenario['coefficients'])
        metrics, trace, applied = core_module('polynomial_study').full_command_trace(controller, coefficients, case)
    # Match the integrator's ms-to-s operation order exactly; multiplying by
    # a precomputed 0.0001 differs by one floating-point ULP at some samples.
    np.testing.assert_array_equal(trace[:, 0], np.arange(len(trace), dtype=float) * .1 / 1000.)
    np.testing.assert_array_equal(trace[0, 1:3], case[1:3])
    if metrics[0] != -1.:
        recomputed = dynamics.measure_trace(trace[1:], target_rad=case[0], initial_theta=case[1])
        np.testing.assert_allclose(metrics, recomputed, atol=1e-10, rtol=1e-10)
    termination = ('numerical_failure' if metrics[0] == -1. else
                   'angle_limit' if abs(trace[-1, 1]) > np.pi / 2 else 'duration')
    path = Path(directory) / scenario['name'] / bank / f'case_{trial:03d}.npz'
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, trace=trace, columns=np.array(dynamics.TRACE_COLUMNS),
                        case=case, case_columns=np.array(CASE_COLUMNS), metrics=metrics,
                        metric_columns=np.array(dynamics.METRIC_COLUMNS), physical_controller=applied,
                        fixed_controller=controller, coefficients=coefficients, dt_ms=.1,
                        duration_s=10., mode=0, muscle_torque_factor=1., termination=termination)
    temporary.replace(path)
    return dict(scenario=scenario['name'], bank=bank, trial=trial,
                file=str(path.relative_to(directory)), sha256=sha256(path), samples=len(trace),
                target_deg=float(scenario['target_deg']), theta0_deg=float(np.rad2deg(case[1])),
                omega0_rad_s=float(case[2]), success=bool(metrics[5]), termination=termination,
                end_time_s=float(trace[-1, 0]), late_rmse_deg=float(np.rad2deg(metrics[7])),
                metrics=metrics.tolist(), metrics_recomputed=bool(metrics[0] != -1.), bytes=path.stat().st_size)


def compute(root, output, workers):
    """Create a new study, run 880 prescribed cases and save a verified manifest."""
    runtime = runtime_directory(root)
    specs = scenarios(root, runtime)
    output.mkdir(parents=True, exist_ok=False)
    protocol = dict(created_at=datetime.now().astimezone().isoformat(), source_study=str(root),
                    runtime=str(runtime), runtime_provenance_sha256=sha256(runtime.parent / 'runtime_provenance.json'),
                    script_sha256=sha256(Path(__file__)), model=core_module('config').MODEL,
                    dt_ms=.1, duration_s=10., mode=0, pd_used=False,
                    mechanics=asdict(core_module('config').MECHANICS), angles_deg=np.linspace(-50., 50., 10).tolist(),
                    velocities_rad_s=np.linspace(-1., 1., 10).tolist(), phase_cases_per_scenario=100,
                    zero_velocity_cases_per_scenario=10, signal_zero_velocity_index=SIGNAL_INDEX,
                    signal_selection='Prespecified index 2: theta0=-27.7777777778 deg, omega0=0; same case for all panels C/D.',
                    kicks=False, seed_selection='Existing validation selections; no training or reselection.',
                    scenarios=specs)
    write_json(output / 'protocol.json', protocol)
    # Keep the exact orchestration source alongside the independently hashed
    # numerical runtime. Later plotting changes do not obscure what ran.
    (output / 'article_figures_source.py').write_bytes(Path(__file__).read_bytes())
    tasks = []
    for spec in specs:
        phase, zero = case_banks(spec['target_deg'])
        for bank, cases in [('phase_grid', phase), ('zero_velocity', zero)]:
            (output / spec['name'] / bank).mkdir(parents=True)
            for trial, case in enumerate(cases):
                tasks.append((str(output), spec, bank, trial, case.tolist()))
    records = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
                             initializer=initialize_worker, initargs=(str(runtime),)) as executor:
        futures = [executor.submit(simulate_case, task) for task in tasks]
        for future in as_completed(futures):
            records.append(future.result())
            if len(records) % 20 == 0 or len(records) == len(tasks):
                write_json(output / 'progress.json', dict(completed=len(records), total=len(tasks)))
                print(f'Completed {len(records)}/{len(tasks)}', flush=True)
    records.sort(key=lambda row: (row['scenario'], row['bank'], row['trial']))
    write_json(output / 'traces.json', records)
    import csv
    with (output / 'metrics.csv').open('w', newline='') as stream:
        fields = [key for key in records[0] if key != 'metrics']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in records)
    summary = []
    for spec in specs:
        row = dict(scenario=spec['name'], target_deg=spec['target_deg'])
        for bank in ('phase_grid', 'zero_velocity'):
            subset = [r for r in records if r['scenario'] == spec['name'] and r['bank'] == bank]
            row[bank] = dict(cases=len(subset), successes=sum(r['success'] for r in subset),
                             falls=sum(r['termination'] == 'angle_limit' for r in subset),
                             numerical_failures=sum(r['termination'] == 'numerical_failure' for r in subset))
        summary.append(row)
    write_json(output / 'summary.json', summary)
    write_json(output / 'verification.json', dict(status='passed', cases=len(records),
               full_resolution=True, dt_ms=.1, mode=0, independent_metric_reconstruction=True,
               numerical_failures=sum(r['termination'] == 'numerical_failure' for r in records),
               trace_bytes=sum(r['bytes'] for r in records), numerical_source_hashes_verified=True))
    return protocol


def display_indices(length, stride=10):
    """Decimate display only, always retaining the initial and terminal samples."""
    if length < 1 or stride < 1:
        raise ValueError('Positive length and stride required')
    return np.unique(np.r_[np.arange(0, length, stride), length - 1])


def load_trace(root, record):
    """Verify and read one saved trajectory before plotting or derived diagnostics."""
    path = root / record['file']
    if sha256(path) != record['sha256']:
        raise ValueError(f'Trajectory checksum mismatch: {path}')
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


@lru_cache(maxsize=2)
def figure_data(output, scenario_name, stride=10):
    """Read checked trajectories and derive signals without running simulations.

    Cache two display datasets, not all 880 full-resolution trajectories.
    A/B retain both endpoints; C/D retain every sample. Clear this cache after
    replacing source files. Rendering code and style belong to the notebook.
    """
    output = Path(output).resolve()
    protocol = read_json(output / 'protocol.json')
    spec = next(item for item in protocol['scenarios'] if item['name'] == scenario_name)
    records = read_json(output / 'traces.json')
    selected = [row for row in records if row['scenario'] == scenario_name]
    phase = sorted((r for r in selected if r['bank'] == 'phase_grid'), key=lambda r: r['trial'])
    zero = sorted((r for r in selected if r['bank'] == 'zero_velocity'), key=lambda r: r['trial'])
    assert len(phase) == protocol['phase_cases_per_scenario'] == 100
    assert len(zero) == protocol['zero_velocity_cases_per_scenario'] == 10
    signal_index = protocol['signal_zero_velocity_index']
    phase_lines, zero_lines = [], []
    example = None
    for bank, lines in ((phase, phase_lines), (zero, zero_lines)):
        for row in bank:
            data = load_trace(output, row)
            trace = data['trace']
            lines.append(trace[display_indices(len(trace), stride), :3].copy())
            if bank is zero and row['trial'] == signal_index:
                example = data
    if example is None:
        raise ValueError(f'Missing prescribed signal example: {signal_index}')
    activate_runtime(Path(protocol['runtime']))
    trace = example['trace']
    signals = core_module('wide_replay').nonlinear_signals(trace, example['physical_controller'])
    terms = signals['afferent_terms']
    derived = np.stack((terms[:, :, :4].sum(axis=2),
                        terms[:, :, 4:7].sum(axis=2), terms[:, :, 7]), axis=2)
    np.testing.assert_allclose(derived[1:].reshape(-1, 6), trace[1:, 11:17],
                               atol=1e-10, rtol=1e-10)
    assert np.all((signals['activation'] >= 0.) & (signals['activation'] <= 1.))
    return dict(spec=spec, protocol=protocol, phase=phase, zero=zero,
                phase_lines=phase_lines, zero_lines=zero_lines,
                example=example, signals=signals,
                example_record=next(row for row in zero if row['trial'] == signal_index))


def verify_library(output):
    """Independently replay the A1 and +22.5-degree examples through library classes.

    Compare complete trajectories and metrics against the saved Numba output.
    This checks the neuron, muscle and full Afferents update sequence without
    training or adding trajectories to the prescribed evaluation banks.
    """
    protocol = read_json(output / 'protocol.json')
    activate_runtime(Path(protocol['runtime']))
    dynamics = core_module('dynamics')
    records = read_json(output / 'traces.json')
    directory = output / 'library_reference'
    directory.mkdir(exist_ok=True)
    comparisons = []
    for scenario in ('balance_A1', 'setpoint_plus_22p5'):
        row = next(r for r in records if r['scenario'] == scenario
                   and r['bank'] == 'zero_velocity' and r['trial'] == SIGNAL_INDEX)
        saved = load_trace(output, row)
        case = saved['case']
        reference = dynamics.reference_trace(saved['physical_controller'], case[1:],
                                             target_rad=case[0], dt_ms=.1, mode=0)
        metrics = dynamics.measure_trace(reference, target_rad=case[0], initial_theta=case[1])
        np.testing.assert_allclose(metrics, saved['metrics'], atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(reference, saved['trace'][1:], atol=1e-6, rtol=1e-6)
        path = directory / f'{scenario}.npz'
        np.savez_compressed(path, trace=reference, metrics=metrics, case=case,
                            columns=saved['columns'], physical_controller=saved['physical_controller'])
        comparisons.append(dict(scenario=scenario, source=row['file'], samples=len(reference),
                                success_agrees=bool(metrics[5] == saved['metrics'][5]),
                                max_absolute_error_by_channel=dict(zip(dynamics.TRACE_COLUMNS,
                                    np.max(np.abs(reference - saved['trace'][1:]), axis=0).tolist())),
                                file=str(path.relative_to(output)), sha256=sha256(path)))
        print('Library reference verified:', scenario, flush=True)
    write_json(output / 'library_reference_verification.json', dict(
        status='passed', comparisons=comparisons, script_sha256=sha256(Path(__file__)),
        note='Original library classes; no Numba; first reference sample at t=0.1 ms.'))


def main():
    """Compute or verify trajectories; edit figures in article_figures.ipynb."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('compute', 'verify', 'all'))
    parser.add_argument('--root', type=Path, help='Completed unscaled study; defaults to the current pointer')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    output = args.output.resolve()
    if args.action in ('compute', 'all'):
        compute(args.root.resolve() if args.root else current_results(), output, args.workers)
    if args.action in ('verify', 'all'):
        verify_library(output)


if __name__ == '__main__':
    main()
