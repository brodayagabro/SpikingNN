"""Re-evaluate saved winners after the tensile-torque correction without retraining."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from scripts.AfferentBalance2026.experiments.core.genome import load_genome
from scripts.AfferentBalance2026.experiments.simulation import (
    MODEL, TRACE_COLUMNS, METRIC_COLUMNS, balance_banks, case_grid, simulate)
from scripts.AfferentBalance2026.experiments.C_setpoint_balance.joint_study import (
    load_joint_genome, target_banks)
from scripts.AfferentBalance2026.experiments.D_trajectory.controller import load_solution
from scripts.AfferentBalance2026.experiments.D_trajectory.trajectory import (
    METRIC_COLUMNS as TRACKING_COLUMNS, replay, sine_reference, trajectory_banks)
from scripts.AfferentBalance2026.experiments.visualization import configure_fonts

EXPERIMENTS = REPO/'scripts/AfferentBalance2026/experiments'
TARGETS = (-45., -22.5, 0., 22.5, 45.)


def digest(path):
    """Hash a source or input file without changing its contents."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    """Write portable metadata and reject nonfinite JSON numbers."""
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def candidates(experiment):
    """Find published winners, including both retained C studies and all D study winners."""
    folder = EXPERIMENTS/experiment
    if experiment in ('A_balance', 'B_balance'):
        return [('current', folder/'results/best_genome.json')]
    if experiment == 'C_setpoint_balance':
        result = [('current', folder/'best_genome.json')]
        for study in sorted((folder/'results').glob('joint_sweep*')):
            for method in ('joint', 'staged'):
                path = study/'method_winners'/method/'best_genome.json'
                if path.is_file():
                    result.append((f'{study.name}_{method}', path))
        return result
    result = [('current', folder/'results/best_genome.json')]
    for study in sorted((folder/'results').iterdir()):
        if study.is_dir() and (study/'banks.json').is_file():
            for method in ('frozen', 'command_only', 'joint'):
                path = study/method/'best_genome.json'
                if path.is_file():
                    result.append((f'{study.name}_{method}', path))
    return result


def load_candidate(experiment, label, path):
    """Decode a genome using its saved mechanics, bounds and current law."""
    result = dict(label=label, source=str(path.relative_to(REPO)), source_sha256=digest(path))
    if experiment in ('A_balance', 'B_balance'):
        cfg, physical = load_genome(path)
        coefficients, tau, joint, command = np.zeros((2, 4)), 0., None, None
    elif experiment == 'C_setpoint_balance':
        joint, unit = load_joint_genome(path)
        physical, command = joint.decode(unit)
        cfg, coefficients, tau = joint.controller, command, 0.
        assert joint.kind == 'polynomial'
    else:
        cfg, physical, coefficients, tau, _ = load_solution(path)
        joint, command = None, None
    result.update(cfg=cfg, physical=physical, coefficients=coefficients, tau=tau,
                  joint=joint, command=command, path=path, aliases=[])
    # Identical copies are checked once per experiment and retain all source aliases.
    payload = dict(configuration=cfg.data, physical=physical.tolist(),
                   coefficients=coefficients.tolist(), current_tau_s=tau)
    result['identity'] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return result


def summarize(metrics, durations, final_angles):
    """Count completion and strict success; failed episodes never receive invented tail data."""
    survived = (np.isclose(metrics[:, 1], durations, rtol=0, atol=1e-9)
                & (metrics[:, 0] >= 0) & (np.abs(final_angles) <= np.pi/2))
    return dict(cases=len(metrics), survived=int(survived.sum()),
                successes=int(metrics[:, 5].sum()), nonfinite=int((metrics[:, 0] < 0).sum()),
                min_survival_s=float(metrics[:, 1].min()),
                median_survival_s=float(np.median(metrics[:, 1])),
                max_survival_s=float(metrics[:, 1].max()),
                mean_rmse_before_stop_deg=float(np.rad2deg(metrics[:, 2]).mean()))


def run_bank(item, experiment, name, cases, folder, *, specs=None):
    """Evaluate an unchanged winner and save metrics plus traces on grids and trajectory tests."""
    cfg, physical = item['cfg'], item['physical']
    tracking = experiment == 'D_trajectory'
    duration = 10. if tracking else float(cfg.data['simulation']['duration_s'])
    save_traces = name != 'test' or tracking
    # One full-resolution trajectory per bank supports independent signal inspection.
    metrics, traces, currents, final_angles, durations = [], {}, [], [], []
    if not tracking:
        applied = (item['joint'].currents(item['command'], cases) if item['joint'] is not None
                   else np.zeros((len(cases), 2)))
    for index, case in enumerate(cases):
        stride = 1 if index == 0 else (10 if save_traces else 0)
        if tracking:
            if specs is None:
                spec = dict(duration_s=duration, hold_s=0., initial_angle_deg=float(np.rad2deg(case[1])),
                            initial_speed_rad_s=float(case[2]))
                reference = np.zeros((round(duration/.0001)+1, 2))
                reference[:, 0] = case[0]
            else:
                spec = specs[index]
                reference = sine_reference(spec)
            result = replay(physical, item['coefficients'], cfg.mechanics, spec,
                            tau_s=item['tau'], stride=stride, reference=reference)
            values, trace = result['metrics'], result['trace']
            durations.append(spec['duration_s'])
            final_angles.append(float(trace[-1, 1]))
            if stride:
                traces[f'reference_{index:03d}'] = result['reference']
                traces[f'currents_{index:03d}'] = result['commands']
        else:
            values, trace = simulate(physical, case, cfg.mechanics, currents=applied[index],
                                     duration_s=duration, stride=stride)
            durations.append(duration)
            final_angles.append(float(values[8]))
            currents.append(applied[index])
        metrics.append(values)
        if stride:
            traces[f'trace_{index:03d}'] = trace
            traces[f'stride_{index:03d}'] = np.array(stride)
    metrics = np.asarray(metrics)
    columns = TRACKING_COLUMNS if tracking else METRIC_COLUMNS
    np.savez_compressed(folder/f'{name}_metrics.npz', cases=cases, metrics=metrics,
                        metric_columns=columns, durations_s=durations,
                        final_angles_rad=final_angles, currents=np.asarray(currents))
    np.savez_compressed(folder/f'{name}_traces.npz', **traces, trace_columns=TRACE_COLUMNS,
                        cases=cases, dt_ms=.1, model=MODEL)
    summary = summarize(metrics, np.asarray(durations), np.asarray(final_angles))
    summary['by_target'] = []
    if specs is None:
        for target in np.unique(cases[:, 0]):
            mask = cases[:, 0] == target
            summary['by_target'].append(dict(target_deg=float(np.rad2deg(target)),
                **summarize(metrics[mask], np.asarray(durations)[mask], np.asarray(final_angles)[mask])))
    if name == 'grid':
        targets = np.unique(cases[:, 0])
        figure, axes = plt.subplots(1, len(targets), figsize=(5*len(targets), 4.5),
                                    squeeze=False, constrained_layout=True)
        for axis, target in zip(axes[0], targets):
            for index in np.flatnonzero(cases[:, 0] == target):
                trace = traces[f'trace_{index:03d}']
                color = plt.cm.coolwarm((np.rad2deg(cases[index, 1])+50)/100)
                axis.plot(np.rad2deg(trace[:, 1]), trace[:, 2], color=color, alpha=.6, lw=.7)
                if abs(trace[-1, 1]) > np.pi/2:
                    axis.plot(np.rad2deg(trace[-1, 1]), trace[-1, 2], 'kx', ms=3)
            axis.plot(np.rad2deg(target), 0, 'k*', ms=10)
            axis.set(title=f'Target {np.rad2deg(target):g}°', xlabel='Angle (deg)', ylabel='Velocity (rad/s)')
            axis.grid(alpha=.2)
        figure.suptitle(f'{experiment}: {item["label"]} — corrected tensile torque')
        figure.savefig(folder/'phase_grid.png', dpi=130)
        figure.savefig(folder/'phase_grid.pdf')
        plt.close(figure)
    return summary


def main():
    """Write separate result directories, preserving every historical genome and metric."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', default='torque_sign_check_20261026')
    args = parser.parse_args()
    if Path(args.tag).name != args.tag or not args.tag:
        parser.error('--tag must be a directory name')
    assert MODEL == 'SpikingNN.Afferents/v3-tensile'
    configure_fonts(12)
    code_paths = [Path(__file__), REPO/'src/SpikingNN/core/Izh_net.py',
                  *(p for p in EXPERIMENTS.rglob('*.py')
                    if not {'results', '__pycache__', '.ipynb_checkpoints'}.intersection(p.parts))]
    hashes = {str(path.relative_to(REPO)): digest(path) for path in code_paths}
    provenance = dict(model=MODEL, dt_ms=.1, trained_again=False,
                      checked_at_utc=datetime.now(timezone.utc).isoformat(),
                      mechanics_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                      source_hashes=hashes, note='Frozen old genomes; no selection or optimization on these checks.')
    for experiment in ('A_balance', 'B_balance', 'C_setpoint_balance', 'D_trajectory'):
        unique = {}
        for label, path in candidates(experiment):
            item = load_candidate(experiment, label, path)
            alias = dict(label=label, source=item['source'], sha256=item['source_sha256'])
            if item['identity'] in unique:
                unique[item['identity']]['aliases'].append(alias)
            else:
                item['aliases'].append(alias)
                unique[item['identity']] = item
        output = EXPERIMENTS/experiment/'results'/args.tag
        output.mkdir(exist_ok=False)
        shutil.copy2(__file__, output/'verification_source.py')
        write_json(output/'provenance.json', provenance)
        summaries = []
        for item in unique.values():
            folder = output/item['label']; folder.mkdir()
            shutil.copy2(item['path'], folder/'evaluated_genome.json')
            targets = (0.,) if experiment in ('A_balance', 'B_balance') else TARGETS
            grid = np.concatenate([case_grid(np.linspace(-50, 50, 10), np.linspace(-1, 1, 10), t)
                                   for t in targets])
            small = np.concatenate([case_grid(np.array([-5., 0., 5.])+t, [0.], t) for t in targets])
            specs = None
            if experiment in ('A_balance', 'B_balance'):
                test = balance_banks(item['cfg'].data['simulation'])['test']
            elif experiment == 'C_setpoint_balance':
                test = target_banks(item['cfg'])['test']
            else:
                banks = trajectory_banks(item['cfg'].data['trajectory'])
                specs = banks['test']
                # Cross-check the regenerated protocol against any saved study bank.
                saved = next((p/'banks.json' for p in item['path'].parents
                              if (p/'banks.json').is_file()), None)
                if saved is not None:
                    assert json.loads(saved.read_text()) == banks
                write_json(folder/'trajectory_banks.json', banks)
                test = np.array([[sine_reference(s)[0, 0], np.deg2rad(s['initial_angle_deg']),
                                  s['initial_speed_rad_s'], -1., 0.] for s in specs])
            summary = dict(label=item['label'], source=item['source'], aliases=item['aliases'])
            for name, cases in (('test', test), ('grid', grid), ('near_target', small)):
                summary[name] = run_bank(item, experiment, name, cases, folder,
                                          specs=specs if name == 'test' else None)
                print(experiment, item['label'], name,
                      f'survived={summary[name]["survived"]}/{len(cases)}',
                      f'successes={summary[name]["successes"]}', flush=True)
            for alias in item['aliases']:
                assert digest(REPO/alias['source']) == alias['sha256']
            write_json(folder/'summary.json', summary)
            summaries.append(summary)
        for name, expected in hashes.items():
            assert digest(REPO/name) == expected, f'Calculation source changed during replay: {name}'
        write_json(output/'summary.json', dict(experiment=experiment, **provenance, genomes=summaries))
        for english in (False, True):
            title = '# Frozen-genome check after torque correction' if english else '# Проверка прежних геномов после исправления момента'
            intro = ('No retraining. Historical inputs/results were preserved. Each genome folder contains the evaluated JSON, test/grid/near-target metrics and traces, and phase_grid.png/pdf. The grid uses 10 angles from −50 to +50 degrees and 10 velocities from −1 to +1 rad/s; C/D use five targets. Failed trajectories stop at the ±90 degree boundary and are included. RMSE covers only the interval before stopping. One trace per bank is recorded at 0.1 ms; other saved traces use 1 ms samples.\n' if english else
                     'Без переобучения. Исторические геномы и результаты сохранены. В папках геномов находятся проверенный JSON, метрики и траектории test/grid/near_target, фазовые портреты phase_grid.png/pdf. Сетка: 10 углов −50…+50° × 10 скоростей −1…+1 рад/с; для C/D — пять целей. Падения до границы ±90° включены в результаты. RMSE учитывает только время до остановки. Одна траектория каждого набора записана с шагом 0,1 мс; остальные сохранённые траектории — 1 мс.\n')
            lines=[title, '', intro, '| Genome | Test survival | Grid survival | Near-target survival |', '|---|---:|---:|---:|']
            for s in summaries:
                lines.append('| '+s['label']+' | '+' | '.join(f'{s[b]["survived"]}/{s[b]["cases"]}' for b in ('test','grid','near_target'))+' |')
            (output/('README.en.md' if english else 'README.md')).write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
