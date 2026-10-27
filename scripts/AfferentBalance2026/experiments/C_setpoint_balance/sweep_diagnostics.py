"""Publish validation-selected C method winners and their actual state trajectories.

Dense grids are post-selection diagnostics. They never choose a controller,
polynomial degree, mutation rate or representative example. All arrays retain
0.1 ms samples; figures alone may decimate smooth channels for readability.
"""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from scripts.AfferentBalance2026.experiments.core.genome import file_hash, write_json
from scripts.AfferentBalance2026.experiments.core.run import save_replays, source_hashes, write_csv
from scripts.AfferentBalance2026.experiments.simulation import case_grid
from scripts.AfferentBalance2026.experiments.visualization import (
    configure_fonts, plot_phase, plot_angles, plot_states,
)
from scripts.AfferentBalance2026.experiments.C_setpoint_balance.joint_study import (
    finish_solution, load_joint_genome,
)


def read_rows(path):
    """Read CSV records without losing their source-relative path labels."""
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def choose_method_winners(directory):
    """Freeze one winner per method using validation fitness alone."""
    rows = read_rows(Path(directory) / 'summary.csv')
    if not rows:
        raise ValueError('A completed nonempty sweep is required')
    winners = {}
    for method in sorted({row['method'] for row in rows}):
        candidates = [row for row in rows if row['method'] == method]
        winners[method] = max(candidates, key=lambda row: float(row['validation_fitness']))
    return winners


def save_figure(figure, path):
    """Save both a reviewable raster and an exportable PDF, then release memory."""
    import matplotlib.pyplot as plt
    for suffix in ('.png', '.pdf'):
        figure.savefig(Path(path).with_suffix(suffix), dpi=180)
    plt.close(figure)


def load_records(folder, stride=10):
    """Load a compact plotting view; leave full-resolution trajectory files intact."""
    records = []
    for row in read_rows(Path(folder) / 'metrics.csv'):
        with np.load(Path(folder) / row['file'], allow_pickle=False) as data:
            trace = data['trace']
            indices = np.unique(np.r_[np.arange(0, len(trace), stride), len(trace)-1])
            records.append(dict(case=data['case'], metrics=data['metrics'],
                                trace=trace[indices], currents=data['currents']))
    return records


def plot_example(folder):
    """Plot one predeclared case, including exact currents and resolved spike windows."""
    import matplotlib.pyplot as plt
    with np.load(folder / 'case_000.npz', allow_pickle=False) as data:
        trace, case, currents = data['trace'], data['case'], data['currents']
        record = dict(trace=trace, case=case, metrics=data['metrics'])
    for name, figure in plot_states(record).items():
        save_figure(figure, folder / name)
    t = trace[:, 0]
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    for i, current in enumerate(currents):
        ax.plot([t[0], t[-1]], [current, current], label=f'Neuron {i}: {current:.6g} nA')
    ax.set(xlabel='Time (s)', ylabel='Applied current (nA)', title='Constant target commands')
    ax.legend(); ax.grid(alpha=.2)
    save_figure(fig, folder / 'input_currents')
    # Raw receptor sums are visualization only, not Q-weighted synaptic currents.
    sums = np.column_stack((trace[:, 11:14].sum(axis=1), trace[:, 14:17].sum(axis=1)))
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    for i, label in enumerate(('Flexor', 'Extensor')):
        ax.plot(t[::10], sums[::10, i], label=label)
    ax.set(xlabel='Time (s)', ylabel='Ia + II + Ib (model units)', title='Raw afferent sums per muscle')
    ax.legend(); ax.grid(alpha=.2)
    save_figure(fig, folder / 'afferent_sums')
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), constrained_layout=True)
    for column, start in enumerate((0., max(0., t[-1]-.2))):
        mask = (t >= start) & (t <= start+.2)
        for i in range(2):
            axes[i, column].plot(t[mask], trace[mask, 5+i], lw=.7)
            axes[i, column].set(xlabel='Time (s)', ylabel=f'Neuron {i}: V (mV)')
            axes[i, column].grid(alpha=.2)
    save_figure(fig, folder / 'membrane_zoom')
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
    smooth = slice(None, None, 10)
    axes[0, 0].plot(t[smooth], np.rad2deg(trace[smooth, 1]))
    axes[0, 0].axhline(np.rad2deg(case[0]), color='black', ls='--', label='Target')
    axes[0, 0].set_ylabel('Limb angle (deg)'); axes[0, 0].legend()
    mask = t <= .2
    for i, label in enumerate(('Flexor', 'Extensor')):
        axes[0, 1].plot(t[mask], trace[mask, 5+i], label=f'Neuron {i}')
        axes[1, 0].plot(t[smooth], trace[smooth, 17+i], label=label)
        axes[1, 1].plot(t[smooth], trace[smooth, 3+i], label=label)
        axes[2, 0].plot(t[smooth], sums[smooth, i], label=label)
        axes[2, 1].plot([t[0], t[-1]], [currents[i]]*2, label=f'Neuron {i}: {currents[i]:.4g} nA')
    axes[0, 1].set_ylabel('Membrane V (mV), first 0.2 s')
    axes[1, 0].set_ylabel('Muscle activation x'); axes[1, 1].set_ylabel('Muscle force (N)')
    axes[2, 0].set_ylabel('Ia + II + Ib (model units)'); axes[2, 1].set_ylabel('Applied current (nA)')
    for i, ax in enumerate(axes.flat):
        ax.set_xlabel('Time (s)'); ax.grid(alpha=.2)
        if i: ax.legend()
    fig.suptitle(f'Target {np.rad2deg(case[0]):g} deg; initial angle {np.rad2deg(case[1]):g} deg, speed {case[2]:g} rad/s')
    save_figure(fig, folder / 'state_overview')
    write_json(folder / 'example.json', dict(target_deg=float(np.rad2deg(case[0])),
               theta0_deg=float(np.rad2deg(case[1])), omega0_rad_s=float(case[2]),
               currents_nA=currents.tolist(), selection='predeclared case, independent of outcome',
               afferent_sum='Ia + II + Ib, raw model units; not weighted neural feedback'))


def diagnose_winner(genome_path, output, *, grid_size=10, figures=True):
    """Save full phase-grid, zero-speed and fixed-example trajectories for one genome."""
    if not isinstance(grid_size, int) or grid_size < 2:
        raise ValueError('Use at least two points on each initial-condition axis')
    genome, unit = load_joint_genome(genome_path)
    if genome.kind != 'polynomial':
        raise ValueError('Diagnostics require a polynomial command')
    physical, law = genome.decode(unit); cfg = genome.controller
    duration = cfg.data['simulation']['duration_s']
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    hashes = source_hashes()
    angles, speeds = np.linspace(-50., 50., grid_size), np.linspace(-1., 1., grid_size)
    protocol = dict(status='running',genome_sha256=file_hash(genome_path),source_sha256=hashes,
                    grid_size=grid_size,angles_deg=angles.tolist(),speeds_rad_s=speeds.tolist(),
                    targets_deg=genome.targets.tolist(),dt_ms=.1,duration_s=duration,
                    selection_used_grid=False,example=dict(target_deg=22.5,theta0_deg=-30.,omega0_rad_s=0.))
    write_json(output / 'protocol.json', protocol)
    summaries, commands = [], []
    if figures:
        import matplotlib.pyplot as plt
        configure_fonts(14)
        nrows = int(np.ceil(len(genome.targets)/3))
        multi, axes = plt.subplots(nrows, 3, figsize=(17, 5*nrows), squeeze=False, constrained_layout=True)
    for index, target in enumerate(genome.targets):
        folder = output / f'target_{target:+06.2f}'; folder.mkdir()
        cases = case_grid(angles, speeds, target)
        currents = genome.currents(law, cases)
        rows = save_replays(folder / 'grid', cfg, physical, cases, duration_s=duration, currents=currents)
        zero = case_grid(angles, [0.], target)
        zero_rows = save_replays(folder / 'zero_speed', cfg, physical, zero,
                                 duration_s=duration, currents=genome.currents(law, zero))
        summaries.append(dict(target_deg=float(target),grid_cases=len(rows),
                          grid_successes=int(sum(r['success'] for r in rows)),
                          grid_late_rmse_deg=float(np.rad2deg([r['late_rmse_rad'] for r in rows]).mean()),
                          zero_speed_cases=len(zero_rows),zero_speed_successes=int(sum(r['success'] for r in zero_rows))))
        commands.append(dict(target_deg=float(target),neuron_0_nA=float(currents[0,0]),neuron_1_nA=float(currents[0,1])))
        write_csv(output / 'grid_summary.csv', summaries)
        print(json.dumps(dict(target_deg=float(target),**{k:v for k,v in summaries[-1].items() if k!='target_deg'})),flush=True)
        if figures:
            records = load_records(folder / 'grid')
            figure = plot_phase(records)
            figure.axes[0].set_title(f'Target {target:g} deg: {summaries[-1]["grid_successes"]}/{len(rows)} successes')
            for line in figure.axes[0].lines: line.set_rasterized(True)
            save_figure(figure, folder / 'phase_grid')
            save_figure(plot_angles(load_records(folder / 'zero_speed')), folder / 'angles_zero_speed')
            ax = axes.flat[index]
            for rec in records:
                tr, case = rec['trace'], rec['case']
                color = plt.cm.coolwarm((np.rad2deg(case[1])+50)/100)
                ax.plot(np.rad2deg(tr[:,1]),tr[:,2],color=color,lw=.6,alpha=.7,rasterized=True)
                ax.plot(np.rad2deg(case[1]),case[2],'.',color=color,ms=2)
                if abs(tr[-1,1])>np.pi/2: ax.plot(np.rad2deg(tr[-1,1]),tr[-1,2],'kx',ms=3)
            ax.plot(target,0,'k*',ms=10)
            ax.set(title=f'Target {target:g} deg: {summaries[-1]["grid_successes"]}/{len(rows)}',
                   xlabel='Angle from upright (deg)',ylabel='Speed (rad/s)'); ax.grid(alpha=.2)
    if figures:
        for ax in list(axes.flat)[len(genome.targets):]: ax.axis('off')
        multi.suptitle(f'{grid_size} × {grid_size} initial states per target; stars: desired equilibria')
        save_figure(multi, output / 'phase_all_targets')
    write_csv(output / 'input_currents.csv', commands)
    example_cases = case_grid([-30.], [0.], 22.5)
    save_replays(output / 'example',cfg,physical,example_cases,duration_s=duration,
                 currents=genome.currents(law,example_cases))
    if figures: plot_example(output / 'example')
    if source_hashes()!=hashes:
        raise RuntimeError('Calculation sources changed while saving diagnostics')
    result = dict(status='completed',grid_cases=sum(r['grid_cases'] for r in summaries),
                  grid_successes=sum(r['grid_successes'] for r in summaries),
                  zero_speed_cases=sum(r['zero_speed_cases'] for r in summaries),
                  zero_speed_successes=sum(r['zero_speed_successes'] for r in summaries),
                  full_traces=sum(r['grid_cases']+r['zero_speed_cases'] for r in summaries)+1,
                  genome_sha256=file_hash(genome_path))
    write_json(output / 'result.json', result)
    write_json(output / 'protocol.json', {**protocol,'status':'completed'})
    return result


def publish_methods(directory, *, grid_size=10, workers=1, figures=True):
    """Test and diagnose each method's validation winner, without reselection on tests."""
    directory = Path(directory).resolve()
    completed = json.loads((directory / 'completion.json').read_text())
    if completed['status'] != 'completed': raise ValueError('Finish the sweep before diagnostics')
    winners = choose_method_winners(directory)
    output = directory / 'method_winners'; output.mkdir(exist_ok=False)
    write_json(output / 'selection.json', dict(selection='validation only',winners=winners))
    overall = json.loads((directory / 'result.json').read_text())
    rows = []
    for method, selected in winners.items():
        source = directory / selected['directory'] / 'polynomial'
        destination = output / method
        genome, _ = load_joint_genome(source / 'best_genome.json')
        if selected['directory'] == overall['directory']:
            # Reuse the already evaluated overall winner instead of adding another test.
            shutil.copytree(directory / 'selected', destination)
            test = json.loads((destination / 'result.json').read_text())
        else:
            test = finish_solution(genome.controller, source, destination, workers=workers)
        write_json(destination / 'selection.json', selected)
        diagnostics = diagnose_winner(destination/'best_genome.json',destination/'diagnostics',
                                       grid_size=grid_size,figures=figures)
        rows.append(dict(method=method,degree=int(selected['degree']),population=int(selected['population']),
                         mutation_probability=float(selected['mutation_probability']),seed=int(selected['seed']),
                         generation=int(selected['generation']),validation_fitness=float(selected['validation_fitness']),
                         test_successes=round(test['test_success']*test['test_cases']),test_cases=test['test_cases'],
                         test_late_rmse_deg=test['test_late_rmse_deg'],grid_successes=diagnostics['grid_successes'],
                         grid_cases=diagnostics['grid_cases'],directory=str(destination.relative_to(directory))))
        write_csv(output / 'summary.csv', rows)
    write_json(output / 'completion.json',dict(status='completed',methods=len(rows),selection='validation only'))
    return rows


def main():
    """Expose repeatable method-winner publication for a completed C sweep."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--grid-size', type=int, default=10)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(publish_methods(args.directory,grid_size=args.grid_size,workers=args.workers),indent=2))


if __name__ == '__main__':
    main()
