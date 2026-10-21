"""Audit completed ABC studies and save selected-genome grids beside each experiment.

This command never runs the balance GA. A/B selection uses validation only.
C replays the selected A controller with configured currents and does not train.
"""
import argparse
import csv
import json
import platform
import importlib.metadata
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parents[1]))
from scripts.AfferentBalance2026.experiments.core.fitness import fitness
from scripts.AfferentBalance2026.experiments.core.genome import (
    GenomeConfig, SCHEMA, file_hash, load_genome, write_json,
)
from scripts.AfferentBalance2026.experiments.core.run import (
    PopulationEvaluator, save_replays, source_hashes,
)
from scripts.AfferentBalance2026.experiments.simulation import case_grid, METRIC_COLUMNS
from scripts.AfferentBalance2026.experiments.visualization import (
    configure_fonts, plot_phase, plot_angles, plot_states,
)
from scripts.AfferentBalance2026.experiments.C_setpoint_balance.run_experiment import run as replay_setpoints

EXPERIMENTS = BASE / 'experiments'


def read_json(path):
    """Read a local JSON record without changing its recorded provenance."""
    return json.loads(Path(path).read_text())


def save_figure(figure, path):
    """Export a plot with readable serif labels, then release its memory."""
    for extension in ('png', 'pdf'):
        figure.savefig(Path(path).with_suffix('.'+extension), dpi=180)
    plt.close(figure)


def multipanel(phase, zero, example, target):
    """Combine real phase trajectories, angle responses and one full signal example."""
    fig, axes = plt.subplots(2, 2, figsize=(13,9), constrained_layout=True)
    phase_ax, angle_ax, muscle_ax, afferent_ax = axes.ravel()
    for record in phase:
        trace = record['trace']
        phase_ax.plot(np.rad2deg(trace[:,1]),trace[:,2],alpha=.35,linewidth=.65)
        phase_ax.scatter(np.rad2deg(trace[0,1]),trace[0,2],s=6,color='grey')
        if abs(trace[-1,1])>np.pi/2:
            phase_ax.scatter(np.rad2deg(trace[-1,1]),trace[-1,2],s=15,color='red',marker='x')
    phase_ax.scatter([target],[0],s=120,marker='*',color='black',zorder=5)
    phase_ax.set(xlabel='Angle from vertical (deg)',ylabel='Speed (rad/s)',title='A. Phase trajectories (100 starts)')
    for record in zero:
        trace = record['trace']
        angle_ax.plot(trace[:,0],np.rad2deg(trace[:,1]),linewidth=1,
                      label=f"{np.rad2deg(record['case'][1]):.1f}°")
    angle_ax.axhline(target,color='black',linestyle='--',linewidth=1)
    angle_ax.set(xlabel='Time (s)',ylabel='Angle (deg)',title='B. Zero initial speed (10 starts)')
    angle_ax.legend(ncol=2,fontsize=9)
    trace = example['trace']
    for index,label in ((17,'Flexor'),(18,'Extensor')):
        muscle_ax.plot(trace[:,0],trace[:,index],label=label,linewidth=.8)
    muscle_ax.set(xlabel='Time (s)',ylabel='Muscle activation x',title='C. Muscle activity: one example')
    muscle_ax.legend(fontsize=11)
    for index,label in enumerate(('Ia flexor','II flexor','Ib flexor','Ia extensor','II extensor','Ib extensor'),11):
        afferent_ax.plot(trace[:,0],trace[:,index],label=label,linewidth=.8)
    afferent_ax.set(xlabel='Time (s)',ylabel='Afferent response (model units)',title='D. Afferent activity: same example')
    afferent_ax.legend(ncol=2,fontsize=9)
    for ax in axes.ravel():
        ax.grid(alpha=.2)
    fig.suptitle(f"Target {target:+g}°; signal example: initial angle={np.rad2deg(example['case'][1]):.1f}°, initial speed=0 rad/s",fontsize=15)
    return fig


def plot_saved_grid(directory, cases_per_target=110):
    """Plot recorded grid trajectories and full state groups for one zero-speed case."""
    directory = Path(directory)
    files = sorted((directory/'traces').glob('case_*.npz'))
    configure_fonts(14)
    summaries = []
    for first in range(0, len(files), cases_per_target):
        records = []
        for path in files[first:first+cases_per_target]:
            with np.load(path, allow_pickle=False) as saved:
                trace = saved['trace']
                indices = np.unique(np.r_[np.arange(0, len(trace), 25), len(trace)-1])
                records.append(dict(case=saved['case'].copy(), trace=trace[indices].copy(),
                                    metrics=saved['metrics'].copy(), currents=saved['currents'].copy(), file=path))
        target = float(np.rad2deg(records[0]['case'][0]))
        folder = directory/f'target_{target:+06.2f}'
        folder.mkdir(exist_ok=True)
        phase = [r for r in records if r['case'][2] != 0]
        zero = [r for r in records if r['case'][2] == 0]
        save_figure(plot_phase(phase), folder/'phase')
        save_figure(plot_angles(zero), folder/'angles')
        example = min(zero, key=lambda r: abs(np.rad2deg(r['case'][1])+30))
        # Keep the single signal example at full resolution, including every spike.
        with np.load(example['file'], allow_pickle=False) as saved:
            example = {**example, 'trace': saved['trace'].copy()}
        save_figure(multipanel(phase, zero, example, target), folder/'multipanel')
        for name, figure in plot_states(example).items():
            save_figure(figure, folder/name)
        metrics = np.array([r['metrics'] for r in phase])
        summaries.append(dict(target_deg=target, phase_cases=len(phase), zero_speed_cases=len(zero),
                              success_count=int(metrics[:,5].sum()), success_fraction=float(metrics[:,5].mean()),
                              late_rmse_deg=float(np.rad2deg(metrics[:,7]).mean()),
                              example_angle_deg=float(np.rad2deg(example['case'][1]))))
    write_json(directory/'grid_summary.json', summaries)
    return summaries


def audit_sweep(root):
    """Verify every generation, physical bounds, validation ranking and source hashes."""
    cfg = GenomeConfig(root/'genome_cfg.json')
    rows = []
    generations = 0
    for line in sorted(root.glob('population_*/mutation_*/seed_*')):
        run = read_json(line/'run.json')
        if run['status'] != 'completed' or run['source_sha256'] != source_hashes():
            raise ValueError(f'Unfinished or changed-source run: {line}')
        files = sorted((line/'generations').glob('generation_*.npz'))
        if len(files) != run['generations']:
            raise ValueError(f'Missing generations: {line}')
        with np.load(line/'cases.npz') as saved:
            cases = {name: saved[name] for name in ('training','validation','test')}
        for number, path in enumerate(files, 1):
            with np.load(path, allow_pickle=False) as saved:
                assert str(saved['schema']) == SCHEMA and int(saved['generation']) == number
                assert saved['physical'].shape == (run['population'], 52)
                for physical, unit in zip(saved['physical'], saved['normalized']):
                    np.testing.assert_allclose(cfg.decode(unit), physical, atol=1e-12)
                    cfg.encode(physical)
                np.testing.assert_allclose(saved['training_scores'],
                                           [fitness(m) for m in saved['training_metrics']], atol=1e-12)
                assert saved['training_metrics'].shape[1] == len(cases['training'])
            generations += 1
        for entry in read_json(line/'checkpoints.json'):
            folder = line/entry['directory']
            with np.load(folder/'final_population.npz') as saved:
                selected = int(saved['selected_index'])
                assert selected == int(np.argmax(saved['validation_scores']))
                np.testing.assert_allclose(saved['validation_scores'],
                                           [fitness(m) for m in saved['validation_metrics']], atol=1e-12)
                _, best = load_genome(folder/'best_genome.json')
                np.testing.assert_array_equal(best, saved['physical'][selected])
            rows.append({**entry, 'directory':str(folder.relative_to(root))})
    winner = max(rows, key=lambda row: row['validation_fitness'])
    folder = root/winner['directory']
    cfg, physical = load_genome(folder/'best_genome.json')
    with np.load(root/'cases.npz') as saved:
        test_cases = saved['test']
    with PopulationEvaluator(cfg.mechanics, workers=1) as evaluator:
        scores, metrics = evaluator.evaluate(physical[None,:], test_cases, cfg.data['simulation']['duration_s'])
    np.testing.assert_allclose(scores[0], winner['test_fitness'], atol=1e-12)
    np.testing.assert_allclose(metrics[0,:,5].mean(), winner['test_success'], atol=1e-12)
    audit = dict(status='passed', generations=generations, checkpoints=len(rows),
                 test_used_for_selection=False, selected_test_recomputed=True,
                 source_hashes_verified=True, analysis_source_sha256=file_hash(__file__))
    return cfg, physical, rows, winner, audit


def balance(name, sweep_name):
    """Select a completed A/B study winner and save a broad independent replay grid."""
    results = EXPERIMENTS/name/'results'
    root = results/sweep_name
    if not (root/'completion.json').is_file():
        raise ValueError(f'Sweep has not completed: {root}')
    cfg, physical, rows, winner, audit = audit_sweep(root)
    cfg.save(results/'best_genome.json', physical, selection='maximum validation fitness across sweep checkpoints',
             source=f'{sweep_name}/{winner["directory"]}/best_genome.json',
             generation=winner['generation'], population=winner['population'],
             mutation_probability=winner['mutation_probability'], seed=winner['seed'])
    write_json(results/'audit.json', audit)
    write_json(results/'environment.json', dict(python=sys.version, executable=sys.executable,
               platform=platform.platform(), captured_at='post-training audit',
               packages={name:importlib.metadata.version(name)
                         for name in ('numpy','numba','matplotlib')}))
    baseline, baseline_arrays = {}, {}
    with np.load(root/'cases.npz') as banks, PopulationEvaluator(cfg.mechanics, workers=1) as evaluator:
        for bank in ('training','validation','test'):
            scores, metrics = evaluator.evaluate(cfg.initial_physical[None,:], banks[bank],
                                                  cfg.data['simulation']['duration_s'])
            baseline[bank] = dict(fitness=float(scores[0]), success_fraction=float(metrics[0,:,5].mean()),
                                  late_rmse_deg=float(np.rad2deg(metrics[0,:,7]).mean()))
            baseline_arrays[bank] = metrics[0]
    write_json(results/'initial_baseline.json', baseline)
    np.savez_compressed(results/'initial_baseline_metrics.npz', **baseline_arrays,
                        physical=cfg.initial_physical, metric_columns=np.array(METRIC_COLUMNS))
    configure_fonts(13)
    fig, axes = plt.subplots(1, 3, figsize=(15,4), constrained_layout=True)
    for ax, probability in zip(axes, (.1,.3,.6)):
        for line in sorted(root.glob(f'population_*/mutation_{probability:.8g}/seed_*')):
            with (line/'history.csv').open() as stream:
                history = list(csv.DictReader(stream))
            size = read_json(line/'run.json')['population']
            ax.plot([int(r['generation']) for r in history],
                    [float(r['best_fitness']) for r in history],label=f'N={size}')
        ax.set(xlabel='Generation',ylabel='Training fitness',title=f'Mutation probability = {probability}')
        ax.grid(alpha=.2)
        if ax.lines:
            ax.legend()
    save_figure(fig, results/'convergence')
    diagnostic = results/'diagnostics_20261021'
    diagnostic.mkdir(exist_ok=False)
    angles, speeds = np.linspace(-50,50,10), np.r_[np.linspace(-1,1,10),0.]
    cases = case_grid(angles, speeds)
    print(name,'audit passed; recording',len(cases),'grid trajectories',flush=True)
    cfg.save(diagnostic/'genome.json',physical)
    write_json(diagnostic/'genome_cfg.json',cfg.data)
    save_replays(diagnostic/'traces',cfg,physical,cases,duration_s=cfg.data['simulation']['duration_s'])
    grids = plot_saved_grid(diagnostic)
    summary = dict(experiment=name, sweep=sweep_name, selected=winner, selection='validation only',
                   training_condition_count=cfg.data['simulation']['training_count'],
                   initial_conditions=cfg.data['simulation']['initial_conditions'],
                   grid=grids, audit=audit, initial_baseline=baseline, scientific_training_completed=True)
    write_json(results/'summary.json',summary)
    write_json(results/'current.json',dict(sweep=sweep_name,genome='best_genome.json',diagnostics=diagnostic.name))
    shutil.copy2(__file__,results/'analysis_source.py')
    write_summary(results,summary)
    print(name,'analysis completed',json.dumps(winner),flush=True)
    return summary


def write_summary(results, summary):
    """Write parallel Russian/English result indexes with actual measured outcomes."""
    w=summary['selected']; grid=summary['grid'][0]
    body=(f"Population: **{w['population']}**, mutation probability: **{w['mutation_probability']}**, "
          f"generation: **{w['generation']}**, seed: **{w['seed']}**.\n\n"
          f"Validation fitness: **{w['validation_fitness']:.6f}**; independent test success: "
          f"**{w['test_success']:.1%}**; test late RMSE: **{w['test_late_rmse_deg']:.3f}°**.\n\n"
          f"Separate 100-case phase grid: **{grid['success_count']}/100** successes.\n\n")
    links=(f"- [Full sweep]({summary['sweep']}/summary.csv): 60 evaluated checkpoints from 12 continuous GA runs.\n"
           f"- [Initial conditions]({summary['sweep']}/initial_conditions.png).\n"
           "- [Convergence](convergence.png).\n- [Selected genome](best_genome.json).\n"
           "- [Generation audit](audit.json).\n- [Untrained baseline](initial_baseline.json).\n- [Grid metrics](diagnostics_20261021/grid_summary.json).\n"
           "- `diagnostics_20261021/traces/`: 110 full 0.1 ms trajectories; 100 phase-grid cases and 10 zero-speed cases.\n"
           "- `diagnostics_20261021/target_+00.00/`: phase, angles, neurons, muscles and afferents.\n"
           f"- `{summary['sweep']}/heatmap_*.png` and `genome_rmse.npz`: fitness/test maps and normalized 50-gene RMSE.\n")
    en=f"# {summary['experiment']}: computed results, 2026-10-21\n\n"+body+links+"\nSelection used validation, never test or the diagnostic grid. One seed was used; these results do not establish global stability or statistical robustness. Early falls remain in the saved trajectories.\n"
    ru=(f"# {summary['experiment']}: результаты расчёта 21 октября 2026\n\n"
        f"Выбрана популяция **{w['population']}**, вероятность мутации **{w['mutation_probability']}**, "
        f"эпоха **{w['generation']}**, seed **{w['seed']}**.\n\n"
        f"Фитнесс валидации **{w['validation_fitness']:.6f}**. Успех на независимом тесте "
        f"**{w['test_success']:.1%}**, поздняя RMSE **{w['test_late_rmse_deg']:.3f}°**. "
        f"На отдельной фазовой сетке успех **{grid['success_count']}/100**.\n\n"
        f"[Сводка 60 контрольных точек]({summary['sweep']}/summary.csv), [сходимость](convergence.png), "
        "[лучший геном](best_genome.json), [аудит всех поколений](audit.json), "
        "[исходный необученный геном](initial_baseline.json).\n\n"
        "Полные траектории: `diagnostics_20261021/traces/` (100 условий фазовой сетки и 10 при нулевой скорости). "
        "Графики: `diagnostics_20261021/target_+00.00/`. Тепловые карты и RMSE геномов находятся в папке sweep. "
        "Победитель выбирался только по валидации; тест и диагностическая сетка не использованы для отбора. "
        "Один seed не доказывает статистическую устойчивость или глобальную стабилизацию. Падения сохранены.\n")
    (results/'README.en.md').write_text(en);(results/'README.md').write_text(ru)


def setpoint(sweep_name):
    """Replay the new A winner at five targets with the configured, unfitted currents."""
    a_results=EXPERIMENTS/'A_balance/results'
    if not (a_results/'summary.json').is_file():
        raise ValueError('Complete the A sweep analysis before replaying C')
    cfg=GenomeConfig(EXPERIMENTS/'C_setpoint_balance/genome_cfg.json')
    results=EXPERIMENTS/'C_setpoint_balance/results'
    output=results/'replay_20261021'
    shutil.copy2(a_results/'environment.json',results/'environment.json')
    result=replay_setpoints(cfg,output,genome=a_results/'best_genome.json',
                           targets=cfg.data['setpoint']['targets_deg'],angles=np.linspace(-50,50,10),
                           speeds=np.r_[np.linspace(-1,1,10),0.],currents=cfg.data['setpoint']['currents'])
    grids=plot_saved_grid(output)
    shutil.copy2(output/'best_genome.json',results/'best_genome.json')
    shutil.copy2(__file__,results/'analysis_source.py')
    write_json(results/'summary.json',dict(**result,grid=grids,
               controller_selection='A sweep validation winner',current_law_fitted=False))
    write_json(results/'current.json',dict(replay=output.name,genome='best_genome.json'))
    for language in ('ru','en'):
        text = ('# C: проверка заданных углов без обучения токов\n\n'
                'Использован лучший по валидации геном новой серии A. Токи взяты из конфига; '
                'подбор закона токов не выполнялся. При нулевых токах целевой угол влияет только на оценку, '
                'а не на динамику: этот расчёт является контрольной проверкой, не обученным регулятором заданного угла.\n\n'
                if language=='ru' else '# C: target-angle control check without current fitting\n\n'
                'The controller is the validation winner of the new A study. Currents come from the configuration; '
                'no current law was fitted. With zero currents, target angle changes scoring only, '
                'not dynamics. This is a control check, not a trained setpoint controller.\n\n')
        text+='| Target (deg) | Successful phase-grid cases | Late RMSE (deg) |\n|---|---|---|\n'
        for row in grids:
            text+=f"| {row['target_deg']} | {row['success_count']}/100 | {row['late_rmse_deg']:.3f} |\n"
        text+='\n[Summary](summary.json) · [Full replay](replay_20261021/result.json).\n\n`replay_20261021/traces/`: 550 full trajectories. `replay_20261021/target_*/`: phase portraits, angle traces and all states for one example per target.\n'
        (results/('README.md' if language=='ru' else 'README.en.md')).write_text(text)
    print('C analysis completed',json.dumps(grids),flush=True)


def compare(sweep_name):
    """Compare matched A/B budgets and genome distances in one shared physical domain."""
    records = {}
    for name in ('A_balance', 'B_balance'):
        root = EXPERIMENTS/name/'results'/sweep_name
        with (root/'summary.csv').open() as stream:
            records[name] = {tuple(row[k] for k in ('population','mutation_probability','generation','seed')): row
                             for row in csv.DictReader(stream)}
    if records['A_balance'].keys() != records['B_balance'].keys():
        raise ValueError('A/B sweeps do not contain the same experimental budgets')
    common = GenomeConfig(EXPERIMENTS/'A_balance/genome_cfg.json')
    rows = []
    for key, a in records['A_balance'].items():
        b = records['B_balance'][key]
        units = []
        for name, row in (('A_balance', a), ('B_balance', b)):
            _, physical = load_genome(EXPERIMENTS/name/'results'/sweep_name/row['directory']/'best_genome.json')
            units.append(common.encode(physical))
        rows.append(dict(population=int(key[0]), mutation_probability=float(key[1]), generation=int(key[2]),
                         seed=int(key[3]), A_validation=float(a['validation_fitness']),
                         B_validation=float(b['validation_fitness']), A_test_success=float(a['test_success']),
                         B_test_success=float(b['test_success']),
                         success_difference_A_minus_B=float(a['test_success'])-float(b['test_success']),
                         common_domain_genome_rmse=float(np.sqrt(np.mean((units[0]-units[1])**2)))))
    from scripts.AfferentBalance2026.experiments.core.run import write_csv
    for name in ('A_balance','B_balance'):
        write_csv(EXPERIMENTS/name/'results/paired_comparison.csv', rows)
    configure_fonts(13)
    probabilities = sorted({row['mutation_probability'] for row in rows})
    sizes = sorted({row['population'] for row in rows})
    epochs = sorted({row['generation'] for row in rows})
    fig, axes = plt.subplots(1,len(probabilities),figsize=(5*len(probabilities),4),squeeze=False,constrained_layout=True)
    for ax, probability in zip(axes[0], probabilities):
        grid = np.full((len(sizes),len(epochs)),np.nan)
        for row in rows:
            if row['mutation_probability']==probability:
                grid[sizes.index(row['population']),epochs.index(row['generation'])]=row['success_difference_A_minus_B']
        im=ax.imshow(grid,origin='lower',aspect='auto',cmap='coolwarm',vmin=-1,vmax=1)
        ax.set(xticks=range(len(epochs)),xticklabels=epochs,yticks=range(len(sizes)),yticklabels=sizes,
               xlabel='Generation',ylabel='Population',title=f'p={probability}: success A − B')
        fig.colorbar(im,ax=ax)
    for name in ('A_balance','B_balance'):
        for extension in ('png','pdf'):
            fig.savefig(EXPERIMENTS/name/'results'/f'comparison_A_minus_B.{extension}',dpi=180)
        for language in ('md','en.md'):
            path=EXPERIMENTS/name/'results'/f'README.{language}'
            description=('Сопоставление одинаковых бюджетов A/B: [таблица](paired_comparison.csv), '
                         '[разность долей успеха](comparison_A_minus_B.png). RMSE геномов A/B вычислена '
                         'по 50 координатам в общей нормировке диапазона A.\n' if language=='md' else
                         'Matched A/B budgets: [table](paired_comparison.csv), '
                         '[success-fraction difference](comparison_A_minus_B.png). Cross-experiment genome RMSE '
                         'uses 50 coordinates normalized to the common A domain.\n')
            path.write_text(path.read_text()+'\n'+description)
    plt.close(fig)
    print('Matched A/B comparison saved:',len(rows),'checkpoints',flush=True)


def main():
    """Analyze one completed experiment without launching or changing training."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment',choices=['A_balance','B_balance','C_setpoint_balance','compare'])
    parser.add_argument('--sweep',default='sweep_20261021')
    args=parser.parse_args()
    if args.experiment=='compare':
        compare(args.sweep)
    elif args.experiment=='C_setpoint_balance':
        setpoint(args.sweep)
    else:
        balance(args.experiment,args.sweep)


if __name__=='__main__':
    main()
