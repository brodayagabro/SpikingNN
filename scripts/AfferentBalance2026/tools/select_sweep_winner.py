"""Select a completed restored-weight B sweep winner and save its full replay.

The selected genome is published to the experiment-local results directory only
after all checks and replays finish. Previous selection metadata is retained with
its original sweep. No training or current-law fitting is performed here.
"""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from scripts.AfferentBalance2026.experiments.core.fitness import fitness
from scripts.AfferentBalance2026.experiments.core.genome import file_hash, load_genome, write_json
from scripts.AfferentBalance2026.experiments.core.run import PopulationEvaluator, save_replays
from scripts.AfferentBalance2026.experiments.simulation import (
    balance_banks, case_grid, reference_trace, simulate,
)


def audit(sweep, config):
    """Verify every stored population, score and completed run against its domain."""
    count = 0
    for line in sorted(sweep.glob('population_*/mutation_*/seed_*')):
        run = json.loads((line/'run.json').read_text())
        assert run['status'] == 'completed'
        assert run['configuration_sha256'] == config.signature
        assert run['dt_ms'] == .1 and run['muscle_torque_factor'] == 1.
        assert run['pd_used'] is False and run['currents'] == [0.,0.]
        for name, expected in run['source_sha256'].items():
            assert file_hash(REPO/name) == expected, name
        files = sorted((line/'generations').glob('generation_*.npz'))
        assert len(files) == run['generations']
        for number, path in enumerate(files, 1):
            with np.load(path, allow_pickle=False) as data:
                assert int(data['generation']) == number
                assert str(data['configuration_sha256']) == config.signature
                physical = np.array([config.decode(row) for row in data['normalized']])
                np.testing.assert_array_equal(physical, data['physical'])
                scores = np.array([fitness(m) for m in data['training_metrics']])
                np.testing.assert_allclose(scores, data['training_scores'], rtol=0, atol=1e-13)
            count += 1
    return dict(status='passed', generations=count, sources_verified=True,
                population_domains_verified=True, training_scores_verified=True)


def plot_replays(selected):
    """Plot dense saved replays; decimate phase lines only for display."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scripts.AfferentBalance2026.experiments.visualization import (
        configure_fonts, plot_angles, plot_phase, plot_states,
    )
    configure_fonts(13)
    records = []
    for path in sorted((selected/'traces').glob('case_*.npz')):
        with np.load(path, allow_pickle=False) as data:
            trace = data['trace']
            indices = np.unique(np.r_[np.arange(0,len(trace),25),len(trace)-1])
            records.append(dict(case=data['case'], trace=trace[indices], metrics=data['metrics']))
    phase = [r for r in records if abs(r['case'][2]) > 1e-12]
    zero = [r for r in records if abs(r['case'][2]) <= 1e-12]
    figures = {'phase':plot_phase(phase), 'angles':plot_angles(zero)}
    with np.load(selected/'traces/case_102.npz', allow_pickle=False) as data:
        figures.update(plot_states(dict(case=data['case'], trace=data['trace'], metrics=data['metrics'])))
    for name, figure in figures.items():
        for extension in ('png','pdf'):
            figure.savefig(selected/f'{name}.{extension}', dpi=180)
        plt.close(figure)
    return dict(phase_cases=len(phase), zero_speed_cases=len(zero),
                success_count=int(sum(r['metrics'][5] for r in phase)),
                late_rmse_deg=float(np.rad2deg(np.mean([r['metrics'][7] for r in phase]))))


def select(sweep):
    """Choose the validation winner, verify its independent test, then publish it."""
    sweep = Path(sweep).resolve()
    completion = json.loads((sweep/'completion.json').read_text())
    assert completion['status'] == 'completed'
    with (sweep/'summary.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == completion['checkpoints']
    chosen = max(rows, key=lambda row: float(row['validation_fitness']))
    folder = sweep/chosen['directory']
    config, physical = load_genome(folder/'best_genome.json')
    result = json.loads((folder/'result.json').read_text())
    assert config.data['experiment'] == 'B_balance'
    fields = config.data['parameters']
    if (config.size != 48 or any(fields[i]['mutable'] or fields[i]['value'] != .37
                                for i in (18,25))
            or any(fields[i]['minimum'] != .02 or fields[i]['maximum'] != np.nextafter(.1,0.)
                   for i in (16,23))):
        raise ValueError('Expected restored B weights 0.02 <= w < 0.1 and fixed muscle A=0.37')
    duration = result['duration_s']
    checks = audit(sweep, config)
    banks = balance_banks(config.data['simulation'])
    with PopulationEvaluator(config.mechanics) as evaluator:
        scores, metrics = evaluator.evaluate(physical[None], banks['test'], duration)
    np.testing.assert_allclose(scores[0], result['test_fitness'], rtol=0, atol=1e-13)
    assert metrics[0,:,5].mean() == result['test_success']
    selected = sweep/'selected'
    selected.mkdir(exist_ok=False)
    config.save(selected/'best_genome.json', physical, selection='maximum validation fitness across sweep',
                source=str(folder.relative_to(sweep)), generation=result['generation'],
                population=result['population'], mutation_probability=result['mutation_probability'],
                seed=result['seed'], warm_start=False)
    write_json(selected/'genome_cfg.json', config.data)
    np.savez_compressed(selected/'test_metrics.npz', cases=banks['test'], metrics=metrics[0])
    reference_case = case_grid([50.], [1.])[0]
    reference_duration = min(1., duration)
    _, actual = simulate(physical, reference_case, config.mechanics, duration_s=reference_duration)
    expected = reference_trace(physical, reference_case, config.mechanics, duration_s=reference_duration)
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-8)
    write_json(selected/'model_reference_check.json', dict(
        status='passed', duration_s=reference_duration, samples=len(actual),
        theta0_deg=50., omega0_rad_s=1., channels=actual.shape[1],
        max_absolute_difference=float(np.max(np.abs(actual-expected))),
        genome_sha256=file_hash(selected/'best_genome.json')))
    print('Validated winner:', json.dumps(result), flush=True)
    cases = case_grid(np.linspace(-50.,50.,10), np.r_[np.linspace(-1.,1.,10),0.])
    save_replays(selected/'traces', config, physical, cases, duration_s=duration)
    grid = plot_replays(selected)
    summary = dict(experiment=config.data['experiment'], sweep=sweep.name, selected=result,
                   selection='validation only', grid=grid, audit=checks,
                   muscle_A_N_per_ms=physical[[18,25]].tolist(),
                   neuromuscular_weights=physical[[16,23]].tolist(),
                   configuration_sha256=config.signature,
                   training_performed=True, warm_start=False, checkpoints=len(rows),
                   training_lines=completion['lines'], test_count=len(banks['test']),
                   source_directory=str(folder.relative_to(sweep)))
    write_json(selected/'summary.json', summary)
    write_json(selected/'audit.json', checks)
    shutil.copy2(__file__, selected/'selection_source.py')
    results = sweep.parent
    pointer = results/'current.json'
    if pointer.exists():
        previous = json.loads(pointer.read_text())
        prior_sweep = results/previous['sweep']
        preserved = prior_sweep/'previous_selection'
        preserved.mkdir(exist_ok=False)
        for filename in ('best_genome.json','current.json','summary.json','audit.json',
                         'README.md','README.en.md','initial_baseline.json','initial_baseline_metrics.npz',
                         'analysis_source.py','environment.json','notebook_validation.json'):
            path = results/filename
            if path.is_file():
                shutil.move(str(path), preserved/filename)
        for pattern in ('comparison_A_minus_B.*','paired_comparison.csv','convergence.*'):
            for path in results.glob(pattern):
                shutil.move(str(path), preserved/path.name)
    for filename in ('best_genome.json','summary.json','audit.json'):
        shutil.copy2(selected/filename,results/filename)
    write_json(pointer, dict(sweep=sweep.name, genome='best_genome.json', selected=str(selected.relative_to(results))))
    write_indexes(results, sweep, summary)
    print('Saved current best genome:', results/'best_genome.json', flush=True)


def write_indexes(results, sweep, summary):
    """Write bilingual result indexes identifying the current weight domain."""
    selected = summary['selected']
    grid = summary['grid']
    for language in ('ru','en'):
        ru = language == 'ru'
        text = ('# B: обучение с восстановленным диапазоном весов\n\n' if ru else
                '# B: training with the restored weight range\n\n')
        text += ('`0.02 ≤ w < 0.1`, начальные веса 0.08, фиксированное `A=0.37` для обеих мышц (48 свободных параметров). '
                 f"Шаг 0.1 мс, горизонт {selected['duration_s']:g} с, обучение −50…+50° × −1…+1 рад/с.\n\n" if ru else
                 '`0.02 ≤ w < 0.1`, initial weights 0.08 and fixed `A=0.37` for both muscles (48 free parameters). '
                 f"Step 0.1 ms, episodes {selected['duration_s']:g} s, training −50…+50° × −1…+1 rad/s.\n\n")
        text += (f"Population **{selected['population']}**, mutation **{selected['mutation_probability']}**, "
                 f"generation **{selected['generation']}**, seed **{selected['seed']}**.\n\n"
                 f"Validation fitness **{selected['validation_fitness']:.6f}**; "
                 f"test **{round(selected['test_success']*summary['test_count'])}/{summary['test_count']}**; "
                 f"phase grid **{grid['success_count']}/100**.\n\n")
        text += ('Сеть обучена с нуля. Выбор только по валидации; тест и фазовая сетка не использованы для отбора. '
                 f"Сохранены {summary['training_lines']} линий обучения, {summary['checkpoints']} контрольных точек "
                 f"и все {summary['audit']['generations']} поколений.\n\n" if ru else
                 'Fresh GA training. Selection uses validation only; test and phase-grid results were not used for selection. '
                 f"All {summary['training_lines']} training lines, {summary['checkpoints']} checkpoints "
                 f"and {summary['audit']['generations']} generations are saved.\n\n")
        text += (f"[Genome](best_genome.json) · [Summary](summary.json) · [Sweep]({sweep.name}/summary.csv) · "
                 f"[Phase portrait]({sweep.name}/selected/phase.png) · [Angle traces]({sweep.name}/selected/angles.png).\n\n"
                 f"`{sweep.name}/selected/traces/`: 110 full trajectories. "
                 f"`{sweep.name}/selected/`: neurons, muscles and afferents for one example.\n")
        (results/('README.md' if ru else 'README.en.md')).write_text(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sweep', type=Path)
    select(parser.parse_args().sweep)
