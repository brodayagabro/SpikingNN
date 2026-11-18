"""Audit and summarize a completed joint/staged C sweep and its saved diagnostics."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.AfferentBalance2026.experiments.core.genome import file_hash, write_json
from scripts.AfferentBalance2026.experiments.core.fitness import fitness
from scripts.AfferentBalance2026.experiments.core.run import PopulationEvaluator, source_hashes, write_csv
from scripts.AfferentBalance2026.experiments.simulation import measure_trace, reference_trace, simulate
from scripts.AfferentBalance2026.experiments.visualization import configure_fonts
from scripts.AfferentBalance2026.experiments.C_setpoint_balance.joint_study import (
    JointGenome, load_joint_genome, evaluate_joint,
)
from scripts.AfferentBalance2026.experiments.C_setpoint_balance.sweep_diagnostics import (
    choose_method_winners, read_rows, save_figure,
)


def audit(directory):
    """Verify every generation and frozen snapshot, then replay both selected tests."""
    directory = Path(directory)
    stages, generations, individuals, checkpoints = 0, 0, 0, 0
    live = source_hashes()
    for path in sorted(directory.glob('*_degree_*/population_*/mutation_*/seed_*/*/run.json')):
        run = json.loads(path.read_text()); stage = path.parent
        assert run['status'] == 'completed'
        assert run['source_sha256'] == live
        spec = run['command']; genome = JointGenome(stage/'genome_cfg.json',spec['kind'],spec['degree'])
        assert genome.controller.signature == run['configuration_sha256']
        with zipfile.ZipFile(stage/'source_snapshot.zip') as archive:
            for name, digest in run['source_sha256'].items():
                assert hashlib.sha256(archive.read(name)).hexdigest() == digest
        files = sorted((stage/'generations').glob('generation_*.npz'))
        assert len(files) == run['generations']
        for number, file in enumerate(files, 1):
            with np.load(file, allow_pickle=False) as data:
                assert int(data['generation']) == number
                assert data['training_metrics'].shape == (run['population'],run['training_cases'],9)
                assert str(data['configuration_sha256']) == genome.controller.signature
                assert list(data['names']) == list(genome.names)
                for unit, physical in zip(data['normalized'],data['physical']):
                    p, c = genome.decode(unit)
                    np.testing.assert_array_equal(np.r_[p,c.ravel()],physical)
                    genome.controller.encode(p)
                    np.testing.assert_array_equal(p[[18,25]],[.37,.37])
                np.testing.assert_allclose([fitness(m) for m in data['training_metrics']],data['training_scores'],rtol=0,atol=1e-14)
        checks = read_rows(stage/'checkpoints.csv')
        chosen = max(checks,key=lambda r:float(r['validation_fitness']))
        assert file_hash(stage/'best_genome.json') == file_hash(stage/chosen['directory']/'best_genome.json')
        for row in checks:
            with np.load(stage/row['directory']/'final_population.npz') as data:
                np.testing.assert_allclose([fitness(m) for m in data['validation_metrics']],data['validation_scores'],rtol=0,atol=1e-14)
                assert int(np.argmax(data['validation_scores'])) == int(row['selected_index'])
        generations += len(files); individuals += len(files)*run['population']
        stages += 1; checkpoints += len(checks)
    seed_hashes = set()
    for path in directory.glob('*_degree_*/population_*/mutation_*/seed_*/pipeline.json'):
        record = json.loads(path.read_text())
        assert record['status'] == 'completed' and not record['final_test']
        if record['seed_genome'] is not None:
            assert file_hash(record['seed_genome']) == record['seed_genome_sha256']
            seed_hashes.add(record['seed_genome_sha256'])
    assert len(seed_hashes) <= 1, 'All compared lines must start from the same controller'
    winners = choose_method_winners(directory)
    trace_count = 0; test_checks = []; max_time_error = 0.
    for method, row in winners.items():
        folder = directory/'method_winners'/method
        assert file_hash(folder/'best_genome.json') == file_hash(directory/row['directory']/'best_genome.json')
        genome, unit = load_joint_genome(folder/'best_genome.json')
        p, law = genome.decode(unit)
        duration = genome.controller.data['simulation']['duration_s']
        with np.load(folder/'test_metrics.npz') as data:
            cases, expected = data['cases'], data['metrics']
        with PopulationEvaluator(genome.controller.mechanics,1) as evaluator:
            scores, metrics = evaluate_joint(evaluator,genome,unit[None],cases,duration)
        np.testing.assert_array_equal(metrics[0],expected)
        test_checks.append(dict(method=method,test_cases=len(cases),successes=int(expected[:,5].sum()),exact_replay=True))
        for index_path in (folder/'diagnostics').rglob('metrics.csv'):
            for record in read_rows(index_path):
                path = index_path.parent/record['file']
                assert file_hash(path) == record['sha256']
                with np.load(path) as data:
                    trace, case = data['trace'],data['case']
                    np.testing.assert_array_equal(data['physical'],p)
                    np.testing.assert_array_equal(data['currents'],genome.currents(law,case[None])[0])
                    assert float(data['dt_ms']) == .1 and trace.shape[1] == 26
                    assert np.isfinite(trace).all()
                    # Compare absolute sample times: subtracting neighboring floats
                    # can lose a few ULPs at the end of a ten-second trajectory.
                    expected_times = np.arange(len(trace),dtype=float)*.1/1000
                    time_error = float(np.max(abs(trace[:,0]-expected_times)))
                    max_time_error = max(max_time_error,time_error)
                    np.testing.assert_allclose(trace[:,0],expected_times,rtol=0,
                                               atol=4*np.spacing(max(1.,duration)))
                    assert len(trace) == int(record['samples'])
                    assert abs(trace[0,1]-case[1]) < 1e-14
                    rebuilt = measure_trace(trace[1:],case[0],duration_s=duration,initial_theta=case[1])
                    np.testing.assert_allclose(rebuilt,data['metrics'],rtol=2e-10,atol=2e-10)
                trace_count += 1
        # A nonzero command also has to reproduce the original library classes.
        with np.load(folder/'diagnostics/example/case_000.npz') as data:
            case, currents = data['case'],data['currents']
        _, fast = simulate(p,case,genome.controller.mechanics,currents=currents,duration_s=.1)
        library = reference_trace(p,case,genome.controller.mechanics,currents=currents,duration_s=.1)
        delta = float(np.max(abs(fast-library)))
        np.testing.assert_allclose(fast,library,rtol=1e-8,atol=1e-8)
        write_json(folder/'model_reference_check.json',dict(duration_s=.1,samples=len(fast),channels=26,
                   max_absolute_difference=delta,case=case.tolist(),currents=currents.tolist(),genome_sha256=file_hash(folder/'best_genome.json')))
    report = dict(status='passed',stages=stages,generations=generations,individuals=individuals,
                  validation_checkpoints=checkpoints,full_traces_checked=trace_count,
                  fixed_A=.37,source_snapshots_match=True,test_replays=test_checks,
                  seed_controller_hashes=sorted(seed_hashes),
                  max_time_axis_error_s=max_time_error,
                  selection='validation only; dense grids never used for selection')
    write_json(directory/'audit.json',report)
    return report


def plot_comparison(directory):
    """Draw validation-only hyperparameter maps and within-degree genome distances."""
    import matplotlib.pyplot as plt
    configure_fonts(14)
    rows = read_rows(directory/'summary.csv')
    methods = sorted({r['method'] for r in rows}); degrees = sorted({int(r['degree']) for r in rows})
    sizes = sorted({int(r['population']) for r in rows}); rates = sorted({float(r['mutation_probability']) for r in rows})
    seeds = sorted({int(r['seed']) for r in rows})
    for seed in seeds:
        fig, axes = plt.subplots(len(methods),len(degrees),figsize=(6*len(degrees),4.5*len(methods)),squeeze=False,constrained_layout=True)
        for i, method in enumerate(methods):
            for j, degree in enumerate(degrees):
                ax = axes[i,j]; values = np.full((len(sizes),len(rates)),np.nan)
                for row in rows:
                    if row['method']==method and int(row['degree'])==degree and int(row['seed'])==seed:
                        values[sizes.index(int(row['population'])),rates.index(float(row['mutation_probability']))]=float(row['validation_fitness'])
                image = ax.imshow(values,vmin=0,vmax=1,origin='lower',aspect='auto')
                for a in range(len(sizes)):
                    for b in range(len(rates)):
                        ax.text(b,a,f'{values[a,b]:.4f}',ha='center',va='center',color='white' if values[a,b]<.55 else 'black')
                ax.set(xticks=range(len(rates)),xticklabels=rates,yticks=range(len(sizes)),yticklabels=sizes,
                       xlabel='Per-gene mutation probability',ylabel='Population size',title=f'{method}, degree {degree}, seed {seed}')
                fig.colorbar(image,ax=ax,label='Best checkpoint validation fitness')
        save_figure(fig,directory/f'validation_heatmap_seed_{seed}')
    for degree in degrees:
        with np.load(directory/f'genome_rmse_degree_{degree}.npz') as data:
            distances, labels = data['rmse'],data['labels']
        fig, ax = plt.subplots(figsize=(9,7),constrained_layout=True)
        image = ax.imshow(distances,origin='lower')
        ax.set(xlabel='Run index',ylabel='Run index',title=f'Normalized joint-genome RMSE: degree {degree}',
               xticks=range(len(labels)),yticks=range(len(labels)))
        fig.colorbar(image,ax=ax);save_figure(fig,directory/f'genome_rmse_degree_{degree}')
        (directory/f'genome_rmse_degree_{degree}_labels.json').write_text(json.dumps(labels.tolist(),indent=2)+'\n')
    fig, axes = plt.subplots(len(methods),len(degrees),figsize=(13,9),squeeze=False,constrained_layout=True)
    for row in rows:
        ax = axes[methods.index(row['method']),degrees.index(int(row['degree']))]
        peers = [r for r in rows if r['method']==row['method'] and r['degree']==row['degree']]
        color = plt.cm.tab10(peers.index(row) % 10)
        stages = ('polynomial',) if row['method']=='joint' else ('table','polynomial')
        offset = 0
        for stage in stages:
            history = read_rows(directory/row['directory']/stage/'history.csv')
            label = f"N={row['population']}, p={row['mutation_probability']}, seed={row['seed']}" if stage=='polynomial' else None
            ax.plot([offset+int(r['generation']) for r in history],[float(r['best_fitness']) for r in history],
                    label=label,color=color,linestyle='--' if stage=='table' else '-')
            offset += len(history)
    for i,method in enumerate(methods):
        for j,degree in enumerate(degrees):
            ax=axes[i,j];ax.set(title=f'{method}, degree {degree}',xlabel='Total generation',ylabel='Best training fitness',ylim=(0,1.05))
            ax.grid(alpha=.2);ax.legend(fontsize=9)
    save_figure(fig,directory/'training_histories')


def write_fit_effects(directory):
    """Measure the LS transition on identical training episodes, without test selection."""
    records = []
    for row in read_rows(directory/'summary.csv'):
        if row['method'] != 'staged':
            continue
        source = directory/row['directory']
        saved = json.loads((source/'table/best_genome.json').read_text())
        meta = saved['metadata']
        checkpoint = source/'table/checkpoints'/f"epochs_{meta['generation']:03d}"
        with np.load(checkpoint/'final_population.npz') as data:
            table_score = float(data['training_scores'][meta['population_index']])
            table_controller = data['controller_physical'][meta['population_index']]
        with np.load(source/'polynomial/generations/generation_0001.npz') as data:
            fitted_score = float(data['training_scores'][0])
            np.testing.assert_allclose(data['controller_physical'][0],table_controller,rtol=1e-13,atol=1e-13)
        with np.load(source/'table/cases.npz') as table_cases, np.load(source/'polynomial/cases.npz') as poly_cases:
            np.testing.assert_array_equal(table_cases['training'],poly_cases['training'])
        fit = json.loads((source/'least_squares.json').read_text())
        records.append(dict(degree=int(row['degree']),population=int(row['population']),
                       mutation_probability=float(row['mutation_probability']),seed=int(row['seed']),
                       table_training_fitness=table_score,fitted_training_fitness=fitted_score,
                       fitted_minus_table=fitted_score-table_score,current_fit_rmse_nA=fit['rmse_nA'],
                       final_validation_fitness=float(row['validation_fitness']),directory=row['directory']))
    if records:
        write_csv(directory/'least_squares_effect.csv',records)


def write_summary(directory, checks):
    """Write bilingual result indexes and publish an explicit notebook pointer."""
    rows = read_rows(directory/'method_winners/summary.csv')
    for row in rows:
        folder = directory/row['directory']
        saved = json.loads((folder/'best_genome.json').read_text())
        write_csv(folder/'best_genome_parameters.csv', [
            dict(index=i, name=field['name'], value=value, mutable=field['mutable'],
                 unit=field['unit'], description=field['description'])
            for i, (field, value) in enumerate(zip(saved['parameters'],saved['physical']))])
    protocol_path = directory/'study_protocol.json'
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text())
    else:
        runs = read_rows(directory/'summary.csv')
        genome, _ = load_joint_genome(directory/'best_genome.json')
        protocol = dict(planned_runs=len(runs),
                        populations=sorted({int(r['population']) for r in runs}),
                        mutation_probabilities=sorted({float(r['mutation_probability']) for r in runs}),
                        seeds=sorted({int(r['seed']) for r in runs}),
                        total_epochs_per_run=int(runs[0]['total_generations']),
                        test_bank_seed=genome.controller.data['simulation']['bank_seeds']['test'],
                        provenance='Reconstructed from completed run metadata')
    budget = protocol['total_epochs_per_run']
    selected = json.loads((directory/'result.json').read_text())
    table = ('| Method | Degree | N | Mutation | Validation | Test | Late RMSE (deg) | Phase grid |\n'
             '|---|---:|---:|---:|---:|---:|---:|---:|\n')
    for row in rows:
        table += (f"| {row['method']} | {row['degree']} | {row['population']} | {row['mutation_probability']} | "
                  f"{float(row['validation_fitness']):.6f} | {row['test_successes']}/{row['test_cases']} | "
                  f"{float(row['test_late_rmse_deg']):.6f} | {row['grid_successes']}/{row['grid_cases']} |\n")
    for suffix, ru in [('md', True), ('en.md', False)]:
        text = ('# C: sweep двух способов обучения\n\n[English](README.en.md) · [Протокол](../../README.md)\n\n' if ru else
                '# C: two-method training sweep\n\n[Русский](README.md) · [Protocol](../../README.en.md)\n\n')
        text += (f"Завершено {protocol['planned_runs']} обучений: методы joint/staged, степени 2/3, "
                 f"популяции {protocol['populations']}, вероятности мутации {protocol['mutation_probabilities']}, "
                 f"seed={protocol['seeds']}. По {budget} эпох общего бюджета (staged: {budget//2}+{budget-budget//2}), "
                 "эпизоды 10 с, шаг 0,1 мс, полные Afferents, без ПД/ПИД и усиления момента. "
                 "Обучение использует пять целей и 40 эпизодов на особь. "
                 "Валидация: 9 целей × 12 состояний; тест: 17 × 24. "
                 "Начальные углы −50…50°, скорости −1…1 рад/с. Оба A=0,37 фиксированы; веса C в [0,0001; 1].\n\n" if ru else
                 f"Completed {protocol['planned_runs']} runs: joint/staged, degrees 2/3, "
                 f"populations {protocol['populations']}, mutation probabilities {protocol['mutation_probabilities']}, "
                 f"seeds {protocol['seeds']}. Each uses {budget} total generations (staged: {budget//2}+{budget-budget//2}), "
                 "10 s episodes, 0.1 ms steps, full Afferents, no PD/PID or torque amplification. "
                 "Training evaluates five targets and 40 episodes per candidate; validation uses 9 targets × 12 starts, "
                 "and testing 17 × 24. Initial angles span −50…50°, speeds −1…1 rad/s. Both A=0.37 values are fixed; C weights are in [0.0001, 1].\n\n")
        text += table + '\n'
        text += (f"Общий победитель по валидации: **{selected['method']}**, степень {selected['degree']}, "
                 f"N={selected['population']}, p={selected['mutation_probability']}. "
                 "Для каждого метода отдельно зафиксирован лучший геном по валидации, затем выполнены тест и диагностика. "
                 "Тест и фазовая сетка не использовались для повторного выбора. Новый seed тестового банка: "
                 f"{protocol['test_bank_seed']}; это новые начальные состояния после пилота. "
                 "Один seed ГА не позволяет делать вывод о статистическом превосходстве метода.\n\n" if ru else
                 f"Overall validation winner: **{selected['method']}**, degree {selected['degree']}, "
                 f"N={selected['population']}, p={selected['mutation_probability']}. "
                 "Each method's best validation genome was frozen before tests and diagnostics. "
                 "Neither test nor dense-grid outcomes reselected a controller. The new test-bank seed is "
                 f"{protocol['test_bank_seed']}, giving fresh initial states after the pilot. "
                 "One GA seed cannot establish statistical method superiority.\n\n")
        text += ('## Геномы, траектории и рисунки\n\n' if ru else '## Genomes, trajectories and figures\n\n')
        for method in ('joint', 'staged'):
            base = f'method_winners/{method}'
            text += (f"- **{method}**: [genome]({base}/best_genome.json), "
                     f"[parameter values]({base}/best_genome_parameters.csv), "
                     f"[phase portraits]({base}/diagnostics/phase_all_targets.png), "
                     f"[state overview]({base}/diagnostics/example/state_overview.png), "
                     f"[input currents]({base}/diagnostics/input_currents.csv), "
                     f"[test by target]({base}/test_by_target.csv), "
                     f"[grid by target]({base}/diagnostics/grid_summary.csv).\n")
        text += ('\nКаждый метод сохраняет 551 полную траекторию: 500 фазовых стартов, 50 при нулевой скорости и один заранее заданный пример. '
                 'Пример: цель +22,5°, начальный угол −30°, скорость 0. Неудачи также сохранены. '
                 '`diagnostics/example/` содержит `input_currents`, `neurons`, `membrane_zoom`, `muscles`, '
                 '`afferents`, `afferent_sums`, `limb`, `state_overview` в PNG/PDF и полный `case_000.npz`. '
                 'Сумма Ia+II+Ib — ненормированная сумма каналов мышцы, не взвешенный ток обратной связи. '
                 'Первые и последние 0,2 с мембранного потенциала показаны без прореживания.\n\n' if ru else
                 '\nEach method saves 551 full trajectories: 500 phase-grid starts, 50 at zero velocity and one predeclared example. '
                 'The example targets +22.5° from −30° with zero speed. Failures are retained. '
                 '`diagnostics/example/` contains `input_currents`, `neurons`, `membrane_zoom`, `muscles`, '
                 '`afferents`, `afferent_sums`, `limb`, `state_overview` in PNG/PDF and the full `case_000.npz`. '
                 'Ia+II+Ib is a raw per-muscle channel sum, not weighted feedback current. '
                 'The first and last 0.2 s membrane windows preserve every sample.\n\n')
        text += ''.join(f'- [Validation heatmap, seed {seed}](validation_heatmap_seed_{seed}.png)\n'
                        for seed in protocol['seeds'])
        text += ('- [Training histories](training_histories.png)\n'
                 '- [Run summary](summary.csv) · [Method winners](method_winners/summary.csv)\n'
                 '- [Least-squares transition on identical training episodes](least_squares_effect.csv)\n'
                 '- [Genome RMSE, degree 2](genome_rmse_degree_2.png) · [Degree 3](genome_rmse_degree_3.png)\n'
                 '- [Audit](audit.json) · [Prespecified protocol](study_protocol.json)\n\n')
        text += (f"Аудит: {checks['generations']} поколений, {checks['individuals']} геномов, "
                 f"{checks['full_traces_checked']} полных траекторий. Проверены границы, фиксированные параметры, "
                 "снимки исходников, метрики из полной динамики и точное повторение независимого теста. "
                 "Все исходные популяции и последние поколения сохранены в каталогах обучений; лучший геном может относиться к более ранней контрольной точке. "
                 "Результаты исключены из Git.\n" if ru else
                 f"Audit: {checks['generations']} generations, {checks['individuals']} genomes, "
                 f"{checks['full_traces_checked']} complete trajectories. Verified bounds, fixed values, "
                 "source snapshots, metrics rebuilt from trajectories and exact independent-test replay. "
                 "Training directories retain all populations and final generations; the selected genome may come from an earlier checkpoint. "
                 "Results are excluded from Git.\n")
        (directory/f'README.{suffix}').write_text(text)
    pointer = directory.parent/'current_joint.json'
    if pointer.exists() and not (directory/'previous_current_joint.json').exists():
        (directory/'previous_current_joint.json').write_bytes(pointer.read_bytes())
    write_json(pointer,dict(study_type='hyperparameter_sweep',genome=str((directory/'best_genome.json').relative_to(directory.parent)),
               summary=str((directory/'result.json').relative_to(directory.parent)),method=selected['method'],
               note='Selected by validation; both method winners and dense phase grids are saved.'))
    write_json(directory/'study_protocol.json',{**protocol,'status':'completed'})


def main():
    """Audit completed C outputs, then create comparison figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args();directory=args.directory.resolve()
    result=audit(directory);plot_comparison(directory);write_fit_effects(directory);write_summary(directory,result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
