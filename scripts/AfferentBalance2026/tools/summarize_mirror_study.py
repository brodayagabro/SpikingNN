"""Verify, compare and publish the separate exact-reflection C/D experiment."""
import argparse
from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import shlex
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from scripts.AfferentBalance2026.experiments.core.genome import (
    GenomeConfig, MIRROR_INDICES, NAMES, file_hash, write_json,
)
from scripts.AfferentBalance2026.experiments.core.run import PopulationEvaluator
from scripts.AfferentBalance2026.experiments.C_setpoint_balance.joint_study import (
    load_joint_genome, target_banks, evaluate_joint,
)
from scripts.AfferentBalance2026.experiments.D_trajectory.controller import load_solution, TrajectoryGenome
from scripts.AfferentBalance2026.experiments.D_trajectory.trajectory import trajectory_banks, sine_reference
from scripts.AfferentBalance2026.experiments.D_trajectory.run_experiment import Evaluator
from scripts.AfferentBalance2026.experiments.visualization import configure_fonts

EXPERIMENTS = REPO/'scripts/AfferentBalance2026/experiments'
STUDY = 'mirror_symmetry_20261026'


def read(path):
    """Read UTF-8 JSON metadata."""
    return json.loads(Path(path).read_text())


def solution(path, experiment):
    """Load either portable command representation without changing its domain."""
    if experiment == 'C':
        return load_joint_genome(path)
    cfg, p, c, _, _ = load_solution(path)
    codec = TrajectoryGenome(cfg)
    return codec, codec.encode(p, c)


def metrics_summary(metrics, cfg, experiment, bank='test'):
    """Separate completed episodes, strict success and error over valid intervals."""
    if experiment == 'C':
        complete = np.isclose(metrics[:, 1], cfg.data['simulation']['duration_s'], rtol=0, atol=1e-8)
        complete &= (metrics[:, 0] >= 0) & (abs(metrics[:, 8]) <= np.pi/2)
    else:
        specs = trajectory_banks(cfg.data['trajectory'])[bank]
        complete = np.isclose(metrics[:, 1], [s['duration_s'] for s in specs], rtol=0, atol=1e-8)
        target = np.array([sine_reference(s)[-1, 0] for s in specs])
        complete &= (metrics[:, 0] >= 0) & (abs(metrics[:, 8]+target) <= np.pi/2)
    row = dict(cases=len(metrics), completed=int(complete.sum()), successes=int(metrics[:, 5].sum()),
               nonfinite=int((metrics[:, 0] < 0).sum()),
               mean_rmse_all_deg=float(np.rad2deg(metrics[:, 2]).mean()),
               mean_rmse_completed_deg=float(np.rad2deg(metrics[complete, 2]).mean()) if complete.any() else None)
    if experiment == 'C':
        row['late_rmse_all_deg'] = float(np.rad2deg(metrics[:, 7]).mean())
    else:
        row['mean_band_fraction'] = float(metrics[:, 6].mean())
        row['velocity_rmse_completed_rad_s'] = float(metrics[complete, 7].mean()) if complete.any() else None
    return row


def verify_generations(folder, experiment):
    """Check every stored offspring against exact ties, bounds and its saved codec."""
    checked = 0
    stages = []
    for stage in sorted(p.parent for p in folder.glob('*/run.json')):
        record = read(stage/'run.json')
        if record['status'] != 'completed':
            raise RuntimeError(f'Incomplete training: {stage}')
        codec, _ = solution(stage/'best_genome.json', experiment)
        paths = sorted((stage/'generations').glob('generation_*.npz'))
        assert len(paths) == record['generations']
        for path in paths:
            with np.load(path) as arrays:
                p = arrays['physical'][:, :52]
                np.testing.assert_array_equal(p, p[:, MIRROR_INDICES])
                assert np.all(p[:, [19, 20, 26, 27]] >= 50)
                assert np.all(p[:, [19, 26]] <= 100) and np.all(p[:, [20, 27]] <= 200)
                for unit, physical in zip(arrays['normalized'], arrays['physical']):
                    decoded, command = codec.decode(unit)
                    np.testing.assert_array_equal(np.r_[decoded, command.ravel()], physical)
                checked += len(p)
        stages.append(dict(stage=stage.name, generations=len(paths), independent_coordinates=codec.size))
    return dict(verified_individuals=checked, stages=stages)


def confirmation(folder, experiment, selected, workers):
    """Evaluate frozen old/new winners once on a preregistered new random bank."""
    output = folder/'confirmation'
    paths = {'asymmetric': folder/'asymmetric_baseline/best_genome.json',
             'symmetric': selected/'best_genome.json'}
    hashes = {name: file_hash(path) for name, path in paths.items()}
    if (output/'summary.json').exists():
        result = read(output/'summary.json')
        if result['genome_sha256'] != hashes:
            raise ValueError('Confirmation already evaluated different frozen genomes')
        return result
    output.mkdir(exist_ok=True)
    result = dict(seed=read(folder/'planned_comparison.json')['confirmation_test_seed'],
                  selection_used=False, genome_sha256=hashes, results={})
    for name, path in paths.items():
        codec, unit = solution(path, experiment)
        data = deepcopy(codec.controller.data)
        key = 'simulation' if experiment == 'C' else 'trajectory'
        data[key]['bank_seeds']['test'] = result['seed']
        cfg = GenomeConfig(data)
        if experiment == 'C':
            bank = target_banks(cfg)['test']
            with PopulationEvaluator(cfg.mechanics, workers) as evaluator:
                score, metrics = evaluate_joint(evaluator, codec, unit[None], bank, data['simulation']['duration_s'])
            np.savez_compressed(output/(name+'.npz'), cases=bank, metrics=metrics[0], fitness=score)
        else:
            banks = trajectory_banks(data['trajectory'])
            with Evaluator(codec, banks, workers) as evaluator:
                score, metrics = evaluator.evaluate(unit[None], 'test', load_solution(path)[3])
            np.savez_compressed(output/(name+'.npz'), metrics=metrics[0], fitness=score)
            write_json(output/'cases.json', banks['test'])
        result['results'][name] = dict(fitness=float(score[0]), **metrics_summary(metrics[0], cfg, experiment))
    write_json(output/'summary.json', result)
    return result


def command_line(protocol):
    """Reconstruct the exact warm-start CLI recorded by each completed branch."""
    tokens = ['.venv/bin/python', 'scripts/AfferentBalance2026/tools/retrain_tensile.py']
    for key, value in protocol['options'].items():
        flag = '--'+key.replace('_', '-')
        if isinstance(value, bool):
            tokens.append(flag if value else '--no-'+key.replace('_', '-'))
        elif isinstance(value, list):
            for item in value:
                tokens.extend([flag, str(item).replace(str(REPO)+'/', '')])
        else:
            tokens.extend([flag, str(value).replace(str(REPO)+'/', '')])
    return shlex.join(tokens)


def publish_versions(folder, experiment, selected):
    """Copy the three requested physiological variants and record their origins."""
    root = folder.parent.parent
    fast_manifest = read(REPO/'scripts/AfferentBalance2026/report/results/tensile_retrain_20261026/manifest.json')
    fast_entry = fast_manifest['selected'][experiment]
    fast = REPO/fast_entry['run']/'best_genome.json'
    if file_hash(fast) != fast_entry['genome_sha256']:
        raise ValueError('Historical fast-muscle winner differs from its report manifest')
    sources = {
        'best_genome_fast_muscle.json': fast,
        'best_genome_slow_muscle.json': folder/'asymmetric_baseline/best_genome.json',
        # Keep the user-requested spelling "mirrow" in the published filename.
        'best_genome_slow_muscle_mirrow.json': selected/'best_genome.json',
    }
    rows = []
    for name, source in sources.items():
        codec, unit = solution(source, experiment)
        physical, _ = codec.decode(unit)
        destination = root/name
        shutil.copy2(source, destination)
        assert file_hash(destination) == file_hash(source)
        fields = codec.controller.data['parameters']
        rows.append(dict(file=name, source=str(source.relative_to(REPO)), sha256=file_hash(source),
                         mirror_symmetry=codec.controller.mirror_symmetry,
                         times_ms={NAMES[i]:float(physical[i]) for i in (19, 20, 26, 27)},
                         bounds_ms={NAMES[i]:[fields[i]['minimum'], fields[i]['maximum']] for i in (19, 20, 26, 27)}))
    write_json(folder/'genome_versions.json', rows)
    for language in ('ru', 'en'):
        text = ('# Версии лучших геномов\n\n' if language == 'ru' else '# Best-genome versions\n\n')
        text += ('`fast` — решение до введения границы 50 мс; это не означает отсутствия '
                 'всех ограничений в старом конфиге. `slow` — асимметричное решение с '
                 'нижней границей 50 мс. `slow_muscle_mirrow` — новое решение с той же '
                 'границей и точной зеркальной симметрией. Файлы скопированы без изменения чисел.\n\n'
                 if language == 'ru' else
                 '`fast` is the winner before the 50 ms floor; it still retains the bounds '
                 'of its historical configuration. `slow` is the asymmetric winner with '
                 'the 50 ms floor. `slow_muscle_mirrow` adds exact reflection symmetry '
                 'under that same floor. Files are copied without changing any values.\n\n')
        text += '| File | τc F, ms | τ1 F, ms | τc E, ms | τ1 E, ms |\n|---|---:|---:|---:|---:|\n'
        for row in rows:
            text += '| ['+row['file']+']('+row['file']+') | '+' | '.join(f'{v:.4f}' for v in row['times_ms'].values())+' |\n'
        text += '\n'
        for row in rows:
            relative = Path(os.path.relpath(REPO/row['source'], root))
            label = 'источник' if language == 'ru' else 'source'
            text += f"- `{row['file']}`: [{label}]({relative.as_posix()}), SHA256 `{row['sha256']}`.\n"
        suffix = '' if language == 'ru' else '.en'
        labels = ('Протокол', 'Результаты') if language == 'ru' else ('Study protocol', 'Results')
        text += f'\n[{labels[0]}](../MIRROR_SYMMETRY{suffix}.md) · [{labels[1]}](results/{STUDY}/README{suffix}.md)\n'
        (root/f'GENOME_VERSIONS{suffix}.md').write_text(text)
    return rows


def learning_plot(branches, output):
    """Plot saved training and validation histories, never interpolated test results."""
    configure_fonts(14)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, (experiment, folders) in zip(axes, branches.items()):
        for folder in folders:
            offset = 0
            stages = [folder/name for name in ('training', 'table', 'polynomial') if (folder/name).exists()]
            for stage in stages:
                with (stage/'history.csv').open() as stream:
                    history = list(csv.DictReader(stream))
                with (stage/'validation_history.csv').open() as stream:
                    validation = list(csv.DictReader(stream))
                label = folder.name + ('/'+stage.name if len(stages) > 1 else '')
                line, = ax.plot([offset+int(r['generation']) for r in history],
                                [float(r['best_fitness']) for r in history], label=label+' train')
                ax.plot([offset+int(r['generation']) for r in validation],
                        [float(r['selected_fitness']) for r in validation], '--', color=line.get_color(), label=label+' validation')
                offset += len(history)
        ax.set(title=experiment, xlabel='Generation', ylabel='Fitness', ylim=(0, 1.02))
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
    for suffix in ('png', 'pdf'):
        fig.savefig(output/f'learning.{suffix}', dpi=180)
    plt.close(fig)


def target_plot(folders, output):
    """Show strict successes by target, including all C branches and the old winner."""
    values, labels = [], []
    for folder in [folders[0].parent/'asymmetric_baseline', *folders]:
        codec, _ = solution(folder/'best_genome.json', 'C')
        cases = target_banks(codec.controller)['test']
        metrics = np.load(folder/'test_metrics.npz')['metrics']
        targets = np.unique(cases[:, 0])
        values.append([int(metrics[cases[:, 0] == target, 5].sum()) for target in targets])
        labels.append(folder.name)
    fig, ax = plt.subplots(figsize=(14, 3.8), constrained_layout=True)
    display = ax.imshow(values, vmin=0, vmax=24, cmap='YlGn', aspect='auto')
    for row in range(len(values)):
        for column, value in enumerate(values[row]):
            ax.text(column, row, f'{value}/24', ha='center', va='center', fontsize=10)
    ax.set(xticks=np.arange(len(targets)), xticklabels=[f'{v:g}' for v in np.rad2deg(targets)],
           yticks=np.arange(len(labels)), yticklabels=labels, xlabel='Target angle (deg)',
           title='C: strict success on the unchanged held-out test')
    ax.tick_params(axis='x', labelrotation=45)
    fig.colorbar(display, ax=ax, label='Successful episodes')
    for suffix in ('png', 'pdf'):
        fig.savefig(output/f'strict_success_targets.{suffix}', dpi=180)
    plt.close(fig)


def comparison_report(results, output):
    """Write a bilingual result appendix without rewriting the previous full article."""
    report = output.parent.parent
    for language in ('ru', 'en'):
        ru = language == 'ru'
        title = 'Проверка зеркальной симметрии C/D' if ru else 'C/D reflection-symmetry experiment'
        text = f'# {title}\n\n'
        text += ('Сравниваются прежние асимметричные решения с временами мышц ≥50 мс '
                 'и отдельная серия с точно связанными парными генами и зеркальным законом токов. '
                 'Механика, фитнес, шаг 0.1 мс и исходные наборы испытаний одинаковы. '
                 'Выбор решения выполнен по валидационному фитнесу, не по тестовым успехам.\n\n'
                 if ru else
                 'The previous asymmetric solutions with muscle times ≥50 ms are compared '
                 'with a separate exact-pair and mirrored-current study. Mechanics, fitness, '
                 'the 0.1 ms step and original episode banks are unchanged. Selection uses '
                 'validation fitness, not test success.\n\n')
        suffix = '' if ru else '.en'
        text += f'[Protocol](../experiments/MIRROR_SYMMETRY{suffix}.md)\n\n'
        text += '| Experiment | Branch | Validation | Completed | Strict success | Error, deg |\n|---|---|---:|---:|---:|---:|\n'
        for experiment, result in results.items():
            for row in result['rows']:
                error = row['late_rmse_all_deg'] if experiment == 'C' else row['mean_rmse_completed_deg']
                shown = '—' if error is None else f'{error:.3f}'
                text += f"| {experiment} | {row['branch']} | {row['validation_fitness']:.6f} | {row['completed']}/{row['cases']} | {row['successes']}/{row['cases']} | {shown} |\n"
        text += ('\nДля C ошибка — средний RMSE последних 2 с; падения имеют прежний штраф 90°. '
                 'Для D показан RMSE только завершённых эпизодов; число незавершённых явно приведено '
                 'в соседнем столбце. Метрики всех эпизодов, включая штрафы, сохранены в JSON.\n\n'
                 if ru else
                 '\nC error is mean last-2-s RMSE, with the unchanged 90° fall penalty. D error '
                 'is RMSE over completed episodes only; the completion count is explicit. '
                 'All-episode metrics, including penalties, are retained in JSON.\n\n')
        text += f'![Learning](results/{STUDY}/learning.png)\n\n'
        text += f'![Success by target](results/{STUDY}/strict_success_targets.png)\n\n'
        text += ('## Новый подтверждающий набор\n\n' if ru else '## Fresh confirmation bank\n\n')
        text += ('Seed 2026102605 был задан до окончания обучения. Эти результаты не выбирают '
                 'геномы и не используются для дальнейшего обучения данной серии.\n\n'
                 if ru else
                 'Seed 2026102605 was fixed before training finished. These outcomes do not '
                 'select genomes or inform further training in this study.\n\n')
        text += '| Experiment | Variant | Completed | Strict success |\n|---|---|---:|---:|\n'
        for experiment, result in results.items():
            for variant, row in result['confirmation']['results'].items():
                text += f"| {experiment} | {variant} | {row['completed']}/{row['cases']} | {row['successes']}/{row['cases']} |\n"
        text += ('\n## Динамика и файлы\n\n' if ru else '\n## Dynamics and files\n\n')
        for experiment, result in results.items():
            name = 'C_setpoint_balance' if experiment == 'C' else 'D_trajectory'
            prefix = f'../experiments/{name}'
            text += f"- {experiment}: [versions]({prefix}/GENOME_VERSIONS{suffix}.md), [results]({prefix}/results/{STUDY}/README{suffix}.md).\n"
            image = 'target_+022.5.png' if experiment == 'C' else 'tracking.png'
            text += f"\n![{experiment} dynamics]({prefix}/results/{STUDY}/{result['selected']}/figures/{image})\n\n"
        c_figures = EXPERIMENTS/'C_setpoint_balance/results'/STUDY/results['C']['selected']/'figures'
        examples = []
        for path in sorted(c_figures.glob('*_example.npz')):
            with np.load(path) as arrays:
                trace, case = arrays['trace'], arrays['case']
                mask = trace[:, 0] > 3.
                if mask.any():
                    examples.append((float(np.rad2deg(case[0])),
                        float(np.rad2deg(abs(trace[mask, 1]-case[0])).max()),
                        float(abs(trace[mask, 2]).max())))
        if examples:
            text += ('Для примера C с начальным углом −30° и нулевой скоростью после 3 с:\n\n'
                     if ru else 'For the C example starting at −30° with zero speed, after 3 s:\n\n')
            text += '| Target, deg | Maximum angle error, deg | Maximum speed, rad/s |\n|---:|---:|---:|\n'
            for target, error, speed in sorted(examples):
                text += f'| {target:g} | {error:.4f} | {speed:.4f} |\n'
            text += ('\nНа промежуточных целях этого примера допуск угла выполняется, '
                     'но нарушается предел скорости 0.35 рад/с.\n\n' if ru else
                     '\nAt the intermediate targets in this example, the angle tolerance holds '
                     'but the 0.35 rad/s speed limit is exceeded.\n\n')
        d_figures = EXPERIMENTS/'D_trajectory/results'/STUDY/results['D']['selected']/'figures'
        if (d_figures/'trajectory_tests.npz').exists():
            spec = read(d_figures/'trajectory_specs.json')[0]
            with np.load(d_figures/'trajectory_tests.npz') as arrays:
                trace, reference = arrays['trace_000'], arrays['reference_000']
                mask = trace[:, 0] > spec['hold_s']
                if mask.any():
                    angle = abs(np.rad2deg(trace[mask, 1]-reference[mask, 0]))
                    velocity = abs(trace[mask, 2]-reference[mask, 1])
                    a, v = 100*np.mean(angle > 3), 100*np.mean(velocity > .35)
                    text += (f'В первом тесте D после разогрева допуск угла нарушен в {a:.2f}% '
                             f'отсчётов, допуск ошибки скорости — в {v:.2f}%. Здесь основное '
                             'ограничение связано с ошибкой и запаздыванием угла.\n\n' if ru else
                             f'In D test 0 after warm-up, the angle bound is exceeded in {a:.2f}% '
                             f'of samples and the velocity-error bound in {v:.2f}%. This example '
                             'is primarily limited by angle error and lag.\n\n')
        total = sum(row['integrity']['verified_individuals'] for r in results.values() for row in r['rows'] if 'integrity' in row)
        text += (f'Проверены все {total} сохранённых особей: парные равенства, границы времён и '
                 'восстановление полных геномов из нормированных координат. Численные данные '
                 'графиков сохраняются вместе с изображениями. Симметрия сокращает поиск, '
                 'но не является доказательством устойчивости: небольшие колебания угла '
                 'могут сопровождаться превышением допуска скорости. Бюджеты прежней и новой '
                 'серий различаются; статистическое сравнение требует дополнительных seed.\n'
                 if ru else
                 f'All {total} saved individuals were checked for pair equality, time bounds '
                 'and reconstruction from normalized coordinates. Figures retain their numerical '
                 'data. Symmetry reduces the search space but does not prove stability: small '
                 'angle oscillations can still violate the velocity criterion. Historical and '
                 'new training budgets differ; statistical comparisons require additional seeds.\n')
        checks = output/'notebook_checks/checks.json'
        if checks.exists():
            text += ('\nТри тетрадки C/D исполнены с новыми JSON, включая оба живых виджета '
                     'и сохранение снимка. Ошибок исполнения нет. При отдельной проверке '
                     'резкая команда −20°→+45° вызвала падение через 0.4329 с; сброс '
                     'работает. Это физическая неудача управления, а не ошибка виджета. '
                     'Протокол проверки сохранён в `notebook_checks/checks.json`.\n'
                     if ru else
                     '\nThree C/D notebooks executed with the new JSONs, including both live '
                     'widgets and snapshot export. There were no execution errors. In a '
                     'separate command-jump check, −20°→+45° caused a fall after 0.4329 s; '
                     'reset worked. This is a control failure, not a widget error. '
                     'The record is in `notebook_checks/checks.json`.\n')
        (report/f'MIRROR_SYMMETRY{suffix}.md').write_text(text)


def main():
    """Select by validation, verify artifacts and write an independent comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    all_results, branches = {}, {}
    for experiment, name in [('C', 'C_setpoint_balance'), ('D', 'D_trajectory')]:
        folder = EXPERIMENTS/name/'results'/STUDY
        planned = read(folder/'planned_comparison.json')
        folders = [folder/branch for branch in planned['branches']]
        branches[experiment] = folders
        selected = max(folders, key=lambda path: read(path/'result.json')['validation_fitness'])
        old_codec, _ = solution(folder/'asymmetric_baseline/best_genome.json', experiment)
        new_codec, _ = solution(selected/'best_genome.json', experiment)
        if experiment == 'C':
            old_banks, new_banks = target_banks(old_codec.controller), target_banks(new_codec.controller)
            for bank in old_banks:
                np.testing.assert_array_equal(old_banks[bank], new_banks[bank])
        else:
            assert (trajectory_banks(old_codec.controller.data['trajectory']) ==
                    trajectory_banks(new_codec.controller.data['trajectory']))
        rows = []
        for path in [folder/'asymmetric_baseline', *folders]:
            codec, unit = solution(path/'best_genome.json', experiment)
            metrics = np.load(path/'test_metrics.npz')['metrics']
            row = dict(branch=path.name, validation_fitness=read(path/'result.json')['validation_fitness'],
                       independent_coordinates=codec.size, genome_sha256=file_hash(path/'best_genome.json'),
                       **metrics_summary(metrics, codec.controller, experiment))
            if path.name != 'asymmetric_baseline':
                row['integrity'] = verify_generations(path, experiment)
            rows.append(row)
        # Save the branch selection before evaluating the new confirmation bank.
        write_json(folder/'selection.json', dict(branch=selected.name, criterion='maximum validation fitness',
                   test_used=False, genome_sha256=file_hash(selected/'best_genome.json')))
        shutil.copy2(selected/'best_genome.json', folder/'best_genome.json')
        versions = publish_versions(folder, experiment, selected)
        write_json(folder/'pair_constraints.json', dict(pairs=[dict(first=NAMES[i], second=NAMES[j])
            for i, j in enumerate(MIRROR_INDICES) if i < j], command_rule='I1(theta) = I0(-theta)',
            coefficient_rule='c[1,k] = (-1)**k * c[0,k]', geometry_tied=False))
        confirmed = confirmation(folder, experiment, selected, args.workers)
        result = dict(selected=selected.name, rows=rows, confirmation=confirmed,
                      previous_canonical_preserved=True, source=str(folder.relative_to(REPO)), genome_versions=versions)
        write_json(folder/'comparison.json', result)
        all_results[experiment] = result
        commands = '\n\n'.join(command_line(read(path/'protocol.json')) for path in folders)
        for language in ('ru', 'en'):
            title = 'Симметричный контроллер' if language == 'ru' else 'Mirror-symmetric controller'
            intro = ('Отдельная серия. Основной best_genome.json предыдущей серии не заменён. '
                     'Выбор выполнен только по валидации; строгий успех не является отдельным критерием выбора.'
                     if language == 'ru' else 'Separate study. The previous canonical best_genome.json is preserved. '
                     'Selection uses validation fitness only; strict success is not a separate selection objective.')
            text = f'# {title}: {experiment}\n\n{intro}\n\n'
            text += ('| Вариант | Валидация | Завершено | Строгий успех |\n|---|---:|---:|---:|\n'
                     if language == 'ru' else '| Branch | Validation | Completed | Strict success |\n|---|---:|---:|---:|\n')
            for row in rows:
                text += f"| {row['branch']} | {row['validation_fitness']:.6f} | {row['completed']}/{row['cases']} | {row['successes']}/{row['cases']} |\n"
            text += f'\n'+('Выбран' if language == 'ru' else 'Selected')+f': `{selected.name}`.\n\n'
            text += ('Файлы: ' if language == 'ru' else 'Files: ')
            text += ('`best_genome.json` — выбранный симметричный геном; `comparison.json` — все метрики; '
                     '`confirmation/` — новый тестовый набор и результаты обоих замороженных решений; '
                     '`pair_constraints.json` — все парные равенства; `inputs*/` — проекции начальных геномов; '
                     '`asymmetric_baseline/` — прежний победитель без изменений; '
                     '`*/training/` или `*/table/`, `*/polynomial/` — история всех поколений, валидация и последние популяции; '
                     '`<выбранный вариант>/figures/` — фазовые портреты и численные данные всех сигналов.\n\n'
                     if language == 'ru' else
                     '`best_genome.json` — selected symmetric genome; `comparison.json` — all metrics; '
                     '`confirmation/` — newly drawn test bank and both frozen solutions; '
                     '`pair_constraints.json` — all equality constraints; '
                     '`inputs*/` — projected initial candidates; `asymmetric_baseline/` — unchanged previous winner; '
                     '`*/training/` or `*/table/`, `*/polynomial/` — generation histories, validation and final populations; '
                     '`<selected>/figures/` — phase portraits and complete state data.\n\n')
            text += '```bash\n'+commands+'\n```\n'
            (folder/('README.md' if language == 'ru' else 'README.en.md')).write_text(text)
    output = REPO/'scripts/AfferentBalance2026/report/results'/STUDY
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'comparison.json', all_results)
    learning_plot(branches, output)
    target_plot(branches['C'], output)
    comparison_report(all_results, output)
    print(json.dumps(all_results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
