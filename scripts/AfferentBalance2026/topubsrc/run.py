#!/usr/bin/env python3
"""One-folder entry point: train/sweep/check/simulate A|B|C, suite, or repeat.

The seven suite searches are independent and start from configuration values.
No published or previously trained genome is read by suite or repeat.
"""
import argparse
import importlib
import json
from pathlib import Path
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

from evolution import HERE, cli, profile_inputs, run_training, write_csv
from genome import file_hash, write_json

EXPERIMENTS = {'A':'A_balance', 'B':'B_balance', 'C':'C_setpoint_balance'}
PROFILES = ('A:fast', 'A:slow', 'B:fast', 'B:slow', 'C:fast', 'C:slow', 'C:mirror')


def inputs(label):
    """Resolve a declared profile and its task-specific bank/staged runner."""
    experiment, profile = label.split(':')
    folder = HERE/EXPERIMENTS[experiment]
    config, settings = profile_inputs(folder, profile)
    module = importlib.import_module(f'{folder.name}.experiment')
    return config, settings, module


def table_row(label, settings, summary):
    """Report all three banks for the SAME validation-selected individual."""
    return dict(profile=label, population=settings['population'], generations=settings['generations'],
                seed=settings['seed'], method=settings.get('method', 'joint'),
                selected_generation=summary['selected_generation'],
                **{f'{bank}_{key}':value for bank in ('training','validation','test')
                   for key,value in summary[bank].items()})


def suite(output, *, profiles=PROFILES, overrides=None):
    """Train all requested domains independently with full episode banks."""
    output = Path(output).resolve()
    plan = []
    for label in profiles:
        config, settings, module = inputs(label)
        settings.update(overrides or {})
        from evolution import validate_search
        validate_search(settings)
        plan.append((label, config, settings, module))
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'suite.json', dict(initialization='configuration', archive_genomes_used=False,
        profiles=[dict(profile=label, configuration=config, training=settings)
                  for label,config,settings,_ in plan]))
    rows = []
    for label, config, settings, module in plan:
        result = run_training(config, settings, output/label.replace(':','_'), module.make_banks,
                              runner=getattr(module, 'train_experiment', None))
        rows.append(table_row(label, settings, result))
        write_csv(output/'comparison.csv', rows)
        write_json(output/'comparison.json', rows)
    return rows


def repeat(source, output, workers=None):
    """Restart a completed from-scratch run using its recorded inputs, not its winner."""
    source = Path(source).resolve()
    provenance = json.loads((source/'provenance.json').read_text())
    if provenance.get('initialization') != 'configuration':
        raise ValueError('repeat requires a recorded run initialized from configuration')
    for filename, key in [('genome_cfg.json','configuration_sha256'), ('training.json','training_sha256')]:
        if file_hash(source/filename) != provenance[key]:
            raise ValueError(f'Changed recorded input: {filename}')
    for filename, expected in provenance['code_sha256'].items():
        if file_hash(HERE/filename) != expected:
            raise ValueError(f'Changed code: {filename}; restore source_snapshot.zip to a separate folder first')
    config = json.loads((source/'genome_cfg.json').read_text())
    settings = json.loads((source/'training.json').read_text())
    if workers is not None:
        settings['workers'] = workers
    module = importlib.import_module(f"{config['experiment']}.experiment")
    return run_training(config, settings, output, module.make_banks,
                        runner=getattr(module, 'train_experiment', None),
                        smoke=provenance['smoke_test'])


def activity(run):
    """Replay every test case of a frozen winner and measure both spike outputs."""
    import numpy as np
    from genome import load_solution
    from model import simulate, measure_activity
    run = Path(run).resolve()
    genome, unit = load_solution(run/'best_genome.json')
    stage = run/('polynomial' if (run/'polynomial').exists() else 'training')
    with np.load(stage/'cases.npz') as saved:
        cases = saved['test']
    with np.load(run/'test_metrics.npz') as saved:
        expected = saved['metrics']
    physical, command = genome.decode(unit)
    duration=genome.controller.data['simulation']['duration_s']
    window=genome.controller.data['simulation'].get('fitness_window_s',2.)
    currents=genome.currents(command,cases)
    rows = []
    for index, case in enumerate(cases):
        # Preserve the stored SI case bit-for-bit, including any perturbation.
        # A radians -> degrees -> radians round trip can alter a spike threshold.
        metrics, trace = simulate(physical,case,genome.controller.mechanics,
                                  currents=currents[index],duration_s=duration,stride=1,
                                  fitness_window_s=window)
        replay=dict(metrics=metrics,trace=trace,duration_s=duration,fitness_window_s=window)
        np.testing.assert_allclose(replay['metrics'], expected[index], atol=1e-9, rtol=1e-9)
        measured = measure_activity(replay)
        rows.append(dict(index=index, case=case.tolist(), strict_success=bool(expected[index, 5]), **measured))
    valid = [r for r in rows if r['late_spike_count'] is not None]
    report = dict(genome_sha256=file_hash(run/'best_genome.json'), used_for_selection=False,
                  bank='test', cases=len(rows), complete_late_windows=len(valid),
                  output_P=physical[48:52].reshape(2,2).tolist(),
                  definition='Spikes counted at every 0.1 ms sample. Both active means at least one spike each; common bins are 1 s, not coincident spikes.',
                  both_active_late_cases=sum(r['both_active_late'] for r in valid),
                  both_active_every_late_bin_cases=sum(r['both_active_every_late_bin'] for r in valid),
                  silent_late_cases=[sum(r['late_spike_count'][i]==0 for r in valid) for i in range(2)],
                  mean_late_rate_hz=[float(np.mean([r['late_rate_hz'][i] for r in valid])) if valid else None for i in range(2)],
                  episodes=rows)
    write_json(run/'activity.json', report)
    return report


def refresh_study(root, plan):
    """Report completed searches; rank runs only by their saved validation fitness."""
    rows = []
    for item in plan:
        run = root/item['run']
        if not (run/'activity.json').exists():
            continue
        result = json.loads((run/'result.json').read_text())
        measured = json.loads((run/'activity.json').read_text())
        rows.append(dict(run=item['run'], mutation_probability=item['training']['mutation_probability'],
                         test_completed=result['test_completed'],
                         activity_complete_cases=measured['complete_late_windows'],
                         activity_both_each_bin=measured['both_active_every_late_bin_cases'],
                         **table_row(item['profile'], item['training'], result)))
    write_json(root/'comparison.json', rows)
    if rows: write_csv(root/'comparison.csv', rows)
    groups = [('baseline', [r for r in rows if '/' not in r['run']], 7)]
    groups += [(f'expanded_{task}', [dict(r, run=r['run'].split('/',1)[1]) for r in rows
                                     if r['run'].startswith(f'expanded_{task}/')], 12) for task in EXPERIMENTS]
    for name, measured, expected in groups:
        write_json(root/f'{name}.json', measured)
        if measured: write_csv(root/f'{name}.csv', measured)
        if measured and name!='baseline':
            # Canonical planned order is G, seed, mutation probability. Ties keep the first.
            selected = max(measured, key=lambda r:r['validation_fitness'])
            write_json(root/f'{name}_selection.json', dict(status='completed' if len(measured)==expected else 'in_progress',
                selection='maximum validation fitness; ties keep first in planned order', test_used_for_selection=False,
                completed=len(measured), expected=expected, selected=selected))
    lines=['| Run | G | Seed | p | Training F; success | Validation F; success | Test F; success | Both neurons active in each late 1 s bin / completed |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        cells=[f"{row[f'{bank}_fitness']:.6f}; {row[f'{bank}_success']}/{row[f'{bank}_cases']}" for bank in ('training','validation','test')]
        lines.append(f"| {row['run']} | {row['generations']} | {row['seed']} | {row['mutation_probability']} | "+' | '.join(cells)+f" | {row['activity_both_each_bin']}/{row['activity_complete_cases']} |")
    content=f"Completed **{len(rows)}/{len(plan)}** searches, including test activity replay.\n\n"+'\n'.join(lines)
    (root/'RESULTS.md').write_text(content+'\n')
    readme=HERE/'README.md'
    if readme.exists():
        text=readme.read_text(); start='<!-- STUDY_RESULTS_START -->'; end='<!-- STUDY_RESULTS_END -->'
        if start in text and end in text:
            a=text.index(start)+len(start); b=text.index(end,a)
            readme.write_text(text[:a]+'\n'+content+'\n'+text[b:])
    return rows


def export_study(root, rows):
    """Publish only completed constrained-study winners; keep training inputs untouched."""
    import numpy as np
    import numba
    import shutil
    selected=[]
    for task in EXPERIMENTS:
        selection=json.loads((root/f'expanded_{task}_selection.json').read_text())
        assert selection['status']=='completed'
        selected.append((task,'fast',root/f'expanded_{task}'/selection['selected']['run']))
        selected.append((task,'slow',root/f'{task}_slow'))
    selected.append(('C','mirror',root/'C_mirror'))
    published=[]
    for task,domain,run in selected:
        folder=HERE/EXPERIMENTS[task]
        (folder/'genomes').mkdir(exist_ok=True); (folder/'reference').mkdir(exist_ok=True)
        target=folder/'genomes'/f'{domain}.json'; reference=folder/'reference'/f'{domain}.npz'
        shutil.copy2(run/'best_genome.json',target)
        result=json.loads((run/'result.json').read_text())
        stage=run/('polynomial' if (run/'polynomial').exists() else 'training')
        with np.load(run/'test_metrics.npz') as m, np.load(stage/'cases.npz') as c:
            np.savez_compressed(reference, metrics=m['metrics'], scores=m['scores'], cases=c['test'])
        published.append(dict(experiment=task,domain=domain,genome=str(target.relative_to(HERE)),
            source=str((run/'best_genome.json').relative_to(HERE.parent)),
            validation_fitness=result['validation_fitness'],test_success=result['test_success'],test_cases=result['test_cases'],
            original_file_sha256=file_hash(target),published_file_sha256=file_hash(target),
            reference_metrics=str(reference.relative_to(HERE)),reference_metrics_sha256=file_hash(reference),
            training=dict(initialization='configuration',archive_genomes_used=False,
                          options=json.loads((run/'training.json').read_text()))))
    write_json(HERE/'published_results.json',published)
    subprocess.run([sys.executable,str(HERE/'verify.py'),'--workers',str(min(8,numba.config.NUMBA_NUM_THREADS)),
                    '--output',str(HERE/'verification.json')],check=True,cwd=HERE)


def study(output, workers=10, jobs=2):
    """Run the predeclared 36-case fast grid and seven independent baseline profiles.

    This is the P_cross < 0.1 study. Activity is diagnostic only. Every run starts
    from generic configuration. No resume, implicit warm start or test-based search.
    """
    import numpy as np
    import numba
    if jobs < 1 or workers < 1 or workers > numba.config.NUMBA_NUM_THREADS:
        raise ValueError('Positive process/worker counts within the Numba thread limit are required')
    root=Path(output).resolve()
    root.mkdir(parents=True,exist_ok=False)
    (root/'logs').mkdir(); (root/'plan_inputs').mkdir()
    plan=[]
    def add(label, relative, overrides):
        config,settings,_=inputs(label)
        settings.update(overrides,workers=workers)
        for i in (49,50):
            p=config['parameters'][i]
            assert p['minimum']==0 and p['maximum']==float(np.nextafter(.1,0.))
            assert p['value']==0 and p['mutable']
        key=relative.replace('/','_')
        cfg=root/'plan_inputs'/f'{key}.config.json'; training=root/'plan_inputs'/f'{key}.training.json'
        write_json(cfg,config);write_json(training,settings)
        command=[sys.executable,str(HERE/'run.py'),'train',label[0],'--config',str(cfg),'--training',str(training),'--output',str(root/relative)]
        plan.append(dict(profile=label,run=relative,configuration=config,training=settings,command=command))
    for label in ('A:fast','B:fast','C:fast','A:slow','B:slow','C:slow','C:mirror'):
        add(label,label.replace(':','_'),{})
    for budget in (200,500,1000):
        for seed in (42,43):
            for probability in (.1,.3):
                for task in EXPERIMENTS:
                    add(task+':fast',f'expanded_{task}/G{budget}/N50_p{probability:g}_seed{seed}',
                        dict(population=50,generations=budget,seed=seed,mutation_probability=probability))
    state=dict(study='cross_output_20261116',status='running',fitness_changed=False,
               cross_output_upper_exclusive=.1,archive_genomes_used=False,
               activity_used_for_selection=False,workers_per_process=workers,processes=jobs,
               expected=len(plan),completed=0,runs=plan)
    write_json(root/'study.json',state);refresh_study(root,plan)
    def execute(item):
        run=root/item['run']; log=root/'logs'/(item['run'].replace('/','_')+'.log')
        with log.open('w') as stream:
            print('Started',item['run'],flush=True)
            subprocess.run(item['command'],cwd=HERE,stdout=stream,stderr=subprocess.STDOUT,check=True)
            subprocess.run([sys.executable,str(HERE/'run.py'),'activity','--run',str(run)],cwd=HERE,
                           stdout=stream,stderr=subprocess.STDOUT,check=True)
        return item
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures=[pool.submit(execute,item) for item in plan]
        try:
            for future in as_completed(futures):
                item=future.result(); print('Completed',item['run'],flush=True)
                rows=refresh_study(root,plan);state['completed']=len(rows);write_json(root/'study.json',state)
        except Exception as exc:
            for future in futures:future.cancel()
            state.update(status='failed',error=str(exc));write_json(root/'study.json',state)
            raise
    rows=refresh_study(root,plan)
    state.update(status='finalizing',completed=len(rows));write_json(root/'study.json',state)
    try:
        export_study(root,rows)
    except Exception as exc:
        state.update(status='failed',error=str(exc));write_json(root/'study.json',state)
        raise
    state.update(status='completed');write_json(root/'study.json',state)
    write_json(root/'manifest.json',dict(sha256={str(p.relative_to(root)):file_hash(p) for p in sorted(root.rglob('*'))
                                                 if p.is_file() and p.name!='manifest.json'}))
    print('Constrained study completed; seven new validation-selected controllers exported.',flush=True)
    return rows


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # Route old experiment commands through the same parser and implementation.
    if argv and argv[0] in ('train','sweep','check','simulate'):
        if len(argv) < 2 or argv[1] not in EXPERIMENTS:
            raise SystemExit('Specify A, B or C after the command, e.g. python run.py train A --profile fast')
        command, task = argv[:2]
        module = importlib.import_module(f'{EXPERIMENTS[task]}.experiment')
        return cli(EXPERIMENTS[task], HERE/EXPERIMENTS[task], module.make_banks,
                   getattr(module, 'train_experiment', None), [command, *argv[2:]])
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('suite', help='Independent fresh training of all seven article domains')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--profiles', nargs='+', choices=PROFILES, default=list(PROFILES))
    for option in ('population','generations','seed','workers'):
        p.add_argument(f'--{option}', type=int)
    p = sub.add_parser('repeat', help='Repeat a fresh search from saved inputs with unchanged code')
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int)
    p = sub.add_parser('activity', help='Measure both neurons on all tests of a frozen run; no selection')
    p.add_argument('--run', type=Path, required=True)
    p = sub.add_parser('study', help='Fresh P_cross < 0.1 sweep and seven baseline profiles; 43 searches')
    p.add_argument('--output', type=Path, default=Path('fresh_results'))
    p.add_argument('--workers', type=int, default=10)
    p.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args(argv)
    try:
        if args.command == 'activity':
            report=activity(args.run)
            print(json.dumps({k:v for k,v in report.items() if k!='episodes'},indent=2))
            return report
        if args.command == 'study':
            return study(args.output,args.workers,args.jobs)
        if args.command == 'repeat':
            return repeat(args.run, args.output, args.workers)
        if len(set(args.profiles)) != len(args.profiles):
            raise ValueError('Suite profiles must not repeat')
        overrides = {key:getattr(args,key) for key in ('population','generations','seed','workers')
                     if getattr(args,key) is not None}
        return suite(args.output, profiles=args.profiles, overrides=overrides)
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
