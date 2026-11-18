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
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

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
    rate_limit=genome.controller.data['simulation'].get('max_firing_rate_hz')
    currents=genome.currents(command,cases)
    rows = []
    for index, case in enumerate(cases):
        # Preserve the stored SI case bit-for-bit, including any perturbation.
        # A radians -> degrees -> radians round trip can alter a spike threshold.
        metrics, trace = simulate(physical,case,genome.controller.mechanics,
                                  currents=currents[index],duration_s=duration,stride=1,
                                  fitness_window_s=window,max_firing_rate_hz=rate_limit)
        replay=dict(metrics=metrics,trace=trace,duration_s=duration,fitness_window_s=window,
                    max_firing_rate_hz=rate_limit)
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
    if rate_limit is not None:
        report.update(max_firing_rate_hz=rate_limit,
                      rate_limit_definition='each ISI >= 10 ms at 100 Hz; full observed episode, including transients',
                      rate_limit_observed_pass_cases=sum(r['rate_limit_observed_pass'] for r in rows),
                      rate_limit_failed_cases=sum(not r['rate_limit_observed_pass'] for r in rows),
                      max_instantaneous_rate_hz=[max((r['peak_instantaneous_rate_hz'][i] or 0.) for r in rows) for i in range(2)])
    write_json(run/'activity.json', report)
    return report


def refresh_study(root, plan, readme_marker='STUDY_RESULTS'):
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
        if result.get('max_firing_rate_hz') is not None:
            rows[-1].update(rate_limit_feasible=result['rate_limit_feasible'],
                            max_instantaneous_rate_hz=measured['max_instantaneous_rate_hz'],
                            rate_limit_failed_cases=measured['rate_limit_failed_cases'])
    write_json(root/'comparison.json', rows)
    if rows: write_csv(root/'comparison.csv', rows)
    groups = [('baseline', [r for r in rows if '/' not in r['run']], sum('/' not in p['run'] for p in plan))]
    groups += [(f'expanded_{task}', [dict(r, run=r['run'].split('/',1)[1]) for r in rows
                                     if r['run'].startswith(f'expanded_{task}/')],
                sum(p['run'].startswith(f'expanded_{task}/') for p in plan)) for task in EXPERIMENTS]
    for name, measured, expected in groups:
        write_json(root/f'{name}.json', measured)
        if measured: write_csv(root/f'{name}.csv', measured)
        if measured and name!='baseline':
            # Ties keep the first candidate in the declared plan order.
            selected = max(measured, key=lambda r:r['validation_fitness'])
            write_json(root/f'{name}_selection.json', dict(status='completed' if len(measured)==expected else 'in_progress',
                selection='maximum validation fitness; ties keep first in planned order', test_used_for_selection=False,
                completed=len(measured), expected=expected, selected=selected))
    lines=['| Run | N | G | Seed | p | Training F; success | Validation F; success | Test F; success | Both neurons active in each late 1 s bin / completed |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        cells=[f"{row[f'{bank}_fitness']:.6f}; {row[f'{bank}_success']}/{row[f'{bank}_cases']}" for bank in ('training','validation','test')]
        lines.append(f"| {row['run']} | {row['population']} | {row['generations']} | {row['seed']} | {row['mutation_probability']} | "+' | '.join(cells)+f" | {row['activity_both_each_bin']}/{row['activity_complete_cases']} |")
    if any('rate_limit_feasible' in row for row in rows):
        lines += ['', '| Run | Rate feasible: train / validation / test | Peak instantaneous Hz: neuron 0 / 1 | Test episodes violating rate |',
                  '|---|---|---:|---:|']
        for row in rows:
            if 'rate_limit_feasible' not in row: continue
            feasible=' / '.join('yes' if row['rate_limit_feasible'][b] else 'no' for b in ('training','validation','test'))
            peak=' / '.join(f'{x:.3f}' for x in row['max_instantaneous_rate_hz'])
            lines.append(f"| {row['run']} | {feasible} | {peak} | {row['rate_limit_failed_cases']} |")
    content=f"Completed **{len(rows)}/{len(plan)}** searches, including test activity replay.\n\n"+'\n'.join(lines)
    (root/'RESULTS.md').write_text(content+'\n')
    readme=HERE/'README.md'
    if readme_marker and readme.exists():
        text=readme.read_text(); start=f'<!-- {readme_marker}_START -->'; end=f'<!-- {readme_marker}_END -->'
        if start in text and end in text:
            a=text.index(start)+len(start); b=text.index(end,a)
            readme.write_text(text[:a]+'\n'+content+'\n'+text[b:])
    return rows


def live_leaderboard(root, plan, persist=False):
    """Read atomic incumbent pointers; rank comparable FINAL-stage validations.

    C's table stage uses a different validation bank and is never ranked next
    to polynomial controllers. Test metrics are displayed only after the
    frozen winner has been evaluated; they never enter ranking.
    """
    root=Path(root).resolve()
    rows=[];preparatory=[];started=completed=0
    for item in plan:
        run=root/item['run'];settings=item['training']
        started+=int((run/'training.json').exists())
        done=(run/'activity.json').exists();completed+=int(done)
        stage=run/('polynomial' if settings.get('method')=='staged' else 'training')
        if not (stage/'selection.json').exists():
            table=run/'table'/'progress.json'
            if table.exists():preparatory.append(dict(run=item['run'],**json.loads(table.read_text())))
            continue
        selected=json.loads((stage/'selection.json').read_text())
        progress=json.loads((stage/'progress.json').read_text()) if (stage/'progress.json').exists() else {}
        result=json.loads((run/'result.json').read_text()) if (run/'result.json').exists() else None
        archive=stage/selected.get('genome','best_genome.json')
        row=dict(profile=item['profile'],run=item['run'],
                 status='completed' if done else ('test_activity' if result else 'running'),
                 population=settings['population'],generations=settings['generations'],
                 seed=settings['seed'],mutation_probability=settings['mutation_probability'],
                 selected_generation=selected['generation'],stage=stage.name,
                 current_stage_generation=progress.get('generation',selected['generation']),
                 stage_generations=progress.get('generations',settings['generations']),
                 validation_fitness=selected['fitness'],
                 validation_success=selected.get('validation_success'),validation_cases=selected.get('validation_cases'),
                 validation_rate_feasible=selected.get('rate_limit_feasible',selected['fitness']>=0),
                 training_fitness=None,training_success=None,training_cases=None,
                 test_fitness=None,test_success=None,test_cases=None,
                 genome=str(archive.relative_to(root)),genome_sha256=selected.get('genome_sha256'))
        if result:
            for bank in ('training','validation','test'):
                for key in ('fitness','success','cases'):row[f'{bank}_{key}']=result[bank][key]
        rows.append(row)
    profiles=list(dict.fromkeys(p['profile'] for p in plan))
    def winners(candidates):
        return [max(group,key=lambda r:r['validation_fitness']) for profile in profiles
                if (group:=[r for r in candidates if r['profile']==profile])]
    leaders=winners(rows);finished=winners([r for r in rows if r['status']=='completed'])
    report=dict(expected=len(plan),started=started,completed=completed,queued=len(plan)-started,
                active=started-completed,selection='maximum validation fitness; ties retain plan order',
                test_used_for_selection=False,leaders=leaders,completed_leaders=finished,
                table_stage_runs=preparatory,runs=rows)
    def render(chosen,total,complete):
        lines=[f'Completed **{complete}/{total}** searches. Running candidates are provisional; ranking uses validation only.',
               '', '| Profile | Status | N | G budget | Best stage generation | Seed | p | Training F; success | Validation F; success | Test F; success | Rate feasible on validation |',
               '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
        for row in chosen:
            banks=[]
            for bank in ('training','validation','test'):
                value=row[f'{bank}_fitness'];success=row[f'{bank}_success'];cases=row[f'{bank}_cases']
                banks.append('—' if value is None else f'{value:.6f}; {success}/{cases}')
            lines.append(f"| {row['profile']} | {row['status']} | {row['population']} | {row['generations']} | {row['selected_generation']} | {row['seed']} | {row['mutation_probability']} | "+' | '.join(banks)+f" | {row['validation_rate_feasible']} |")
        if not chosen:lines+=['','No final-stage validation candidate is available yet.']
        for row in chosen:
            lines+=['',f"{row['profile']}: `{row['run']}`; genome `{row['genome']}`."]
        return '\n'.join(lines)
    content=render(leaders,len(plan),completed)
    report['markdown']=content
    if persist:
        write_json(root/'leaderboard.json',{k:v for k,v in report.items() if k!='markdown'})
        if rows:write_csv(root/'leaderboard.csv',rows)
        (root/'leaderboard.md').write_text(content+'\n')
        def update_readme(path,marker,text):
            if not path.exists():return
            source=path.read_text();start=f'<!-- {marker}_START -->';end=f'<!-- {marker}_END -->'
            if start in source and end in source:
                a=source.index(start)+len(start);b=source.index(end,a)
                path.write_text(source[:a]+'\n'+text+'\n'+source[b:])
        tasks=list(dict.fromkeys(item['profile'][0] for item in plan))
        for task in tasks:
            task_plan=[p for p in plan if p['profile'][0]==task]
            task_done=sum((root/p['run']/'activity.json').exists() for p in task_plan)
            block=render([r for r in leaders if r['profile'][0]==task],len(task_plan),task_done)
            update_readme(HERE/EXPERIMENTS[task]/'README.md',f'GRID_{task}_RESULTS',block)
            update_readme(HERE/'README.md',f'GRID_{task}_RESULTS',block)
    return report


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


def study_specs(bc_rate100=False):
    """Return the declared, deterministic order without reading any trained genome."""
    plan=[]
    baseline=('B:slow','C:slow','C:mirror') if bc_rate100 else PROFILES
    for label in baseline:
        plan.append((label,label.replace(':','_'),dict(generations=500) if bc_rate100 else {}))
    for budget in ((200,500) if bc_rate100 else (200,500,1000)):
        for seed in (42,43):
            for probability in (.1,.3):
                for task in (('B','C') if bc_rate100 else EXPERIMENTS):
                    plan.append((task+':fast',f'expanded_{task}/G{budget}/N50_p{probability:g}_seed{seed}',
                                 dict(population=50,generations=budget,seed=seed,mutation_probability=probability)))
    return plan


def inclusive_range(values):
    """Expand a CLI start/stop/step triple, with an inclusive aligned stop."""
    start,stop,step=values
    if step<=0 or stop<start or (stop-start)%step:
        raise ValueError('Use START <= STOP and positive STEP dividing STOP-START exactly')
    return list(range(start,stop+1,step))


def grid_specs(experiments=('B','C'), populations=tuple(range(10,51,5)),
               generations=tuple(range(100,501,50)), seeds=(42,43), mutation_probability=.3):
    """A complete fresh fast-profile grid; order is G, population, seed, task."""
    axes=[experiments,populations,generations,seeds]
    if any(not axis or len(set(axis))!=len(axis) for axis in axes):
        raise ValueError('Grid axes must be nonempty and contain distinct values')
    if any(task not in ('B','C') for task in experiments):
        raise ValueError('This constrained grid supports experiments B and C')
    for axis,minimum in ((populations,3),(generations,2),(seeds,0)):
        if any(isinstance(x,bool) or not isinstance(x,int) or x<minimum for x in axis):
            raise ValueError('Grid needs population >=3, generations >=2 and nonnegative integer seeds')
    if not 0<=mutation_probability<=1:
        raise ValueError('Mutation probability must lie in [0,1]')
    return [(task+':fast',f'expanded_{task}/G{budget}/N{size}_p{mutation_probability:g}_seed{seed}',
             dict(population=size,generations=budget,seed=seed,mutation_probability=mutation_probability))
            for budget in generations for size in populations for seed in seeds for task in experiments]


def export_bc_study(root, experiments=('B','C'), include_baselines=True, suffix='rate100'):
    """Export validation-feasible B/C candidates; report test failures without reselection."""
    import numpy as np
    import numba
    import shutil
    from verify import verify
    selected=[]
    for task in experiments:
        selection=json.loads((root/f'expanded_{task}_selection.json').read_text())
        assert selection['status']=='completed'
        selected.append((task,'fast',root/f'expanded_{task}'/selection['selected']['run']))
        if include_baselines: selected.append((task,'slow',root/f'{task}_slow'))
    if include_baselines and 'C' in experiments: selected.append(('C','mirror',root/'C_mirror'))
    published=[]; excluded=[]
    for task,domain,run in selected:
        result=json.loads((run/'result.json').read_text())
        if not result['rate_limit_feasible']['validation']:
            excluded.append(dict(experiment=task,domain=domain,run=str(run.relative_to(HERE)),
                                 reason='No validation-feasible genome found; candidate remains in run directory'))
            continue
        folder=HERE/EXPERIMENTS[task]
        (folder/'genomes').mkdir(exist_ok=True);(folder/'reference').mkdir(exist_ok=True)
        target=folder/'genomes'/f'{domain}_{suffix}.json';reference=folder/'reference'/f'{domain}_{suffix}.npz'
        shutil.copy2(run/'best_genome.json',target)
        stage=run/('polynomial' if (run/'polynomial').exists() else 'training')
        with np.load(run/'test_metrics.npz') as metrics, np.load(stage/'cases.npz') as cases:
            np.savez_compressed(reference,metrics=metrics['metrics'],scores=metrics['scores'],cases=cases['test'])
        published.append(dict(experiment=task,domain=domain,genome=str(target.relative_to(HERE)),
            source=str((run/'best_genome.json').relative_to(HERE.parent)),
            validation_fitness=result['validation_fitness'],test_success=result['test_success'],test_cases=result['test_cases'],
            rate_limit_feasible=result['rate_limit_feasible'],max_firing_rate_hz=100.,
            original_file_sha256=file_hash(target),published_file_sha256=file_hash(target),
            reference_metrics=str(reference.relative_to(HERE)),reference_metrics_sha256=file_hash(reference),
            training=dict(initialization='configuration',archive_genomes_used=False,
                          options=json.loads((run/'training.json').read_text()))))
    write_json(root/'published_results.json',published)
    catalogue=HERE/f'{suffix}_results.json'
    existing=json.loads(catalogue.read_text()) if catalogue.exists() else []
    replaced={(task,domain) for task,domain,_ in selected}
    write_json(catalogue,[row for row in existing if (row['experiment'],row['domain']) not in replaced]+published)
    write_json(root/'export.json',dict(exported=len(published),excluded=excluded,test_used_for_selection=False))
    verification=(verify(min(8,numba.config.NUMBA_NUM_THREADS),root/'published_results.json') if published
                  else dict(status='skipped',reason='No validation-feasible candidates',genomes=0))
    write_json(root/'verification.json',verification)
    return published


def study(output, workers=10, jobs=2, bc_rate100=False, grid=None, plan_only=False):
    """Run a predeclared fresh plan, with bounded parallelism and live incumbents."""
    import numpy as np
    import numba
    if jobs < 1 or workers < 1 or workers > numba.config.NUMBA_NUM_THREADS:
        raise ValueError('Positive process/worker counts within the Numba thread limit are required')
    root=Path(output).resolve()
    # Validate the entire plan before creating output files.
    if grid is not None: bc_rate100=True
    specs=grid_specs(**grid) if grid is not None else study_specs(bc_rate100)
    for label,relative,overrides in specs:
        config,settings,_=inputs(label)
        from evolution import validate_search, make_genome
        settings.update(overrides,workers=workers);validate_search(settings);make_genome(config)
        rate=config['simulation'].get('max_firing_rate_hz')
        if bc_rate100:
            if rate!=100.: raise ValueError('study-bc requires max_firing_rate_hz=100 in both B/C configurations')
            for i in (31,34,37,40):
                if (config['parameters'][i]['minimum'],config['parameters'][i]['maximum'])!=(-50,50):
                    raise ValueError('study-bc requires four II-afferent weights in [-50,50]')
            if not all(config['parameters'][i]['mutable'] for i in (17,24)):
                raise ValueError('study-bc requires mutable muscle N')
        elif rate is not None:
            raise ValueError('The historical 43-run study has no rate constraint. Use study-bc for current B/C inputs, or restore a historical source snapshot.')
    if plan_only:
        report=dict(searches=len(specs),profiles=list(dict.fromkeys(label for label,_,_ in specs)),
                    grid=grid,workers_per_process=workers,processes=jobs,
                    initialization='configuration',archive_genomes_used=False,
                    training_performed=False,output=str(root))
        print(json.dumps(report,indent=2));return report
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
    for label,relative,overrides in specs: add(label,relative,overrides)
    from datetime import datetime, timezone
    marker=None if grid is not None else ('BC_RATE100_RESULTS' if bc_rate100 else 'STUDY_RESULTS')
    state=dict(study='bc_grid_rate100_20261117' if grid is not None else ('bc_rate100_20261117' if bc_rate100 else 'cross_output_20261116'),
               status='running',fitness_changed=bc_rate100,neuron_dynamics_changed=False,
               max_firing_rate_hz=100. if bc_rate100 else None,
               started_utc=datetime.now(timezone.utc).isoformat(),
               cross_output_upper_exclusive=.1,archive_genomes_used=False,
               coactivity_used_for_selection=False,test_activity_used_for_selection=False,
               firing_rate_used_for_selection=bc_rate100,workers_per_process=workers,processes=jobs,
               expected=len(plan),completed=0,grid=grid,runs=plan)
    write_json(root/'study.json',state);refresh_study(root,plan,marker)
    if grid is not None:live_leaderboard(root,plan,persist=True)
    def execute(item):
        run=root/item['run']; log=root/'logs'/(item['run'].replace('/','_')+'.log')
        with log.open('w') as stream:
            print('Started',item['run'],flush=True)
            subprocess.run(item['command'],cwd=HERE,stdout=stream,stderr=subprocess.STDOUT,check=True)
            subprocess.run([sys.executable,str(HERE/'run.py'),'activity','--run',str(run)],cwd=HERE,
                           stdout=stream,stderr=subprocess.STDOUT,check=True)
        return item
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures={pool.submit(execute,item) for item in plan}
        try:
            while futures:
                done,pending=wait(futures,timeout=15,return_when=FIRST_COMPLETED)
                for future in done:
                    item=future.result(); print('Completed',item['run'],flush=True)
                if done:
                    rows=refresh_study(root,plan,marker);state['completed']=len(rows);write_json(root/'study.json',state)
                if grid is not None:live_leaderboard(root,plan,persist=True)
                futures=pending
        except Exception as exc:
            for future in futures:future.cancel()
            state.update(status='failed',error=str(exc));write_json(root/'study.json',state)
            raise
    rows=refresh_study(root,plan,marker)
    state.update(status='finalizing',completed=len(rows));write_json(root/'study.json',state)
    try:
        if grid is not None: export_bc_study(root,experiments=grid['experiments'],include_baselines=False,suffix='grid_rate100')
        elif bc_rate100: export_bc_study(root)
        else: export_study(root,rows)
    except Exception as exc:
        state.update(status='failed',error=str(exc));write_json(root/'study.json',state)
        raise
    state.update(status='completed',finished_utc=datetime.now(timezone.utc).isoformat());write_json(root/'study.json',state)
    if grid is not None:live_leaderboard(root,plan,persist=True)
    write_json(root/'manifest.json',dict(sha256={str(p.relative_to(root)):file_hash(p) for p in sorted(root.rglob('*'))
                                                 if p.is_file() and p.name!='manifest.json'}))
    print('Study completed; export and verification reports saved.',flush=True)
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
    p = sub.add_parser('grid', help='Fresh B/C fast sweep over population, generation budget and seeds; live genome checkpoints')
    p.add_argument('--experiments',nargs='+',choices=('B','C'),default=['B','C'])
    p.add_argument('--population-range',type=int,nargs=3,metavar=('START','STOP','STEP'),default=[10,50,5])
    p.add_argument('--generation-range',type=int,nargs=3,metavar=('START','STOP','STEP'),default=[100,500,50])
    p.add_argument('--seeds',type=int,nargs='+',default=[42,43])
    p.add_argument('--mutation-probability',type=float,default=.3)
    p.add_argument('--output',type=Path,default=Path('results/bc_grid_20261117'))
    p.add_argument('--workers',type=int,default=10)
    p.add_argument('--jobs',type=int,default=2)
    p.add_argument('--plan-only',action='store_true',help='Validate the entire grid without starting training or writing files')
    p = sub.add_parser('status',help='Read live validation leaders and their hyperparameters without changing the study')
    p.add_argument('--output',type=Path,required=True,help='Study output directory containing study.json')
    p = sub.add_parser('study-bc', help='B/C: II weights +/-50, mutable N, 100 Hz ISI constraint; 19 searches up to G500')
    p.add_argument('--output', type=Path, default=Path('results/bc_rate100_20261117'))
    p.add_argument('--workers', type=int, default=10)
    p.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args(argv)
    try:
        if args.command=='grid':
            grid=dict(experiments=args.experiments,populations=inclusive_range(args.population_range),
                      generations=inclusive_range(args.generation_range),seeds=args.seeds,
                      mutation_probability=args.mutation_probability)
            return study(args.output,args.workers,args.jobs,grid=grid,plan_only=args.plan_only)
        if args.command=='status':
            state=json.loads((args.output/'study.json').read_text())
            report=live_leaderboard(args.output,state['runs'])
            print(f"Study {state['status']}; active {report['active']}, queued {report['queued']}")
            print(report['markdown']);return report
        if args.command == 'activity':
            report=activity(args.run)
            print(json.dumps({k:v for k,v in report.items() if k!='episodes'},indent=2))
            return report
        if args.command in ('study','study-bc'):
            return study(args.output,args.workers,args.jobs,args.command=='study-bc')
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
