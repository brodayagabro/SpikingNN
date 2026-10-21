"""Resume the historical October-16 series in its immutable factor-1 runtime.

Run with .venv/bin/python scripts/AfferentBalance2026/recompute_unscaled.py all.
Results and source snapshots are deliberately outside the Git index.
"""
import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
BASE = REPO/'scripts/AfferentBalance2026'
sys.path.insert(0, str(REPO))
from scripts.AfferentBalance2026.runtime_support import (
    activate_runtime, core_module, current_results, resolve_recorded_path,
    runtime_directory, runtime_package_directory)

DEFAULT = BASE/'report/results/recompute_unscaled_20261016'


def write(path, data):
    """Atomically replace a pipeline JSON status or pointer file."""
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def prepare(root):
    """Verify an existing historical snapshot without importing the new ABC core.

    New studies must use experiments/A_balance or B_balance entry points.
    Historical files retain the old current-slot genome and their exact sources.
    """
    runtime = Path(root) / "factor_1/runtime"
    provenance = runtime.parent / "runtime_provenance.json"
    if not provenance.is_file():
        raise ValueError("Historical runtime missing; use A_balance/B_balance for new studies")
    for name, expected in json.loads(provenance.read_text())["runtime_sha256"].items():
        if hashlib.sha256((runtime / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen runtime source changed: {name}")
    runtime_package_directory(runtime)
    return runtime


def verify_runtime(root,runtime):
    """Test the exact frozen sources, and reuse only their matching test record."""
    signature=hashlib.sha256((root/'factor_1/runtime_provenance.json').read_bytes()).hexdigest()
    record=root/'preflight.json'
    if record.exists() and json.loads(record.read_text()).get('runtime_provenance_sha256')==signature:
        if json.loads(record.read_text()).get('status')=='passed':return
    env=dict(os.environ,PYTHONPATH=str(runtime),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
             MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1',MPLBACKEND='Agg')
    with (root/'preflight.log').open('w') as log:
        subprocess.run([sys.executable,'-m','pytest',str(runtime_package_directory(runtime)/'tests'),
                        '-q','-p','no:cacheprovider',f'--junitxml={root / "runtime_tests.xml"}'],
                       cwd=runtime,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    shutil.copy2(root/'runtime_tests.xml',root/'tests.xml')
    write(record,dict(status='passed',runtime_provenance_sha256=signature))


def winner(sweep):
    """Locate the final population of the sweep checkpoint selected by validation."""
    analysis = json.loads((sweep/'analysis.json').read_text())
    return sweep/analysis['best_by_validation']['directory']/'final_population.npz'


def train_once(experiment, output, **kwargs):
    """Allow a planned run to start early without racing the main pipeline."""
    import fcntl
    run = core_module('run').run
    output.parent.mkdir(parents=True,exist_ok=True)
    with (output.parent/(output.name+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not (output/'result.json').exists():
            return run(experiment,output,**kwargs)
        return json.loads((output/'result.json').read_text())


def replay_once(sweep, output, workers=8):
    """Acquire a replay lock and reuse a completed result or calculate missing trials."""
    import fcntl
    run_replay = core_module('wide_replay').run_replay
    output.parent.mkdir(parents=True,exist_ok=True)
    with (output.parent/(output.name+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        progress=output/'progress.json'
        if progress.exists() and json.loads(progress.read_text())['finished']:
            return json.loads((output/'summary.json').read_text())
        return run_replay(sweep,output,workers=workers)


def stage(root, name):
    """Execute one scheduled experiment stage with the selected frozen factor-1 runtime."""
    runtime = root/'factor_1/runtime'
    activate_runtime(runtime)
    MECHANICS = core_module('config').MECHANICS
    assert MECHANICS.muscle_torque_factor == 1.
    run = core_module('run').run
    Schedule = core_module('evolution').Schedule
    schedule = Schedule(probability_start=.3, probability_end=.3)
    if name in ('equilibrium', 'small_nm_weights'):
        launch = core_module('sweep').launch
        audit = core_module('sweep_verification').audit
        out = root/name/'sweep'
        launch(out, experiment=name, jobs=2, workers_per_line=4, resume=out.exists())
        audit(out)
    elif name == 'extra_balance':
        wide_balance_banks = core_module('cases').wide_balance_banks
        WIDE_TRAINING_SEEDS = core_module('cases').WIDE_TRAINING_SEEDS
        for seed in (43, 44):
            out = root/'equilibrium/replicates'/f'seed_{seed}'
            if not (out/'result.json').exists():
                train_once('equilibrium', out, population_size=50, generations=200, seed=seed,
                    workers=8, plot=False, checkpoint_every=10, schedule=schedule)
        # Three four-process pools plus the eight-process command study fit
        # twenty CPUs. A run started earlier is joined through its file lock.
        with ThreadPoolExecutor(3) as executor:
            futures=[]
            for seed in (42, 43, 44):
                out = root/'A1'/f'seed_{seed}'
                futures.append(executor.submit(train_once,'equilibrium',out,
                    population_size=50,generations=200,seed=seed,workers=4,
                    plot=False,checkpoint_every=10,schedule=schedule,
                    case_banks=wide_balance_banks(),case_seeds=WIDE_TRAINING_SEEDS))
            for future in futures:future.result()
    elif name == 'angle_setpoint':
        run_study = core_module('polynomial_study').run_study
        source = winner(root/'small_nm_weights/sweep')
        out = root/'angle_setpoint/polynomial'
        run_study(out, source, workers=8, resume=(out/'protocol.json').exists())
        out = root/'angle_setpoint/compensated_cubic'
        if not (out/'result.json').exists():
            run('angle_setpoint', out, population_size=50, generations=200, seed=42,
                population_from=source, workers=8, plot=False, checkpoint_every=10, schedule=schedule)
    elif name == 'replays':
        for experiment in ('equilibrium', 'small_nm_weights'):
            out = root/experiment/'wide_replay'
            replay_once(root/experiment/'sweep',out,workers=8)
        diagnose(root)
    else:
        raise ValueError(name)


def diagnose(root):
    """Evaluate A1 transfer and distant setpoint starts after validation selection is complete."""
    import numpy as np
    banks = core_module('cases').banks
    wide_balance_banks = core_module('cases').wide_balance_banks
    balance_cases = core_module('cases').balance_cases
    SMALL_TRANSFER_SEED = core_module('cases').SMALL_TRANSFER_SEED
    PopulationEvaluator = core_module('population').PopulationEvaluator
    summarize = core_module('polynomial_study').summarize
    full_command_trace = core_module('polynomial_study').full_command_trace
    wide_cases = core_module('wide_replay').wide_cases
    full_replay = core_module('wide_replay').full_replay
    TRACE_COLUMNS = core_module('dynamics').TRACE_COLUMNS
    METRIC_COLUMNS = core_module('dynamics').METRIC_COLUMNS
    write_json = core_module('run').write_json
    write_csv = core_module('run').write_csv
    metric_rows = core_module('run').metric_rows
    protocols = {'large_test':wide_balance_banks()['test'], 'wide_grid':wide_cases(),
                 'small_test':banks('equilibrium')['test'], 'small_fresh':balance_cases(SMALL_TRANSFER_SEED,96)}
    out = root/'A1/diagnostics';out.mkdir(exist_ok=True, parents=True)
    paths = [root/'equilibrium/sweep/mutation_0.300/population_050/seed_42/epochs_200']
    paths += [root/'equilibrium/replicates'/f'seed_{seed}' for seed in (43,44)]
    paths += [root/'A1'/f'seed_{seed}' for seed in (42,43,44)]
    rows = []
    with PopulationEvaluator(8) as evaluator:
        for path in paths:
            with np.load(path/'final_population.npz') as data:
                p = data['physical_controllers'][int(data['selected_index'])]
            row = dict(directory=str(path.relative_to(root)),banks={})
            for name, cases in protocols.items():
                _, m = evaluator.evaluate(p[None,:],None,cases)
                row['banks'][name] = summarize(m[0])
                row['banks'][name]['mean_total_force_N'] = float(m[0,:,4].mean())
                np.savez_compressed(out/(path.parent.name+'_'+path.name+'_'+name+'.npz'),cases=cases,metrics=m[0])
            rows.append(row)
    candidates = [json.loads((p/'result.json').read_text()) for p in paths[-3:]]
    selected = max(candidates,key=lambda r:r['validation_fitness'])
    path = root/'A1'/f"seed_{selected['seed']}"
    with np.load(path/'final_population.npz') as data:
        p = data['physical_controllers'][int(data['selected_index'])]
    for bank, cases in [('large',protocols['wide_grid'][[0,18,38,56,94,114,132]]),
                        ('small',protocols['small_fresh'][[0,12,24,48,72]])]:
        for i, case in enumerate(cases):
            metrics, trace = full_replay(p,case)
            np.savez_compressed(out/f'{bank}_{i}.npz',trace=trace,metrics=metrics,case=case,
                                columns=np.array(TRACE_COLUMNS),physical_controller=p)
    write_json(out/'summary.json',dict(selected_seed=selected['seed'],selection='large-angle validation only',runs=rows))
    study = root/'angle_setpoint/polynomial'
    selected = json.loads((study/'selection.json').read_text())['overall']
    law = json.loads((study/selected/'current_law.json').read_text())
    p = np.array(law['physical_controller']);coeff=np.array(law['coefficients'])
    targets = [-45,-22.5,0,22.5,45];starts=[45,45,-45,-45,-45]
    out = root/'angle_setpoint/distant_replay';out.mkdir(exist_ok=True)
    metrics=[]
    for i,(target,start) in enumerate(zip(targets,starts)):
        case=np.array([np.deg2rad(target),np.deg2rad(start),.15*(-1)**i,-1,0])
        m,tr,commanded=full_command_trace(p,coeff,case);metrics.append(m)
        np.savez_compressed(out/f'case_{i}.npz',trace=tr,metrics=m,case=case,
                            columns=np.array(TRACE_COLUMNS),physical_controller=commanded)
    write_csv(out/'metrics.csv',metric_rows(np.array(metrics)))
    write_json(out/'summary.json',dict(selected=selected,summary=summarize(np.array(metrics))))


def all_stages(root):
    """Prepare, test and execute the complete unscaled study with persistent stage status."""
    root.mkdir(exist_ok=True, parents=True)
    runtime=prepare(root)
    verify_runtime(root,runtime)
    if not (root/'pipeline_source.py').exists():
        shutil.copy2(__file__,root/'pipeline_source.py')
    manifest=dict(status='running',started=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                  root=str(root),runtime=str(runtime),muscle_torque_factor=1.,dt_ms=.1,
                  completed_stages=[],stages=['equilibrium','small_nm_weights','extra_balance','angle_setpoint','replays'])
    if (root/'manifest.json').exists():
        manifest=json.loads((root/'manifest.json').read_text());manifest['status']='running'
    write(root/'manifest.json',manifest)
    write(BASE/'report/results/current_unscaled.json',dict(root=os.path.relpath(root, BASE/'report/results')))
    env=dict(os.environ,PYTHONPATH=str(runtime),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
             MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1',MPLBACKEND='Agg')
    def execute(name):
        """Run one unfinished stage in a subprocess and append its output to a log."""
        if name in manifest['completed_stages']:return name
        with (root/f'{name}.log').open('a') as log:
            subprocess.run([sys.executable,str(Path(__file__).resolve()),name,'--output',str(root)],
                           env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        return name
    try:
        for batch in [('equilibrium','small_nm_weights'),('extra_balance','angle_setpoint'),('replays',)]:
            with ThreadPoolExecutor(len(batch)) as pool:
                for name in pool.map(execute,batch):
                    if name not in manifest['completed_stages']:manifest['completed_stages'].append(name)
                    write(root/'manifest.json',manifest)
                    print('Completed:',name,flush=True)
        manifest['status']='completed';manifest['finished']=time.strftime('%Y-%m-%dT%H:%M:%S%z')
    except Exception as error:
        manifest['status']='failed';manifest['error']=repr(error)
        raise
    finally:
        write(root/'manifest.json',manifest)


def main():
    """Dispatch the requested pipeline stage and output directory."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['all','equilibrium','small_nm_weights','extra_balance','angle_setpoint','replays'])
    parser.add_argument('--output',type=Path,default=DEFAULT)
    args=parser.parse_args()
    if args.stage=='all':all_stages(args.output.resolve())
    else:stage(args.output.resolve(),args.stage)


if __name__=='__main__':main()
