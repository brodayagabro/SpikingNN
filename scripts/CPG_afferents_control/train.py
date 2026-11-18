#!/usr/bin/env python3
"""Train, sweep and analyze the Markin-connectivity spiking CPG.

python train.py verify
python train.py train --population 30 --generations 150 --seed 11 --workers 6
python train.py sweep --populations 20 50 --seeds 11 29 --generations 150 --workers 6
python train.py scan --genome runs/pilot_sweep/best_genome.json --output runs/drive_scan

The sine is a fitness reference, never an injected neural signal. d1 is constant,
d2=1. See README.md and training.json for timing, phase, fitness and gene groups.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import csv
import hashlib
import json
import os
import multiprocessing as mp
from pathlib import Path
import time
import zipfile

import numpy as np
import numba
from matplotlib import pyplot as plt
from simulate import build_config, Simulation, run_simulation
from dynamics import pack, unpack, PATH_INDEX, DT_S, TRACE_COLUMNS, replay
from fitness import task_vector, evaluate_population, measure, diagnostics, reference, METRIC_COLUMNS
from evolution import optimize

HERE = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    """Atomically replace JSON; checkpoints never contain NaN or pickle objects."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


class SearchSpace:
    """Map 22 interpretable groups (or all 95 genes) to complete API parameters."""

    def __init__(self, system, genome, search):
        self.system = build_config(system, genome)
        self.genome = deepcopy(genome)
        self.base = pack(self.system)
        lookup = {g['name']:g for g in genome['genes']}
        if search['space'] == 'all':
            self.groups = [dict(name=g['name'], targets=[g['name']], value=g['value'],
                                bounds=g['bounds'], scale=g['scale'], mode='absolute') for g in genome['genes']]
        elif search['space'] == 'groups':
            self.groups = deepcopy(search['groups'])
        else:
            raise ValueError('search.space must be groups or all')
        self.indices, self.factors = [], []
        used, names = set(), set()
        for group in self.groups:
            low, high = group['bounds']
            if (not np.isfinite([low, high, group['value']]).all() or not low < high
                    or not low <= group['value'] <= high or group['scale'] not in ('linear','log')
                    or group['mode'] not in ('absolute','multiply') or not group['targets']
                    or (group['scale']=='log' and low<=0) or group['name'] in names):
                raise ValueError(f'Invalid search group: {group["name"]}')
            names.add(group['name'])
            indices, factors = [], []
            for name in group['targets']:
                if name in used or name not in lookup:
                    raise ValueError(f'Unknown or overlapping target: {name}')
                used.add(name)
                g = lookup[name]
                index = PATH_INDEX[tuple(g['path'])]
                factor = g['value'] if group['mode']=='multiply' else 1.
                proposed = sorted([factor*low, factor*high])
                if proposed[0] < g['bounds'][0]-1e-12 or proposed[1] > g['bounds'][1]+1e-12:
                    raise ValueError(f'Group bounds exceed physical gene bounds: {name}')
                indices.append(index)
                factors.append(factor)
            self.indices.append(indices)
            self.factors.append(np.array(factors))
        self.low = np.array([g['bounds'][0] for g in self.groups])
        self.high = np.array([g['bounds'][1] for g in self.groups])
        self.logarithmic = np.array([g['scale']=='log' for g in self.groups])
        self.lo = self.low.copy()
        self.hi = self.high.copy()
        self.lo[self.logarithmic] = np.log(self.low[self.logarithmic])
        self.hi[self.logarithmic] = np.log(self.high[self.logarithmic])
        self.center = self.encode([g['value'] for g in self.groups])

    def encode(self, values):
        values = np.asarray(values, dtype=float).copy()
        if values.shape != self.low.shape or not np.isfinite(values).all() or np.any(values<self.low) or np.any(values>self.high):
            raise ValueError('Search values outside group bounds')
        values[self.logarithmic] = np.log(values[self.logarithmic])
        return (values-self.lo)/(self.hi-self.lo)

    def values(self, units):
        z = np.atleast_2d(np.asarray(units, dtype=float))
        if z.shape[1] != len(self.groups) or not np.isfinite(z).all() or np.any((z<0)|(z>1)):
            raise ValueError('Invalid normalized chromosome')
        values = self.lo + z*(self.hi-self.lo)
        values[:, self.logarithmic] = np.exp(values[:, self.logarithmic])
        return np.clip(values, self.low, self.high)

    def decode(self, units):
        values = self.values(units)
        physical = np.tile(self.base, (len(values),1))
        for i, (indices, factors) in enumerate(zip(self.indices,self.factors)):
            physical[:,indices] = values[:,i,None]*factors
        return physical

    def payload(self, unit, metadata):
        """Export an ordinary 95-gene document usable by simulate.py and the API."""
        physical = self.decode(unit)[0]
        result = deepcopy(self.genome)
        result['name'] = 'cpg_sine_ga_solution'
        result['status'] = 'Геном после генетической оптимизации; качество и ограничения указаны в training и metrics.json.'
        for gene in result['genes']:
            value = float(physical[PATH_INDEX[tuple(gene['path'])]])
            gene['value'] = float(np.clip(value, *gene['bounds']))
        result['training'] = dict(metadata, search_values={g['name']:float(v) for g,v in zip(self.groups,self.values(unit)[0])})
        # Preserve provenance of the hand-designed model without mislabelling new values.
        result['adaptation_note'] = 'adaptation описывает исходную модель; обученные численные значения находятся в genes.'
        build_config(self.system, result)
        return result

    def initial_candidates(self):
        """Documented exploratory starts with slower RG adaptation and balanced force."""
        names = [g['name'] for g in self.groups]
        if not {'RG_a','extensor_A'} <= set(names):
            return []
        candidates = []
        for a in (.0007,.0015,.003):
            values = np.array([g['value'] for g in self.groups])
            values[names.index('RG_a')] = a
            values[names.index('extensor_A')] = .5
            candidates.append(self.encode(values))
        return candidates


def source_hashes():
    files = [HERE/name for name in ('simulate.py','dynamics.py','fitness.py','evolution.py','train.py')]
    repo = HERE.parent.parent
    for name in ('api.py','schema.py','signals.py','core/Izh_net.py','core/multi_limb.py','core/factories.py'):
        path = repo/'src'/'SpikingNN'/name
        if path.exists():
            files.append(path)
    return {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def save_sources(folder, hashes):
    with zipfile.ZipFile(folder/'source_snapshot.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in hashes:
            path=Path(name)
            label='experiment/'+path.name if path.parent==HERE else 'SpikingNN/'+name.split('/SpikingNN/src/SpikingNN/')[-1]
            archive.write(path,label)


def api_trace(config, *, duration_s=10., drive=1.):
    """Full reference histories from SpikingNN; the target is not an input."""
    reference_config=deepcopy(config)
    reference_config['simulation']['duration']=duration_s*1000
    for row in reference_config['network']['Q_app']:row[0]*=drive
    data=run_simulation(Simulation(reference_config)).get_data()
    limb=next(iter(data['limbs'].values()))
    spikes=(data['V'][:,[0,1,8,9]]>=30).astype(float)
    counts=np.cumsum(spikes,axis=0)-spikes
    trace=np.column_stack((data['time']/1000,limb['angle_q'],limb['angular_velocity_w'],
                           limb['force_flex'],limb['force_ext'],data['V'],data['U'],limb['afferents'],counts))
    if not np.isfinite(trace).all():raise FloatingPointError('API produced nonfinite dynamics')
    return trace,data


def verify_api(config, *, duration_s=2., drive=1.):
    """Strict state parity for a short trajectory; report all errors on the full run.

    Adaptive bursting can amplify round-off and shift spikes over long runs.
    Final winners are therefore always replayed and scored by the actual API.
    """
    p = pack(config)
    fast, _, failure = replay(p,duration_s=duration_s,drive=drive,stride=1)
    if failure:
        raise ValueError('Fast reference run failed')
    _,data = api_trace(config,duration_s=duration_s,drive=drive)
    limb = next(iter(data['limbs'].values()))
    channels = {'V':(fast[:,5:15],data['V']), 'U':(fast[:,15:25],data['U']),
                'q':(fast[:,1],limb['angle_q']), 'w':(fast[:,2],limb['angular_velocity_w']),
                'force_flex':(fast[:,3],limb['force_flex']), 'force_ext':(fast[:,4],limb['force_ext']),
                'afferents':(fast[:,25:31],limb['afferents'])}
    errors={}
    for name,(actual,expected) in channels.items():
        errors[name]=float(np.max(abs(actual-expected)))
        # Floating-point summation order differs between NumPy/BLAS and sparse loops.
        tolerance = 1e-4 if name=='V' else (1e-6 if name in ('U','afferents') else 1e-8)
        short=min(len(actual),round(1./DT_S))
        np.testing.assert_allclose(actual[:short],expected[:short],rtol=0,atol=tolerance,err_msg=name)
    return errors, data


def plot_tracking(folder, trace, baseline, task, history, title):
    t=trace[:,0]
    desired,_=reference(t,task)
    fig,ax=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    ax[0,0].plot(t,np.rad2deg(desired),'k--',lw=1,label='Target')
    ax[0,0].plot(baseline[:,0],np.rad2deg(baseline[:,1]),color='.7',lw=.8,label='Initial genome')
    ax[0,0].plot(t,np.rad2deg(trace[:,1]),color='#2878a5',lw=1,label='Trained genome')
    ax[0,0].axvspan(0,task['score_start_s'],color='.85',alpha=.3)
    ax[0,0].axvline(task['duration_s'],color='black',ls=':',label='End of training window')
    ax[0,0].set(xlabel='Time (s)',ylabel='q (deg)',title='A  Fixed-frequency, fixed-phase tracking')
    ax[0,0].legend(fontsize=8)
    ax[0,1].plot(t,np.rad2deg(trace[:,1]-desired),color='#2878a5')
    ax[0,1].axhline(0,color='.5',lw=.7)
    ax[0,1].axvline(task['duration_s'],color='black',ls=':')
    ax[0,1].set(xlabel='Time (s)',ylabel='q - target (deg)',title='B  Phase is never aligned after simulation')
    ax[1,0].plot([r['generation'] for r in history],[r['best_rmse_deg'] for r in history])
    ax[1,0].set(xlabel='Generation',ylabel='Leader RMSE (deg)',title='C  Training history (leader chosen by fitness)')
    ax[1,1].plot(t,trace[:,3],label='Flexor',color='#2878a5')
    ax[1,1].plot(t,trace[:,4],label='Extensor',color='#d6652a')
    ax[1,1].set(xlabel='Time (s)',ylabel='Force (N)',title='D  Muscles driven only by CPG output')
    ax[1,1].legend()
    for a in ax.flat:a.grid(alpha=.15)
    fig.suptitle(title+' | constant tonic input; no sine injected into network')
    fig.savefig(folder/'tracking.png',dpi=170)
    plt.close(fig)


def finalize(folder, space, unit, config, history, *, api_check=True):
    task=config['task']; duration=config['evaluation']['long_duration_s']
    if duration <= task['duration_s']:
        raise ValueError('Long evaluation must extend beyond the training duration')
    physical=space.decode(unit)[0]
    trace,completed,failure=replay(physical,duration_s=duration,drive=task['tonic_drive'],angle_limit_deg=task['angle_limit_deg'])
    fast_trace=trace.copy()
    baseline,_,baseline_failure=replay(space.base,duration_s=duration,drive=task['tonic_drive'],angle_limit_deg=task['angle_limit_deg'])
    report=dict(numba_training=dict(zip(METRIC_COLUMNS,map(float,measure(trace, min(completed,round(task['duration_s']/DT_S)),
                              failure if completed < round(task['duration_s']/DT_S) else 0,task_vector(task))))),
                long_run_failure=int(failure),baseline_failure=int(baseline_failure),long_run_survival_s=completed*DT_S,
                selection='Maximum training fitness; the continuation after training duration is reported only, never used for selection.')
    if api_check:
        exported=build_config(space.system,read_json(folder/'best_genome.json'))
        full_trace,_=api_trace(exported,duration_s=duration,drive=task['tonic_drive'])
        outside=abs(full_trace[:,1]-np.pi/2)>np.deg2rad(task['angle_limit_deg'])
        completed=int(np.flatnonzero(outside)[0]) if outside.any() else len(full_trace)
        failure=int(outside.any())
        train_steps=round(task['duration_s']/DT_S)
        report['training']=dict(zip(METRIC_COLUMNS,map(float,measure(full_trace,min(completed,train_steps),
                            int(completed<train_steps),task_vector(task)))))
        report['trajectory_backend']='SpikingNN API'
        trace=full_trace[::10].copy()
        if len(trace)==len(fast_trace):
            report['numba_vs_api_max_q_error_deg']=float(np.rad2deg(np.max(abs(trace[:,1]-fast_trace[:,1]))))
        report.update(long_run_failure=failure,long_run_survival_s=completed*DT_S)
    else:
        report['training']=report['numba_training'].copy()
        report['trajectory_backend']='Numba (development only)'
    if completed >= round(task['duration_s']/DT_S):
        report['training_diagnostics']=diagnostics(trace,task)
    if not failure:
        report['continuation_diagnostics']=diagnostics(trace,task,start_s=task['duration_s'],end_s=duration)
    if not baseline_failure:
        report['baseline_diagnostics']=diagnostics(baseline,task)
    np.savez_compressed(folder/'tracking.npz',trace=trace,baseline=baseline,
                        numba_trace=fast_trace,target_rad=reference(trace[:,0],task)[0],columns=np.array(TRACE_COLUMNS))
    best=read_json(folder/'best_genome.json')
    best['training']['final_evaluation']=report
    write_json(folder/'best_genome.json',best)
    write_json(folder/'metrics.json',report)
    plot_tracking(folder,trace,baseline,task,history,folder.name)
    return report


def train_run(folder, system, genome, config, *, resume=False, api_check=True):
    folder=Path(folder).resolve()
    task=task_vector(config['task']); ga=config['ga']
    if config['evaluation']['long_duration_s'] <= config['task']['duration_s']:
        raise ValueError('Long evaluation must be longer than training')
    space=SearchSpace(system,genome,config['search'])
    identity=dict(system=system,genome=genome,configuration=config)
    signature=digest(identity)
    hashes=source_hashes()
    resume_state=None
    if resume:
        manifest=read_json(folder/'manifest.json')
        if manifest['signature']!=signature or manifest['source_hashes']!=hashes:
            raise ValueError('Resume configuration or source hashes differ from the saved run')
        with np.load(folder/'checkpoint.npz',allow_pickle=False) as saved:
            checkpoint=json.loads(str(saved['metadata_json']))
            resume_state=dict(generation=checkpoint['generation'],rng_state=checkpoint['rng_state'],
                              population=saved['population'],scores=saved['scores'])
        history=checkpoint['history']
        if checkpoint['generation']>=ga['generations']:
            if (folder/'metrics.json').exists():
                return read_json(folder/'metrics.json')
            leader=int(np.argmax(resume_state['scores']))
            return finalize(folder,space,resume_state['population'][leader],config,history,api_check=api_check)
    else:
        folder.mkdir(parents=True,exist_ok=False)
        write_json(folder/'manifest.json',dict(signature=signature,configuration=config,source_hashes=hashes,
                    numpy_version=np.__version__,numba_version=numba.__version__,threads=numba.get_num_threads(),
                    initial_candidates='Untrained center, nearby mutations, uniform immigrants; three slower-RG/balanced-force warm starts.'))
        write_json(folder/'system.json',system)
        write_json(folder/'initial_genome.json',genome)
        write_json(folder/'training.json',config)
        save_sources(folder,hashes)
        history=[]
    # Compile before timing the generations and before entering Numba's thread pool.
    evaluate_population(space.decode(space.center),task)
    start=time.monotonic()-(history[-1]['elapsed_s'] if history else 0.)
    def evaluate(units):
        return evaluate_population(space.decode(units),task)
    def checkpoint(state):
        leader=int(np.argmax(state['scores']))
        best=state['metrics'][leader]
        row=dict(generation=int(state['generation']),evaluations=(state['generation']+1)*ga['population'],
                 best_fitness=float(best[0]),median_fitness=float(np.median(state['scores'])),best_rmse_deg=float(best[1]),
                 failed=int(np.count_nonzero(state['metrics'][:,-1])),diversity=float(state['population'].std(axis=0).mean()),
                 elapsed_s=time.monotonic()-start)
        history.append(row)
        if state['generation']%10==0 or state['generation']==ga['generations']:
            metadata=dict(generation=state['generation'],rng_state=state['rng_state'],history=history)
            np.savez_compressed(folder/'checkpoint.tmp.npz',population=state['population'],scores=state['scores'],metrics=state['metrics'],
                                metadata_json=np.array(json.dumps(metadata,allow_nan=False)))
            (folder/'checkpoint.tmp.npz').replace(folder/'checkpoint.npz')
            metadata=dict(task=config['task'],ga=ga,generation=state['generation'],fitness=float(best[0]),
                          rmse_deg=float(best[1]),configuration_sha256=signature)
            write_json(folder/'best_genome.json',space.payload(state['population'][leader],metadata))
            write_csv(folder/'history.csv',history)
            print(f'{folder.name}: {state["generation"]}/{ga["generations"]}, fitness={best[0]:.5f}, '
                  f'RMSE={best[1]:.3f} deg, failed={row["failed"]}/{ga["population"]}, elapsed={row["elapsed_s"]:.1f}s',flush=True)
    state=optimize(space.center,evaluate,population=ga['population'],generations=ga['generations'],seed=ga['seed'],
                   settings=ga,callback=checkpoint,candidates=space.initial_candidates(),resume=resume_state)
    leader=int(np.argmax(state['scores']))
    return finalize(folder,space,state['population'][leader],config,history,api_check=api_check)


def run_sweep(folder, system, genome, config, *, populations, seeds, generations):
    """Cross population sizes with RNG seeds, select by training fitness only."""
    folder=Path(folder).resolve()
    if len(set(populations))!=len(populations) or len(set(seeds))!=len(seeds):
        raise ValueError('Sweep sizes and seeds must be unique')
    if any(n<4 for n in populations) or any(s<0 for s in seeds) or generations<1:
        raise ValueError('Invalid sweep settings')
    folder.mkdir(parents=True,exist_ok=False)
    write_json(folder/'sweep.json',dict(populations=populations,seeds=seeds,generations=generations,configuration=config,
                                     selection='Maximum training fitness; continuation metrics never select the winner.'))
    rows=[]; histories=[]
    for population in populations:
        for seed in seeds:
            settings=deepcopy(config)
            settings['ga'].update(population=population,generations=generations,seed=seed)
            name=f'N{population}_G{generations}_seed{seed}'
            report=train_run(folder/name,system,genome,settings,api_check=True)
            continuation=report.get('continuation_diagnostics',{})
            row=dict(run=name,population=population,generations=generations,seed=seed,
                     fitness=report['training']['fitness'],training_rmse_deg=report['training']['rmse_deg'],
                     continuation_rmse_deg=continuation.get('rmse_deg'),
                     continuation_amplitude_deg=continuation.get('fitted_target_amplitude_deg'),
                     continuation_phase_error_deg=continuation.get('phase_error_deg'),
                     continuation_frequency_hz=continuation.get('dominant_motion_frequency_hz'),
                     failure=report['long_run_failure'])
            rows.append(row)
            write_csv(folder/'summary.csv',rows)
            with (folder/name/'history.csv').open() as stream:
                histories.append((name,list(csv.DictReader(stream))))
    winner=max(rows,key=lambda row:row['fitness'])
    best=read_json(folder/winner['run']/'best_genome.json')
    write_json(folder/'best_genome.json',best)
    effective=build_config(system,best)
    errors,data=verify_api(effective,duration_s=config['task']['duration_s'],drive=config['task']['tonic_drive'])
    q=next(iter(data['limbs'].values()))['angle_q'];t=data['time']/1000
    desired,_=reference(t,config['task']); mask=t>=config['task']['score_start_s']
    write_json(folder/'selection.json',dict(winner=winner,api_max_absolute_errors=errors,
              api_training_rmse_deg=float(np.rad2deg(np.sqrt(np.mean((q[mask]-desired[mask])**2)))),
              note='A fixed-initial-state pilot. No robust phase-locking to an external clock is claimed.'))
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    for name,history in histories:
        axes[0].plot([int(r['generation']) for r in history],[float(r['best_rmse_deg']) for r in history],label=name)
    axes[0].set(xlabel='Generation',ylabel='Leader training RMSE (deg)',title='Population / seed sweep')
    axes[0].legend(fontsize=7)
    with np.load(folder/winner['run']/'tracking.npz') as saved:
        trace=saved['trace'];target=saved['target_rad']
    axes[1].plot(trace[:,0],np.rad2deg(target),'k--',lw=1,label='Target')
    axes[1].plot(trace[:,0],np.rad2deg(trace[:,1]),lw=1,label='Selected CPG')
    axes[1].axvline(config['task']['duration_s'],color='black',ls=':',label='Training horizon')
    axes[1].set(xlabel='Time (s)',ylabel='q (deg)',title=winner['run']+'; frozen continuation')
    axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.15)
    fig.savefig(folder/'sweep.png',dpi=180);plt.close(fig)
    print('Selected:',json.dumps(winner,ensure_ascii=False),flush=True)
    drive_scan(folder/'drive_scan',system,best,config)
    return rows


def _scan_api_task(arguments):
    system,duration,drive,limit=arguments
    trace,_=api_trace(system,duration_s=duration,drive=drive)
    outside=abs(trace[:,1]-np.pi/2)>np.deg2rad(limit)
    completed=int(np.flatnonzero(outside)[0]) if outside.any() else len(trace)
    return trace[::10].copy(),completed,int(outside.any())


def drive_scan(folder, system, genome, config):
    """Measure the actual frequency/amplitude response of one frozen genome."""
    folder=Path(folder).resolve();folder.mkdir(parents=True,exist_ok=False)
    effective=build_config(system,genome)
    scan=config['drive_scan'];task=deepcopy(config['task'])
    task.update(duration_s=scan['duration_s'],score_start_s=scan['score_start_s'])
    rows=[];traces=[]
    arguments=[(effective,scan['duration_s'],drive,task['angle_limit_deg']) for drive in scan['values']]
    # Spawn isolated API simulations; never fork an initialized Numba thread pool.
    with ProcessPoolExecutor(max_workers=min(4,numba.get_num_threads()),mp_context=mp.get_context('spawn')) as pool:
        results=list(pool.map(_scan_api_task,arguments))
    for drive,(trace,completed,failure) in zip(scan['values'],results):
        d=diagnostics(trace,task) if not failure else {}
        row=dict(tonic_drive=drive,failure=failure,survival_s=completed*DT_S,
                 motion_frequency_hz=d.get('dominant_motion_frequency_hz'),
                 motion_half_range_deg=d.get('robust_half_range_deg'),mean_angle_deg=d.get('mean_angle_deg'),
                 RG_F_frequency_hz=d.get('RG_F_burst_frequency_hz'),RG_E_frequency_hz=d.get('RG_E_burst_frequency_hz'))
        rows.append(row);traces.append(trace)
    write_csv(folder/'response.csv',rows)
    write_json(folder/'protocol.json',dict(scan=scan,genome_sha256=digest(genome),configuration=config,
              backend='SpikingNN API',note='Independent identical-state resets. d1 scales only Q_app[:,0], including RG and PF; d2 stays fixed. Spectral peaks are finite-window diagnostics.'))
    write_json(folder/'genome.json',genome)
    np.savez_compressed(folder/'traces.npz',**{f'drive_{i:02d}':trace for i,trace in enumerate(traces)},columns=np.array(TRACE_COLUMNS))
    x=np.array([r['tonic_drive'] for r in rows])
    def y(key):return np.array([r[key] if r[key] is not None else np.nan for r in rows])
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    axes[0,0].plot(x,y('motion_frequency_hz'),'o-',label='Limb motion')
    axes[0,0].plot(x,y('RG_F_frequency_hz'),'s--',label='RG-F bursts')
    axes[0,0].plot(x,y('RG_E_frequency_hz'),'d:',label='RG-E bursts')
    axes[0,0].set(ylabel='Dominant frequency (Hz)',title='A  Frequency need not be monotone');axes[0,0].legend()
    axes[0,1].plot(x,y('motion_half_range_deg'),'o-')
    axes[0,1].set(ylabel='Motion half-range: 5–95% / 2 (deg)',title='B  Oscillation amplitude')
    axes[1,0].plot(x,y('mean_angle_deg'),'o-')
    axes[1,0].set(ylabel='Mean q (deg)',title='C  Tonic drive can also shift the mean posture')
    for i in sorted(set([int(np.argmin(abs(x-v))) for v in (.5,1.,2.)])):
        trace=traces[i];late=trace[:,0]>=scan['score_start_s']
        axes[1,1].plot(trace[late,0],np.rad2deg(trace[late,1]),lw=.8,label=f'd1={x[i]:g}')
    axes[1,1].set(xlabel='Time (s)',ylabel='q (deg)',title='D  Frozen genome at different constant drives');axes[1,1].legend()
    for ax in axes.flat:ax.grid(alpha=.15)
    for ax in (axes[0,0],axes[0,1],axes[1,0]):ax.set_xlabel('Constant supraspinal drive d1')
    fig.savefig(folder/'response.png',dpi=170);plt.close(fig)
    print('Drive response saved:',folder/'response.csv',flush=True)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command',choices=['train','sweep','verify','scan'])
    parser.add_argument('--config',type=Path,default=HERE/'training.json')
    parser.add_argument('--system',type=Path,default=HERE/'system.json')
    parser.add_argument('--genome',type=Path,default=HERE/'genome.json')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--workers',type=int,default=min(6,len(os.sched_getaffinity(0))) if hasattr(os,'sched_getaffinity') else 4)
    parser.add_argument('--population',type=int)
    parser.add_argument('--populations',nargs='+',type=int)
    parser.add_argument('--generations',type=int)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--seeds',nargs='+',type=int)
    parser.add_argument('--space',choices=['groups','all'])
    parser.add_argument('--frequency',type=float,help='Target frequency in Hz')
    parser.add_argument('--amplitude',type=float,help='Target amplitude in degrees')
    parser.add_argument('--phase',type=float,help='Target phase at t=0, in radians')
    parser.add_argument('--drive',type=float,help='Constant multiplier of the supraspinal input only')
    parser.add_argument('--resume',action='store_true',help='Resume an interrupted single training run with unchanged configuration')
    args=parser.parse_args()
    if args.workers<1 or args.workers>numba.config.NUMBA_NUM_THREADS:
        parser.error('--workers outside the available Numba thread range')
    numba.set_num_threads(args.workers)
    config=read_json(args.config); system=read_json(args.system); genome=read_json(args.genome)
    for key in ('population','generations','seed'):
        if getattr(args,key) is not None:config['ga'][key]=getattr(args,key)
    for arg,key in [('frequency','frequency_hz'),('amplitude','amplitude_deg'),('phase','phase_rad'),('drive','tonic_drive')]:
        if getattr(args,arg) is not None:config['task'][key]=getattr(args,arg)
    if args.space:config['search']['space']=args.space
    task_vector(config['task'])
    if args.command=='verify':
        c=build_config(system,genome)
        result,_=verify_api(c,duration_s=config['task']['duration_s'],drive=config['task']['tonic_drive'])
        print(json.dumps(result,indent=2))
    elif args.command=='train':
        ga=config['ga']
        output=args.output or HERE/'runs'/f'train_N{ga["population"]}_G{ga["generations"]}_seed{ga["seed"]}'
        train_run(output,system,genome,config,resume=args.resume)
    elif args.command=='sweep':
        run_sweep(args.output or HERE/'runs'/'pilot_sweep',system,genome,config,
                  populations=args.populations or config['sweep']['populations'],
                  seeds=args.seeds or config['sweep']['seeds'],
                  generations=args.generations or config['sweep']['generations'])
    else:
        drive_scan(args.output or HERE/'runs'/'drive_scan',system,genome,config)


if __name__=='__main__':
    main()
