"""Verify a retrained winner and save publication figures with their numerical data."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from scripts.AfferentBalance2026.tools.recheck_tensile_genomes import load_candidate, summarize, digest, write_json
from scripts.AfferentBalance2026.experiments.simulation import (
    MODEL, TRACE_COLUMNS, METRIC_COLUMNS, case_grid, balance_banks, simulate,
    reference_trace, measure_trace)
from scripts.AfferentBalance2026.experiments.C_setpoint_balance.joint_study import target_banks
from scripts.AfferentBalance2026.experiments.D_trajectory.trajectory import (
    replay, trajectory_banks, measure_tracking_trace)
from scripts.AfferentBalance2026.experiments.visualization import configure_fonts


def save_figure(figure, folder, name):
    """Export identical raster and vector figures and release Matplotlib memory."""
    figure.savefig(folder/(name+'.png'), dpi=180)
    figure.savefig(folder/(name+'.pdf'))
    plt.close(figure)


def label_axes(axes):
    """Use large Times New Roman labels and consistent lettered panels."""
    for letter, axis in zip('ABCDEF', axes.flat):
        axis.text(.015, .98, letter, transform=axis.transAxes, va='top',
                  weight='bold', fontsize=19, bbox=dict(facecolor='white', alpha=.8, edgecolor='none'))
        axis.grid(alpha=.2)


def signals(axes, trace, currents, reference, *, zoom=True):
    """Plot one complete mechanical example and a resolved neural time window."""
    time=trace[:,0]
    mask=time>=max(0., time[-1]-.2) if zoom else np.ones(len(time),dtype=bool)
    for i, color in enumerate(('crimson','royalblue')):
        axes[1,0].plot(time[mask],trace[mask,5+i],color=color,lw=.8,label=f'Neuron {i}')
        axes[1,1].plot(time,trace[:,3+i],color=color,lw=1,label=('Flexor','Extensor')[i])
        axes[2,0].plot(time,trace[:,11+3*i:14+3*i].sum(axis=1),color=color,lw=.8,label=('Flexor','Extensor')[i])
        axes[2,1].plot(time,currents[:,i],color=color,lw=1,label=f'Neuron {i}')
    axes[1,0].set(xlabel='Time (s)',ylabel='Membrane potential (mV)')
    axes[1,1].set(xlabel='Time (s)',ylabel='Muscle force (N)')
    axes[2,0].set(xlabel='Time (s)',ylabel='Ia + II + Ib (model units)')
    axes[2,1].set(xlabel='Time (s)',ylabel='Applied current (model units)')
    for ax in axes[1:].flat:ax.legend(fontsize=10,loc='upper right')


def evaluate_static(item, experiment, output):
    """Save all grid trials, zero-speed oscillograms and one full-resolution example per target."""
    p,cfg=item['physical'],item['cfg']
    targets=(0.,) if experiment in ('A_balance','B_balance') else (-45.,-22.5,0.,22.5,45.)
    summaries=[]
    for target in targets:
        cases=case_grid(np.linspace(-50,50,10),np.linspace(-1,1,10),target)
        traces={};metrics=[]
        currents=np.zeros((len(cases),2)) if item['joint'] is None else item['joint'].currents(item['command'],cases)
        for i,case in enumerate(cases):
            values,trace=simulate(p,case,cfg.mechanics,currents=currents[i],duration_s=10.,stride=20)
            metrics.append(values);traces[f'trace_{i:03d}']=trace
        metrics=np.array(metrics)
        tag=f'target_{target:+06.1f}'
        np.savez_compressed(output/(tag+'_grid.npz'),cases=cases,metrics=metrics,currents=currents,
                            trace_columns=TRACE_COLUMNS,metric_columns=METRIC_COLUMNS,**traces)
        stats=summarize(metrics,np.full(len(cases),10.),metrics[:,8]);stats['target_deg']=target
        figure,axes=plt.subplots(3,2,figsize=(12,12),constrained_layout=True)
        for i,case in enumerate(cases):
            tr=traces[f'trace_{i:03d}'];color=plt.cm.coolwarm((np.rad2deg(case[1])+50)/100)
            axes[0,0].plot(np.rad2deg(tr[:,1]),tr[:,2],color=color,alpha=.6,lw=.65)
            if abs(tr[-1,1])>np.pi/2:axes[0,0].plot(np.rad2deg(tr[-1,1]),tr[-1,2],'kx',ms=3)
        axes[0,0].plot(target,0,'k*',ms=12)
        axes[0,0].set(xlabel='Angle (deg)',ylabel='Angular velocity (rad/s)',title=f'Grid success: {stats["successes"]}/100')
        zero_cases=case_grid(np.linspace(-50,50,10),[0.],target);zero_traces={}
        for i,case in enumerate(zero_cases):
            current=(0.,0.) if item['joint'] is None else item['joint'].currents(item['command'],case[None])[0]
            _,tr=simulate(p,case,cfg.mechanics,currents=current,duration_s=10.,stride=20)
            zero_traces[f'trace_{i:03d}']=tr
            axes[0,1].plot(tr[:,0],np.rad2deg(tr[:,1]),color=plt.cm.coolwarm(i/9),lw=1)
        axes[0,1].axhline(target,color='black',ls='--',lw=1)
        axes[0,1].set(xlabel='Time (s)',ylabel='Angle (deg)',title='10 initial angles; initial velocity = 0')
        np.savez_compressed(output/(tag+'_zero_speed.npz'),cases=zero_cases,**zero_traces)
        case=case_grid([-30.],[0.],target)[0]
        current=np.zeros(2) if item['joint'] is None else item['joint'].currents(item['command'],case[None])[0]
        values,tr=simulate(p,case,cfg.mechanics,currents=current,duration_s=10.,stride=1)
        np.testing.assert_allclose(measure_trace(tr[1:],np.deg2rad(target),initial_theta=case[1]),values,rtol=1e-9,atol=1e-9)
        # Compare all state channels against the independent original library.
        ref=reference_trace(p,case,cfg.mechanics,currents=current,duration_s=min(.05,tr[-1,0]))
        np.testing.assert_allclose(tr[:len(ref)],ref,rtol=1e-9,atol=1e-9)
        np.savez_compressed(output/(tag+'_example.npz'),case=case,trace=tr,currents=current,metrics=values,trace_columns=TRACE_COLUMNS)
        signals(axes,tr,np.tile(current,(len(tr),1)),None)
        label_axes(axes)
        figure.suptitle(f'{experiment} | target {target:g}° | C–F: initial angle −30°, velocity 0',fontsize=17)
        save_figure(figure,output,tag)
        summaries.append(stats)
        print(experiment,target,stats,flush=True)
    small=np.concatenate([case_grid(np.array([-5.,0.,5.])+t,[0.],t) for t in targets])
    sm=[]
    for case in small:
        current=np.zeros(2) if item['joint'] is None else item['joint'].currents(item['command'],case[None])[0]
        sm.append(simulate(p,case,cfg.mechanics,currents=current,duration_s=10.,stride=0)[0])
    sm=np.array(sm)
    np.savez_compressed(output/'near_target.npz',cases=small,metrics=sm)
    return dict(grid=summaries,near_target=summarize(sm,np.full(len(sm),10.),sm[:,8]))


def evaluate_dynamic(item, output):
    """Replay every held-out sinusoid without resets and plot a prescribed first test."""
    specs=trajectory_banks(item['cfg'].data['trajectory'])['test']
    metrics=[];traces={}
    for i,spec in enumerate(specs):
        result=replay(item['physical'],item['coefficients'],item['cfg'].mechanics,spec,tau_s=item['tau'],stride=1 if i==0 else 50)
        metrics.append(result['metrics'])
        for name in ('trace','commands','reference'):traces[f'{name}_{i:03d}']=result[name]
        if i==0:
            np.testing.assert_allclose(measure_tracking_trace(result['trace'],result['reference'],spec),result['metrics'],rtol=1e-9,atol=1e-9)
    np.savez_compressed(output/'trajectory_tests.npz',metrics=metrics,**traces)
    write_json(output/'trajectory_specs.json',specs)
    figure,axes=plt.subplots(3,2,figsize=(12,12),constrained_layout=True)
    tr,ref,cur=(traces[f'{name}_000'] for name in ('trace','reference','commands'))
    axes[0,0].plot(np.rad2deg(tr[:,1]),tr[:,2],lw=.8,label='Limb')
    axes[0,0].plot(np.rad2deg(ref[:,0]),ref[:,1],'k--',lw=1,label='Target')
    axes[0,0].set(xlabel='Angle (deg)',ylabel='Angular velocity (rad/s)',title='Phase trajectory: time-dependent target')
    axes[0,1].plot(tr[:,0],np.rad2deg(tr[:,1]),lw=1,label='Limb')
    axes[0,1].plot(tr[:,0],np.rad2deg(ref[:,0]),'k--',lw=1,label='Target')
    axes[0,1].set(xlabel='Time (s)',ylabel='Angle (deg)',title='First held-out test, specified before replay')
    for ax in axes[0]:ax.legend(fontsize=10)
    signals(axes,tr,cur[:,:2],ref);label_axes(axes)
    figure.suptitle('D: tracking, neurons, muscle forces, afferents and applied currents',fontsize=16)
    save_figure(figure,output,'tracking')
    figure,ax=plt.subplots(figsize=(11,5),constrained_layout=True)
    for i in range(len(specs)):
        tr,ref=(traces[f'{name}_{i:03d}'] for name in ('trace','reference'))
        ax.plot(tr[:,0],np.rad2deg(tr[:,1]-ref[:,0]),lw=.6,alpha=.7)
    ax.axhspan(-3,3,color='green',alpha=.1)
    ax.set(xlabel='Time (s)',ylabel='Tracking error (deg)',title='All 18 held-out trajectories, including early falls')
    ax.grid(alpha=.2);save_figure(figure,output,'all_tracking_errors')
    return dict(test=summarize(np.array(metrics),np.array([s['duration_s'] for s in specs]),np.array([traces[f'trace_{i:03d}'][-1,1] for i in range(len(specs))])))


def main():
    """Validate the frozen winner and save a complete, auditable figure dataset."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',choices=['A_balance','B_balance','C_setpoint_balance','D_trajectory'],required=True)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();args.run=args.run.resolve();configure_fonts(14)
    item=load_candidate(args.experiment,args.run.name,args.run/'best_genome.json')
    output=args.run/'figures';output.mkdir(exist_ok=True)
    if args.experiment=='D_trajectory':summary=evaluate_dynamic(item,output)
    else:
        summary=evaluate_static(item,args.experiment,output)
        bank=balance_banks(item['cfg'].data['simulation'])['test'] if item['joint'] is None else target_banks(item['cfg'])['test']
        applied=np.zeros((len(bank),2)) if item['joint'] is None else item['joint'].currents(item['command'],bank)
        metrics=np.array([simulate(item['physical'],case,item['cfg'].mechanics,currents=cur,duration_s=10.,stride=0)[0] for case,cur in zip(bank,applied)])
        np.testing.assert_allclose(metrics,np.load(args.run/'test_metrics.npz')['metrics'],rtol=1e-10,atol=1e-10)
        summary['test']=summarize(metrics,np.full(len(bank),10.),metrics[:,8])
        summary['test']['late_rmse_mean_deg']=float(np.rad2deg(metrics[:,7]).mean())
    if args.experiment=='D_trajectory':
        np.testing.assert_allclose(np.load(output/'trajectory_tests.npz')['metrics'],np.load(args.run/'test_metrics.npz')['metrics'],rtol=1e-10,atol=1e-10)
    summary.update(model=MODEL,genome_sha256=digest(item['path']),trace_columns=TRACE_COLUMNS,
                   example_rule='static: theta0=-30 deg, omega0=0; dynamic: first held-out test')
    write_json(output/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
