"""Five-angle current-law search using the shared Numba dynamics, fitness and GA.

No mechanical, neural or muscle gene is trained here. All free coordinates
describe the two constant-in-time input currents as functions of the command.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import time
import zipfile

import numpy as np

from .cases import setpoint_cases
from .commands import input_currents, polynomial_currents, polynomial_parameters
from .config import DT_MS, FITNESS, MECHANICS, MODEL
from .dynamics import rollout, measure_trace, TRACE_COLUMNS, METRIC_COLUMNS
from .evolution import optimize, Schedule
from .population import PopulationEvaluator, load_controller_population
from .run import write_json, write_csv, source_hashes, metric_rows, CASE_COLUMNS

TARGETS_DEG = np.array([-45., -22.5, 0., 22.5, 45.])
INTERPOLATION_DEG = np.array([-33.75, -11.25, 11.25, 33.75])
BANK_SEEDS = dict(training=2026101501, validation=2026101502,
                  test=2026101503, interpolation=2026101504)


def study_banks():
    return {name:setpoint_cases(BANK_SEEDS[name],count,
                INTERPOLATION_DEG if name == 'interpolation' else TARGETS_DEG)
            for name,count in (('training',3),('validation',5),('test',10),('interpolation',8))}


def summarize(metrics):
    from .fitness import fitness
    return dict(fitness=fitness(metrics),cases=len(metrics),
                successes=int(np.sum(metrics[:,5])),
                late_rmse_deg=float(np.rad2deg(metrics[:,7]).mean()),
                max_late_rmse_deg=float(np.rad2deg(metrics[:,7]).max()),
                max_settling_s=float(metrics[:,6].max()),
                falls=int(np.sum(metrics[:,1] < 10.-1e-9)),
                numerical_failures=int(np.sum(metrics[:,0] == -1.)))


def optimize_law(output,controller,bank,evaluator,low,high,center,*,
                 population,generations,seed,candidates=(),resume=False):
    """Same GA for constant calibration and both polynomial degrees."""
    output.mkdir(exist_ok=True)
    shape=center.shape
    def decode(units):
        return low[None,:,:] + units.reshape(-1,*shape)*(high-low)[None,:,:]
    def evaluate(units):
        return evaluator.evaluate(np.repeat(controller[None,:],len(units),axis=0),
                                  decode(units),bank,10.,command_kind='polynomial')
    def progress(history):
        write_csv(output/'history.csv',history)
        row=history[-1]
        if row['generation']==1 or row['generation']%10==0 or row['generation']==generations:
            print(f"{output.name}: {row['generation']}/{generations}; "
                  f"fitness={row['best_fitness']:.6f}; success={row['best_success']:.1%}",flush=True)
    def checkpoint(generation,units,scores,metrics,history,rng):
        if generation%10==0 or generation==generations:
            temporary=output/'checkpoint.tmp.npz'
            np.savez_compressed(temporary,generation=generation,population=units,scores=scores,
                                history_json=json.dumps(history),rng_state=json.dumps(rng))
            temporary.replace(output/'checkpoint.npz')
    state=None
    if resume and (output/'checkpoint.npz').exists():
        with np.load(output/'checkpoint.npz') as d:
            if int(d['generation'])<generations:
                state=dict(generation=int(d['generation']),population=d['population'],scores=d['scores'],
                           history=json.loads(str(d['history_json'])),rng_state=json.loads(str(d['rng_state'])))
    encode=lambda c: ((c-low)/(high-low)).ravel()
    schedule=Schedule(sigma_start=.18,sigma_end=.001,probability_start=.5,probability_end=.2)
    units,scores,metrics,history,rng=optimize(encode(center),evaluate,size=population,
        generations=generations,seed=seed,schedule=schedule,on_generation=progress,on_state=checkpoint,
        resume_state=state,initial_candidates=None if not len(candidates) else np.array([encode(c) for c in candidates]))
    coefficients=decode(units)
    np.savez_compressed(output/'training_population.npz',population=units,coefficients=coefficients,
                        scores=scores,metrics=metrics,lower=low,upper=high,
                        history_json=json.dumps(history),rng_state=json.dumps(rng))
    return coefficients,scores,metrics


def full_command_trace(controller,coefficients,case,duration_s=10.):
    commanded=polynomial_parameters(controller,coefficients,case[0])
    metrics,trace=rollout(commanded,case[1:],target_rad=case[0],duration_s=duration_s,stride=1,mode=0)
    initial=np.zeros(len(TRACE_COLUMNS));initial[1:3]=case[1:3]
    initial[5:7]=controller[4:6];initial[7:9]=controller[4:6]*controller[2:4]
    if metrics[0] != -1.:
        np.testing.assert_allclose(metrics,measure_trace(trace,target_rad=case[0],
                     duration_s=duration_s,initial_theta=case[1]),atol=1e-10,rtol=1e-10)
    return metrics,np.vstack((initial,trace)),commanded


def run_study(output,population_from,*,population=50,generations=150,seeds=(42,43),workers=10,resume=False):
    output=Path(output).resolve();population_from=Path(population_from).resolve()
    if (output/'protocol.json').exists() and not resume:
        raise FileExistsError('This study already exists; use --resume for its saved protocol')
    output.mkdir(exist_ok=True,parents=True)
    controllers=load_controller_population(population_from)
    with np.load(population_from,allow_pickle=False) as data:
        index=int(data['selected_index'])
        assert index==int(np.argmax(data['validation_scores']))
    controller=controllers[index].copy()
    assert np.all(controller[46:48]==0)
    banks=study_banks();hashes=source_hashes()
    protocol=dict(model=MODEL,fitness=FITNESS,dt_ms=DT_MS,duration_s=10.,mode=0,pd_used=False,
        targets_deg=TARGETS_DEG.tolist(),interpolation_deg=INTERPOLATION_DEG.tolist(),
        population=population,generations=generations,seeds=list(seeds),workers=workers,
        calibration_population=24,calibration_generations=60,calibration_current_limit=1000.,
        coefficient_domain='min(calibration_fit,static_fit,0)-200 to max(...)+200, separately per coefficient',
        polynomial_degrees=[2,3],powers_include_constant=True,
        schedule=asdict(Schedule(.18,.001,.5,.2)),
        banks={name:len(cases) for name,cases in banks.items()},case_seeds=BANK_SEEDS,
        hyperparameters=asdict(MECHANICS),source_population=str(population_from),
        source_population_sha256=hashlib.sha256(population_from.read_bytes()).hexdigest(),
        fixed_controller_index=index,controller_selection='existing balance validation winner; no setpoint test selection',
        source_sha256=hashes)
    if resume:
        old=json.loads((output/'protocol.json').read_text())
        assert old==protocol,'Resume requires the same protocol and source bytes'
    else:
        write_json(output/'protocol.json',protocol)
        shutil.copy2(population_from.with_name('best_genome.json'),output/'fixed_controller.json')
        np.save(output/'physical_controller.npy',controller)
        repo=Path(__file__).resolve().parents[4]
        with zipfile.ZipFile(output/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for path in hashes:archive.write(path,Path(path).relative_to(repo))
        for name,cases in banks.items():
            write_csv(output/f'{name}_cases.csv',[dict(zip(CASE_COLUMNS,map(float,c))) for c in cases])
    started=time.monotonic()
    runs=[]
    with PopulationEvaluator(workers,cache_size=12000) as evaluator:
        calibrated=[]
        for target in TARGETS_DEG:
            path=output/f'calibration_{target:+06.1f}'
            if resume and (path/'best.json').exists():
                calibrated.append(json.loads((path/'best.json').read_text())['currents']);continue
            bank=banks['training'][np.isclose(banks['training'][:,0],np.deg2rad(target))]
            center=input_currents(controller,np.zeros((2,3)),np.deg2rad(target)).reshape(2,1)
            coeff,scores,metrics=optimize_law(path,controller,bank,evaluator,
                np.full((2,1),-1000.),np.full((2,1),1000.),center,population=24,generations=60,
                seed=20261015+len(calibrated),resume=resume)
            chosen=int(np.argmax(scores));currents=coeff[chosen,:,0]
            write_json(path/'best.json',dict(target_deg=float(target),currents=currents.tolist(),
                        selected_index=chosen,training=summarize(metrics[chosen]),selection='training only'))
            calibrated.append(currents)
        calibrated=np.asarray(calibrated)
        write_csv(output/'calibration_currents.csv',[dict(target_deg=float(a),I_flex=float(c[0]),I_ext=float(c[1]))
                                                    for a,c in zip(TARGETS_DEG,calibrated)])
        static=np.array([input_currents(controller,np.zeros((2,3)),np.deg2rad(a)) for a in TARGETS_DEG])
        for degree in (2,3):
            center=np.polynomial.polynomial.polyfit(TARGETS_DEG/45.,calibrated,degree).T
            prior=np.polynomial.polynomial.polyfit(TARGETS_DEG/45.,static,degree).T
            zeros=np.zeros_like(center)
            low=np.minimum.reduce([center,prior,zeros])-200.
            high=np.maximum.reduce([center,prior,zeros])+200.
            for seed in seeds:
                path=output/f'degree_{degree}_seed_{seed}'
                if resume and (path/'selection.json').exists():
                    runs.append(json.loads((path/'selection.json').read_text()));continue
                coeff,scores,training=optimize_law(path,controller,banks['training'],evaluator,low,high,center,
                    population=population,generations=generations,seed=seed,candidates=[prior,zeros],resume=resume)
                val_scores,val=evaluator.evaluate(np.repeat(controller[None,:],len(coeff),axis=0),
                    coeff,banks['validation'],10.,command_kind='polynomial')
                chosen=int(np.argmax(val_scores))
                np.savez_compressed(path/'final_population.npz',model=MODEL,dt_ms=DT_MS,
                    command_coefficients=coeff,training_scores=scores,training_metrics=training,
                    validation_scores=val_scores,validation_metrics=val,selected_index=chosen,
                    fixed_controller=controller,degree=degree,generation=generations,seed=seed)
                genes=[dict(index=i*(degree+1)+k,neuron=i,power=k,value=float(coeff[chosen,i,k]),
                            low=float(low[i,k]),high=float(high[i,k]),unit='model current',
                            description=f'Ток мотонейрона {i}: коэффициент при (целевой угол / 45°)^{k}; '+
                            ('постоянное смещение тока.' if k==0 else 'зависит только от команды, не от текущего угла.'))
                       for i in range(2) for k in range(degree+1)]
                write_json(path/'current_law.json',dict(degree=degree,coefficients=coeff[chosen].tolist(),
                    law='I_i(theta*)=sum_{k=0}^degree c[i,k]*(theta*/45deg)^k',genes=genes,
                    physical_controller=controller.tolist(),model=MODEL,dt_ms=DT_MS))
                row=dict(directory=path.name,degree=degree,seed=seed,selected_index=chosen,
                         training=summarize(training[chosen]),validation=summarize(val[chosen]))
                write_json(path/'selection.json',row);runs.append(row)
                print('Selected by validation:',json.dumps(row),flush=True)
        # Freeze every choice before looking at any test or interpolation score.
        winners={str(d):max([r for r in runs if r['degree']==d],key=lambda r:r['validation']['fitness'])['directory']
                 for d in (2,3)}
        overall=max(runs,key=lambda r:r['validation']['fitness'])['directory']
        write_json(output/'selection.json',dict(by_degree=winners,overall=overall,criterion='validation fitness only'))
        for row in runs:
            path=output/row['directory'];law=json.loads((path/'current_law.json').read_text())
            coeff=np.array(law['coefficients'])
            for name in ('test','interpolation'):
                _,metrics=evaluator.evaluate(controller[None,:],coeff[None,:,:],banks[name],10.,command_kind='polynomial')
                row[name]=summarize(metrics[0])
                write_csv(path/f'{name}_metrics.csv',metric_rows(metrics[0]))
                write_csv(path/f'{name}_by_target.csv',[dict(target_deg=float(np.rad2deg(target)),
                    **summarize(metrics[0][np.isclose(banks[name][:,0],target)])) for target in np.unique(banks[name][:,0])])
                if path.name in winners.values():
                    trace_dir=path/f'{name}_traces';trace_dir.mkdir(exist_ok=True)
                    trace_info=[]
                    for trial,case in enumerate(banks[name]):
                        # Two histories per target: start at vertical; local start + late kick.
                        per_target=10 if name=='test' else 8
                        if trial%per_target not in (0,2):continue
                        m,trace,commanded=full_command_trace(controller,coeff,case)
                        np.testing.assert_allclose(m,metrics[0,trial],atol=1e-12,rtol=1e-12)
                        file=trace_dir/f'case_{trial:03d}.npz'
                        np.savez_compressed(file,trace=trace,columns=np.array(TRACE_COLUMNS),case=case,
                            physical_controller=commanded,fixed_controller=controller,coefficients=coeff,
                            metrics=m,metric_columns=np.array(METRIC_COLUMNS),model=MODEL,dt_ms=DT_MS,mode=0)
                        trace_info.append(dict(trial=trial,target_deg=float(np.rad2deg(case[0])),samples=len(trace),
                            file=str(file.relative_to(output)),sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
                    write_json(path/f'{name}_traces.json',trace_info)
            write_csv(path/'input_currents.csv',[dict(target_deg=float(a),I_flex=float(c[0]),I_ext=float(c[1]))
                for a in np.linspace(-45,45,181) for c in [polynomial_currents(coeff,np.deg2rad(a))]])
            write_json(path/'result.json',row)
            print('Final tests:',json.dumps(row),flush=True)
        # Same new banks, same strong controller, with the former zero-residual compensation.
        baseline={}
        for name in ('training','validation','test','interpolation'):
            _,metrics=evaluator.evaluate(controller[None,:],np.zeros((1,2,3)),banks[name],10.)
            baseline[name]=summarize(metrics[0]);write_csv(output/f'baseline_{name}_metrics.csv',metric_rows(metrics[0]))
    if source_hashes()!=hashes:raise RuntimeError('Simulation/search sources changed during this study')
    summary=dict(protocol=protocol,runs=runs,selected_by_degree=winners,selected_overall=overall,
                 baseline_static_compensation=baseline,elapsed_s=time.monotonic()-started)
    write_json(output/'summary.json',summary)
    print('STUDY COMPLETE',json.dumps(summary),flush=True)
    return summary
