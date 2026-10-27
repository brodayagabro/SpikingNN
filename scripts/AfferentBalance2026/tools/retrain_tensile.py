"""Warm-start the A/B/C/D experiments on tensile mechanics using the shared GA."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import zipfile

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from scripts.AfferentBalance2026.experiments.core.evolution import Schedule, optimize
from scripts.AfferentBalance2026.experiments.core.genome import GenomeConfig, NAMES, load_genome, file_hash, write_json
from scripts.AfferentBalance2026.experiments.core.run import PopulationEvaluator, source_hashes, write_csv
from scripts.AfferentBalance2026.experiments.core.fitness import fitness
from scripts.AfferentBalance2026.experiments.simulation import MODEL, balance_banks
from scripts.AfferentBalance2026.experiments.C_setpoint_balance.joint_study import (
    JointGenome, load_joint_genome, target_banks, evaluate_joint, fit_table_polynomial)
from scripts.AfferentBalance2026.experiments.D_trajectory.controller import TrajectoryGenome, load_solution
from scripts.AfferentBalance2026.experiments.D_trajectory.trajectory import trajectory_banks
from scripts.AfferentBalance2026.experiments.D_trajectory.run_experiment import Evaluator as TrajectoryEvaluator

EXPERIMENTS = REPO/'scripts/AfferentBalance2026/experiments'


def polarity_seeds(physical, config):
    """Generate valid genetic alternatives to the old loop polarity, never changing physics."""
    variants = []
    for operation in ('negate_feedback', 'swap_feedback', 'swap_output', 'swap_muscles_output'):
        p = physical.copy()
        if operation == 'negate_feedback':
            p[30:42] *= -1
        elif operation == 'swap_feedback':
            p[30:42] = physical[30:42].reshape(2,2,3)[:,::-1,:].ravel()
        else:
            p[48:52] = physical[48:52].reshape(2,2)[::-1].ravel()
            if operation == 'swap_muscles_output':
                p[16:30] = physical[16:30].reshape(2,7)[::-1].ravel()
        try:
            config.encode(p)
        except ValueError:
            continue
        variants.append((operation,p))
    return variants


def decode(codec, unit):
    """Return physical controller and an optional command matrix from one chromosome."""
    return (codec.decode(unit), None) if isinstance(codec, GenomeConfig) else codec.decode(unit)


def save_genome(codec, path, unit, tau, **metadata):
    """Save the experiment-specific portable genome with corrected-model provenance."""
    metadata.update(model=MODEL)
    if isinstance(codec, GenomeConfig):
        codec.save(path, codec.decode(unit), **metadata)
    elif isinstance(codec, TrajectoryGenome):
        codec.save(path, unit, current_tau_s=tau, **metadata)
    else:
        codec.save(path, unit, **metadata)


def train_stage(codec, center, extras, folder, args, *, generations, kind, tau=0.):
    """Train on fixed banks, selecting checkpoints by validation and withholding tests."""
    folder.mkdir(parents=True,exist_ok=False)
    cfg = codec if isinstance(codec, GenomeConfig) else codec.controller
    banks = (trajectory_banks(cfg.data['trajectory']) if kind=='D' else
             balance_banks(cfg.data['simulation']) if kind in ('A','B') else target_banks(cfg,codec.kind))
    if kind=='D':
        context=TrajectoryEvaluator(codec,banks,args.workers)
    else:
        context=PopulationEvaluator(cfg.mechanics,args.workers)
    indices = np.arange(cfg.size,codec.size) if args.method=='command_only' else np.arange(codec.size)
    save_genome(codec,folder/'initial_genome.json',center,tau,selection='old winner / prescribed stage start')
    write_json(folder/'genome_cfg.json',cfg.data)
    schema = json.loads((folder/'initial_genome.json').read_text())['schema']
    population_metadata = dict(schema=schema, configuration_sha256=cfg.signature)
    if kind=='D':write_json(folder/'banks.json',banks)
    else:np.savez_compressed(folder/'cases.npz',**banks)
    folder.joinpath('generations').mkdir()
    hashes=source_hashes(); hashes[str(Path(__file__).relative_to(REPO))]=file_hash(__file__)
    with zipfile.ZipFile(folder/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in hashes:archive.write(REPO/name,name)
    write_json(folder/'run.json',dict(status='running',model=MODEL,dt_ms=.1,method=args.method,
        population=args.population,generations=generations,seed=args.seed,workers=args.workers,
        mutation_probability=args.mutation_probability,sigma_start=args.sigma_start,sigma_end=args.sigma_end,
        validation_every=args.validation_every,source_hashes=hashes,search_indices=indices.tolist(),
        test_used_for_selection=False,
        command=codec.specification if isinstance(codec, JointGenome) else None))
    with context as evaluator:
        def evaluate_full(population,bank):
            """Use identical training and validation reductions with experiment-specific episodes."""
            if kind=='D':return evaluator.evaluate(population,bank,tau)
            if kind in ('A','B'):
                return evaluator.evaluate(np.array([codec.decode(u) for u in population]),banks[bank],cfg.data['simulation']['duration_s'])
            return evaluate_joint(evaluator,codec,population,banks[bank],cfg.data['simulation']['duration_s'])
        baseline, baseline_metrics=evaluate_full(center[None],'validation')
        np.savez_compressed(folder/'baseline_validation.npz',scores=baseline,metrics=baseline_metrics)
        winner=dict(score=float(baseline[0]),unit=center.copy(),generation=0,individual=0)
        validation=[]
        def expand(population):
            """Insert the searched coordinates into the complete fixed-width genome."""
            full=np.tile(center,(len(population),1));full[:,indices]=population;return full
        def evaluate(population):
            """Return training scores for a complete candidate population."""
            return evaluate_full(expand(population),'training')
        def checkpoint(generation,population,scores,metrics,history,rng_state):
            """Save every evaluated generation and validate all members at fixed checkpoints."""
            full=expand(population)
            decoded=[decode(codec,u) for u in full]
            values=np.array([p if c is None else np.r_[p,c.ravel()] for p,c in decoded])
            np.savez_compressed(folder/'generations'/f'generation_{generation:04d}.npz',
                normalized=full,physical=values,training_scores=scores,training_metrics=metrics,
                generation=generation,rng_state=json.dumps(rng_state),
                names=codec.names if hasattr(codec,'names') else NAMES, **population_metadata)
            write_csv(folder/'history.csv',history)
            write_json(folder/'progress.json',dict(generation=generation,total=generations,
                training_fitness=history[-1]['best_fitness'],training_success=history[-1]['best_success'],
                best_validation_fitness=winner['score']))
            if generation==1 or generation==generations or generation%args.validation_every==0:
                val,vm=evaluate_full(full,'validation');i=int(np.argmax(val))
                np.savez_compressed(folder/f'validation_{generation:04d}.npz',normalized=full,scores=val,metrics=vm)
                if generation == generations:
                    np.savez_compressed(folder/'final_population.npz',
                        normalized=full, physical=values, training_scores=scores, training_metrics=metrics,
                        validation_scores=val, validation_metrics=vm, generation=generation,
                        names=codec.names if hasattr(codec,'names') else NAMES, **population_metadata)
                if val[i]>winner['score']:winner.update(score=float(val[i]),unit=full[i].copy(),generation=generation,individual=i)
                validation.append(dict(generation=generation,best_validation_fitness=float(val[i]),
                                       selected_fitness=winner['score'],selected_generation=winner['generation']))
                write_csv(folder/'validation_history.csv',validation)
                save_genome(codec,folder/'best_genome.json',winner['unit'],tau,
                    selection='validation only',generation=winner['generation'],individual=winner['individual'],seed=args.seed)
        extra=np.array([u[indices] for u in extras[:args.population-1]]) if extras else None
        schedule=Schedule(args.sigma_start,args.sigma_end,args.mutation_probability,args.mutation_probability)
        optimize(center[indices],evaluate,size=args.population,generations=generations,seed=args.seed,
                 schedule=schedule,on_state=checkpoint,initial_candidates=extra)
        # Generation zero remains eligible when no child improves validation.
        save_genome(codec,folder/'best_genome.json',winner['unit'],tau,selection='validation only',
                    generation=winner['generation'],individual=winner['individual'],seed=args.seed)
        final_score, final_metrics=evaluate_full(winner['unit'][None],'validation')
        np.savez_compressed(folder/'selected_validation.npz',scores=final_score,metrics=final_metrics)
        write_json(folder/'selection.json',{k:v for k,v in winner.items() if k!='unit'})
        # Only the final, frozen winner is evaluated on held-out tests.
        if not isinstance(codec, JointGenome) or codec.kind != 'table':
            score,metrics=evaluate_full(winner['unit'][None],'test')
            np.savez_compressed(folder/'test_metrics.npz',metrics=metrics[0],scores=score)
        for name,expected in hashes.items():
            if file_hash(REPO/name)!=expected:raise RuntimeError('Calculation source changed: '+name)
    result=dict(status='completed',model=MODEL,selected_generation=winner['generation'],
                baseline_validation_fitness=float(baseline[0]),validation_fitness=winner['score'],
                population=args.population,generations=generations,seed=args.seed)
    write_json(folder/'result.json',result)
    record=json.loads((folder/'run.json').read_text());record['status']='completed';write_json(folder/'run.json',record)
    return winner['unit']


def main():
    """Parse explicit warm-start options and train one reproducible experimental branch."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',choices=['A','B','C','D'],required=True)
    parser.add_argument('--method',choices=['joint','staged','command_only'],default='joint')
    parser.add_argument('--seed-genome',type=Path,required=True)
    parser.add_argument('--extra-seed',type=Path,action='append',default=[])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--population',type=int,default=50)
    parser.add_argument('--generations',type=int,default=200)
    parser.add_argument('--seed',type=int,default=43)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--mutation-probability',type=float,default=.3)
    parser.add_argument('--sigma-start',type=float,default=.2)
    parser.add_argument('--sigma-end',type=float,default=.015)
    parser.add_argument('--validation-every',type=int,default=20)
    parser.add_argument('--polarity-seeds',action=argparse.BooleanOptionalAction,default=True)
    args=parser.parse_args()
    if args.population<3 or args.generations<1 or args.validation_every<1:parser.error('Invalid population or horizon')
    if args.method=='staged' and (args.experiment!='C' or args.generations<2):parser.error('Staged training needs C and at least two generations')
    if args.method=='command_only' and args.experiment!='D':parser.error('Command-only training belongs to D')
    if MODEL!='SpikingNN.Afferents/v3-tensile':raise RuntimeError('Corrected tensile model required')
    if args.output.exists():raise FileExistsError(args.output)
    kind=args.experiment
    if kind in ('A','B'):
        cfg,p=load_genome(args.seed_genome);codec=cfg;center=cfg.encode(p);c=None;tau=0.
        assert cfg.data['experiment'].startswith(kind+'_')
    elif kind=='C':
        codec,center=load_joint_genome(args.seed_genome);cfg=codec.controller;p,c=codec.decode(center);tau=0.
    else:
        cfg,p,c,tau,_=load_solution(args.seed_genome);codec=TrajectoryGenome(cfg);center=codec.encode(p,c)
    extras=[];labels=[]
    if args.polarity_seeds and args.method!='command_only':
        for label,variant in polarity_seeds(p,cfg):
            extras.append(cfg.encode(variant) if kind in ('A','B') else codec.encode(variant,c));labels.append(label)
    for path in args.extra_seed:
        if kind in ('A','B'):
            old,p_extra=load_genome(path);u=cfg.encode(p_extra)
        elif kind == 'C':
            # Preserve the C polynomial width; load_solution pads to D's cubic format.
            extra_codec, extra_unit = load_joint_genome(path)
            if extra_codec.specification != codec.specification:
                raise ValueError('Extra C seed command specification differs')
            p_extra, c_extra = extra_codec.decode(extra_unit)
            old = extra_codec.controller
            u = codec.encode(p_extra, c_extra)
        else:
            old,p_extra,c_extra,_,_=load_solution(path);u=codec.encode(p_extra,c_extra)
        if not np.array_equal(old.mechanics,cfg.mechanics):raise ValueError('Extra seed mechanics differ')
        extras.append(u);labels.append(str(path))
    # Compile one evaluation in the parent before starting worker processes.
    if kind=='D':
        from scripts.AfferentBalance2026.experiments.D_trajectory.trajectory import replay
        replay(p,c,cfg.mechanics,trajectory_banks(cfg.data['trajectory'])['training'][0],tau_s=tau,stride=0)
    else:
        from scripts.AfferentBalance2026.experiments.simulation import simulate
        simulate(p,np.array([0.,0.,0.,-1.,0.]),cfg.mechanics,duration_s=.0001,stride=0)
    args.output.mkdir(parents=True)
    shutil.copy2(args.seed_genome,args.output/'source_genome.json')
    if args.extra_seed:
        (args.output/'extra_seed_inputs').mkdir()
        for index, path in enumerate(args.extra_seed):
            shutil.copy2(path, args.output/'extra_seed_inputs'/f'{index:02d}.json')
    write_json(args.output/'protocol.json',dict(model=MODEL,options={k:str(v) if isinstance(v,Path) else [str(p) for p in v] if k=='extra_seed' else v for k,v in vars(args).items()},
        source_sha256=file_hash(args.seed_genome),initial_variants=labels,extra_seed_sha256={str(p):file_hash(p) for p in args.extra_seed}))
    if args.method=='staged':
        table=JointGenome(cfg,'table',codec.degree)
        cases=np.array([[np.deg2rad(t),0.,0.,-1.,0.] for t in table.targets])
        currents=codec.currents(c,cases).T
        if np.any(abs(currents)>table.bound):raise ValueError('Warm-start table exceeds bounds')
        table_center=table.encode(p,currents)
        table_extras=[table.encode(*((decode(codec,u)[0],codec.currents(decode(codec,u)[1],cases).T))) for u in extras]
        n_table=args.generations//2
        best_table=train_stage(table,table_center,table_extras,args.output/'table',args,generations=n_table,kind='C')
        p,values=table.decode(best_table)
        polynomial,fit=fit_table_polynomial(table.targets,values,codec.degree,codec.bound)
        write_json(args.output/'least_squares.json',fit)
        center=codec.encode(p,polynomial)
        winner=train_stage(codec,center,[],args.output/'polynomial',args,generations=args.generations-n_table,kind='C')
        selected_folder=args.output/'polynomial'
    else:
        winner=train_stage(codec,center,extras,args.output/'training',args,generations=args.generations,kind=kind,tau=tau)
        selected_folder=args.output/'training'
    shutil.copy2(selected_folder/'best_genome.json',args.output/'best_genome.json')
    shutil.copy2(selected_folder/'result.json',args.output/'result.json')
    shutil.copy2(selected_folder/'test_metrics.npz',args.output/'test_metrics.npz')
    print(json.dumps(dict(output=str(args.output),result=json.loads((args.output/'result.json').read_text())),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
