"""Joint controller/current-law genomes and multi-target training for experiment C.

The 52 controller fields retain their shared schema. This experiment appends
polynomial coefficients or a table of target-specific currents to the search
vector. Every individual uses one controller across all target/initial states.
Current commands depend on the desired angle only and are constant per episode.
"""
from copy import deepcopy
from itertools import product
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
from scipy.optimize import lsq_linear

from ..core.evolution import Schedule, optimize
from ..core.genome import GenomeConfig, NAMES, file_hash, load_genome, write_json
from ..core.run import PopulationEvaluator, REPO, save_replays, source_hashes, write_csv
from ..simulation import CASE_COLUMNS, METRIC_COLUMNS, balance_banks, case_grid, simulate
from .polynomial_study import polynomial_currents

SCHEMA = 'afferent_setpoint_joint_v1'


class JointGenome:
    """Map one normalized chromosome to a controller and two command functions.

    Parameters
    ----------
    config : GenomeConfig, path or dict
        C controller domain and independent multi-target protocol.
    kind : {'polynomial', 'table'}
        Append ascending-power coefficients or one current per training target.
    degree : {2, 3}
        Polynomial degree; table width is the number of training targets.
    """

    def __init__(self, config, kind='polynomial', degree=3):
        """Validate the controller and append an explicitly named command block."""
        self.controller = config if isinstance(config, GenomeConfig) else GenomeConfig(config)
        if self.controller.data['experiment'] != 'C_setpoint_balance':
            raise ValueError('Joint training belongs to C_setpoint_balance')
        if kind not in ('polynomial', 'table') or degree not in (2, 3):
            raise ValueError('Use a table or a degree-2/3 polynomial')
        self.kind, self.degree = kind, degree
        settings = self.controller.data['setpoint']
        self.targets = np.asarray(settings['targets_deg'], dtype=float)
        if (self.targets.ndim != 1 or len(self.targets) < 5
                or not np.isfinite(self.targets).all() or np.any(np.diff(self.targets) <= 0)
                or self.targets[0] != -45 or self.targets[-1] != 45):
            raise ValueError('Use at least five increasing training targets spanning -45 to +45 degrees')
        self.width = degree+1 if kind == 'polynomial' else len(self.targets)
        if self.controller.mirror_symmetry and not np.array_equal(self.targets, -self.targets[::-1]):
            raise ValueError('Mirror symmetry requires a reflected target table')
        self.bound = float(settings['coefficient_bound_nA' if kind == 'polynomial' else 'table_current_bound_nA'])
        if not np.isfinite(self.bound) or self.bound <= 0:
            raise ValueError('Command bounds must be finite and positive')
        suffixes = range(self.width) if kind == 'polynomial' else self.targets
        self.names = (*NAMES, *(f'command.{kind}.neuron_{i}.{k:g}' for i in range(2) for k in suffixes))

    @property
    def size(self):
        """Return independent controller and command coordinates, honoring reflection."""
        return self.controller.size + (1 if self.controller.mirror_symmetry else 2)*self.width

    @property
    def specification(self):
        """Return the portable definition of the appended command genes."""
        spec = dict(kind=self.kind, degree=self.degree, targets_deg=self.targets.tolist(),
                    bound_nA=self.bound, variable='target_rad / radians(45)', order='neuron-major')
        if self.controller.mirror_symmetry:
            spec['mirror_symmetry'] = True
        return spec

    def reflected_command(self, first):
        """Return neuron 1's command from I1(theta) = I0(-theta)."""
        return first * (-1.)**np.arange(self.width) if self.kind == 'polynomial' else first[::-1]

    def encode(self, physical, command):
        """Validate and concatenate controller and command coordinates in [0,1]."""
        command = np.asarray(command, dtype=float)
        if command.shape != (2,self.width) or not np.isfinite(command).all() or np.any(abs(command) > self.bound):
            raise ValueError('Command shape or coefficient/current bounds are invalid')
        if self.controller.mirror_symmetry:
            if not np.allclose(command[1], self.reflected_command(command[0]), rtol=0, atol=1e-10):
                raise ValueError('Commands violate I1(theta) = I0(-theta)')
            command = command[0]
        return np.r_[self.controller.encode(physical), .5+.5*command.ravel()/self.bound]

    def decode(self, unit):
        """Decode a chromosome, restoring every fixed controller hyperparameter."""
        unit = np.asarray(unit, dtype=float)
        if unit.shape != (self.size,) or not np.isfinite(unit).all() or np.any((unit<0)|(unit>1)):
            raise ValueError(f'Expected {self.size} finite normalized joint coordinates')
        command = (unit[self.controller.size:]*2-1)*self.bound
        command = (np.array([command, self.reflected_command(command)])
                   if self.controller.mirror_symmetry else command.reshape(2, self.width))
        return self.controller.decode(unit[:self.controller.size]), command

    def currents(self, command, cases):
        """Return K-by-2 constant commands; a table never interpolates unseen targets."""
        targets = np.asarray(cases)[:,0]
        if self.kind == 'polynomial':
            return np.array([polynomial_currents(command, target) for target in targets])
        match = np.isclose(targets[:,None], np.deg2rad(self.targets)[None,:], rtol=0, atol=1e-12)
        if not np.all(match.sum(axis=1)==1):
            raise ValueError('A current table can only replay its training target angles')
        return np.asarray(command)[:,np.argmax(match,axis=1)].T

    def save(self, path, unit, **metadata):
        """Save all physical genes, English mappings and the original controller domain."""
        physical, command = self.decode(unit)
        fields = deepcopy(self.controller.data['parameters'])
        for field, value in zip(fields, physical):
            field['value'] = float(value)
        for i, name in enumerate(self.names[52:]):
            neuron, index = divmod(i,self.width)
            meaning = (f'Coefficient of (target_rad / radians(45))**{index}' if self.kind=='polynomial'
                       else f'Constant command at target {self.targets[index]:g} degrees')
            fields.append(dict(index=52+i, name=name, value=float(command.ravel()[i]), mutable=True,
                               minimum=-self.bound, maximum=self.bound, scale='linear', unit='nA',
                               description=f'{meaning} applied to neuron {neuron}; independent of measured state.',
                               system_parameter=f'Izhikevich_IO_Network applied current, neuron {neuron}'))
        write_json(path, dict(schema=SCHEMA, configuration=self.controller.data,
                   configuration_sha256=self.controller.signature, command=self.specification,
                   names=list(self.names), parameters=fields, physical=np.r_[physical,command.ravel()].tolist(),
                   normalized=np.asarray(unit).tolist(), metadata=metadata))


def load_joint_genome(path):
    """Load and cross-check a self-contained joint genotype; reject changed fields."""
    data = json.loads(Path(path).read_text())
    if data.get('schema') != SCHEMA:
        raise ValueError('Expected a joint C genome')
    spec = data['command']
    genome = JointGenome(data['configuration'], spec['kind'], spec['degree'])
    if (genome.specification != spec or list(genome.names) != data['names']
            or genome.controller.signature != data['configuration_sha256']):
        raise ValueError('Joint genome specification or configuration checksum mismatch')
    unit = np.asarray(data['normalized'], dtype=float)
    physical, command = genome.decode(unit)
    np.testing.assert_allclose(np.r_[physical,command.ravel()], data['physical'], rtol=0, atol=1e-11)
    return genome, unit


def target_banks(config, kind='polynomial'):
    """Cross independent initial-state banks with equally weighted target angles.

    Training always uses the five or more anchor targets. Polynomial validation
    and tests include configured intermediate targets; table stages use anchors
    only. Initial conditions are absolute angles, not errors around each target.
    """
    cfg = config if isinstance(config, GenomeConfig) else GenomeConfig(config)
    if kind not in ('polynomial','table'): raise ValueError('Unknown command representation')
    settings = cfg.data['setpoint']
    initial = balance_banks(cfg.data['simulation'])
    result = {}
    for name in ('training','validation','test'):
        targets = np.asarray(settings['targets_deg' if name=='training' or kind=='table'
                                      else f'{name}_targets_deg'], dtype=float)
        if (targets.ndim != 1 or not len(targets) or not np.isfinite(targets).all()
                or len(np.unique(targets)) != len(targets) or np.any(abs(targets)>45)):
            raise ValueError('Target banks must contain unique finite angles within -45..45 degrees')
        if not set(settings['targets_deg']).issubset(set(targets)):
            raise ValueError('Each bank must include all training target angles')
        blocks=[]
        for target in targets:
            block=initial[name].copy(); block[:,0]=np.deg2rad(target); blocks.append(block)
        result[name]=np.concatenate(blocks)
    sets=[set(map(tuple,result[name])) for name in result]
    if any(sets[i]&sets[j] for i in range(3) for j in range(i+1,3)):
        raise ValueError('Training, validation and test episodes must be disjoint')
    return result


def fit_table_polynomial(targets_deg, currents, degree, bound):
    """Fit two bounded ascending-power polynomials by least squares.

    This approximates currents, not dynamics. The returned residuals must be
    followed by fresh rollouts and joint refinement. No test outcomes enter fitting.
    """
    targets=np.asarray(targets_deg,dtype=float); currents=np.asarray(currents,dtype=float)
    if (degree not in (2,3) or targets.ndim!=1 or len(targets)<degree+1
            or len(np.unique(targets))!=len(targets) or currents.shape!=(2,len(targets))
            or not np.isfinite(targets).all() or not np.isfinite(currents).all()
            or not np.isfinite(bound) or bound<=0):
        raise ValueError('Invalid least-squares table, polynomial degree or coefficient bound')
    design=np.vander(targets/45.,degree+1,increasing=True)
    solutions=[lsq_linear(design,row,bounds=(-bound,bound),tol=1e-12) for row in currents]
    if not all(sol.success for sol in solutions):
        raise RuntimeError('Bounded least-squares solver failed')
    coefficients=np.array([sol.x for sol in solutions])
    residual=coefficients@design.T-currents
    return coefficients, dict(rmse_nA=float(np.sqrt(np.mean(residual**2))),
                              max_absolute_error_nA=float(abs(residual).max()),
                              solver='scipy.optimize.lsq_linear', degree=degree)


def evaluate_joint(evaluator, genome, unit, cases, duration):
    """Evaluate every candidate on every target using one shared physical controller."""
    decoded=[genome.decode(row) for row in unit]
    physical=np.array([p for p,c in decoded])
    currents=np.array([genome.currents(c,cases) for p,c in decoded])
    return evaluator.evaluate(physical,cases,duration,currents=currents)


def _save_solution(folder, genome, unit, **metadata):
    """Export the unified genome and compatibility views for replay tooling."""
    physical, command=genome.decode(unit)
    genome.save(folder/'best_genome.json',unit,**metadata)
    genome.controller.save(folder/'best_controller.json',physical,**metadata)
    payload={**genome.specification, 'values':command.tolist()}
    if genome.kind=='polynomial':
        payload['coefficients']=command.tolist()
    write_json(folder/'current_law.json',payload)


def train_joint(config, output, *, kind='polynomial', degree=3, population=50,
                generations=200, seed=42, workers=1, probability=.1, center=None):
    """Train a joint population, saving all generations and validation checkpoints.

    No independent test is evaluated here. The returned winner maximizes
    validation over the saved checkpoints, including generation 1 so that a
    fitted initial solution cannot be lost merely through training selection.
    """
    genome=JointGenome(config,kind,degree); cfg=genome.controller
    if population<3 or generations<1 or seed<0 or not 0<=probability<=1:
        raise ValueError('Invalid population, generation, seed or mutation setting')
    if center is None:
        center=genome.encode(cfg.initial_physical,np.zeros((2,genome.width)))
    genome.decode(center)
    banks=target_banks(cfg,kind); duration=cfg.data['simulation']['duration_s']
    physical,command=genome.decode(center)
    simulate(physical,banks['training'][0],cfg.mechanics,
             currents=genome.currents(command,banks['training'][:1])[0],duration_s=duration,stride=0)
    output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=False)
    (output/'generations').mkdir()
    write_json(output/'genome_cfg.json',cfg.data)
    genome.save(output/'initial_genome.json',center,selection='initial candidate')
    np.savez_compressed(output/'cases.npz',columns=np.array(CASE_COLUMNS),**banks)
    hashes=source_hashes()
    with zipfile.ZipFile(output/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in hashes: archive.write(REPO/name,name)
        archive.write(output/'genome_cfg.json','genome_cfg.json')
    checkpoints={1,generations,*[n for n in cfg.data['setpoint']['checkpoints'] if 1<=n<=generations]}
    run=dict(status='running',schema=SCHEMA,command=genome.specification,dt_ms=.1,
             pd_used=False,muscle_torque_factor=1.,population=population,generations=generations,
             seed=seed,mutation_probability=probability,workers=workers,duration_s=duration,
             configuration_sha256=cfg.signature,source_sha256=hashes,checkpoints=sorted(checkpoints),
             training_cases=len(banks['training']),validation_cases=len(banks['validation']),
             evaluation_budget=population*generations*len(banks['training']))
    write_json(output/'run.json',run); rows=[]
    with PopulationEvaluator(cfg.mechanics,workers) as evaluator:
        def evaluate(unit):
            """Use all training targets in each fitness call; never use test episodes."""
            return evaluate_joint(evaluator,genome,unit,banks['training'],duration)

        def save_state(generation,unit,scores,metrics,history,rng_state):
            """Persist the complete chromosome, measured metrics and RNG for each epoch."""
            decoded=[genome.decode(row) for row in unit]
            arrays=dict(schema=SCHEMA,configuration_sha256=cfg.signature,
                        normalized=unit,physical=np.array([np.r_[p,c.ravel()] for p,c in decoded]),
                        controller_physical=np.array([p for p,c in decoded]),
                        command_values=np.array([c for p,c in decoded]),
                        training_scores=scores,training_metrics=metrics,generation=generation,
                        rng_state=json.dumps(rng_state),names=np.array(genome.names))
            np.savez_compressed(output/'generations'/f'generation_{generation:04d}.npz',**arrays)
            write_csv(output/'history.csv',history)
            write_json(output/'progress.json',dict(generation=generation,total=generations,
                       best_fitness=history[-1]['best_fitness']))
            if generation not in checkpoints: return
            folder=output/'checkpoints'/f'epochs_{generation:03d}'; folder.mkdir(parents=True)
            values,validation=evaluate_joint(evaluator,genome,unit,banks['validation'],duration)
            winner=int(np.argmax(values))
            np.savez_compressed(folder/'final_population.npz',**arrays,
                                validation_scores=values,validation_metrics=validation,selected_index=winner)
            _save_solution(folder,genome,unit[winner],selection='validation',generation=generation,
                           population_index=winner,seed=seed)
            row=dict(directory=str(folder.relative_to(output)),generation=generation,
                     validation_fitness=float(values[winner]),population=population,
                     mutation_probability=probability,seed=seed,selected_index=winner)
            write_json(folder/'result.json',row); rows.append(row)
            write_csv(output/'checkpoints.csv',rows)
        optimize(center,evaluate,size=population,generations=generations,seed=seed,
                 schedule=Schedule(probability_start=probability,probability_end=probability),on_state=save_state)
    if source_hashes()!=hashes: raise RuntimeError('Calculation sources changed during C training')
    chosen=max(rows,key=lambda r:r['validation_fitness'])
    for name in ('best_genome.json','best_controller.json','current_law.json'):
        shutil.copy2(output/chosen['directory']/name,output/name)
    shutil.copy2(output/'checkpoints'/f'epochs_{generations:03d}'/'final_population.npz',output/'final_population.npz')
    result={**chosen,'kind':kind,'degree':degree,'free_parameters':genome.size,
            'physical_fields':len(genome.names),'training_performed':True,'test_used_for_selection':False}
    write_json(output/'result.json',result); write_json(output/'run.json',{**run,'status':'completed'})
    return result


def summarize_targets(cases, metrics):
    """Return separate success and error summaries for every target angle."""
    rows=[]
    for target in np.unique(cases[:,0]):
        values=metrics[cases[:,0]==target]
        rows.append(dict(target_deg=float(np.rad2deg(target)),cases=len(values),
                         successes=int(values[:,5].sum()),success_fraction=float(values[:,5].mean()),
                         late_rmse_deg=float(np.rad2deg(values[:,7]).mean())))
    return rows


def load_joint_solution(path, rank=1):
    """Load a joint JSON or a ranked saved generation with its original domain."""
    path=Path(path).resolve()
    if path.suffix=='.json':
        genome,unit=load_joint_genome(path)
        return genome,unit,dict(selection='saved joint genome',file=str(path))
    parent=next((p for p in path.parents if (p/'run.json').is_file() and (p/'genome_cfg.json').is_file()),None)
    if parent is None: raise FileNotFoundError('No saved C training configuration found')
    record=json.loads((parent/'run.json').read_text()); spec=record['command']
    genome=JointGenome(parent/'genome_cfg.json',spec['kind'],spec['degree'])
    with np.load(path,allow_pickle=False) as data:
        if str(data['schema'])!=SCHEMA or str(data['configuration_sha256'])!=genome.controller.signature:
            raise ValueError('Joint population schema or configuration checksum mismatch')
        key='validation_scores' if 'validation_scores' in data else 'training_scores'
        if not isinstance(rank,int) or not 1<=rank<=len(data[key]): raise ValueError('Rank is outside the population')
        index=int(np.argsort(-data[key],kind='stable')[rank-1]); unit=data['normalized'][index].copy()
        physical,command=genome.decode(unit)
        np.testing.assert_array_equal(np.r_[physical,command.ravel()],data['physical'][index])
    return genome,unit,dict(selection=key,rank=rank,population_index=index,file=str(path))


def plot_solution(output, genome, law):
    """Draw actual saved target trajectories, test errors and the commanded-current curves."""
    import matplotlib.pyplot as plt
    from ..visualization import configure_fonts, plot_phase, plot_angles, plot_states
    configure_fonts(13); records=[]
    for path in sorted((output/'traces').glob('case_*.npz')):
        with np.load(path,allow_pickle=False) as data:
            records.append(dict(case=data['case'],trace=data['trace'],metrics=data['metrics']))
    for target in genome.targets:
        group=[r for r in records if np.isclose(r['case'][0],np.deg2rad(target))]
        folder=output/f'target_{target:+06.2f}'; folder.mkdir()
        for name,figure in [('phase',plot_phase(group)),('angles',plot_angles(group))]:
            figure.savefig(folder/f'{name}.png',dpi=160);plt.close(figure)
    for name,figure in plot_states(records[len(records)//2]).items():
        figure.savefig(output/f'{name}.png',dpi=160);plt.close(figure)
    targets=np.linspace(-45,45,181)
    currents=np.array([polynomial_currents(law,np.deg2rad(t)) for t in targets])
    fig,ax=plt.subplots(figsize=(8,5),constrained_layout=True)
    for i in range(2):ax.plot(targets,currents[:,i],label=f'Neuron {i}')
    ax.set(xlabel='Commanded target (deg)',ylabel='Applied current (nA)');ax.legend();ax.grid(alpha=.2)
    fig.savefig(output/'current_law.png',dpi=160);plt.close(fig)


def finish_solution(config, source, output, *, workers=1):
    """Independently test one validation-selected polynomial and save real trajectories."""
    genome,unit=load_joint_genome(source/'best_genome.json')
    physical,law=genome.decode(unit); cfg=genome.controller
    if genome.kind!='polynomial': raise ValueError('Final deployment requires a polynomial, not a current table')
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    for name in ('best_genome.json','best_controller.json','current_law.json'):
        shutil.copy2(source/name,output/name)
    banks=target_banks(cfg); duration=cfg.data['simulation']['duration_s']
    with PopulationEvaluator(cfg.mechanics,workers) as evaluator:
        scores,metrics=evaluate_joint(evaluator,genome,unit[None],banks['test'],duration)
    np.savez_compressed(output/'test_metrics.npz',cases=banks['test'],metrics=metrics[0],
                        case_columns=np.array(CASE_COLUMNS),metric_columns=np.array(METRIC_COLUMNS))
    by_target=summarize_targets(banks['test'],metrics[0]); write_csv(output/'test_by_target.csv',by_target)
    cases=np.concatenate([case_grid(cfg.data['simulation']['replay_angles_deg'],[0.],target)
                          for target in genome.targets])
    save_replays(output/'traces',cfg,physical,cases,duration_s=duration,currents=genome.currents(law,cases))
    result=dict(test_fitness=float(scores[0]),test_success=float(metrics[0,:,5].mean()),
                test_cases=len(banks['test']),worst_target_success=min(row['success_fraction'] for row in by_target),
                test_late_rmse_deg=float(np.rad2deg(metrics[0,:,7]).mean()),duration_s=duration,
                test_used_for_selection=False,source=str(source.resolve()))
    write_json(output/'result.json',result)
    if cfg.data['setpoint'].get('save_figures',True): plot_solution(output,genome,law)
    return result


def run_pipeline(config, output, *, method='joint', degree=3, population=50,
                 generations=200, table_generations=None, seed=42, workers=1,
                 probability=.1, seed_genome=None, final_test=True):
    """Run direct joint optimization or table -> least squares -> joint refinement.

    ``generations`` is the total training horizon. Staged runs split that budget
    between a common controller/current table and a common controller/polynomial.
    The table never consists of independently trained controllers for each angle.
    """
    cfg=config if isinstance(config,GenomeConfig) else GenomeConfig(config)
    if method not in ('joint','staged') or generations<1:
        raise ValueError('Use joint or staged training with a positive epoch budget')
    if table_generations is None: table_generations=generations//2
    if method=='staged' and not 1<=table_generations<generations:
        raise ValueError('A staged budget must leave at least one epoch for each stage')
    physical=cfg.initial_physical.copy()
    # Validate the full protocol before leaving a partial output tree.
    target_banks(cfg)
    if seed_genome is not None:
        seed_cfg,physical=load_genome(seed_genome)
        if not np.array_equal(seed_cfg.mechanics,cfg.mechanics): raise ValueError('Seed mechanics differ from C')
        cfg.encode(physical)  # Never silently clip a seed or change fixed muscle/geometry values.
    poly=JointGenome(cfg,'polynomial',degree); command=np.zeros((2,degree+1))
    output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=False)
    record=dict(status='running',method=method,degree=degree,population=population,
                generations=generations,table_generations=table_generations if method=='staged' else 0,
                seed=seed,mutation_probability=probability,configuration_sha256=cfg.signature,
                seed_genome=None if seed_genome is None else str(Path(seed_genome).resolve()),
                seed_genome_sha256=None if seed_genome is None else file_hash(seed_genome),
                selection='validation only',final_test=final_test)
    write_json(output/'pipeline.json',record)
    common=dict(population=population,seed=seed,workers=workers,probability=probability)
    if method=='staged':
        table=JointGenome(cfg,'table',degree)
        train_joint(cfg,output/'table',kind='table',degree=degree,generations=table_generations,
                    center=table.encode(physical,np.zeros((2,table.width))),**common)
        saved,unit=load_joint_genome(output/'table/best_genome.json')
        physical,values=saved.decode(unit)
        command,fit=fit_table_polynomial(table.targets,values,degree,poly.bound)
        write_json(output/'least_squares.json',{**fit,'targets_deg':table.targets.tolist(),
                   'table_currents':values.tolist(),'coefficients':command.tolist(),
                   'note':'Current approximation is re-evaluated dynamically in polynomial generation 1.'})
    source=output/'polynomial'
    trained=train_joint(cfg,source,degree=degree,
                        generations=generations-(table_generations if method=='staged' else 0),
                        center=poly.encode(physical,command),**common)
    result={**trained,'method':method,'total_generations':generations}
    if final_test:
        result.update(finish_solution(cfg,source,output/'selected',workers=workers))
    for name in ('best_genome.json','best_controller.json','current_law.json'):
        shutil.copy2(source/name,output/name)
    write_json(output/'result.json',result)
    write_json(output/'pipeline.json',{**record,'status':'completed'})
    return result


def sweep_joint(config, output, *, methods=('joint','staged'), degrees=(2,3),
                populations=(30,), probabilities=(.1,), seeds=(42,), generations=200,
                workers=1, seed_genome=None):
    """Compare methods at equal total epoch budgets; test only the validation winner."""
    cfg=config if isinstance(config,GenomeConfig) else GenomeConfig(config)
    axes=[tuple(x) for x in (methods,degrees,populations,probabilities,seeds)]
    if any(not x or len(set(x))!=len(x) for x in axes): raise ValueError('Sweep axes must be nonempty and unique')
    output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for method,degree,population,probability,seed in product(*axes):
        folder=output/f'{method}_degree_{degree}'/f'population_{population:03d}'/f'mutation_{probability:g}'/f'seed_{seed}'
        result=run_pipeline(cfg,folder,method=method,degree=degree,population=population,
                            generations=generations,seed=seed,workers=workers,probability=probability,
                            seed_genome=seed_genome,final_test=False)
        rows.append({**result,'checkpoint_directory':result['directory'],
                     'directory':str(folder.relative_to(output))})
        write_csv(output/'summary.csv',rows)
    chosen=max(rows,key=lambda r:r['validation_fitness'])
    source=output/chosen['directory']/'polynomial'
    result=finish_solution(cfg,source,output/'selected',workers=workers)
    for name in ('best_genome.json','best_controller.json','current_law.json'):
        shutil.copy2(source/name,output/name)
    write_json(output/'result.json',{**chosen,**result})
    # Different degrees have different chromosome dimensions: never mix their RMSE.
    for degree in axes[1]:
        group=[row for row in rows if row['degree']==degree]
        vectors=np.array([load_joint_genome(output/row['directory']/'best_genome.json')[1] for row in group])
        distances=np.sqrt(np.mean((vectors[:,None]-vectors[None,:])**2,axis=2))
        np.savez_compressed(output/f'genome_rmse_degree_{degree}.npz',normalized=vectors,rmse=distances,
                            labels=np.array([row['directory'] for row in group]))
    if cfg.data['setpoint'].get('save_figures',True):
        import matplotlib.pyplot as plt
        from ..visualization import configure_fonts
        configure_fonts(12)
        fig,ax=plt.subplots(figsize=(max(8,len(rows)*.5),5),constrained_layout=True)
        labels=[f"{r['method']}, d={r['degree']}\nN={r['population']}, p={r['mutation_probability']}, s={r['seed']}" for r in rows]
        ax.bar(np.arange(len(rows)),[r['validation_fitness'] for r in rows])
        ax.set(xticks=np.arange(len(rows)),xticklabels=labels,ylabel='Validation fitness',ylim=(-.05,1.05))
        ax.tick_params(axis='x',labelrotation=45)
        fig.savefig(output/'validation_comparison.png',dpi=160);plt.close(fig)
    write_json(output/'completion.json',dict(status='completed',lines=len(rows),selection='validation only'))
    return rows
