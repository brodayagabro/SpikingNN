"""The declared Cartesian grid, live validation checkpoints and hyperparameter ranking."""
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest

SOURCE=Path(__file__).resolve().parents[1]/'topubsrc'
sys.path.insert(0,str(SOURCE))
import evolution
import run
from genome import file_hash, write_json, load_solution


def test_default_grid_is_exactly_324_unique_fresh_searches():
    specs=run.grid_specs()
    assert len(specs)==324 and len({path for _,path,_ in specs})==324
    actual={(label,s['population'],s['generations'],s['seed'],s['mutation_probability']) for label,_,s in specs}
    expected={(task+':fast',n,g,seed,.3) for task in 'BC' for n in range(10,51,5)
              for g in range(100,501,50) for seed in (42,43)}
    assert actual==expected
    assert sum(s['population']*s['generations'] for _,_,s in specs)==2916000
    for task in 'BC':
        separate=run.grid_specs(experiments=(task,))
        assert len(separate)==162
        assert separate==[row for row in specs if row[0]==task+':fast']


@pytest.mark.parametrize('triple',[(10,50,0),(50,10,5),(10,51,5)])
def test_range_is_positive_and_inclusive(triple):
    with pytest.raises(ValueError):run.inclusive_range(triple)


@pytest.mark.parametrize('kwargs',[{'seeds':(42,42)},{'populations':(2,)},{'generations':(1,)},
                                  {'experiments':('A',)},{'mutation_probability':float('nan')}])
def test_invalid_axes_are_rejected(kwargs):
    with pytest.raises(ValueError):run.grid_specs(**kwargs)


@pytest.mark.parametrize('tasks,count',[(['B','C'],324),(['B'],162),(['C'],162)])
def test_documented_cli_plan_only_has_no_training_or_output(tasks,count,tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('Plan-only must not start a process')
    monkeypatch.setattr(run.subprocess,'run',forbidden)
    out=tmp_path/'grid'
    report=run.main(['grid','--experiments',*tasks,'--population-range','10','50','5',
                    '--generation-range','100','500','50','--seeds','42','43',
                    '--mutation-probability','0.3','--workers','1','--jobs','2',
                    '--output',str(out),'--plan-only'])
    assert report['searches']==count and not report['training_performed']
    assert not out.exists()


def test_new_best_genomes_exist_while_training_and_all_improvements_survive(tmp_path,monkeypatch):
    config,settings=evolution.profile_inputs(SOURCE/'B_balance','fast')
    settings.update(population=3,generations=3,validation_every=1,workers=1)
    genome=evolution.make_genome(config)
    banks={name:np.zeros((2,5)) for name in ('training','validation','test')}
    validations=0
    def fake_evaluate(g,population,cases,duration):
        nonlocal validations
        assert cases is banks['training'] or cases is banks['validation']
        if cases is banks['validation']:validations+=1
        scores=np.arange(len(population))*.01+(validations*.1 if cases is banks['validation'] else .01)
        metrics=np.zeros((len(population),len(cases),9));metrics[:,:,0]=scores[:,None]
        return scores,metrics
    monkeypatch.setattr(evolution,'evaluate_population',fake_evaluate)
    original=evolution.optimize;observed=[];output=tmp_path/'training'
    def checked_optimize(*args,**kwargs):
        observe=kwargs['on_state']
        def check(generation,*rest):
            observe(generation,*rest)
            selection=json.loads((output/'selection.json').read_text())
            archive=output/selection['genome']
            assert archive.exists() and file_hash(archive)==selection['genome_sha256']
            assert file_hash(archive)==file_hash(output/'best_genome.json')
            assert selection['generation']==generation
            assert not (output/'final_population.npz').exists()
            observed.append(generation)
        kwargs['on_state']=check
        return original(*args,**kwargs)
    monkeypatch.setattr(evolution,'optimize',checked_optimize)
    winner=evolution.train_phase(genome,banks,settings,output)
    assert observed==[1,2,3] and validations==4
    assert len(list((output/'best_genomes').glob('*_genome.json')))==4
    assert winner['generation']==3
    saved=json.loads((output/'best_genome.json').read_text())
    assert 'workers' not in saved['metadata']['phase_training']
    assert saved['metadata']['phase_training']['population']==3
    restored,_=load_solution(output/'best_genome.json')
    assert restored.controller.data==config
    history=list(csv.DictReader((output/'best_history.csv').open()))
    assert [int(row['generation']) for row in history]==[0,1,2,3]


def test_live_ranking_uses_validation_and_excludes_C_table_stage(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'HERE',tmp_path/'source')
    settings=dict(population=10,generations=100,seed=42,mutation_probability=.3)
    plan=[dict(profile='B:fast',run='expanded_B/run1',training=settings),
          dict(profile='B:fast',run='expanded_B/run2',training={**settings,'seed':43}),
          dict(profile='C:fast',run='expanded_C/run3',training={**settings,'method':'staged'})]
    def stage(item,name,fitness):
        folder=tmp_path/item['run'];folder.mkdir(parents=True,exist_ok=True)
        write_json(folder/'training.json',item['training']);out=folder/name;out.mkdir(exist_ok=True)
        write_json(out/'best_genome.json',{})
        write_json(out/'selection.json',dict(fitness=fitness,generation=20,genome='best_genome.json',
                    genome_sha256=file_hash(out/'best_genome.json'),validation_cases=48,validation_success=0,
                    rate_limit_feasible=True))
        write_json(out/'progress.json',dict(generation=40,generations=100,status='running'))
    stage(plan[0],'training',.5);stage(plan[1],'training',.4);stage(plan[2],'table',999.)
    result={bank:dict(fitness=value,cases=48,success=48) for bank,value in
            [('training',1.),('validation',.4),('test',1.)]}
    write_json(tmp_path/plan[1]['run']/'result.json',result)
    write_json(tmp_path/plan[1]['run']/'activity.json',{})
    report=run.live_leaderboard(tmp_path,plan)
    assert len(report['leaders'])==1 and report['leaders'][0]['run']==plan[0]['run']
    assert report['leaders'][0]['status']=='running' and report['leaders'][0]['test_fitness'] is None
    assert report['completed_leaders'][0]['run']==plan[1]['run']
    assert len(report['table_stage_runs'])==1
    stage(plan[2],'polynomial',.3)
    report=run.live_leaderboard(tmp_path,plan,persist=True)
    assert len(report['leaders'])==2
    assert report['leaders'][1]['validation_fitness']==.3
    assert (tmp_path/'leaderboard.csv').exists()
    assert 'G budget' in (tmp_path/'leaderboard.md').read_text()
