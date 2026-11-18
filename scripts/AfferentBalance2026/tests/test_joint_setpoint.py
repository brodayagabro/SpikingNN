"""Check the multi-target chromosome, least squares, data isolation and C artifacts."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.AfferentBalance2026.experiments.core.genome import GenomeConfig,load_genome
from scripts.AfferentBalance2026.experiments.core.evolution import reproduce
from scripts.AfferentBalance2026.experiments.core.run import PopulationEvaluator
from scripts.AfferentBalance2026.experiments.simulation import reference_trace, simulate
from scripts.AfferentBalance2026.experiments.C_setpoint_balance.joint_study import (
    JointGenome,load_joint_genome,load_joint_solution,target_banks,fit_table_polynomial,
    evaluate_joint,run_pipeline,sweep_joint,
)
from scripts.AfferentBalance2026.experiments.C_setpoint_balance.polynomial_study import polynomial_currents
from scripts.AfferentBalance2026.experiments.C_setpoint_balance.run_experiment import run

CONFIG=Path(__file__).resolve().parents[1]/'experiments/C_setpoint_balance/genome_cfg.json'


def tiny_config():
    """Preserve all target angles while shortening episodes for integration tests."""
    data=deepcopy(GenomeConfig(CONFIG).data)
    data['simulation'].update(duration_s=.01,training_count=4,validation_count=4,test_count=4,
                              replay_angles_deg=[-30,30])
    data['setpoint'].update(checkpoints=[1,2],save_figures=False)
    return GenomeConfig(data)


def test_muscle_time_floor_survives_genetic_operations_and_matches_library():
    """Keep calcium/force times at least 50 ms through mutation and library replay."""
    cfg = tiny_config()
    indices = np.array([19, 20, 26, 27])
    rng = np.random.default_rng(43)
    initial = np.vstack([np.zeros(cfg.size), np.ones(cfg.size),
                         rng.random((30, cfg.size))])
    mutated = reproduce(initial, rng.random(len(initial)), rng, .2, 1.)
    for unit in np.vstack([initial, mutated]):
        physical = cfg.decode(unit)
        assert np.all(physical[indices] >= 50.)
        assert np.all(physical[indices] <= np.array([100., 200., 100., 200.]))
    physical = cfg.initial_physical.copy()
    for index in indices:
        invalid = physical.copy()
        invalid[index] = 49.999
        with pytest.raises(ValueError):
            cfg.encode(invalid)
    physical[indices] = 50.
    cfg.encode(physical)
    case = np.array([np.deg2rad(22.5), np.deg2rad(-30), .2, -1, 0])
    currents = np.array([40., 60.])
    _, fast = simulate(physical, case, cfg.mechanics, currents=currents, duration_s=.15)
    library = reference_trace(physical, case, cfg.mechanics,
                              currents=currents, duration_s=.15)
    np.testing.assert_allclose(fast, library, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize('kind,degree,fields,size',[('polynomial',3,60,56),('polynomial',2,58,54),('table',3,62,58)])
def test_joint_roundtrip_and_fixed_physiology(kind,degree,fields,size,tmp_path):
    """Reproduce whole chromosomes while keeping geometry and A fixed; export all genes."""
    genome=JointGenome(tiny_config(),kind,degree)
    assert len(genome.names)==fields and genome.size==size
    rng=np.random.default_rng(21)
    population=reproduce(rng.random((5,size)),rng.random(5),rng,.2,1.)
    for unit in population:
        p,c=genome.decode(unit)
        np.testing.assert_array_equal(p[[18,25,46,47]],[.37,.37,.2,.05])
        np.testing.assert_allclose(genome.encode(p,c),unit,atol=1e-14)
    path=tmp_path/'genome.json';genome.save(path,population[0])
    restored,unit=load_joint_genome(path)
    np.testing.assert_array_equal(unit,population[0])
    payload=json.loads(path.read_text())
    assert len(payload['physical'])==fields and len(payload['parameters'])==fields
    assert all(f['description'] and f['system_parameter'] for f in payload['parameters'])
    payload['physical'][-1]+=1;path.write_text(json.dumps(payload))
    with pytest.raises(AssertionError):load_joint_genome(path)


def test_balanced_targets_and_held_out_interpolation():
    """Every individual sees all anchors; test-only targets and reset states remain unseen."""
    cfg=tiny_config();banks=target_banks(cfg)
    assert [len(np.unique(banks[k][:,0])) for k in ('training','validation','test')]==[5,9,17]
    for bank in banks.values():
        _,counts=np.unique(bank[:,0],return_counts=True)
        np.testing.assert_array_equal(counts,np.full(len(counts),4))
        assert np.max(abs(np.rad2deg(bank[:,1])))<=50+1e-10
    sets=[set(map(tuple,banks[k])) for k in banks]
    assert not any(sets[i]&sets[j] for i in range(3) for j in range(i+1,3))
    assert len(set(banks['test'][:,0])-set(banks['validation'][:,0]))==8
    assert all(len(np.unique(b[:,0]))==5 for b in target_banks(cfg,'table').values())
    bad=deepcopy(cfg.data);bad['setpoint']['targets_deg']=[-45,0,45]
    with pytest.raises(ValueError,match='at least five'):JointGenome(bad)


def test_bounded_least_squares_and_table_target_safety():
    """Recover a known cubic without inventing table values at untrained target angles."""
    cfg=tiny_config();genome=JointGenome(cfg,'table')
    expected=np.array([[3.,4.,-2.,1.],[-5.,2.,3.,-.5]])
    targets=np.linspace(-45,45,5)
    values=np.array([polynomial_currents(expected,np.deg2rad(t)) for t in targets]).T
    actual,fit=fit_table_polynomial(targets,values,3,1000.)
    np.testing.assert_allclose(actual,expected,atol=1e-12)
    assert fit['rmse_nA']<1e-12
    limited,report=fit_table_polynomial(targets,values,2,1.)
    assert abs(limited).max()<=1 and report['rmse_nA']>0
    with pytest.raises(ValueError,match='only replay'):
        genome.currents(values,np.array([[np.deg2rad(11.25),0,0,-1,0]]))


def test_joint_evaluation_applies_target_currents_to_the_same_controller():
    """Check all five target commands against direct integrations, with no target feedback."""
    cfg=tiny_config();genome=JointGenome(cfg)
    law=np.array([[1.,2.,3.,4.],[-2.,1.,0.,-1.]])
    unit=genome.encode(cfg.initial_physical,law)
    cases=target_banks(cfg)['training']
    with PopulationEvaluator(cfg.mechanics) as evaluator:
        _,metrics=evaluate_joint(evaluator,genome,unit[None],cases,.01)
    for i,case in enumerate(cases):
        expected,_=simulate(cfg.initial_physical,case,cfg.mechanics,
                            currents=polynomial_currents(law,case[0]),duration_s=.01,stride=0)
        np.testing.assert_allclose(metrics[0,i],expected,rtol=1e-12,atol=1e-12)


@pytest.mark.parametrize('method',['joint','staged'])
def test_pipeline_replays_joint_winner_and_preserves_all_generations(method,tmp_path):
    """Exercise both real training paths, LS seeding, test export and replay compatibility."""
    cfg=tiny_config();out=tmp_path/method
    result=run_pipeline(cfg,out,method=method,population=3,generations=4,workers=1)
    genome,unit=load_joint_genome(out/'best_genome.json')
    assert result['test_cases']==68 and result['test_used_for_selection'] is False
    assert len(list(out.glob('*/generations/*.npz')))==4
    p,law=genome.decode(unit)
    _,compat=load_genome(out/'best_controller.json');np.testing.assert_array_equal(p,compat)
    restored,chosen,meta=load_joint_solution(out/'polynomial/final_population.npz')
    assert meta['selection']=='validation_scores' and restored.size==56
    with np.load(out/'polynomial/generations/generation_0001.npz') as first:
        assert first['physical'].shape==(3,60) and first['normalized'].shape==(3,56)
        assert first['training_metrics'].shape==(3,20,9)
        assert len(np.unique(first['physical'][:,:52],axis=0))==3
        if method=='staged':
            fitted=json.loads((out/'least_squares.json').read_text())['coefficients']
            np.testing.assert_allclose(first['command_values'][0],fitted,atol=1e-12)
            table,table_unit=load_joint_genome(out/'table/best_genome.json')
            np.testing.assert_allclose(first['controller_physical'][0],table.decode(table_unit)[0])
    replay=tmp_path/f'{method}_replay'
    run(cfg,replay,genome=out/'best_genome.json',targets=[22.5],angles=[-30],duration_s=.01)
    with np.load(replay/'traces/case_000.npz') as data:
        np.testing.assert_allclose(data['currents'],polynomial_currents(law,np.deg2rad(22.5)))
        np.testing.assert_array_equal(data['physical'],p)
    with pytest.raises(FileExistsError):run_pipeline(cfg,out,method=method,population=3,generations=4)


def test_sweep_tests_only_the_validation_winner(tmp_path):
    """Compare equal epoch budgets and keep every trial's test bank out of selection."""
    out=tmp_path/'sweep'
    rows=sweep_joint(tiny_config(),out,methods=['joint','staged'],degrees=[2],populations=[3],
                     probabilities=[.1],seeds=[42],generations=2)
    assert len(rows)==2 and all(r['total_generations']==2 for r in rows)
    assert all('test_fitness' not in row for row in rows)
    best=max(rows,key=lambda r:r['validation_fitness'])
    assert json.loads((out/'result.json').read_text())['directory']==best['directory']
    assert len(list(out.rglob('test_metrics.npz')))==1


def test_method_diagnostics_preserve_selection_and_full_state_traces(tmp_path):
    """Publish each method by validation and retain real full-resolution grid inputs."""
    from scripts.AfferentBalance2026.experiments.C_setpoint_balance.sweep_diagnostics import publish_methods
    out=tmp_path/'study'
    rows=sweep_joint(tiny_config(),out,methods=['joint','staged'],degrees=[2],populations=[3],
                     probabilities=[.1],seeds=[42],generations=2)
    summary=publish_methods(out,grid_size=2,workers=1,figures=False)
    assert {r['method'] for r in summary}=={'joint','staged'}
    for row in summary:
        selected=next(r for r in rows if r['method']==row['method'])
        assert selected['validation_fitness']==row['validation_fitness']
        folder=out/row['directory'];genome,unit=load_joint_genome(folder/'best_genome.json')
        physical,law=genome.decode(unit)
        assert row['grid_cases']==20
        assert len(list((folder/'diagnostics').rglob('case_*.npz')))==31
        with np.load(folder/'diagnostics/example/case_000.npz') as data:
            expected=polynomial_currents(law,np.deg2rad(22.5))
            np.testing.assert_allclose(data['currents'],expected,atol=1e-12)
            np.testing.assert_allclose(data['case'][:3],[np.deg2rad(22.5),np.deg2rad(-30.),0.])
            metrics,trace=simulate(physical,data['case'],genome.controller.mechanics,
                                   currents=expected,duration_s=.01,stride=1)
            np.testing.assert_array_equal(data['trace'],trace)
            np.testing.assert_array_equal(data['metrics'],metrics)
    with pytest.raises(FileExistsError):publish_methods(out,grid_size=2,figures=False)
