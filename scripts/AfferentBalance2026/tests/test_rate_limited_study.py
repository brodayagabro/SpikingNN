"""B/C rate-constrained search: unchanged trajectories, strict ISIs and explicit bounds."""
from pathlib import Path
import sys

import numpy as np
import pytest

SOURCE=Path(__file__).resolve().parents[1]/'topubsrc'
sys.path.insert(0,str(SOURCE))
from evolution import profile_inputs, make_genome, initial_population, seed_coordinates
from genome import load_solution
from model import (TRACE_COLUMNS, firing_interval_steps, simulate, measure_trace,
                   measure_activity, evaluate_population, fitness)
from run import study_specs


@pytest.mark.parametrize('folder,profile',[
    ('B_balance','fast'),('B_balance','slow'),('C_setpoint_balance','fast'),
    ('C_setpoint_balance','slow'),('C_setpoint_balance','mirror')])
def test_bounds_mutable_N_and_recorded_constraint(folder,profile,tmp_path):
    cfg,settings=profile_inputs(SOURCE/folder,profile)
    assert cfg['simulation']['max_firing_rate_hz']==100
    assert settings['generations']==500
    for i in (31,34,37,40):
        assert (cfg['parameters'][i]['minimum'],cfg['parameters'][i]['maximum'])==(-50,50)
        assert abs(cfg['parameters'][i]['value'])==15
    for kind in (('table','polynomial') if folder.startswith('C') else (None,)):
        genome=make_genome(cfg,kind)
        population=initial_population(seed_coordinates(genome),50,np.random.default_rng(42))
        physical=np.array([genome.decode(row)[0] for row in population])
        for i in (17,24):
            assert cfg['parameters'][i]['mutable']
            assert (physical[:,i]>=1).all() and (physical[:,i]<=40).all()
            assert np.ptp(physical[:,i])>1
        assert (abs(physical[:,[31,34,37,40]])<=50).all()
        assert (physical[:,[49,50]]<.1).all()
        path=tmp_path/f'{kind}.json';genome.save(path,population[0])
        restored,_=load_solution(path)
        assert restored.controller.data['simulation']['max_firing_rate_hz']==100


@pytest.mark.parametrize('invalid',[0,-1,float('nan'),float('inf'),True,'100'])
def test_rate_validation(invalid):
    with pytest.raises(ValueError): firing_interval_steps(invalid)


def trace_fixture(steps=100000):
    trace=np.zeros((steps,len(TRACE_COLUMNS)))
    trace[:,0]=np.arange(1,steps+1)*.0001
    return trace


def test_ISI_boundary_and_single_violating_episode_cannot_be_hidden():
    assert firing_interval_steps(100)==100
    assert firing_interval_steps(None)==0
    trace=trace_fixture()
    trace[[99,199],19]=30
    allowed=measure_trace(trace,max_firing_rate_hz=100)
    assert allowed[0]==pytest.approx(1) and allowed[5]==1
    trace[199,19]=0;trace[198,19]=30
    violation=measure_trace(trace,max_firing_rate_hz=100)
    assert violation[0]==pytest.approx(-100/99) and violation[5]==0
    bank=np.tile(allowed,(100,1));bank[-1]=violation
    assert fitness(bank)==violation[0]<0
    # A burst straddling a reporting-bin boundary still violates the limit.
    trace[:,19]=0;trace[[9998,10000],19]=30
    assert measure_trace(trace,max_firing_rate_hz=100)[0]==-50


def test_no_interval_does_not_create_a_false_violation():
    trace=trace_fixture(1);trace[0,19]=30
    assert measure_trace(trace,duration_s=.0001,max_firing_rate_hz=100)[0]>=0


def test_constraint_changes_scoring_not_neural_or_muscle_dynamics():
    cfg,_=profile_inputs(SOURCE/'B_balance','fast')
    genome=make_genome(cfg);p,_=genome.decode(seed_coordinates(genome))
    case=np.array([0.,0.,0.,-1.,0.])
    kw=dict(currents=(1000.,1000.),duration_s=.1,stride=1,fitness_window_s=.05)
    old,trace=simulate(p,case,genome.controller.mechanics,**kw)
    capped,capped_trace=simulate(p,case,genome.controller.mechanics,max_firing_rate_hz=100,**kw)
    np.testing.assert_array_equal(trace,capped_trace)
    assert capped[0]<-1 and old[0]>=0
    np.testing.assert_array_equal(capped[[1,2,3,4,6,7,8]],old[[1,2,3,4,6,7,8]])
    independent=measure_trace(trace[1:],duration_s=.1,fitness_window_s=.05,max_firing_rate_hz=100,initial_theta=0.)
    np.testing.assert_allclose(capped,independent,rtol=1e-12,atol=1e-12)
    activity=measure_activity(dict(metrics=capped,trace=trace,duration_s=.1,fitness_window_s=.05,max_firing_rate_hz=100))
    assert activity['completed'] and activity['late_rate_hz'] is not None
    assert not activity['rate_limit_observed_pass']
    assert max(activity['peak_instantaneous_rate_hz'])>100


def test_population_scoring_uses_saved_constraint():
    cfg,_=profile_inputs(SOURCE/'C_setpoint_balance','fast')
    cfg['simulation']['fitness_window_s']=.05
    genome=make_genome(cfg,'polynomial')
    physical,_=genome.decode(seed_coordinates(genome))
    coefficients=np.zeros((2,3));coefficients[:,0]=1000
    unit=genome.encode(physical,coefficients)
    case=np.array([[0.,0.,0.,-1.,0.]])
    scores,metrics=evaluate_population(genome,unit[None],case,.1)
    expected,_=simulate(physical,case[0],genome.controller.mechanics,currents=(1000.,1000.),
                        duration_s=.1,stride=0,fitness_window_s=.05,max_firing_rate_hz=100)
    np.testing.assert_array_equal(metrics[0,0],expected)
    assert scores[0]==expected[0]<-1


def test_new_sweep_is_nineteen_searches_with_no_budget_above_500():
    specs=study_specs(True)
    assert len(specs)==19
    assert {label for label,_,_ in specs}=={'B:fast','B:slow','C:fast','C:slow','C:mirror'}
    assert max(settings['generations'] for _,_,settings in specs)==500
    for task in 'BC':
        fast=[settings for label,_,settings in specs if label==task+':fast']
        assert {(s['generations'],s['seed'],s['mutation_probability']) for s in fast}=={
            (g,seed,p) for g in (200,500) for seed in (42,43) for p in (.1,.3)}
