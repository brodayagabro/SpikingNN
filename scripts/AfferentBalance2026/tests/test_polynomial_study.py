"""Polynomial commands must preserve dynamics, parallel evaluation and selection boundaries."""
import numpy as np
import pytest
from scripts.SteppingReflexGA.experiments.core.commands import polynomial_currents, polynomial_parameters
from scripts.SteppingReflexGA.experiments.core.genome import ControllerDomain
from scripts.SteppingReflexGA.experiments.core.population import PopulationEvaluator
from scripts.SteppingReflexGA.experiments.core.dynamics import rollout
from scripts.SteppingReflexGA.experiments.core.polynomial_study import study_banks, full_command_trace
from scripts.SteppingReflexGA.experiments.core.evolution import optimize


def test_polynomial_has_constant_term_and_changes_only_currents():
    p=ControllerDomain(True).decode(ControllerDomain(True).initial());before=p.copy()
    coefficients=np.array([[1.,2.,3.,4.],[-3.,7.,-2.,1.]])
    np.testing.assert_array_equal(polynomial_currents(coefficients,0),[1.,-3.])
    np.testing.assert_allclose(polynomial_currents(coefficients,np.pi/8),[3.25,.125])
    commanded=polynomial_parameters(p,coefficients,np.pi/8)
    np.testing.assert_array_equal(p,before)
    unchanged=np.r_[np.arange(46),np.arange(48,52)]
    np.testing.assert_array_equal(commanded[unchanged],p[unchanged])
    for invalid in (np.zeros((2,2)),np.zeros((3,3)),np.full((2,3),np.nan)):
        with pytest.raises(ValueError):polynomial_currents(invalid,0)
    with pytest.raises(ValueError):polynomial_currents(coefficients,np.pi/3)


def test_polynomial_parallel_matches_direct_rollout_and_cache_isolates_laws():
    domain=ControllerDomain(True);p=domain.decode(domain.initial())
    cases=np.array([[0.,.03,.1,-1.,0.],[.2,.1,-.1,-1.,0.]])
    coeff=np.array([[[30.,-20.,15.],[10.,5.,-3.]]])
    with PopulationEvaluator(1,cache_size=10) as serial,PopulationEvaluator(2) as parallel:
        a=serial.evaluate(p[None,:],coeff,cases,.1,command_kind='polynomial')
        b=parallel.evaluate(p[None,:],coeff,cases,.1,command_kind='polynomial')
        legacy=serial.evaluate(p[None,:],coeff,cases,.1)
        repeated=serial.evaluate(p[None,:],coeff,cases,.1,command_kind='polynomial')
    for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
    for x,y in zip(a,repeated):np.testing.assert_array_equal(x,y)
    assert not np.array_equal(a[1],legacy[1])
    expected=np.array([rollout(polynomial_parameters(p,coeff[0],c[0]),c[1:],target_rad=c[0],duration_s=.1)[0] for c in cases])
    np.testing.assert_array_equal(a[1][0],expected)


def test_study_banks_keep_test_and_interpolation_separate():
    banks=study_banks()
    assert {k:len(v) for k,v in banks.items()}==dict(training=15,validation=25,test=50,interpolation=32)
    for name in ('training','validation','test'):
        np.testing.assert_allclose(np.rad2deg(np.unique(banks[name][:,0])),[-45,-22.5,0,22.5,45])
    assert not set(banks['training'][:,0])&set(banks['interpolation'][:,0])
    for first,second in [('training','validation'),('training','test'),('validation','test')]:
        assert not set(map(tuple,banks[first]))&set(map(tuple,banks[second]))


def test_full_command_recording_preserves_every_state():
    domain=ControllerDomain(True);p=domain.decode(domain.initial())
    c=np.array([[5.,1.,2.],[7.,-2.,1.]])
    case=np.array([.1,.05,.1,.015,-.2])
    m,trace,commanded=full_command_trace(p,c,case,.03)
    expected,raw=rollout(commanded,case[1:],target_rad=case[0],duration_s=.03,stride=1)
    np.testing.assert_array_equal(m,expected)
    np.testing.assert_array_equal(trace[1:],raw)
    assert trace.shape==(301,17) and trace[0,0]==0
    np.testing.assert_array_equal(trace[0,1:3],case[1:3])


def test_warm_starts_are_evaluated_without_changing_ga_selection():
    center=np.array([.5,.5]);starts=np.array([[.1,.2],[.7,.8]])
    def evaluate(population):
        scores=-np.sum(population**2,axis=1)
        metrics=np.zeros((len(population),1,9));metrics[:,0,0]=scores
        return scores,metrics
    population,*_=optimize(center,evaluate,size=5,generations=1,initial_candidates=starts)
    np.testing.assert_array_equal(population[0],center)
    np.testing.assert_array_equal(population[1:3],starts)
