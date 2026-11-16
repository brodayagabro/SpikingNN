"""Check standalone publication code against the existing scientific implementation."""
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numba
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PUBLICATION = ROOT/'topubsrc'
sys.path.insert(0, str(PUBLICATION))
pub_model = importlib.import_module('model')
pub_genome = importlib.import_module('genome')
pub_evolution = importlib.import_module('evolution')

from scripts.AfferentBalance2026.experiments.simulation import (
    PARAMETERS, simulate, reference_trace, balance_banks,
)
from scripts.AfferentBalance2026.experiments.core.evolution import reproduce
from scripts.AfferentBalance2026.experiments.C_setpoint_balance.joint_study import (
    target_banks, JointGenome,
)

EXAMPLES = sorted(PUBLICATION.glob('*/genomes/*.json'))


def test_initial_population_matches_documented_draw_order():
    """Keep the centre, use reflected Gaussian neighbours and a uniform last quarter."""
    center = np.linspace(0, 1, 48)
    rng, reference = np.random.default_rng(42), np.random.default_rng(42)
    actual = pub_evolution.initial_population(center, 50, rng)
    raw = center + reference.normal(0, .1, (50, 48))
    expected = 1-abs(np.mod(raw, 2)-1)
    expected[0] = center
    expected[-12:] = reference.random((12, 48))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-16)
    np.testing.assert_array_equal(actual[0], center)
    np.testing.assert_array_equal(actual[-12:], expected[-12:])
    assert rng.bit_generator.state == reference.bit_generator.state


@pytest.mark.parametrize('folder,domain', [
    ('A_balance','fast'), ('A_balance','slow'), ('B_balance','fast'), ('B_balance','slow'),
    ('C_setpoint_balance','fast'), ('C_setpoint_balance','slow'), ('C_setpoint_balance','mirror')])
def test_fresh_profiles_retain_article_domains_without_loading_controllers(folder, domain, monkeypatch):
    reference, _ = pub_genome.load_solution(PUBLICATION/folder/'genomes'/f'{domain}.json')
    def forbidden(*args, **kwargs):
        raise AssertionError('A fresh profile must not read a saved controller')
    monkeypatch.setattr(pub_evolution, 'load_solution', forbidden)
    config, settings = pub_evolution.profile_inputs(PUBLICATION/folder, domain)
    genome = pub_evolution.make_genome(config)
    for actual, expected in zip(config['parameters'], reference.controller.data['parameters']):
        for key in ('name','value','minimum','maximum','scale','mutable'):
            assert actual.get(key) == expected.get(key)
    assert genome.controller.mirror_symmetry == reference.controller.mirror_symmetry
    p, command = genome.decode(pub_evolution.seed_coordinates(genome))
    np.testing.assert_allclose(p, genome.controller.initial_physical, rtol=1e-14)
    np.testing.assert_array_equal(command, np.zeros_like(command))


def test_sweep_cartesian_grid_and_validation_selection(tmp_path, monkeypatch):
    module = importlib.import_module('C_setpoint_balance.experiment')
    calls = []
    def fake_run(config, settings, output, bank_factory, **kwargs):
        assert kwargs['saved'] is None
        calls.append((settings.copy(), deepcopy(config)))
        val = len(calls)/100
        return {bank:dict(fitness=(val if bank=='validation' else 1-val),
                          success=0, cases=1, mean_late_rmse_deg=90.)
                for bank in ('training','validation','test')}
    monkeypatch.setattr(pub_evolution, 'run_training', fake_run)
    out = tmp_path/'sweep'
    pub_evolution.cli('C_setpoint_balance', PUBLICATION/'C_setpoint_balance', module.make_banks,
        module.train_experiment, ['sweep','--profile','fast','--populations','3','--seeds','42',
        '--mutation-probabilities','.1','.3','--methods','joint','staged','--degrees','2','3',
        '--output',str(out)])
    assert len(calls)==8
    assert {(s['mutation_probability'],s['method'],c['setpoint']['degree']) for s,c in calls} == {
        (p,m,d) for p in [.1,.3] for m in ['joint','staged'] for d in [2,3]}
    selected = json.loads((out/'selection.json').read_text())['selected']
    assert selected['validation_fitness']==.08
    assert selected['test_fitness']==.92


@pytest.mark.parametrize('payload', [b'', b'abc', bytes(range(256))*4097],
                         ids=['empty', 'text', 'binary_over_one_megabyte'])
def test_file_hash_without_python_311_api(tmp_path, monkeypatch, payload):
    """Provenance and verification must work without hashlib.file_digest."""
    path = tmp_path/'input.bin'
    path.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    monkeypatch.delattr(hashlib, 'file_digest', raising=False)
    assert pub_genome.file_hash(path) == expected


@pytest.mark.parametrize('path', EXAMPLES, ids=lambda p:f'{p.parent.parent.name}/{p.name}')
def test_published_genomes_roundtrip_and_dynamics_match_original(path):
    """Every included example retains all physical fields and the full neural trace."""
    genome, unit = pub_genome.load_solution(path)
    saved = json.loads(path.read_text())
    physical, command = genome.decode(unit)
    np.testing.assert_allclose(np.r_[physical, command.ravel()], saved['physical'], rtol=0, atol=1e-11)
    restored, restored_unit = pub_genome.load_solution(genome.payload(unit))
    np.testing.assert_array_equal(restored_unit, unit)
    np.testing.assert_array_equal(restored.controller.mechanics, genome.controller.mechanics)
    assert pub_model.PARAMETERS == PARAMETERS
    target = np.deg2rad(22.5) if genome.kind == 'polynomial' else 0.
    case = np.array([target, np.deg2rad(-30.), .2, .04, -.1])
    currents = genome.currents(command, case[None])[0]
    expected_metrics, expected_trace = simulate(physical, case, genome.controller.mechanics,
                                               currents=currents, duration_s=.08)
    metrics, trace = pub_model.simulate(physical, case, genome.controller.mechanics,
                                        currents=currents, duration_s=.08)
    np.testing.assert_allclose(trace, expected_trace, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(metrics, expected_metrics, atol=1e-12, rtol=1e-12)
    library = reference_trace(physical, case, genome.controller.mechanics,
                              currents=currents, duration_s=.08)
    np.testing.assert_allclose(trace, library, atol=1e-8, rtol=1e-8)


def test_population_parallelism_preserves_individuals_cases_and_fitness():
    """Thread count must not affect data ordering, metrics or the selected rank."""
    genome, unit = pub_genome.load_solution(PUBLICATION/'C_setpoint_balance/genomes/mirror.json')
    population = np.vstack([unit, np.clip(unit+.0001, 0, 1), np.clip(unit-.0001, 0, 1)])
    cases = np.array([[np.deg2rad(t), np.deg2rad(a), .1, -1., 0.]
                      for t, a in [(45,-30),(-45,30),(0,5)]])
    old_threads = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        serial = pub_model.evaluate_population(genome, population, cases, .03)
        numba.set_num_threads(min(2, numba.config.NUMBA_NUM_THREADS))
        parallel = pub_model.evaluate_population(genome, population, cases, .03)
    finally:
        numba.set_num_threads(old_threads)
    for first, second in zip(serial, parallel):
        np.testing.assert_array_equal(first, second)
    for i, candidate in enumerate(population):
        physical, command = genome.decode(candidate)
        for j, case in enumerate(cases):
            current = genome.currents(command, case[None])[0]
            metrics, _ = simulate(physical, case, genome.controller.mechanics,
                                  currents=current, duration_s=.03, stride=0)
            np.testing.assert_allclose(serial[1][i, j], metrics, rtol=1e-12, atol=1e-12)
        assert serial[0][i] == pub_model.fitness(serial[1][i])


def test_published_case_banks_match_archived_protocol():
    """All target/initial-state combinations and independent seeds are retained."""
    for folder in ('A_balance','B_balance','C_setpoint_balance'):
        module = importlib.import_module(f'{folder}.experiment')
        data = json.loads((PUBLICATION/folder/'genome_cfg.json').read_text())
        for kind in (('polynomial','table') if folder.startswith('C') else ('none',)):
            actual = module.make_banks(data, kind)
            expected = target_banks(data,kind) if folder.startswith('C') else balance_banks(data['simulation'])
            for bank in ('training','validation','test'):
                np.testing.assert_array_equal(actual[bank], expected[bank])


def test_ga_preserves_original_random_operator_order():
    """The standalone GA must not change elitism, tournament, mutation or reflection."""
    rng = np.random.default_rng(19)
    population = rng.random((12, 48));scores = rng.random(12)
    actual = pub_evolution.reproduce(population,scores,np.random.default_rng(43),.2,.3)
    expected = reproduce(population,scores,np.random.default_rng(43),.2,.3)
    np.testing.assert_array_equal(actual, expected)


def test_fixed_and_mirror_coordinates_survive_changed_configuration():
    """Changing trainable fields recalculates dimensions without changing their order."""
    data = json.loads((PUBLICATION/'A_balance/genome_cfg.json').read_text())
    before = pub_genome.GenomeConfig(data)
    data['parameters'][0]['mutable'] = False
    fixed = pub_genome.GenomeConfig(data)
    assert fixed.size == before.size-1
    physical = fixed.decode(np.full(fixed.size,.8))
    assert physical[0] == data['parameters'][0]['value']
    mirror, unit = pub_genome.load_solution(PUBLICATION/'C_setpoint_balance/genomes/mirror.json')
    physical, command = mirror.decode(np.clip(unit+.01,0,1))
    np.testing.assert_array_equal(physical,physical[pub_genome.MIRROR_INDICES])
    np.testing.assert_array_equal(command[1],command[0]*[1,-1,1])


def test_validation_incumbent_is_retained_and_tests_are_not_used_in_search(tmp_path, monkeypatch):
    """An early validation winner must survive a better final training champion."""
    cfg = json.loads((PUBLICATION/'A_balance/genome_cfg.json').read_text())
    genome = pub_evolution.make_genome(cfg)
    center = np.full(genome.size,.5)
    banks = {name:np.array([[tag,0.,0.,-1.,0.]])
             for name,tag in [('training',0.),('validation',.1),('test',.2)]}
    calls = []

    def evaluate(codec, population, cases, duration):
        tag = cases[0,0];calls.append(tag)
        scores = population[:,0] if tag == 0 else 1-population[:,0]
        metrics = np.zeros((len(population),len(cases),9));metrics[:,:,0] = scores[:,None]
        metrics[:,:,1] = duration
        return scores,metrics

    def optimize(center, evaluate_training, *, on_state, **kwargs):
        history=[]
        for generation,value in [(1,.1),(2,.9)]:
            population=np.full((3,len(center)),value)
            scores,metrics=evaluate_training(population)
            history.append(dict(generation=generation,best_fitness=float(scores.max())))
            on_state(generation,population,scores,metrics,history,{})
        return population,scores,metrics,history,{}

    monkeypatch.setattr(pub_evolution,'evaluate_population',evaluate)
    monkeypatch.setattr(pub_evolution,'optimize',optimize)
    settings=dict(population=3,generations=2,seed=42,mutation_probability=.3,
                  sigma_start=.2,sigma_end=.015,validation_every=1,workers=1)
    winner=pub_evolution.train_phase(genome,banks,settings,tmp_path/'training',center=center)
    assert winner['generation']==1 and winner['fitness']==.9
    assert .2 not in calls
    summary=pub_evolution.finish_run(genome,winner,banks,tmp_path)
    assert calls.count(.2)==1 and summary['test_used_for_selection'] is False
    selected,unit=pub_genome.load_solution(tmp_path/'best_genome.json')
    np.testing.assert_array_equal(unit,np.full(genome.size,.1))
    with np.load(tmp_path/'training/final_population.npz') as saved:
        np.testing.assert_array_equal(saved['normalized'],np.full((3,genome.size),.9))


def test_replay_does_not_apply_nonzero_targets_to_balance():
    genome,unit=pub_genome.load_solution(PUBLICATION/'A_balance/genomes/fast.json')
    with pytest.raises(ValueError,match='target_deg must be zero'):
        pub_model.replay_genome(genome,unit,target_deg=10.)


def test_minimum_staged_budget_completes_table_and_polynomial(tmp_path):
    """Two total generations are valid even though each phase has just one."""
    module = importlib.import_module('C_setpoint_balance.experiment')
    config = json.loads((PUBLICATION/'C_setpoint_balance/genome_cfg.json').read_text())
    config['simulation'].update(duration_s=.02, training_count=4, validation_count=4, test_count=4)
    settings = dict(population=3, generations=2, seed=43, mutation_probability=.3,
                    sigma_start=.2, sigma_end=.015, validation_every=1, workers=1, method='staged')
    output = tmp_path/'staged'
    result = pub_evolution.run_training(config, settings, output, module.make_banks,
                                         runner=module.train_experiment, smoke=True)
    assert result['status']=='completed' and result['test_cases']==68
    for stage in ('table','polynomial'):
        with np.load(output/stage/'final_population.npz') as population:
            assert int(population['generation'])==1
    selected, unit = pub_genome.load_solution(output/'best_genome.json')
    assert selected.kind=='polynomial' and selected.degree==2
    assert settings['generations']==2 and settings['method']=='staged'


@pytest.mark.parametrize('window', [0, -.1, float('nan'), .00015, '2'])
def test_invalid_fitness_windows_are_rejected(window):
    with pytest.raises(ValueError, match='fitness_window_s'):
        pub_model.validate_fitness_window(window)


def test_fitness_window_changes_scoring_but_preserves_the_physical_trace():
    """Check two late intervals against a separate NumPy reduction of the trace."""
    genome, unit = pub_genome.load_solution(PUBLICATION/'C_setpoint_balance/genomes/fast.json')
    results = []
    for window in (.1, .9):
        result = pub_model.replay_genome(genome, unit, target_deg=45., duration_s=1.,
                                          stride=1, fitness_window_s=window)
        trace = result['trace'][1:]
        assert len(trace) == 10000
        error = trace[:, 1]-np.deg2rad(45.)
        speed = trace[:, 2]
        tail = error[-round(window*10000):]
        reward = np.maximum(0., np.exp(-(error/np.deg2rad(15.))**2)-.05*np.minimum(speed**2,4.))
        outside = (abs(error)>np.deg2rad(3.)) | (abs(speed)>.35)
        last_outside = trace[outside,0][-1] if outside.any() else 0.
        success = float(last_outside<=min(3.,1.-window))
        score = .10+.30*reward.mean()+.45*np.exp(-(tail/np.deg2rad(3.))**2).mean()+.15*success
        np.testing.assert_allclose(result['metrics'][[0,5,7]],
                                   [score,success,np.sqrt(np.mean(tail**2))], rtol=1e-11,atol=1e-11)
        reconstructed = pub_model.measure_trace(trace, target_rad=np.deg2rad(45.), duration_s=1.,
                                                  initial_theta=np.deg2rad(-30.), fitness_window_s=window)
        np.testing.assert_allclose(result['metrics'],reconstructed,rtol=1e-11,atol=1e-11)
        assert result['fitness_window_s']==window
        results.append(result)
    np.testing.assert_array_equal(results[0]['trace'],results[1]['trace'])
    assert results[0]['metrics'][0]!=results[1]['metrics'][0]


def test_population_and_replay_use_the_saved_fitness_window():
    """The window in the embedded configuration must reach training and replay."""
    source, unit = pub_genome.load_solution(PUBLICATION/'C_setpoint_balance/genomes/fast.json')
    config = deepcopy(source.controller.data)
    config['simulation']['fitness_window_s'] = .1
    genome = pub_evolution.make_genome(config)
    result = pub_model.replay_genome(genome,unit,target_deg=45.,duration_s=1.,stride=1)
    scores,metrics = pub_model.evaluate_population(genome,unit[None],result['case'][None],1.)
    np.testing.assert_array_equal(metrics[0,0],result['metrics'])
    assert scores[0]==pub_model.fitness(result['metrics'][None])
    restored, restored_unit = pub_genome.load_solution(genome.payload(unit))
    assert restored.controller.data['simulation']['fitness_window_s']==.1
    np.testing.assert_array_equal(restored_unit,unit)
