"""Validate physical equivalence, fixed genes and complete A/B/C run artifacts."""
from copy import deepcopy
import json
from pathlib import Path
import re
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

from SpikingNN.core.Izh_net import Afferents
from scripts.AfferentBalance2026.experiments.core.genome import (
    GenomeConfig, NAMES, SCHEMA, import_legacy, load_genome,
)
from scripts.AfferentBalance2026.experiments.core.evolution import reproduce
from scripts.AfferentBalance2026.experiments.core.run import PopulationEvaluator, train, sweep
from scripts.AfferentBalance2026.experiments.simulation import (
    PARAMETERS, TRACE_COLUMNS, afferent_values, balance_banks, case_grid,
    measure_trace, reference_trace, simulate,
)
from scripts.AfferentBalance2026.experiments.C_setpoint_balance.run_experiment import run as replay
from scripts.AfferentBalance2026.experiments.C_setpoint_balance.polynomial_study import polynomial_currents
from scripts.AfferentBalance2026.experiments.visualization import (
    load_solution, replay_grid, plot_phase, plot_angles, plot_states,
)

EXPERIMENTS = Path(__file__).resolve().parents[1] / 'experiments'


def config(name='A_balance', tiny=False):
    """Load real bounds with optional small banks for fast integration checks."""
    cfg = GenomeConfig(EXPERIMENTS / name / 'genome_cfg.json')
    if not tiny:
        return cfg
    data = deepcopy(cfg.data)
    data['simulation'].update(training_count=4, validation_count=4, test_count=4,
                              replay_angles_deg=[-45, 45], save_top=1, duration_s=.01)
    data['ga'].update(population=3, generations=2, workers=1)
    return GenomeConfig(data)


@pytest.mark.parametrize('name,cap', [('A_balance', 1.), ('B_balance', .02), ('C_setpoint_balance', 1.)])
def test_schema_bounds_fixed_geometry_and_english_descriptions(name, cap, tmp_path):
    """Keep a1/a2 exact through reproduction and export all English field metadata."""
    cfg = config(name)
    assert len(NAMES) == 52 and cfg.size == 50
    for field in cfg.data['parameters']:
        for key in ('description', 'unit', 'system_parameter'):
            assert field[key] and not re.search('[А-Яа-яЁё]', field[key])
    rng = np.random.default_rng(108)
    population = reproduce(rng.random((100, 50)), rng.random(100), rng, .8, 1.)
    physical = np.array([cfg.decode(row) for row in population])
    np.testing.assert_array_equal(physical[:, 46:48], np.tile([.2, .05], (100, 1)))
    assert np.all(physical[:, [16, 23]] <= cap)
    assert np.all(physical[:, [16, 23]] >= .0001)
    assert np.all(cfg.decode(np.ones(50))[[16, 23]] <= cap)
    invalid = cfg.initial_physical.copy(); invalid[46] += .001
    with pytest.raises(ValueError, match='Fixed parameters'):
        cfg.encode(invalid)
    path = tmp_path / 'genome.json'
    cfg.save(path, physical[0])
    saved_cfg, saved = load_genome(path)
    assert saved_cfg.signature == cfg.signature
    np.testing.assert_array_equal(saved, physical[0])
    assert json.loads(path.read_text())['configuration']['parameters'][46]['mutable'] is False


def test_legacy_conversion_is_explicit_and_never_clips(tmp_path):
    """Reject old slots, unknown order, nonzero old currents and invalid B weights."""
    cfg = config()
    names = list(NAMES); names[46:48] = ['input.bias_nA.0', 'input.bias_nA.1']
    values = cfg.initial_physical.copy(); values[46:48] = 0
    old = dict(version=1, genes=[dict(name=name, value=float(value)) for name, value in zip(names, values)])
    path = tmp_path/'legacy.json'; path.write_text(json.dumps(old))
    with pytest.raises(ValueError, match='explicit import_legacy'):
        load_genome(path)
    np.testing.assert_array_equal(import_legacy(path, cfg), cfg.initial_physical)
    with pytest.raises(ValueError, match='bounds'):
        import_legacy(path, config('B_balance'))
    old['genes'][46]['value'] = 2; path.write_text(json.dumps(old))
    with pytest.raises(ValueError, match='currents'):
        import_legacy(path, cfg)
    old['genes'][0]['name'] = 'unknown'; path.write_text(json.dumps(old))
    with pytest.raises(ValueError, match='version-1'):
        import_legacy(path, cfg)


@pytest.mark.parametrize('velocity,length,force,activation', [(-.5,.1,0.,0.), (.7,.25,8.,.8), (0.,.2,3.,.1)])
def test_full_afferents_match_library(velocity, length, force, activation):
    """Exercise rectification, fractional velocity exponent and activation feedback."""
    receptor = Afferents(); receptor.L_th = .2; receptor.F_th = 3.
    expected = [receptor.Ia(velocity, length, activation), receptor.II(length, activation), receptor.Ib(force)]
    actual = afferent_values(velocity, length, force, activation, .2, 3., PARAMETERS)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize('theta,speed,currents,geometry', [(-45,.7,[0.,0.],[.2,.05]), (40,-.9,[120.,75.],[.23,.06])])
def test_numba_all_state_channels_match_library(theta, speed, currents, geometry):
    """Check actual limb, muscle and neuron trajectories under nondefault inputs."""
    cfg = config(); physical = cfg.initial_physical.copy(); physical[46:48] = geometry
    mechanics = np.array([.35, .32, .0015, 9.7])
    case = case_grid([theta], [speed], target_deg=22.5)[0]
    metrics, trace = simulate(physical, case, mechanics, currents=currents, duration_s=.15)
    expected = reference_trace(physical, case, mechanics, currents=currents, duration_s=.15)
    assert trace.shape == (1501, len(TRACE_COLUMNS))
    np.testing.assert_allclose(trace, expected, rtol=1e-9, atol=1e-8)
    # Targets change scoring, never control: the same inputs give identical states.
    changed = case.copy(); changed[0] = 0.
    other, trajectory = simulate(physical, changed, mechanics, currents=currents, duration_s=.15)
    np.testing.assert_array_equal(trace, trajectory)
    assert metrics[0] != other[0]
    np.testing.assert_allclose(metrics, measure_trace(trace[1:], case[0], .15, initial_theta=case[1]), rtol=1e-9, atol=1e-9)


def test_full_angle_banks_identical_and_disjoint():
    """Use the same broad conditions across weight caps without leaking test cases."""
    a = balance_banks(config().data['simulation'])
    b = balance_banks(config('B_balance').data['simulation'])
    sets = []
    for name, cases in a.items():
        np.testing.assert_array_equal(cases, b[name])
        degrees = np.abs(np.rad2deg(cases[:, 1]))
        assert np.all(degrees <= 50+1e-12)
        assert np.any(degrees < 5) and np.any(degrees > 40)
        assert np.max(abs(cases[:, 2])) <= 1
        assert np.all(cases[:, 0] == 0) and np.all(cases[:, 3] == -1)
        sets.append(set(map(tuple, cases)))
    assert all(not sets[i] & sets[j] for i in range(3) for j in range(i+1, 3))
    np.testing.assert_allclose(np.rad2deg(a['training'][:3, 1]), [-50, 0, 50])
    np.testing.assert_array_equal(a['training'][:3, 2], [0, 0, 0])


def test_process_pool_matches_serial():
    """Numerical evaluations are independent of process scheduling and worker count."""
    cfg = config(); cases = case_grid([-30, 40], [-.5, .5])
    population = np.array([cfg.initial_physical, cfg.decode(np.full(50, .5))])
    with PopulationEvaluator(cfg.mechanics, workers=1) as serial:
        expected = serial.evaluate(population, cases, .01)
    with PopulationEvaluator(cfg.mechanics, workers=2) as parallel:
        actual = parallel.evaluate(population, cases, .01)
    for lhs, rhs in zip(expected, actual):
        np.testing.assert_array_equal(lhs, rhs)


@pytest.mark.parametrize('name', ['A_balance', 'B_balance'])
def test_training_retains_every_population_and_best_dynamics(name, tmp_path):
    """Run two real short generations and validate provenance, ranking and replays."""
    cfg = config(name, tiny=True); output = tmp_path/name
    result = train(cfg, output)
    assert result['generation'] == 2 and result['free_parameters'] == 50
    assert json.loads((output/'run.json').read_text())['status'] == 'completed'
    generations = list((output/'generations').glob('*.npz')); assert len(generations) == 2
    with np.load(output/'final_population.npz') as saved:
        assert saved['physical'].shape == (3, 52)
        winner = int(saved['selected_index'])
        assert winner == int(np.argmax(saved['validation_scores']))
        _, physical = load_genome(output/'best_genome.json')
        np.testing.assert_array_equal(physical, saved['physical'][winner])
    for path in generations:
        with np.load(path) as saved:
            assert str(saved['schema']) == SCHEMA
            np.testing.assert_array_equal(saved['physical'][:,46:48], np.tile([.2,.05],(3,1)))
    with np.load(output/'best/rank_001/case_000.npz') as saved:
        assert saved['trace'].shape == (101,26)
        np.testing.assert_allclose(np.diff(saved['trace'][:,0]), .0001, atol=1e-15)
        np.testing.assert_array_equal(saved['currents'], [0.,0.])
    with zipfile.ZipFile(output/'source_snapshot.zip') as archive:
        assert 'genome_cfg.json' in archive.namelist()
        assert any(name.endswith('Izh_net.py') for name in archive.namelist())
    loaded_cfg, loaded, metadata = load_solution(generations[0])
    assert loaded_cfg.signature == cfg.signature and loaded.shape == (52,)
    with pytest.raises(FileExistsError):
        train(cfg, output)


def test_sweep_checkpoints_do_not_change_training(tmp_path):
    """Validate continuous checkpoint semantics, normalized RMSE and heatmap output."""
    cfg = config(tiny=True); out = tmp_path/'sweep'
    rows = sweep(cfg, out, populations=[3], generations=[1,2], probabilities=[.3], seeds=[42])
    assert len(rows) == 2
    direct = tmp_path/'direct'; train(cfg, direct)
    with np.load(out/rows[-1]['directory']/'final_population.npz') as lhs, np.load(direct/'final_population.npz') as rhs:
        np.testing.assert_array_equal(lhs['physical'], rhs['physical'])
    with np.load(out/'genome_rmse.npz') as saved:
        assert saved['rmse'].shape == (2,2)
        np.testing.assert_array_equal(np.diag(saved['rmse']), [0.,0.])
        assert saved['normalized'].shape == (2,50)
    assert list(out.glob('*.png'))


def test_c_replay_uses_external_currents_and_never_trains(tmp_path):
    """Store a fixed genome and actual grid trajectories with explicit polynomial inputs."""
    cfg = config('C_setpoint_balance')
    law = np.array([[2,3,4], [5,-2,1.]])
    np.testing.assert_allclose(polynomial_currents(law, np.deg2rad(45)), [9,4])
    path = tmp_path/'law.json'; path.write_text(json.dumps({'coefficients': law.tolist()}))
    out = tmp_path/'C'
    result = replay(cfg, out, targets=[-22.5,22.5], angles=[-50,50], speeds=[0], current_law=path, duration_s=.01)
    assert result['training_performed'] is False and result['cases'] == 4
    assert not (out/'generations').exists()
    _, physical = load_genome(out/'best_genome.json')
    np.testing.assert_array_equal(physical, cfg.initial_physical)
    with np.load(out/'traces/case_000.npz') as saved:
        np.testing.assert_array_equal(saved['currents'], polynomial_currents(law, np.deg2rad(-22.5)))
        assert saved['trace'].shape == (101,26)
    with pytest.raises(ValueError, match='A_balance and B_balance'):
        train(cfg, tmp_path/'not_allowed')


def test_figures_draw_real_recorded_state_channels():
    """Render all requested state groups from real short trajectories."""
    cfg = config()
    records = replay_grid(cfg, cfg.initial_physical, [-30,30], [0], duration_s=.01)
    figures = [plot_phase(records), plot_angles(records), *plot_states(records[0]).values()]
    for figure in figures:
        # Plot helpers return figures; rendering catches invalid labels and data paths.
        figure.canvas.draw()
    plt.close('all')


def test_early_fall_metrics_and_terminal_state_are_retained():
    """Check failure scoring against recorded dynamics instead of extrapolating success."""
    cfg = config(); physical = cfg.initial_physical.copy()
    physical[8:12] = 0.; physical[30:42] = 0.
    case = case_grid([50], [1.])[0]
    metrics, trace = simulate(physical, case, cfg.mechanics, duration_s=2.)
    assert len(trace) < 20001 and abs(trace[-1, 1]) > np.pi/2
    assert metrics[5] == 0 and metrics[7] == np.pi/2
    np.testing.assert_allclose(metrics, measure_trace(trace[1:], 0., 2., initial_theta=case[1]), atol=1e-12)


def test_optional_current_fit_keeps_the_controller_fixed(tmp_path):
    """Exercise the separate fitting API briefly without starting a scientific study."""
    from scripts.AfferentBalance2026.experiments.C_setpoint_balance.polynomial_study import fit_current_law
    cfg = config('C_setpoint_balance'); data = deepcopy(cfg.data)
    data['simulation']['duration_s'] = .01; cfg = GenomeConfig(data)
    banks = {name: case_grid([angle], [0], target_deg=22.5)
             for name, angle in [('training', -30), ('validation', 0), ('test', 30)]}
    out = tmp_path/'optional_fit'
    fit_current_law(cfg, cfg.initial_physical, banks, out, population=3, generations=1, degree=2)
    _, saved = load_genome(out/'fixed_genome.json')
    np.testing.assert_array_equal(saved, cfg.initial_physical)
    assert json.loads((out/'current_law.json').read_text())['selection'] == 'validation'
    with np.load(out/'final_coefficients.npz') as payload:
        assert payload['coefficients'].shape == (3,2,3)


def test_initial_condition_ranges_are_configurable():
    """Accept asymmetric rectangles without dropping small angles or imposing parity."""
    data = deepcopy(config().data['simulation'])
    data['initial_conditions'] = {'angle_range_deg': [-12., 35.], 'speed_range_rad_s': [-.4, .8]}
    data['training_count'] = 7
    for cases in balance_banks(data).values():
        angles = np.rad2deg(cases[:, 1])
        assert np.all((angles >= -12.-1e-12) & (angles <= 35.+1e-12))
        assert np.all((cases[:, 2] >= -.4) & (cases[:, 2] <= .8))
    data['initial_conditions']['angle_range_deg'] = [10, -10]
    with pytest.raises(ValueError, match='ranges'):
        balance_banks(data)


@pytest.mark.parametrize('seed', [17, 39, 82])
def test_mutated_controllers_match_library(seed):
    """Verify neuron, projection, receptor and muscle changes beyond default parameters."""
    cfg = config()
    physical = cfg.decode(np.random.default_rng(seed).uniform(.1, .9, cfg.size))
    case = case_grid([-20], [.3])[0]
    _, trace = simulate(physical, case, cfg.mechanics, currents=[40,60], duration_s=.2)
    expected = reference_trace(physical, case, cfg.mechanics, currents=[40,60], duration_s=.2)
    np.testing.assert_allclose(trace, expected, rtol=1e-8, atol=1e-8)
