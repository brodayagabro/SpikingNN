"""Independent model parity, full nonlinear receptors, and numerical contracts."""
import json
from dataclasses import asdict

import numpy as np
import pytest
from SpikingNN.core.Izh_net import Afferents, OneDOFLimb, Afferented_Limb
from scripts.SteppingReflexGA.experiments.core import genome
from scripts.SteppingReflexGA.experiments.core.afferents import afferent_response, PARAMETERS, PARAMETER_NAMES
from scripts.SteppingReflexGA.experiments.core.config import DT_MS, MECHANICS
from scripts.SteppingReflexGA.experiments.core.dynamics import rollout, reference_trace, measure_trace
from scripts.SteppingReflexGA.experiments.core.commands import input_currents, commanded_parameters
from scripts.SteppingReflexGA.experiments.core.fitness import fitness


@pytest.mark.parametrize("velocity", [-.3, 0., 1e-8, .2, 2.])
@pytest.mark.parametrize("length,force,activation", [(.1, 0., 0.), (.2, 3.38, .5), (.3, 8., 1.)])
def test_all_afferent_nonlinearities_match_library(velocity, length, force, activation):
    receptor = Afferents()
    receptor.L_th = .2
    # Nondefault constants ensure no terms can accidentally disappear.
    receptor.const_I, receptor.const_II, receptor.k_f = .7, .3, 1.4
    parameters = tuple(float(getattr(receptor, name)) for name in PARAMETER_NAMES)
    observed = afferent_response(velocity, length, force, activation, .2, receptor.F_th, parameters)
    expected = [receptor.Ia(velocity, length, activation), receptor.II(length, activation), receptor.Ib(force)]
    np.testing.assert_allclose(observed, expected, atol=1e-14, rtol=1e-14)


def controller():
    p = genome.decode(genome.initial_genome())
    p[46:48] = [5., 7.]
    p[30:42] *= .15
    p[42:44] *= [.99, 1.01]
    p[44:46] = [.1, .2]
    p[48:52] = [1., .2, .1, 1.2]
    return p


@pytest.mark.parametrize("mode", [0, 1, 2, 3])
@pytest.mark.parametrize("theta", [-np.pi/4, .03, np.pi/4])
def test_full_trajectory_matches_original_classes(mode, theta):
    p = controller()
    case = np.array([theta, .1, .05, -.2])
    metrics, actual = rollout(p, case, duration_s=.12, mode=mode, stride=1)
    expected = reference_trace(p, case, duration_s=.12, mode=mode)
    assert actual.shape == expected.shape == (1200, 17)
    np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=1e-9)
    np.testing.assert_allclose(metrics, measure_trace(expected, duration_s=.12, initial_theta=theta), atol=1e-10)


def test_receptor_activation_is_muscle_x_in_library_composite():
    system = Afferented_Limb()
    system.Flexor.x, system.Extensor.x = .2, .8
    system.calc_afferents()
    assert type(system.Afferents) is Afferents
    # At upright and zero speed/force only the activation term remains.
    system.Limb.q = -np.pi/2
    system.Limb.w = 0.
    system.calc_afferents()
    np.testing.assert_allclose(system.output, [.012, .012, 0., .048, .048, 0.], atol=1e-14)


def test_full_model_retains_activation_feedback_without_stretch():
    low = afferent_response(-.1, .1, 0., 0., .2, 1.)
    high = afferent_response(-.1, .1, 0., 1., .2, 1.)
    np.testing.assert_allclose(high-low, [.06, .06, 0.])
    assert PARAMETERS[0] == Afferents.p_v == .6


@pytest.mark.parametrize("step", [.5, 0., -.1, float("nan")])
def test_nonstandard_step_is_rejected(step):
    with pytest.raises(ValueError):
        rollout(controller(), np.zeros(4), duration_s=.01, dt_ms=step)
    with pytest.raises(ValueError):
        reference_trace(controller(), np.zeros(4), duration_s=.01, dt_ms=step)


def test_stride_keeps_last_sample_for_nondivisible_step_count():
    p, case = controller(), np.array([.05, 0., -1., 0.])
    m, full = rollout(p, case, duration_s=.0008, stride=1)
    n, sampled = rollout(p, case, duration_s=.0008, stride=3)
    np.testing.assert_array_equal(sampled, full[[0, 3, 6, 7]])
    np.testing.assert_array_equal(m, n)
    np.testing.assert_allclose(np.diff(full[:, 0]), .0001)


def test_target_does_not_enter_neural_or_mechanical_dynamics():
    p, case = controller(), np.array([.1, .05, -1., 0.])
    a = rollout(p, case, target_rad=-.2, duration_s=.1, stride=1)[1]
    b = rollout(p, case, target_rad=.2, duration_s=.1, stride=1)[1]
    np.testing.assert_array_equal(a, b)


def test_upright_is_unstable_and_pd_diagnostic_recovers():
    p, case = controller(), np.array([np.pi/4, 0., -1., 0.])
    passive, _ = rollout(p, case, mode=2)
    active, trace = rollout(p, case, mode=3, stride=1)
    assert passive[1] < 1. and passive[5] == 0
    assert active[1] == 10. and active[5] == 1
    np.testing.assert_allclose(active, measure_trace(trace, initial_theta=case[0]), atol=1e-10)


def test_survival_or_late_arrival_is_not_success():
    trace = np.zeros((100000, 17))
    trace[:, 0] = np.arange(1, 100001)*DT_MS/1000
    trace[:, 1] = .2
    assert measure_trace(trace)[5] == 0
    trace[40000:, 1] = 0.
    assert measure_trace(trace)[5] == 0
    trace[20000:, 1] = 0.
    assert measure_trace(trace)[5] == 1
    trace[-1, 2] = .4
    assert measure_trace(trace)[5] == 0


def test_same_fitness_penalizes_brittle_controllers():
    robust = np.zeros((12, 9)); robust[:, 0] = .85
    brittle = np.zeros((12, 9)); brittle[:, 0] = 1.; brittle[0, 0] = .1
    assert fitness(robust) > fitness(brittle)


@pytest.mark.parametrize("theta", [-.7, -.1, 0., .1, .7])
def test_command_uses_full_static_afferents_and_preserves_controller(theta):
    p = controller()
    original = p.copy()
    coeff = np.arange(6.).reshape(2, 3)
    limb = OneDOFLimb(a1=.2, a2=.05)
    delta = []
    for i in range(2):
        receptor = Afferents()
        receptor.L_th, receptor.F_th = p[42+i], p[44+i]
        q = theta-np.pi/2 if i == 0 else np.pi-(theta-np.pi/2)
        length = limb.L(q)
        l0 = np.hypot(.2, .05)
        delta.extend([receptor.Ia(0., l0, 0.)-receptor.Ia(0., length, 0.),
                      receptor.II(l0, 0.)-receptor.II(length, 0.), 0.])
    x = theta/(np.pi/4)
    expected = p[46:48] + p[30:42].reshape(2, 6) @ delta + coeff @ [x, x*x, x*x*x]
    np.testing.assert_allclose(input_currents(p, coeff, theta), expected, atol=1e-11)
    commanded = commanded_parameters(p, coeff, theta)
    np.testing.assert_array_equal(p, original)
    np.testing.assert_array_equal(commanded[genome.FREE_INDICES], p[genome.FREE_INDICES])


def test_export_explains_every_gene_and_separates_hyperparameters(tmp_path):
    domain = genome.ControllerDomain()
    p = domain.decode(domain.initial())
    path = tmp_path / "genome.json"
    genome.save_genome(path, genome.encode(p))
    payload = json.loads(path.read_text())
    np.testing.assert_allclose(genome.decode(genome.load_genome(path)), p)
    for i, entry in enumerate(payload["genes"]):
        assert entry["index"] == i
        assert entry["value"] == p[i]
        assert entry["description"] and entry["system_parameter"] and entry["unit"]
        assert not any(term in entry["name"] for term in ("mass", "geometry", "a1", "a2", "gravity"))
    assert payload["hyperparameters"]["mechanics"] == asdict(MECHANICS)
    assert payload["parameters"]["bias_nA"] == [0., 0.]
