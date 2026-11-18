"""Check continuous angle commands against the batch solver and library objects."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.AfferentBalance2026.experiments.D_trajectory.controller import (
    AngleController, TrajectoryGenome, load_solution)
from scripts.AfferentBalance2026.experiments.D_trajectory.trajectory import (
    DT_S, command_currents, measure_tracking_trace, replay, sine_case, sine_reference,
    step_count, stream, trajectory_banks)
from scripts.AfferentBalance2026.experiments.D_trajectory.run_experiment import Evaluator, train_mode
from scripts.AfferentBalance2026.experiments.simulation import initial_state, reference_trace, simulate

CONFIG = Path(__file__).resolve().parents[1]/'experiments/D_trajectory/genome_cfg.json'


@pytest.fixture
def solution(tmp_path):
    """Create a portable, deterministic combined genome without stored run artifacts."""
    codec = TrajectoryGenome(CONFIG)
    p = codec.controller.initial_physical.copy()
    # Moderate calcium drive keeps this untrained fixture alive for block comparisons.
    p[[16, 23]] = .01
    c = np.array([[30., 20., -5., 1.], [50., -10., 3., -1.]])
    unit = codec.encode(p, c)
    path = tmp_path/'genome.json'
    codec.save(path, unit)
    return codec, unit, path


def test_streaming_blocks_preserve_full_hidden_state(solution):
    """Block boundaries and target assignments must not change a constant-command solution."""
    codec, unit, path = solution
    p, c = codec.decode(unit)
    x = AngleController(path, initial_angle_deg=-20, initial_velocity_rad_s=.2, target_angle_deg=15)
    for _ in range(15):
        result = x.advance(15, .01)
    current = command_currents(c, np.deg2rad(15))
    _, trace = simulate(p, np.array([np.deg2rad(15), np.deg2rad(-20), .2, -1, 0]),
                        codec.controller.mechanics, currents=current, duration_s=.15)
    np.testing.assert_array_equal(result['membrane_mV'], trace[-1, 5:7])
    np.testing.assert_array_equal(result['synapses'].ravel(), trace[-1, 21:25])
    np.testing.assert_array_equal(result['afferents'], trace[-1, 11:17])
    assert result['angle_deg'] == np.rad2deg(trace[-1, 1])
    assert result['time_s'] == .15
    before = x.snapshot()
    x.target_angle_deg = -15
    after = x.snapshot()
    for key in ('time_s', 'membrane_mV', 'force_N', 'afferents', 'calcium', 'synapses', 'angle_deg'):
        np.testing.assert_array_equal(before[key], after[key])
    updated = x.step()
    np.testing.assert_array_equal(updated['input_currents'], command_currents(c, np.deg2rad(-15)))
    assert not np.array_equal(updated['input_currents'], before['input_currents'])
    updated['membrane_mV'][:] = 999
    assert np.all(x.snapshot()['membrane_mV'] != 999)


def test_variable_commands_match_original_library_and_filter_units(solution):
    """Changing angle commands and a 0.1 s filter reproduce original library physics."""
    codec, unit, path = solution
    p, c = codec.decode(unit)
    steps = 1500
    reference = np.zeros((steps+1, 2))
    reference[:, 0] = np.deg2rad(np.linspace(-20, 25, steps+1))
    state = initial_state(p, np.deg2rad(-30), .2)
    initial_current = command_currents(c, reference[0, 0])
    result = stream(p, c, codec.controller.mechanics, state, initial_current.copy(), reference, .1, 0, 1)
    _, trace, commands, _, completed, failure = result
    library = reference_trace(p, np.array([0., np.deg2rad(-30), .2, -1, 0]),
                              codec.controller.mechanics, currents=commands[1:, :2], duration_s=completed*DT_S)
    # An untrained controller may fall; compare the complete executed prefix.
    assert failure in (0, 1) and 0 < completed <= steps
    assert len(trace) == completed+1
    np.testing.assert_allclose(trace, library, rtol=1e-8, atol=1e-8)
    expected = initial_current.copy()
    alpha = -np.expm1(-DT_S/.1)
    for row in range(completed):
        expected += alpha*(command_currents(c, reference[row, 0])-expected)
    np.testing.assert_array_equal(expected, commands[-1, :2])


def test_reference_derivative_and_bank_isolation():
    """References stay inside +/-45 degrees and analytic speed matches angle derivative."""
    spec = sine_case(40, .2, offset=5, phase=.8, hold=.05, ramp=.1)
    reference = sine_reference(spec)
    numerical = np.gradient(reference[:, 0], DT_S)
    np.testing.assert_allclose(reference[2:-2, 1], numerical[2:-2], rtol=1e-3, atol=4e-4)
    assert np.max(abs(reference[:, 0])) <= np.pi/4+1e-14
    assert np.all(reference[:500, 1] == 0)
    settings = json.loads(CONFIG.read_text())['trajectory']
    banks = trajectory_banks(settings)
    assert banks == trajectory_banks(settings)
    sets = [set(json.dumps(spec, sort_keys=True) for spec in bank) for bank in banks.values()]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    for bank in banks.values():
        for spec in bank:
            assert abs(spec['offset_deg'])+spec['amplitude_deg'] <= 45
            assert spec['duration_s'] >= spec['hold_s']+spec['ramp_s']+2*spec['period_s']-1e-8


def test_metrics_independent_recomputation_and_recording_stride(solution):
    """Streaming scores agree with full trace analysis and do not depend on recording."""
    codec, unit, path = solution
    p, c = codec.decode(unit)
    spec = sine_case(10, .06, hold=.02, ramp=.04)
    result = replay(p, c, codec.controller.mechanics, spec)
    measured = measure_tracking_trace(result['trace'], result['reference'], spec)
    np.testing.assert_allclose(measured, result['metrics'], atol=1e-10, rtol=1e-10)
    for stride in (0, 7, 100):
        sparse = replay(p, c, codec.controller.mechanics, spec, stride=stride)
        np.testing.assert_array_equal(sparse['metrics'], result['metrics'])
        if stride:
            assert sparse['trace'][-1, 0] == result['trace'][-1, 0]


@pytest.mark.parametrize('duration', [0, -.1, .00015, float('nan')])
def test_invalid_duration_rejected(duration):
    """Reject invalid times instead of silently rounding simulation steps."""
    with pytest.raises(ValueError):
        step_count(duration)


def test_genome_roundtrip_bounds_and_input_validation(solution, tmp_path):
    """Portable genes retain all fixed physiology and reject invalid angle inputs."""
    codec, unit, path = solution
    cfg, p, c, tau, data = load_solution(path)
    assert len(data['physical']) == 60 and len(unit) == 56
    np.testing.assert_array_equal(p[[18, 25, 46, 47]], [.37, .37, .2, .05])
    for boundary in (np.zeros(codec.size), np.ones(codec.size)):
        physical, _ = codec.decode(boundary)
        assert np.all(physical[[19, 20, 26, 27]] >= 50.)
    x = AngleController(path)
    for bad in (-46, 46, float('inf'), float('nan')):
        with pytest.raises(ValueError):
            x.advance(bad)
    assert x.snapshot()['time_s'] == 0
    x.reset(89.9, 100, 0)
    result = x.advance(0, .01)
    assert result['failure'] == 'fall'
    with pytest.raises(RuntimeError):
        x.step()
    x.reset()
    assert x.step()['failure'] == 'none'


def test_training_checkpoints_and_parallel_determinism(solution, tmp_path):
    """Execute both search modes, check validation selection and fixed physical genes."""
    codec, unit, path = solution
    banks = {name:[sine_case(10, .01, initial_angle=angle, hold=.002, ramp=.002),
                    sine_case(15, .012, initial_angle=-angle, hold=.002, ramp=.002)]
             for name, angle in (('training', 2), ('validation', 3), ('test', 4))}
    with Evaluator(codec, banks, 1) as evaluator:
        serial = evaluator.evaluate([unit, unit], 'training', .1)
        settings = dict(population=3, generations=2, seed=44, validation_every=1,
                        sigma_start=.02, sigma_end=.004, mutation_probability=.1)
        for mode in ('command_only', 'joint'):
            folder = tmp_path/mode
            winner = train_mode(folder, evaluator, unit, .1, mode, settings)
            data = np.load(folder/'generation_002.npz')
            assert data['population'].shape == (3, 56)
            np.testing.assert_array_equal(data['physical'][:, [18, 25, 46, 47]],
                                          np.tile([.37, .37, .2, .05], (3, 1)))
            if mode == 'command_only':
                np.testing.assert_array_equal(data['population'][:, :48], np.tile(unit[:48], (3, 1)))
            initial = np.load(folder/'validation_000.npz')['scores'][0]
            assert evaluator.evaluate([winner], 'validation', .1)[0][0] >= initial
            _, p, c, tau, metadata = load_solution(folder/'best_genome.json')
            np.testing.assert_array_equal(np.r_[p, c.ravel()], np.r_[codec.decode(winner)[0], codec.decode(winner)[1].ravel()])
    with Evaluator(codec, banks, 2) as evaluator:
        parallel = evaluator.evaluate([unit, unit], 'training', .1)
    for a, b in zip(serial, parallel):
        np.testing.assert_array_equal(a, b)
