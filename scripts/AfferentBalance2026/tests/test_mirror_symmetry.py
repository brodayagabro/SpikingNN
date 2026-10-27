"""Test constrained genetic operations, saved genomes and reflected dynamics."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from scripts.AfferentBalance2026.experiments.core.genome import (
    GenomeConfig, MIRROR_INDICES, project_mirror,
)
from scripts.AfferentBalance2026.experiments.core.evolution import reproduce
from scripts.AfferentBalance2026.experiments.simulation import simulate, reference_trace
from scripts.AfferentBalance2026.experiments.C_setpoint_balance.joint_study import (
    JointGenome, load_joint_genome, fit_table_polynomial, run_pipeline,
)
from scripts.AfferentBalance2026.experiments.D_trajectory.controller import TrajectoryGenome, load_solution

EXPERIMENTS = Path(__file__).resolve().parents[1]/'experiments'


def symmetric_config(experiment='C_setpoint_balance'):
    """Make an independent test domain without changing the active experiment."""
    data = deepcopy(GenomeConfig(EXPERIMENTS/experiment/'genome_cfg.json').data)
    data['mirror_symmetry'] = True
    initial = project_mirror([p['value'] for p in data['parameters']])
    for field, value in zip(data['parameters'], initial):
        field['value'] = float(value)
    return GenomeConfig(data)


@pytest.mark.parametrize('kind,degree,size', [('polynomial', 2, 27), ('polynomial', 3, 28), ('table', 3, 29)])
def test_mutation_roundtrip_and_command_reflection(kind, degree, size, tmp_path):
    """Tie every decoded offspring and reject asymmetric physical/command inputs."""
    cfg = symmetric_config()
    codec = JointGenome(cfg, kind, degree)
    assert cfg.size == 24 and codec.size == size
    np.testing.assert_array_equal(MIRROR_INDICES[MIRROR_INDICES], np.arange(52))
    rng = np.random.default_rng(713)
    parents = np.vstack([np.zeros(size), np.ones(size), rng.random((8, size))])
    children = reproduce(parents, rng.random(len(parents)), rng, .4, 1.)
    for unit in np.vstack([parents, children]):
        p, c = codec.decode(unit)
        np.testing.assert_array_equal(p, p[MIRROR_INDICES])
        np.testing.assert_array_equal(c[1], codec.reflected_command(c[0]))
        np.testing.assert_allclose(codec.encode(p, c), unit, atol=1e-14)
        assert np.all(p[[19, 20, 26, 27]] >= 50)
        np.testing.assert_array_equal(p[[18, 25, 46, 47]], [.37, .37, .2, .05])
    path = tmp_path/'genome.json'
    codec.save(path, children[0])
    loaded, actual = load_joint_genome(path)
    assert loaded.size == size
    np.testing.assert_array_equal(actual, children[0])
    p, c = codec.decode(children[0])
    p[1] *= 1.0001
    with pytest.raises(ValueError, match='mirror symmetry'):
        codec.encode(p, c)
    p, c = codec.decode(children[0])
    c[1, 0] += .01
    with pytest.raises(ValueError, match='Commands violate'):
        codec.encode(p, c)


def test_symmetric_d_codec_and_legacy_roundtrip(tmp_path):
    """Keep portable 60 physical fields while optimizing only 28 coordinates."""
    codec = TrajectoryGenome(symmetric_config('D_trajectory'))
    assert codec.size == 28 and len(codec.names) == 60
    u = np.random.default_rng(9).random(codec.size)
    p, c = codec.decode(u)
    path = tmp_path/'d.json'
    codec.save(path, u)
    cfg, restored_p, restored_c, tau, _ = load_solution(path)
    np.testing.assert_array_equal(p, restored_p)
    np.testing.assert_array_equal(c, restored_c)
    np.testing.assert_allclose(TrajectoryGenome(cfg).encode(p, c), u, atol=1e-14)
    assert tau == 0
    legacy = TrajectoryGenome(EXPERIMENTS/'D_trajectory/genome_cfg.json')
    assert legacy.size == 56
    legacy.save(path, legacy.encode(p, c))
    assert not load_solution(path)[0].mirror_symmetry


def test_reflected_physical_dynamics_and_original_library():
    """Check all 26 channels under reflection, including nonzero initial velocity."""
    cfg = symmetric_config()
    p = cfg.initial_physical.copy()
    # Keep this reference fixture moderate; arbitrary high-gain genomes may fall.
    p[[16, 23]] = .01
    p[30:42] *= .05
    case = np.array([.2, -.6, .4, -1., 0.])
    reflected = case.copy()
    reflected[[0, 1, 2, 4]] *= -1
    currents = np.array([40., 65.])
    _, left = simulate(p, case, cfg.mechanics, currents=currents, duration_s=.1)
    _, right = simulate(p, reflected, cfg.mechanics, currents=currents[::-1].copy(), duration_s=.1)
    columns = [0, 1, 2, 4, 3, 6, 5, 8, 7, 10, 9, 14, 15, 16, 11, 12, 13,
               18, 17, 20, 19, 24, 23, 22, 21, 25]
    expected = left[:, columns].copy()
    expected[:, [1, 2, 25]] *= -1
    assert len(left) == 1001
    np.testing.assert_allclose(right, expected, rtol=1e-9, atol=1e-8)
    library = reference_trace(p, case, cfg.mechanics, currents=currents, duration_s=.1)
    np.testing.assert_allclose(left, library, rtol=1e-9, atol=1e-8)


def test_bounds_and_fixed_pairs_must_agree():
    """Reject ambiguous reflected domains rather than silently changing constraints."""
    data = symmetric_config().data
    data['parameters'][23]['maximum'] = .5
    with pytest.raises(ValueError, match='identical maximum'):
        GenomeConfig(data)


@pytest.mark.parametrize('method', ['joint', 'staged'])
def test_symmetric_training_preserves_constraints_and_artifacts(method, tmp_path):
    """Exercise real training, table-to-polynomial fitting and complete exports."""
    data = symmetric_config().data
    data['simulation'].update(duration_s=.002, training_count=4, validation_count=4, test_count=4)
    data['setpoint'].update(checkpoints=[1, 2], save_figures=False)
    cfg = GenomeConfig(data)
    out = tmp_path/method
    run_pipeline(cfg, out, method=method, population=3, generations=4, workers=1)
    codec, unit = load_joint_genome(out/'best_genome.json')
    assert codec.size == 28
    p, c = codec.decode(unit)
    np.testing.assert_array_equal(p, p[MIRROR_INDICES])
    np.testing.assert_array_equal(c[1], c[0]*[1, -1, 1, -1])
    for path in out.glob('*/generations/*.npz'):
        with np.load(path) as arrays:
            p = arrays['physical'][:, :52]
            np.testing.assert_array_equal(p, p[:, MIRROR_INDICES])


def test_table_least_squares_preserves_parity():
    """Ensure a fitted symmetric table can initialize the polynomial codec."""
    codec = JointGenome(symmetric_config(), degree=3)
    first = np.array([-300., 40., 160., 270., 350.])
    c, _ = fit_table_polynomial(codec.targets, np.array([first, first[::-1]]), 3, codec.bound)
    unit = codec.encode(codec.controller.initial_physical, c)
    _, actual = codec.decode(unit)
    np.testing.assert_allclose(actual, c, rtol=0, atol=1e-10)


def test_explicit_projection_preserves_source_and_exports_bounds(tmp_path):
    """Build a symmetric seed without silently editing the original candidate."""
    from scripts.AfferentBalance2026.tools.prepare_mirror_genome import prepare
    import json

    original = JointGenome(EXPERIMENTS/'C_setpoint_balance/genome_cfg.json', degree=2)
    p = original.controller.initial_physical.copy()
    p[[16, 23]] = [.1, .3]
    c = np.array([[20., 30., 40.], [80., 90., 60.]])
    source = tmp_path/'source.json'
    original.save(source, original.encode(p, c))
    before = source.read_bytes()
    output = prepare(source, EXPERIMENTS/'C_setpoint_balance/genome_cfg.json', tmp_path/'projected', 2)
    codec, unit = load_joint_genome(output)
    actual, command = codec.decode(unit)
    np.testing.assert_allclose(actual, project_mirror(p), rtol=0, atol=1e-12)
    first = .5*(c[0]+c[1]*[1, -1, 1])
    np.testing.assert_allclose(command, [first, first*[1, -1, 1]], atol=1e-12)
    assert source.read_bytes() == before
    report = json.loads((output.parent/'projection.json').read_text())
    assert report['independent_coordinates'] == 27
    assert all(bounds[0] == 50 for bounds in report['muscle_time_bounds_ms'].values())


@pytest.mark.parametrize('side', ['flexor', 'extensor'])
def test_parameter_scan_moves_both_mirrored_members(side):
    """Keep notebook parameter scans inside the constrained physical domain."""
    from scripts.AfferentBalance2026.experiments.visualization import parameter_scan
    cfg = symmetric_config()
    original = cfg.initial_physical.copy()
    results = parameter_scan(cfg, original, f'muscles.{side}.tau_c', [50., 71., 100.], duration_s=.002)
    for result, value in zip(results, [50., 71., 100.]):
        p = original.copy()
        p[[19, 26]] = value
        expected, trace = simulate(p, result['case'], cfg.mechanics, duration_s=.002, stride=10)
        np.testing.assert_array_equal(result['trace'], trace)
        np.testing.assert_array_equal(result['metrics'], expected)
    np.testing.assert_array_equal(original, cfg.initial_physical)
