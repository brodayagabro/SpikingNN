"""Full-history replay preserves dynamics and exposes receptor nonlinearities."""
import csv
import hashlib
import json
import numpy as np
import pytest
from scripts.SteppingReflexGA.experiments.core import wide_replay, wide_replay_plots
from scripts.SteppingReflexGA.experiments.core.genome import ControllerDomain
from scripts.SteppingReflexGA.experiments.core.dynamics import rollout, METRIC_COLUMNS
from scripts.SteppingReflexGA.experiments.core.run import run, write_json


def test_grid_covers_angle_velocity_and_kick_combinations():
    cases = wide_replay.wide_cases()
    assert cases.shape == (171, 5)
    assert np.all(cases[:, 0] == 0.)
    np.testing.assert_allclose(np.rad2deg(np.unique(cases[:, 1])), np.arange(-45, 46, 5))
    assert set(cases[:, 2]) == {-.5, 0., .5}
    assert set(cases[:, 4]) == {-.3, 0., .3}
    assert np.all(cases[cases[:, 4] == 0., 3] == -1.)
    assert np.all(cases[cases[:, 4] != 0., 3] == 5.)
    with pytest.raises(ValueError):
        wide_replay.wide_cases(angles=[-90., 0.])
    with pytest.raises(ValueError):
        wide_replay.wide_cases(velocities=[0., 0.])


@pytest.mark.parametrize("small", [False, True])
def test_full_recording_keeps_every_step_and_does_not_change_dynamics(small):
    domain = ControllerDomain(small)
    p = domain.decode(domain.initial())
    case = np.array([0., np.pi/4, .5, .02, -.3])
    metrics, full = wide_replay.full_replay(p, case, .05)
    original, raw = rollout(p, case[1:], duration_s=.05, mode=0, stride=1)
    assert full.shape == (501, 17)
    assert full[0, 0] == 0. and full[0, 1] == case[1]
    np.testing.assert_array_equal(full[0, 11:17], np.zeros(6))
    np.testing.assert_array_equal(full[1:], raw)
    np.testing.assert_array_equal(metrics, original)
    signals = wide_replay.nonlinear_signals(full, p)
    terms = signals["afferent_terms"]
    expected = np.stack((terms[:, :, :4].sum(axis=2), terms[:, :, 4:7].sum(axis=2),
                         terms[:, :, 7]), axis=2).reshape(-1, 6)
    np.testing.assert_allclose(expected[1:], full[1:, 11:17], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(signals["afferent_current_used"][1:],
                               full[:-1, 11:17] @ p[30:42].reshape(2, 6).T)
    assert np.all((signals["activation"] >= 0.) & (signals["activation"] <= 1.))


def test_balance_replay_rejects_input_bias_or_a_different_target():
    domain = ControllerDomain()
    p = domain.decode(domain.initial())
    with pytest.raises(ValueError, match="upright target"):
        wide_replay.full_replay(p, [.1, .2, 0., -1., 0.], .01)
    p[46] = 1.
    with pytest.raises(ValueError, match="zero input currents"):
        wide_replay.full_replay(p, [0., .2, 0., -1., 0.], .01)


def test_ten_second_grid_does_not_reject_binary_time_roundoff():
    domain = ControllerDomain(True)
    p = domain.decode(domain.initial())
    _, trace = wide_replay.full_replay(p, [0., 0., 0., -1., 0.], 10.)
    assert trace.shape == (100001, 17)
    np.testing.assert_array_equal(trace[:, 0], np.arange(100001)*.1/1000)


def test_fall_keeps_last_step_instead_of_padding_the_trace():
    domain = ControllerDomain(True)
    p = domain.decode(domain.initial())
    case = np.array([0., np.deg2rad(89.), 2., -1., 0.])
    metrics, trace = wide_replay.full_replay(p, case, .1)
    assert metrics[5] == 0. and metrics[1] < .1
    assert abs(trace[-1, 1]) > np.pi/2
    assert trace[-1, 0] == metrics[1]
    assert len(trace) == round(metrics[1]/.0001)+1


def test_saved_pipeline_preserves_genome_and_trace_metrics(tmp_path, monkeypatch):
    sweep = tmp_path/"sweep"
    source = sweep/"control"
    result = run("small_nm_weights", source, population_size=3, generations=1,
                 duration_s=.01, plot=False, verbose=False)
    write_json(sweep/"analysis.json", {"best_by_validation": dict(result, directory="control")})
    monkeypatch.setattr(wide_replay_plots, "analyze", lambda root: None)
    output = tmp_path/"wide"
    summary = wide_replay.run_replay(sweep, output, workers=1, angles=[-45., 45.],
                                    velocities=[0.], kicks=[0.], duration_s=.03)
    assert summary["cases"] == 2 and summary["total_samples"] == 602
    assert (output/"best_genome.json").read_bytes() == (source/"best_genome.json").read_bytes()
    assert json.loads((output/"original_replay_check.json").read_text())["max_abs_errors"] == {"training": 0., "validation": 0.}
    with (output/"metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        path = output/row["trace_file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["trace_sha256"]
        with np.load(path, allow_pickle=False) as data:
            assert int(data["mode"]) == 0
            assert data["trace"].shape == (301, 17)
            np.testing.assert_array_equal(data["metrics"], [float(row[k]) for k in METRIC_COLUMNS])
            assert np.all(data["physical_controller"][[46, 47]] == 0.)
