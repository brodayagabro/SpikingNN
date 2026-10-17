"""Continuation must reproduce an uninterrupted trajectory; distances use gene scales."""
from copy import deepcopy
import json
from pathlib import Path
import zipfile
import numpy as np
import pytest
from scripts.SteppingReflexGA.experiments.core.evolution import optimize, Schedule
from scripts.SteppingReflexGA.experiments.core.sweep import load_state
from scripts.SteppingReflexGA.experiments.core.sweep_analysis import pairwise_rmse, grid_values, analyze
from scripts.SteppingReflexGA.experiments.core import population as population_module
from scripts.SteppingReflexGA.experiments.core.genome import ControllerDomain
from scripts.SteppingReflexGA.experiments.core.cases import banks
from scripts.SteppingReflexGA.experiments.core.run import run, write_json, source_hashes
from scripts.SteppingReflexGA.experiments.core.sweep_verification import audit


def test_continuation_preserves_rng_population_and_annealing():
    def evaluate(population):
        scores = 1.-np.mean((population-.7)**2, axis=1)
        metrics = np.zeros((len(population), 2, 9))
        metrics[:, :, 0] = scores[:, None]
        return scores, metrics

    schedule = Schedule(probability_start=.3, probability_end=.3)
    kwargs = dict(size=9, seed=11, schedule=schedule, schedule_generations=17)
    complete = optimize(np.full(6, .5), evaluate, generations=17, **kwargs)
    first = optimize(np.full(6, .5), evaluate, generations=7, **kwargs)
    state = dict(generation=7, population=first[0], scores=first[1], history=first[3], rng_state=first[4])
    before = deepcopy(state)
    continued = optimize(np.full(6, .5), evaluate, generations=17, resume_state=state, **kwargs)
    for i in (0, 1, 2):
        np.testing.assert_array_equal(continued[i], complete[i])
    assert continued[4] == complete[4]
    assert state["rng_state"] == before["rng_state"]
    np.testing.assert_array_equal(state["population"], before["population"])
    for a, b in zip(continued[3], complete[3]):
        assert {k: v for k, v in a.items() if k != "elapsed_s"} == {k: v for k, v in b.items() if k != "elapsed_s"}
        assert a["mutation_probability"] == .3


def test_cached_evaluation_is_exact_and_case_sensitive(monkeypatch):
    p = ControllerDomain().decode(ControllerDomain().initial())
    cases = banks("equilibrium")["training"][:1].copy()
    original = population_module.evaluate_individual
    calls = []

    def counted(task):
        calls.append(1)
        return original(task)

    monkeypatch.setattr(population_module, "evaluate_individual", counted)
    with population_module.PopulationEvaluator(cache_size=3) as evaluator:
        first = evaluator.evaluate(np.array([p, p]), None, cases, .01)
        assert len(calls) == 1
        second = evaluator.evaluate(np.array([p, p]), None, cases, .01)
        assert len(calls) == 1
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)
        cases[0, 1] += .01
        evaluator.evaluate(np.array([p]), None, cases, .01)
        assert len(calls) == 2


def test_rmse_and_grid_coordinates():
    vectors = [[0., 0.], [1., 0.], [1., 1.]]
    distance = pairwise_rmse(vectors)
    np.testing.assert_allclose(distance, [[0, 2**-.5, 1], [2**-.5, 0, 2**-.5], [1, 2**-.5, 0]])
    rows = [dict(population=10, generations=30, mutation_probability=.1, value=.7),
            dict(population=100, generations=200, mutation_probability=.1, value=.9)]
    grid = grid_values(rows, [10, 100], [30, 200], .1, "value")
    assert grid[0, 0] == .7 and grid[1, 1] == .9
    assert np.isnan(grid[0, 1]) and np.isnan(grid[1, 0])


def test_population_uses_neural_control_without_target_error_feedback(monkeypatch):
    import inspect
    from scripts.SteppingReflexGA.experiments.core.dynamics import rollout
    domain = ControllerDomain(True)
    p = domain.decode(domain.initial())
    traces, modes = [], []
    default_mode = inspect.signature(rollout.py_func).parameters["mode"].default

    def observed(*args, **kwargs):
        modes.append(kwargs.get("mode", default_mode))
        metrics, trace = rollout(*args, stride=1, **kwargs)
        traces.append(trace)
        return metrics, trace

    monkeypatch.setattr(population_module, "rollout", observed)
    metrics = []
    for target in (0., .1):
        cases = np.array([[target, .08, .04, -1., 0.]])
        metrics.append(population_module.evaluate_individual((p, None, cases, .2)))
    assert modes == [0, 0]
    np.testing.assert_array_equal(traces[0], traces[1])
    assert not np.array_equal(metrics[0], metrics[1])


@pytest.mark.parametrize("experiment", ["equilibrium", "small_nm_weights"])
def test_saved_continuation_and_analysis(tmp_path, experiment):
    root = tmp_path / "sweep"
    root.mkdir()
    write_json(root / "sweep.json", dict(experiment=experiment, populations=[3], epochs=[2, 3],
               probabilities=[.3], seeds=[9], source_sha256=source_hashes()))
    repo = Path(__file__).resolve().parents[3]
    with zipfile.ZipFile(root / "source_snapshot.zip", "w") as archive:
        for filename in source_hashes():
            archive.write(filename, str(Path(filename).relative_to(repo)))
    cases = banks(experiment)
    state = None
    for epochs in (2, 3):
        directory = root / f"mutation_0.300/population_003/seed_9/epochs_{epochs:03d}"
        run(experiment, directory, population_size=3, generations=epochs, seed=9,
            schedule=Schedule(probability_start=.3, probability_end=.3), schedule_generations=3,
            resume_state=state, duration_s=.01, case_banks=cases, checkpoint_every=1,
            plot=False, verbose=False)
        state = load_state(directory / "final_population.npz")
        assert state["generation"] == epochs and len(state["history"]) == epochs
        partial = load_state(directory / "checkpoint.npz")
        np.testing.assert_array_equal(state["population"], partial["population"])
    result = analyze(root)
    assert result["completed"] == result["expected"] == 2
    assert (root / "genome_rmse_final.png").is_file()
    with np.load(root / "genome_comparison.npz") as comparison:
        assert comparison["normalized"].shape == (2, 50)
        np.testing.assert_array_equal(comparison["rmse"], comparison["rmse"].T)
        assert not comparison["rmse"].diagonal().any()
    verification = audit(root)
    assert verification["completed"] == 2
    if experiment == "small_nm_weights":
        assert verification["small_weight_bound_verified"]
        assert verification["neuromuscular_weight_max"] < .1
        assert "0.02 ≤ w < 0.1" in (root / "REPORT.md").read_text()
        # A corrupt population must not pass the audit merely because its
        # exported winner and scores still look valid.
        path = directory / "final_population.npz"
        with np.load(path, allow_pickle=False) as saved:
            changed = {key: saved[key].copy() for key in saved.files}
        changed["physical_controllers"][0, 16] = .1
        np.savez_compressed(path, **changed)
        with pytest.raises(AssertionError, match="Neuromuscular weight"):
            audit(root)
