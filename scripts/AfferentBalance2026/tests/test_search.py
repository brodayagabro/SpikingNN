"""Search constraints, deterministic parallel evaluation, selection, and saved results."""
import json
from pathlib import Path
import numpy as np
import pytest
from scripts.SteppingReflexGA.experiments.core.genome import ControllerDomain, WEIGHT_INDICES, BIAS_INDICES
from scripts.SteppingReflexGA.experiments.core.evolution import initial_population, reproduce, Schedule
from scripts.SteppingReflexGA.experiments.core.cases import banks
from scripts.SteppingReflexGA.experiments.core.population import PopulationEvaluator, load_controller_population, select_controller
from scripts.SteppingReflexGA.experiments.core.run import run


@pytest.mark.parametrize("small", [False, True])
def test_genome_constraints_survive_mutation_and_endpoints(small):
    domain, rng = ControllerDomain(small), np.random.default_rng(8)
    population = initial_population(domain.initial(), 9, rng)
    for _ in range(20):
        scores = rng.random(9)
        best = population[np.argmax(scores)].copy()
        population = reproduce(population, scores, rng, .8, 1.)
        np.testing.assert_array_equal(population[0], best)
        assert population.shape == (9, 50)
        assert np.all((population >= 0) & (population <= 1))
        for unit in population:
            p = domain.decode(unit)
            assert np.all(p[BIAS_INDICES] == 0.)
            assert np.all(p[WEIGHT_INDICES] < .1) if small else np.all(p[WEIGHT_INDICES] <= 1.5)
            np.testing.assert_allclose(domain.encode(p), unit, atol=1e-14)
    for edge in [np.zeros(50), np.ones(50)]:
        p = domain.decode(edge)
        assert np.all(p[BIAS_INDICES] == 0.)
        if small:
            assert np.all(p[WEIGHT_INDICES] < .1)


def test_only_current_and_weight_constraints_differ_between_protocols():
    broad, narrow = ControllerDomain(False), ControllerDomain(True)
    a, b = broad.decode(np.ones(50)), narrow.decode(np.ones(50))
    other = np.setdiff1d(np.arange(52), WEIGHT_INDICES)
    np.testing.assert_array_equal(a[other], b[other])


def test_schedule_includes_single_generation():
    schedule = Schedule()
    np.testing.assert_allclose(schedule.at(1, 1), [.2, .6])
    values = np.array([schedule.at(i, 100) for i in range(1, 101)])
    np.testing.assert_allclose(values[-1], [.015, .1])
    assert np.all(np.diff(values, axis=0) <= 0)


def test_banks_are_reproducible_and_disjoint():
    for experiment in ("equilibrium", "small_nm_weights", "angle_setpoint"):
        one, two = banks(experiment), banks(experiment)
        for name in one:
            np.testing.assert_array_equal(one[name], two[name])
        for a, b in (("training", "validation"), ("training", "test"), ("validation", "test")):
            assert not set(map(tuple, one[a])) & set(map(tuple, one[b]))
    for key in banks("equilibrium"):
        np.testing.assert_array_equal(banks("equilibrium")[key], banks("small_nm_weights")[key])


def test_population_serial_equals_spawn_workers():
    domain = ControllerDomain()
    population = initial_population(domain.initial(), 3, np.random.default_rng(42))
    controllers = np.array([domain.decode(u) for u in population])
    cases = banks("equilibrium")["training"][:2]
    with PopulationEvaluator(1) as serial, PopulationEvaluator(2) as parallel:
        a = serial.evaluate(controllers, None, cases, .02)
        b = parallel.evaluate(controllers, None, cases, .02)
        p, index, scores, metrics = select_controller(controllers, cases, serial, .02)
    for first, second in zip(a, b):
        np.testing.assert_array_equal(first, second)
    assert index == np.argmax(a[0])
    np.testing.assert_array_equal(p, controllers[index])


def short_banks(experiment):
    return {key: values[:2] for key, values in banks(experiment).items()}


def test_three_experiments_export_and_freeze_selected_controller(tmp_path):
    directories = {}
    for experiment in ("equilibrium", "small_nm_weights", "angle_setpoint"):
        output = tmp_path / experiment
        population_from = directories["small_nm_weights"] / "final_population.npz" if experiment == "angle_setpoint" else None
        result = run(experiment, output, population_size=3, generations=2, seed=5,
                     duration_s=.01, case_banks=short_banks(experiment),
                     population_from=population_from, plot=False)
        directories[experiment] = output
        assert result["dt_ms"] == .1 and result["evaluations"] == 6
        assert result["free_genes"] == (6 if experiment == "angle_setpoint" else 50)
        assert (output / "test_metrics.csv").is_file()
        with np.load(output / "final_population.npz") as saved:
            winner = int(saved["selected_index"])
            assert winner == int(np.argmax(saved["validation_scores"]))
            if experiment == "angle_setpoint":
                with np.load(output / "controller_selection.npz") as selected:
                    fixed = selected["physical_controller"]
                    assert int(selected["selected_index"]) == int(np.argmax(selected["scores"]))
                np.testing.assert_array_equal(saved["physical_controllers"], np.tile(fixed, (3, 1)))
                law = json.loads((output / "current_law.json").read_text())
                assert len(law["genes"]) == 6 and all(g["description"] for g in law["genes"])
            else:
                np.testing.assert_array_equal(load_controller_population(output / "final_population.npz"), saved["physical_controllers"])
    with pytest.raises(FileExistsError):
        run("equilibrium", directories["equilibrium"], population_size=3, generations=1)


def test_fixed_seed_repeats_entire_evolution(tmp_path):
    for name in ("a", "b"):
        run("equilibrium", tmp_path / name, population_size=5, generations=3, seed=14,
            duration_s=.01, case_banks=short_banks("equilibrium"), plot=False)
    with np.load(tmp_path / "a/final_population.npz") as a, np.load(tmp_path / "b/final_population.npz") as b:
        for key in ("population", "scores", "evaluation", "validation_scores", "selected_index"):
            np.testing.assert_array_equal(a[key], b[key])


def test_historical_population_cannot_be_silently_reused(tmp_path):
    old = tmp_path / "old.npz"
    np.savez(old, population=np.zeros((3, 52)))
    with pytest.raises(ValueError, match="rerun"):
        load_controller_population(old)
