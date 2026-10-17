"""Audit saved numerical results without rerunning or selecting on test cases."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
from . import genome
from .config import DT_MS, MODEL, MECHANICS
from .fitness import fitness
from .dynamics import METRIC_COLUMNS
from .evolution import Schedule
from .sweep_analysis import pairwise_rmse
from dataclasses import asdict


def audit(root):
    root = Path(root).resolve()
    sweep = json.loads((root / "sweep.json").read_text())
    assert sweep["experiment"] in ("equilibrium", "small_nm_weights")
    domain = genome.ControllerDomain(small_weights=sweep["experiment"] == "small_nm_weights")
    expected = len(sweep["populations"])*len(sweep["epochs"])*len(sweep["probabilities"])*len(sweep["seeds"])
    results = sorted(root.glob("mutation_*/population_*/seed_*/epochs_*/result.json"))
    assert len(results) == expected, (len(results), expected)
    reference_banks = {}
    prefixes = {}
    weight_min, weight_max = np.inf, -np.inf
    for path in results:
        result = json.loads(path.read_text())
        directory = path.parent
        settings = json.loads((directory / "run.json").read_text())
        assert settings["experiment"] == result["experiment"] == sweep["experiment"]
        assert settings["dt_ms"] == DT_MS == .1 and settings["model"] == MODEL
        assert settings["hyperparameters"] == asdict(MECHANICS)
        assert settings["source_sha256"] == sweep["source_sha256"]
        probability = result["mutation_probability"]
        assert settings["schedule"]["probability_start"] == settings["schedule"]["probability_end"] == probability
        for name in ("training", "validation", "test"):
            content = (directory / f"{name}_cases.csv").read_bytes()
            if name not in reference_banks:
                reference_banks[name] = content
            assert reference_banks[name] == content
        with np.load(directory / "final_population.npz", allow_pickle=False) as data:
            assert data["population"].shape == (result["population"], 50)
            assert int(data["generation"]) == result["generations"]
            assert np.all((data["population"] >= 0) & (data["population"] <= 1))
            assert np.all(data["physical_controllers"][:, genome.BIAS_INDICES] == 0.)
            weights = data["physical_controllers"][:, genome.WEIGHT_INDICES]
            assert np.all(weights >= .02)
            if domain.small_weights:
                assert np.all(weights < .1), "Neuromuscular weight reaches or exceeds 0.1"
            weight_min = min(weight_min, float(weights.min()))
            weight_max = max(weight_max, float(weights.max()))
            for unit, p in zip(data["population"], data["physical_controllers"]):
                np.testing.assert_array_equal(domain.decode(unit), p)
            np.testing.assert_array_equal(data["scores"], [fitness(m) for m in data["evaluation"]])
            np.testing.assert_array_equal(data["validation_scores"], [fitness(m) for m in data["validation_metrics"]])
            winner = int(data["selected_index"])
            assert winner == int(np.argmax(data["validation_scores"]))
            assert result["training_fitness"] == data["scores"][winner]
            assert result["validation_fitness"] == data["validation_scores"][winner]
            exported = genome.decode(genome.load_genome(directory / "best_genome.json"))
            np.testing.assert_allclose(exported, data["physical_controllers"][winner], atol=1e-13, rtol=1e-13)
            history = json.loads(str(data["history_json"]))
        assert len(history) == result["generations"]
        assert [h["generation"] for h in history] == list(range(1, result["generations"]+1))
        assert np.all(np.diff([h["best_fitness"] for h in history]) >= 0)
        schedule = Schedule(**settings["schedule"])
        for row in history:
            assert row["mutation_probability"] == probability
            assert row["mutation_sigma"] == schedule.at(row["generation"], max(sweep["epochs"]))[0]
        key = (result["population"], probability, result["seed"])
        if key in prefixes:
            assert history[:len(prefixes[key])] == prefixes[key]
        prefixes[key] = history
        with (directory / "test_metrics.csv").open() as stream:
            tests = list(csv.DictReader(stream))
        metrics = np.array([[float(row[k]) for k in METRIC_COLUMNS] for row in tests])
        assert result["test_fitness"] == fitness(metrics)
        assert result["test_success"] == metrics[:, 5].mean()
        assert len(tests) == 48
    with np.load(root / "genome_comparison.npz", allow_pickle=False) as data:
        assert data["normalized"].shape == (expected, 50)
        np.testing.assert_array_equal(data["rmse"], pairwise_rmse(data["normalized"]))
        np.testing.assert_array_equal(data["differences_to_best"], data["normalized"]-data["normalized"][int(data["best_index"])] )
    repo = Path(__file__).resolve().parents[4]
    with zipfile.ZipFile(root / "source_snapshot.zip") as archive:
        for filename, digest in sweep["source_sha256"].items():
            assert hashlib.sha256(archive.read(str(Path(filename).relative_to(repo)))).hexdigest() == digest
    summary = dict(status="passed", experiment=sweep["experiment"], completed=expected, trajectories=len(prefixes),
                   small_weight_bound_verified=domain.small_weights,
                   neuromuscular_weight_min=weight_min, neuromuscular_weight_max=weight_max,
                   shared_case_banks=True, dt_ms=DT_MS, external_currents_all_zero=True,
                   hyperparameters_unchanged=True, validation_selection_verified=True,
                   epoch_continuations_verified=True, rmse_verified=True, source_snapshot_verified=True)
    (root / "verification.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    audit(parser.parse_args().output)


if __name__ == "__main__":
    main()
