"""Setpoint-specific current laws and an explicitly invoked optional fitting API.

Nothing in the balance core imports this module. The default C scripts only
replay a supplied controller/current law; they do not call ``fit_current_law``.
Historical searches are preserved in archive/pre_ABC_20261020 and frozen runtimes.
"""
import json
from pathlib import Path

import numpy as np

from ..core.evolution import Schedule, optimize
from ..core.genome import GenomeConfig, write_json
from ..core.run import PopulationEvaluator, save_replays, write_csv


def polynomial_currents(coefficients, target_rad):
    """Evaluate two polynomials in target/45 degrees, including constant terms.

    Coefficients have shape (2, degree+1), with degree 0, 2 or 3. The resulting
    two currents remain constant throughout an episode; no state feedback is added.
    """
    coefficients = np.asarray(coefficients, dtype=float)
    if (coefficients.ndim != 2 or coefficients.shape[0] != 2
            or coefficients.shape[1] not in (1, 3, 4)
            or not np.isfinite(coefficients).all()
            or not np.isfinite(target_rad) or abs(target_rad) >= np.pi/2):
        raise ValueError("Expected two finite constant/quadratic/cubic coefficient rows and a valid target")
    x = target_rad / np.deg2rad(45.)
    return coefficients @ (x ** np.arange(coefficients.shape[1]))


def load_current_law(path):
    """Load explicit coefficients, also accepting historical current_law.json files."""
    payload = json.loads(Path(path).read_text())
    coefficients = np.asarray(payload["coefficients"], dtype=float)
    polynomial_currents(coefficients, 0.)
    return coefficients


def fit_current_law(config, physical, banks, output, *, degree=3, population=50,
                    generations=150, seed=42, workers=1, bound=1000.):
    """Explicitly fit currents on supplied independent banks, keeping the genome fixed.

    This optional API has no automatic callers. The caller must supply training,
    validation and test banks with five target angles as needed. It optimizes
    current coefficients only and saves all coefficient generations separately
    from controller genomes. No bank is silently borrowed from balance training.
    """
    config = config if isinstance(config, GenomeConfig) else GenomeConfig(config)
    config.encode(physical)
    if (degree not in (2, 3) or set(banks) != {"training", "validation", "test"}
            or not np.isfinite(bound) or bound <= 0):
        raise ValueError("Expected degree 2/3, positive coefficient bound and three explicit banks")
    row_sets = [set(map(tuple, banks[name])) for name in ("training", "validation", "test")]
    if any(row_sets[i] & row_sets[j] for i in range(3) for j in range(i+1, 3)):
        raise ValueError("Training, validation and test cases must be disjoint")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "generations").mkdir()
    config.save(output / "fixed_genome.json", physical)
    np.savez_compressed(output / "cases.npz", **banks)
    duration = config.data["simulation"]["duration_s"]

    def decode(unit):
        """Map normalized current-law coefficients to a symmetric physical interval."""
        return ((np.asarray(unit) * 2 - 1) * bound).reshape(-1, 2, degree+1)

    with PopulationEvaluator(config.mechanics, workers) as evaluator:
        def evaluate_on(unit, cases):
            """Evaluate explicit current arrays with a frozen physical controller."""
            laws = decode(unit)
            currents = np.array([[polynomial_currents(law, case[0]) for case in cases] for law in laws])
            controllers = np.repeat(np.asarray(physical)[None, :], len(laws), axis=0)
            return evaluator.evaluate(controllers, cases, duration, currents=currents)

        def evaluate(unit):
            """Score coefficient candidates on training only."""
            return evaluate_on(unit, banks["training"])

        def save_state(generation, unit, scores, metrics, history, rng_state):
            """Save all coefficient candidates and training metrics before reproduction."""
            np.savez_compressed(output / "generations" / f"generation_{generation:04d}.npz",
                                kind="current_polynomial", normalized=unit, coefficients=decode(unit),
                                training_scores=scores, training_metrics=metrics,
                                generation=generation, rng_state=json.dumps(rng_state))
            write_csv(output / "history.csv", history)

        unit, scores, metrics, history, rng = optimize(
            np.full(2*(degree+1), .5), evaluate, size=population, generations=generations, seed=seed,
            schedule=Schedule(probability_start=.3, probability_end=.3), on_state=save_state)
        validation_scores, validation = evaluate_on(unit, banks["validation"])
        winner = int(np.argmax(validation_scores))
        test_scores, test = evaluate_on(unit[winner:winner+1], banks["test"])
    laws = decode(unit)
    write_json(output / "current_law.json", dict(coefficients=laws[winner].tolist(),
               variable="target_rad / radians(45)", degree=degree, selected_index=winner,
               selection="validation", seed=seed))
    np.savez_compressed(output / "final_coefficients.npz", coefficients=laws,
                        normalized=unit, training_scores=scores, validation_scores=validation_scores,
                        validation_metrics=validation, selected_index=winner, test_metrics=test[0])
    currents = np.array([polynomial_currents(laws[winner], case[0]) for case in banks["test"]])
    save_replays(output / "test_traces", config, physical, banks["test"],
                 duration_s=duration, currents=currents)
    result = dict(validation_fitness=float(validation_scores[winner]),
                  test_fitness=float(test_scores[0]), test_success=float(test[0, :, 5].mean()))
    write_json(output / "result.json", result)
    return result
