"""Measure the fixed initial controllers; these results never affect selection."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
import numpy as np
from scripts.SteppingReflexGA.experiments.core import genome
from scripts.SteppingReflexGA.experiments.core.cases import banks
from scripts.SteppingReflexGA.experiments.core.population import PopulationEvaluator
from scripts.SteppingReflexGA.experiments.core.run import source_hashes, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    root = parser.parse_args().output.resolve()
    experiments = ("equilibrium", "small_nm_weights")
    controllers = []
    for experiment in experiments:
        domain = genome.ControllerDomain(experiment == "small_nm_weights")
        controllers.append(domain.decode(domain.initial()))
    controllers = np.array(controllers)
    records, arrays = {}, {"physical_controllers": controllers}
    with PopulationEvaluator(workers=1) as evaluator:
        for name, cases in banks("equilibrium").items():
            scores, metrics = evaluator.evaluate(controllers, None, cases, duration_s=10.)
            arrays[f"{name}_cases"] = cases
            arrays[f"{name}_metrics"] = metrics
            for index, experiment in enumerate(experiments):
                records.setdefault(experiment, {})[name] = dict(
                    fitness=float(scores[index]), success=float(metrics[index, :, 5].mean()),
                    late_rmse_deg=float(np.rad2deg(metrics[index, :, 7]).mean()),
                    trials=len(cases))
    for index, experiment in enumerate(experiments):
        records[experiment]["weights"] = controllers[index, genome.WEIGHT_INDICES].tolist()
    write_json(root / "initial_controller_baseline.json", dict(
        controllers=records, duration_s=10., dt_ms=.1,
        note="Fixed initial controllers, evaluated without training; no test-based selection.",
        source_sha256=source_hashes()))
    np.savez_compressed(root / "initial_controller_baseline.npz", **arrays)
    print(records)


if __name__ == "__main__":
    main()
