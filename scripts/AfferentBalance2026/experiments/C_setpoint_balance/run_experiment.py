"""Replay a fixed controller at a target angle with explicit constant currents."""
import argparse
from datetime import datetime
from pathlib import Path
import sys
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from scripts.AfferentBalance2026.experiments.core.genome import GenomeConfig, load_genome, write_json
from scripts.AfferentBalance2026.experiments.core.run import REPO, save_replays, source_hashes
from scripts.AfferentBalance2026.experiments.simulation import MODEL, PARAMETERS, case_grid, simulate
from scripts.AfferentBalance2026.experiments.C_setpoint_balance.polynomial_study import (
    load_current_law, polynomial_currents,
)


def run(config, output, *, genome=None, targets=(0.,), angles=(30.,), speeds=(0.,),
        currents=(0., 0.), current_law=None, duration_s=None):
    """Save fixed-controller replays; never train or select a network on test outcomes."""
    config = config if isinstance(config, GenomeConfig) else GenomeConfig(config)
    if config.data["experiment"] != "C_setpoint_balance":
        raise ValueError("Use the C_setpoint_balance configuration")
    physical = config.initial_physical.copy()
    if genome is not None:
        saved_config, physical = load_genome(genome)
        if not np.array_equal(saved_config.mechanics, config.mechanics):
            raise ValueError("Saved controller mechanics differ from the C configuration")
        config.encode(physical)  # Includes exact fixed-geometry agreement.
    duration_s = config.data["simulation"]["duration_s"] if duration_s is None else duration_s
    cases = np.concatenate([case_grid(angles, speeds, target) for target in targets])
    law = None if current_law is None else load_current_law(current_law)
    applied = (np.tile(np.asarray(currents, dtype=float), (len(cases), 1)) if law is None
               else np.array([polynomial_currents(law, case[0]) for case in cases]))
    if applied.shape != (len(cases), 2) or not np.isfinite(applied).all():
        raise ValueError("Expected two finite currents per case")
    # Validate numerical inputs before creating an output directory.
    simulate(physical, cases[0], config.mechanics, currents=applied[0],
             duration_s=duration_s, stride=0)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config.save(output / "best_genome.json", physical,
                source="untrained configuration" if genome is None else str(Path(genome).resolve()),
                training_performed=False, selection="supplied fixed controller")
    write_json(output / "genome_cfg.json", config.data)
    hashes = source_hashes()
    with zipfile.ZipFile(output / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in hashes:
            archive.write(REPO / name, name)
        archive.write(output / "genome_cfg.json", "genome_cfg.json")
    record = dict(model=MODEL, dt_ms=.1, muscle_torque_factor=1., pd_used=False,
                  training_performed=False, source_sha256=hashes,
                  afferent_parameters=list(PARAMETERS), configuration_sha256=config.signature,
                  status="running")
    write_json(output / "run.json", record)
    rows = save_replays(output / "traces", config, physical, cases,
                        duration_s=duration_s, currents=applied)
    result = dict(experiment="C_setpoint_balance", training_performed=False, cases=len(cases),
                  dt_ms=.1, currents=applied.tolist(), duration_s=duration_s,
                  successes=int(sum(row["success"] for row in rows)),
                  controller_source="untrained configuration" if genome is None else str(genome),
                  coefficients=None if law is None else law.tolist())
    if source_hashes() != hashes:
        raise RuntimeError("Calculation sources changed during replay")
    write_json(output / "run.json", {**record, "status": "completed"})
    write_json(output / "result.json", result)
    return result


def main(grid=False):
    """Parse replay options; a grid sweep is a simulation sweep, not a GA search."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("genome_cfg.json"))
    parser.add_argument("--genome", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--targets", nargs="+", type=float)
    parser.add_argument("--angles", nargs="+", type=float)
    parser.add_argument("--speeds", nargs="+", type=float)
    command = parser.add_mutually_exclusive_group()
    command.add_argument("--currents", nargs=2, type=float)
    command.add_argument("--current-law", type=Path)
    args = parser.parse_args()
    cfg = GenomeConfig(args.config)
    targets = args.targets if args.targets is not None else (cfg.data["setpoint"]["targets_deg"] if grid else [0.])
    angles = args.angles if args.angles is not None else (np.linspace(-50, 50, 10) if grid else [30.])
    speeds = args.speeds if args.speeds is not None else (np.linspace(-1, 1, 10) if grid else [0.])
    output = args.output or Path(__file__).parent / "results" / (
        ("grid_" if grid else "replay_") + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    result = run(cfg, output, genome=args.genome, targets=targets, angles=angles, speeds=speeds,
                 currents=args.currents if args.currents is not None else cfg.data["setpoint"]["currents"],
                 current_law=args.current_law, duration_s=args.duration)
    print({"output": str(output), "cases": result["cases"], "training_performed": False})


if __name__ == "__main__":
    main()
