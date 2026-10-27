"""Replay a C solution or explicitly train its controller and command genes jointly."""
import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import sys
import json
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from scripts.AfferentBalance2026.experiments.core.genome import GenomeConfig, load_genome, write_json
from scripts.AfferentBalance2026.experiments.core.run import REPO, save_replays, source_hashes
from scripts.AfferentBalance2026.experiments.simulation import MODEL, PARAMETERS, case_grid, simulate
from scripts.AfferentBalance2026.experiments.C_setpoint_balance.polynomial_study import (
    load_current_law, polynomial_currents,
)
from scripts.AfferentBalance2026.experiments.C_setpoint_balance.joint_study import (
    SCHEMA as JOINT_SCHEMA, load_joint_genome, run_pipeline, sweep_joint,
)


def run(config, output, *, genome=None, targets=(0.,), angles=(30.,), speeds=(0.,),
        currents=(0., 0.), current_law=None, duration_s=None):
    """Save fixed-controller replays; never train or select a network on test outcomes."""
    config = config if isinstance(config, GenomeConfig) else GenomeConfig(config)
    if config.data["experiment"] != "C_setpoint_balance":
        raise ValueError("Use the C_setpoint_balance configuration")
    physical = config.initial_physical.copy()
    joint = None
    if genome is not None:
        if json.loads(Path(genome).read_text()).get('schema') == JOINT_SCHEMA:
            joint, unit = load_joint_genome(genome)
            physical, joint_command = joint.decode(unit)
            saved_config = joint.controller
            if current_law is not None or np.any(np.asarray(currents) != 0):
                raise ValueError('A joint genome already contains its current command')
        else:
            saved_config, physical = load_genome(genome)
        if not np.array_equal(saved_config.mechanics, config.mechanics):
            raise ValueError("Saved controller mechanics differ from the C configuration")
        config.encode(physical)  # Includes exact fixed-geometry agreement.
        if joint is not None:
            config = saved_config
    duration_s = config.data["simulation"]["duration_s"] if duration_s is None else duration_s
    cases = np.concatenate([case_grid(angles, speeds, target) for target in targets])
    law = None if current_law is None else load_current_law(current_law)
    applied = (np.tile(np.asarray(currents, dtype=float), (len(cases), 1)) if law is None
               else np.array([polynomial_currents(law, case[0]) for case in cases]))
    if joint is not None:
        applied = joint.currents(joint_command, cases)
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
    if joint is not None:
        (output/'best_genome.json').rename(output/'best_controller.json')
        joint.save(output/'best_genome.json',unit,training_performed=False,selection='supplied joint genome')
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
                  coefficients=(joint_command.tolist() if joint is not None and joint.kind=='polynomial'
                                else None if law is None else law.tolist()))
    if source_hashes() != hashes:
        raise RuntimeError("Calculation sources changed during replay")
    write_json(output / "run.json", {**record, "status": "completed"})
    write_json(output / "result.json", result)
    return result


def main(grid=False):
    """Keep replay explicit and expose joint/staged training and method comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("genome_cfg.json"))
    parser.add_argument("--genome", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--duration", type=float)
    parser.add_argument('--mode', choices=['replay','joint','staged','compare'], default='replay')
    parser.add_argument('--population', type=int)
    parser.add_argument('--generations', type=int)
    parser.add_argument('--mutation-probability', type=float)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--degree', type=int, choices=[2,3])
    parser.add_argument('--table-generations', type=int)
    parser.add_argument('--populations', nargs='+', type=int)
    parser.add_argument('--probabilities', nargs='+', type=float)
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--degrees', nargs='+', type=int, choices=[2,3])
    start = parser.add_mutually_exclusive_group()
    start.add_argument('--seed-genome', type=Path)
    start.add_argument('--from-scratch', action='store_true')
    parser.add_argument("--targets", nargs="+", type=float)
    parser.add_argument("--angles", nargs="+", type=float)
    parser.add_argument("--speeds", nargs="+", type=float)
    command = parser.add_mutually_exclusive_group()
    command.add_argument("--currents", nargs=2, type=float)
    command.add_argument("--current-law", type=Path)
    args = parser.parse_args()
    cfg = GenomeConfig(args.config)
    if args.mode != 'replay':
        if any(x is not None for x in (args.genome,args.targets,args.angles,args.speeds,args.currents,args.current_law)):
            parser.error('Training uses config banks; use --seed-genome for an initial controller')
        if args.mode=='compare' and not grid: parser.error('Use run_sweep.py for method comparisons')
        if args.duration is not None:
            data=deepcopy(cfg.data); data['simulation']['duration_s']=args.duration; cfg=GenomeConfig(data)
        ga=cfg.data['ga']; study=cfg.data['sweep']
        seed_path = None if args.from_scratch else (args.seed_genome or
                     (args.config.resolve().parent/cfg.data['setpoint']['seed_genome']))
        output=args.output or Path(__file__).parent/'results'/(
            args.mode+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        common=dict(generations=args.generations if args.generations is not None else ga['generations'],
                    workers=args.workers if args.workers is not None else ga['workers'],seed_genome=seed_path)
        if grid:
            result=sweep_joint(cfg,output,methods=study['methods'] if args.mode=='compare' else [args.mode],
                degrees=args.degrees or ([args.degree] if args.degree else study['degrees']),
                populations=args.populations or ([args.population] if args.population else study['populations']),
                probabilities=args.probabilities or ([args.mutation_probability] if args.mutation_probability is not None else study['mutation_probabilities']),
                seeds=args.seeds or ([args.seed] if args.seed is not None else study['seeds']),**common)
        else:
            result=run_pipeline(cfg,output,method=args.mode,degree=args.degree or cfg.data['setpoint']['degree'],
                population=args.population if args.population is not None else ga['population'],
                probability=args.mutation_probability if args.mutation_probability is not None else ga['mutation_probability'],
                seed=args.seed if args.seed is not None else ga['seed'],table_generations=args.table_generations,**common)
        print(json.dumps(dict(output=str(output),result=result),ensure_ascii=False))
        return
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
