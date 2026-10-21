"""Population evaluation, GA execution, checkpoints and parameter sweeps.

The core has no current-law model. External inputs, when explicitly supplied to
the evaluator, are ordinary arrays. Balance training always uses zero currents.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime
from itertools import product
import json
import multiprocessing as mp
from pathlib import Path
import zipfile

import numpy as np

from .evolution import Schedule, optimize
from .fitness import fitness
from .genome import GenomeConfig, SCHEMA, NAMES, file_hash, write_json
from ..simulation import (
    CASE_COLUMNS, METRIC_COLUMNS, TRACE_COLUMNS, PARAMETERS, MODEL,
    balance_banks, case_grid, simulate,
)

EXPERIMENTS = Path(__file__).resolve().parents[1]
REPO = EXPERIMENTS.parents[2]


def write_csv(path, rows):
    """Write nonempty records using the first record's key order."""
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot write an empty table without columns")
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_hashes():
    """Hash active calculation modules and library sources, excluding result trees."""
    files = [p for p in EXPERIMENTS.rglob("*.py")
             if not {"results", "__pycache__"}.intersection(p.relative_to(EXPERIMENTS).parts)]
    files += list((REPO / "src/SpikingNN").rglob("*.py"))
    return {str(p.relative_to(REPO)): file_hash(p) for p in sorted(files)}


def _evaluate_task(task):
    """Evaluate one genome on a bank; all worker inputs are explicit and deterministic."""
    physical, cases, mechanics, duration_s, currents = task
    return np.array([simulate(physical, case, mechanics, currents=currents[i],
                              duration_s=duration_s, stride=0)[0]
                     for i, case in enumerate(cases)])


class PopulationEvaluator:
    """Evaluate populations through a persistent spawn pool while preserving order."""

    def __init__(self, mechanics, workers=1):
        """Configure mechanics and worker count; create processes only on entry."""
        if not isinstance(workers, int) or workers < 1:
            raise ValueError("workers must be a positive integer")
        self.mechanics = np.asarray(mechanics, dtype=float)
        self.workers = workers
        self.pool = None

    def __enter__(self):
        """Start a persistent process pool when multiple workers are requested."""
        if self.workers > 1:
            self.pool = ProcessPoolExecutor(self.workers, mp_context=mp.get_context("spawn"))
        return self

    def __exit__(self, *args):
        """Wait for submitted evaluations and release worker processes."""
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)

    def evaluate(self, physical, cases, duration_s=10., currents=None):
        """Return scores (N,) and metrics (N,K,9), with optional explicit currents.

        Currents have shape (N,K,2); omitted currents are exactly zero.
        No target-to-current transformation occurs in this method.
        """
        physical, cases = np.asarray(physical, dtype=float), np.asarray(cases, dtype=float)
        if physical.ndim != 2 or physical.shape[1] != 52 or not len(physical):
            raise ValueError("Expected a nonempty N-by-52 population")
        if cases.ndim != 2 or cases.shape[1] != 5 or not len(cases):
            raise ValueError("Expected a nonempty K-by-5 case bank")
        shape = (len(physical), len(cases), 2)
        currents = np.zeros(shape) if currents is None else np.asarray(currents, dtype=float)
        if currents.shape != shape or not np.isfinite(currents).all():
            raise ValueError("Expected finite currents with shape (N,K,2)")
        tasks = [(p, cases, self.mechanics, duration_s, currents[i]) for i, p in enumerate(physical)]
        values = self.pool.map(_evaluate_task, tasks) if self.pool else map(_evaluate_task, tasks)
        metrics = np.array(list(values))
        return np.array([fitness(m) for m in metrics]), metrics


def save_replays(output, config, physical, cases, *, duration_s, currents=None):
    """Save full 0.1 ms trajectories and their inputs, including early failures."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    currents = np.zeros((len(cases), 2)) if currents is None else np.asarray(currents, dtype=float)
    rows = []
    for i, case in enumerate(cases):
        metrics, trace = simulate(physical, case, config.mechanics, duration_s=duration_s,
                                  currents=currents[i], stride=1)
        filename = f"case_{i:03d}.npz"
        np.savez_compressed(output / filename, trace=trace, columns=np.array(TRACE_COLUMNS),
                            metrics=metrics, metric_columns=np.array(METRIC_COLUMNS),
                            case=case, case_columns=np.array(CASE_COLUMNS),
                            physical=physical, names=np.array(NAMES), schema=SCHEMA,
                            currents=currents[i], mechanics=config.mechanics, dt_ms=.1)
        rows.append(dict(file=filename, theta0_deg=float(np.rad2deg(case[1])),
                         omega0_rad_s=float(case[2]), target_deg=float(np.rad2deg(case[0])),
                         samples=len(trace), sha256=file_hash(output / filename),
                         **dict(zip(METRIC_COLUMNS, map(float, metrics)))))
    write_csv(output / "metrics.csv", rows)
    return rows


def train(config, output, *, population=None, generations=None, seed=None,
          workers=None, mutation_probability=None, duration_s=None, checkpoints=()):
    """Train a zero-current balance population and select winners by validation.

    Save every evaluated generation and the complete final population. Save the
    best ``save_top`` final-validation genomes and their full-resolution dynamics
    on the configured replay angles. The independent test bank scores only the
    validation winner. Optional sweep checkpoints share one uninterrupted GA.
    Existing output directories are rejected.
    """
    config = config if isinstance(config, GenomeConfig) else GenomeConfig(config)
    if config.data["experiment"] not in ("A_balance", "B_balance"):
        raise ValueError("Balance GA accepts only A_balance and B_balance configurations")
    ga, settings = config.data["ga"], config.data["simulation"]
    population = ga["population"] if population is None else population
    generations = ga["generations"] if generations is None else generations
    seed = ga["seed"] if seed is None else seed
    workers = ga["workers"] if workers is None else workers
    probability = ga["mutation_probability"] if mutation_probability is None else mutation_probability
    duration_s = settings["duration_s"] if duration_s is None else duration_s
    if (not all(isinstance(v, int) for v in (population, generations, seed, workers))
            or population < 3 or generations < 1 or seed < 0 or workers < 1
            or not np.isfinite(duration_s) or duration_s < .0001):
        raise ValueError("Invalid GA settings or duration")
    checkpoints = tuple(sorted(set(checkpoints)))
    if any(not isinstance(g, int) or not 1 <= g <= generations for g in checkpoints):
        raise ValueError("Checkpoints must lie inside the training horizon")
    schedule = Schedule(probability_start=probability, probability_end=probability)
    cases = balance_banks(settings)
    replay_cases = case_grid(settings["replay_angles_deg"], [0.])
    if not isinstance(settings["save_top"], int) or settings["save_top"] < 1:
        raise ValueError("save_top must be a positive integer")
    # Fail invalid numerical durations before creating a partial result directory.
    simulate(config.initial_physical, cases["training"][0], config.mechanics,
             duration_s=.0001, stride=0)
    if not np.isclose(duration_s * 10000, round(duration_s * 10000), rtol=0, atol=1e-7):
        raise ValueError("Duration must be on the 0.1 ms grid")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "generations").mkdir()
    write_json(output / "genome_cfg.json", config.data)
    np.savez_compressed(output / "cases.npz", columns=np.array(CASE_COLUMNS), **cases,
                        replay=replay_cases)
    hashes = source_hashes()
    with zipfile.ZipFile(output / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in hashes:
            archive.write(REPO / name, name)
        archive.write(output / "genome_cfg.json", "genome_cfg.json")
    run_record = dict(schema=SCHEMA, model=MODEL, dt_ms=.1, muscle_torque_factor=1.,
                      pd_used=False, currents=[0., 0.], experiment=config.data["experiment"],
                      configuration_sha256=config.signature, population=population,
                      generations=generations, seed=seed, workers=workers,
                      mutation_probability=probability, duration_s=duration_s,
                      checkpoints=list(checkpoints), afferent_parameters=list(PARAMETERS),
                      source_sha256=hashes, status="running")
    write_json(output / "run.json", run_record)
    checkpoint_results = []
    with PopulationEvaluator(config.mechanics, workers) as evaluator:
        def evaluate(unit):
            """Evaluate normalized candidates only on the training bank."""
            physical = np.array([config.decode(row) for row in unit])
            return evaluator.evaluate(physical, cases["training"], duration_s)

        def finalize(folder, generation, unit, scores, metrics):
            """Validate all candidates; save winners, independent test and replay traces."""
            folder.mkdir(parents=True, exist_ok=True)
            write_json(folder / "genome_cfg.json", config.data)
            physical = np.array([config.decode(row) for row in unit])
            validation_scores, validation = evaluator.evaluate(physical, cases["validation"], duration_s)
            order = np.argsort(-validation_scores, kind="stable")
            winner = int(order[0])
            test_scores, test = evaluator.evaluate(physical[winner:winner+1], cases["test"], duration_s)
            np.savez_compressed(
                folder / "final_population.npz", schema=SCHEMA, names=np.array(NAMES),
                configuration_sha256=config.signature, normalized=unit, physical=physical,
                training_scores=scores, training_metrics=metrics,
                validation_scores=validation_scores, validation_metrics=validation,
                selected_index=winner, generation=generation)
            config.save(folder / "best_genome.json", physical[winner],
                        selection="final-population validation", population_index=winner,
                        generation=generation, seed=seed)
            write_csv(folder / "validation_ranking.csv", [
                dict(rank=rank, population_index=int(index),
                     training_fitness=float(scores[index]),
                     validation_fitness=float(validation_scores[index]))
                for rank, index in enumerate(order, 1)])
            write_csv(folder / "test_metrics.csv", [
                dict(trial=i, **dict(zip(METRIC_COLUMNS, map(float, row))))
                for i, row in enumerate(test[0])])
            for rank, index in enumerate(order[:settings["save_top"]], 1):
                directory = folder / "best" / f"rank_{rank:03d}"
                directory.mkdir(parents=True)
                config.save(directory / "genome.json", physical[index],
                            rank=rank, population_index=int(index), generation=generation,
                            selection="final-population validation")
                save_replays(directory, config, physical[index], replay_cases, duration_s=duration_s)
            result = dict(experiment=config.data["experiment"], generation=generation,
                          population=population, seed=seed, mutation_probability=probability,
                          selected_index=winner, validation_fitness=float(validation_scores[winner]),
                          test_fitness=float(test_scores[0]), test_success=float(test[0, :, 5].mean()),
                          test_late_rmse_deg=float(np.rad2deg(test[0, :, 7]).mean()),
                          free_parameters=config.size, fixed_geometry_m=physical[winner, 46:48].tolist(),
                          best_saved=min(settings["save_top"], population), duration_s=duration_s)
            write_json(folder / "result.json", result)
            return result

        def save_state(generation, unit, scores, metrics, history, rng_state):
            """Persist the full training population before reproduction."""
            physical = np.array([config.decode(row) for row in unit])
            np.savez_compressed(
                output / "generations" / f"generation_{generation:04d}.npz",
                schema=SCHEMA, configuration_sha256=config.signature,
                names=np.array(NAMES), normalized=unit, physical=physical,
                training_scores=scores, training_metrics=metrics, generation=generation,
                rng_state=json.dumps(rng_state))
            write_csv(output / "history.csv", history)
            write_json(output / "progress.json", dict(generation=generation,
                       total=generations, best_fitness=history[-1]["best_fitness"]))
            if generation in checkpoints and generation != generations:
                folder = output / "checkpoints" / f"epochs_{generation:03d}"
                result = finalize(folder, generation, unit, scores, metrics)
                checkpoint_results.append(dict(directory=str(folder.relative_to(output)), **result))

        units, scores, metrics, history, rng = optimize(
            config.initial(), evaluate, size=population, generations=generations,
            seed=seed, schedule=schedule, on_state=save_state)
        result = finalize(output, generations, units, scores, metrics)
    if source_hashes() != hashes:
        raise RuntimeError("Calculation sources changed during training; this run is not certified")
    run_record["status"] = "completed"
    write_json(output / "run.json", run_record)
    write_json(output / "checkpoints.json", [*checkpoint_results, dict(directory=".", **result)])
    return result


def sweep(config, output, *, populations=None, generations=None, probabilities=None,
          seeds=None, workers=None, duration_s=None):
    """Run each population/mutation/seed line once up to the largest checkpoint.

    Save summary tables, normalized-genome RMSE and heatmaps. Checkpoints along
    one line share their training history; they are not independent GA runs.
    """
    config = config if isinstance(config, GenomeConfig) else GenomeConfig(config)
    if config.data["experiment"] not in ("A_balance", "B_balance"):
        raise ValueError("Only balance configurations participate in a GA sweep")
    defaults = config.data["sweep"]
    populations = list(defaults["populations"] if populations is None else populations)
    generations = sorted(set(defaults["generations"] if generations is None else generations))
    probabilities = list(defaults["mutation_probabilities"] if probabilities is None else probabilities)
    seeds = list(defaults["seeds"] if seeds is None else seeds)
    if not all((populations, generations, probabilities, seeds)):
        raise ValueError("Sweep axes cannot be empty")
    if (any(not isinstance(n, int) or n < 3 for n in populations)
            or any(not isinstance(n, int) or n < 1 for n in generations)
            or any(not np.isfinite(p) or not 0 <= p <= 1 for p in probabilities)
            or any(not isinstance(seed, int) or seed < 0 for seed in seeds)
            or any(len(axis) != len(set(axis)) for axis in (populations, probabilities, seeds))):
        raise ValueError("Invalid or repeated sweep settings")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "genome_cfg.json", config.data)
    write_json(output / "sweep.json", dict(populations=populations, generations=generations,
               mutation_probabilities=probabilities, seeds=seeds,
               configuration_sha256=config.signature, checkpoint_semantics="one continuous GA per line"))
    from ..visualization import plot_condition_banks
    banks = balance_banks(config.data["simulation"])
    np.savez_compressed(output / "cases.npz", columns=np.array(CASE_COLUMNS), **banks)
    plot_condition_banks(output, banks)
    rows, vectors, labels = [], [], []
    for size, probability, seed in product(populations, probabilities, seeds):
        line = output / f"population_{size:03d}" / f"mutation_{probability:.8g}" / f"seed_{seed}"
        train(config, line, population=size, generations=generations[-1], seed=seed,
              workers=workers, mutation_probability=probability,
              duration_s=duration_s, checkpoints=generations)
        for entry in json.loads((line / "checkpoints.json").read_text()):
            folder = line / entry["directory"]
            row = {**entry, "directory": str(folder.relative_to(output))}
            rows.append(row)
            with np.load(folder / "final_population.npz", allow_pickle=False) as saved:
                vectors.append(saved["normalized"][int(saved["selected_index"])])
            labels.append(f"N={size},p={probability},seed={seed},E={entry['generation']}")
        write_csv(output / "summary.csv", rows)
    units = np.asarray(vectors)
    distances = np.sqrt(np.mean((units[:, None] - units[None, :])**2, axis=2))
    np.savez_compressed(output / "genome_rmse.npz", rmse=distances, labels=np.array(labels),
                        normalized=units, free_parameter_indices=config.free)
    from ..visualization import plot_sweep
    plot_sweep(output, rows)
    write_json(output / "completion.json", dict(status="completed", lines=len(populations)*len(probabilities)*len(seeds),
                                               checkpoints=len(rows)))
    return rows


def balance_main(config_path, sweep_mode=False):
    """Parse the concise A/B entry points with result paths local to the experiment."""
    config_path = Path(config_path)
    parser = argparse.ArgumentParser(description="Large-angle balance, full Afferents, dt=0.1 ms")
    parser.add_argument("--config", type=Path, default=config_path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--duration", type=float, help="Seconds; short durations are for implementation smoke tests")
    if sweep_mode:
        parser.add_argument("--populations", nargs="+", type=int)
        parser.add_argument("--generations", nargs="+", type=int)
        parser.add_argument("--probabilities", nargs="+", type=float)
        parser.add_argument("--seeds", nargs="+", type=int)
    else:
        parser.add_argument("--population", type=int)
        parser.add_argument("--generations", type=int)
        parser.add_argument("--seed", type=int)
        parser.add_argument("--mutation-probability", type=float)
    args = parser.parse_args()
    output = args.output or config_path.parent / "results" / (
        ("sweep_" if sweep_mode else "run_") + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    if sweep_mode:
        result = sweep(args.config, output, populations=args.populations, generations=args.generations,
                       probabilities=args.probabilities, seeds=args.seeds, workers=args.workers,
                       duration_s=args.duration)
        print(json.dumps(dict(output=str(output), checkpoints=len(result))))
    else:
        result = train(args.config, output, population=args.population, generations=args.generations,
                       seed=args.seed, workers=args.workers, mutation_probability=args.mutation_probability,
                       duration_s=args.duration)
        print(json.dumps(dict(output=str(output), **result), indent=2))
