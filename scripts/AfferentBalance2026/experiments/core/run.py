"""Population evaluation, GA execution, checkpoints and parameter sweeps.

The core has no current-law model. External inputs, when explicitly supplied to
the evaluator, are ordinary arrays. Balance training always uses zero currents.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
from itertools import product
import json
import multiprocessing as mp
import shutil
from pathlib import Path
import zipfile

import numpy as np

from .evolution import Schedule, optimize
from .fitness import fitness
from .genome import (GenomeConfig, SCHEMA, NAMES, file_hash, write_json,
                     FEEDBACK_INDICES, MUSCLE_INDICES, OUTPUT_INDICES)
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
    path = Path(path)
    temporary = path.with_suffix('.tmp.csv')
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


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
    """Compatibility alias for the local argument translator."""
    from scripts.AfferentBalance2026.run import legacy_main
    return legacy_main(Path(config_path).parent.name[0], sweep=sweep_mode)


# New-run lifecycle. Legacy functions above retain their persisted format for
# reading/reproducing earlier studies; local CLI requests use the functions below.
from copy import deepcopy
from dataclasses import dataclass
import time

from . import artifacts
from .config import controller_config, content_hash, project_root, read_json


@dataclass
class ExperimentTask:
    """Numerical adapter used by the shared executor, independent of UI and files.

    A future model adapter supplies a codec, banks, metric/channel descriptions
    and ``measure``. The GA and persistence lifecycle do not assume two neurons,
    one limb or 52 parameters; those constraints belong to this current adapter.
    """
    codec: object
    banks: dict
    duration_s: float | None
    trajectory: bool = False

    @property
    def config(self):
        """Return the physical interpretation owned by the codec."""
        return artifacts.controller(self.codec)

    @property
    def metric_columns(self):
        """Distinguish trajectory error columns from equilibrium columns."""
        if self.trajectory:
            from ..D_trajectory.trajectory import METRIC_COLUMNS as columns
            return columns
        return METRIC_COLUMNS

    def measure(self, unit, bank, tau=0., references=None):
        """Evaluate one entire bank; no recording is performed in training."""
        physical, command = artifacts.decode(self.codec, unit)
        if self.trajectory:
            from ..D_trajectory.trajectory import replay, sine_reference
            cache = references if references is not None else {}
            if bank not in cache:
                cache[bank] = [sine_reference(spec) for spec in self.banks[bank]]
            return np.array([replay(physical, command, self.config.mechanics, spec,
                                    reference=ref, tau_s=tau, stride=0)['metrics']
                             for spec, ref in zip(self.banks[bank], cache[bank])])
        currents = (np.zeros((len(self.banks[bank]), 2)) if command is None else
                    self.codec.currents(command, self.banks[bank]))
        return _evaluate_task((physical, self.banks[bank], self.config.mechanics, self.duration_s, currents))


def make_task(effective, *, kind='polynomial'):
    """Construct the current model adapter without numerical integration."""
    cfg = controller_config(effective)
    codec = artifacts.make_codec(cfg, kind, effective['genome']['command'].get('degree', 3))
    if effective['experiment'] == 'D':
        from ..D_trajectory.trajectory import trajectory_banks, step_count
        settings = effective['protocol']['trajectory']
        banks = trajectory_banks(settings)
        # Explicit smoke horizons change evaluation duration, not the time step
        # or generated random commands. Their truncated specifications are saved.
        duration = settings.get('duration_s')
        if duration is not None:
            step_count(duration)
            for bank in banks.values():
                for spec in bank:
                    spec['duration_s'] = duration
        return ExperimentTask(codec, banks, None, True)
    from ..D_trajectory.trajectory import step_count
    duration = effective['protocol']['balance']['duration_s']
    step_count(duration)
    if effective['experiment'] == 'C':
        from ..C_setpoint_balance.joint_study import target_banks
        banks = target_banks(cfg, kind)
    else:
        banks = balance_banks(cfg.data['simulation'])
    return ExperimentTask(codec, banks, duration)


_EVALUATION_CONTEXT = {}


def _initialize_evaluation(task):
    """Cache the immutable model and trajectory references once per spawned worker."""
    _EVALUATION_CONTEXT.clear()
    _EVALUATION_CONTEXT.update(task=task, references={})


def _measure_candidate(request):
    """Process-pool entry point; the parent owns all GA randomness."""
    unit, bank, tau = request
    return _EVALUATION_CONTEXT['task'].measure(unit, bank, tau, _EVALUATION_CONTEXT['references'])


class EvaluationPool:
    """One deterministic population pool shared by all new experiment procedures."""

    def __init__(self, task, workers=1):
        """Store a model adapter and keep per-bank accounting outside hot kernels."""
        self.task, self.workers, self.pool = task, workers, None
        self.references, self.counts, self.seconds = {}, {}, {}

    def __enter__(self):
        """Start one spawn pool; sweep never nests these pools."""
        if self.workers > 1:
            self.pool = ProcessPoolExecutor(self.workers, mp_context=mp.get_context('spawn'),
                                           initializer=_initialize_evaluation, initargs=(self.task,))
        return self

    def __exit__(self, *args):
        """Release workers on success, failure or interruption."""
        if self.pool:
            self.pool.shutdown(wait=True, cancel_futures=True)

    def evaluate(self, population, bank, tau=0.):
        """Preserve candidate/episode order and count every actual evaluation."""
        started = time.perf_counter()
        requests = [(unit, bank, tau) for unit in population]
        values = (self.pool.map(_measure_candidate, requests) if self.pool else
                  (self.task.measure(unit, bank, tau, self.references) for unit in population))
        metrics = np.array(list(values))
        self.counts[bank] = self.counts.get(bank, 0)+len(population)*len(self.task.banks[bank])
        self.seconds[bank] = self.seconds.get(bank, 0.)+time.perf_counter()-started
        return np.array([fitness(values) for values in metrics]), metrics


def _initial_candidates(effective, task):
    """Check source domains and return one center plus explicitly declared seeds."""
    codec, cfg = task.codec, task.config
    command = None if isinstance(codec, GenomeConfig) else np.zeros((2, getattr(codec, 'width', 4)))
    center = artifacts.encode(codec, cfg.initial_physical, command)
    initial = effective['initialization']
    if initial['solution']:
        center = artifacts.transfer(artifacts.load_solution(initial['solution']), codec)
    extras = [artifacts.transfer(artifacts.load_solution(path), codec) for path in initial['extra_solutions']]
    sources = {path:artifacts.transfer(artifacts.load_solution(path), codec) for path in initial['sources']}
    if effective['search']['polarity_seeds'] and effective['search']['method'] != 'command_only':
        p, c = artifacts.decode(codec, center)
        # The operation is genetic; it does not modify the sign of physical torque.
        extras = [artifacts.encode(codec, variant, c) for variant in polarity_candidates(p, cfg)]+extras
    if len(extras) >= effective['search']['population']:
        raise ValueError('initialization.extra_solutions: all seeds must fit after the central individual')
    return center, extras, sources


def polarity_candidates(physical, config):
    """Return the four historical genetic polarity alternatives in original order."""
    values = []
    for operation in ('negate_feedback', 'swap_feedback', 'swap_output', 'swap_muscles_output'):
        p = physical.copy()
        if operation == 'negate_feedback':
            p[FEEDBACK_INDICES] *= -1
        elif operation == 'swap_feedback':
            p[FEEDBACK_INDICES] = physical[FEEDBACK_INDICES].reshape(2, 2, 3)[:, ::-1, :].reshape(2, 6)
        else:
            p[OUTPUT_INDICES] = physical[OUTPUT_INDICES][::-1]
            if operation == 'swap_muscles_output':
                p[MUSCLE_INDICES] = physical[MUSCLE_INDICES][::-1]
        try:
            config.encode(p)
        except ValueError:
            continue
        values.append(p)
    return values


def stage_checkpoints(effective, generations, policy):
    """Return validation generations without changing a line's cooling horizon."""
    search = effective['search']
    checkpoints = {generations, *[g for g in search['checkpoints'] if g <= generations]}
    if policy == 'initial_checkpoint_validation_v1':
        checkpoints |= {1, *range(search['validation_every'], generations+1, search['validation_every'])}
    elif policy == 'checkpoint_validation_v1' and effective['experiment'] == 'C':
        checkpoints.add(1)
    return sorted(checkpoints)


def table_initialization(task, table, center, extras):
    """Convert explicit polynomial starts to table currents, checking their own bound."""
    targets = np.array([[np.deg2rad(t), 0, 0, -1, 0] for t in table.codec.targets])
    def convert(unit):
        physical, command = artifacts.decode(task.codec, unit)
        return table.codec.encode(physical, task.codec.currents(command, targets).T)
    return convert(center), [convert(unit) for unit in extras]


def inspect_request(effective, *, check_output=True):
    """Resolve banks, preparation and per-stage budgets before creating any output."""
    task = make_task(effective)
    action, search = effective['action'], effective['search']
    preparation = None
    if any(effective['initialization']['prepare'].values()):
        preparation = artifacts.prepare_solution(effective, dry_run=True)
    elif action != 'prepare':
        center, extras, _ = _initial_candidates(effective, task)
        if search['method'] == 'staged':
            table_initialization(task, make_task(effective, kind='table'), center, extras)
    output = Path(effective['storage']['output'])
    if check_output and output.exists():
        raise ValueError(f'storage.output already exists: {output}; choose a new directory')
    policy = effective['protocol']['selection']
    stages = [(search['method'], search['generations'], task, policy)]
    if search['method'] == 'staged':
        stages = [('table', search['table_generations'], make_task(effective, kind='table'), policy),
                  ('polynomial', search['generations']-search['table_generations'], task, policy)]
    elif search['method'] == 'compare':
        stages = [('frozen', 0, task, policy),
                  ('command_only', search['generations'], task, 'initial_checkpoint_validation_v1'),
                  ('joint', search['generations'], task, 'initial_checkpoint_validation_v1')]
    budgets = []
    for name, generations, stage, selection in stages:
        counts = {key:len(bank) for key, bank in stage.banks.items()}
        steps = {key:(sum(round(spec['duration_s']*10000) for spec in bank) if stage.trajectory else
                      round(stage.duration_s*10000)*len(bank)) for key, bank in stage.banks.items()}
        checkpoints = stage_checkpoints(effective, generations, selection) if generations else []
        initial = int(selection == 'initial_checkpoint_validation_v1')
        validation = search['population']*len(checkpoints)+initial
        if name == 'frozen':
            validation = len(effective['initialization']['sources'])*len(effective['protocol']['trajectory']['filter_times_s'])
        budgets.append(dict(name=name, generations=generations, banks=counts, checkpoints=checkpoints,
                            generation_zero_eligible=bool(initial or name == 'frozen'),
                            training_candidates=search['population']*generations,
                            training_episodes=search['population']*generations*counts['training'],
                            training_steps=search['population']*generations*steps['training'],
                            validation_candidates=validation, validation_episodes=validation*counts['validation'],
                            validation_steps=validation*steps['validation'], policy=selection))
    cfg = task.config
    initial = effective['initialization']
    sources = ([initial['solution']] if initial['solution'] else [])+initial['sources']+initial['extra_solutions']
    test_solutions = 3 if search['method'] == 'compare' else 1
    description = dict(experiment=effective['experiment'], method=search['method'], model=effective['model'],
        physical_fields=len(getattr(task.codec, 'names', NAMES)), independent_coordinates=task.codec.size,
        fixed_fields=[f['name'] for f in cfg.data['parameters'] if not f['mutable']],
        fixed_values={f['name']:f['value'] for f in cfg.data['parameters'] if not f['mutable']},
        tied=cfg.mirror_symmetry, banks={k:len(v) for k,v in task.banks.items()}, stages=budgets,
        generations_total=sum(b['generations'] for b in budgets), population=search['population'],
        training_candidate_evaluations=sum(b['training_candidates'] for b in budgets),
        training_episode_evaluations=sum(b['training_episodes'] for b in budgets),
        training_step_upper_bound=sum(b['training_steps'] for b in budgets),
        validation_episode_evaluations=sum(b['validation_episodes'] for b in budgets),
        test_episode_evaluations=test_solutions*len(task.banks['test']) if effective['protocol']['evaluate_test'] else 0,
        selection=policy, test_used_for_selection=False,
        sweep_lines=int(np.prod([len(v) for v in effective['sweep']['axes'].values()])) if effective['sweep']['axes'] else 1,
        checkpoints=search['checkpoints'], search=search,
        compatibility_messages=effective['provenance']['compatibility_messages'],
        current_filter_time_s=initial['current_tau_s'], current_domain=effective['genome']['command'],
        workers=effective['runtime']['workers'], output=str(output),
        recording_stride=effective['storage']['recording_stride'], save_generations=effective['storage']['save_generations'],
        sources=[artifacts.load_solution(p).describe() for p in sources], preparation=preparation,
        changed_fields={name:cfg.data['parameters'][NAMES.index(name)] for name in effective['genome']['overrides']})
    return description


def _save_banks(folder, task):
    """Save actual generated cases, not only random seeds or counts."""
    folder.mkdir(parents=True, exist_ok=True)
    if task.trajectory:
        write_json(folder/'specifications.json', task.banks)
    else:
        artifacts.atomic_npz(folder/'cases.npz', **task.banks, columns=CASE_COLUMNS)


def _population_arrays(task, unit, scores, metrics, generation, rng_state, effective, tau):
    """Export one common population schema with an embedded interpretation template."""
    decoded = [artifacts.decode(task.codec, row) for row in unit]
    physical = np.array([p if c is None else np.r_[p, c.ravel()] for p, c in decoded])
    template = dict(schema=artifacts.SOLUTION_SCHEMA, model=MODEL,
                    genome=artifacts.genome_payload(task.codec, unit[0], tau),
                    effective_config=effective, selection={}, provenance={})
    template['genome_sha256'] = content_hash(template['genome'])
    return dict(schema=artifacts.POPULATION_SCHEMA, normalized=unit, physical=physical,
                controller_physical=np.array([p for p, _ in decoded]),
                command_values=np.empty((len(unit), 0, 0)) if decoded[0][1] is None else np.array([c for _, c in decoded]),
                training_scores=scores, training_metrics=metrics, generation=generation,
                names=np.asarray(getattr(task.codec, 'names', NAMES)),
                metric_columns=np.asarray(task.metric_columns), configuration_sha256=task.config.signature,
                rng_state=json.dumps(rng_state), solution_template=json.dumps(template))


def execute_stage(task, evaluator, center, extras, effective, folder, *, name,
                  generations=None, tau=0., mode='joint', policy=None):
    """Run one stage with common storage and explicit task-specific selection.

    The unchanged optimize() owns all genetic operations. This observer never
    draws from its RNG; checkpoint validation cannot restart or perturb a line.
    """
    folder.mkdir(parents=True, exist_ok=False)
    for directory in ('generations', 'checkpoints'):
        (folder/directory).mkdir()
    search = effective['search']
    generations = search['generations'] if generations is None else generations
    policy = effective['protocol']['selection'] if policy is None else policy
    indices = (np.arange(task.config.size, task.codec.size) if mode == 'command_only'
               else np.arange(task.codec.size))
    include_initial = policy == 'initial_checkpoint_validation_v1'
    checkpoints = stage_checkpoints(effective, generations, policy)
    winner, validations = None, []
    storage_seconds = 0.
    artifacts.save_solution(folder/'initial_solution.json', task.codec, center, effective, tau=tau,
                            selection=dict(stage=name, generation=0, individual=0, role='initial'))

    def consider(generation, unit, arrays=None):
        """Evaluate validation only and preserve the earliest winner when scores tie."""
        nonlocal winner
        scores, metrics = evaluator.evaluate(unit, 'validation', tau)
        index = int(np.argmax(scores))
        checkpoint = folder/'checkpoints'/f'generation_{generation:04d}'
        checkpoint.mkdir(exist_ok=False)
        if arrays is None:
            artifacts.atomic_npz(checkpoint/'validation.npz', normalized=unit, validation_scores=scores,
                                 validation_metrics=metrics, metric_columns=task.metric_columns, generation=generation)
        else:
            artifacts.atomic_npz(checkpoint/'population.npz', **arrays,
                                 validation_scores=scores, validation_metrics=metrics, selected_index=index)
        selected = dict(stage=name, generation=generation, individual=index,
                        validation_fitness=float(scores[index]), policy=policy, test_used_for_selection=False)
        artifacts.save_solution(checkpoint/'solution.json', task.codec, unit[index], effective, tau=tau, selection=selected)
        if (winner is None or (policy == 'final_validation_v1' and generation == generations)
                or (policy != 'final_validation_v1' and scores[index] > winner['selection']['validation_fitness'])):
            winner = dict(unit=unit[index].copy(), metrics=metrics[index].copy(), selection=selected)
        validations.append(dict(generation=generation, best_validation_fitness=float(scores[index]),
                                selected_generation=winner['selection']['generation'],
                                selected_fitness=winner['selection']['validation_fitness']))
        write_csv(folder/'validation_history.csv', validations)
        return scores, metrics

    if include_initial:
        consider(0, center[None])

    def expand(population):
        """Restore coordinates outside the command-only search subspace."""
        full = np.tile(center, (len(population), 1))
        full[:, indices] = population
        return full

    def evaluate(population):
        """Evaluate training without trajectories, file I/O or test access."""
        return evaluator.evaluate(expand(population), 'training', tau)

    def observe(generation, population, scores, metrics, history, rng_state):
        """Save every evaluated generation before reproduction and validate checkpoints."""
        nonlocal storage_seconds
        started = time.perf_counter()
        full = expand(population)
        arrays = _population_arrays(task, full, scores, metrics, generation, rng_state, effective, tau)
        if effective['storage']['save_generations']:
            artifacts.atomic_npz(folder/'generations'/f'generation_{generation:04d}.npz', **arrays)
        write_csv(folder/'history.csv', history)
        storage_seconds += time.perf_counter()-started
        if generation in checkpoints:
            val, vm = consider(generation, full, arrays)
            if generation == generations:
                artifacts.atomic_npz(folder/'final_population.npz', **arrays, validation_scores=val, validation_metrics=vm)

    candidates = np.array([row[indices] for row in extras]) if extras else None
    if effective['initialization']['strategy'] == 'local_d':
        from .evolution import reflect
        rng = np.random.default_rng(search['seed']+effective['initialization']['candidate_seed_offset'])
        candidates = reflect(center[indices]+rng.normal(0, effective['initialization']['local_sigma'],
                             (search['population']-1, len(indices))))
        if extras:
            candidates[:len(extras)] = np.asarray(extras)[:, indices]
    schedule = Schedule(search['sigma_start'], search['sigma_end'],
                        search['mutation_probability'], search['mutation_probability'])
    optimize(center[indices], evaluate, size=search['population'], generations=generations,
             seed=search['seed'], schedule=schedule, on_state=observe, initial_candidates=candidates)
    artifacts.save_solution(folder/'solution.json', task.codec, winner['unit'], effective,
                            tau=tau, selection=winner['selection'])
    write_json(folder/'selection.json', winner['selection'])
    write_json(folder/'timing.json', dict(storage_s=storage_seconds, evaluation_s=evaluator.seconds))
    return winner


def _record_metrics(folder, label, task, unit, evaluator, tau, selection, *, measured=None):
    """Evaluate a frozen solution and export explicitly named metric columns."""
    if measured is None:
        scores, metrics = evaluator.evaluate(unit[None], label, tau)
    else:
        scores = np.array([selection['validation_fitness']])
        metrics = np.asarray(measured)[None]
    folder.mkdir(exist_ok=True, parents=True)
    artifacts.atomic_npz(folder/f'{label}.npz', scores=scores, metrics=metrics[0],
                         metric_columns=task.metric_columns, solution_selection=json.dumps(selection))
    return dict(fitness=float(scores[0]), cases=len(metrics[0]), successes=int(metrics[0, :, 5].sum()))


def _write_traces(folder, task, unit, effective, tau, solution_hash, *, test_bank=False):
    """Replay with explicit recording stride; retain failure, command and reference data."""
    folder.mkdir(parents=True, exist_ok=False)
    p, command = artifacts.decode(task.codec, unit)
    stride = effective['storage']['recording_stride']
    specs = effective['protocol']['replay']
    if task.trajectory:
        from ..D_trajectory.trajectory import replay
        cases = [{**case, 'initial_angle_deg':angle, 'initial_speed_rad_s':speed,
                  'duration_s':specs['duration_s'] or case['duration_s']}
                 for case in task.banks['test']
                 for angle in (specs['angles_deg'] or [case['initial_angle_deg']])
                 for speed in (specs['speeds_rad_s'] or [case['initial_speed_rad_s']])]
    else:
        cases = np.concatenate([case_grid(specs['angles_deg'], specs['speeds_rad_s'], target)
                                for target in specs['targets_deg']])
    records = []
    for index, case in enumerate(cases):
        if task.trajectory:
            result = replay(p, command, task.config.mechanics, case, tau_s=tau, stride=stride)
            values = result
            expected_s = case['duration_s']
        else:
            duration = specs['duration_s'] if specs['duration_s'] is not None else task.duration_s
            currents = np.zeros(2) if command is None else task.codec.currents(command, case[None])[0]
            metrics, trace = simulate(p, case, task.config.mechanics, currents=currents, duration_s=duration, stride=stride)
            failure = 2 if metrics[0] < 0 else 1 if metrics[1] < duration-1e-10 or abs(trace[-1, 1]) > np.pi/2 else 0
            values = dict(trace=trace, metrics=metrics, currents=currents, case=case, failure=failure)
            expected_s = duration
        trace = values['trace']
        file = folder/f'case_{index:04d}.npz'
        artifacts.atomic_npz(file, **values, columns=TRACE_COLUMNS, metric_columns=task.metric_columns,
                             units=['s', 'rad', 'rad/s', 'N', 'N', 'mV', 'mV']+
                                   ['model units']*10+['1', '1', 'binary', 'binary']+['model units']*4+['N m'],
                             dt_ms=.1, recording_stride=stride, recording_interval_ms=.1*stride,
                             expected_duration_s=expected_s, actual_duration_s=float(values['metrics'][1]),
                             recorded_duration_s=float(trace[-1, 0]), sample_count=len(trace),
                             solution_sha256=solution_hash, case_json=json.dumps(case if task.trajectory else case.tolist()),
                             current_tau_s=tau, physical=p,
                             command_columns=['applied_flex', 'applied_ext', 'desired_flex', 'desired_ext'],
                             reference_columns=['target_rad', 'target_velocity_rad_s'],
                             termination=['completed', 'angle_limit', 'nonfinite'][int(values['failure'])],
                             model=MODEL,
                             command_coefficients=np.empty((0, 0)) if command is None else command,
                             mechanics=task.config.mechanics)
        records.append(dict(file=file.name, samples=len(trace), duration_s=float(trace[-1, 0]), failure=int(values['failure'])))
    write_json(folder/'index.json', records)
    return len(records)


def execute(effective):
    """Train or fine-tune A/B/C/D with one validated and auditable result lifecycle."""
    started_at = time.perf_counter()
    inspection = inspect_request(effective)
    effective = deepcopy(effective)
    effective['inspection'] = inspection
    task = make_task(effective)
    center, extras, sources = _initial_candidates(effective, task)
    output = Path(effective['storage']['output'])
    output.mkdir(parents=True, exist_ok=False)
    for name in ('inputs', 'banks', 'stages', 'metrics', 'figures'):
        (output/name).mkdir()
    write_json(output/'effective_config.json', effective)
    manifest = dict(schema=artifacts.RUN_SCHEMA, status='running', model=MODEL, action=effective['action'],
                    experiment=effective['experiment'], protocol=effective['protocol'], started_utc=datetime.now(timezone.utc).isoformat(),
                    versions=artifacts.environment_versions(), effective_config_sha256=content_hash(effective),
                    selection_policy=effective['protocol']['selection'], inputs=[], artifacts={})
    write_json(output/'manifest.json', manifest)
    try:
        manifest['source_sha256'] = artifacts.source_snapshot(output)
        initial = effective['initialization']
        paths = ([initial['solution']] if initial['solution'] else [])+initial['extra_solutions']+initial['sources']
        for index, path in enumerate(paths):
            source = artifacts.load_solution(path)
            target = output/'inputs'/f'{index:03d}_{Path(path).name}'
            shutil.copy2(path, target)
            portable = target.with_name(target.stem+'.portable.json')
            artifacts.save_solution(portable, source.codec, source.unit, source.effective_config, tau=source.current_tau_s,
                                    selection=source.metadata.get('selection', {}), provenance={'source_sha256':source.sha256})
            manifest['inputs'].append(dict(original=str(path), file=str(target.relative_to(output)), sha256=file_hash(path),
                                           resolved_solution=str(source.path), resolved_solution_sha256=source.sha256,
                                           portable=str(portable.relative_to(output))))
        _save_banks(output/'banks', task)
        tau = initial['current_tau_s']
        method = effective['search']['method']
        counts, timings = {}, {}
        if method == 'staged':
            from ..C_setpoint_balance.joint_study import fit_table_polynomial
            table = make_task(effective, kind='table')
            table_center, table_extras = table_initialization(task, table, center, extras)
            _save_banks(output/'banks/table', table)
            with EvaluationPool(table, effective['runtime']['workers']) as evaluator:
                table_winner = execute_stage(table, evaluator, table_center, table_extras, effective,
                                             output/'stages/table', name='table', generations=effective['search']['table_generations'])
                counts['table'], timings['table'] = evaluator.counts, evaluator.seconds
            p, currents = artifacts.decode(table.codec, table_winner['unit'])
            coefficients, fit = fit_table_polynomial(table.codec.targets, currents, task.codec.degree, task.codec.bound)
            write_json(output/'least_squares.json', dict(**fit, currents=currents.tolist(), coefficients=coefficients.tolist()))
            center, extras = task.codec.encode(p, coefficients), []
        with EvaluationPool(task, effective['runtime']['workers']) as evaluator:
            if method == 'compare':
                candidates, screen = [], []
                for filter_time in effective['protocol']['trajectory']['filter_times_s']:
                    labels, population = list(sources), np.array(list(sources.values()))
                    scores, metrics = evaluator.evaluate(population, 'validation', filter_time)
                    artifacts.atomic_npz(output/'metrics'/f'sources_tau_{filter_time:g}.npz', normalized=population,
                                         scores=scores, metrics=metrics, sources=labels, metric_columns=task.metric_columns)
                    for label, unit, score, measured in zip(labels, population, scores, metrics):
                        candidates.append((float(score), label, filter_time, unit.copy(), measured.copy()))
                score, label, tau, center, measured = max(candidates, key=lambda row: row[0])
                frozen = dict(unit=center, metrics=measured, selection=dict(stage='frozen', generation=0, individual=0,
                               validation_fitness=score, source=label, policy='d_comparison_validation_v1', test_used_for_selection=False))
                folder = output/'stages/frozen';folder.mkdir()
                artifacts.save_solution(folder/'solution.json', task.codec, center, effective, tau=tau, selection=frozen['selection'])
                solutions = {'frozen': frozen}
                for mode in ('command_only', 'joint'):
                    solutions[mode] = execute_stage(task, evaluator, center, [], effective, output/'stages'/mode,
                                                    name=mode, tau=tau, mode=mode, policy='initial_checkpoint_validation_v1')
                winner = max(solutions.values(), key=lambda item: item['selection']['validation_fitness'])
            else:
                stage = 'polynomial' if effective['experiment'] == 'C' else 'training'
                generations = effective['search']['generations']-(effective['search']['table_generations'] if method == 'staged' else 0)
                winner = execute_stage(task, evaluator, center, extras, effective, output/'stages'/stage,
                                       name=stage, generations=generations, tau=tau, mode=method)
                solutions = {stage: winner}
            selection = dict(winner['selection'])
            if method == 'compare':
                selection.update(stage_policy=selection['policy'], policy='d_comparison_validation_v1', current_tau_s=tau)
            artifacts.save_solution(output/'solution.json', task.codec, winner['unit'], effective, tau=tau, selection=selection)
            solution_hash = file_hash(output/'solution.json')
            write_json(output/'selection.json', {**selection, 'solution_sha256': solution_hash})
            _record_metrics(output/'metrics', 'validation', task, winner['unit'], evaluator, tau, selection,
                            measured=winner['metrics'])
            if effective['protocol']['evaluate_test']:
                manifest['test'] = _record_metrics(output/'metrics', 'test', task, winner['unit'], evaluator, tau, selection)
                if method == 'compare':
                    for name, item in solutions.items():
                        if item is winner:
                            shutil.copy2(output/'metrics/test.npz', output/'stages'/name/'test.npz')
                        else:
                            _record_metrics(output/'stages'/name, 'test', task, item['unit'], evaluator, tau, item['selection'])
            counts['main'], timings['main'] = evaluator.counts, evaluator.seconds
        if effective['storage']['replay']:
            manifest['replays'] = _write_traces(output/'traces', task, winner['unit'], effective, tau, solution_hash)
        if effective['storage']['figures']:
            from ..visualization import plot_run_artifacts
            started = time.perf_counter()
            plot_run_artifacts(output)
            timings['plotting'] = time.perf_counter()-started
        if artifacts.source_snapshot() != manifest['source_sha256']:
            raise RuntimeError('Calculation sources changed during the run; result is not certified')
        manifest.update(status='completed', completed_utc=datetime.now(timezone.utc).isoformat(),
                        elapsed_s=time.perf_counter()-started_at, selection=selection, solution_sha256=solution_hash,
                        evaluation_counts=counts, evaluation_seconds=timings,
                        artifacts=dict(solution='solution.json', selection='selection.json', configuration='effective_config.json'))
    except BaseException as error:
        manifest.update(status='failed', error=dict(type=type(error).__name__, message=str(error)))
        write_json(output/'manifest.json', manifest)
        raise
    write_json(output/'manifest.json', manifest)
    return dict(status='completed', output=str(output), solution=str(output/'solution.json'), selection=selection)


def replay_solution(effective):
    """Replay an explicit solution with its own physical domain and requested banks."""
    initial = effective['initialization']
    if not initial['solution']:
        raise ValueError('replay requires --solution')
    solution = artifacts.load_solution(initial['solution'])
    if effective['experiment'] != solution.config.data['experiment'][0]:
        raise ValueError('replay experiment differs from solution; use prepare for task conversion')
    if not np.array_equal(controller_config(effective).mechanics, solution.config.mechanics):
        raise ValueError('replay cannot change saved mechanics; use an explicitly prepared new experiment')
    effective = deepcopy(effective)
    effective['genome'].update(parameters=deepcopy(solution.config.data['parameters']), overrides={},
                              mirror_symmetry=solution.config.mirror_symmetry)
    if solution.command is not None:
        effective['genome']['command']['degree'] = solution.degree
        key = 'table_current_bound_nA' if solution.kind == 'table' else 'coefficient_bound_nA'
        effective['genome']['command'][key] = solution.codec.bound
    effective['controller'] = controller_config(effective).data
    task = make_task(effective, kind=solution.kind)
    unit = artifacts.transfer(solution, task.codec)
    output = Path(effective['storage']['output'])
    if output.exists():
        raise ValueError(f'storage.output already exists: {output}')
    output.mkdir(parents=True, exist_ok=False)
    (output/'inputs').mkdir()
    shutil.copy2(solution.path, output/'inputs'/solution.path.name)
    write_json(output/'effective_config.json', effective)
    manifest = dict(schema=artifacts.RUN_SCHEMA, action='replay', status='running', model=MODEL,
                    protocol=effective['protocol'], source_solution_sha256=solution.sha256,
                    versions=artifacts.environment_versions())
    write_json(output/'manifest.json', manifest)
    try:
        manifest['source_sha256'] = artifacts.source_snapshot(output)
        _save_banks(output/'banks', task)
        artifacts.save_solution(output/'solution.json', task.codec, unit, effective, tau=solution.current_tau_s,
                                selection=solution.metadata.get('selection', {}),
                                provenance={'replayed_source_sha256':solution.sha256})
        sha = file_hash(output/'solution.json')
        manifest['replays'] = _write_traces(output/'traces', task, unit, effective, solution.current_tau_s, sha)
        with EvaluationPool(task, effective['runtime']['workers']) as evaluator:
            for bank in ('validation', 'test'):
                if bank == 'validation' or effective['protocol']['evaluate_test']:
                    _record_metrics(output/'metrics', bank, task, unit, evaluator,
                                    solution.current_tau_s, solution.metadata.get('selection', {}))
            manifest['evaluation_counts'] = evaluator.counts
        if effective['storage']['figures']:
            from ..visualization import plot_run_artifacts
            plot_run_artifacts(output)
        manifest.update(status='completed', solution_sha256=sha)
    except BaseException as error:
        manifest.update(status='failed', error={'type':type(error).__name__, 'message':str(error)})
        write_json(output/'manifest.json', manifest)
        raise
    write_json(output/'manifest.json', manifest)
    return dict(status='completed', output=str(output), solution=str(output/'solution.json'))


def sweep_configurations(effective):
    """Resolve independent lines; generations on an axis are separate GA horizons."""
    from .config import _validate_effective
    axes = effective['sweep']['axes']
    lines = []
    for number, combination in enumerate(product(*axes.values())):
        line = deepcopy(effective)
        line['sweep']['axes'] = {}
        line['action'] = 'train'
        for name, value in zip(axes, combination):
            block, key = ('genome', 'command') if name == 'degree' else ('search', name)
            if name == 'degree':
                line[block][key]['degree'] = value
            else:
                line[block][key] = value
            line['provenance']['sources'][f'genome.command.degree' if name == 'degree' else f'search.{name}'] = 'sweep.axis'
        if line['search']['method'] == 'staged' and 'generations' in axes:
            if effective['provenance']['sources'].get('search.table_generations', '').startswith('preset:'):
                line['search']['table_generations'] = line['search']['generations']//2
        line['storage']['output'] = str(Path(effective['storage']['output'])/f'line_{number:04d}')
        _validate_effective(line)
        if line['experiment'] in 'AB':
            line['protocol']['selection'] = 'checkpoint_validation_v1'
        line['action'] = 'sweep_line'
        line['controller'] = controller_config(line).data
        inspect_request(line)
        lines.append(line)
    return lines


def execute_sweep(effective):
    """Execute serial independent lines, each owning one persistent worker pool."""
    lines = sweep_configurations(effective)
    output = Path(effective['storage']['output'])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'effective_config.json', effective)
    manifest = dict(schema=artifacts.RUN_SCHEMA, action='sweep', status='running', lines=[])
    write_json(output/'manifest.json', manifest)
    try:
        for line in lines:
            result = execute(line)
            manifest['lines'].append(result)
            write_json(output/'manifest.json', manifest)
        # Validation selects the sweep winner; test is descriptive only.
        winner = max(manifest['lines'], key=lambda row: row['selection']['validation_fitness'])
        source = Path(winner['solution'])
        shutil.copy2(source, output/'solution.json')
        write_json(output/'selection.json', {**winner['selection'], 'line':str(source.parent.relative_to(output)),
                   'solution_sha256':file_hash(source), 'test_used_for_selection':False})
        manifest.update(status='completed', solution_sha256=file_hash(source))
        write_csv(output/'sweep.csv', [dict(line=Path(row['output']).name, **row['selection']) for row in manifest['lines']])
    except BaseException as error:
        manifest.update(status='failed', error=dict(type=type(error).__name__, message=str(error)))
        write_json(output/'manifest.json', manifest)
        raise
    write_json(output/'manifest.json', manifest)
    return dict(status='completed', output=str(output), solution=str(output/'solution.json'), lines=len(lines))
