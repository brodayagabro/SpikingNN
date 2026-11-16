"""One real-coded GA with elitism, tournament selection and cosine annealing."""
import argparse
from copy import deepcopy
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time
import zipfile
from importlib.metadata import version
from itertools import product

import numba
import numpy as np

from genome import BalanceGenome, GenomeConfig, JointGenome, file_hash, load_solution, write_json
from model import (METRIC_COLUMNS, MODEL, PARAMETERS, PARAMETER_NAMES, evaluate_population,
                   plot_replay, replay_genome, validate_duration, validate_fitness_window,
                   DEFAULT_FITNESS_WINDOW_S)

HERE = Path(__file__).resolve().parent
SEARCH_KEYS = {'population', 'generations', 'seed', 'mutation_probability',
               'sigma_start', 'sigma_end', 'validation_every', 'workers', 'method'}


def reflect(unit):
    """Reflect arbitrary real coordinates into [0, 1] with period two."""
    folded = np.mod(unit, 2.)
    return np.where(folded > 1, 2-folded, folded)


@dataclass(frozen=True)
class Schedule:
    """Store immutable endpoints for cosine mutation annealing.

    sigma_start/sigma_end are standard deviations in normalized gene units.
    probability_start/probability_end are independent per-gene mutation
    probabilities. Equal probability endpoints give a constant probability.
    """
    sigma_start: float = .20
    sigma_end: float = .015
    probability_start: float = .60
    probability_end: float = .10

    def __post_init__(self):
        """Validate finite sigma endpoints and mutation probabilities."""
        if not np.isfinite(list(asdict(self).values())).all() or not 0 < self.sigma_end <= self.sigma_start:
            raise ValueError("Expected finite 0 < sigma_end <= sigma_start")
        if not 0 <= self.probability_end <= self.probability_start <= 1:
            raise ValueError("Expected 0 <= probability_end <= probability_start <= 1")

    def at(self, generation, generations):
        """Return (sigma, probability) at a one-based generation.

        Use the same total horizon when resuming training to preserve the schedule.
        """
        if not 1 <= generation <= generations:
            raise ValueError("Expected 1 <= generation <= generations")
        cooling = .5*(1+np.cos(np.pi*(generation-1)/max(1, generations-1)))
        return (self.sigma_end+(self.sigma_start-self.sigma_end)*cooling,
                self.probability_end+(self.probability_start-self.probability_end)*cooling)


def initial_population(center, size, rng):
    """Create a normalized population around center using the supplied RNG.

    Keep center as the first individual, perturb nearby individuals, and fill
    the last quarter with uniform immigrants. Require at least three members.
    """
    if size < 3:
        raise ValueError("Population must contain at least three individuals")
    population = reflect(center + rng.normal(0., .1, (size, len(center))))
    population[0] = center
    immigrants = max(1, size//4)
    population[-immigrants:] = rng.random((immigrants, len(center)))
    return population


def reproduce(population, scores, rng, sigma, probability):
    """Build one generation with elitism, tournaments, crossover and mutation.

    Preserve the best ceil(5% of N) individuals. Other children are either
    uniform immigrants or mixtures of tournament winners with Gaussian
    mutation and reflection into [0, 1]. All randomness uses rng.
    """
    size, dimensions = population.shape
    order = np.argsort(-scores, kind="stable")
    children = [population[i].copy() for i in order[:max(1, int(np.ceil(.05*size)))]]
    while len(children) < size:
        if rng.random() < .05:
            child = rng.random(dimensions)
        else:
            parents = []
            for _ in range(2):
                candidates = rng.integers(size, size=3)
                parents.append(population[candidates[np.argmax(scores[candidates])]])
            alpha = rng.uniform(-.15, 1.15, dimensions)
            child = alpha*parents[0] + (1-alpha)*parents[1]
            child += (rng.random(dimensions) < probability)*rng.normal(0., sigma, dimensions)
            child = reflect(child)
        children.append(child)
    return np.asarray(children)


def optimize(center, evaluate, *, size=30, generations=100, seed=42,
             schedule=Schedule(), on_generation=None, on_state=None,
             resume_state=None, schedule_generations=None, initial_candidates=None):
    """Run or resume the shared real-coded genetic algorithm.

    Parameters
    ----------
    center : array_like
        Initial candidate in normalized [0, 1] coordinates.
    evaluate : callable
        Map an N-by-D population to N fitness scores and N-by-K-by-9 metrics.
    size, generations, seed : int
        Population size, final generation number, and parent-process RNG seed.
    schedule : Schedule
        Mutation rule; schedule_generations optionally fixes a longer horizon.
    on_generation, on_state : callable, optional
        Observers of history and complete evaluated state. Observers must not
        mutate the population or draw from the optimizer's RNG.
    resume_state : dict, optional
        Evaluated population, scores, generation, history, and RNG state.
    initial_candidates : array_like, optional
        Additional normalized warm starts inserted after the central candidate.

    Returns
    -------
    tuple
        Final population, training scores, episode metrics, history, RNG state.
        Validation and test selection belong to the caller, not this loop.
    """
    if generations < 1 or seed < 0:
        raise ValueError("Positive generations and a nonnegative seed are required")
    rng = np.random.default_rng(seed)
    horizon = generations if schedule_generations is None else schedule_generations
    if horizon < generations:
        raise ValueError("Schedule horizon must cover all requested generations")
    if resume_state is None:
        population = initial_population(np.asarray(center), size, rng)
        if initial_candidates is not None:
            candidates = np.asarray(initial_candidates, dtype=float)
            if (candidates.ndim != 2 or candidates.shape[1] != len(center) or
                    len(candidates) >= size or not np.isfinite(candidates).all() or
                    np.any((candidates < 0) | (candidates > 1))):
                raise ValueError("Initial candidates must fit the population and normalized domain")
            population[1:1+len(candidates)] = candidates
        history, first, elapsed = [], 1, 0.
    else:
        population = np.array(resume_state["population"], copy=True)
        history = [dict(row) for row in resume_state["history"]]
        first = int(resume_state["generation"]) + 1
        if population.shape != (size, len(center)) or not 1 < first <= generations:
            raise ValueError("Incompatible population or generation in checkpoint")
        rng.bit_generator.state = resume_state["rng_state"]
        elapsed = history[-1]["elapsed_s"]
        population = reproduce(population, resume_state["scores"], rng,
                               *schedule.at(first, horizon))
    started = time.monotonic() - elapsed
    for generation in range(first, generations+1):
        scores, metrics = evaluate(population)
        leader = int(np.argmax(scores))
        sigma, probability = schedule.at(generation, horizon)
        row = dict(generation=generation, evaluations=generation*size,
                   best_fitness=float(scores[leader]), median_fitness=float(np.median(scores)),
                   best_success=float(metrics[leader, :, 5].mean()),
                   diversity=float(population.std(axis=0).mean()), mutation_sigma=sigma,
                   mutation_probability=probability, elapsed_s=time.monotonic()-started)
        history.append(row)
        if on_generation:
            on_generation(history)
        if on_state:
            on_state(generation, population, scores, metrics, history, rng.bit_generator.state)
        if generation < generations:
            population = reproduce(population, scores, rng, *schedule.at(generation+1, horizon))
    return population, scores, metrics, history, rng.bit_generator.state


# Experiment workflow. These functions keep file handling and CLI decisions out
# of the numerical model and keep validation/test data out of GA selection.
def write_csv(path, rows):
    """Replace a small human-readable history/summary with its current rows."""
    if not rows:
        return
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_search(settings):
    """Reject misspelled options before creating output or compiling a kernel."""
    unknown = set(settings)-SEARCH_KEYS
    missing = SEARCH_KEYS-{'method'}-set(settings)
    if unknown or missing:
        raise ValueError(f'Training options: unknown={sorted(unknown)}, missing={sorted(missing)}')
    for key, minimum in (('population', 3), ('generations', 1), ('seed', 0),
                         ('validation_every', 1), ('workers', 1)):
        if type(settings[key]) is not int or settings[key] < minimum:
            raise ValueError(f'{key} must be an integer >= {minimum}')
    if settings['workers'] > numba.config.NUMBA_NUM_THREADS:
        raise ValueError(f'workers exceeds the Numba thread limit {numba.config.NUMBA_NUM_THREADS}')
    Schedule(settings['sigma_start'], settings['sigma_end'],
             settings['mutation_probability'], settings['mutation_probability'])
    if settings.get('method', 'joint') not in ('joint', 'staged'):
        raise ValueError('method must be joint or staged')
    if settings.get('method') == 'staged' and settings['generations'] < 2:
        raise ValueError('Staged training needs at least two generations')


def make_genome(config, kind=None):
    """Derive dimensions from fixed fields, mirror ties and the command degree."""
    cfg = config if isinstance(config, GenomeConfig) else GenomeConfig(config)
    if cfg.data['experiment'] == 'C_setpoint_balance':
        return JointGenome(cfg, kind or 'polynomial', cfg.data['setpoint']['degree'])
    return BalanceGenome(cfg)


def profile_inputs(folder, profile):
    """Apply named domain overrides to generic parameters, without loading a solution."""
    folder = Path(folder)
    config = json.loads((folder/'genome_cfg.json').read_text())
    settings = json.loads((folder/'training.json').read_text())
    profiles = json.loads((folder/'profiles.json').read_text())
    if profile not in profiles:
        raise ValueError(f'Unknown profile {profile!r}; choose {list(profiles)}')
    patch = profiles[profile]
    fields = {field['name']:field for field in config['parameters']}
    for name, changes in patch.get('parameters', {}).items():
        if name not in fields or set(changes)-{'value','minimum','maximum','mutable','scale'}:
            raise ValueError(f'Invalid profile field {name}')
        fields[name].update(changes)
    config['mirror_symmetry'] = patch.get('mirror_symmetry', False)
    settings.update(patch.get('training', {}))
    make_genome(config)
    validate_search(settings)
    return config, settings


def seed_coordinates(genome, saved=None):
    """Transfer a saved physical genome explicitly, without clipping its bounds.

    A/B seeds give zero initial commands in C. Polynomial seeds can initialize
    a C table by evaluating its training targets. Increasing polynomial degree
    inserts zero coefficients; reducing it requires zero discarded terms.
    """
    if saved is None:
        p = genome.controller.initial_physical
        command = np.zeros((2, getattr(genome, 'width', 0)))
    else:
        source, unit = saved
        p, command = source.decode(unit)
        if genome.kind == 'none':
            if command.size and np.any(command):
                raise ValueError('Cannot discard nonzero command currents when transferring to A/B')
        elif source.kind == 'none':
            command = np.zeros((2, genome.width))
        elif genome.kind == 'table' and source.kind == 'polynomial':
            cases = np.zeros((len(genome.targets), 5));cases[:, 0] = np.deg2rad(genome.targets)
            command = source.currents(command, cases).T
        elif genome.kind == 'polynomial' and source.kind == 'table':
            raise ValueError('Use the staged procedure to fit table commands to a polynomial')
        elif source.kind == 'polynomial' and command.shape[1] != genome.width:
            if np.any(command[:, genome.width:]):
                raise ValueError('Cannot discard nonzero polynomial coefficients')
            changed = np.zeros((2, genome.width))
            width = min(genome.width, command.shape[1]);changed[:, :width] = command[:, :width]
            command = changed
    try:
        return genome.encode(p, command)
    except ValueError as exc:
        raise ValueError(f'Seed does not fit the requested genome configuration: {exc}. '
                         'Use compatible bounds or a compatible symmetric seed; values are not clipped.') from exc


def train_phase(genome, banks, settings, output, *, center=None):
    """Train one stage and select its validation incumbent; never evaluate tests.

    Validation includes the initial genome, generation 1, periodic checkpoints
    and the final generation. Save the final population separately from the
    selected genome. Every array retains candidate and episode order.
    """
    validate_search(settings)
    output = Path(output);output.mkdir(parents=True, exist_ok=True)
    if (output/'best_genome.json').exists():
        raise FileExistsError(f'Existing stage: {output}')
    numba.set_num_threads(settings['workers'])
    duration = genome.controller.data['simulation']['duration_s']
    center = seed_coordinates(genome) if center is None else np.asarray(center)
    genome.decode(center)
    write_json(output/'genome_cfg.json', genome.controller.data)
    np.savez_compressed(output/'cases.npz', **banks)
    initial_scores, initial_metrics = evaluate_population(genome, center[None], banks['validation'], duration)
    winner = dict(unit=center.copy(), fitness=float(initial_scores[0]),
                  metrics=initial_metrics[0].copy(), generation=0, individual=0, seed=settings['seed'])
    genome.save(output/'initial_genome.json', center, generation=0, model=MODEL)
    np.savez_compressed(output/'baseline_validation.npz', scores=initial_scores, metrics=initial_metrics)
    validation_history = []
    last_validation = {}

    def evaluate(population):
        return evaluate_population(genome, population, banks['training'], duration)

    def observe(generation, population, scores, metrics, history, rng_state):
        if generation == 1:
            np.savez_compressed(output/'initial_population.npz', normalized=population,
                                rng_state=json.dumps(rng_state))
        due = generation == 1 or generation % settings['validation_every'] == 0 or generation == settings['generations']
        if due:
            val, measured = evaluate_population(genome, population, banks['validation'], duration)
            index = int(np.argmax(val))
            if val[index] > winner['fitness']:
                winner.update(unit=population[index].copy(), fitness=float(val[index]),
                              metrics=measured[index].copy(), generation=generation, individual=index)
            validation_history.append(dict(generation=generation, best_fitness=float(val[index]),
                                           selected_fitness=winner['fitness']))
            np.savez_compressed(output/f'validation_{generation:04d}.npz', normalized=population,
                                scores=val, metrics=measured)
            last_validation.update(scores=val, metrics=measured)
            write_csv(output/'validation_history.csv', validation_history)
        write_csv(output/'history.csv', history)
        np.savez_compressed(output/'checkpoint.npz', normalized=population, scores=scores,
                            generation=generation, rng_state=json.dumps(rng_state))
        if due:
            print(f'{output.name}: generation {generation}/{settings["generations"]}; '
                  f'train={scores.max():.6f}; validation={winner["fitness"]:.6f}', flush=True)

    schedule = Schedule(settings['sigma_start'], settings['sigma_end'],
                        settings['mutation_probability'], settings['mutation_probability'])
    population, scores, measured, history, rng = optimize(
        center, evaluate, size=settings['population'], generations=settings['generations'],
        seed=settings['seed'], schedule=schedule, on_state=observe)
    physical = [np.r_[p, c.ravel()] for p, c in (genome.decode(row) for row in population)]
    np.savez_compressed(output/'final_population.npz', normalized=population, physical=np.asarray(physical),
                        training_scores=scores, training_metrics=measured,
                        validation_scores=last_validation['scores'], validation_metrics=last_validation['metrics'],
                        generation=settings['generations'], rng_state=json.dumps(rng))
    genome.save(output/'best_genome.json', winner['unit'], selection='validation only',
                generation=winner['generation'], individual=winner['individual'], seed=settings['seed'], model=MODEL)
    np.savez_compressed(output/'selected_validation.npz', scores=[winner['fitness']], metrics=winner['metrics'][None])
    write_json(output/'selection.json', {k:v for k,v in winner.items() if k not in ('unit','metrics')})
    return winner


def finish_run(genome, winner, banks, output):
    """Evaluate the frozen validation winner once on its held-out test bank."""
    duration = genome.controller.data['simulation']['duration_s']
    scores, measured = evaluate_population(genome, winner['unit'][None], banks['test'], duration)
    np.savez_compressed(output/'test_metrics.npz', scores=scores, metrics=measured[0],
                        metric_columns=np.array(METRIC_COLUMNS))
    genome.save(output/'best_genome.json', winner['unit'], selection='validation only',
                generation=winner['generation'], individual=winner['individual'], seed=winner['seed'], model=MODEL)
    m = measured[0]
    training_scores, training_metrics = evaluate_population(
        genome, winner['unit'][None], banks['training'], duration)
    def bank_summary(score, metrics):
        return dict(fitness=float(score), cases=len(metrics), success=int(metrics[:, 5].sum()),
                    mean_late_rmse_deg=float(np.degrees(metrics[:, 7]).mean()))
    summary = dict(status='completed', model=MODEL, validation_fitness=winner['fitness'],
                   duration_s=duration,
                   fitness_window_s=genome.controller.data['simulation'].get('fitness_window_s', DEFAULT_FITNESS_WINDOW_S),
                   selected_generation=winner['generation'], selected_individual=winner['individual'],
                   test_fitness=float(scores[0]), test_cases=len(m),
                   test_completed=int((m[:, 1] >= duration-1e-8).sum()), test_success=int(m[:, 5].sum()),
                   mean_late_rmse_deg=float(np.degrees(m[:, 7]).mean()),
                   free_coordinates=genome.size, test_used_for_selection=False,
                   training=bank_summary(training_scores[0], training_metrics[0]),
                   validation=bank_summary(winner['fitness'], winner['metrics']),
                   test=bank_summary(scores[0], m))
    np.savez_compressed(output/'selected_training.npz', scores=training_scores,
                        metrics=training_metrics[0], metric_columns=np.array(METRIC_COLUMNS))
    write_json(output/'result.json', summary)
    return summary


def train_joint(config, settings, output, bank_factory, saved=None):
    """Default A/B/C joint training; C may supply a staged runner instead."""
    genome = make_genome(config)
    banks = bank_factory(config, genome.kind)
    winner = train_phase(genome, banks, settings, output/'training',
                         center=seed_coordinates(genome, saved))
    return finish_run(genome, winner, banks, output)


def run_training(config, settings, output, bank_factory, *, saved=None, runner=None, smoke=False):
    """Create a new result directory and persist the complete effective inputs."""
    validate_search(settings)
    validate_duration(config['simulation']['duration_s'])
    fitness_window_s = validate_fitness_window(
        config['simulation'].get('fitness_window_s', DEFAULT_FITNESS_WINDOW_S))
    make_genome(config)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'genome_cfg.json', config)
    write_json(output/'training.json', settings)
    provenance = dict(model=MODEL, smoke_test=smoke, python=platform.python_version(),
                      numpy=np.__version__, numba=numba.__version__,
                      scipy=version('scipy'), llvmlite=version('llvmlite'),
                      platform=platform.platform(), machine=platform.machine(),
                      argv=sys.argv, initialization='configuration' if saved is None else 'explicit_warm_start',
                      archive_genomes_used=saved is not None, rng='numpy.random.PCG64',
                      configuration_sha256=file_hash(output/'genome_cfg.json'),
                      training_sha256=file_hash(output/'training.json'),
                      fitness_window_s=fitness_window_s,
                      receptor_constants=dict(zip(PARAMETER_NAMES, PARAMETERS)),
                      code_sha256={str(p.relative_to(HERE)):file_hash(p)
                                   for p in sorted([*HERE.glob('*.py'), *HERE.glob('*/experiment.py')])})
    if saved is not None:
        source, unit = saved
        write_json(output/'source_genome.json', source.payload(unit, model=MODEL))
    write_json(output/'provenance.json', provenance)
    sources = sorted([*HERE.glob('*.py'), *HERE.glob('requirements*.txt'),
                      *HERE.glob('*/experiment.py'), *HERE.glob('*/genome_cfg.json'),
                      *HERE.glob('*/training.json'), *HERE.glob('*/profiles.json')])
    with zipfile.ZipFile(output/'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            archive.write(path, str(path.relative_to(HERE)))
    print(f'Output: {output}', flush=True)
    summary = (runner or train_joint)(config, settings, output, bank_factory, saved)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def _training_inputs(args, folder, experiment):
    """Resolve paths once and distinguish a new domain from an embedded seed domain."""
    saved = load_solution(args.genome) if args.genome else None
    if args.profile and (args.config or args.genome):
        raise ValueError('--profile defines a fresh search; do not combine it with --config or --genome')
    config = (deepcopy(saved[0].controller.data) if saved and not args.config
              else json.loads(Path(args.config or folder/'genome_cfg.json').read_text()))
    if config['experiment'] != experiment:
        raise ValueError(f'Expected {experiment}; cross-task transfers require its explicit --config')
    if saved and not args.config and saved[0].kind == 'polynomial':
        config['setpoint']['degree'] = saved[0].degree
    if args.duration is not None:
        config['simulation']['duration_s'] = args.duration
    if args.fitness_window is not None:
        config['simulation']['fitness_window_s'] = args.fitness_window
    settings = json.loads(Path(args.training or folder/'training.json').read_text())
    if args.profile:
        config, profile_settings = profile_inputs(folder, args.profile)
        if not args.training:
            settings = profile_settings
        if args.duration is not None:
            config['simulation']['duration_s'] = args.duration
        if args.fitness_window is not None:
            config['simulation']['fitness_window_s'] = args.fitness_window
    if getattr(args, 'degree', None) is not None:
        config['setpoint']['degree'] = args.degree
    for key in ('population','generations','seed','workers','method',
                'mutation_probability','sigma_start','sigma_end','validation_every'):
        value = getattr(args, key, None)
        if value is not None:
            settings[key] = value
    if experiment != 'C_setpoint_balance' and settings.get('method', 'joint') != 'joint':
        raise ValueError('Staged table-to-polynomial training belongs to experiment C')
    if args.smoke:
        config['simulation'].update(duration_s=.02, training_count=4, validation_count=4, test_count=4)
        settings.update(population=3, generations=2, validation_every=1)
        print('SMOKE TEST: shortened episodes and population; these are not scientific results.', flush=True)
    validate_search(settings)
    validate_duration(config['simulation']['duration_s'])
    validate_fitness_window(config['simulation'].get('fitness_window_s', DEFAULT_FITNESS_WINDOW_S))
    return config, settings, saved


def cli(experiment, folder, bank_factory, runner=None, argv=None):
    """Small uniform command interface shared by the per-experiment entry files."""
    folder = Path(folder).resolve()
    parser = argparse.ArgumentParser(description=f'{experiment}: standalone Numba controller experiment')
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('check','train','sweep'):
        p = sub.add_parser(command, help={'check':'Validate inputs without simulation', 'train':'Train and test a validation-selected genome',
                                         'sweep':'Compare population sizes and random seeds'}[command])
        p.add_argument('--config', type=Path, help='Genome configuration; with a seed, omitted means its embedded configuration')
        p.add_argument('--profile', choices=list(json.loads((folder/'profiles.json').read_text())),
                       help='Named physical domain, initialized from generic parameters')
        p.add_argument('--training', type=Path, help='Training settings JSON (default: beside experiment.py)')
        p.add_argument('--genome', type=Path, help='Optional warm-start genome; bounds are checked, never clipped')
        p.add_argument('--population', type=int)
        p.add_argument('--generations', type=int)
        p.add_argument('--seed', type=int)
        p.add_argument('--workers', type=int, help='Numba threads, not independent GA runs')
        p.add_argument('--mutation-probability', type=float)
        p.add_argument('--sigma-start', type=float)
        p.add_argument('--sigma-end', type=float)
        p.add_argument('--validation-every', type=int)
        p.add_argument('--duration', type=float, help='Episode duration in seconds; overrides the genome configuration')
        p.add_argument('--fitness-window', type=float, help='Final accuracy window in seconds; overrides the genome configuration')
        p.add_argument('--smoke', action='store_true', help='Quick technical run: N=3, G=2, 0.02 s episodes')
        if experiment == 'C_setpoint_balance':
            p.add_argument('--method', choices=('joint','staged'))
            p.add_argument('--degree', type=int, choices=(2,3))
        if command != 'check':
            p.add_argument('--output', type=Path, help='New directory; existing output is never overwritten')
        if command == 'sweep':
            p.add_argument('--populations', type=int, nargs='+', default=[30,50])
            p.add_argument('--seeds', type=int, nargs='+', default=[42,43])
            p.add_argument('--mutation-probabilities', type=float, nargs='+')
            if experiment == 'C_setpoint_balance':
                p.add_argument('--methods', choices=('joint','staged'), nargs='+')
                p.add_argument('--degrees', type=int, choices=(2,3), nargs='+')
    p = sub.add_parser('simulate', help='Replay a saved genome and save dynamics/plots')
    p.add_argument('--genome', type=Path, default=folder/'genomes/fast.json')
    p.add_argument('--target', type=float, default=0., help='Desired angle in degrees; zero for A/B')
    p.add_argument('--angle', type=float, default=-30., help='Initial limb angle in degrees')
    p.add_argument('--speed', type=float, default=0., help='Initial speed in rad/s')
    p.add_argument('--duration', type=float, default=10., help='Episode duration in seconds')
    p.add_argument('--fitness-window', type=float, help='Final accuracy window; default: saved genome setting, or 2 seconds')
    p.add_argument('--stride', type=int, default=1, help='Record every n-th integration step; 1 resolves all spikes')
    p.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == 'simulate':
            import matplotlib
            matplotlib.use('Agg')
            genome, unit = load_solution(args.genome)
            if genome.controller.data['experiment'] != experiment:
                raise ValueError(f'This entry point expects a {experiment} genome')
            if args.stride < 1:
                raise ValueError('Use stride >= 1 for saved dynamics')
            output = args.output or Path('results')/f'{experiment}_replay_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}'
            if output.exists():
                raise FileExistsError(f'Existing output: {output}')
            result = replay_genome(genome, unit, target_deg=args.target, angle_deg=args.angle,
                                   speed_rad_s=args.speed, duration_s=args.duration, stride=args.stride,
                                   fitness_window_s=args.fitness_window)
            output.mkdir(parents=True, exist_ok=False)
            np.savez_compressed(output/'dynamics.npz', **result)
            write_json(output/'metrics.json', dict(zip(METRIC_COLUMNS, result['metrics'].tolist())))
            fig = plot_replay(result);fig.savefig(output/'dynamics.png', dpi=160)
            print(f'Saved dynamics: {output.resolve()}')
            return
        config, settings, saved = _training_inputs(args, folder, experiment)
        genome = make_genome(config)
        initial_genome = make_genome(config, 'table') if settings.get('method') == 'staged' else genome
        seed_coordinates(initial_genome, saved)
        banks = bank_factory(config, genome.kind)
        if args.command == 'check':
            print(json.dumps(dict(experiment=experiment, physical_parameters=len(genome.names),
                                  free_coordinates=genome.size, mirror=genome.controller.mirror_symmetry,
                                  episodes={name:len(bank) for name,bank in banks.items()},
                                  duration_s=config['simulation']['duration_s'],
                                  fitness_window_s=config['simulation'].get('fitness_window_s', DEFAULT_FITNESS_WINDOW_S),
                                  initialization='configuration' if saved is None else 'explicit_warm_start',
                                  training=settings), indent=2))
            return
        output = args.output or Path('results')/f'{experiment}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}'
        if args.command == 'train':
            run_training(config, settings, output, bank_factory, saved=saved, runner=runner, smoke=args.smoke)
            return
        if args.smoke:
            raise ValueError('For a small sweep use --populations 3 --generations 2; --smoke is a single-run check')
        probabilities = args.mutation_probabilities or [settings['mutation_probability']]
        methods = getattr(args, 'methods', None) or [settings.get('method', 'joint')]
        degrees = getattr(args, 'degrees', None) or [config.get('setpoint', {}).get('degree', 0)]
        axes = [args.populations, args.seeds, probabilities, methods, degrees]
        if any(len(set(axis)) != len(axis) for axis in axes):
            raise ValueError('Sweep axis values must not repeat')
        combinations = []
        for population, seed, probability, method, degree in product(*axes):
            candidate = {**settings, 'population':population, 'seed':seed,
                         'mutation_probability':probability, 'method':method}
            domain = deepcopy(config)
            if degree:
                domain['setpoint']['degree'] = degree
            validate_search(candidate);make_genome(domain)
            combinations.append((candidate, domain))
        output.mkdir(parents=True, exist_ok=False)
        write_json(output/'sweep.json', dict(selection='validation only',
                   runs=[dict(training=s, configuration=c) for s,c in combinations]))
        rows = []
        for candidate, domain in combinations:
            degree = domain.get('setpoint', {}).get('degree', 0)
            name = f'N{candidate["population"]}_p{candidate["mutation_probability"]:g}_seed{candidate["seed"]}'
            if degree:
                name += f'_{candidate["method"]}_degree{degree}'
            run = output/name
            summary = run_training(domain, candidate, run, bank_factory, saved=saved, runner=runner)
            rows.append(dict(run=run.name, population=candidate['population'], seed=candidate['seed'],
                             mutation_probability=candidate['mutation_probability'],
                             method=candidate['method'], degree=degree,
                             **{f'{bank}_{key}':value for bank in ('training','validation','test')
                                for key,value in summary[bank].items()}))
            write_csv(output/'comparison.csv', rows)
        write_json(output/'selection.json', dict(selection='maximum validation fitness; tests unused for selection',
                                                 selected=max(rows, key=lambda r:r['validation_fitness'])))
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        parser.error(str(exc))
