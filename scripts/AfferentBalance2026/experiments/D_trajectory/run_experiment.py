"""Compare frozen C controllers with command-only and joint trajectory refinement.

Run this file directly or as a module. Selection uses validation trajectories;
test trajectories are evaluated only after all three solutions have been frozen.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import zipfile

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from scripts.AfferentBalance2026.experiments.core.evolution import Schedule, optimize, reflect
from scripts.AfferentBalance2026.experiments.core.fitness import fitness
from scripts.AfferentBalance2026.experiments.core.genome import file_hash, write_json
from scripts.AfferentBalance2026.experiments.core.run import REPO, source_hashes, write_csv
from scripts.AfferentBalance2026.experiments.simulation import TRACE_COLUMNS
from scripts.AfferentBalance2026.experiments.D_trajectory.controller import TrajectoryGenome, load_solution
from scripts.AfferentBalance2026.experiments.D_trajectory.trajectory import (
    METRIC_COLUMNS, replay, sine_case, sine_reference, trajectory_banks)

HERE = Path(__file__).resolve().parent
_WORKER = {}


def initialize_worker(mechanics, banks):
    """Cache immutable reference arrays once per worker, avoiding repeated IPC."""
    _WORKER.clear()
    _WORKER.update(mechanics=np.asarray(mechanics), banks=banks, references={})


def evaluate_task(task):
    """Evaluate one physical solution on a named bank, preserving episode order."""
    physical, coefficients, bank, tau = task
    if bank not in _WORKER['references']:
        _WORKER['references'][bank] = [sine_reference(spec) for spec in _WORKER['banks'][bank]]
    return np.array([replay(physical, coefficients, _WORKER['mechanics'], spec,
                            reference=reference, tau_s=tau, stride=0)['metrics']
                     for spec, reference in zip(_WORKER['banks'][bank], _WORKER['references'][bank])])


class Evaluator:
    """Evaluate complete candidate banks with a persistent deterministic pool."""

    def __init__(self, genome, banks, workers=1):
        """Store a genome codec and banks; defer process creation until entry."""
        if not isinstance(workers, int) or workers < 1:
            raise ValueError('workers must be a positive integer')
        self.genome, self.banks, self.workers, self.pool = genome, banks, workers, None

    def __enter__(self):
        """Start spawn workers, or initialize the same serial evaluation path."""
        args = (self.genome.controller.mechanics, self.banks)
        if self.workers > 1:
            self.pool = ProcessPoolExecutor(self.workers, mp_context=mp.get_context('spawn'),
                                           initializer=initialize_worker, initargs=args)
        else:
            initialize_worker(*args)
        return self

    def __exit__(self, *args):
        """Release all workers after outstanding evaluations have completed."""
        if self.pool:
            self.pool.shutdown(wait=True, cancel_futures=True)

    def evaluate(self, population, bank, tau):
        """Return robust fitness and all per-trajectory metrics for every genome."""
        tasks = [(*self.genome.decode(unit), bank, float(tau)) for unit in population]
        metrics = np.array(list(self.pool.map(evaluate_task, tasks) if self.pool else map(evaluate_task, tasks)))
        return np.array([fitness(row) for row in metrics]), metrics


def default_sources():
    """Return all four retained C method winners, before and after the tau study."""
    base = HERE.parent/'C_setpoint_balance/results'
    return {f'{study}_{method}': base/f'joint_sweep{suffix}_20261022/method_winners/{method}/best_genome.json'
            for study, suffix in (('original', ''), ('tau_gt5', '_tau_gt5'))
            for method in ('joint', 'staged')}


def summarize(name, metrics, specs):
    """Report aggregate tracking errors and failures without dropping hard cases."""
    full = np.isclose(metrics[:, 1], [spec['duration_s'] for spec in specs], rtol=0, atol=1e-8)
    final_target = np.array([sine_reference(spec)[-1, 0] for spec in specs])
    full &= (metrics[:, 0] >= 0) & (abs(metrics[:, 8]+final_target) <= np.pi/2)
    return dict(method=name, fitness=float(fitness(metrics)), count=len(metrics),
                survived=int(full.sum()), successes=int(metrics[:, 5].sum()),
                mean_rmse_deg=float(np.rad2deg(metrics[:, 2]).mean()),
                worst_rmse_deg=float(np.rad2deg(metrics[:, 2]).max()),
                mean_in_band_fraction=float(metrics[:, 6].mean()),
                mean_velocity_rmse_rad_s=float(metrics[:, 7].mean()))


def train_mode(output, evaluator, center, tau, mode, settings):
    """Fine-tune one search subspace while retaining generation zero for validation."""
    output.mkdir(parents=True, exist_ok=False)
    genome = evaluator.genome
    indices = np.arange(genome.controller.size, genome.size) if mode == 'command_only' else np.arange(genome.size)
    checkpoints = {1, settings['generations'], *range(settings['validation_every'], settings['generations']+1,
                                                      settings['validation_every'])}
    initial_score, initial_metrics = evaluator.evaluate([center], 'validation', tau)
    winner = dict(score=float(initial_score[0]), unit=center.copy(), generation=0, individual=0)
    validation_rows = [dict(generation=0, best_validation_fitness=winner['score'], selected_generation=0)]
    np.savez_compressed(output/'validation_000.npz', population=center[None], scores=initial_score,
                        metrics=initial_metrics)

    def expand(population):
        """Insert optimized coordinates into a copied full chromosome."""
        full = np.tile(center, (len(population), 1))
        full[:, indices] = population
        return full

    def evaluate(population):
        """Evaluate the current search subspace on training trajectories only."""
        return evaluator.evaluate(expand(population), 'training', tau)

    def checkpoint(generation, population, scores, metrics, history, rng_state):
        """Persist every population; validate prescribed checkpoints independently."""
        full = expand(population)
        decoded = [genome.decode(row) for row in full]
        np.savez_compressed(output/f'generation_{generation:03d}.npz', population=full,
                            search_population=population, scores=scores, metrics=metrics,
                            physical=np.array([np.r_[p, c.ravel()] for p, c in decoded]))
        write_csv(output/'history.csv', history)
        write_json(output/'resume.json', dict(generation=generation, rng_state=rng_state, history=history,
                                             search_indices=indices.tolist()))
        if generation in checkpoints:
            val_scores, val_metrics = evaluator.evaluate(full, 'validation', tau)
            np.savez_compressed(output/f'validation_{generation:03d}.npz', population=full,
                                scores=val_scores, metrics=val_metrics)
            best = int(np.argmax(val_scores))
            if val_scores[best] > winner['score']:
                winner.update(score=float(val_scores[best]), unit=full[best].copy(), generation=generation,
                              individual=best)
            validation_rows.append(dict(generation=generation, best_validation_fitness=float(val_scores[best]),
                                        selected_generation=winner['generation']))
            write_csv(output/'validation_history.csv', validation_rows)
            genome.save(output/'best_genome.json', winner['unit'], current_tau_s=tau, mode=mode,
                         selected_generation=winner['generation'], individual=winner['individual'],
                         validation_fitness=winner['score'])

    center_search = center[indices]
    rng = np.random.default_rng(settings['seed']+1000)
    candidates = reflect(center_search+rng.normal(0, .012, (settings['population']-1, len(indices))))
    schedule = Schedule(settings['sigma_start'], settings['sigma_end'],
                        settings['mutation_probability'], settings['mutation_probability'])
    optimize(center_search, evaluate, size=settings['population'], generations=settings['generations'],
             seed=settings['seed'], schedule=schedule, on_state=checkpoint, initial_candidates=candidates)
    write_json(output/'selection.json', {k:v for k, v in winner.items() if k != 'unit'})
    return winner['unit']


def save_dynamics(folder, genome, unit, specs, tau):
    """Save full 0.1 ms traces, applied currents and reference states for every case."""
    folder.mkdir(parents=True, exist_ok=True)
    p, c = genome.decode(unit)
    rows, metrics = [], []
    for index, spec in enumerate(specs):
        result = replay(p, c, genome.controller.mechanics, spec, tau_s=tau, stride=1)
        path = folder/f'case_{index:03d}.npz'
        np.savez_compressed(path, **result, physical=p, coefficients=c, mechanics=genome.controller.mechanics,
                            current_tau_s=tau, dt_ms=.1, columns=TRACE_COLUMNS,
                            command_columns=['applied_flex', 'applied_ext', 'desired_flex', 'desired_ext'],
                            reference_columns=['target_rad', 'target_velocity_rad_s'], metric_columns=METRIC_COLUMNS)
        metrics.append(result['metrics'])
        rows.append(dict(case=index, **spec, failure=int(result['failure']),
                         **dict(zip(METRIC_COLUMNS, map(float, result['metrics'])))))
    write_csv(folder/'metrics.csv', rows)
    write_json(folder/'cases.json', specs)
    return np.array(metrics)


def run_study(output, *, config=HERE/'genome_cfg.json', workers=1, settings_override=None,
              sources=None):
    """Run a reproducible frozen-baseline comparison and two fine-tuning methods."""
    output = Path(output).resolve()
    cfg = json.loads(Path(config).read_text()) if not isinstance(config, dict) else deepcopy(config)
    if settings_override:
        cfg['trajectory']['ga'].update(settings_override)
    genome = TrajectoryGenome(cfg)
    settings = cfg['trajectory']
    banks = trajectory_banks(settings)
    sources = default_sources() if sources is None else sources
    # Validate all sources before creating an output directory or launching work.
    seeds = {}
    for name, path in sources.items():
        oldcfg, p, c, _, _ = load_solution(path)
        if not np.array_equal(oldcfg.mechanics, genome.controller.mechanics):
            raise ValueError('Source mechanics differ from D mechanics')
        seeds[name] = genome.encode(p, c)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'genome_cfg.json', cfg)
    write_json(output/'banks.json', banks)
    hashes = source_hashes()
    write_json(output/'provenance.json', dict(source_hashes=hashes,
               source_genomes={name:dict(path=str(Path(path).resolve()), sha256=file_hash(path))
                               for name, path in sources.items()}, workers=workers,
               selection='Frozen baseline and fine-tuned genomes selected by validation only. Test is evaluated afterwards.'))
    with zipfile.ZipFile(output/'source_snapshot.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in hashes:
            archive.write(REPO/name, name)
    (output/'sources').mkdir()
    for name, path in sources.items():
        shutil.copy2(path, output/'sources'/f'{name}.json')
    # Compile once before workers start, avoiding simultaneous Numba cache writes.
    p, c = genome.decode(next(iter(seeds.values())))
    replay(p, c, genome.controller.mechanics, banks['training'][0], stride=0)
    candidates, frozen_rows = [], []
    with Evaluator(genome, banks, workers) as evaluator:
        for tau in settings['filter_times_s']:
            labels, population = list(seeds), np.array(list(seeds.values()))
            scores, metrics = evaluator.evaluate(population, 'validation', tau)
            np.savez_compressed(output/f'frozen_tau_{tau:g}.npz', population=population, scores=scores, metrics=metrics)
            for name, unit, score, values in zip(labels, population, scores, metrics):
                frozen_rows.append(dict(source=name, tau_s=tau, **summarize(name, values, banks['validation'])))
                candidates.append((float(score), name, tau, unit.copy()))
        write_csv(output/'frozen_validation.csv', frozen_rows)
        score, source, tau, center = max(candidates, key=lambda row: row[0])
        print(f'Frozen validation winner: {source}, tau={tau}, fitness={score:.6f}', flush=True)
        frozen = output/'frozen'
        frozen.mkdir()
        genome.save(frozen/'best_genome.json', center, current_tau_s=tau,
                    source=source, validation_fitness=score, selected_generation=0)
        solutions = {'frozen': center}
        for mode in ('command_only', 'joint'):
            print(f'Training {mode}', flush=True)
            solutions[mode] = train_mode(output/mode, evaluator, center, tau, mode, settings['ga'])
        # Freeze selection before computing any held-out test metric.
        validations = {name:float(evaluator.evaluate([unit], 'validation', tau)[0][0])
                       for name, unit in solutions.items()}
        selected = max(validations, key=validations.get)
        write_json(output/'selection.json', dict(method=selected, current_tau_s=tau,
                   validation_fitness=validations, frozen_source=source,
                   rule='Maximum validation fitness; test results have no role in selection.'))
        shutil.copy2(output/selected/'best_genome.json', output/'best_genome.json')
    summary = []
    diagnostic_specs = [sine_case(a, period) for a in (15., 30., 45.) for period in (10., 20., 40.)]
    for name, unit in solutions.items():
        print(f'Saving full dynamics: {name}', flush=True)
        metrics = save_dynamics(output/name/'test', genome, unit, banks['test'], tau)
        np.savez_compressed(output/name/'test_metrics.npz', metrics=metrics, metric_columns=METRIC_COLUMNS)
        summary.append(summarize(name, metrics, banks['test']))
        save_dynamics(output/name/'diagnostics', genome, unit, diagnostic_specs, tau)
    write_csv(output/'test_summary.csv', summary)
    from scripts.AfferentBalance2026.experiments.D_trajectory.plots import plot_study
    plot_study(output)
    write_json(output/'result.json', dict(status='complete', selected=selected, tau_s=tau,
                                         test=summary, dt_ms=.1))
    return output


def main():
    """Parse explicit reproducible study settings and run the D comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=HERE/'genome_cfg.json')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--population', type=int)
    parser.add_argument('--generations', type=int)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    overrides = {k:getattr(args, k) for k in ('population', 'generations', 'seed') if getattr(args, k) is not None}
    print(run_study(args.output, config=args.config, workers=args.workers, settings_override=overrides))


if __name__ == '__main__':
    main()
