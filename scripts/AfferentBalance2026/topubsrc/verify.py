#!/usr/bin/env python3
"""Reproduce every published held-out episode using only this source directory.

Reference arrays are frozen numerical artifacts, not a runtime dependency on
the original package. This command performs no training and selects no genome.
"""
import argparse
import importlib
import json
from pathlib import Path
import platform
import time

import numba
import numpy as np

from genome import file_hash, load_solution, write_json
from model import evaluate_population

HERE = Path(__file__).resolve().parent


def compare_runs(first, second):
    """Require identical genomes, populations, RNG states, banks and metrics.

    Elapsed time, machine provenance, ZIP headers and the worker count are not
    numerical research outputs and are intentionally excluded.
    """
    first, second = Path(first), Path(second)
    for name in ('genome_cfg.json','result.json'):
        if json.loads((first/name).read_text()) != json.loads((second/name).read_text()):
            raise AssertionError(f'Different {name}: {first}, {second}')
    settings = [json.loads((root/'training.json').read_text()) for root in (first,second)]
    for row in settings:
        row.pop('workers')
    if settings[0] != settings[1]:
        raise AssertionError('Different search settings')
    arrays, genomes = 0, 0
    files = sorted(p.relative_to(first) for p in first.rglob('*.npz'))
    if files != sorted(p.relative_to(second) for p in second.rglob('*.npz')):
        raise AssertionError('Different numerical artifact lists')
    for relative in files:
        with np.load(first/relative, allow_pickle=False) as a, np.load(second/relative, allow_pickle=False) as b:
            if a.files != b.files:
                raise AssertionError(f'Different array names: {relative}')
            for name in a.files:
                np.testing.assert_array_equal(a[name], b[name], err_msg=f'{relative}: {name}')
                arrays += 1
    for path in sorted(first.rglob('*genome.json')):
        relative = path.relative_to(first)
        if file_hash(path) != file_hash(second/relative):
            raise AssertionError(f'Different genome: {relative}')
        genomes += 1
    import csv
    for path in sorted(first.rglob('*history.csv')):
        relative = path.relative_to(first)
        histories = []
        for root in (first,second):
            with (root/relative).open() as stream:
                rows = list(csv.DictReader(stream))
            for row in rows:
                row.pop('elapsed_s', None)
            histories.append(rows)
        if histories[0] != histories[1]:
            raise AssertionError(f'Different history: {relative}')
    return dict(status='passed', exact_genomes=genomes, exact_arrays=arrays,
                best_genome_sha256=file_hash(first/'best_genome.json'))


def verify_training(output, workers=1, population=6, generations=4):
    """Repeat seven fresh searches with serial/parallel evaluation and full banks.

    This is an execution/repeatability check with a small search budget, not an
    estimate of attainable performance. It never reads published genomes.
    """
    from run import suite, PROFILES
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    settings = dict(population=population, generations=generations)
    suite(output/'serial', overrides={**settings, 'workers':1})
    suite(output/'parallel', overrides={**settings, 'workers':workers})
    rows = []
    for label in PROFILES:
        folder = label.replace(':','_')
        row = compare_runs(output/'serial'/folder, output/'parallel'/folder)
        rows.append(dict(profile=label, **row))
    result = dict(status='passed', training_performed=True, archive_genomes_used=False,
                  population=population, generations=generations, duration_s=10,
                  full_case_banks=True, workers=[1,workers], python=platform.python_version(),
                  numpy=np.__version__, numba=numba.__version__, results=rows)
    write_json(output/'verification.json', result)
    return result


def verify(workers=1):
    """Check portable genomes, independent case generation and all frozen metrics."""
    if not isinstance(workers, int) or not 1 <= workers <= numba.config.NUMBA_NUM_THREADS:
        raise ValueError(f'workers must be between 1 and {numba.config.NUMBA_NUM_THREADS}')
    old_threads = numba.get_num_threads()
    numba.set_num_threads(workers)
    rows = []
    try:
        for item in json.loads((HERE/'published_results.json').read_text()):
            path = HERE/item['genome']
            reference = HERE/item['reference_metrics']
            assert file_hash(path) == item['published_file_sha256'], f'Changed example genome: {path}'
            assert file_hash(reference) == item['reference_metrics_sha256'], f'Changed reference: {reference}'
            genome, unit = load_solution(path)
            module = importlib.import_module(f"{genome.controller.data['experiment']}.experiment")
            banks = module.make_banks(genome.controller.data, genome.kind)
            started = time.perf_counter()
            scores, metrics = evaluate_population(genome, unit[None], banks['test'],
                                                   genome.controller.data['simulation']['duration_s'])
            with np.load(reference, allow_pickle=False) as saved:
                np.testing.assert_array_equal(banks['test'], saved['cases'])
                np.testing.assert_allclose(metrics[0], saved['metrics'], rtol=1e-9, atol=1e-9)
                np.testing.assert_allclose(scores, saved['scores'], rtol=1e-11, atol=1e-11)
                difference = float(np.abs(metrics[0]-saved['metrics']).max())
            assert int(metrics[0, :, 5].sum()) == item['test_success']
            row = dict(experiment=item['experiment'], domain=item['domain'], episodes=len(banks['test']),
                       success=int(metrics[0, :, 5].sum()), max_absolute_metric_difference=difference,
                       elapsed_s=round(time.perf_counter()-started, 3))
            rows.append(row)
            print(f"{row['experiment']}/{row['domain']}: {row['success']}/{row['episodes']} successes; "
                  f"max metric difference={difference:.3g}", flush=True)
    finally:
        numba.set_num_threads(old_threads)
    return dict(status='passed', python=platform.python_version(), numpy=np.__version__, numba=numba.__version__,
                genomes=len(rows), test_episodes=sum(r['episodes'] for r in rows), results=rows,
                training_performed=False, code_sha256={str(p.relative_to(HERE)):file_hash(p)
                                                       for p in sorted([*HERE.glob('*.py'), *HERE.glob('*/experiment.py')])})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=1, help='Numba threads for independent episodes')
    parser.add_argument('--output', type=Path, help='Optional local JSON report; otherwise print only')
    parser.add_argument('--training', action='store_true', help='Repeat seven from-scratch searches (full 10 s banks)')
    parser.add_argument('--directory', type=Path, default=Path('results/repeatability'),
                        help='New directory for the two training suites')
    parser.add_argument('--population', type=int, default=6, help='Repeatability-check population')
    parser.add_argument('--generations', type=int, default=4, help='Repeatability-check generation budget')
    parser.add_argument('--compare', type=Path, nargs=2, metavar=('FIRST','SECOND'),
                        help='Require exact equality of two completed training runs')
    args = parser.parse_args()
    if args.compare and args.training:
        parser.error('Choose --compare or --training')
    result = (compare_runs(*args.compare) if args.compare else
              verify_training(args.directory, args.workers, args.population, args.generations)
              if args.training else verify(args.workers))
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2))
