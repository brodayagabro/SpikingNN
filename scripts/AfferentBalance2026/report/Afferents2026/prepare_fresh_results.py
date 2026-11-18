#!/usr/bin/env python3
"""Export completed fresh searches and replay prescribed figure cases; never train.

The fast-domain winners come solely from the recorded validation selections.
Slow/mirror rows are the untuned N=50, G=200 baseline, not comparable sweeps.
Numerical training artifacts remain immutable. Figure caches live here.
"""
from pathlib import Path
import hashlib
import json
import shutil
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SOURCE = ROOT / 'topubsrc'
FRESH = SOURCE / 'fresh_results'
CACHE = HERE / 'fresh_replays'
sys.path.insert(0, str(SOURCE))
from genome import file_hash, load_solution, write_json
from model import MODEL, case_grid, replay_genome

EXPS = {'A': 'A_balance', 'B': 'B_balance', 'C': 'C_setpoint_balance'}


def read(path):
    return json.loads(Path(path).read_text())


def selected_runs():
    selections = {}
    for task in EXPS:
        record = read(FRESH / f'expanded_{task}_selection.json')
        assert record['status'] == 'completed' and not record['test_used_for_selection']
        selections[task, 'fast'] = FRESH / f'expanded_{task}' / record['selected']['run']
        selections[task, 'slow'] = FRESH / f'{task}_slow'
    selections['C', 'mirror'] = FRESH / 'C_mirror'
    return selections


def catalogue():
    rows = []
    for task, folder in EXPS.items():
        paths = [(FRESH/f'{task}_{group}', group) for group in ('fast', 'slow', 'mirror')
                 if (FRESH/f'{task}_{group}').exists()]
        paths += [(p.parent, 'fast') for p in sorted((FRESH/f'expanded_{task}').glob('G*/*/result.json'))]
        for path, group in paths:
            provenance = read(path/'provenance.json')
            assert provenance['initialization'] == 'configuration'
            assert not provenance['archive_genomes_used'] and not provenance['smoke_test']
            rows.append(dict(task=task, group=group, run=str(path.relative_to(ROOT)),
                             options=read(path/'training.json'), result=read(path/'result.json')))
    assert len(rows) == 43
    return rows


def stage_for(path):
    return path / ('polynomial' if (path/'polynomial').exists() else 'training')


def export_examples():
    published = []
    for (task, group), path in selected_runs().items():
        folder = SOURCE/EXPS[task]
        target = folder/'genomes'/f'{group}.json'
        shutil.copy2(path/'best_genome.json', target)
        result, settings = read(path/'result.json'), read(path/'training.json')
        reference = folder/'reference'/f'{group}.npz'
        with np.load(path/'test_metrics.npz') as m, np.load(stage_for(path)/'cases.npz') as c:
            np.savez_compressed(reference, metrics=m['metrics'], scores=m['scores'], cases=c['test'])
        command = ['python', 'run.py', 'repeat', '--run', str(path.relative_to(SOURCE)),
                   '--workers', str(settings['workers']), '--output', f'results/{task}_{group}_repeated']
        published.append(dict(experiment=task, domain=group, genome=str(target.relative_to(SOURCE)),
                              source=str((path/'best_genome.json').relative_to(ROOT)),
                              validation_fitness=result['validation_fitness'], test_success=result['test_success'],
                              test_cases=result['test_cases'], original_file_sha256=file_hash(path/'best_genome.json'),
                              published_file_sha256=file_hash(target), portability_edit='None: byte-for-byte fresh winner.',
                              reference_metrics=str(reference.relative_to(SOURCE)), reference_metrics_sha256=file_hash(reference),
                              training=dict(initialization='configuration', archive_genomes_used=False,
                                            options=settings, command_argv=command,
                                            selected_generation=result['selected_generation']), appendix_genomes=[]))
    write_json(SOURCE/'published_results.json', published)


def prepare_replays():
    for (task, group), path in selected_runs().items():
        # Prescribed diagnostic banks: only the four branches shown in phase figures.
        if (task, group) not in {('A','fast'), ('B','fast'), ('C','fast'), ('C','mirror')}:
            continue
        target = 45. if task == 'C' else 0.
        output = CACHE/f'{task}_{group}'
        output.mkdir(parents=True, exist_ok=True)
        identity = dict(genome_sha256=file_hash(path/'best_genome.json'), model=MODEL,
                        model_sha256=file_hash(SOURCE/'model.py'), target_deg=target,
                        duration_s=10., fitness_window_s=2., grid_stride=20, example_stride=1)
        summary = output/'summary.json'
        if summary.exists() and read(summary).get('identity') == identity:
            saved=read(summary)
            if all(file_hash(output/name)==value for name,value in saved['files'].items()):
                print('Cached:', task, group, flush=True)
                continue
        genome, unit = load_solution(path/'best_genome.json')
        files, counts = {}, {}
        for kind, cases, stride in [
            ('grid', case_grid(target_deg=target), 20),
            ('zero_speed', case_grid(speeds=[0.], target_deg=target), 20),
            ('example', case_grid(angles_deg=[-30.], speeds=[0.], target_deg=target), 1)]:
            arrays, metrics = {}, []
            for i, case in enumerate(cases):
                replay = replay_genome(genome, unit, target_deg=target, angle_deg=np.degrees(case[1]),
                                       speed_rad_s=case[2], duration_s=10., stride=stride)
                metrics.append(replay['metrics'])
                arrays[f'trace_{i:03d}'] = replay['trace']
            metrics = np.asarray(metrics)
            if kind == 'example':
                arrays = dict(trace=arrays['trace_000'], metrics=metrics[0], case=cases[0])
            else:
                arrays.update(cases=cases, metrics=metrics)
            name = f'target_{target:+06.1f}_{kind}.npz'
            np.savez_compressed(output/name, **arrays)
            files[name] = file_hash(output/name)
            counts[kind] = dict(cases=len(cases), success=int(metrics[:,5].sum()),
                                completed=int((metrics[:,1]>=10-1e-5).sum()))
        write_json(summary, dict(identity=identity, files=files, counts=counts))
        print('Replayed:', task, group, counts, flush=True)


def main():
    catalogue()  # Reject missing/incomplete or archive-initialized runs before any export.
    export_examples()
    prepare_replays()


if __name__ == '__main__':
    main()
