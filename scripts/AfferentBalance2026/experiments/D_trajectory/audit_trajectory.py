"""Audit a completed D study against saved populations, traces and independent physics."""
import argparse
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from scripts.AfferentBalance2026.experiments.core.fitness import fitness
from scripts.AfferentBalance2026.experiments.core.genome import file_hash, write_json
from scripts.AfferentBalance2026.experiments.core.run import REPO, source_hashes
from scripts.AfferentBalance2026.experiments.D_trajectory.controller import TrajectoryGenome, load_solution
from scripts.AfferentBalance2026.experiments.D_trajectory.run_experiment import Evaluator
from scripts.AfferentBalance2026.experiments.D_trajectory.trajectory import measure_tracking_trace, sine_reference
from scripts.AfferentBalance2026.experiments.simulation import reference_trace


def audit(output, workers=1):
    """Verify provenance, all generations, selection, tests and complete dynamics."""
    output = Path(output).resolve()
    assert json.loads((output/'result.json').read_text())['status'] == 'complete'
    genome = TrajectoryGenome(output/'genome_cfg.json')
    banks = json.loads((output/'banks.json').read_text())
    provenance = json.loads((output/'provenance.json').read_text())
    expected_hashes = provenance['source_hashes']
    live_hashes = source_hashes()
    audit_source = str(Path(__file__).resolve().relative_to(REPO))
    # Older studies predate this auditor's move into experiments/. Newly created
    # Jupyter backup files are also outside the calculation. Keep every recorded
    # hash strict, including any backups already present in the saved manifest.
    for name in list(live_hashes):
        if name not in expected_hashes and (
                name == audit_source or '.ipynb_checkpoints' in Path(name).parts):
            live_hashes.pop(name)
    assert live_hashes == expected_hashes, 'Live computational sources differ from the study snapshot'
    import hashlib
    with zipfile.ZipFile(output/'source_snapshot.zip') as archive:
        for name, expected in expected_hashes.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected, name
    for name, source in provenance['source_genomes'].items():
        assert file_hash(output/'sources'/f'{name}.json') == source['sha256']
    generations, individuals = 0, 0
    for mode in ('command_only', 'joint'):
        folder = output/mode
        seed = np.load(folder/'validation_000.npz')['population'][0]
        for path in sorted(folder.glob('generation_*.npz')):
            with np.load(path) as saved:
                pop, scores, metrics = saved['population'], saved['scores'], saved['metrics']
                decoded = [genome.decode(row) for row in pop]
                physical = np.array([np.r_[p, c.ravel()] for p, c in decoded])
                np.testing.assert_array_equal(physical, saved['physical'])
                np.testing.assert_array_equal(physical[:, [18, 25, 46, 47]],
                                              np.tile([.37, .37, .2, .05], (len(pop), 1)))
                assert (physical[:, [19, 20, 26, 27]] >= 5.1).all()
                np.testing.assert_array_equal(scores, [fitness(m) for m in metrics])
                if mode == 'command_only':
                    np.testing.assert_array_equal(pop[:, :48], np.tile(seed[:48], (len(pop), 1)))
                generations += 1
                individuals += len(pop)
        candidates = []
        for path in sorted(folder.glob('validation_*.npz')):
            generation = int(path.stem.split('_')[-1])
            with np.load(path) as saved:
                np.testing.assert_array_equal(saved['scores'], [fitness(m) for m in saved['metrics']])
                for index, score in enumerate(saved['scores']):
                    candidates.append((float(score), generation, index, saved['population'][index].copy()))
        winner = max(candidates, key=lambda row: row[0])
        data = json.loads((folder/'best_genome.json').read_text())
        np.testing.assert_array_equal(data['normalized'], winner[3])
        assert data['metadata']['selected_generation'] == winner[1]
    selection = json.loads((output/'selection.json').read_text())
    assert selection['method'] == max(selection['validation_fitness'], key=selection['validation_fitness'].get)
    assert file_hash(output/'best_genome.json') == file_hash(output/selection['method']/'best_genome.json')
    trace_count, samples, max_metric_error = 0, 0, 0.
    library_errors, units = {}, []
    for mode in ('frozen', 'command_only', 'joint'):
        cfg, p, coefficients, tau, data = load_solution(output/mode/'best_genome.json')
        units.append(np.asarray(data['normalized']))
        for bank in ('test', 'diagnostics'):
            specs = json.loads((output/mode/bank/'cases.json').read_text())
            for index, spec in enumerate(specs):
                with np.load(output/mode/bank/f'case_{index:03d}.npz') as saved:
                    trace, ref = saved['trace'], saved['reference']
                    assert trace[0, 0] == 0
                    np.testing.assert_array_equal(saved['physical'], p)
                    np.testing.assert_array_equal(saved['coefficients'], coefficients)
                    np.testing.assert_array_equal(ref, sine_reference(spec)[:len(trace)])
                    measured = measure_tracking_trace(trace, ref, spec)
                    error = abs(measured-saved['metrics']).max()
                    max_metric_error = max(max_metric_error, float(error))
                    np.testing.assert_allclose(measured, saved['metrics'], rtol=1e-10, atol=1e-10)
                    # Reconstruct actual applied currents, including left-endpoint timing.
                    x = ref[:-1, 0]/(np.pi/4)
                    desired = sum(coefficients[:, k][None, :]*x[:, None]**k for k in range(4))
                    np.testing.assert_allclose(desired, saved['commands'][1:, 2:], rtol=1e-12, atol=1e-10)
                    if tau == 0:
                        np.testing.assert_array_equal(saved['commands'][:, :2], saved['commands'][:, 2:])
                    else:
                        alpha = -np.expm1(-.0001/tau)
                        previous = saved['commands'][:-1, :2]
                        np.testing.assert_allclose(previous+alpha*(desired-previous), saved['commands'][1:, :2],
                                                   rtol=1e-12, atol=1e-10)
                    trace_count += 1
                    samples += len(trace)
        # Include 0.1 s of changing command after the four-second hold.
        with np.load(output/mode/'diagnostics/case_004.npz') as saved:
            duration = 4.1
            last = round(duration*10000)
            case = np.array([0., 0., 0., -1., 0.])
            library = reference_trace(p, case, cfg.mechanics,
                                      currents=saved['commands'][1:last+1, :2], duration_s=duration)
            fast = saved['trace'][:len(library)]
            library_errors[mode] = float(np.max(abs(library-fast)))
            np.testing.assert_allclose(library, fast, rtol=1e-7, atol=1e-7)
    with Evaluator(genome, banks, workers) as evaluator:
        scores, metrics = evaluator.evaluate(units, 'test', selection['current_tau_s'])
    for i, mode in enumerate(('frozen', 'command_only', 'joint')):
        with np.load(output/mode/'test_metrics.npz') as saved:
            np.testing.assert_array_equal(metrics[i], saved['metrics'])
    result = dict(status='passed', generations=generations, individuals=individuals,
                  full_traces=trace_count, samples=samples,
                  max_metric_recomputation_error=max_metric_error,
                  library_duration_s=4.1, library_max_absolute_errors=library_errors,
                  test_reproduced_exactly=True, source_hashes_verified=True,
                  selected_genome_sha256=file_hash(output/'best_genome.json'))
    write_json(output/'audit.json', result)
    return result


def main():
    """Run a read-only numerical audit and write its machine-readable result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(audit(args.output, args.workers), indent=2))


if __name__ == '__main__':
    main()
