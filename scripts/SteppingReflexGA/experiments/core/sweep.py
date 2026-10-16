"""Reproducible mutation/population sweeps with exact epoch continuations."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import threading
import time
import zipfile

import numpy as np
from .config import DT_MS, MODEL
from .evolution import Schedule
from .run import run, source_hashes, write_json, EXPERIMENTS


def load_state(path):
    with np.load(path, allow_pickle=False) as data:
        return dict(generation=int(data["generation"]), population=data["population"].copy(),
                    scores=data["scores"].copy(), history=json.loads(str(data["history_json"])),
                    rng_state=json.loads(str(data["rng_state"])))


def line_path(root, size, probability, seed):
    return Path(root) / f"mutation_{probability:.3f}" / f"population_{size:03d}" / f"seed_{seed}"


def train_line(root, settings, size, probability, seed, notify, resume=False):
    path = line_path(root, size, probability, seed)
    schedule = Schedule(probability_start=probability, probability_end=probability)
    state = None
    results = []
    for epochs in settings["epochs"]:
        destination = path / f"epochs_{epochs:03d}"
        if (destination / "result.json").exists():
            if not resume:
                raise FileExistsError(destination)
            result = json.loads((destination / "result.json").read_text())
            state = load_state(destination / "final_population.npz")
        else:
            if destination.exists():
                if not resume:
                    raise FileExistsError(destination)
                checkpoint = destination / "checkpoint.npz"
                if checkpoint.exists():
                    saved = load_state(checkpoint)
                    # The final evaluated state needs postprocessing only. Reuse
                    # the previous completed budget when an interruption occurred
                    # during validation, so no unevaluated state is accepted.
                    if saved["generation"] < epochs:
                        state = saved
                interrupted = path / "interrupted" / f"epochs_{epochs:03d}_{time.time_ns()}"
                interrupted.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(destination, interrupted)
            result = run(settings["experiment"], destination,
                         population_size=size, generations=epochs, seed=seed,
                         workers=settings["workers_per_line"], plot=False,
                         population_from=settings.get("population_from"),
                         schedule=schedule, schedule_generations=max(settings["epochs"]),
                         resume_state=state, checkpoint_every=10, verbose=False)
            state = load_state(destination / "final_population.npz")
        result = dict(result, directory=str(destination.relative_to(root)))
        results.append(result)
        notify(result)
    return results


def launch(root, *, experiment="equilibrium", populations=(10, 30, 50, 100),
           epochs=(30, 60, 100, 150, 200), probabilities=(.1, .3, .6), seeds=(42,),
           jobs=3, workers_per_line=4, population_from=None, resume=False):
    root = Path(root).resolve()
    if min(populations) < 3 or min(epochs) < 1 or jobs < 1 or workers_per_line < 1:
        raise ValueError("Invalid sweep sizes")
    if len(set(populations)) != len(populations) or len(set(epochs)) != len(epochs) or len(set(probabilities)) != len(probabilities) or len(set(seeds)) != len(seeds):
        raise ValueError("Sweep axes must contain unique values")
    for probability in probabilities:
        Schedule(probability_start=probability, probability_end=probability)
    settings = dict(experiment=experiment, populations=sorted(populations), epochs=sorted(epochs),
                    probabilities=sorted(probabilities), seeds=sorted(seeds),
                    jobs=jobs, workers_per_line=workers_per_line,
                    population_from=None if population_from is None else str(Path(population_from).resolve()),
                    dt_ms=DT_MS, model=MODEL, duration_s=10.,
                    sigma_schedule={k: v for k, v in asdict(Schedule()).items() if k.startswith("sigma_")},
                    probability_policy="Constant per gene throughout training; same value in every generation",
                    epoch_policy="Snapshots of one continued trajectory; sigma anneals over max(epochs), not each snapshot budget",
                    source_sha256=source_hashes())
    manifest = root / "sweep.json"
    if root.exists():
        if not resume or not manifest.exists() or json.loads(manifest.read_text()) != settings:
            raise ValueError("Resume requires identical sweep settings and source hashes")
    else:
        root.mkdir(parents=True)
        write_json(manifest, settings)
        repo = Path(__file__).resolve().parents[4]
        with zipfile.ZipFile(root / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as snapshot:
            for filename in settings["source_sha256"]:
                p = Path(filename)
                snapshot.write(p, str(p.relative_to(repo)))

    total = len(populations)*len(epochs)*len(probabilities)*len(seeds)
    lock = threading.Lock()
    completed = {}
    started = time.monotonic()

    def notify(result):
        with lock:
            completed[result["directory"]] = result
            write_json(root / "progress.json", dict(completed=len(completed), total=total,
                       elapsed_s=time.monotonic()-started, last=result, finished=False))
            print(f"[{len(completed)}/{total}] N={result['population']} p={result['mutation_probability']:g} "
                  f"epochs={result['generations']} validation={result['validation_fitness']:.6f} "
                  f"test_success={result['test_success']:.1%}", flush=True)

    # The heaviest lines start first. Threads coordinate independent RNGs and
    # process pools; numerical workers total jobs * workers_per_line.
    specifications = [(n, p, seed) for n in sorted(populations, reverse=True)
                      for p in probabilities for seed in seeds]
    with ThreadPoolExecutor(jobs) as executor:
        futures = [executor.submit(train_line, root, settings, n, p, seed, notify, resume)
                   for n, p, seed in specifications]
        for future in as_completed(futures):
            future.result()
    from .sweep_analysis import analyze
    summary = analyze(root)
    write_json(root / "progress.json", dict(completed=total, total=total,
               elapsed_s=time.monotonic()-started, finished=True))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment", choices=EXPERIMENTS, default="equilibrium")
    parser.add_argument("--populations", type=int, nargs="+", default=[10, 30, 50, 100])
    parser.add_argument("--epochs", type=int, nargs="+", default=[30, 60, 100, 150, 200])
    parser.add_argument("--mutation-probabilities", type=float, nargs="+", default=[.1, .3, .6])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--workers-per-line", type=int, default=4)
    parser.add_argument("--population-from", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if args.analyze_only:
        from .sweep_analysis import analyze
        print(json.dumps(analyze(args.output), indent=2))
    else:
        # Numerical libraries in spawned workers should not start nested BLAS pools.
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
            os.environ[name] = "1"
        print(json.dumps(launch(args.output, experiment=args.experiment,
              populations=args.populations, epochs=args.epochs,
              probabilities=args.mutation_probabilities, seeds=args.seeds,
              jobs=args.jobs, workers_per_line=args.workers_per_line,
              population_from=args.population_from, resume=args.resume), indent=2))


if __name__ == "__main__":
    main()
