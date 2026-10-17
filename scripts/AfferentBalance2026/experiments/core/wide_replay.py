"""Full-resolution, out-of-training-range replay of a fixed validation winner."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import time
import zipfile

import numpy as np
from . import genome
from .afferents import PARAMETERS
from .cases import banks
from .config import DT_MS, MODEL, MECHANICS
from .dynamics import rollout, measure_trace, METRIC_COLUMNS, TRACE_COLUMNS
from .fitness import fitness
from .run import source_hashes, write_csv, write_json

CASE_COLUMNS = ("target_rad", "theta0_rad", "omega0_rad_s", "kick_time_s", "kick_delta_omega")


def wide_cases(angles=range(-45, 46, 5), velocities=(-.5, 0., .5), kicks=(-.3, 0., .3), kick_time_s=5.):
    """Cartesian test grid, fixed upright target; no RNG or genome selection."""
    for axis in (angles, velocities, kicks):
        if not len(axis) or not np.isfinite(axis).all() or len(set(axis)) != len(axis):
            raise ValueError("Case axes must be nonempty, finite and unique")
    if np.any(np.abs(angles) >= 90.) or not np.isfinite(kick_time_s) or kick_time_s < 0.:
        raise ValueError("Initial angles must lie inside (-90, 90), and kick time must be nonnegative")
    return np.array([[0., np.deg2rad(angle), velocity, kick_time_s if kick else -1., kick]
                     for kick in kicks for velocity in velocities for angle in angles], dtype=float)


def full_replay(controller, case, duration_s=10.):
    """Keep every integration step and explicitly prepend the reset state at t=0."""
    controller, case = np.asarray(controller, dtype=float), np.asarray(case, dtype=float)
    if controller.shape != (52,) or case.shape != (5,) or not np.isfinite(controller).all() or not np.isfinite(case).all():
        raise ValueError("Expected a finite 52-field controller and 5-field case")
    if np.any(controller[genome.BIAS_INDICES] != 0.) or case[0] != 0.:
        raise ValueError("Balance replay requires zero input currents and the upright target")
    metrics, trace = rollout(controller, case[1:], target_rad=case[0],
                             duration_s=duration_s, dt_ms=DT_MS, mode=0, stride=1)
    initial = np.zeros(len(TRACE_COLUMNS))
    initial[1:3] = case[1:3]
    initial[5:7] = controller[4:6]
    initial[7:9] = controller[4:6] * controller[2:4]
    full = np.vstack((initial, trace))
    if metrics[0] != -1.:
        independently_measured = measure_trace(trace, target_rad=case[0], duration_s=duration_s,
                                                initial_theta=case[1])
        np.testing.assert_allclose(metrics, independently_measured, atol=1e-10, rtol=1e-10)
    # Compare the actual grid, not subtractions of nearby binary floats at t=10 s.
    np.testing.assert_array_equal(full[:, 0], np.arange(len(full), dtype=float)*DT_MS/1000)
    return metrics, full


def nonlinear_signals(trace, controller):
    """Algebraic diagnostics from recorded states; no second dynamics integrator.

    Receptor terms below describe the recorded state, including t=0. Actual
    afferents are reset to zero at t=0 and applied to neurons one step later.
    """
    p, t = np.asarray(controller), np.asarray(trace)
    q = t[:, 1] - np.pi/2
    a1, a2 = MECHANICS.a1_m, MECHANICS.a2_m
    angle = np.column_stack((q, np.pi-q))
    length = np.sqrt(a1*a1 + a2*a2 - 2*a1*a2*np.cos(angle))
    arm = a1*a2*np.sin(angle)/length
    velocity = arm * t[:, 2, None] * np.array([1., -1.])
    power = p[[21, 28]]
    half = p[[22, 29]]
    calcium = t[:, 9:11]
    activation = calcium**power / (calcium**power + half**power)
    stretch = np.maximum((length-p[42:44])/p[42:44], 0.)
    speed = np.maximum(velocity/p[42:44], 0.)
    tension = np.maximum((t[:, 3:5]-p[44:46])/p[44:46], 0.)
    pv, kv, kd1, kd2, kn1, kn2, kf, c1, c2 = PARAMETERS
    terms = np.stack((kv*speed**pv, kd1*stretch, kn1*activation,
                      np.full_like(stretch, c1), kd2*stretch, kn2*activation,
                      np.full_like(stretch, c2), kf*tension), axis=2)
    currents = np.zeros((len(t), 2))
    # Each row is the afferent current actually used in the step ending there.
    currents[1:] = t[:-1, 11:17] @ p[30:42].reshape(2, 6).T
    return dict(length_m=length, velocity_m_s=velocity, activation=activation,
                afferent_terms=terms, afferent_current_used=currents)


def _save_case(task):
    root, index, case, p, duration_s = task
    metrics, trace = full_replay(p, case, duration_s)
    termination = ("numerical_failure" if metrics[0] == -1. else
                   "angle_limit" if abs(trace[-1, 1]) > np.pi/2 else "duration")
    path = Path(root) / "traces" / f"case_{index:03d}.npz"
    np.savez_compressed(path, trace=trace, columns=np.array(TRACE_COLUMNS),
                        metrics=metrics, metric_columns=np.array(METRIC_COLUMNS),
                        case=case, case_columns=np.array(CASE_COLUMNS),
                        physical_controller=p, gene_names=np.array([g.name for g in genome.GENES]),
                        spike_times_flex_s=trace[trace[:, 5] >= 30., 0],
                        spike_times_ext_s=trace[trace[:, 6] >= 30., 0],
                        dt_ms=DT_MS, duration_s=duration_s, mode=0, model=MODEL,
                        termination=termination)
    return dict(trial=index, theta0_deg=float(np.rad2deg(case[1])), omega0_rad_s=float(case[2]),
                kick_time_s=float(case[3]), kick_delta_omega=float(case[4]),
                **dict(zip(METRIC_COLUMNS, map(float, metrics))),
                late_rmse_deg=float(np.rad2deg(metrics[7])), samples=len(trace),
                termination=termination, metrics_recomputed=metrics[0] != -1.,
                trace_file=str(path.relative_to(root)),
                trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def run_replay(sweep_root, output, workers=4, *, angles=range(-45, 46, 5),
               velocities=(-.5, 0., .5), kicks=(-.3, 0., .3), duration_s=10., kick_time_s=5.):
    sweep_root, output = Path(sweep_root).resolve(), Path(output).resolve()
    cases = wide_cases(angles, velocities, kicks, kick_time_s)
    if workers < 1 or not np.isfinite(duration_s) or duration_s < DT_MS/1000:
        raise ValueError("Need positive workers and duration")
    if np.any(cases[:, 3] >= duration_s):
        raise ValueError("Kicks must happen before the episode ends")
    analysis = json.loads((sweep_root / "analysis.json").read_text())
    source = sweep_root / analysis["best_by_validation"]["directory"]
    original = json.loads((source / "run.json").read_text())
    if original["experiment"] not in ("equilibrium", "small_nm_weights"):
        raise ValueError("Choose a completed balance sweep")
    current_hashes = source_hashes()
    numerical_files = {"config.py", "genome.py", "afferents.py", "dynamics.py", "fitness.py", "Izh_net.py"}
    for filename, digest in original["source_sha256"].items():
        if Path(filename).name in numerical_files:
            assert current_hashes[filename] == digest, f"Numerical source changed: {filename}"
    with np.load(source / "final_population.npz", allow_pickle=False) as data:
        index = int(data["selected_index"])
        assert index == int(np.argmax(data["validation_scores"]))
        p = data["physical_controllers"][index].copy()
    assert np.all(p[genome.BIAS_INDICES] == 0.)
    domain = genome.ControllerDomain(original["experiment"] == "small_nm_weights")
    domain.encode(p)
    output.mkdir(parents=True, exist_ok=False)
    (output / "traces").mkdir()
    shutil.copy2(source / "best_genome.json", output / "best_genome.json")
    np.save(output / "physical_controller.npy", p)
    write_csv(output / "cases.csv", [dict(trial=i, **dict(zip(CASE_COLUMNS, map(float, case))),
              theta0_deg=float(np.rad2deg(case[1]))) for i, case in enumerate(cases)])
    manifest = dict(experiment=original["experiment"], dt_ms=DT_MS, model=MODEL,
                    mode=0, pd_used=False, duration_s=duration_s, target_rad=0., trace_stride=1,
                    initial_state_included=True, angles_deg=list(map(float, angles)),
                    velocities_rad_s=list(map(float, velocities)), kicks_rad_s=list(map(float, kicks)),
                    kick_time_s=kick_time_s,
                    hyperparameters=asdict(MECHANICS), workers=workers, cases=len(cases),
                    source_sweep=str(sweep_root), source_run=str(source),
                    source_selection="Best original validation checkpoint; no selection on wide-angle results",
                    original_best=analysis["best_by_validation"],
                    original_population_sha256=hashlib.sha256((source/"final_population.npz").read_bytes()).hexdigest(),
                    original_genome_sha256=hashlib.sha256((source/"best_genome.json").read_bytes()).hexdigest(),
                    source_sha256=current_hashes)
    write_json(output / "run.json", manifest)
    repo = Path(__file__).resolve().parents[4]
    with zipfile.ZipFile(output / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for filename in current_hashes:
            archive.write(filename, str(Path(filename).relative_to(repo)))
    # Verify exact replay of original training/validation metrics before using
    # a new case bank. Original physical NPZ values avoid an encode/decode roundtrip.
    errors = {}
    with np.load(source / "final_population.npz", allow_pickle=False) as data:
        for bank, key in (("training", "evaluation"), ("validation", "validation_metrics")):
            actual = np.array([rollout(p, case[1:], target_rad=case[0], mode=0,
                                      duration_s=original["duration_s"])[0]
                               for case in banks(original["experiment"])[bank]])
            expected = data[key][index]
            np.testing.assert_array_equal(actual, expected)
            errors[bank] = float(np.max(np.abs(actual-expected)))
    write_json(output / "original_replay_check.json", dict(status="passed", max_abs_errors=errors,
               numerical_sources_unchanged=True, physical_controller_taken_from_original_population=True))
    started = time.monotonic()
    tasks = [(str(output), i, case, p, duration_s) for i, case in enumerate(cases)]
    rows = []
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as executor:
        for row in executor.map(_save_case, tasks):
            rows.append(row)
            if len(rows) % 10 == 0 or len(rows) == len(cases):
                write_json(output / "progress.json", dict(completed=len(rows), total=len(cases),
                           elapsed_s=time.monotonic()-started, finished=False))
                print(f"{original['experiment']}: {len(rows)}/{len(cases)} full traces saved", flush=True)
    write_csv(output / "metrics.csv", rows)
    metric_array = np.array([[r[k] for k in METRIC_COLUMNS] for r in rows])
    summary = dict(cases=len(rows), successes=int(metric_array[:, 5].sum()),
                   success_fraction=float(metric_array[:, 5].mean()),
                   falls=sum(r["termination"] == "angle_limit" for r in rows),
                   numerical_failures=sum(r["termination"] == "numerical_failure" for r in rows),
                   fitness=fitness(metric_array), max_settling_s=float(metric_array[:, 6].max()),
                   mean_late_rmse_deg=float(np.rad2deg(metric_array[:, 7]).mean()),
                   max_late_rmse_deg=float(np.rad2deg(metric_array[:, 7]).max()),
                   total_samples=int(sum(r["samples"] for r in rows)),
                   trace_bytes=sum((output/r["trace_file"]).stat().st_size for r in rows),
                   elapsed_s=time.monotonic()-started)
    write_json(output / "summary.json", summary)
    write_json(output / "verification.json", dict(status="passed", cases=len(rows),
               full_resolution_metrics_recomputed=all(r["metrics_recomputed"] for r in rows),
               time_step_verified=True,
               mode=0, pd_used=False, external_currents_zero=True,
               original_metrics_exact=True, fixed_genome_for_all_cases=True))
    from .wide_replay_plots import analyze
    analyze(output)
    write_json(output / "progress.json", dict(completed=len(rows), total=len(cases),
               elapsed_s=time.monotonic()-started, finished=True))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--angles", type=float, nargs="+", default=list(range(-45, 46, 5)))
    parser.add_argument("--velocities", type=float, nargs="+", default=[-.5, 0., .5])
    parser.add_argument("--kicks", type=float, nargs="+", default=[-.3, 0., .3])
    parser.add_argument("--kick-time", type=float, default=5.)
    parser.add_argument("--duration", type=float, default=10.)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if args.analyze_only:
        from .wide_replay_plots import analyze
        analyze(args.output)
        return
    if args.sweep_root is None or args.workers < 1:
        parser.error("Provide --sweep-root and positive --workers")
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[key] = "1"
    print(json.dumps(run_replay(args.sweep_root, args.output, args.workers,
          angles=args.angles, velocities=args.velocities, kicks=args.kicks,
          duration_s=args.duration, kick_time_s=args.kick_time), indent=2))


if __name__ == "__main__":
    main()
