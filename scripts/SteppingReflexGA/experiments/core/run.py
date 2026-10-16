"""Shared experiment lifecycle: train, validate all individuals, then test once."""
import argparse
import csv
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
from SpikingNN.core import Izh_net
from . import genome
from .afferents import PARAMETER_NAMES, PARAMETERS
from .cases import banks, balance_cases, SEEDS
from .commands import decode_command, COMMAND_SIZE, commanded_parameters, input_currents
from .config import DT_MS, MODEL, FITNESS, POPULATION_SCHEMA, MECHANICS
from .dynamics import rollout, METRIC_COLUMNS, TRACE_COLUMNS
from .evolution import optimize, Schedule
from .population import PopulationEvaluator, load_controller_population, select_controller

EXPERIMENTS = ("equilibrium", "small_nm_weights", "angle_setpoint")
CASE_COLUMNS = ["target_rad", "theta0_rad", "omega0_rad_s", "kick_time_s", "kick_delta_omega"]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path, rows):
    rows = list(rows)
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_hashes():
    files = sorted(Path(__file__).parent.glob("*.py")) + [Path(Izh_net.__file__)]
    return {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def metric_rows(metrics):
    return [dict(trial=i, **dict(zip(METRIC_COLUMNS, map(float, row))))
            for i, row in enumerate(metrics)]


def run(experiment, output, *, population_size=30, generations=100, seed=42,
        workers=1, population_from=None, duration_s=10., case_banks=None, plot=True,
        schedule=Schedule(), schedule_generations=None, resume_state=None,
        checkpoint_every=0, verbose=True):
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment}")
    if population_size < 3 or generations < 1 or workers < 1 or seed < 0:
        raise ValueError("Need population >= 3, generations >= 1, workers >= 1, seed >= 0")
    if not np.isfinite(duration_s) or duration_s < DT_MS/1000:
        raise ValueError("Invalid duration")
    if (experiment == "angle_setpoint") != (population_from is not None):
        raise ValueError("Only angle_setpoint requires --population-from from a balance run")
    controllers_from = load_controller_population(population_from) if population_from else None
    case_banks = banks(experiment) if case_banks is None else case_banks
    if set(case_banks) != {"training", "validation", "test"}:
        raise ValueError("Provide separate training, validation and test banks")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    hashes = source_hashes()
    settings = dict(experiment=experiment, model=MODEL, fitness=FITNESS, dt_ms=DT_MS,
                    duration_s=duration_s, population=population_size, generations=generations,
                    seed=seed, workers=workers, schedule=asdict(schedule), case_seeds=SEEDS,
                    schedule_generations=schedule_generations or generations,
                    continued_from_generation=0 if resume_state is None else int(resume_state["generation"]),
                    afferent_parameters=dict(zip(PARAMETER_NAMES, PARAMETERS)),
                    hyperparameters=asdict(MECHANICS),
                    python=platform.python_version(),
                    packages={k: importlib.metadata.version(k) for k in ("numpy", "numba")},
                    source_sha256=hashes)
    if population_from:
        settings["population_from"] = str(Path(population_from).resolve())
        settings["population_sha256"] = hashlib.sha256(Path(population_from).read_bytes()).hexdigest()
    write_json(output / "run.json", settings)
    for name, cases in case_banks.items():
        write_csv(output / f"{name}_cases.csv", [dict(zip(CASE_COLUMNS, map(float, c))) for c in cases])

    fixed_controller = None
    with PopulationEvaluator(workers, cache_size=10000) as evaluator:
        if controllers_from is not None:
            selection_cases = balance_cases(SEEDS["controller_selection"])
            fixed_controller, selected_index, selection_scores, selection_metrics = select_controller(
                controllers_from, selection_cases, evaluator, duration_s)
            np.savez_compressed(output / "controller_selection.npz", cases=selection_cases,
                                scores=selection_scores, metrics=selection_metrics,
                                selected_index=selected_index, physical_controller=fixed_controller)
            write_json(output / "controller_selection.json", dict(
                selected_index=selected_index, candidates=len(controllers_from),
                selection_fitness=float(selection_scores[selected_index]),
                selection_success=float(selection_metrics[selected_index, :, 5].mean()),
                note="Controller frozen for all command generations; selected on an independent balance bank."))
            genome.save_genome(output / "fixed_controller.json", genome.encode(fixed_controller),
                               model=MODEL, dt_ms=DT_MS, source_population_index=selected_index)
            center = np.full(COMMAND_SIZE, .5)
        else:
            domain = genome.ControllerDomain(experiment == "small_nm_weights")
            center = domain.initial()

        def decode_population(population):
            if fixed_controller is not None:
                return (np.repeat(fixed_controller[None, :], len(population), axis=0),
                        np.array([decode_command(unit) for unit in population]))
            return np.array([domain.decode(unit) for unit in population]), None

        def evaluate(population):
            controllers, coefficients = decode_population(population)
            return evaluator.evaluate(controllers, coefficients, case_banks["training"], duration_s)

        def progress(history):
            write_csv(output / "history.csv", history)
            row = history[-1]
            if verbose:
                print(f"{experiment}: generation {row['generation']}/{generations}, "
                      f"fitness={row['best_fitness']:.6f}, success={row['best_success']:.1%}", flush=True)

        def checkpoint(generation, population, scores, metrics, history, rng_state):
            if checkpoint_every and (generation % checkpoint_every == 0 or generation == generations):
                temporary = output / "checkpoint.tmp.npz"
                np.savez_compressed(temporary, generation=generation, population=population,
                                    scores=scores, history_json=json.dumps(history),
                                    rng_state=json.dumps(rng_state))
                temporary.replace(output / "checkpoint.npz")

        population, scores, evaluation, history, rng_state = optimize(
            center, evaluate, size=population_size, generations=generations, seed=seed,
            schedule=schedule, schedule_generations=schedule_generations,
            resume_state=resume_state, on_generation=progress, on_state=checkpoint)
        controllers, coefficients = decode_population(population)
        validation_scores, validation = evaluator.evaluate(
            controllers, coefficients, case_banks["validation"], duration_s)
        order = np.argsort(-validation_scores, kind="stable")
        winner = int(order[0])
        best_coefficients = None if coefficients is None else coefficients[winner]
        test_scores, test = evaluator.evaluate(
            controllers[winner:winner+1], None if coefficients is None else coefficients[winner:winner+1],
            case_banks["test"], duration_s)

    np.savez_compressed(output / "final_population.npz", schema=POPULATION_SCHEMA,
                        model=MODEL, dt_ms=DT_MS, experiment=experiment,
                        population=population, physical_controllers=controllers,
                        command_coefficients=np.empty((0, 2, 3)) if coefficients is None else coefficients,
                        scores=scores, evaluation=evaluation, validation_scores=validation_scores,
                        validation_metrics=validation, selected_index=winner,
                        generation=generations, history_json=json.dumps(history),
                        rng_state=json.dumps(rng_state))
    p = controllers[winner]
    genome.save_genome(output / "best_genome.json", genome.encode(p),
                       experiment=experiment, model=MODEL, dt_ms=DT_MS,
                       selected_index=winner, selection="validation of every final-population individual")
    if best_coefficients is not None:
        write_json(output / "current_law.json", dict(
            model=MODEL, dt_ms=DT_MS, normalized=population[winner].tolist(),
            coefficients=best_coefficients.tolist(), physical_controller=p.tolist(),
            genes=[dict(index=3*i+j, neuron=i, power=j+1,
                        value=float(best_coefficients[i,j]),
                        description=f"Коэффициент тока нейрона {i} при x^{j+1}; x=целевой угол/45 градусов.",
                        unit="ед. тока модели") for i in range(2) for j in range(3)],
            target_degrees=[-45., 45.], law="Q_aff @ (aff_static(0)-aff_static(target)) + C @ [x,x^2,x^3]; x=target_deg/45"))
        write_csv(output / "input_currents.csv", [dict(target_deg=float(angle),
                  I_flex=float(currents[0]), I_ext=float(currents[1]))
                  for angle in np.linspace(-45., 45., 181)
                  for currents in [input_currents(p, best_coefficients, np.deg2rad(angle))]])
    write_csv(output / "validation_ranking.csv", [dict(rank=rank, population_index=int(index),
              training_fitness=float(scores[index]), validation_fitness=float(validation_scores[index]),
              success=float(validation[index, :, 5].mean())) for rank, index in enumerate(order, 1)])
    write_csv(output / "test_metrics.csv", metric_rows(test[0]))
    traces = {}
    for i, case in enumerate(case_banks["test"]):
        # One representative per target; metrics above use all trials, full step.
        if i and case[0] in case_banks["test"][:i, 0]:
            continue
        commanded = p if best_coefficients is None else commanded_parameters(p, best_coefficients, case[0])
        traces[f"case_{i}"] = rollout(commanded, case[1:], target_rad=case[0],
                                     duration_s=duration_s, stride=10)[1]
    np.savez_compressed(output / "traces.npz", columns=np.array(TRACE_COLUMNS), **traces)
    result = dict(experiment=experiment, model=MODEL, dt_ms=DT_MS, selected_index=winner,
                  training_fitness=float(scores[winner]), validation_fitness=float(validation_scores[winner]),
                  test_fitness=float(test_scores[0]), test_success=float(test[0, :, 5].mean()),
                  test_late_rmse_deg=float(np.rad2deg(test[0, :, 7]).mean()),
                  evaluations=population_size*generations,
                  population=population_size, generations=generations, seed=seed,
                  mutation_probability=schedule.probability_start,
                  free_genes=len(center), fixed_controller=fixed_controller is not None)
    if source_hashes() != hashes:
        raise RuntimeError("Calculation sources changed during this run")
    write_json(output / "result.json", result)
    (output / "REPORT.md").write_text(
        f"# {experiment}\n\nМодель: `{MODEL}`, шаг: {DT_MS} мс.\n\n"
        f"Поколений: {generations}; особей: {population_size}; seed: {seed}.\n"
        f"Индекс победителя по validation: {winner}.\n\n"
        f"Fitness на независимом test: {result['test_fitness']:.6f}; "
        f"успешность: {result['test_success']:.1%}.\n\n"
        "Результат одного seed; для оценки вариативности требуются независимые повторения.\n")
    if plot:
        from .plotting import plot_run
        plot_run(output, history, traces, case_banks["test"], p, best_coefficients)
    return result


def main(experiment):
    parser = argparse.ArgumentParser(description=f"{experiment}: shared GA, full Afferents, dt=0.1 ms")
    parser.add_argument("--population", type=int, default=30)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--mutation-probability", type=float, help="Constant per-gene mutation probability; otherwise use annealing")
    parser.add_argument("--output", type=Path, required=True, help="A new directory; existing runs are never overwritten")
    if experiment == "angle_setpoint":
        parser.add_argument("--population-from", type=Path, required=True)
    args = parser.parse_args()
    schedule = Schedule() if args.mutation_probability is None else Schedule(
        probability_start=args.mutation_probability, probability_end=args.mutation_probability)
    result = run(experiment, args.output, population_size=args.population,
                 generations=args.generations, seed=args.seed, workers=args.workers,
                 population_from=getattr(args, "population_from", None), schedule=schedule)
    print(json.dumps(result, indent=2))
