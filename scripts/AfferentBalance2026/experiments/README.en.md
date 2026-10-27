# Experiment layout

[Русский](README.md)

```text
experiments/
  core/                     genome.py, fitness.py, evolution.py, run.py
  simulation.py             one Numba model, library reference, condition banks
  visualization.py          solution loading and shared plots
  A_balance/                run_experiment.py, run_sweep.py, genome_cfg.json
  B_balance/                run_experiment.py, run_sweep.py, genome_cfg.json
  C_setpoint_balance/       the same entry files plus polynomial_study.py, joint_study.py
  D_trajectory/             controller.py, trajectory.py, run_experiment.py, run_sweep.py
```

Each experiment also contains `README.md`, `README.en.md`, and `explore_genome.ipynb`. Calculations create a local `results/` tree. There is no nested `source/` directory or wrapper import chain. Core has no current-law implementation; shared physics lives in `simulation.py` and current laws in C/D.

## A/B output files

| File within a run | Contents |
|---|---|
| `genome_cfg.json` | Full configuration, including English descriptions of 52 fields |
| `run.json`, `source_snapshot.zip` | Actual run settings, hashes and source snapshot |
| `cases.npz` | Training, validation, test and demonstration conditions |
| `generations/generation_XXXX.npz` | Every evaluated population: 50 (A) / 48 (B) normalized and 52 physical parameters, metrics, RNG state |
| `history.csv`, `progress.json` | Convergence and last completed generation |
| `final_population.npz` | Complete last population with training and validation scores |
| `best_genome.json` | Final-population validation winner with self-contained configuration |
| `validation_ranking.csv`, `test_metrics.csv`, `result.json` | Ranking, independent winner test and summary |
| `best/rank_001/` etc. | Three validation leaders: `genome.json`, five `case_XXX.npz` files, `metrics.csv` |

Winner trajectories retain the full 0.1 ms resolution and initial state: ±50°, ±25°, 0°, zero initial speed. NPZ `columns` and `metric_columns` name every channel. Recorded variables include angle, speed, neuron potentials and recovery states, spikes, synaptic currents, muscle calcium, activation and forces, six afferent channels, and muscle torque. Failed trajectories are retained and stop after crossing ±90°.

In A/B, `run_sweep.py` uses populations 10/30/50/100, constant per-free-gene mutation probabilities 0.1/0.3/0.6, seed 42, and generations 30/60/100/150/200. Each size/probability/seed combination is one uninterrupted 200-generation run; earlier generations are checkpoints of that run, not independently scheduled short runs. Outputs include `checkpoints/epochs_XXX/`, `checkpoints.json`, `summary.csv`, heatmaps and `genome_rmse.npz`. RMSE uses normalized free coordinates (50 for A, 48 for current B) and excludes every fixed field. One seed cannot establish statistical robustness.

## Notebooks

The local `results/best_genome.json` is loaded by default when it exists. `GENOME_PATH=None` selects the **untrained** configured genome. You can select `best_genome.json`, `final_population.npz` or a generation file; `RANK` selects an individual. Generation ranking uses training; final ranking uses validation. Saved solution configurations are respected.

`run_on_grid` evaluates initial conditions; `phase_portrait` plots phase trajectories; `view_states` shows all state groups for one example; `scan_parameter` evaluates values of a free parameter. Defaults are a 10×10 grid, −50…50° and −1…1 rad/s, plus ten angle traces at zero initial speed. A/B training requires explicit `RUN_TRAINING` or `RUN_SWEEP`, both false by default. C joint training requires `RUN_STUDY`, also false by default.

## October 21, 2026 study

A/B train across the full −50…50° × −1…1 rad/s rectangle, including small angles. Ranges are configured through `simulation.initial_conditions`. Each experiment’s `results/sweep_20261021/` stores 12 GA lines and 60 checkpoints. After analysis, `results/best_genome.json` is loaded automatically by its notebook; previous diagnostic folders were moved to trash at the user’s request. C uses the selected A network and configured currents without fitting a current law.

The study is complete. Result indexes: [A](A_balance/results/README.en.md), [B](B_balance/results/README.en.md), [C](C_setpoint_balance/results/README.en.md). Summary late RMSE is averaged across episodes; a fall receives a 90° penalty, so B's value must not be interpreted as the tracking error of successful trajectories alone.


Current B uses `0.02 ≤ w < 0.1` and fixed `A=0.37` for both muscles. [B postprocessing](../tools/select_sweep_winner.py) audits stored populations, selects the validation maximum and saves 110 full trajectories under the sweep’s `selected/` directory:

```bash
.venv/bin/python scripts/AfferentBalance2026/tools/select_sweep_winner.py scripts/AfferentBalance2026/experiments/B_balance/results/sweep_restored_weights_A037_20261021
```

Run this once after completion, with no existing `selected/`; the previous selection is preserved inside its original sweep.

## C: joint controller and command training, 22 October 2026

The 52-field controller remains the common base. C fixes both muscle A values at 0.37, giving 48 mutable controller coordinates. Cubic command functions append eight genes: 60 physical fields / 56 mutable coordinates. Quadratics use 58/54; a five-target current table uses 62/58. `joint_study.py` contains both `joint` and `staged` (table → bounded least squares → joint refinement) pipelines. Shared core retains physics-independent genetic operators and fitness; current-law logic stays in C.

Five targets are evaluated per individual and generation. Training/validation/test contain 40/108/408 episodes; only the overall validation winner is tested. `best_genome.json` stores the combined schema, `best_controller.json` exports the base 52-field block, and `current_law.json` exports the command function. All generations retain controller and command genes. The C notebook reads `results/current_joint.json`, including a clear pilot label. See [C documentation](C_setpoint_balance/README.en.md) for commands and the complete output map. The previous zero-current C study above remains a historical control. The new 20-individual, 30-generation pilot achieved 76/408 tests; it has not solved the full target range.

New full C sweep: [joint_sweep_20261022](C_setpoint_balance/results/joint_sweep_20261022/README.en.md), 16 × 200 epochs.

C retraining with τ_c, τ₁ ≥5.1 ms: [protocol and results](C_setpoint_balance/results/joint_sweep_tau_gt5_20261022/README.en.md), 4 × 200 generations, seed 43. Force pulsations remain.

## D: continuous tracking, October 23, 2026

[Documentation and API](D_trajectory/README.en.md), [notebook](D_trajectory/explore_genome.ipynb).
The shared Numba physical step in `simulation.py` serves both episodic integration and the stateful `AngleController`. The class accepts angles within −45…45°, preserves every state and returns actual motion. D owns references, current filtering and tracking scores. No trajectory logic was added to core.

The study compares four retained C winners × three filter times, then 30 individuals × 80 generations for `command_only` and `joint`. Banks contain 8/10/18 episodes; separate diagnostics cover three amplitudes × three periods. Every generation and complete 0.1 ms dynamics are saved. D success is relative to a moving target and is not equivalent to A/B/C static success.
