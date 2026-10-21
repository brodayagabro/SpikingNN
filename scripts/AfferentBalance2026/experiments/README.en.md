# Experiment layout

[Русский](README.md)

```text
experiments/
  core/                     genome.py, fitness.py, evolution.py, run.py
  simulation.py             one Numba model, library reference, condition banks
  visualization.py          solution loading and shared plots
  A_balance/                run_experiment.py, run_sweep.py, genome_cfg.json
  B_balance/                run_experiment.py, run_sweep.py, genome_cfg.json
  C_setpoint_balance/       the same entry files plus polynomial_study.py
```

Each experiment also contains `README.md`, `README.en.md`, and `explore_genome.ipynb`. Calculations create a local `results/` tree. There is no nested `source/` directory or wrapper import chain. Core has no current-law implementation; shared physics lives in `simulation.py` and current laws in C.

## A/B output files

| File within a run | Contents |
|---|---|
| `genome_cfg.json` | Full configuration, including English descriptions of 52 fields |
| `run.json`, `source_snapshot.zip` | Actual run settings, hashes and source snapshot |
| `cases.npz` | Training, validation, test and demonstration conditions |
| `generations/generation_XXXX.npz` | Every evaluated population: 50 normalized and 52 physical parameters, metrics, RNG state |
| `history.csv`, `progress.json` | Convergence and last completed generation |
| `final_population.npz` | Complete last population with training and validation scores |
| `best_genome.json` | Final-population validation winner with self-contained configuration |
| `validation_ranking.csv`, `test_metrics.csv`, `result.json` | Ranking, independent winner test and summary |
| `best/rank_001/` etc. | Three validation leaders: `genome.json`, five `case_XXX.npz` files, `metrics.csv` |

Winner trajectories retain the full 0.1 ms resolution and initial state: ±50°, ±25°, 0°, zero initial speed. NPZ `columns` and `metric_columns` name every channel. Recorded variables include angle, speed, neuron potentials and recovery states, spikes, synaptic currents, muscle calcium, activation and forces, six afferent channels, and muscle torque. Failed trajectories are retained and stop after crossing ±90°.

`run_sweep.py` uses populations 10/30/50/100, constant per-free-gene mutation probabilities 0.1/0.3/0.6, seed 42, and generations 30/60/100/150/200. Each size/probability/seed combination is one uninterrupted 200-generation run; earlier generations are checkpoints of that run, not independently scheduled short runs. Outputs include `checkpoints/epochs_XXX/`, `checkpoints.json`, `summary.csv`, heatmaps and `genome_rmse.npz`. RMSE uses 50 normalized free coordinates and excludes fixed geometry. One seed cannot establish statistical robustness.

## Notebooks

The local `results/best_genome.json` is loaded by default when it exists. `GENOME_PATH=None` selects the **untrained** configured genome. You can select `best_genome.json`, `final_population.npz` or a generation file; `RANK` selects an individual. Generation ranking uses training; final ranking uses validation. Saved solution configurations are respected.

`run_on_grid` evaluates initial conditions; `phase_portrait` plots phase trajectories; `view_states` shows all state groups for one example; `scan_parameter` evaluates values of a free parameter. Defaults are a 10×10 grid, −50…50° and −1…1 rad/s, plus ten angle traces at zero initial speed. A/B training requires explicit `RUN_TRAINING` or `RUN_SWEEP`, both false by default. C only runs replays.

## October 21, 2026 study

A/B train across the full −50…50° × −1…1 rad/s rectangle, including small angles. Ranges are configured through `simulation.initial_conditions`. Each experiment’s `results/sweep_20261021/` stores 12 GA lines and 60 checkpoints. After analysis, `results/best_genome.json` is loaded automatically by its notebook; `results/diagnostics_20261021/` holds an independent phase grid. C uses the selected A network and configured currents without fitting a current law.

The study is complete. Result indexes: [A](A_balance/results/README.en.md), [B](B_balance/results/README.en.md), [C](C_setpoint_balance/results/README.en.md). Summary late RMSE is averaged across episodes; a fall receives a 90° penalty, so B's value must not be interpreted as the tracking error of successful trajectories alone.

To reproduce postprocessing after new training runs finish (diagnostic directories must not exist):

```bash
.venv/bin/python scripts/AfferentBalance2026/analyze_abc_study.py A_balance
.venv/bin/python scripts/AfferentBalance2026/analyze_abc_study.py B_balance
.venv/bin/python scripts/AfferentBalance2026/analyze_abc_study.py compare
.venv/bin/python scripts/AfferentBalance2026/analyze_abc_study.py C_setpoint_balance
```

`--sweep` selects a series directory within each local `results/`; the default is `sweep_20261021`. Analysis does not overwrite existing trajectories. It checks every generation, genome bounds, source hashes and an independent repeated test of the selected solution.
