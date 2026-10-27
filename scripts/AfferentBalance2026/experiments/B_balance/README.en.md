<!-- muscle-tau50-20261026 -->

# Retraining with muscle relaxation times ≥50 ms

Current winner: `joint`. Completed tests: **96/96**; strict successes: **96/96**. Selected by validation only. Both muscle A parameters are fixed at 0.37. Both muscles use tau_c in [50,100] ms and tau_1 in [50,200] ms. Integration step: 0.1 ms. No PD/PID controller or torque multiplier.

[Current results and file guide](results/README.en.md). [Full article](../../report/REPORT.en.md).

## Historical documentation (before retraining)

<!-- retraining-20261026 -->

# Retraining on corrected tensile mechanics

Current winner: `seed_43`. Completed tests: **96/96**; strict successes: **96/96**. Selected by validation only. Both muscle A parameters are fixed at 0.37. Integration step: 0.1 ms. No PD/PID controller or torque multiplier.

[Current results and file guide](results/README.en.md). [Full article](../../report/REPORT.en.md).

## Historical documentation (before retraining)

# B_balance: balance from large deviations

[Русский](README.md) · [Output map and notebooks](../README.en.md) · [Model](../MODEL_METHODS.en.md)

Two neurons, two muscles and full nonlinear Afferents. Applied currents are zero; there is no PD/PID controller. Integration uses 0.1 ms steps and muscle torque has no ×10 multiplier.

The current `genome_cfg.json` restores the historical neuromuscular weight domain **`0.02 ≤ w < 0.1`** for both muscles, with initial value **0.08**. The upper endpoint is the largest float below 0.1. At the user's request, the physiological parameter **`A=0.37 N/ms`** is fixed in both muscles. `A`, `a1=0.2 m` and `a2=0.05 m` remain in the saved genome but are excluded from genetic operations. There are 52 physical fields and **48 mutable coordinates**. Field descriptions and mappings to model parameters are English.

Training, validation and independent testing use 24/48/96 cases across the entire **−50…+50° × −1…+1 rad/s** rectangle, including small deviations. Banks use Latin hypercube sampling with independent fixed seeds. Training also includes −50°, 0°, +50° at zero speed; no impulses are applied. Banks match A and remain fixed during training. Each episode lasts 10 s.

Configure ranges in `simulation.initial_conditions`:

```json
"initial_conditions": {
  "angle_range_deg": [-50.0, 50.0],
  "speed_range_rad_s": [-1.0, 1.0]
}
```

## Tensile-torque correction, 2026-10-26

The active mechanics is `SpikingNN.Afferents/v3-tensile`: positive tension gives `M = -F_flex*h_flex + F_ext*h_ext`. The retained old winner completed **0/96** original test episodes after the correction; no retraining was performed. Historical training scores below describe the old sign. [New metrics and phase portraits](results/torque_sign_check_20261026/README.en.md).

Restart the Jupyter kernel and reopen the updated notebook before running it. Old outputs were backed up under `results/torque_sign_check_20261026/notebooks_before_correction/` and cleared from the active notebooks.

## Running and selecting a solution

From the repository root:

```bash
.venv/bin/python scripts/AfferentBalance2026/experiments/B_balance/run_experiment.py --population 50 --generations 200 --workers 16
.venv/bin/python scripts/AfferentBalance2026/experiments/B_balance/run_sweep.py --workers 16 --output scripts/AfferentBalance2026/experiments/B_balance/results/sweep_restored_weights_A037_20261021
.venv/bin/python scripts/AfferentBalance2026/tools/select_sweep_winner.py scripts/AfferentBalance2026/experiments/B_balance/results/sweep_restored_weights_A037_20261021
```

The first two commands are alternative training modes: one run or the full sweep. The third processes a completed sweep; an existing `selected/` is not overwritten. Use a fresh `--output` for a repeat. `--config` selects a configuration. Single runs support `--seed`, `--mutation-probability`; sweeps support `--populations`, `--generations`, `--probabilities`, `--seeds`. A short `--duration` is for implementation checks only.

The full sweep uses populations 10/30/50/100, constant per-mutable-gene mutation probabilities 0.1/0.3/0.6 and seed 42: 12 separate 200-generation runs with checkpoints at 30/60/100/150/200. Each starts with a fresh population; the transferred A genome is not used. Each checkpoint selects the final population's validation winner; the overall winner maximizes that same validation score. Test and phase-grid outcomes do not guide selection.

## Retraining result

All 12 lines and 60 checkpoints completed. Validation winner: **50 individuals, mutation 0.1, generation 200, seed 42**. Independent test: **96/96**, mean late RMSE **0.004013°**. Phase grid: **100/100**; zero-speed runs: **10/10**. Flexor/extensor weights: **0.0493838 / 0.0528917**; both A values are exactly **0.37**. All 2400 generations passed the audit.

Across the 110 checked trajectories, peak absolute speed was **14.699 rad/s** and peak muscle forces were **114.833 / 87.438 N**. Successful balance does not establish the physiological realism of these transients. Measurements and file-integrity checks are stored in `selected/trace_audit.json`; the library-model comparison is in `selected/model_reference_check.json`.

## Output locations

All outputs remain in local `results/`, excluded from Git.

- `best_genome.json`: overall winner, embedded configuration, English field descriptions and provenance.
- `current.json`, `summary.json`, `README.md` / `README.en.md`: current sweep pointer, scores and index.
- `sweep_restored_weights_A037_20261021/summary.csv`: all 60 checkpoints, alongside heatmaps and `genome_rmse.npz` / PNG / PDF.
- `population_*/mutation_*/seed_*/` within the sweep: all 200 populations, source snapshots and hashes, condition banks, three best genomes and their dynamics at each checkpoint.
- `selected/` within the sweep: winner, repeated independent test, all-generation audit, `phase.png/pdf`, `angles.png/pdf`, and `limb`, `neurons`, `muscles`, `afferents` plots.
- `selected/traces/`: 100 phase trajectories from 10 angles −50…+50° × 10 speeds −1…+1 rad/s, plus 10 trajectories at zero speed. All states are recorded every 0.1 ms. Subsystem signals show one example starting at −27.78° and zero speed.

Set `GENOME_PATH` in `explore_genome.ipynb` to the selected genome, final population or any generation file. `RANK` selects an individual; `run_on_grid` evaluates a grid; `phase_portrait` and `view_states` display dynamics; `scan_parameter` varies a mutable parameter. Fixed A requires a separate configuration and cannot be changed with that function. `GENOME_PATH=None` selects the untrained genome. Explicitly enable `RUN_TRAINING` or `RUN_SWEEP` to retrain.

Genome RMSE uses the 48 normalized mutable coordinates; it does not measure differences in dynamics. Phase lines are decimated for display only. One seed and a finite grid do not establish global stability.

## Previous calculations

`results/sweep_20261021/` and earlier notebook runs retain their original `w≤0.02` bounds and mutable A in embedded configurations. Once the new winner is published, previous selection metadata is stored under `sweep_20261021/previous_selection/`. Comparison changes two factors simultaneously: the w domain and fixing A. It cannot isolate the effect of changing weights alone.

The [historical diagnosis](DIAGNOSIS.en.md) explains the earlier 38/100 result and comparison with the old protocol. `diagnosis_20261021` and `diagnostics_20261021` were moved to trash at the user's request; see `results/cleanup_diagnostics_20261021.json`. Interrupted training with the previous mutable A was also moved to trash and is excluded from the new results.
