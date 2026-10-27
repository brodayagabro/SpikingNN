<!-- muscle-tau50-20261026 -->

# Retraining with muscle relaxation times ≥50 ms

Current winner: `joint`. Completed tests: **96/96**; strict successes: **96/96**. Selected by validation only. Both muscle A parameters are fixed at 0.37. Both muscles use tau_c in [50,100] ms and tau_1 in [50,200] ms. Integration step: 0.1 ms. No PD/PID controller or torque multiplier.

[Current results and file guide](results/README.en.md). [Full article](../../report/REPORT.en.md).

## Historical documentation (before retraining)

<!-- retraining-20261026 -->

# Retraining on corrected tensile mechanics

Current winner: `B_transfer_refined`. Completed tests: **96/96**; strict successes: **96/96**. Selected by validation only. Both muscle A parameters are fixed at 0.37. Integration step: 0.1 ms. No PD/PID controller or torque multiplier.

[Current results and file guide](results/README.en.md). [Full article](../../report/REPORT.en.md).

## Historical documentation (before retraining)

# A_balance: balance throughout −50…+50°

[Русский](README.md) · [Output map and notebook functions](../README.en.md) · [Model](../MODEL_METHODS.en.md)

Two neurons, two muscles and full Afferents. Zero applied currents; no PD/PID. Each neuromuscular weight satisfies `0.0001 ≤ w ≤ 1`. Initial values and fixed `a1=0.2 m`, `a2=0.05 m` are specified in `genome_cfg.json`. The integration step is always 0.1 ms.

Training/validation/test banks contain 24/48/96 cases with independent fixed seeds. Initial angles span the entire −50…+50° interval and speeds −1…+1 rad/s, including small deviations. Seeded Latin hypercube sampling is used; training additionally includes −50°, 0°, +50° at zero speed. No impulses are applied. A/B banks are identical and remain fixed during training. Duration is 10 s; success and fitness are defined in the shared model document.

From the repository root:

```bash
.venv/bin/python scripts/AfferentBalance2026/experiments/A_balance/run_experiment.py --population 50 --generations 200 --workers 4
.venv/bin/python scripts/AfferentBalance2026/experiments/A_balance/run_sweep.py --workers 4
```

`--config` selects a configuration and `--output` a new directory; existing directories are not overwritten. Single-run options include `--seed`, `--mutation-probability`; sweep options include `--populations`, `--generations`, `--probabilities`, `--seeds`. A short `--duration` is intended for implementation checks, not scientific comparison.

Set `GENOME_PATH` in `explore_genome.ipynb` to inspect a trained solution. Select any saved generation and ranked individual, then plot phase grids, time series and all states for one case. By default, `results/best_genome.json` is loaded when available; `GENOME_PATH=None` selects the untrained configuration. The current study is saved in `results/sweep_20261021/`; its post-analysis index is `results/README.en.md`.

## Tensile-torque correction, 2026-10-26

The active mechanics is `SpikingNN.Afferents/v3-tensile`: positive tension gives `M = -F_flex*h_flex + F_ext*h_ext`. The retained old winner completed **0/96** original test episodes after the correction; no retraining was performed. Historical training scores below describe the old sign. [New metrics and phase portraits](results/torque_sign_check_20261026/README.en.md).

Restart the Jupyter kernel and reopen the updated notebook before running it. Old outputs were backed up under `results/torque_sign_check_20261026/notebooks_before_correction/` and cleared from the active notebooks.

## Ranges and local results

Set two simple ranges inside `simulation` in `genome_cfg.json`. Asymmetric ranges are supported; there is no excluded central interval.
```json
"initial_conditions": {
  "angle_range_deg": [-50.0, 50.0],
  "speed_range_rad_s": [-1.0, 1.0]
}
```
Actual samples are stored in `cases.npz` and plotted in `initial_conditions.png` within the sweep. This experiment’s `results/` contains its own calculations, complete generation histories, best genomes, full traces, heatmaps and RMSE. The previous analysis script was removed during reorganization; stored results remain in `results/`.

After analysis completes:

- `results/initial_baseline.json`: the initial genome on the same banks.
- `results/paired_comparison.csv`: matched-budget A/B comparison.

Genome RMSE measures normalized parameter differences, not differences in dynamics or stability. Single-example signals retain the full 0.1 ms recording step; phase curves are decimated for display only.

`results/diagnostics_20261021/` was moved to trash at the user’s request. Genomes and training histories remain; use the notebook for new trajectories. Manifest: `../B_balance/results/cleanup_diagnostics_20261021.json`.
