# Project tools

[Русский](README.md) · [Project](../README.en.md)

This directory contains shared helpers and project maintenance commands.
Experiment entry points remain in `experiments/A_balance`, `B_balance`,
`C_setpoint_balance` and `D_trajectory`; figure editing lives in the [article notebook](../report/article_figures.ipynb).

| File | Purpose |
|---|---|
| [select_sweep_winner.py](select_sweep_winner.py) | Select a completed B sweep winner by validation, audit populations and save full replays. |
| [analyze_joint_sweep.py](analyze_joint_sweep.py) | Audit both C method winners, generations and dynamics; validation heatmaps, genome RMSE and RU/EN indexes. |
| [compare_muscle_relaxation.py](compare_muscle_relaxation.py) | Compare frozen winners of two C studies on one test bank; relaxation times, force variation and unsmoothed oscillograms. |
| [validate_layout.py](validate_layout.py) | Validate experiment structure, configurations, notebooks and documentation. |
| [analyze_unscaled.py](analyze_unscaled.py) | Audit the historical unscaled study and rebuild its report. |
| [runtime_support.py](runtime_support.py) | Resolve saved-result paths and load verified immutable numerical runtimes. |
| [layout.py](layout.py) | Resolve image and archived-source paths and verify checksums. |

Commands from the SpikingNN repository root:

```bash
.venv/bin/python scripts/AfferentBalance2026/tools/validate_layout.py
.venv/bin/python scripts/AfferentBalance2026/tools/analyze_unscaled.py --help
```

Paths are resolved relative to the sources, independently of the working directory.
Historical analysis requires the corresponding data previously moved to the desktop
trash; moving these utilities does not restore those datasets.

Other modules should import from `scripts.AfferentBalance2026.tools`.
Test collection excludes generated directories through [pytest.ini](../pytest.ini);
test import paths are configured in [tests/conftest.py](../tests/conftest.py).

For C, run `experiments/C_setpoint_balance/sweep_diagnostics.py <sweep directory>` after training, then `tools/analyze_joint_sweep.py <sweep directory>`. The second step verifies all trajectories and updates the notebook pointer only after the audit passes.

For D: `experiments/D_trajectory/audit_trajectory.py <D study directory> --workers 18`. It writes `audit.json` without altering archived sources. Use `validate_layout.py --allow-notebook-outputs` for local structure checks that preserve user-generated notebook output.

## Check retained genomes after the torque correction

`recheck_tensile_genomes.py` replays the unchanged A/B/C/D winners on their original test banks, a 10×10 initial-state grid and near-target states. It deduplicates identical copies and writes separate metrics, traces and phase portraits under each experiment’s `results/`. It never retrains or replaces a winner. The default tag is `torque_sign_check_20261026`; an existing directory is never overwritten.

```bash
.venv/bin/python scripts/AfferentBalance2026/tools/recheck_tensile_genomes.py --tag torque_check_repeat
```

## Retraining with muscle times ≥50 ms

`prepare_muscle_times.py --genome OLD.json --config EXPERIMENT/genome_cfg.json --output NEW_INPUTS` preserves the source and creates four admissible starts: clipped/reference times, with or without approximate steady-gain compensation. Reference times come from the destination configuration (71/130 ms in the current series).

`retrain_tensile.py` trains from a self-contained prepared genome and saves every population. `retraining_figures.py --experiment FOLDER_NAME --run RUN_DIRECTORY` independently replays the winner, full phase grids and signal examples. `finalize_retraining.py` selects completed winners by validation and updates notebook defaults, preserving previous canonical files. The active study is `muscle_tau50_20261026`; exact run commands and flag meanings are in [the report](../report/REPORT.en.md).
