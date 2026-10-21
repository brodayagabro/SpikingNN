# B_balance: balance from large deviations

[Русский](README.md) · [Output map and notebook functions](../README.en.md) · [Model](../MODEL_METHODS.en.md)

Two neurons, two muscles and full Afferents. Zero applied currents; no PD/PID. Each neuromuscular weight satisfies `0.0001 ≤ w ≤ 0.02`. Initial values and fixed `a1=0.2 m`, `a2=0.05 m` are specified in `genome_cfg.json`. The integration step is always 0.1 ms.

Training/validation/test banks contain 24/48/96 cases with independent fixed seeds. Initial angles span the entire −50…+50° interval and speeds −1…+1 rad/s, including small deviations. Seeded Latin hypercube sampling is used; training additionally includes −50°, 0°, +50° at zero speed. No impulses are applied. A/B banks are identical and remain fixed during training. Duration is 10 s; success and fitness are defined in the shared model document.

From the repository root:

```bash
.venv/bin/python scripts/AfferentBalance2026/experiments/B_balance/run_experiment.py --population 50 --generations 200 --workers 4
.venv/bin/python scripts/AfferentBalance2026/experiments/B_balance/run_sweep.py --workers 4
```

`--config` selects a configuration and `--output` a new directory; existing directories are not overwritten. Single-run options include `--seed`, `--mutation-probability`; sweep options include `--populations`, `--generations`, `--probabilities`, `--seeds`. A short `--duration` is intended for implementation checks, not scientific comparison.

Set `GENOME_PATH` in `explore_genome.ipynb` to inspect a trained solution. Select any saved generation and ranked individual, then plot phase grids, time series and all states for one case. By default, `results/best_genome.json` is loaded when available; `GENOME_PATH=None` selects the untrained configuration. The current study is saved in `results/sweep_20261021/`; its post-analysis index is `results/README.en.md`.

## Ranges and local results

Set two simple ranges inside `simulation` in `genome_cfg.json`. Asymmetric ranges are supported; there is no excluded central interval.
```json
"initial_conditions": {
  "angle_range_deg": [-50.0, 50.0],
  "speed_range_rad_s": [-1.0, 1.0]
}
```
Actual samples are stored in `cases.npz` and plotted in `initial_conditions.png` within the sweep. This experiment’s `results/` contains its own calculations, complete generation histories, best genomes, full traces, heatmaps and RMSE. Run `../../analyze_abc_study.py` after the sweep for auditing and diagnostic grids.

After analysis completes:

- `results/initial_baseline.json`: the initial genome on the same banks.
- `results/diagnostics_20261021/target_+00.00/multipanel.png`: phase trajectories, angles and signals for one example.
- `results/paired_comparison.csv`: matched-budget A/B comparison.

Genome RMSE measures normalized parameter differences, not differences in dynamics or stability. Single-example signals retain the full 0.1 ms recording step; phase curves are decimated for display only.
