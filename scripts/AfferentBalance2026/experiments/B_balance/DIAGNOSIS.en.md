# Why B reaches only 38–40%: comparison on 21 October 2026

> Historical diagnosis before the protocol change. Its result folders were moved to trash at the user’s request; paths below describe their former locations. Manifest: `results/cleanup_diagnostics_20261021.json`. Current B uses `0.02 ≤ w < 0.1` and fixed `A=0.37`; scores below belong to the previous genomes.

[Русский](DIAGNOSIS.md) · [Experiment B](README.en.md)

The current B runs found weak, one-sided controllers. A good controller exists within the original `0.0001 ≤ w ≤ 0.02` domain: transferring the selected A controller with coordinated changes to `w` and `k` achieved 96/96 independent tests and 100/100 phase-grid cases. This is a diagnostic transfer of an existing controller, **not new B training**. Original training results and selected genomes were not replaced.

## Paired comparison

| Controller | Training: population / mutation / generation | Old test, 48 | New test, 96 | Common phase grid, 100 |
|---|---|---:|---:|---:|
| Historical `small_nm_weights`, October 16 | 100 / 0.1 / 200 | 48/48 | 78/96 | 75/100 |
| Selected B sweep controller, October 21 | 30 / 0.1 / 100 | 12/48 | 39/96 | 40/100 |
| B from `notebook_run_20261021_2024` | 50 / 0.3 / 200 | 10/48 | 35/96 | 38/100 |
| Historical controller: `w` and `k` mapped into B | No training | 48/48 | 78/96 | 75/100 |
| Selected A controller: `w` and `k` mapped into B | No training | 48/48 | 96/96 | 100/100 |

The common phase grid combines ten angles from −50 to +50° with ten speeds from −1 to +1 rad/s. All episodes use 10 s, 0.1 ms, zero applied currents, full Afferents, no PD and no torque multiplier. Success requires reaching ±3° and ±0.35 rad/s by 3 s and staying inside until the episode ends. New test and phase-grid conditions were not used for additional training. Conclusions concern these finite banks and selected controllers.


The reported 38/100 belongs to the separate notebook run. At inspection, `GENOME_PATH` selected `results/notebook_run_20261021_2024/best_genome.json`, not the sweep's `results/best_genome.json`. Both new controllers fail all 50 positive-angle grid cases; they succeed on 38/50 and 40/50 negative-angle cases, respectively.

## Changes from the earlier successful study

| Setting | Historical `small_nm_weights` | Saved B study, October 21 |
|---|---|---|
| Each `w` | `0.02 ≤ w < 0.1` | `0.0001 ≤ w ≤ 0.02` |
| Initial `w` | 0.08 | 0.01 |
| Training angles | −8…+8° | −50…+50° |
| Initial speeds | −0.15…+0.15 rad/s | −1…+1 rad/s |
| Perturbations | Half of episodes, at 4–6 s | None |
| Train / validation / test counts | 12 / 24 / 48 | 24 / 48 / 96 |
| Model, mechanics, step, success rule | Full Afferents, ×1, 0.1 ms | Unchanged |

The historical 0.1 upper bound was imposed by `ControllerDomain(small_weights=True)`. Old JSON `low/high` fields describe the canonical genome, so `high=1.5` is not the actual bound of that training run. The effective domain is exported to `historical_domain.npz`.

The initial population places about 75% of candidates near the configured initial genome and samples the rest uniformly in normalized coordinates. For logarithmic `w`, the median of this uniform part changed from 0.04472 to 0.001414, a factor of about 31.6. The initial value fell eightfold. Thus initialization and search geometry changed substantially despite an unchanged population/mutation sweep grid and fitness formula. Quantifying each factor's effect on convergence requires separate repeated-seed studies.

## Weak parameters in the selected controllers

| Controller | `w_F` / `w_E` | `k_F` / `k_E` | Maximum F / E force, N |
|---|---|---|---|
| Historical | 0.05024 / 0.07700 | 0.79563 / 0.68560 | 15.30 / 9.08 |
| B sweep | 0.01713 / 0.00777 | 0.55384 / 0.36503 | 11.51 / 21.30 |
| B notebook | 0.00328 / 0.00443 | 0.47388 / 0.45550 | **0.563 / 16.36** |
| A mapped into B | 0.02000 / 0.02000 | 0.19981 / 0.12880 | 62.48 / 41.36 |

The equation `dF/dt = A N x − F/τ₁`, with `0 ≤ x ≤ 1`, bounds force from zero initial force by `F ≤ A N τ₁`. The notebook flexor has `N=7.72178`, `A=0.00461702`, `τ₁=15.80415 ms`: even at saturation it cannot exceed 0.56344 N. Balancing gravity at +50° and zero speed with the flexor alone requires **8.67332 N**. This genome's ideal saturated flexor, with the extensor off, has a positive static angle limit of only about +3.596°. This bounds that individual, not the whole B domain.

The sweep winner has greater nominal force but differs substantially in feedback, cross projections P and activation times. Its force ceiling alone does not explain every dynamic failure. The controlled interventions here separately clip `w` and jointly rescale `w/k`; all 52 coordinates are recorded in `parameters.csv`.

## Why the A-to-B transfer is admissible

The model uses

$$
\dot C = w u - C/\tau_c,\qquad x = \frac{C^m}{C^m+k^m}.
$$

Starting from `C(0)=0`, replacing `w′=s w` and `k′=s k` gives `C′=s C` while preserving `x`. In exact arithmetic this preserves force, afferent, neural and mechanical dynamics; only the calcium scale changes. Each muscle of the selected A genome uses `s=0.02/w`. The resulting `k=0.1998051072` and `0.1288020123` lie within `[0.1,2]`; all other 48 fields are unchanged. Every B-domain bound and fixed geometry value was verified.

Floating-point trajectories may differ slightly because of rounding and spike thresholds. The transferred A genome has late RMSE 0.01816° on the new grid and 0.01847° on the new test, with success on every tested case. Clipping its weights alone also gives 100/100 grid successes but RMSE 0.516° and only 43/48 on the old perturbed test, so that is a different intervention.

With free `k`, `P`, `A`, `N` and `τ₁`, a small `w` does not by itself imply weak actuation. A useful next training step is to insert the admissible transferred controller into the initial population, or search the ratio `w/k` with coordinated mutations. A study specifically targeting weakened neuromuscular transmission should state which compensating parameters remain free. Enlarging the weight cap is unnecessary for a good solution to exist in the present domain.

## Verification and output files

All 71 frozen runtime source files passed SHA256 verification. The historical genome was replayed independently with old and current kernels: all nine metrics on the old test and common grid matched exactly, as did all 17 common channels of a dense 0.5 s trace. No numerical regression from the reorganization was found in these comparisons. Two new tests verify rescaling invariance and rejection when the required `k` leaves its domain.

Within `results/diagnosis_20261021/`:

- `comparison.csv`, `summary.json`: all eight controllers/interventions on five banks;
- `parameters.csv`, `muscle_parameters.csv`: all 52 fields and derived muscle parameters;
- `*_metrics.npz`: actual conditions, genomes and per-episode metrics;
- `model_parity.json`, `frozen_reference.npz`, `historical_domain.npz`: historical-kernel verification and effective bounds;
- `A_rescaled_genome.json`: B-admissible genome reaching 100/100; `historical_B_rescaled_genome.json`: transferred historical genome;
- `B_sweep/`, `A_rescaled/`: full 0.1 ms traces from −50° and +50° at zero speed;
- `success_basins.png/pdf`: success-region comparison; `static_force_check.json`: static estimate;
- `provenance.json`, `diagnosis_source.py`: calculation sources and provenance.

Code: [diagnose_balance_b.py](../../tools/diagnose_balance_b.py). Historical data were read from trash without restoring them over current files. Supply the available historical root and a fresh output directory to repeat:

```bash
.venv/bin/python scripts/AfferentBalance2026/tools/diagnose_balance_b.py \
  --historical-root "$HOME/.local/share/Trash/files/results.2/recompute_unscaled_20261016" \
  --output scripts/AfferentBalance2026/experiments/B_balance/results/diagnosis_repeat \
  --notebook-genome scripts/AfferentBalance2026/experiments/B_balance/results/notebook_run_20261021_2024/best_genome.json
```

During diagnosis the working `genome_cfg.json` was edited: the flexor's `w` and `A` upper bounds alone became 0.1. This `w` conflicts with the B-protocol validation (`w≤0.02`), while extending `A` separately changes available force. These ongoing edits were preserved and excluded from comparison; the calculation reads configurations embedded in saved genomes.
