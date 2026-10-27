# Mirror symmetry in C and D

[Русский](MIRROR_SYMMETRY.md)

The separate `mirror_symmetry_20261026` study tests whether tying antagonistic
controller halves improves setpoint holding and tracking. It imposes **exact
symmetry**, rather than a penalty for unequal genes. The previous asymmetric
`muscle_tau50_20261026` solutions are retained.

## Constraints

Reflection reverses upright-relative angle, angular velocity and target angle,
while swapping neurons 0↔1 and muscles F↔E:

| Block | Constraint |
|---|---|
| Neurons | `a[0]=a[1]`, likewise `b`, `c`, `d` |
| Recurrent connections | `W[0,0]=W[1,1]`, `W[0,1]=W[1,0]` |
| Synaptic times | The same pairs for `tau_ms` |
| Muscles | Equal F/E values of `w, N, A, tau_c, tau_1, m, k` |
| Afferent projections | `Q[0,Ia_F]=Q[1,Ia_E]`, likewise II and Ib; also `Q[0,*_E]=Q[1,*_F]` |
| Receptor thresholds | `L_th[F]=L_th[E]`, `F_th[F]=F_th[E]` |
| Output projection | `P[F,0]=P[E,1]`, `P[F,1]=P[E,0]` |
| Command currents | `I1(theta_target)=I0(-theta_target)` |

The distinct attachment distances `a1=0.2 m` and `a2=0.05 m` are **not tied**.
Mass, length, viscosity and gravity remain hyperparameters. Both muscles retain
fixed `A=0.37`, `tau_c∈[50,100] ms`, and `tau_1∈[50,200] ms`. Full nonlinear
`Afferents`, tensile mechanics, the 0.1 ms step and the fitness function are
unchanged. No PD/PID controller is added.

For `Ii(x)=sum(c[i,k]*x**k)`, `x=theta_target/45°`, symmetry requires
`c[1,k]=(-1)**k*c[0,k]`: equal even coefficients and opposite odd coefficients.
In C's table stage, the second current row is the first row in reversed target
order. Simply supplying equal currents to both neurons would not encode this rule.

## Implementation and dimensions

`core/genome.py` defines the 52-field permutation `MIRROR_INDICES`. With
`"mirror_symmetry": true`, `GenomeConfig` retains one independent coordinate
per pair. `decode` reconstructs both values, so crossover, mutation and uniform
immigrants automatically satisfy the constraints. `encode` rejects asymmetric
physical values. Paired bounds, scales and mutable flags must agree. Configurations
without the flag retain the previous behavior.

There are still 52 physical controller fields, but 24 independent coordinates
instead of 48. Quadratic C has 27 independent coordinates and 58 physical fields;
cubic D has 28 coordinates and 60 fields. JSON exports preserve all physical fields.
Command parity is implemented in `C_setpoint_balance/joint_study.py` and
`D_trajectory/controller.py`, outside the common core.

## Protocol

1. `tools/prepare_mirror_genome.py` explicitly averages paired physical parameters
   and projects polynomial coefficients onto the required parity. This creates a
   new candidate whose performance must be measured again; it does not modify or
   preserve the behavior of the source genome.
2. Projected A/B/C seeds for C and A/B/C/D seeds for D are compared on **training**
   data only. The best starts the population; the others are additional candidates.
   `seed_screen.json/.npz` records these evaluations and source hashes.
3. C compares direct joint quadratic training against 100 table generations →
   bounded least squares → 100 joint polynomial generations. Each method uses
   50 individuals, 200 total generations, seed 61, mutation probability 0.3,
   and normalized mutation standard deviation 0.12→0.01.
4. D first uses 50×200 with the same settings. A second run transfers the best
   symmetric C, adding the first D winner as another seed: 50×200, seed 62,
   probability 0.2, deviation 0.06→0.005. Current filtering remains zero.
5. Training, validation and test banks are unchanged. C jointly trains five
   targets −45°, −22.5°, 0°, +22.5°, +45°; its test crosses 17 targets with 24
   initial conditions spanning ±50° and ±1 rad/s. D retains 8 training,
   10 validation and 18 test sinusoids. Whole populations are validated at
   generations 1, 20, 40, …, 200; the original candidate is also eligible.
   Selection maximizes validation fitness.
6. After freezing the winners, a new bank with test seed 2026102605 evaluates both
   the old and new solutions. It is not used for training or selection.

Strict C success requires a complete episode and uninterrupted
`|theta-target|≤3°`, `|omega|≤0.35 rad/s` after 3 seconds. After warm-up, D requires
a complete episode, angle RMSE ≤3°, and at least 95% occupancy of the joint angle
±3° / velocity-error ±0.35 rad/s band. Parameter equality alone neither proves
stability nor guarantees these criteria.

This is an exploratory comparison of individual runs. The old winner came from
a longer series with several refinements; historical and new computational
budgets are unequal. Statistical conclusions require multiple independent seeds.

## Running and locating results

From the repository root, prepare a quadratic C seed:

```bash
.venv/bin/python scripts/AfferentBalance2026/tools/prepare_mirror_genome.py \
  --genome scripts/AfferentBalance2026/experiments/C_setpoint_balance/results/best_genome.json \
  --config scripts/AfferentBalance2026/experiments/C_setpoint_balance/genome_cfg.json \
  --output /tmp/my_symmetric_C --degree 2

.venv/bin/python scripts/AfferentBalance2026/tools/retrain_tensile.py \
  --experiment C --method joint --seed-genome /tmp/my_symmetric_C/best_genome.json \
  --output /tmp/my_symmetric_C_training --population 50 --generations 200 \
  --seed 61 --workers 8 --mutation-probability .3 --sigma-start .12 --sigma-end .01 \
  --validation-every 20 --no-polarity-seeds
```

`--genome` supplies the source solution; `--config` supplies destination bounds
and protocol; `--degree` selects C's polynomial degree (D always uses cubic format).
`--output` must be new. `--seed-genome` supplies the prepared start and repeated
`--extra-seed` options add alternatives. `--method staged` enables C's table stage.
`--population` and `--generations` set population and total generation budget;
`--mutation-probability` applies to each **independent paired coordinate**;
`--sigma-start/end` define its deviation in [0,1]. `--seed` controls GA randomness,
`--workers` sets evaluator processes, and `--validation-every` sets checkpoint
spacing. `--no-polarity-seeds` disables the older polarity alternatives.

Results: [C](C_setpoint_balance/results/mirror_symmetry_20261026/README.en.md),
[D](D_trajectory/results/mirror_symmetry_20261026/README.en.md).
Each experiment root exposes `best_genome_slow_muscle_mirrow.json`. The previous canonical
`best_genome.json` remains available. Select the new JSON explicitly in a demo
notebook; the loading interface is unchanged.

`tests/test_mirror_symmetry.py` checks codecs, mutation, least squares, both GA
pipelines, the original library and all reflected state channels.
`tools/summarize_mirror_study.py` additionally verifies every saved individual,
publishes the comparison and runs the independent confirmation bank.
`tools/retraining_figures.py --experiment C_setpoint_balance --run <winner_directory>`
saves 100 phase trajectories at each of five targets, zero-speed oscillograms and
one full-resolution state example per target. For `D_trajectory` it replays all
held-out sinusoids.

Both C/D folders also publish `best_genome_fast_muscle.json` (the earlier 5.1 ms-floor study) and `best_genome_slow_muscle.json` (the previous asymmetric winner with the 50 ms floor). `GENOME_VERSIONS.md/.en.md` list source paths, SHA256 hashes and actual time constants.
