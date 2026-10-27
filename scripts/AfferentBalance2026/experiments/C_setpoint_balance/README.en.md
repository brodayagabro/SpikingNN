<!-- mirror-symmetry-20261026 -->

Separate study: [exact reflection symmetry of controller and currents](../MIRROR_SYMMETRY.en.md). Outputs are in `results/mirror_symmetry_20261026/`; the previous canonical genome is preserved.

<!-- muscle-tau50-20261026 -->

# Retraining with muscle relaxation times ≥50 ms

Current winner: `joint_refined`. Completed tests: **408/408**; strict successes: **0/408**. Selected by validation only. Both muscle A parameters are fixed at 0.37. Both muscles use tau_c in [50,100] ms and tau_1 in [50,200] ms. Integration step: 0.1 ms. No PD/PID controller or torque multiplier.

[Current results and file guide](results/README.en.md). [Full article](../../report/REPORT.en.md).

## Historical documentation (before retraining)

<!-- retraining-20261026 -->

# Retraining on corrected tensile mechanics

Current winner: `staged_refined`. Completed tests: **407/408**; strict successes: **381/408**. Selected by validation only. Both muscle A parameters are fixed at 0.37. Integration step: 0.1 ms. No PD/PID controller or torque multiplier.

[Current results and file guide](results/README.en.md). [Full article](../../report/REPORT.en.md).

## Historical documentation (before retraining)

# C: joint learning of a controller and a position command

[Русский](README.md) · [Model](../MODEL_METHODS.en.md) · [Layout](../README.en.md)

One spiking network must hold different commanded angles. The chromosome contains the system parameters and two current-polynomial coefficient vectors. Each current depends **only on the desired angle**, remains constant during an episode, and never uses measured angle, velocity or error. There is no PD/PID controller. Dynamics retain full Afferents, 0.1 ms integration and no ×10 torque gain; fitness and genetic operators come from the shared core.

## Tensile-torque correction, 2026-10-26

The active mechanics is `SpikingNN.Afferents/v3-tensile`: positive tension gives `M = -F_flex*h_flex + F_ext*h_ext`. The retained old winner completed **0/408** original test episodes after the correction; no retraining was performed. Historical training scores below describe the old sign. [New metrics and phase portraits](results/torque_sign_check_20261026/README.en.md).

Restart the Jupyter kernel and reopen the updated notebook before running it. Old outputs were backed up under `results/torque_sign_check_20261026/notebooks_before_correction/` and cleared from the active notebooks.

## Ready-to-use genome in the experiment directory

[best_genome.json](best_genome.json) is an exact copy of the winner of the **latest study**, `joint_sweep_tau_gt5_20261022`: quadratic `joint`, 50 individuals, 200 generations, seed 43, selected by validation. The file is self-contained: it includes its configuration, 52 system parameters and six coefficients for the two current commands. Muscle times have a 5.1 ms lower bound. The original file and complete history remain under `results/`.

[Combined phase portraits — PNG](phase_portraits/all_targets.png) · [PDF](phase_portraits/all_targets.pdf).

| Target angle | Individual portrait | Successful starts |
|---|---|---:|
| −45° | [PNG](phase_portraits/target_m45.png) · [PDF](phase_portraits/target_m45.pdf) | 100/100 |
| −22.5° | [PNG](phase_portraits/target_m22p5.png) · [PDF](phase_portraits/target_m22p5.pdf) | 99/100 |
| 0° | [PNG](phase_portraits/target_0.png) · [PDF](phase_portraits/target_0.pdf) | 100/100 |
| +22.5° | [PNG](phase_portraits/target_p22p5.png) · [PDF](phase_portraits/target_p22p5.pdf) | 100/100 |
| +45° | [PNG](phase_portraits/target_p45.png) · [PDF](phase_portraits/target_p45.pdf) | 100/100 |

Each target uses a 10×10 grid of absolute initial angles −50…50° and velocities −1…1 rad/s. Episodes last 10 s with 0.1 ms integration steps. Figures are redrawn from this genome's saved actual dynamics; hashes, physical parameters and commands are checked for all 500 trajectories. Plotting samples every 1 ms without smoothing. Color indicates initial angle, dots mark initial states, and stars mark desired equilibria. The font is Times New Roman.

[manifest.json](phase_portraits/manifest.json) links the exported JSON, source trajectories and figures by SHA256. [input_currents.csv](phase_portraits/input_currents.csv) lists commands for the five targets. These exported numerical results remain excluded from Git.

## Genome and parameters

The physical system block retains its 52 fields and ordering. C fixes `a1`, `a2` and both muscle `A=0.37` values, leaving 48 mutable system parameters. Original C weight bounds remain `0.0001 ≤ w ≤ 1`. `genome_cfg.json` contains the settings and English descriptions; exported genomes also describe every appended command gene.

With `x=target_deg/45`, commands are `I_i(x)=c_i0+c_i1*x+c_i2*x²+c_i3*x³`. Constant terms are learned too.

| Command representation | Physical fields | Mutable coordinates |
|---|---:|---:|
| Quadratic, 2×3 coefficients | 58 | 54 |
| Cubic, 2×4 coefficients | 60 | 56 |
| Current table for five targets, 2×5 values | 62 | 58 |

Default coefficient bounds are ±1000 nA; table-current bounds are also ±1000 nA. These are **different constraints**: a polynomial sum can exceed ±1000 nA. Currents are not clipped; `current_law.png` shows the actual command over the whole interval. All mutable search coordinates are normalized to [0,1]. Mass and the other mechanical settings remain external hyperparameters.

The default central candidate uses `../B_balance/results/best_genome.json` and a zero command. Other candidates are perturbed copies and uniform immigrants. The controller and command then evolve jointly. Seed mechanics, geometry, fixed A and parameter bounds are checked without clipping. Use `--from-scratch` to omit B, or `--seed-genome` for another admissible 52-field controller.

## Banks and selection

Every candidate sees **all five targets**: −45°, −22.5°, 0°, +22.5°, +45°. Each target uses the same eight initial states, totaling 40 training episodes per individual. Initial angles are absolute −50…+50°, with velocities −1…+1 rad/s, rather than errors relative to the target. Neurons and muscles reset for each episode; these are separate held commands, not online target switching.

- Training: 5 targets × 8 initial states = 40 episodes.
- Polynomial validation: 9 targets spaced by 11.25° × 12 independent states = 108 episodes.
- Final test: 17 targets spaced by 5.625° × 24 fresh states = 408 episodes. Eight test targets are absent from both training and validation.

Targets receive equal episode counts. The unchanged robust fitness reduction includes worst-case contributions; per-target quality and the worst target success fraction are also saved. Training, validation and test episodes are disjoint. Episodes last 10 s; short `--duration` values are for implementation checks.

Validation selects among checkpoints, including the first and last generations. Generation 1 contains the initial candidate, preserving a good least-squares solution even if subsequent training selection loses it. A comparison sweep evaluates the independent test **only for its overall validation winner**. Test outcomes never choose the method, degree or GA settings.

## Two pipelines

**`joint`** learns the common controller and polynomial coefficients from the first generation. This is the direct baseline.

**`staged`**:

1. Jointly learn one shared controller and ten currents for the five targets. The whole table uses the same network. Checkpoints are validated at those five targets with independent initial states.
2. Fit two bounded polynomials to the winning table with `scipy.optimize.lsq_linear`. Save the original currents, coefficients and approximation residuals.
3. Use that controller and fitted law as the initial individual for joint refinement. Polynomial validation now includes intermediate targets.

Least squares minimizes current error, not movement error. Small current changes can alter spiking regimes, so the fitted result must undergo fresh dynamical evaluations. A table genome cannot be replayed at an intermediate target before conversion to a polynomial.

`--generations 200` is the **total** budget: staged defaults to 100 table generations plus 100 polynomial generations, while joint uses 200 polynomial generations. Training simulation counts match at equal population sizes. Validation costs differ and are recorded. For a single staged run, set the split with `--table-generations`.

## Commands

From the repository root:

```bash
# Direct joint learning.
.venv/bin/python scripts/AfferentBalance2026/experiments/C_setpoint_balance/run_experiment.py --mode joint --degree 3 --population 50 --generations 200 --workers 8

# Table -> least squares -> joint refinement, with the same total budget.
.venv/bin/python scripts/AfferentBalance2026/experiments/C_setpoint_balance/run_experiment.py --mode staged --degree 3 --population 50 --generations 200 --workers 8

# Compare two methods and two degrees; test only the winner.
.venv/bin/python scripts/AfferentBalance2026/experiments/C_setpoint_balance/run_sweep.py --mode compare --degrees 2 3 --populations 30 --probabilities 0.1 --seeds 42 --generations 200 --workers 8

# A full 10×10 phase grid for each of the five targets from a joint genome.
.venv/bin/python scripts/AfferentBalance2026/experiments/C_setpoint_balance/run_sweep.py --mode replay --genome /path/to/joint/best_genome.json
```

The default mode is `replay`; opening the notebook or running old commands does not start training. `--output` must be a fresh directory. Configure training target/state banks in JSON, not with replay-only CLI options. Controller-only legacy solutions still accept `--current-law` or `--currents`; a joint genome already contains its command.

`explore_genome.ipynb` loads joint JSON genomes, populations or generation files through `load_joint_solution`. Generation ranking uses training scores; final-population ranking uses validation. Changing the target automatically recalculates currents. Phase trajectories, angle histories, neuron, muscle and afferent states are available. Explicit training is behind the disabled `RUN_STUDY` flag. When present, `results/current_joint.json` selects the new joint study; otherwise the notebook retains compatibility with the earlier `results/best_genome.json`.

## Output files

| File/directory within a new run | Contents |
|---|---|
| `pipeline.json` | Method, total budget, seed-controller source/hash and status |
| `table/` | Staged table training; absent for joint |
| `least_squares.json` | Current table, fitted coefficients and residuals |
| `polynomial/` | Joint learning of controller and polynomial |
| `*/generations/generation_XXXX.npz` | Every population: complete physical/normalized chromosomes, controller/command blocks, all measured training metrics and RNG |
| `*/checkpoints/epochs_XXX/` | Population validation scores and the winning genome |
| `*/final_population.npz` | Last stage population, potentially different from its best checkpoint |
| `best_genome.json` | Self-contained **joint** winner, schema `afferent_setpoint_joint_v1` |
| `best_controller.json`, `current_law.json` | Compatible separate views of that same solution for older tools |
| `selected/test_metrics.npz`, `test_by_target.csv` | Independent test and target-specific summaries |
| `selected/traces/` | 15 full traces: 5 targets × initial angles −50°, 0°, +50°, zero speed; recorded every 0.1 ms |
| `selected/target_*/` | Phase and angle-time plots for each training target |
| `selected/current_law.png`, `neurons.png`, `muscles.png`, `afferents.png` | Command law and subsystem states for one example |

Comparison sweeps also store `summary.csv`, `validation_comparison.png`, `completion.json` and `genome_rmse_degree_*.npz`. RMSE uses all normalized mutable genes, including command genes, separately for each degree. Different dimensions are not mixed. Test artifacts and `selected/` are created only for the overall winner.

Each stage stores its configuration and source code in `source_snapshot.zip`; hashes are checked on completion. Earlier zero-current C results remain as controls, separate from joint learning. `polynomial_study.py:fit_current_law` remains an optional API for learning currents while keeping the network fixed.

## Pilot, 22 October 2026

Compared `joint` and `staged`: 20 individuals, 30 total generations, cubic polynomials, p=0.1, seed=42. Validation selected `joint`; independent test **76/408**, mean late RMSE **7.95°**. Reliable control across the range has not yet been achieved. The historical pilot directory `results/joint_pilot_20261022/` is absent from the current working tree.

## Two-method sweep and full diagnostics

The preceding full sweep used `joint`/`staged`, degrees 2/3, populations 30/50, mutation probabilities 0.1/0.3 and seed=42: 16 lines with 200 total generations each. Staged splits this into 100 table and 100 polynomial generations. The new test-bank seed is 2026102213; training and validation banks remain unchanged for comparison with the pilot. One GA seed does not establish statistical robustness.

```bash
.venv/bin/python scripts/AfferentBalance2026/experiments/C_setpoint_balance/run_sweep.py --config scripts/AfferentBalance2026/experiments/C_setpoint_balance/results/joint_sweep_20261022/joint_degree_2/population_030/mutation_0.1/seed_42/polynomial/genome_cfg.json --seed-genome scripts/AfferentBalance2026/experiments/B_balance/results/best_genome.json --mode compare --generations 200 --workers 18 --output scripts/AfferentBalance2026/experiments/C_setpoint_balance/results/joint_sweep_20261022_reproduction
.venv/bin/python scripts/AfferentBalance2026/experiments/C_setpoint_balance/sweep_diagnostics.py scripts/AfferentBalance2026/experiments/C_setpoint_balance/results/joint_sweep_20261022_reproduction --grid-size 10
.venv/bin/python scripts/AfferentBalance2026/tools/analyze_joint_sweep.py scripts/AfferentBalance2026/experiments/C_setpoint_balance/results/joint_sweep_20261022_reproduction
```

The additional `sweep_diagnostics.py` step freezes the validation winner of **each method** before independent tests and dense phase grids. The overall winner's existing test is reused; test outcomes do not reselect the method. Genomes live at `method_winners/joint/best_genome.json` and `method_winners/staged/best_genome.json`; every training run also retains its complete last population.

Within `method_winners/<method>/diagnostics/`:

- `phase_all_targets.png/.pdf`: five phase portraits, 100 starts each on −50…50° × −1…1 rad/s. Stars mark requested equilibria and crosses mark falls; failed traces are retained.
- `target_*/grid/`: full 0.1 ms trajectories for all 500 cases, currents, genome, mechanics and metrics. `target_*/phase_grid.png/.pdf` shows each target separately.
- `target_*/zero_speed/` and `angles_zero_speed.png/.pdf`: ten angle-time trajectories per target at zero initial velocity, another 50 episodes.
- `input_currents.csv`: both exact commanded currents for every target.
- `example/`: a predeclared case, target +22.5°, initial angle −30°, zero speed. Selection is independent of success; `case_000.npz` contains the complete dynamics.
- `example/state_overview.png/.pdf`: limb angle, membrane voltages, muscle activation/force, summed afferents per muscle and input currents.
- `example/input_currents`, `neurons`, `membrane_zoom`, `muscles`, `afferents`, `afferent_sums`, `limb` (`.png/.pdf`): detailed plots. `membrane_zoom` preserves every sample over the first and last 0.2 s. Ia+II+Ib sums use raw model units and are not weighted afferent feedback currents.

Each method saves 551 full trajectories. Smooth channels may be decimated for display only; integration and saved arrays always use 0.1 ms. `analyze_joint_sweep.py` audits every generation, snapshot/trace hashes and fixed fields, rebuilds metrics from traces, repeats tests and compares an example with the original library classes. It also plots `validation_heatmap_seed_42`, `training_histories` and `genome_rmse_degree_*`.

## Completed C sweep, 22 October 2026

Completed 16 lines: two methods, degrees 2/3, populations 30/50, mutation probabilities 0.1/0.3, seed=42, 200 total generations each. Staged splits the budget into 100 table and 100 polynomial generations. Training uses five targets, validation nine and independent testing seventeen. Fresh test initial-state seed: 2026102213.

| Method | Degree | Population | Mutation | Validation | Independent test | Late RMSE (deg) | Phase grid |
|---|---:|---:|---:|---:|---:|---:|---:|
| joint | 2 | 30 | 0.1 | 0.969244 | 408/408 | 0.469522 | 500/500 |
| staged | 3 | 50 | 0.1 | 0.902371 | 402/408 | 0.886729 | 498/500 |

Both method winners and the overall winner were selected by validation alone. Quadratic joint performed best in this sweep; one seed cannot establish statistical method superiority. Each solution has five targets × a 10×10 start grid (−50…50°, −1…1 rad/s), plus 50 zero-speed trajectories and one predeclared example: target +22.5°, start −30°, zero speed. Integration and recording use 0.1 ms; episodes last 10 s.

[Complete result index, genomes and figures](results/joint_sweep_20261022/README.en.md). The joint example uses constant currents **−31.4123513 and 238.3142774 nA**. This study is preserved; retraining with a 5.1 ms floor is described below.

## C retraining: relaxation times >5 ms

`genome_cfg.json` now constrains `muscles.flexor.tau_c`, `muscles.flexor.tau_1`, `muscles.extensor.tau_c` and `muscles.extensor.tau_1` to **[5.1, 100] ms**, with logarithmic encoding. Genome encoding rejects values at or below 5 ms; boundary values and mutation are covered by a test. The library internally stores reciprocal rates (1/ms) in its `tau_c` and `tau_1` attributes, whereas the genome and constructor arguments specify times in ms.

The preceding study already used a 5 ms floor. Its joint winner had times 7.276, 19.739, 6.675 and 23.041 ms (flexor τ_c, τ₁; extensor τ_c, τ₁), already satisfying >5 ms. Force pulsations therefore cannot be explained by violating this bound. In `dF/dt = A·N·x − F/τ₁`, τ₁ also changes steady force magnitude.

The repeat study uses both methods, degrees 2/3, 50 individuals, mutation probability 0.1, GA seed 43 and 200 total generations per line (4 runs). The fresh test-bank seed is 2026102223; training and validation remain unchanged. The B seed controller is unchanged and admissible under the new bounds. This is retraining with a new seed, not an isolated estimate of the 0.1 ms lower-bound change.

Outputs are stored in [joint_sweep_tau_gt5_20261022](results/joint_sweep_tau_gt5_20261022/README.en.md). The [comparison](results/joint_sweep_tau_gt5_20261022/relaxation_comparison/README.en.md) evaluates the previously frozen old and new winners on the same new test bank. Force panels show the final 0.2 s without smoothing; force statistics use the last 2 s. Both methods retain full phase grids and state traces under the existing protocol.

```bash
.venv/bin/python scripts/AfferentBalance2026/experiments/C_setpoint_balance/run_sweep.py --mode compare --degrees 2 3 --populations 50 --probabilities 0.1 --seeds 43 --generations 200 --workers 18 --output scripts/AfferentBalance2026/experiments/C_setpoint_balance/results/joint_sweep_tau_gt5_20261022
.venv/bin/python scripts/AfferentBalance2026/experiments/C_setpoint_balance/sweep_diagnostics.py scripts/AfferentBalance2026/experiments/C_setpoint_balance/results/joint_sweep_tau_gt5_20261022 --grid-size 10
.venv/bin/python scripts/AfferentBalance2026/tools/analyze_joint_sweep.py scripts/AfferentBalance2026/experiments/C_setpoint_balance/results/joint_sweep_tau_gt5_20261022
.venv/bin/python scripts/AfferentBalance2026/tools/compare_muscle_relaxation.py scripts/AfferentBalance2026/experiments/C_setpoint_balance/results/joint_sweep_20261022 scripts/AfferentBalance2026/experiments/C_setpoint_balance/results/joint_sweep_tau_gt5_20261022 --workers 18
```

Choose a fresh `--output` directory to repeat training; completed outputs are never overwritten. After a successful audit, `results/current_joint.json` points the notebook to the new study. The previous pointer and results are retained.

### Retraining outcome

| Method | Degree | Validation | Independent test | Late RMSE (deg) | Phase grid |
|---|---:|---:|---:|---:|---:|
| joint | 2 | 0.910907 | 402/408 | 0.878920 | 499/500 |
| staged | 2 | 0.687306 | 354/408 | 1.600976 | 495/500 |

Retraining did not improve the preceding best solution. The new joint has no test falls; six episodes miss the 3 s settling deadline (3.05–3.41 s). Force pulsations also remain: in the fixed example, flexor force standard deviation changes from 2.419 to 2.265 N, and extensor force standard deviation from 1.108 to 1.793 N. This compares different trained genomes and seeds, not the causal effect of τ alone. [Common test, times and oscillograms](results/joint_sweep_tau_gt5_20261022/relaxation_comparison/README.en.md).

In this example, the new `staged` reduces flexor force standard deviation from 6.188 to 1.370 N and extensor force standard deviation from 2.522 to 0.418 N. Relative variation also decreases: 0.406→0.216 and 0.178→0.127. Mean forces change as well; this is a different learned regime. A calmer example does not imply better control across targets: on the same new bank, the previous staged wins 401/408 cases and the new one 354/408.
