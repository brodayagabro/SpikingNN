# Local workflow development (2026-10-27)

[Русский](DEVELOPMENT.md) · [Structure](STRUCT.en.md) · [Guide](docs/USER_GUIDE.en.md)

The experiment remains local. No changes to src/SpikingNN, setup.py, dependencies, package entry points or numerical equations
are required by this refactor. No full scientific retraining was performed.

## Extending the model

The current architecture is `two_neurons_one_limb_v1`, with controller schema `afferent_controller_52_v4`.
JSON overrides can change existing domains/fixed values/ties; they cannot safely change the meaning or size of that schema.
For more neurons or MultyLimbSystem:

1. Define a new versioned codec with named physical fields and encode/decode/size/names; specify state and channel units.
2. Implement the numerical adapter and banks at `make_task`/`ExperimentTask`. Keep model-specific dynamics and command semantics there.
3. Extend the explicit schema dispatcher in config/artifacts; never reinterpret an old 52-field solution as a larger system.
4. Reuse the executor, deterministic GA, source snapshots, stages, selection and result lifecycle.
5. Verify one physical step against the relevant SpikingNN classes, force directions, units and time grid; test complete hidden-state continuity,
   metrics from dense traces, controller round trips, deterministic populations, and process ordering before a scientific run.

No plugin registry or inheritance framework is needed for the four current tasks. A/B/C/D are small explicit adapters.
Generic exports infer physical width from the codec; current model-specific channels and banks remain explicit.
A larger network/multilimb integrator is an extension point, **not an implemented or validated new scientific model**.

## Automated verification

Run from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -c scripts/AfferentBalance2026/pytest.ini scripts/AfferentBalance2026/tests -q
.venv/bin/python scripts/AfferentBalance2026/tools/check_workflow.py --check-docs
.venv/bin/python scripts/AfferentBalance2026/tools/check_workflow.py --notebooks /tmp/afferent-notebook-check
.venv/bin/python scripts/AfferentBalance2026/tools/check_workflow.py --profile /tmp/afferent-profile
```

Output directories must be new. Notebook execution reads code in smoke mode; it does not overwrite notebook files, outputs or notes.
Use `--write-docs` only when intentionally updating generated parser/preset references.

## Refactor verification record

Baseline: 117 tests passed. Final verification: **135 tests passed in 15.75 s**. The isolated end-to-end check executed 21 CLI calls, including the expected rejection of an existing output. The extended suite adds common-run contracts, exact A/B/C/D population comparisons,
parallel/serial equality, explicit preparation and symmetry, generation-zero selection, validation/test separation,
independent sweep horizons, portable records, failures, and cwd-independent resolution.
Eight actual historical JSON genomes and one historical NPZ population per A/B/C/D task were also read with unchanged file hashes. All five experiment notebooks passed smoke execution with locally generated untrained fixtures; neither ignored scientific results nor GA were needed.
The original outputs, execution counts, notes and user-tuned settings in both D notebooks were preserved;
the user's original STRUCT.md is retained verbatim in its historical section.

| TZ acceptance | Evidence |
|---|---|
| 1–3: boundaries, clean fixtures, cwd | unchanged library/setup; smoke studies and test_cwd_paths_validate_does_not_integrate |
| 4–8: keys, provenance, domains, preparation, symmetry | test_run_contract.py; existing genome/mirror tests |
| 9: explicit D source | no default_sources; field-specific transfer diagnostics; D source tests |
| 10–12: physics, streaming, metrics | unchanged 26-channel kernel; existing tensile/trajectory/abc tests and stride checks |
| 13–15: GA, selection, budgets | exact populations/scores/RNG A/B/C and D initialization; generation-zero and sweep tests |
| 16–18: artifacts, legacy data, failures | portable copy/NPZ adapters, source hashes, failed manifests and overwrite rejection |
| 19: notebooks | tools/check_workflow.py --notebooks; read-only outputs/notes check |
| 20: references | generated CONFIGURATION/COMMANDS and checked source links; guide smoke commands |
| 21: publication/report | explicit publish; report takes a manifest; comparison/replay source hashes stay unchanged |

## Fixed-protocol timing

Same environment, six candidates × four episodes × 0.05 s; seven warm repeats; separate empty Numba cache for each cold process.
These millisecond workloads emphasize process overhead and do not predict large-run throughput.

| Component | Historical evaluator | Common evaluator |
|---|---:|---:|
| Cold integration including JIT | 2.765 s | 2.770 s |
| Warm training median | 2.107 ms | 2.186 ms |
| Warm validation median | 2.070 ms | 2.136 ms |
| First two-worker evaluation including spawn | 0.941 s | 0.948 s |
| Warm process evaluation median | 1.724 ms | 2.021 ms |
| Candidate IPC payload | 929 bytes | 557 bytes |
| Compressed array write median | 0.287 ms | 0.244 ms |
| Small plot | 30.25 ms | 29.97 ms |

The numerical metric bytes have identical SHA-256. The common wrapper adds about 0.08 ms serial and 0.30 ms process overhead
on this tiny workload; no speedup is claimed. Recording remains outside numerical hot loops. The profiling utility retains raw JSON and plots.

## Compatibility boundaries

Historical Python training APIs remain available for old layouts and regression; current CLI wrappers route to the shared executor.
Existing results, frozen ZIPs and scientific claims were not migrated or recomputed. `report/build_retraining_article.py`
remains the dated historical article builder; new evidence reports use `run.py report --manifest ...`.
Current-model replay is recorded as a new evaluation protocol and is not claimed as exact archived-runtime reproduction.
Archived runtimes require explicit isolated activation. Restart Jupyter after changing numerical code.

The earlier developer history follows unchanged.

---

# Development and validation

[Русский](DEVELOPMENT.md) · [Overview](README.en.md)

From the repository root:

```bash
.venv/bin/python -m pytest -c scripts/AfferentBalance2026/pytest.ini scripts/AfferentBalance2026/tests -q
.venv/bin/python scripts/AfferentBalance2026/tools/validate_layout.py
```

Tests compare Numba against independent library-class replay; check nonlinear receptors, all 52 fields and fixed geometry, English descriptions, A/B bounds, condition banks, process-pool reproducibility, generation/winner storage and sweeps. Short real integrations validate implementation rather than scientific convergence. Historical tests are excluded from active collection.

Notebooks use the `spikingnn` kernel. Environment variable `SPIKINGNN_NOTEBOOK_SMOKE=1` shortens demonstration grids and duration for automated implementation checks; normal runs use full values. Notebook training remains off until its flag is explicitly changed.

Change shared physics in `experiments/simulation.py` and check all channels against `SpikingNN.core.Izh_net`. Current-law changes belong only to C. New comments, docstrings and JSON field descriptions are English; documentation has RU/EN pairs. Do not add results or source snapshots to Git.

Do not reinterpret old arrays using the new field order. Never edit frozen historical runtimes or replace their recorded hashes. Validate historical and current runtimes in separate processes.

## Verified on 20 October 2026

All 37 tests passed. The three `explore_genome.ipynb` notebooks executed with the `spikingnn` kernel in smoke mode (0.02 s). A CLI, B sweep and C grid entry points were also checked from `/tmp`, including two worker processes for A (0.01 s episodes). Historical ×1 runtime hashes matched their recorded values. No scientific training was launched for the new protocol.

## Verified on 21 October 2026

All 41 tests passed. Added checks cover the full initial-condition domain, asymmetric ranges and odd bank sizes, and Numba/library agreement for randomly sampled admissible genomes. All three notebooks were executed again in smoke mode with their actual saved winning genomes; records are in `results/notebook_validation.json`. Notebook checks do not launch training.

Scientific A/B studies completed: 12 continuous training runs, 60 checkpoints and 2400 saved generations each. `analyze_abc_study.py` checked every population, gene bounds, fixed `a1/a2`, fitness reconstruction and source hashes. Selection used validation, followed by independently repeating the winner's test. Each experiment recorded 110 diagnostic trajectories. C recorded 550 trajectories without fitting currents. See local `results/README.en.md` and A/B `audit.json` files.

An additional one-second trajectory of a trained A controller was compared with the original `SpikingNN.core.Izh_net` classes: initial conditions +50°, +1 rad/s, 10001 samples and all 26 channels. The maximum absolute discrepancy was approximately `4.19e-12`; `experiments/A_balance/results/model_reference_check.json` identifies the checked genome and its hash. This verifies the numerical implementation, not stability over an infinite horizon.

Old numerical results were moved to the desktop trash; the manifest is `archive/cleanup_results_20261021.json`. Historical source code was retained. New results are excluded from Git.

## B retraining with fixed A

All 45 tests pass. Coverage includes the strict `w<0.1` cap, lower bound 0.02, initial weights 0.08, both `A=0.37` values remaining fixed through genetic operations and storage, 48 mutable coordinates and loading earlier 50-coordinate configurations. A separate check compares every Numba channel with library classes at the new A, initial angles ±50° and speeds ±0.7 rad/s.

`tools/select_sweep_winner.py` was checked with a short series in `/tmp`, including preserving the previous winner when publishing a new one. It rejects historical protocols instead of labeling them as the current A=0.37 study. Previous diagnostic output folders were moved to trash; tests and source code remain.

The B rerun completed: 12 lines, 60 checkpoints and an audit of 2400 generations. Independent test scores matched on replay; the phase grid passed 100/100 cases, plus 10/10 at zero speed. The selected genome was compared with library classes over 10001 samples and all 26 channels (+50°, +1 rad/s, 1 s); the maximum absolute discrepancy was `1.047e-11`. Record: `experiments/B_balance/results/sweep_restored_weights_A037_20261021/selected/model_reference_check.json`.

The B notebook executed with the new winning genome in the `spikingnn` kernel, using 0.02 s smoke mode without training. Record: `experiments/B_balance/results/notebook_validation.json`. Checksums and genome identity were verified for all 110 full trajectories (100001 samples and 26 channels each).

## C joint training: 22 October 2026

All 54 tests passed; the nine C-specific tests passed again after adding the comparison figure and genome RMSE. `test_joint_setpoint.py` checks 58/60/62-field combined genomes, exact fixed A values and geometry, English descriptions, one shared controller per multi-target evaluation, all five targets per candidate, independent banks, bounded least squares, changes to both mutable blocks, every-generation storage, JSON/NPZ replay and one test for the overall sweep winner. Short integrations validate implementation.

A separate C pilot compared `joint` and `staged`: 20 individuals, 30 total generations, cubic laws, p=0.1, seed=42, 10 s episodes. The audit checked 60 generations / 1200 individuals, bounds and fixed A=0.37 values, reconstructed fitness and snapshot hashes. All 408 independent tests of the selected joint solution reproduced exactly: 76 successes, mean late RMSE 7.95°. Full-range control was not achieved. Artifacts: `experiments/C_setpoint_balance/results/joint_pilot_20261022/`.

C's notebook executed with the new combined genome in the `spikingnn` kernel using 0.02 s episodes and no training: nine nonempty code cells, no errors. Checked 120 active-document local links, RU/EN pairs and C docstrings. A/B notebooks containing user-generated output were not altered to satisfy the general validator's output-free rule. C numerical results are excluded from Git.

## Completed C sweep, 22 October 2026

All **55 tests** pass. The new check verifies separate validation selection per method and exact full-trajectory/current exports. All new figure types were rendered and inspected using short real episodes.

Full sweep: 3200 generations, 128000 checked individuals and 112 validation checkpoints. The audit verified source snapshots, bounds and fixed fields, a common seed controller, population metrics and 1102 complete trajectories. Both independent tests reproduced exactly; each example was compared against original library classes over 0.1 s across all 26 channels. The notebook executed with the new genome in spikingnn smoke mode (0.02 s, no training). Records: `experiments/C_setpoint_balance/results/joint_sweep_20261022/audit.json` and `notebook_validation.json`.

## C muscle-time boundary check

All 56 tests passed after raising the lower bound to 5.1 ms. `test_muscle_time_floor_survives_genetic_operations_and_matches_library` checks boundary genomes, mutation of every gene, rejection of 5 ms times, and library agreement across all channels with all four times set to 5.1 ms. The shared loader remains compatible with archived configurations; new bounds belong to the C configuration.

The C repeat study was fully audited: 800 generations, 40000 individuals, 28 checkpoints and 1102 full traces. Both tests reproduce exactly; accelerated dynamics agree with the library. All muscle times are ≥5.1 ms. The notebook executed in spikingnn with the new genome; the smoke check did not train or modify the source ipynb. Audit records are in `joint_sweep_tau_gt5_20261022/audit.json`, `relaxation_bounds_audit.json` and `notebook_validation.json`.

## D: streaming interface, October 23, 2026

All 66 tests pass, including 10 new D checks (with parametrization). `simulation.py` now exposes one complete-state physical step shared by the original episodic solver and streaming D. A saved ten-second C example (100001 samples × 26 channels) and its metrics were reproduced bit for bit. Independent library objects were also checked with varying currents.

D tests block-boundary continuity, command changes without reset, filter units, analytic target velocity, metrics recomputed from complete traces, independent banks, both training modes, fixed A/geometry, the 5.1 ms time floor, and identical serial/parallel results. A complete technical smoke study with short episodes was run separately from the scientific study.

The scientific D study completed and passed an independent audit: 160 generations, 4800 individuals and 81 full trajectories (34,012,998 samples). `experiments/D_trajectory/audit_trajectory.py` checks sources, bounds, every fitness value and validation selection, then repeats held-out tests. Variable-current dynamics were compared with library objects for 4.1 s in three solutions; the largest discrepancy is ≤5.83e−11. Records are in `D_trajectory/results/trajectory_sweep_20261023/audit.json`. The final notebook was executed with spikingnn without training or modifying the source ipynb.

Active-layout validation: `tools/validate_layout.py --allow-notebook-outputs`. The flag checks links and code without deleting user-generated A outputs; strict mode still requires empty outputs by default. All four experiments, 11 RU/EN pairs, English docstrings and genome descriptions were verified.
