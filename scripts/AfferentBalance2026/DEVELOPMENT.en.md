# Development and validation

[Русский](DEVELOPMENT.md) · [Overview](README.en.md)

From the repository root:

```bash
.venv/bin/python -m pytest -c scripts/AfferentBalance2026/pytest.ini scripts/AfferentBalance2026/tests -q
.venv/bin/python scripts/AfferentBalance2026/validate_layout.py
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
