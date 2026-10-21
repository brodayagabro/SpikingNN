# Afferent controller: the A/B/C protocol

[Русский](README.md)

The active implementation uses the full nonlinear `SpikingNN.core.Izh_net.Afferents`, a fixed 0.1 ms integration step, physical muscle torque without ×10, and no PD/PID controller.

| Experiment | Task | Bounds |
|---|---|---|
| [A_balance](experiments/A_balance/README.en.md) | Zero-current balance throughout −50…+50° | `0.0001 ≤ w ≤ 1` |
| [B_balance](experiments/B_balance/README.en.md) | The same balance protocol | `0.0001 ≤ w ≤ 0.02` |
| [C_setpoint_balance](experiments/C_setpoint_balance/README.en.md) | Fixed-network setpoint replay with external currents | No automatic training |

See [layout and output files](experiments/README.en.md), [equations](experiments/MODEL_METHODS.en.md), and [development and tests](DEVELOPMENT.en.md).

The genome contains 52 fields: 50 controller parameters and fixed `a1`, `a2`. All field descriptions, units and system mappings in `genome_cfg.json` are English and are embedded in exported genomes. Mass, link length, viscosity and gravity are hyperparameters outside the genome. Applied currents are also separate inputs.

## Calculations completed on 21 October 2026

Each of A/B completed 12 training runs: populations 10/30/50/100, per-free-gene mutation probabilities 0.1/0.3/0.6, seed 42. All 60 checkpoints (generations 30/60/100/150/200) and 2400 generations per experiment are saved. Training, validation and test conditions span the full −50…+50° and −1…+1 rad/s domain. Episodes last 10 s.

| Results | Validation selection: population / mutation / generation | Independent test | Phase grid |
|---|---|---|---|
| [A](experiments/A_balance/results/README.en.md) | 100 / 0.1 / 200 | 96/96 | 100/100 |
| [B](experiments/B_balance/results/README.en.md) | 30 / 0.1 / 100 | 39/96 | 40/100 |

[C](experiments/C_setpoint_balance/results/README.en.md) evaluated five targets through 550 trajectories of the selected A genome with currents `[0, 0]`; no current law was fitted. Only the 0° target is held. Local `results/` directories contain genomes, complete training histories, heatmaps, RMSE and phase/time plots. Notebooks automatically load `results/best_genome.json`. One seed and a finite grid do not establish global stability.

## Existing report and historical results

The [report](report/REPORT.md), [PDF](report/REPORT.pdf) and [article figures](ARTICLE_FIGURES.en.md) describe earlier studies, not the new A/B/C protocol. Full A/B studies spanning −50…+50° completed on October 21. Their outputs are stored beside the corresponding experiments; this historical report has not been rebuilt for the new protocol.

Sources preceding reorganization remain in `archive/pre_ABC_20261020/` (`manifest.json`, `sources_before_ABC.zip`). Following the October-21 request, old numerical results, including the shared ×1 series and its runtime, were moved to the desktop trash. Recovery manifest: `archive/cleanup_results_20261021.json`. Historical helpers require restoring their data. New results live only in `experiments/A_balance/results/`, `B_balance/results/`, and `C_setpoint_balance/results/`.

Old 52-component arrays are incompatible: slots 46/47 used to contain currents and now contain `a1/a2`. Use `core.genome.import_legacy` explicitly, without transferring old performance claims to the new protocol. Results, archives and generated reports are excluded from Git; source and notebooks are retained.
