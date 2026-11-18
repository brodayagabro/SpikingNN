# AfferentBalance2026

**Серия B/C от 17 ноября:** веса обратной связи II ограничены ±50, N изменяемый, предел 100 Гц проверяется по всем межспайковым интервалам. Новый sweep до 500 поколений: [инструкции](topubsrc/README.md).

[Русский](README.md)

**Article source:** [topubsrc/README.md](topubsrc/README.md) is the standalone A–C bundle,
with eight Python files, configurations, frozen genomes and three notebooks.
Only that directory is needed; SpikingNN is not a dependency.

Local experiments with a spiking controller, two antagonistic muscles and the full nonlinear
`SpikingNN.core.Izh_net.Afferents` model. Integration uses 0.1 ms steps and tensile muscle forces.
There is no PD/PID controller or ×10 torque multiplier.

| Experiment | Task |
|---|---|
| A_balance | Unstable upright equilibrium, zero currents, w ≤ 1 |
| B_balance | The same task with 0.02 ≤ w < 0.1 |
| C_setpoint_balance | Fixed targets −45…45°: joint controller/polynomial search or table → least squares → polynomial |
| D_trajectory | Continuous tracking: joint, command_only and frozen comparison |

Starting presets fix A=0.37, τc=50…100 ms and τ1=50…200 ms. The controller has 52 physical fields;
independent dimensionality is computed from fixed fields and mirror ties.

From the repository root:

```bash
.venv/bin/python scripts/AfferentBalance2026/run.py validate --config scripts/AfferentBalance2026/configs/studies/A_smoke.json --json
.venv/bin/python scripts/AfferentBalance2026/run.py train --config scripts/AfferentBalance2026/configs/studies/A_smoke.json
```

This is a short infrastructure check, not scientific training. A new result appears in `results/smoke_A`;
use another `--output` to repeat it. Historical results are neither relocated nor overwritten.

- [Guide: nine working scenarios](docs/USER_GUIDE.en.md).
- [Configuration](docs/CONFIGURATION.en.md), [commands](docs/COMMANDS.en.md), [result schemas](docs/RESULTS.en.md).
- [Module structure](docs/STRUCT.en.md), [development and verification](docs/DEVELOPMENT.en.md).
- [Model equations and criteria](experiments/MODEL_METHODS.en.md), [mirror symmetry](experiments/MIRROR_SYMMETRY.en.md).
- [Technical specification, Russian](docs/TZ.md), [article materials](report/README.en.md).

`solution.json` is the selected individual; the final population is stored separately.
Only explicit `publish` updates a current-solution pointer. Notebook training is disabled by default.
Larger networks and MultyLimbSystem require a new codec and a validated numerical adapter;
the current configuration intentionally cannot change the meaning of the 52-field schema.

## Constrained study, 16 November 2026

The previous publication results have been deleted. The new A/B/C study bounds both cross projections strictly below 0.1, retains the fitness and independently measures both neurons. See [topubsrc/README.md](topubsrc/README.md) for commands and live results.
