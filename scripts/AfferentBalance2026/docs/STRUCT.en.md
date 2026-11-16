# Current structure after refactoring

[Русский](STRUCT.md) · [Guide](docs/USER_GUIDE.en.md)

The local run.py parses and routes requests. The SpikingNN library, installation and package CLI are unchanged.

| Module | Responsibility |
|---|---|
| [run.py](run.py) | User commands and legacy flag translation |
| [core/config.py](experiments/core/config.py) | One versioned preset → JSON → CLI, strict keys/applicability, provenance and paths |
| [core/genome.py](experiments/core/genome.py) | Named 52-field schema, domain, fixed/tied coordinates and encoding |
| [core/evolution.py](experiments/core/evolution.py) | Pure GA/RNG, independent of scientific tasks and files |
| [core/fitness.py](experiments/core/fitness.py) | Shared robust score reduction |
| [core/run.py](experiments/core/run.py) | Small model adapter, persistent evaluation pool, stages, selection, test, sweep and replay |
| [core/artifacts.py](experiments/core/artifacts.py) | Portable Solution, explicit schema adapters, preparation, atomic writes, comparison and publication |
| [simulation.py](experiments/simulation.py) | Unchanged Numba physical step and balance integration |
| [C/joint_study.py](experiments/C_setpoint_balance/joint_study.py) | Command codec, target banks and bounded least squares |
| [D/controller.py](experiments/D_trajectory/controller.py), [trajectory.py](experiments/D_trajectory/trajectory.py) | D codec, continuous AngleController, reference signals and tracking criterion |
| [notebook_runtime.py](experiments/notebook_runtime.py) | Shared solution selection, in-memory smoke fixtures and restart checks |
| [notebook_ui.py](experiments/notebook_ui.py) | One configurable UI with angle-only and activity views |
| [visualization.py](experiments/visualization.py) | Plots of explicit data and reports from an explicit manifest |
| `configs/presets/`, `configs/studies/` | Complete versioned domains and concise studies |
| `experiments/*/run_experiment.py`, `run_sweep.py` | Compatibility entry points into the common executor |
| `tools/retrain_tensile.py`, `finalize_retraining.py` | Thin finetune/publication entry points; no dependency on article selection code |

The extension boundary is ExperimentTask/make_task plus a model-specific codec. Larger networks and multilimb mechanics
must receive a new schema and numerical verification; they do not require duplicating the generic lifecycle.
See [development](DEVELOPMENT.en.md).

New runs live under results/. Historical experiment-local results and source ZIPs remain unchanged.
Historical Python executor APIs remain for old formats and exact numerical regression; new CLI commands do not use them.
Dated tools/articles describe historical studies, not current workflow defaults.
The Russian STRUCT.md retains the user's complete original analysis and notes under a clearly marked historical section.
Current references: [commands](docs/COMMANDS.en.md), [configuration](docs/CONFIGURATION.en.md), [results](docs/RESULTS.en.md).
