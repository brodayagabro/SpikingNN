# Working with the experiments

[Русский](USER_GUIDE.md) · [Configuration](CONFIGURATION.en.md) · [Commands](COMMANDS.en.md) · [Results](RESULTS.en.md)

Run the examples from `scripts/AfferentBalance2026` using the existing SpikingNN environment.
From the repository: `source .venv/bin/activate`, then `cd scripts/AfferentBalance2026`.
Absolute paths to `run.py` and the configuration work from any cwd. JSON paths are relative to the declaring file;
CLI paths are relative to cwd. This experiment is not an installed package or a library CLI.

`validate` neither integrates dynamics nor compiles Numba nor creates a run directory. Training requires `--config`.
The `*_smoke.json` examples use 0.02 s horizons and 2–3 generations: they verify infrastructure, **not scientific performance**.
Full starting protocols live in `configs/presets/*_tau50_v1.json`. D always requires explicit compatible sources.
Each output must be new; use a different `--output` to repeat a completed example.

## 1. Change a domain and train A/B

```bash
python run.py validate --config configs/studies/A_domain_smoke.json --json
python run.py train --config configs/studies/A_domain_smoke.json
python run.py train --config configs/studies/B_smoke.json
```

A overrides `neurons.a.0` to 0.01…0.2. `value` is the fresh population center; bounds and scale define encoding.
B retains `0.02 ≤ w < 0.1`. Domain changes never reinterpret foreign normalized coordinates.

## 2. Fix and free a field

```bash
python run.py validate --config configs/studies/A_fixed_smoke.json --json
python run.py train --config configs/studies/A_fixed_smoke.json
```

Fixing `neurons.a.0=0.1` leaves 47 independent coordinates. To free it in the saved domain, specify the complete new range and scale:

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path('results/smoke_A_fixed/effective_config.json')
c = json.loads(p.read_text())
c['genome']['overrides'] = {'neurons.a.0': {'mutable': True, 'minimum': 0.005, 'maximum': 0.15, 'scale': 'log'}}
c['storage']['output'] = str(Path('results/smoke_A_free').resolve())
Path('results/free_config.json').write_text(json.dumps(c))
PY
python run.py validate --config results/free_config.json --json
```

This restores 48 coordinates. Geometry remains fixed in the current schema. Misspellings, nonpositive logarithmic bounds,
and inconsistent reflected bounds fail before creating a result.

## 3. Finetune in the saved domain

```bash
python run.py train --config configs/studies/A_smoke.json
python run.py finetune --solution results/smoke_A/solution.json --config configs/studies/A_smoke.json --output results/smoke_A_warm
```

The explicit config keeps this example short. Without `--config`, finetuning inherits the solution's domain and mechanics;
it uses the historical warm-start defaults: 50 individuals, 200 generations, seed 43, p=0.3, σ 0.2→0.015,
validation every 20 generations, 8 workers, and genetic polarity alternatives. That is a full run.
Generation zero is eligible; the winner need not belong to the last population. Every setting's origin is recorded in provenance.

## 4. Reject an incompatible domain, then prepare explicitly

Copy `prepare_domain.json`, remove `initialization.prepare`, and validate it with
`--solution results/smoke_A/solution.json`. If the source lies outside [0.005,0.02], the error names `neurons.a.0` and its destination bounds.
Loading never clips silently.

```bash
python run.py prepare --solution results/smoke_A/solution.json --config configs/studies/prepare_domain.json --output results/prepared_A
```

Only the named field changes. `preparation.json` records before/after values and the source hash.
Finetune using the same destination domain **without** preparation operations, or use
`finetune --solution results/prepared_A/solution.json --output results/warm_prepared_A` with full defaults.
For muscle-time studies, `prepare.clip_fields`, `prepare.values`, and `prepare.gain_match` are available.
Gain matching explicitly permits changes to w, N, and the corresponding P row. It approximates steady gain, not transient behavior.

## 5. C joint and staged with equal total budgets

```bash
python run.py train --config configs/studies/C_smoke.json
python run.py train --config configs/studies/C_staged_smoke.json
python run.py compare --runs results/smoke_C results/smoke_C_staged --output results/compare_C
```

Both use two total generations. Joint optimizes the controller and polynomial together.
Staged allocates 1+1: controller and five-target current table → bounded least squares → controller and polynomial.
`search.table_generations` sets the first-stage budget; the remainder belongs to the second stage.
Table validation uses anchor targets; polynomial validation also includes intermediate targets.
Commands depend only on the desired angle. A coefficient bound is not a bound on the resulting polynomial current.

## 6. Mirror symmetry

```bash
python run.py prepare --solution results/smoke_C/solution.json --config configs/studies/prepare_mirror.json --output results/prepared_C_mirror
python run.py finetune --solution results/prepared_C_mirror/solution.json --config configs/studies/C_mirror_smoke.json --output results/smoke_C_mirror_warm
```

Reflection ties both controller halves and enforces `I1(θ)=I0(−θ)`. A cubic solution has 28 independent coordinates instead of 56.
Projection averages physical pairs and preserves the source file; it does not guarantee unchanged performance.
Notebook variants accept both `slow_muscle_mirror` and the published filename spelling `slow_muscle_mirrow`.

## 7. D and continuous commands

```bash
python run.py validate --config configs/studies/D_smoke.json --json
python run.py train --config configs/studies/D_smoke.json
python run.py train --config configs/studies/D_compare_smoke.json
```

These examples explicitly use the C solution created above. D compare screens sources and filter times on validation,
then compares frozen, command_only, and joint. Test is evaluated after selection.

```python
from scripts.AfferentBalance2026.experiments.D_trajectory.controller import AngleController
controller = AngleController('results/smoke_D/solution.json')
controller.advance(10, duration_s=0.01)
controller.advance(-10, duration_s=0.01)  # Preserves the complete hidden state.
```

Use an absolute solution path from another cwd. Desired angle is an input command, not guaranteed tracking.

## 8. Replay the selected individual

```bash
python run.py replay --solution results/smoke_C/solution.json --output results/replay_C --recording-stride 10
python run.py report --manifest results/replay_C/manifest.json --output results/report_C
```

`solution.json` contains the selected individual, including early or generation-zero winners.
`stages/*/final_population.npz` represents the last population, a different object.
Recording stride affects stored samples only; metrics still use every 0.1 ms step.
The report reads the explicit manifest and traces. The existing dated article remains historical evidence.

## 9. Sweeps, checkpoints and publication

```bash
python run.py validate --config configs/studies/sweep_smoke.json --json
python run.py sweep --config configs/studies/sweep_smoke.json
python run.py publish --solution results/smoke_C/solution.json --pointer results/current_C.json
```

`sweep.axes.generations=[2,3]` means independent GA lines with different cooling horizons.
`search.checkpoints=[1,2]` validates an already-running line without restarting its GA.
Legacy A/B `run_sweep.py --generations 30 60 100` retains checkpoint semantics within one line up to 100.
Training, comparison, and report building never update current solutions. Only explicit publication writes a checked pointer.

## Notebooks and legacy data

All views share one selected `solution`, an explicit `protocol`, and `recording_stride`.
D's shared UI is in `experiments/notebook_ui.py`; fonts, colors, initial conditions, filenames, and window settings remain in cells.
The user's 0.5 s activity window, saved outputs, and notes are preserved.
`SPIKINGNN_NOTEBOOK_SMOKE=1` uses an untrained in-memory fixture and does not require ignored results or start a GA.
Restart Jupyter after numerical code changes; running integrators are never automatically replaced.

Legacy JSON/NPZ use explicit schema/domain adapters without rewriting their bytes.
`SpikingNN.Afferents/v3-tensile` is a model identifier, not an import path.
Historical current fields at positions 46/47 require explicit `genome.import_legacy` conversion.
Archived numerical runtime must be executed separately; the current loader does not activate it automatically.
