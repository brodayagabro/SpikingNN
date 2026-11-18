# Справочник команд

[English](COMMANDS.en.md) · [Руководство](USER_GUIDE.md)

Генерируется из локального `build_parser()`. Каждая команда принимает `--help`; установленной команды библиотеки нет.
Флаги другой операции отклоняются parser. Output должен быть новым.

| Команда | Флаг | Обязательность | Смысл |
|---|---|---|---|
| `validate` | `--config` | required | JSON; paths inside it are relative to that file. |
| `validate` | `--solution` | optional | Portable JSON, explicit legacy JSON/NPZ, or current-solution pointer. |
| `validate` | `--workers` | optional | Override persistent process count. |
| `validate` | `--seed` | optional | Override GA seed (bank seeds are separate). |
| `validate` | `--json` | optional | Print the full effective configuration, provenance and inspection. |
| `train` | `--config` | required | JSON; paths inside it are relative to that file. |
| `train` | `--output` | optional | New result directory; CLI paths are relative to cwd. |
| `train` | `--workers` | optional | Override persistent process count. |
| `train` | `--seed` | optional | Override GA seed (bank seeds are separate). |
| `finetune` | `--config` | optional | JSON; paths inside it are relative to that file. |
| `finetune` | `--solution` | required | Portable JSON, explicit legacy JSON/NPZ, or current-solution pointer. |
| `finetune` | `--output` | required | New result directory; CLI paths are relative to cwd. |
| `finetune` | `--workers` | optional | Override persistent process count. |
| `finetune` | `--seed` | optional | Override GA seed (bank seeds are separate). |
| `sweep` | `--config` | required | JSON; paths inside it are relative to that file. |
| `sweep` | `--output` | optional | New result directory; CLI paths are relative to cwd. |
| `sweep` | `--workers` | optional | Override persistent process count. |
| `sweep` | `--seed` | optional | Override GA seed (bank seeds are separate). |
| `replay` | `--config` | optional | JSON; paths inside it are relative to that file. |
| `replay` | `--solution` | required | Portable JSON, explicit legacy JSON/NPZ, or current-solution pointer. |
| `replay` | `--output` | required | New result directory; CLI paths are relative to cwd. |
| `replay` | `--workers` | optional | Override persistent process count. |
| `replay` | `--recording-stride` | optional | Record every Nth 0.1 ms step; metric sampling stays unchanged. |
| `prepare` | `--config` | required | JSON; paths inside it are relative to that file. |
| `prepare` | `--solution` | required | Portable JSON, explicit legacy JSON/NPZ, or current-solution pointer. |
| `prepare` | `--output` | required | New result directory; CLI paths are relative to cwd. |
| `compare` | `--runs` | required | — |
| `compare` | `--output` | required | — |
| `publish` | `--solution` | required | — |
| `publish` | `--pointer` | required | — |
| `report` | `--manifest` | required | — |
| `report` | `--output` | required | — |

From the experiment directory / Из каталога эксперимента:

```bash
python run.py validate --config configs/studies/A_smoke.json --json
```

From another cwd, with absolute paths / Из другого cwd с абсолютными путями:

```bash
REPO=$(pwd)  # Run this assignment at the repository root / Выполнить в корне репозитория.
cd /tmp
"$REPO/.venv/bin/python" "$REPO/scripts/AfferentBalance2026/run.py" validate --config "$REPO/scripts/AfferentBalance2026/configs/studies/A_smoke.json" --json
```

Legacy wrappers / Совместимые обёртки:

- `experiments/{A_balance,B_balance}/run_experiment.py`: --config, --output, --population, --generations, --seed, --workers, --mutation-probability, --duration.
- A/B `run_sweep.py`: --populations, --generations, --probabilities, --seeds; **generations are checkpoints in one continuous line**.
- C: --mode joint/staged/compare; --degree; --table-generations now sets the actual table budget; no ignored training flag.
- D: --sources explicitly lists compatible C/D solutions; no dated source fallback.
- `tools/retrain_tensile.py`: explicit --experiment/--seed-genome/--output; historical warm defaults, shared executor.
- `tools/finalize_retraining.py`: --solution/--pointer, explicit publication only; no archive annotations or notebook clearing.

Compatibility messages identify ignored legacy metadata such as setpoint.training_enabled.
Old constant-current C replay remains an explicit historical API; prefer the portable solution command for new work.
