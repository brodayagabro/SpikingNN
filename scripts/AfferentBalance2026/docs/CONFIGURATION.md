# Справочник конфигурации

[English](CONFIGURATION.en.md) · [Руководство](USER_GUIDE.md)

Генерируется `python tools/check_workflow.py --write-docs`, проверяется `--check-docs`.
Схема запуска: `afferent_experiment_run_v1`. Один локальный versioned preset → JSON запроса → явные CLI-замены.
Массивы заменяются целиком. Произвольной цепочки наследования нет. Сохранённый полный effective config допустим как самостоятельный домен.
Неизвестные/неуместные ключи дают ошибку. Пути JSON отсчитываются от объявившего их файла, CLI — от cwd.
В `provenance` сохраняются источник каждого значения и исходные/разрешённые пути.

Таблица показывает фактические defaults. Пустые блоки задач неприменимы. Для D нужны явные пути источников.
Finetune без конфига наследует физический домен решения и использует явные defaults дообучения из руководства.
Операции prepare применимы только к prepare/validate и не выполняются скрыто при обучении.

| Ключ | Тип | Единица | Default по preset | Применимые presets |
|---|---|---|---|---|
| `experiment` | string | — | A: "A"; B: "B"; C: "C"; D: "D" | A, B, C, D |
| `genome.command` | object | — | {} | A, B |
| `genome.command.coefficient_bound_nA` | number | nA | 1000.0 | C, D |
| `genome.command.degree` | integer | — | 3 | C, D |
| `genome.command.table_current_bound_nA` | number | nA | 1000.0 | C, D |
| `genome.mirror_symmetry` | boolean | — | false | A, B, C, D |
| `genome.overrides` | object | — | {} | A, B, C, D |
| `genome.parameters` | array | — | 52 named parameter records | A, B, C, D |
| `genome.schema` | string | — | "afferent_controller_52_v4" | A, B, C, D |
| `initialization.candidate_seed_offset` | integer | — | 1000 | A, B, C, D |
| `initialization.current_tau_s` | number | s | 0.0 | A, B, C, D |
| `initialization.extra_solutions` | array | — | [] | A, B, C, D |
| `initialization.local_sigma` | number | — | 0.012 | A, B, C, D |
| `initialization.prepare.clip_fields` | array | — | [] | A, B, C, D |
| `initialization.prepare.gain_match` | boolean | — | false | A, B, C, D |
| `initialization.prepare.mirror_projection` | boolean | — | false | A, B, C, D |
| `initialization.prepare.values` | object | — | {} | A, B, C, D |
| `initialization.solution` | null/path | — | null | A, B, C, D |
| `initialization.sources` | array | — | [] | A, B, C, D |
| `initialization.strategy` | string | — | A: "standard"; B: "standard"; C: "standard"; D: "local_d" | A, B, C, D |
| `model.architecture` | string | — | "two_neurons_one_limb_v1" | A, B, C, D |
| `model.dt_ms` | number | ms | 0.1 | A, B, C, D |
| `model.mechanics.gravity_m_s2` | number | m/s² | 9.81 | A, B, C, D |
| `model.mechanics.length_m` | number | m | 0.3 | A, B, C, D |
| `model.mechanics.mass_kg` | number | kg | 0.3 | A, B, C, D |
| `model.mechanics.viscosity` | number | — | 0.001 | A, B, C, D |
| `model.name` | string | — | "SpikingNN.Afferents/v3-tensile" | A, B, C, D |
| `preset` | string | — | A: "A_balance_tau50_v1"; B: "B_balance_tau50_v1"; C: "C_setpoint_balance_tau50_v1"; D: "D_trajectory_tau50_v1" | A, B, C, D |
| `protocol.balance` | object | — | {} | D |
| `protocol.balance.bank_seeds.test` | integer | — | A: 2026102003; B: 2026102003; C: 2026102223 | A, B, C |
| `protocol.balance.bank_seeds.training` | integer | — | A: 2026102001; B: 2026102001; C: 2026102201 | A, B, C |
| `protocol.balance.bank_seeds.validation` | integer | — | A: 2026102002; B: 2026102002; C: 2026102202 | A, B, C |
| `protocol.balance.duration_s` | number | s | 10.0 | A, B, C |
| `protocol.balance.initial_conditions.angle_range_deg` | array | deg | [-50.0,50.0] | A, B, C |
| `protocol.balance.initial_conditions.speed_range_rad_s` | array | s | [-1.0,1.0] | A, B, C |
| `protocol.balance.test_count` | integer | — | A: 96; B: 96; C: 24 | A, B, C |
| `protocol.balance.training_count` | integer | — | A: 24; B: 24; C: 8 | A, B, C |
| `protocol.balance.validation_count` | integer | — | A: 48; B: 48; C: 12 | A, B, C |
| `protocol.criterion` | string | — | A: "balance_v1"; B: "balance_v1"; C: "balance_v1"; D: "trajectory_v1" | A, B, C, D |
| `protocol.evaluate_test` | boolean | — | true | A, B, C, D |
| `protocol.replay.angles_deg` | array | deg | A: [-50.0,-25.0,0.0,25.0,50.0]; B: [-50.0,-25.0,0.0,25.0,50.0]; C: [-50.0,0.0,50.0]; D: null | A, B, C, D |
| `protocol.replay.duration_s` | null/path | s | null | A, B, C, D |
| `protocol.replay.speeds_rad_s` | array | s | A: [0.0]; B: [0.0]; C: [0.0]; D: null | A, B, C, D |
| `protocol.replay.targets_deg` | array | deg | A: [0.0]; B: [0.0]; C: [-45.0,-22.5,0.0,22.5,45.0]; D: null | A, B, C, D |
| `protocol.selection` | string | — | A: "final_validation_v1"; B: "final_validation_v1"; C: "checkpoint_validation_v1"; D: "d_comparison_validation_v1" | A, B, C, D |
| `protocol.setpoint` | object | — | {} | A, B, D |
| `protocol.setpoint.targets_deg` | array | deg | [-45.0,-22.5,0.0,22.5,45.0] | C |
| `protocol.setpoint.test_targets_deg` | array | deg | [-45.0,-39.375,-33.75,-28.125,-22.5,-16.875,-11.25,-5.625,0.0,5.625,11.25,16.875,22.5,28.125,33.75,39.375,45.0] | C |
| `protocol.setpoint.validation_targets_deg` | array | deg | [-45.0,-33.75,-22.5,-11.25,0.0,11.25,22.5,33.75,45.0] | C |
| `protocol.trajectory` | object | — | {} | A, B, C |
| `protocol.trajectory.bank_seeds.test` | integer | — | 2026102303 | D |
| `protocol.trajectory.bank_seeds.training` | integer | — | 2026102301 | D |
| `protocol.trajectory.bank_seeds.validation` | integer | — | 2026102302 | D |
| `protocol.trajectory.filter_times_s` | array | s | [0.0,0.1,0.5] | D |
| `protocol.trajectory.hold_s` | number | s | 4.0 | D |
| `protocol.trajectory.ramp_s` | number | s | 2.0 | D |
| `protocol.trajectory.test_count` | integer | — | 18 | D |
| `protocol.trajectory.training_count` | integer | — | 8 | D |
| `protocol.trajectory.validation_count` | integer | — | 10 | D |
| `runtime.workers` | integer | — | A: 8; B: 8; C: 18; D: 1 | A, B, C, D |
| `schema` | string | — | "afferent_experiment_run_v1" | A, B, C, D |
| `search.checkpoints` | array | — | A: []; B: []; C: [1,30,60,100,150,200]; D: [] | A, B, C, D |
| `search.generations` | integer | — | A: 200; B: 200; C: 200; D: 80 | A, B, C, D |
| `search.method` | string | — | A: "joint"; B: "joint"; C: "joint"; D: "compare" | A, B, C, D |
| `search.mutation_probability` | number | — | A: 0.3; B: 0.3; C: 0.1; D: 0.1 | A, B, C, D |
| `search.polarity_seeds` | boolean | — | false | A, B, C, D |
| `search.population` | integer | — | A: 50; B: 50; C: 50; D: 30 | A, B, C, D |
| `search.seed` | integer | — | A: 42; B: 42; C: 43; D: 44 | A, B, C, D |
| `search.sigma_end` | number | — | A: 0.015; B: 0.015; C: 0.015; D: 0.004 | A, B, C, D |
| `search.sigma_start` | number | — | A: 0.2; B: 0.2; C: 0.2; D: 0.04 | A, B, C, D |
| `search.table_generations` | null/path | — | null | A, B, C, D |
| `search.validation_every` | integer | — | 20 | A, B, C, D |
| `storage.figures` | boolean | — | false | A, B, C, D |
| `storage.output` | null/path | — | null | A, B, C, D |
| `storage.recording_stride` | integer | — | 1 | A, B, C, D |
| `storage.replay` | boolean | — | true | A, B, C, D |
| `storage.save_generations` | boolean | — | true | A, B, C, D |
| `sweep.axes` | object | — | {} | A, B, C, D |

`genome.parameters` содержит ровно именованную схему контроллера из preset: `index`, `name`, `value`,
`minimum`, `maximum`, `scale`, `mutable`, `unit`, `description`, `system_parameter`.
`genome.overrides.<точное имя>` принимает только value/mutable/minimum/maximum/scale. Фиксация требует value;
освобождение ранее fixed-поля требует обе границы и scale. Геометрия остаётся фиксированной; зеркальные пары имеют одинаковые области.
Версии схемы запуска и генома разделены. Структурное изменение сети требует нового кодека/модели, а не добавочного JSON-поля.

`search.generations` — общий бюджет C staged и бюджет каждой обучаемой ветки D compare.
`table_generations=null` выделяет таблице floor(total/2). `search.checkpoints` не перезапускает GA.
`sweep.axes` принимает непустые массивы уникальных population/generations/seed/mutation_probability/method/degree;
каждая комбинация — независимая линия. Validate показывает развёрнутый бюджет.
`coefficient_bound_nA` ограничивает отдельные коэффициенты, `table_current_bound_nA` — значения таблицы токов.
`storage.recording_stride` — положительное целое; метрики всегда используют шаг 0,1 мс.

Defaults команд определяет parser, научные defaults — эти presets.
`validate --json` показывает значения, источники, fixed-поля, размерности, банки и политику выбора.
