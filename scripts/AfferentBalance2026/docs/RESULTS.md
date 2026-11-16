# Формат результатов

[English](RESULTS.en.md) · [Руководство](USER_GUIDE.md)

Новый run: `afferent_result_v1`; конфиг: `afferent_experiment_run_v1`; решение: `afferent_solution_v1`;
популяция: `afferent_population_v1`; указатель: `afferent_solution_pointer_v1`.
Результаты по умолчанию создаются в `AfferentBalance2026/results/<run_id>`. Старые каталоги экспериментов остаются на месте.

| Путь внутри run | Содержимое |
|---|---|
| `effective_config.json` | Разрешённые значения, interpreted controller, источники отдельных полей, исходные и абсолютные пути, бюджет |
| `manifest.json` | running/completed/failed, тип операции, модель, протокол, версии, хеши исходников, входы, выбранное решение, фактическое число оценок и время |
| `source_snapshot.zip` | Локальные Python-обработчики, presets/studies, доступные исходники библиотеки; без venv и старых результатов |
| `inputs/` | Неизменные копии входных файлов и переносимые экспорты; исходники не редактируются |
| `banks/` | Фактически использованные условия; A/B/C `cases.npz`, D `specifications.json`; отдельный банк таблицы C |
| `stages/<stage>/initial_solution.json` | Исходный кандидат; это не последнее поколение |
| `stages/<stage>/generations/generation_NNNN.npz` | Все оценённые поколения по умолчанию |
| `stages/<stage>/checkpoints/` | Полная validation-популяция или нулевая особь, решения отдельных checkpoints |
| `stages/<stage>/final_population.npz` | Последняя популяция этой стадии, включая validation |
| `stages/<stage>/history.csv`, `validation_history.csv`, `timing.json` | Обучение, ход выбора, время оценки/записи |
| `selection.json` | Политика, стадия, поколение, индекс особи, validation fitness, SHA-256 решения; test не участвует |
| `solution.json` | Полный домен, физические значения, независимые координаты, закон токов и фильтр D; переносим одним файлом |
| `metrics/validation.npz`, `test.npz` | Независимые метрики выбранной особи; отсутствие test не восполняется скрытым расчётом |
| `traces/`, `figures/` | Запрошенные записи и производные изображения |

Подготовка имеет `preparation.json` вместо GA-стадий. Sweep содержит отдельные `line_NNNN/` с полными runs и общий выбор.
Сравнение создаёт `comparison.json/csv` без симуляций. Отчёт хранит копию manifest и его SHA-256.

## Массивы популяции

`normalized: (N,d)` — независимые координаты; `physical: (N,P)` — полные поля контроллера и команд.
`controller_physical: (N,52)` и `command_values: (N,2,width)` разделяют блоки; A/B имеют пустой командный блок.
`training_scores: (N,)`, `training_metrics: (N,K,9)`, `generation`, `names`, `metric_columns`,
`configuration_sha256`, `rng_state`, `solution_template` задают оценку и интерпретацию.
Validation-архив дополнительно содержит `validation_scores`, `validation_metrics`, `selected_index`.
В genotype-0 архиве сохраняются только фактически вычисленные validation-данные; фиктивных training-метрик нет.

`load_solution(path, rank=1)` из `core.artifacts` читает JSON/указатель/NPZ. Для NPZ rank определяется validation,
если он записан, иначе training. Это правило не заменяет `selection.json`: выбранное решение загружайте из `solution.json`.

## Девять столбцов метрик

| Индекс | A/B/C | D |
|---|---|---|
| 0 | score | score |
| 1 | survival_s | survival_s |
| 2 | rmse_rad | tracking_rmse_rad |
| 3 | peak_error_rad | peak_error_rad |
| 4 | mean_force_N | mean_force_N |
| 5 | success | success |
| 6 | settling_s | in_band_fraction |
| 7 | late_rmse_rad | velocity_rmse_rad_s |
| 8 | final_theta_rad | final_error_rad |

Общий fitness = `0.50 mean(score) + 0.30 mean(worst ceil(K/4)) + 0.20 min(score)`.
Строгий успех и выживание — разные показатели. Сравнение объединяет только одинаковые задачи, физику, критерий и фактические банки.
Даже одинаковые численные fitness разных критериев не образуют общий рейтинг.

## Траектории и воспроизводимость

26 каналов в `columns`, единицы в `units`: время, угол от вертикали, скорость, силы мышц,
V/U, кальций, Ia/II/Ib, активации, спайки, состояния синапсов, мышечный момент.
`dt_ms=0.1`, `recording_stride`, `recording_interval_ms`, `sample_count`, `expected_duration_s`,
`actual_duration_s`, `recorded_duration_s`, `termination` различают интегрирование, запись и раннее завершение.
D дополнительно хранит `reference` (угол/скорость цели) и `commands` (применённые/желаемые токи).
Ток C постоянен внутри одного эпизода. `solution_sha256` связывает запись с конкретным геномом.

`storage.save_generations=false` явно ограничивает последующий аудит поколений, но оставляет checkpoints/final/метрики.
`storage.replay=false` исключает записи; `figures=true` требует replay. Метрики не зависят от stride и наличия картинок.
Метаданные пишутся атомарно; существующие outputs не перезаписываются. Ошибка выполнения оставляет failed manifest.

Старые `afferent_controller_52_v4`, `afferent_setpoint_joint_v1`, `afferent_trajectory_v1` поддерживаются без изменения файлов.
Legacy NPZ требует рядом сохранённый `genome_cfg.json`, а C — описание команд из `run.json`.
Массив без достаточной схемы не угадывается по длине. Архивный runtime не смешивается с текущим процессом.
