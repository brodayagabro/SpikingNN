# Версии лучших геномов

`fast` — решение до введения границы 50 мс; это не означает отсутствия всех ограничений в старом конфиге. `slow` — асимметричное решение с нижней границей 50 мс. `slow_muscle_mirrow` — новое решение с той же границей и точной зеркальной симметрией. Файлы скопированы без изменения чисел.

| File | τc F, ms | τ1 F, ms | τc E, ms | τ1 E, ms |
|---|---:|---:|---:|---:|
| [best_genome_fast_muscle.json](best_genome_fast_muscle.json) | 7.4435 | 27.3347 | 6.7552 | 14.7166 |
| [best_genome_slow_muscle.json](best_genome_slow_muscle.json) | 53.3485 | 54.3107 | 53.7710 | 53.8773 |
| [best_genome_slow_muscle_mirrow.json](best_genome_slow_muscle_mirrow.json) | 53.2255 | 55.2849 | 53.2255 | 55.2849 |

- `best_genome_fast_muscle.json`: [источник](results/tensile_retrain_20261026/staged_refined/best_genome.json), SHA256 `93d7cf09502d2cd245fff8fc40850d778a90bebc95b8327e938e7b046933e5ea`.
- `best_genome_slow_muscle.json`: [источник](results/mirror_symmetry_20261026/asymmetric_baseline/best_genome.json), SHA256 `065c320ead2f1d44061eaafe0a72aaced29e3c14b0eb770b2d65b4be363fae37`.
- `best_genome_slow_muscle_mirrow.json`: [источник](results/mirror_symmetry_20261026/joint/best_genome.json), SHA256 `3b99adb3877c4df3043d88aeedb767cd8195831061ed1f65f7b47a08138d69d9`.

[Протокол](../MIRROR_SYMMETRY.md) · [Результаты](results/mirror_symmetry_20261026/README.md)

## Выбор в тетрадке

В начале ячейки задайте `GENOME_VARIANT = "slow_muscle_mirrow"` и
`GENOME_PATH = None` для зеркального решения. Другие значения:
`"fast_muscle"`, `"slow_muscle"`, `"current"` (`results/best_genome.json`).
Явный `GENOME_PATH` имеет приоритет; короткое имя файла ищется в папке этого
опыта. После изменения выбора повторно выполните ячейку. Загрузчики
обновляются вместе, поэтому ранее открытое ядро также читает зеркальные
геномы с сокращённым набором независимых координат. Сами JSON не меняются.
