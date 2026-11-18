# Версии лучших геномов

`fast` — решение до введения границы 50 мс; это не означает отсутствия всех ограничений в старом конфиге. `slow` — асимметричное решение с нижней границей 50 мс. `slow_muscle_mirrow` — новое решение с той же границей и точной зеркальной симметрией. Файлы скопированы без изменения чисел.

| File | τc F, ms | τ1 F, ms | τc E, ms | τ1 E, ms |
|---|---:|---:|---:|---:|
| [best_genome_fast_muscle.json](best_genome_fast_muscle.json) | 7.8309 | 8.2873 | 8.0565 | 23.1892 |
| [best_genome_slow_muscle.json](best_genome_slow_muscle.json) | 52.3115 | 56.8259 | 59.4942 | 52.9767 |
| [best_genome_slow_muscle_mirrow.json](best_genome_slow_muscle_mirrow.json) | 53.7446 | 50.3009 | 53.7446 | 50.3009 |

- `best_genome_fast_muscle.json`: [источник](results/tensile_retrain_20261026/joint_refined/best_genome.json), SHA256 `26f7a3998ac1e1190fec16c6a5c24538f20f8d7a438e92b86f8cb2b941fcd7fa`.
- `best_genome_slow_muscle.json`: [источник](results/mirror_symmetry_20261026/asymmetric_baseline/best_genome.json), SHA256 `2ee96743a6049fa9ac653e9d7e041b29696431dbb01f1ccb5c7c751f36fd8ca4`.
- `best_genome_slow_muscle_mirrow.json`: [источник](results/mirror_symmetry_20261026/C_transfer/best_genome.json), SHA256 `682c56afce76dfdb99b746e1b1087d7b6a704149cd4d5d61a579f2d65ee736b3`.

[Протокол](../MIRROR_SYMMETRY.md) · [Результаты](results/mirror_symmetry_20261026/README.md)

## Выбор в тетрадке

В начале ячейки задайте `GENOME_VARIANT = "slow_muscle_mirrow"` и
`GENOME_PATH = None` для зеркального решения. Другие значения:
`"fast_muscle"`, `"slow_muscle"`, `"current"` (`results/best_genome.json`).
Явный `GENOME_PATH` имеет приоритет; короткое имя файла ищется в папке этого
опыта. После изменения выбора повторно выполните ячейку. Загрузчики
обновляются вместе, поэтому ранее открытое ядро также читает зеркальные
геномы с сокращённым набором независимых координат. Сами JSON не меняются.
