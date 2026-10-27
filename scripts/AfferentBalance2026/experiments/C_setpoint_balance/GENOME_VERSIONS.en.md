# Best-genome versions

`fast` is the winner before the 50 ms floor; it still retains the bounds of its historical configuration. `slow` is the asymmetric winner with the 50 ms floor. `slow_muscle_mirrow` adds exact reflection symmetry under that same floor. Files are copied without changing any values.

| File | τc F, ms | τ1 F, ms | τc E, ms | τ1 E, ms |
|---|---:|---:|---:|---:|
| [best_genome_fast_muscle.json](best_genome_fast_muscle.json) | 7.4435 | 27.3347 | 6.7552 | 14.7166 |
| [best_genome_slow_muscle.json](best_genome_slow_muscle.json) | 53.3485 | 54.3107 | 53.7710 | 53.8773 |
| [best_genome_slow_muscle_mirrow.json](best_genome_slow_muscle_mirrow.json) | 53.2255 | 55.2849 | 53.2255 | 55.2849 |

- `best_genome_fast_muscle.json`: [source](results/tensile_retrain_20261026/staged_refined/best_genome.json), SHA256 `93d7cf09502d2cd245fff8fc40850d778a90bebc95b8327e938e7b046933e5ea`.
- `best_genome_slow_muscle.json`: [source](results/mirror_symmetry_20261026/asymmetric_baseline/best_genome.json), SHA256 `065c320ead2f1d44061eaafe0a72aaced29e3c14b0eb770b2d65b4be363fae37`.
- `best_genome_slow_muscle_mirrow.json`: [source](results/mirror_symmetry_20261026/joint/best_genome.json), SHA256 `3b99adb3877c4df3043d88aeedb767cd8195831061ed1f65f7b47a08138d69d9`.

[Study protocol](../MIRROR_SYMMETRY.en.md) · [Results](results/mirror_symmetry_20261026/README.en.md)

## Notebook selection

At the top of the cell, set `GENOME_VARIANT = "slow_muscle_mirrow"` and
`GENOME_PATH = None` for the mirror solution. Other choices are
`"fast_muscle"`, `"slow_muscle"`, and `"current"` (`results/best_genome.json`).
An explicit `GENOME_PATH` takes precedence; bare filenames resolve inside this
experiment. Rerun the cell after changing the selection. Dependent loaders
are refreshed together so an existing kernel can read mirror genomes with
fewer independent coordinates. The JSON files themselves are unchanged.
