# Best-genome versions

`fast` is the winner before the 50 ms floor; it still retains the bounds of its historical configuration. `slow` is the asymmetric winner with the 50 ms floor. `slow_muscle_mirrow` adds exact reflection symmetry under that same floor. Files are copied without changing any values.

| File | τc F, ms | τ1 F, ms | τc E, ms | τ1 E, ms |
|---|---:|---:|---:|---:|
| [best_genome_fast_muscle.json](best_genome_fast_muscle.json) | 7.8309 | 8.2873 | 8.0565 | 23.1892 |
| [best_genome_slow_muscle.json](best_genome_slow_muscle.json) | 52.3115 | 56.8259 | 59.4942 | 52.9767 |
| [best_genome_slow_muscle_mirrow.json](best_genome_slow_muscle_mirrow.json) | 53.7446 | 50.3009 | 53.7446 | 50.3009 |

- `best_genome_fast_muscle.json`: [source](results/tensile_retrain_20261026/joint_refined/best_genome.json), SHA256 `26f7a3998ac1e1190fec16c6a5c24538f20f8d7a438e92b86f8cb2b941fcd7fa`.
- `best_genome_slow_muscle.json`: [source](results/mirror_symmetry_20261026/asymmetric_baseline/best_genome.json), SHA256 `2ee96743a6049fa9ac653e9d7e041b29696431dbb01f1ccb5c7c751f36fd8ca4`.
- `best_genome_slow_muscle_mirrow.json`: [source](results/mirror_symmetry_20261026/C_transfer/best_genome.json), SHA256 `682c56afce76dfdb99b746e1b1087d7b6a704149cd4d5d61a579f2d65ee736b3`.

[Study protocol](../MIRROR_SYMMETRY.en.md) · [Results](results/mirror_symmetry_20261026/README.en.md)

## Notebook selection

At the top of the cell, set `GENOME_VARIANT = "slow_muscle_mirrow"` and
`GENOME_PATH = None` for the mirror solution. Other choices are
`"fast_muscle"`, `"slow_muscle"`, and `"current"` (`results/best_genome.json`).
An explicit `GENOME_PATH` takes precedence; bare filenames resolve inside this
experiment. Rerun the cell after changing the selection. Dependent loaders
are refreshed together so an existing kernel can read mirror genomes with
fewer independent coordinates. The JSON files themselves are unchanged.
