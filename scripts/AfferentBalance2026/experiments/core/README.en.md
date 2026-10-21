# Shared core

[Русский](README.md) · [Layout and outputs](../README.en.md)

- `genome.py`: `afferent_controller_52_v4`, named fields, bounds, 50→52 mapping, export and explicit legacy import.
- `fitness.py`: shared aggregation of scenario metrics.
- `evolution.py`: tournament selection, elitism, crossover, mutation and generation observers.
- `run.py`: population evaluation, process pool, A/B training, checkpoints, winner saving and sweeps.

Dynamics is imported directly from `experiments/simulation.py`. Explicit external currents may be passed to the evaluator as an `(individuals, cases, 2)` array; core does not calculate a target-to-current law. A/B training always uses zero currents. Current laws and their separate fitting API live in C.

`GenomeConfig` accepts a JSON path or dictionary. `decode` inserts fixed values after genetic operations; `encode` rejects geometry mismatches and values outside bounds. Set `a1`, `a2` in the configuration before running: every physical genome retains them, but genetic operators never mutate them.
