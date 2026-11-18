# Committed genome snapshot

This is a **provisional** snapshot of the constrained study, taken after 33 of 43 searches completed. The sweep is still running; these are not its final winners.

Fast profiles are selected by maximum validation fitness among completed expanded runs; ties retain the declared run order. Slow and mirror profiles are the independent baseline searches. Test scores and neuron activity do not select winners. Every genome satisfies `0 <= P.0.1, P.1.0 < 0.1`.

`genome_snapshot.json` records the source runs, file hashes, settings and commands. Each genome JSON contains the complete controller configuration. The notebooks load `genomes/fast.json` directly; no result archives or reference arrays are needed to simulate it.

Run the following commands from `topubsrc` in the pinned environment. They start from generic configuration values, without reading the saved genomes. Output directories must not already exist. Exact numerical agreement is conditional on the recorded numerical environment and hardware; a fixed seed alone does not guarantee it on every platform.

| Experiment | Profile | Saved genome | Source run |
|---|---|---|---|
| A | fast | [A_balance/genomes/fast.json](A_balance/genomes/fast.json) | `expanded_A/G200/N50_p0.3_seed42` |
| A | slow | [A_balance/genomes/slow.json](A_balance/genomes/slow.json) | `A_slow` |
| B | fast | [B_balance/genomes/fast.json](B_balance/genomes/fast.json) | `expanded_B/G500/N50_p0.1_seed42` |
| B | slow | [B_balance/genomes/slow.json](B_balance/genomes/slow.json) | `B_slow` |
| C | fast | [C_setpoint_balance/genomes/fast.json](C_setpoint_balance/genomes/fast.json) | `expanded_C/G200/N50_p0.3_seed43` |
| C | slow | [C_setpoint_balance/genomes/slow.json](C_setpoint_balance/genomes/slow.json) | `C_slow` |
| C | mirror | [C_setpoint_balance/genomes/mirror.json](C_setpoint_balance/genomes/mirror.json) | `C_mirror` |

```bash
python run.py train A --profile fast --population 50 --generations 200 --seed 42 --mutation-probability 0.3 --sigma-start 0.2 --sigma-end 0.015 --validation-every 20 --workers 10 --output results/snapshot_A_fast
python run.py train A --profile slow --population 50 --generations 200 --seed 42 --mutation-probability 0.3 --sigma-start 0.2 --sigma-end 0.015 --validation-every 20 --workers 10 --output results/snapshot_A_slow
python run.py train B --profile fast --population 50 --generations 500 --seed 42 --mutation-probability 0.1 --sigma-start 0.2 --sigma-end 0.015 --validation-every 20 --workers 10 --output results/snapshot_B_fast
python run.py train B --profile slow --population 50 --generations 200 --seed 42 --mutation-probability 0.3 --sigma-start 0.2 --sigma-end 0.015 --validation-every 20 --workers 10 --output results/snapshot_B_slow
python run.py train C --profile fast --population 50 --generations 200 --seed 43 --mutation-probability 0.3 --sigma-start 0.2 --sigma-end 0.015 --validation-every 20 --workers 10 --method staged --output results/snapshot_C_fast
python run.py train C --profile slow --population 50 --generations 200 --seed 43 --mutation-probability 0.3 --sigma-start 0.2 --sigma-end 0.015 --validation-every 20 --workers 10 --method joint --output results/snapshot_C_slow
python run.py train C --profile mirror --population 50 --generations 200 --seed 43 --mutation-probability 0.3 --sigma-start 0.2 --sigma-end 0.015 --validation-every 20 --workers 10 --method joint --output results/snapshot_C_mirror
```

The repository intentionally excludes populations, traces, reference arrays, logs, figures and numerical result tables. Run `python run.py study --output fresh_results --jobs 2 --workers 10` to generate the complete study locally. The live result table in `README.md`, `published_results.json`, references and final exports are created by that command. `python verify.py --workers 8 --output verification.json` requires those generated references; it is not available from the genome-only snapshot.
