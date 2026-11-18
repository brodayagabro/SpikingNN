# Result formats

[Русский](RESULTS.md) · [Guide](USER_GUIDE.en.md)

Run: `afferent_result_v1`; config: `afferent_experiment_run_v1`; solution: `afferent_solution_v1`;
population: `afferent_population_v1`; pointer: `afferent_solution_pointer_v1`.
New outputs default to `AfferentBalance2026/results/<run_id>`. Existing experiment-local results remain untouched.

| Run-relative path | Meaning |
|---|---|
| `effective_config.json` | Resolved settings, interpreted controller, leaf provenance, original/resolved paths, budgets |
| `manifest.json` | running/completed/failed, operation, model/protocol, versions, source/input hashes, selection, actual evaluation counts/timings |
| `source_snapshot.zip` | Local Python handlers, presets/studies and available library sources; excludes environments and old results |
| `inputs/` | Unmodified input copies and portable exports |
| `banks/` | Actual cases: A/B/C `cases.npz`, D `specifications.json`; separate C table bank |
| `stages/<stage>/initial_solution.json` | Initial candidate, distinct from the final population |
| `stages/<stage>/generations/generation_NNNN.npz` | Every evaluated generation by default |
| `stages/<stage>/checkpoints/` | Validation populations or initial candidate; individual checkpoint solutions |
| `stages/<stage>/final_population.npz` | Last population of that stage with validation |
| `stages/<stage>/history.csv`, `validation_history.csv`, `timing.json` | Learning, selection history and evaluation/storage timing |
| `selection.json` | Policy, stage, generation, individual index, validation fitness and solution hash; test never selects |
| `solution.json` | Full domain, physical values, independent coordinates, commands and D filter; portable as one file |
| `metrics/validation.npz`, `test.npz` | Frozen winner metrics; missing test is never silently recomputed by compare |
| `traces/`, `figures/` | Requested recordings and derived plots |

Preparation records transformations in `preparation.json` instead of GA stages.
A sweep contains complete independent `line_NNNN/` runs and an overall selection.
Comparison writes JSON/CSV without simulation. Reports retain the explicit manifest and its checksum.

## Population arrays

`normalized: (N,d)` contains independent coordinates; `physical: (N,P)` contains complete controller and command fields.
`controller_physical: (N,52)` and `command_values: (N,2,width)` explicitly separate blocks; A/B commands are empty.
`training_scores: (N,)`, `training_metrics: (N,K,9)`, `generation`, `names`, `metric_columns`,
`configuration_sha256`, `rng_state`, and `solution_template` preserve scores and interpretation.
Validation archives additionally contain `validation_scores`, `validation_metrics`, and `selected_index`.
Generation-zero archives only contain measured validation data; no fictitious training metrics are created.

`core.artifacts.load_solution(path, rank=1)` reads JSON/pointers/NPZ. NPZ ranking uses validation when present,
otherwise training. To load the actual selected winner, use `solution.json`, not the final population's first row.

## Nine metric columns

| Index | A/B/C | D |
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

Common fitness is `0.50 mean(score) + 0.30 mean(worst ceil(K/4)) + 0.20 min(score)`.
Strict success differs from survival. Comparison groups runs by task, mechanics, criterion and actual banks;
fitness values from different criteria must not form one ranking.

## Recordings and provenance

`columns` names all 26 channels; `units` declares their units: time, angle from vertical, velocity, muscle forces,
V/U, calcium, Ia/II/Ib, activations, spikes, synaptic states and muscle torque.
`dt_ms=0.1`, `recording_stride`, `recording_interval_ms`, `sample_count`, `expected_duration_s`,
`actual_duration_s`, `recorded_duration_s`, and `termination` distinguish integration, recording and early termination.
D additionally stores `reference` (target angle/velocity) and `commands` (applied/desired currents).
C currents remain constant within an episode. `solution_sha256` identifies the exact solution.

`storage.save_generations=false` explicitly limits generation audits while preserving checkpoints/final/metrics.
`storage.replay=false` omits traces; figures require replay. Metrics do not depend on recording stride or plots.
Metadata writes are atomic; existing outputs are rejected. Runtime failures retain failed manifests.

Legacy controller-52, joint-setpoint and trajectory JSON schemas remain readable without rewriting files.
Legacy NPZ requires its saved `genome_cfg.json`; C also requires the command specification in `run.json`.
Ambiguous arrays are never guessed from their length. Historical numerical runtime must remain isolated.
