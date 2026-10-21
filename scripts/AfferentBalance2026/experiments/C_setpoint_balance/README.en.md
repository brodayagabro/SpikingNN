# C_setpoint_balance: prescribed-angle replay

[Русский](README.md) · [Layout](../README.en.md) · [Model](../MODEL_METHODS.en.md)

This experiment uses a fixed 52-field genome. Currents are separate inputs, not genes. `run_experiment.py` and `run_sweep.py` **do not train**: the first replays selected conditions; the second evaluates an initial-condition grid for five targets −45°, −22.5°, 0°, 22.5°, 45°.

```bash
.venv/bin/python scripts/AfferentBalance2026/experiments/C_setpoint_balance/run_experiment.py --genome /path/to/best_genome.json --targets 22.5 --angles -50 0 50 --speeds 0 --currents 100 80
.venv/bin/python scripts/AfferentBalance2026/experiments/C_setpoint_balance/run_sweep.py --genome /path/to/best_genome.json --current-law /path/to/current_law.json
```

Example currents 100/80 illustrate the interface, not a successful solution. Without `--genome`, the untrained configured genome is used. Loaded geometry, mechanics and bounds must agree with the C configuration. The default grid is 10×10 (−50…50°, −1…1 rad/s) for each target. `--currents` and `--current-law` are mutually exclusive. CLI angles are degrees from vertical; speed is rad/s.

`polynomial_study.py` implements `I_i(target)=Σ c_ik (target/45°)^k`: constant, quadratic or cubic. Currents remain constant throughout an episode; measured angle and speed are not used for PD control. `load_current_law` accepts historical coefficient files without implicitly importing historical genomes.

The separate, explicitly invoked `fit_current_law` API optimizes coefficients only, using caller-supplied independent banks and a fixed network. Neither entry scripts nor notebooks invoke fitting automatically. Its `final_coefficients.npz` format is distinct from a population of 52-field genomes.

Replay outputs are `genome_cfg.json`, `best_genome.json` (the supplied fixed controller, without selection), `run.json`, `source_snapshot.zip`, `result.json`, `traces/case_XXX.npz`, and `traces/metrics.csv`. Recording retains every 0.1 ms step. Each trajectory stores its target, initial conditions and currents; `training_performed=false` is explicit. The notebook plots state variables for one case and phase portraits by target; `PLOT_ALL_TARGETS` enables all five targets.

## October 21 control check

`../../analyze_abc_study.py C_setpoint_balance` uses the validation-selected `A_balance/results/best_genome.json`. It saves 550 full trajectories in `results/replay_20261021/`, five-target summaries and `target_*/multipanel.png` figures. Currents come from the configuration (currently `[0, 0]`); this is a control check without current-law training. With zero currents, target changes affect only scoring of the same dynamics. Fitting the current law remains a separately invoked study.
