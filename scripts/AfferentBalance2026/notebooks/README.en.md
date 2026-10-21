[Русский](README.md)

# Notebooks

[training_phase_trajectories.ipynb](training_phase_trajectories.ipynb) shows
actual trajectories of saved individuals: full Afferents, 0.1 ms step, unscaled
torque and neural-control mode 0.

## Run

```bash
.venv/bin/jupyter lab scripts/AfferentBalance2026/notebooks/training_phase_trajectories.ipynb
```

Select the `SpikingNN (.venv)` kernel and execute cells in order. The repository
is located through `src/SpikingNN/core/Izh_net.py`.
`runtime_support.current_results()` reads `../report/results/current_unscaled.json`.
For another series, set `ROOT` explicitly before calling `runtime_directory(ROOT)`.

The notebook loads the original `factor_1/runtime` and checks its SHA256 hashes.
Historical package names inside snapshots remain unchanged; the relocated outer
path is resolved automatically. Restart the kernel if another numerical version
has already been imported. Dependencies: `../requirements-notebook.txt`.

## Features

- Fitness/success curves and comparisons across generations.
- The leader or up to five best individuals by training fitness at a chosen generation.
- Separately, the actual validation winner of the latest completed checkpoint;
  its training rank only locates that same individual.
- ±8°, ±45° or mixed initial angles with nonzero velocities.
- Different commanded angles and five distant starts for current laws.
- θ–ω phase trajectories, angle over time, metrics and early falls.
- Custom `CASES`: `[target,theta0,omega0,kick_t,kick_dw]`, in radians and seconds;
  `kick_t=-1` disables the disturbance.

Balance uses a zero target and zero applied currents. For a target-angle law,
the saved mapping determines a time-constant current without a PD signal.
Hidden states are reset identically for each start. Curves come from integrating
the complete system, not interpolating between genomes.

Training and validation use different selection rules; generations are not one
ancestral lineage. A fall's late RMSE is the 90° penalty rather than a measured
steady-state error.

## Code and files

[phase_viewer.py](phase_viewer.py) implements the catalogue, replay, widgets,
plotting and export. [training_history.py](../experiments/core/training_history.py)
defines generation snapshots. Edit the clean source notebook directly; there
is no separate `build_training_phase_notebook.py` in the working directory.

Sweep `epochs_*/generations/` checkpoints form one 1…200 history. During training,
rerun the catalogue and widget cells; incomplete atomic writes are skipped.
Exports go to the selected series' `notebook_exports/`: PNG/PDF, complete NPZ
traces, metric CSV and JSON containing the source-generation hash. Git ignores
these files; the source notebook contains no outputs.

See [complete recomputation](../experiments/RECOMPUTE_UNSCALED.en.md).
The historical `../report/edit_figures.ipynb` retains its code with outputs
cleared. Its ×10 data are in the trash; it is not the current population viewer.
