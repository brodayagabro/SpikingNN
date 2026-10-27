<!-- mirror-symmetry-20261026 -->

Separate study: [exact reflection symmetry of controller and currents](../MIRROR_SYMMETRY.en.md). Outputs are in `results/mirror_symmetry_20261026/`; the previous canonical genome is preserved.

<!-- muscle-tau50-20261026 -->

**Live step check.** The new genome completed 0°, +20°, −20° and +45° holds, but the abrupt +45°→−45° change caused a fall after 0.5864 s. The widget stops at a fall; Reset restores the initial state. This is measured controller behavior in this series, not an execution error.

# Retraining with muscle relaxation times ≥50 ms

Current winner: `C_transfer`. Completed tests: **14/18**; strict successes: **0/18**. Selected by validation only. Both muscle A parameters are fixed at 0.37. Both muscles use tau_c in [50,100] ms and tau_1 in [50,200] ms. Integration step: 0.1 ms. No PD/PID controller or torque multiplier.

[Current results and file guide](results/README.en.md). [Full article](../../report/REPORT.en.md).

## Historical documentation (before retraining)

<!-- retraining-20261026 -->

# Retraining on corrected tensile mechanics

Current winner: `joint_refined`. Completed tests: **18/18**; strict successes: **0/18**. Selected by validation only. Both muscle A parameters are fixed at 0.37. Integration step: 0.1 ms. No PD/PID controller or torque multiplier.

[Current results and file guide](results/README.en.md). [Full article](../../report/REPORT.en.md).

## Historical documentation (before retraining)

# D — tracking a changing target angle

[Русский](README.md)

C already learns a mapping from the desired angle to two neural input currents.
D tests continuous position tracking with those controllers. Actual motion may
lag or oscillate around the reference; a fitted current polynomial alone does
not guarantee accurate tracking.

```text
desired angle → two polynomials → [current filter] → neurons → muscles → limb
                                                     ↑         ↓        ↓
                                                     └──── full Afferents
```

Measured angle, velocity and tracking error are absent from the command law.
There is no PD/PID controller. The network retains its nonlinear Ia/II/Ib
feedback. All simulations use **0.1 ms**, mechanics without the ×10 multiplier,
fixed `A=0.37`, `a1=0.2`, `a2=0.05`, and muscle time constants ≥5.1 ms.

## Tensile-torque correction, 2026-10-26

The active mechanics is `SpikingNN.Afferents/v3-tensile`: positive tension gives `M = -F_flex*h_flex + F_ext*h_ext`. The retained old winner completed **0/18** original test episodes after the correction; no retraining was performed. Historical training scores below describe the old sign. [New metrics and phase portraits](results/torque_sign_check_20261026/README.en.md).

Restart the Jupyter kernel and reopen the updated notebook before running it. Old outputs were backed up under `results/torque_sign_check_20261026/notebooks_before_correction/` and cleared from the active notebooks.

## Stateful angle interface

[controller.py](controller.py) implements `AngleController`. From the repository root:

```python
from scripts.AfferentBalance2026.experiments.D_trajectory import AngleController

controller = AngleController(
    "scripts/AfferentBalance2026/experiments/D_trajectory/results/best_genome.json",
    initial_angle_deg=-30,
    initial_velocity_rad_s=0.2,
    target_angle_deg=0,
)
state = controller.advance(target_angle_deg=15, duration_s=0.01)
print(state["angle_deg"], state["input_currents"])

controller.target_angle_deg = 20  # Preserve all physical state.
state = controller.step()         # Advance exactly 0.1 ms.
```

A combined C `best_genome.json` can be loaded directly; quadratic polynomials
are padded with zero cubic terms. External angles are degrees relative to the
unstable vertical, speeds are rad/s, and durations are seconds. Commands outside
the calibrated −45…45° interval raise an exception instead of silently clipping.

- `step(angle)` advances one 0.1 ms step; the angle is optional.
- `advance(angle, duration_s)` holds a command over a block, such as 10 ms.
- Assigning `target_angle_deg` changes only the command, without advancing time.
- `snapshot()` returns copied observations: currents, membrane potentials,
  forces, calcium, activation, afferents and synaptic currents.
- `reset(...)` explicitly clears all dynamic history. A fall beyond ±90° or
  nonfinite state stops integration and requires a reset before continuing.

All neuronal, synaptic, muscle, afferent and limb states persist between commands.
This is an online simulation interface; wall-clock scheduling and hard real-time
hardware guarantees are outside its scope. The first call can compile Numba;
warm up once and reset before starting an external command loop.

For a signal updated every 10 ms:

```python
import numpy as np

controller.reset(target_angle_deg=0)
history = []
for k in range(6000):
    t = k * 0.01
    target = 30 * np.sin(2 * np.pi * t / 20)
    state = controller.advance(target, 0.01)
    history.append(state)
    if state["failure"] != "none":
        break
```

This example uses held commands. Scientific D batch evaluations update the
smooth reference at every internal 0.1 ms step.

## Slider animation in one self-contained cell

The final `LIVE_ANGLE_SIMULATION` cell in
[explore_genome.ipynb](explore_genome.ipynb) can run directly in the `spikingnn`
kernel without any preceding cells. Imports, settings, the
`LivePendulumSimulation` class, button callbacks and launch are all defined
in that cell. No additional UI source files are needed; the existing
`AngleController` computes physics using `results/best_genome.json`.

- The −45…45° slider sets a green dashed desired-direction line. The dark segment
  shows the actual limb. Positive angles point right of the upward vertical.
- The red string is the flexor; the blue string is the extensor. The brown `mg`
  arrow points down from the midpoint of the uniform rod. Colored muscle arrows
  show **tendon tension toward the fixed anchors along the strings**.
  String lengths match the model geometry. Numerical forces and signed moments
  are labeled; arrow lengths use a display scale and cap.
- The adjacent chart shows desired and actual angles over the latest **10 s**.
  Commands/history update every 10 ms; physical integration remains at 0.1 ms.
- **Стоп / Старт** (stop/start) freezes and resumes the same state; paused time
  does not count as simulated time. **Сброс** (reset) restores initial angle and
  speed, clears history and retains the current slider target.
- **Снимок PNG** (PNG snapshot) pauses the simulation and saves both panels at
  high resolution under `results/live_snapshots/`. The saved path appears below
  the animation. Stopping is also sufficient for an ordinary screen capture.

At the top of the same cell, edit `FONT_FAMILY`, `FONT_SIZE`, `TITLE_SIZE`,
`LABEL_SIZE`, `TICK_SIZE`, `LEGEND_SIZE`, `ANNOTATION_SIZE` and `WIDGET_FONT_SIZE`.
Other settings include initial conditions, genome path, `FPS`, `FIGSIZE`,
`FRAME_DPI`/`SNAPSHOT_DPI`, arrow scale and snapshot directory.
`GRAVITY_MIN_ARROW_M=0.06` sets the minimum visible gravity-arrow length in
diagram coordinates without changing physical `mg`. Rerun the cell
when changing settings: its old background task is closed before a new
simulation is created.

The default is `AUTO_START=True`, `TIME_SCALE=1`: animation starts immediately
and follows wall-clock time. The asynchronous loop leaves the kernel available
for widget callbacks. Slow rendering can cause lag; no physical steps are
skipped. Crossing ±90° stops integration and requires explicit reset. The
mathematical model, genome and current law are unchanged for animation.

## Separate notebook with live neuronal and afferent activity

[live_activity.ipynb](live_activity.ipynb) contains an extended copy of the live
simulation in **one code cell**, `LIVE_ACTIVITY_SIMULATION`. Run it in the
`spikingnn` kernel.

At the top of the cell, `GENOME_VARIANT = "slow_muscle_mirrow"` selects the mirror
genome. Other choices are `"fast_muscle"`, `"slow_muscle"` and `"current"`
(the latter loads `results/best_genome.json`). With `GENOME_PATH = None`, the
named variant is used. An explicit override such as
`GENOME_PATH = "best_genome_slow_muscle_mirrow.json"` resolves the bare filename
inside D regardless of Jupyter's working directory. Absolute paths also override
`GENOME_VARIANT`. The actual path is printed when the cell runs.

If the notebook is already open, reload it from disk and rerun this cell.
It refreshes dependent Python loaders that may still be cached from before
symmetry support was added: first the helper itself, then `core.genome`, and
then its dependants. This order also fixes
`ImportError: cannot import name 'MIRROR_INDICES'` in an existing kernel.
This fixes the reproduced
`Expected 56 finite coordinates in [0, 1]` error for a mirror D genome containing
28 independent coordinates. All 60 physical fields remain present; the genome
does not require reformatting or zero padding.
[Sources of the three versions](GENOME_VERSIONS.en.md).

- **A** — limb, muscles, desired direction and forces.
- **B** — desired and actual angle.
- **C** — membrane potentials of neurons N0/N1 (mV), or a spike raster. The
  “Нейроны” selector switches views without resetting the physical state.
- **D** — the sum of **Ia + II + Ib** separately for the flexor and extensor.
  These are model units, not Hz or the weighted afferent feedback current.
- **E** — flexor and extensor forces in newtons.

The four right-hand plots share a time axis; `WINDOW_S` sets the rolling window.
`RecordingAngleController`, defined in the same cell, uses the existing Numba
integrator and records **every 0.1 ms step**. Voltage and summed afferent samples
are retained at that resolution in a bounded rolling window; all spike times
are retained. Curves use time-bin minima/maxima (`MAX_DISPLAY_POINTS`) to limit
drawing cost while retaining narrow peaks. This changes display resolution,
not the model's integration step. N0/N1 are neuron indices, not exclusive
connections to individual muscles.

The slider, start/stop, reset, font settings and enlarged `mg` arrow are retained.
PNG snapshots include all five panels and go to `results/live_activity_snapshots/`.
`NEURON_VIEW` sets the initial neuronal view, `FIGSIZE` sets figure size, and
`FPS` controls refresh rate. Reduce `FPS` or `FRAME_DPI` if rendering is slow;
simulated time never skips physical integration steps to catch up.

## Genome and current commands

`TrajectoryGenome` contains **60 physical fields** and **56 independent coordinates
without symmetry or 28 with mirror symmetry**:
the original 52 fields, including fixed geometry and A, plus eight coefficients
of two cubic polynomials. All parameter descriptions are English. For neuron i:

`x = target_angle_deg / 45`, `I_desired[i] = Σ(k=0…3) c[i,k] x**k`.

For nonzero filter time `tau`:
`I[n] = I[n−1] + (1 − exp(−0.0001/tau)) (I_desired[n] − I[n−1])`.
With `tau=0`, the command is applied directly. Initial currents equal the
polynomial at the initial desired angle. Filter time is a hyperparameter, not
another gene. Currents use model units; `*_nA` field names are inherited from C.

## Training protocol

[genome_cfg.json](genome_cfg.json) defines the protocol. Four C winners are
imported: `joint` and `staged` from `joint_sweep_20261022` and
`joint_sweep_tau_gt5_20261022`. All imported values are checked against the new
bounds and mechanics. Existing C results are preserved.

1. Compare four frozen genomes and filter times 0 / 0.1 / 0.5 s on validation.
2. Refine the best pair in two modes: `command_only` changes eight current
   coefficients; `joint` changes all 56 mutable coordinates. Both reuse core GA.
3. Each run uses 30 individuals × 80 generations, seed 44, per-gene mutation
   probability 0.1, normalized σ annealed from 0.04 to 0.004. Initial candidates
   are close to the seed (σ=0.012).
4. Validate the full population at generations 1/20/40/60/80. Include the original
   seed as generation 0, so refinement is allowed to retain the unchanged seed.
5. Select the overall winner using validation **before** evaluating held-out tests.

Banks contain 8 training, 10 validation and 18 test trajectories, with independent
seeds 2026102301/02/03. Sine amplitudes are 10…45°, frequencies 0.04…0.10 Hz,
with random phases and offsets keeping the complete command inside ±45°.
Initial angles span −45…45°, speeds −0.5…0.5 rad/s. Two training anchors use
amplitude 45° / period 20 s and 30° / 10 s. All specifications are saved.

Each episode holds its initial target for 4 s, starts the sinusoid smoothly over
2 s using `6u⁵−15u⁴+10u³`, then covers at least two complete periods. No state is
reset at a reversal or cycle boundary. A separate diagnostic grid uses amplitudes
15/30/45° × periods 10/20/40 s. The 40 s period is outside the training range;
these diagnostics do not select a genome.

## Fitness

The initial 4 s acquisition interval is excluded from tracking quality. Errors
are relative to the **moving** desired angle and its analytic velocity:
`e=theta−theta_target`, `ev=omega−omega_target`.

Episode score:

`0.10 survival_fraction + 0.55 mean(exp(−(e/5°)²))`
`+ 0.20 mean(exp(−(ev/0.15 rad/s)²)) + 0.15 in_band_fraction`.

The band is `|e|≤3°` and `|ev|≤0.35 rad/s`. Success requires full survival,
at least 95% of scored time inside the band and RMSE≤3°. Missing samples after
a fall earn zero reward; denominators retain the planned horizon. RMSE covers
the observed part and must be interpreted alongside survival/failure status.
Core aggregates episodes as 0.5 mean + 0.3 mean of the worst quarter + 0.2 minimum.
D replaces C's static settling criterion with an explicit tracking criterion.

## Execution and files

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
.venv/bin/python scripts/AfferentBalance2026/experiments/D_trajectory/run_sweep.py \
  --output scripts/AfferentBalance2026/experiments/D_trajectory/results/my_study \
  --workers 18
```

Output must be a new directory. `run_experiment.py` runs the same protocol.
`--population`, `--generations`, `--seed` and `--config` explicitly override settings.

| File | Purpose |
|---|---|
| [controller.py](controller.py) | Angle input class, combined genome, C/D import |
| [trajectory.py](trajectory.py) | References, streaming integration and tracking metrics |
| [run_experiment.py](run_experiment.py) | Frozen comparison, two refinements, selection and export |
| [audit_trajectory.py](audit_trajectory.py) | Verify sources, generations, selection and full dynamics of a completed study |
| [plots.py](plots.py) | Plots from saved numerical trajectories |
| [explore_genome.ipynb](explore_genome.ipynb) | Interactive angle commands and state visualization |
| `results/best_genome.json` | Copy of the validation-selected winner for class/notebook use |
| `results/trajectory_sweep_20261023/` | Completed study and its local detailed README |

Inside a study, `sources/` stores four original JSON genomes, `banks.json` stores
all episodes, and `genome_cfg.json` records the actual configuration.
`provenance.json` and `source_snapshot.zip` retain hashes and source code.
`frozen_validation.csv` compares all 12 frozen variants. `command_only/` and
`joint/` contain **every** `generation_XXX.npz`, `history.csv`,
`validation_XXX.npz`, `validation_history.csv`, `resume.json`, `selection.json`
and `best_genome.json`. The final population is `generation_080.npz`; the selected
genome may come from an earlier generation. `resume.json` retains RNG state and
the search subspace for programmatic continuation through core
`optimize(resume_state=...)`; automatic CLI resume is not implemented.

Each of `frozen/`, `command_only/`, `joint/` contains `test/` (18 trajectories)
and `diagnostics/` (9 trajectories), including falls. Each NPZ stores `trace`
(26 channels), `reference` (desired angle/velocity), `commands` (two applied and
two desired currents), parameters, metrics and column names. Every **0.1 ms**
sample is retained, including reset. Current row n>0 applies to the interval
ending at that row; its angle command is taken from reference row n−1.
`figures/` includes tracking comparison, amplitude/period RMSE, phase
trajectories and all states for one shared example (30°, period 20 s).
Numerical results are excluded from Git. One seed and finite trajectories
do not establish global stability or statistical superiority.

## Verification

`tests/test_trajectory.py` checks hidden-state continuity, command switches without
reset, filter units, variable-current agreement with original library objects,
analytic reference velocity, disjoint banks, independent metric recomputation,
fixed physiology, generation persistence, generation-zero selection and identical
serial/parallel evaluations. Constant commands also match the previous solver;
a saved ten-second C example was reproduced bit for bit across all channels.

## Completed study

[October 23, 2026 results](results/trajectory_sweep_20261023/README.en.md): joint refinement reduced independent-test mean angular RMSE from 2.80° to 1.09° (18/18 survive), but velocity oscillations remain; strict success is 2/18. The study retains 160 generations, 4800 individuals and 81 complete trajectories. The audit reproduced tests exactly. [Best genome](results/best_genome.json), [comparison](results/trajectory_sweep_20261023/figures/tracking_comparison.png), [phase trajectories](results/trajectory_sweep_20261023/figures/phase_trajectories.png).

To repeat the audit:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
.venv/bin/python scripts/AfferentBalance2026/experiments/D_trajectory/audit_trajectory.py \
  scripts/AfferentBalance2026/experiments/D_trajectory/results/trajectory_sweep_20261023 --workers 18
```

The auditor was moved from `tools/` into D. Earlier result snapshots remain compatible: the auditor itself and newly created Jupyter `.ipynb_checkpoints` backups may be absent from an older source manifest. All recorded hashes and active computational modules are still checked in full.
