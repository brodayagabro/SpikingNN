> October 21, 2026: numerical data for this historical series were moved to the desktop trash at the user’s request. Restore them using `archive/cleanup_results_20261021.json` to rebuild historical figures. New data live in `experiments/*/results/`.

[Русский](ARTICLE_FIGURES.md)

# Article multipanels: balance and target-angle holding

[report/article_figures.ipynb](report/article_figures.ipynb) builds editable figures
from saved trajectories. Computation and verification now live in
[report/article_figure_data.py](report/article_figure_data.py), which replays saved
validation winners through their immutable numerical runtime. It performs no training or selection
on the new grid. All runs use full `Afferents`, **0.1 ms** steps, **10 s** episodes,
neural `mode=0` and **unscaled muscle torque** (factor 1).

## Eight figures

Three balance figures use the ordinary `equilibrium` sweep winner, the
`small_nm_weights` winner and A1 trained from large initial deviations. A1 uses
the seed previously selected by large-angle validation (43 in the current study).
Five additional figures use the same validation-selected cubic current law
(`degree_3_seed_43`) at targets −45°, −22.5°, 0°, +22.5°, +45°.
Selections precede the new evaluation. Every outcome, including falls, is retained.

## Panels

- **A — phase trajectories:** Cartesian product of ten initial angles
  `linspace(-50,50,10)` degrees and ten velocities `linspace(-1,1,10)` rad/s,
  giving 100 trajectories. Angles are absolute deviations from upright, not
  target-relative errors. Color encodes initial angle; dots are starts, the
  star is the target and crosses mark falls. Green region: ±3° target error and
  ±0.35 rad/s. These are projections of the full dynamics, not an autonomous
  two-dimensional vector field.
- **B — angle oscillograms:** the same ten angles at **ω₀=0**, through 10 s.
  These are additional integrations: the even-sized velocity grid in A excludes
  zero. The thicker line is the example used in C/D.
- **C — one example only:** flexor/extensor activation `x=Cn^m/(Cn^m+k^m)` and
  forces. Every figure uses the same prespecified start **θ₀=−27.7777778°, ω₀=0**
  (index 2 of bank B). The first 2 s are shown, or until an earlier fall.
  Blue denotes flexor, orange extensor.
- **D — the same single example:** Ia, II and Ib of both muscles on separate
  axes so different scales remain visible. Receptor values are in model units,
  not measured spike rates. The time window matches C.

There are **110 cases per figure, 880 cases total**. Both endpoints of each
phase-grid axis are included; neither even-sized axis contains zero. Bank B has
exactly ten zero-speed cases. No velocity impulses are applied. Every episode
resets force, calcium, afferents and neural states as in training. Command
currents are constant within an angle-holding episode.

Crossing |θ|>90° terminates a case. Curves are never extended beyond a fall.
Standard success requires entering |θ−θ*|≤3° and |ω|≤0.35 rad/s by 3 s and
remaining there until the end. Falls receive the usual 90° late-RMSE penalty,
not a measured steady-state error. This evaluation extends the earlier ±45°,
±0.5 rad/s checks; reported fractions apply specifically to the new grid.

## Commands

Open the notebook from the repository root:

```bash
.venv/bin/python -m pip install -r scripts/AfferentBalance2026/requirements-notebook.txt
.venv/bin/jupyter lab scripts/AfferentBalance2026/report/article_figures.ipynb
```

Choose **SpikingNN (.venv)** and **Run All**. The default figure is A1 from the
completed study. Choose another figure using `SCENARIO` or the widget controls.
Fonts, colors, sizes, time windows and the code for panels A–D are editable
in separate cells. `ipympl` enables mouse zoom and pan; without it, static
output, widgets and cell-based editing remain available.

Edit the current `fig` and `axes` directly, or retrieve previously built objects
from `FIGURES` and `AXES`. Set `BUILD_ALL = True` to prepare all eight figures.
Set `EXPORT = True` to save selected current objects as PNG/PDF/SVG and a bundled
PDF under a new `report/figures_edited/article_<timestamp>/` directory, alongside
a notebook copy and `export.json`. Manual edits are retained; source data and
original figures are preserved. Normal notebook execution performs no new integrations.

For a new computation, use a new output directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  scripts/AfferentBalance2026/report/article_figure_data.py all \
  --output scripts/AfferentBalance2026/report/results/article_phase_grid_new \
  --workers 6
```

The source study defaults to `report/results/current_unscaled.json`; `--root`
selects another completed recomputation with the same layout. Computation does
not overwrite existing output directories. `compute` calculates trajectories,
`verify` runs the library comparison, and `all` performs both actions.
Set the notebook's `STUDY` to the resulting directory to draw its figures.

Dependencies are listed in `requirements-notebook.txt`. The original style uses
Times New Roman, also searched under `~/.wine/drive_c/windows/Fonts`; the notebook
falls back to DejaVu Serif for viewing if unavailable. Numerical computation is font-independent.
The saved runtime's sources and hashes remain unchanged; workers import that
runtime, including its historical internal package namespace.

## Outputs and verification

Completed study: `report/results/article_phase_grid_20261019/`.

- `figures/<scenario>.pdf/.svg/.png`: eight figures with English Times New Roman
  labels. PDF/SVG retain vector curves; PNG provides convenient previews.
- `figures/all_panels.pdf`: one figure per page, eight pages total.
- `protocol.json`: banks, selection rules, genomes, current coefficients and
  original population/generation hashes. `article_figures_source.py` preserves
  the exact computation runner.
- `<scenario>/phase_grid/case_XXX.npz`: 100 complete phase trajectories.
- `<scenario>/zero_velocity/case_XXX.npz`: ten complete oscillogram trajectories.
- `<scenario>/example_signals.npz`: complete activation, force, afferents,
  lengths, velocities, nonlinear terms and actually applied afferent currents
  for the C/D example in the original study. The notebook reconstructs these
  signals in memory from the saved trajectories.
- `traces.json`, `metrics.csv`, `summary.json`: hashes, metrics and outcomes.
- `verification.json`: time grid, initial state and independent metric
  reconstruction from recorded states. `plot_manifest.json`: per-figure sources,
  output hashes and checks of nonlinear receptor-term sums.

NPZ retains every **0.1 ms** step including t=0: a full episode has 100001 rows
and 17 channels. A/B display every 1 ms while retaining the final sample; C/D
show every sample within their window. Scientific records are never decimated.
No smoothing or trajectory interpolation is used. New checks are in
[test_article_figures.py](tests/test_article_figures.py). Numerical outputs and
figures are excluded from the Git index.

The `verify` action additionally replays the C/D examples for A1 and target +22.5°
through the original library classes without Numba. Full reference trajectories
are saved in `library_reference/`; comparisons in `library_reference_verification.json`.
The `all` action includes this check. Panel B's inset shows the first 0.7 s of
the same ten oscillograms without any additional selection.
