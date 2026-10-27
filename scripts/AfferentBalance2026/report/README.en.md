<!-- mirror-symmetry-results -->

[Detailed supplement: symmetric C/D](MIRROR_SYMMETRY.en.md). This series is also included in the main article: §2.10 describes constraints and training, §4.2 provides comparisons and two figures, and §7.2 lists exact commands. The asymmetric series remains in §4.1 for comparison.

# Current scientific report

[PDF](REPORT.pdf) · [English article](REPORT.en.md) · [Russian article](REPORT.md) · [Genome appendix](APPENDIX_PARAMETERS.en.md)

The report covers corrected tensile mechanics, full nonlinear Afferents and the 26 October 2026 retraining of A–D. It includes all model equations, genetic algorithm steps, conditional analytical stability, independent test results, phase grids, neuronal/muscle/afferent signals and exact commands with option meanings. Muscle A is fixed at 0.37 in every main experiment.

The two symmetry figures show +45° holding across an initial-condition grid and the first held-out sinusoidal tracking example. Each includes phase trajectories, angles, neuronal potentials, muscle forces, afferents and currents. Target-dependent C success and the remaining strict-criterion failure in D are stated explicitly. Figures use stored numerical calculations rather than schematic trajectories.

`article_body.en.txt` and `article_body.ru.txt` contain the scientific prose. `build_retraining_article.py` inserts verified numerical tables and figures from completed runs. `build_pdf.py` typesets the Russian article and full genome appendix. `refresh_report.py` routes to the current builder. Numerical outputs remain under `results/muscle_tau50_20261026/`; its `manifest.json` records selected runs and hashes. Its `mirror_symmetry` entry records separate symmetric winners, the comparison JSON, source metrics and both figure files; the comparison itself is in `results/mirror_symmetry_20261026/comparison.json`. `previous_report/` preserves the old report. Historical figures in other directories are not evidence for the corrected model.

Run from the repository root:

```bash
.venv/bin/python scripts/AfferentBalance2026/report/build_retraining_article.py
.venv/bin/python scripts/AfferentBalance2026/report/build_pdf.py \
  --pandoc /home/nickolai/miniconda3/envs/CPG/bin/pandoc
```

Simulation and phase-figure commands are fully described in section 7 of the article. Rebuilding the article does not train models. Never use `--allow-incomplete` for a final report; it is a development preview.

The current series constrains both muscles to tau_c ∈ [50,100] ms and tau_1 ∈ [50,200] ms. Previous winners remain under each experiment’s `results/tensile_retrain_20261026/`; the preceding report is archived in `results/muscle_tau50_20261026/previous_report/`. The comparison table distinguishes old and new domains.
