> October 21, 2026: numerical data for this historical series were moved to the desktop trash at the user’s request. Restore them using `archive/cleanup_results_20261021.json` to rebuild historical figures. New data live in `experiments/*/results/`.

[Русский](README.md)

# Report on the full Afferents model without ×10

[REPORT.md](REPORT.en.md) and [REPORT.pdf](REPORT.pdf) describe the recomputation
of 16 October 2026: physical muscle torque without amplification, a 0.1 ms step,
full Afferents and neural control without PD.

The `results/recompute_unscaled_20261016` series contains 39 training histories,
6900 generations, final populations and complete wide-angle trajectories.
Failed trials remain included in the results.

## Main files

- [Interactive article figures](article_figures.ipynb): editable A–D panels,
  scenario selection and PNG/PDF/SVG export from saved trajectories.
  [Instructions and protocol](../ARTICLE_FIGURES.en.md).
- [Summary](results/recompute_unscaled_20261016/summary.json).
- [Generation audit](results/recompute_unscaled_20261016/history_verification.json).
- [Notebook verification](results/recompute_unscaled_20261016/notebook_verification.json).
- [Report provenance](results/recompute_unscaled_20261016/report_verification.json).
- [Winner parameters](APPENDIX_PARAMETERS.en.md).
- [Analytical controllability and holding estimates](ANALYTICAL_CONTROL.en.md):
  recovered from the historical report and recomputed without ×10; section 27 of the current report.
- [Numerical checks of the analytical formulas](analytical/estimates.json)
  and [checking script](../analytical_control.py).
- [Generation notebook](../notebooks/training_phase_trajectories.ipynb).
- [Calculation-file map](../archive/pre_ABC_20261020/experiments/RECOMPUTE_UNSCALED.en.md).

Verification records refer to document versions at build time. Translation and
link updates do not alter numerical NPZ/CSV files or frozen runtime sources.

## Build

```bash
.venv/bin/python scripts/AfferentBalance2026/analyze_unscaled.py
.venv/bin/python scripts/AfferentBalance2026/report/build_pdf.py --pandoc /path/to/pandoc
```

Analysis verifies saved data, creates figures and assembles the Russian report
without retraining. `build_pdf.py` only typesets it with Pandoc and LuaLaTeX.
The English Markdown edition is adjacent to the Russian source; this builder
does not produce a separate English PDF.

Historical ×10 results, old PDFs and rendered presentation files were moved
to the trash. Builder sources, the TeX article and system diagrams remain;
historical builds require their data to be restored.
See the [cleanup audit](results/cleanup_factor10_20261017.json).
Historical analytical excerpts are recovered separately:
[original text](analytical/legacy_excerpt.txt),
[provenance](analytical/legacy_source.json).

[10×10 phase grids and oscillograms, 19 October](results/article_phase_grid_20261019/README.en.md).
