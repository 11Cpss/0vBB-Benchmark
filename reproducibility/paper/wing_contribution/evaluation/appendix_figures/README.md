# Appendix figure replay

`../../scripts/plot_appendix_figures.py` redraws the three SuperNEMO appendix
figures used by the manuscript. It uses the shared `../paper_style.py`
typography and exports at the actual LaTeX inclusion widths: 4.95 inches for
the energy spectrum, 4.675 inches for the threshold scan, and 5.5 inches for
tracker extent. This preserves the intended font sizes when LaTeX includes
the PDFs. Plot titles and long notes already supplied by the figure captions
are omitted from the image to keep the figures compact.

From the repository root, with NumPy, pandas, and Matplotlib installed:

```bash
python wing_contribution/scripts/plot_appendix_figures.py \
  --extent-cdf /path/to/supernemo_cohort_ecdf_LOCAL_ONLY.csv.gz
```

The `--extent-cdf` file contains the complete original empirical CDFs, with
every distinct extent retained. It is an event-derived local-only input and
is deliberately excluded from the repository. The script checks its hash
against `data/sources.json`. Its original local location is recorded there.
Without this option, the script rebuilds the spectrum and threshold figures
and leaves the tracker-extent PDF as supplied.

The checked-in inputs preserve the original display bins, event counts,
energy threshold scan, and conditional extent probabilities. The median
curve file contains only the original per-energy-bin aggregate medians.
`data/sources.json` records source locations, SHA-256 hashes, and derivations.
No models, matching weights, thresholds, or scientific metrics are changed.

Optional `--output-dir` and `--preview-dir` arguments redirect PDF output
and request PNG previews, respectively.
