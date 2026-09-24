# Table and figure rendering

This folder contains rendering code and final plot inputs. Event-level metric
computation is provided only by the top-level `benchmark/` package.

From the `reproducibility/` root:

```bash
python reproduction/reproduce.py render --output outputs/render
```

- `tables.py` renders the main table and five numerical appendices.
- `plot_next.py` renders the NEXT capacity plot.
- `paper_style.py` loads the bundled fonts and plotting style.
- `result_profiles.py` validates profile identifiers on saved results; it does
  not calculate any metric.
- `data/diagnostics.json` contains final full-precision class/support diagnostics.
- `appendix_figures/data/` and `mjd_style/data/` contain final plotting inputs.

The main table reads `../figures/data/main_table_results.json`. Final aggregate
metrics and per-record protocol fingerprints are in
`../figures/data/unified_results.json`. These files are display inputs, not
substitutes for event-level predictions when recalculating metrics. The paper
retains different named profiles for different result groups; profile identifiers
must be preserved when reproducing its values.
