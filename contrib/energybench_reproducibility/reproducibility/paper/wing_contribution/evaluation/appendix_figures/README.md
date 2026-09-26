# SuperNEMO figure inputs

These final CSV/JSON inputs render the energy spectrum, energy-threshold scan,
and tracker-extent appendix figures. The complete CDF and event-derived geometry
descriptors are in `code/figure_sources/data/` at the reproduction root.

```bash
python paper/wing_contribution/scripts/plot_appendix_figures.py \
  --extent-cdf code/figure_sources/data/supernemo_cohort_ecdf.csv.gz \
  --output-dir outputs/appendix
```

The spectrum and threshold panels use the separate 0nu/Bi214 energy-only
illustration. They are not learned classifier results. The descriptor reconstruction
command is documented in `code/figure_sources/README.md`.
