# Paper figures

Run these commands from the `reproducibility/` directory using the environment in
`environments/requirements-core.txt`.

```bash
python reproduction/reproduce.py render --output outputs/render
python code/figure_sources/validate_renders.py --output-dir outputs/render/figures
python code/figure_sources/rebuild_extent_inputs.py --output-dir outputs/extent_inputs
```

The first command generates the six active figures and six numerical tables. The
second compares the generated PDFs with the paper assets. PDF byte identity is
expected with the pinned environment and bundled fonts; other Matplotlib versions
can change PDF serialization. The third reconstructs the extent histograms,
medians, and complete CDF from the supplied event-derived descriptors.

| Figure | Renderer under `paper/wing_contribution/` | Inputs |
| --- | --- | --- |
| Motivation, three panels | `scripts/plot_mjd_motivation_v2.py` | MJD waveform CSV and SuperNEMO spectra/ROC CSVs |
| MJD low/high waveforms | `evaluation/mjd_style/render.py` | Exact selected waveform arrays and event metadata |
| NEXT model capacity | `scripts/plot_next_capacity.py` | Final full-precision results and capacity CSV |
| Energy spectrum | `scripts/plot_appendix_figures.py` | Energy histogram CSV |
| Energy threshold | Same | Threshold-scan CSV |
| SuperNEMO tracker extent | Same | Conditional histogram, medians, summary, and complete CDF |

The motivation figure's first panel uses MJD waveforms. Its second and third panels
use the separate SuperNEMO **0nu/Bi214 energy-only illustration**, not the trained
2nu/Bi214 classification task. Histogram display bins and energy-matching bins
have distinct meanings; the saved figure inputs preserve both.

`data/supernemo_geometry_event_descriptors.csv.gz` contains 458,752 test-event
energy/geometry descriptors. `data/supernemo_cohort_ecdf.csv.gz` contains all 229,376
CDF rows used in the figure. Rebuilding from these descriptors does not replace
raw detector preprocessing or model inference.

Eight additional detector examples are supplied as PDFs in
`paper/dataset_description/Images/`. Their original generators and exact event
selections were unavailable. The PDFs compile into the paper; this release does
not claim regeneration of those examples from raw events.
