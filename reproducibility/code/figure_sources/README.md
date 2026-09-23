# Active manuscript figures and their sources

This directory completes the figure inputs missing from the original Overleaf repository. It also preserves the upstream preprocessing scripts **byte for byte**. The frozen manuscript is in `../../paper/`; nothing in the original working tree was edited. All commands below are run from the **bundle root** and write only into `validation/figures/`.

The active include tree contains 35 TeX files, 14 external figure assets, and 11 table environments. Six figure assets have recovered code and plotting inputs and were regenerated successfully. Eight dataset-description panels are preserved PDF assets whose original generators and selected event identities were not located. Compiling those eight supplied PDFs reproduces their appearance; it does not establish regeneration from raw events.

The machine-readable inventory is `provenance/figure_inventory.json`. It follows `iclr2027_conference.tex` recursively after removing commented lines, so it excludes older images not included by the current manuscript. The inventory records all active tables, including the inline Transformer tokenization table.

## Replay the six generated active figures

Use the bundle's scientific Python environment. The verified local environment was Python 3.11.15, NumPy 2.4.6, SciPy 1.17.1, pandas 3.0.5, Matplotlib 3.11.1, and Pillow 12.3.0. NumPy 2 or newer is required by the new motivation renderer's `numpy.trapezoid` call. Bundled Nimbus font files are loaded by relative path; no external font download is required.

```bash
# Current Figure 2, including the uncommitted 2026-09-22/23 revision.
python paper/wing_contribution/scripts/plot_mjd_motivation_v2.py \
  --output-stem validation/figures/mjd_motivation_main_v2_generated

# The active MJD low/high waveform appendix figure.
python paper/wing_contribution/evaluation/mjd_style/render.py \
  --output-dir validation/figures/mjd

# The active NEXT model-capacity figure.
python paper/wing_contribution/scripts/plot_next_capacity.py \
  --output-dir validation/figures/next

# All three active SuperNEMO appendix figures, including complete extent CDFs.
python paper/wing_contribution/scripts/plot_appendix_figures.py \
  --extent-cdf code/figure_sources/data/supernemo_cohort_ecdf.csv.gz \
  --output-dir validation/figures/appendix

# Compare all six regenerated active PDF files to the frozen paper assets.
python code/figure_sources/validate_renders.py
```

`mjd_style/render.py` also emits the older `mjd_motivation_main.pdf` in its isolated output directory. That secondary output is not an active manuscript figure and is not substituted for the new `mjd_motivation_main_v2_generated.pdf`.

All six active PDFs are **byte-identical** to the supplied manuscript assets in the verified environment. See `validation/figures/render_validation.json`. Exact PDF bytes can depend on library/font versions; numerical inputs and hashes are separately recorded.

| Active asset under `paper/` | Final renderer | Plotting inputs |
| --- | --- | --- |
| `wing_contribution/figures/mjd_motivation_main_v2_generated.pdf` | `wing_contribution/scripts/plot_mjd_motivation_v2.py` | MJD waveform CSV; SuperNEMO 50 keV display histogram, complete ROC CSV, and matching evidence JSON |
| `wing_contribution/figures/mjd_low_high_waveforms.pdf` | `wing_contribution/evaluation/mjd_style/render.py` | Exact selected waveform arrays and frozen waveform/selection summaries in `evaluation/mjd_style/data/` |
| `wing_contribution/figures/next_capacity_scores.pdf` | `wing_contribution/scripts/plot_next_capacity.py` → `evaluation/plot_next.py` | `figures/data/next_capacity_plot.csv`, full-precision result records and applicable v2 profile |
| `wing_contribution/figures/energy_bias_spectrum.pdf` | `wing_contribution/scripts/plot_appendix_figures.py` | `evaluation/appendix_figures/data/energy_shortcut_histograms.csv` |
| `wing_contribution/figures/energy_threshold_tradeoff.pdf` | Same | `evaluation/appendix_figures/data/energy_threshold_scan.csv` |
| `wing_contribution/figures/supernemo_extent_energy_population.pdf` | Same | Frozen conditional histogram, energy-bin medians, extent summary, and the complete CDF supplied here |

The latest motivation figure deliberately combines different datasets: panel (a) is MJD, while panels (b,c) are the separate SuperNEMO **0nu/Bi214 energy-only test illustration**. They are not MJD training distributions or a trained MJD classifier. The renderer checks all three AUCs by integrating the full archived ROC curves and reconstructs the 283 retained 5 keV bins from the unsampled threshold sequence. It does not rerun a model or change a metric profile.

## Event-derived extent preprocessing

`data/supernemo_geometry_event_descriptors.csv.gz` contains the 458,752 event-derived energy and raw tracker-extent descriptors used by the active appendix: 262,144 Bi214 and 196,608 0nu events. `data/supernemo_cohort_ecdf.csv.gz` preserves all 229,376 ECDF rows; it is the exact original compressed input expected by the final renderer. These files were omitted from the original public repository but are included in this local reproducibility bundle under the user's authorization.

The following new portable verifier reconstructs the displayed bins, medians, and complete CDF from those descriptors, without changing any archived source:

```bash
python code/figure_sources/rebuild_extent_inputs.py
```

It writes to `validation/figures/rebuilt_extent_inputs/`. Normalized CSV content matches all three original plotting inputs exactly: 100 keV energy bins, 20 mm extent bins, the 20-event display threshold, and the lowest/highest energy quartiles sorted by energy and then event ordinal. The full extent tails remain in normalization and the CDF. This operation reproduces the plots' source arrays from event-derived descriptors; it is not a new raw-HDF5 extraction or a model evaluation.

## Original preprocessing sources, preserved unchanged

`archived_preprocessing/` contains original code for:

- MJD official-shard scalar metadata extraction and deterministic low/high waveform event selection/preprocessing.
- SuperNEMO event energy/geometry extraction from hit-level HDF5 and the saved split manifest.
- Conditional extent histograms, energy quartiles and complete CDF preparation.
- SuperNEMO energy histograms and threshold scans.

The archived scripts retain their original absolute paths, output-directory safeguards, historical plotting styles, and, where present, historical metric calls. They are preserved as source provenance and are **not** the portable figure replay entry points. In particular, `analyze_energy_shortcut.py` also calls the old quantile-bin evaluator; those calls must not be presented as the paper's current matching implementation. No archived code was corrected or silently modernized.

Re-extracting from raw detector files requires the original MJD `MJD_Test_*.hdf5` shards or SuperNEMO HDF5 files, SuperNEMO split manifest and event-offset index arrays, respectively. Original paths and source audit details remain in the preserved scripts and provenance JSON. Those large detector datasets are not copied by this figure component. The final figure replay commands above require none of them.

## Eight dataset-description panels: precise limitation

The active files are `paper/dataset_description/Images/{mjd,next,supernemo,exo}_{signal,background}.pdf`. All eight were imported by Overleaf commit `9360b04fa52568562adcf1702231527d06a1146a` on 2026-09-21. PDF metadata identifies Matplotlib 3.10.6; MJD/NEXT/SuperNEMO PDFs were created September 18 and the EXO PDFs September 20 in the local timezone.

Local searches of `/home/wenyu` and the accessible relevant `/home/klz` source trees used the exact output basenames and distinctive embedded labels, including `No shared vertex`, `U-wire Channel`, and `MJD: Signal`. No corresponding plotting script, notebook cell, exact selected event manifest, or plot-input archive was located. The Overleaf commit adds the PDF binaries without their generators. Event identity therefore cannot be inferred from the caption or visual appearance. No replacement events or invented reconstruction have been introduced.

These eight PDFs are included intact and their SHA256 hashes appear in `provenance/figure_inventory.json`. A future raw-data regeneration requires obtaining the original author's figure generator and event-selection records. Until then their status is **archived asset only**.

## Active tables and Transformer material

There are no external figure assets in the active Transformer contribution. Its appendix contains a native LaTeX tokenization table in `transformer_contribution/sections/appendix_tokenization.tex`; the separate `transformer_contribution/tables/tokenization_summary.tex` is not included by the current paper.

The classification table is generated by `paper/wing_contribution/scripts/build_classification_table.py`. Five numerical diagnostic/extended tables use `paper/wing_contribution/evaluation/tables.py`. The conditioning-variable, model-adaptation, training-protocol and inline energy-illustration/Transformer tables are maintained directly in their active TeX sources; these exact sources are the reproducible table artifacts. Do not invent a missing generator or rewrite their descriptions while replaying figures. The root bundle's evaluation instructions cover regeneration of the numerical tables and the per-record protocol registry.

To refresh the active include inventory using only the bundle copy:

```bash
python code/figure_sources/inventory_active_paper.py \
  --paper-dir paper --output provenance/figure_inventory.json
```

`provenance/figure_source_manifest.json` gives original locations, bundled destinations, sizes, and SHA256 values for every externally copied input and historical script. Original files and the manuscript snapshot remain untouched by all validation commands above.
