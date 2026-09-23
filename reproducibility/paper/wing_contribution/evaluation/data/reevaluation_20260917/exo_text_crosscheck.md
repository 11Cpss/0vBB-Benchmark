# EXO-200 and MJD final manuscript cross-check

Reviewed at 2026-09-17T18:52:29.158457+00:00.

## Scope and result

PASS for numerical values, label/score orientation, strict-range definitions, and the EXO/MJD ranking/Pareto claims checked below. This was a read-only review of paper and source data; only this audit report was written. The requested final scope is EXO-200 and MJD: eight classic models, nine EXO Transformers, six MJD MLP/Fourier Transformers. Three MJD RoPE rows retain their previously reported three-decimal values and are excluded from reevaluation-based comparisons. NEXT, SuperNEMO, and the independent SuperNEMO energy illustration are out of scope and intentionally retain their previous profiles.

Cross-checks compared 23 final full-precision records against current paper evidence with exact scalar equality, and 52 displayed EXO/MJD main-table cells (including six retained MJD RoPE cells) against the appropriate three-decimal formatting. Numerical mismatches: 0. All 23 reevaluated records carry v3 fingerprint `6d696d87f1307b3f6d27bb9ef7ec443e1d8d98b9d95e90713f553c70edb42953`.

The active input graph from `iclr2027_conference.tex` contains 36 TeX files. In addition to the main table and quantitative appendices, the Transformer model/tokenization sections, top-level model wrapper, experimental setup, benchmarking-metrics section, results, conclusion, MJD motivation appendix and dataset descriptions were read/searched. Commented template material and `format_backup.tex` were distinguished from active content.

## EXO positive class, score, energy, and definitions

- `experimental_setup.tex`, `metric_protocols.tex` and `appendix_metrics.tex` consistently identify stored EXO label 0 as single-cluster signal and stored label 1 as multi-cluster background. The latter explicitly writes `y = 1 - y_stored` and negation of the background logit for all nine Transformers, preserving the scalar-logit score representation without a sigmoid. This agrees with the original source adapter, historical manifest and canonical raw-event reconstruction.
- EXO conditions on physical reconstructed rotated energy in keV. Its saved metadata match classic CNN event identities and source energies exactly; no changed dataset or checkpoint is implied by the paper.
- Strict MJD/EXO protocol text correctly specifies 600 bins, left-closed/right-open interior intervals, exact 3000 keV in the last bin, outside-range exclusion before support estimation, original finite-population coverage denominators, and unchanged inclusive AUC population.
- Historical EXO matching is correctly separated as six pooled class-balanced energy-quantile bins; historical I is separately eight within-category energy-quantile bins. The score-quantile histogram rule remains up to 20 bins. Old protocol terms appear as historical explanations, not current EXO/MJD rules.
- Per-class I and matching diagnostics agree with final strict-range results: EXO matching 426 bins, support [538.1772625732422, 2662.5670739746092] keV, coverage signal/background 0.9456571099875875/0.9414028336540143, I bins 438/484. Out-of-range losses 3/1423 are separated from I sparse losses 152/257; no matching sparse loss.

## MJD consistency

- Text correctly restores original float64 calibrated energies, removes old classic clipping, excludes 67 above-range events (one clean, 66 nonclean), and preserves all 390,000 events in inclusive AUC. The six Transformer historical caches are described as 836 fixed 5-keV bins to 4180 keV; the strict600 reevaluation is not mislabeled as unchanged merely because the width was already 5 keV.
- MJD diagnostic tables and the motivation appendix use support [18.151, 2613.790] keV, 493 valid matching bins, 146,383 clean and 232,824 nonclean retained events, and CNN inclusive/matched/I 0.971283/0.969851/0.759637. These agree at displayed precision with final results.
- The additional four-PSD supervision for classic GINE is disclosed; the table excludes it from best-score highlighting. Other classic models and the six reevaluated Transformers retain their clean-oriented single scores.
- All three MJD RoPE rows have `comparison_eligible=false`, recorded-display-only status and source provenance; results prose explicitly excludes them from new-protocol comparisons.

## Rankings and claims

- EXO Transformer I descending: summary_fourier, region_mlp, region_fourier, region_rope, summary_mlp, summary_rope, entity_mlp, entity_fourier, entity_rope.
- EXO Transformer matched AUC descending: region_fourier, region_rope, region_mlp, summary_fourier, summary_rope, entity_fourier, summary_mlp, entity_rope, entity_mlp.
- Historical EXO Transformer Pareto set: region_mlp, region_fourier, region_rope, summary_fourier. Final set: region_mlp, region_fourier, summary_fourier. The text correctly removes region_rope because region_fourier now exceeds it on both axes; it does not assert significance.
- EXO region_fourier final matched AUC 0.9722610766369408 versus region_rope 0.9709096841000631; final I 0.7900516557727041 versus 0.7892311644895167. The prose rounds these correctly to six decimals. Summary_fourier retains the highest Transformer I (0.8019020777424724).
- EXO Transformer I decreases by 0.061012807–0.089056841; the prose range 0.061–0.089 is correct. Both physical-category scores decrease relative to history for all nine models.
- Both MJD MLP/Fourier I ordering and matched-AUC ordering are unchanged: True. Region Fourier remains the matched-AUC leader; entity MLP retains the highest I among the reevaluated MJD Transformers.
- Classic EXO GINE remains the matched-AUC and I leader (display 0.949/0.801). MJD BiGRU remains the classic matched-AUC leader; GINE has the highest I with its separately disclosed supervision. No new statistical-significance claim was introduced.

## Presentation finding

Low-severity finding: the standalone main-table caption currently does not identify the three MJD RoPE rows as unrecomputed historical results or explicitly reference the dataset-specific protocol profiles. The immediately preceding results paragraph and appendix do explain this accurately, so this is not a numerical contradiction. A short caption note would prevent a reader viewing the table in isolation from assuming every row was evaluated under v3. This was sent to the root editor; no paper edits were made by this reviewer.

Out-of-scope NEXT/SuperNEMO mentions of overflow, including the separate SuperNEMO illustration, are intentionally retained. They must not be mistaken for obsolete EXO/MJD protocol statements or edited as part of this restricted update.

## Inspected evidence and file state

- `wing_contribution/tables/benchmark_main.tex` SHA256 `a976c718857eb282fcda479f6dd398d48e2bf9506813f1894a24c919797fde46`
- `wing_contribution/tables/metric_protocols.tex` SHA256 `1e38598b6232d9669195c67b300b0de41fd540a15b90d6eb0bc5a7f59ff75b97`
- `wing_contribution/tables/transformer_diagnostics.tex` SHA256 `79d0475322ac6780d6e53e27e8e4d61c32a6665d8b4071e0d015a6dc0f48dedf`
- `wing_contribution/tables/matching_diagnostics.tex` SHA256 `af5d3c30619d316c233b3957dc863b6798587c3e0eb3ebe1fc88cf02d757ac18`
- `wing_contribution/tables/independence_diagnostics.tex` SHA256 `2c553006061ab1a0791cc97f8f19171c05ac9320be2da8d9e27acc41b8c46de5`
- `wing_contribution/tables/evaluation_losses.tex` SHA256 `4e40a6b4f3f702ed11b52479c84270fd5c98e7e516ecef7a2c2e003049eb91d6`
- `wing_contribution/sections/results_analysis.tex` SHA256 `410db940ad6814cc25b03301013407b516eb28417d9df4eb6773b658f87ad55e`
- `wing_contribution/sections/benchmarking_metrics.tex` SHA256 `cb0de43dd3c157ee0e98611a7e4a5973efe63d460af76cefdd9b2d0ee1c3905b`
- `wing_contribution/sections/appendix_metrics.tex` SHA256 `00359a3b5da2b42c6fd9875c7ac2cac534aabe656a39a4835f8cd7a6fe293d19`
- `wing_contribution/sections/appendix_mjd_motivation.tex` SHA256 `b2c9e8ecd7485b69b51237fb32cbd723bb836efbc38f43efe3607611a95464bb`
- `wing_contribution/sections/conclusion.tex` SHA256 `82cd775fd842db75133639b2aa2ca02f00f159bc209980b82532d80373125fc1`
- `transformer_contribution/sections/transformer_models.tex` SHA256 `26489349f7e7820074abd84066645f3d659c8625cac22559019842744b2e18e9`
- `iclr2027_conference.tex` SHA256 `e1396d927d44e369b913e06c619237f5c74ce2d0a60903804671c88f63208ef9`

Source records: `strict_600bin_20260917/results/results_full_precision.json`, `transformer_exo_mjd_20260917/exo/strict600/summary.json`, `transformer_exo_mjd_20260917/mjd/strict600/summary.json`, and the linked original-source audits/manifests. The summary values are used only for manuscript cross-checking; the reevaluations themselves used aligned per-event scores, labels, physical energies, groups and weights.
