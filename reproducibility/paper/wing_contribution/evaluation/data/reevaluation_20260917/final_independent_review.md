# Independent review — final MJD/EXO-200 scope

The final scope is 23 reevaluations: four classic models plus six MLP/Fourier Transformers on MJD, and four classic models plus nine Transformers on EXO-200. The final protocol is `EnergyBench-unified-5keV-range-v3.0.0`, fingerprint `6d696d87f1307b3f6d27bb9ef7ec443e1d8d98b9d95e90713f553c70edb42953`. The independently tested evaluator SHA256 is `a243160b0a529a26a38dc6be14baa6f17a09618fea57c5a80de21822369e06b0`.

NEXT, SuperNEMO, and the three MJD RoPE entries remain outside the final reevaluation. The 38 other strict-range evaluations performed before the last scope reduction are audit-only and are not substituted into the paper. In particular, the unchanged SuperNEMO energy-only illustration retains its v2 result and figures.

## Numerical and event-source verification: PASS

- `independent_numeric_active23.json` independently reconstructs every selected metric from every event, including original inclusive AUC, weighted score quantiles and histograms, entropy-form Jensen–Shannon divergence, group aggregation, support quantiles, class/bin counts and mass, overlap target, weighted ROC, coverage, effective sample size and loss decomposition. No shared metric helper is imported. Alternative floating-point summation of the full weighted ROC agrees within 5e-12; manuscript numbers exactly equal the original hashed shared-evaluator details.
- `independent_synthetic_v3.json` passes 48 independent full-equation synthetic cases plus explicit constants, missingness, empty/sparse bins, weights, ties, endpoints, all 601 energy boundaries, MeV/keV equivalence, direct enumeration of cross-class score pairs, range exclusions and coverage gates. Exactly constant common energy waives only the bin-count gate.
- `independent_sources.json` passes all 15 Transformer source checks, directly reading scalar event metadata for 530,383 distinct test events: 390,000 MJD and 140,383 EXO-200. Original exported and native prediction arrays, physical identities, labels, categories, original energies, original source hashes and common classic metadata agree. No dataset loader or shared metric implementation is imported by this source check.
- EXO-200 uses stored label 0 (`n_CCL=1`) as positive and negates the original background-oriented logit. MJD Transformers retain their clean-oriented scalar logits; clean requires all four PSD flags. The classic MJD GINE's different four-output construction and supervision are retained and disclosed.
- MJD physical energies are recovered as float64 through verified official shard and row identities; no clipped or substituted energies are used. EXO-200 uses original rotated energy in keV. Inputs and checkpoints are unchanged; no fitting is performed.
- Both metrics use exactly 600 fixed bins. Zero and interior edges are assigned correctly; 3000 keV belongs to the last bin. Values below zero or above 3000 keV are excluded before support estimation and histograms. Inclusive AUC preserves its original score/weight population.
- Every selected formal matched AUC satisfies the existing reporting gates. Coverage denominators remain original finite-score/finite-energy class mass before range filtering. The upper exclusions are MJD 1 positive/66 negative and EXO-200 3 positive/1423 negative events.

The full intermediate 61-model mathematical audit also passes, but `independent_numeric_active23.json` is the authoritative review for the final selected scope.

## Manuscript and result provenance: PASS

`review_final_paper.py` checks the manuscript against Git baseline `4851898c40bdd38e6fad64bf0d6068860fb91f93` and the hashed numerical review. `final_paper_review.json` records 32 passing checks and final manuscript input hashes:

- All 46 selected main-table metric cells and 23 full-precision result records match the audited results. Every selected group-level, matching, coverage, effective-sample-size and loss diagnostic matches the original details.
- The current 10-column layout, 13 model rows, parameter-count ranges, two group titles and `bestscore` mechanism remain. Highlighting follows full-precision eligible winners. MJD GINE remains excluded from best-result highlighting because of its extra supervision.
- All 52 NEXT/SuperNEMO main-table metric cells, the six MJD RoPE cells, 54 unselected result records, 29 unselected table-evidence records and 15 unselected workbook rows remain unchanged. The 33 protected figures, plotting data files and independent NEXT/SuperNEMO sections are byte-identical to the baseline; their rows in the three shared diagnostic tables are also unchanged.
- The mixed-profile registry preserves the original v2 top-level profile and provides v3 separately. Both registry fingerprints validate, and every result resolves to its declared profile. The portable legacy v2 evaluator is byte-identical to the previously validated core; selected v3 evaluator bytes match this audit.
- The main workflow and appendix explicitly scope strict600 to the selected MJD/EXO-200 records. They correctly distinguish energy bins from score-quantile histograms, retained pooling from pre-sparse group weighting, range and support filtering from inclusive AUC, and retained v2 results from the current reevaluation.
- EXO-200 Transformer independence decreases by 0.0610128073–0.0890568412 relative to its historical estimator. Region–Fourier now dominates region–RoPE in both reported metrics, removing only region–RoPE from the former Transformer Pareto set. Region–Fourier retains the highest matched AUC and Summary–Fourier the highest independence. MJD's six selected Transformer rankings and Pareto set remain unchanged. These are descriptive single-run comparisons, not statistical-significance claims.

## Reproducing this independent review

Run from any working directory:

```bash
python3 '/home/wenyu/iclr final paper/unified_5kev_evaluation/transformer_exo_mjd_20260917/audit/review_final_paper.py'
/home/wenyu/summer/.venv/bin/python -B '/home/wenyu/iclr final paper/unified_5kev_evaluation/transformer_exo_mjd_20260917/audit/independent_synthetic_v3.py'
/home/wenyu/summer/.venv/bin/python -B '/home/wenyu/iclr final paper/unified_5kev_evaluation/transformer_exo_mjd_20260917/audit/independent_sources.py'
```

The numerical comparison script and exact input/detail manifests are retained alongside these files. Source arrays stay external to the paper repository; their original locations and hashes are preserved in the manifests.

## Final PDF: PASS

`final_pdf_review.json` records the final 25-page PDF hash `e0b2d5e6d3a3b8cf710f5ecc27ad8985294942c22fa03ed08ac40a2a738ddaff`. Independent raster inspection covered page 7 (main table and scope note), page 14 (support, coverage and loss tables), and page 15 (classic class-level and 15-Transformer diagnostics). Tables are complete, the intended best-result numerals are bold, and no text overlaps or table clipping were found. Extracted text contains the selected final values and the MJD RoPE scope note. The build log has no undefined references or citations, overfull boxes, or missing glyphs. Existing Nimbus Roman small-caps fallback and underfull-box warnings remain.

The final manuscript source review passes all 32 checks; the PDF review passes all nine automated checks plus visual inspection. There are no outstanding requested corrections from this independent audit.
