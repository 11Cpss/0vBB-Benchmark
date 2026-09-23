# EXO-200 Transformer source audit and final strict-600 reevaluation

Completed all nine published Transformer variants from their original per-event predictions under `EnergyBench-unified-5keV-range-v3.0.0`, fingerprint `6d696d87f1307b3f6d27bb9ef7ec443e1d8d98b9d95e90713f553c70edb42953`. Final results are in `strict600/`. Exactly 600 energy bins cover 0–3000 keV; the last bin includes exactly 3000 keV. Energies above 3000 keV are excluded from the energy metrics. No inference, training or benchmark-source changes were made.

## Reproduce final evaluation

```bash
/home/wenyu/summer/.venv/bin/python -B \
  "/home/wenyu/iclr final paper/unified_5kev_evaluation/transformer_exo_mjd_20260917/exo/recompute_strict600.py"
```

The script verifies the unchanged input hashes and frozen v3 implementation/configuration hashes before evaluation. `manifest.json` records the full original source audit and historical replay. `strict600/manifest.json` links every final detail to its original checkpoint, per-event prediction, standardized file and source audit. `strict600/summary.json`, `summary.csv` and `group_summary.csv` hold complete-precision final results. `strict600/detail/*.json` contain every group and bin diagnostic. The original full-population `standardized/*.npz` files are unchanged.

The initially requested overflow-bin computation is preserved only as a historical intermediate in `v2_historical_snapshot/`, with original evaluator and hash inventory. Root-level `detail/` and `summary.*` are likewise v2 intermediate outputs and must not be used for the final paper. The original `recompute_exo_transformers.py` is a v2 provenance/replay script and intentionally rejects the current v3 protocol.

## Inputs, identities and orientation

The nine source runs are under `/home/klz/Data/zeronu_benchmark/Transformer_Approach/exo200_detector/results/transformer_official_v1/`, with names `classification__{pulse_entities,raw_patches,segment_summary}__{coordinate_mlp,fourier_coordinates,rope}`. A separate `segment_summary__rope_partial_20260902_141826` directory is an incomplete archival run and was excluded; its checkpoint was never substituted.

All nine original `best.pt` hashes match the saved prediction provenance. Each exact architecture was instantiated from its source configuration and strict-loaded with the original checkpoint using `torch.load(..., weights_only=True)`. Trainable parameters were counted exactly and match both `run_config.json` and `run_summary.json`. This validation performs no forward pass.

The canonical EXOBench test metadata was rebuilt from the raw HDF5 files by the original `build_test_metadata` adapter and its deterministic run-group split. Every saved model’s event IDs, labels, physical categories, energies, weights, runs and split flags match it exactly, in order. These seven arrays also match the classic CNN evaluation bundle exactly. There are 140,383 unique test events: 66,062 signal and 74,321 background, from runs 8967, 8970, 8971 and 9010. The recorded train and validation run groups are disjoint.

This is the same cluster-count classification task as the classic EXO models: stored label 0 is signal (`Charge_cluster_number==1`), stored label 1 is background (`Charge_cluster_number>1`). Training produces one scalar logit increasing toward background. Historical EnergyBench already treated signal as positive by negating that logit. Standardized files use `label=1-stored_label` and `score=-stored_logit`; no sigmoid, score calibration or ranking alteration is applied. The original native float32 logits equal the EnergyBench float64 export exactly.

Conditioning energy is raw physical `Rotated_energy` in keV, read by event index from the unchanged test HDF5 shards. The adapter and frozen manifest specify keV, and every exported value was checked against the canonical raw-source reconstruction. No energy conversion, clipping or energy recovery is required. There are no negative or missing energies. Strict energy evaluation excludes 1,423 background events and 3 signal events above 3000 keV; it does not clip or assign them an overflow bin. All source test events remain in inclusive AUC.

## Historical replay and final protocol differences

Historical I uses eight energy quantile bins separately within each physical category, with up to 20 weighted score-quantile bins. Historical matched AUC independently uses six class-balanced pooled energy quantile bins. The frozen `roc.py`, `dependence.py` and `utils.py` hashes match every saved source provenance. Exact replay of I, matched AUC and inclusive AUC yields zero absolute error for all nine models. Current paper values agree after stored decimal rounding; no workbook or paper summary is a recomputation input.

The final evaluator preserves natural-log JS, the weighted mean of within-group JS before the square root, the final aggregation of group I, score-quantile histogram rules, overlap target, class normalization, pooled weighted AUC, cross-bin pairs and half ties. It replaces adaptive energy quantiles with the global 600-bin grid. Unlike the historical I implementation’s local threshold of two, each eligible group energy bin requires 20 effective observations. Score quantiles and the pooled score histogram are computed on retained eligible group events. Group aggregation masses are the in-range, finite group populations before sparse exclusions.

Final I uses background aggregation mass 72,898 and signal mass 66,059. Background retains 72,641 events in 484 bins after 257 sparse exclusions; signal retains 65,907 events in 438 bins after 152 sparse exclusions. The separate range losses are 1,423 and 3. Retained fractions with original finite denominators are 0.9773953525921342 background and 0.9976537192334474 signal. Both physical-category I values decline from the historical estimate for every model; `strict600/group_summary.csv` reports each value, so the aggregate cannot conceal an improving category.

Final matching support is [538.1772625732422, 2662.5670739746092] keV, estimated after the range cut. All 426 intersecting 5-keV bins pass 20 observations per class, so there is no additional matching sparse loss. After range filtering, support exclusions are 2,932 background and 3,587 signal events. Matching retains 69,966 background and 62,472 signal events. Coverage uses the original finite class populations (74,321 and 66,062), yielding background 0.9414028336540143 and signal 0.9456571099875875. ESS is 61,626.880359525014 background and 51,923.64134977649 signal. All nine matching estimates pass the reporting gates.

The matched-AUC changes from history chiefly reflect overlap weighting on 426 fixed bins rather than six broad quantile bins, with the additional explicitly recorded support shift from strict range selection. The old/v2 support was [538.5922875976562, 2662.7574169921877] keV; final support includes 45 more signal and 3 fewer background events despite the upper range cut. This follows from recomputing class quantiles after range selection, not an input or test-set change. The I changes combine finer conditioning, the documented histogram/qualification harmonization, and range selection. Relative to v2, signal I_g is unchanged because its three overflow events had failed the old 20-event threshold; background pooling and final group weights change. These are estimator changes on fixed models, not evidence that the underlying models changed.

## Final results

| Model | Parameters | Historical I | Final I | Historical matched AUC | Final matched AUC |
|---|---:|---:|---:|---:|---:|
| entity_mlp | 111297 | 0.803336097861 | 0.737165079973 | 0.941942758282 | 0.941321621005 |
| entity_fourier | 115905 | 0.797026444378 | 0.729183537355 | 0.947602884139 | 0.947120535171 |
| entity_rope | 106689 | 0.776349424366 | 0.715336617072 | 0.946470744831 | 0.945924979925 |
| region_mlp | 120769 | 0.888355354664 | 0.800938157908 | 0.969817658399 | 0.969557320963 |
| region_fourier | 125377 | 0.872379074836 | 0.790051655773 | 0.972585937531 | 0.972261076637 |
| region_rope | 116161 | 0.878288005706 | 0.789231164490 | 0.971190767911 | 0.970909684100 |
| summary_mlp | 111425 | 0.859148204962 | 0.776141011515 | 0.947539227320 | 0.946954508099 |
| summary_fourier | 116033 | 0.889024185161 | 0.801902077742 | 0.960438763528 | 0.960023743125 |
| summary_rope | 106817 | 0.846467714151 | 0.772731187603 | 0.955702690172 | 0.955170674747 |

Inclusive AUC remains on all 140,383 test events and changes by at most 1.11e-16, due to floating-point accumulation. I decreases by 0.061013–0.089057; matched AUC decreases by 0.000260–0.000621.

Summary + Fourier retains the largest Transformer I; Region + Fourier retains the largest matched AUC. Region + Fourier now also exceeds Region + RoPE in I, removing Region + RoPE from the within-Transformer Pareto set. The final Pareto set is {Region + MLP, Region + Fourier, Summary + Fourier}. Matched-AUC ordering otherwise remains unchanged. These are single-run point estimates; statistical significance has not been established.

## Limits and preserved evidence

The raw HDF5 inventory records path, size and mtime; entire multi-GB detector shards were not hashed. Every selected physical energy, class label, event ID and test membership was reconstructed from those unchanged shards and compared exactly. Original predictions, checkpoints, configurations, source metric files and evaluator code were hashed. The current raw-source README describes six variants, but the training notebook includes all three RoPE variants and complete canonical artifacts exist for all nine. The partial archival RoPE run was not used.

The v2 protocol’s historical scope text is superseded for this campaign by explicit user authorization to include EXO Transformers. The final v3 protocol implements the user’s later instruction to use exactly 600 bins without overflow. Source, checkpoint, label direction, test membership and inclusive population were held fixed through both intermediate and final evaluations.
