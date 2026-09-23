# EXO-200 / SuperNEMO source and change audit

Final protocol: `EnergyBench-unified-5keV-overflow-v2.0.0`. The user subsequently requested one additional overflow bin for **all E>3000 keV**. The final evaluation therefore has 600 ordinary 5-keV bins on [0,3000] plus a separate overflow bin. Exactly 3000 belongs to the last ordinary bin. Negative energies are excluded; overflow values are neither clipped nor discarded.

The final requested model scope is classic models plus the separate SuperNEMO energy-only illustration. Transformer extraction/replay completed before the user excluded Transformers; those files are audit history only.

## Source identity and score direction

- EXO-200: four saved `outputs/classification/<architecture>/energybench_predictions.npz` bundles, each containing 140,383 held-out events (66,062 signal, 74,321 background). Every bundle agrees exactly with its native `predictions.npz`, and the saved checkpoint SHA256 matches the actual checkpoint. All metadata columns agree across models. Test runs 8967, 8970, 8971 and 9010 are disjoint from the recorded training/validation runs. Every event ID was joined to the raw HDF5 `event_number`; `Rotated_energy` and `Charge_cluster_number` were checked exactly. Stored label 0 means signal (`n_CCL=1`), while the stored logit grows toward background label 1. The standard NPZ uses `label=1-stored_label`, `score=-stored_logit`, with no sigmoid.
- SuperNEMO trained task: four classic `test_evaluation/test_predictions.npz` bundles, each containing 589,824 events (327,680 `2nu=1`, 262,144 `Bi214=0`). The saved scalar logit grows toward 2nu. Original checkpoint hashes and the exact split-manifest hash match. All test events were reconstructed from raw tracker HDF5 rows in manifest order; repeated event ID, raw class label, E1 and E2 were checked, then E1/E2 were separately cast to float64 before addition. All metadata columns agree across the four models.
- Physical energies are keV, with no source clipping. EXO contains 3 signal and 1,423 background overflow events; SuperNEMO training contains 0 signal and 40 background overflow events. None have negative energy. The standard NPZs have always retained these events, so the user-requested overflow revision needs no score, checkpoint or test-set changes.
- Exact per-model source paths, hashes, original metadata, full-precision old values and paper locations are in `manifest.json`; its `raw_source_checks` records all raw source checks. Original experiment files and checkpoints were not modified.

## Historical protocol and replay

EXO-200 and trained SuperNEMO used **six class-balanced pooled energy quantile bins for matched AUC**, separately from **eight energy quantile bins per physical category for I**. Matching used 0.5% class-tail support trimming, overlap target, within-class normalized event weights, pooled weighted ROC with cross-energy-bin pairs and half credit for ties, 20 events of each class per eligible bin, and the documented count/coverage gates.

Historical I used up to 20 weighted score-quantile bins, natural-log JS, group-local weighted mean JS before the square root, and a base-mass-weighted mean of group I. Its nominal min_per_bin=20 determined the requested energy-bin count, but the local energy-bin retention threshold was two events. Its score edges and pooled histogram included all group events. The unified reference instead uses the fixed energy grid plus overflow, 20 events per group energy bin, and retained-event score quantiles/pooling. These deliberate estimator differences are documented rather than interpreted as model changes.

`replay_history.py` replayed all 14 already-extracted models from events. The maximum error over I, matched AUC and inclusive AUC is 0.0. Eight are in the final classic scope. `historical_replay_and_range_attribution.json` also retains intermediate range-exclusion/Q8/Q6 counterfactuals from superseded v1; **those range-exclusion counterfactuals are not the final protocol**.

## Final classic changes

| dataset | model | old_I | new_I | old_matched | new_matched |
|---|---|---|---|---|---|
| EXO-200 | mvcnn | 0.817986575 | 0.751932408 | 0.863853364 | 0.862813707 |
| EXO-200 | gine | 0.891950005 | 0.799472255 | 0.949997478 | 0.949430853 |
| EXO-200 | bigru | 0.871303032 | 0.789095619 | 0.930117535 | 0.929478223 |
| EXO-200 | mamba | 0.865789705 | 0.785369397 | 0.934678844 | 0.934079879 |
| SuperNEMO | mvcnn | 0.950361798 | 0.910148819 | 0.586360100 | 0.585941331 |
| SuperNEMO | gine | 0.955183978 | 0.912544729 | 0.590556347 | 0.590184094 |
| SuperNEMO | bigru | 0.974394804 | 0.920886348 | 0.579662041 | 0.579485174 |
| SuperNEMO | mamba | 0.975342039 | 0.921531494 | 0.579640686 | 0.579492745 |

Every classic matched AUC remains reportable; inclusive AUC remains unchanged to numerical precision on its original population. Full-precision deltas, matching counts and coverage are in `classic_change_attribution.csv`. Physical range filtering excludes no events in these two datasets under v2 because every event has nonnegative energy. All changes arise from the finer global grid, explicit overflow, and the harmonized I qualification/pooling rules.

I decreases in both physical groups for every classic model. The aggregate therefore does not hide an improving group. It does average unequal group effects, so `classic_group_changes.csv` reports each I_g and retention separately. The changes reflect estimator resolution and finite-bin statistics; they do not show that the unchanged models became less independent.

EXO-200: I order gine > bigru > mamba > mvcnn → gine > bigru > mamba > mvcnn; matched AUC order gine > mamba > bigru > mvcnn → gine > mamba > bigru > mvcnn. Old Pareto set: gine; new Pareto set: gine.

SuperNEMO: I order mamba > bigru > gine > mvcnn → mamba > bigru > gine > mvcnn; matched AUC order gine > mvcnn > bigru > mamba → gine > mvcnn > mamba > bigru. Old Pareto set: gine, bigru, mamba; new Pareto set: gine, mamba.

The SuperNEMO BiGRU/PointMamba-lite matched-AUC ordering changes by a very small single-run point-estimate difference. No statistical significance is asserted. EXO GINE remains best in both metrics among its four classic models.

## Separate 0nu energy-only illustration

The illustration contains the existing 196,608 0nubb and 262,144 Bi214 test events and is separate from the trained 2nu/Bi214 task. `illustration_manifest.json` preserves the raw-source checks. Its NPZ uses `score=energy=float64(E1)+float64(E2)`, with 0nubb positive. Original extraction and exact energy/offset checks were reused. The 758 signal and 40 background energies above 3000 are now assigned to the additional overflow bin in I and remain available to support estimation.

Unified I is 0.079209565131697732. Common support is [1169.39, 2734.1949999999997] keV, with 283 eligible matching bins. Matching status is `ok` and formal matched AUC is 0.5000239466039934. Signal coverage is 0.50424702962239587; background coverage is 0.57445907592773438; denominators are the complete original finite event populations.

The new overflow instruction restores support estimation on all nonnegative energies. The earlier v1 result that failed signal coverage applied range exclusion and is superseded; it is preserved only in `v1_before_overflow/`. No reporting threshold has been relaxed. Inclusive ROC and the original 2200-keV operating point stay unchanged.

## Figures and numerical checks

`code/plot_unified_illustration.py` reads standardized event arrays plus `results/illustration_weights.npz` from the shared evaluator. It writes `figures/supernemo_matching_50keV.csv`, `supernemo_matching_roc.csv`, `supernemo_matching_evidence.json` and `supernemo_energy_matching_auc.pdf/.png`. The original paper reference is `wing_contribution/sections/energybench_auc_figure.tex`, pointing to `wing_contribution/figures/supernemo_energy_matching_auc.pdf`.

Histograms retain the 50-keV display width and normalize using all original class events for inclusive spectra. Every displayed histogram integrates to one. Full ROC points are independently accumulated by score ties, checked for monotonicity, and agree with the shared result to 1e-10. The renderer uses the saved formal reporting status to distinguish any diagnostic-only weighted curve.

The final v2 PNG was opened and visually inspected: all three panels, legends, axes and overflow/coverage notes are readable and within the image bounds. `figures/visual_validation.json` records the reviewed PNG/PDF hashes.

## Limitations and preserved earlier work

No retraining or substitute inference was performed. Full raw HDF5 files were not hashed, but every selected event ID, label and energy was re-read and checked; offset indices and extracted event-energy arrays were hashed. Original model checkpoints were also hashed. All selected events have unit base weight.

A new hidden-inclusive file search enumerated 514,454 account paths; evidence is in `missing_transformer_search.json` and `artifact_path_search.txt.gz`. The user then excluded Transformers, so their absent source artifacts are not a final-scope blocker. Earlier Transformer NPZ/replay artifacts remain optional audit history, not paper updates.
