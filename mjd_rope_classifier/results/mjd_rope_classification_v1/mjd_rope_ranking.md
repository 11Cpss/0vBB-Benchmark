# MJD RoPE classification -- 3-tokenization matrix

Ranked by `test_auc` descending (signal = clean, all four PSD cuts). Chance level: AUC 0.5, accuracy 0.62 (majority class, non-clean).

`energy independence` / `worst-group` are EnergyBench's class-conditional score/energy independence score (run post hoc by `scripts/energy_independence.py` from each run's `predictions.npz`; 1.0 = the classifier's output distribution is identical across energy within each class, 0.0 = maximal class-conditional energy dependence). Blank when that script hasn't been run yet.

## Official matrix

| # | tokenization | test AUC | test accuracy | energy independence | worst-group | rope_base | params |
|--:|---|--:|--:|--:|--:|--:|--:|
| 1 | raw_patches | 0.9157 | 0.8405 | 0.7684 | 0.7632 | 2.00 | 107841 |
| 2 | segment_summary | 0.8780 | 0.7969 | 0.7768 | 0.7679 | 2.00 | 106689 |
| 3 | pulse_entities | 0.7932 | 0.6782 | 0.7491 | 0.7447 | 2.00 | 106689 |

Parameter counts differ across the tokenization axis (`raw_patches` has feature_dim = patch_size vs. 2 for the other two). All runs are single-seed: differences smaller than the within-family spread are not interpretable.
