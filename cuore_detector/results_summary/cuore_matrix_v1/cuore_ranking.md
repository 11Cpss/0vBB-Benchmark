# CUORE Δt regression -- 3 x 3 matrix

Ranked by `test_rmse_ms_pileup_only` ascending, tie-broken by `test_r2` descending.

`ers` and `fractional_resolution_68` are NOT used: both are fraction-based and degenerate here, because ~51% of the Δt targets are exactly 0. `clean_ms.r2` is NaN by construction (zero variance).

## Official matrix

| # | tokenization | encoding | pile-up RMSE [ms] | RMSE [ms] | MAE [ms] | bias [ms] | R² | params |
|--:|---|---|--:|--:|--:|--:|--:|--:|
| 1 | raw_patches | rope | 749.0 | 575.9 | 174.7 | -31.3 | 0.9270 | 107841 |
| 2 | raw_patches | fourier_coordinates | 761.0 | 560.0 | 213.3 | 6.6 | 0.9309 | 114561 |
| 3 | segment_summary | fourier_coordinates | 782.3 | 623.0 | 215.0 | 54.1 | 0.9145 | 113409 |
| 4 | pulse_entities | fourier_coordinates | 797.7 | 612.1 | 177.6 | -40.3 | 0.9175 | 113409 |
| 5 | raw_patches | coordinate_mlp | 837.9 | 607.7 | 188.3 | -66.6 | 0.9187 | 112257 |
| 6 | segment_summary | rope | 840.5 | 608.5 | 256.8 | -53.4 | 0.9184 | 106689 |
| 7 | pulse_entities | rope | 872.5 | 631.5 | 293.3 | -23.2 | 0.9122 | 106689 |
| 8 | segment_summary | coordinate_mlp | 1025.0 | 744.0 | 230.4 | 11.4 | 0.8781 | 111105 |
| 9 | pulse_entities | coordinate_mlp | 1164.7 | 848.1 | 397.1 | -4.1 | 0.8416 | 111105 |

## Sweeps and diagnostics (NOT part of the official matrix)

These vary one hyperparameter of the RoPE cells. They are listed separately because every one of them shares `position_encoding == "rope"`, so ranking them alongside the official cells would compare a tuned family against untuned single configurations.

| # | tokenization | encoding | variant | pile-up RMSE [ms] | RMSE [ms] | MAE [ms] | bias [ms] | R² | params |
|--:|---|---|---|--:|--:|--:|--:|--:|--:|
| 1 | raw_patches | fourier_coordinates | seed4 | 717.4 | 564.2 | 189.4 | 29.1 | 0.9299 | 114561 |
| 2 | pulse_entities | rope | base4_seed4 | 721.5 | 584.9 | 177.7 | 36.6 | 0.9247 | 106689 |
| 3 | segment_summary | fourier_coordinates | seed3 | 723.6 | 547.7 | 189.2 | -6.0 | 0.9339 | 113409 |
| 4 | raw_patches | rope | seed4 | 735.0 | 615.4 | 233.1 | 69.5 | 0.9166 | 107841 |
| 5 | segment_summary | rope | seed1 | 739.2 | 603.8 | 246.7 | 97.7 | 0.9197 | 106689 |
| 6 | raw_patches | rope | base4 | 740.7 | 578.2 | 202.6 | 20.5 | 0.9264 | 107841 |
| 7 | segment_summary | rope | base32 | 756.2 | 574.6 | 229.8 | 76.4 | 0.9273 | 106689 |
| 8 | raw_patches | rope | timeonly | 757.8 | 555.3 | 238.8 | 45.6 | 0.9321 | 107841 |
| 9 | raw_patches | fourier_coordinates | seed3 | 761.5 | 573.9 | 247.6 | -16.2 | 0.9275 | 114561 |
| 10 | pulse_entities | rope | seed4 | 765.4 | 560.2 | 170.2 | -27.3 | 0.9309 | 106689 |
| 11 | segment_summary | rope | base4_seed4 | 770.0 | 585.1 | 191.0 | 32.0 | 0.9246 | 106689 |
| 12 | raw_patches | fourier_coordinates | seed1 | 773.8 | 586.3 | 167.2 | -25.6 | 0.9243 | 114561 |
| 13 | raw_patches | rope | base4_seed4 | 781.7 | 590.4 | 214.7 | 35.1 | 0.9232 | 107841 |
| 14 | segment_summary | fourier_coordinates | seed2 | 782.0 | 578.3 | 194.2 | -95.8 | 0.9263 | 113409 |
| 15 | pulse_entities | rope | base4 | 782.8 | 579.4 | 258.9 | 25.4 | 0.9261 | 106689 |
| 16 | pulse_entities | rope | base32 | 789.8 | 599.8 | 257.8 | 45.0 | 0.9208 | 106689 |
| 17 | segment_summary | rope | seed3 | 790.6 | 586.7 | 160.5 | -6.3 | 0.9242 | 106689 |
| 18 | segment_summary | fourier_coordinates | seed1 | 792.1 | 634.5 | 265.4 | 8.3 | 0.9113 | 113409 |
| 19 | segment_summary | rope | seed4 | 792.7 | 607.7 | 173.5 | 7.0 | 0.9187 | 106689 |
| 20 | pulse_entities | fourier_coordinates | seed1 | 795.9 | 589.7 | 179.7 | 22.9 | 0.9234 | 113409 |
| 21 | segment_summary | rope | base4_seed3 | 799.3 | 584.9 | 188.7 | 25.2 | 0.9246 | 106689 |
| 22 | pulse_entities | rope | base4_seed1 | 804.6 | 678.7 | 247.7 | 86.0 | 0.8985 | 106689 |
| 23 | pulse_entities | rope | timeonly | 805.1 | 595.1 | 260.9 | 49.8 | 0.9220 | 106689 |
| 24 | raw_patches | rope | seed2 | 806.1 | 574.0 | 182.3 | 21.1 | 0.9275 | 107841 |
| 25 | raw_patches | rope | base4_seed1 | 806.2 | 605.5 | 270.0 | 29.9 | 0.9193 | 107841 |
| 26 | raw_patches | coordinate_mlp | seed3 | 807.6 | 588.6 | 196.0 | -32.6 | 0.9237 | 112257 |
| 27 | pulse_entities | fourier_coordinates | seed3 | 809.3 | 639.4 | 232.5 | 51.8 | 0.9100 | 113409 |
| 28 | raw_patches | rope | base16 | 810.0 | 584.9 | 210.0 | -62.3 | 0.9247 | 107841 |
| 29 | raw_patches | rope | seed3 | 814.7 | 638.9 | 291.1 | 129.3 | 0.9101 | 107841 |
| 30 | pulse_entities | fourier_coordinates | seed2 | 815.8 | 588.5 | 206.4 | 19.4 | 0.9237 | 113409 |
| 31 | segment_summary | rope | base4 | 818.3 | 605.7 | 227.8 | -31.3 | 0.9192 | 106689 |
| 32 | raw_patches | rope | base32 | 821.8 | 607.2 | 306.8 | 69.6 | 0.9188 | 107841 |
| 33 | segment_summary | fourier_coordinates | seed4 | 822.0 | 592.5 | 220.6 | 87.1 | 0.9227 | 113409 |
| 34 | raw_patches | rope | base4_seed2 | 822.2 | 612.5 | 219.5 | -54.4 | 0.9174 | 107841 |
| 35 | pulse_entities | rope | seed2 | 824.5 | 625.5 | 274.1 | -18.5 | 0.9138 | 106689 |
| 36 | segment_summary | rope | timeonly | 826.3 | 610.4 | 284.8 | -34.6 | 0.9179 | 106689 |
| 37 | pulse_entities | fourier_coordinates | seed4 | 826.5 | 621.1 | 199.9 | 20.9 | 0.9150 | 113409 |
| 38 | pulse_entities | rope | seed3 | 829.0 | 665.0 | 202.6 | 70.2 | 0.9026 | 106689 |
| 39 | segment_summary | rope | base16 | 840.1 | 609.3 | 259.2 | 26.1 | 0.9182 | 106689 |
| 40 | segment_summary | rope | base4_seed2 | 841.8 | 633.2 | 256.3 | 32.4 | 0.9117 | 106689 |
| 41 | raw_patches | coordinate_mlp | seed1 | 843.1 | 620.0 | 292.0 | -2.1 | 0.9153 | 112257 |
| 42 | raw_patches | rope | seed1 | 846.4 | 653.4 | 184.7 | 17.4 | 0.9060 | 107841 |
| 43 | pulse_entities | rope | base4_seed2 | 846.6 | 619.1 | 231.3 | -95.0 | 0.9156 | 106689 |
| 44 | segment_summary | rope | base4_seed1 | 856.0 | 622.3 | 222.5 | -99.7 | 0.9147 | 106689 |
| 45 | segment_summary | rope | seed2 | 860.9 | 623.3 | 214.9 | -109.4 | 0.9144 | 106689 |
| 46 | raw_patches | fourier_coordinates | seed2 | 867.6 | 620.9 | 223.1 | 54.2 | 0.9151 | 114561 |
| 47 | pulse_entities | rope | base16 | 879.5 | 698.5 | 335.8 | 42.8 | 0.8925 | 106689 |
| 48 | raw_patches | rope | base4_seed3 | 881.0 | 639.3 | 293.4 | -42.1 | 0.9100 | 107841 |
| 49 | pulse_entities | rope | seed1 | 881.9 | 694.8 | 251.9 | 31.4 | 0.8937 | 106689 |
| 50 | segment_summary | coordinate_mlp | seed2 | 882.3 | 624.6 | 203.2 | 23.4 | 0.9141 | 111105 |
| 51 | raw_patches | coordinate_mlp | seed2 | 888.3 | 629.9 | 210.9 | 6.1 | 0.9126 | 112257 |
| 52 | segment_summary | coordinate_mlp | seed3 | 892.5 | 632.4 | 204.5 | 27.7 | 0.9119 | 111105 |
| 53 | raw_patches | coordinate_mlp | seed4 | 969.4 | 703.8 | 221.4 | -8.9 | 0.8909 | 112257 |
| 54 | pulse_entities | rope | base4_seed3 | 975.7 | 760.3 | 327.7 | 97.2 | 0.8727 | 106689 |
| 55 | segment_summary | coordinate_mlp | seed4 | 1047.2 | 748.1 | 280.5 | 19.5 | 0.8767 | 111105 |
| 56 | segment_summary | coordinate_mlp | seed1 | 1089.6 | 796.9 | 262.2 | 77.8 | 0.8601 | 111105 |
| 57 | pulse_entities | coordinate_mlp | seed1 | 1142.1 | 862.4 | 355.5 | 6.1 | 0.8362 | 111105 |
| 58 | pulse_entities | coordinate_mlp | seed4 | 1189.7 | 951.0 | 355.0 | 171.6 | 0.8008 | 111105 |
| 59 | pulse_entities | coordinate_mlp | seed3 | 1200.7 | 874.7 | 333.3 | 16.3 | 0.8315 | 111105 |
| 60 | pulse_entities | coordinate_mlp | seed2 | 1220.3 | 912.5 | 423.9 | 41.2 | 0.8166 | 111105 |

Parameter counts differ across the tokenization axis (`raw_patches` has feature_dim 20 vs 2) and across the encoding axis (`rope` adds no positional parameters). Capacity is not matched -- do not read these cells as a controlled capacity comparison. All runs are single-seed: differences smaller than the within-family spread are not interpretable.
