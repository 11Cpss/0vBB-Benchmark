# CUORE first-two-pulse Δt regression -- final benchmark

Task: regress the time gap between the first two detected pulses in a CUORE waveform, from `pulseFinder/dtBetweenPeaks[:, 0]`.
Test set: all 1,000 events of `cuoreTest.h5`, identical for every run.
Single-pulse events are kept with Δt = 0 (~51% of events).

Aggregated over up to **5 seeds** per configuration (seed drives both weight initialization and the train/val partition; the test set is fixed). Cells show mean ± sd.

**Seed noise floor: ±51.1 ms** on pile-up RMSE (pooled within-configuration standard deviation). Differences smaller than roughly twice this are not interpretable.

`ers` and `fractional_resolution_68` are excluded throughout: both are fraction-based and degenerate against the ~51% point mass at Δt = 0.

## Official matrix (3 tokenizations × 3 positional encodings)

| # | tokenization | encoding | n | pile-up RMSE [ms] | RMSE [ms] | MAE [ms] | bias [ms] | R² |
|--:|---|---|--:|--:|--:|--:|--:|--:|
| 1 | raw_patches | fourier_coordinates | 5 | 776.3 ± 55.4 | 581.1 ± 24.5 | 208.1 ± 31.0 | 9.6 ± 32.7 | 0.9255 ± 0.0064 |
| 2 | segment_summary | fourier_coordinates | 5 | 780.4 ± 35.7 | 595.2 ± 34.9 | 216.9 ± 30.2 | 9.5 ± 69.5 | 0.9218 ± 0.0091 |
| 3 | raw_patches | rope | 5 | 790.2 ± 46.8 | 611.5 ± 36.0 | 213.2 ± 49.2 | 41.2 ± 60.8 | 0.9174 ± 0.0097 |
| 4 | segment_summary | rope | 5 | 804.8 ± 47.6 | 606.0 ± 13.1 | 210.5 ± 42.8 | -12.9 ± 76.9 | 0.9191 ± 0.0035 |
| 5 | pulse_entities | fourier_coordinates | 5 | 809.0 ± 12.8 | 610.2 ± 21.6 | 199.2 ± 22.4 | 15.0 ± 33.6 | 0.9179 ± 0.0058 |
| 6 | pulse_entities | rope | 5 | 834.6 ± 46.4 | 635.4 ± 50.4 | 238.4 ± 51.0 | 6.5 ± 42.8 | 0.9106 ± 0.0139 |
| 7 | raw_patches | coordinate_mlp | 5 | 869.3 ± 63.0 | 630.0 ± 44.1 | 221.7 ± 41.4 | -20.8 ± 29.4 | 0.9123 ± 0.0126 |
| 8 | segment_summary | coordinate_mlp | 5 | 987.3 ± 94.2 | 709.2 ± 76.6 | 236.2 ± 34.5 | 32.0 ± 26.3 | 0.8882 ± 0.0237 |
| 9 | pulse_entities | coordinate_mlp | 5 | 1183.5 ± 30.6 | 889.7 ± 41.8 | 372.9 ± 36.7 | 46.2 ± 72.1 | 0.8254 ± 0.0166 |

## RoPE hyperparameter variants (not part of the official matrix)

| # | tokenization | variant | n | pile-up RMSE [ms] | RMSE [ms] | MAE [ms] | bias [ms] | R² |
|--:|---|---|--:|--:|--:|--:|--:|--:|
| 1 | segment_summary | base32 | 1 | 756.2 | 574.6 | 229.8 | 76.4 | 0.9273 |
| 2 | raw_patches | timeonly | 1 | 757.8 | 555.3 | 238.8 | 45.6 | 0.9321 |
| 3 | pulse_entities | base32 | 1 | 789.8 | 599.8 | 257.8 | 45.0 | 0.9208 |
| 4 | pulse_entities | timeonly | 1 | 805.1 | 595.1 | 260.9 | 49.8 | 0.9220 |
| 5 | raw_patches | base4 | 5 | 806.4 ± 51.8 | 605.2 ± 23.3 | 240.0 ± 39.4 | -2.2 ± 42.6 | 0.9192 ± 0.0062 |
| 6 | raw_patches | base16 | 1 | 810.0 | 584.9 | 210.0 | -62.3 | 0.9247 |
| 7 | segment_summary | base4 | 5 | 817.1 ± 34.1 | 606.3 ± 21.7 | 217.2 ± 28.1 | -8.3 ± 57.6 | 0.9190 ± 0.0058 |
| 8 | raw_patches | base32 | 1 | 821.8 | 607.2 | 306.8 | 69.6 | 0.9188 |
| 9 | pulse_entities | base4 | 5 | 826.2 ± 95.0 | 644.5 ± 75.8 | 248.7 ± 54.1 | 30.0 ± 76.4 | 0.9075 ± 0.0223 |
| 10 | segment_summary | timeonly | 1 | 826.3 | 610.4 | 284.8 | -34.6 | 0.9179 |
| 11 | segment_summary | base16 | 1 | 840.1 | 609.3 | 259.2 | 26.1 | 0.9182 |
| 12 | pulse_entities | base16 | 1 | 879.5 | 698.5 | 335.8 | 42.8 | 0.8925 |

## Marginal means on pile-up RMSE [ms]

**Tokenization** (averaged over the other axis):

- `raw_patches`: 811.9
- `segment_summary`: 857.5
- `pulse_entities`: 942.4

**Encoding** (averaged over the other axis):

- `fourier_coordinates`: 788.6
- `rope`: 809.9
- `coordinate_mlp`: 1013.4

## Differences that clear the noise floor

Best configuration: **raw_patches x fourier_coordinates (776.3 ms)**.

Statistically indistinguishable from it (within 2 x 51.1 ms):

- segment_summary x fourier_coordinates (780.4 ms, +4.1)
- raw_patches x rope (790.2 ms, +14.0)
- segment_summary x rope (804.8 ms, +28.5)
- pulse_entities x fourier_coordinates (809.0 ms, +32.8)
- pulse_entities x rope (834.6 ms, +58.4)
- raw_patches x coordinate_mlp (869.3 ms, +93.0)

Genuinely worse than the best configuration:

- segment_summary x coordinate_mlp (987.3 ms, +211.1)
- pulse_entities x coordinate_mlp (1183.5 ms, +407.2)

## Caveats

- Parameter counts are not matched: `raw_patches` has feature_dim 20 vs 2 for the other tokenizations, and `rope` adds no positional parameters while `coordinate_mlp` adds ~4.4k and `fourier_coordinates` ~6.7k.
- The tokenizations differ in information content, not just layout: `raw_patches` preserves all 10,000 samples, `segment_summary` compresses to 500 mean/RMS pairs, `pulse_entities` keeps 500 of 10,000 raw samples.
- Labels are heuristic `pulseFinder` output, not ground truth; `tailElevated` events carry lower-bound Δt.
- Reported RMSE/MAE/bias are dominated by a heavy tail: median absolute error is several times smaller than the mean.
