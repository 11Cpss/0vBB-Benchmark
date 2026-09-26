# Energy Regression Metric Design

## Problem

Some benchmark tasks are not naturally classification tasks. For energy regression, ordinary MSE is not enough because rare-event physics cares about the **energy spectrum**, not only average pointwise error.

A model can have a good global MSE while still failing in the physics-relevant energy region. It can also perform well on common energy ranges while doing poorly in rare or high-value regions.

The benchmark should therefore prioritize two things:

1. Spectrum-aware regression performance across energy bins
2. Event-level regression accuracy as a secondary metric

## Core Idea

For a dataset with true energy `E_true` and predicted energy `E_pred`, divide the true energy range into bins and compute regression error inside each bin.

This tests whether the model performs consistently across the spectrum instead of only optimizing the densest part of the distribution.

## Metric 1: Global Regression Error

Use a standard event-level metric:

```text
MSE = mean((E_pred - E_true)^2)
RMSE = sqrt(MSE)
MAE = mean(abs(E_pred - E_true))
```

This gives the usual regression performance.

## Metric 2: Binned Spectrum Regression Error

Split true energy into `K` bins:

```text
B_1, B_2, ..., B_K
```

For each bin:

```text
MSE_k = mean((E_pred - E_true)^2 | E_true in B_k)
RMSE_k = sqrt(MSE_k)
MAE_k = mean(abs(E_pred - E_true) | E_true in B_k)
```

Then aggregate across bins:

```text
Binned-MSE = mean_k(MSE_k)
Binned-RMSE = mean_k(RMSE_k)
Binned-MAE = mean_k(MAE_k)
```

This gives each energy region equal importance, instead of letting high-density regions dominate.

## Metric 3: Weighted Binned Error

Some bins may be more important for rare-event searches, especially the region of interest near a physics signal.

Define weights:

```text
w_k = importance of energy bin k
```

Then:

```text
Weighted-Binned-MSE = sum_k w_k * MSE_k / sum_k w_k
```

Possible weights:

- uniform weights across bins
- higher weight near signal region / ROI
- inverse-frequency weights to avoid common-spectrum domination
- uncertainty-aware weights if bin statistics differ strongly

## Metric 4: Spectrum Shape Agreement

Pointwise energy error is not the whole story. We also care whether the predicted energy distribution has the right shape.

Construct histograms:

```text
H_true = histogram(E_true)
H_pred = histogram(E_pred)
```

Then compare the two distributions using:

```text
Histogram L1 = sum_k abs(H_pred_k - H_true_k)
Histogram L2 = sqrt(sum_k (H_pred_k - H_true_k)^2)
Jensen-Shannon divergence = JSD(H_pred, H_true)
Wasserstein distance = earth mover distance between spectra
```

For physics, Wasserstein distance and binned histogram error are intuitive because they measure how much the predicted spectrum is shifted or distorted.

## Recommended Benchmark Metrics

For the paper, make binned energy regression the standard regression evaluation. Report global regression error as a secondary metric:

### 1. Primary Metric: Binned Energy Regression

```text
Uniform-Binned RMSE across true-energy bins
```

This answers:

> Does the model perform consistently across the full energy range?

### 2. Secondary Metric: Global Regression Error

```text
RMSE or MAE
```

This answers:

> How close is the predicted energy for each event on average?

## Why Global MSE Is Not Enough

If the energy spectrum is non-uniform, most events may lie in a few common regions. A model can minimize global MSE by doing well there while ignoring low-statistics regions.

That is dangerous for rare-event searches because the important region may be rare, narrow, or near the tail of the distribution.

## Example Benchmark Table

If there are 5 datasets and 2 metric families, each model should produce 10 headline results:

```text
5 datasets x 2 metrics = 10 benchmark entries per model
```

Example:

| Dataset | Model | Event RMSE | Spectrum Metric |
|---|---|---:|---:|
| CUORE | specialized | ... | ... |
| CUORE | unified transformer | ... | ... |
| MJD | specialized | ... | ... |
| MJD | unified transformer | ... | ... |
| NEXT | specialized | ... | ... |
| NEXT | unified transformer | ... | ... |
| SuperNEMO | specialized | ... | ... |
| SuperNEMO | unified transformer | ... | ... |
| KLZ | specialized | ... | ... |
| KLZ | unified transformer | ... | ... |

## Practical Design Recommendation

Use:

```text
Primary regression metric: uniform-binned RMSE across true-energy bins
Secondary regression metric: global RMSE or MAE
Secondary spectrum metric: Wasserstein distance or Jensen-Shannon divergence between true and predicted energy histograms
```

This keeps the benchmark understandable while adding a physics-aware check.

## Meeting Sentence

"For regression, global MSE is not enough because our energy spectra are not uniform. I think we should report both an event-level error like RMSE and a spectrum-aware metric where we bin true energy and compute per-bin MSE/RMSE, so models are judged on whether they reproduce the full energy spectrum instead of only the densest regions."
