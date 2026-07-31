# Weekly Task Summary

## Main Task

Start the unified transformer benchmark direction with one concrete dataset, one experiment, and the standard physics-aware metrics.

## Best Starting Dataset

Use **CUORE** first because:

- it is already downloaded locally
- it is small enough to run first experiments quickly
- it has clean labels
- the data modality is simple: 1D waveform
- it is a good testbed for tokenization and positional encoding

## Experiment Question

For CUORE pile-up classification:

> How do different waveform tokenization and positional encoding strategies affect transformer performance?

## Metrics

Use energy-matched AUC as the primary classification metric. Regular ROC-AUC can be reported later as a secondary comparison.

```text
Energy-matched ROC-AUC
```

For regression tasks, use binned energy regression as the primary regression evaluation: divide true energy into bins and compute error within each bin. Overall RMSE or MAE can be reported later as secondary metrics.

Additional checks later:

- accuracy
- F1
- background rejection at fixed signal efficiency
- score dependence on waveform amplitude / peak / integral

## Tokenization Strategies To Test

### 1. Fixed Patch Tokens

Split each 10,000-sample waveform into equal contiguous patches.

Examples:

- 100 tokens x 100 samples
- 200 tokens x 50 samples
- 50 tokens x 200 samples

Each token is a small local waveform segment.

### 2. Downsampled Point Tokens

Downsample the waveform into fewer time points.

Examples:

- 10,000 samples to 500 time tokens
- 10,000 samples to 250 time tokens

Each token is one compressed time point.

### 3. Summary-Feature Tokens

Convert waveform regions into engineered feature tokens.

Example token features:

- patch mean
- patch max
- patch standard deviation
- patch slope
- patch integral
- patch peak position

This tests whether physics-lite waveform summaries work better than raw patches.

### 4. Hybrid Tokens

Combine raw patch embeddings with summary-feature embeddings.

This may be a strong middle ground between black-box transformer tokens and physics-aware features.

## Positional Encoding Strategies To Test

### 1. Learned Absolute Position

Give each patch index a learned embedding.

Good first baseline.

### 2. Sinusoidal Absolute Position

Use classic transformer sinusoidal time encoding.

Good because waveform time has natural order.

### 3. Relative Position / Local Attention Bias

Let nearby waveform patches know they are close together.

Potentially useful because pulse shape is local in time.

### 4. Physics-Aware Time Encoding

Encode time relative to known trigger structure:

- pretrigger baseline region
- trigger time around 3 seconds
- post-trigger pulse/tail region

This is promising because CUORE waveforms include a 3-second pretrigger baseline.

## One-Week Deliverable

Produce a small experiment table:

| Dataset | Task | Tokenization | Positional Encoding | Metric |
|---|---|---|---|---:|
| CUORE | pile-up classification | fixed patches | learned absolute | energy-matched AUC |
| CUORE | pile-up classification | fixed patches | sinusoidal | energy-matched AUC |
| CUORE | pile-up classification | summary-feature patches | learned absolute | energy-matched AUC |
| CUORE | pile-up classification | hybrid patches | physics-aware time | energy-matched AUC |

The broader comparison will use approximately eight selected combinations of tokenization and positional encoding, alongside a specialized model for each dataset.

## Connection To Bigger Paper

This is the first prototype for the larger question:

> Across rare-event datasets with different structures, should we use dataset-specific models or unified tokenized transformer-style models?

CUORE is the simplest first test. Later, the same logic can extend to:

- MJD: waveform tokens
- NEXT: hit / voxel / point-cloud tokens
- SuperNEMO: feature tokens
- EXO-200 or KLZ: detector-event tokens, depending on access and paper direction

## Separate Metric Thread

There was also a separate regression-metric discussion:

- global MSE/RMSE is not enough
- energy spectra are non-uniform
- models should be evaluated across true-energy bins
- proposed metric: uniform-binned RMSE or binned MSE
- optional spectrum metric: Wasserstein distance or Jensen-Shannon divergence between predicted and true energy histograms

That matters for energy regression datasets such as MJD or EXO-200. The immediate transformer prototype can start with CUORE classification, using energy/amplitude-matched AUC where the necessary control variable is available.
