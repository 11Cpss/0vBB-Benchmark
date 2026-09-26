# CUORE Transformer Experiment Spec

## Purpose

This document describes a first transformer benchmark experiment for the CUORE dataset. It is written so another AI assistant or collaborator can implement the experiment without needing prior conversation context.

## Research Goal

Use CUORE as the first testbed for a unified transformer benchmark on rare-event physics data.

The core question is:

> For detector waveforms, how do different tokenization strategies and positional encodings affect transformer performance?

This is part of a larger benchmark-paper direction:

> Across rare-event physics datasets with different data structures, can unified tokenized transformer-style models compete with dataset-specific models?

## Dataset

Dataset: **CUORE Pulse Classification Dataset**

Local path:

```text
data/raw/cuore/20721645/cuoreTraining.h5
data/raw/cuore/20721645/cuoreTest.h5
```

Training file keys:

```text
Waveform: shape=(9000, 10000), dtype=float64
Labels: shape=(9000,), dtype=int64
eventId: shape=(9000,), dtype=int64
normalizationMaximum: shape=(9000, 1), dtype=float64
normalizationOffset: shape=(9000, 1), dtype=float64
normalizationScale: shape=(9000, 1), dtype=float64
```

Task:

```text
Binary classification
```

Labels:

```text
0 = clean single-pulse event
1 = pile-up event
```

Input:

```text
10,000-sample 1D waveform
```

Physical interpretation:

- Each waveform is 10 seconds long.
- Sampling rate is 1 kHz.
- The first 3 seconds are pretrigger baseline.
- The model should distinguish clean single-pulse events from pile-up events, where two or more pulses overlap.

## Data Normalization

The HDF5 waveforms are raw. Use the provided normalization arrays:

```python
waveforms_normalized = (waveforms_raw - normalizationOffset) / (normalizationScale + 1e-8)
waveforms_normalized = np.clip(waveforms_normalized, 0.0, 1.0)
```

Do not fit normalization statistics on train+validation together. The provided per-event normalization is the intended starting point.

## Primary Metric

Use:

```text
Energy-matched AUC, or an amplitude-matched AUC proxy where calibrated energy is unavailable
```

Secondary metrics:

```text
regular ROC-AUC
accuracy
F1
background rejection at fixed signal efficiency
```

For the first experiment, use the primary matched metric for model selection and report regular ROC-AUC as a secondary comparison.

## Baseline Models

Before transformer variants, implement at least one simple baseline:

1. Logistic regression or random forest on waveform summary features
2. Simple 1D CNN on the normalized waveform

The transformer should be compared against at least one non-transformer baseline.

## Transformer Model Skeleton

General pipeline:

```text
raw waveform
-> normalize
-> tokenize
-> embed tokens
-> add positional encoding
-> transformer encoder
-> pooling or [CLS] token
-> binary classification head
-> ROC-AUC evaluation
```

Keep the model small for initial experiments:

```text
embedding dimension: 64 or 128
number of transformer layers: 2-4
number of attention heads: 4
dropout: 0.1
batch size: based on memory
epochs: small first pass, e.g. 5-20
```

Use a fixed train/validation split for all comparisons.

## Tokenization Strategies

### 1. Raw Patch Tokens

Split each waveform into equal contiguous patches.

Example:

```text
10,000 samples -> 100 patches
patch size = 100 samples
sequence length = 100 tokens
```

Each token is a raw waveform segment.

Suggested patch sizes:

```text
50 samples  -> 200 tokens
100 samples -> 100 tokens
200 samples -> 50 tokens
```

Implementation idea:

```python
patches = waveform.reshape(num_patches, patch_size)
tokens = Linear(patch_size, embed_dim)(patches)
```

Why test it:

- This is the direct 1D equivalent of Vision Transformer patching.
- It preserves local waveform shape while reducing sequence length.

Possible weakness:

- Patch size matters.
- Large patches may blur sharp pile-up details.
- Raw patches may let the model learn amplitude shortcuts.

### 2. Downsampled Point Tokens

Compress each waveform into fewer time points.

Example:

```text
10,000 samples -> 500 downsampled points
each point = one token
```

Downsampling methods to test:

```text
average pooling
max pooling
strided sampling
```

Implementation idea:

```python
downsampled = average_pool_1d(waveform, factor=20)
tokens = Linear(1, embed_dim)(downsampled[..., None])
```

Why test it:

- Keeps natural time order.
- Very simple.
- Reduces sequence length.

Possible weakness:

- May smooth away small secondary pulses.
- Strided sampling may miss localized pile-up artifacts.

### 3. Summary-Feature Patch Tokens

Split waveform into patches, then compute summary features per patch.

Example patch features:

```text
mean
max
min
standard deviation
integral / sum
slope
peak position within patch
```

Each token is a feature vector instead of raw samples.

Implementation idea:

```python
features = [
    patch.mean(),
    patch.max(),
    patch.min(),
    patch.std(),
    patch.sum(),
    patch[-1] - patch[0],
    patch.argmax() / patch_size,
]
tokens = Linear(num_features, embed_dim)(features)
```

Why test it:

- Adds physics-lite signal processing.
- Pile-up may appear as abnormal slopes, secondary peaks, broadening, or tail changes.
- More interpretable than raw patch tokens.

Possible weakness:

- Feature choices may throw away useful raw waveform shape.
- Less generic than raw patch tokenization.

### 4. Hybrid Tokens

Combine raw patch embeddings and summary-feature embeddings.

Example:

```text
raw patch -> raw embedding
summary features -> feature embedding
hybrid token = raw embedding + feature embedding
```

or:

```text
hybrid token = concat(raw embedding, feature embedding)
```

Implementation idea:

```python
raw_emb = Linear(patch_size, embed_dim)(raw_patch)
feat_emb = Linear(num_features, embed_dim)(patch_features)
tokens = raw_emb + feat_emb
```

Why test it:

- Combines raw waveform shape with physics-inspired summaries.
- Likely strong for scientific ML.
- Good benchmark-paper story: pure raw tokens vs physics-aware tokens vs hybrid tokens.

Possible weakness:

- More complex.
- Harder to identify which part drove improvement.
- More risk of overfitting.

## Positional Encoding Strategies

Transformers do not naturally know token order. Positional encodings tell the model where each token sits in time.

### 1. Learned Absolute Positional Encoding

Each token index gets a learned vector.

Example:

```text
token 0 -> learned position vector 0
token 1 -> learned position vector 1
...
```

Implementation idea:

```python
pos = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
tokens = tokens + pos
```

Why test it:

- Easiest first baseline.
- Flexible.
- Often works well.

Possible weakness:

- Tied to a fixed sequence length.
- Does not explicitly encode physical time.

### 2. Sinusoidal Positional Encoding

Use fixed sine/cosine encodings like the original Transformer.

Why test it:

- Waveforms are ordered time sequences.
- Sinusoidal encodings provide smooth position information.
- No learned position parameters.

Possible weakness:

- Less flexible than learned embeddings.
- Does not know CUORE-specific trigger structure.

### 3. Physics-Aware Positional Encoding

Use CUORE timing structure as additional position information.

Known CUORE structure:

```text
0-3 seconds: pretrigger baseline
around 3 seconds: trigger / pulse region
after 3 seconds: post-trigger pulse and tail
```

Possible position features per token:

```text
token_center_time_seconds
time_since_trigger = token_center_time_seconds - 3.0
is_pretrigger
is_posttrigger
normalized_position
```

Implementation idea:

```python
position_features = [
    token_center_time / 10.0,
    (token_center_time - 3.0) / 7.0,
    token_center_time < 3.0,
    token_center_time >= 3.0,
]
pos_emb = Linear(num_position_features, embed_dim)(position_features)
tokens = tokens + pos_emb
```

Why test it:

- Uses real detector structure.
- Pile-up before and after the trigger may have different physical meanings.
- Strong scientific-ML story.

Possible weakness:

- Dataset-specific.
- Less universal than learned/sinusoidal encodings.

## Additional Tokenizations To Try Later

These are not required for the first pass, but may be useful for a benchmark paper.

### 5. Multi-Scale Patch Tokens

Use multiple patch sizes.

Example:

```text
small patches capture sharp pulse details
large patches capture broad tail and baseline trends
```

This can help because pile-up can happen at different time scales.

### 6. Event-Centric Tokens

Detect important waveform regions first:

```text
baseline
rising edge
peak region
tail region
late-time region
```

Then tokenize those regions.

This is more physics-aware, but less generic.

### 7. Frequency / Spectral Tokens

Convert waveform patches into FFT or wavelet features.

Useful if pile-up creates frequency-domain differences, ringing, or high-frequency artifacts.

### 8. Peak / Shape Landmark Tokens

Detect landmarks:

```text
main peak
candidate secondary peak
rise start
tail slope
baseline window
```

Each landmark becomes a token.

This is very interpretable but more handcrafted.

## Additional Positional Encodings To Try Later

### 4. Relative Positional Encoding

Encode distance between tokens rather than only absolute token index.

Useful because waveform shape is local.

### 5. Rotary Positional Encoding / RoPE

Modern positional encoding used in many transformer models.

Potentially useful but not necessary for the first pass.

### 6. Local Attention Bias

Bias attention toward nearby patches.

Useful because adjacent waveform patches are more likely to be related.

### 7. Multi-Scale Position Encoding

If using multi-scale tokens, encode both:

```text
where the token is
what scale the token represents
```

## First Experiment Grid

Minimum viable grid:

| Dataset | Task | Tokenization | Positional Encoding | Metric |
|---|---|---|---|---|
| CUORE | pile-up classification | raw patches | learned absolute | ROC-AUC |
| CUORE | pile-up classification | raw patches | sinusoidal | ROC-AUC |
| CUORE | pile-up classification | summary-feature patches | learned absolute | ROC-AUC |
| CUORE | pile-up classification | hybrid patches | physics-aware time | ROC-AUC |

Preferred first grid:

```text
3 tokenizations x 3 positional encodings = 9 experiments
```

Tokenizations:

```text
raw patches
summary-feature patches
hybrid patches
```

Positional encodings:

```text
learned absolute
sinusoidal
physics-aware time
```

If time allows:

```text
4 tokenizations x 3 positional encodings = 12 experiments
```

Add:

```text
downsampled point tokens
```

## Implementation Requirements

Use the same:

```text
train/validation split
random seed
optimizer
number of epochs
batch size
embedding size
transformer depth
metric calculation
```

across experiments unless the change is explicitly part of the experiment.

Save results to a table:

```text
outputs/reports/cuore_transformer_tokenization_results.csv
```

Suggested columns:

```text
dataset
task
tokenization
patch_size
num_tokens
position_encoding
model_dim
num_layers
num_heads
validation_auc
validation_accuracy
validation_f1
notes
```

## Expected Deliverable

Produce:

1. A runnable training script or notebook
2. A small experiment table
3. A short written interpretation:

```text
Which tokenization worked best?
Which positional encoding worked best?
Did physics-aware tokenization help?
Did the transformer beat a simple baseline?
What should be tried next?
```

## Paper-Framing Sentence

"We evaluate how tokenization and positional encoding choices affect transformer performance on rare-event detector waveforms, using CUORE pile-up classification as an initial benchmark for unified transformer-style modeling across heterogeneous detector datasets."
