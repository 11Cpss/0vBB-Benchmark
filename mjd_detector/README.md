# MJD Transformer Benchmark

This folder benchmarks Transformer representations for Majorana Demonstrator
(MJD) waveforms using the same collaborator-provided data split,
preprocessing, optimizer, checkpoint selection, and evaluation as the paired
1D CNN workflow.

The Transformer tokenizer runs inside the model, so CNN and Transformer models
receive the same preprocessed waveform tensor from `mjdbench`.

## Tasks and data protocol

Each event contains a 10,000-sample waveform.

- **Classification:** predict whether all four PSD indicators mark the event as
  clean.
- **Regression:** retain clean events only and predict event energy in keV.

The official `MJD_Train_*.hdf5` files form the development pool. A seeded
permutation (`seed=42`) assigns 90% to training and 10% to validation. The
official `MJD_Test_*.hdf5` files remain the untouched test set. This is not a
global 80/10/10 split.

For classification, the verified counts are:

| Partition | Events |
|---|---:|
| Train | 936,000 |
| Validation | 104,000 |
| Official test | 390,000 |

Regression counts are smaller because `mjdbench` filters to clean events
before constructing the development split.

Shared preprocessing subtracts the mean of the first 200 waveform samples.
Classification additionally divides each waveform by its maximum absolute
amplitude. Regression retains the configured physical scale.

## Folder layout

```text
mjd_detector/
├── mjdbench/                    # Shared data, CNN, training, and evaluation
├── mjd_transformer/
│   ├── tokenization.py
│   ├── positional_encoding.py
│   ├── model.py
│   └── tests/
└── notebooks/
    ├── mjd_transformer_train.ipynb
    └── mjd_transformer_results.ipynb
```

`mjdbench` is the collaborator-provided standardized workflow. Keep its data,
training, and evaluation behavior unchanged for direct CNN/Transformer
comparisons.

## Transformer representations

The official matrix uses 500 tokens per event so sequence length and attention
cost are comparable:

| Tokenization | One token represents | Content features |
|---|---|---|
| `raw_patches` | One non-overlapping 20-sample region | All 20 ordered amplitudes |
| `segment_summary` | The same 20-sample region | Segment mean and RMS |
| `pulse_entities` | One selected waveform time point | Raw amplitude and 9-sample local RMS |

`pulse_entities` selects 250 uniformly spaced positions and 250 positions with
the largest normalized amplitude/change importance score. Selection uses only
the input waveform, never labels or target energy, and selected positions are
returned to chronological order.

Each tokenization returns three coordinates:

```text
normalized time
normalized amplitude or region mean
local or first difference
```

The benchmark compares two coordinate encoders:

- `coordinate_mlp`;
- `fourier_coordinates`.

Content and coordinate vectors are independently projected to 64 dimensions,
added, normalized, processed by a two-layer four-head Transformer, and pooled
into one event representation. The shared task head returns one classification
logit or energy prediction per event.

The coordinate vectors include waveform-derived amplitude and change, so this
is a waveform-geometry encoding comparison rather than a pure time-only
positional-encoding comparison.

## Dataset layout

Raw data are not included. Place the official shards in one directory:

```text
MJD/
├── MJD_Train_*.hdf5
├── MJD_Test_*.hdf5
└── MJD_NPML_*.hdf5            # Optional inference files
```

Point the notebook to the data:

```bash
export MJD_BENCH_DATA=/absolute/path/to/MJD
```

## Run the official benchmark

From the repository root, install the package and start Jupyter:

```bash
python -m pip install -e . --no-deps
jupyter lab
```

Open and run:

```text
mjd_detector/notebooks/mjd_transformer_train.ipynb
```

The complete design contains 12 runs:

```text
3 tokenizations × 2 coordinate encodings × 2 tasks
```

The notebook seeds before every model construction, prepares each task's data
once, invokes the unchanged `train_model` and `evaluate_model`, and records
configuration, counts, parameters, epochs, runtime, validation selection, and
held-out test metrics.

## Assign runs to collaborators

By default the notebook runs all incomplete experiments. Assign a subset with
a comma-separated environment variable:

```bash
export MJD_RUN_IDS=classification__raw_patches__coordinate_mlp
```

Multiple runs are supported:

```bash
export MJD_RUN_IDS=classification__raw_patches__coordinate_mlp,regression__segment_summary__fourier_coordinates
```

Run IDs follow:

```text
<task>__<tokenization>__<position_encoding>
```

Useful environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `MJD_BENCH_DATA` | `data/MJD` | Raw MJD shard directory |
| `MJD_OUTPUT_ROOT` | `mjd_detector/results/transformer_official_v1` | Generated results |
| `MJD_RUN_IDS` | all 12 | Comma-separated run assignment |
| `MJD_TRANSFORMER_PROJECT_ROOT` | auto-detected | Explicit `mjd_detector` location |

## Outputs and restart behavior

Each completed run writes:

```text
results/transformer_official_v1/<run_id>/
├── run_config.json
├── best.pt
├── history.json
├── metrics.json
├── predictions.npz
└── run_summary.json
```

The root results directory also contains `transformer_results.csv`, updated
after each completed evaluation. A run is skipped only when all required
artifacts exist. An interrupted run starts again at epoch 1 because the shared
checkpoint does not save optimizer and scheduler state; the presence of a
partial folder alone does not mark it complete.

Open the results-only notebook at any time:

```text
mjd_detector/notebooks/mjd_transformer_results.ipynb
```

It reads saved JSON/CSV artifacts, ranks classification by test ROC-AUC, ranks
regression by test RMSE, reports epochs and minutes per epoch, and plots
validation curves. It does not load raw data or train a model.

## Tests

From the repository root:

```bash
python -m unittest discover -s mjd_detector/mjd_transformer/tests -v
python -m compileall -q mjd_detector
```

The tests cover all three tokenizers, both coordinate encodings, both tasks,
deterministic entity selection, raw-patch preservation, forward/backward
passes, and the shared training/evaluation contract using synthetic data.

## Scientific comparison rules

- Use the same `DataConfig` and `TrainingConfig` for directly comparable runs.
- Validation selects the checkpoint; only the untouched official test set is
  used for final reporting.
- Raw patches use a larger content projection (`20 → 64`) than the two-feature
  strategies, so parameter counts must be reported.
- Do not describe `pulse_entities` as reconstructed physics pulses; it is a
  deterministic hybrid time-point selector.
- Treat single-seed results as one benchmark realization, not an uncertainty
  estimate.
