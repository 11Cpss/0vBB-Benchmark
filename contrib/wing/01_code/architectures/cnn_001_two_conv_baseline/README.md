# CNN-001: Two-Convolution Baseline

## Purpose

This folder defines the first paired NEXT architecture baseline. The same CNN
topology is trained independently for:

1. binary classification of `0nubb` signal versus Bi-214 background; and
2. regression of the event's summed deposited energy.

These are separate models, objectives, checkpoints, and training runs. This is
not a multitask network.

## Files

```text
cnn_001_two_conv_baseline/
├── train_classification.py
├── train_energy_regression.py
└── README.md
```

Both programs use the shared NEXT data reader and three detector-coordinate
projections:

- channel 0: XY;
- channel 1: XZ;
- channel 2: YZ.

## Network Architecture

The default input is `(batch, 3, 128, 128)`, and the default base channel count
is `C = 8`.

| Stage | Output shape | Trainable parameters |
|---|---:|---:|
| Input | `(B, 3, 128, 128)` | 0 |
| Conv2d, `3 -> C`, 3 x 3, padding 1 | `(B, C, 128, 128)` | `28C` |
| ReLU + MaxPool2d, 2 x 2 | `(B, C, 64, 64)` | 0 |
| Conv2d, `C -> 2C`, 3 x 3, padding 1 | `(B, 2C, 64, 64)` | `18C^2 + 2C` |
| ReLU + MaxPool2d, 2 x 2 | `(B, 2C, 32, 32)` | 0 |
| Adaptive average pooling, 1 x 1 | `(B, 2C, 1, 1)` | 0 |
| Flatten | `(B, 2C)` | 0 |
| Linear, `2C -> 1` | `(B,)` | `2C + 1` |

The total trainable parameter count is:

```text
N_params = 18C^2 + 32C + 1
```

For the default `C = 8`:

- first convolution: 224 parameters;
- second convolution: 1,168 parameters;
- scalar output head: 17 parameters;
- total: **1,409 trainable parameters**.

The classification and regression models each have one scalar head, so both
contain exactly 1,409 trainable parameters. Changing the image size does not
change this count because adaptive pooling produces a fixed-size feature
vector. Both training programs also verify the formula at runtime.

## Classification Task

`train_classification.py` predicts one logit per event:

- label 1: `0nubb`;
- label 0: Bi-214.

The objective is `BCEWithLogitsLoss`. Reported validation quantities include
BCE loss, accuracy, inclusive AUC, event count, and projection coverage. The
best checkpoint is selected by the highest validation AUC, with negative
validation loss used when AUC is unavailable.

Classification projections are normalized by each event's total deposited
energy and then scaled by 100. This emphasizes spatial topology while
suppressing direct use of total energy.

## Energy-Regression Task

`train_energy_regression.py` predicts summed voxel-deposited energy in MeV. The
target is:

```text
energy_target = sum(voxel_energy for all voxels in one event)
```

The sum is accumulated in float64 from HDF5 rows grouped by event ID.

Unlike classification, regression uses raw-energy projections with event-energy
normalization disabled. Dividing every image by its own total energy would
remove the main quantity the model must predict. The default raw-image scale is
40, which keeps the numerical input magnitude similar to the normalized
classification input.

The target is standardized before calculating the regression loss:

```text
z_target = (energy_target - training_mean) / training_std
```

The mean and population standard deviation are fitted using the selected
training files only. They are saved in the checkpoint and reused for validation;
validation and test events never contribute to these statistics. Predictions
are converted back to physical units with:

```text
energy_prediction = training_mean + training_std * z_prediction
```

The standalone regression loader uses every event in the selected training
files rather than class balancing, because class labels are not part of the
regression objective. The fitted normalizer and effective regression training
distribution therefore cover the same event set during uncapped training.
`--smoke` and `NEXT_MAX_TRAIN_BATCHES` intentionally train on an early subset
while retaining train-only normalizer statistics from all selected files. The
classification loader remains class-balanced for its binary objective.

Training uses `SmoothL1Loss(beta=1)` in standardized target space. Physical-unit
metrics are MAE, RMSE, bias, and R-squared. The best regression checkpoint is
selected by the lowest validation Smooth-L1 loss, while all reported energy
errors remain in MeV.

## Running

Activate the project environment first:

```bash
cd /home/wenyu/summer
source .venv/bin/activate
```

Run quick GPU workflow checks:

```bash
python 01_code/architectures/cnn_001_two_conv_baseline/train_classification.py --smoke

python 01_code/architectures/cnn_001_two_conv_baseline/train_energy_regression.py --smoke
```

Run full training:

```bash
python 01_code/architectures/cnn_001_two_conv_baseline/train_classification.py

python 01_code/architectures/cnn_001_two_conv_baseline/train_energy_regression.py
```

Load the default best checkpoint and run validation only:

```bash
python 01_code/architectures/cnn_001_two_conv_baseline/train_classification.py \
  --full-validation

python 01_code/architectures/cnn_001_two_conv_baseline/train_energy_regression.py \
  --full-validation
```

To validate a smoke checkpoint, include both `--smoke` and
`--full-validation`. Validation-only runs restore the checkpoint's data root,
batch size, worker count, validation-batch cap, projection, and AMP settings;
regression also restores its target normalizer and Smooth-L1 beta. CUDA is
mandatory; the programs do not fall back to CPU.

## Configuration

The configuration blocks near the top of each program are authoritative.
Common settings can be overridden with environment variables, including:

- `NEXT_DATA_ROOT`;
- `NEXT_BATCH_SIZE`;
- `NEXT_NUM_EPOCHS`;
- `NEXT_MAX_FILES_PER_CLASS`;
- `NEXT_NUM_WORKERS`;
- `NEXT_LEARNING_RATE`;
- `NEXT_WEIGHT_DECAY`;
- `NEXT_GRAD_CLIP`;
- `NEXT_BASE_CHANNELS`;
- `NEXT_SEED`;
- `NEXT_DETERMINISTIC`;
- `NEXT_USE_AMP`;
- `NEXT_AMP_PRECISION`;
- `NEXT_MAX_TRAIN_BATCHES`;
- `NEXT_MAX_VALIDATION_BATCHES`;
- `NEXT_MODEL_SUFFIX`;
- `NEXT_RUN_ID`.

The regression program additionally accepts `NEXT_SMOOTH_L1_BETA`.

Example:

```bash
NEXT_BATCH_SIZE=32 \
NEXT_NUM_EPOCHS=20 \
NEXT_MAX_FILES_PER_CLASS=200 \
python 01_code/architectures/cnn_001_two_conv_baseline/train_energy_regression.py
```

Set `NEXT_MAX_FILES_PER_CLASS=0` to remove the per-class file limit. Use
`--allow-existing` only when intentionally replacing best/last checkpoints for
an existing model suffix.

## Output Naming

The default artifact identities are:

- `cnn_001_two_conv_baseline_classification`;
- `cnn_001_two_conv_baseline_energy_regression`.

Checkpoints are written under `02_models/checkpoints/`:

```text
NEXTCNN_cnn_001_two_conv_baseline_classification_best.pt
NEXTCNN_cnn_001_two_conv_baseline_classification_last.pt
NEXTCNN_cnn_001_two_conv_baseline_energy_regression_best.pt
NEXTCNN_cnn_001_two_conv_baseline_energy_regression_last.pt
```

Training CSV/JSON logs are written under `03_training_runs/logs/`. History and
diagnostic figures are written under `03_training_runs/history_plots/`. A
timestamped run ID is included in log and figure names.

Existing checkpoints are preserved by default. If a requested name already
exists, `_run2`, `_run3`, and so forth are appended automatically.

## Caveats

- The two programs share a network topology but intentionally use different
  energy preprocessing.
- Classification and regression losses are on different scales and must not be
  compared directly.
- Architecture comparisons should keep data inventory, file-level splits,
  preprocessing, optimization settings, and evaluation protocols fixed within
  each task.
- Splits are assigned at the HDF5-file level to prevent events from the same
  source file appearing in multiple splits.
- Regression normalization is recomputed for the selected training files and
  stored with every run.
- `--smoke` checks execution only; its metrics and target statistics are not
  scientific results.
- The regression target is derived from the same voxel energies used to build
  the raw projections. With full projection coverage, their summed amplitudes
  are closely related by construction. This is a data-flow regression baseline,
  not independent evidence of experimental energy reconstruction.
- The target energy range is narrow, so R-squared can be unstable. Interpret it
  together with MAE, RMSE, and bias.
- The standalone regression checkpoint is distinct from the existing
  classification and multitask checkpoint contracts; formal EnergyBench
  inference requires a compatible regression adapter.
