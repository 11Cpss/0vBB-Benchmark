# CNN-003: Residual Spatial CNN

## Purpose

CNN-003 replaces the shallow two-convolution backbone used by CNN-001 and
CNN-002. Classification and energy regression are separate tasks that share
only the residual convolutional core.

The directory contains two independent training programs:

```text
cnn_003_residual_spatial/
├── train_classification.py
├── train_energy_regression.py
└── README.md
```

Classification uses `ResidualSpatialNextCNN`; regression uses the distinct
`ResidualSpatialEnergyRegressor`. They have separate inputs, output heads,
losses, batch contracts, checkpoint identities, logs, and validation metrics.

## What Changes from CNN-002

CNN-002 used two 3 x 3 convolutions followed by global average pooling. Its
last convolutional cells had an effective receptive field of about 10 x 10,
and global averaging discarded their spatial arrangement.

CNN-003 makes four structural changes:

1. eight residual blocks replace the two plain convolutions;
2. three learned stride-2 transitions grow the receptive field to about
   109 x 109 pixels;
3. average pooling retains a 4 x 4 spatial feature grid instead of reducing
   each channel to one number; and
4. the output head also receives 15 explicit global geometry features: the
   log mass, x/y centroid, and x/y variance of each of the three views.

This addresses the spatial bottleneck diagnosed in CNN-002. The dedicated
regressor additionally receives energy-preserving projections and uses their
per-view sum as a physical baseline for the learned CNN residual.

## Architecture

The classification defaults remain `C = 16`, pooled size `P = 4`, and head
width `H = 256`. Energy regression uses `C = 4`, `P = 1`, and `H = 32`. The
shared convolutional shape is:

| Stage | Output shape |
|---|---:|
| Input | `(B, 3, 128, 128)` |
| 3 x 3 stem + GroupNorm + SiLU | `(B, C, 128, 128)` |
| 2 residual blocks | `(B, C, 128, 128)` |
| stride-2 residual block + residual block | `(B, 2C, 64, 64)` |
| stride-2 residual block + residual block | `(B, 4C, 32, 32)` |
| stride-2 residual block + residual block | `(B, 8C, 16, 16)` |
| deterministic average pooling | `(B, 8C, P, P)` |
| flatten + 15 global geometry features | `(B, 8C P^2 + 15)` |
| linear + SiLU | `(B, H)` |
| task output | `(B,)` |

Every residual block contains two 3 x 3 convolutions. Downsampling blocks use
a learned 1 x 1 projection on the skip path. GroupNorm is used instead of
BatchNorm so behavior does not depend on the batch composition.

The classification defaults have **1,228,817 trainable parameters**.  The
compact energy-regression defaults have **45,861 trainable parameters**,
compared with 1,409 parameters in the original CNN-002 regressor
and 75,777 in its width-only large variant.

Architecture dimensions can be overridden with:

- `NEXT_BASE_CHANNELS` (default `16`);
- `NEXT_POOLED_SIZE` (default `4`); and
- `NEXT_HEAD_FEATURES` (default `256`).

The fixed 128 x 128 projection becomes a 16 x 16 encoded map after the three
downsampling stages, so `NEXT_POOLED_SIZE` must be a divisor of 16. Ordinary
average pooling is used instead of adaptive pooling because its CUDA backward
path supports the project's strict deterministic mode.

## Classification Task

Classification deliberately keeps the CNN-002 task protocol unchanged:

- input: per-event energy-normalized XY/XZ/YZ projections scaled by 100;
- target: `0nubb` signal versus Bi-214 background;
- loss: `BCEWithLogitsLoss`;
- balanced class iteration during training;
- checkpoint selection: highest validation AUC, with negative validation loss
  used only if AUC is unavailable.

The old global-energy skip is not used. At full projection coverage, each
normalized view has the same sum, so that path does not provide discriminating
energy information.

## Energy-Regression Task

Regression has a task-specific protocol:

- model: `ResidualSpatialEnergyRegressor`, never the classification model;
- input: unnormalised energy-weighted XY/XZ/YZ projections scaled by 100;
- target: float64 sum of voxel-deposited energy in MeV;
- batch contract: no classification label or category is exposed;
- target transform: mean and population standard deviation fitted on train;
- objective: MSE in standardized target space;
- metrics: objective loss, Smooth-L1, physical-unit MAE/RMSE/bias/R-squared,
  prediction spread, and Pearson correlation;
- checkpoint selection: lowest validation RMSE, with validation MAE as an
  exact-tie fallback;
- early stopping: 12 epochs without an RMSE improvement larger than
  `1e-6 MeV`;
- baselines: the train-target mean and an ordinary least-squares model over the
  same 15 global geometry features;
- training order: every selected event is retained and mixed through a
  deterministic 512-event shuffle buffer.

The learned regression residual is initialized to zero. Epoch 0 therefore
predicts the energy-preserving mean of the three projection sums; the CNN can
learn corrections when projection coverage or detector response is imperfect.

In the current Monte Carlo data, the regression target is derived from the same
voxel energies used to construct the input and projection coverage is complete.
Near-perfect performance is consequently a data-flow reconstruction baseline,
not independent evidence of experimental energy resolution.

## Running

```bash
cd /home/wenyu/summer
source .venv/bin/activate

python 01_code/architectures/cnn_003_residual_spatial/train_classification.py --smoke
python 01_code/architectures/cnn_003_residual_spatial/train_energy_regression.py --smoke

python 01_code/architectures/cnn_003_residual_spatial/train_classification.py
python 01_code/architectures/cnn_003_residual_spatial/train_energy_regression.py
```

Validation-only runs use `--full-validation`. Existing checkpoints are
preserved automatically with `_run2`, `_run3`, and later suffixes. The default
artifact identities are:

- `cnn_003_residual_spatial_classification`; and
- `cnn_003_residual_spatial_energy_regression`.

The remaining `NEXT_*` environment overrides, CUDA-only execution, deterministic
mode, AMP behavior, data split, output directories, and logging format are the
same as CNN-002.

Regression additionally accepts:

- `NEXT_REGRESSION_LOSS` (`mse` by default; legacy `smooth_l1` is supported);
- `NEXT_EVENT_SHUFFLE_BUFFER_SIZE` (default `512`);
- `NEXT_EARLY_STOPPING_PATIENCE` (default `12`); and
- `NEXT_EARLY_STOPPING_MIN_DELTA` (default `1e-6 MeV`).
