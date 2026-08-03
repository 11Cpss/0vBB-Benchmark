# CNN-002: Global Energy Skip

## Purpose

CNN-002 keeps the two-convolution topology used by CNN-001. Classification
retains the explicit FP32 global-energy path. Energy regression now uses a
topology-only configuration so the deposited-energy target cannot be recovered
by directly summing input pixels. The two programs train independently for:

1. binary classification of `0nubb` signal versus Bi-214 background; and
2. regression of the event's summed deposited energy.

The tasks have separate objectives, checkpoints, and training runs. CNN-001 is
left unchanged so its completed runs remain reproducible.

## Files

```text
cnn_002_global_energy_skip/
├── train_classification.py
├── train_energy_regression.py
├── train_energy_regression_large.py
└── README.md
```

Classification uses `GlobalEnergySkipCNN`. Regression uses
`SimpleNextEnergyRegressor` with binary XY/XZ/YZ voxel-occupancy projections.

## Network Architectures

The default input is `(batch, 3, 128, 128)`, and the default base channel count
is `C = 8`.

The shared convolutional trunk is:

| Stage | Output shape | Trainable parameters |
|---|---:|---:|
| Input | `(B, 3, 128, 128)` | 0 |
| Conv2d, `3 -> C`, 3 x 3, padding 1 | `(B, C, 128, 128)` | `28C` |
| ReLU + MaxPool2d, 2 x 2 | `(B, C, 64, 64)` | 0 |
| Conv2d, `C -> 2C`, 3 x 3, padding 1 | `(B, 2C, 64, 64)` | `18C^2 + 2C` |
| ReLU + MaxPool2d, 2 x 2 | `(B, 2C, 32, 32)` | 0 |
| Adaptive average pooling + flatten | `(B, 2C)` | 0 |

Classification appends the following global-energy correction path:

| Stage | Output shape | Trainable parameters |
|---|---:|---:|
| FP32 sums of the XY, XZ, YZ views | `(B, 3)` | 0 |
| Concatenate topology and global features | `(B, 2C + 3)` | 0 |
| Linear residual correction, `2C + 3 -> 1` | `(B,)` | `2C + 4` |
| Mean standardized view energy + correction | `(B,)` | 0 |

For view `v`, the global feature is:

```text
g_v = (sum(image_v) / input_scale - global_center) / global_scale
output = mean(g_XY, g_XZ, g_YZ) + correction(topology, g_XY, g_XZ, g_YZ)
```

The classification correction head is initialized to zero. This model has
`18C^2 + 32C + 4` parameters, or 1,412 for `C = 8`.

Topology-only regression appends a `2C -> 1` linear head to the convolutional
trunk. It has:

```text
N_params = 18C^2 + 32C + 1
```

For `C = 8`, this is **1,409 trainable parameters**. It has no global sum or
energy-preserving skip.

### Enlarged Energy-Regressor Variant

`train_energy_regression_large.py` is an enlarged-width control experiment.
It uses exactly the same `SimpleNextEnergyRegressor` architecture, binary
occupancy input, data split, target normalization, loss, optimizer, training
schedule, and evaluation logic as `train_energy_regression.py`. The only model
change is the default base channel count: `C = 64` instead of `C = 8`.

This widens the two convolutional layers from `3 -> 8 -> 16` to
`3 -> 64 -> 128`, increasing the trainable parameter count from **1,409** to
**75,777** (about 53.8 times as many). `NEXT_BASE_CHANNELS` remains available
as an explicit override. The enlarged program uses the separate default
artifact identity `cnn_002_topology_only_energy_regression_large`, so it does
not overwrite the original regression checkpoints.

## Classification Task

Classification uses per-event energy-normalized projections scaled by 100. At
full coverage, each view sum divided by 100 is 1, so `global_center=1` and
`global_scale=1` make the direct skip exactly zero. The learned output is then a
binary logit trained with `BCEWithLogitsLoss`; the global features remain
available to the correction head for coverage diagnostics.

The best checkpoint is selected by validation AUC, with negative validation
loss used only when AUC is unavailable.

## Energy-Regression Task

Regression uses binary occupancy projections: each in-range voxel marks its
XY, XZ, and YZ pixels as 1, independently of deposited-energy amplitude. A
standard two-convolution regressor predicts standardized energy from topology;
the target mean and population standard deviation are fitted on train only.

The target and physical-unit metrics are unchanged from CNN-001:

- target: float64 sum of voxel-deposited energy for one event, in MeV;
- loss: `SmoothL1Loss(beta=1)` in standardized target space;
- metrics: MAE, RMSE, bias, and R-squared in physical units;
- checkpoint selection: lowest validation Smooth-L1 loss.

The model input therefore contains coordinates/topology but not the MC energy
amplitudes used to define the target.

## Running

```bash
cd /home/wenyu/summer
source .venv/bin/activate

python 01_code/architectures/cnn_002_global_energy_skip/train_classification.py --smoke
python 01_code/architectures/cnn_002_global_energy_skip/train_energy_regression.py --smoke
python 01_code/architectures/cnn_002_global_energy_skip/train_energy_regression_large.py --smoke

python 01_code/architectures/cnn_002_global_energy_skip/train_classification.py
python 01_code/architectures/cnn_002_global_energy_skip/train_energy_regression.py
python 01_code/architectures/cnn_002_global_energy_skip/train_energy_regression_large.py
```

Validation-only runs use `--full-validation`. To validate a smoke checkpoint,
combine `--smoke --full-validation`.

The default artifact identities are:

- `cnn_002_global_energy_skip_classification`;
- `cnn_002_topology_only_energy_regression`; and
- `cnn_002_topology_only_energy_regression_large`.

Existing checkpoints are preserved automatically by appending `_run2`,
`_run3`, and so on. Common `NEXT_*` environment overrides are the same as in
CNN-001; regression additionally accepts `NEXT_SMOOTH_L1_BETA`.

## Interpretation Caveat

The topology-only regression is intentionally harder than the old raw-energy
sum baseline. Good performance must come from correlations between track
geometry/occupancy and deposited energy. It is still not a full detector
reconstruction study unless the input is replaced by realistic detector
response rather than MC voxel coordinates.
