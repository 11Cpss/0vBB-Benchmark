# CNN-006: Dense-Voxel 3-D Residual CNN

[中文](README.md) | English

## 1. Model Positioning

CNN-006 does not generate 2-D projections. Instead, it places event-centered energy
depositions into a fixed 3-D voxel tensor and uses 3-D convolution directly to learn
track connectivity, endpoint blobs, and multi-site structure.

> **Naming boundary: “Dense” means that the input is a densely allocated
> `(C,D,H,W)` voxel tensor. The model is not DenseNet and does not use DenseNet-style
> dense connectivity.**

| Item | Description |
|---|---|
| architecture ID | `cnn_006_dense_3d_resnet` |
| checkpoint model name | `Dense3DResidualCNN` |
| model family | dense-voxel 3-D residual CNN |
| registry input kind | `dense3d` |
| task | binary classification of `0νββ` signal versus `Bi214` background |
| output | one unnormalized signal logit per event, shape `(B,)` |
| default trainable parameters | **688,433** |
| central comparison question | whether preserving native 3-D adjacency outperforms 2-D projections and whether the cost of dense 3-D computation is justified |

The implementation is
[`Dense3DResidualCNN`](../../../src/next_alt/models/cnn.py). This directory contains
the model-specific configuration, training entry point, and documentation.

## 2. Exact Input and Preprocessing

The HDF5 reader extracts event-level `x,y,z,energy` from `/MC/hits/table`. The
event representation is constructed in this order:

1. validate the coordinates, energy values, and total energy of the complete event;
2. compute the energy-weighted 3-D centroid of the original input rows and subtract it
   from all coordinates;
3. aggregate input rows within the same voxel using 15 mm cells;
4. recenter the quantized voxel centers once more using voxel energy, removing the
   half-bin quantization offset while preserving relative distances;
5. place the centered voxels into a `96x96x96` cube;
6. discard voxels outside the cube without renormalizing the retained energy.

The final batch key is `volume`, with shape `(B,2,96,96,96)`. PyTorch spatial
dimensions follow `(D,H,W)=(z,y,x)`:

| Channel | Value |
|---:|---|
| 0 | deposited voxel energy / total energy of the complete event |
| 1 | `log1p(number of original input rows in the voxel)` |

With 15 mm cells and 96 bins, the centered cube spans 1,440 mm on each axis.
`representation_coverage` is the sum of the energy fraction inside the cube. It may
be below 1, and the model does not conceal cropping through renormalization. This
representation does not use `max_points`, although the fixed cube itself imposes
spatial cropping.

## 3. Network Architecture and Tensor Shapes

### 3.1 3-D Stem and Residual Stages

| Layer/stage | Operation | Output shape, excluding batch |
|---|---|---|
| input | two-channel dense voxel volume | `(2,96,96,96)` |
| stem | `Conv3d(2,12,5x5x5,stride=2,pad=2)` + GroupNorm + SiLU | `(12,48,48,48)` |
| stage 0 | one 3-D residual block, width 12 | `(12,48,48,48)` |
| stage 1 | two blocks, first block stride 2, `12 -> 24` | `(24,24,24,24)` |
| stage 2 | two blocks, first block stride 2, `24 -> 48` | `(48,12,12,12)` |
| stage 3 | one block, stride 2, `48 -> 96` | `(96,6,6,6)` |

The main path of each 3-D residual block is
`3x3x3 Conv -> GroupNorm -> SiLU -> 3x3x3 Conv -> GroupNorm`. When the channel count
or stride changes, the skip path uses `1x1x1 Conv + GroupNorm`; SiLU is applied
after the main and skip paths are added. The stem first downsamples `96^3` to
`48^3`, which is essential for controlling activation memory.

### 3.2 Event Pooling and Classification Head

The final `(96,6,6,6)` feature volume is pooled independently over the three spatial
dimensions using:

- global mean: 96 dimensions;
- global max: 96 dimensions.

The two vectors are concatenated into 192 dimensions, followed by
`Linear(192,128) -> SiLU -> Dropout(0.1) -> Linear(128,1)`, which outputs
`(B,)` logits.

## 4. Relationship to Reference Methods and Scope Boundaries

This implementation applies 3-D convolution and residual shortcuts to static detector
voxels. The citations document the technical origins and do not imply a reproduction
of video models or the original experiments.

- [Hara, Kataoka and Satoh, *Learning Spatio-Temporal Features with 3D Residual Networks for Action Recognition*, ICCV Workshops 2017](https://arxiv.org/abs/1708.07632),
  arXiv:1708.07632, DOI
  [10.1109/ICCVW.2017.373](https://doi.org/10.1109/ICCVW.2017.373). The common element
  is the use of 3-D residual blocks. In the original paper, the third dimension
  includes time; all three dimensions here are detector-space axes.
- [Tran et al., *Learning Spatiotemporal Features with 3D Convolutional Networks*, ICCV 2015](https://arxiv.org/abs/1412.0767),
  arXiv:1412.0767. This is background for 3-D ConvNets. The current network is not
  C3D; its depth, pooling, normalization, and task all differ.
- [He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016](https://arxiv.org/abs/1512.03385),
  arXiv:1512.03385, DOI
  [10.1109/CVPR.2016.90](https://doi.org/10.1109/CVPR.2016.90).
- [Wu and He, *Group Normalization*, ECCV 2018](https://arxiv.org/abs/1803.08494),
  arXiv:1803.08494, DOI
  [10.1007/978-3-030-01261-8_1](https://doi.org/10.1007/978-3-030-01261-8_1).
  GroupNorm is especially appropriate for the memory-constrained default batch size
  of 2 used here.

## 5. Key Configuration

See [`config.yaml`](config.yaml) for the authoritative configuration.

| Category | Parameter | Default |
|---|---|---:|
| representation | dense grid / bin | `96 / 15 mm` |
| representation | input channels | `energy_fraction, log1p(hit_count)` |
| model | base channels | `12` |
| model | stage blocks | `[1,2,2,1]` |
| model | head features | `128` |
| model | dropout | `0.1` |
| data | max files per class | `100` |
| data | split seed / fractions | `42 / [0.8,0.1,0.1]` |
| data | balanced training / shuffle buffer | `true / 512` |
| training | batch size / epochs | `2 / 50` |
| training | learning rate / weight decay | `1e-3 / 1e-4` |
| training | gradient clip norm | `1.0` |
| training | early-stop patience | `12` |
| training | AMP / deterministic | `auto / false` |

## 6. Shared Training Procedure and Commands

The model entry point invokes the unified trainer, which uses
`BCEWithLogitsLoss`, AdamW, `CosineAnnealingLR(T_max=50)`, gradient clipping, and
CUDA AMP. The best checkpoint is selected by validation AUC, and training stops after
12 consecutive epochs fail to exceed the current best. The training stream alternates
signal and background, while validation preserves the original distribution and a
deterministic order. The current `balance_training_classes=true` branch does not
perform event-buffer shuffling: `512` is a retained configuration value, while only
source-file order is shuffled in practice at each epoch.

Official configuration entry point:

```bash
cd /home/wenyu/summer
source .venv/bin/activate
python 01_code/architectures/cnn_006_dense_3d_resnet/train_classification.py
```

A custom YAML may also be supplied. The official best/last checkpoints, history
CSV/JSON, and training plot already exist, and `allow_overwrite: false` is the
default. Direct execution will reject the rerun before allocating the GPU or starting
training; it will neither resume nor overwrite. New experiments should use a copied
YAML with separate checkpoint, log, and plot directories.

## 7. Completed Experiment Results

| Metric | Result |
|---|---:|
| epochs actually trained | 22 |
| best epoch | **10** |
| best validation AUC | **0.929562** |
| full-test files / events | `1,490 / 115,499` |
| full-test rank | **7 / 10** |
| matched AUC | **0.928210** |
| inclusive AUC | **0.928082** |
| energy independence | **0.977320** |

The official full-test evaluation is strict, reports `comparable=True`, and has zero
warnings and errors. This configuration preserves 3-D connectivity, but inference is
substantially slower than with the lighter projection, point-cloud, and graph models,
and it did not achieve a higher AUC in the current experiment.

Use a new `_rerun` output directory for reevaluation:

```bash
cd /home/wenyu/summer
source .venv/bin/activate
energybench next \
  02_models/checkpoints/NEXTALT_cnn_006_dense_3d_resnet_classification_best.pt \
  --model-id cnn_006_dense_3d_resnet \
  --manifest manifests/next_0nubb_vs_bi214.yaml \
  --split test \
  --device cuda:0 \
  --batch-size 2 \
  --num-workers 0 \
  --output-dir 04_evaluations/NEXTALT_cnn_006_dense_3d_resnet_test_rerun
```

The command omits `--max-files-per-class`, so it uses the full test split. The
output directory must not exist beforehand.

## 8. Checkpoints, Training History, and Evaluation Artifacts

- [best checkpoint](../../../02_models/checkpoints/NEXTALT_cnn_006_dense_3d_resnet_classification_best.pt)
- [last checkpoint](../../../02_models/checkpoints/NEXTALT_cnn_006_dense_3d_resnet_classification_last.pt)
- [epoch CSV](../../../03_training_runs/logs/NEXTALT_cnn_006_dense_3d_resnet_classification_epochs.csv)
- [complete history JSON](../../../03_training_runs/logs/NEXTALT_cnn_006_dense_3d_resnet_classification_history.json)
- [training-history plot](../../../03_training_runs/history_plots/NEXTALT_cnn_006_dense_3d_resnet_classification_history.png)
- [test predictions](../../../04_evaluations/NEXTALT_cnn_006_dense_3d_resnet_test/predictions_test.npz)
- [test results.csv](../../../04_evaluations/NEXTALT_cnn_006_dense_3d_resnet_test/evaluation_test/results.csv)
- [strict metrics.json](../../../04_evaluations/NEXTALT_cnn_006_dense_3d_resnet_test/evaluation_test/.energybench/metrics.json)
- [energy-matched ROC](../../../04_evaluations/NEXTALT_cnn_006_dense_3d_resnet_test/evaluation_test/energy_matched_roc.png)
- [score-energy dependence](../../../04_evaluations/NEXTALT_cnn_006_dense_3d_resnet_test/evaluation_test/score_energy_dependence.png)
- [official ten-model leaderboard](../../../04_evaluations/NEXTALT_all_models_comparison/leaderboard.csv)

## 9. Related Local Files

| File | Purpose |
|---|---|
| [config.yaml](config.yaml) | dense representation, network, and training hyperparameters |
| [train_classification.py](train_classification.py) | model-specific training entry point |
| [src/next_alt/models/cnn.py](../../../src/next_alt/models/cnn.py) | 3-D residual block and `Dense3DResidualCNN` |
| [src/next_alt/data.py](../../../src/next_alt/data.py) | centered voxelization, dense cube construction, and coverage |
| [src/next_cnn/data.py](../../../src/next_cnn/data.py) | HDF5 reading and file-level split |
| [src/next_alt/registry.py](../../../src/next_alt/registry.py) | `dense3d` input-kind registration |
| [src/next_alt/training.py](../../../src/next_alt/training.py) | unified loss, optimizer, scheduler, and early stopping |
| [src/next_alt/checkpoint.py](../../../src/next_alt/checkpoint.py) | format-v3 checkpoint and provenance |
| [src/next_alt/adapter.py](../../../src/next_alt/adapter.py) | inference adaptation for the dense volume |
| [evaluation manifest](../../../manifests/next_0nubb_vs_bi214.yaml) | strict NEXT evaluation contract |
| [complete comparison results](../../../docs/ALTERNATIVE_EVALUATION_RESULTS.md) | ten-model ranking and evaluation conclusions |
| [usage guide](../../../docs/USAGE_GUIDE.md) | EnergyBench commands and result directories |

## 10. Limitations and Interpretation Boundaries

- Most voxels in the dense `96^3` tensor are usually zero, yet 3-D convolution still
  computes over every position. Memory and compute utilization are lower than with
  sparse convolution or graph methods.
- The fixed centered cube discards out-of-range energy. Coverage records that loss,
  but the model does not receive coverage explicitly as an input feature.
- Event centering removes absolute detector position. This helps the model focus on
  relative topology but also discards potentially useful positional information.
- A 15 mm voxelization merges finer-scale structure. Increasing the grid size or
  reducing the bin size rapidly increases activation cost cubically.
- Global mean/max pooling compresses the final `6^3` feature map into one event
  vector and cannot directly provide interpretable local positions.
- This model is neither DenseNet nor a sparse CNN; its name should not be taken to
  imply dense skip connections.
- The current checkpoint is trained for classification only. A `not applicable`
  result for energy regression is expected.
