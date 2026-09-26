# CNN-005: Global and Local Dual-Scale Projection CNN

[中文](README.md) | English

## 1. Model Positioning

CNN-005 jointly observes detector-wide 30 mm coarse projections and event-centered
15 mm fine projections to test whether global track geometry and local topology such
as endpoint blobs provide complementary information. Both scales use the same
residual encoder, so the model capacity is not simply doubled by the two branches.

| Item | Description |
|---|---|
| architecture ID | `cnn_005_multiscale_projection` |
| checkpoint model name | `MultiScaleProjectionCNN` |
| model family | shared-weight multi-scale 2-D residual CNN |
| registry input kind | `multiscale2d` |
| task | binary classification of `0νββ` signal versus `Bi214` background |
| output | one unnormalized signal logit per event, shape `(B,)` |
| default trainable parameters | **865,939** |
| core processing strategy | **plane early fusion within each scale; scale late fusion between the two scales** |

The model class is implemented centrally as
[`MultiScaleProjectionCNN`](../../../src/next_alt/models/cnn.py). This directory
contains only its YAML configuration, training entry point, and documentation.

## 2. Exact Input and Preprocessing

The shared reader obtains `x,y,z,energy` from HDF5 `/MC/hits/table` and performs a
deterministic split over complete files. Each event produces the following two
representations, both with shape `(3,128,128)`.

### 2.1 Coarse Detector-Wide Projection

- batch key: `projections`, shape `(B,3,128,128)`;
- plane order: XY, XZ, YZ;
- 30 mm/bin;
- fixed detector coordinates;
- origin `[-1920,-1920,-120]` mm;
- 128 bins per plane.

### 2.2 Fine Event-Centered Projection

- batch key: `fine_projections`, shape `(B,3,128,128)`;
- first compute the energy-weighted centroid from the original input rows and subtract
  it from all coordinates;
- 15 mm/bin;
- the projection cube has origin `-960 mm` on all three axes and covers approximately
  `[-960,+960) mm`;
- plane order is again XY, XZ, YZ.

Both projection sets retain only input rows that fall inside their respective 3-D
ranges. Their values are divided by the **total energy of the complete event** and
then multiplied by `projection_input_scale=100`. Out-of-range energy is not
redistributed; coarse and fine coverage are recorded separately in the batch and
provenance.

Two different fusion levels must be distinguished:

- **plane early fusion:** within each scale, XY/XZ/YZ are three input channels, and
  the first `Conv2d` already mixes the three planes;
- **scale late fusion:** coarse and fine inputs pass independently through the full
  encoder and are fused only after event-level vectors have been produced. Both
  forward passes call the same encoder instance and share all parameters.

## 3. Network Architecture and Tensor Shapes

### 3.1 Residual Encoder Shared by Both Scales

The table applies independently to both the coarse and fine branches:

| Layer/stage | Operation | Single-scale output shape |
|---|---|---|
| input | three orthogonal projection channels | `(3,128,128)` |
| stem | `Conv2d(3,16,5x5,stride=2,pad=2)` + GroupNorm + SiLU | `(16,64,64)` |
| stage 0 | two residual blocks, width 16 | `(16,64,64)` |
| stage 1 | two blocks, first block stride 2, `16 -> 32` | `(32,32,32)` |
| stage 2 | two blocks, first block stride 2, `32 -> 64` | `(64,16,16)` |
| stage 3 | two blocks, first block stride 2, `64 -> 128` | `(128,8,8)` |
| scale pooling | spatial mean | `(128,)` |

The main residual-block path is
`3x3 Conv -> GroupNorm -> SiLU -> 3x3 Conv -> GroupNorm`. When downsampling or a
channel change is required, the skip path uses `1x1 Conv + GroupNorm`. SiLU is
applied after the two paths are added.

### 3.2 Scale Gate and Explicit Interaction Features

After the coarse and fine branches each produce a 128-dimensional vector:

1. stack them into `(B,2,128)` and add the learned `scale_identity`;
2. concatenate coarse and fine into 256 dimensions;
3. apply the gate:
   `Linear(256,128) -> SiLU -> Linear(128,2) -> softmax`;
4. obtain `coarse_weighted` and `fine_weighted`, each 128-dimensional;
5. also compute `abs(coarse-fine)` and `coarse*fine`, each 128-dimensional;
6. concatenate the four feature groups into `128*4=512` dimensions;
7. use the classification head:
   `Linear(512,256) -> SiLU -> Dropout(0.1) -> Linear(256,1)`;
8. squeeze the result to `(B,)` logits.

## 4. Relationship to Reference Methods and Scope Boundaries

This implementation is a custom dual-scale input model for NEXT, not a layer-by-layer
reproduction of any paper architecture.

- [He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016](https://arxiv.org/abs/1512.03385),
  arXiv:1512.03385, DOI
  [10.1109/CVPR.2016.90](https://doi.org/10.1109/CVPR.2016.90). Only the basic
  residual-shortcut idea is adopted here.
- [Bromley et al., *Signature Verification using a “Siamese” Time Delay Neural Network*, NeurIPS 1993](https://proceedings.neurips.cc/paper/1993/hash/288cc0ff022877bd3df94bc9360b9c5d-Abstract.html).
  This work is a historical reference for shared-weight dual branches. The current
  model is not a metric-learning Siamese network, and its two branch inputs are two
  scales of the same event rather than two samples to be compared.
- [Lin et al., *Feature Pyramid Networks for Object Detection*, CVPR 2017](https://arxiv.org/abs/1612.03144),
  arXiv:1612.03144, DOI
  [10.1109/CVPR.2017.106](https://doi.org/10.1109/CVPR.2017.106). FPN is only a
  motivation for multi-scale representation. This model has no top-down pathway,
  lateral connections, or pyramid detection heads and therefore should not be called
  an FPN.
- [Wu and He, *Group Normalization*, ECCV 2018](https://arxiv.org/abs/1803.08494),
  arXiv:1803.08494, DOI
  [10.1007/978-3-030-01261-8_1](https://doi.org/10.1007/978-3-030-01261-8_1).

## 5. Key Configuration

See [`config.yaml`](config.yaml) for the authoritative configuration.

| Category | Parameter | Default |
|---|---|---:|
| coarse representation | grid / bin / input scale | `128 / 30 mm / 100` |
| fine representation | grid / bin | `128 / 15 mm` |
| model | base channels | `16` |
| model | stage blocks | `[2,2,2,2]` |
| model | fusion features | `256` |
| model | dropout | `0.1` |
| data | max files per class | `100` |
| data | split seed / fractions | `42 / [0.8,0.1,0.1]` |
| data | balanced training / shuffle buffer | `true / 512` |
| training | batch size / epochs | `8 / 50` |
| training | learning rate / weight decay | `1e-3 / 1e-4` |
| training | gradient clip norm | `1.0` |
| training | early-stop patience | `12` |
| training | AMP / deterministic | `auto / false` |

## 6. Shared Training Procedure and Commands

The model entry point passes the architecture ID and YAML to the unified trainer. The
trainer uses `BCEWithLogitsLoss`, AdamW, `CosineAnnealingLR(T_max=50)`, gradient
clipping, and CUDA AMP. It computes AUC over the complete validation stream after each
epoch, writes the best checkpoint according to validation AUC, and uses early
stopping with a patience of 12. The training stream alternates between the two classes
to remain balanced, while validation retains the original distribution and a
deterministic order. The current `balance_training_classes=true` branch does not
perform event-buffer shuffling: `512` is a retained configuration value, while only
source-file order is shuffled in practice at each epoch.

Official configuration entry point:

```bash
cd /home/wenyu/summer
source .venv/bin/activate
python 01_code/architectures/cnn_005_multiscale_projection/train_classification.py
```

A custom YAML may be supplied as the sole positional argument. The official
checkpoints, history CSV/JSON, and history plot already exist. Because
`allow_overwrite: false` is the default, direct execution will reject the rerun
before training starts and will not resume or overwrite automatically. For a new
experiment, copy the YAML and specify entirely new checkpoint, log, and plot output
directories.

## 7. Completed Experiment Results

| Metric | Result |
|---|---:|
| epochs actually trained | 19 |
| best epoch | **7** |
| best validation AUC | **0.949000** |
| full-test files / events | `1,490 / 115,499` |
| full-test rank | **6 / 10** |
| matched AUC | **0.947787** |
| inclusive AUC | **0.948255** |
| energy independence | **0.980444** |

Among the ten models, this model has the highest energy-independence score, but its
discrimination is below the Multi-view CNN, GINE, ParticleNet, GravNet, and
PointNet++. The official evaluation is strict, reports `comparable=True`, and has
zero warnings and errors.

Use a new directory when rerunning the complete test evaluation:

```bash
cd /home/wenyu/summer
source .venv/bin/activate
energybench next \
  02_models/checkpoints/NEXTALT_cnn_005_multiscale_projection_classification_best.pt \
  --model-id cnn_005_multiscale_projection \
  --manifest manifests/next_0nubb_vs_bi214.yaml \
  --split test \
  --device cuda:0 \
  --batch-size 8 \
  --num-workers 0 \
  --output-dir 04_evaluations/NEXTALT_cnn_005_multiscale_projection_test_rerun
```

`--max-files-per-class` is omitted, so the test split is not reduced. The `_rerun`
directory must also not exist beforehand.

## 8. Checkpoints, Training History, and Evaluation Artifacts

- [best checkpoint](../../../02_models/checkpoints/NEXTALT_cnn_005_multiscale_projection_classification_best.pt)
- [last checkpoint](../../../02_models/checkpoints/NEXTALT_cnn_005_multiscale_projection_classification_last.pt)
- [epoch CSV](../../../03_training_runs/logs/NEXTALT_cnn_005_multiscale_projection_classification_epochs.csv)
- [complete history JSON](../../../03_training_runs/logs/NEXTALT_cnn_005_multiscale_projection_classification_history.json)
- [training-history plot](../../../03_training_runs/history_plots/NEXTALT_cnn_005_multiscale_projection_classification_history.png)
- [test predictions](../../../04_evaluations/NEXTALT_cnn_005_multiscale_projection_test/predictions_test.npz)
- [test results.csv](../../../04_evaluations/NEXTALT_cnn_005_multiscale_projection_test/evaluation_test/results.csv)
- [strict metrics.json](../../../04_evaluations/NEXTALT_cnn_005_multiscale_projection_test/evaluation_test/.energybench/metrics.json)
- [energy-matched ROC](../../../04_evaluations/NEXTALT_cnn_005_multiscale_projection_test/evaluation_test/energy_matched_roc.png)
- [score-energy dependence](../../../04_evaluations/NEXTALT_cnn_005_multiscale_projection_test/evaluation_test/score_energy_dependence.png)
- [official ten-model leaderboard](../../../04_evaluations/NEXTALT_all_models_comparison/leaderboard.csv)

## 9. Related Local Files

| File | Purpose |
|---|---|
| [config.yaml](config.yaml) | representation, network, and training hyperparameters for this model |
| [train_classification.py](train_classification.py) | model-specific training entry point |
| [src/next_alt/models/cnn.py](../../../src/next_alt/models/cnn.py) | shared residual encoder, scale gate, and classification head |
| [src/next_alt/data.py](../../../src/next_alt/data.py) | coarse/fine representations, centering, collation, and loader |
| [src/next_cnn/data.py](../../../src/next_cnn/data.py) | HDF5 reading, splitting, and orthogonal-projection implementation |
| [src/next_alt/registry.py](../../../src/next_alt/registry.py) | architecture/model/input-kind registration |
| [src/next_alt/training.py](../../../src/next_alt/training.py) | unified training and early stopping |
| [src/next_alt/checkpoint.py](../../../src/next_alt/checkpoint.py) | format-v3 checkpoint and provenance |
| [src/next_alt/adapter.py](../../../src/next_alt/adapter.py) | inference adaptation for `projections` and `fine_projections` |
| [evaluation manifest](../../../manifests/next_0nubb_vs_bi214.yaml) | strict NEXT evaluation contract |
| [complete comparison results](../../../docs/ALTERNATIVE_EVALUATION_RESULTS.md) | ten-model ranking and unified evaluation documentation |
| [usage guide](../../../docs/USAGE_GUIDE.md) | EnergyBench commands and directory structure |

## 10. Limitations and Interpretation Boundaries

- Within each scale, the model still mixes the unaligned XY/XZ/YZ planes early as
  channels. This differs from the processing hypothesis of CNN-004 and is an important
  experimental distinction between the two models.
- Centering the fine projection improves local utilization but removes absolute
  detector position. The coarse branch retains that information, so the branches may
  learn different kinds of shortcuts.
- The same convolutional kernels process both 30 mm and 15 mm pixels, assuming that
  local patterns can share parameters across scales.
- Both scales remain 2-D projections and cannot preserve complete 3-D hit/voxel
  correspondence.
- The gate provides two event-dependent scalar weights; it is not a validated physical
  interpretation of scale importance.
- This architecture is not an FPN and contains no explicit cross-layer feature
  pyramid.
- The current checkpoint is classification-only. A `not applicable` result for
  energy regression is expected.
