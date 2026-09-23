# CNN-004: Three-View Shared Encoder with Late Fusion

[中文](README.md) | English

## 1. Model Positioning

CNN-004 tests a specific hypothesis: XY, XZ, and YZ are three orthogonal projections
with different geometric meanings and spatially unaligned pixels, so they should not
be mixed by the first convolution as if they were the channels of an RGB image. The
model first processes each view independently with the same single-channel residual
encoder, then performs late fusion after each view has been reduced to an event-level
representation.

| Item | Description |
|---|---|
| architecture ID | `cnn_004_multiview_late_fusion` |
| checkpoint model name | `MultiViewLateFusionCNN` |
| model family | multi-view 2-D residual CNN |
| registry input kind | `projection2d` |
| task | binary classification of `0νββ` signal versus `Bi214` background |
| output | one unnormalized signal logit per event, shape `(B,)` |
| default trainable parameters | **807,666** |
| main comparison question | whether view-level late fusion outperforms early fusion that treats the three views as ordinary channels |

The model class is not redefined in this directory. This directory contains only the
model-specific configuration, entry point, and documentation. The implementation is
[`MultiViewLateFusionCNN`](../../../src/next_alt/models/cnn.py), constructed by the
shared registry from the architecture ID.

## 2. Exact Input and Preprocessing

Raw events come from `/MC/hits/table` in HDF5; each row provides
`x,y,z,energy`. The shared reader first performs a deterministic
train/validation/test split over complete HDF5 files, preventing events from the same
file from crossing splits.

The model input `projections` has shape `(B,3,128,128)`, with a fixed view order:

1. channel 0: XY;
2. channel 1: XZ;
3. channel 2: YZ.

The default projection parameters are:

| Parameter | Value | Meaning |
|---|---:|---|
| `projection_grid_size` | 128 | height and width of each plane |
| `projection_bin_size` | 30 mm | projection-pixel size |
| `projection_origin` | `[-1920,-1920,-120]` mm | fixed detector-coordinate origin |
| `projection_input_scale` | 100 | scale applied after normalization |

An input row contributes to all three projections only if its discretized x, y, and z
coordinates all lie inside the configured 3-D range. Retained energy is normalized by
the total energy of the complete event, rather than renormalized by the energy that
falls within the range, and is then multiplied by 100. Consequently,
`projection_coverage` may be below 1, and missing energy is not hidden.

Before entering the network, `(B,3,128,128)` is reshaped to
`(3B,1,128,128)`. The three planes share one encoder, but they are never mixed as
three ordinary image channels in the first layer.

## 3. Network Architecture and Tensor Shapes

### 3.1 Shared Single-View Encoder

The shapes below omit the batch dimension and apply to one single-channel view:

| Layer/stage | Operation | Output shape |
|---|---|---|
| input | one XY/XZ/YZ projection | `(1,128,128)` |
| stem | `Conv2d(1,16,5x5,stride=2,pad=2)` + GroupNorm + SiLU | `(16,64,64)` |
| stage 0 | two 2-D residual blocks, width 16 | `(16,64,64)` |
| stage 1 | two blocks, first block stride 2, `16 -> 32` | `(32,32,32)` |
| stage 2 | two blocks, first block stride 2, `32 -> 64` | `(64,16,16)` |
| stage 3 | two blocks, first block stride 2, `64 -> 128` | `(128,8,8)` |
| view pooling | mean over both spatial axes | `(128,)` |

The main path of every residual block is
`3x3 Conv -> GroupNorm -> SiLU -> 3x3 Conv -> GroupNorm`. When the channel count or
stride changes, the skip path uses `1x1 Conv + GroupNorm`; SiLU is applied after the
main and skip paths are added. GroupNorm does not depend on within-batch statistics,
which suits a unified experiment in which different architectures use different batch
sizes.

### 3.2 View Identity, Attention, and Classification Head

The three 128-dimensional view vectors form `(B,3,128)`:

1. add the learnable `view_identity`, shape `(1,3,128)`, to preserve XY/XZ/YZ
   identity;
2. pass each view through a shared scorer:
   `Linear(128,64) -> SiLU -> Linear(64,1)`;
3. apply softmax over the view axis to obtain three event-dependent weights;
4. multiply every view by its corresponding weight;
5. **do not sum the views**; instead, flatten them in their fixed order to 384
   dimensions;
6. use the classification head:
   `Linear(384,256) -> SiLU -> Dropout(0.1) -> Linear(256,1)`;
7. squeeze the result to `(B,)` logits.

This design retains shared local feature rules, plane identity, and high-level
information from each of the three views.

## 4. Relationship to Reference Methods and Scope Boundaries

This model is a lightweight variant designed for NEXT energy-deposition data. The
following papers document the origins of the methods; they **do not imply a
reproduction of the papers' original networks, inputs, or training procedures**.

- [Su et al., *Multi-view Convolutional Neural Networks for 3D Shape Recognition*, ICCV 2015](https://arxiv.org/abs/1505.00880),
  arXiv:1505.00880, DOI
  [10.1109/ICCV.2015.114](https://doi.org/10.1109/ICCV.2015.114). Both approaches
  encode multiple 2-D views before fusion. The original paper processes rendered
  views, whereas this model processes fixed orthogonal detector-energy projections
  and uses learned identities, softmax weighting, and ordered concatenation. It is
  not a reproduction of the original MVCNN view pooling.
- [He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016](https://arxiv.org/abs/1512.03385),
  arXiv:1512.03385, DOI
  [10.1109/CVPR.2016.90](https://doi.org/10.1109/CVPR.2016.90). This model adopts the
  basic residual-shortcut idea, but its widths, depth, stem, activations, and
  normalization are project-specific.
- [Wu and He, *Group Normalization*, ECCV 2018](https://arxiv.org/abs/1803.08494),
  arXiv:1803.08494, DOI
  [10.1007/978-3-030-01261-8_1](https://doi.org/10.1007/978-3-030-01261-8_1).

## 5. Key Configuration

The adjacent [`config.yaml`](config.yaml) is the authoritative source for these
parameters.

| Category | Parameter | Default |
|---|---|---:|
| representation | grid / bin | `128 / 30 mm` |
| representation | input scale | `100` |
| model | base channels | `16` |
| model | stage blocks | `[2,2,2,2]` |
| model | fusion features | `256` |
| model | dropout | `0.1` |
| data | max files per class | `100` |
| data | split seed / fractions | `42 / [0.8,0.1,0.1]` |
| data | balanced training / shuffle buffer | `true / 512` |
| training | batch size / epochs | `16 / 50` |
| training | learning rate / weight decay | `1e-3 / 1e-4` |
| training | gradient clip norm | `1.0` |
| training | early-stop patience | `12` |
| training | AMP / deterministic | `auto / false` |

## 6. Shared Training Procedure and Commands

[`train_classification.py`](train_classification.py) only fixes the architecture ID
and passes the YAML configuration to the shared trainer. The shared training procedure
uses:

- `BCEWithLogitsLoss`;
- AdamW;
- `CosineAnnealingLR(T_max=epochs)`;
- gradient clipping at every step;
- CUDA AMP, where `auto` resolves to BF16 when supported by the device and FP16
  otherwise;
- best-checkpoint selection by validation AUC;
- early stopping after 12 consecutive epochs without improvement;
- alternating signal/background sampling to balance training data, with no balancing
  or shuffling in validation. The current `balance_training_classes=true` branch
  does not perform event-buffer shuffling: `512` is a retained configuration value,
  while only source-file order is shuffled in practice at each epoch.

The entry command for the official configuration is:

```bash
cd /home/wenyu/summer
source .venv/bin/activate
python 01_code/architectures/cnn_004_multiview_late_fusion/train_classification.py
```

An alternative YAML may be supplied as the sole positional argument. The official
checkpoint, CSV, JSON, and history plot already exist, and
`allow_overwrite: false` is the default. The command above will therefore reject a
rerun before training starts; it will neither resume training nor overwrite existing
results. To run a new experiment, copy the YAML and configure new checkpoint, log, and
plot output directories so the official artifacts are not overwritten accidentally.

## 7. Completed Experiment Results

The best checkpoint was selected by validation AUC during training. The test metrics
below come from a strict evaluation over the complete test set, without a file limit.

| Metric | Result |
|---|---:|
| epochs actually trained | 24 |
| best epoch | **12** |
| best validation AUC | **0.956498** |
| full-test files / events | `1,490 / 115,499` |
| full-test rank | **3 / 10** |
| matched AUC | **0.955819** |
| inclusive AUC | **0.955936** |
| energy independence | **0.978224** |

This evaluation shares the same evaluation, protocol, and code fingerprint as the
other nine models. Its status is `comparable=True`, with zero warnings and errors.

To reevaluate the model, use a new directory that does not already exist, for example:

```bash
cd /home/wenyu/summer
source .venv/bin/activate
energybench next \
  02_models/checkpoints/NEXTALT_cnn_004_multiview_late_fusion_classification_best.pt \
  --model-id cnn_004_multiview_late_fusion \
  --manifest manifests/next_0nubb_vs_bi214.yaml \
  --split test \
  --device cuda:0 \
  --batch-size 16 \
  --num-workers 0 \
  --output-dir 04_evaluations/NEXTALT_cnn_004_multiview_late_fusion_test_rerun
```

The command omits `--max-files-per-class`, so it uses the full test split.

## 8. Checkpoints, Training History, and Evaluation Artifacts

- [best checkpoint](../../../02_models/checkpoints/NEXTALT_cnn_004_multiview_late_fusion_classification_best.pt)
- [last checkpoint](../../../02_models/checkpoints/NEXTALT_cnn_004_multiview_late_fusion_classification_last.pt)
- [epoch CSV](../../../03_training_runs/logs/NEXTALT_cnn_004_multiview_late_fusion_classification_epochs.csv)
- [complete history JSON](../../../03_training_runs/logs/NEXTALT_cnn_004_multiview_late_fusion_classification_history.json)
- [training-history plot](../../../03_training_runs/history_plots/NEXTALT_cnn_004_multiview_late_fusion_classification_history.png)
- [test predictions](../../../04_evaluations/NEXTALT_cnn_004_multiview_late_fusion_test/predictions_test.npz)
- [test results.csv](../../../04_evaluations/NEXTALT_cnn_004_multiview_late_fusion_test/evaluation_test/results.csv)
- [strict metrics.json](../../../04_evaluations/NEXTALT_cnn_004_multiview_late_fusion_test/evaluation_test/.energybench/metrics.json)
- [energy-matched ROC](../../../04_evaluations/NEXTALT_cnn_004_multiview_late_fusion_test/evaluation_test/energy_matched_roc.png)
- [score-energy dependence](../../../04_evaluations/NEXTALT_cnn_004_multiview_late_fusion_test/evaluation_test/score_energy_dependence.png)
- [official ten-model leaderboard](../../../04_evaluations/NEXTALT_all_models_comparison/leaderboard.csv)

## 9. Related Local Files

| File | Purpose |
|---|---|
| [config.yaml](config.yaml) | representation, network, and training hyperparameters for this model |
| [train_classification.py](train_classification.py) | model-specific training entry point |
| [src/next_alt/models/cnn.py](../../../src/next_alt/models/cnn.py) | residual block, shared encoder, and model implementation |
| [src/next_alt/data.py](../../../src/next_alt/data.py) | representation dispatch, batch collation, and loader |
| [src/next_cnn/data.py](../../../src/next_cnn/data.py) | HDF5 reading, file-level split, and XY/XZ/YZ projections |
| [src/next_alt/registry.py](../../../src/next_alt/registry.py) | architecture-ID, model-class, and input-kind registration |
| [src/next_alt/training.py](../../../src/next_alt/training.py) | loss, optimizer, scheduler, early stopping, and artifact writing |
| [src/next_alt/checkpoint.py](../../../src/next_alt/checkpoint.py) | format-v3 checkpoint provenance |
| [src/next_alt/adapter.py](../../../src/next_alt/adapter.py) | adaptation from checkpoint to EnergyBench inference |
| [evaluation manifest](../../../manifests/next_0nubb_vs_bi214.yaml) | strict NEXT evaluation contract |
| [complete comparison results](../../../docs/ALTERNATIVE_EVALUATION_RESULTS.md) | unified results for the ten non-Transformer architectures |
| [usage guide](../../../docs/USAGE_GUIDE.md) | EnergyBench CLI and artifact documentation |

## 10. Limitations and Interpretation Boundaries

- The three orthogonal projections lose 3-D correspondence. Late fusion cannot recover
  information already discarded by projection.
- The shared encoder assumes that local patterns in different planes can use the same
  convolutional kernels, which may not fully reflect physical anisotropy across
  detector axes.
- Learned view identities preserve plane names but do not provide rotation
  equivariance or rotation invariance.
- Detector-fixed projections retain absolute position and may allow the model to
  exploit positional differences unrelated to topology.
- The attention weights only scale the three view vectors. Because the final
  representation uses ordered concatenation, the weights cannot be interpreted as
  strict causal attributions of each view's contribution.
- The current experiment trains classification only. A `not applicable` status for
  energy regression in evaluation is expected.
