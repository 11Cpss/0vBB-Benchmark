# HYBRID-001: Three-View CNN with a Dynamic EdgeConv Graph Network

[中文](README.md) | English

## 1. Model Positioning

HYBRID-001 tests whether two complementary representations can jointly improve
classification performance: detector-fixed 2-D projections provide dense global
views, while an event-centered 3-D voxel graph preserves local spatial adjacency.
The image and graph branches independently produce event-level embeddings, which are
then fused at the feature level.

| Item | Description |
|---|---|
| architecture ID | `hybrid_001_cnn_gnn` |
| checkpoint model name | `CNNGNNHybridClassifier` |
| model family | shared-view CNN + dynamic EdgeConv GNN |
| registry input kind | `hybrid` |
| task | binary classification of `0νββ` signal versus `Bi214` background |
| output | one unnormalized signal logit per event, shape `(B,)` |
| default trainable parameters | **341,969** |
| dependencies | pure PyTorch; no dependency on PyG, DGL, or compiled scatter/kNN extensions |

The implementation is
[`CNNGNNHybridClassifier`](../../../src/next_alt/models/point_graph.py). This
directory contains only the model-specific YAML, entry point, and documentation.

## 2. Exact Input and Preprocessing

Each HDF5 event must generate both image and graph representations. The shared
collation step checks that their batch sizes agree, pads graphs with different node
counts, and creates a boolean mask.

### 2.1 Three-View CNN Input

- batch key: `projections`;
- shape: `(B,3,128,128)`;
- plane order: XY, XZ, YZ;
- 30 mm/bin in fixed detector coordinates;
- origin: `[-1920,-1920,-120]` mm;
- retained energy is divided by the total energy of the complete event and multiplied
  by 100;
- an input row is projected only when x, y, and z all lie within the configured cube;
  unretained energy is not renormalized.

The model then reshapes the three views into `(3B,1,128,128)` and sends them
independently through a shared CNN. The image branch therefore does not mix
XY/XZ/YZ as RGB-like channels in its first layer.

### 2.2 3-D Voxel-Graph Input

The graph branch uses a **voxel graph, not a raw-hit graph**:

1. compute the 3-D energy-weighted centroid from the input energy and center the
   coordinates;
2. aggregate input rows using 15 mm cells;
3. recenter the quantized voxel centers with voxel energy to remove the half-bin
   offset;
4. if the node count exceeds 512, retain the 512 nodes with the highest deposited
   energy, using coordinates as deterministic tie-breakers;
5. after truncation, calculate energy fractions relative to the total energy of the
   complete event without renormalization;
6. pad each graph to the largest node count in its batch and create `mask`.

| Batch tensor | Shape | Meaning |
|---|---|---|
| `coords` | `(B,N,3)` | event-centered voxel centers in mm, divided by `coordinate_scale=1000` |
| `features` | `(B,N,2)` | `[energy_fraction, log1p(hit_count)]` |
| `mask` | `(B,N)` | `true` for valid nodes and `false` for padding |
| `num_points` | `(B,)` | number of valid nodes per event after truncation |

`point_coverage` records the fraction of energy retained after truncation. The
projection and point representations share the same event ID and label, but one
retains absolute detector coordinates while the other deliberately preserves only
relative 3-D topology.

## 3. Network Architecture and Tensor Shapes

### 3.1 Shared-View CNN Branch

Each single-channel view independently passes through the same
`_SharedViewEncoder`:

| Layer | Operation | Single-view output shape |
|---|---|---|
| input | one XY/XZ/YZ projection | `(1,128,128)` |
| conv 1 | `Conv2d(1,16,3x3,stride=2,pad=1)` + GroupNorm + SiLU | `(16,64,64)` |
| conv 2 | `Conv2d(16,32,3x3,stride=2,pad=1)` + GroupNorm + SiLU | `(32,32,32)` |
| conv 3 | `Conv2d(32,64,3x3,stride=2,pad=1)` + GroupNorm + SiLU | `(64,16,16)` |
| conv 4 | `Conv2d(64,128,3x3,stride=2,pad=1)` + GroupNorm + SiLU | `(128,8,8)` |
| pooling | `AdaptiveAvgPool2d(1)` | `(128,)` |
| projection | `Linear(128,128)` | `(128,)` |

The three view embeddings are stacked into `(B,3,128)`, then the learned
`view_identity` is added. Mean and max are taken separately over the view axis and
concatenated into 256 dimensions. Finally,
`Linear(256,128) -> SiLU` produces a 128-dimensional image embedding. This branch
has no residual blocks and no attention gate.

### 3.2 Dynamic EdgeConv Graph Branch

Each node first concatenates its three coordinates with its two features:

1. node encoder:
   `Linear(5,96) -> LayerNorm -> SiLU -> Dropout(0.1) -> Linear(96,96) -> SiLU`;
2. pass through three residual EdgeConv blocks in sequence, with `k=16` and self
   edges excluded;
3. construct kNN from centered `coords` in the first layer;
4. rebuild kNN in feature space in the second and third layers after dividing current
   node features by `sqrt(96)`;
5. because the neighborhoods are recomputed discretely at every layer, this is
   dynamic kNN.

For center node `i` and neighbor `j`, the edge vector is:

```text
[h_i, h_j - h_i, x_j - x_i, ||x_j - x_i||]
```

Its dimension is `96+96+3+1=196`. Even though the final two layers select neighbors
in feature space, their messages still explicitly use the relative vector and
distance from the original centered coordinates. Each block performs:

- edge MLP: `196 -> 96 -> 96`, with LayerNorm, SiLU, and dropout between the linear
  layers;
- masked max aggregation over valid neighbors;
- residual addition with the original node feature;
- LayerNorm, followed by resetting padded nodes to zero.

After three layers, masked mean and max are taken separately over valid nodes and
concatenated into 192 dimensions. `Linear(192,192) -> SiLU` produces the graph
embedding.

### 3.3 Cross-Modal Fusion and Classification Head

- image embedding: 128 dimensions;
- graph embedding: 192 dimensions;
- concatenation: 320 dimensions;
- head: `Linear(320,192) -> SiLU -> Dropout(0.1) -> Linear(192,1)`;
- output: `(B,)` logits.

## 4. Relationship to Reference Methods and Scope Boundaries

This model is a representation-level hybrid for NEXT, not a direct reproduction of
any of the following paper models.

- [Wang et al., *Dynamic Graph CNN for Learning on Point Clouds*, ACM TOG 2019](https://arxiv.org/abs/1801.07829),
  arXiv:1801.07829, DOI
  [10.1145/3326362](https://doi.org/10.1145/3326362). The graph branch directly draws
  on EdgeConv and feature-space dynamic kNN. Here, however, the inputs are centered
  detector voxels, and the implementation uses padded dense `torch.cdist`, explicit
  coordinate edge features, a fixed three-layer width, and project-specific residual
  and normalization choices.
- [Su et al., *Multi-view Convolutional Neural Networks for 3D Shape Recognition*, ICCV 2015](https://arxiv.org/abs/1505.00880),
  arXiv:1505.00880, DOI
  [10.1109/ICCV.2015.114](https://doi.org/10.1109/ICCV.2015.114). The only shared idea
  is encoding multiple 2-D views with a shared CNN before fusion. The current input is
  detector-energy projections and its fusion uses mean/max; it does not reproduce the
  original MVCNN.
- [Wu and He, *Group Normalization*, ECCV 2018](https://arxiv.org/abs/1803.08494),
  arXiv:1803.08494, DOI
  [10.1007/978-3-030-01261-8_1](https://doi.org/10.1007/978-3-030-01261-8_1).
  Only the image branch uses GroupNorm; the graph MLP and blocks use LayerNorm.

The final CNN/GNN concatenation is an experiment-specific design and has no single
corresponding “original paper architecture.” The terms hybrid, EdgeConv, and
multi-view in this document identify method families rather than reproduction claims.

## 5. Key Configuration

See [`config.yaml`](config.yaml) for the authoritative configuration.

| Category | Parameter | Default |
|---|---|---:|
| image representation | grid / bin / input scale | `128 / 30 mm / 100` |
| graph representation | point bin / coordinate scale | `15 mm / 1000` |
| graph representation | max points | `512` |
| model | node feature dim | `2` |
| model | image base channels / embedding | `16 / 128` |
| model | graph hidden / embedding | `96 / 192` |
| model | graph layers / k | `3 / 16` |
| model | classifier dim / dropout | `192 / 0.1` |
| data | max files per class | `100` |
| data | split seed / fractions | `42 / [0.8,0.1,0.1]` |
| data | balanced training / shuffle buffer | `true / 512` |
| training | batch size / epochs | `8 / 50` |
| training | learning rate / weight decay | `3e-4 / 1e-4` |
| training | gradient clip norm | `1.0` |
| training | early-stop patience | `12` |
| training | AMP / deterministic | `auto / false` |

## 6. Shared Training Procedure and Commands

The entry point only fixes the architecture ID. The unified trainer is responsible for
`BCEWithLogitsLoss`, AdamW, `CosineAnnealingLR(T_max=50)`, gradient clipping,
CUDA AMP, validation-AUC checkpoint selection, and early stopping. The training stream
alternates signal and background to remain balanced; validation is neither balanced
nor shuffled. The current `balance_training_classes=true` branch does not perform
event-buffer shuffling: `512` is a retained configuration value, while only
source-file order is shuffled in practice at each epoch. The checkpoint also records
the model configuration, representation configuration, split, source inventory,
training configuration, and history.

Official configuration entry point:

```bash
cd /home/wenyu/summer
source .venv/bin/activate
python 01_code/architectures/hybrid_001_cnn_gnn/train_classification.py
```

A custom YAML may be supplied. The official checkpoints, epoch CSV, history JSON, and
training plot already exist. The default configuration inherits
`allow_overwrite: false` from the shared defaults, so direct execution rejects a
rerun before training starts and neither overwrites nor resumes automatically. For a
new experiment, copy the YAML and configure independent checkpoint, log, and plot
directories.

## 7. Completed Experiment Results

| Metric | Result |
|---|---:|
| epochs actually trained | 17 |
| best epoch | **5** |
| best validation AUC | **0.916308** |
| full-test files / events | `1,490 / 115,499` |
| full-test rank | **9 / 10** |
| matched AUC | **0.912542** |
| inclusive AUC | **0.912792** |
| energy independence | **0.976863** |

The official evaluation is strict, reports `comparable=True`, and has zero warnings
and errors. The current hybrid configuration does not outperform the standalone
Multi-view CNN or the best graph models, showing that “more representations” do not
automatically improve optimization or generalization. This result evaluates only the
current fusion capacity and training setup; it does not invalidate the hybrid
direction itself.

The complete reevaluation example uses a new `_rerun` directory:

```bash
cd /home/wenyu/summer
source .venv/bin/activate
energybench next \
  02_models/checkpoints/NEXTALT_hybrid_001_cnn_gnn_classification_best.pt \
  --model-id hybrid_001_cnn_gnn \
  --manifest manifests/next_0nubb_vs_bi214.yaml \
  --split test \
  --device cuda:0 \
  --batch-size 8 \
  --num-workers 0 \
  --output-dir 04_evaluations/NEXTALT_hybrid_001_cnn_gnn_test_rerun
```

`--max-files-per-class` is omitted, so the command evaluates the complete test
split. The output directory must not exist beforehand.

## 8. Checkpoints, Training History, and Evaluation Artifacts

- [best checkpoint](../../../02_models/checkpoints/NEXTALT_hybrid_001_cnn_gnn_classification_best.pt)
- [last checkpoint](../../../02_models/checkpoints/NEXTALT_hybrid_001_cnn_gnn_classification_last.pt)
- [epoch CSV](../../../03_training_runs/logs/NEXTALT_hybrid_001_cnn_gnn_classification_epochs.csv)
- [complete history JSON](../../../03_training_runs/logs/NEXTALT_hybrid_001_cnn_gnn_classification_history.json)
- [training-history plot](../../../03_training_runs/history_plots/NEXTALT_hybrid_001_cnn_gnn_classification_history.png)
- [test predictions](../../../04_evaluations/NEXTALT_hybrid_001_cnn_gnn_test/predictions_test.npz)
- [test results.csv](../../../04_evaluations/NEXTALT_hybrid_001_cnn_gnn_test/evaluation_test/results.csv)
- [strict metrics.json](../../../04_evaluations/NEXTALT_hybrid_001_cnn_gnn_test/evaluation_test/.energybench/metrics.json)
- [energy-matched ROC](../../../04_evaluations/NEXTALT_hybrid_001_cnn_gnn_test/evaluation_test/energy_matched_roc.png)
- [score-energy dependence](../../../04_evaluations/NEXTALT_hybrid_001_cnn_gnn_test/evaluation_test/score_energy_dependence.png)
- [official ten-model leaderboard](../../../04_evaluations/NEXTALT_all_models_comparison/leaderboard.csv)

## 9. Related Local Files

| File | Purpose |
|---|---|
| [config.yaml](config.yaml) | both representations, the dual-branch model, and training parameters |
| [train_classification.py](train_classification.py) | model-specific training entry point |
| [src/next_alt/models/point_graph.py](../../../src/next_alt/models/point_graph.py) | shared-view encoder, dense kNN, EdgeConv, and hybrid classifier |
| [src/next_alt/data.py](../../../src/next_alt/data.py) | projections, centered voxel graph, truncation, and padding/collation |
| [src/next_cnn/data.py](../../../src/next_cnn/data.py) | HDF5 reading, file-level split, and coarse projections |
| [src/next_alt/registry.py](../../../src/next_alt/registry.py) | `hybrid` input-kind and model registration |
| [src/next_alt/training.py](../../../src/next_alt/training.py) | unified training, AUC selection, and early stopping |
| [src/next_alt/checkpoint.py](../../../src/next_alt/checkpoint.py) | format-v3 checkpoint provenance |
| [src/next_alt/adapter.py](../../../src/next_alt/adapter.py) | inference adaptation for images and padded graphs |
| [evaluation manifest](../../../manifests/next_0nubb_vs_bi214.yaml) | strict NEXT evaluation contract |
| [complete comparison results](../../../docs/ALTERNATIVE_EVALUATION_RESULTS.md) | ten-model ranking and unified conclusions |
| [usage guide](../../../docs/USAGE_GUIDE.md) | EnergyBench CLI and evaluation-directory documentation |

## 10. Limitations and Interpretation Boundaries

- `_knn` uses dense `torch.cdist` over padded tensors. Runtime and distance-matrix
  memory grow approximately with the square of the valid-node limit, making
  `max_points=512` a necessary computational boundary.
- Energy-based truncation favors high-energy voxels and may remove low-energy but
  topologically important thin tracks. Point coverage records retained energy but is
  not supplied as an explicit model input.
- The first layer builds its graph from physical coordinates, while later layers use
  learned features. Dynamic adjacency is discrete, so the neighbor indices themselves
  are not differentiable.
- The projection retains absolute detector position while the graph is deliberately
  centered. Their complementarity is intentional, but it also creates branches with
  different statistical properties and optimization difficulty.
- The capacities and gradient scales of the image and graph branches are not
  explicitly balanced, and there is no auxiliary loss. Simple concatenation may allow
  one branch to be ignored.
- The three views remain lossy projections; the voxel graph is additionally limited
  by 15 mm aggregation and truncation to 512 nodes.
- Mean/max pooling and the final logit do not provide reliable causal explanations at
  the node, edge, or plane level.
- The current checkpoint is trained for classification only. A `not applicable`
  result for energy regression is expected.
