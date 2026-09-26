# GNN-001: Static Geometric GINE-Style Classifier

[中文](README.md) | English

This model constructs a geometric k-nearest-neighbor graph once from the event's centered
voxel coordinates, then reuses the same adjacency in five residual edge-aware GIN/GINE-style
layers. Its central question is whether fixed three-dimensional local connectivity and edge
geometry distinguish `0nubb` and `Bi214` topology better than point-set models without graphs.

## 1. Model Identity

| Item | Value |
|---|---|
| `architecture_id` | `gnn_001_static_gine` |
| checkpoint `model_name` | `StaticGINEClassifier` |
| Python class | `next_alt.models.point_graph.StaticGINEClassifier` |
| registry `input_kind` | `graph` |
| Task | Binary classification of NEXT `0nubb` (label 1) versus `Bi214` (label 0) |
| Output | One uncalibrated signal logit per event, shape `(B,)`; larger values favor `0nubb` |
| Trainable parameters | **479,942** |
| Checkpoint format | EnergyBench/NEXT format version 3 |

## 2. Raw Data and Exact Preprocessing

### 2.1 Data loading, labels, and split

1. Each event is read from contiguous rows with the same `event_id` in HDF5
   `/MC/hits/table`, using the `x/y/z/energy` input columns. The shared reader rejects empty
   events, non-finite values, negative energy, class inconsistencies, and noncontiguous repeated
   event IDs.
2. Directories matching `0nubb_part_*` map to signal/label 1, while directories matching
   `Bi_part_*` map to background/label 0.
3. The complete relative HDF5 path is used as the group, and a stable hash with seed 42 assigns
   files to train/validation/test at 0.8/0.1/0.1. A file cannot leak across splits.
4. Official training selects at most 100 files per class independently in the train and
   validation splits. The official test evaluation has no file limit and uses 1,490 files
   containing 115,499 events.

### 2.2 Nodes from 15 mm voxels

Given hit coordinates $\mathbf r_i$ and deposited energies $e_i$:

1. Compute the complete-event energy $E=\sum_i e_i$ and energy-weighted centroid
   $\mathbf c=\sum_i e_i\mathbf r_i/E$, then center the event.
2. Merge rows in the same cell using
   $\mathbf q_i=\lfloor(\mathbf r_i-\mathbf c)/(15\,\mathrm{mm})\rfloor$.
3. Use $(\mathbf q+0.5)\times15\,\mathrm{mm}$ as each voxel center, then subtract the
   energy-weighted centroid of the quantized voxel centers to remove the common half-bin offset.
4. If there are more than 512 nodes, retain the 512 voxels with the highest energy, using cell
   coordinates as a lexicographic tie-breaker. After truncation, `energy_fraction` remains
   divided by the complete-event energy and is not renormalized.
5. Coordinates are divided by 1000 mm. Node features are
   `[voxel_energy / complete_event_energy, log1p(merged_row_count)]`.

### 2.3 Batch tensors and static graph

| Field | Shape / dtype | Purpose |
|---|---|---|
| `coords` | `(B,N,3)`, float32 | Centered, scaled 3D coordinates; used by both the node encoder and kNN |
| `features` | `(B,N,2)`, float32 | Energy fraction and log hit count |
| `mask` | `(B,N)`, bool | Excludes padded nodes and their edges |
| graph | Implicit `(B,N,K)` neighbor indices | Up to 12 other valid nodes per valid node |
| output | `(B,)` | Signal logit |

The graph is constructed from Euclidean distance before the first message-passing layer, with
self-neighbors excluded. The implementation chooses nearest neighbors independently for each
query, so it is a **directed query-to-neighbor kNN relation**, with no mutual-edge closure or
explicit undirected symmetrization. For events with fewer than 13 nodes, each node has at most
`N-1` valid neighbors; a single-node event has an empty neighbor aggregate. Adjacency indices
are generated under `no_grad` and reused in all five layers.

Total event energy and absolute detector position do not enter the model. Total energy is
exported only as `energy_condition` for evaluation.

## 3. Layer-by-Layer Architecture and Dimensions

The shared `_mlp(a,b,c,p)` helper is
`Linear(a,b) → LayerNorm(b) → SiLU → Dropout(p) → Linear(b,c) → SiLU`.

### 3.1 Input encoding and graph construction

| Stage | Operation | Input → output |
|---|---|---|
| Node input | `concat(coords, features)` | `(B,N,3)+(B,N,2) → (B,N,5)` |
| Node encoder | `_mlp(5,128,128,0.10)` | `(B,N,5) → (B,N,128)` |
| Static graph | Masked Euclidean kNN, `k=12`, excluding self | `(B,N,3) → indices/mask (B,N,K)` |

### 3.2 Five residual GINE-style layers

For center node $i$ and its neighbor $j$, let
$\Delta\mathbf r_{ij}=\mathbf r_j-\mathbf r_i$. Each layer owns an independent set of the
following parameters:

$$
\begin{aligned}
\mathbf e_{ij} &= \mathrm{MLP}_e(
  \Delta\mathbf r_{ij}\Vert\lVert\Delta\mathbf r_{ij}\rVert_2),\\
\mathbf m_{ij} &= \mathrm{MLP}_m(\mathbf h_j+\mathbf e_{ij}),\\
\mathbf a_i &= \sum_{j\in\mathcal N(i)}\mathbf m_{ij},\\
\mathbf u_i &= \mathrm{MLP}_u((1+\epsilon)\mathbf h_i+\mathbf a_i),\\
\mathbf h'_i &= \mathrm{LayerNorm}(
  \mathbf h_i+\mathrm{Dropout}(\mathbf u_i)).
\end{aligned}
$$

| Submodule | Dimensions |
|---|---|
| Edge input | Relative XYZ 3 + Euclidean distance 1 = 4 |
| Edge encoder | `_mlp(4,128,128,0.10)` |
| Message MLP | `_mlp(128,128,128,0.10)` |
| Neighbor aggregate | Masked sum, `K≤12` |
| Update MLP | `_mlp(128,128,128,0.10)` |
| Epsilon | One learnable scalar per layer, initialized to 0 |
| Residual normalization | Node residual + dropout update + LayerNorm(128) |
| Number of layers | 5; every layer reuses the initial static kNN graph |

### 3.3 Event readout

| Stage | Operation | Input → output |
|---|---|---|
| Graph pooling | Concatenate masked node mean and max | `(B,N,128) → (B,256)` |
| Classifier hidden | `Linear + SiLU + Dropout(0.10)` | `256 → 160` |
| Classifier output | `Linear` | `160 → 1` |

## 4. Differences from the Original GIN/GINE Methods

This model is inspired by GIN's learnable-$\epsilon$ sum aggregation and the edge-feature-aware
GINE message form, but it is a custom residual variant for this project:

- edge features are not bond types supplied by the data; they are learned by an edge encoder
  from relative XYZ and distance;
- canonical GINE is often expressed as summing a nonlinear transformation of
  $h_j+e_{ij}$. This implementation adds a complete message MLP, update MLP, outer residual,
  and LayerNorm;
- the graph is constructed from each event's geometric coordinates using fixed kNN rather than
  from discrete molecular or general-graph edges in the original papers;
- adjacency remains unchanged across the five layers and is not explicitly symmetrized;
- graph-level readout uses masked mean+max rather than claiming a standard sum readout;
- the implementation is pure PyTorch with dense masked kNN and does not call PyG `GINEConv`,
  DGL, or `torch_scatter`.

It should therefore be described as “static geometric GINE-style” or an “edge-aware GIN
variant,” not as a line-by-line reproduction of PyG GINEConv or the paper's network.

## 5. Key Configuration

| Category | Parameter | Value |
|---|---|---:|
| Representation | `point_bin_size` | 15.0 mm |
| Representation | `coordinate_scale` | 1000.0 mm |
| Representation | `max_points` | 512 |
| Model | `feature_dim` | 2 |
| Model | `hidden_dim` | 128 |
| Model | `num_layers` | 5 |
| Model | `k` | 12 |
| Model | `classifier_dim` | 160 |
| Model | `dropout` | 0.10 |
| Model | `train_eps` | true |
| Data | `max_files_per_class` | 100 |
| Data | `split_seed` / `split_fractions` | 42 / `[0.8, 0.1, 0.1]` |
| Data | `balance_training_classes` | true |
| Data | `event_shuffle_buffer_size` | 512 (see the actual mechanism below) |
| Data | `num_workers` | 0 |
| Training | `batch_size` | 16 |
| Training | `epochs` | 50 |
| Training | `learning_rate` | 5e-4 |
| Training | `weight_decay` | 1e-4 |
| Training | `gradient_clip_norm` | 1.0 |
| Training | `early_stopping_patience` / `min_delta` | 12 / 0.0 |
| Training | `seed` / `deterministic` | 42 / false |
| Training | AMP | `auto`; the official checkpoint records bfloat16 |

## 6. Training Mechanism and Command

- `BCEWithLogitsLoss` + AdamW;
- `CosineAnnealingLR(T_max=50)`, stepped each epoch;
- gradient norm clipping at 1.0;
- files are shuffled every epoch, while balanced mode alternates signal and background events;
- the current balanced branch does not pass through the bounded event shuffle buffer. Therefore,
  `event_shuffle_buffer_size=512` in the configuration does not mean that buffer-level event
  shuffling actually took place in this run;
- the best checkpoint is selected by validation AUC, with early-stopping patience 12;
- GPU only, using `cuda:0` and AMP, with no CPU fallback or smoke branch.

Standard training entry point:

```bash
cd /home/wenyu/summer
source .venv/bin/activate
python 01_code/architectures/gnn_001_static_gine/train_classification.py
```

The official best/last checkpoints, history, and plot already exist. Because the runner defaults
to `allow_overwrite=false`, running the command again will refuse to overwrite them before
training begins. For an independent retraining run, copy `config.yaml`, configure new output
paths for checkpoints, logs, and plots in the copy, and pass the copied path as the training
script's sole positional argument.

## 7. Completed Training and Official Evaluation Results

The best checkpoint is at epoch 42. Its AUC on 16,786 validation events is **0.981898**, with
mean representation coverage of 1.0.

The strict official results on the complete test split are:

| Metric | Result |
|---|---:|
| Classification rank among the 10 alternative architectures | **1 / 10** |
| Energy-matched AUC | **0.981035** |
| Inclusive AUC | **0.981224** |
| Energy-independence score (mean) | **0.977111** |
| Test files / events | 1,490 / 115,499 |

The result has 0 warnings and 0 errors and achieves the highest matched AUC in the ten-model
comparison. The model outputs only a classification logit, so `not_applicable` for energy
regression is expected.

### Rerunning the complete test evaluation in a new directory

```bash
cd /home/wenyu/summer
source .venv/bin/activate
energybench next \
  02_models/checkpoints/NEXTALT_gnn_001_static_gine_classification_best.pt \
  --split test \
  --device cuda:0 \
  --output-dir 04_evaluations/NEXTALT_gnn_001_static_gine_test_rerun
```

This uses a new `_rerun` directory to avoid overwriting the existing official results. Do not
specify `--max-files-per-class`, because that would not be a full-test evaluation.

## 8. Existing Artifacts

| Artifact | Path |
|---|---|
| Best checkpoint | [`NEXTALT_gnn_001_static_gine_classification_best.pt`](../../../02_models/checkpoints/NEXTALT_gnn_001_static_gine_classification_best.pt) |
| Last checkpoint | [`NEXTALT_gnn_001_static_gine_classification_last.pt`](../../../02_models/checkpoints/NEXTALT_gnn_001_static_gine_classification_last.pt) |
| Epoch metrics | [`..._epochs.csv`](../../../03_training_runs/logs/NEXTALT_gnn_001_static_gine_classification_epochs.csv) |
| Complete history | [`..._history.json`](../../../03_training_runs/logs/NEXTALT_gnn_001_static_gine_classification_history.json) |
| Training curve | [`..._history.png`](../../../03_training_runs/history_plots/NEXTALT_gnn_001_static_gine_classification_history.png) |
| Official test predictions | [`predictions_test.npz`](../../../04_evaluations/NEXTALT_gnn_001_static_gine_test/predictions_test.npz) |
| Official test summary | [`results.csv`](../../../04_evaluations/NEXTALT_gnn_001_static_gine_test/evaluation_test/results.csv) |
| Complete evaluation metrics | [`metrics.json`](../../../04_evaluations/NEXTALT_gnn_001_static_gine_test/evaluation_test/.energybench/metrics.json) |
| Matched ROC plot | [`energy_matched_roc.png`](../../../04_evaluations/NEXTALT_gnn_001_static_gine_test/evaluation_test/energy_matched_roc.png) |
| Score–energy plot | [`score_energy_dependence.png`](../../../04_evaluations/NEXTALT_gnn_001_static_gine_test/evaluation_test/score_energy_dependence.png) |
| Ten-model leaderboard | [`leaderboard.csv`](../../../04_evaluations/NEXTALT_all_models_comparison/leaderboard.csv) |

## 9. Related Local Files

| File | Role |
|---|---|
| [`config.yaml`](./config.yaml) | Official representation, architecture, and training parameters for this model |
| [`train_classification.py`](./train_classification.py) | Training entry point with a fixed architecture ID |
| [`src/next_alt/models/point_graph.py`](../../../src/next_alt/models/point_graph.py) | Authoritative implementation of masked kNN, the GINE layer, and `StaticGINEClassifier` |
| [`src/next_alt/data.py`](../../../src/next_alt/data.py) | Voxelization, truncation, point/graph representations, and padded collation |
| [`src/next_cnn/data.py`](../../../src/next_cnn/data.py) | HDF5 loading, labels, and file-level splitting |
| [`src/next_alt/registry.py`](../../../src/next_alt/registry.py) | Architecture/class/input-kind registration |
| [`src/next_alt/training.py`](../../../src/next_alt/training.py) | Shared loss, optimizer, scheduler, and checkpoint selection |
| [`src/next_alt/checkpoint.py`](../../../src/next_alt/checkpoint.py) | Format-v3 checkpoint schema and provenance |
| [`src/next_alt/adapter.py`](../../../src/next_alt/adapter.py) | Checkpoint reconstruction, data-leakage checks, and official inference |
| [`docs/ALTERNATIVE_ARCHITECTURES.md`](../../../docs/ALTERNATIVE_ARCHITECTURES.md) | Overview of representations and methods for all alternative models |
| [`docs/ALTERNATIVE_EVALUATION_RESULTS.md`](../../../docs/ALTERNATIVE_EVALUATION_RESULTS.md) | Official comparison contract and ten-model results |

This directory has no separate `model.py`; training and inference use the same model definition
from `src/next_alt/models/point_graph.py`.

## 10. Original Method References

- Keyulu Xu et al.,
  [*How Powerful are Graph Neural Networks?*](https://openreview.net/forum?id=ryGs6iA5Km),
  ICLR 2019. This paper provides the theoretical basis for GIN and learnable-$\epsilon$
  aggregation.
- Weihua Hu et al.,
  [*Strategies for Pre-training Graph Neural Networks*](https://openreview.net/forum?id=HJlWWJSFDH),
  ICLR 2020. This work uses GIN updates with edge features and is the principal methodological
  basis for GINE-style implementations.

## 11. Limitations

- distance computation for dense `torch.cdist` kNN is $O(BN^2)$; the 512-node cap is both an
  information and computational tradeoff;
- directed kNN has no mutual/symmetric closure, and fixed `k` corresponds to different physical
  radii at different node densities;
- all five layers reuse the initial geometric graph and cannot update neighborhoods from learned
  node embeddings as a dynamic graph network can;
- direct use of centered XYZ and relative XYZ provides no guarantee of rotational invariance or
  E(3) equivariance;
- truncation discards low-energy nodes, and absolute total energy is not an input;
- rank 1 is a result under the current data, training budget, model size, and single seed.
  Multi-seed, systematic-uncertainty, and computational-cost studies are still needed to
  support more general conclusions.
