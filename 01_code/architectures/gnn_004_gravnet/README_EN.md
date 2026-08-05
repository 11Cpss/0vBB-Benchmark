# GNN-004: GravNet-Style Learned-Space GNN

[中文](README.md) | English

This directory defines a learned-space graph network for the NEXT `0nubb` versus `Bi214` binary classification task. Each block learns a low-dimensional coordinate space from the current node states, constructs a neighborhood in that space, and propagates features with distance-dependent weights. It adopts GravNet's core inductive bias, but its blocks, pooling, classification head, and training protocol are all custom to this project rather than a component-by-component reproduction of the original paper's implementation.

## Model identity

| Item | Value |
|---|---|
| `architecture_id` | `gnn_004_gravnet` |
| PyTorch class | `next_alt.models.point_graph.GravNetClassifier` |
| `input_kind` | `graph` |
| Task | Binary classification; `0nubb=1`, `Bi214=0` |
| Output | One uncalibrated signal logit per event, shape `(B,)` |
| Trainable parameters | 293,585 |
| Checkpoint format | EnergyBench/NEXT format version 3 |

The model is implemented entirely with PyTorch tensor operations. It does not depend on PyG, DGL, `torch_scatter`, or a compiled GravNet operator.

## Input and exact preprocessing

Denote an event's raw coordinates and deposited energies by
$\{(\mathbf r_a,e_a)\}$, with total energy $E=\sum_a e_a$. The actual preprocessing is:

1. Compute the energy-weighted center
   $\mathbf c=\sum_a e_a\mathbf r_a/E$, and transform the raw coordinates to
   $\mathbf r_a-\mathbf c$.
2. Aggregate with `15 mm` voxels:
   $\mathbf q_a=\lfloor(\mathbf r_a-\mathbf c)/15\,\mathrm{mm}\rfloor$.
3. For each occupied voxel, compute its deposited energy
   $E_i=\sum_{a:q_a=i}e_a$ and the number of merged raw rows $n_i$.
4. The geometric voxel center is $(\mathbf q_i+0.5)\times15\,\mathrm{mm}$. The energy-weighted center of the quantized voxels is then subtracted to eliminate the half-cell quantization offset.
5. The input coordinates are
   $\mathbf x_i=\mathbf x_i^{\mathrm{centered}}/1000.0$.
6. The input node features are
   $\mathbf f_i=[E_i/E,\ \log(1+n_i)]$, namely `energy_fraction` and `log1p(hit_count)`.
7. If there are more than 512 occupied voxels, retain the 512 with the highest deposited energy; XYZ is used as a deterministic tie-breaker. Energy fractions are not renormalized after truncation.
8. Zero-pad within each batch and generate `coords: (B,N,3)`, `features: (B,N,2)`, and a boolean `mask: (B,N)`. The mask protects learned-space kNN, aggregation, and event pooling from padded entries.

The total event energy and absolute detector position are not model inputs. Total energy is emitted with predictions only as the EnergyBench evaluation condition.

## Network architecture

### 1. Node encoder

```text
[x, y, z, energy_fraction, log1p(hit_count)]  # 5 dims
    -> Linear(5, 128)
    -> LayerNorm(128) -> SiLU -> Dropout(0.10)
    -> Linear(128, 128) -> SiLU
```

Physical XYZ appears directly only in this initial node-encoding stage; the neighborhood in each subsequent block is determined by learned coordinates.

### 2. Four GravNet-style residual blocks

Each block receives $\mathbf h_i\in\mathbb R^{128}$ and independently projects a learned coordinate and a propagated feature:

$$
\mathbf s_i=W_s\mathbf h_i\in\mathbb R^4,
\qquad
\mathbf p_i=W_p\mathbf h_i\in\mathbb R^{64}.
$$

A self-excluding `k=16` kNN graph is built in $\mathbf s$-space, followed by:

$$
d_{ij}^2=\lVert\mathbf s_i-\mathbf s_j\rVert_2^2,
\qquad
w_{ij}=\exp(-d_{ij}^2).
$$

The two aggregations are a normalized distance-weighted mean and a masked maximum:

$$
\mathbf a_i^{\rm mean}=
\frac{\sum_{j\in\mathcal N(i)}w_{ij}\mathbf p_j}
{\max(\sum_{j\in\mathcal N(i)}w_{ij},10^{-8})},
$$

$$
\mathbf a_i^{\rm max}=\max_{j\in\mathcal N(i)}\mathbf p_j.
$$

The current state and the two 64-dimensional aggregations are then concatenated:

```text
[h_i, weighted_mean_i, masked_max_i]  # 128 + 64 + 64 = 256
    -> MLP 256 -> 128 -> 128
    -> Dropout(0.10)
    -> residual add with h_i
    -> LayerNorm(128)
```

All four blocks relearn a 4D coordinate space from the updated current node state. The kNN index selection itself is discrete; the code recomputes distances for the selected edges from the live learned coordinates, allowing gradients from the `exp(-d²)` weights to continue flowing into `space_projection`.

### 3. Event pooling and classification head

```text
masked_mean(nodes) || masked_max(nodes)  # 128 + 128 = 256
    -> Linear(256, 160) -> SiLU -> Dropout(0.10)
    -> Linear(160, 1)
    -> squeeze -> (B,)
```

## Relationship to the original method

- The original reference is Qasim, Kieseler, Iiyama, and Pierini's [Learning representations of irregular particle-detector geometry with distance-weighted graph networks](https://arxiv.org/abs/1902.07987), which introduced the GarNet and GravNet layers.
- This project retains the central idea of learning a coordinate space from features, building neighborhoods in that learned space, and propagating features with distance decay.
- The original paper primarily targets reconstruction and clustering in irregular particle detectors; this project applies four custom residual GravNet-style blocks to event-level binary classification.
- This implementation uses an explicit normalized `exp(-d²)` weighted mean, an unweighted masked maximum, LayerNorm residuals, and mean+max event pooling. These specific choices should not be treated as a reproduction of the paper's complete network.
- The formal description should therefore use **GravNet-style learned-space GNN**, rather than claiming a reproduction of the original GravNet architecture or official implementation.

## Key configuration

| Category | Configuration item | Value |
|---|---|---:|
| Data | `max_files_per_class` | 100 |
| Data | `split_fractions` | `[0.8, 0.1, 0.1]` |
| Data | `split_seed` | 42 |
| Representation | `point_bin_size` | 15.0 mm |
| Representation | `coordinate_scale` | 1000.0 |
| Representation | `max_points` | 512 |
| Model | `feature_dim` | 2 |
| Model | `hidden_dim` | 128 |
| Model | `num_layers` | 4 |
| Model | `space_dim` | 4 |
| Model | `propagate_dim` | 64 |
| Model | `k` | 16 |
| Model | `classifier_dim` | 160 |
| Model | `dropout` | 0.10 |
| Training | `batch_size` | 12 |
| Training | `epochs` | 50 |
| Training | `learning_rate` | `5e-4` |
| Training | `weight_decay` | `1e-4` |
| Training | `gradient_clip_norm` | 1.0 |
| Training | `early_stopping_patience` | 12 |
| Training | `seed` | 42 |
| Training | `use_amp` / `amp_precision` | `true` / `auto` |
| Training | `deterministic` | `false` |

See [`config.yaml`](config.yaml) for the complete configuration, and [`src/next_alt/config.py`](../../../src/next_alt/config.py) for shared defaults and validation.

## Training behavior and commands

- Complete HDF5 source files are assigned to train/validation/test by stable hash, avoiding source-file leakage.
- The training set is kept class-balanced by alternating signal and background events and stopping when the smaller class is exhausted.
- The current `balance_training_classes=true` code path does not activate the configured `event_shuffle_buffer_size=512`; in practice, file order is shuffled once per epoch.
- Training uses `BCEWithLogitsLoss`, AdamW, and a 50-epoch CosineAnnealingLR.
- Validation AUC selects the best checkpoint, the last checkpoint is saved every epoch, and training stops early after 12 epochs without improvement.
- Formal training requires CUDA and provides neither a CPU fallback nor a smoke-training mode.

The original formal training entry point was:

```bash
cd /home/wenyu/summer
source .venv/bin/activate
python 01_code/architectures/gnn_004_gravnet/train_classification.py
```

The formal training artifacts already exist. The shared default is `allow_overwrite: false`, so directly rerunning now refuses to overwrite the existing files and exits. A new experiment must copy the YAML and specify new checkpoint, log, and plot output directories:

```bash
python 01_code/architectures/gnn_004_gravnet/train_classification.py \
  /path/to/gnn_004_new_experiment.yaml
```

## Completed formal results

| Metric | Result |
|---|---:|
| best epoch | 42 |
| best validation AUC | 0.956015 |
| epochs actually completed | 50 |
| overall full-test rank | **4 / 10** |
| full-test matched AUC | **0.955680** |
| full-test inclusive AUC | 0.955733 |
| energy-independence mean | 0.977351 |

The full test used the complete test split of 1,490 files and 115,499 events. Strict evaluation produced 0 warnings and 0 errors. The formal results directory already exists; reruns must use a new `_rerun` directory without limiting the number of files:

```bash
cd /home/wenyu/summer
source .venv/bin/activate
energybench next \
  02_models/checkpoints/NEXTALT_gnn_004_gravnet_classification_best.pt \
  --device cuda:0 \
  --split test \
  --batch-size 12 \
  --num-workers 0 \
  --model-id gnn_004_gravnet \
  --output-dir 04_evaluations/NEXTALT_gnn_004_gravnet_test_rerun
```

Do not pass `--max-files-per-class` for a formal full-test rerun.

## Artifacts

| Type | File |
|---|---|
| best checkpoint | [`NEXTALT_gnn_004_gravnet_classification_best.pt`](../../../02_models/checkpoints/NEXTALT_gnn_004_gravnet_classification_best.pt) |
| last checkpoint | [`NEXTALT_gnn_004_gravnet_classification_last.pt`](../../../02_models/checkpoints/NEXTALT_gnn_004_gravnet_classification_last.pt) |
| epoch CSV | [`classification_epochs.csv`](../../../03_training_runs/logs/NEXTALT_gnn_004_gravnet_classification_epochs.csv) |
| history JSON | [`classification_history.json`](../../../03_training_runs/logs/NEXTALT_gnn_004_gravnet_classification_history.json) |
| history plot | [`classification_history.png`](../../../03_training_runs/history_plots/NEXTALT_gnn_004_gravnet_classification_history.png) |
| test predictions | [`predictions_test.npz`](../../../04_evaluations/NEXTALT_gnn_004_gravnet_test/predictions_test.npz) |
| test results | [`evaluation_test/results.csv`](../../../04_evaluations/NEXTALT_gnn_004_gravnet_test/evaluation_test/results.csv) |
| strict metrics | [`evaluation_test/.energybench/metrics.json`](../../../04_evaluations/NEXTALT_gnn_004_gravnet_test/evaluation_test/.energybench/metrics.json) |
| ROC | [`energy_matched_roc.png`](../../../04_evaluations/NEXTALT_gnn_004_gravnet_test/evaluation_test/energy_matched_roc.png) |
| score-energy plot | [`score_energy_dependence.png`](../../../04_evaluations/NEXTALT_gnn_004_gravnet_test/evaluation_test/score_energy_dependence.png) |
| ten-model leaderboard | [`leaderboard.csv`](../../../04_evaluations/NEXTALT_all_models_comparison/leaderboard.csv) |

## Related local files

| Responsibility | File |
|---|---|
| Model configuration | [`config.yaml`](config.yaml) |
| Model training entry point | [`train_classification.py`](train_classification.py) |
| GravNet block and classifier | [`src/next_alt/models/point_graph.py`](../../../src/next_alt/models/point_graph.py) |
| Voxelization, padding, and loader | [`src/next_alt/data.py`](../../../src/next_alt/data.py) |
| Architecture registration | [`src/next_alt/registry.py`](../../../src/next_alt/registry.py) |
| Shared training loop | [`src/next_alt/training.py`](../../../src/next_alt/training.py) |
| v3 checkpoint contract | [`src/next_alt/checkpoint.py`](../../../src/next_alt/checkpoint.py) |
| v3 inference adapter | [`src/next_alt/adapter.py`](../../../src/next_alt/adapter.py) |
| HDF5 reader and file-level split | [`src/next_cnn/data.py`](../../../src/next_cnn/data.py) |
| All-architecture documentation | [`docs/ALTERNATIVE_ARCHITECTURES.md`](../../../docs/ALTERNATIVE_ARCHITECTURES.md) |
| Formal evaluation results | [`docs/ALTERNATIVE_EVALUATION_RESULTS.md`](../../../docs/ALTERNATIVE_EVALUATION_RESULTS.md) |
| EnergyBench usage instructions | [`docs/USAGE_GUIDE.md`](../../../docs/USAGE_GUIDE.md) |

## Limitations and interpretation boundaries

- All four layers execute dense `torch.cdist` in learned space, with time and intermediate-memory complexity of approximately `O(4BN²)`.
- Learned-space top-k indices are discrete and non-differentiable; only the continuous distance weights on selected neighbors propagate gradients to the space projection.
- Learned coordinates can expand in scale, contract, or collapse locally, and `exp(-d²)` decays rapidly at large distances. The current implementation mitigates rather than theoretically eliminates these effects through residual connections, LayerNorm, and the normalized denominator.
- Top-512, high-energy-first truncation may remove low-energy long-range topology.
- Physical coordinates are used directly only in the initial encoder. If a learned neighborhood discards useful detector geometry, subsequent blocks do not explicitly recover a fixed geometric graph.
- Input normalization and centering do not constitute a strict guarantee of energy independence.
- The formal result comes from a single split, seed, and hyperparameter configuration and cannot be treated as the performance ceiling of a fully tuned original GravNet method.
- The output is an uncalibrated logit and should not be interpreted directly as a probability.
