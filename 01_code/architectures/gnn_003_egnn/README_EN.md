# GNN-003: E(n)-Equivariant GNN

[中文](README.md) | English

This directory defines an EGNN-style dynamic-graph model for the NEXT `0nubb` versus `Bi214` binary classification task. Node states remain scalars under rotations, translations, and reflections; messages use node states and pairwise distances, while coordinates are updated equivariantly along relative directions. Final classification aggregates both invariant node states and invariant squared-radius statistics.

## Model identity

| Item | Value |
|---|---|
| `architecture_id` | `gnn_003_egnn` |
| PyTorch class | `next_alt.models.point_graph.EGNNClassifier` |
| `input_kind` | `graph` |
| Task | Binary classification; `0nubb=1`, `Bi214=0` |
| Output | One uncalibrated invariant signal logit per event, shape `(B,)` |
| Trainable parameters | 640,518 |
| Checkpoint format | EnergyBench/NEXT format version 3 |

The implementation uses only PyTorch. It does not depend on PyG, DGL, e3nn, or custom CUDA extensions.

## Input and exact preprocessing

For an event, denote the coordinates and deposited energies of its raw rows by
$\{(\mathbf r_a,e_a)\}$, with total energy $E=\sum_a e_a$. The shared preprocessing performs the following steps:

1. Compute the energy-weighted center
   $\mathbf c=\sum_a e_a\mathbf r_a/E$, then translate all raw coordinates to
   $\mathbf r_a-\mathbf c$.
2. Re-aggregate occupied voxels using a `15 mm` side length:
   $\mathbf q_a=\lfloor(\mathbf r_a-\mathbf c)/15\,\mathrm{mm}\rfloor$.
3. For each voxel, obtain its deposited energy
   $E_i=\sum_{a:q_a=i}e_a$ and the number of merged rows $n_i$.
4. The geometric voxel center is $(\mathbf q_i+0.5)\times15\,\mathrm{mm}$. The energy-weighted center of the quantized voxels is then subtracted to remove the half-cell quantization offset.
5. The initial model coordinates are
   $\mathbf x_i=\mathbf x_i^{\mathrm{centered}}/1000.0$. In other words, the coordinates in mm are divided by `coordinate_scale=1000.0`.
6. The scalar node features are
   $\mathbf f_i=[E_i/E,\ \log(1+n_i)]$.
7. If there are more than 512 occupied voxels, retain the 512 with the highest deposited energy, using XYZ as a deterministic tie-breaker. The truncated `energy_fraction` values are not renormalized.
8. Zero-pad within each batch and produce `coords: (B,N,3)`, `features: (B,N,2)`, and `mask: (B,N)`. The mask excludes padded nodes and neighbors from coordinate updates and pooling.

The total event energy and absolute detector position are not fed directly to the classifier. Total energy is exported with the predictions only as EnergyBench's `energy_condition`.

## Network architecture

### 1. Invariant node encoder

The initial node encoder **does not read XYZ**; it reads only the two scalar features:

```text
[energy_fraction, log1p(hit_count)]  # 2 dims
    -> Linear(2, 128)
    -> LayerNorm(128) -> SiLU -> Dropout(0.10)
    -> Linear(128, 128) -> SiLU
```

Coordinate components are therefore not mixed directly into the scalar node state.

### 2. Five EGNN-style update layers

Each layer first builds a self-excluding `k=16` kNN graph in the current evolving coordinates $\mathbf x$. For edge $i\leftarrow j$:

$$
d_{ij}^2=\lVert\mathbf x_i-\mathbf x_j\rVert_2^2,
\qquad
\mathbf m_{ij}=\phi_e([\mathbf h_i,\mathbf h_j,d_{ij}^2]),
$$

where $\phi_e$ is a `257 -> 128 -> 128` MLP. Messages are aggregated with a masked mean, followed by a residual feature update:

$$
\bar{\mathbf m}_i=\operatorname{mean}_{j\in\mathcal N(i)}\mathbf m_{ij},
$$

$$
\mathbf h_i'=\operatorname{LayerNorm}\left(
\mathbf h_i+\operatorname{Dropout}
\left[\phi_h([\mathbf h_i,\bar{\mathbf m}_i])\right]
\right),
$$

where $\phi_h$ is `256 -> 128 -> 128`. The coordinate update uses normalized relative directions:

$$
\mathbf u_{ij}=\frac{\mathbf x_i-\mathbf x_j}
{\sqrt{d_{ij}^2+10^{-8}}},
\qquad
\alpha_{ij}=\tanh(\phi_x(\mathbf m_{ij})),
$$

$$
\mathbf x_i'=\mathbf x_i+0.10\cdot
\operatorname{mean}_{j\in\mathcal N(i)}
(\mathbf u_{ij}\alpha_{ij}).
$$

`coord_scale=0.10` limits coordinate drift in each layer. The updated coordinates are used to reconstruct the graph in the next layer.

### 3. Invariant geometric statistics and classification head

After five layers, the scalar node states are first pooled:

```text
masked_mean(h) || masked_max(h)  # 128 + 128 = 256
```

The model then computes the masked arithmetic center of the evolving coordinates,
$\bar{\mathbf x}=\operatorname{mean}_i\mathbf x_i$, and
$r_i^2=\lVert\mathbf x_i-\bar{\mathbf x}\rVert_2^2$, followed by:

```text
masked_mean(radius_squared) || masked_max(radius_squared)  # 2 dims
```

The final classification head is:

```text
[node mean/max, radius_squared mean/max]  # 256 + 2 = 258
    -> Linear(258, 160) -> SiLU -> Dropout(0.10)
    -> Linear(160, 1)
    -> squeeze -> (B,)
```

This corrects an earlier documentation error stating that the final model pools only scalar node states: the actual model also explicitly uses two squared-radius geometric statistics. These remain invariant under a common translation, rotation, or reflection and therefore do not break the E(3) invariance of the event logit.

## Relationship to the original method

- The original method is Satorras, Hoogeboom, and Welling's [E(n) Equivariant Graph Neural Networks](https://arxiv.org/abs/2102.09844); the formal ICML version is available in [PMLR 139](https://proceedings.mlr.press/v139/satorras21a.html).
- This project retains the equivariant core in which scalar messages depend on pairwise distances and coordinates are updated along relative directions.
- The original paper permits operation on a given graph; this implementation reconstructs dynamic kNN from the evolving coordinates in every layer.
- This implementation normalizes relative displacements to unit directions, uses a `tanh`-bounded coefficient, and fixes `coord_scale=0.10`; these are not a literal implementation of the original paper's equations.
- It additionally concatenates the mean and maximum squared radius of the evolving coordinates into the event classification head and uses NEXT-specific voxels, features, and training protocol.
- It should therefore be described as an **EGNN-style dynamic-kNN classifier**, not as a reproduction of the original EGNN code.

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
| Model | `message_dim` | 128 |
| Model | `num_layers` | 5 |
| Model | `k` | 16 |
| Model | `coord_scale` | 0.10 |
| Model | `classifier_dim` | 160 |
| Model | `dropout` | 0.10 |
| Training | `batch_size` | 12 |
| Training | `epochs` | 50 |
| Training | `learning_rate` | `3e-4` |
| Training | `weight_decay` | `1e-4` |
| Training | `gradient_clip_norm` | 1.0 |
| Training | `early_stopping_patience` | 12 |
| Training | `seed` | 42 |
| Training | `use_amp` / `amp_precision` | `true` / `auto` |
| Training | `deterministic` | `false` |

See [`config.yaml`](config.yaml) for the complete configuration, and [`src/next_alt/config.py`](../../../src/next_alt/config.py) for shared defaults and validation.

## Training behavior and commands

- Data is split by complete HDF5 file through a stable hash, preventing a source file from leaking into multiple splits.
- With `balance_training_classes=true`, the stream alternates signal and background events and stops when the smaller class is exhausted.
- The current balanced branch does not use the YAML's `event_shuffle_buffer_size=512`; in practice, source-file order is shuffled once per epoch.
- The loss is `BCEWithLogitsLoss`, the optimizer is AdamW, and the scheduler is a 50-epoch CosineAnnealingLR.
- Validation AUC selects the best checkpoint, the last checkpoint is updated every epoch, and patience is 12 epochs.
- Formal training supports CUDA only, with no CPU fallback or smoke mode.

The original formal training entry point was:

```bash
cd /home/wenyu/summer
source .venv/bin/activate
python 01_code/architectures/gnn_003_egnn/train_classification.py
```

The formal training artifacts already exist. Because `allow_overwrite: false` by default, directly rerunning the command above refuses to start when the target files are present. For a new experiment, copy the YAML and set new `output.checkpoint_dir`, `output.log_dir`, and `output.plot_dir` values:

```bash
python 01_code/architectures/gnn_003_egnn/train_classification.py \
  /path/to/gnn_003_new_experiment.yaml
```

## Completed formal results

| Metric | Result |
|---|---:|
| best epoch | 7 |
| best validation AUC | 0.889481 |
| early-stop epoch | 19 |
| overall full-test rank | **10 / 10** |
| full-test matched AUC | **0.886558** |
| full-test inclusive AUC | 0.887098 |
| energy-independence mean | 0.977462 |

The full test covers the complete test split of 1,490 files and 115,499 events. Strict evaluation produced 0 warnings and 0 errors. The formal directory must not be overwritten; evaluation reruns should write to a new `_rerun` directory and retain the complete test split:

```bash
cd /home/wenyu/summer
source .venv/bin/activate
energybench next \
  02_models/checkpoints/NEXTALT_gnn_003_egnn_classification_best.pt \
  --device cuda:0 \
  --split test \
  --batch-size 12 \
  --num-workers 0 \
  --model-id gnn_003_egnn \
  --output-dir 04_evaluations/NEXTALT_gnn_003_egnn_test_rerun
```

Do not pass `--max-files-per-class` for a formal full-test rerun.

## Artifacts

| Type | File |
|---|---|
| best checkpoint | [`NEXTALT_gnn_003_egnn_classification_best.pt`](../../../02_models/checkpoints/NEXTALT_gnn_003_egnn_classification_best.pt) |
| last checkpoint | [`NEXTALT_gnn_003_egnn_classification_last.pt`](../../../02_models/checkpoints/NEXTALT_gnn_003_egnn_classification_last.pt) |
| epoch CSV | [`classification_epochs.csv`](../../../03_training_runs/logs/NEXTALT_gnn_003_egnn_classification_epochs.csv) |
| history JSON | [`classification_history.json`](../../../03_training_runs/logs/NEXTALT_gnn_003_egnn_classification_history.json) |
| history plot | [`classification_history.png`](../../../03_training_runs/history_plots/NEXTALT_gnn_003_egnn_classification_history.png) |
| test predictions | [`predictions_test.npz`](../../../04_evaluations/NEXTALT_gnn_003_egnn_test/predictions_test.npz) |
| test results | [`evaluation_test/results.csv`](../../../04_evaluations/NEXTALT_gnn_003_egnn_test/evaluation_test/results.csv) |
| strict metrics | [`evaluation_test/.energybench/metrics.json`](../../../04_evaluations/NEXTALT_gnn_003_egnn_test/evaluation_test/.energybench/metrics.json) |
| ROC | [`energy_matched_roc.png`](../../../04_evaluations/NEXTALT_gnn_003_egnn_test/evaluation_test/energy_matched_roc.png) |
| score-energy plot | [`score_energy_dependence.png`](../../../04_evaluations/NEXTALT_gnn_003_egnn_test/evaluation_test/score_energy_dependence.png) |
| ten-model leaderboard | [`leaderboard.csv`](../../../04_evaluations/NEXTALT_all_models_comparison/leaderboard.csv) |

## Related local files

| Responsibility | File |
|---|---|
| Model configuration | [`config.yaml`](config.yaml) |
| Model training entry point | [`train_classification.py`](train_classification.py) |
| EGNN layer and classifier | [`src/next_alt/models/point_graph.py`](../../../src/next_alt/models/point_graph.py) |
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

- All five layers reconstruct kNN with dense `torch.cdist`, giving compute and intermediate-memory complexity of approximately `O(5BN²)`; this model also has the largest parameter count among these three GNNs.
- The top-k neighbor indices are selected discretely. Ties at equal distances, finite precision, and numerical differences between GPU kernels mean that theoretical equivariance holds only approximately at the numerical level.
- Top-512 truncation may remove low-energy long-range structure.
- Energy normalization and centering remove direct total-energy and absolute-position inputs but do not guarantee complete score independence from energy.
- The coordinate update is this project's normalized-direction variant; this model's result cannot be interpreted directly as the performance limit of the standard EGNN in the original paper.
- This run reached its best validation AUC at epoch 7, after which validation performance declined substantially and training stopped early at epoch 19. The current configuration is sensitive to optimization and regularization.
- The current ranking represents one split, one seed, and one hyperparameter configuration, not a method ranking after exhaustive tuning.
