# GNN-002: ParticleNet-lite EdgeConv

[中文](README.md) | English

This directory defines a dynamic-graph classifier for the NEXT `0nubb` versus `Bi214` binary classification task. It draws on the EdgeConv ideas of ParticleNet and Dynamic Graph CNN (DGCNN), but its network widths, edge features, residual blocks, event pooling, and training protocol are all custom implementations for this project rather than a component-by-component reproduction of the original papers' code.

## Model identity

| Item | Value |
|---|---|
| `architecture_id` | `gnn_002_particlenet_edgeconv` |
| PyTorch class | `next_alt.models.point_graph.ParticleNetLiteClassifier` |
| `input_kind` | `graph` |
| Task | Binary classification; `0nubb=1`, `Bi214=0` |
| Output | One uncalibrated signal logit per event, shape `(B,)` |
| Trainable parameters | 150,497 |
| Checkpoint format | EnergyBench/NEXT format version 3 |

The implementation uses only PyTorch tensor operations. It does not depend on PyG, DGL, `torch_scatter`, or custom CUDA neighbor operators.

## Input and exact preprocessing

For an event, denote the coordinates and deposited energies of its raw rows by
$\{(\mathbf r_a,e_a)\}$, with total energy $E=\sum_a e_a$. Preprocessing is performed exactly by the shared data module:

1. Compute the energy-weighted center of the raw rows,
   $\mathbf c=\sum_a e_a\mathbf r_a/E$, and translate the coordinates to
   $\mathbf r_a-\mathbf c$.
2. Aggregate voxels with a `15 mm` side length:
   $\mathbf q_a=\lfloor(\mathbf r_a-\mathbf c)/15\,\mathrm{mm}\rfloor$.
3. For each occupied voxel, obtain its deposited energy
   $E_i=\sum_{a:q_a=i}e_a$ and the number of merged rows $n_i$.
4. Start from the geometric voxel center $(\mathbf q_i+0.5)\times15\,\mathrm{mm}$, then subtract the energy-weighted center of the quantized voxels to remove the half-cell quantization offset.
5. The model coordinates are
   $\mathbf x_i=\mathbf x_i^{\mathrm{centered}}/1000.0$: the centered coordinates in mm are divided by `coordinate_scale=1000.0`, not multiplied by 1000.
6. The two node features are
   $\mathbf f_i=[E_i/E,\ \log(1+n_i)]$, corresponding to `energy_fraction` and `log1p(hit_count)`.
7. If there are more than 512 occupied voxels, retain the 512 with the highest $E_i$, using XYZ as a deterministic tie-breaker. The truncated `energy_fraction` values are not renormalized, so `point_coverage` reflects the actual fraction of energy retained.
8. Within each batch, zero-pad to the largest node count in that batch, producing `coords: (B,N,3)`, `features: (B,N,2)`, and a boolean `mask: (B,N)`. The mask remains active during graph construction, message aggregation, and event pooling.

The total event energy and absolute detector position are not fed directly to the model. Total energy is exported with the predictions only as EnergyBench's `energy_condition`. Events with fewer than 17 valid nodes, including single-node events, are handled safely by the neighbor mask.

## Network architecture

### 1. Node encoder

Each node first concatenates its coordinates and two scalar features:

```text
[x, y, z, energy_fraction, log1p(hit_count)]  # 5 dims
    -> Linear(5, 64)
    -> LayerNorm(64) -> SiLU -> Dropout(0.10)
    -> Linear(64, 64) -> SiLU
```

### 2. Dynamic EdgeConv

For center node $i$ and neighbor $j$, the edge input is

$$
\mathbf z_{ij}=
[\mathbf h_i,\ \mathbf h_j-\mathbf h_i,\
 \mathbf x_j-\mathbf x_i,\
 \lVert\mathbf x_j-\mathbf x_i\rVert_2].
$$

After the edge MLP, messages are aggregated with a masked maximum over neighbors and the node is updated through a residual branch:

$$
\mathbf h_i' = \operatorname{LayerNorm}\!\left(
W_{\rm skip}\mathbf h_i+
\operatorname{Dropout}\left[
\max_{j\in\mathcal N(i)}\operatorname{MLP}(\mathbf z_{ij})
\right]\right).
$$

| Stage | Node dimension | kNN search space | Edge input dimension | Output dimension |
|---:|---:|---|---:|---:|
| 1 | 64 | Physical coordinates `coords` | `2×64+4=132` | 64 |
| 2 | 64 | Current `nodes / sqrt(64)` | `2×64+4=132` | 96 |
| 3 | 96 | Current `nodes / sqrt(96)` | `2×96+4=196` | 128 |

Every layer uses self-excluding kNN with `k=16`. The first layer uses a geometric graph, while the next two reconstruct the graph in learned feature space. Importantly, the learned space in the latter two layers is used only to select neighbors; the relative XYZ values and Euclidean distances supplied to the edge messages still come from physical coordinates.

### 3. Event pooling and classification head

```text
masked_mean(nodes) || masked_max(nodes)  # 128 + 128 = 256
    -> Linear(256, 192) -> SiLU -> Dropout(0.10)
    -> Linear(192, 1)
    -> squeeze -> (B,)
```

Mean/max pooling makes the output insensitive to node ordering.

## Relationship to the original methods

- [ParticleNet: Jet Tagging via Particle Clouds](https://arxiv.org/abs/1902.08570) provides the main motivation for treating particle clouds as dynamic graphs for high-energy-physics classification.
- [Dynamic Graph CNN for Learning on Point Clouds](https://arxiv.org/abs/1801.07829) introduced EdgeConv and the basic approach of reconstructing neighborhoods in feature space layer by layer.
- This project uses 15 mm NEXT voxels, two project-specific node features, three `[64,96,128]` residual EdgeConv layers, explicit relative XYZ/distance features, mean+max event pooling, and a custom classification head.
- The implementation does not reproduce ParticleNet's complete block depth, cross-block feature combination, training setup, or official accelerated operators. It should therefore be described in documentation and papers as **ParticleNet-lite** or **ParticleNet-inspired EdgeConv**, not as a reproduction of ParticleNet.

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
| Model | `hidden_dims` | `[64, 96, 128]` |
| Model | `k` | 16 |
| Model | `classifier_dim` | 192 |
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

For the complete values, see the adjacent [`config.yaml`](config.yaml). Shared defaults and validation rules are defined in [`src/next_alt/config.py`](../../../src/next_alt/config.py).

## Training behavior and commands

- Complete HDF5 files are assigned to train/validation/test through a stable hash, preventing a source file from crossing split boundaries.
- The training stream alternates signal and background events and stops when the smaller class is exhausted, keeping the training sample balanced.
- Although `event_shuffle_buffer_size=512` is present in the YAML, event-buffer shuffling is not enabled in the current `balance_training_classes=true` branch; in practice, source-file order is shuffled once per epoch.
- The loss is `BCEWithLogitsLoss`, the optimizer is AdamW, and the scheduler is a 50-epoch CosineAnnealingLR.
- The best checkpoint is saved only when validation AUC strictly improves, while the last checkpoint is overwritten every epoch. Training stops early after 12 consecutive epochs without improvement.
- CUDA is a hard requirement for formal training; no CPU fallback or smoke-training parameter is provided.

The original formal training entry point was:

```bash
cd /home/wenyu/summer
source .venv/bin/activate
python 01_code/architectures/gnn_002_particlenet_edgeconv/train_classification.py
```

The checkpoints, history, and plots from that formal run already exist. Because `allow_overwrite: false` by default, directly repeating the command above now refuses to overwrite the existing outputs and exits; this is the intended protection. For a new experiment, copy the YAML, set new directories for `output.checkpoint_dir`, `output.log_dir`, and `output.plot_dir`, and pass the new YAML as the sole positional argument:

```bash
python 01_code/architectures/gnn_002_particlenet_edgeconv/train_classification.py \
  /path/to/gnn_002_new_experiment.yaml
```

## Completed formal results

| Metric | Result |
|---|---:|
| best epoch | 47 |
| best validation AUC | 0.971676 |
| epochs actually completed | 50 |
| overall full-test rank | **2 / 10** |
| full-test matched AUC | **0.970363** |
| full-test inclusive AUC | 0.970754 |
| energy-independence mean | 0.979700 |

The full test used the complete test split: 1,490 files and 115,499 events. Strict evaluation produced 0 warnings and 0 errors. The formal results already exist and must not be overwritten. If a rerun is genuinely required, use a new `_rerun` directory and do not pass `--max-files-per-class`:

```bash
cd /home/wenyu/summer
source .venv/bin/activate
energybench next \
  02_models/checkpoints/NEXTALT_gnn_002_particlenet_edgeconv_classification_best.pt \
  --device cuda:0 \
  --split test \
  --batch-size 12 \
  --num-workers 0 \
  --model-id gnn_002_particlenet_edgeconv \
  --output-dir 04_evaluations/NEXTALT_gnn_002_particlenet_edgeconv_test_rerun
```

## Artifacts

| Type | File |
|---|---|
| best checkpoint | [`NEXTALT_gnn_002_particlenet_edgeconv_classification_best.pt`](../../../02_models/checkpoints/NEXTALT_gnn_002_particlenet_edgeconv_classification_best.pt) |
| last checkpoint | [`NEXTALT_gnn_002_particlenet_edgeconv_classification_last.pt`](../../../02_models/checkpoints/NEXTALT_gnn_002_particlenet_edgeconv_classification_last.pt) |
| epoch CSV | [`classification_epochs.csv`](../../../03_training_runs/logs/NEXTALT_gnn_002_particlenet_edgeconv_classification_epochs.csv) |
| history JSON | [`classification_history.json`](../../../03_training_runs/logs/NEXTALT_gnn_002_particlenet_edgeconv_classification_history.json) |
| history plot | [`classification_history.png`](../../../03_training_runs/history_plots/NEXTALT_gnn_002_particlenet_edgeconv_classification_history.png) |
| test predictions | [`predictions_test.npz`](../../../04_evaluations/NEXTALT_gnn_002_particlenet_edgeconv_test/predictions_test.npz) |
| test results | [`evaluation_test/results.csv`](../../../04_evaluations/NEXTALT_gnn_002_particlenet_edgeconv_test/evaluation_test/results.csv) |
| strict metrics | [`evaluation_test/.energybench/metrics.json`](../../../04_evaluations/NEXTALT_gnn_002_particlenet_edgeconv_test/evaluation_test/.energybench/metrics.json) |
| ROC | [`energy_matched_roc.png`](../../../04_evaluations/NEXTALT_gnn_002_particlenet_edgeconv_test/evaluation_test/energy_matched_roc.png) |
| score-energy plot | [`score_energy_dependence.png`](../../../04_evaluations/NEXTALT_gnn_002_particlenet_edgeconv_test/evaluation_test/score_energy_dependence.png) |
| ten-model leaderboard | [`leaderboard.csv`](../../../04_evaluations/NEXTALT_all_models_comparison/leaderboard.csv) |

## Related local files

| Responsibility | File |
|---|---|
| Model configuration | [`config.yaml`](config.yaml) |
| Model training entry point | [`train_classification.py`](train_classification.py) |
| ParticleNet-lite / EdgeConv implementation | [`src/next_alt/models/point_graph.py`](../../../src/next_alt/models/point_graph.py) |
| Voxelization, padding, and loader | [`src/next_alt/data.py`](../../../src/next_alt/data.py) |
| Architecture ID and input-kind registration | [`src/next_alt/registry.py`](../../../src/next_alt/registry.py) |
| Shared training loop | [`src/next_alt/training.py`](../../../src/next_alt/training.py) |
| v3 checkpoint contract | [`src/next_alt/checkpoint.py`](../../../src/next_alt/checkpoint.py) |
| v3 inference adapter | [`src/next_alt/adapter.py`](../../../src/next_alt/adapter.py) |
| HDF5 reader and file-level split | [`src/next_cnn/data.py`](../../../src/next_cnn/data.py) |
| All-architecture documentation | [`docs/ALTERNATIVE_ARCHITECTURES.md`](../../../docs/ALTERNATIVE_ARCHITECTURES.md) |
| Formal evaluation results | [`docs/ALTERNATIVE_EVALUATION_RESULTS.md`](../../../docs/ALTERNATIVE_EVALUATION_RESULTS.md) |
| EnergyBench usage instructions | [`docs/USAGE_GUIDE.md`](../../../docs/USAGE_GUIDE.md) |

## Limitations and interpretation boundaries

- kNN uses dense `torch.cdist`, with time and intermediate-memory complexity of approximately `O(BN²)`; `N=512` is an important resource cap.
- The top-512 strategy prioritizes high-energy voxels and may discard low-energy branches that are nevertheless topologically meaningful.
- The discrete kNN neighbor selection is not differentiable; only the continuous network parameters on selected edges participate in backpropagation.
- Energy normalization and removal of absolute position reduce direct energy/position shortcuts, but hit count, geometric extent, and truncation rate may still correlate with energy. Input design alone therefore cannot establish complete energy independence.
- The current result comes from one data split, one seed, and one hyperparameter configuration. The ten-model ranking is not equivalent to the theoretical limits of the methods after equally extensive tuning.
- The model output is a logit, not a calibrated posterior probability.
