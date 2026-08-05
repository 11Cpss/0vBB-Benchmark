# POINT-001: Deep Sets Permutation-Invariant Set Baseline

[中文](README.md) | English

This model represents a NEXT event as a variable-length set of voxels. A shared MLP
encodes every node independently, and mean/max pooling that is insensitive to node
ordering produces the event representation. It does not construct a graph or perform
neighborhood message passing, making it a low-cost reference baseline for the point-cloud
and graph models.

## 1. Model Identity

| Item | Value |
|---|---|
| `architecture_id` | `point_001_deepsets` |
| checkpoint `model_name` | `DeepSetsClassifier` |
| Python class | `next_alt.models.point_graph.DeepSetsClassifier` |
| registry `input_kind` | `points` |
| Task | Binary classification of NEXT `0nubb` (label 1) versus `Bi214` (label 0) |
| Output | One uncalibrated signal logit per event, shape `(B,)`; larger values favor `0nubb` |
| Trainable parameters | **75,585** |
| Checkpoint format | EnergyBench/NEXT format version 3 |

## 2. Raw Data and Exact Preprocessing

### 2.1 Data loading and splitting

1. Input is read from `/MC/hits/table` in each HDF5 file. The shared reader validates
   `event_id`, `x/y/z/energy`, and the class declared by the file directory, then assembles
   events from contiguous rows with the same `event_id`.
2. Directories matching `0nubb_part_*` map to signal/label 1, while directories matching
   `Bi_part_*` map to background/label 0.
3. The split is not a random event-level split. The complete relative HDF5 path is used as
   the `group_id`, and a stable file-level hash split with seed 42 assigns files to
   train/validation/test at 0.8/0.1/0.1. This prevents one source file from crossing splits.
4. Official training selects at most 100 files per class independently in the train and
   validation splits. The completed official test evaluation has no file limit and uses
   1,490 files containing 115,499 events.

### 2.2 From hits to a set of 15 mm voxels

For the coordinates $\mathbf r_i$ and nonnegative deposited energies $e_i$ of one event:

1. Compute the complete-event total energy $E=\sum_i e_i$ and energy-weighted centroid
   $\mathbf c=\sum_i e_i\mathbf r_i/E$, then translate all coordinates to
   $\mathbf r_i-\mathbf c$.
2. Bin with
   $\mathbf q_i=\lfloor(\mathbf r_i-\mathbf c)/(15\,\mathrm{mm})\rfloor$;
   input rows falling into the same cell are merged into one node.
3. The initial center of a voxel is
   $(\mathbf q+0.5)\times15\,\mathrm{mm}$. The voxel-energy-weighted centroid of these
   quantized centers is then subtracted, removing the common half-bin quantization offset
   while preserving relative distances between nodes.
4. If there are more than 512 nodes, retain the 512 voxels with the largest deposited
   energy. Cell coordinates provide a stable lexicographic tie-breaker. After truncation,
   the energy feature is still divided by the complete-event $E$ and is not renormalized
   over retained nodes, so `point_coverage` reflects the retained energy fraction.
5. Voxel coordinates are divided by 1000 mm before being supplied numerically to the network.

The final batch fields are:

| Field | Shape / dtype | Meaning |
|---|---|---|
| `coords` | `(B, N, 3)`, float32 | Twice-centered voxel XYZ, numerically expressed as coordinate/1000 mm |
| `features[..., 0]` | `(B, N)`, float32 | `voxel_energy / complete_event_energy` |
| `features[..., 1]` | `(B, N)`, float32 | `log1p(number_of_merged_rows)` |
| `mask` | `(B, N)`, bool | True for valid nodes; batches are padded only to that batch's largest node count |
| model output | `(B,)`, float32 | Signal logit; the logits version of BCE is applied internally during training |

Neither total event energy nor absolute detector position is supplied directly to the model.
Total energy is retained only as `energy_condition` in the prediction table for EnergyBench's
energy-matched evaluation.

## 3. Layer-by-Layer Architecture and Dimensions

For each valid node, the three coordinates and two node features are first concatenated into
a five-dimensional vector.

| Stage | Operation | Input → output |
|---|---|---|
| Node input | `concat(coords, features)` | `(B,N,3) + (B,N,2) → (B,N,5)` |
| Shared node encoder 1 | `Linear + LayerNorm + SiLU` | `5 → 128` |
| Shared node encoder 2 | `Linear + LayerNorm + SiLU` | `128 → 192` |
| Padding handling | Zero invalid nodes using `mask` | `(B,N,192) → (B,N,192)` |
| Event pooling | Concatenate masked mean and masked max | `(B,N,192) → (B,384)` |
| Classifier hidden | `Linear + SiLU + Dropout(0.10)` | `384 → 128` |
| Classifier output | `Linear` | `128 → 1` |

The forward pass can be summarized as

$$
\mathrm{logit}=\rho\!\left(
  \operatorname{mean}_{i\in V}\phi(\mathbf x_i)
  \;\Vert\;
  \operatorname{max}_{i\in V}\phi(\mathbf x_i)
\right).
$$

The shared encoder and symmetric pooling make the output independent of node ordering.
Different nodes do not exchange information before pooling.

## 4. Relationship to the Original Deep Sets Method

This implementation follows the Deep Sets principle of a shared element-wise mapping,
symmetric aggregation, and a set-level mapping, but it is not a layer-by-layer reproduction
of the experimental networks in the original paper:

- the aggregator concatenates **masked mean and masked max**, rather than using only sum pooling;
- element inputs are the centered voxel coordinates, energy fraction, and merged-hit count
  defined by this project;
- LayerNorm, SiLU, and dropout in the classification head are added;
- each event retains at most 512 high-energy voxels;
- this is a binary-classification-specific implementation and does not include the paper's
  other set-learning tasks or generative components;
- the code is pure PyTorch and does not depend on `torch_geometric`, DGL, or compiled scatter
  extensions.

The documentation and results should therefore call this a “Deep Sets baseline/variant,” not
a literal reproduction of the paper's model.

## 5. Key Configuration

The following values come from this directory's `config.yaml`. Output defaults not specified
explicitly are supplied by the shared configuration module.

| Category | Parameter | Value |
|---|---|---:|
| Representation | `point_bin_size` | 15.0 mm |
| Representation | `coordinate_scale` | 1000.0 mm |
| Representation | `max_points` | 512 |
| Model | `feature_dim` | 2 |
| Model | `hidden_dim` | 128 |
| Model | `embedding_dim` | 192 |
| Model | `classifier_dim` | 128 |
| Model | `dropout` | 0.10 |
| Data | `max_files_per_class` | 100 |
| Data | `split_seed` / `split_fractions` | 42 / `[0.8, 0.1, 0.1]` |
| Data | `balance_training_classes` | true |
| Data | `event_shuffle_buffer_size` | 512 (see the actual mechanism below) |
| Data | `num_workers` | 0 |
| Training | `batch_size` | 64 |
| Training | `epochs` | 50 |
| Training | `learning_rate` | 1e-3 |
| Training | `weight_decay` | 1e-4 |
| Training | `gradient_clip_norm` | 1.0 |
| Training | `early_stopping_patience` / `min_delta` | 12 / 0.0 |
| Training | `seed` / `deterministic` | 42 / false |
| Training | AMP | `auto`; the official checkpoint records bfloat16 |

## 6. Training Mechanism and Command

- loss: `BCEWithLogitsLoss`;
- optimizer: AdamW;
- scheduler: `CosineAnnealingLR(T_max=50)`, stepped once per epoch;
- global gradient norm is clipped to 1.0 before every parameter update;
- source files are shuffled each epoch; with `balance_training_classes=true`, signal and
  background events are read alternately, and the epoch ends when either class is exhausted;
- the current balanced-class branch does not pass through the bounded event shuffle buffer.
  Consequently, YAML's `event_shuffle_buffer_size=512` is a recorded configuration value and
  should not be described as buffer-level event shuffling that actually occurred in this run;
- the best checkpoint is selected by validation AUC, with patience 12 controlling early stopping;
- training requires `cuda:0`, with no CPU fallback and no smoke-training branch.

The standard command from the repository root is:

```bash
cd /home/wenyu/summer
source .venv/bin/activate
python 01_code/architectures/point_001_deepsets/train_classification.py
```

The current best/last checkpoints, history CSV/JSON, and training curve already exist. The
shared runner defaults to `allow_overwrite=false`, so rerunning the command above directly
will refuse to overwrite existing artifacts before training begins. For an independent
retraining run, copy the YAML file, configure new output directories for checkpoints, logs,
and plots in the copy, and pass that YAML path as the training script's sole positional argument.

## 7. Completed Training and Official Evaluation Results

The best checkpoint corresponds to epoch 45. Its AUC on 16,786 validation events is
**0.923009**, with mean representation coverage of 1.0.

The official results use the best checkpoint, the complete test split, and a strict manifest,
with no warnings or errors:

| Metric | Result |
|---|---:|
| Classification rank among the 10 alternative architectures | **8 / 10** |
| Energy-matched AUC | **0.920475** |
| Inclusive AUC | **0.920727** |
| Energy-independence score (mean) | **0.974339** |
| Test files / events | 1,490 / 115,499 |

This checkpoint performs classification only. Therefore, energy regression appearing as
`not_applicable` is expected and is not an evaluation failure.

### Rerunning the complete test evaluation in a new directory

The command below deliberately uses an unused `_rerun` output directory and will not overwrite
the official results. Do not add `--max-files-per-class`, because that would no longer be a
complete test evaluation.

```bash
cd /home/wenyu/summer
source .venv/bin/activate
energybench next \
  02_models/checkpoints/NEXTALT_point_001_deepsets_classification_best.pt \
  --split test \
  --device cuda:0 \
  --output-dir 04_evaluations/NEXTALT_point_001_deepsets_test_rerun
```

## 8. Existing Artifacts

| Artifact | Path |
|---|---|
| Best checkpoint | [`NEXTALT_point_001_deepsets_classification_best.pt`](../../../02_models/checkpoints/NEXTALT_point_001_deepsets_classification_best.pt) |
| Last checkpoint | [`NEXTALT_point_001_deepsets_classification_last.pt`](../../../02_models/checkpoints/NEXTALT_point_001_deepsets_classification_last.pt) |
| Epoch metrics | [`..._epochs.csv`](../../../03_training_runs/logs/NEXTALT_point_001_deepsets_classification_epochs.csv) |
| Complete history | [`..._history.json`](../../../03_training_runs/logs/NEXTALT_point_001_deepsets_classification_history.json) |
| Training curve | [`..._history.png`](../../../03_training_runs/history_plots/NEXTALT_point_001_deepsets_classification_history.png) |
| Official test predictions | [`predictions_test.npz`](../../../04_evaluations/NEXTALT_point_001_deepsets_test/predictions_test.npz) |
| Official test summary | [`results.csv`](../../../04_evaluations/NEXTALT_point_001_deepsets_test/evaluation_test/results.csv) |
| Complete evaluation metrics | [`metrics.json`](../../../04_evaluations/NEXTALT_point_001_deepsets_test/evaluation_test/.energybench/metrics.json) |
| Matched ROC plot | [`energy_matched_roc.png`](../../../04_evaluations/NEXTALT_point_001_deepsets_test/evaluation_test/energy_matched_roc.png) |
| Score–energy plot | [`score_energy_dependence.png`](../../../04_evaluations/NEXTALT_point_001_deepsets_test/evaluation_test/score_energy_dependence.png) |
| Ten-model leaderboard | [`leaderboard.csv`](../../../04_evaluations/NEXTALT_all_models_comparison/leaderboard.csv) |

## 9. Related Local Files

| File | Role |
|---|---|
| [`config.yaml`](./config.yaml) | Official training hyperparameters for this model |
| [`train_classification.py`](./train_classification.py) | Training entry point with a fixed architecture ID |
| [`src/next_alt/models/point_graph.py`](../../../src/next_alt/models/point_graph.py) | Authoritative implementation of `DeepSetsClassifier`, masked pooling, and point-input checks |
| [`src/next_alt/data.py`](../../../src/next_alt/data.py) | 15 mm voxelization, truncation, representation conversion, and padded collation |
| [`src/next_cnn/data.py`](../../../src/next_cnn/data.py) | HDF5 schema, labels, file-level splitting, and event loading |
| [`src/next_alt/registry.py`](../../../src/next_alt/registry.py) | Registration of architecture ID, class, and `input_kind` |
| [`src/next_alt/training.py`](../../../src/next_alt/training.py) | Loss, optimizer, scheduler, AMP, best-checkpoint selection, and training loop |
| [`src/next_alt/checkpoint.py`](../../../src/next_alt/checkpoint.py) | Format-v3 checkpoint contract |
| [`src/next_alt/adapter.py`](../../../src/next_alt/adapter.py) | Checkpoint-driven EnergyBench inference adapter |
| [`docs/ALTERNATIVE_ARCHITECTURES.md`](../../../docs/ALTERNATIVE_ARCHITECTURES.md) | Overall description of the ten alternative architectures |
| [`docs/ALTERNATIVE_EVALUATION_RESULTS.md`](../../../docs/ALTERNATIVE_EVALUATION_RESULTS.md) | Unified official evaluation contract and leaderboard |

This directory has no separate `model.py`; both training and inference use the
`src/next_alt/models/point_graph.py` file listed above, preventing architectural drift between
the two paths.

## 10. Original Method Reference

- Manzil Zaheer et al., [*Deep Sets*](https://arxiv.org/abs/1703.06114),
  NeurIPS 2017.

This citation identifies the methodological source. The specific inputs, mean+max aggregation,
and classification head used here are defined by this repository's implementation.

## 11. Limitations

- Independent node encoding cannot explicitly learn voxel adjacency, local curvature,
  branching, or track connectivity;
- events with more than 512 voxels lose low-energy nodes, although coverage is recorded;
- energy-fraction input removes absolute total energy and may discard useful energy information;
- although coordinates are centered, the network reads XYZ directly and therefore has no
  guarantee of rotational invariance or E(3) equivariance;
- the test rank applies only to the current data version, preprocessing, training budget, and
  single seed. It is not a general upper bound for the model family, and no multi-seed variance
  is reported.
