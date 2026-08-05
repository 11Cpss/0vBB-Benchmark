# POINT-002: Lightweight PointNet++-Style Hierarchical Point-Cloud Network

[中文](README.md) | English

This model adds explicit local scales to the same centered voxel point set used by POINT-001.
It first selects centers with farthest-point sampling (FPS), then encodes and pools each
center's geometric neighbors, progressively reducing at most 512 input nodes to at most 64
centers and then at most 16 centers.

## 1. Model Identity

| Item | Value |
|---|---|
| `architecture_id` | `point_002_pointnetpp` |
| checkpoint `model_name` | `PointNetPPClassifier` |
| Python class | `next_alt.models.point_graph.PointNetPPClassifier` |
| registry `input_kind` | `points` |
| Task | Binary classification of NEXT `0nubb` (label 1) versus `Bi214` (label 0) |
| Output | One uncalibrated signal logit per event, shape `(B,)`; larger values favor `0nubb` |
| Trainable parameters | **164,513** |
| Checkpoint format | EnergyBench/NEXT format version 3 |

## 2. Raw Data and Exact Preprocessing

### 2.1 Data loading and splitting

1. Input is read from `/MC/hits/table` in each HDF5 file. The shared reader validates
   `event_id`, `x/y/z/energy`, and the class declared by the file directory, then assembles
   events from contiguous rows with the same `event_id`.
2. Directories matching `0nubb_part_*` map to signal/label 1, while directories matching
   `Bi_part_*` map to background/label 0.
3. The complete relative HDF5 path is used as the `group_id`. A stable file-level hash split
   with seed 42 assigns files to train/validation/test at 0.8/0.1/0.1. The same source file
   never crosses splits.
4. Official training selects at most 100 files per class independently in the train and
   validation splits. The official test evaluation has no file limit and uses 1,490 files
   containing 115,499 events.

### 2.2 From hits to a set of 15 mm voxels

For coordinates $\mathbf r_i$ and deposited energies $e_i$:

1. Use the complete-event total energy $E=\sum_i e_i$ to compute the energy-weighted centroid
   $\mathbf c=\sum_i e_i\mathbf r_i/E$, then translate the event to that centroid.
2. Merge input rows in the same cell using
   $\mathbf q_i=\lfloor(\mathbf r_i-\mathbf c)/(15\,\mathrm{mm})\rfloor$.
3. The initial voxel center is $(\mathbf q+0.5)\times15\,\mathrm{mm}$. The
   voxel-energy-weighted centroid of the quantized centers is then subtracted to remove the
   common half-bin offset.
4. If there are more than 512 nodes, select the 512 voxels with the highest energy, using cell
   coordinates as a stable lexicographic tie-breaker. Node energy fractions continue to use
   complete-event energy as their denominator and are not renormalized after truncation.
5. Center coordinates are divided by 1000 mm. Each node also receives two features:
   `voxel_energy / complete_event_energy` and `log1p(merged_row_count)`.

The final inputs are:

| Field | Shape / dtype | Meaning |
|---|---|---|
| `coords` | `(B, N, 3)`, float32 | Centered voxel XYZ divided by 1000 mm |
| `features` | `(B, N, 2)`, float32 | Energy fraction and `log1p(hit_count)` |
| `mask` | `(B, N)`, bool | Excludes batch padding; the valid node count may differ by event |
| model output | `(B,)`, float32 | Signal logit |

Total event energy and absolute detector position do not enter the model. Total energy is kept
only as `energy_condition` in the output table for matched evaluation.

## 3. Layer-by-Layer Architecture and Dimensions

The internal order of the shared `_mlp(a,b,c,p)` helper is
`Linear(a,b) → LayerNorm(b) → SiLU → Dropout(p) → Linear(b,c) → SiLU`.

| Stage | Operation | Input → output |
|---|---|---|
| Node input | Concatenate XYZ and two features | `(B,N,5)` |
| Input encoder | `_mlp(5,96,96,0.10)` | `(B,N,5) → (B,N,96)` |
| FPS-1 | Deterministic farthest-point sampling | At most `N=512 → Q1=64` centers |
| Local group-1 | Up to 16 Euclidean kNN support points per center | `96 + ΔXYZ(3) + distance(1) = 100` message-input dimensions |
| Set abstraction-1 | `_mlp(100,128,128,0.10)` + masked neighbor max | `(B,Q1,16,100) → (B,Q1,128)` |
| FPS-2 | Deterministic FPS over first-stage centers | `Q1≤64 → Q2≤16` |
| Local group-2 | Up to 16 kNN support points per second-stage center | `128 + 3 + 1 = 132` message-input dimensions |
| Set abstraction-2 | `_mlp(132,192,192,0.10)` + masked neighbor max | `(B,Q2,16,132) → (B,Q2,192)` |
| Event pooling | Concatenate masked mean and max over valid second-stage centers | `(B,Q2,192) → (B,384)` |
| Classifier hidden | `Linear + SiLU + Dropout(0.10)` | `384 → 160` |
| Classifier output | `Linear` | `160 → 1` |

FPS is deterministic: its first step selects the point farthest from the arithmetic mean of
the valid support nodes, and subsequent steps repeatedly select the point farthest from the
set already chosen. If an event has fewer than 64 or 16 nodes, `sampled_mask` prevents filler
centers from entering later pooling. Local grouping uses fixed-`k` kNN, and the query center
itself may be one of its support neighbors.

## 4. Differences from the Original PointNet++ Method

This implementation retains the core PointNet++ ideas—FPS, local grouping, shared local
networks, symmetric pooling, and hierarchical downsampling—but it is a lightweight,
classification-only **PointNet++-style** variant:

- the original paper primarily uses radius/ball query and introduces multi-scale grouping to
  handle density variation. This implementation has one fixed `k=16` Euclidean kNN scale per
  layer, with no radius cutoff or MSG;
- there is no feature-propagation/upsampling path for segmentation;
- local messages explicitly concatenate relative XYZ and Euclidean distance;
- the event-level head concatenates mean and max rather than claiming to reproduce a particular
  classification head from the paper;
- point sets come from this project's 15 mm energy voxelization and are capped at 512 nodes;
- FPS, kNN, and aggregation are implemented in pure PyTorch, without the custom CUDA point
  operators commonly used with the paper and without `torch_geometric` or DGL.

The results can therefore be attributed only to this lightweight implementation and cannot be
treated as a reproduction of the standard official PointNet++ implementation.

## 5. Key Configuration

| Category | Parameter | Value |
|---|---|---:|
| Representation | `point_bin_size` | 15.0 mm |
| Representation | `coordinate_scale` | 1000.0 mm |
| Representation | `max_points` | 512 |
| Model | `feature_dim` | 2 |
| Model | `hidden_dim` | 96 |
| Model | `stage1_dim` / `stage2_dim` | 128 / 192 |
| Model | `stage1_points` / `stage2_points` | 64 / 16 |
| Model | `k` | 16 |
| Model | `classifier_dim` | 160 |
| Model | `dropout` | 0.10 |
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

- loss: `BCEWithLogitsLoss`; optimizer: AdamW;
- scheduler: `CosineAnnealingLR(T_max=50)`, stepped every epoch;
- gradient norm clipping at 1.0;
- source files are shuffled each epoch, and balanced mode alternates signal and background
  events until either class is exhausted;
- the current `balance_training_classes=true` path does not invoke the bounded event shuffle
  buffer. Therefore, `event_shuffle_buffer_size=512` in the configuration should not be
  described as event shuffling that actually took place;
- the best checkpoint is selected by validation AUC, with patience 12;
- official training uses `cuda:0` and AMP, with no CPU fallback or smoke branch.

Standard entry point:

```bash
cd /home/wenyu/summer
source .venv/bin/activate
python 01_code/architectures/point_002_pointnetpp/train_classification.py
```

This model's checkpoints and all history artifacts already exist. Because
`allow_overwrite=false` by default, rerunning the command directly will refuse to overwrite
them. For an independent retraining run, copy the YAML file, configure new checkpoint/log/plot
output directories, and pass the copied file's path as the training script's positional argument.

## 7. Completed Training and Official Evaluation Results

The best checkpoint comes from epoch 48. Its AUC on 16,786 validation events is
**0.953117**, with mean representation coverage of 1.0.

The strict evaluation of the best checkpoint on the complete test split gives:

| Metric | Result |
|---|---:|
| Classification rank among the 10 alternative architectures | **5 / 10** |
| Energy-matched AUC | **0.953342** |
| Inclusive AUC | **0.953552** |
| Energy-independence score (mean) | **0.977201** |
| Test files / events | 1,490 / 115,499 |

The official evaluation has 0 warnings and 0 errors. This checkpoint outputs only a
classification logit, so `not_applicable` for energy regression is expected.

### Rerunning the complete test evaluation in a new directory

```bash
cd /home/wenyu/summer
source .venv/bin/activate
energybench next \
  02_models/checkpoints/NEXTALT_point_002_pointnetpp_classification_best.pt \
  --split test \
  --device cuda:0 \
  --output-dir 04_evaluations/NEXTALT_point_002_pointnetpp_test_rerun
```

`_rerun` is a new output directory. Do not add `--max-files-per-class`, because the result
would no longer correspond to the complete test.

## 8. Existing Artifacts

| Artifact | Path |
|---|---|
| Best checkpoint | [`NEXTALT_point_002_pointnetpp_classification_best.pt`](../../../02_models/checkpoints/NEXTALT_point_002_pointnetpp_classification_best.pt) |
| Last checkpoint | [`NEXTALT_point_002_pointnetpp_classification_last.pt`](../../../02_models/checkpoints/NEXTALT_point_002_pointnetpp_classification_last.pt) |
| Epoch metrics | [`..._epochs.csv`](../../../03_training_runs/logs/NEXTALT_point_002_pointnetpp_classification_epochs.csv) |
| Complete history | [`..._history.json`](../../../03_training_runs/logs/NEXTALT_point_002_pointnetpp_classification_history.json) |
| Training curve | [`..._history.png`](../../../03_training_runs/history_plots/NEXTALT_point_002_pointnetpp_classification_history.png) |
| Official test predictions | [`predictions_test.npz`](../../../04_evaluations/NEXTALT_point_002_pointnetpp_test/predictions_test.npz) |
| Official test summary | [`results.csv`](../../../04_evaluations/NEXTALT_point_002_pointnetpp_test/evaluation_test/results.csv) |
| Complete evaluation metrics | [`metrics.json`](../../../04_evaluations/NEXTALT_point_002_pointnetpp_test/evaluation_test/.energybench/metrics.json) |
| Matched ROC plot | [`energy_matched_roc.png`](../../../04_evaluations/NEXTALT_point_002_pointnetpp_test/evaluation_test/energy_matched_roc.png) |
| Score–energy plot | [`score_energy_dependence.png`](../../../04_evaluations/NEXTALT_point_002_pointnetpp_test/evaluation_test/score_energy_dependence.png) |
| Ten-model leaderboard | [`leaderboard.csv`](../../../04_evaluations/NEXTALT_all_models_comparison/leaderboard.csv) |

## 9. Related Local Files

| File | Role |
|---|---|
| [`config.yaml`](./config.yaml) | Official training configuration for this model |
| [`train_classification.py`](./train_classification.py) | Entry point with a fixed architecture ID |
| [`src/next_alt/models/point_graph.py`](../../../src/next_alt/models/point_graph.py) | Authoritative implementation of FPS, set abstraction, kNN, and `PointNetPPClassifier` |
| [`src/next_alt/data.py`](../../../src/next_alt/data.py) | Voxelization, node truncation, representation conversion, and padding |
| [`src/next_cnn/data.py`](../../../src/next_cnn/data.py) | HDF5 loading, labels, and file-level splitting |
| [`src/next_alt/registry.py`](../../../src/next_alt/registry.py) | Architecture/class/input-kind registration |
| [`src/next_alt/training.py`](../../../src/next_alt/training.py) | Shared training loop, optimizer, and best-checkpoint selection |
| [`src/next_alt/checkpoint.py`](../../../src/next_alt/checkpoint.py) | Format-v3 checkpoint contract |
| [`src/next_alt/adapter.py`](../../../src/next_alt/adapter.py) | Official inference and EnergyBench interface |
| [`docs/ALTERNATIVE_ARCHITECTURES.md`](../../../docs/ALTERNATIVE_ARCHITECTURES.md) | Overview of the alternative architectures |
| [`docs/ALTERNATIVE_EVALUATION_RESULTS.md`](../../../docs/ALTERNATIVE_EVALUATION_RESULTS.md) | Complete-test evaluation contract and leaderboard |

This directory has no separate `model.py`; training and inference both construct the same
class from `src/next_alt/models/point_graph.py`.

## 10. Original Method References

- Charles R. Qi et al.,
  [*PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space*](https://arxiv.org/abs/1706.02413),
  NeurIPS 2017.
- Charles R. Qi et al.,
  [*PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation*](https://arxiv.org/abs/1612.00593),
  CVPR 2017. This is the precursor method for shared point-wise networks and symmetric pooling.

## 11. Limitations

- fixed-`k` kNN does not impose an explicit physical radius; density changes alter the actual
  neighborhood scale;
- the two FPS stages and dense `torch.cdist` cost more time and memory than simple Deep Sets,
  and no high-performance point-ops CUDA extension is used;
- truncation at 512 nodes may discard low-energy topology details;
- although the input is event-centered, the model directly uses XYZ and relative XYZ and has
  no guarantee of rotational invariance or equivariance;
- single-scale grouping and a classification head do not represent the full capabilities of
  PointNet++;
- the current rank comes from one seed, one model size, and a fixed budget, with no multi-seed
  uncertainty estimate.
