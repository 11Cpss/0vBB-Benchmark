# POINT-003: Fixed-kNN Residual PointMLP-style Classifier

[中文](README.md) | English

## 1. Identity and hypothesis

| Item | Value |
|---|---|
| `architecture_id` | `point_003_pointmlp` |
| checkpoint `model_name` | `PointMLPClassifier` |
| Python class | `next_alt.models.point_sequence.PointMLPClassifier` |
| registry `input_kind` | `points` |
| Task | NEXT `0nubb` (label 1) versus `Bi214` (label 0) classification |
| Output | One uncalibrated signal logit per event, shape `(B,)` |
| Configuration-derived parameters | **443,521** |

The hypothesis is that a single fixed Euclidean kNN graph, lightweight local geometric messages, and deep residual MLPs can capture the local morphology that separates double-electron signal tracks from single-electron backgrounds, without attention or dynamic graph rebuilding.

## 2. Raw data, labels, and split

The shared reader loads `/MC/hits/table` from every HDF5 file, groups contiguous rows by `event_id`, and uses `x/y/z/energy`. `0nubb_part_*` maps to label 1 and `Bi_part_*` to label 0. The complete relative file path is the `group_id`; a stable seed-42 file-level split assigns 0.8/0.1/0.1 without leaking one source file across splits. This campaign builds only train and validation loaders, selecting at most 100 files per class in each. Validation is used only for early stopping and best-checkpoint selection.

For hits $(\mathbf r_i,e_i)$, compute complete-event energy $E=\sum_i e_i$ and centroid $\mathbf c=\sum_i e_i\mathbf r_i/E$. Merge cells

$$
\mathbf q_i=\lfloor(\mathbf r_i-\mathbf c)/(15\ \mathrm{mm})\rfloor.
$$

The initial voxel center is $(\mathbf q+0.5)15\ \mathrm{mm}$; subtract the voxel-energy-weighted quantized centroid once more. If more than 512 nodes remain, keep the highest-energy 512 with lexicographic cell tie-breaking. Coordinates are divided by 1000 mm and node features are

$$
[e_i^{\mathrm{voxel}}/E,\ \log(1+n_i^{\mathrm{rows}})].
$$

Energy fractions retain the complete-event denominator after truncation. Neither total energy nor absolute detector position is an input.

## 3. Input contract

| Field | dtype / shape | Meaning |
|---|---|---|
| `coords` | float32 `(B,N,3)` | centered 15 mm voxel XYZ divided by 1000 mm |
| `features` | float32 `(B,N,2)` | energy fraction and `log1p(hit_count)` |
| `mask` | bool `(B,N)` | valid nodes versus batch padding |
| output | float `(B,)` | positive values favor `0nubb` |

Here `1 ≤ N ≤ 512`. Padding cannot become a neighbor and is excluded from pooling.

## 4. Layer-by-layer architecture

Defaults are (d=128,k=16,r=2,L=4). kNN is computed once from input coordinates, excludes self-edges, and is shared by all blocks.

| Stage | Operation | Input → output |
|---|---|---|
| Input encoder | `Linear(5,128) → LayerNorm → GELU` | `(B,N,5) → (B,N,128)` |
| Fixed neighborhood | masked `cdist`, self exclusion, top-16 | indices `(B,N,K)` |
| Local branch | concatenate $h_j-h_i,\Delta xyz,\|\Delta xyz\|_2$ | `132 → 128 → 128` |
| Local aggregation | masked neighbor max, `LayerNorm(h + aggregate)` | `(B,N,K,128) → (B,N,128)` |
| Residual FFN | pre-LN, `128 → 256 → 128`, GELU/dropout | `(B,N,128)` |
| Repetition | four local-plus-FFN blocks | `(B,N,128)` |
| Event pooling | concatenate masked mean and max | `(B,N,128) → (B,256)` |
| Head | `256 → 160 → 1`, LN/GELU/dropout | `(B,256) → (B,)` |

The block computes

$$
m_i=\max_{j\in\mathcal N_k(i)}\phi([h_j-h_i,\mathbf r_j-\mathbf r_i,\|\mathbf r_j-\mathbf r_i\|]),
\quad
h_i'=\tilde h_i+\mathrm{FFN}(\mathrm{LN}(\tilde h_i)),
$$

where $\tilde h_i=\mathrm{LN}(h_i+m_i)$. Parameter accounting is 1,024 for the input, 100,224 per block, and 41,601 for the head: 443,521 total.

## 5. Frozen configuration and training

Representation is 15 mm / 1000 mm / 512 points. Model values are `hidden_dim=128`, `num_blocks=4`, `expansion=2`, `k=16`, `classifier_dim=160`, and dropout 0.10. Training uses batch 12, 50 epochs, learning rate $5\times10^{-4}$, weight decay $10^{-4}$, gradient clipping 1.0, patience 12, seed 42, and AMP `auto`.

The shared runner uses `BCEWithLogitsLoss`, AdamW, and `CosineAnnealingLR(T_max=50)`. Best means maximum validation AUC; last is saved separately. The balanced-class path alternates classes; the configured shuffle-buffer size must not be described as event-buffer shuffling that necessarily executes in this path.

## 6. Complexity and memory

The dense distance matrix costs (O(BN^2)) time and temporary memory and is the primary VRAM bottleneck. Block messages cost approximately (O(BLNkd^2)) time and (O(BNkd)) message storage. Reusing one graph saves repeated neighbor searches but prevents feature-dependent neighborhoods.

## 7. Boundary relative to PointMLP

This is **PointMLP-inspired/style**, not a reproduction. It does not implement the paper's hierarchical FPS stages, exact pre/post extraction layout, official geometric-affine equation, augmentation, or training recipe. It instead uses one fixed kNN graph, explicit relative XYZ/distance, masked max, and a mean-plus-max event head. It is pure PyTorch without the official point CUDA operators.

## 8. Running and immutable artifacts

The campaign launches all models serially:

```bash
cd /home/wenyu/summer
RUN_ID=<timestamp-or-explicit-name>
tmux new-session -d -s "next-nontransformer-v2-${RUN_ID}" -n gpu-queue \
  "cd /home/wenyu/summer && bash 01_code/architectures/run_nontransformer_training_queue.sh --run-id ${RUN_ID}"
```

The queue invokes `python 01_code/architectures/point_003_pointmlp/train_classification.py <config.snapshot.yaml>`. Outputs belong under `02_models/checkpoints/<RUN_ID>/point_003_pointmlp/attempt_001/` and `03_training_runs/campaigns/<RUN_ID>/point_003_pointmlp/attempt_001/`. Ctrl-C stops the queue pane. `--resume-queue` skips `DONE`; a failed rerun creates `attempt_002` from scratch. A surviving `last.pt` is not true checkpoint resume, and prior attempts must never be overwritten.

## 9. Training result (filled by the campaign)

| Status | epochs / best epoch | best val AUC / loss | duration | early stop / retry | artifacts |
|---|---|---|---|---|---|
| `PRE-CAMPAIGN PLACEHOLDER` (see appended result) | — | — | — | — | — |

Only train/validation results belong here.

## 10. Limitations

- The 512-node cap can remove low-energy topology; fixed k corresponds to different physical radii at different densities.
- The representation is not rotation invariant or equivariant.
- Dense `cdist` and pure-PyTorch top-k are less memory-efficient than compiled point kernels.
- One fixed graph, model scale, and seed do not represent the full PointMLP family.

## 11. References

1. Xu Ma, Can Qin, Haoxuan You, Haoxi Ran, Yun Fu, “Rethinking Network Design and Local Geometry in Point Cloud: A Simple Residual MLP Framework,” ICLR 2022, arXiv:2202.07123, [OpenReview](https://openreview.net/forum?id=3Pbra-_u76D), [arXiv](https://arxiv.org/abs/2202.07123).
2. NEXT Collaboration (J. Renner et al.), “Background rejection in NEXT using deep neural networks,” *JINST* 12 (2017) T01004, DOI: 10.1088/1748-0221/12/01/T01004, [official article](https://doi.org/10.1088/1748-0221/12/01/T01004).
3. NEXT Collaboration (F. Monrabal et al.), “The NEXT White (NEW) detector,” *JINST* 13 (2018) P12010, DOI: 10.1088/1748-0221/13/12/P12010, [official article](https://doi.org/10.1088/1748-0221/13/12/P12010).


<!-- campaign-result:20260803_200356:start -->
## Campaign `20260803_200356` training result

This section is written after successful serial training. It contains train/validation information only; the test split was not read in this stage.

| Item | Observed value |
|---|---|
| Status / attempt | `DONE` / `attempt_001` |
| Backend | `pytorch` |
| Trainable parameters | 443,521 |
| Trees / tree nodes | N/A / N/A |
| Completed / best epoch | 30 / 18 |
| Best validation AUC | **0.976662** |
| Best validation loss | 0.214011 |
| Training time | 00:27:55 |
| Early stopped | `true` |
| Failed-attempt retry | `no` |
| Python / framework | `3.11.15` / `2.11.0+cu128` |
| Device | `NVIDIA GeForce RTX 5090` |
| Best / last checkpoint | `/home/wenyu/summer/02_models/checkpoints/20260803_200356/point_003_pointmlp/attempt_001/best.pt` / `/home/wenyu/summer/02_models/checkpoints/20260803_200356/point_003_pointmlp/attempt_001/last.pt` |
| Training history | `/home/wenyu/summer/03_training_runs/campaigns/20260803_200356/point_003_pointmlp/attempt_001/history.json` |
<!-- campaign-result:20260803_200356:end -->
