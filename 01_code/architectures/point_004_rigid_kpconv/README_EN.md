# POINT-004: Pure-PyTorch Rigid KPConv-Style Classifier

[中文](README.md) | English

## 1. Positioning and hypothesis

POINT-004 tests whether fixed 3-D kernel points provide a useful geometric bias for NEXT
two-electron tracks and blobs compared with ordinary point MLPs. It contains no Transformer
or attention.

| Item | Definition |
|---|---|
| `architecture_id` | `point_004_rigid_kpconv` |
| checkpoint `model_name` | `RigidKPConvClassifier` |
| Python class | `next_alt.models.mixer_sparse.RigidKPConvClassifier` |
| registry `input_kind` | `points` |
| Task/output | `0nubb` (1) versus `Bi214` (0); one signal logit `(B,)` |
| Default trainable parameters | **390,977** |
| Backend | Dense kNN plus a pure-PyTorch rigid-kernel fallback |

## 2. Data, split, and exact point representation

The reader groups `event_id,x,y,z,energy` rows from `/MC/hits/table`. `0nubb_part_*` is label
1 and `Bi_part_*` label 0. Complete relative HDF5 paths define a deterministic file-level split
with seed 42 and fractions `[0.8,0.1,0.1]`. Only train and validation are instantiated in this
stage; the reserved third split is not read. Selection is capped at 100 files per class.

For total energy $E=\sum_i e_i$,


$$
\mathbf c=\sum_i e_i\mathbf r_i/E,\qquad
\mathbf q_i=\lfloor(\mathbf r_i-\mathbf c)/(15\;\mathrm{mm})\rfloor.
$$

Rows in the same cell are merged. Initial centers $(\mathbf q+0.5)15$ mm are recentered by
their energy-weighted centroid. If there are more than 512 voxels, the highest-energy 512 are
kept with lexicographic coordinate tie-breakers; energy fractions still use complete-event
energy and are not renormalised.

| Field | dtype / shape | Meaning |
|---|---|---|
| `coords` | float32 `(B,N,3)` | centered voxel centers / 1000 mm; `N≤512`, padded per batch |
| `features` | float32 `(B,N,2)` | `[voxel_energy/E, log1p(hit_count)]` |
| `mask` | bool `(B,N)` | valid nodes versus padding for neighbourhoods and pooling |
| `label` | float32 `(B,)` | supervision only |
| output | floating `(B,)` | signal logits |

Neither total energy nor absolute detector position enters the model.

## 3. Rigid kernel-point operation

Each layer uses float32 `cdist`, retains at most 24 nearest neighbours, and removes neighbours
outside radius $r_l$. Fifteen fixed kernel points comprise the origin plus 14 points in
Fibonacci-sphere order at radius $0.65r_l$. They are non-trainable and non-deformable.


$$
a_{ijk}=\max(0,1-\|\mathbf x_j-\mathbf x_i-\mathbf p_k\|_2/\sigma_l),\quad
\bar a_{ijk}=a_{ijk}/\max(\epsilon,\sum_j a_{ijk}),
$$


$$
\mathbf g_i=\sum_{k=1}^{15}(\sum_{j\in\mathcal N(i)}\bar a_{ijk}\mathbf h_j)
\mathbf W_k+\mathbf b.
$$

The mask applies to queries, supports, residual outputs, and event pooling.

## 4. Layer table

| Stage | Operation | Input → output |
|---|---|---|
| Input encoder | concatenate XYZ/features; `Linear(5,64) → LN → SiLU` | `(B,N,5) → (B,N,64)` |
| KP residual 0 | rigid `64→64`, `r=.12`, `σ=.06`, 24-NN, 15 kernels; LN/SiLU/dropout; identity skip | `(B,N,64) → (B,N,64)` |
| KP residual 1 | rigid `64→96`, `r=.18`, `σ=.09`; `Linear(64,96)` skip | `(B,N,64) → (B,N,96)` |
| KP residual 2 | rigid `96→128`, `r=.27`, `σ=.135`; `Linear(96,128)` skip | `(B,N,96) → (B,N,128)` |
| Pool | concatenate masked mean/max | `(B,N,128) → (B,256)` |
| Head | `Linear(256,128) → SiLU → Dropout(.1) → Linear(128,1)` | `(B,256) → (B,1) → (B,)` |

Because coordinates are divided by 1000 mm, the radii mean 120/180/270 mm. No point
subsampling occurs.

## 5. Parameters, complexity, memory

Encoder: 512 parameters; blocks: 61,632 / 98,688 / 197,120; head: 33,025; total
**390,977**. For batch (B), padded points (N), neighbours (K=24), kernels (M=15),
dense neighbour search is (O(BN^2)) time and storage per layer, while influence storage is
(O(BNKM)). The float32 distance matrix and `(B,N,24,15)` influence tensor are expected
bottlenecks; all three blocks recompute neighbours. Batch size is therefore frozen at 8.

## 6. Frozen configuration

[config.yaml](config.yaml) fixes 15 mm voxels, coordinate scale 1000, cap 512; dimensions
`[64,96,128]`, 24 neighbours, 15 kernels, base radius/sigma `.12/.06`, radius multiplier 1.5,
classifier 128, dropout .10. Data uses 100 files/class, seed 42, `[.8,.1,.1]`, zero workers,
balanced mode, buffer 512. Training uses batch 8, 50 epochs, BCEWithLogitsLoss, AdamW `5e-4`,
weight decay `1e-4`, cosine schedule, clip 1.0, patience 12, seed 42, AMP auto, and validation
AUC checkpoint selection.

## 7. Boundary from KPConv

This is a **rigid KPConv-style lightweight fallback**, not the official implementation. It
uses a simple Fibonacci sphere, per-kernel neighbour normalisation, dense kNN plus a radius,
no grid subsampling/striding/deformable kernels/decoder, a project-specific full-resolution
head, and no compiled C++/CUDA operator. Official speed, accuracy, deformability, and large
point-cloud capacity are not claimed.

## 8. Reference

- Hugues Thomas, Charles R. Qi, Jean-Emmanuel Deschaud, Beatriz Marcotegui, François
  Goulette, and Leonidas J. Guibas, “KPConv: Flexible and Deformable Convolution for Point
  Clouds,” *ICCV*, 2019, pp. 6411–6420, arXiv:1904.08889,
  DOI [10.1109/ICCV.2019.00651](https://doi.org/10.1109/ICCV.2019.00651),
  [arXiv](https://arxiv.org/abs/1904.08889).

## 9. tmux campaign and recovery

This is item 7 in the serial queue. Launch with `RUN_ID=<...>` using
`run_nontransformer_training_queue.sh --run-id ${RUN_ID}` in detached session
`next-nontransformer-v2-${RUN_ID}`, and add the read-only monitor window using
`monitor_nontransformer_training.sh --run-id ${RUN_ID}`. Checkpoints go to
`02_models/checkpoints/<RUN_ID>/point_004_rigid_kpconv/attempt_NNN/{best.pt,last.pt}`; logs,
snapshot, CSV, JSON, and plot go to the matching campaign attempt directory. Stop with Ctrl-C
in `gpu-queue` while retaining the session. `--resume-queue` skips DONE and creates a fresh
attempt for FAILED/PENDING; `last.pt` is not true resume and old attempts are immutable.

## 10. Limitations

The 512-node cap can discard low-energy tails; fixed neighbourhoods do not adapt to density;
centering removes detector position; dense `cdist` scales quadratically; rigid kernels do not
provide deformable-KPConv capability; and pooled features are not physical explanations.

## 11. Training result

Pre-campaign placeholder: **PENDING** (see the appended result for actual status). After the real campaign, append actual parameters/environment, completed
and best epochs, best validation AUC/loss, duration, artifacts, early stop, and retry attempts.
No reserved-split metric belongs in this stage.


<!-- campaign-result:20260803_200356:start -->
## Campaign `20260803_200356` training result

This section is written after successful serial training. It contains train/validation information only; the test split was not read in this stage.

| Item | Observed value |
|---|---|
| Status / attempt | `DONE` / `attempt_001` |
| Backend | `pytorch` |
| Trainable parameters | 390,977 |
| Trees / tree nodes | N/A / N/A |
| Completed / best epoch | 17 / 5 |
| Best validation AUC | **0.933581** |
| Best validation loss | 0.353042 |
| Training time | 00:16:13 |
| Early stopped | `true` |
| Failed-attempt retry | `no` |
| Python / framework | `3.11.15` / `2.11.0+cu128` |
| Device | `NVIDIA GeForce RTX 5090` |
| Best / last checkpoint | `/home/wenyu/summer/02_models/checkpoints/20260803_200356/point_004_rigid_kpconv/attempt_001/best.pt` / `/home/wenyu/summer/02_models/checkpoints/20260803_200356/point_004_rigid_kpconv/attempt_001/last.pt` |
| Training history | `/home/wenyu/summer/03_training_runs/campaigns/20260803_200356/point_004_rigid_kpconv/attempt_001/history.json` |
<!-- campaign-result:20260803_200356:end -->
