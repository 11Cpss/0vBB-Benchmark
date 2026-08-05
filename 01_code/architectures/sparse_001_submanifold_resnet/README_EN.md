# SPARSE-001: Pure-PyTorch Submanifold Sparse 3-D ResNet

[中文](README.md) | English

## 1. Positioning and hypothesis

SPARSE-001 asks whether preserving every occupied 15 mm voxel and applying 3×3×3
submanifold convolutions only between active coordinates can learn NEXT track topology without
materialising a dense 3-D volume. It contains no Transformer or attention, and its active set
never expands.

| Item | Definition |
|---|---|
| `architecture_id` | `sparse_001_submanifold_resnet` |
| checkpoint `model_name` | `SubmanifoldSparseResNetClassifier` |
| Python class | `next_alt.models.mixer_sparse.SubmanifoldSparseResNetClassifier` |
| registry `input_kind` | `sparse3d` |
| Task/output | `0nubb` (1) versus `Bi214` (0); one signal logit `(B,)` |
| Default trainable parameters | **298,177** |
| Backend | Pure PyTorch sorted integer hashes and `searchsorted` |

This fallback is explicitly not a reproduction of MinkowskiEngine, spconv, or an official
high-performance C++/CUDA kernel.

## 2. Raw data, file split, and untruncated sparse input

The shared reader groups `event_id,x,y,z,energy` rows from `/MC/hits/table`.
`0nubb_part_*` maps to label 1 and `Bi_part_*` to label 0. Complete relative HDF5 paths define
a deterministic file-level split with seed 42 and `[0.8,0.1,0.1]`. This stage creates only
train/validation loaders and does not read the third reserved split; each class is capped at
100 files.

For total energy $E=\sum_i e_i$,


$$
\mathbf c=\sum_i e_i\mathbf r_i/E,\qquad
\mathbf q_i=\lfloor(\mathbf r_i-\mathbf c)/(15\;\mathrm{mm})\rfloor\in\mathbb Z^3.
$$

Hits sharing $\mathbf q$ form one active voxel with


$$
\mathbf f_q=[E_q/E,\log(1+n_q)].
$$

`max_points=null`: occupied voxels are never energy-truncated. Total energy and absolute
position are absent; integer coordinates are used only for relative neighbourhood lookup.

| Batch field | dtype / shape | Semantics |
|---|---|---|
| `voxel_coords` | int64 `(B,V,3)` | centered 15 mm lattice cells; padded rows are ignored |
| `voxel_features` | float32 `(B,V,2)` | energy fraction and `log1p(hit_count)`; zero-padded |
| `voxel_mask` | bool `(B,V)` | active versus batch padding; `V` is the batch maximum |
| `label` | float32 `(B,)` | supervision only |
| output | floating `(B,)` | signal logits |

Integration must supply these three `voxel_*` keys. A dense `(B,2,D,H,W)` volume or the
512-capped point representation is not this model's input.

## 3. Submanifold operation

For active set $\mathcal A\subset\mathbb Z^3$ and offsets
$\mathcal D=\{-1,0,1\}^3$,


$$
\mathbf g(\mathbf x)=\mathbf b+\sum_{\delta\in\mathcal D}
\mathbb1[\mathbf x+\delta\in\mathcal A]\mathbf h(\mathbf x+\delta)\mathbf W_\delta,
\quad\mathbf x\in\mathcal A.
$$

Outputs exist only on the original active set. Per-event integer coordinates receive a
collision-free linear hash. All 27 offsets expand to `(V,27,3)`, flatten into one
`torch.searchsorted` call, and use `einsum('voc,ocd->vd')` to apply their weights together.
No detector-sized dense grid is allocated.

## 4. Layer table

Each residual main branch is `SubMConv → LayerNorm → SiLU → Dropout → SubMConv →
LayerNorm`, followed by identity/Linear skip and SiLU. The mask is restored after each block.

| Stage | Operation | Input → output |
|---|---|---|
| Stem | 27-neighbour SubMConv `2→24`, LN, SiLU | `(B,V,2) → (B,V,24)` |
| Stage 0 | one `24→24→24` residual block | `(B,V,24) → (B,V,24)` |
| Stage 1 | one `24→40→40` block, `Linear(24,40)` skip | `(B,V,24) → (B,V,40)` |
| Stage 2 | one `40→64→64` block, `Linear(40,64)` skip | `(B,V,40) → (B,V,64)` |
| Pool | concatenate masked active-site mean/max | `(B,V,64) → (B,128)` |
| Head | `Linear(128,96) → SiLU → Dropout(.1) → Linear(96,1)` | `(B,128) → (B,1) → (B,)` |

No stride or pooling changes the active coordinates.

## 5. Parameters, complexity, and memory

Stem: 1,368; stages: 31,248 / 70,360 / 182,720; head: 12,481; total **298,177**.
One convolution on $V$ active sites costs $O(V\log V)$ sorting/lookups plus up to
$O(27VC_{in}C_{out})$ offset matrix products. There are seven SubMConv layers and each
currently repeats the sort. Feature/padding memory is $O(BVC)$, not $O(BD^3C)$. All 27
offsets are vectorised within an event; the remaining per-event Python loop, repeated lookup,
and unequal-count padding are expected bottlenecks.

## 6. Frozen configuration

[config.yaml](config.yaml) fixes 15 mm cells, coordinate provenance scale 1000, no point cap;
channels `[24,40,64]`, blocks `[1,1,1]`, classifier 96, dropout .10. Data uses 100
files/class, seed 42, `[.8,.1,.1]`, zero workers, balanced mode, buffer 512. Training uses
batch 8, 50 epochs, BCEWithLogitsLoss, AdamW `5e-4`, weight decay `1e-4`, cosine schedule,
clip 1.0, patience 12, seed 42, AMP auto, and validation-AUC checkpoint selection.

## 7. Boundary from papers and sparse libraries

This is a **submanifold sparse-convolution-inspired ResNet with a PyTorch fallback**. It keeps
same-active-site semantics and per-offset weights, but has no compiled hash table, kernel-map
cache, coordinate manager, fused kernel, strided sparse hierarchy, U-Net, or full library API.
Batches are padded tensors processed with Python event loops and vectorised offsets. LayerNorm,
SiLU, widths, and mean/max head are project choices. Official backend performance and scalability are not
claimed.

## 8. References

- Benjamin Graham and Laurens van der Maaten, “Submanifold Sparse Convolutional Networks,”
  arXiv:1706.01307, 2017. [arXiv](https://arxiv.org/abs/1706.01307); no DOI.
- Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, “Deep Residual Learning for Image
  Recognition,” *CVPR*, 2016, pp. 770–778, arXiv:1512.03385,
  DOI [10.1109/CVPR.2016.90](https://doi.org/10.1109/CVPR.2016.90).

## 9. tmux campaign, artifacts, and recovery

This is item 10 in the serial single-GPU queue. Run `run_nontransformer_training_queue.sh
--run-id ${RUN_ID}` in detached session `next-nontransformer-v2-${RUN_ID}` and add the
read-only monitor with `monitor_nontransformer_training.sh --run-id ${RUN_ID}`. Checkpoints
go under `02_models/checkpoints/<RUN_ID>/sparse_001_submanifold_resnet/attempt_NNN/`; stdout,
snapshot, CSV, JSON, and plot use the matching campaign attempt directory. Stop with Ctrl-C in
`gpu-queue` while retaining the session. `--resume-queue` skips DONE and creates a fresh
attempt for FAILED/PENDING; `last.pt` is not true resume, and prior attempts are immutable.

## 10. Limitations

Untruncated outlier events can enlarge padding/runtime; the fallback repeats kernel-map work;
the shallow non-strided receptive field limits long-range propagation; quantisation is
discontinuous at voxel boundaries; and centering/translation-invariant adjacency removes
detector position that might carry useful information.

## 11. Training result

Pre-campaign placeholder: **PENDING** (see the appended result for actual status). After the real campaign, append actual parameters/environment, completed
and best epochs, best validation AUC/loss, duration, artifacts, early stop, and attempts. No
reserved-split metric belongs in this stage.


<!-- campaign-result:20260803_200356:start -->
## Campaign `20260803_200356` training result

This section is written after successful serial training. It contains train/validation information only; the test split was not read in this stage.

| Item | Observed value |
|---|---|
| Status / attempt | `DONE` / `attempt_001` |
| Backend | `pytorch` |
| Trainable parameters | 298,177 |
| Trees / tree nodes | N/A / N/A |
| Completed / best epoch | 22 / 10 |
| Best validation AUC | **0.971691** |
| Best validation loss | 0.280406 |
| Training time | 00:49:40 |
| Early stopped | `true` |
| Failed-attempt retry | `no` |
| Python / framework | `3.11.15` / `2.11.0+cu128` |
| Device | `NVIDIA GeForce RTX 5090` |
| Best / last checkpoint | `/home/wenyu/summer/02_models/checkpoints/20260803_200356/sparse_001_submanifold_resnet/attempt_001/best.pt` / `/home/wenyu/summer/02_models/checkpoints/20260803_200356/sparse_001_submanifold_resnet/attempt_001/last.pt` |
| Training history | `/home/wenyu/summer/03_training_runs/campaigns/20260803_200356/sparse_001_submanifold_resnet/attempt_001/history.json` |
<!-- campaign-result:20260803_200356:end -->
