# GNN-005: DimeNet-lite Directional Classifier

[中文](README.md) | [English](README_EN.md)

## 1. Identity and hypothesis

| Item | Definition |
|---|---|
| `architecture_id` | `gnn_005_dimenet_lite` |
| checkpoint `model_name` | `DimeNetLiteClassifier` |
| Python class | `next_alt.models.graph_topology.DimeNetLiteClassifier` |
| `input_kind` | `graph` |
| Task | NEXT `0nubb` (1) versus `Bi214` (0) classification |
| Output | One uncalibrated signal logit per event, shape `(B,)` |
| Exact trainable parameters | **169,553** |

The hypothesis is that explicit three-point angles $k\to j\to i$ expose double-blob and
curved-track structure that distance-only graph layers miss. There is no Transformer or attention.

## 2. Data and exact preprocessing

Rows sharing one contiguous `event_id` are read from HDF5 `/MC/hits/table`, using
`x/y/z/energy`. `0nubb_part_*` maps to label 1 and `Bi_part_*` to 0. A stable hash of the complete
relative file path with seed 42 assigns whole files to train/validation/test at 0.8/0.1/0.1. This
stage opens train and validation only, at most 100 files per class in each split; validation is used
only for early stopping and best-checkpoint selection.

For hit position $\mathbf r_i$ and energy $e_i$:

$$
E=\sum_i e_i,\quad \mathbf c=\sum_i e_i\mathbf r_i/E,\quad
\mathbf q_i=\lfloor(\mathbf r_i-\mathbf c)/(15\,\mathrm{mm})\rfloor.
$$

Rows in one cell are merged. Centers are `(q+0.5)*15 mm`, then recentered by their energy-weighted
center. If more than 512 voxels remain, the highest-energy 512 are retained with cell-coordinate
tie breakers. Truncated energy fractions are not renormalized:

$$
\mathbf x_v=\mathbf r^{centered}_v/(1000\,\mathrm{mm}),\qquad
\mathbf f_v=[e_v/E,\log(1+n_v)].
$$

Total energy only defines the fraction and the external `energy_condition`; it and absolute
detector position are not model inputs.

## 3. Tensor contract

| Key | dtype and exact shape | Meaning |
|---|---|---|
| `coords` | float32 `(B,N,3)` | centered, scaled voxel centers |
| `features` | float32 `(B,N,2)` | energy fraction and log hit count |
| `mask` | bool `(B,N)` | valid point versus padding |
| output | floating `(B,)` | binary logit |

A directed masked kNN graph with at most `K=8` other nodes per query is selected under `no_grad`;
live distances are recomputed from the input tensor.

## 4. Layer table and forward equations

| Stage | Operation | Shape |
|---|---|---|
| Node encoder | `Linear(5,96)-LN-SiLU-Linear(96,96)-SiLU` | `(B,N,5)→(B,N,96)` |
| Radial basis | 6 smooth-cutoff sine/Bessel-style terms, cutoff 1.0 | `(B,N,K)→(B,N,K,6)` |
| Edge encoder | `Linear(198,96)-LN-SiLU` on `[h_i,h_j,RBF(d_ij)]` | `(B,N,K,198)→(B,N,K,96)` |
| Directional block ×3 | triplet interaction, edge residual/LN, edge-to-node sum, node residual/LN | width 96 |
| Pool | masked node mean concatenated with max | `(B,N,96)→(B,192)` |
| Head | `Linear(192,128)-SiLU-Dropout(.1)-Linear(128,1)` | `(B,192)→(B,)` |

For edge $j\to i$, enumerate $k\to j$, $k\ne i$, and use
`[cos(0θ),...,cos(3θ)]` for angle $\theta_{kji}$:

$$
u_{ji}=|T_{ji}|^{-1/2}\sum_{k\in T_{ji}}
W_hh_{kj}\odot W_rR(d_{jk})\odot W_aC(\theta_{kji}),
$$
$$
h'_{ji}=\mathrm{LN}(h_{ji}+\mathrm{Dropout}(W_u[u_{ji}\odot W_rR(d_{ij})])).
$$

Transformed edges are summed into nodes and divided by the square root of the valid degree.

## 5. Parameters, complexity, and memory

The exact parameter count is 169,553. With $N\le512,K=8,H=96,I=48$, dense kNN costs
`O(BN²)` time/memory and directional interaction costs roughly `O(BNK²I)`. The dominant saved
activation is `(B,N,K,K,H)`, so the frozen batch size is 4.

## 6. Frozen YAML settings

- Data: root `/home/klz/Data/zeronu_benchmark/NEXT`; max 100 files/class; split seed 42 and
  `[0.8,0.1,0.1]`; workers 0; balanced training true; shuffle buffer 512.
- Representation: 15 mm bins, 1000 mm coordinate scale, at most 512 points.
- Model: feature/hidden/interaction 2/96/48; 3 blocks; k 8; radial/angular 6/4; cutoff 1.0;
  classifier 128; dropout 0.10.
- Training: batch 4, 50 epochs, learning rate 4e-4, weight decay 1e-4, clip 1.0, patience 12,
  min delta 0, seed 42, nondeterministic, AMP auto.

Training uses `BCEWithLogitsLoss`, AdamW, and epoch-wise `CosineAnnealingLR(T_max=50)`; best is
chosen by validation AUC.

## 7. Boundary relative to DimeNet

This is a **DimeNet-inspired lite classifier**, not a reproduction. Its graph is voxel kNN rather
than a molecular cutoff graph; it has no atom types. The node encoder concatenates centered XYZ,
so the complete model is not rotation invariant. The bases are compact sine/Bessel-style and
cosine terms, not the paper's full spherical Bessel/spherical-harmonic construction. It omits the
full DimeNet output blocks and bilinear machinery, predicts event labels rather than molecular
energy, and uses a pure-PyTorch dense implementation. No quantum-chemistry accuracy or unimplemented
paper capability is claimed.

## 8. Run and campaign artifacts

```bash
cd /home/wenyu/summer
source .venv/bin/activate
python 01_code/architectures/gnn_005_dimenet_lite/train_classification.py CONFIG_SNAPSHOT
```

The detached session is `next-nontransformer-v2-<RUN_ID>`; `gpu-queue` is serial and `monitor` is
read-only. Checkpoints are
`02_models/checkpoints/<RUN_ID>/gnn_005_dimenet_lite/attempt_NNN/{best.pt,last.pt}`; logs, config
snapshot, CSV, JSON, and plot are under the matching campaign attempt directory. Stop with
`tmux send-keys -t <session>:gpu-queue C-c`. `--resume-queue` skips DONE and starts FAILED models
from scratch in a new attempt; `last.pt` is not true checkpoint resume.

## 9. Limitations

Dense kNN and explicit triplets scale quickly; the 512-point cap can remove low-energy tails;
cutoff-zero edges may remain in kNN; tie breaking and AMP affect strict reproducibility; excluding
total energy and absolute position can reduce raw accuracy. This stage has no test-split result.

## 10. Training result placeholder

This is a pre-campaign placeholder; see the appended result for actual status. After the campaign, append the actual environment and parameter
count, completed/best epoch, best validation AUC/loss, duration, checkpoint/log paths, early-stop
state, and retries. Do not add test metrics.

## 11. References

1. Johannes Gasteiger, Janek Groß, Stephan Günnemann, “Directional Message Passing for Molecular
   Graphs,” *ICLR 2020*, arXiv:2003.03123.
   [OpenReview](https://openreview.net/forum?id=B1eWbxStPH) · [arXiv](https://arxiv.org/abs/2003.03123)
2. NEXT Collaboration, P. Ferrario et al., “First proof of topological signature in the high
   pressure xenon gas TPC with electroluminescence amplification for the NEXT experiment,”
   *JHEP* 2016, 104 (2016), DOI `10.1007/JHEP01(2016)104`, arXiv:1507.05902.
   [Journal](https://link.springer.com/article/10.1007/JHEP01%282016%29104) ·
   [arXiv](https://arxiv.org/abs/1507.05902)


<!-- campaign-result:20260803_200356:start -->
## Campaign `20260803_200356` training result

This section is written after successful serial training. It contains train/validation information only; the test split was not read in this stage.

| Item | Observed value |
|---|---|
| Status / attempt | `DONE` / `attempt_001` |
| Backend | `pytorch` |
| Trainable parameters | 169,553 |
| Trees / tree nodes | N/A / N/A |
| Completed / best epoch | 38 / 26 |
| Best validation AUC | **0.980804** |
| Best validation loss | 0.246705 |
| Training time | 00:51:49 |
| Early stopped | `true` |
| Failed-attempt retry | `no` |
| Python / framework | `3.11.15` / `2.11.0+cu128` |
| Device | `NVIDIA GeForce RTX 5090` |
| Best / last checkpoint | `/home/wenyu/summer/02_models/checkpoints/20260803_200356/gnn_005_dimenet_lite/attempt_001/best.pt` / `/home/wenyu/summer/02_models/checkpoints/20260803_200356/gnn_005_dimenet_lite/attempt_001/last.pt` |
| Training history | `/home/wenyu/summer/03_training_runs/campaigns/20260803_200356/gnn_005_dimenet_lite/attempt_001/history.json` |
<!-- campaign-result:20260803_200356:end -->
