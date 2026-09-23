# TOPO-001: Exact H0 Persistence with PersLay-style Pooling

[中文](README.md) | [English](README_EN.md)

## 1. Identity and hypothesis

| Item | Definition |
|---|---|
| `architecture_id` | `topo_001_persistence_perslay` |
| checkpoint `model_name` | `PersistencePersLayClassifier` |
| Python class | `next_alt.models.graph_topology.PersistencePersLayClassifier` |
| `input_kind` | `topology` (tensor-compatible with `points`) |
| Task/output | `0nubb=1`, `Bi214=0`; one signal logit of shape `(B,)` |
| Exact trainable parameters | **63,274** |

The hypothesis is that multiscale component merging complements ordinary point/graph pooling. The
model computes exact zero-dimensional Vietoris--Rips persistence only. It does not claim H1/H2.
The diagram is pooled with learned positive weights and no attention or softmax.

## 2. Data, split, and preprocessing

Contiguous rows sharing an `event_id` are read from HDF5 `/MC/hits/table` using `x/y/z/energy`.
Directories `0nubb_part_*` and `Bi_part_*` map to labels 1 and 0. A stable hash of the complete
relative file path with seed 42 assigns whole files at fractions 0.8/0.1/0.1. This stage opens only
train and validation, at most 100 files per class in each; validation selects best and early stop.

$$
E=\sum_i e_i,\quad \mathbf c=\sum_i e_i\mathbf r_i/E,\quad
\mathbf q_i=\lfloor(\mathbf r_i-\mathbf c)/(15\,\mathrm{mm})\rfloor.
$$

Hits in one cell are merged, `(q+0.5)*15 mm` centers are energy-recentered, and at most 512
highest-energy voxels are retained deterministically. Fractions are not renormalized after the cap:

$$
\mathbf x_v=\mathbf r^{centered}_v/(1000\,\mathrm{mm}),\qquad
\mathbf f_v=[e_v/E,\log(1+n_v)].
$$

Topology then retains at most 96 points by energy fraction. Total energy and absolute position are
not inputs.

## 3. Exact tensor and diagram contracts

| Field | dtype / shape | Meaning |
|---|---|---|
| `coords` | float32 `(B,N,3)` | centered and scaled voxel centers |
| `features` | float32 `(B,N,2)` | energy fraction, log hit count |
| `mask` | bool `(B,N)` | valid point versus padding |
| selected | `(B,M,*)`, `M≤96` | energy-ranked subset |
| diagram | float32 `(B,max(M-1,1),6)` | finite H0 pairs with endpoint marks |
| diagram mask | bool `(B,max(M-1,1))` | valid finite pairs |
| output | floating `(B,)` | signal logit |

Finite H0 death times of the complete Euclidean Rips filtration equal minimum-spanning-tree edge
lengths. Batched Prim computes the exact MST under `no_grad`; the essential final component is
omitted. An edge `(u,v)` becomes

$$
[0,d_{uv},d_{uv},e_u/E+e_v/E,|e_u/E-e_v/E|,
(\log(1+n_u)+\log(1+n_v))/2].
$$

The first three entries are birth/death/persistence and the last three are project-specific marks.

## 4. Layer-by-layer structure

| Stage | Operation | Shape |
|---|---|---|
| Select | top energy, at most 96 | `(B,N,*)→(B,M,*)` |
| H0 | pairwise distance plus batched Prim | `(B,M,3)→(B,M-1,6)` |
| Encoder | `Linear(6,96)-LN-SiLU-Linear(96,128)-SiLU` | row → 128 |
| Weight | persistence through `Linear(1,24)-SiLU-Linear(24,1)-Softplus` | row → scalar |
| Pool | weighted sum/√count, ordinary mean, ordinary max | diagram → 384 |
| Statistics | count/96, total/√count, mean and max persistence | 4 |
| Head | `Linear(388,128)-SiLU-Dropout(.1)-Linear(128,1)` | `(B,388)→(B,)` |

For diagram row (p),

$$
z_p=\phi(p),\quad w_p=\mathrm{Softplus}(g(\mathrm{pers}(p))),\quad
z_{sum}=|D|^{-1/2}\sum_{p\in D}w_pz_p.
$$

Weights are independent and unnormalised, so this is not attention.

## 5. Parameters, complexity, and memory

The encoder has 13,280 parameters, weighting network 73, and head 49,921: **63,274 total**.
With (M≤96), pairwise distances and Prim cost `O(BM²)` time and distance memory; neural pooling
costs `O(BMH)`. MST is outside autograd. The frozen batch size is 16. No GUDHI or Ripser is used.

## 6. Frozen YAML settings

- Data: root `/home/klz/Data/zeronu_benchmark/NEXT`; 100 files/class; split 42 and
  `[0.8,0.1,0.1]`; workers 0; balanced true; buffer 512.
- Representation: 15 mm bin, 1000 mm scale, 512 shared points.
- Model: feature dim 2; 96 topology points; hidden/embedding/head 96/128/128; dropout .10.
- Training: batch 16; 50 epochs; lr 7e-4; AdamW decay 1e-4; clip 1.0; patience 12; min delta 0;
  seed 42; nondeterministic; AMP auto.

The loss is `BCEWithLogitsLoss`, with AdamW and `CosineAnnealingLR(T_max=50)`; best is selected by
validation AUC.

## 7. Boundary relative to Persistence Images and PersLay

This implementation has no persistence-image raster; that paper supplies vectorisation context.
It has no H1, extended persistence, heat-kernel signature, or PersLay graph filtration, and no
GUDHI/Ripser. Endpoint energy/hit marks are project-specific. Learned rowwise representation,
positive weighting, and invariant aggregation are PersLay-style, while mean/max/statistics differ
from the paper examples. Pairing and top-energy selection are discrete. The accurate name is
“exact-H0 persistence plus PersLay-style pooling,” not a complete PersLay reproduction.

## 8. Run and campaign artifacts

```bash
cd /home/wenyu/summer
source .venv/bin/activate
python 01_code/architectures/topo_001_persistence_perslay/train_classification.py CONFIG_SNAPSHOT
```

The detached session is `next-nontransformer-v2-<RUN_ID>`, with one serial `gpu-queue` and a
read-only monitor. Best/last checkpoints are under
`02_models/checkpoints/<RUN_ID>/topo_001_persistence_perslay/attempt_NNN/`; logs, snapshot, CSV,
JSON, and plot use the matching campaign attempt. Stop with `C-c`; queue resume skips DONE and
starts a FAILED model from scratch in a new attempt, not from `last.pt`.

## 9. Limitations

H0 cannot directly represent loops/cavities; the 96-point cap can remove low-energy connections;
ties change discrete MST pairing; persistence depends on the fixed 1000 mm scale; topology is
recomputed on every forward; total energy and absolute position are excluded. No test split or test
metric belongs to this stage.

## 10. Training result placeholder

This is a pre-campaign placeholder; see the appended result for actual status. Append the actual environment/parameter count, completed and best
epoch, best validation AUC/loss, duration, artifact paths, early stop, and retries after the campaign.
Do not add test metrics.

## 11. References

1. Mathieu Carrière, Frédéric Chazal, Yuichi Ike, Theo Lacombe, Martin Royer, Yuhei Umeda,
   “PersLay: A Neural Network Layer for Persistence Diagrams and New Graph Topological
   Signatures,” *AISTATS 2020*, PMLR 108:2786–2796.
   [PMLR](https://proceedings.mlr.press/v108/carriere20a.html)
2. Henry Adams et al., “Persistence Images: A Stable Vector Representation of Persistent
   Homology,” *JMLR* 18(8):1–35, 2017. [JMLR](https://jmlr.org/papers/v18/16-337.html)
3. NEXT Collaboration, P. Ferrario et al., “First proof of topological signature in the high
   pressure xenon gas TPC with electroluminescence amplification for the NEXT experiment,”
   *JHEP* 2016, 104, DOI `10.1007/JHEP01(2016)104`, arXiv:1507.05902.
   [Journal](https://link.springer.com/article/10.1007/JHEP01%282016%29104) ·
   [arXiv](https://arxiv.org/abs/1507.05902)


<!-- campaign-result:20260803_200356:start -->
## Campaign `20260803_200356` training result

This section is written after successful serial training. It contains train/validation information only; the test split was not read in this stage.

| Item | Observed value |
|---|---|
| Status / attempt | `DONE` / `attempt_001` |
| Backend | `pytorch` |
| Trainable parameters | 63,274 |
| Trees / tree nodes | N/A / N/A |
| Completed / best epoch | 50 / 49 |
| Best validation AUC | **0.909366** |
| Best validation loss | 0.380932 |
| Training time | 01:08:03 |
| Early stopped | `false` |
| Failed-attempt retry | `no` |
| Python / framework | `3.11.15` / `2.11.0+cu128` |
| Device | `NVIDIA GeForce RTX 5090` |
| Best / last checkpoint | `/home/wenyu/summer/02_models/checkpoints/20260803_200356/topo_001_persistence_perslay/attempt_001/best.pt` / `/home/wenyu/summer/02_models/checkpoints/20260803_200356/topo_001_persistence_perslay/attempt_001/last.pt` |
| Training history | `/home/wenyu/summer/03_training_runs/campaigns/20260803_200356/topo_001_persistence_perslay/attempt_001/history.json` |
<!-- campaign-result:20260803_200356:end -->
