# TOPO-001：H0 Persistence + PersLay-style 分类器

[中文](README.md) | [English](README_EN.md)

## 1. 模型定位

| 项目 | 定义 |
|---|---|
| `architecture_id` | `topo_001_persistence_perslay` |
| checkpoint `model_name` | `PersistencePersLayClassifier` |
| Python 类 | `next_alt.models.graph_topology.PersistencePersLayClassifier` |
| `input_kind` | `topology`（底层 tensor 与 `points` 相同） |
| 任务/输出 | `0nubb=1`、`Bi214=0`；每事件一个 `(B,)` signal logit |
| 精确可训练参数量 | **63,274** |

研究假设是：track 在多尺度连通性上的合并顺序可以提供与普通 point/graph pooling
互补的信息。该模型只计算精确的 Vietoris–Rips **零维持久同调 (H_0)**；不声称实现
(H_1/H_2)。持久图之后是无 attention、无归一化 softmax 的 PersLay-style learned
weighting。

## 2. 数据、split 与表示公式

原始事件为 HDF5 `/MC/hits/table` 中连续相同 `event_id` 的 `x/y/z/energy` 行。
`0nubb_part_*` 为 1，`Bi_part_*` 为 0。完整相对文件路径用 seed 42 稳定 hash 后按
`[0.8,0.1,0.1]` 做 file-level split。本阶段只读取 train/validation，各 split 每类
最多 100 文件；validation 仅用于 best checkpoint 与 patience 12 early stopping。

预处理与其他 point 模型一致：

$$
E=\sum_i e_i,\quad \mathbf c=\sum_i e_i\mathbf r_i/E,\quad
\mathbf q_i=\lfloor(\mathbf r_i-\mathbf c)/(15\,\mathrm{mm})\rfloor.
$$

同 cell 合并，center 为 `(q+0.5)*15 mm` 并再次 energy-weighted recenter。若点数超过
512，按能量和 cell 坐标确定性保留 512；energy fraction 不重归一化。最终：

$$
\mathbf x_v=\mathbf r^{centered}_v/(1000\,\mathrm{mm}),\qquad
\mathbf f_v=[e_v/E,\log(1+n_v)].
$$

进入 topology 算法前再按 energy fraction 取最多 96 点。这是独立于共享 512 点 cap
的计算预算限制。总能量与绝对位置不进入模型。

## 3. tensor 与 persistence diagram

| 字段 | dtype / shape | 语义 |
|---|---|---|
| `coords` | float32 `(B,N,3)` | centered、scaled voxel center |
| `features` | float32 `(B,N,2)` | energy fraction、log hit count |
| `mask` | bool `(B,N)` | 有效点/padding |
| selected | float32 `(B,M,*)`, `M≤96` | energy-ranked topology subset |
| diagram | float32 `(B,max(M-1,1),6)` | finite (H_0) pairs 与 endpoint marks |
| diagram mask | bool `(B,max(M-1,1))` | 排除 padding 与单点事件的空 diagram |
| 输出 | floating `(B,)` | signal logit |

对完整 Euclidean Rips filtration，有限 (H_0) death times 等于 complete weighted graph
最小生成树的 edge lengths。实现以 batched Prim 在 `no_grad` 中精确求 MST；最后一个
essential component 不放入 finite diagram。每个 MST edge `(u,v)` 的 6 维 row 是：

$$
[0,d_{uv},d_{uv},e_u/E+e_v/E,|e_u/E-e_v/E|,
(\log(1+n_u)+\log(1+n_v))/2].
$$

前 3 项分别是 birth、death、persistence，后 3 项是本项目的 marked-diagram 扩展。

## 4. 逐层架构与 forward

| 阶段 | 运算 | shape |
|---|---|---|
| selection | top-energy，最多 96 点 | `(B,N,*)→(B,M,*)` |
| exact (H_0) | pairwise distance + batched Prim MST | `(B,M,3)→(B,M-1,6)` |
| diagram encoder | `Linear(6,96) → LayerNorm → SiLU → Linear(96,128) → SiLU` | row → 128 |
| scalar weight | persistence `Linear(1,24) → SiLU → Linear(24,1) → Softplus` | row → positive scalar |
| pool | weighted sum/√count、unweighted mean、unweighted max | `(B,M-1,128)→(B,384)` |
| raw stats | count/96、total/√count、mean、max persistence | `(B,4)` |
| head | `Linear(388,128) → SiLU → Dropout(.1) → Linear(128,1)` | `(B,388)→(B,)` |

令 diagram row 为 (p)，有效集合为 (D)：

$$
z_p=\phi(p),\quad w_p=\mathrm{Softplus}(g(\mathrm{pers}(p))),\quad
z_{sum}=|D|^{-1/2}\sum_{p\in D}w_pz_p.
$$

`w_p` 彼此独立、不经 softmax，因此不是 attention。最终拼接 `z_sum`、mean、max 与
4 个 raw statistics。

## 5. 参数量、复杂度与显存

参数分解：diagram encoder 13,280；weight network 73；classifier 49,921；合计
**63,274**。设 (M≤96)，pairwise distance 与 Prim 时间 (O(BM^2))、distance memory
(O(BM^2))；neural pooling 为 (O(BMH))。MST 位于 `no_grad`，显存主要来自 diagram
MLP activation；正式 batch size 为 16。CPU/GPU 上都使用纯 PyTorch，无 GUDHI/Ripser。

## 6. 完整 YAML 参数

| 类别 | 参数 | 值 |
|---|---|---:|
| data | root / max files | `/home/klz/Data/zeronu_benchmark/NEXT` / 100 每类 |
| data | seed / fractions | 42 / `[0.8,0.1,0.1]` |
| data | workers / balanced / buffer | 0 / true / 512 |
| representation | bin / scale / max points | 15 mm / 1000 mm / 512 |
| model | feature / topology points | 2 / 96 |
| model | diagram hidden / embedding / classifier | 96 / 128 / 128 |
| model | dropout | 0.10 |
| training | batch / epochs / lr | 16 / 50 / 7e-4 |
| training | weight decay / clip | 1e-4 / 1.0 |
| training | patience / min delta | 12 / 0.0 |
| training | seed / deterministic / AMP | 42 / false / true-auto |

训练协议为 `BCEWithLogitsLoss`、AdamW、`CosineAnnealingLR(T_max=50)`；每 epoch 做
validation，按 validation AUC 保存 best。

## 7. 与 Persistence Images / PersLay 的差异

- 本实现使用 exact Euclidean Rips (H_0) finite pairs；没有 persistence image raster；
  Persistence Images 论文用于说明稳定向量化背景，不应把本模型称作 PI 网络。
- 没有 (H_1)、extended persistence、heat-kernel signature 或 PersLay 论文的 graph
  filtration；也不依赖 GUDHI/Ripser。
- diagram row 加入 MST edge endpoint 的能量/hit-count marks，这是项目专属设计。
- learned positive weighting、rowwise representation 与 permutation-invariant aggregation 是
  PersLay-style；额外 mean/max/statistics 与论文实例不同。
- MST pairing 和 top-energy selection 不可微；只训练 diagram encoder、weight 与 head。

因此正确名称是“exact-H0 persistence + PersLay-style pooling”，不是 PersLay 全复现。

## 8. 运行与 campaign

```bash
cd /home/wenyu/summer
source .venv/bin/activate
python 01_code/architectures/topo_001_persistence_perslay/train_classification.py CONFIG_SNAPSHOT
```

tmux session 为 `next-nontransformer-v2-<RUN_ID>`；只允许 `gpu-queue` 串行训练。
checkpoint 位于
`02_models/checkpoints/<RUN_ID>/topo_001_persistence_perslay/attempt_NNN/{best.pt,last.pt}`，
日志/config/CSV/JSON/plot 位于相应 campaign attempt 目录。`monitor` 不启动训练。
`C-c` 停止后，`--resume-queue` 对 FAILED 创建新 attempt 并从头训练；不是 checkpoint resume。

## 9. 限制

- 只包含 (H_0)，不能直接表示 loop/cavity；96 点 energy cap 可能删掉低能连接；
- complete-graph MST 对坐标小扰动稳定，但 edge pairing 在 tie 时离散变化；
- (H_0) death 对 overall coordinate scale 敏感，因此 scale 必须固定为 1000 mm；
- topological computation 每 forward 重做，且 Prim 有 96 次 Python-level loop；
- 不使用总能量/绝对位置；本阶段不读 test，也没有 test 指标。

## 10. 训练结果（待追加）

本段为训练前占位说明；实际状态见文末追加结果。结束后追加实际参数量与环境、完成/best epoch、best validation
AUC/loss、耗时、best/last/log 路径、early-stop 与 attempt 重试；不得写 test 指标。

## 11. 参考文献

1. Mathieu Carrière, Frédéric Chazal, Yuichi Ike, Theo Lacombe, Martin Royer, Yuhei Umeda,
   “PersLay: A Neural Network Layer for Persistence Diagrams and New Graph Topological
   Signatures,” *AISTATS 2020*, PMLR 108:2786–2796.
   [PMLR](https://proceedings.mlr.press/v108/carriere20a.html)
2. Henry Adams et al., “Persistence Images: A Stable Vector Representation of Persistent
   Homology,” *Journal of Machine Learning Research* 18(8):1–35, 2017.
   [JMLR](https://jmlr.org/papers/v18/16-337.html)
3. NEXT Collaboration, P. Ferrario et al., “First proof of topological signature in the high
   pressure xenon gas TPC with electroluminescence amplification for the NEXT experiment,”
   *JHEP* 2016, 104, DOI `10.1007/JHEP01(2016)104`, arXiv:1507.05902.
   [Journal](https://link.springer.com/article/10.1007/JHEP01%282016%29104) ·
   [arXiv](https://arxiv.org/abs/1507.05902)


<!-- campaign-result:20260803_200356:start -->
## Campaign `20260803_200356` 训练结果

本节由串行训练队列在训练成功后写入，只包含 train/validation 信息；本阶段没有读取 test split。

| 项目 | 实际值 |
|---|---|
| 状态 / attempt | `DONE` / `attempt_001` |
| 后端 | `pytorch` |
| 实际参数量 | 63,274 |
| 树数量 / 树节点数 | N/A / N/A |
| 完成 epoch / best epoch | 50 / 49 |
| best validation AUC | **0.909366** |
| best validation loss | 0.380932 |
| 总训练时间 | 01:08:03 |
| early stop | `false` |
| 失败重试 | `no` |
| Python / 框架 | `3.11.15` / `2.11.0+cu128` |
| 设备 | `NVIDIA GeForce RTX 5090` |
| best / last checkpoint | `/home/wenyu/summer/02_models/checkpoints/20260803_200356/topo_001_persistence_perslay/attempt_001/best.pt` / `/home/wenyu/summer/02_models/checkpoints/20260803_200356/topo_001_persistence_perslay/attempt_001/last.pt` |
| 训练日志 | `/home/wenyu/summer/03_training_runs/campaigns/20260803_200356/topo_001_persistence_perslay/attempt_001/history.json` |
<!-- campaign-result:20260803_200356:end -->
