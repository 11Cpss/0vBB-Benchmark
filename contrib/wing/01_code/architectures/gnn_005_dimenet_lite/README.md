# GNN-005：DimeNet-lite 定向消息传递分类器

[中文](README.md) | [English](README_EN.md)

## 1. 定位与研究假设

| 项目 | 定义 |
|---|---|
| `architecture_id` | `gnn_005_dimenet_lite` |
| checkpoint `model_name` | `DimeNetLiteClassifier` |
| Python 类 | `next_alt.models.graph_topology.DimeNetLiteClassifier` |
| `input_kind` | `graph` |
| 任务 | NEXT `0nubb`（1）与 `Bi214`（0）二分类 |
| 输出 | 每事件一个未校准 signal logit，shape `(B,)`；值越大越偏向 `0nubb` |
| 精确可训练参数量 | **169,553** |

研究问题是：相对于仅用距离或普通 edge features 的图网络，显式的
三点夹角 $k\to j\to i$ 是否能更好地区分双电子信号的双 blob/弯曲轨迹与背景拓扑。
本实现没有 Transformer 或 attention。

## 2. 原始数据、标签与预处理

每个事件来自 HDF5 `/MC/hits/table` 中同一连续 `event_id` 的行，读取
`x/y/z/energy`。目录 `0nubb_part_*` 映射为 1，`Bi_part_*` 映射为 0。
完整相对文件路径作为 group；seed 42 的稳定 file-level hash 按
`[0.8, 0.1, 0.1]` 分配 train/validation/test。本阶段只打开 train 和 validation，
各 split 每类最多 100 个文件；validation 只用于 early stopping 和 best checkpoint。

对命中位置 $\mathbf r_i$ 与能量 $e_i$：

$$
E=\sum_i e_i,\qquad
\mathbf c=\frac{\sum_i e_i\mathbf r_i}{E},\qquad
\mathbf q_i=\left\lfloor\frac{\mathbf r_i-\mathbf c}{15\ \mathrm{mm}}\right\rfloor .
$$

同一 cell 合并；voxel center 取 $(\mathbf q+0.5)15$ mm，并再次减去量化后
energy-weighted center。超过 512 个 voxel 时按 voxel energy 降序、cell 坐标字典序
确定性保留 512 个。截断后 energy fraction 仍除以完整事件能量，不重新归一化：

$$
\mathbf x_v=\mathbf r^{\rm centered}_v/(1000\ \mathrm{mm}),\qquad
\mathbf f_v=[e_v/E,\ \log(1+n_v)].
$$

总能量只用于构造无量纲 fraction 和外部 `energy_condition`，不进入模型；绝对位置已
被 centering 删除。

## 3. 输入合同

| key | dtype 与 shape | 语义 |
|---|---|---|
| `coords` | float32 `(B,N,3)` | centered、除以 1000 mm 的 voxel center |
| `features` | float32 `(B,N,2)` | `[energy_fraction, log1p(hit_count)]` |
| `mask` | bool `(B,N)` | `true` 为真实点，`false` 为 batch padding |
| 输出 | floating `(B,)` | binary logit |

每个有效点在原始坐标上选最多 `k=8` 个其他有效点，得到有向 kNN 关系
`indices/mask: (B,N,K)`。邻接选择位于 `no_grad`；距离从 live tensor 重算。

## 4. 逐层架构

| 阶段 | 运算 | 输入 → 输出 |
|---|---|---|
| node encoder | `Linear(5,96) → LayerNorm → SiLU → Linear(96,96) → SiLU` | `(B,N,5) → (B,N,96)` |
| radial basis | 6 个 smooth-cutoff sine/Bessel-style basis，cutoff 1.0 scaled unit | `(B,N,K) → (B,N,K,6)` |
| edge encoder | `Linear(198,96) → LayerNorm → SiLU`，输入 `[h_i,h_j,RBF(d_ij)]` | `(B,N,K,198) → (B,N,K,96)` |
| directional block ×3 | triplet interaction、edge residual+LN、edge-to-node sum、node residual+LN | edge/node 维度保持 96 |
| event pool | masked node mean 与 max 拼接 | `(B,N,96) → (B,192)` |
| head | `Linear(192,128) → SiLU → Dropout(.1) → Linear(128,1)` | `(B,192) → (B,)` |

对 edge $j\to i$，枚举 $k\to j$ 且 $k\ne i$。令
$\theta_{kji}$ 为 $\mathbf r_i-\mathbf r_j$ 与
$\mathbf r_k-\mathbf r_j$ 的夹角，角基为
$[\cos(0\theta),\ldots,\cos(3\theta)]$。一个 block 的核心为：

$$
u_{ji}=\frac{1}{\sqrt{|T_{ji}|}}\sum_{k\in T_{ji}}
W_hh_{kj}\odot W_rR(d_{jk})\odot W_aC(\theta_{kji}),
$$
$$
h'_{ji}=\mathrm{LN}(h_{ji}+\mathrm{Dropout}(W_u[u_{ji}\odot W_rR(d_{ij})])).
$$

edge 经 MLP 后按邻居求和并除以 $\sqrt{\deg(i)}$，作为 node residual update。

## 5. 复杂度与显存

设点数为 $N\le512$、邻居数 $K=8$、hidden $H=96$、interaction $I=48$。
kNN 的 dense distance 时间/空间为 (O(BN^2))；triplet interaction 时间约
(O(BNK^2I))，主要中间张量 `(B,N,K,K,H)`。正式 batch size 固定为 4；预期显存
瓶颈是反向传播保留的 triplet embeddings，而不是参数本身。

## 6. 完整配置

| 类别 | 参数 | 值 |
|---|---|---:|
| data | root / max files | `/home/klz/Data/zeronu_benchmark/NEXT` / 100 每类 |
| data | split seed / fractions | 42 / `[0.8,0.1,0.1]` |
| data | workers / balanced / buffer | 0 / true / 512 |
| representation | bin / coordinate scale / max points | 15 mm / 1000 mm / 512 |
| model | feature / hidden / interaction | 2 / 96 / 48 |
| model | blocks / k | 3 / 8 |
| model | radial / angular / cutoff | 6 / 4 / 1.0 |
| model | classifier / dropout | 128 / 0.10 |
| training | batch / epochs / lr | 4 / 50 / 4e-4 |
| training | AdamW weight decay | 1e-4 |
| training | clip / patience / min delta | 1.0 / 12 / 0.0 |
| training | seed / deterministic | 42 / false |
| training | AMP | true / auto |

训练采用 `BCEWithLogitsLoss`、AdamW、`CosineAnnealingLR(T_max=50)`，每 epoch
validation；best 按 validation AUC，early stop patience 12。

## 7. 与原论文的边界

这是 **DimeNet-inspired lite classifier**，不是论文复现：

- 图由 voxel kNN 构造，不是分子 cutoff graph；输入也没有原子种类；
- node encoder 直接拼接 centered XYZ，因此整体模型并非论文式 rotation-invariant 模型；
- 径向基为 smooth-cutoff sine/Bessel-style，角基为 Chebyshev 递推的
  `cos(mθ)`，没有论文完整 spherical Bessel/spherical harmonics basis；
- 只用 3 个紧凑 block，没有 DimeNet 的完整 output blocks、bilinear layers 与
  molecular-energy readout；
- 任务是 event binary classification，不声称量子化学精度或论文的旋转等变消息表示；
- 纯 PyTorch dense masked implementation，没有 PyG 专用 kernel。

## 8. 运行、tmux 与产物

直接入口（正式 campaign 由 queue 调用）：

```bash
cd /home/wenyu/summer
source .venv/bin/activate
python 01_code/architectures/gnn_005_dimenet_lite/train_classification.py CONFIG_SNAPSHOT
```

campaign session 为 `next-nontransformer-v2-<RUN_ID>`，`gpu-queue` 串行训练，
`monitor` 只读。产物位于：

```text
02_models/checkpoints/<RUN_ID>/gnn_005_dimenet_lite/attempt_NNN/{best.pt,last.pt}
03_training_runs/campaigns/<RUN_ID>/gnn_005_dimenet_lite/attempt_NNN/
  stdout.log  config.snapshot.yaml  epochs.csv  history.json  history.png
```

停止使用 `tmux send-keys -t <session>:gpu-queue C-c`。`--resume-queue` 跳过 DONE；
FAILED 会创建下一个 attempt 并从头训练，不能把 `last.pt` 描述为真正断点续训。

## 9. 已知限制

- dense kNN 与显式 (K^2) triplets 随点数增长快；512 点截断可能丢失低能尾迹；
- cutoff 1.0 等于 1000 mm，但 kNN 仍可能产生 cutoff 外、basis 为零的 edge；
- kNN tie、triplet 枚举及 AMP 会影响严格可重复性；
- total energy 与绝对位置被刻意排除，可能降低原始分类性能但减少能量捷径；
- 本阶段不加载 test split，不提供 test 指标或排行榜。

## 10. 训练结果（campaign 完成后追加）

本段为训练前占位说明；实际状态见文末追加结果。campaign 完成后在此追加实际参数量、环境、完成/best epoch、best
validation AUC/loss、总耗时、best/last 路径、early-stop 与失败重试；不得写 test 指标。

## 11. 参考文献

1. Johannes Gasteiger, Janek Groß, Stephan Günnemann, “Directional Message Passing for
   Molecular Graphs,” *ICLR 2020*, arXiv:2003.03123.
   [OpenReview](https://openreview.net/forum?id=B1eWbxStPH) ·
   [arXiv](https://arxiv.org/abs/2003.03123)
2. NEXT Collaboration, P. Ferrario et al., “First proof of topological signature in the high
   pressure xenon gas TPC with electroluminescence amplification for the NEXT experiment,”
   *JHEP* 2016, 104 (2016), DOI `10.1007/JHEP01(2016)104`, arXiv:1507.05902.
   [Journal](https://link.springer.com/article/10.1007/JHEP01%282016%29104) ·
   [arXiv](https://arxiv.org/abs/1507.05902)


<!-- campaign-result:20260803_200356:start -->
## Campaign `20260803_200356` 训练结果

本节由串行训练队列在训练成功后写入，只包含 train/validation 信息；本阶段没有读取 test split。

| 项目 | 实际值 |
|---|---|
| 状态 / attempt | `DONE` / `attempt_001` |
| 后端 | `pytorch` |
| 实际参数量 | 169,553 |
| 树数量 / 树节点数 | N/A / N/A |
| 完成 epoch / best epoch | 38 / 26 |
| best validation AUC | **0.980804** |
| best validation loss | 0.246705 |
| 总训练时间 | 00:51:49 |
| early stop | `true` |
| 失败重试 | `no` |
| Python / 框架 | `3.11.15` / `2.11.0+cu128` |
| 设备 | `NVIDIA GeForce RTX 5090` |
| best / last checkpoint | `/home/wenyu/summer/02_models/checkpoints/20260803_200356/gnn_005_dimenet_lite/attempt_001/best.pt` / `/home/wenyu/summer/02_models/checkpoints/20260803_200356/gnn_005_dimenet_lite/attempt_001/last.pt` |
| 训练日志 | `/home/wenyu/summer/03_training_runs/campaigns/20260803_200356/gnn_005_dimenet_lite/attempt_001/history.json` |
<!-- campaign-result:20260803_200356:end -->
