# POINT-004：纯 PyTorch 刚性 KPConv-style 点云分类器

中文 | [English](README_EN.md)

## 1. 定位与研究假设

POINT-004 检验一个几何归纳偏置：相对于普通 point MLP，围绕每个点放置一组固定的
三维 kernel points，并让邻居按几何距离对这些 kernel 贡献，是否更适合识别 NEXT 的
双电子轨迹与 blob 拓扑。模型不含 Transformer 或 attention。

| 项目 | 定义 |
|---|---|
| `architecture_id` | `point_004_rigid_kpconv` |
| checkpoint `model_name` | `RigidKPConvClassifier` |
| Python 实现 | `next_alt.models.mixer_sparse.RigidKPConvClassifier` |
| registry `input_kind` | `points` |
| 任务 | `0nubb`（1）与 `Bi214`（0）二分类 |
| 输出 | 每事件一个未校准 signal logit，shape `(B,)` |
| 默认可训练参数 | **390,977** |
| backend | dense kNN + 纯 PyTorch rigid-kernel fallback |

## 2. 原始数据、file-level split 与点云构造

共享读取器从 `/MC/hits/table` 读取 `event_id,x,y,z,energy`，按连续 event ID 聚合事件；
`0nubb_part_*` 为标签 1，`Bi_part_*` 为标签 0。完整相对 HDF5 路径作为 group，以 seed
42 和 `[0.8,0.1,0.1]` 做确定性 file-level split。本阶段只实例化 train/validation，
不读取第三个保留 split；每类最多选择 100 文件。

令事件 hit 为 $(\mathbf r_i,e_i)$，总能量 $E=\sum_i e_i$。预处理为：


$$
\mathbf c=\frac{\sum_i e_i\mathbf r_i}{E},\qquad
\mathbf q_i=\left\lfloor\frac{\mathbf r_i-\mathbf c}{15\;\mathrm{mm}}\right\rfloor.
$$

相同 $\mathbf q$ 的行合并，得到 voxel energy $E_q$ 与 hit count $n_q$。初始中心
$(\mathbf q+0.5)15$ mm 再减去其能量加权中心，以去掉共同的 half-bin offset。
若 voxel 超过 512，只保留能量最大的 512 个，坐标字典序作为稳定 tie breaker；能量
特征仍除以完整事件 $E$，截断后不重新归一化。

| batch 字段 | dtype / shape | 语义 |
|---|---|---|
| `coords` | float32 `(B,N,3)` | centered voxel center / 1000 mm；`N≤512` 后按 batch pad |
| `features` | float32 `(B,N,2)` | `[E_q/E, log1p(n_q)]` |
| `mask` | bool `(B,N)` | true 为有效 voxel，false 为 padding；所有邻域与池化都排除 false |
| `label` | float32 `(B,)` | 监督标签，不作为输入特征 |
| 输出 | floating `(B,)` | signal logits |

总能量和绝对 detector 位置不进入模型。

## 3. 刚性 kernel point convolution

每层先在 centered coordinates 上用 float32 `cdist` 建立最多 24 个 kNN，并进一步丢弃
半径 $r_l$ 外邻居。每个卷积有 15 个固定 kernel points：一个位于原点，其余 14 个
以 Fibonacci-sphere 顺序放在半径 $0.65r_l$ 的球面上。它们是 buffer，不训练、不
deform。

对 query $i$、邻居 $j$、kernel $k$，相对坐标
$\Delta_{ij}=\mathbf x_j-\mathbf x_i$，线性 influence 为


$$
a_{ijk}=\max\left(0,1-\frac{\lVert\Delta_{ij}-\mathbf p_k\rVert_2}{\sigma_l}\right),
\qquad
\bar a_{ijk}=\frac{a_{ijk}}{\max(\epsilon,\sum_j a_{ijk})}.
$$

本实现的输出为


$$
\mathbf g_i=\sum_{k=1}^{15}\left(\sum_{j\in\mathcal N(i)}
\bar a_{ijk}\mathbf h_j\right)\mathbf W_k+\mathbf b.
$$

mask 同时作用于 query、support、residual 输出和 event pooling。

## 4. 逐层结构与 shape

| 阶段 | 运算 | 输入 → 输出 |
|---|---|---|
| node input | concatenate XYZ 与两特征 | `(B,N,3)+(B,N,2) → (B,N,5)` |
| encoder | `Linear(5,64) → LayerNorm(64) → SiLU` | `(B,N,5) → (B,N,64)` |
| KP residual 0 | rigid KPConv `64→64`, `r=0.12`, `σ=0.06`, 24-NN, 15 kernels；LN→SiLU→Dropout；identity residual→SiLU | `(B,N,64) → (B,N,64)` |
| KP residual 1 | rigid KPConv `64→96`, `r=0.18`, `σ=0.09`；同样 norm/activation；`Linear(64,96)` shortcut | `(B,N,64) → (B,N,96)` |
| KP residual 2 | rigid KPConv `96→128`, `r=0.27`, `σ=0.135`；`Linear(96,128)` shortcut | `(B,N,96) → (B,N,128)` |
| event pooling | masked mean 与 masked max 拼接 | `(B,N,128) → (B,256)` |
| head | `Linear(256,128) → SiLU → Dropout(0.1) → Linear(128,1)` | `(B,256) → (B,1)` |
| 输出 | squeeze | `(B,1) → (B,)` |

坐标单位是 1000 mm，因此默认 radius 0.12/0.18/0.27 分别对应 120/180/270 mm。
网络不做 point subsampling，三个 block 始终保留相同的 valid-node mask。

## 5. 参数量、复杂度和显存

参数分解：input encoder 512；三个 residual blocks 分别 61,632、98,688、197,120；
head 33,025；总计 **390,977**。

对 batch (B)、padded 点数 (N)、邻居 (K=24)、kernel 数 (M=15)，每层 dense
kNN 的时间/空间分别为 (O(BN^2))；influence tensor 是 (O(BNKM))。kernel 加权还
需要约 (O(BNM C_{in}C_{out})) 的矩阵运算。三层都会重新计算 cdist；因此 `N×N`
float32 distance matrix 和 `(B,N,24,15)` influence 是预期显存/速度瓶颈，这也是正式
batch size 冻结为 8 的原因。该 fallback 优先可移植性，不保证官方 KPConv CUDA 的吞吐。

## 6. 完整冻结配置

[config.yaml](config.yaml) 是权威来源。表示参数：15 mm voxels、coordinates /1000、
最多 512 点。模型参数：`feature_dim=2`、`hidden_dims=[64,96,128]`、
`neighbour_count=24`、`kernel_point_count=15`、`base_radius=0.12`、
`base_sigma=0.06`、`radius_multiplier=1.5`、`classifier_dim=128`、dropout 0.10。

数据：每类最多 100 文件、seed 42、fractions `[0.8,0.1,0.1]`、workers 0、balanced
training、buffer 512。训练：batch 8、50 epochs、BCEWithLogitsLoss、AdamW learning
rate `5e-4`、weight decay `1e-4`、CosineAnnealingLR、clip 1.0、patience 12、min delta
0、seed 42、非 deterministic、AMP auto；best 只按 validation AUC。

## 7. 与 KPConv 论文的差异和声明边界

这是 **rigid KPConv-style lightweight fallback**，不是官方实现复现：

- kernel points 使用简单确定性 Fibonacci sphere，而不是论文/官方初始化与优化流程；
- influence 在每个 kernel 上对邻居归一化，是本项目的数值稳定选择；
- dense kNN+radius 代替高效 radius search；
- 不做 grid subsampling、strided KPConv、deformable KPConv 或 encoder-decoder；
- 三层全分辨率分类网络、LayerNorm、SiLU、mean+max head 均为项目专属；
- 没有官方 C++/CUDA kernel，性能不可与官方 backend 等同。

不能声称复现论文速度、精度、deformability 或大规模点云能力。

## 8. 参考文献

- Hugues Thomas, Charles R. Qi, Jean-Emmanuel Deschaud, Beatriz Marcotegui,
  François Goulette, and Leonidas J. Guibas, “KPConv: Flexible and Deformable
  Convolution for Point Clouds,” *IEEE/CVF International Conference on Computer
  Vision (ICCV)*, 2019, pp. 6411–6420, arXiv:1904.08889,
  DOI [10.1109/ICCV.2019.00651](https://doi.org/10.1109/ICCV.2019.00651),
  [arXiv](https://arxiv.org/abs/1904.08889).

## 9. tmux campaign 与产物

该模型是单 GPU 队列第 7 项：

```bash
RUN_ID=<timestamp-or-explicit-name>
tmux new-session -d -s "next-nontransformer-v2-${RUN_ID}" -n gpu-queue \
  "cd /home/wenyu/summer && bash 01_code/architectures/run_nontransformer_training_queue.sh --run-id ${RUN_ID}"
tmux new-window -t "next-nontransformer-v2-${RUN_ID}" -n monitor \
  "cd /home/wenyu/summer && bash 01_code/architectures/monitor_nontransformer_training.sh --run-id ${RUN_ID}"
```

checkpoint 路径为
`02_models/checkpoints/<RUN_ID>/point_004_rigid_kpconv/attempt_NNN/{best.pt,last.pt}`；
config snapshot、stdout、epochs CSV、history JSON/PNG 位于
`03_training_runs/campaigns/<RUN_ID>/point_004_rigid_kpconv/attempt_NNN/`。
停止时对 `gpu-queue` 发送 Ctrl-C 并保留 session。`--resume-queue` 跳过 DONE；FAILED/
PENDING 使用同一 RUN_ID 建新 attempt 并从头训练。`last.pt` 不能描述为 checkpoint resume，
旧 attempt 禁止覆盖。

## 10. 已知限制

- 512 点截断可能丢失低能量长尾；coverage 不会被偷偷归一化，但信息仍已丢失。
- 固定 k、radius 和 kernel placement 未适应不同事件密度。
- centered 输入移除绝对位置，也可能移除有判别力的 detector 信息。
- dense cdist 是二次复杂度，纯 PyTorch fallback 对更大 N 不经济。
- rigid kernels 不具备论文 deformable variant 的能力；mean/max 不是可解释物理量。

## 11. 训练结果（campaign 后填写）

训练前占位状态：**PENDING**（实际状态见文末追加结果）。只在真实训练后追加实际参数量/环境、完成与 best epoch、best
validation AUC/loss、耗时、checkpoint/log、early-stop 与 attempt/retry；不写保留 split
指标。


<!-- campaign-result:20260803_200356:start -->
## Campaign `20260803_200356` 训练结果

本节由串行训练队列在训练成功后写入，只包含 train/validation 信息；本阶段没有读取 test split。

| 项目 | 实际值 |
|---|---|
| 状态 / attempt | `DONE` / `attempt_001` |
| 后端 | `pytorch` |
| 实际参数量 | 390,977 |
| 树数量 / 树节点数 | N/A / N/A |
| 完成 epoch / best epoch | 17 / 5 |
| best validation AUC | **0.933581** |
| best validation loss | 0.353042 |
| 总训练时间 | 00:16:13 |
| early stop | `true` |
| 失败重试 | `no` |
| Python / 框架 | `3.11.15` / `2.11.0+cu128` |
| 设备 | `NVIDIA GeForce RTX 5090` |
| best / last checkpoint | `/home/wenyu/summer/02_models/checkpoints/20260803_200356/point_004_rigid_kpconv/attempt_001/best.pt` / `/home/wenyu/summer/02_models/checkpoints/20260803_200356/point_004_rigid_kpconv/attempt_001/last.pt` |
| 训练日志 | `/home/wenyu/summer/03_training_runs/campaigns/20260803_200356/point_004_rigid_kpconv/attempt_001/history.json` |
<!-- campaign-result:20260803_200356:end -->
