# SSM-001：纯 PyTorch Selective-Scan PointMamba-style 分类器

中文 | [English](README_EN.md)

## 1. 定位和研究假设

| 项目 | 值 |
|---|---|
| `architecture_id` | `ssm_001_pointmamba` |
| checkpoint `model_name` | `PointMambaLiteClassifier` |
| Python class | `next_alt.models.point_sequence.PointMambaLiteClassifier` |
| registry `input_kind` | `sequence` |
| 任务 | NEXT `0nubb`（label 1）与 `Bi214`（label 0）二分类 |
| 输出 | 每事件一个 `(B,)` 未校准 signal logit |
| 配置推导参数量 | **316,993** |
| backend | 纯 PyTorch、分块向量化 diagonal selective scan |

研究假设：若用 Hilbert 与 Trans-Hilbert 把稀疏 3D 轨迹转成局部连续序列，输入依赖的状态空间参数可在线性序列复杂度下传播全局拓扑信息；先读完整 Hilbert，再读 Trans-Hilbert，还允许第二个扫描继承第一个扫描的全局状态。

本实现没有 Transformer、self-attention 或 `mamba_ssm`/Triton/CUDA 扩展。

## 2. HDF5、标签、切分与 voxel 公式

共享 reader 读取每个 HDF5 的 `/MC/hits/table`，校验并按连续 `event_id` 聚合 `x/y/z/energy`。`0nubb_part_*` 为 label 1，`Bi_part_*` 为 label 0。完整相对文件路径作为 group；seed 42 做 file-level 0.8/0.1/0.1 稳定切分。同一源文件不跨 split。

本阶段只构建 train 与 validation loader，每类每个 split 最多 100 文件；validation 仅用于 early stopping 与 validation-AUC best selection。

对 event hits：

$$
E=\sum_i e_i,\qquad \mathbf c=\frac{\sum_i e_i\mathbf r_i}{E},\qquad
\mathbf q_i=\left\lfloor\frac{\mathbf r_i-\mathbf c}{15\ \mathrm{mm}}\right\rfloor.
$$

同 cell 行合并；中心 $(\mathbf q+0.5)15$ mm 再减去 voxel-energy 加权质心。若多于 512 voxel，按能量降序取 512 个并用 cell 字典序破同分。坐标除以 1000 mm；节点特征为

$$
f_i=[e_i^{\mathrm{voxel}}/E,\log(1+n_i^{\mathrm{rows}})].
$$

截断不重归一化 energy fraction。总事件能量和绝对位置不进入模型。

## 3. 输入与统一空间曲线

| 字段 | dtype / shape | 语义 |
|---|---|---|
| `coords` | float32 `(B,N,3)` | 居中 voxel XYZ / 1000 mm |
| `features` | float32 `(B,N,2)` | energy fraction、`log1p(hit_count)` |
| `mask` | bool `(B,N)` | 有效点，`1≤N≤512` |
| Hilbert tokens | float `(B,N,5)` | Hilbert 排序的 XYZ+feature |
| Trans-Hilbert tokens | float `(B,N,5)` | x/y 交换后 Hilbert 排序 |
| compact SSM sequence | float `(B,2N,128)` | 每事件有效的 Hilbert 后紧接有效 Trans-Hilbert，再 padding |
| output | float `(B,)` | signal logit |

空间曲线实现由三个序列模型共享：每事件按有效点包围盒把各轴量化到 10-bit `[0,1023]`，用 Skilling transpose algorithm 得到 3D Hilbert code 并稳定排序。Trans-Hilbert 明确定义为量化后交换 x/y 再编码；这是本项目 convention，不宣称是论文唯一轴约定。

两条有效序列在 batch 内被压紧为 `[Hilbert(valid), Trans-Hilbert(valid), padding]`，状态在两种扫描之间不断开。两类 token 分别使用可训练的 channel-wise scale/shift order indicator：

$$
z^{(o)}=e^{(o)}\odot\gamma_o+\beta_o,\qquad o\in\{H,T\}.
$$

## 4. 逐层架构

| 阶段 | 运算 | 输入 → 输出 |
|---|---|---|
| 共享 token encoder | `Linear(5,128) → LayerNorm → SiLU` | `(B,N,5) → (B,N,128)` |
| order indicator | 两组 `scale(128)+shift(128)` | 两条序列各 `(B,N,128)` |
| compact concat | 有效 H 后接有效 T | `(B,2N,128)` |
| SSM block ×3 | pre-LN、input/gate projection `128→384×2`、causal depthwise Conv1d `k=4`、selective SSM、gate、`192→128`、residual | `(B,2N,128)` |
| final norm | LayerNorm + mask | `(B,2N,128)` |
| event pool | masked mean + max | `(B,256)` |
| head | `Linear(256,160) → LN → SiLU → Dropout(0.1) → Linear(160,1)` | `(B,)` |

每个 SSM block 中，`inner_dim=192`、`state_dim=16`、`dt_rank=16`。depthwise causal convolution 后，由 token 投影产生 $\Delta_t,B_t,C_t$，并令 $A=-\exp(A_{\log})<0$：

$$
\bar A_t=\exp(\Delta_t A),\qquad
h_t=\bar A_t\odot h_{t-1}+\Delta_t\odot B_t\odot x_t,
$$

$$
y_t=\sum_s C_{t,s}h_{t,:,s}+D\odot x_t.
$$

输出再与 SiLU gate 相乘、线性投影并 residual add。`B_t` 和 `C_t` 随 token 变化；`A` 与 skip `D` 为 learned channel/state 参数。

参数量分解：token encoder 1,024；order indicators 512；每 block 91,200，三层 273,600；final norm 256；head 41,601；合计 316,993。

## 5. 纯 PyTorch scan fallback

这不是官方 Mamba hardware-aware kernel。为避免每 token 一个 Python 循环，recurrence 每 32 token 分块，并在块内使用 diagonal recurrence 的 prefix closed form：

$$
P_t=\prod_{j\le t}\bar A_j,\qquad
h_t=P_t\left(h_0+\sum_{i\le t}\frac{u_i}{P_i}\right),
\quad u_i=\Delta_iB_ix_i.
$$

块末状态精确传给下一块；scan 算术即使在 AMP 下也强制 float32。块内累计 `log(P)` 限制到 `[-60,0]` 以避免指数溢出，因此在极端 learned transition 下它是数值保护后的近似。默认较小的初始 transition rate 与 32-token chunk 用于降低触发保护的概率。fallback 会显式物化 channel×state activation，显存和速度均不能与官方 fused selective-scan kernel 相提并论。

## 6. YAML 与训练合同

| 类别 | 参数 | 值 |
|---|---|---:|
| 表示 | bin / scale / point cap | 15 mm / 1000 mm / 512 |
| 模型 | `model_dim` / `inner_dim` / `state_dim` | 128 / 192 / 16 |
| 模型 | `dt_rank` / layers / conv kernel | 16 / 3 / 4 |
| 模型 | Hilbert bits / chunk / head / dropout | 10 / 32 / 160 / 0.10 |
| 训练 | batch / epochs / lr | 4 / 50 / 3e-4 |
| 训练 | weight decay / clip / patience | 1e-4 / 1.0 / 12 |
| 训练 | seed / AMP | 42 / auto（scan 内部 float32） |

共享 runner 使用 `BCEWithLogitsLoss`、AdamW、`CosineAnnealingLR(T_max=50)`；按 validation AUC 选 best，同时保存 last。balanced-class 路径交替读取类别，配置 buffer 不能被描述为实际已执行 event buffer shuffle。

## 7. 复杂度与显存瓶颈

- 两次序列化排序约 $O(BN\log N)$。SSM sequence 长度为 $S=2N\le1024$。
- selective recurrence 时间为 (O(BSLd_{inner}d_{state}))，相对序列长度线性；depthwise conv 为 (O(BSLkd_{inner}))。
- 纯 PyTorch autograd 需保存分块 prefix/state，最坏 activation 约 (O(BSLd_{inner}d_{state}))，而不是官方 fused kernel 的内存行为；这是主要 VRAM 瓶颈，所以 batch 固定为 4。
- 没有 (O(S^2)) attention matrix。

## 8. 与 Mamba / PointMamba 的差异

这是 **PointMamba-inspired lite fallback**，不是官方复现：

- 原 PointMamba 先 FPS 选 keypoints，再以 kNN patch + lightweight PointNet tokenizer；本实现直接使用项目已有 15 mm voxel tokens，没有 FPS/patch tokenizer。
- 原论文默认 12 个 384-dim Mamba block；此处为 3 个 128-dim block。
- 保留 Hilbert、Trans-Hilbert、order indicator、两序列串联以及 plain non-hierarchical SSM 的思想。
- 不执行论文的 masked pretraining、ShapeNet transfer、augmentation 或下游 recipe。
- 官方 Mamba 使用硬件感知 fused scan；此处为分块 pure-PyTorch diagonal scan，数值与性能边界不同。
- 因而模型卡和结果只能使用 “inspired/lite/style”，不能声称官方 PointMamba 或 Mamba kernel 复现。

## 9. tmux、产物、停止和恢复

```bash
cd /home/wenyu/summer
RUN_ID=<timestamp-or-explicit-name>
tmux new-session -d -s "next-nontransformer-v2-${RUN_ID}" -n gpu-queue \
  "cd /home/wenyu/summer && bash 01_code/architectures/run_nontransformer_training_queue.sh --run-id ${RUN_ID}"
```

队列调用 `python 01_code/architectures/ssm_001_pointmamba/train_classification.py <config.snapshot.yaml>`。产物：

```text
02_models/checkpoints/<RUN_ID>/ssm_001_pointmamba/attempt_001/{best.pt,last.pt}
03_training_runs/campaigns/<RUN_ID>/ssm_001_pointmamba/attempt_001/
```

向 `gpu-queue` pane 发送 Ctrl-C 停止。`--resume-queue` 只跳过 `DONE`；失败时保留 attempt 并新建下一个 attempt 从 epoch 1 开始。存在 last checkpoint 不等于真正 resume。

## 10. 训练结果（campaign 后填写）

| 状态 | epoch / best epoch | best val AUC / loss | 耗时 | early stop / retry | artifacts |
|---|---|---|---|---|---|
| `PRE-CAMPAIGN PLACEHOLDER`（实际状态见文末追加结果） | — | — | — | — | — |

此处只允许 train/validation 结果。

## 11. 已知限制

- 人工曲线顺序、每事件 min-max 量化和 x/y Trans-Hilbert convention 会引入方向偏置。
- 512 点截断可能丢低能分支；直接 voxel token 缺少原论文局部 patch tokenizer。
- 分块 closed form 有 `log(P)` 数值截断，极端 transition 下不是逐步 recurrence 的位级等价。
- pure-PyTorch fallback 比官方 kernel 慢且耗显存；没有证明 batch 4 是所有 GPU 的最优值。
- 单 seed、固定模型规模不代表 Mamba/PointMamba 方法族能力。

## 12. 参考文献

1. Albert Gu, Tri Dao, “Mamba: Linear-Time Sequence Modeling with Selective State Spaces,” 2023/2024, arXiv:2312.00752, [arXiv](https://arxiv.org/abs/2312.00752).
2. Dingkang Liang, Xin Zhou, Wei Xu, Xingkui Zhu, Zhikang Zou, Xiaoqing Ye, Xiao Tan, Xiang Bai, “PointMamba: A Simple State Space Model for Point Cloud Analysis,” *Advances in Neural Information Processing Systems 37*, 2024, arXiv:2402.10739, [NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/395371f778ebd4854b88521100af30ad-Abstract-Conference.html).
3. John Skilling, “Programming the Hilbert Curve,” *AIP Conference Proceedings* 707 (2004), DOI: 10.1063/1.1751381, [official article](https://doi.org/10.1063/1.1751381).
4. NEXT Collaboration (J. Renner et al.), “Background rejection in NEXT using deep neural networks,” *JINST* 12 (2017) T01004, DOI: 10.1088/1748-0221/12/01/T01004, [official article](https://doi.org/10.1088/1748-0221/12/01/T01004).


<!-- campaign-result:20260803_200356:start -->
## Campaign `20260803_200356` 训练结果

本节由串行训练队列在训练成功后写入，只包含 train/validation 信息；本阶段没有读取 test split。

| 项目 | 实际值 |
|---|---|
| 状态 / attempt | `DONE` / `attempt_001` |
| 后端 | `pytorch` |
| 实际参数量 | 316,993 |
| 树数量 / 树节点数 | N/A / N/A |
| 完成 epoch / best epoch | 24 / 12 |
| best validation AUC | **0.928102** |
| best validation loss | 0.407766 |
| 总训练时间 | 01:14:38 |
| early stop | `true` |
| 失败重试 | `no` |
| Python / 框架 | `3.11.15` / `2.11.0+cu128` |
| 设备 | `NVIDIA GeForce RTX 5090` |
| best / last checkpoint | `/home/wenyu/summer/02_models/checkpoints/20260803_200356/ssm_001_pointmamba/attempt_001/best.pt` / `/home/wenyu/summer/02_models/checkpoints/20260803_200356/ssm_001_pointmamba/attempt_001/last.pt` |
| 训练日志 | `/home/wenyu/summer/03_training_runs/campaigns/20260803_200356/ssm_001_pointmamba/attempt_001/history.json` |
<!-- campaign-result:20260803_200356:end -->
