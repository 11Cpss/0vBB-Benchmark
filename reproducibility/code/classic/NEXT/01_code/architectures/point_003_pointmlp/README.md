# POINT-003：固定 kNN 的 Residual PointMLP-style 分类器

中文 | [English](README_EN.md)

## 1. 定位与研究假设

| 项目 | 值 |
|---|---|
| `architecture_id` | `point_003_pointmlp` |
| checkpoint `model_name` | `PointMLPClassifier` |
| Python class | `next_alt.models.point_sequence.PointMLPClassifier` |
| registry `input_kind` | `points` |
| 任务 | NEXT `0nubb`（label 1）与 `Bi214`（label 0）二分类 |
| 输出 | 每个事件一个未校准 signal logit，shape `(B,)` |
| 配置推导参数量 | **443,521** |

研究假设是：对居中能量 voxel 点云，一次固定 Euclidean kNN 建图、轻量局部几何仿射消息和深 residual MLP，可能已足以刻画双电子轨迹与单电子背景的局部形态差异，不必引入 attention 或动态图重建。

## 2. 原始数据、标签和切分

- 每个 HDF5 文件读取 `/MC/hits/table`，按连续 `event_id` 聚合事件，使用 `x`、`y`、`z`、`energy` 字段。
- `0nubb_part_*` 映射为 label 1，`Bi_part_*` 映射为 label 0。
- 完整相对 HDF5 路径作为 `group_id`；seed 42 按文件稳定切分为 0.8/0.1/0.1。同一文件不会跨 split。
- 本阶段只构建 train 和 validation loader，每类每个 split 最多 100 个文件。第三个 split 不读取；validation 只用于 early stopping 与 best checkpoint 选择。

对事件 hit $(\mathbf r_i,e_i)$，令完整事件能量 $E=\sum_i e_i$ 和能量加权质心

$$
\mathbf c=\frac{\sum_i e_i\mathbf r_i}{E}.
$$

先按

$$
\mathbf q_i=\left\lfloor\frac{\mathbf r_i-\mathbf c}{15\ \mathrm{mm}}\right\rfloor
$$

合并同一 cell。初始 voxel 中心为 $(\mathbf q+0.5)15\ \mathrm{mm}$，随后减去量化 voxel 的能量加权质心。若节点超过 512，按 voxel energy 降序保留 512 个，cell 坐标字典序破同分。坐标再除以 1000 mm。节点特征为

$$
f_i=\left[e_i^{\mathrm{voxel}}/E,\ \log(1+n_i^{\mathrm{rows}})\right].
$$

截断后 energy fraction 仍以完整事件 (E) 为分母。总能量和绝对探测器位置都不作为模型输入。

## 3. 输入合同

| 字段 | dtype / shape | 语义 |
|---|---|---|
| `coords` | float32 `(B,N,3)` | 居中、15 mm voxel 化、除以 1000 mm 的 XYZ |
| `features` | float32 `(B,N,2)` | energy fraction、`log1p(hit_count)` |
| `mask` | bool `(B,N)` | `true` 为有效节点，`false` 为 batch padding |
| output | float `(B,)` | 正值更倾向 `0nubb` 的 logit |

`1 ≤ N ≤ 512`。模型在有效节点上构造 kNN；padding 既不能成为邻居，也不参与 pooling。

## 4. 逐层架构

默认 (d=128,k=16,r=2,L=4)。kNN 只根据输入坐标计算一次并在四个 block 间共享；自环被排除。

| 阶段 | 运算 | 输入 → 输出 |
|---|---|---|
| 输入编码 | `Linear(5,128) → LayerNorm → GELU` | `(B,N,5) → (B,N,128)` |
| 固定邻域 | masked `cdist`、排除自身、top-16 | `(B,N,3) → indices (B,N,K)` |
| 每个 local branch | 拼接 $h_j-h_i,\Delta xyz,\|\Delta xyz\|_2$ | `132 → 128 → 128` |
| local 聚合 | masked neighbor max；`LayerNorm(h + aggregate)` | `(B,N,K,128) → (B,N,128)` |
| 每个 residual FFN | pre-LN，`Linear(128,256) → GELU → Dropout → Linear(256,128)` | `(B,N,128)` |
| block 重复 | 上述 local + FFN 共 4 次 | `(B,N,128)` |
| 事件池化 | masked mean 与 max 拼接 | `(B,N,128) → (B,256)` |
| 分类头 | `Linear(256,160) → LN → GELU → Dropout(0.1) → Linear(160,1)` | `(B,256) → (B,)` |

一个 block 的核心为

$$
m_i=\max_{j\in\mathcal N_k(i)}\phi([h_j-h_i,\mathbf r_j-\mathbf r_i,\|\mathbf r_j-\mathbf r_i\|]),
$$

$$
\tilde h_i=\mathrm{LN}(h_i+m_i),\qquad
h_i'=\tilde h_i+\mathrm{FFN}(\mathrm{LN}(\tilde h_i)).
$$

参数量分解：输入编码 1,024；四个 block 各 100,224；分类头 41,601；总计 443,521。

## 5. 配置和训练协议

| 类别 | 参数 | 值 |
|---|---|---:|
| 表示 | `point_bin_size` / `coordinate_scale` / `max_points` | 15 / 1000 / 512 |
| 模型 | `hidden_dim` / `num_blocks` / `expansion` | 128 / 4 / 2 |
| 模型 | `k` / `classifier_dim` / `dropout` | 16 / 160 / 0.10 |
| 数据 | `max_files_per_class` / `split_seed` | 100 / 42 |
| 训练 | `batch_size` / `epochs` | 12 / 50 |
| 训练 | `learning_rate` / `weight_decay` | 5e-4 / 1e-4 |
| 训练 | `gradient_clip_norm` / patience | 1.0 / 12 |
| 训练 | seed / AMP | 42 / `auto` |

共享 runner 使用 `BCEWithLogitsLoss`、AdamW、`CosineAnnealingLR(T_max=50)`，按 validation AUC 保存 best，另存 last。当前 balanced-class 路径交替读取两个类别；配置的 shuffle buffer 不应被误述为该路径已执行的 event-level buffer shuffle。

## 6. 复杂度和显存

- 距离矩阵时间与临时空间均为 (O(BN^2))，是主要显存瓶颈。
- block 消息大致为 (O(BLNkd^2))，邻居消息张量空间为 (O(BNkd))。
- 固定图避免每个 block 重算 kNN，但它也无法随 learned feature 更新邻域。

## 7. 与 PointMLP 原论文的边界

这是 **PointMLP-inspired/style** 模型，而不是原论文复现：它保留 residual MLP 与轻量几何归一化思想，但没有论文的 FPS 分层 stage、pre/post extraction block 数量、官方 geometric affine 公式、数据增强和训练 recipe；本实现用一次固定 kNN、显式 relative XYZ/distance、masked max 与 mean+max event head。代码是纯 PyTorch，也没有官方 point operator CUDA 扩展。因此训练结果只能归因于本项目变体。

## 8. 运行、tmux 和不可变产物

正式运行由 campaign 串行队列启动：

```bash
cd /home/wenyu/summer
RUN_ID=<timestamp-or-explicit-name>
tmux new-session -d -s "next-nontransformer-v2-${RUN_ID}" -n gpu-queue \
  "cd /home/wenyu/summer && bash 01_code/architectures/run_nontransformer_training_queue.sh --run-id ${RUN_ID}"
```

单模型入口为 `python 01_code/architectures/point_003_pointmlp/train_classification.py <config.snapshot.yaml>`，由队列调用。目标路径：

```text
02_models/checkpoints/<RUN_ID>/point_003_pointmlp/attempt_001/{best.pt,last.pt}
03_training_runs/campaigns/<RUN_ID>/point_003_pointmlp/attempt_001/
```

停止整个 campaign 可向 `gpu-queue` pane 发送 Ctrl-C。`--resume-queue` 跳过 `DONE`；失败重跑创建 `attempt_002` 并从头开始，存在 `last.pt` 不代表真正断点续训。禁止覆盖历史 attempt。

## 9. 训练结果（由 campaign 完成后填写）

| 状态 | epoch / best epoch | best val AUC / loss | 耗时 | early stop / retry | artifacts |
|---|---|---|---|---|---|
| `PRE-CAMPAIGN PLACEHOLDER`（实际状态见文末追加结果） | — | — | — | — | — |

这里只记录 train/validation，不记录延后阶段的正式评测指标。

## 10. 已知限制

- 512 点截断可能丢弃低能量拓扑；固定 k 在密度变化时对应不同物理半径。
- 输入未提供旋转不变或等变保证；绝对位置被刻意移除。
- dense `cdist` 对大 batch 显存不友好；纯 PyTorch top-k 不是高性能 point kernel。
- 单次固定图和单 seed 不能代表 PointMLP 方法族的完整能力。

## 11. 参考文献

1. Xu Ma, Can Qin, Haoxuan You, Haoxi Ran, Yun Fu, “Rethinking Network Design and Local Geometry in Point Cloud: A Simple Residual MLP Framework,” ICLR 2022, arXiv:2202.07123, [OpenReview](https://openreview.net/forum?id=3Pbra-_u76D), [arXiv](https://arxiv.org/abs/2202.07123).
2. NEXT Collaboration (J. Renner et al.), “Background rejection in NEXT using deep neural networks,” *JINST* 12 (2017) T01004, DOI: 10.1088/1748-0221/12/01/T01004, [official article](https://doi.org/10.1088/1748-0221/12/01/T01004).
3. NEXT Collaboration (F. Monrabal et al.), “The NEXT White (NEW) detector,” *JINST* 13 (2018) P12010, DOI: 10.1088/1748-0221/13/12/P12010, [official article](https://doi.org/10.1088/1748-0221/13/12/P12010).


<!-- campaign-result:20260803_200356:start -->
## Campaign `20260803_200356` 训练结果

本节由串行训练队列在训练成功后写入，只包含 train/validation 信息；本阶段没有读取 test split。

| 项目 | 实际值 |
|---|---|
| 状态 / attempt | `DONE` / `attempt_001` |
| 后端 | `pytorch` |
| 实际参数量 | 443,521 |
| 树数量 / 树节点数 | N/A / N/A |
| 完成 epoch / best epoch | 30 / 18 |
| best validation AUC | **0.976662** |
| best validation loss | 0.214011 |
| 总训练时间 | 00:27:55 |
| early stop | `true` |
| 失败重试 | `no` |
| Python / 框架 | `3.11.15` / `2.11.0+cu128` |
| 设备 | `NVIDIA GeForce RTX 5090` |
| best / last checkpoint | `/home/wenyu/summer/02_models/checkpoints/20260803_200356/point_003_pointmlp/attempt_001/best.pt` / `/home/wenyu/summer/02_models/checkpoints/20260803_200356/point_003_pointmlp/attempt_001/last.pt` |
| 训练日志 | `/home/wenyu/summer/03_training_runs/campaigns/20260803_200356/point_003_pointmlp/attempt_001/history.json` |
<!-- campaign-result:20260803_200356:end -->
