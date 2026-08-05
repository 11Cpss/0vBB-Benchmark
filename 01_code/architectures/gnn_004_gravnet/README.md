# GNN-004：GravNet-style learned-space GNN

中文 | [English](README_EN.md)

本目录定义一个面向 NEXT `0nubb` 与 `Bi214` 二分类任务的 learned-space 图网络。每个 block 从当前节点状态学习一个低维坐标空间，在该空间中构造邻域并用距离加权传播特征。它借鉴 GravNet 的核心 inductive bias，但 block、pooling、分类头和训练协议均为本项目定制，并非原论文实现的逐项复现。

## 模型身份

| 项目 | 值 |
|---|---|
| `architecture_id` | `gnn_004_gravnet` |
| PyTorch 类 | `next_alt.models.point_graph.GravNetClassifier` |
| `input_kind` | `graph` |
| 任务 | 二分类；`0nubb=1`，`Bi214=0` |
| 输出 | 每个事件一个未校准的 signal logit，shape `(B,)` |
| 可训练参数量 | 293,585 |
| checkpoint 格式 | EnergyBench/NEXT format version 3 |

模型完全由 PyTorch tensor 运算实现，不依赖 PyG、DGL、`torch_scatter` 或编译版 GravNet operator。

## 输入与精确预处理

对事件原始坐标和 deposited energy 记为
$\{(\mathbf r_a,e_a)\}$，总能量 $E=\sum_a e_a$。实际预处理如下：

1. 计算能量加权中心
   $\mathbf c=\sum_a e_a\mathbf r_a/E$，并将原始坐标变换为
   $\mathbf r_a-\mathbf c$。
2. 用 `15 mm` voxel 聚合：
   $\mathbf q_a=\lfloor(\mathbf r_a-\mathbf c)/15\,\mathrm{mm}\rfloor$。
3. 对每个 occupied voxel 计算 deposited energy
   $E_i=\sum_{a:q_a=i}e_a$ 和原始合并行数 $n_i$。
4. voxel 几何中心为 $(\mathbf q_i+0.5)\times15\,\mathrm{mm}$，然后再减去量化 voxel 的能量加权中心，消除半格量化偏移。
5. 输入坐标是
   $\mathbf x_i=\mathbf x_i^{\mathrm{centered}}/1000.0$。
6. 输入节点特征是
   $\mathbf f_i=[E_i/E,\ \log(1+n_i)]$，即 `energy_fraction` 与 `log1p(hit_count)`。
7. occupied voxel 多于 512 时，按 deposited energy 降序保留 512 个；XYZ 用作确定性 tie-break。截断后不重新归一化能量分数。
8. batch 内补零并生成 `coords: (B,N,3)`、`features: (B,N,2)` 与 boolean `mask: (B,N)`。mask 保护 learned-space kNN、聚合和 event pooling。

总事件能量和绝对 detector position 不作为模型输入。总能量仅作为 EnergyBench 评测条件随预测输出。

## 网络结构

### 1. 节点编码

```text
[x, y, z, energy_fraction, log1p(hit_count)]  # 5 dims
    -> Linear(5, 128)
    -> LayerNorm(128) -> SiLU -> Dropout(0.10)
    -> Linear(128, 128) -> SiLU
```

物理 XYZ 只在这个初始节点编码阶段直接出现；后续每个 block 的邻居由 learned coordinate 决定。

### 2. 四个 GravNet-style residual block

每个 block 接收 $\mathbf h_i\in\mathbb R^{128}$，独立投影出 learned coordinate 与 propagated feature：

$$
\mathbf s_i=W_s\mathbf h_i\in\mathbb R^4,
\qquad
\mathbf p_i=W_p\mathbf h_i\in\mathbb R^{64}.
$$

在 $\mathbf s$ 空间中构建排除 self 的 `k=16` kNN，并计算：

$$
d_{ij}^2=\lVert\mathbf s_i-\mathbf s_j\rVert_2^2,
\qquad
w_{ij}=\exp(-d_{ij}^2).
$$

两个聚合分别为归一化的 distance-weighted mean 和 masked max：

$$
\mathbf a_i^{\rm mean}=
\frac{\sum_{j\in\mathcal N(i)}w_{ij}\mathbf p_j}
{\max(\sum_{j\in\mathcal N(i)}w_{ij},10^{-8})},
$$

$$
\mathbf a_i^{\rm max}=\max_{j\in\mathcal N(i)}\mathbf p_j.
$$

随后把当前状态与两个 64 维聚合拼接：

```text
[h_i, weighted_mean_i, masked_max_i]  # 128 + 64 + 64 = 256
    -> MLP 256 -> 128 -> 128
    -> Dropout(0.10)
    -> residual add with h_i
    -> LayerNorm(128)
```

四个 block 都从更新后的当前节点状态重新学习 4D coordinate space。kNN index 的选择本身是离散的；代码会用 live learned coordinates 重算被选中边的距离，使 `exp(-d²)` 权重仍能向 `space_projection` 传播梯度。

### 3. 事件池化与分类头

```text
masked_mean(nodes) || masked_max(nodes)  # 128 + 128 = 256
    -> Linear(256, 160) -> SiLU -> Dropout(0.10)
    -> Linear(160, 1)
    -> squeeze -> (B,)
```

## 与原始方法的边界

- 原始参考是 Qasim、Kieseler、Iiyama 与 Pierini 的 [Learning representations of irregular particle-detector geometry with distance-weighted graph networks](https://arxiv.org/abs/1902.07987)，该工作提出 GarNet 与 GravNet layer。
- 本项目保留“从 feature 学习 coordinate space、在 learned space 建邻域、使用距离衰减传播特征”的核心思想。
- 原论文主要面向不规则粒子探测器的重建/聚类；本项目把定制的四层 residual GravNet-style block 用于事件级二分类。
- 本实现采用明确的 `exp(-d²)` normalized weighted mean、unweighted masked max、LayerNorm residual 和 mean+max event pooling；这些具体选择不应视作论文完整网络的复现。
- 因此正式描述应使用 **GravNet-style learned-space GNN**，而不是声称复现原始 GravNet architecture 或官方实现。

## 关键配置

| 类别 | 配置项 | 值 |
|---|---|---:|
| 数据 | `max_files_per_class` | 100 |
| 数据 | `split_fractions` | `[0.8, 0.1, 0.1]` |
| 数据 | `split_seed` | 42 |
| 表示 | `point_bin_size` | 15.0 mm |
| 表示 | `coordinate_scale` | 1000.0 |
| 表示 | `max_points` | 512 |
| 模型 | `feature_dim` | 2 |
| 模型 | `hidden_dim` | 128 |
| 模型 | `num_layers` | 4 |
| 模型 | `space_dim` | 4 |
| 模型 | `propagate_dim` | 64 |
| 模型 | `k` | 16 |
| 模型 | `classifier_dim` | 160 |
| 模型 | `dropout` | 0.10 |
| 训练 | `batch_size` | 12 |
| 训练 | `epochs` | 50 |
| 训练 | `learning_rate` | `5e-4` |
| 训练 | `weight_decay` | `1e-4` |
| 训练 | `gradient_clip_norm` | 1.0 |
| 训练 | `early_stopping_patience` | 12 |
| 训练 | `seed` | 42 |
| 训练 | `use_amp` / `amp_precision` | `true` / `auto` |
| 训练 | `deterministic` | `false` |

完整配置见 [`config.yaml`](config.yaml)，共享默认值和校验见 [`src/next_alt/config.py`](../../../src/next_alt/config.py)。

## 训练机制与命令

- 完整 HDF5 source file 按稳定 hash 切分为 train/validation/test，避免 source-file leakage。
- 训练集通过交替 signal/background event 保持类别平衡，并在较小类耗尽时结束。
- `balance_training_classes=true` 的当前代码路径不会启用配置中的 `event_shuffle_buffer_size=512`；实际每个 epoch shuffle 文件顺序。
- 使用 `BCEWithLogitsLoss`、AdamW 和 50-epoch CosineAnnealingLR。
- validation AUC 选择 best checkpoint；last checkpoint 每 epoch 保存；12 epoch 不改善时 early stop。
- 正式训练强制 CUDA，不提供 CPU fallback 或 smoke-training mode。

原始正式训练入口为：

```bash
cd /home/wenyu/summer
source .venv/bin/activate
python 01_code/architectures/gnn_004_gravnet/train_classification.py
```

正式训练产物已经存在。共享默认配置为 `allow_overwrite: false`，因此现在直接重跑会拒绝覆盖并退出。新实验必须复制 YAML 并指定新的 checkpoint/log/plot 输出目录：

```bash
python 01_code/architectures/gnn_004_gravnet/train_classification.py \
  /path/to/gnn_004_new_experiment.yaml
```

## 已完成的正式结果

| 指标 | 结果 |
|---|---:|
| best epoch | 42 |
| best validation AUC | 0.956015 |
| 实际完成 epoch | 50 |
| full-test 总排名 | **4 / 10** |
| full-test matched AUC | **0.955680** |
| full-test inclusive AUC | 0.955733 |
| energy-independence mean | 0.977351 |

full-test 使用完整 test split 的 1,490 个文件、115,499 个事件，strict evaluation 为 0 warning、0 error。正式结果目录已存在；重跑时必须使用新的 `_rerun` 目录，并且不限制文件数：

```bash
cd /home/wenyu/summer
source .venv/bin/activate
energybench next \
  02_models/checkpoints/NEXTALT_gnn_004_gravnet_classification_best.pt \
  --device cuda:0 \
  --split test \
  --batch-size 12 \
  --num-workers 0 \
  --model-id gnn_004_gravnet \
  --output-dir 04_evaluations/NEXTALT_gnn_004_gravnet_test_rerun
```

不要为正式 full-test 重跑传 `--max-files-per-class`。

## 产物

| 类型 | 文件 |
|---|---|
| best checkpoint | [`NEXTALT_gnn_004_gravnet_classification_best.pt`](../../../02_models/checkpoints/NEXTALT_gnn_004_gravnet_classification_best.pt) |
| last checkpoint | [`NEXTALT_gnn_004_gravnet_classification_last.pt`](../../../02_models/checkpoints/NEXTALT_gnn_004_gravnet_classification_last.pt) |
| epoch CSV | [`classification_epochs.csv`](../../../03_training_runs/logs/NEXTALT_gnn_004_gravnet_classification_epochs.csv) |
| history JSON | [`classification_history.json`](../../../03_training_runs/logs/NEXTALT_gnn_004_gravnet_classification_history.json) |
| history plot | [`classification_history.png`](../../../03_training_runs/history_plots/NEXTALT_gnn_004_gravnet_classification_history.png) |
| test predictions | [`predictions_test.npz`](../../../04_evaluations/NEXTALT_gnn_004_gravnet_test/predictions_test.npz) |
| test results | [`evaluation_test/results.csv`](../../../04_evaluations/NEXTALT_gnn_004_gravnet_test/evaluation_test/results.csv) |
| strict metrics | [`evaluation_test/.energybench/metrics.json`](../../../04_evaluations/NEXTALT_gnn_004_gravnet_test/evaluation_test/.energybench/metrics.json) |
| ROC | [`energy_matched_roc.png`](../../../04_evaluations/NEXTALT_gnn_004_gravnet_test/evaluation_test/energy_matched_roc.png) |
| score-energy 图 | [`score_energy_dependence.png`](../../../04_evaluations/NEXTALT_gnn_004_gravnet_test/evaluation_test/score_energy_dependence.png) |
| 十模型排行榜 | [`leaderboard.csv`](../../../04_evaluations/NEXTALT_all_models_comparison/leaderboard.csv) |

## 本地关联文件

| 职责 | 文件 |
|---|---|
| 本模型配置 | [`config.yaml`](config.yaml) |
| 本模型训练入口 | [`train_classification.py`](train_classification.py) |
| GravNet block 与 classifier | [`src/next_alt/models/point_graph.py`](../../../src/next_alt/models/point_graph.py) |
| voxel、padding、loader | [`src/next_alt/data.py`](../../../src/next_alt/data.py) |
| 架构注册 | [`src/next_alt/registry.py`](../../../src/next_alt/registry.py) |
| 共享训练循环 | [`src/next_alt/training.py`](../../../src/next_alt/training.py) |
| v3 checkpoint 合同 | [`src/next_alt/checkpoint.py`](../../../src/next_alt/checkpoint.py) |
| v3 推理适配 | [`src/next_alt/adapter.py`](../../../src/next_alt/adapter.py) |
| HDF5 读取与 file-level split | [`src/next_cnn/data.py`](../../../src/next_cnn/data.py) |
| 全架构说明 | [`docs/ALTERNATIVE_ARCHITECTURES.md`](../../../docs/ALTERNATIVE_ARCHITECTURES.md) |
| 正式评测结果 | [`docs/ALTERNATIVE_EVALUATION_RESULTS.md`](../../../docs/ALTERNATIVE_EVALUATION_RESULTS.md) |
| EnergyBench 使用方法 | [`docs/USAGE_GUIDE.md`](../../../docs/USAGE_GUIDE.md) |

## 局限与解释边界

- 四层都在 learned space 中执行 dense `torch.cdist`，时间和中间显存约为 `O(4BN²)`。
- learned-space top-k index 是离散且不可微的；只有被选中邻居上的连续距离权重向 space projection 传梯度。
- learned coordinate 可能发生尺度膨胀、收缩或局部塌缩；`exp(-d²)` 在距离较大时会快速衰减。当前实现依靠 residual、LayerNorm 和归一化分母缓解，而不是从理论上消除这些现象。
- top-512 高能优先截断可能移除低能长程拓扑。
- 物理坐标只在初始 encoder 中直接使用；若 learned neighbourhood 丢失有用的 detector geometry，后续 block 不会显式恢复固定几何图。
- 输入归一化和中心化不构成严格的 energy-independence 保证。
- 正式结果来自单一 split、seed 和超参数配置，不能当作原始 GravNet 方法经过完整调参后的性能上限。
- 输出是未校准 logit，不应直接解释为概率。
