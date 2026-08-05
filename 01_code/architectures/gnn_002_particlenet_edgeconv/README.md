# GNN-002：ParticleNet-lite EdgeConv

中文 | [English](README_EN.md)

本目录定义一个面向 NEXT `0nubb` 与 `Bi214` 二分类任务的动态图分类器。它借鉴 ParticleNet 和 Dynamic Graph CNN（DGCNN）的 EdgeConv 思想，但网络宽度、边特征、残差块、事件池化和训练协议均为本项目的定制实现，不是原论文代码的逐项复现。

## 模型身份

| 项目 | 值 |
|---|---|
| `architecture_id` | `gnn_002_particlenet_edgeconv` |
| PyTorch 类 | `next_alt.models.point_graph.ParticleNetLiteClassifier` |
| `input_kind` | `graph` |
| 任务 | 二分类；`0nubb=1`，`Bi214=0` |
| 输出 | 每个事件一个未校准的 signal logit，shape `(B,)` |
| 可训练参数量 | 150,497 |
| checkpoint 格式 | EnergyBench/NEXT format version 3 |

实现只使用 PyTorch tensor 运算，不依赖 PyG、DGL、`torch_scatter` 或自定义 CUDA 邻居算子。

## 输入与精确预处理

对一个事件中的原始行记坐标和 deposited energy 为
$\{(\mathbf r_a,e_a)\}$，总能量为 $E=\sum_a e_a$。预处理严格由共享数据模块完成：

1. 计算原始行的能量加权中心
   $\mathbf c=\sum_a e_a\mathbf r_a/E$，并将坐标平移为
   $\mathbf r_a-\mathbf c$。
2. 用 `15 mm` 边长聚合 voxel：
   $\mathbf q_a=\lfloor(\mathbf r_a-\mathbf c)/15\,\mathrm{mm}\rfloor$。
3. 对每个 occupied voxel 得到 deposited energy
   $E_i=\sum_{a:q_a=i}e_a$ 和合并行数 $n_i$。
4. 以 voxel 几何中心 $(\mathbf q_i+0.5)\times15\,\mathrm{mm}$ 为起点，再减去量化后 voxel 的能量加权中心，以消除半格量化偏移。
5. 模型坐标为
   $\mathbf x_i=\mathbf x_i^{\mathrm{centered}}/1000.0$，即把以 mm 表示的中心坐标除以 `coordinate_scale=1000.0`；不是乘以 1000。
6. 两个节点特征为
   $\mathbf f_i=[E_i/E,\ \log(1+n_i)]$，对应 `energy_fraction` 和 `log1p(hit_count)`。
7. occupied voxel 多于 512 时，按 $E_i$ 从高到低保留 512 个，并用 XYZ 作确定性 tie-break。截断后的 `energy_fraction` 不重新归一化，因此 `point_coverage` 能反映实际保留的能量比例。
8. batch 内按该 batch 的最大节点数补零，产生 `coords: (B,N,3)`、`features: (B,N,2)` 和 boolean `mask: (B,N)`。mask 在建图、消息聚合和事件池化中始终生效。

总事件能量和绝对 detector position 不直接输入模型；总能量只随预测导出为 EnergyBench 的 `energy_condition`。少于 17 个有效节点以及单节点事件均由 neighbour mask 安全处理。

## 网络结构

### 1. 节点编码

每个节点先拼接坐标和两个标量特征：

```text
[x, y, z, energy_fraction, log1p(hit_count)]  # 5 dims
    -> Linear(5, 64)
    -> LayerNorm(64) -> SiLU -> Dropout(0.10)
    -> Linear(64, 64) -> SiLU
```

### 2. 动态 EdgeConv

对中心节点 $i$ 和邻居 $j$，边输入为

$$
\mathbf z_{ij}=
[\mathbf h_i,\ \mathbf h_j-\mathbf h_i,\
 \mathbf x_j-\mathbf x_i,\
 \lVert\mathbf x_j-\mathbf x_i\rVert_2].
$$

Edge MLP 后对邻居做 masked max，并通过残差支路更新节点：

$$
\mathbf h_i' = \operatorname{LayerNorm}\!\left(
W_{\rm skip}\mathbf h_i+
\operatorname{Dropout}\left[
\max_{j\in\mathcal N(i)}\operatorname{MLP}(\mathbf z_{ij})
\right]\right).
$$

| Stage | 节点维度 | kNN 搜索空间 | 边输入维度 | 输出维度 |
|---:|---:|---|---:|---:|
| 1 | 64 | 物理坐标 `coords` | `2×64+4=132` | 64 |
| 2 | 64 | 当前 `nodes / sqrt(64)` | `2×64+4=132` | 96 |
| 3 | 96 | 当前 `nodes / sqrt(96)` | `2×96+4=196` | 128 |

每一层都用 `k=16`、排除 self 的 kNN。第一层是几何图，后两层在 learned feature space 中重新建图。需要注意：后两层的 learned space 只负责选择邻居；送入 edge message 的 relative XYZ 和欧氏距离仍来自物理坐标。

### 3. 事件池化与分类头

```text
masked_mean(nodes) || masked_max(nodes)  # 128 + 128 = 256
    -> Linear(256, 192) -> SiLU -> Dropout(0.10)
    -> Linear(192, 1)
    -> squeeze -> (B,)
```

mean/max 池化使输出对节点排列不敏感。

## 与原始方法的边界

- [ParticleNet: Jet Tagging via Particle Clouds](https://arxiv.org/abs/1902.08570) 提供了把粒子云视为动态图并用于高能物理分类的主要动机。
- [Dynamic Graph CNN for Learning on Point Clouds](https://arxiv.org/abs/1801.07829) 提出了 EdgeConv 和逐层在特征空间重建邻居的基本方法。
- 本项目使用 15 mm NEXT voxel、两种项目专用节点特征、三层 `[64,96,128]` 残差 EdgeConv、显式 relative XYZ/distance、mean+max 事件池化以及自定义分类头。
- 本实现没有照搬 ParticleNet 的完整 block 深度、跨 block 特征组合、训练设置或官方加速算子。因此文档和论文中应称为 **ParticleNet-lite** 或 **ParticleNet-inspired EdgeConv**，不应称为 ParticleNet 复现。

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
| 模型 | `hidden_dims` | `[64, 96, 128]` |
| 模型 | `k` | 16 |
| 模型 | `classifier_dim` | 192 |
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

完整取值以相邻的 [`config.yaml`](config.yaml) 为准；共享默认值和校验规则见 [`src/next_alt/config.py`](../../../src/next_alt/config.py)。

## 训练机制与命令

- 完整 HDF5 文件按稳定 hash 分配到 train/validation/test，避免同一 source file 跨 split。
- 训练流在事件级交替输出 signal/background，并在较小类耗尽时结束，因此训练样本保持平衡。
- `event_shuffle_buffer_size=512` 虽存在于 YAML，但在当前 `balance_training_classes=true` 分支中不会启用 event-buffer shuffle；训练时实际逐 epoch shuffle 文件顺序。
- loss 为 `BCEWithLogitsLoss`，optimizer 为 AdamW，scheduler 为 50-epoch CosineAnnealingLR。
- best checkpoint 按 validation AUC 严格提升保存；同时每个 epoch 覆盖 last checkpoint。连续 12 epoch 没有改善则 early stop。
- CUDA 是正式训练的硬要求，不提供 CPU fallback 或 smoke-training 参数。

原始正式训练入口为：

```bash
cd /home/wenyu/summer
source .venv/bin/activate
python 01_code/architectures/gnn_002_particlenet_edgeconv/train_classification.py
```

该正式训练的 checkpoint、history 和 plot 已存在。默认 `allow_overwrite: false`，现在直接重复上述命令会拒绝覆盖并退出，这是预期保护。要做新实验，应复制 YAML、为 `output.checkpoint_dir`、`output.log_dir` 和 `output.plot_dir` 设置新的目录，再把新 YAML 作为唯一位置参数传入：

```bash
python 01_code/architectures/gnn_002_particlenet_edgeconv/train_classification.py \
  /path/to/gnn_002_new_experiment.yaml
```

## 已完成的正式结果

| 指标 | 结果 |
|---|---:|
| best epoch | 47 |
| best validation AUC | 0.971676 |
| 实际完成 epoch | 50 |
| full-test 总排名 | **2 / 10** |
| full-test matched AUC | **0.970363** |
| full-test inclusive AUC | 0.970754 |
| energy-independence mean | 0.979700 |

full-test 使用完整 test split：1,490 个文件、115,499 个事件；strict evaluation 为 0 warning、0 error。正式结果已经存在，不应覆盖。若确实需要重跑，使用新的 `_rerun` 目录且不要传 `--max-files-per-class`：

```bash
cd /home/wenyu/summer
source .venv/bin/activate
energybench next \
  02_models/checkpoints/NEXTALT_gnn_002_particlenet_edgeconv_classification_best.pt \
  --device cuda:0 \
  --split test \
  --batch-size 12 \
  --num-workers 0 \
  --model-id gnn_002_particlenet_edgeconv \
  --output-dir 04_evaluations/NEXTALT_gnn_002_particlenet_edgeconv_test_rerun
```

## 产物

| 类型 | 文件 |
|---|---|
| best checkpoint | [`NEXTALT_gnn_002_particlenet_edgeconv_classification_best.pt`](../../../02_models/checkpoints/NEXTALT_gnn_002_particlenet_edgeconv_classification_best.pt) |
| last checkpoint | [`NEXTALT_gnn_002_particlenet_edgeconv_classification_last.pt`](../../../02_models/checkpoints/NEXTALT_gnn_002_particlenet_edgeconv_classification_last.pt) |
| epoch CSV | [`classification_epochs.csv`](../../../03_training_runs/logs/NEXTALT_gnn_002_particlenet_edgeconv_classification_epochs.csv) |
| history JSON | [`classification_history.json`](../../../03_training_runs/logs/NEXTALT_gnn_002_particlenet_edgeconv_classification_history.json) |
| history plot | [`classification_history.png`](../../../03_training_runs/history_plots/NEXTALT_gnn_002_particlenet_edgeconv_classification_history.png) |
| test predictions | [`predictions_test.npz`](../../../04_evaluations/NEXTALT_gnn_002_particlenet_edgeconv_test/predictions_test.npz) |
| test results | [`evaluation_test/results.csv`](../../../04_evaluations/NEXTALT_gnn_002_particlenet_edgeconv_test/evaluation_test/results.csv) |
| strict metrics | [`evaluation_test/.energybench/metrics.json`](../../../04_evaluations/NEXTALT_gnn_002_particlenet_edgeconv_test/evaluation_test/.energybench/metrics.json) |
| ROC | [`energy_matched_roc.png`](../../../04_evaluations/NEXTALT_gnn_002_particlenet_edgeconv_test/evaluation_test/energy_matched_roc.png) |
| score-energy 图 | [`score_energy_dependence.png`](../../../04_evaluations/NEXTALT_gnn_002_particlenet_edgeconv_test/evaluation_test/score_energy_dependence.png) |
| 十模型排行榜 | [`leaderboard.csv`](../../../04_evaluations/NEXTALT_all_models_comparison/leaderboard.csv) |

## 本地关联文件

| 职责 | 文件 |
|---|---|
| 本模型配置 | [`config.yaml`](config.yaml) |
| 本模型训练入口 | [`train_classification.py`](train_classification.py) |
| ParticleNet-lite / EdgeConv 实现 | [`src/next_alt/models/point_graph.py`](../../../src/next_alt/models/point_graph.py) |
| voxel、padding、loader | [`src/next_alt/data.py`](../../../src/next_alt/data.py) |
| 架构 ID 与 input kind 注册 | [`src/next_alt/registry.py`](../../../src/next_alt/registry.py) |
| 共享训练循环 | [`src/next_alt/training.py`](../../../src/next_alt/training.py) |
| v3 checkpoint 合同 | [`src/next_alt/checkpoint.py`](../../../src/next_alt/checkpoint.py) |
| v3 推理适配 | [`src/next_alt/adapter.py`](../../../src/next_alt/adapter.py) |
| HDF5 读取与 file-level split | [`src/next_cnn/data.py`](../../../src/next_cnn/data.py) |
| 全架构说明 | [`docs/ALTERNATIVE_ARCHITECTURES.md`](../../../docs/ALTERNATIVE_ARCHITECTURES.md) |
| 正式评测结果 | [`docs/ALTERNATIVE_EVALUATION_RESULTS.md`](../../../docs/ALTERNATIVE_EVALUATION_RESULTS.md) |
| EnergyBench 使用方法 | [`docs/USAGE_GUIDE.md`](../../../docs/USAGE_GUIDE.md) |

## 局限与解释边界

- kNN 使用 dense `torch.cdist`，时间和中间显存约为 `O(BN²)`；`N=512` 是重要的资源上限。
- top-512 策略优先保留高能 voxel，可能丢失低能但拓扑上有意义的细枝结构。
- kNN 的离散邻居选择不可微；只有选中边上的连续网络参数参与反向传播。
- 归一化能量和移除绝对位置降低了直接能量/位置捷径，但 hit count、几何范围和截断率仍可能与能量相关，因此不能仅凭输入设计宣称完全 energy-independent。
- 当前结果来自一个数据 split、一个 seed 和一组超参数；十模型排名不等价于各方法经过同等规模调参后的理论上限。
- 模型输出是 logit，不是经过校准的后验概率。
