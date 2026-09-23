# GNN-003：E(n)-equivariant GNN

中文 | [English](README_EN.md)

本目录定义一个面向 NEXT `0nubb` 与 `Bi214` 二分类任务的 EGNN-style 动态图模型。节点状态保持为旋转、平移和反射下的标量，消息使用节点状态与成对距离，坐标沿相对方向做等变更新；最终分类同时汇聚 invariant 节点状态和 invariant 半径平方统计。

## 模型身份

| 项目 | 值 |
|---|---|
| `architecture_id` | `gnn_003_egnn` |
| PyTorch 类 | `next_alt.models.point_graph.EGNNClassifier` |
| `input_kind` | `graph` |
| 任务 | 二分类；`0nubb=1`，`Bi214=0` |
| 输出 | 每个事件一个未校准的 invariant signal logit，shape `(B,)` |
| 可训练参数量 | 640,518 |
| checkpoint 格式 | EnergyBench/NEXT format version 3 |

实现只使用 PyTorch，不依赖 PyG、DGL、e3nn 或自定义 CUDA extension。

## 输入与精确预处理

对一个事件中的原始行记坐标和 deposited energy 为
$\{(\mathbf r_a,e_a)\}$，总能量为 $E=\sum_a e_a$。共享预处理执行以下步骤：

1. 计算能量加权中心
   $\mathbf c=\sum_a e_a\mathbf r_a/E$，先将所有原始坐标平移为
   $\mathbf r_a-\mathbf c$。
2. 用 `15 mm` 边长重新聚合 occupied voxel：
   $\mathbf q_a=\lfloor(\mathbf r_a-\mathbf c)/15\,\mathrm{mm}\rfloor$。
3. 每个 voxel 得到 deposited energy
   $E_i=\sum_{a:q_a=i}e_a$ 与合并行数 $n_i$。
4. voxel 几何中心为 $(\mathbf q_i+0.5)\times15\,\mathrm{mm}$，随后再减去量化 voxel 的能量加权中心，以移除半格量化偏移。
5. 模型初始坐标为
   $\mathbf x_i=\mathbf x_i^{\mathrm{centered}}/1000.0$。这里是把 mm 坐标除以 `coordinate_scale=1000.0`。
6. scalar node feature 为
   $\mathbf f_i=[E_i/E,\ \log(1+n_i)]$。
7. 超过 512 个 occupied voxel 时，按 deposited energy 从高到低保留 512 个，并以 XYZ 确定性打破平局。截断后不重新归一化 `energy_fraction`。
8. batch 内补零并生成 `coords: (B,N,3)`、`features: (B,N,2)` 和 `mask: (B,N)`；mask 会排除 padding node、padding neighbour、坐标更新和池化值。

总事件能量与绝对 detector position 不直接输入分类器。总能量仅作为 EnergyBench 的 `energy_condition` 随预测导出。

## 网络结构

### 1. invariant 节点编码

初始节点编码器**不读取 XYZ**，只读取两个 scalar feature：

```text
[energy_fraction, log1p(hit_count)]  # 2 dims
    -> Linear(2, 128)
    -> LayerNorm(128) -> SiLU -> Dropout(0.10)
    -> Linear(128, 128) -> SiLU
```

这样坐标分量不会被直接混入 scalar node state。

### 2. 五层 EGNN-style 更新

每层先在当前演化坐标 $\mathbf x$ 中构建排除 self 的 `k=16` kNN。对边 $i\leftarrow j$：

$$
d_{ij}^2=\lVert\mathbf x_i-\mathbf x_j\rVert_2^2,
\qquad
\mathbf m_{ij}=\phi_e([\mathbf h_i,\mathbf h_j,d_{ij}^2]),
$$

其中 $\phi_e$ 是 `257 -> 128 -> 128` 的 MLP。消息用 masked mean 聚合，随后做残差 feature update：

$$
\bar{\mathbf m}_i=\operatorname{mean}_{j\in\mathcal N(i)}\mathbf m_{ij},
$$

$$
\mathbf h_i'=\operatorname{LayerNorm}\left(
\mathbf h_i+\operatorname{Dropout}
\left[\phi_h([\mathbf h_i,\bar{\mathbf m}_i])\right]
\right),
$$

其中 $\phi_h$ 为 `256 -> 128 -> 128`。坐标更新使用归一化相对方向：

$$
\mathbf u_{ij}=\frac{\mathbf x_i-\mathbf x_j}
{\sqrt{d_{ij}^2+10^{-8}}},
\qquad
\alpha_{ij}=\tanh(\phi_x(\mathbf m_{ij})),
$$

$$
\mathbf x_i'=\mathbf x_i+0.10\cdot
\operatorname{mean}_{j\in\mathcal N(i)}
(\mathbf u_{ij}\alpha_{ij}).
$$

`coord_scale=0.10` 限制每层的坐标漂移。更新后的坐标用于下一层重新建图。

### 3. invariant 几何统计与分类头

五层之后先汇聚 scalar node state：

```text
masked_mean(h) || masked_max(h)  # 128 + 128 = 256
```

然后计算演化坐标的 masked arithmetic center
$\bar{\mathbf x}=\operatorname{mean}_i\mathbf x_i$，以及
$r_i^2=\lVert\mathbf x_i-\bar{\mathbf x}\rVert_2^2$，再汇聚：

```text
masked_mean(radius_squared) || masked_max(radius_squared)  # 2 dims
```

最终分类头是：

```text
[node mean/max, radius_squared mean/max]  # 256 + 2 = 258
    -> Linear(258, 160) -> SiLU -> Dropout(0.10)
    -> Linear(160, 1)
    -> squeeze -> (B,)
```

这修正了旧文档中“最终只汇聚 scalar node state”的误述：实际模型还显式使用两个 radius-squared 几何统计。它们仍对共同平移、旋转和反射保持不变，因此不会破坏 event logit 的 E(3) invariance。

## 与原始方法的边界

- 原始方法来自 Satorras、Hoogeboom 与 Welling 的 [E(n) Equivariant Graph Neural Networks](https://arxiv.org/abs/2102.09844)，正式 ICML 版本见 [PMLR 139](https://proceedings.mlr.press/v139/satorras21a.html)。
- 本项目保留“scalar message 依赖成对距离、坐标沿相对方向更新”的等变核心。
- 原论文允许在给定图上工作；本实现每层都基于演化坐标重建 dynamic kNN。
- 本实现把 relative displacement 归一化为单位方向，使用 `tanh` bounded coefficient 和固定 `coord_scale=0.10`；这不是原论文公式的逐字实现。
- 本实现额外把演化坐标的 radius-squared mean/max 拼入事件分类头，并使用 NEXT 专用 voxel、特征和训练协议。
- 因此应称为 **EGNN-style dynamic-kNN classifier**，而不是原始 EGNN 代码复现。

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
| 模型 | `message_dim` | 128 |
| 模型 | `num_layers` | 5 |
| 模型 | `k` | 16 |
| 模型 | `coord_scale` | 0.10 |
| 模型 | `classifier_dim` | 160 |
| 模型 | `dropout` | 0.10 |
| 训练 | `batch_size` | 12 |
| 训练 | `epochs` | 50 |
| 训练 | `learning_rate` | `3e-4` |
| 训练 | `weight_decay` | `1e-4` |
| 训练 | `gradient_clip_norm` | 1.0 |
| 训练 | `early_stopping_patience` | 12 |
| 训练 | `seed` | 42 |
| 训练 | `use_amp` / `amp_precision` | `true` / `auto` |
| 训练 | `deterministic` | `false` |

完整配置见 [`config.yaml`](config.yaml)，共享默认值和校验见 [`src/next_alt/config.py`](../../../src/next_alt/config.py)。

## 训练机制与命令

- 数据按完整 HDF5 文件进行稳定 hash split，防止同一 source file 泄漏到多个 split。
- `balance_training_classes=true` 时交替产生 signal/background event，并在较小类耗尽时结束。
- 当前 balanced 分支不会使用 YAML 中的 `event_shuffle_buffer_size=512`；实际逐 epoch shuffle 文件顺序。
- loss 为 `BCEWithLogitsLoss`；optimizer 为 AdamW；scheduler 为 50-epoch CosineAnnealingLR。
- best checkpoint 按 validation AUC 选择，last checkpoint 每 epoch 更新；patience 为 12。
- 正式训练仅支持 CUDA，不提供 CPU fallback 或 smoke mode。

原始正式训练入口为：

```bash
cd /home/wenyu/summer
source .venv/bin/activate
python 01_code/architectures/gnn_003_egnn/train_classification.py
```

正式训练产物已经存在。默认 `allow_overwrite: false`，直接重跑上述命令会因为目标文件存在而拒绝启动。新的实验应复制 YAML 并设置新的 `output.checkpoint_dir`、`output.log_dir`、`output.plot_dir`：

```bash
python 01_code/architectures/gnn_003_egnn/train_classification.py \
  /path/to/gnn_003_new_experiment.yaml
```

## 已完成的正式结果

| 指标 | 结果 |
|---|---:|
| best epoch | 7 |
| best validation AUC | 0.889481 |
| early stop epoch | 19 |
| full-test 总排名 | **10 / 10** |
| full-test matched AUC | **0.886558** |
| full-test inclusive AUC | 0.887098 |
| energy-independence mean | 0.977462 |

full-test 覆盖完整 test split 的 1,490 个文件和 115,499 个事件，strict evaluation 为 0 warning、0 error。正式目录不可覆盖；评测重跑应写入新的 `_rerun` 目录，并保持完整 test split：

```bash
cd /home/wenyu/summer
source .venv/bin/activate
energybench next \
  02_models/checkpoints/NEXTALT_gnn_003_egnn_classification_best.pt \
  --device cuda:0 \
  --split test \
  --batch-size 12 \
  --num-workers 0 \
  --model-id gnn_003_egnn \
  --output-dir 04_evaluations/NEXTALT_gnn_003_egnn_test_rerun
```

不要为正式 full-test 重跑传 `--max-files-per-class`。

## 产物

| 类型 | 文件 |
|---|---|
| best checkpoint | [`NEXTALT_gnn_003_egnn_classification_best.pt`](../../../02_models/checkpoints/NEXTALT_gnn_003_egnn_classification_best.pt) |
| last checkpoint | [`NEXTALT_gnn_003_egnn_classification_last.pt`](../../../02_models/checkpoints/NEXTALT_gnn_003_egnn_classification_last.pt) |
| epoch CSV | [`classification_epochs.csv`](../../../03_training_runs/logs/NEXTALT_gnn_003_egnn_classification_epochs.csv) |
| history JSON | [`classification_history.json`](../../../03_training_runs/logs/NEXTALT_gnn_003_egnn_classification_history.json) |
| history plot | [`classification_history.png`](../../../03_training_runs/history_plots/NEXTALT_gnn_003_egnn_classification_history.png) |
| test predictions | [`predictions_test.npz`](../../../04_evaluations/NEXTALT_gnn_003_egnn_test/predictions_test.npz) |
| test results | [`evaluation_test/results.csv`](../../../04_evaluations/NEXTALT_gnn_003_egnn_test/evaluation_test/results.csv) |
| strict metrics | [`evaluation_test/.energybench/metrics.json`](../../../04_evaluations/NEXTALT_gnn_003_egnn_test/evaluation_test/.energybench/metrics.json) |
| ROC | [`energy_matched_roc.png`](../../../04_evaluations/NEXTALT_gnn_003_egnn_test/evaluation_test/energy_matched_roc.png) |
| score-energy 图 | [`score_energy_dependence.png`](../../../04_evaluations/NEXTALT_gnn_003_egnn_test/evaluation_test/score_energy_dependence.png) |
| 十模型排行榜 | [`leaderboard.csv`](../../../04_evaluations/NEXTALT_all_models_comparison/leaderboard.csv) |

## 本地关联文件

| 职责 | 文件 |
|---|---|
| 本模型配置 | [`config.yaml`](config.yaml) |
| 本模型训练入口 | [`train_classification.py`](train_classification.py) |
| EGNN layer 与 classifier | [`src/next_alt/models/point_graph.py`](../../../src/next_alt/models/point_graph.py) |
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

- 五层都用 dense `torch.cdist` 重建 kNN，计算和中间显存约为 `O(5BN²)`；该模型也是这组三个 GNN 中参数量最大的。
- top-k neighbour index 是离散选择，且距离相等时的 tie、有限精度和 GPU kernel 数值差异会使理论等变性在数值层面只能近似成立。
- top-512 截断可能移除低能的长程结构。
- 能量归一化和中心化移除了直接的总能量与绝对位置，但并不能保证 score 与能量完全独立。
- coordinate update 是本项目的 normalized-direction 变体；不能把本模型的结果直接解释为原论文标准 EGNN 的能力上限。
- 该次训练在 epoch 7 达到最好 validation AUC，之后 validation 明显下降并于 epoch 19 early stop；当前配置对优化和正则化较敏感。
- 当前排名只代表一个 split、一个 seed 和一组超参数，而不是完整调参后的方法排序。
