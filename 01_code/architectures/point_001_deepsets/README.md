# POINT-001：Deep Sets 置换不变集合基线

中文 | [English](README_EN.md)

本模型把一个 NEXT 事件表示为不定长 voxel 集合。所有节点使用同一个 MLP
独立编码，再用对节点排列不敏感的 mean/max pooling 形成事件表示。它不构图、
不做邻域消息传递，因而是点云与图模型的低成本参照基线。

## 1. 模型定位

| 项目 | 值 |
|---|---|
| `architecture_id` | `point_001_deepsets` |
| checkpoint `model_name` | `DeepSetsClassifier` |
| Python class | `next_alt.models.point_graph.DeepSetsClassifier` |
| registry `input_kind` | `points` |
| 任务 | NEXT `0nubb`（label 1）与 `Bi214`（label 0）二分类 |
| 输出 | 每个事件一个未校准的 signal logit，shape 为 `(B,)`，值越大越倾向 `0nubb` |
| 可训练参数量 | **75,585** |
| checkpoint 格式 | EnergyBench/NEXT format version 3 |

## 2. 原始数据与精确预处理

### 2.1 数据读取与切分

1. 输入来自每个 HDF5 文件的 `/MC/hits/table`。共享读取器校验
   `event_id`、`x/y/z/energy` 与文件目录所声明的类别，并按连续
   `event_id` 组装事件。
2. `0nubb_part_*` 目录映射为 signal/label 1，`Bi_part_*` 目录映射为
   background/label 0。
3. split 不是 event-level 随机切分。完整 HDF5 相对路径作为 `group_id`，
   用 seed 42 做稳定的 file-level hash 切分，比例为 train/validation/test
   = 0.8/0.1/0.1，避免同一源文件跨 split。
4. 正式训练在 train 和 validation split 中分别最多选择每类 100 个文件；
   已完成的正式 test 评测没有文件上限，共使用 1,490 个文件、115,499 个事件。

### 2.2 从 hit 到 15 mm voxel 点集

对一个事件的坐标 $\mathbf r_i$ 和非负沉积能量 $e_i$：

1. 计算完整事件总能量 $E=\sum_i e_i$ 与能量加权质心
   $\mathbf c=\sum_i e_i\mathbf r_i/E$，并将所有坐标平移为
   $\mathbf r_i-\mathbf c$。
2. 用
   $\mathbf q_i=\lfloor(\mathbf r_i-\mathbf c)/(15\,\mathrm{mm})\rfloor$
   分箱；落入同一个 cell 的原始行合并为一个节点。
3. 一个 voxel 的初始中心为
   $(\mathbf q+0.5)\times15\,\mathrm{mm}$。随后再减去这些量化中心的
   voxel-energy 加权质心，去掉统一的 half-bin 量化偏移，同时保持节点间相对距离。
4. 若节点数超过 512，按 voxel 沉积能量从高到低保留 512 个；同能量时用
   cell 坐标字典序稳定破同分。被截断后能量特征仍除以完整事件的 $E$，
   不对保留节点重新归一化，因此 `point_coverage` 可以反映保留能量比例。
5. voxel 坐标再除以 1000 mm，作为神经网络的数值输入。

最终 batch 字段为：

| 字段 | shape / dtype | 含义 |
|---|---|---|
| `coords` | `(B, N, 3)`, float32 | 二次居中后的 voxel XYZ，单位数值为坐标/1000 mm |
| `features[..., 0]` | `(B, N)`, float32 | `voxel_energy / complete_event_energy` |
| `features[..., 1]` | `(B, N)`, float32 | `log1p(number_of_merged_rows)` |
| `mask` | `(B, N)`, bool | 有效节点为 true；batch 内只 pad 到该 batch 的最大节点数 |
| model output | `(B,)`, float32 | signal logit；训练损失内部再应用 logits 版本 BCE |

总事件能量和绝对探测器位置不直接送入模型。总能量只作为预测表中的
`energy_condition`，供 EnergyBench 做 energy-matched evaluation。

## 3. 逐层架构与维度

每个有效节点先把三维坐标和两个节点特征拼成 5 维向量。

| 阶段 | 运算 | 输入 → 输出 |
|---|---|---|
| 节点输入 | `concat(coords, features)` | `(B,N,3) + (B,N,2) → (B,N,5)` |
| shared node encoder 1 | `Linear + LayerNorm + SiLU` | `5 → 128` |
| shared node encoder 2 | `Linear + LayerNorm + SiLU` | `128 → 192` |
| padding 处理 | 用 `mask` 将无效节点清零 | `(B,N,192) → (B,N,192)` |
| event pooling | masked mean 与 masked max 拼接 | `(B,N,192) → (B,384)` |
| classifier hidden | `Linear + SiLU + Dropout(0.10)` | `384 → 128` |
| classifier output | `Linear` | `128 → 1` |

可以把前向过程概括为

$$
\mathrm{logit}=\rho\!\left(
  \operatorname{mean}_{i\in V}\phi(\mathbf x_i)
  \;\Vert\;
  \operatorname{max}_{i\in V}\phi(\mathbf x_i)
\right).
$$

shared encoder 与对称池化使输出不依赖节点排列；不同节点在 pooling 前没有信息交换。

## 4. 与原始 Deep Sets 方法的边界

本实现借鉴 Deep Sets 的“共享逐元素映射 + 对称聚合 + 集合级映射”原则，
但不是原论文实验网络的逐层复现：

- 聚合器使用 **masked mean 与 masked max 的拼接**，不是只用 sum pooling；
- 元素输入是本项目定义的居中 voxel 坐标、能量占比和合并 hit 数；
- 加入 LayerNorm、SiLU 和分类头 dropout；
- 每个事件最多保留 512 个高能 voxel；
- 这是二分类专用实现，没有原论文中的其它 set-learning 任务或生成模块；
- 代码为纯 PyTorch，不依赖 `torch_geometric`、DGL 或 compiled scatter 扩展。

因此文档和结果应称为“Deep Sets baseline/变体”，不应声称逐字复现论文模型。

## 5. 关键配置

以下值来自本目录的 `config.yaml`；未显式写出的 output 默认值由共享配置模块补全。

| 类别 | 参数 | 值 |
|---|---|---:|
| 表示 | `point_bin_size` | 15.0 mm |
| 表示 | `coordinate_scale` | 1000.0 mm |
| 表示 | `max_points` | 512 |
| 模型 | `feature_dim` | 2 |
| 模型 | `hidden_dim` | 128 |
| 模型 | `embedding_dim` | 192 |
| 模型 | `classifier_dim` | 128 |
| 模型 | `dropout` | 0.10 |
| 数据 | `max_files_per_class` | 100 |
| 数据 | `split_seed` / `split_fractions` | 42 / `[0.8, 0.1, 0.1]` |
| 数据 | `balance_training_classes` | true |
| 数据 | `event_shuffle_buffer_size` | 512（见下方实际机制说明） |
| 数据 | `num_workers` | 0 |
| 训练 | `batch_size` | 64 |
| 训练 | `epochs` | 50 |
| 训练 | `learning_rate` | 1e-3 |
| 训练 | `weight_decay` | 1e-4 |
| 训练 | `gradient_clip_norm` | 1.0 |
| 训练 | `early_stopping_patience` / `min_delta` | 12 / 0.0 |
| 训练 | `seed` / `deterministic` | 42 / false |
| 训练 | AMP | `auto`；本次正式 checkpoint 记录为 bfloat16 |

## 6. 训练机制与命令

- loss：`BCEWithLogitsLoss`；
- optimizer：AdamW；
- scheduler：`CosineAnnealingLR(T_max=50)`，每个 epoch 更新一次；
- 每次参数更新前将梯度全局 norm clip 到 1.0；
- 每个 epoch 打乱源文件；`balance_training_classes=true` 时按 signal、background
  事件交替读取，任一类别耗尽即结束该 epoch；
- 当前 balanced-class 分支不经过 bounded event shuffle buffer，因此 YAML 中
  `event_shuffle_buffer_size=512` 是已记录配置值，但不应表述为本次训练实际执行了
  buffer-level event shuffle；
- best checkpoint 按 validation AUC 选择，patience 12 控制提前停止；
- 训练要求 `cuda:0`，没有 CPU fallback，也没有 smoke-training 分支。

从仓库根目录启动的标准命令是：

```bash
cd /home/wenyu/summer
source .venv/bin/activate
python 01_code/architectures/point_001_deepsets/train_classification.py
```

当前 best/last checkpoint、history CSV/JSON 和训练曲线都已经存在。共享 runner
默认 `allow_overwrite=false`，所以直接再次运行上述命令会在训练开始前拒绝覆盖已有产物。
若要做独立重训，请复制 YAML，并在副本中为 checkpoint、log 和 plot 配置新的 output
目录，再把该 YAML 路径作为唯一位置参数传给训练脚本。

## 7. 已完成训练与正式评测结果

best checkpoint 对应 epoch 45；该 epoch 在 16,786 个 validation 事件上的 AUC 为
**0.923009**，mean representation coverage 为 1.0。

正式结果来自 best checkpoint、完整 test split、strict manifest，且没有
warning/error：

| 指标 | 结果 |
|---|---:|
| 10 个替代架构中的 classification rank | **8 / 10** |
| Energy-matched AUC | **0.920475** |
| Inclusive AUC | **0.920727** |
| Energy-independence score（mean） | **0.974339** |
| 测试文件 / 事件数 | 1,490 / 115,499 |

该 checkpoint 只做分类，因此 energy regression 显示 `not_applicable` 是预期行为，
不是评测失败。

### 在新目录重新执行完整 test 评测

下面故意使用尚未占用的 `_rerun` 输出目录，不会覆盖正式结果。不要添加
`--max-files-per-class`，否则不再是完整 test 评测。

```bash
cd /home/wenyu/summer
source .venv/bin/activate
energybench next \
  02_models/checkpoints/NEXTALT_point_001_deepsets_classification_best.pt \
  --split test \
  --device cuda:0 \
  --output-dir 04_evaluations/NEXTALT_point_001_deepsets_test_rerun
```

## 8. 已有产物

| 产物 | 路径 |
|---|---|
| best checkpoint | [`NEXTALT_point_001_deepsets_classification_best.pt`](../../../02_models/checkpoints/NEXTALT_point_001_deepsets_classification_best.pt) |
| last checkpoint | [`NEXTALT_point_001_deepsets_classification_last.pt`](../../../02_models/checkpoints/NEXTALT_point_001_deepsets_classification_last.pt) |
| epoch metrics | [`..._epochs.csv`](../../../03_training_runs/logs/NEXTALT_point_001_deepsets_classification_epochs.csv) |
| 完整 history | [`..._history.json`](../../../03_training_runs/logs/NEXTALT_point_001_deepsets_classification_history.json) |
| 训练曲线 | [`..._history.png`](../../../03_training_runs/history_plots/NEXTALT_point_001_deepsets_classification_history.png) |
| 正式 test predictions | [`predictions_test.npz`](../../../04_evaluations/NEXTALT_point_001_deepsets_test/predictions_test.npz) |
| 正式 test 汇总 | [`results.csv`](../../../04_evaluations/NEXTALT_point_001_deepsets_test/evaluation_test/results.csv) |
| 完整评测指标 | [`metrics.json`](../../../04_evaluations/NEXTALT_point_001_deepsets_test/evaluation_test/.energybench/metrics.json) |
| matched ROC 图 | [`energy_matched_roc.png`](../../../04_evaluations/NEXTALT_point_001_deepsets_test/evaluation_test/energy_matched_roc.png) |
| score-energy 图 | [`score_energy_dependence.png`](../../../04_evaluations/NEXTALT_point_001_deepsets_test/evaluation_test/score_energy_dependence.png) |
| 十模型排行榜 | [`leaderboard.csv`](../../../04_evaluations/NEXTALT_all_models_comparison/leaderboard.csv) |

## 9. 本地关联文件

| 文件 | 作用 |
|---|---|
| [`config.yaml`](./config.yaml) | 本模型正式训练超参数 |
| [`train_classification.py`](./train_classification.py) | 固定 architecture ID 的训练入口 |
| [`src/next_alt/models/point_graph.py`](../../../src/next_alt/models/point_graph.py) | `DeepSetsClassifier`、masked pooling 与点输入检查的权威实现 |
| [`src/next_alt/data.py`](../../../src/next_alt/data.py) | 15 mm voxelization、截断、表示转换与 padded collate |
| [`src/next_cnn/data.py`](../../../src/next_cnn/data.py) | HDF5 schema、标签、file-level split 与事件读取 |
| [`src/next_alt/registry.py`](../../../src/next_alt/registry.py) | architecture ID、class 和 `input_kind` 注册 |
| [`src/next_alt/training.py`](../../../src/next_alt/training.py) | loss、optimizer、scheduler、AMP、best selection 与训练循环 |
| [`src/next_alt/checkpoint.py`](../../../src/next_alt/checkpoint.py) | format-v3 checkpoint contract |
| [`src/next_alt/adapter.py`](../../../src/next_alt/adapter.py) | checkpoint 驱动的 EnergyBench 推理适配器 |
| [`docs/ALTERNATIVE_ARCHITECTURES.md`](../../../docs/ALTERNATIVE_ARCHITECTURES.md) | 十个替代架构的总体说明 |
| [`docs/ALTERNATIVE_EVALUATION_RESULTS.md`](../../../docs/ALTERNATIVE_EVALUATION_RESULTS.md) | 正式统一评测合同与排行榜 |

本目录没有单独的 `model.py`；训练和推理共同使用上表中的
`src/next_alt/models/point_graph.py`，避免两条路径出现架构漂移。

## 10. 原始方法参考

- Manzil Zaheer et al., [*Deep Sets*](https://arxiv.org/abs/1703.06114),
  NeurIPS 2017。

该引用说明方法来源；本项目的具体输入、mean+max 聚合和分类头以本仓库实现为准。

## 11. 局限

- 独立节点编码无法显式学习 voxel 邻接、局部曲率、分叉或 track 连通性；
- 高于 512 个 voxel 的事件会丢弃低能节点，尽管 coverage 会被记录；
- 能量占比输入去除了绝对总能量，可能丢失有用能量信息；
- 坐标虽经居中，但网络直接读取 XYZ，因此没有旋转不变或 E(3) 等变保证；
- test 排名只适用于当前数据版本、预处理、训练预算和单个 seed，不能视为模型家族的
  普遍上限，也没有提供多 seed 方差。
