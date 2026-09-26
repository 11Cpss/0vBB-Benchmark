# POINT-002：轻量 PointNet++ 风格层次点云网络

中文 | [English](README_EN.md)

本模型在与 POINT-001 相同的居中 voxel 点集上加入显式局部尺度：先用
farthest-point sampling（FPS）选取中心，再对每个中心的几何近邻编码与池化，
逐级把最多 512 个输入节点压缩为最多 64、再到最多 16 个中心。

## 1. 模型定位

| 项目 | 值 |
|---|---|
| `architecture_id` | `point_002_pointnetpp` |
| checkpoint `model_name` | `PointNetPPClassifier` |
| Python class | `next_alt.models.point_graph.PointNetPPClassifier` |
| registry `input_kind` | `points` |
| 任务 | NEXT `0nubb`（label 1）与 `Bi214`（label 0）二分类 |
| 输出 | 每个事件一个未校准的 signal logit，shape 为 `(B,)`，值越大越倾向 `0nubb` |
| 可训练参数量 | **164,513** |
| checkpoint 格式 | EnergyBench/NEXT format version 3 |

## 2. 原始数据与精确预处理

### 2.1 数据读取与切分

1. 输入来自每个 HDF5 文件的 `/MC/hits/table`。共享读取器校验
   `event_id`、`x/y/z/energy` 与文件目录所声明的类别，并按连续
   `event_id` 组装事件。
2. `0nubb_part_*` 目录映射为 signal/label 1，`Bi_part_*` 目录映射为
   background/label 0。
3. 完整 HDF5 相对路径作为 `group_id`，用 seed 42 做稳定 file-level hash
   切分，train/validation/test 比例为 0.8/0.1/0.1。同一源文件不会跨 split。
4. 正式训练在 train 和 validation split 中分别最多选择每类 100 个文件；
   正式 test 评测不限制文件数，共使用 1,490 个文件、115,499 个事件。

### 2.2 从 hit 到 15 mm voxel 点集

对坐标 $\mathbf r_i$ 和沉积能量 $e_i$：

1. 用完整事件总能量 $E=\sum_i e_i$ 计算能量加权质心
   $\mathbf c=\sum_i e_i\mathbf r_i/E$，先将事件平移到该质心。
2. 用
   $\mathbf q_i=\lfloor(\mathbf r_i-\mathbf c)/(15\,\mathrm{mm})\rfloor$
   合并同 cell 的输入行。
3. 初始 voxel 中心为 $(\mathbf q+0.5)\times15\,\mathrm{mm}$，随后再减去
   量化中心的 voxel-energy 加权质心，以去除 half-bin 统一偏移。
4. 若超过 512 个节点，按 voxel 能量降序选 512 个，cell 坐标字典序稳定破同分。
   节点能量占比仍以完整事件能量为分母，不在截断后重新归一化。
5. 中心坐标除以 1000 mm。每节点附加
   `voxel_energy / complete_event_energy` 与 `log1p(merged_row_count)` 两个特征。

最终输入为：

| 字段 | shape / dtype | 含义 |
|---|---|---|
| `coords` | `(B, N, 3)`, float32 | 居中并除以 1000 mm 的 voxel XYZ |
| `features` | `(B, N, 2)`, float32 | energy fraction 与 `log1p(hit_count)` |
| `mask` | `(B, N)`, bool | 排除 batch padding；每个事件的有效节点数可以不同 |
| model output | `(B,)`, float32 | signal logit |

总事件能量与绝对探测器位置不进入模型；总能量仅作为 `energy_condition`
留在输出表中供匹配评测。

## 3. 逐层架构与维度

共用 `_mlp(a,b,c,p)` 的内部顺序为
`Linear(a,b) → LayerNorm(b) → SiLU → Dropout(p) → Linear(b,c) → SiLU`。

| 阶段 | 运算 | 输入 → 输出 |
|---|---|---|
| 节点输入 | 拼接 XYZ 与两个特征 | `(B,N,5)` |
| input encoder | `_mlp(5,96,96,0.10)` | `(B,N,5) → (B,N,96)` |
| FPS-1 | 确定性 farthest-point sampling | 最多 `N=512 → Q1=64` 个中心 |
| local group-1 | 每个中心在输入 support 中取最多 16 个 Euclidean kNN | `96 + ΔXYZ(3) + distance(1) = 100` 维消息输入 |
| set abstraction-1 | `_mlp(100,128,128,0.10)` + masked neighbor max | `(B,Q1,16,100) → (B,Q1,128)` |
| FPS-2 | 在第一层中心上再次确定性 FPS | `Q1≤64 → Q2≤16` |
| local group-2 | 每个第二层中心在第一层 support 中取最多 16 个 kNN | `128 + 3 + 1 = 132` 维消息输入 |
| set abstraction-2 | `_mlp(132,192,192,0.10)` + masked neighbor max | `(B,Q2,16,132) → (B,Q2,192)` |
| event pooling | 对第二层有效中心做 masked mean 与 max 并拼接 | `(B,Q2,192) → (B,384)` |
| classifier hidden | `Linear + SiLU + Dropout(0.10)` | `384 → 160` |
| classifier output | `Linear` | `160 → 1` |

FPS 是确定性的：第一步选择离当前 support 有效节点算术均值最远的点，之后反复选择
距已选集合最远的点。若事件节点少于 64 或 16，`sampled_mask` 保证补位中心不会参与
后续 pooling。局部分组使用 fixed-`k` kNN，查询中心本身可以作为其 support 邻居之一。

## 4. 与原始 PointNet++ 的差异边界

本实现保留了 PointNet++ 的核心思路——FPS、局部分组、共享局部网络、对称池化和
层次下采样——但它是 classification-only 的轻量 **PointNet++-style** 变体：

- 原论文主要使用 radius/ball query，并提出 multi-scale grouping 适应密度变化；
  本实现每层只有一个固定 `k=16` 的 Euclidean kNN 尺度，没有 radius cutoff 或 MSG；
- 本实现没有 segmentation 所需的 feature-propagation/上采样路径；
- 局部消息显式拼接 relative XYZ 与 Euclidean distance；
- 事件级 head 使用 mean+max 拼接，而不是声称复现论文的特定分类 head；
- 点集来自本项目的 15 mm 能量 voxelization，并有 512 节点上限；
- FPS、kNN 与聚合均为纯 PyTorch，没有论文常见的自定义 CUDA point operators，
  也不依赖 `torch_geometric` 或 DGL。

因此结果只能归因于当前轻量实现，不能作为标准 PointNet++ 官方实现的复现结果。

## 5. 关键配置

| 类别 | 参数 | 值 |
|---|---|---:|
| 表示 | `point_bin_size` | 15.0 mm |
| 表示 | `coordinate_scale` | 1000.0 mm |
| 表示 | `max_points` | 512 |
| 模型 | `feature_dim` | 2 |
| 模型 | `hidden_dim` | 96 |
| 模型 | `stage1_dim` / `stage2_dim` | 128 / 192 |
| 模型 | `stage1_points` / `stage2_points` | 64 / 16 |
| 模型 | `k` | 16 |
| 模型 | `classifier_dim` | 160 |
| 模型 | `dropout` | 0.10 |
| 数据 | `max_files_per_class` | 100 |
| 数据 | `split_seed` / `split_fractions` | 42 / `[0.8, 0.1, 0.1]` |
| 数据 | `balance_training_classes` | true |
| 数据 | `event_shuffle_buffer_size` | 512（见下方实际机制说明） |
| 数据 | `num_workers` | 0 |
| 训练 | `batch_size` | 16 |
| 训练 | `epochs` | 50 |
| 训练 | `learning_rate` | 5e-4 |
| 训练 | `weight_decay` | 1e-4 |
| 训练 | `gradient_clip_norm` | 1.0 |
| 训练 | `early_stopping_patience` / `min_delta` | 12 / 0.0 |
| 训练 | `seed` / `deterministic` | 42 / false |
| 训练 | AMP | `auto`；本次正式 checkpoint 记录为 bfloat16 |

## 6. 训练机制与命令

- loss：`BCEWithLogitsLoss`；optimizer：AdamW；
- scheduler：`CosineAnnealingLR(T_max=50)`，每个 epoch 更新；
- gradient norm clip 为 1.0；
- 每个 epoch 打乱源文件，并在 balanced 模式下交替读取 signal/background 事件，
  任一类别耗尽后结束该 epoch；
- 当前 `balance_training_classes=true` 路径不调用 bounded event shuffle buffer，
  因而不应把配置中的 `event_shuffle_buffer_size=512` 写成实际执行的 event shuffle；
- best 按 validation AUC 选择，patience 为 12；
- 正式训练使用 `cuda:0` 和 AMP，没有 CPU fallback 或 smoke branch。

标准入口：

```bash
cd /home/wenyu/summer
source .venv/bin/activate
python 01_code/architectures/point_002_pointnetpp/train_classification.py
```

本模型的 checkpoint 与全部 history 产物已经存在。默认
`allow_overwrite=false`，直接重跑上述命令会拒绝覆盖。独立重训应复制 YAML，
为 checkpoint/log/plot 指定新的 output 目录，再将副本路径作为训练脚本的位置参数。

## 7. 已完成训练与正式评测结果

best checkpoint 来自 epoch 48；在 16,786 个 validation 事件上的 AUC 为
**0.953117**，mean representation coverage 为 1.0。

best checkpoint 在完整 test split 上的 strict 评测结果为：

| 指标 | 结果 |
|---|---:|
| 10 个替代架构中的 classification rank | **5 / 10** |
| Energy-matched AUC | **0.953342** |
| Inclusive AUC | **0.953552** |
| Energy-independence score（mean） | **0.977201** |
| 测试文件 / 事件数 | 1,490 / 115,499 |

正式评测为 0 warning、0 error。该 checkpoint 只输出分类 logit，所以 energy
regression 为 `not_applicable` 属于正常状态。

### 在新目录重新执行完整 test 评测

```bash
cd /home/wenyu/summer
source .venv/bin/activate
energybench next \
  02_models/checkpoints/NEXTALT_point_002_pointnetpp_classification_best.pt \
  --split test \
  --device cuda:0 \
  --output-dir 04_evaluations/NEXTALT_point_002_pointnetpp_test_rerun
```

`_rerun` 是新的输出目录；不要加入 `--max-files-per-class`，否则结果不再对应完整 test。

## 8. 已有产物

| 产物 | 路径 |
|---|---|
| best checkpoint | [`NEXTALT_point_002_pointnetpp_classification_best.pt`](../../../02_models/checkpoints/NEXTALT_point_002_pointnetpp_classification_best.pt) |
| last checkpoint | [`NEXTALT_point_002_pointnetpp_classification_last.pt`](../../../02_models/checkpoints/NEXTALT_point_002_pointnetpp_classification_last.pt) |
| epoch metrics | [`..._epochs.csv`](../../../03_training_runs/logs/NEXTALT_point_002_pointnetpp_classification_epochs.csv) |
| 完整 history | [`..._history.json`](../../../03_training_runs/logs/NEXTALT_point_002_pointnetpp_classification_history.json) |
| 训练曲线 | [`..._history.png`](../../../03_training_runs/history_plots/NEXTALT_point_002_pointnetpp_classification_history.png) |
| 正式 test predictions | [`predictions_test.npz`](../../../04_evaluations/NEXTALT_point_002_pointnetpp_test/predictions_test.npz) |
| 正式 test 汇总 | [`results.csv`](../../../04_evaluations/NEXTALT_point_002_pointnetpp_test/evaluation_test/results.csv) |
| 完整评测指标 | [`metrics.json`](../../../04_evaluations/NEXTALT_point_002_pointnetpp_test/evaluation_test/.energybench/metrics.json) |
| matched ROC 图 | [`energy_matched_roc.png`](../../../04_evaluations/NEXTALT_point_002_pointnetpp_test/evaluation_test/energy_matched_roc.png) |
| score-energy 图 | [`score_energy_dependence.png`](../../../04_evaluations/NEXTALT_point_002_pointnetpp_test/evaluation_test/score_energy_dependence.png) |
| 十模型排行榜 | [`leaderboard.csv`](../../../04_evaluations/NEXTALT_all_models_comparison/leaderboard.csv) |

## 9. 本地关联文件

| 文件 | 作用 |
|---|---|
| [`config.yaml`](./config.yaml) | 本模型正式训练配置 |
| [`train_classification.py`](./train_classification.py) | 固定 architecture ID 的入口 |
| [`src/next_alt/models/point_graph.py`](../../../src/next_alt/models/point_graph.py) | FPS、set abstraction、kNN 与 `PointNetPPClassifier` 权威实现 |
| [`src/next_alt/data.py`](../../../src/next_alt/data.py) | voxelization、节点截断、表示转换和 padding |
| [`src/next_cnn/data.py`](../../../src/next_cnn/data.py) | HDF5 读取、标签和 file-level split |
| [`src/next_alt/registry.py`](../../../src/next_alt/registry.py) | architecture/class/input-kind 注册 |
| [`src/next_alt/training.py`](../../../src/next_alt/training.py) | 共享训练循环、优化器和 best selection |
| [`src/next_alt/checkpoint.py`](../../../src/next_alt/checkpoint.py) | format-v3 checkpoint contract |
| [`src/next_alt/adapter.py`](../../../src/next_alt/adapter.py) | 正式推理与 EnergyBench 接口 |
| [`docs/ALTERNATIVE_ARCHITECTURES.md`](../../../docs/ALTERNATIVE_ARCHITECTURES.md) | 替代架构总览 |
| [`docs/ALTERNATIVE_EVALUATION_RESULTS.md`](../../../docs/ALTERNATIVE_EVALUATION_RESULTS.md) | 完整 test 评测合同和排行榜 |

本目录没有独立 `model.py`；训练与推理都从
`src/next_alt/models/point_graph.py` 构建同一个 class。

## 10. 原始方法参考

- Charles R. Qi et al.,
  [*PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space*](https://arxiv.org/abs/1706.02413),
  NeurIPS 2017。
- Charles R. Qi et al.,
  [*PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation*](https://arxiv.org/abs/1612.00593),
  CVPR 2017。该文是共享逐点网络与对称池化的前置方法。

## 11. 局限

- fixed-`k` kNN 不会显式约束物理半径；节点密度变化会改变实际邻域尺度；
- 两层 FPS 与 dense `torch.cdist` 的时间/显存成本高于简单 Deep Sets，且没有使用
  高性能 point-ops CUDA 扩展；
- 512 节点截断可能丢失低能拓扑细节；
- 输入虽然事件居中，但直接使用 XYZ 和 relative XYZ，不具备旋转不变/等变保证；
- 只有 single-scale grouping 和 classification head，不能代表完整 PointNet++ 能力；
- 当前排名来自一个 seed、一个模型规模与固定预算，没有多 seed 不确定性。
