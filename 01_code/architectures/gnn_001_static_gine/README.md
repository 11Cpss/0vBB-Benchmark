# GNN-001：静态几何图 GINE 风格分类器

中文 | [English](README_EN.md)

本模型先在事件的居中 voxel 坐标上构建一次几何 k-nearest-neighbor 图，随后在
五个 residual edge-aware GIN/GINE 风格层中重复使用同一邻接关系。它测试的核心问题
是：固定的三维局部连通性与 edge geometry，能否比不构图的点集模型更好地识别
`0nubb` 与 `Bi214` 拓扑。

## 1. 模型定位

| 项目 | 值 |
|---|---|
| `architecture_id` | `gnn_001_static_gine` |
| checkpoint `model_name` | `StaticGINEClassifier` |
| Python class | `next_alt.models.point_graph.StaticGINEClassifier` |
| registry `input_kind` | `graph` |
| 任务 | NEXT `0nubb`（label 1）与 `Bi214`（label 0）二分类 |
| 输出 | 每个事件一个未校准 signal logit，shape 为 `(B,)`，值越大越倾向 `0nubb` |
| 可训练参数量 | **479,942** |
| checkpoint 格式 | EnergyBench/NEXT format version 3 |

## 2. 原始数据与精确预处理

### 2.1 数据读取、标签与 split

1. 每个事件从 HDF5 `/MC/hits/table` 的连续 `event_id` 行读取，输入列为
   `x/y/z/energy`。共享读取器拒绝空事件、非有限值、负能量、类别不一致和非连续重复
   event ID。
2. `0nubb_part_*` 映射为 signal/label 1；`Bi_part_*` 映射为
   background/label 0。
3. 用完整 HDF5 相对路径作为 group，seed 42 的稳定 hash 做 file-level
   train/validation/test = 0.8/0.1/0.1 切分。同一文件不会泄漏到不同 split。
4. 正式训练的 train 与 validation split 各自最多选每类 100 个文件；正式 test
   评测不设文件上限，共 1,490 个文件、115,499 个事件。

### 2.2 15 mm voxel 节点

给定 hit 坐标 $\mathbf r_i$ 与沉积能量 $e_i$：

1. 计算完整事件能量 $E=\sum_i e_i$ 和能量加权质心
   $\mathbf c=\sum_i e_i\mathbf r_i/E$，先做事件居中。
2. 用
   $\mathbf q_i=\lfloor(\mathbf r_i-\mathbf c)/(15\,\mathrm{mm})\rfloor$
   合并同 cell 的行。
3. 以 $(\mathbf q+0.5)\times15\,\mathrm{mm}$ 为 voxel 中心，再减去量化
   voxel 中心的能量加权质心，消除 half-bin 公共偏移。
4. 节点数超过 512 时，按 voxel energy 降序保留 512 个，cell 坐标字典序破同分。
   截断后 `energy_fraction` 仍除以完整事件能量，不重新归一化。
5. 坐标除以 1000 mm。节点特征是
   `[voxel_energy / complete_event_energy, log1p(merged_row_count)]`。

### 2.3 batch 张量与静态图

| 字段 | shape / dtype | 用途 |
|---|---|---|
| `coords` | `(B,N,3)`, float32 | 居中、缩放的三维坐标；同时用于 node encoder 和 kNN |
| `features` | `(B,N,2)`, float32 | energy fraction 与 log hit count |
| `mask` | `(B,N)`, bool | 排除 padded 节点及其边 |
| graph | 隐式 `(B,N,K)` neighbor index | 每个有效节点最多 12 个其它有效节点 |
| output | `(B,)` | signal logit |

图在第一次 message passing 前用 Euclidean distance 构建，排除 self。实现对每个
query 独立选择最近邻，因此是 **query-to-neighbor 的有向 kNN 关系**，没有 mutual-edge
闭包或显式无向对称化。对少于 13 个节点的事件，每个节点只有至多 `N-1` 个有效邻居；
单节点事件的 neighbor aggregate 为空。邻接 index 在 `no_grad` 中生成，并在全部五层复用。

总事件能量和绝对探测器位置不进入模型；总能量仅作为评测用 `energy_condition` 导出。

## 3. 逐层架构与维度

共享 `_mlp(a,b,c,p)` 为
`Linear(a,b) → LayerNorm(b) → SiLU → Dropout(p) → Linear(b,c) → SiLU`。

### 3.1 输入编码与图构建

| 阶段 | 运算 | 输入 → 输出 |
|---|---|---|
| 节点输入 | `concat(coords, features)` | `(B,N,3)+(B,N,2) → (B,N,5)` |
| node encoder | `_mlp(5,128,128,0.10)` | `(B,N,5) → (B,N,128)` |
| static graph | masked Euclidean kNN，`k=12`，排除 self | `(B,N,3) → indices/mask (B,N,K)` |

### 3.2 五个 residual GINE 风格层

对中心节点 $i$ 与其邻居 $j$，令
$\Delta\mathbf r_{ij}=\mathbf r_j-\mathbf r_i$。每一层独立拥有以下参数：

$$
\begin{aligned}
\mathbf e_{ij} &= \mathrm{MLP}_e(
  \Delta\mathbf r_{ij}\Vert\lVert\Delta\mathbf r_{ij}\rVert_2),\\
\mathbf m_{ij} &= \mathrm{MLP}_m(\mathbf h_j+\mathbf e_{ij}),\\
\mathbf a_i &= \sum_{j\in\mathcal N(i)}\mathbf m_{ij},\\
\mathbf u_i &= \mathrm{MLP}_u((1+\epsilon)\mathbf h_i+\mathbf a_i),\\
\mathbf h'_i &= \mathrm{LayerNorm}(
  \mathbf h_i+\mathrm{Dropout}(\mathbf u_i)).
\end{aligned}
$$

| 子模块 | 维度 |
|---|---|
| edge input | relative XYZ 3 + Euclidean distance 1 = 4 |
| edge encoder | `_mlp(4,128,128,0.10)` |
| message MLP | `_mlp(128,128,128,0.10)` |
| neighbor aggregate | masked sum，`K≤12` |
| update MLP | `_mlp(128,128,128,0.10)` |
| epsilon | 每层一个可学习标量，初始化为 0 |
| residual normalization | node residual + dropout update + LayerNorm(128) |
| 层数 | 5；所有层复用初始 static kNN graph |

### 3.3 事件读出

| 阶段 | 运算 | 输入 → 输出 |
|---|---|---|
| graph pooling | masked node mean 与 max 拼接 | `(B,N,128) → (B,256)` |
| classifier hidden | `Linear + SiLU + Dropout(0.10)` | `256 → 160` |
| classifier output | `Linear` | `160 → 1` |

## 4. 与 GIN/GINE 原始方法的差异边界

本模型从 GIN 的 learnable-$\epsilon$ sum aggregation 与带 edge feature 的 GINE
消息形式得到启发，但属于本项目的 custom residual variant：

- edge feature 不是数据中预先给定的 bond type，而是由 relative XYZ 与 distance
  经过可学习 edge encoder 得到；
- canonical GINE 常写成对 $h_j+e_{ij}$ 做非线性后求和；本实现额外使用完整
  message MLP、update MLP、外层 residual 与 LayerNorm；
- 图由当前事件几何坐标的 fixed kNN 构造，而不是原论文中的离散分子/一般图边；
- 邻接关系五层不变，且没有显式对称化；
- graph-level readout 使用 masked mean+max，而不是只声称标准 sum readout；
- 实现为纯 PyTorch dense masked kNN，没有调用 PyG `GINEConv`、DGL 或
  `torch_scatter`。

因此应称为“static geometric GINE-style”或“edge-aware GIN variant”，不能视为
PyG GINEConv 或论文网络的逐行复现。

## 5. 关键配置

| 类别 | 参数 | 值 |
|---|---|---:|
| 表示 | `point_bin_size` | 15.0 mm |
| 表示 | `coordinate_scale` | 1000.0 mm |
| 表示 | `max_points` | 512 |
| 模型 | `feature_dim` | 2 |
| 模型 | `hidden_dim` | 128 |
| 模型 | `num_layers` | 5 |
| 模型 | `k` | 12 |
| 模型 | `classifier_dim` | 160 |
| 模型 | `dropout` | 0.10 |
| 模型 | `train_eps` | true |
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
| 训练 | AMP | `auto`；正式 checkpoint 记录为 bfloat16 |

## 6. 训练机制与命令

- `BCEWithLogitsLoss` + AdamW；
- `CosineAnnealingLR(T_max=50)`，每 epoch step；
- gradient norm clip 1.0；
- 每 epoch 打乱文件，balanced 模式按 signal/background 事件交替读取；
- 当前 balanced 分支不经过 bounded event shuffle buffer，所以配置中的
  `event_shuffle_buffer_size=512` 不等于本次实际执行了 buffer-level event shuffle；
- best checkpoint 按 validation AUC 选择，early-stopping patience 12；
- GPU-only，使用 `cuda:0` 与 AMP，无 CPU fallback、无 smoke branch。

标准训练入口：

```bash
cd /home/wenyu/summer
source .venv/bin/activate
python 01_code/architectures/gnn_001_static_gine/train_classification.py
```

正式 best/last checkpoint、history 与 plot 已存在。runner 默认
`allow_overwrite=false`，直接再次执行会在开始训练前拒绝覆盖。独立重训应复制
`config.yaml`，在副本中设置新的 checkpoint/log/plot output 路径，然后把副本路径
作为训练脚本的唯一位置参数。

## 7. 已完成训练与正式评测结果

best checkpoint 位于 epoch 42；该 epoch 在 16,786 个 validation 事件上的 AUC 为
**0.981898**，mean representation coverage 为 1.0。

完整 test split 的 strict 正式结果：

| 指标 | 结果 |
|---|---:|
| 10 个替代架构中的 classification rank | **1 / 10** |
| Energy-matched AUC | **0.981035** |
| Inclusive AUC | **0.981224** |
| Energy-independence score（mean） | **0.977111** |
| 测试文件 / 事件数 | 1,490 / 115,499 |

该结果为 0 warning、0 error，并且是十模型比较中的最高 matched AUC。模型只输出
分类 logit，energy regression 的 `not_applicable` 状态属于预期。

### 在新目录重新执行完整 test 评测

```bash
cd /home/wenyu/summer
source .venv/bin/activate
energybench next \
  02_models/checkpoints/NEXTALT_gnn_001_static_gine_classification_best.pt \
  --split test \
  --device cuda:0 \
  --output-dir 04_evaluations/NEXTALT_gnn_001_static_gine_test_rerun
```

这里使用新的 `_rerun` 目录，避免覆盖已有正式结果。不要指定
`--max-files-per-class`，否则不是 full-test evaluation。

## 8. 已有产物

| 产物 | 路径 |
|---|---|
| best checkpoint | [`NEXTALT_gnn_001_static_gine_classification_best.pt`](../../../02_models/checkpoints/NEXTALT_gnn_001_static_gine_classification_best.pt) |
| last checkpoint | [`NEXTALT_gnn_001_static_gine_classification_last.pt`](../../../02_models/checkpoints/NEXTALT_gnn_001_static_gine_classification_last.pt) |
| epoch metrics | [`..._epochs.csv`](../../../03_training_runs/logs/NEXTALT_gnn_001_static_gine_classification_epochs.csv) |
| 完整 history | [`..._history.json`](../../../03_training_runs/logs/NEXTALT_gnn_001_static_gine_classification_history.json) |
| 训练曲线 | [`..._history.png`](../../../03_training_runs/history_plots/NEXTALT_gnn_001_static_gine_classification_history.png) |
| 正式 test predictions | [`predictions_test.npz`](../../../04_evaluations/NEXTALT_gnn_001_static_gine_test/predictions_test.npz) |
| 正式 test 汇总 | [`results.csv`](../../../04_evaluations/NEXTALT_gnn_001_static_gine_test/evaluation_test/results.csv) |
| 完整评测指标 | [`metrics.json`](../../../04_evaluations/NEXTALT_gnn_001_static_gine_test/evaluation_test/.energybench/metrics.json) |
| matched ROC 图 | [`energy_matched_roc.png`](../../../04_evaluations/NEXTALT_gnn_001_static_gine_test/evaluation_test/energy_matched_roc.png) |
| score-energy 图 | [`score_energy_dependence.png`](../../../04_evaluations/NEXTALT_gnn_001_static_gine_test/evaluation_test/score_energy_dependence.png) |
| 十模型排行榜 | [`leaderboard.csv`](../../../04_evaluations/NEXTALT_all_models_comparison/leaderboard.csv) |

## 9. 本地关联文件

| 文件 | 作用 |
|---|---|
| [`config.yaml`](./config.yaml) | 本模型正式表示、架构和训练参数 |
| [`train_classification.py`](./train_classification.py) | 固定 architecture ID 的训练入口 |
| [`src/next_alt/models/point_graph.py`](../../../src/next_alt/models/point_graph.py) | masked kNN、GINE layer 与 `StaticGINEClassifier` 权威实现 |
| [`src/next_alt/data.py`](../../../src/next_alt/data.py) | voxelization、截断、点/图表示与 padded collate |
| [`src/next_cnn/data.py`](../../../src/next_cnn/data.py) | HDF5 读取、标签和 file-level split |
| [`src/next_alt/registry.py`](../../../src/next_alt/registry.py) | architecture/class/input-kind 注册 |
| [`src/next_alt/training.py`](../../../src/next_alt/training.py) | shared loss、optimizer、scheduler 与 checkpoint selection |
| [`src/next_alt/checkpoint.py`](../../../src/next_alt/checkpoint.py) | format-v3 checkpoint schema 与 provenance |
| [`src/next_alt/adapter.py`](../../../src/next_alt/adapter.py) | checkpoint 重建、数据防泄漏检查与正式推理 |
| [`docs/ALTERNATIVE_ARCHITECTURES.md`](../../../docs/ALTERNATIVE_ARCHITECTURES.md) | 所有替代模型的表示与方法总览 |
| [`docs/ALTERNATIVE_EVALUATION_RESULTS.md`](../../../docs/ALTERNATIVE_EVALUATION_RESULTS.md) | 正式比较合同与十模型结果 |

本目录没有单独 `model.py`；训练和推理都使用
`src/next_alt/models/point_graph.py` 中同一个模型定义。

## 10. 原始方法参考

- Keyulu Xu et al.,
  [*How Powerful are Graph Neural Networks?*](https://openreview.net/forum?id=ryGs6iA5Km),
  ICLR 2019。该文给出 GIN 与 learnable-$\epsilon$ 聚合的理论基础。
- Weihua Hu et al.,
  [*Strategies for Pre-training Graph Neural Networks*](https://openreview.net/forum?id=HJlWWJSFDH),
  ICLR 2020。该工作使用带 edge feature 的 GIN 更新，是 GINE 类实现的主要方法依据。

## 11. 局限

- dense `torch.cdist` kNN 的距离计算为 $O(BN^2)$，512 节点上限同时是信息与
  计算折中；
- directed kNN 没有 mutual/symmetric closure，固定 `k` 对不同节点密度对应不同
  物理半径；
- 五层始终复用初始几何图，不能像动态图网络那样依据学习到的 node embedding
  更新邻域；
- 直接使用 centered XYZ、relative XYZ，因此没有旋转不变或 E(3) 等变保证；
- 截断会丢弃低能节点，绝对总能量也不作为输入；
- rank 1 是当前数据、训练预算、模型规模和单 seed 下的结果，仍需多 seed、系统误差
  与计算成本研究才能支持更一般的结论。
