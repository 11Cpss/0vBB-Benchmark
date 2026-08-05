# HYBRID-001：三视图 CNN 与动态 EdgeConv 图网络

中文 | [English](README_EN.md)

## 1. 模型定位

HYBRID-001 检验两种互补表示能否共同提高分类性能：detector-fixed 二维投影提供稠密
全局视图，event-centered 三维 voxel graph 保留局部空间邻接。图像和图分支分别生成
事件级 embedding，最后进行特征级融合。

| 项目 | 内容 |
|---|---|
| architecture ID | `hybrid_001_cnn_gnn` |
| checkpoint model name | `CNNGNNHybridClassifier` |
| 模型族 | shared-view CNN + dynamic EdgeConv GNN |
| registry input kind | `hybrid` |
| 任务 | `0νββ` signal 与 `Bi214` background 二分类 |
| 输出 | 每事件一个未归一化 signal logit，shape `(B,)` |
| 默认可训练参数 | **341,969** |
| 依赖 | pure PyTorch；不依赖 PyG、DGL 或编译 scatter/kNN 扩展 |

实际实现位于
[`CNNGNNHybridClassifier`](../../../src/next_alt/models/point_graph.py)。本目录只保存
模型专属 YAML、入口与文档。

## 2. 精确输入与预处理

同一个 HDF5 事件必须同时生成图像和 graph 两套表示。shared collate 会检查 batch size
一致，并为不同 node 数的图补零和生成 boolean mask。

### 2.1 三视图 CNN 输入

- batch key：`projections`；
- shape：`(B,3,128,128)`；
- plane 顺序：XY、XZ、YZ；
- 30 mm/bin，fixed detector coordinates；
- origin：`[-1920,-1920,-120]` mm；
- 保留能量除以完整事件总能量，再乘 100；
- 只有 x、y、z 都处于配置立方体内的输入行才被投影，未保留能量不重新归一化。

模型随后把三视图 reshape 为 `(3B,1,128,128)`，分别送入共享 CNN，因此 image
branch 不会在第一层把 XY/XZ/YZ 当 RGB 通道混合。

### 2.2 三维 voxel graph 输入

图分支使用的是 **voxel graph，不是 raw-hit graph**：

1. 用输入能量计算三维质心并中心化坐标；
2. 以 15 mm cell 聚合输入行；
3. 量化后的 voxel center 再按 voxel energy 中心化，去除 half-bin offset；
4. 若 node 数超过 512，按 deposited energy 从高到低保留 512 个，坐标提供确定性
   tie-break；
5. 截断后 energy fraction 仍相对完整事件总能量计算，不重新归一化；
6. batch 内 pad 到该 batch 的最大 node 数，并生成 `mask`。

| batch tensor | shape | 含义 |
|---|---|---|
| `coords` | `(B,N,3)` | event-centered voxel center，mm 除以 `coordinate_scale=1000` |
| `features` | `(B,N,2)` | `[energy_fraction, log1p(hit_count)]` |
| `mask` | `(B,N)` | 有效 node 为 `true`，padding 为 `false` |
| `num_points` | `(B,)` | 每事件截断后的有效 node 数 |

`point_coverage` 记录截断后保留的能量比例。projection 与 point 表示共享 event ID 和
label，但一个保留绝对 detector 坐标，另一个刻意只保留相对三维拓扑。

## 3. 网络架构与张量 shape

### 3.1 Shared-view CNN 分支

每个单通道视图独立经过同一个 `_SharedViewEncoder`：

| 层 | 运算 | 单视图输出 shape |
|---|---|---|
| 输入 | 单个 XY/XZ/YZ 投影 | `(1,128,128)` |
| conv 1 | `Conv2d(1,16,3x3,stride=2,pad=1)` + GroupNorm + SiLU | `(16,64,64)` |
| conv 2 | `Conv2d(16,32,3x3,stride=2,pad=1)` + GroupNorm + SiLU | `(32,32,32)` |
| conv 3 | `Conv2d(32,64,3x3,stride=2,pad=1)` + GroupNorm + SiLU | `(64,16,16)` |
| conv 4 | `Conv2d(64,128,3x3,stride=2,pad=1)` + GroupNorm + SiLU | `(128,8,8)` |
| pooling | `AdaptiveAvgPool2d(1)` | `(128,)` |
| projection | `Linear(128,128)` | `(128,)` |

三个 view embedding stack 为 `(B,3,128)` 并加 learned `view_identity`。在 view 轴上
分别取 mean 和 max，拼接为 256 维，再经 `Linear(256,128) -> SiLU` 得到 128 维
image embedding。该分支没有 residual blocks，也没有 attention gate。

### 3.2 动态 EdgeConv graph 分支

每个 node 首先拼接 3 维坐标与 2 维特征：

1. node encoder：`Linear(5,96) -> LayerNorm -> SiLU -> Dropout(0.1) ->
   Linear(96,96) -> SiLU`；
2. 依次通过 3 个 residual EdgeConv blocks，`k=16`，self edge 排除；
3. 第一层用 centered `coords` 建 kNN；
4. 第二、三层用当前 node feature 除以 `sqrt(96)` 后在 feature space 重建 kNN；
5. 每层邻域都是离散重算的，因此是 dynamic kNN。

对中心 node `i` 和邻居 `j`，edge vector 为：

```text
[h_i, h_j - h_i, x_j - x_i, ||x_j - x_i||]
```

维度为 `96+96+3+1=196`。即使后两层用 feature space 选邻居，message 中仍显式使用
原始 centered coordinate 的相对向量和距离。每个 block 执行：

- edge MLP：`196 -> 96 -> 96`，中间含 LayerNorm、SiLU、dropout；
- 对有效邻居做 masked max aggregation；
- 与原 node feature 做 residual addition；
- LayerNorm，并把 padding node 重新置零。

三层后，对有效 nodes 分别做 masked mean 和 max，拼接为 192 维；
`Linear(192,192) -> SiLU` 得到 graph embedding。

### 3.3 跨模态融合与分类头

- image embedding：128 维；
- graph embedding：192 维；
- 拼接：320 维；
- head：`Linear(320,192) -> SiLU -> Dropout(0.1) -> Linear(192,1)`；
- 输出：`(B,)` logits。

## 4. 与参考方法的关系和边界

本模型是 NEXT 表示级 hybrid，不是下列论文模型的直接复现。

- [Wang et al., *Dynamic Graph CNN for Learning on Point Clouds*, ACM TOG 2019](https://arxiv.org/abs/1801.07829)，
  arXiv:1801.07829，DOI
  [10.1145/3326362](https://doi.org/10.1145/3326362)。本图分支直接借鉴 EdgeConv 与
  feature-space dynamic kNN；但这里输入是中心化 detector voxels，使用 padded dense
  `torch.cdist`、显式坐标 edge features、固定三层宽度和本项目自己的 residual/norm。
- [Su et al., *Multi-view Convolutional Neural Networks for 3D Shape Recognition*, ICCV 2015](https://arxiv.org/abs/1505.00880)，
  arXiv:1505.00880，DOI
  [10.1109/ICCV.2015.114](https://doi.org/10.1109/ICCV.2015.114)。共同点仅是共享 CNN
  编码多个二维视图再融合；当前输入是 detector 能量投影，融合为 mean/max，不复现
  原 MVCNN。
- [Wu and He, *Group Normalization*, ECCV 2018](https://arxiv.org/abs/1803.08494)，
  arXiv:1803.08494，DOI
  [10.1007/978-3-030-01261-8_1](https://doi.org/10.1007/978-3-030-01261-8_1)。
  仅 image branch 使用 GroupNorm；graph MLP/block 使用 LayerNorm。

最后的 CNN/GNN concatenation 是本项目实验设计，没有对应的单一“原论文架构”。文档
中的 hybrid、EdgeConv 或 multi-view 名称都应理解为方法家族，不是复现声明。

## 5. 关键配置

权威配置见 [`config.yaml`](config.yaml)。

| 类别 | 参数 | 默认值 |
|---|---|---:|
| 图像表示 | grid / bin / input scale | `128 / 30 mm / 100` |
| graph 表示 | point bin / coordinate scale | `15 mm / 1000` |
| graph 表示 | max points | `512` |
| 模型 | node feature dim | `2` |
| 模型 | image base channels / embedding | `16 / 128` |
| 模型 | graph hidden / embedding | `96 / 192` |
| 模型 | graph layers / k | `3 / 16` |
| 模型 | classifier dim / dropout | `192 / 0.1` |
| 数据 | max files per class | `100` |
| 数据 | split seed / fractions | `42 / [0.8,0.1,0.1]` |
| 数据 | balanced training / shuffle buffer | `true / 512` |
| 训练 | batch size / epochs | `8 / 50` |
| 训练 | learning rate / weight decay | `3e-4 / 1e-4` |
| 训练 | gradient clip norm | `1.0` |
| 训练 | early-stop patience | `12` |
| 训练 | AMP / deterministic | `auto / false` |

## 6. 共享训练机制与命令

入口只固定 architecture ID，统一训练器负责 `BCEWithLogitsLoss`、AdamW、
`CosineAnnealingLR(T_max=50)`、gradient clipping、CUDA AMP、validation AUC checkpoint
selection 和 early stopping。训练 stream 按 signal/background 交替平衡；validation
不平衡、不 shuffle。当前 `balance_training_classes=true` 分支不会执行 event-buffer
shuffle；表中的 `512` 是保留的配置值，实际每个 epoch 只打乱源文件顺序。checkpoint
同时记录 model config、representation config、split、source inventory、训练配置与历史。

正式配置入口：

```bash
cd /home/wenyu/summer
source .venv/bin/activate
python 01_code/architectures/hybrid_001_cnn_gnn/train_classification.py
```

可以传入自定义 YAML。当前正式 checkpoint、epoch CSV、history JSON 和训练图都已存在；
默认配置从 shared defaults 继承 `allow_overwrite: false`，所以直接执行会在训练开始前
拒绝重跑，不会覆盖或自动续训。重新实验时应复制 YAML，并配置独立的 checkpoint、
log 和 plot 目录。

## 7. 已完成实验结果

| 指标 | 结果 |
|---|---:|
| 实际训练 epochs | 17 |
| best epoch | **5** |
| best validation AUC | **0.916308** |
| full-test 文件 / events | `1,490 / 115,499` |
| full-test 排名 | **9 / 10** |
| matched AUC | **0.912542** |
| inclusive AUC | **0.912792** |
| energy independence | **0.976863** |

正式评测为 strict、`comparable=True`，warning/error 均为 0。当前 hybrid 配置没有超过
单独的 Multi-view CNN 或表现最好的 graph 模型，说明“表示更多”并不自动带来更好的
优化或泛化；该结果只评价当前融合容量和训练设置，不否定 hybrid 方向本身。

完整重评示例使用新的 `_rerun` 目录：

```bash
cd /home/wenyu/summer
source .venv/bin/activate
energybench next \
  02_models/checkpoints/NEXTALT_hybrid_001_cnn_gnn_classification_best.pt \
  --model-id hybrid_001_cnn_gnn \
  --manifest manifests/next_0nubb_vs_bi214.yaml \
  --split test \
  --device cuda:0 \
  --batch-size 8 \
  --num-workers 0 \
  --output-dir 04_evaluations/NEXTALT_hybrid_001_cnn_gnn_test_rerun
```

不传 `--max-files-per-class`，因此评测完整 test split；输出目录必须事先不存在。

## 8. Checkpoint、训练历史与评测产物

- [best checkpoint](../../../02_models/checkpoints/NEXTALT_hybrid_001_cnn_gnn_classification_best.pt)
- [last checkpoint](../../../02_models/checkpoints/NEXTALT_hybrid_001_cnn_gnn_classification_last.pt)
- [epoch CSV](../../../03_training_runs/logs/NEXTALT_hybrid_001_cnn_gnn_classification_epochs.csv)
- [完整 history JSON](../../../03_training_runs/logs/NEXTALT_hybrid_001_cnn_gnn_classification_history.json)
- [训练历史图](../../../03_training_runs/history_plots/NEXTALT_hybrid_001_cnn_gnn_classification_history.png)
- [test predictions](../../../04_evaluations/NEXTALT_hybrid_001_cnn_gnn_test/predictions_test.npz)
- [test results.csv](../../../04_evaluations/NEXTALT_hybrid_001_cnn_gnn_test/evaluation_test/results.csv)
- [strict metrics.json](../../../04_evaluations/NEXTALT_hybrid_001_cnn_gnn_test/evaluation_test/.energybench/metrics.json)
- [energy-matched ROC](../../../04_evaluations/NEXTALT_hybrid_001_cnn_gnn_test/evaluation_test/energy_matched_roc.png)
- [score-energy dependence](../../../04_evaluations/NEXTALT_hybrid_001_cnn_gnn_test/evaluation_test/score_energy_dependence.png)
- [十模型正式排行榜](../../../04_evaluations/NEXTALT_all_models_comparison/leaderboard.csv)

## 9. 本地关联文件

| 文件 | 作用 |
|---|---|
| [config.yaml](config.yaml) | 两种表示、双分支模型和训练参数 |
| [train_classification.py](train_classification.py) | 模型专属训练入口 |
| [src/next_alt/models/point_graph.py](../../../src/next_alt/models/point_graph.py) | shared-view encoder、dense kNN、EdgeConv 和 hybrid classifier |
| [src/next_alt/data.py](../../../src/next_alt/data.py) | 投影、中心化 voxel graph、截断、padding/collate |
| [src/next_cnn/data.py](../../../src/next_cnn/data.py) | HDF5 读取、文件级 split 和 coarse projection |
| [src/next_alt/registry.py](../../../src/next_alt/registry.py) | `hybrid` input-kind 与模型注册 |
| [src/next_alt/training.py](../../../src/next_alt/training.py) | 统一训练、AUC selection 和 early stopping |
| [src/next_alt/checkpoint.py](../../../src/next_alt/checkpoint.py) | format-v3 checkpoint provenance |
| [src/next_alt/adapter.py](../../../src/next_alt/adapter.py) | 图像与 padded graph 的推理适配 |
| [评测 manifest](../../../manifests/next_0nubb_vs_bi214.yaml) | strict NEXT evaluation contract |
| [完整对比结果](../../../docs/ALTERNATIVE_EVALUATION_RESULTS.md) | 十模型排名和统一结论 |
| [使用指南](../../../docs/USAGE_GUIDE.md) | EnergyBench CLI 与评测目录说明 |

## 10. 局限与解释边界

- `_knn` 使用 padded tensor 上的 dense `torch.cdist`；时间和距离矩阵内存近似随有效
  node 上限的平方增长，`max_points=512` 是必要的计算边界。
- 按能量截断偏向高能 voxel，可能删除低能但拓扑重要的细轨迹；point coverage 记录
  retained energy，但没有作为显式模型输入。
- 第一层按物理坐标建图，后续按学习到的 feature 建图；动态邻接是离散操作，邻居索引
  本身不可微。
- projection 保留绝对 detector 位置，graph 刻意中心化；互补性是设计目标，也会形成
  两个统计性质不同、优化难度不同的分支。
- image branch 和 graph branch 的容量、梯度尺度没有显式平衡或辅助 loss；简单拼接
  可能导致某一分支被忽略。
- 三视图仍是有损投影；voxel graph 又受 15 mm 聚合和 512 nodes 截断限制。
- mean/max pooling 和最终 logit 不提供可靠的 node、edge 或 plane 因果解释。
- 当前 checkpoint 只训练分类；energy regression not applicable 是预期状态。
