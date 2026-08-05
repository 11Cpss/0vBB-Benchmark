# CNN-004：三视图共享编码器与后期融合

中文 | [English](README_EN.md)

## 1. 模型定位

CNN-004 用来检验一个明确假设：XY、XZ、YZ 是几何含义不同且像素位置不对齐的三
个正交投影，因此不应像 RGB 图像一样在第一层卷积中直接混合。模型先用同一个
单通道残差编码器分别处理三个视图，到事件级表示后再进行后期融合。

| 项目 | 内容 |
|---|---|
| architecture ID | `cnn_004_multiview_late_fusion` |
| checkpoint model name | `MultiViewLateFusionCNN` |
| 模型族 | multi-view 2-D residual CNN |
| registry input kind | `projection2d` |
| 任务 | `0νββ` signal 与 `Bi214` background 二分类 |
| 输出 | 每事件一个未归一化 signal logit，shape `(B,)` |
| 默认可训练参数 | **807,666** |
| 主要对照问题 | view-level late fusion 是否优于把三视图当普通通道早期融合 |

模型类不在本目录中重复定义；本目录只保存模型专属配置、入口和说明。实际实现是
[`MultiViewLateFusionCNN`](../../../src/next_alt/models/cnn.py)，由共享 registry 按
architecture ID 构造。

## 2. 精确输入与预处理

原始事件来自 HDF5 的 `/MC/hits/table`，每一行提供 `x,y,z,energy`。共享读取器先按
完整 HDF5 文件做确定性 train/validation/test 划分，避免同一文件中的事件跨 split。

模型输入 `projections` 的 shape 为 `(B,3,128,128)`，视图顺序固定为：

1. channel 0：XY；
2. channel 1：XZ；
3. channel 2：YZ。

默认投影参数为：

| 参数 | 值 | 含义 |
|---|---:|---|
| `projection_grid_size` | 128 | 每个平面的高和宽 |
| `projection_bin_size` | 30 mm | 投影像素尺寸 |
| `projection_origin` | `[-1920,-1920,-120]` mm | 固定 detector 坐标原点 |
| `projection_input_scale` | 100 | 归一化后的数值缩放 |

一个输入行只有在 x、y、z 三个离散坐标都处于配置的三维范围内时才参与三个投影。
保留的能量按完整事件总能量归一化，而不是按落入范围内的能量重新归一化，随后乘
以 100。因此 `projection_coverage` 可以小于 1，且缺失能量不会被隐藏。

进入网络后，`(B,3,128,128)` 被 reshape 为 `(3B,1,128,128)`。三个 plane 共享
同一个 encoder，但不会在第一层被作为三个普通图像通道混合。

## 3. 网络架构与张量 shape

### 3.1 共享单视图编码器

下表的 shape 不含 batch 维，并针对一个单通道视图：

| 层/阶段 | 运算 | 输出 shape |
|---|---|---|
| 输入 | 单个 XY/XZ/YZ 投影 | `(1,128,128)` |
| stem | `Conv2d(1,16,5x5,stride=2,pad=2)` + GroupNorm + SiLU | `(16,64,64)` |
| stage 0 | 2 个 2-D residual blocks，宽度 16 | `(16,64,64)` |
| stage 1 | 2 blocks，首块 stride 2，`16 -> 32` | `(32,32,32)` |
| stage 2 | 2 blocks，首块 stride 2，`32 -> 64` | `(64,16,16)` |
| stage 3 | 2 blocks，首块 stride 2，`64 -> 128` | `(128,8,8)` |
| view pooling | 对两个空间轴取 mean | `(128,)` |

每个 residual block 的主支路为
`3x3 Conv -> GroupNorm -> SiLU -> 3x3 Conv -> GroupNorm`。当通道数或 stride 改变
时，skip 支路使用 `1x1 Conv + GroupNorm`；主支路与 skip 相加后再经过 SiLU。
GroupNorm 不依赖 batch 内统计量，适合不同架构使用不同 batch size 的统一实验。

### 3.2 视图身份、attention 与分类头

三个 128 维视图向量组成 `(B,3,128)`：

1. 加上可学习的 `view_identity`，shape `(1,3,128)`，保留 XY/XZ/YZ 身份；
2. 每个视图经过共享 scorer：`Linear(128,64) -> SiLU -> Linear(64,1)`；
3. 在 view 轴上做 softmax，得到三个事件相关权重；
4. 每个视图乘对应权重；
5. **不对视图求和**，而是按固定顺序 flatten 为 384 维；
6. 分类头：`Linear(384,256) -> SiLU -> Dropout(0.1) -> Linear(256,1)`；
7. squeeze 为 `(B,)` logits。

这种设计同时保留共享局部特征规则、plane 身份和三个视图各自的高层信息。

## 4. 与参考方法的关系和边界

本模型是针对 NEXT 能量沉积数据设计的轻量变体，以下论文用于说明方法来源，**不
表示复现论文中的原始网络、输入数据或训练方案**。

- [Su et al., *Multi-view Convolutional Neural Networks for 3D Shape Recognition*, ICCV 2015](https://arxiv.org/abs/1505.00880)，
  arXiv:1505.00880，DOI
  [10.1109/ICCV.2015.114](https://doi.org/10.1109/ICCV.2015.114)。共同点是先编码
  多个二维视图再融合；原论文处理渲染视图，而这里处理固定 detector 正交能量投影，
  并使用 learned identity、softmax weighting 和有序拼接，不是原始 MVCNN view
  pooling 的复现。
- [He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016](https://arxiv.org/abs/1512.03385)，
  arXiv:1512.03385，DOI
  [10.1109/CVPR.2016.90](https://doi.org/10.1109/CVPR.2016.90)。本模型采用基本
  residual shortcut 思想，但宽度、深度、stem、激活和归一化均为本项目配置。
- [Wu and He, *Group Normalization*, ECCV 2018](https://arxiv.org/abs/1803.08494)，
  arXiv:1803.08494，DOI
  [10.1007/978-3-030-01261-8_1](https://doi.org/10.1007/978-3-030-01261-8_1)。

## 5. 关键配置

参数的权威来源是相邻的 [`config.yaml`](config.yaml)。

| 类别 | 参数 | 默认值 |
|---|---|---:|
| 表示 | grid / bin | `128 / 30 mm` |
| 表示 | input scale | `100` |
| 模型 | base channels | `16` |
| 模型 | stage blocks | `[2,2,2,2]` |
| 模型 | fusion features | `256` |
| 模型 | dropout | `0.1` |
| 数据 | max files per class | `100` |
| 数据 | split seed / fractions | `42 / [0.8,0.1,0.1]` |
| 数据 | balanced training / shuffle buffer | `true / 512` |
| 训练 | batch size / epochs | `16 / 50` |
| 训练 | learning rate / weight decay | `1e-3 / 1e-4` |
| 训练 | gradient clip norm | `1.0` |
| 训练 | early-stop patience | `12` |
| 训练 | AMP / deterministic | `auto / false` |

## 6. 共享训练机制与命令

[`train_classification.py`](train_classification.py) 只固定 architecture ID 并把 YAML
交给共享训练器。共享训练流程为：

- `BCEWithLogitsLoss`；
- AdamW；
- `CosineAnnealingLR(T_max=epochs)`；
- 每步 gradient clipping；
- CUDA AMP，`auto` 在设备支持时解析为 BF16，否则 FP16；
- 按 validation AUC 保存 best checkpoint；
- 连续 12 epochs 未提升时 early stopping；
- 训练数据按 signal/background 交替平衡，validation 不做平衡或 shuffle；当前
  `balance_training_classes=true` 分支不会执行 event-buffer shuffle，表中的 `512`
  是保留的配置值，实际每个 epoch 只打乱源文件顺序。

正式配置的入口命令是：

```bash
cd /home/wenyu/summer
source .venv/bin/activate
python 01_code/architectures/cnn_004_multiview_late_fusion/train_classification.py
```

也可以把另一个 YAML 作为唯一位置参数传入。当前正式 checkpoint、CSV、JSON 和历史
图已经存在，且默认 `allow_overwrite: false`，因此上面的命令现在会在训练开始前拒绝
重跑，不会续训或覆盖已有结果。需要重新实验时，应复制 YAML 并为 checkpoint、log、
plot 配置新的输出目录；不要无意中把正式产物覆盖掉。

## 7. 已完成实验结果

best checkpoint 来自训练过程中的 validation AUC，而下表 test 指标来自完整、无文件
上限的 strict test evaluation。

| 指标 | 结果 |
|---|---:|
| 实际训练 epochs | 24 |
| best epoch | **12** |
| best validation AUC | **0.956498** |
| full-test 文件 / events | `1,490 / 115,499` |
| full-test 排名 | **3 / 10** |
| matched AUC | **0.955819** |
| inclusive AUC | **0.955936** |
| energy independence | **0.978224** |

该次评测与其余九个模型具有相同 evaluation、protocol 和 code fingerprint，状态为
`comparable=True`，且 warning/error 均为 0。

如果需要重新评测，应使用不存在的新目录，例如：

```bash
cd /home/wenyu/summer
source .venv/bin/activate
energybench next \
  02_models/checkpoints/NEXTALT_cnn_004_multiview_late_fusion_classification_best.pt \
  --model-id cnn_004_multiview_late_fusion \
  --manifest manifests/next_0nubb_vs_bi214.yaml \
  --split test \
  --device cuda:0 \
  --batch-size 16 \
  --num-workers 0 \
  --output-dir 04_evaluations/NEXTALT_cnn_004_multiview_late_fusion_test_rerun
```

命令没有传 `--max-files-per-class`，因此使用完整 test split。

## 8. Checkpoint、训练历史与评测产物

- [best checkpoint](../../../02_models/checkpoints/NEXTALT_cnn_004_multiview_late_fusion_classification_best.pt)
- [last checkpoint](../../../02_models/checkpoints/NEXTALT_cnn_004_multiview_late_fusion_classification_last.pt)
- [epoch CSV](../../../03_training_runs/logs/NEXTALT_cnn_004_multiview_late_fusion_classification_epochs.csv)
- [完整 history JSON](../../../03_training_runs/logs/NEXTALT_cnn_004_multiview_late_fusion_classification_history.json)
- [训练历史图](../../../03_training_runs/history_plots/NEXTALT_cnn_004_multiview_late_fusion_classification_history.png)
- [test predictions](../../../04_evaluations/NEXTALT_cnn_004_multiview_late_fusion_test/predictions_test.npz)
- [test results.csv](../../../04_evaluations/NEXTALT_cnn_004_multiview_late_fusion_test/evaluation_test/results.csv)
- [strict metrics.json](../../../04_evaluations/NEXTALT_cnn_004_multiview_late_fusion_test/evaluation_test/.energybench/metrics.json)
- [energy-matched ROC](../../../04_evaluations/NEXTALT_cnn_004_multiview_late_fusion_test/evaluation_test/energy_matched_roc.png)
- [score-energy dependence](../../../04_evaluations/NEXTALT_cnn_004_multiview_late_fusion_test/evaluation_test/score_energy_dependence.png)
- [十模型正式排行榜](../../../04_evaluations/NEXTALT_all_models_comparison/leaderboard.csv)

## 9. 本地关联文件

| 文件 | 作用 |
|---|---|
| [config.yaml](config.yaml) | 本模型表示、网络和训练超参数 |
| [train_classification.py](train_classification.py) | 模型专属训练入口 |
| [src/next_alt/models/cnn.py](../../../src/next_alt/models/cnn.py) | residual block、共享 encoder 和模型实现 |
| [src/next_alt/data.py](../../../src/next_alt/data.py) | representation dispatch、batch collate 和 loader |
| [src/next_cnn/data.py](../../../src/next_cnn/data.py) | HDF5 读取、文件级 split、XY/XZ/YZ 投影 |
| [src/next_alt/registry.py](../../../src/next_alt/registry.py) | architecture ID、model class、input kind 注册 |
| [src/next_alt/training.py](../../../src/next_alt/training.py) | loss、optimizer、scheduler、early stopping 和产物写入 |
| [src/next_alt/checkpoint.py](../../../src/next_alt/checkpoint.py) | format-v3 checkpoint provenance |
| [src/next_alt/adapter.py](../../../src/next_alt/adapter.py) | checkpoint 到 EnergyBench inference 的适配 |
| [评测 manifest](../../../manifests/next_0nubb_vs_bi214.yaml) | NEXT strict evaluation contract |
| [完整对比结果](../../../docs/ALTERNATIVE_EVALUATION_RESULTS.md) | 十个非 Transformer 架构的统一结果 |
| [使用指南](../../../docs/USAGE_GUIDE.md) | EnergyBench CLI 与产物说明 |

## 10. 局限与解释边界

- 三个正交投影会丢失三维对应关系；late fusion 不能恢复已经在投影时丢失的信息。
- 共享 encoder 假设不同 plane 的局部模式可以使用同一组卷积核，未必完全符合 detector
  在不同轴上的物理各向异性。
- learned view identity 保留 plane 名称，但不是旋转等变或旋转不变机制。
- detector-fixed 投影保留绝对位置，也可能让模型利用与拓扑无关的位置差异。
- attention 权重只调节三个视图向量的幅度；由于最终使用有序拼接，它不能解释为对
  三个视图贡献的严格因果归因。
- 当前实验只训练分类任务；评测中的 energy regression 为 not applicable 是预期行为。
