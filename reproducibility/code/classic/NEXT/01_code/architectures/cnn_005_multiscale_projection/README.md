# CNN-005：全局与局部双尺度投影 CNN

中文 | [English](README_EN.md)

## 1. 模型定位

CNN-005 同时观察 detector-wide 的 30 mm coarse 投影和 event-centered 的 15 mm
fine 投影，用来检验全局轨迹几何与 endpoint blob 等局部拓扑是否能够互补。两个
尺度使用同一个 residual encoder；模型容量没有因为双分支简单翻倍。

| 项目 | 内容 |
|---|---|
| architecture ID | `cnn_005_multiscale_projection` |
| checkpoint model name | `MultiScaleProjectionCNN` |
| 模型族 | shared-weight multi-scale 2-D residual CNN |
| registry input kind | `multiscale2d` |
| 任务 | `0νββ` signal 与 `Bi214` background 二分类 |
| 输出 | 每事件一个未归一化 signal logit，shape `(B,)` |
| 默认可训练参数 | **865,939** |
| 核心处理方式 | **每个尺度内 plane early fusion，两个尺度之间 scale late fusion** |

实际模型类集中定义在
[`MultiScaleProjectionCNN`](../../../src/next_alt/models/cnn.py)；本目录只保存专属
YAML、训练入口和文档。

## 2. 精确输入与预处理

共享读取器从 HDF5 `/MC/hits/table` 取得 `x,y,z,energy`，并按完整文件做确定性
split。一个事件同时生成下面两套 shape 都为 `(3,128,128)` 的表示。

### 2.1 Coarse detector-wide 投影

- batch key：`projections`，shape `(B,3,128,128)`；
- plane 顺序：XY、XZ、YZ；
- 30 mm/bin；
- 固定 detector 坐标；
- origin `[-1920,-1920,-120]` mm；
- 每个平面覆盖 128 bins。

### 2.2 Fine event-centered 投影

- batch key：`fine_projections`，shape `(B,3,128,128)`；
- 先用原始输入行的能量计算事件质心，并从所有坐标中减去该质心；
- 15 mm/bin；
- 三个轴的投影立方体 origin 均为 `-960 mm`，覆盖约 `[-960,+960) mm`；
- plane 顺序同样是 XY、XZ、YZ。

两套投影都只保留同时落在各自三维范围内的输入行，除以**完整事件总能量**后乘
`projection_input_scale=100`。出界能量不会重新分配；coarse 和 fine coverage 分别
写入 batch/provenance。

必须区分两种融合层级：

- **plane early fusion**：在每一个尺度内部，XY/XZ/YZ 作为 3 个输入通道，第一层
  `Conv2d` 就会混合三个 plane；
- **scale late fusion**：coarse 和 fine 分别通过完整 encoder 得到事件级向量，之后
  才融合；两次 forward 调用的是同一个 encoder 实例，参数完全共享。

## 3. 网络架构与张量 shape

### 3.1 两尺度共享 residual encoder

下表分别适用于 coarse 和 fine 分支：

| 层/阶段 | 运算 | 单尺度输出 shape |
|---|---|---|
| 输入 | 三个正交投影通道 | `(3,128,128)` |
| stem | `Conv2d(3,16,5x5,stride=2,pad=2)` + GroupNorm + SiLU | `(16,64,64)` |
| stage 0 | 2 个 residual blocks，宽度 16 | `(16,64,64)` |
| stage 1 | 2 blocks，首块 stride 2，`16 -> 32` | `(32,32,32)` |
| stage 2 | 2 blocks，首块 stride 2，`32 -> 64` | `(64,16,16)` |
| stage 3 | 2 blocks，首块 stride 2，`64 -> 128` | `(128,8,8)` |
| scale pooling | spatial mean | `(128,)` |

residual block 主支路为
`3x3 Conv -> GroupNorm -> SiLU -> 3x3 Conv -> GroupNorm`；需要降采样或改通道时，
skip 支路使用 `1x1 Conv + GroupNorm`。相加后经过 SiLU。

### 3.2 Scale gate 与显式交互特征

coarse/fine 各得到一个 128 维向量后：

1. stack 为 `(B,2,128)` 并加 learned `scale_identity`；
2. 拼接 coarse/fine 为 256 维；
3. gate：`Linear(256,128) -> SiLU -> Linear(128,2) -> softmax`；
4. 得到 `coarse_weighted` 和 `fine_weighted`，各 128 维；
5. 同时计算 `abs(coarse-fine)` 和 `coarse*fine`，各 128 维；
6. 四组特征拼接为 `128*4=512` 维；
7. 分类头：`Linear(512,256) -> SiLU -> Dropout(0.1) -> Linear(256,1)`；
8. squeeze 为 `(B,)` logits。

## 4. 与参考方法的关系和边界

本实现是 NEXT 双尺度输入的定制模型，不是任何论文架构的逐层复现。

- [He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016](https://arxiv.org/abs/1512.03385)，
  arXiv:1512.03385，DOI
  [10.1109/CVPR.2016.90](https://doi.org/10.1109/CVPR.2016.90)。这里只采用基本
  residual shortcut 思想。
- [Bromley et al., *Signature Verification using a “Siamese” Time Delay Neural Network*, NeurIPS 1993](https://proceedings.neurips.cc/paper/1993/hash/288cc0ff022877bd3df94bc9360b9c5d-Abstract.html)。
  这篇工作是共享权重双分支的历史参考；当前模型不是度量学习 Siamese network，
  两个分支输入也不是待比较的两个样本，而是同一事件的两个尺度。
- [Lin et al., *Feature Pyramid Networks for Object Detection*, CVPR 2017](https://arxiv.org/abs/1612.03144)，
  arXiv:1612.03144，DOI
  [10.1109/CVPR.2017.106](https://doi.org/10.1109/CVPR.2017.106)。FPN 仅作为多尺度
  表示的动机参考；本模型没有 top-down pathway、lateral connection 或 pyramid
  detection heads，因此不应称为 FPN。
- [Wu and He, *Group Normalization*, ECCV 2018](https://arxiv.org/abs/1803.08494)，
  arXiv:1803.08494，DOI
  [10.1007/978-3-030-01261-8_1](https://doi.org/10.1007/978-3-030-01261-8_1)。

## 5. 关键配置

权威配置见 [`config.yaml`](config.yaml)。

| 类别 | 参数 | 默认值 |
|---|---|---:|
| coarse 表示 | grid / bin / input scale | `128 / 30 mm / 100` |
| fine 表示 | grid / bin | `128 / 15 mm` |
| 模型 | base channels | `16` |
| 模型 | stage blocks | `[2,2,2,2]` |
| 模型 | fusion features | `256` |
| 模型 | dropout | `0.1` |
| 数据 | max files per class | `100` |
| 数据 | split seed / fractions | `42 / [0.8,0.1,0.1]` |
| 数据 | balanced training / shuffle buffer | `true / 512` |
| 训练 | batch size / epochs | `8 / 50` |
| 训练 | learning rate / weight decay | `1e-3 / 1e-4` |
| 训练 | gradient clip norm | `1.0` |
| 训练 | early-stop patience | `12` |
| 训练 | AMP / deterministic | `auto / false` |

## 6. 共享训练机制与命令

模型入口把 architecture ID 与 YAML 交给统一训练器。训练器使用
`BCEWithLogitsLoss`、AdamW、`CosineAnnealingLR(T_max=50)`、gradient clipping 和
CUDA AMP；每个 epoch 在完整 validation stream 上计算 AUC，按 validation AUC 写入
best checkpoint，并在 patience 12 时 early stop。训练 stream 按两类交替平衡，
validation 保持原始分布与确定性顺序。当前 `balance_training_classes=true` 分支不会
执行 event-buffer shuffle；表中的 `512` 是保留的配置值，实际每个 epoch 只打乱源文件顺序。

正式配置入口：

```bash
cd /home/wenyu/summer
source .venv/bin/activate
python 01_code/architectures/cnn_005_multiscale_projection/train_classification.py
```

可把自定义 YAML 作为唯一位置参数。当前正式 checkpoint、history CSV/JSON 和历史图
已经存在，默认 `allow_overwrite: false`，所以直接执行会在训练开始前拒绝重跑，不会
自动续训或覆盖。重新实验时应复制 YAML 并指定全新的 checkpoint/log/plot 输出目录。

## 7. 已完成实验结果

| 指标 | 结果 |
|---|---:|
| 实际训练 epochs | 19 |
| best epoch | **7** |
| best validation AUC | **0.949000** |
| full-test 文件 / events | `1,490 / 115,499` |
| full-test 排名 | **6 / 10** |
| matched AUC | **0.947787** |
| inclusive AUC | **0.948255** |
| energy independence | **0.980444** |

在这十个模型中，本模型的 energy-independence score 最高，但 discrimination 低于
Multi-view CNN、GINE、ParticleNet、GravNet 和 PointNet++。正式评测为 strict、
`comparable=True`，warning/error 均为 0。

重新执行完整 test evaluation 时使用新的目录：

```bash
cd /home/wenyu/summer
source .venv/bin/activate
energybench next \
  02_models/checkpoints/NEXTALT_cnn_005_multiscale_projection_classification_best.pt \
  --model-id cnn_005_multiscale_projection \
  --manifest manifests/next_0nubb_vs_bi214.yaml \
  --split test \
  --device cuda:0 \
  --batch-size 8 \
  --num-workers 0 \
  --output-dir 04_evaluations/NEXTALT_cnn_005_multiscale_projection_test_rerun
```

未传 `--max-files-per-class`，因此不会缩小 test split；`_rerun` 目录也必须事先不存在。

## 8. Checkpoint、训练历史与评测产物

- [best checkpoint](../../../02_models/checkpoints/NEXTALT_cnn_005_multiscale_projection_classification_best.pt)
- [last checkpoint](../../../02_models/checkpoints/NEXTALT_cnn_005_multiscale_projection_classification_last.pt)
- [epoch CSV](../../../03_training_runs/logs/NEXTALT_cnn_005_multiscale_projection_classification_epochs.csv)
- [完整 history JSON](../../../03_training_runs/logs/NEXTALT_cnn_005_multiscale_projection_classification_history.json)
- [训练历史图](../../../03_training_runs/history_plots/NEXTALT_cnn_005_multiscale_projection_classification_history.png)
- [test predictions](../../../04_evaluations/NEXTALT_cnn_005_multiscale_projection_test/predictions_test.npz)
- [test results.csv](../../../04_evaluations/NEXTALT_cnn_005_multiscale_projection_test/evaluation_test/results.csv)
- [strict metrics.json](../../../04_evaluations/NEXTALT_cnn_005_multiscale_projection_test/evaluation_test/.energybench/metrics.json)
- [energy-matched ROC](../../../04_evaluations/NEXTALT_cnn_005_multiscale_projection_test/evaluation_test/energy_matched_roc.png)
- [score-energy dependence](../../../04_evaluations/NEXTALT_cnn_005_multiscale_projection_test/evaluation_test/score_energy_dependence.png)
- [十模型正式排行榜](../../../04_evaluations/NEXTALT_all_models_comparison/leaderboard.csv)

## 9. 本地关联文件

| 文件 | 作用 |
|---|---|
| [config.yaml](config.yaml) | 本模型表示、网络和训练超参数 |
| [train_classification.py](train_classification.py) | 模型专属训练入口 |
| [src/next_alt/models/cnn.py](../../../src/next_alt/models/cnn.py) | shared residual encoder、scale gate 和分类头 |
| [src/next_alt/data.py](../../../src/next_alt/data.py) | coarse/fine representation、中心化、collate 和 loader |
| [src/next_cnn/data.py](../../../src/next_cnn/data.py) | HDF5 读取、split 和正交投影实现 |
| [src/next_alt/registry.py](../../../src/next_alt/registry.py) | architecture/model/input-kind 注册 |
| [src/next_alt/training.py](../../../src/next_alt/training.py) | 统一训练与 early stopping |
| [src/next_alt/checkpoint.py](../../../src/next_alt/checkpoint.py) | format-v3 checkpoint 与 provenance |
| [src/next_alt/adapter.py](../../../src/next_alt/adapter.py) | `projections`/`fine_projections` 推理适配 |
| [评测 manifest](../../../manifests/next_0nubb_vs_bi214.yaml) | strict NEXT evaluation contract |
| [完整对比结果](../../../docs/ALTERNATIVE_EVALUATION_RESULTS.md) | 十模型排名和统一评测说明 |
| [使用指南](../../../docs/USAGE_GUIDE.md) | EnergyBench 命令和目录结构 |

## 10. 局限与解释边界

- 每个尺度内部仍然把不对齐的 XY/XZ/YZ 当作通道早期混合；这与 CNN-004 的处理假设
  不同，也是两个模型的重要实验差异。
- fine 投影通过中心化改善局部利用率，但丢失绝对 detector 位置；coarse 分支保留该
  信息，二者可能学习到不同类型的 shortcut。
- 同一组卷积核同时处理 30 mm 和 15 mm 像素，假设跨尺度局部模式可以共享参数。
- 双尺度仍是二维投影，无法保留完整三维 hit/voxel 对应关系。
- gate 是事件相关的两个标量权重，不等于可验证的物理尺度重要性解释。
- 该架构不是 FPN，也没有显式跨层特征金字塔。
- 当前 checkpoint 只适用于分类；energy regression 的 not applicable 状态是预期结果。
