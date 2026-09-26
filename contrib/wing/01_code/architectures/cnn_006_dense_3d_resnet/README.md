# CNN-006：稠密体素 3-D Residual CNN

中文 | [English](README_EN.md)

## 1. 模型定位

CNN-006 不生成二维投影，而是把 event-centered 能量沉积放入固定的三维体素张量，
直接用 3-D convolution 学习轨迹连通性、endpoint blob 和 multi-site 结构。

> **名称边界：这里的 Dense 表示输入是稠密分配的 `(C,D,H,W)` voxel tensor；模型
> 不是 DenseNet，也没有 DenseNet 式 dense connectivity。**

| 项目 | 内容 |
|---|---|
| architecture ID | `cnn_006_dense_3d_resnet` |
| checkpoint model name | `Dense3DResidualCNN` |
| 模型族 | dense-voxel 3-D residual CNN |
| registry input kind | `dense3d` |
| 任务 | `0νββ` signal 与 `Bi214` background 二分类 |
| 输出 | 每事件一个未归一化 signal logit，shape `(B,)` |
| 默认可训练参数 | **688,433** |
| 核心对照问题 | 保留原生三维邻接是否优于二维投影，以及稠密 3-D 计算代价是否值得 |

实际模型实现在
[`Dense3DResidualCNN`](../../../src/next_alt/models/cnn.py)，本目录保存专属配置、
训练入口和文档。

## 2. 精确输入与预处理

HDF5 读取器从 `/MC/hits/table` 提取每事件的 `x,y,z,energy`。事件表示按以下顺序
构造：

1. 验证坐标、能量和完整事件总能量；
2. 计算原始输入行的能量加权三维质心并从所有坐标中减去；
3. 用 15 mm cell 聚合相同 voxel 中的输入行；
4. 用 voxel energy 对量化后的 voxel center 再做一次中心校正，移除 half-bin 量化
   offset，同时保留相对距离；
5. 把中心化 voxel 放入 `96x96x96` cube；
6. cube 外 voxel 丢弃，但保留的能量不重新归一化。

最终 batch key 是 `volume`，shape 为 `(B,2,96,96,96)`。PyTorch 空间维顺序对应
`(D,H,W)=(z,y,x)`：

| channel | 数值 |
|---:|---|
| 0 | voxel deposited energy / 完整事件总能量 |
| 1 | `log1p(该 voxel 中原始输入行数)` |

15 mm、96 bins 对应每轴 1,440 mm 的中心 cube。`representation_coverage` 是 cube 内
energy fraction 的总和；它可以小于 1，且模型不会通过重新归一化掩盖裁剪。该表示
不使用 `max_points`，但固定 cube 本身会做空间裁剪。

## 3. 网络架构与张量 shape

### 3.1 3-D stem 与 residual stages

| 层/阶段 | 运算 | 输出 shape，不含 batch |
|---|---|---|
| 输入 | 双通道 dense voxel volume | `(2,96,96,96)` |
| stem | `Conv3d(2,12,5x5x5,stride=2,pad=2)` + GroupNorm + SiLU | `(12,48,48,48)` |
| stage 0 | 1 个 3-D residual block，宽度 12 | `(12,48,48,48)` |
| stage 1 | 2 blocks，首块 stride 2，`12 -> 24` | `(24,24,24,24)` |
| stage 2 | 2 blocks，首块 stride 2，`24 -> 48` | `(48,12,12,12)` |
| stage 3 | 1 block，stride 2，`48 -> 96` | `(96,6,6,6)` |

每个 3-D residual block 的主支路为
`3x3x3 Conv -> GroupNorm -> SiLU -> 3x3x3 Conv -> GroupNorm`。当通道或 stride 改变
时，skip 使用 `1x1x1 Conv + GroupNorm`；主支路与 skip 相加后经过 SiLU。stem 先将
`96^3` 降采样到 `48^3`，是控制 activation memory 的关键步骤。

### 3.2 Event pooling 与分类头

最后的 `(96,6,6,6)` feature volume 分别对三维空间取：

- global mean：96 维；
- global max：96 维。

二者拼接成 192 维，再经过
`Linear(192,128) -> SiLU -> Dropout(0.1) -> Linear(128,1)`，输出 `(B,)` logits。

## 4. 与参考方法的关系和边界

本实现把 3-D convolution 和 residual shortcut 用于静态 detector voxel；引用用于说明
技术来源，不表示复现视频模型或原论文实验。

- [Hara, Kataoka and Satoh, *Learning Spatio-Temporal Features with 3D Residual Networks for Action Recognition*, ICCV Workshops 2017](https://arxiv.org/abs/1708.07632)，
  arXiv:1708.07632，DOI
  [10.1109/ICCVW.2017.373](https://doi.org/10.1109/ICCVW.2017.373)。共同点是 3-D
  residual blocks；原论文的第三维包含时间，本项目三个维度全部是 detector 空间。
- [Tran et al., *Learning Spatiotemporal Features with 3D Convolutional Networks*, ICCV 2015](https://arxiv.org/abs/1412.0767)，
  arXiv:1412.0767。它是 3-D ConvNet 的背景参考；当前网络不是 C3D，层数、pooling、
  normalization 和任务均不同。
- [He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016](https://arxiv.org/abs/1512.03385)，
  arXiv:1512.03385，DOI
  [10.1109/CVPR.2016.90](https://doi.org/10.1109/CVPR.2016.90)。
- [Wu and He, *Group Normalization*, ECCV 2018](https://arxiv.org/abs/1803.08494)，
  arXiv:1803.08494，DOI
  [10.1007/978-3-030-01261-8_1](https://doi.org/10.1007/978-3-030-01261-8_1)。
  GroupNorm 尤其适合这里默认 batch size 2 的显存受限训练。

## 5. 关键配置

权威配置见 [`config.yaml`](config.yaml)。

| 类别 | 参数 | 默认值 |
|---|---|---:|
| 表示 | dense grid / bin | `96 / 15 mm` |
| 表示 | 输入 channels | `energy_fraction, log1p(hit_count)` |
| 模型 | base channels | `12` |
| 模型 | stage blocks | `[1,2,2,1]` |
| 模型 | head features | `128` |
| 模型 | dropout | `0.1` |
| 数据 | max files per class | `100` |
| 数据 | split seed / fractions | `42 / [0.8,0.1,0.1]` |
| 数据 | balanced training / shuffle buffer | `true / 512` |
| 训练 | batch size / epochs | `2 / 50` |
| 训练 | learning rate / weight decay | `1e-3 / 1e-4` |
| 训练 | gradient clip norm | `1.0` |
| 训练 | early-stop patience | `12` |
| 训练 | AMP / deterministic | `auto / false` |

## 6. 共享训练机制与命令

模型入口调用统一训练器，使用 `BCEWithLogitsLoss`、AdamW、
`CosineAnnealingLR(T_max=50)`、gradient clipping 和 CUDA AMP。best checkpoint 按
validation AUC 选择；连续 12 epochs 没有超过当前 best 即停止。训练 stream 交替读取
signal/background，validation 保持原始分布和确定性顺序。当前
`balance_training_classes=true` 分支不会执行 event-buffer shuffle；表中的 `512`
是保留的配置值，实际每个 epoch 只打乱源文件顺序。

正式配置入口：

```bash
cd /home/wenyu/summer
source .venv/bin/activate
python 01_code/architectures/cnn_006_dense_3d_resnet/train_classification.py
```

也可传入一个自定义 YAML。当前正式 best/last checkpoint、history CSV/JSON 和训练图
已经存在，且默认 `allow_overwrite: false`；直接执行会在分配 GPU/开始训练前拒绝重跑，
不会续训或覆盖。新的实验应使用复制的 YAML 并设置独立 checkpoint、log、plot 目录。

## 7. 已完成实验结果

| 指标 | 结果 |
|---|---:|
| 实际训练 epochs | 22 |
| best epoch | **10** |
| best validation AUC | **0.929562** |
| full-test 文件 / events | `1,490 / 115,499` |
| full-test 排名 | **7 / 10** |
| matched AUC | **0.928210** |
| inclusive AUC | **0.928082** |
| energy independence | **0.977320** |

正式 full-test 评测为 strict、`comparable=True`，warning/error 均为 0。该配置保留
三维连接，但推理明显慢于较轻的投影、点云和图模型，且在当前实验中没有取得更高 AUC。

重评时应使用新的 `_rerun` 输出目录：

```bash
cd /home/wenyu/summer
source .venv/bin/activate
energybench next \
  02_models/checkpoints/NEXTALT_cnn_006_dense_3d_resnet_classification_best.pt \
  --model-id cnn_006_dense_3d_resnet \
  --manifest manifests/next_0nubb_vs_bi214.yaml \
  --split test \
  --device cuda:0 \
  --batch-size 2 \
  --num-workers 0 \
  --output-dir 04_evaluations/NEXTALT_cnn_006_dense_3d_resnet_test_rerun
```

命令未传 `--max-files-per-class`，因此使用完整 test split；输出目录必须事先不存在。

## 8. Checkpoint、训练历史与评测产物

- [best checkpoint](../../../02_models/checkpoints/NEXTALT_cnn_006_dense_3d_resnet_classification_best.pt)
- [last checkpoint](../../../02_models/checkpoints/NEXTALT_cnn_006_dense_3d_resnet_classification_last.pt)
- [epoch CSV](../../../03_training_runs/logs/NEXTALT_cnn_006_dense_3d_resnet_classification_epochs.csv)
- [完整 history JSON](../../../03_training_runs/logs/NEXTALT_cnn_006_dense_3d_resnet_classification_history.json)
- [训练历史图](../../../03_training_runs/history_plots/NEXTALT_cnn_006_dense_3d_resnet_classification_history.png)
- [test predictions](../../../04_evaluations/NEXTALT_cnn_006_dense_3d_resnet_test/predictions_test.npz)
- [test results.csv](../../../04_evaluations/NEXTALT_cnn_006_dense_3d_resnet_test/evaluation_test/results.csv)
- [strict metrics.json](../../../04_evaluations/NEXTALT_cnn_006_dense_3d_resnet_test/evaluation_test/.energybench/metrics.json)
- [energy-matched ROC](../../../04_evaluations/NEXTALT_cnn_006_dense_3d_resnet_test/evaluation_test/energy_matched_roc.png)
- [score-energy dependence](../../../04_evaluations/NEXTALT_cnn_006_dense_3d_resnet_test/evaluation_test/score_energy_dependence.png)
- [十模型正式排行榜](../../../04_evaluations/NEXTALT_all_models_comparison/leaderboard.csv)

## 9. 本地关联文件

| 文件 | 作用 |
|---|---|
| [config.yaml](config.yaml) | dense representation、网络和训练超参数 |
| [train_classification.py](train_classification.py) | 模型专属训练入口 |
| [src/next_alt/models/cnn.py](../../../src/next_alt/models/cnn.py) | 3-D residual block 和 `Dense3DResidualCNN` |
| [src/next_alt/data.py](../../../src/next_alt/data.py) | 中心化 voxelization、dense cube 和 coverage |
| [src/next_cnn/data.py](../../../src/next_cnn/data.py) | HDF5 读取与文件级 split |
| [src/next_alt/registry.py](../../../src/next_alt/registry.py) | `dense3d` input-kind 注册 |
| [src/next_alt/training.py](../../../src/next_alt/training.py) | 统一 loss、optimizer、scheduler 与 early stopping |
| [src/next_alt/checkpoint.py](../../../src/next_alt/checkpoint.py) | format-v3 checkpoint 和 provenance |
| [src/next_alt/adapter.py](../../../src/next_alt/adapter.py) | dense volume 的推理适配 |
| [评测 manifest](../../../manifests/next_0nubb_vs_bi214.yaml) | strict NEXT evaluation contract |
| [完整对比结果](../../../docs/ALTERNATIVE_EVALUATION_RESULTS.md) | 十模型排名和评测结论 |
| [使用指南](../../../docs/USAGE_GUIDE.md) | EnergyBench 命令和结果目录 |

## 10. 局限与解释边界

- 稠密 `96^3` tensor 中大多数 voxel 通常为零，3-D convolution 仍对全部位置计算，
  显存和算力利用率低于稀疏卷积或图方法。
- 固定中心 cube 会丢弃范围外能量；coverage 记录损失，但模型并未显式把 coverage
  作为输入特征。
- 事件中心化移除绝对 detector 位置，有助于聚焦相对拓扑，也会丢失可能有用的位置
  信息。
- 15 mm voxelization 会合并更细尺度的结构；增大 grid 或减小 bin 会快速增加三次方
  级别的 activation 开销。
- global mean/max 会把最终 `6^3` feature map 压缩为单个事件向量，无法直接提供局部
  可解释定位。
- 这不是 DenseNet，也不是 sparse CNN；不要据名称推断 dense skip connections。
- 当前 checkpoint 只训练分类，energy regression not applicable 是预期行为。
