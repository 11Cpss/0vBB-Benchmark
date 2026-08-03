# EnergyBench / NEXT CNN

本项目用于 NEXT 逐事件预测、分类评测与能量回归评测。当前正式评分包括：

- energy-matched ROC/AUC：在信号与背景具有相同能谱后比较分类能力；
- ERS-v1：同时评价逐事件能量误差与总体能谱相似性；
- class-conditional energy dependence：诊断分类分数是否随能量变化。

评分定义、表格字段、全部产物和图像解释见
[评测与数据标准](docs/EVALUATION_STANDARD.md)（[English](docs/EVALUATION_STANDARD_EN.md)）。调用、目录结构和修改位置见
[调用与维护指南](docs/USAGE_GUIDE.md)（[English](docs/USAGE_GUIDE_EN.md)）。

## 快速开始

所有命令都在同一个项目环境中运行：

```bash
cd /home/wenyu/summer
source .venv/bin/activate

python --version
energybench --version
```

评测已有 NEXT checkpoint：

```bash
energybench next --dry-run
energybench next 02_models/checkpoints/NEXTCNN_next_cnn_v1_run2_best.pt
```

评测已经导出的逐事件预测表：

```bash
energybench inspect predictions_test.npz

energybench evaluate predictions_test.npz \
  --manifest manifests/next_0nubb_vs_bi214.yaml \
  --model-id my-model \
  --output-dir 04_evaluations/my-model \
  --strict
```

训练当前可复现的 CNN-001、CNN-002，或深层残差空间模型 CNN-003：

```bash
python 01_code/architectures/cnn_001_two_conv_baseline/train_classification.py --smoke
python 01_code/architectures/cnn_001_two_conv_baseline/train_energy_regression.py --smoke

python 01_code/architectures/cnn_002_global_energy_skip/train_classification.py --smoke
python 01_code/architectures/cnn_002_global_energy_skip/train_energy_regression.py --smoke

python 01_code/architectures/cnn_003_residual_spatial/train_classification.py --smoke
python 01_code/architectures/cnn_003_residual_spatial/train_energy_regression.py --smoke
```

网络结构、两种任务的差异和精确参数量见
[`cnn_001_two_conv_baseline/README.md`](01_code/architectures/cnn_001_two_conv_baseline/README.md)
、
[`cnn_002_global_energy_skip/README.md`](01_code/architectures/cnn_002_global_energy_skip/README.md)
和
[`cnn_003_residual_spatial/README.md`](01_code/architectures/cnn_003_residual_spatial/README.md)。

## 当前边界

- 当前源码可分别训练 CNN-001、CNN-002、CNN-003 的分类和能量回归模型；EnergyBench adapter 仍只对其明确支持的 checkpoint 类型做正式推理和评分。
- 仓库中的部分 v2 图表与评测目录是历史产物；对应 v2 训练源码和 checkpoint 已不在当前项目中，因此不能从本目录完整复现训练。
- `.venv` 是唯一项目环境；`envs/python-builds/` 只是它依赖的 Python 3.11 解释器，不是第二个虚拟环境。
- `results.csv` 是面向汇总的单行结果；完整审计信息位于评测目录的 `.energybench/metrics.json`。

## 主要目录

```text
01_code/architectures/  按英文结构 ID 组织的分类/回归训练入口
01_code/src/            共享输出路径工具
02_models/            checkpoint
03_training_runs/     训练日志与训练图
04_evaluations/       预测表与评分结果
src/energybench/      通用评分器
src/next_cnn/         NEXT 模型、数据读取与 adapter
manifests/            冻结的任务/评分配置
examples/             通用 prediction-table 示例
docs/                 两份项目文档
```

完整命令以本机帮助为准：

```bash
energybench --help
energybench next --help
energybench evaluate --help
```
