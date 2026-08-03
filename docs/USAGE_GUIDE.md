# EnergyBench 调用与维护指南

[English version](USAGE_GUIDE_EN.md)

本文只说明怎样运行、目录如何分工、修改哪一处。评分公式和输出解释见
[EVALUATION_STANDARD.md](EVALUATION_STANDARD.md)（[English](EVALUATION_STANDARD_EN.md)）。

## 1. 唯一 Python 环境

项目统一使用 `/home/wenyu/summer/.venv`。每次打开新终端先执行：

```bash
cd /home/wenyu/summer
source .venv/bin/activate
```

激活后只使用 `python`、`python -m pip` 和 `energybench`，不要混用系统的
`python3`、裸 `pip` 或旧环境路径。检查环境：

```bash
which python
python --version
python -m pip check
energybench --version

python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

预期 Python 为 3.11，`which python` 指向本项目 `.venv/bin/python`。退出环境使用
`deactivate`。

### 从零重建 `.venv`

通常不需要重建。确有必要时，先确认
`envs/python-builds/cpython-3.11-linux-x86_64-gnu/bin/python3.11` 存在，然后执行：

```bash
cd /home/wenyu/summer

envs/python-builds/cpython-3.11-linux-x86_64-gnu/bin/python3.11 \
  -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements/next-cnn-cu128.txt
python -m pip install -e .
```

`requirements/next-cnn-cu128.txt` 单独安装 RTX 5090 使用的 CUDA 12.8 PyTorch；
`pyproject.toml` 安装评分器、HDF5/Parquet I/O、绘图和 NEXT 数据读取依赖。

## 2. 常用调用

### 2.1 一条命令评测 NEXT checkpoint

先只查看解析结果，不做推理或写文件：

```bash
energybench next \
  02_models/checkpoints/NEXTCNN_next_cnn_v1_run2_best.pt \
  --dry-run
```

正式运行：

```bash
energybench next \
  02_models/checkpoints/NEXTCNN_next_cnn_v1_run2_best.pt
```

程序从 checkpoint 读取数据根目录、任务类型与预处理设置，自动选择分类、独立能量回归
或多任务 manifest，并输出到一个新的 `04_evaluations/<model-id>/` 目录。常用覆盖参数：

```bash
energybench next CHECKPOINT.pt \
  --data /path/to/NEXT \
  --device cuda:0 \
  --batch-size 32 \
  --output-dir 04_evaluations/my-run
```

用 `--max-files-per-class 1` 做小规模流程检查；用 `--no-plots` 只写表格和审计数据。

### 2.2 评测已有 prediction table

推荐先检查，再严格评分：

```bash
energybench inspect predictions_test.npz

energybench evaluate predictions_test.npz \
  --manifest manifests/next_0nubb_vs_bi214.yaml \
  --model-id model-v1-run2 \
  --output-dir 04_evaluations/model-v1-run2/evaluation_test \
  --strict
```

多任务表（同时有 `energy_target` 和 `energy_pred`）改用：

```bash
--manifest manifests/next_0nubb_vs_bi214_multitask.yaml
```

没有可用于能谱匹配的 energy 时才使用
`manifests/next_0nubb_vs_bi214_no_energy.yaml`；此时 matched AUC 不适用，只报告
inclusive AUC。

输出目录原则上应是新目录。只有明确要替换其中已知 EnergyBench 产物时才加
`--allow-existing`；未知文件不会被自动删除。

### 2.3 从其它模型导出预测

模型 adapter 接收 checkpoint 和数据路径，返回逐事件列：

```bash
energybench predict \
  --adapter examples/adapter_template.py:predict \
  --model /path/to/checkpoint \
  --data /path/to/data \
  --output predictions_test.npz
```

实现新 adapter 时复制 `examples/adapter_template.py`，至少返回任务需要的字段，并保证
所有列第一维长度相同。adapter 是可执行 Python，只运行可信代码。

### 2.4 比较模型

只有 event/truth、评测协议、评分源码指纹一致且均来自成功 strict 运行的结果才可正式排名：

```bash
energybench compare \
  04_evaluations/model-a/evaluation_test \
  04_evaluations/model-b/evaluation_test \
  --output-dir 04_evaluations/comparison
```

`--allow-mixed-data` 只生成不可排名的 inventory，不表示结果已经可比。

### 2.5 分数去相关（可选）

只能在独立 calibration background 上拟合，再应用于 test：

```bash
energybench decorrelate predictions_test.npz \
  --calibration predictions_calibration.npz \
  --output predictions_test_decorrelated.npz \
  --score-column score \
  --energy-column energy_condition \
  --label-column label \
  --event-id-column event_id \
  --split-column split
```

不要在正式结果中使用 `--allow-overlap`。

## 3. 训练 CNN-001 分类与能量回归

两个训练程序都只使用 CUDA，不自动回退到 CPU。先跑 smoke：

```bash
python 01_code/architectures/cnn_001_two_conv_baseline/train_classification.py --smoke
python 01_code/architectures/cnn_001_two_conv_baseline/train_energy_regression.py --smoke
```

再按需运行完整训练：

```bash
python 01_code/architectures/cnn_001_two_conv_baseline/train_classification.py
python 01_code/architectures/cnn_001_two_conv_baseline/train_energy_regression.py
```

常用配置位于各程序顶部，也可临时用环境变量覆盖：

```bash
NEXT_DATA_ROOT=/path/to/NEXT \
NEXT_BATCH_SIZE=32 \
NEXT_NUM_EPOCHS=20 \
NEXT_MODEL_SUFFIX=_cnn_001_regression_experiment \
python 01_code/architectures/cnn_001_two_conv_baseline/train_energy_regression.py
```

默认不会覆盖同名 checkpoint，而是自动增加 `_run2`、`_run3`。只有确认覆盖时才使用
`--allow-existing`。网络结构、两个任务的预处理差异和精确参数量见
`01_code/architectures/cnn_001_two_conv_baseline/README.md`。

当前目录没有历史 v2 训练源码和 v2 checkpoint。`src/next_cnn/model.py` 与 adapter 仍能读取兼容的
v2 checkpoint，但历史 v2 评测目录不能据此完整重现训练。

## 4. 程序结构

```text
summer/
├── 01_code/
│   ├── architectures/
│   │   └── cnn_001_two_conv_baseline/
│   │       ├── train_classification.py
│   │       ├── train_energy_regression.py
│   │       └── README.md
│   └── src/project_paths.py  # checkpoint、日志、图像输出位置
├── 02_models/checkpoints/    # best/last checkpoint
├── 03_training_runs/
│   ├── logs/                 # epoch CSV 与 history JSON
│   └── history_plots/        # 训练曲线与 validation score 图
├── 04_evaluations/           # prediction table 与评分结果
├── src/energybench/
│   ├── cli.py                # 命令入口和参数
│   ├── config.py             # manifest 默认值与校验
│   ├── data.py               # NPZ/CSV/HDF5/Parquet 表读取
│   ├── evaluation.py         # 评测总流程、strict 检查、产物写出
│   ├── roc.py                # inclusive/matched ROC
│   ├── regression.py         # ERS-v1 与回归指标
│   ├── dependence.py         # score-energy dependence
│   ├── plotting.py           # 四张评测图
│   ├── reporting.py          # results.csv 与 leaderboard.csv
│   ├── decorrelation.py      # background conditional ECDF
│   └── next_workflow.py      # `energybench next`
├── src/next_cnn/
│   ├── model.py              # CNN-001 与历史 multitask 网络定义
│   ├── data.py               # HDF5 发现、split、三视图投影、Dataset
│   └── adapter.py            # checkpoint → canonical prediction table
├── manifests/                # 固定任务和评分参数
├── examples/                 # synthetic table 与 adapter 示例
├── docs/                     # 标准与本指南
├── pyproject.toml            # 包、入口和默认依赖
└── requirements/             # CUDA PyTorch 版本
```

## 5. 怎样修改

| 目标 | 修改位置 | 修改后必须检查 |
|---|---|---|
| 改类别、正负角色或评分参数 | `manifests/*.yaml` | task/model/data 语义、能量单位、split 均明确 |
| 改 canonical 列名 | manifest 的 `columns`；必要时 `src/energybench/data.py` | `energybench inspect` 映射正确 |
| 改 matched AUC | `src/energybench/roc.py`，策略门槛在 `evaluation.py` | 公式、NA 规则、protocol/schema 版本同步更新 |
| 改 ERS-v1 | `src/energybench/regression.py` | 分数组件、范围、bootstrap 和标准文档同步更新 |
| 改结果表 | `src/energybench/reporting.py` | CSV header、字段解释和 schema 版本同步更新 |
| 改评测图 | `src/energybench/plotting.py` | 坐标、单位、抽样/权重说明不被误导 |
| 改 NEXT 网络 | `src/next_cnn/model.py` | adapter 的 checkpoint 识别和 state dict 仍一致 |
| 改投影或数据 split | `src/next_cnn/data.py` | 训练与推理使用同一实现，旧 checkpoint provenance 不被伪装成新协议 |
| 改训练超参数/循环 | `01_code/architectures/` 下对应任务程序 | 新 model suffix、validation 选择规则和日志字段明确 |
| 接入新模型框架 | 新 adapter + manifest | 每事件对齐、稳定 event ID、可信 split、单位和 score 方向 |

评分定义或字段语义发生变化时，不要沿用旧 protocol/schema 名称；应提升版本并将新旧结果分开。
已有 `04_evaluations/` 是结果证据，不要手工改数值或当作源码模板。

## 6. 最小验证

修改后不需要创建 `tests/` 目录，可用临时输出做流程验证：

```bash
PYTHONDONTWRITEBYTECODE=1 python -c "import energybench, next_cnn; print('imports ok')"
energybench --help
energybench next \
  02_models/checkpoints/NEXTCNN_next_cnn_v1_run2_best.pt \
  --dry-run
```

需要跑完整 synthetic 流程时，将输出写到 `/tmp`：

```bash
tmp_dir="$(mktemp -d /tmp/energybench-check.XXXXXX)"

python examples/make_synthetic_predictions.py \
  --output "$tmp_dir/predictions.npz" \
  --events 2000

energybench evaluate "$tmp_dir/predictions.npz" \
  --manifest examples/manifest.yaml \
  --output-dir "$tmp_dir/evaluation" \
  --roc-bootstrap 10 \
  --regression-bootstrap 10 \
  --strict
```

如修改评分算法，10 次 bootstrap 只能检查流程是否通畅，不能作为正式置信区间。
