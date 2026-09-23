# SuperNEMO workflow

本项目实现 SuperNEMO 的统一数据 workflow，并保持 MJD 的目录、配置和入口风格。支持两类任务：

- `classification`：只比较原始类别 `2nubb` 与 `Bi214`。`2nubb` 规范化为物理语义 `2nu`，作为 signal，数值标签为 `1`；`Bi214` 作为 background，数值标签为 `0`。其他类别会被统计并明确排除，不会被归入任一类别。
- `energy`：回归目标为逐事件 `E1 + E2`，单位保持为 `keV`；不做单位变换、对数变换或裁剪。

active registry 包含十个精确、区分大小写的实验 ID：

- `cnn_004_multiview_late_fusion`
- `gnn_001_static_gine`
- `seq_001_bigru`
- `ssm_001_pointmamba`
- `transformer_001_sampled_hits_coordinate_mlp`（仅 classification）
- `transformer_002_voxel_coordinate_mlp`（仅 classification）
- `transformer_003_voxel_fourier_xyz`（仅 classification）
- `transformer_004_sampled_hits_fourier_xyz`（仅 classification）
- `transformer_005_summary_features_coordinate_mlp`（仅 classification）
- `transformer_006_summary_features_fourier_xyz`（仅 classification）

六个 Transformer 是组员方法的完整 `3 tokenizers × 2 positional encodings` 设计，而不是只比较两种位置编码。三种 tokenizer 为 `sampled_hits`、`voxel` 和 `summary_features`；每一种分别配 learned coordinate MLP 与 6-frequency Fourier XYZ。token content 先经 content MLP，位置编码后相加并进入 2 层 pre-norm Transformer encoder（`d_model=64`、4 heads、FFN 256），最后用 mask-aware mean pooling 输出 raw logit。4-feature 模型的参数量为 111,233（coordinate）或 113,537（Fourier），6-feature summary 模型分别为 111,361 和 113,665。

这里按 SuperNEMO tracker 观测量对组员 NEXT tokenizer 做了有记录的适配。原 NEXT 数据中的逐 hit energy 在 SuperNEMO release 中不存在；`E1/E2` 是重复在每个 tracker row 上的事件级 calorimeter 量，不能冒充逐 hit energy。因此三种表示只使用 `tX/tY/tZ/tR`：

- `sampled_hits`：先按 `(x,y,z,tR_valid,tR)` 对 hits 做 canonical lexicographic 排序，再从最多 128 个位置按 `(seed, local ev_no)` 确定性抽取；sampling key 不含 `source_key`、类别或标签，原始 HDF5 行顺序也不影响结果；feature 为 `[1/N, log1p(N), tR/(24 mm), tR_valid]`。
- `voxel`：中心化坐标上的 60 mm 三维 voxel，最多保留 128 个、按 occupancy 稳定截断；坐标是真实 hits 的 centroid，不是 voxel cell center；feature 为 `[count/N, log1p(count), mean_valid_tR/(24 mm), valid_fraction]`。
- `summary_features`：10-bit Morton 空间排序后均衡分为最多 16 组，所有 hits 都进入某一组；feature 为 `[count/N, log1p(count), 1/N, spatial_RMS/(1000 mm), mean_valid_tR/(24 mm), valid_fraction]`。

三者都先按完整事件的 raw-hit 均值中心化坐标，再除以 `1000 mm`。`tR=NaN` 不会被伪造成真实零半径：数值槽填零的同时有显式 validity feature；无穷值和负的有限半径会直接拒绝。抽样和 voxel 截断产生的 hit coverage 会汇总为每次 train/validation/test 的 mean、minimum、truncated-event count/fraction 并写入原生指标；coverage 不作为模型输入、loss 权重或 EnergyBench canonical column。summary coverage 恒为 1。

旧目录 `outputs/classification/trf_001_tracker_coordinate_mlp` 和 `outputs/classification/trf_002_tracker_fourier_xyz` 是第一轮单一 point adapter 的 legacy 产物。它们不属于上述六个 active ID，不可作为 CLI alias，也不能和新 tokenizer checkpoint 混用；目录仅保留用于追溯旧训练。

## 数据和防泄漏约束

原始数据位于只读目录 `/home/klz/Data/zeronu_benchmark/SuperNEMO`。模型输入只从事件拓扑构造；原始类别名称、数值标签、直接编码类别的元数据、文件或目录名称，以及 `E1`、`E2` 和由其得到的 energy target 都不会进入模型输入。release 中实验不可直接测量的 simulation-truth `theta/phiS` 也不作为输入。

所有任务和模型共用确定性的 split manifest：训练集、验证集和测试集比例约为 `80/10/10`，随机种子为 `42`。全局事件身份为 `(source_key, ev_no)`；原始 release 没有 run 或 simulation-batch 字段，因此完整事件是能验证的防泄漏分组单位。每个物理类别独立按 4096-event locality blocks 分层划分，同一事件不会跨 split；classification 再从同一基础 manifest 精确过滤 `2nu/Bi214`。不做重采样、class weight 或数据增强。

训练读取使用 `proportional_without_replacement`：先将 manifest 中因连续同 split 而合并的区间无损拆回最多 4096-event locality chunks，在每个来源内按 seed/epoch 确定性重排，再按各来源的原始事件数比例逐事件交织。每个事件每 epoch 恰好出现一次，类别和来源计数不变；validation/test 保持固定顺序。

DataLoader 使用与 MJD 相同的两批 FIFO 后台线程预取，batch 内容与顺序不变。训练循环在设备端累计 loss/metric 数值，并把 prediction、loss 和 gradient 的逐 batch 有限性检查合并为 optimizer step 前的一次同步；这不改变 batch size、优化器更新次数或模型输入。

第一次加载会在 `data/manifests/` 生成一份共享 JSON manifest 和四个只含事件行边界的 mmap 索引。不会复制 HDF5、缓存模型表示或为不同模型重新划分。当前实际计数为：`2nu=3,284,116`、`Bi214=2,634,002`；classification 排除 `0nubb=1,987,943` 与 `Tl208=2,549,143`。manifest 会审计 `tR` 的 NaN 与受影响事件；Transformer 通过数值槽加 validity feature 使用它，不丢事件也不静默插补为有效测量。

CNN adapter 生成 binary-occupancy XY/XZ/YZ `[B,3,128,128]`；原有 GNN、SEQ、SSM 保持 2-feature point adapter；Transformer 按各自冻结 tokenizer 返回 `coords[B,N,3]`、`features[B,N,4或6]`、`mask[B,N]`。collator 只对当前 batch 的最大 token 数做 padding。energy 采用参考 workflow 的 topology-only regression 语义，不伪造逐 hit 能量，也不把 target 广播回输入；六个 Transformer 只支持 classification。

## 环境

从项目目录运行，并使用 summer 虚拟环境。设置 `PYTHONDONTWRITEBYTECODE=1` 可避免校验或运行时生成 Python bytecode 文件：

```bash
cd /home/wenyu/SuperNEMO
export PYTHONDONTWRITEBYTECODE=1
/home/wenyu/summer/.venv/bin/python -m supernemobench.workflow --help
```

项目要求 Python `>=3.11`。训练依赖 `h5py`、`numpy` 和 `torch`；严格评测直接复用 summer 环境中同一份 `energybench`，不在本项目复制 evaluator。

## 统一入口

CLI 形式为：

```bash
/home/wenyu/summer/.venv/bin/python -m supernemobench.workflow \
  --task {classification,energy} \
  --model <MODEL_ID> \
  --mode {train,validate,test}
```

其中 `<MODEL_ID>` 必须是上列十个 active ID 之一。Transformer 只用于当前冻结的 `2nu vs Bi214` classification。例如：

```bash
/home/wenyu/summer/.venv/bin/python -m supernemobench.workflow --task classification --model transformer_001_sampled_hits_coordinate_mlp --mode train
/home/wenyu/summer/.venv/bin/python -m supernemobench.workflow --task classification --model transformer_001_sampled_hits_coordinate_mlp --mode test
```

训练目录非空时不会静默覆盖。中断后使用完全相同的参数并显式加 `--resume`，workflow 会从每个 epoch 原子提交的 `last.pt` 恢复 model/optimizer/scheduler/scaler、early-stopping 状态和随机数状态：

```bash
/home/wenyu/summer/.venv/bin/python -m supernemobench.workflow \
  --task classification --model transformer_001_sampled_hits_coordinate_mlp \
  --mode train --resume
```

checkpoint 由 validation energy-matched AUC 选择；test 不参与选择。`--mode test` 导出 canonical `test_evaluation/test_predictions.npz`（`event_id,label,category,score,energy_condition,group_id,split` 等），其中 `score` 是 raw logit，`energy_condition` 是 float64 `E1+E2 keV`，不会在 3000 keV 截断；随后用 `evaluation/supernemo_2nu_vs_bi214.json` 自动执行 `strict=True` EnergyBench。native summary 与完整报告分别位于 `test_evaluation/test_metrics.json` 和 `test_evaluation/energybench/`；整套结果一次原子提交，已有目录时拒绝覆盖。

checkpoint 同时绑定 split manifest、完整 DataConfig、冻结 tokenizer 配置及 tokenizer 源码、data adapter 源码、本地 evaluation adapter 源码、Transformer 源码、训练源码、EnergyBench manifest 和 evaluator code fingerprint；任一 tokenization、输入语义或协议发生变化都会在 resume/test 时拒绝混用。

## 只读参考

SuperNEMO 的字段与 energy 语义来自原始 HDF5 及随附 release 文档，通用事件级 `80/10/10` manifest 和 topology-only regression 语义参考最早的 `summer/evalutaions_workflow`。原四个模型来自 `summer/src/next_alt` 的最小本地副本；六个 Transformer 的三种 tokenization、两种位置编码、核心层级和超参数对齐组员 `Transformer_Approach/next_detector/next_transformer` 的方法，并明确记录缺少逐 hit energy 时的 SuperNEMO feature 替代。运行时只使用本地 registry；参考目录和原始数据目录保持只读，代码、manifest 和输出只写入 `/home/wenyu/SuperNEMO`。
