# CLASSIC-001：NEXT 拓扑特征 + XGBoost

[中文](README.md) | [English](README_EN.md)

## 1. 模型身份与假设

| 项目 | 定义 |
|---|---|
| `architecture_id` | `classic_001_topology_xgboost` |
| model name | `TopologyBoostedTreeClassifier` |
| feature class | `next_alt.models.classic_topology.TopologyFeatureExtractor` |
| estimator class | `next_alt.models.classic_topology.TopologyBoostedTreeClassifier` |
| `input_kind` | `topology`（物化时使用与 `points` 相同的 batch tensor） |
| 任务/输出 | `0nubb=1`、`Bi214=0`；每事件一个未校准 logit |
| 正式 backend | **`xgboost`**（配置锁定；环境版本 3.0.5） |
| 参数量 | 树模型不使用固定神经参数量；训练后记录 tree count/node count |

假设是 NEXT 物理启发的 endpoint blob、轨迹长度/弯曲度、连通性和能量分布等低维特征
已能提供强分类基线，并可检查深度模型增益是否来自真正的表示学习。模型不使用
Transformer、attention、总事件能量或绝对 detector position。

## 2. 原始数据、标签和 split

事件从 HDF5 `/MC/hits/table` 读取连续相同 `event_id` 的 `x/y/z/energy`。
`0nubb_part_*` 映射为 1，`Bi_part_*` 为 0。完整相对文件路径按 seed 42 的稳定
file-level hash 分到 train/validation/test，比例 `[0.8,0.1,0.1]`。本脚本只调用
`build_training_loaders` 并物化 train/validation；各 split 每类最多 100 文件。validation
AUC 用于 early stopping 与 best tree limit，任何代码路径都不请求 test split。

## 3. voxel 输入与公式

共享点表示先计算：

$$
E=\sum_i e_i,\quad \mathbf c=\sum_i e_i\mathbf r_i/E,\quad
\mathbf q_i=\lfloor(\mathbf r_i-\mathbf c)/(15\,\mathrm{mm})\rfloor.
$$

同 cell 合并，center `(q+0.5)*15 mm` 再做 energy-weighted recenter。最多确定性保留
512 个高能 voxel，energy fraction 不重归一化。tensor 为：

| key | dtype / shape | 语义 |
|---|---|---|
| `coords` | float32 `(B,N,3)` | centered voxel center / 1000 mm |
| `features` | float32 `(B,N,2)` | `[voxel_energy/E, log1p(hit_count)]` |
| `mask` | bool `(B,N)` | padding mask |
| `label` | float32 `(B,)` | 0/1 target，只给 trainer |
| topology matrix | float32 `(events,32)` | 固定 feature vector |
| prediction | float64 `(events,)` | XGBoost output-margin logit |

feature extractor 为控制 (O(N^2)) 成本再取最多 192 个高能点，并重新计算该 subset 的
energy-weighted center。总能量 (E) 只作为 fraction 分母，数值本身不进入矩阵。

## 4. 32 个拓扑特征

| 组 | 特征（按 checkpoint 顺序） | 定义摘要 |
|---|---|---|
| size/energy | `log_num_voxels` | `log1p(M)` |
|  | `topology_retained_energy_fraction` | top-192 voxel fraction 之和 |
|  | `max_voxel_energy_fraction`, `top2_energy_fraction`, `top5_energy_fraction` | 最大/前 2/前 5 energy fractions |
|  | `normalized_energy_entropy` | `-Σw log(w)/log(M)`，subset 内归一化 (w) |
| hit count | `mean/std/max_log_hit_count` | `log1p(merged_row_count)` 的统计量 |
| axis extent | `extent_x/y/z` | centered subset 各轴 peak-to-peak |
| PCA | `rms_major/middle/minor` | energy-weighted covariance eigenvalue 的平方根 |
| PCA ratios | `linearity`, `planarity`, `sphericity` | `(λ1-λ2)/λ1`, `(λ2-λ3)/λ1`, `λ3/λ1` |
| radius | `radial_mean/std/max` | 相对 energy-weighted center 的半径统计 |
| endpoint | `principal_length` | 第一主轴投影 max-min |
| blob | `endpoint_blob_low/high/min` | 主轴两端 30 mm 球内 energy fraction sum |
| blob | `endpoint_blob_asymmetry` | `abs(high-low)/(high+low+eps)` |
| radius graph | `radius_graph_components` | 26 mm 邻接图 connected components |
|  | `radius_graph_mean_degree` | 26 mm 图 mean degree |
| MST | `radius_graph_branch_fraction` | Euclidean MST 中 degree≥3 的节点比例 |
|  | `mst_total_length`, `mst_max_edge` | complete-graph MST edge sum/max |
|  | `mst_tortuosity` | MST total / principal length |

PCA eigenvalues 记为 `lambda_1 >= lambda_2 >= lambda_3 >= 0`。MST 用 Prim 精确计算；
26 mm 约为 15 mm voxel 的 1.73 倍。第一主轴的最大绝对 Cartesian 分量固定为正，
以消除 eigenvector sign ambiguity；所有长度均为除以 1000 mm 后的 scaled units。

## 5. XGBoost 结构与 objective

正式 backend 调用 XGBoost `train` 的 histogram tree method：

$$
\hat y(x)=b+\eta\sum_{t=1}^{T}f_t(x),\qquad
\mathcal L=\sum_i\mathrm{BCEWithLogits}(y_i,\hat y_i)+\Omega(f_t).
$$

每轮由 logistic gradient/Hessian 建一棵 max-depth 4 tree；validation eval metrics 为
`logloss` 与 `auc`，连续 12 rounds 无 AUC 改善停止。`best.json` 存同一 Booster raw UBJ
及 `prediction_tree_limit=best_iteration+1`；`last.json` 的 limit 为完成轮数。输出 margin
而不是概率，以保持与神经模型 `(B,)` logit contract 一致。

### 纯 NumPy备用边界

模块包含 `numpy_hist_gbdt`：quantile candidate splits + Newton leaf weights 的小型纯 NumPy
fallback。它没有 XGBoost 的 sparsity-aware algorithm、weighted quantile sketch、并行 kernel
或完整正则化，**不是 XGBoost 复现**。正式 YAML 显式设 `backend: xgboost`；若依赖缺失会
直接失败而非悄悄切换。只有用户把 config 改为 `numpy_hist_gbdt` 时才使用 fallback，且
checkpoint/run summary 会记录真实 backend。

## 6. 参数、复杂度和内存

feature extraction 对每事件最多 (M=192) 点，pairwise distance/MST 为 (O(M^2)) time
和 memory。XGBoost histogram boosting 近似随 `T * depth * rows * selected_features` 增长；
输入矩阵约 `(train_events,32)`，内存远小于 dense images。树节点数由数据与 early stopping
决定，因此训练后写入 `run_summary.json.tree_node_count`；`parameter_count=null` 是预期。
正式 `tree_method=hist` 使用 CPU（未设置 XGBoost `device=cuda`），但仍在统一 tmux queue
中串行执行以保持 campaign 次序。

## 7. 完整 YAML 配置

| 类别 | 参数 | 值 |
|---|---|---:|
| data | root / max files | `/home/klz/Data/zeronu_benchmark/NEXT` / 100 每类 |
| data | split seed/fractions | 42 / `[0.8,0.1,0.1]` |
| data | workers/balanced/buffer | 0 / true / 512 |
| representation | bin/scale/max points | 15 mm / 1000 mm / 512 |
| extractor | max topology points | 192 |
| extractor | connectivity/blob radius | 0.026 / 0.030 scaled units |
| estimator | backend | `xgboost` |
| estimator | rounds/depth/lr/bins | 500 / 4 / 0.04 / 32 |
| estimator | min child setting | `min_samples_leaf=24` 映射到 XGBoost `min_child_weight=24` |
| estimator | L2/gamma | 1.0 / 0.0 |
| estimator | row/column sample | 0.85 / 0.90 |
| estimator | seed/early stop | 42 / 12 |
| loader | batch size | 128（只控制 feature materialization） |

YAML 的 `training.epochs=50`、`learning_rate=0.001`、weight decay、clip、AMP 等字段用于
共享 config schema 兼容，**不控制树学习**；真正 tree 参数完全来自 `model.estimator`。

## 8. 与文献方法的差异

- NEXT 拓扑论文强调 track reconstruction 与 endpoint energy；本实现使用 voxel PCA、
  radius graph 和 complete-graph MST 的 32 维工程特征，不是论文 reconstruction chain。
- blob 是主轴极端点 30 mm 球，而不是论文的完整 track-end finding；且使用 energy fraction，
  不使用绝对 blob energy。
- 正式 learner 是 XGBoost histogram backend，但 dataset、特征、超参数与 Chen/Guestrin
  论文 benchmarks 不同；不声称复现论文性能。
- top-192 与 top-512 两级截断、axis extents 和 fixed radii 是本项目选择。

## 9. 运行、产物与恢复

```bash
cd /home/wenyu/summer
source .venv/bin/activate
python 01_code/architectures/classic_001_topology_xgboost/train_classification.py CONFIG_SNAPSHOT
```

detached session `next-nontransformer-v2-<RUN_ID>` 中由 `gpu-queue` 串行执行；monitor 只读。
attempt 产物严格为：

```text
02_models/checkpoints/<RUN_ID>/classic_001_topology_xgboost/attempt_NNN/
  best.json  last.json
03_training_runs/campaigns/<RUN_ID>/classic_001_topology_xgboost/attempt_NNN/
  stdout.log  config.snapshot.yaml  epochs.csv  history.json  history.png  run_summary.json
```

JSON checkpoint 含 backend、32 feature names、extractor config、split/file provenance 和
base64 UBJ Booster。`C-c` 可停止窗口；`--resume-queue` 跳过 DONE，FAILED 创建新 attempt
并从头物化/训练，不做 checkpoint resume，也不覆盖旧 attempt。

## 10. 已知限制与训练结果占位

- PCA axis extents 不是旋转不变；fixed 26/30 mm radii 依赖 15 mm voxelization；
- top-192 可能丢失长的低能 track tail，MST 对 outlier 敏感；
- XGBoost 的 `min_child_weight` 是 Hessian sum，不等同名称 `min_samples_leaf` 的样本数；
- JSON 内嵌 Booster 较大，只应加载可信 campaign artifact；
- 本阶段不读取 test，不产生 inference table、test metric 或 leaderboard。

本段训练前占位说明已由文末追加的 campaign 结果取代。结果包含 backend/version、轮数/best round、best validation AUC/loss、
总耗时、tree/node count、artifact path、early-stop 与失败重试；不得出现 test 指标。

## 11. 参考文献

1. Tianqi Chen, Carlos Guestrin, “XGBoost: A Scalable Tree Boosting System,” *Proceedings of
   the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*,
   2016, pp. 785–794, DOI `10.1145/2939672.2939785`, arXiv:1603.02754.
   [DOI](https://doi.org/10.1145/2939672.2939785) · [arXiv](https://arxiv.org/abs/1603.02754)
2. NEXT Collaboration, P. Ferrario et al., “First proof of topological signature in the high
   pressure xenon gas TPC with electroluminescence amplification for the NEXT experiment,”
   *JHEP* 2016, 104, DOI `10.1007/JHEP01(2016)104`, arXiv:1507.05902.
   [Journal](https://link.springer.com/article/10.1007/JHEP01%282016%29104) ·
   [arXiv](https://arxiv.org/abs/1507.05902)


<!-- campaign-result:20260803_200356:start -->
## Campaign `20260803_200356` 训练结果

本节由串行训练队列在训练成功后写入，只包含 train/validation 信息；本阶段没有读取 test split。

| 项目 | 实际值 |
|---|---|
| 状态 / attempt | `DONE` / `attempt_001` |
| 后端 | `xgboost` |
| 实际参数量 | N/A |
| 树数量 / 树节点数 | 491 / 9769 |
| 完成 epoch / best epoch | 500 / 491 |
| best validation AUC | **0.948048** |
| best validation loss | 0.289180 |
| 总训练时间 | 00:01:23 |
| early stop | `false` |
| 失败重试 | `no` |
| Python / 框架 | `3.11.15` / `3.0.5` |
| 设备 | `not used (XGBoost hist on CPU)` |
| best / last checkpoint | `/home/wenyu/summer/02_models/checkpoints/20260803_200356/classic_001_topology_xgboost/attempt_001/best.json` / `/home/wenyu/summer/02_models/checkpoints/20260803_200356/classic_001_topology_xgboost/attempt_001/last.json` |
| 训练日志 | `/home/wenyu/summer/03_training_runs/campaigns/20260803_200356/classic_001_topology_xgboost/attempt_001/history.json` |
<!-- campaign-result:20260803_200356:end -->
