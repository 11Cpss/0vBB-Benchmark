# MJD / EXO-200 评价执行报告

本次最终范围为23项：MJD四个classic与六个MLP/Fourier Transformer，EXO-200四个classic与九个Transformer。均使用已保存的逐事件预测，未重新训练、推理或更换checkpoint/test set。NEXT、SuperNEMO及其独立energy-only示例保留此前的结果、图和v2配置；MJD三个RoPE按用户要求保留历史值。本报告不表示整篇论文所有记录已统一到同一协议。

## 协议与来源

最终配置为`EnergyBench-unified-5keV-range-v3.0.0`：物理能量先转为keV，使用0–3000 keV的600个固定5 keV箱，3000归入最后一箱；范围外事件排除，不clip、不设overflow。范围筛选先于support估计。I保留20个score-quantile箱、每组/能量箱至少20事件、JS先组内平均再开根、范围内稀疏筛选前的组权重。匹配使用0.005支持裁剪、overlap target、类内权重归一化、pooled weighted AUC与ties半分，至少两箱且每类原始finite人口覆盖率不低于50%。Inclusive AUC保持原始人群。

- 配置指纹：`6d696d87f1307b3f6d27bb9ef7ec443e1d8d98b9d95e90713f553c70edb42953`。
- 共享代码SHA256：`a243160b0a529a26a38dc6be14baa6f17a09618fea57c5a80de21822369e06b0`。
- EXO-200的正类为原始标签0、单charge cluster，分数为原background logit取负。140383个测试事件的ID、标签、能量、run、split和权重与原HDF5元数据及classic测试人口逐项核对。
- MJD正类为四PSD标记全通过。六个Transformer输出单一clean binary logit；四输出聚合仅用于classic GINE。390000个事件按实际loader顺序、逐事件标签与能量核对后恢复原始float64 `energy_label`，不直接假定与classic同序。恢复后物理ID及元数据也与classic完全一致。

15个Transformer的历史I、matched AUC和inclusive AUC均已从逐事件输入精确复现，差值为零。EXO历史I为每类8个能量分位数箱，而matching独立采用6个箱；MJD六个Transformer历史使用延伸到4180 keV的836个固定箱，没有采用classic旧adapter的clip。所有原始来源、checkpoint/config、输入及代码哈希见各数据集manifest。

## 结果与变化

23项均满足报告门槛，没有因固定细分箱而不可估计的项。所有inclusive AUC与历史值一致至浮点累加误差，最大差值为`1.11e-16`。逐事件输入与测试人口保持不变。

| 数据集 | 原始事件数 | 范围外排除：类0 / 类1 | 匹配有效箱 | 原始人口覆盖：类0 / 类1 | 匹配ESS：类0 / 类1 |
|---|---:|---:|---:|---:|---:|
| EXO-200 | 140383 | 1423 / 3 | 426 | 94.1403% / 94.5657% | 61626.88 / 51923.64 |
| MJD | 390000 | 66 / 1 | 493 | 96.3082% / 98.7400% | 213946.76 / 141205.10 |

这些损失均来自大于3000 keV的事件；没有负值或缺失能量。范围损失、之后的support损失与稀疏箱损失独立记录。EXO的I有效箱为类0/1的484/438；MJD为524/496。类覆盖率分母保留范围筛选前的finite事件数，未因范围筛选而抬高。

EXO九个Transformer相对其原始历史协议，I下降0.061013–0.089057，matched AUC下降0.000260–0.000621。主要原因是取消粗能量分位数箱，并统一稀疏资格和pooled histogram人口；严格范围筛选还改变支持分位数。两类I均下降，不是总体平均掩盖某一类改善。Region+Fourier仍有最高matched AUC（0.972261），Summary+Fourier仍有最高Transformer I（0.801902）。Region+Fourier的I现在也超过Region+RoPE，后者退出这九个Transformer的Pareto集合。

MJD六个Transformer相对历史协议，I增加`1.97e-5`至`4.72e-5`，matched AUC变化为`-1.38e-6`至`-4.94e-7`，四位小数展示及模型排序/Pareto关系均不变。变化包括：去掉范围外事件、恢复float64后有三个事件跨箱、修复一个2385 keV事件的旧MeV浮点归箱、重新估计范围筛选后的support。旧I还曾包含[3195,3200) keV箱的20个非clean事件。I最高为Entity+MLP（0.811615），matched AUC最高为Region+Fourier（0.981046）。

Classic结果需区分对照基准：`old_I/old_matched_auc`保存最初历史协议值；`previous_protocol_metrics`及`delta_from_v2_*`另记本轮开始前论文中的v2结果。相对本轮开始前工作稿的v2值，EXO四个classic的I增加0.000204–0.001833，matched AUC减少`4.23e-7`至`1.13e-5`；MJD四个classic的I增加`4.28e-5`至`1.19e-4`，matched AUC减少`3.72e-7`至`1.07e-6`。这些是评价人群/支持的变化，不是模型能力发生变化。所有比较均为单次运行点估计，不作统计显著性判断。

## 文件与复现

最终23项的完整精度及对照表：

- `results/selected_results_full_precision.json`（仅本轮23项）、`results/new_old_comparison.csv`。`results/results_full_precision.json`另保留整张论文的77项混合协议记录，供无损重建。
- `results/class_coverage_losses.csv`和`results/group_independence.csv`各46行，逐类展开23项。
- `results/input_manifest.json`、各模型`results/detail/`及其链接的数据集manifest记录输入和原始来源。
- `audit/numerical_tests.log`：14项数值测试全部通过，包括所有601个有限边界、单位等价、范围处理、ties、常数/稀疏/缺失情况和独立小样本AUC枚举。
- `../transformer_exo_mjd_20260917/audit/independent_numeric_active23.json`：独立审查23/23通过，使用熵形式JS及ROC梯形积分，未调用共享指标函数；同时核对support、coverage、ESS和事件来源。

在论文仓库根目录复现一个标准NPZ的严格v3评价：

```bash
.venv-energybench/bin/python wing_contribution/evaluation/evaluate_npz.py \
  /path/to/standardized_predictions.npz --output /tmp/model_metrics.json
```

全部23项从已核验的逐事件输入重算并生成表格：

```bash
bash "/home/wenyu/iclr final paper/unified_5kev_evaluation/strict_600bin_20260917/reproduce.sh"
```

仅重算15个Transformer：

```bash
/home/wenyu/summer/.venv/bin/python -B '/home/wenyu/iclr final paper/unified_5kev_evaluation/transformer_exo_mjd_20260917/exo/recompute_strict600.py'
/home/wenyu/summer/.venv/bin/python -B '/home/wenyu/iclr final paper/unified_5kev_evaluation/transformer_exo_mjd_20260917/mjd/recompute_strict600.py'
```

论文的可移植README提供环境安装及表图重建命令。重建从已核验汇总数据生成展示，不把汇总值当成重算输入；混合结果注册表逐条保留v3/v2/历史状态。

## 限制与收尾状态

原始预测和大型HDF5/checkpoint不随论文仓库分发，需要另备；多GB原始HDF5未整体计算hash，但使用的事件ID、能量、标签等标量输入已精确重建/核对并记录来源。MJD预测缓存本身未存event ID，身份通过实际loader与完整标签/能量数组恢复；此限制在来源审计中明确保留。MJD RoPE没有在本次重算，历史数值不能作为严格v3结果参与本轮排名。

严格v3数学实现及23项独立审查全部通过。可移植重建成功，六张生成表与论文源文件逐字一致；NEXT/SuperNEMO原图保持不变。论文已编译为25页PDF，主表与诊断表经过视觉检查，无溢出、重叠或裁切；没有未解析引用或Overfull警告。编译保留既有字体small-caps替代及Underfull提示。凭据见`audit/final_validation.json`、`audit/compiled_paper.log`和`portable_render/rebuild_receipt.json`。本目录残留的范围扩展阶段输出属于中间审计；最终交付范围仅以上23项。

最终独立稿件审查32项、PDF审查9项全部通过；记录见`audit/final_independent_review.md`、`audit/final_paper_review.json`与`audit/final_pdf_review.json`。全文审查提出的主表协议说明已在最后图注中补充。
