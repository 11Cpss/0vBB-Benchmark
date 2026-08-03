# EnergyBench 评测标准、表格规范与输出解释

[English version](EVALUATION_STANDARD_EN.md)

> 文档版本：1.0  
> 对应实现：EnergyBench 0.1.0  
> 审计基线：2026-08-01 的 `/home/wenyu/summer` 工作区

## 摘要

本标准规定 EnergyBench 对逐事件模型输出进行分类、能量回归和分类分数—能量依赖评测时使用的数据契约、统计量、质量门槛、缺失值语义、表格字段和图像解释。正式分类主指标是完整固定类别对集合上的 **energy-matched AUC 宏平均**；正式能量回归主指标是项目内版本化复合分数 **ERS-v1**。Inclusive AUC、能量独立性分数和其余误差或分布指标均是必要诊断，但不替代主指标，也不合成为一个总分。

评测器只接收逐事件 prediction table，不在统计阶段读取模型结构。正式比较要求相同事件真值、相同协议、相同评测代码、严格模式成功且模型标识唯一。无法满足能量共同支撑、统计量、覆盖率或完整类别对条件时，matched AUC 必须记为 NA；不得用 inclusive AUC 回填。

## 1. 适用范围与指标层级

本标准适用于二分类或多类别拆成固定 signal/background pair 的排序评测，以及标量能量回归。当前 NEXT 任务中，`0nubb` 为信号、`Bi214` 为背景，能量单位为 MeV；分类匹配能量 `energy_condition` 与回归真值 `energy_target` 数值可以相同，但统计角色和 provenance 必须分别声明。

### 1.1 记号

| 记号 | 定义 |
|---|---|
| $i=1,\ldots,n$ | 事件索引 |
| $Y_i\in\{0,1\}$ | 背景/信号标签，1 为正类 |
| $C_i$ | 物理类别，例如 `0nubb`、`Bi214` |
| $s_i$ | 模型原始分类分数 |
| $S_i$ | 定向后的分数；越大越像信号 |
| $E_i$ | 分类匹配条件 `energy_condition` |
| $T_i$ | 回归目标 `energy_target` |
| $\widehat T_i$ | 回归预测 `energy_pred` |
| $a_i\ge0$ | manifest 指定的 base sample weight；缺省为 1 |
| $\Delta_i=\widehat T_i-T_i$ | 有符号残差 |
| $r_i=\Delta_i/\max(|T_i|,\epsilon)$ | 分数残差，$\epsilon$ 为冻结的能量下限 |

若 `score_direction=higher`，$S_i=s_i$；若为 `lower`，$S_i=-s_i$。阈值、ROC 和能量依赖计算均使用 $S_i$。分数方向必须事先声明，不能根据测试结果临时翻转。

### 1.2 主指标与诊断指标

| 层级 | 指标 | 方向 | 用途 |
|---|---|---:|---|
| 分类主指标 | `matched_auc_macro` | 越高越好 | 完整固定 pair 集在匹配能谱后的排序能力 |
| 回归主指标 | `energy_regression_score` / ERS-v1 | 越高越好 | 同时约束逐事件误差、总体能谱和有限预测率 |
| 分类诊断 | inclusive/common-support/diagnostic matched AUC | 越高越好 | 区分原始性能、共同支撑性能和 policy gate 前性能 |
| 回归诊断 | MAE、RMSE、bias、$R^2$、JSD、$W_1$ 等 | 依定义 | 解释 ERS-v1 的误差来源 |
| 去相关诊断 | independence、相关系数、acceptance flatness、sculpting | 依定义 | 检查分类分数对能量的类内依赖 |
| 质量诊断 | coverage、ESS、状态、warnings/errors | coverage/ESS 越高通常越稳健 | 判断主指标是否有足够适用范围和统计量 |

分类与回归分别排名，不设置综合 rank。高 independence 不能单独证明分类器优秀：常数分数也可能与能量独立。

## 2. 逐事件 canonical prediction table 标准

所有列必须具有相同的第一维事件数，且每列至少为一维。推荐容器是压缩 NPZ；CSV、HDF5 和 Parquet 可作为输入，但正式复现优先使用能保存元数据的 NPZ。NPZ 中普通键是逐事件列，`__metadata__` 是一个标量 JSON 字符串。

| canonical 列 | 推荐类型 | 必需条件 | 精确定义 |
|---|---|---|---|
| `event_id` | string | strict 必需 | 全局稳定且唯一的事件标识；NEXT 使用 `NEXT::<相对HDF5路径>::<文件内事件号>` |
| `label` | integer/string | 分类需 `label` 或合法类别角色 | 正类标签由 manifest 的 `positive_label` 冻结 |
| `category` | string | 使用显式 signal/background categories 时必需 | 物理过程类别；显式类别角色优先于普通 label |
| `score` | float | 分类必需 | 对 manifest 正类的分数；NEXT adapter 导出原始 logit |
| `energy_condition` | float | energy-matched 分类和 dependence 必需 | 仅用于能量匹配/条件诊断的协变量 |
| `energy_target` | float | 回归必需 | 逐事件回归参考真值；映射到内部角色 `energy_true` |
| `energy_pred` | float | 回归必需 | 与 `energy_target` 对齐、同单位的物理空间预测 |
| `sample_weight` | float | 可选 | 有限、非负 base weight；至少一个事件权重大于 0 |
| `split` | string | strict 必需 | 事件所属 split，必须与 manifest 的 evaluation split 一致 |
| `group_id` | string | 强烈建议 | 共享文件/run/campaign 的相关组；进入 fingerprint，但当前不用于 group bootstrap |
| `projection_coverage` | float | NEXT 辅助列 | 投影坐标范围内保留的 voxel 能量/事件 voxel 总能量；不直接进入 EnergyBench 评分 |

NEXT 的 `energy_condition` 与 `energy_target` 都是同一事件 `/MC/hits/table` voxel energy 的 float64 求和。CNN-001 分类输入按事件总能量归一化，而配套 CNN-001 回归输入和 v2 历史输入保留绝对能量幅度。`projection_coverage=1` 时，原始投影像素和与目标成正比，因此该回归是数据流基线，不是独立的实验能量重建证明。

### 2.1 NPZ 元数据

`__metadata__` 应至少保存 adapter、模型/checkpoint 标识与 SHA-256、数据根与 split、数据 inventory、分数空间、能量单位和推导、投影配置。v2 还应保存能量目标类型、来源、单位、训练集拟合的标准化参数和预测空间。元数据是 provenance，不代替 manifest 中冻结的统计语义。

实现依据：`src/energybench/data.py`、`src/next_cnn/adapter.py`、`src/next_cnn/data.py`。

## 3. Manifest 冻结规则

评测前必须冻结并在 `.energybench/resolved_manifest.json` 保存以下内容：

1. `task_id`、唯一 `model_id` 与 manifest schema version；
2. experiment、dataset ID/version、evaluation split、独立的 checkpoint-selection split、能量种类和单位；
3. 每个 canonical role 的显式列映射；
4. 正类、signal/background category 集合、pair mode、score direction/space；
5. 能量 ROI、support trim、bin 数、每类最低事件数、最低有效 bin 数、最低 coverage、matching target 和目标 TPR；
6. 回归 histogram/performance bins、显式 edges（若有）、fractional energy floor；
7. dependence binning、distance-correlation 样本上限；
8. bootstrap 次数、置信水平和随机种子。

Strict 模式要求显式事件 ID/split/任务列、数据 provenance、唯一且非空 event ID、匹配 split，以及 selection split 与 evaluation split 不同；概率分数还必须位于 $[0,1]$。评测器拒绝 `split=train`。

正式比较时，以下三种摘要必须一致：

- `evaluation_fingerprint`：对事件 ID、truth/category、匹配/目标能量、base weight、group 和 split 等排序后哈希；故意排除 `score` 和 `energy_pred`；
- `protocol_fingerprint`：对数据、分类、回归、dependence 与 seed 配置哈希；
- `code_fingerprint`：对 `src/energybench/*.py` 内容哈希。

实现依据：`src/energybench/config.py`、`src/energybench/evaluation.py`。

## 4. 分类评分：inclusive 与 energy-matched ROC

### 4.1 加权 inclusive ROC/AUC

对阈值 $\tau$，定义

$$
\operatorname{TPR}(\tau)=
\frac{\sum_i a_i\,\mathbb 1(Y_i=1,S_i\ge\tau)}
{\sum_i a_i\,\mathbb 1(Y_i=1)},
\qquad
\operatorname{FPR}(\tau)=
\frac{\sum_i a_i\,\mathbb 1(Y_i=0,S_i\ge\tau)}
{\sum_i a_i\,\mathbb 1(Y_i=0)}.
$$

阈值从 $+\infty$ 向下扫描；相同 score 的事件作为同一阈值组处理。AUC 使用经验 ROC 的梯形积分：

$$
\operatorname{AUC}_{\rm inclusive}
=\int_0^1 \operatorname{TPR}(u)\,du.
$$

Inclusive AUC 测量原始样本组成下的排序能力，可能同时包含拓扑信息和信号/背景能谱差异带来的捷径。它必须报告，但不是 matched AUC 的替代品。第一个达到 `target_tpr` 的经验点定义 operating point，并报告 threshold、实际 TPR、FPR 与背景拒绝率 $1-\mathrm{FPR}$。

### 4.2 共同支撑、分 bin 与匹配权重

对每一类先求能量支撑。若 `support_trim_quantile=\alpha>0`，使用各类 base-weighted 中央区间；否则使用样本极值。再与预先声明的物理 ROI 相交：

$$
L=\max\{Q_{1,\alpha},Q_{0,\alpha},E_{\rm ROI}^{\rm low}\},\qquad
U=\min\{Q_{1,1-\alpha},Q_{0,1-\alpha},E_{\rm ROI}^{\rm high}\}.
$$

若 $U<L$，matched estimand 无定义。共同支撑内的 bin edges 来自信号/背景各占一半总质量的 pooled weighted quantiles；若不同能量取值数不大于请求 bin 数，则用相邻离散能级中点分隔。

令第 $k$ 个 bin、类别 $c$ 的 base mass 和类内质量分数为

$$
A_{ck}=\sum_{i:Y_i=c,E_i\in B_k}a_i,
\qquad
p_{ck}=\frac{A_{ck}}{\sum_j A_{cj}}.
$$

只有两个类别的事件数均不小于 `min_per_class` 且 $A_{ck}>0$ 的 bin 才有效。默认 overlap target 为

$$
t_k=
\frac{\mathbb 1(k\text{ valid})\min(p_{1k},p_{0k})}
{\sum_j\mathbb 1(j\text{ valid})\min(p_{1j},p_{0j})}.
$$

`matching_target=uniform` 时，有效 bin 的 $t_k$ 相等。最终事件权重为

$$
w_i=a_i\frac{t_k}{A_{Y_i k}},\qquad E_i\in B_k.
$$

因此每个有效 bin 中两类的最终质量都等于 $t_k$，每类最终总质量为 1。Matched ROC 是对 $(Y_i,S_i,w_i)$ 计算的一条全局阈值 ROC；它不是逐事件精确同能量配对，也不是各 bin AUC 的平均。

### 4.3 正式 gates、宏平均与 NA

核心匹配算法先产生 `diagnostic_matched_auc`。正式 `matched_auc` 还必须通过：

- 两类 matched **base-mass coverage** 的较小值不低于 `min_coverage`；
- 有效 bin 数不低于 `min_valid_bins`；若共同支撑只有一个精确能级，则豁免第二项。

覆盖率的分母是有有限 score、正 base weight 且有有限能量的本类样本，分子是进入有效 matched bins 的事件的原始 base mass：

$$
C_c^{(a)}=
\frac{\sum_{i:Y_i=c,\,i\in\text{matched}}a_i}
{\sum_{i:Y_i=c,\,E_i\text{ finite}}a_i}.
$$

有显式 category pairs 时，正式宏平均对所有固定 category pair 做未加权算术平均；否则对 pooled pair 应用相同规则：

$$
\operatorname{AUC}_{\rm matched,macro}
=\frac1P\sum_{p=1}^{P}\operatorname{AUC}_{{\rm matched},p}.
$$

只有 $P$ 个 pair 全部可评价时该值才存在。`matched_auc_macro_available` 仅是当前可评价 pair 的诊断平均，不能用于正式排名。

下列情况必须输出 NA 和原因，不得以 0.5 或 inclusive AUC 填充：

| 条件 | pair 级状态 |
|---|---|
| 缺少信号或背景事件 | `not_evaluable_missing_class` |
| 缺少 `energy_condition` | `not_applicable_no_energy_condition` |
| 某类有限能量事件不足或无有效 bin | `not_evaluable_insufficient_statistics` |
| 两类无共同能量支撑 | `not_evaluable_no_common_support` |
| base-mass coverage 低于门槛 | `not_evaluable_low_coverage` |
| 有效 bin 数低于门槛 | `not_evaluable_too_few_valid_bins` |
| 固定 pair 集不完整 | 汇总状态 `not_evaluable_incomplete_pair_set` |

### 4.4 匹配诊断与不确定性

`coverage` 同时报告 total、with-energy、common-support、matched 的事件数及 count/base-weight fractions。加权有效样本量为

$$
\operatorname{ESS}(w)=\frac{(\sum_iw_i)^2}{\sum_iw_i^2}.
$$

`balance` 报告匹配前后的一维 Wasserstein-1、加权 KS、bin total variation 和匹配后最大 bin 质量差。`inclusive_common_support_auc` 使用整个共同支撑和 base weights；`shortcut_gap` 定义为

$$
\operatorname{AUC}_{\rm common\ support}
-\operatorname{AUC}_{\rm diagnostic\ matched}.
$$

该差值是两个 estimand 的诊断差，不表示“多少比例信息来自能量”。

分类 bootstrap 按类别分层做 event resampling，并在每个 replicate 中重新估计共同支撑、edges、有效 bins、target 和匹配权重。当前 NEXT 数据同一 HDF5 内事件可能相关，而实现没有 group bootstrap，因此 NEXT manifests 将 bootstrap 设为 0，不报告误导性的 event-level CI。

实现依据：`src/energybench/roc.py`、`src/energybench/evaluation.py`。

## 5. 能量回归：ERS-v1

ERS-v1 是本项目定义的版本化复合分数，不是实验合作组或机器学习领域的通用标准。只有目标物理定义、单位、事件集合、base weights、histogram edges、truth-energy bins 和能量 floor 全部一致时才可比较。

### 5.1 有限预测率与逐事件项

真值必须有限；非有限预测不被静默删除，而是降低加权有限预测率：

$$
f_{\rm finite}=
\frac{\sum_i a_i\mathbb 1(\widehat T_i\text{ finite})}
{\sum_i a_i}.
$$

若 manifest 未显式指定 $\epsilon$，实现取正的 $|T|$ 加权中位数的 $10^{-6}$，且下限为 $10^{-12}$。以 truth 的 base-weighted quantiles 建立 $K$ 个 performance bins。在每个含有限预测的非空 bin 中，

$$
m_k=
\frac{\sum_{i\in B_k}a_i|r_i|}{\sum_{i\in B_k}a_i},
\qquad
\operatorname{BFMAE}=\frac1{K'}\sum_{k\in\mathcal K_{\rm nonempty}}m_k,
$$

其中 $K'$ 是具有有限预测的非空 bin 数。事件项为

$$
S_{\rm event}=\max(0,1-\operatorname{BFMAE}).
$$

各非空 truth-energy bin 等权，防止样本密集区主导逐事件项；完全没有有限预测时该项为 0。

### 5.2 能谱项与总分

共享 histogram edges 默认取正权重 truth 范围上的等宽 bins。概率数组额外包含显式 underflow 和 overflow：truth histogram 用所有正权重真值，predicted histogram 只用有限预测并在其内部重新归一化；非有限预测另由 $f_{\rm finite}$ 惩罚。

对归一化概率 $p$ 与 $q$，$m=(p+q)/2$，以 2 为底的 Jensen–Shannon divergence 为

$$
\operatorname{JSD}_2(p,q)=
\frac12\sum_jp_j\log_2\frac{p_j}{m_j}
+\frac12\sum_jq_j\log_2\frac{q_j}{m_j}.
$$

定义

$$
S_{\rm hist}=1-\sqrt{\operatorname{JSD}_2(p,q)},
\qquad
\boxed{\operatorname{ERS\text{-}v1}
=f_{\rm finite}\sqrt{S_{\rm event}S_{\rm hist}}}.
$$

三因子均位于 $[0,1]$，所以 ERS-v1 也位于 $[0,1]$。几何平均要求逐事件准确和总体能谱相似同时成立；打乱预测可以保留能谱，却不能取得高 event score。

### 5.3 辅助回归指标

| 字段 | 定义与解释 | 方向/单位 |
|---|---|---|
| `mae` | $\sum a_i|\Delta_i|/\sum a_i$ | 越低越好，能量单位 |
| `rmse` | $\sqrt{\sum a_i\Delta_i^2/\sum a_i}$ | 越低越好，能量单位 |
| `bias` | 加权平均 $\Delta$ | 越接近 0 越好，能量单位 |
| `r2` | $1-\sum a_i\Delta_i^2/\sum a_i(T_i-\bar T_w)^2$ | 越高越好；可为负，常真值时为 NA |
| `mae_skill` | $1-\mathrm{MAE}/\mathrm{MAE}_{\text{weighted-median baseline}}$ | 越高越好；基线误差为 0 时 NA |
| `fractional_bias` | $r$ 的加权中位数 | 越接近 0 越好，无量纲 |
| `fractional_resolution_68` | $[Q_{0.84}(r)-Q_{0.16}(r)]/2$ | 越低越好，无量纲 |
| `balanced_fractional_mae` | 上述 truth-bin 等权绝对分数误差 | 越低越好，无量纲 |
| `jsd_bits` | truth/predicted flow histogram 的 $\operatorname{JSD}_2$ | 越低越好，bit |
| `histogram_overlap` | $\sum_j\min(p_j,q_j)$ | 越高越好，$[0,1]$ |
| `wasserstein_1` | truth/predicted 加权一维分布的 $W_1$ | 越低越好，能量单位 |

每个 truth-energy bin 还保存加权 truth 均值、事件数/权重、median response、median residual、绝对/分数 $\sigma_{68}$、MAE 和 RMSE。对正且远离 floor 的能量，`response=1+r=\widehat T/T`。

回归 bootstrap 是 paired event bootstrap；truth、prediction 和 weight 用同一事件索引重采样，每个 replicate 重算全部指标，但 histogram edges、truth-energy edges 与 fractional floor 固定。当前 NEXT manifests 同样将其设为 0。

实现依据：`src/energybench/regression.py`。

## 6. 分类分数的类内能量依赖

依赖诊断按真实 `category` 分组；没有 category 时使用 signal/background。每组过滤非有限 score/energy 和非正权重，并用 weighted energy quantiles 与共享 weighted score quantiles 建 bin。

### 6.1 相关与分布稳定性

每组报告：

- `pearson_abs`：加权 Pearson 相关绝对值，检测线性依赖；
- `spearman_abs`：加权 midrank Spearman 绝对值，检测单调依赖；
- `distance_correlation`：一般非线性依赖。精确计算为 $O(n^2)$；事件超过上限时按 base weight 概率无放回抽取确定性子样本，再在子样本上计算未加权经验 dCor；
- `conditional_score_jsd_nats`：各能量 bin 的 score 分布相对 pooled score 分布的加权平均 JSD。

令 $p_k(s)$ 是第 $k$ 个能量 bin 的 score histogram，$p(s)$ 是本组 pooled histogram，$\pi_k$ 是 bin 的 base-weight fraction，则

$$
J_E=\sum_k\pi_k\operatorname{JSD}_{\ln}(p_k,p),
\qquad
I_E=\operatorname{clip}\!\left(1-\sqrt{J_E/\ln2},0,1\right).
$$

`energy_independence_score=I_E` 越高表示 score 分布随能量变化越小。`overall_energy_independence_score` 是可评价组按组 base mass 加权平均，`worst_group_energy_independence_score` 是最小组分数。正式摘要仅在所有预期组均可评价时填入 mean/worst。

### 6.2 固定阈值接受率与能谱塑形

对阈值 $\tau$，

$$
A_k=\frac{\sum_{i\in B_k}a_i\mathbb1(S_i\ge\tau)}{\sum_{i\in B_k}a_i},
\qquad
A=\frac{\sum_i a_i\mathbb1(S_i\ge\tau)}{\sum_i a_i}.
$$

Flatness 指标为

$$
\operatorname{RMS}_A=\sqrt{\sum_k\pi_k(A_k-A)^2},
\qquad
D_{\max}=\max_k|A_k-A|.
$$

选择前后能谱的自然对数 JSD 记为 `energy_sculpting_jsd_nats`，其归一化距离为 $\sqrt{\mathrm{JSD}/\ln2}$。

当前端到端实现从**同一个 evaluation sample** 的 matched ROC 目标 TPR 点取得 $\tau$，并标记 `threshold_source=matched_roc_target_tpr`。因此 acceptance/sculpting 是探索性诊断，不是独立 calibration 上冻结阈值后的无偏部署评估。

实现依据：`src/energybench/dependence.py`、`src/energybench/evaluation.py`。

## 7. `results.csv` 一行摘要标准

`results.csv` 固定为每次 evaluation 一行、64 列。Python/NumPy 的 `None` 或非有限结果写为空单元格；空单元格表示 NA，不表示 0。字段顺序由 `src/energybench/reporting.py` 冻结。

### 7.1 标识与数据范围

| 字段 | 含义 |
|---|---|
| `results_schema_version` | CSV schema 版本，当前为 1 |
| `model_id` | 本次运行唯一模型/checkpoint 标识 |
| `task_id` | 冻结任务标识 |
| `experiment` | 实验名 |
| `dataset_id` | 不可变数据集标识 |
| `dataset_version` | 数据发布/校验版本 |
| `split` | 评测 split |

### 7.2 分类字段

| 字段 | 含义 |
|---|---|
| `classification_status`, `classification_reason` | 分类任务整体是否适用；整体为 `ok` 不保证固定 pair 集完整 |
| `matched_auc_status`, `matched_auc_reason` | 正式 matched 宏平均状态与聚合后的 NA 原因 |
| `matched_auc_macro` | 仅完整 pair 集通过 gates 时存在的正式分类主指标 |
| `matched_auc_macro_available` | 当前可评价 pair 的诊断均值；不可排名 |
| `inclusive_auc_macro` | 固定 pair 集完整时的 inclusive AUC 未加权宏平均 |
| `matched_pairs_evaluable` | 有正式 matched AUC 的 pair 数 |
| `matched_pairs_expected` | 冻结 pair 集总数 |
| `complete_pair_set` | 前两者是否相等且 pair 集非空 |

### 7.3 能量回归字段

| 字段 | 含义 |
|---|---|
| `energy_regression_status`, `energy_regression_reason` | `ok`、`no_finite_predictions` 或 `not_applicable` 及原因 |
| `energy_regression_score_name` | 当前为 `ERS-v1` |
| `energy_regression_score` | ERS-v1 主指标；无有限预测时为 0 |
| `energy_regression_ci_low`, `energy_regression_ci_high` | ERS-v1 percentile bootstrap 区间 |
| `energy_regression_ci_level` | 区间置信水平 |
| `energy_regression_bootstrap_requested` | 请求 replicate 数 |
| `energy_regression_bootstrap_successful` | 产生有限 ERS 的 replicate 数 |
| `event_score` | $S_{\rm event}$ |
| `histogram_similarity` | $S_{\rm hist}$ |
| `histogram_overlap` | flow histogram 质量交集 |
| `jsd_bits` | truth/predicted histogram 的 base-2 JSD |
| `wasserstein_1` | truth/predicted 能谱的 $W_1$ |
| `mae`, `rmse`, `bias`, `r2` | 加权绝对误差、均方根误差、有符号均值偏差和决定系数 |
| `fractional_bias` | 分数残差加权中位数 |
| `fractional_resolution_68` | 分数残差中央 68% 半宽 |
| `balanced_fractional_mae` | truth-bin 等权分数 MAE |
| `finite_fraction` | 有限预测的 base-weight fraction |

### 7.4 Energy-dependence 字段

| 字段 | 含义 |
|---|---|
| `energy_dependence_status`, `energy_dependence_reason` | 依赖任务整体计算状态 |
| `energy_independence_score_status`, `energy_independence_score_reason` | 是否所有预期类内组均可评价 |
| `dependence_groups_evaluable` | `status=ok` 的组数 |
| `dependence_groups_expected` | 预期组总数 |
| `complete_dependence_group_set` | 所有组是否完整 |
| `energy_independence_score_mean` | 完整时按组 base mass 加权的 $I_E$ |
| `energy_independence_score_worst` | 完整时各组最小 $I_E$ |

### 7.5 分数/能量语义

| 字段 | 含义 |
|---|---|
| `score_column` | 实际解析为分类 score 的列 |
| `score_space` | probability、logit、rank、conditional quantile 等明确空间 |
| `score_direction` | `higher` 或 `lower` |
| `energy_condition_kind` | 分类匹配协变量的物理定义 |
| `energy_target_kind` | 回归目标的物理定义；不适用时为 `not_applicable` |
| `energy_unit` | 能量列和相应误差的单位 |

### 7.6 质量与 provenance

| 字段 | 含义 |
|---|---|
| `n_events` | canonical table 事件数 |
| `strict` | 是否由 strict evaluation 产生 |
| `warning_count`, `error_count` | 完整机器报告中的警告/错误计数 |
| `evaluation_fingerprint` | 排除预测、对事件身份/真值/权重等哈希 |
| `protocol_fingerprint` | 冻结评测协议哈希 |
| `code_fingerprint` | EnergyBench Python 源码哈希 |
| `input_sha256` | 若 evaluator 从实际文件加载，记录该输入文件的 SHA-256；内存 bundle 可为空 |
| `created_at_utc` | UTC ISO-8601 生成时间 |
| `source_schema_version` | 完整 `metrics.json` schema 版本，当前为 1 |

分类 bootstrap CI 不在这张 64 列摘要表中；若启用，应从 `metrics.json` 的 pair-level `matching.*_auc_ci` 读取。

## 8. 完整输出文件及其意义

### 8.1 `energybench inspect` 的标准输出

`inspect` 不生成固定文件，而是在终端输出 JSON：`source` 是输入路径，`n_events` 是事件数，
`columns` 给出每列 shape/dtype，`metadata` 是输入 provenance，`inferred_roles` 是最终推断的
canonical 映射，`duplicate_event_ids` 是重复主键数或 NA。它只做结构盘点，不计算分数，也不能证明
energy 的物理定义、单位或 split 无泄漏；需要保存时可由调用者重定向到 `.json`。

### 8.2 `predictions_<split>.npz`

模型无关的逐事件证据层，保存 canonical columns 和 provenance metadata。它可被 `inspect`、`evaluate`、`decorrelate` 重复读取，也是重新评分而无需重新推理的边界。`energybench next` 先保存该文件，再对内存中的同一 bundle 评分；因此现有部分 v2 `metrics.json` 的 `input.path/input.sha256` 为空，尽管 NPZ 实际存在。

### 8.3 `results.csv`

人读和下游表格处理的一行摘要。正式结论优先读取 status、主指标、完整性和 fingerprint，而不是只读取一个浮点数。

### 8.4 `.energybench/metrics.json`

完整机器报告，顶层包括：

| 对象 | 内容 |
|---|---|
| `evaluator`, `created_at_utc` | 评测器版本、运行时版本、代码指纹与时间 |
| `dataset`, `task_id`, `model_id` | 冻结任务/数据标识 |
| `input`, `resolved_columns` | 输入路径/hash/metadata 与最终列角色 |
| `quality` | 事件数、重复 ID、split、strict、warnings/errors |
| `classification.aggregates` | 完整 pair 集宏平均与完整性 |
| `classification.pairs[]` | 每个 pair 的角色、事件数、正式/诊断 AUC、common-support AUC、gap、状态 |
| `pairs[].matching` | support、edges、每-bin 质量、coverage、ESS、balance、inclusive/matched ROC 数组、工作点和 bootstrap |
| `energy_regression` | ERS-v1 全部组成、flow histograms、分能曲线数组和各指标 bootstrap |
| `energy_dependence` | 各类相关、JSD、independence、acceptance、sculpting 和聚合 |
| `artifacts` | 本报告实际生成的用户可见文件名 |

ROC 数组的首点是 $(0,0)$、内存阈值为 $+\infty$；严格 JSON 序列化会把该非有限阈值写成 `null`。这一个 `null` 是起始哨兵，不是曲线缺失。当前大样本 `metrics.json` 约 23 MB，主要因为保存了 inclusive 与 matched 的完整 FPR/TPR/threshold arrays；它适合审计，不适合手工阅读。

回归对象同时保留 `hist_similarity`/`histogram_similarity`、`truth_histogram`/`true_histogram_probability` 和 `prediction_histogram`/`pred_histogram_probability` 兼容别名；这些是同一数值的重复命名，不应当作不同指标。

### 8.5 `.energybench/resolved_manifest.json`

默认值、输入 YAML/JSON 和 CLI overrides 合并后的实际协议快照。它回答“本次真正使用了什么参数”，而不是“模板默认写了什么”。复现实验时应与 prediction table 和 metrics 一起保存。

### 8.6 Decorrelate 输出

`energybench decorrelate` 产生：

- 新 NPZ：保留原列，首次运行时增加 `score_raw`，并增加 `score_decorrelated`；metadata 中写入 calibration/test hashes、split、背景标签、样本数、bin 数和 disjointness 状态；
- `<output>.decorrelator.json`（或 `--artifact` 指定路径）：保存 background conditional weighted ECDF 的 `energy_edges`、每 bin 排序 score、累计权重、score direction、方法版本和 provenance。

变换使用 mid-distribution ECDF：

$$
U(E,s)=\tfrac12\{F_{0,E}(s^-)+F_{0,E}(s)\}.
$$

默认要求 calibration 与 test 的 event IDs 和 split roles 可验证不相交；`--allow-overlap` 只产生 `unverified_override`，不能用于 strict 正式结论。实现依据：`src/energybench/decorrelation.py`、`src/energybench/cli.py`。

### 8.7 Compare 输出

`energybench compare` 产生：

- `leaderboard.csv`：每个 evaluation 一行，增加 `classification_rank`、`energy_regression_rank`、`comparison_mode`、`comparable` 和 `source_metrics`，并携带主要状态、分数、完整性、数据标识与 fingerprints；
- `.energybench/comparison.json`：保存比较模式、输入数量、fingerprint/code/protocol 集合、缺指纹或非 strict 的输入索引，以及 leaderboard 全部行。

`--allow-mixed-data` 只生成 `non_comparative_inventory`，保持输入顺序并留空两个 rank。

### 8.8 训练产物

当前可由源码重跑的 CNN-001 分类训练输出包括：

| 产物 | 意义 |
|---|---|
| `02_models/checkpoints/*_best.pt` | validation AUC 最高的 checkpoint；AUC 不可算时回退为最小 validation loss |
| `02_models/checkpoints/*_last.pt` | 最后一个 epoch checkpoint |
| `03_training_runs/logs/*_validation_metrics_*.csv` | 每 epoch 的 train/validation BCE、accuracy、inclusive AUC、AMP skip、validation projection coverage、该 epoch 实际使用的 learning rate、是否刷新 best |
| `03_training_runs/logs/*_history_*.json` | 同样历史的嵌套完整记录，另含事件数和 train/validation coverage |
| `03_training_runs/history_plots/*_history_*.png` | 训练/验证 BCE 与 inclusive AUC 的 epoch 曲线 |
| `03_training_runs/history_plots/*_score_*.png` | 最终 best checkpoint 的 validation logit 类别密度 |

Checkpoint 还保存模型/优化器状态、投影、split seed/fractions、训练配置、数据 inventory、实际 train/validation groups 和 history。实现依据：`01_code/architectures/cnn_001_two_conv_baseline/train_classification.py`。

配套的 CNN-001 能量回归程序还会写出 best/last checkpoint、包含标准化 Smooth-L1 loss 与物理单位 MAE/RMSE/bias/$R^2$ 的逐 epoch CSV/JSON、loss/error history 图，以及 validation prediction-target/residual 图。它按最小 validation Smooth-L1 loss 选择 best，并把仅由训练集拟合的能量标准化参数写入 checkpoint。这些训练诊断不能替代锁定 test split 上的正式 ERS-v1 评测。实现依据：`01_code/architectures/cnn_001_two_conv_baseline/train_energy_regression.py`。

目录中还保留 v2 历史 CSV/JSON 以及 joint-loss/AUC/energy-error、score、energy-validation PNG 和 v2 prediction/evaluation 产物。这些历史记录表明 v2 曾记录分类 BCE、标准化能量 Smooth-L1 加权项、MAE/RMSE/bias/$R^2$，并按最小 validation joint loss 选择 best。然而当前工作区缺少 `01_code/src/next_cnn_v2.py` 和相应 v2 `.pt` checkpoint。因此 v2 **历史结果可解释、可从现存 NPZ 重新评分，但不能从当前仓库完整重训或重新推理复现**。唯一残留的 CPython 3.11 字节码痕迹已隔离到 `/home/wenyu/summer_legacy/bytecode/`，仅作恢复证据，不是可执行或可复现源码。

### 8.9 旧版遗留产物

`04_evaluations/next_cnn_v1_smoke/evaluation_5files/` 中的 `report.md`、`pair_metrics.csv`、根目录 `metrics.json` 和 `resolved_manifest.json` 是旧版输出。当前 `run_evaluation` 不再生成 Markdown report、pair table、`energy_regression_bins.csv` 或 `matching_diagnostics/`；当前正式布局是 `results.csv`、可适用 PNG 与 `.energybench/{metrics,resolved_manifest}.json`。历史输出仅用于追溯，不能作为当前目录或表格格式范例。

本次整理修改了评分包的源码内容（不改变主指标数学定义），因此新运行的 `code_fingerprint` 与已存历史结果不同。如需正式横向比较，应使用现存 prediction NPZ 在同一当前版本下全部重新评分，不得将新旧 fingerprint 结果直接混排。

实现依据：`src/energybench/evaluation.py`、`src/energybench/reporting.py`。

## 9. 标准评测图与训练图的逐图意义

### 9.1 `energy_matched_roc.png`

横轴为 FPR/背景接受率，纵轴为 TPR/信号效率；黑色对角线是随机排序。每个 pair 的实线是 energy-matched ROC，浅色虚线是 inclusive ROC。图例给出匹配核心算法的 AUC 和可用 CI。实线明显低于虚线提示原始排序可能受类间能谱差异帮助；两者接近仅说明这两个 estimand 接近，不证明 score 与能量独立。

注意：绘图层直接读取核心 `result.matched.auc`，而不是 policy gate 后的正式 `pair.matched_auc`。若 coverage/bin gate 失败，图仍可能显示一条诊断曲线；正式结论必须以 `results.csv` 的 status/主指标为准。实现：`src/energybench/plotting.py`。

### 9.2 `score_energy_dependence.png`

左图按真实类别绘制各 weighted energy-quantile bin 的加权平均 classifier score；图例同时给出 dCor 和 independence score。曲线斜率或弯曲表示 score 响应随能量变化，但平均值曲线不能替代完整分布 JSD。

右图在报告阈值下绘制每类各 energy bin 的接受率。平坦曲线表示该固定 cut 的效率更稳定；类别间绝对高度差是分类行为本身，不应误判为 sculpting。该阈值来自同一 evaluation sample，因此本图是探索性诊断，且没有置信带。实现：`src/energybench/plotting.py`。

### 9.3 `energy_regression.png`

左图是最多 30,000 个有限事件的 true-vs-predicted hexbin，颜色表示对数事件密度，红色虚线为理想 $\widehat T=T$。靠近对角线表示逐事件响应正确；收缩成水平带表示模型接近常数预测。

右图绘制 residual $\widehat T-T$ 对 true energy 的散点、每个 truth bin 的加权 median residual，以及 `median ± central-68%-half-width` 阴影。红线偏离 0 表示能量依赖 bias；阴影宽度表示分辨率。若残差分布不对称，`median ± half-width` 并不等同于真实的 $[Q_{0.16},Q_{0.84}]$ 两端。可视化子样本是固定 seed 的均匀事件抽样，图上密度不使用 sample weights；数值指标仍使用权重。实现：`src/energybench/plotting.py`。

### 9.4 `energy_histograms.png`

上图叠加 weighted normalized truth/predicted **每-bin 概率质量**，并标注 JS similarity、histogram overlap 和两者的 underflow/overflow 质量。flow bins 不画在横轴可见区域，但进入分数。下图是可见 bin 的 predicted/true 质量比；水平线 1 表示谱形一致，truth 为 0 的 bin 比值为 NA。

该图只检验总体能谱，不能证明事件一一对应正确；必须与 event score 和 `energy_regression.png` 联读。ratio 纵轴为可读性按 95 分位并最多截到 5，不是误差条或置信区间。实现：`src/energybench/plotting.py`。

### 9.5 CNN-001 分类 `*_history_*.png`

左图是 train/validation 的 `BCEWithLogitsLoss` 随 epoch 变化；右图是同一时期的未做能量匹配的 inclusive AUC。训练 loss 下降而 validation loss 上升提示过拟合；train/validation AUC 差距反映泛化差距。该图用于 checkpoint 训练过程诊断，不是锁定 test split 的正式性能图，也不能替代 energy-matched AUC。实现：`01_code/architectures/cnn_001_two_conv_baseline/train_classification.py`。

### 9.6 CNN-001 分类 `*_score_*.png`

图中填充直方图是 validation `0nubb` logit 密度，轮廓直方图是 `Bi214` 密度；标题 AUC 是 validation inclusive AUC。分布越可分通常排序越好，但图形受 logit 尺度和 binning 影响。它没有能量匹配、base weight、coverage、CI 或 test-set 含义，只用于训练/选择阶段的分数形状检查。实现：`01_code/architectures/cnn_001_two_conv_baseline/train_classification.py`。

### 9.7 CNN-001 回归 `*_energy_*.png`

左图是 validation predicted energy 对 target energy 并画出恒等线；右图是残差 `prediction - target` 分布。标题按适用情况报告 validation MAE、RMSE、bias 与 $R^2$，能量单位为 MeV。这只是 validation 训练诊断，不具备锁定 test split、ERS-v1、不确定度或 sample weight 含义，不能作为正式回归结果。实现：`01_code/architectures/cnn_001_two_conv_baseline/train_energy_regression.py`。

## 10. 正式比较与报告规则

正式 leaderboard 必须同时满足：

1. 每个输入来自成功 strict evaluation，且 `quality.errors` 为空；
2. 所有输入具有完全相同的 evaluation、protocol 与 code fingerprints；
3. 每个 `model_id` 非空且唯一；
4. 分类只按 `matched_auc_status=ok` 的 `matched_auc_macro` 降序排名；
5. 回归按 ERS-v1 降序独立排名；`no_finite_predictions` 的 ERS=0 可以排在有效模型之后；
6. 同值使用 competition ranking，例如 1、1、3；
7. 分类和回归 rank 不互相充当 tie-breaker，也不合成总 rank；
8. `matched_auc_macro_available`、inclusive AUC 或 independence score 不得填补正式分类 rank；
9. `--allow-mixed-data` 只能形成不可比较 inventory，不得给出性能名次。

一份论文式结果至少应同时报告：主指标与 status、完整 pair/group 数、inclusive 基线、coverage/ESS、关键回归组成、依赖诊断、单位、manifest/fingerprints、CI 或“不提供 CI”的原因。所有 NA 应连同机器可读 reason 报告。

实现依据：`src/energybench/reporting.py`。

## 11. 限制、稳健性与待补能力

1. **组内相关未进入置信区间。** 当前分类和回归仅实现 event bootstrap；`group_id` 只进入 fingerprint。NEXT 正式 manifest 因此关闭 bootstrap。后续如需 CI，应实现按 HDF5/run 分层的 group bootstrap。
2. **阈值未独立校准。** 当前 acceptance/sculpting 使用 test/evaluation sample 自身的 matched ROC 阈值，只能探索性解读。
3. **Energy matching 是分 bin 近似。** 匹配后 bin 级质量相等，但 bin 内仍可有残余能量差异；应联读匹配后 KS/$W_1$、coverage 和 ESS。
4. **Dependence 不是因果或完全独立证明。** Pearson/Spearman 有形状盲点，dCor 大样本为抽样近似，JSD 依赖 binning，且当前没有不确定性区间。
5. **Score-space 敏感。** ROC 对严格单调变换不变；mean score、Pearson、JSD、slope/threshold 类诊断通常不变性不足，跨 probability/logit 比较必须谨慎。
6. **ERS-v1 是项目内分数。** 它对 histogram/performance bins、floor、目标范围和 weights 敏感；不同任务的 ERS 即使都在 $[0,1]$ 也不能直接比较。
7. **图像是诊断而非充分证据。** 当前图没有统计误差带；回归散点还使用未加权子样本。正式数值以机器结果为准。
8. **v2 历史链不完整。** 缺少 v2 训练源码和 checkpoints，使现存 v2 结果不能端到端重现；在补回与校验前只能作为历史评测样例。

## 12. 实现索引

| 主题 | 当前源码 |
|---|---|
| canonical table、NPZ、role inference | `src/energybench/data.py` |
| manifest 默认值与验证 | `src/energybench/config.py` |
| 分类 pair、policy gates、聚合、fingerprint、输出编排 | `src/energybench/evaluation.py` |
| inclusive/matched ROC、coverage、ESS、balance、bootstrap | `src/energybench/roc.py` |
| ERS-v1 与回归诊断 | `src/energybench/regression.py` |
| 类内 score–energy dependence | `src/energybench/dependence.py` |
| held-out background ECDF | `src/energybench/decorrelation.py` |
| 评测/训练图 | `src/energybench/plotting.py`、`01_code/architectures/cnn_001_two_conv_baseline/train_classification.py` |
| `results.csv` 与 leaderboard | `src/energybench/reporting.py` |
| NEXT adapter 与逐事件物理列 | `src/next_cnn/adapter.py`、`src/next_cnn/data.py` |

## 13. 参考文献

1. Fawcett, T. “An Introduction to ROC Analysis.” *Pattern Recognition Letters* 27 (2006): 861–874. <https://doi.org/10.1016/j.patrec.2005.10.010>
2. Janes, H., Longton, G., and Pepe, M. S. “Accommodating Covariates in ROC Analysis.” *The Stata Journal* 9 (2009). <https://pmc.ncbi.nlm.nih.gov/articles/PMC2758790/>
3. Efron, B. “Bootstrap Methods: Another Look at the Jackknife.” *The Annals of Statistics* 7 (1979): 1–26. <https://doi.org/10.1214/aos/1176344552>
4. Lin, J. “Divergence Measures Based on the Shannon Entropy.” *IEEE Transactions on Information Theory* 37 (1991): 145–151. <https://doi.org/10.1109/18.61115>
5. Székely, G. J., Rizzo, M. L., and Bakirov, N. K. “Measuring and Testing Dependence by Correlation of Distances.” *The Annals of Statistics* 35 (2007): 2769–2794. <https://doi.org/10.1214/009053607000000505>
6. SciPy Developers. “`scipy.stats.wasserstein_distance`.” <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html>

本文档是当前项目的唯一评分与表格规范。若文档、manifest 和源码冲突，应停止正式比较，核对实际 `resolved_manifest.json` 并同步提升协议/结果版本，不得静默沿用旧语义。
