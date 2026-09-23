# NEXT 与 MJD 输入与变化审计（最终 v2）

依据用户后续要求，最终协议为600个0–3000keV常规箱（3000入最后常规箱）加独立第601个 `E>3000keV` overflow箱；不clip，不丢弃overflow。负能量和非有限能量不进入能量指标。此前严格排除overflow的v1结果保留于明确命名的历史文件，不是最终结果。

本目录只修改自身，未训练、未改变 checkpoint、未改变测试集、未修改共享评价代码或论文。逐事件标准化文件保留全部 inclusive 人群，合法能量和独立 overflow 处理交给共享评价器。工作簿和旧审计只用于定位/识别论文条目；数值输入来自原逐事件 NPZ 和官方 HDF5。

## 已完成范围

- `manifest.json`：22 个 canonical NEXT classic 候选（含主表3模型、附表22个可得结果）和 MJD classic 4模型，全部 `ready`。NEXT canonical PointMamba 原来即无完整结果，不能用旧 NEXTALT run 替换。
- `exploratory/manifest.json`：早期 NEXT 独立 115499 事件人群，共11模型。7个在 appendix_models.tex 引用数值，额外4个 graph 候选用于复核 GINE 的排序描述。它们与 canonical NEXT 必须分别报告。
- Transformer 查找已按用户随后“transformet的不用管”停止；既有缺失来源审计仅保留历史记录，不是缩小后任务的阻塞。

## 标签、分数、能量和测试事件

NEXT label=1 对应原始 HDF5 `Signal` / category `0nubb`，label=0 为 `Bkg` / `Bi214`。标准化 score 是原 float32 logit 的无损 float64 转换，未 sigmoid，未翻转方向。分数越大越支持 0nubb。能量是原 hit `MC/hits/table.values_block_1[:,3]` 的 float64 组内和，单位 MeV，使用共享 `to_keV` 转为 keV。22个 canonical 模型的 event_id、label、category、原始 energy、sample_weight、group_id 和 split 全部逐元素相等，均为 116549 测试事件（64879 signal，51670 background）。每条预测也与 reevaluation_record 指向的原训练输出逐元素相等。

`audit_next_raw.py` 已读取全部1498个 test HDF5，按 `event_split.json` 的官方 ordinal slices 恢复116549个 event ID、原 label 和能量和；全部精确等于保存预测，最大能量差为0。文件级验证记录在 `next_raw_event_verification.json`。这超出了只对照汇总数量的检查。canonical split 为分层80/10/10事件数分割；较早 exploratory 则为文件分割，二者不同。

MJD label=1 为四个参考 PSD flags (low_avse, high_avse, dcr, lq) 全为1的 clean，label=0为 nonclean。CNN/BiGRU/PointMamba 保存二分类 clean logit；GINE 训练额外四 PSD 目标，分数为 `logit(product(sigmoid(auxiliary_logits)))`，实现通过 summed logsigmoid 保持稳定。不能把其监督优势仅解释成图架构优势。四个模型均直接复用保存分数，无构造变化。

MJD六个 `MJD_Test_0..5.hdf5` 按原代码字典序拼接、保持各文件行序，总390000事件（clean148251，background241749）。原输出 event_id 为 `mjd-test-<index>`；本次同时保存 shard、row、raw id、run、detector 和独立唯一的 physical_event_id。逐项核验原预测 target、保存评分和官方四标签 conjunction 精确一致；每个模型的测试元数据全部相等。

MJD原评价把官方 float64 `energy_label` 先 cast float32、除1000再 clip到[0,3] MeV。恢复官方原始 calibrated keV 后，67个真实overflow事件（66 background、1 clean）超过3000，最高4177.604856194463 keV。未发现负能量或非有限能量。`float32 -> float64 /1000 -> clip` 完整精确复现原390000个保存能量，因此原值已找回，不存在只持有clip后能量的限制。`MJD_official_test_shards.json` 保存每个原scalar字段的hash，标准文件保留完整float64原能量。

## 历史协议与最终变化归因

canonical NEXT/MJD 历史固定5keV；I以各组≥20事件的能量箱人群估计score20分位箱与pooled histogram，聚合使用稀疏筛选前组权重；JS自然对数，先组内加权平均再开根，最后聚合I_g。Matched AUC独立支持裁剪0.5%、overlap target、每箱每类≥20、至少2有效箱、逐类覆盖≥0.5。早期NEXT exploratory原I为每组8能量分位箱、matched独立6分位箱，两种指标分别改为最终共享网格。

`attribute_changes.py` 对全部26 canonical模型调用原历史函数和原MeV数组，旧I、matched AUC、inclusive AUC全部精确等于saved metrics；dCor样本量4只影响未报告辅助诊断。`change_attribution.json` 为最终v2全精度阶段归因，`change_attribution_v1_strict_range.json` 仅为被用户修改的历史v1。

NEXT22个canonical模型最终I完全不变，matched差异低于7e-13，仅为独立加权排序累加浮点差。真实NEXT没有overflow；共享to_keV对canonical边界的修复影响0事件。早期NEXT重算源为独立115499事件的原score数组，数值变化来自历史分位能量网格统一；不得把此旧run或PointMamba替换到canonical主表。

MJD最终变化来源逐一记录：

1. 原MeV构造边界在2385keV为`2.3850000000000002`，而`mjd-test-145680`为`2.385`，历史错入左侧箱；最终整数keV网格正确归[2385,2390)。
2. 67个原物理overflow（66 background、1 clean）从历史clip尾箱移到独立第601箱。背景overflow满足I的≥20门槛，clean overflow不满足；匹配支持上端约2613.79keV，故overflow自然在matching共同支持外，不因为变独立overflow而改变matched AUC。背景最后常规箱剩1事件，成为稀疏排除。
3. 恢复官方float64能量后，原float32舍入到275、335、2615keV的三个事件实际略低于边界，正确归入左侧箱。

MJD最终指标（完整精度）：

| 模型 | 新I | 新matched AUC | ΔI | Δmatched AUC |
|---|---:|---:|---:|---:|
| mvcnn | 0.7595780033237467 | 0.9698515539237522 | -1.70213251937668e-07 | -8.42513414589519e-08 |
| gine | 0.7965705447318409 | 0.9691831024267327 | +5.4896856395148e-06 | -1.67511647886087e-08 |
| bigru | 0.7752460566782948 | 0.9754459601561317 | +4.03945686777707e-06 | -6.55867028465096e-08 |
| mamba | 0.773278117459327 | 0.9598875942760778 | -4.99375774509758e-07 | -1.16571202690707e-08 |

MJD模型排序不变，inclusive AUC保留原390000人群，差异≤1.2e-16。MJD PointMamba工作簿旧路径误指regression且数值截断，本次旧全精度对照直接读取classification metrics。

## 输出与复现

- `manifest.json` / `alignment_checks.json`：26 canonical模型逐事件来源、旧全精度、方向、能量、原测试身份、hash及代码位置。
- `MJD_official_test_metadata.npz`、`MJD_official_test_shards.json`、`mjd_energy_recovery.json`：原能量/ID恢复证据。
- `next_raw_event_verification.json`：1498官方HDF5 / 116549事件全量核验。
- `exploratory/{manifest,results,alignment}.json`：独立早期NEXT 11模型最终协议重算；其中7条为论文原文数值，4条用于graph排名检查。`results_v1_strict_range.json`仅历史v1。
- `../../code/plot_unified_next.py`：读取主代理生成的论文CSV并逐项核对full precision results，生成工作目录`figures/next_capacity_scores.pdf/png`。源CSV已更新到v2；两种导出均已视觉检查，19点不clip，标签可读。

```bash
PY=/home/wenyu/summer/.venv/bin/python
DIR='/home/wenyu/iclr final paper/unified_5kev_evaluation/sources/next_mjd'
"$PY" -B "$DIR/extract_next_mjd.py"
"$PY" -B "$DIR/audit_next_raw.py"
"$PY" -B "$DIR/attribute_changes.py"
"$PY" -B "$DIR/extract_exploratory.py"
"$PY" -B "$DIR/../../code/plot_unified_next.py"
```

标准NPZ的`energy_keV`是最终原物理能量，不进行clip；`legacy_energy_keV`与MJD `energy_float32_keV`仅作阶段归因。所有模型为单次运行点估计，不构成统计显著性结论。
