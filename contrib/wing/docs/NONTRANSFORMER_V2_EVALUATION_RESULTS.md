# 非 Transformer v2 模型评估结果

## 评估合同

训练 campaign `20260803_200356` 的 10 个非 Transformer 分类模型已于
2026-08-04 使用 `docs/USAGE_GUIDE.md` 规定的 EnergyBench NEXT 流程完成正式评估。

- 数据集：`zeronu-benchmark-next`
- 数据版本：`zenodo-18927784-v1.0-tarset-0b57f1a2c33c`
- 数据范围：完整 `test` split，1,490 个文件、115,499 个事件
- manifest：`manifests/next_0nubb_vs_bi214.yaml`
- 模式：strict、无文件数限制、未混用数据或协议
- 可比性：10 行具有相同的 evaluation、protocol 和 code fingerprint
- 完整性：全部结果 `strict=True`，warning 数和 error 数均为 0

正式分类排名如下。主排名指标是能量匹配后的 AUC；Energy independence
列为各分类组的平均能量独立性得分。

| 排名 | Architecture | 类型 | Matched AUC | Inclusive AUC | Energy independence |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | `gnn_005_dimenet_lite` | directional message passing | 0.979227 | 0.979351 | 0.974925 |
| 2 | `point_003_pointmlp` | residual point MLP | 0.974984 | 0.975353 | 0.978494 |
| 3 | `sparse_001_submanifold_resnet` | submanifold sparse 3D ResNet fallback | 0.970237 | 0.970562 | 0.977391 |
| 4 | `classic_001_topology_xgboost` | topology features + XGBoost | 0.947282 | 0.947664 | 0.979120 |
| 5 | `seq_001_bigru` | Hilbert sequence + BiGRU | 0.933935 | 0.934251 | 0.977601 |
| 6 | `point_004_rigid_kpconv` | rigid KPConv | 0.931301 | 0.931661 | 0.976893 |
| 7 | `seq_002_dilated_tcn` | Hilbert sequence + dilated TCN | 0.926043 | 0.926353 | 0.975525 |
| 8 | `ssm_001_pointmamba` | selective SSM PyTorch fallback | 0.925420 | 0.925742 | 0.977934 |
| 9 | `topo_001_persistence_perslay` | persistence + PersLay-style pooling | 0.907281 | 0.907488 | 0.971912 |
| 10 | `mixer_001_projection_mlp_mixer` | projection MLP-Mixer | 0.890496 | 0.890960 | 0.977347 |

这些 checkpoint 只执行二分类，因此 energy regression 的 `not_applicable` 状态是预期行为，
不是评估失败。

## 主要观察

- DimeNet-lite 的 matched AUC 最高，比 PointMLP 高约 0.004243。
- PointMLP 与 sparse 3D ResNet 分列第二、第三；前三名 matched AUC 均超过 0.97。
- XGBoost 的 matched AUC 为 0.947282，在不使用神经网络的前提下排名第四，并具有本批次最高的平均 energy-independence 得分。
- BiGRU 高于本次 TCN 和 PointMamba；TCN 与 PointMamba 的 matched AUC 仅相差约 0.000624。
- 此处 sparse ResNet 和 PointMamba 的成绩来自项目文档声明的纯 PyTorch fallback，不应解释为官方高性能 CUDA kernel 的复现结果。
- MLP-Mixer 在本次冻结配置中排名最后；这只说明当前三视图投影、patch 与训练设置的组合表现，不代表 MLP-Mixer 家族的普遍上限。

## 产物

权威 leaderboard：

```text
04_evaluations/NEXTALT_nontransformer_v2_20260803_200356_comparison/leaderboard.csv
```

每个模型的独立目录为：

```text
04_evaluations/NEXTALT_<architecture_id>_test/
```

其中包含：

- `predictions_test.npz`
- `evaluation_test/results.csv`
- `evaluation_test/.energybench/metrics.json`
- `evaluation_test/energy_matched_roc.png`
- `evaluation_test/score_energy_dependence.png`

完整队列日志：

```text
04_evaluations/logs/nontransformer_v2_evaluation_queue_20260804_171829.log
```

可复现队列入口：

```text
01_code/architectures/run_nontransformer_evaluation_queue.sh
```

本次队列从 2026-08-04 17:18:29 运行至 18:03:24（America/Los_Angeles），
10 个模型均一次成功，随后完成 formal comparison。
