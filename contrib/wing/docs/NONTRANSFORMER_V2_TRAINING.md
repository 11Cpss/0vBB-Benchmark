# NEXT non-Transformer v2：仅训练 campaign

本文件是 10 个新架构的统一运行合同。每个架构的逐层结构、公式、输入 shape、复杂度、论文差异与完整引用位于对应目录的 `README.md` / `README_EN.md`。

## 严格边界

- 训练只允许发现 `train` 与 `validation` 文件；validation 仅用于 early stopping 和 best checkpoint 选择。
- 本 campaign 不读取 `test` split，不运行测试、smoke training、EnergyBench、checkpoint 推理或排行榜，也不写入 `04_evaluations/`。
- 单 GPU 只运行一个训练进程。tmux 的 `monitor` window 只读取 manifest、epoch CSV、GPU 状态与日志。
- 每类最多选择 100 个源 HDF5 文件；file-level split 固定 seed 42 和比例 `[0.8, 0.1, 0.1]`。

## 固定队列

| 顺序 | architecture ID | 模型卡 | 方法来源 |
|---:|---|---|---|
| 1 | `classic_001_topology_xgboost` | [中文](../01_code/architectures/classic_001_topology_xgboost/README.md) / [English](../01_code/architectures/classic_001_topology_xgboost/README_EN.md) | [NEXT topology](https://link.springer.com/article/10.1007/JHEP01%282016%29104), [XGBoost](https://doi.org/10.1145/2939672.2939785) |
| 2 | `point_003_pointmlp` | [中文](../01_code/architectures/point_003_pointmlp/README.md) / [English](../01_code/architectures/point_003_pointmlp/README_EN.md) | [PointMLP](https://arxiv.org/abs/2202.07123) |
| 3 | `seq_001_bigru` | [中文](../01_code/architectures/seq_001_bigru/README.md) / [English](../01_code/architectures/seq_001_bigru/README_EN.md) | [GRU](https://aclanthology.org/D14-1179/) |
| 4 | `seq_002_dilated_tcn` | [中文](../01_code/architectures/seq_002_dilated_tcn/README.md) / [English](../01_code/architectures/seq_002_dilated_tcn/README_EN.md) | [TCN](https://arxiv.org/abs/1803.01271) |
| 5 | `mixer_001_projection_mlp_mixer` | [中文](../01_code/architectures/mixer_001_projection_mlp_mixer/README.md) / [English](../01_code/architectures/mixer_001_projection_mlp_mixer/README_EN.md) | [MLP-Mixer](https://arxiv.org/abs/2105.01601) |
| 6 | `gnn_005_dimenet_lite` | [中文](../01_code/architectures/gnn_005_dimenet_lite/README.md) / [English](../01_code/architectures/gnn_005_dimenet_lite/README_EN.md) | [DimeNet](https://openreview.net/forum?id=B1eWbxStPH) |
| 7 | `point_004_rigid_kpconv` | [中文](../01_code/architectures/point_004_rigid_kpconv/README.md) / [English](../01_code/architectures/point_004_rigid_kpconv/README_EN.md) | [KPConv](https://arxiv.org/abs/1904.08889) |
| 8 | `topo_001_persistence_perslay` | [中文](../01_code/architectures/topo_001_persistence_perslay/README.md) / [English](../01_code/architectures/topo_001_persistence_perslay/README_EN.md) | [Persistence Images](https://jmlr.org/papers/v18/16-337.html), [PersLay](https://proceedings.mlr.press/v108/carriere20a.html) |
| 9 | `ssm_001_pointmamba` | [中文](../01_code/architectures/ssm_001_pointmamba/README.md) / [English](../01_code/architectures/ssm_001_pointmamba/README_EN.md) | [Mamba](https://arxiv.org/abs/2312.00752), [PointMamba](https://proceedings.neurips.cc/paper_files/paper/2024/hash/395371f778ebd4854b88521100af30ad-Abstract-Conference.html) |
| 10 | `sparse_001_submanifold_resnet` | [中文](../01_code/architectures/sparse_001_submanifold_resnet/README.md) / [English](../01_code/architectures/sparse_001_submanifold_resnet/README_EN.md) | [Submanifold Sparse CNN](https://arxiv.org/abs/1706.01307) |

## 启动

RUN_ID 必须唯一；它同时拥有 checkpoint 与 campaign 目录，已有目录绝不覆盖。

```bash
cd /home/wenyu/summer
RUN_ID="$(date +%Y%m%d_%H%M%S)"

tmux new-session -d \
  -s "next-nontransformer-v2-${RUN_ID}" \
  -n gpu-queue \
  "cd /home/wenyu/summer && bash 01_code/architectures/run_nontransformer_training_queue.sh --run-id ${RUN_ID}"

tmux new-window \
  -t "next-nontransformer-v2-${RUN_ID}" \
  -n monitor \
  "cd /home/wenyu/summer && bash 01_code/architectures/monitor_nontransformer_training.sh --run-id ${RUN_ID}"
```

查看而不改变训练状态：

```bash
tmux attach -t "next-nontransformer-v2-${RUN_ID}"
tmux capture-pane -pt "next-nontransformer-v2-${RUN_ID}:gpu-queue" -S -80
```

停止整个 campaign 使用 `tmux kill-session -t ...`。这可能中断当前 epoch，但不会删除任何已有 attempt。不要把 `last.pt`/`last.json` 称为可恢复断点：当前 runner 不做 optimizer-level checkpoint resume。

## 失败后继续

队列把单模型异常记录为 `FAILED` 并继续后续模型。修复后在同一个 session 或新 detached session 中执行：

```bash
bash 01_code/architectures/run_nontransformer_training_queue.sh \
  --run-id "${RUN_ID}" \
  --resume-queue
```

`--resume-queue` 跳过 `DONE`，其余模型创建下一个 `attempt_NNN` 并从头训练。旧 attempt、日志和 checkpoint 不会覆盖。只有 manifest 中 10 个模型全部为 `DONE` 时，campaign 才标记 `complete=true`。

## 产物

```text
02_models/checkpoints/<RUN_ID>/<architecture_id>/attempt_NNN/
  best.pt|best.json
  last.pt|last.json

03_training_runs/campaigns/<RUN_ID>/
  manifest.json
  queue.log
  README.md
  README_EN.md
  <architecture_id>/attempt_NNN/
    stdout.log
    config.snapshot.yaml
    epochs.csv
    history.json
    history.png
    run_summary.json
```

每次成功训练后，队列会把实际参数量/后端、设备、完成与 best epoch、best validation AUC/loss、耗时、early stop、attempt 和产物路径追加到该模型的中英文模型卡。campaign README 汇总运行状态和 validation 结果，但不是正式测试排行榜。

