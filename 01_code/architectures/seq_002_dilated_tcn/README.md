# SEQ-002：Hilbert / Trans-Hilbert Dilated TCN

中文 | [English](README_EN.md)

## 1. 模型定位

| 项目 | 值 |
|---|---|
| `architecture_id` | `seq_002_dilated_tcn` |
| checkpoint `model_name` | `HilbertTCNClassifier` |
| Python class | `next_alt.models.point_sequence.HilbertTCNClassifier` |
| registry `input_kind` | `sequence` |
| 任务 / 输出 | `0nubb=1`、`Bi214=0`；每事件 `(B,)` signal logit |
| 配置推导参数量 | **694,529** |

研究假设是：空间填充曲线把 3D 点云变成局部连续的一维轨迹后，指数 dilation 的纯卷积网络可并行提取从短程 blob 到长程主轨迹的多尺度模式，并作为 GRU/SSM 之外的无 attention 序列基线。

## 2. 原始数据和表示

共享读取器从 `/MC/hits/table` 按连续 `event_id` 组装 `x/y/z/energy`。目录 `0nubb_part_*` 对应 label 1，`Bi_part_*` 对应 label 0。完整相对文件路径用 seed 42 做 file-level 0.8/0.1/0.1 稳定切分。本阶段只构建 train/validation，每类每 split 最多 100 文件；validation 只选 best 和 early stop。

每事件先用完整能量 $E=\sum_i e_i$ 计算质心 $\sum_i e_i\mathbf r_i/E$，再按 15 mm 合并 cell，并对 voxel 中心第二次能量居中。若超过 512 个点，保留最高能 512 个；特征仍用完整事件分母：

$$
x_i=[\mathbf r_i^{\mathrm{centered}}/(1000\ \mathrm{mm}),\ e_i^{\mathrm{voxel}}/E,\ \log(1+n_i)].
$$

总能量与绝对探测器坐标不输入模型。

## 3. Hilbert / Trans-Hilbert 合同

有效 XYZ 在每事件包围盒中分别量化到 10 bit；用 Skilling algorithm 得到 3D Hilbert code 并稳定排序。Trans-Hilbert 在量化后固定交换 x/y 轴再编码。SEQ-001、SEQ-002 与 SSM-001 调用同一函数；两条序列 shape 都是 `(B,N,5)`，mask 都是 `(B,N)`。code 相同时保留 voxel 输入的稳定字典序。

| 输入 | dtype / shape | 含义 |
|---|---|---|
| `coords` | float32 `(B,N,3)` | 居中并缩放坐标 |
| `features` | float32 `(B,N,2)` | energy fraction 与 hit-count feature |
| `mask` | bool `(B,N)` | 有效 token；`1≤N≤512` |
| output | float `(B,)` | signal logit |

padding 在每个 block 后重新置零，也不参与最终 mean/max。

## 4. 逐层结构

| 阶段 | 运算 | shape |
|---|---|---|
| 共享输入编码 | `Linear(5,128) → LayerNorm → GELU` | `(B,N,5) → (B,N,128)` |
| TCN block 0 | 两个 causal `Conv1d(128,128,k=3,d=1)`，各 LN/GELU/dropout，外层 residual | `(B,N,128)` |
| block 1–5 | 同结构，dilation `2,4,8,16,32` | `(B,N,128)` |
| 每顺序池化 | masked mean + max | `(B,N,128) → (B,256)` |
| 双顺序融合 | 拼接 | `(B,512)` |
| 分类头 | `Linear(512,192) → LN → GELU → Dropout → Linear(192,1)` | `(B,)` |

每次卷积显式仅在左侧 padding (d(k-1))，因此

$$
y_t=\sum_{j=0}^{k-1}W_jx_{t-dj}
$$

不会使用该排序的未来 token。一个 residual block 写为

$$
h'=h+\mathcal D(\mathrm{GELU}(\mathrm{LN}(\mathrm{Conv}_{d,2}(\mathcal D(\mathrm{GELU}(\mathrm{LN}(\mathrm{Conv}_{d,1}(h)))))))).
$$

六个 block、每个两层卷积的理论 receptive field 为

$$
1+2(k-1)\sum_{i=0}^{5}2^i=253\ \text{tokens}.
$$

参数量：input 1,024；每 block 99,072，共 594,432；head 99,073；合计 694,529。

## 5. YAML 与训练

| 参数 | 值 | 参数 | 值 |
|---|---:|---|---:|
| point bin / scale / cap | 15 / 1000 / 512 | hidden dim | 128 |
| blocks / kernel | 6 / 3 | dilation base | 2 |
| Hilbert bits | 10 | classifier / dropout | 192 / 0.10 |
| batch / epochs | 12 / 50 | lr / weight decay | 5e-4 / 1e-4 |
| clip / patience | 1.0 / 12 | seed / AMP | 42 / auto |

共享训练为 `BCEWithLogitsLoss`、AdamW、CosineAnnealingLR；以 validation AUC 选择 best、同时保存 last。balanced-class loader 交替两类；YAML 的 buffer 数值不是该路径实际 event-buffer shuffle 的证明。

## 6. 复杂度和显存

- 两种排序约 $O(BN\log N)$。
- 卷积时间为 (O(2BLNkd^2))，对 (N) 线性且可并行；激活空间约 (O(BLNd))。
- 六层 full-channel Conv1d 权重与 activation 是主要成本；它没有 depthwise/separable 优化。
- receptive field 253 小于最大 512；mean/max head 提供全事件汇聚，但卷积 token 在 head 前不都拥有完整序列上下文。

## 7. 与 TCN 文献的边界

这是 **dilated TCN-style** 点云序列分类器。Bai 等人的论文评估通用序列任务与标准 residual TCN；本实现使用 Hilbert 点云、每 block 两个同 dilation 卷积、LayerNorm/GELU、双顺序共享权重和 event mean+max head，没有复现论文数据、优化 recipe 或 weight normalization。这里的“时间”只是空间曲线顺序，不是物理时间。

## 8. 运行与恢复

```bash
cd /home/wenyu/summer
RUN_ID=<timestamp-or-explicit-name>
tmux new-session -d -s "next-nontransformer-v2-${RUN_ID}" -n gpu-queue \
  "cd /home/wenyu/summer && bash 01_code/architectures/run_nontransformer_training_queue.sh --run-id ${RUN_ID}"
```

队列入口为 `python 01_code/architectures/seq_002_dilated_tcn/train_classification.py <config.snapshot.yaml>`。输出：

```text
02_models/checkpoints/<RUN_ID>/seq_002_dilated_tcn/attempt_001/{best.pt,last.pt}
03_training_runs/campaigns/<RUN_ID>/seq_002_dilated_tcn/attempt_001/
```

Ctrl-C 停止 queue pane；`--resume-queue` 跳过 `DONE`。失败会建立下一 attempt 并从头训练，保留所有日志；不是 checkpoint resume。

## 9. 训练结果（campaign 后填写）

| 状态 | epoch / best epoch | best val AUC / loss | 耗时 | early stop / retry | artifacts |
|---|---|---|---|---|---|
| `PRE-CAMPAIGN PLACEHOLDER`（实际状态见文末追加结果） | — | — | — | — | — |

此处只记录 train/validation。

## 10. 已知限制

- 人为序列化不具备置换不变性，旋转或量化边界变化会改变邻接。
- Trans-Hilbert 的 x/y 交换是项目 convention；双顺序仍不覆盖所有方向。
- 最大 receptive field、512 点截断和 causal 方向都可能限制全局拓扑表达。
- 没有 compiled convolution fusion，单 seed 不提供误差条。

## 11. 参考文献

1. Shaojie Bai, J. Zico Kolter, Vladlen Koltun, “An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling,” 2018, arXiv:1803.01271, [arXiv](https://arxiv.org/abs/1803.01271).
2. John Skilling, “Programming the Hilbert Curve,” *AIP Conference Proceedings* 707 (2004), DOI: 10.1063/1.1751381, [official article](https://doi.org/10.1063/1.1751381).
3. David Hilbert, “Über die stetige Abbildung einer Linie auf ein Flächenstück,” *Mathematische Annalen* 38 (1891), DOI: 10.1007/BF01199431, [official article](https://doi.org/10.1007/BF01199431).
4. NEXT Collaboration (J. Renner et al.), “Background rejection in NEXT using deep neural networks,” *JINST* 12 (2017) T01004, DOI: 10.1088/1748-0221/12/01/T01004, [official article](https://doi.org/10.1088/1748-0221/12/01/T01004).


<!-- campaign-result:20260803_200356:start -->
## Campaign `20260803_200356` 训练结果

本节由串行训练队列在训练成功后写入，只包含 train/validation 信息；本阶段没有读取 test split。

| 项目 | 实际值 |
|---|---|
| 状态 / attempt | `DONE` / `attempt_001` |
| 后端 | `pytorch` |
| 实际参数量 | 694,529 |
| 树数量 / 树节点数 | N/A / N/A |
| 完成 epoch / best epoch | 28 / 16 |
| best validation AUC | **0.927011** |
| best validation loss | 0.354707 |
| 总训练时间 | 00:39:07 |
| early stop | `true` |
| 失败重试 | `no` |
| Python / 框架 | `3.11.15` / `2.11.0+cu128` |
| 设备 | `NVIDIA GeForce RTX 5090` |
| best / last checkpoint | `/home/wenyu/summer/02_models/checkpoints/20260803_200356/seq_002_dilated_tcn/attempt_001/best.pt` / `/home/wenyu/summer/02_models/checkpoints/20260803_200356/seq_002_dilated_tcn/attempt_001/last.pt` |
| 训练日志 | `/home/wenyu/summer/03_training_runs/campaigns/20260803_200356/seq_002_dilated_tcn/attempt_001/history.json` |
<!-- campaign-result:20260803_200356:end -->
