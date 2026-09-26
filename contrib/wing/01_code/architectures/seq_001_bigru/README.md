# SEQ-001：Hilbert / Trans-Hilbert 双向 GRU

中文 | [English](README_EN.md)

## 1. 定位与研究假设

| 项目 | 值 |
|---|---|
| `architecture_id` | `seq_001_bigru` |
| checkpoint `model_name` | `HilbertBiGRUClassifier` |
| Python class | `next_alt.models.point_sequence.HilbertBiGRUClassifier` |
| registry `input_kind` | `sequence` |
| 任务 | `0nubb=1`、`Bi214=0` 二分类 |
| 输出 | `(B,)` 未校准 signal logit |
| 配置推导参数量 | **733,953** |

假设：空间填充曲线能将局部相邻的 3D voxel 尽量映射为相邻 token；共享 BiGRU 从两个互补扫描方向读取轨迹后，可以用门控递归状态表达长程拓扑，而不使用 Transformer/attention。

## 2. 数据与预处理

共享 reader 从 HDF5 `/MC/hits/table` 按连续 `event_id` 读取 `x/y/z/energy`。`0nubb_part_*` 为 label 1，`Bi_part_*` 为 label 0。完整相对文件路径作为 group；seed 42 做 file-level 0.8/0.1/0.1 切分。本阶段只读取 train 和 validation，每类每个 split 最多 100 个文件，validation 只用于 early stopping 和 best selection。

命中先按完整事件能量 (E) 做能量质心平移，再以 15 mm cell 合并；量化 voxel 中心再次按 voxel energy 居中。超过 512 个节点时保留最高能 512 个且不重新归一化 energy fraction。模型数值坐标为中心坐标/1000 mm，特征为

$$
[e_i^{\mathrm{voxel}}/E,\log(1+n_i^{\mathrm{rows}})].
$$

总能量和绝对位置不进入网络。

### 统一序列化

1. 对每个事件、每个 XYZ 轴分别用有效点包围盒线性量化到 10-bit 整数 `[0,1023]`。
2. 用 Skilling transpose algorithm 计算 3D Hilbert code，按 code 稳定升序排列。
3. Trans-Hilbert 在相同量化点上固定交换 x/y 后再计算同一 Hilbert code。这个 x/y 交换是本项目明确冻结的 convention，不宣称是论文唯一实现。
4. 两种排序由 `point_sequence.py` 的同一公共函数生成；SEQ-001、SEQ-002 与 SSM-001 不允许各自定制排序。

## 3. 输入、mask 和 shape

| 字段 | dtype / shape | 语义 |
|---|---|---|
| `coords` | float32 `(B,N,3)` | 居中 voxel 坐标/1000 mm |
| `features` | float32 `(B,N,2)` | energy fraction、`log1p(hit_count)` |
| `mask` | bool `(B,N)` | 有效点为 true；`1≤N≤512` |
| Hilbert / Trans-Hilbert | float `(B,N,5)` 各一条 | 排序后的 XYZ+feature |
| GRU output | float `(B,N,256)` | 每 token 前向 128 + 反向 128 |
| model output | float `(B,)` | signal logit |

两条序列的有效长度相同。`pack_padded_sequence` 保证 padding 不进入正向或反向 recurrence；恢复 padding 后再做 masked pooling。

## 4. 逐层架构和 forward

| 阶段 | 运算 | 输入 → 输出 |
|---|---|---|
| 共享 token encoder | `Linear(5,96) → LayerNorm → SiLU` | `(B,N,5) → (B,N,96)` |
| 共享 BiGRU layer 1 | hidden 128，双向 | `96 → 256` |
| 共享 BiGRU layer 2 | hidden 128，双向，层间 dropout 0.1 | `256 → 256` |
| 每个顺序池化 | masked mean + max | `(B,N,256) → (B,512)` |
| 双顺序融合 | 拼接 Hilbert 与 Trans-Hilbert | `512 + 512 → 1024` |
| head | `Linear(1024,256) → LN → SiLU → Dropout → Linear(256,1)` | `(B,)` |

对方向 $d\in\{\rightarrow,\leftarrow\}$，GRU 使用

$$
z_t=\sigma(W_zx_t+U_zh_{t-1}),\quad
r_t=\sigma(W_rx_t+U_rh_{t-1}),
$$
$$
\tilde h_t=\tanh(W_hx_t+U_h(r_t\odot h_{t-1})),\quad
h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde h_t.
$$

两层两个方向共享于两种空间曲线，但每条曲线独立运行 recurrence。参数分解：token encoder 768；两层双向 GRU 470,016；head 263,169；总计 733,953。

## 5. YAML 和训练参数

| 类别 | 参数 | 值 |
|---|---|---:|
| 表示 | bin / scale / max points | 15 mm / 1000 mm / 512 |
| 模型 | `embedding_dim` / `hidden_dim` | 96 / 128 |
| 模型 | `num_layers` / `hilbert_bits` | 2 / 10 |
| 模型 | `classifier_dim` / dropout | 256 / 0.10 |
| 训练 | batch / epochs / lr | 16 / 50 / 5e-4 |
| 训练 | weight decay / clip / patience | 1e-4 / 1.0 / 12 |
| 训练 | seed / AMP | 42 / auto |

训练合同为 `BCEWithLogitsLoss`、AdamW、`CosineAnnealingLR(T_max=50)`；按 validation AUC 选 best 并保存 last。balanced-class 路径交替读取两类；配置中的 buffer size 不等同于实际执行 event-buffer shuffle。

## 6. 复杂度与显存

- 两次 Hilbert 排序约 $O(BN\log N)$。
- 双顺序、双向、两层 GRU 的主要计算约 (O(BN L h(h+d)))，对序列长度线性。
- 训练需保存两条序列各层的 recurrent activation，约 (O(BNLh))；head 的 1024 维融合不是主要瓶颈。
- packing 会把 lengths 同步到 CPU，可能带来小的 host/device synchronization 开销。

## 7. 与论文方法的差异边界

本模型是 **GRU-based serialized point baseline**。GRU 论文提出的是机器翻译 encoder-decoder，并没有 Hilbert 点云、双向层、voxel 能量特征或本分类头；本实现不声称复现其任务。Hilbert 排序也只是人为定义一维因果结构，BiGRU 的反向分支减弱但不能消除顺序偏置。没有 attention、预训练或语言 decoder。

## 8. tmux、路径、停止与恢复

```bash
cd /home/wenyu/summer
RUN_ID=<timestamp-or-explicit-name>
tmux new-session -d -s "next-nontransformer-v2-${RUN_ID}" -n gpu-queue \
  "cd /home/wenyu/summer && bash 01_code/architectures/run_nontransformer_training_queue.sh --run-id ${RUN_ID}"
```

队列调用 `python 01_code/architectures/seq_001_bigru/train_classification.py <config.snapshot.yaml>`。产物位于：

```text
02_models/checkpoints/<RUN_ID>/seq_001_bigru/attempt_001/{best.pt,last.pt}
03_training_runs/campaigns/<RUN_ID>/seq_001_bigru/attempt_001/
```

向 `gpu-queue` 发送 Ctrl-C 可停止。`--resume-queue` 仅跳过 `DONE`；失败后创建新 attempt 并从 epoch 1 开始，历史 attempt 不覆盖，不能把 `last.pt` 描述为断点续训。

## 9. 训练结果（campaign 后追加）

| 状态 | epoch / best epoch | best val AUC / loss | 耗时 | early stop / retry | artifacts |
|---|---|---|---|---|---|
| `PRE-CAMPAIGN PLACEHOLDER`（实际状态见文末追加结果） | — | — | — | — | — |

这里只放 train/validation 结果。

## 10. 已知限制

- Hilbert/Trans-Hilbert 是人为顺序；每事件 min-max 量化和 code tie 都可能改变局部邻接。
- x/y 交换的 Trans-Hilbert 是本项目 convention，不是唯一标准。
- 512 点截断可能丢失低能分支；双向 GRU 的顺序时间为线性但无法并行 scan。
- 坐标输入没有旋转不变/等变保证，单 seed 结果不含统计不确定性。

## 11. 参考文献

1. Kyunghyun Cho, Bart van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio, “Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation,” EMNLP 2014, DOI: 10.3115/v1/D14-1179, [ACL Anthology](https://aclanthology.org/D14-1179/).
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
| 实际参数量 | 733,953 |
| 树数量 / 树节点数 | N/A / N/A |
| 完成 epoch / best epoch | 32 / 20 |
| best validation AUC | **0.936488** |
| best validation loss | 0.335716 |
| 总训练时间 | 00:44:49 |
| early stop | `true` |
| 失败重试 | `no` |
| Python / 框架 | `3.11.15` / `2.11.0+cu128` |
| 设备 | `NVIDIA GeForce RTX 5090` |
| best / last checkpoint | `/home/wenyu/summer/02_models/checkpoints/20260803_200356/seq_001_bigru/attempt_001/best.pt` / `/home/wenyu/summer/02_models/checkpoints/20260803_200356/seq_001_bigru/attempt_001/last.pt` |
| 训练日志 | `/home/wenyu/summer/03_training_runs/campaigns/20260803_200356/seq_001_bigru/attempt_001/history.json` |
<!-- campaign-result:20260803_200356:end -->
