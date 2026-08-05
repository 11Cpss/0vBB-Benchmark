# MIXER-001：三视图投影 MLP-Mixer

中文 | [English](README_EN.md)

## 1. 定位与研究假设

MIXER-001 检验“不使用卷积或 self-attention，仅在 patch/token 轴和通道轴交替使用
MLP，能否从 NEXT 的 XY/XZ/YZ 投影学习长程拓扑关系”。它不是 Transformer，也不
包含 attention、位置嵌入或卷积残差 backbone；唯一的卷积算子是无重叠 patch
embedding，其作用等价于每个 patch 的共享线性投影。

| 项目 | 定义 |
|---|---|
| `architecture_id` | `mixer_001_projection_mlp_mixer` |
| checkpoint `model_name` | `ProjectionMLPMixerClassifier` |
| Python 实现 | `next_alt.models.mixer_sparse.ProjectionMLPMixerClassifier` |
| registry `input_kind` | `projection2d` |
| 任务 | `0nubb` signal（1）与 `Bi214` background（0）二分类 |
| 输出 | 每事件一个未校准 signal logit，shape `(B,)` |
| 默认可训练参数 | **539,073** |

## 2. 原始数据、划分与预处理

共享读取器从每个 HDF5 文件的 `/MC/hits/table` 读取 `event_id,x,y,z,energy`，把连续
的相同 `event_id` 行组成一个事件。父目录 `0nubb_part_*` 映射为标签 1，`Bi_part_*`
映射为标签 0。划分以完整相对 HDF5 路径为 group，seed 42 和比例
`[0.8,0.1,0.1]` 做确定性 file-level split；本阶段只构造 train 与 validation loader，
第三个保留 split 不会被读取。每类在 train/validation 发现过程中最多选择 100 个文件。

对一个事件，令总能量 $E=\sum_i e_i$，先计算能量质心并平移所有 hit，再以对称原点
做离散化：


$$
\mathbf c=\frac{\sum_i e_i\mathbf r_i}{E},\qquad
\mathbf o=(-1920,-1920,-1920)\;\mathrm{mm},\qquad
\mathbf b_i=\left\lfloor\frac{(\mathbf r_i-\mathbf c)-\mathbf o}{30\;\mathrm{mm}}\right\rfloor.
$$

只有三个索引都在 `[0,127]` 的 hit 才进入投影。三个通道按固定顺序累加：


$$
P_{xy}[b_y,b_x] {+}{=}100e_i/E,\quad
P_{xz}[b_z,b_x] {+}{=}100e_i/E,\quad
P_{yz}[b_z,b_y] {+}{=}100e_i/E.
$$

归一化分母始终是完整事件能量；范围外能量不会被重新归一化。`center_projection=true`
保证总能量和绝对 detector 位置都不进入模型；仍保留相对于事件能量质心的拓扑位置。

| batch 字段 | dtype / shape | 语义 |
|---|---|---|
| `projections` | float32 `(B,3,128,128)` | 依次为 XY、XZ、YZ 的能量分数图，已乘 100 |
| `label` | float32 `(B,)` | 训练目标；不进入 forward 的特征流 |
| 输出 | floating `(B,)` | BCEWithLogitsLoss 使用的 signal logits |

该固定尺寸输入没有 mask。

## 3. 逐层架构

默认 patch size 为 16，因此每个平面方向有 8 个 patch，token 数
$S=8\times8=64$，embedding 宽度 $C=128$。

| 阶段 | 运算 | 输入 → 输出 |
|---|---|---|
| patch embedding | `Conv2d(3,128,kernel=16,stride=16)` | `(B,3,128,128) → (B,128,8,8)` |
| flatten | 展平空间轴并转置 | `(B,128,8,8) → (B,64,128)` |
| Mixer block ×6：token 分支 | `LayerNorm(C)`；转置；`Linear(64,32) → GELU → Dropout → Linear(32,64) → Dropout`；残差相加 | `(B,64,128) → (B,64,128)` |
| Mixer block ×6：channel 分支 | `LayerNorm(C) → Linear(128,256) → GELU → Dropout → Linear(256,128) → Dropout`；残差相加 | `(B,64,128) → (B,64,128)` |
| event pooling | final `LayerNorm(128)` 后对 64 tokens 求 mean | `(B,64,128) → (B,128)` |
| head | `Linear(128,128) → GELU → Dropout(0.1) → Linear(128,1)` | `(B,128) → (B,1)` |
| 输出 | squeeze 最后一维 | `(B,1) → (B,)` |

一个 block 的数据流为


$$
U=X+\operatorname{TransposeBack}\left(
\operatorname{MLP}_{token}(\operatorname{Transpose}(\operatorname{LN}(X)))
\right),
\qquad
Y=U+\operatorname{MLP}_{channel}(\operatorname{LN}(U)).
$$

没有 attention softmax、query/key/value、class token 或显式位置编码。patch 的固定展平
位置由 token-mixing Linear 的参数索引隐式区分。

## 4. 参数量、复杂度与显存

默认参数分解：patch embedding 98,432；每个 Mixer block 70,624，六个共 423,744；
final norm 256；分类头 16,641；总计 **539,073**。

令 batch 为 (B)、token 数 (S)、embedding 宽度 (C)、token/channel 隐层宽度分别
为 (D_t,D_c)、block 数为 (L)。Mixer 主干时间复杂度为
(O(BL(CSD_t+SCD_c)))，activation 空间为 (O(BSC))，没有 attention 的
(O(S^2)) score matrix。默认显存主要来自六层 `(B,64,128)` activation 和 channel
MLP 的 `(B,64,256)` 中间量，而不是原始投影。

## 5. 冻结配置

权威参数在 [config.yaml](config.yaml)。表示参数为 grid 128、bin 30 mm、质心居中、
对称原点 `[-1920,-1920,-1920]`、input scale 100。模型参数为
`input_channels=3`、`patch_size=16`、
`embedding_dim=128`、`depth=6`、`token_mlp_dim=32`、`channel_mlp_dim=256`、
`classifier_dim=128`、`dropout=0.10`。

数据参数为每类最多 100 文件、split seed 42、fractions `[0.8,0.1,0.1]`、
`num_workers=0`、balanced training 和 buffer 512。训练参数为 batch 16、50 epochs、
AdamW learning rate `1e-3`、weight decay `1e-4`、gradient clip 1.0、patience 12、
min delta 0、seed 42、非 deterministic、AMP auto。loss 为 BCEWithLogitsLoss，scheduler
为 `CosineAnnealingLR(T_max=50)`，best checkpoint 只按 validation AUC 选择。

## 6. 与原论文的差异和命名边界

这是 **MLP-Mixer-inspired projection classifier**，不是论文原配置复现：

- 输入是三个具有不同几何含义的 detector 投影通道，而不是自然图像 RGB；
- 采用 128×128 输入、16×16 patch、6 blocks、128 宽度，远小于论文常见配置；
- 没有论文预训练数据、增强、优化 recipe 或模型规模；
- 三视图在 patch embedding 就线性混合，不是独立 view encoder；
- 事件质心居中和能量分数预处理是本项目专属设计。

因此结果只能归因于此处实现，不能声称复现论文精度或能力。

## 7. 参考文献

- Ilya Tolstikhin, Neil Houlsby, Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai,
  Thomas Unterthiner, Jessica Yung, Andreas Steiner, Daniel Keysers, Jakob Uszkoreit,
  Mario Lucic, and Alexey Dosovitskiy, “MLP-Mixer: An all-MLP Architecture for
  Vision,” *Advances in Neural Information Processing Systems 34*, 2021,
  arXiv:2105.01601. [arXiv](https://arxiv.org/abs/2105.01601)；无 DOI。

## 8. tmux campaign、产物与恢复

本模型只能作为串行 campaign 的第 5 项启动：

```bash
RUN_ID=<timestamp-or-explicit-name>
tmux new-session -d -s "next-nontransformer-v2-${RUN_ID}" -n gpu-queue \
  "cd /home/wenyu/summer && bash 01_code/architectures/run_nontransformer_training_queue.sh --run-id ${RUN_ID}"
tmux new-window -t "next-nontransformer-v2-${RUN_ID}" -n monitor \
  "cd /home/wenyu/summer && bash 01_code/architectures/monitor_nontransformer_training.sh --run-id ${RUN_ID}"
```

实现接入后，模型专属入口是 `train_classification.py`，但正式训练由 queue 调用其 campaign
config snapshot。attempt 产物必须位于：

- `02_models/checkpoints/<RUN_ID>/mixer_001_projection_mlp_mixer/attempt_NNN/best.pt`；
- 同目录 `last.pt`；
- `03_training_runs/campaigns/<RUN_ID>/mixer_001_projection_mlp_mixer/attempt_NNN/`
  下的 `stdout.log`、`config.snapshot.yaml`、`epochs.csv`、`history.json`、`history.png`。

停止训练应向 `gpu-queue` window 发送 Ctrl-C 并保留 session/日志。恢复使用同一 RUN_ID
加 `--resume-queue`；DONE 会跳过，FAILED/PENDING 建立下一 attempt 并从头训练，不能把
`last.pt` 描述为真正断点续训。任何 attempt 都不得覆盖已有产物。

## 9. 已知限制

- 2-D 投影丢失三维 hit 对应关系，MLP 无法恢复已丢失信息。
- 早期混合三视图可能掩盖 plane-specific 统计差异。
- 质心居中移除绝对 detector 位置，也可能去掉与边界/漂移相关的有用信息。
- token mixer 固定依赖 64-token 网格；改变 grid 或 patch size 会改变权重 shape。
- 全局 token mixing 没有局部卷积的归纳偏置，数据量较小时可能更易过拟合。

## 10. 训练结果（campaign 完成后填写）

训练前占位状态：**PENDING**（实际状态见文末追加结果）。训练启动前不伪造结果；campaign 完成后只填写实际参数量、CUDA/
PyTorch/AMP 环境、完成 epoch、best epoch、best validation AUC/loss、总耗时、best/last
checkpoint、日志路径、early-stop 和 retry/attempt 情况；本阶段不记录保留 split 指标。


<!-- campaign-result:20260803_200356:start -->
## Campaign `20260803_200356` 训练结果

本节由串行训练队列在训练成功后写入，只包含 train/validation 信息；本阶段没有读取 test split。

| 项目 | 实际值 |
|---|---|
| 状态 / attempt | `DONE` / `attempt_001` |
| 后端 | `pytorch` |
| 实际参数量 | 539,073 |
| 树数量 / 树节点数 | N/A / N/A |
| 完成 epoch / best epoch | 19 / 7 |
| best validation AUC | **0.896199** |
| best validation loss | 0.421741 |
| 总训练时间 | 00:09:02 |
| early stop | `true` |
| 失败重试 | `no` |
| Python / 框架 | `3.11.15` / `2.11.0+cu128` |
| 设备 | `NVIDIA GeForce RTX 5090` |
| best / last checkpoint | `/home/wenyu/summer/02_models/checkpoints/20260803_200356/mixer_001_projection_mlp_mixer/attempt_001/best.pt` / `/home/wenyu/summer/02_models/checkpoints/20260803_200356/mixer_001_projection_mlp_mixer/attempt_001/last.pt` |
| 训练日志 | `/home/wenyu/summer/03_training_runs/campaigns/20260803_200356/mixer_001_projection_mlp_mixer/attempt_001/history.json` |
<!-- campaign-result:20260803_200356:end -->
