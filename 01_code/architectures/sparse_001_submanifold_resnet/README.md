# SPARSE-001：纯 PyTorch Submanifold Sparse 3-D ResNet

中文 | [English](README_EN.md)

## 1. 定位与研究假设

SPARSE-001 检验：保留所有 occupied 15 mm voxels，只在活跃坐标之间进行 3×3×3
submanifold convolution，是否能在不构造稠密 3-D volume 的情况下学习 NEXT 轨迹拓扑。
模型不含 Transformer 或 attention；它的活跃坐标集合在所有卷积中保持不变。

| 项目 | 定义 |
|---|---|
| `architecture_id` | `sparse_001_submanifold_resnet` |
| checkpoint `model_name` | `SubmanifoldSparseResNetClassifier` |
| Python 实现 | `next_alt.models.mixer_sparse.SubmanifoldSparseResNetClassifier` |
| registry `input_kind` | `sparse3d` |
| 任务 | `0nubb`（1）与 `Bi214`（0）二分类 |
| 输出 | 每事件一个未校准 signal logit `(B,)` |
| 默认可训练参数 | **298,177** |
| backend | sorted integer hash + `searchsorted` 的纯 PyTorch fallback |

该 fallback 明确不是 MinkowskiEngine、spconv 或论文官方高性能 C++/CUDA kernel 的复现。

## 2. 原始 HDF5、file-level split 与无截断稀疏表示

共享读取器从 `/MC/hits/table` 读取 `event_id,x,y,z,energy`，聚合连续 event ID；目录
`0nubb_part_*` 映射标签 1，`Bi_part_*` 映射标签 0。完整相对 HDF5 路径是 group；seed
42、fractions `[0.8,0.1,0.1]` 决定 file-level split。本阶段只构造 train/validation
loader，不读取第三个保留 split；每类最多 100 文件。

令 $E=\sum_i e_i$，先用完整事件做能量中心化，再量化：


$$
\mathbf c=\frac{\sum_i e_i\mathbf r_i}{E},\qquad
\mathbf q_i=\left\lfloor\frac{\mathbf r_i-\mathbf c}{15\;\mathrm{mm}}\right\rfloor
\in\mathbb Z^3.
$$

相同 $\mathbf q$ 的 hit 合并为一个 active voxel $q$，特征为


$$
\mathbf f_q=\left[\frac{E_q}{E},\ \log(1+n_q)\right].
$$

`max_points=null`：不按能量裁剪 occupied voxels，能量分数和为 1（除浮点误差）。总能量
和绝对位置不输入模型；整数 $\mathbf q$ 只用于相对 27 邻域查找。

| batch 字段 | dtype / shape | mask 语义 |
|---|---|---|
| `voxel_coords` | int64 `(B,V,3)` | centered 15 mm lattice coordinates；pad 行数值无意义 |
| `voxel_features` | float32 `(B,V,2)` | `[energy_fraction, log1p(hit_count)]`；pad 行为 0 |
| `voxel_mask` | bool `(B,V)` | true 为 active voxel，false 为 batch padding；`V` 是 batch 内最大 active count |
| `label` | float32 `(B,)` | 监督标签，不进入模型特征 |
| 输出 | floating `(B,)` | signal logits |

数据集集成必须直接提供以上三个 `voxel_*` keys；不能把 `(B,2,D,H,W)` 稠密 volume
交给本模型，也不能复用 `max_points=512` 的截断点云。

## 3. Submanifold convolution 定义

令事件 active set 为 $\mathcal A\subset\mathbb Z^3$，offset set
$\mathcal D=\{-1,0,1\}^3$，共 27 个。对每个 $\mathbf x\in\mathcal A$：


$$
\mathbf g(\mathbf x)=\mathbf b+
\sum_{\boldsymbol\delta\in\mathcal D}
\mathbb 1[\mathbf x+\boldsymbol\delta\in\mathcal A]\,
\mathbf h(\mathbf x+\boldsymbol\delta)\mathbf W_{\boldsymbol\delta}.
$$

输出只写回 $\mathcal A$，因此普通 sparse convolution 会产生的新邻近 active sites 在
这里不会出现。实现对每个事件给整数坐标构造无碰撞线性 hash；把 27 个 offset 扩成
`(V,27,3)` 后一次展平并调用 `torch.searchsorted`，再以
`einsum('voc,ocd->vd')` 同时应用 27 组权重。每层都不会分配 detector-sized dense grid。

## 4. 逐层架构与准确 shape

每个 residual block 主路是
`SubMConv → LayerNorm → SiLU → Dropout → SubMConv → LayerNorm`，加 identity 或
Linear shortcut 后 SiLU；每步重新乘 `voxel_mask`。

| 阶段 | 运算 | 输入 → 输出 |
|---|---|---|
| sparse input | 只取 active rows | coords `(B,V,3)`，features `(B,V,2)` |
| stem | 27-neighbour SubMConv `2→24` → LN → SiLU | `(B,V,2) → (B,V,24)` |
| residual stage 0 | 1 block，`24→24→24`，identity shortcut | `(B,V,24) → (B,V,24)` |
| residual stage 1 | 1 block，`24→40→40`，`Linear(24,40)` shortcut | `(B,V,24) → (B,V,40)` |
| residual stage 2 | 1 block，`40→64→64`，`Linear(40,64)` shortcut | `(B,V,40) → (B,V,64)` |
| event pooling | active rows 的 masked mean 与 max 拼接 | `(B,V,64) → (B,128)` |
| head | `Linear(128,96) → SiLU → Dropout(0.1) → Linear(96,1)` | `(B,128) → (B,1)` |
| 输出 | squeeze | `(B,1) → (B,)` |

没有 stride 或 pooling 改变 active set；感受野仅通过连续 3×3×3 卷积扩大。

## 5. 参数量、复杂度和显存

参数分解：stem 1,368；stage 0/1/2 分别 31,248 / 70,360 / 182,720；分类头
12,481；总计 **298,177**。

对一个事件 $V$ 个 active voxels，一层需要 $O(V\log V)$ sort/hash lookup，以及最多
$O(27VC_{in}C_{out})$ 的 offset 矩阵乘。模型共 7 个 SubMConv，每层当前都会重新排序。
feature/padding activation 为 $O(BVC)$，而不是 dense grid 的 $O(BD^3C)$。27 个
offset 已在一个事件内完全向量化；预期瓶颈是仍保留的逐事件 Python loop、每层重复
sort/searchsorted，以及 active-count 差异造成的 batch padding。这套 fallback 追求无需
编译扩展即可训练，不追求生产吞吐。

## 6. 完整冻结配置

[config.yaml](config.yaml) 为权威来源。表示参数：`point_bin_size=15.0` mm、
`coordinate_scale=1000.0`（为跨表示 provenance 保留，整数 sparse coords 不使用该缩放）、
`max_points=null`。模型：`feature_dim=2`、`stage_channels=[24,40,64]`、
`stage_blocks=[1,1,1]`、`classifier_dim=96`、dropout 0.10。

数据：每类最多 100 文件、split seed 42、fractions `[0.8,0.1,0.1]`、workers 0、
balanced training、buffer 512。训练：batch 8、50 epochs、BCEWithLogitsLoss、AdamW
`5e-4`、weight decay `1e-4`、CosineAnnealingLR、clip 1.0、patience 12、min delta 0、
seed 42、非 deterministic、AMP auto；best 只按 validation AUC。

## 7. 与论文/官方 sparse backend 的差异

这是 **submanifold sparse-convolution-inspired ResNet with a PyTorch fallback**：

- 保留“输出 active sites 与输入相同”的 SubMConv 语义和每 offset 独立权重；
- 不使用 C++/CUDA hash table、kernel map cache、coordinate manager 或 fused kernels；
- batch 以 padded tensors 表示，并在 Python 中逐事件处理；27 个 offset 在事件内向量化；
- 没有 strided sparse convolution、层级下采样、U-Net 或 dense/sparse hybrid；
- LayerNorm、SiLU、通道宽度、全局 mean+max head 是本项目选择；
- 未实现论文或库的高性能、可扩展性与完整 API，禁止称为官方 kernel 复现。

## 8. 参考文献

- Benjamin Graham and Laurens van der Maaten, “Submanifold Sparse Convolutional
  Networks,” arXiv preprint arXiv:1706.01307, 2017.
  [arXiv](https://arxiv.org/abs/1706.01307)；无 DOI。
- Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, “Deep Residual Learning for
  Image Recognition,” *IEEE Conference on Computer Vision and Pattern Recognition
  (CVPR)*, 2016, pp. 770–778, arXiv:1512.03385,
  DOI [10.1109/CVPR.2016.90](https://doi.org/10.1109/CVPR.2016.90),
  [arXiv](https://arxiv.org/abs/1512.03385).

## 9. tmux campaign、路径和恢复

本模型是单 GPU 队列第 10 项：

```bash
RUN_ID=<timestamp-or-explicit-name>
tmux new-session -d -s "next-nontransformer-v2-${RUN_ID}" -n gpu-queue \
  "cd /home/wenyu/summer && bash 01_code/architectures/run_nontransformer_training_queue.sh --run-id ${RUN_ID}"
tmux new-window -t "next-nontransformer-v2-${RUN_ID}" -n monitor \
  "cd /home/wenyu/summer && bash 01_code/architectures/monitor_nontransformer_training.sh --run-id ${RUN_ID}"
```

产物为
`02_models/checkpoints/<RUN_ID>/sparse_001_submanifold_resnet/attempt_NNN/{best.pt,last.pt}`
以及
`03_training_runs/campaigns/<RUN_ID>/sparse_001_submanifold_resnet/attempt_NNN/` 内的 stdout、
config snapshot、CSV、JSON、PNG。停止时向 `gpu-queue` 发 Ctrl-C 并保留 session。
`--resume-queue` 跳过 DONE；FAILED/PENDING 在同 RUN_ID 建新 attempt 从头训练，不能把
`last.pt` 称为断点续训，旧 attempt 不可覆盖。

## 10. 已知限制

- 无截断意味着极大事件会扩大 padding、排序和训练时间；batch 8 不保证适合所有尾部事件。
- fallback 每层重复 kernel-map 工作，显著慢于缓存映射的 sparse CUDA library。
- 只有 27-neighbour、无下采样的浅层感受野，远距轨迹关系依赖多层传播与全局池化。
- center-before-floor 量化对跨越 voxel 边界的小扰动不连续。
- 平移不变的 adjacency 和中心化去掉绝对 detector 位置，可能同时去掉有用信息。

## 11. 训练结果（campaign 后填写）

训练前占位状态：**PENDING**（实际状态见文末追加结果）。真实 campaign 完成后只追加实际参数量/环境、完成和 best epoch、
best validation AUC/loss、耗时、best/last/log 路径、early-stop 与 retry/attempt；本阶段
不写保留 split 指标。


<!-- campaign-result:20260803_200356:start -->
## Campaign `20260803_200356` 训练结果

本节由串行训练队列在训练成功后写入，只包含 train/validation 信息；本阶段没有读取 test split。

| 项目 | 实际值 |
|---|---|
| 状态 / attempt | `DONE` / `attempt_001` |
| 后端 | `pytorch` |
| 实际参数量 | 298,177 |
| 树数量 / 树节点数 | N/A / N/A |
| 完成 epoch / best epoch | 22 / 10 |
| best validation AUC | **0.971691** |
| best validation loss | 0.280406 |
| 总训练时间 | 00:49:40 |
| early stop | `true` |
| 失败重试 | `no` |
| Python / 框架 | `3.11.15` / `2.11.0+cu128` |
| 设备 | `NVIDIA GeForce RTX 5090` |
| best / last checkpoint | `/home/wenyu/summer/02_models/checkpoints/20260803_200356/sparse_001_submanifold_resnet/attempt_001/best.pt` / `/home/wenyu/summer/02_models/checkpoints/20260803_200356/sparse_001_submanifold_resnet/attempt_001/last.pt` |
| 训练日志 | `/home/wenyu/summer/03_training_runs/campaigns/20260803_200356/sparse_001_submanifold_resnet/attempt_001/history.json` |
<!-- campaign-result:20260803_200356:end -->
