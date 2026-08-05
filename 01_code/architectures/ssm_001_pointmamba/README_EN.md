# SSM-001: Pure-PyTorch Selective-Scan PointMamba-style Classifier

[中文](README.md) | English

## 1. Identity and hypothesis

| Item | Value |
|---|---|
| `architecture_id` | `ssm_001_pointmamba` |
| checkpoint `model_name` | `PointMambaLiteClassifier` |
| Python class | `next_alt.models.point_sequence.PointMambaLiteClassifier` |
| registry `input_kind` | `sequence` |
| Task / output | NEXT `0nubb=1`, `Bi214=0`; one `(B,)` signal logit |
| Configuration-derived parameters | **316,993** |
| Backend | pure-PyTorch chunk-vectorized diagonal selective scan |

The hypothesis is that Hilbert and Trans-Hilbert can turn a sparse 3-D track into locally coherent sequences, after which input-dependent state-space parameters propagate global topology with linear sequence complexity. Reading all Hilbert tokens before Trans-Hilbert also lets the second scan inherit global state from the first.

There is no Transformer, self-attention, `mamba_ssm`, Triton, or custom CUDA extension.

## 2. HDF5, labels, split, and voxels

The shared reader loads `/MC/hits/table`, validates and groups contiguous `event_id` rows, and uses `x/y/z/energy`. `0nubb_part_*` maps to label 1 and `Bi_part_*` to label 0. Complete relative paths are groups in a stable seed-42 file-level 0.8/0.1/0.1 split. This stage builds train and validation only, at most 100 files per class in each; validation serves early stopping and validation-AUC best selection only.

For hits,

$$
E=\sum_i e_i,\quad \mathbf c=\sum_i e_i\mathbf r_i/E,\quad
\mathbf q_i=\lfloor(\mathbf r_i-\mathbf c)/(15\ \mathrm{mm})\rfloor.
$$

Rows in one cell are merged; voxel centers are recentered by their voxel-energy centroid. Above 512 voxels, keep the highest-energy 512 with lexicographic cell tie-breaking. Coordinates are divided by 1000 mm and features are `[voxel_energy/E, log1p(merged_row_count)]`. Truncation does not renormalize energy fraction. Total energy and absolute position are excluded.

## 3. Inputs and shared space-filling curves

| Field | dtype / shape | Meaning |
|---|---|---|
| `coords` | float32 `(B,N,3)` | centered voxel XYZ / 1000 mm |
| `features` | float32 `(B,N,2)` | energy fraction and `log1p(hit_count)` |
| `mask` | bool `(B,N)` | valid points, `1≤N≤512` |
| each curve | float `(B,N,5)` | ordered XYZ plus features |
| compact SSM input | float `(B,2N,128)` | valid Hilbert, then valid Trans-Hilbert, then padding |
| output | float `(B,)` | signal logit |

All three serialized models share the implementation. Each valid per-event bounding box is quantized per axis to 10-bit `[0,1023]`; Skilling's transpose algorithm supplies 3-D Hilbert codes and a stable sort. Trans-Hilbert is explicitly defined here as an x/y swap after quantization followed by the same encoding. This is a frozen project convention, not a claim of a unique paper-defined axis rule.

The two valid sequences are compacted so no padding separates them. Learned channel-wise order indicators apply

$$
z^{(o)}=e^{(o)}\odot\gamma_o+\beta_o,\quad o\in\{H,T\}.
$$

## 4. Layer-by-layer architecture

| Stage | Operation | Input → output |
|---|---|---|
| Shared encoder | `Linear(5,128) → LayerNorm → SiLU` | `(B,N,5) → (B,N,128)` |
| Order indicator | two sets of scale(128) and shift(128) | each order `(B,N,128)` |
| Compact concatenation | valid H then valid T | `(B,2N,128)` |
| SSM block ×3 | pre-LN; input/gate `128→384×2`; causal depthwise Conv1d k=4; selective SSM; gate; `192→128`; residual | `(B,2N,128)` |
| Final norm | LayerNorm and mask | `(B,2N,128)` |
| Pool | masked mean plus max | `(B,256)` |
| Head | `256 → 160 → 1`, LN/SiLU/dropout | `(B,)` |

Each block uses inner 192, state 16, and delta rank 16. After causal depthwise convolution, token projections produce $\Delta_t,B_t,C_t$, while $A=-\exp(A_{\log})<0$:

$$
\bar A_t=\exp(\Delta_tA),\quad
h_t=\bar A_t\odot h_{t-1}+\Delta_t\odot B_t\odot x_t,
$$
$$
y_t=\sum_s C_{t,s}h_{t,:,s}+D\odot x_t.
$$

The result is multiplied by a SiLU gate, projected, dropped out, and added residually. Parameters: encoder 1,024; order indicators 512; 91,200 per block (273,600); final norm 256; head 41,601; total 316,993.

## 5. Pure-PyTorch scan fallback

This is not the official hardware-aware Mamba kernel. To avoid one Python iteration per token, recurrence is divided into 32-token chunks and evaluated inside a chunk with the diagonal prefix form

$$
P_t=\prod_{j\le t}\bar A_j,\qquad
h_t=P_t\left(h_0+\sum_{i\le t}u_i/P_i\right),\quad u_i=\Delta_iB_ix_i.
$$

The exact chunk-final state is carried forward. Scan arithmetic is float32 even under AMP. Within each chunk, cumulative `log(P)` is clamped to `[-60,0]` against overflow, so extreme learned transitions make this a guarded approximation. Conservative transition initialization and short chunks reduce that risk. The fallback materializes channel-by-state activations and cannot match the official fused kernel's speed or memory behavior.

## 6. Frozen YAML and training

Representation is 15 mm, scale 1000 mm, and 512 points. Model values are model/inner/state 128/192/16, delta rank 16, three blocks, conv kernel 4, 10 Hilbert bits, chunk 32, head 160, dropout 0.10. Training is batch 4, 50 epochs, learning rate $3\times10^{-4}$, weight decay $10^{-4}$, clip 1.0, patience 12, seed 42, and AMP auto with float32 scan arithmetic.

The runner uses BCE-with-logits, AdamW, cosine annealing, validation-AUC best selection, and a separate last checkpoint. Balanced loading alternates classes; the YAML buffer size is not evidence that event-buffer shuffling executes in that path.

## 7. Complexity and VRAM

Two sorts cost $O(BN\log N)$; SSM length is $S=2N\le1024$. Selective recurrence is $O(BSLd_{inner}d_{state})$, linear in $S$; depthwise convolution is $O(BSLkd_{inner})$. Pure-PyTorch autograd may retain $O(BSLd_{inner}d_{state})$ prefix/state activations rather than matching fused-kernel memory behavior. This is the main VRAM bottleneck and motivates batch 4. There is no quadratic attention matrix.

## 8. Boundary relative to Mamba and PointMamba

This is a **PointMamba-inspired lite fallback**, not an official reproduction:

- PointMamba uses FPS keypoints and kNN patches with a lightweight PointNet tokenizer; this model directly uses existing 15 mm voxel tokens.
- The paper's default encoder has twelve 384-dimensional blocks; this one has three 128-dimensional blocks.
- Hilbert, Trans-Hilbert, order indicators, concatenated scans, and a plain non-hierarchical SSM are retained conceptually.
- There is no masked pretraining, ShapeNet transfer, paper augmentation, or downstream recipe.
- Official Mamba uses a hardware-aware fused scan; this is a guarded chunked diagonal PyTorch recurrence.

Accordingly, results must use “inspired/lite/style,” never “official PointMamba reproduction.”

## 9. tmux, artifacts, stopping, and recovery

```bash
cd /home/wenyu/summer
RUN_ID=<timestamp-or-explicit-name>
tmux new-session -d -s "next-nontransformer-v2-${RUN_ID}" -n gpu-queue \
  "cd /home/wenyu/summer && bash 01_code/architectures/run_nontransformer_training_queue.sh --run-id ${RUN_ID}"
```

The queue invokes `python 01_code/architectures/ssm_001_pointmamba/train_classification.py <config.snapshot.yaml>`. Outputs go to `02_models/checkpoints/<RUN_ID>/ssm_001_pointmamba/attempt_001/` and `03_training_runs/campaigns/<RUN_ID>/ssm_001_pointmamba/attempt_001/`. Ctrl-C stops the queue pane. `--resume-queue` skips only `DONE`; failures retain their attempt and start the next attempt from epoch 1. A last checkpoint is not true resume.

## 10. Training result (campaign fills this)

| Status | epochs / best epoch | best val AUC / loss | duration | early stop / retry | artifacts |
|---|---|---|---|---|---|
| `PRE-CAMPAIGN PLACEHOLDER` (see appended result) | — | — | — | — | — |

Only train/validation results belong here.

## 11. Limitations

- Artificial curves, per-event min-max quantization, and the x/y Trans-Hilbert convention introduce orientation bias.
- The 512-node cap can remove low-energy branches; direct voxel tokens omit the paper's local patch tokenizer.
- Clipped chunk-prefix logs are not bitwise equivalent to step recurrence under extreme transitions.
- The fallback is slower and more memory-hungry than official kernels; batch 4 is not guaranteed optimal on every GPU.
- One seed and model size do not represent the Mamba/PointMamba families.

## 12. References

1. Albert Gu, Tri Dao, “Mamba: Linear-Time Sequence Modeling with Selective State Spaces,” 2023/2024, arXiv:2312.00752, [arXiv](https://arxiv.org/abs/2312.00752).
2. Dingkang Liang, Xin Zhou, Wei Xu, Xingkui Zhu, Zhikang Zou, Xiaoqing Ye, Xiao Tan, Xiang Bai, “PointMamba: A Simple State Space Model for Point Cloud Analysis,” *Advances in Neural Information Processing Systems 37*, 2024, arXiv:2402.10739, [NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/395371f778ebd4854b88521100af30ad-Abstract-Conference.html).
3. John Skilling, “Programming the Hilbert Curve,” *AIP Conference Proceedings* 707 (2004), DOI: 10.1063/1.1751381, [official article](https://doi.org/10.1063/1.1751381).
4. NEXT Collaboration (J. Renner et al.), “Background rejection in NEXT using deep neural networks,” *JINST* 12 (2017) T01004, DOI: 10.1088/1748-0221/12/01/T01004, [official article](https://doi.org/10.1088/1748-0221/12/01/T01004).


<!-- campaign-result:20260803_200356:start -->
## Campaign `20260803_200356` training result

This section is written after successful serial training. It contains train/validation information only; the test split was not read in this stage.

| Item | Observed value |
|---|---|
| Status / attempt | `DONE` / `attempt_001` |
| Backend | `pytorch` |
| Trainable parameters | 316,993 |
| Trees / tree nodes | N/A / N/A |
| Completed / best epoch | 24 / 12 |
| Best validation AUC | **0.928102** |
| Best validation loss | 0.407766 |
| Training time | 01:14:38 |
| Early stopped | `true` |
| Failed-attempt retry | `no` |
| Python / framework | `3.11.15` / `2.11.0+cu128` |
| Device | `NVIDIA GeForce RTX 5090` |
| Best / last checkpoint | `/home/wenyu/summer/02_models/checkpoints/20260803_200356/ssm_001_pointmamba/attempt_001/best.pt` / `/home/wenyu/summer/02_models/checkpoints/20260803_200356/ssm_001_pointmamba/attempt_001/last.pt` |
| Training history | `/home/wenyu/summer/03_training_runs/campaigns/20260803_200356/ssm_001_pointmamba/attempt_001/history.json` |
<!-- campaign-result:20260803_200356:end -->
