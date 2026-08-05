# SEQ-002: Hilbert / Trans-Hilbert Dilated TCN

[中文](README.md) | English

## 1. Identity

| Item | Value |
|---|---|
| `architecture_id` | `seq_002_dilated_tcn` |
| checkpoint `model_name` | `HilbertTCNClassifier` |
| Python class | `next_alt.models.point_sequence.HilbertTCNClassifier` |
| registry `input_kind` | `sequence` |
| Task / output | `0nubb=1`, `Bi214=0`; one `(B,)` signal logit |
| Configuration-derived parameters | **694,529** |

The hypothesis is that after a space-filling curve makes a locally coherent 1-D track, exponentially dilated convolutions can extract scales from endpoint blobs to the main trajectory in parallel, providing an attention-free alternative to recurrent and SSM models.

## 2. Raw data and representation

The shared reader assembles contiguous `event_id` rows from HDF5 `/MC/hits/table` using `x/y/z/energy`. `0nubb_part_*` is label 1 and `Bi_part_*` is label 0. Complete relative paths form file groups in a stable seed-42 0.8/0.1/0.1 split. Only train and validation are read here, with at most 100 files per class in each; validation only selects the best checkpoint and early stopping.

Each event is translated by its complete-energy centroid, aggregated in 15 mm cells, and recentered using the voxel-energy centroid. The highest-energy 512 voxels are retained when necessary without renormalizing energy fractions. Inputs are centered coordinate/1000 mm, voxel energy/complete-event energy, and `log1p(merged_row_count)`. Total energy and absolute detector position are excluded.

## 3. Hilbert contract and shapes

Valid XYZ axes are independently quantized to 10 bits within each event's bounding box. Skilling's algorithm supplies 3-D Hilbert codes and a stable sort. Trans-Hilbert swaps x/y after quantization and applies the same encoder. SEQ-001, SEQ-002, and SSM-001 share this exact function.

| Field | dtype / shape | Meaning |
|---|---|---|
| `coords` | float32 `(B,N,3)` | centered scaled coordinates |
| `features` | float32 `(B,N,2)` | energy fraction and hit-count feature |
| `mask` | bool `(B,N)` | valid token; `1≤N≤512` |
| each ordered sequence | float `(B,N,5)` | reordered coordinates and features |
| output | float `(B,)` | signal logit |

Padding is zeroed after every block and excluded from final pooling.

## 4. Architecture

| Stage | Operation | Shape |
|---|---|---|
| Shared input | `Linear(5,128) → LayerNorm → GELU` | `(B,N,5) → (B,N,128)` |
| TCN block 0 | two causal full-channel Conv1d, `k=3,d=1`, LN/GELU/dropout, outer residual | `(B,N,128)` |
| Blocks 1–5 | same, dilation `2,4,8,16,32` | `(B,N,128)` |
| Per-order pool | masked mean plus max | `(B,N,128) → (B,256)` |
| Fusion | concatenate orders | `(B,512)` |
| Head | `512 → 192 → 1`, LN/GELU/dropout | `(B,)` |

Explicit left padding implements

$$
y_t=\sum_{j=0}^{k-1}W_jx_{t-dj}.
$$

With two convolutions at every dilation, the receptive field is

$$
1+2(k-1)\sum_{i=0}^{5}2^i=253\text{ tokens}.
$$

Parameters are 1,024 input, 99,072 per block (594,432 total), and 99,073 head: 694,529 total.

## 5. Frozen YAML and training

Representation: 15 mm, scale 1000 mm, cap 512. Model: hidden 128, six blocks, kernel 3, dilation base 2, 10 Hilbert bits, head 192, dropout 0.10. Training: batch 12, 50 epochs, learning rate $5\times10^{-4}$, weight decay $10^{-4}$, clip 1.0, patience 12, seed 42, AMP auto.

The shared runner uses BCE-with-logits, AdamW, cosine annealing, validation-AUC best selection, and a separate last checkpoint. Balanced loading alternates classes; the buffer size in YAML does not prove event-buffer shuffling occurs in that path.

## 6. Complexity and memory

Sorting costs $O(BN\log N)$. Convolutions cost $O(2BLNkd^2)$, linear in sequence length and parallel across positions, with $O(BLNd)$ activations. Full-channel convolutions dominate. The 253-token receptive field is below the 512-token cap; event pooling is global, but pre-pooling tokens need not contain the entire sequence context.

## 7. Boundary relative to TCN literature

This is a **dilated TCN-style** point-sequence classifier. Bai et al. evaluate generic sequence tasks and a standard residual TCN; this project adds Hilbert points, two same-dilation convolutions per block, LayerNorm/GELU, shared dual orders, and a mean-plus-max event head. It does not reproduce the paper's data, optimizer recipe, or weight normalization. “Time” here means curve order, not physical time.

## 8. Running and recovery

```bash
cd /home/wenyu/summer
RUN_ID=<timestamp-or-explicit-name>
tmux new-session -d -s "next-nontransformer-v2-${RUN_ID}" -n gpu-queue \
  "cd /home/wenyu/summer && bash 01_code/architectures/run_nontransformer_training_queue.sh --run-id ${RUN_ID}"
```

The queue invokes `python 01_code/architectures/seq_002_dilated_tcn/train_classification.py <config.snapshot.yaml>`. Outputs live in `02_models/checkpoints/<RUN_ID>/seq_002_dilated_tcn/attempt_001/` and `03_training_runs/campaigns/<RUN_ID>/seq_002_dilated_tcn/attempt_001/`. Ctrl-C stops the pane. `--resume-queue` skips `DONE`; failures get a new from-scratch attempt, with all earlier attempts retained. This is not checkpoint resume.

## 9. Training result (campaign fills this)

| Status | epochs / best epoch | best val AUC / loss | duration | early stop / retry | artifacts |
|---|---|---|---|---|---|
| `PRE-CAMPAIGN PLACEHOLDER` (see appended result) | — | — | — | — | — |

Only train/validation results belong here.

## 10. Limitations

- Artificial ordering is not permutation invariant; rotation and quantization boundaries alter adjacency.
- The x/y Trans-Hilbert convention is project-specific and two scans do not cover every orientation.
- Receptive field, causal direction, and the 512-node cap can limit global topology.
- There is no compiled convolution fusion, and one seed gives no uncertainty estimate.

## 11. References

1. Shaojie Bai, J. Zico Kolter, Vladlen Koltun, “An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling,” 2018, arXiv:1803.01271, [arXiv](https://arxiv.org/abs/1803.01271).
2. John Skilling, “Programming the Hilbert Curve,” *AIP Conference Proceedings* 707 (2004), DOI: 10.1063/1.1751381, [official article](https://doi.org/10.1063/1.1751381).
3. David Hilbert, “Über die stetige Abbildung einer Linie auf ein Flächenstück,” *Mathematische Annalen* 38 (1891), DOI: 10.1007/BF01199431, [official article](https://doi.org/10.1007/BF01199431).
4. NEXT Collaboration (J. Renner et al.), “Background rejection in NEXT using deep neural networks,” *JINST* 12 (2017) T01004, DOI: 10.1088/1748-0221/12/01/T01004, [official article](https://doi.org/10.1088/1748-0221/12/01/T01004).


<!-- campaign-result:20260803_200356:start -->
## Campaign `20260803_200356` training result

This section is written after successful serial training. It contains train/validation information only; the test split was not read in this stage.

| Item | Observed value |
|---|---|
| Status / attempt | `DONE` / `attempt_001` |
| Backend | `pytorch` |
| Trainable parameters | 694,529 |
| Trees / tree nodes | N/A / N/A |
| Completed / best epoch | 28 / 16 |
| Best validation AUC | **0.927011** |
| Best validation loss | 0.354707 |
| Training time | 00:39:07 |
| Early stopped | `true` |
| Failed-attempt retry | `no` |
| Python / framework | `3.11.15` / `2.11.0+cu128` |
| Device | `NVIDIA GeForce RTX 5090` |
| Best / last checkpoint | `/home/wenyu/summer/02_models/checkpoints/20260803_200356/seq_002_dilated_tcn/attempt_001/best.pt` / `/home/wenyu/summer/02_models/checkpoints/20260803_200356/seq_002_dilated_tcn/attempt_001/last.pt` |
| Training history | `/home/wenyu/summer/03_training_runs/campaigns/20260803_200356/seq_002_dilated_tcn/attempt_001/history.json` |
<!-- campaign-result:20260803_200356:end -->
