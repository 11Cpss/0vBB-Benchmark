# SEQ-001: Hilbert / Trans-Hilbert Bidirectional GRU

[中文](README.md) | English

## 1. Identity and hypothesis

| Item | Value |
|---|---|
| `architecture_id` | `seq_001_bigru` |
| checkpoint `model_name` | `HilbertBiGRUClassifier` |
| Python class | `next_alt.models.point_sequence.HilbertBiGRUClassifier` |
| registry `input_kind` | `sequence` |
| Task / output | `0nubb=1`, `Bi214=0`; one `(B,)` signal logit |
| Configuration-derived parameters | **733,953** |

The hypothesis is that a space-filling curve makes spatially nearby voxels frequently adjacent in a 1-D sequence, allowing a shared BiGRU to model long-range track topology from two complementary scans without Transformer attention.

## 2. Data and preprocessing

The shared reader groups contiguous `event_id` rows from HDF5 `/MC/hits/table` and uses `x/y/z/energy`. `0nubb_part_*` is label 1; `Bi_part_*` is label 0. Complete relative paths are file-level groups in a stable seed-42 0.8/0.1/0.1 split. This campaign reads train and validation only, at most 100 files per class in each; validation serves early stopping and best selection only.

Hits are translated by the complete-event energy centroid, merged in 15 mm cells, and recentered using the voxel-energy centroid. Above 512 voxels, the highest-energy 512 are retained without renormalizing energy fractions. Numerical coordinates are center/1000 mm; features are `[voxel_energy / complete_event_energy, log1p(merged_row_count)]`. Total energy and absolute position are excluded.

### Shared serialization

1. Per event and axis, map the valid bounding box to 10-bit integers `[0,1023]`.
2. Compute 3-D Hilbert codes with Skilling's transpose algorithm and stable-sort by code.
3. Trans-Hilbert uses the same quantized points after a fixed x/y-axis swap. This is the project's frozen convention, not a claim that the paper defines a unique implementation.
4. SEQ-001, SEQ-002, and SSM-001 call the same implementation; backbones cannot customize the orders.

## 3. Inputs, masks, and shapes

| Field | dtype / shape | Meaning |
|---|---|---|
| `coords` | float32 `(B,N,3)` | centered voxel coordinates / 1000 mm |
| `features` | float32 `(B,N,2)` | energy fraction and `log1p(hit_count)` |
| `mask` | bool `(B,N)` | valid nodes; `1≤N≤512` |
| each serialized input | float `(B,N,5)` | reordered XYZ plus features |
| GRU output | float `(B,N,256)` | forward 128 plus backward 128 |
| model output | float `(B,)` | signal logit |

Both orders have the same lengths. Packed sequences prevent padding from entering either recurrent direction; restored outputs use masked pooling.

## 4. Architecture and equations

| Stage | Operation | Input → output |
|---|---|---|
| Shared encoder | `Linear(5,96) → LayerNorm → SiLU` | `(B,N,5) → (B,N,96)` |
| Shared BiGRU layer 1 | hidden 128, bidirectional | `96 → 256` |
| Shared BiGRU layer 2 | hidden 128, bidirectional, dropout 0.1 | `256 → 256` |
| Per-order pool | masked mean plus max | `(B,N,256) → (B,512)` |
| Order fusion | concatenate two curves | `512 + 512 → 1024` |
| Head | `1024 → 256 → 1`, LN/SiLU/dropout | `(B,)` |

For each direction the GRU uses

$$
z_t=\sigma(W_zx_t+U_zh_{t-1}),\quad r_t=\sigma(W_rx_t+U_rh_{t-1}),
$$
$$
\tilde h_t=\tanh(W_hx_t+U_h(r_t\odot h_{t-1})),\quad
h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde h_t.
$$

The encoder and GRU weights are shared across orders, but recurrences run separately. Parameters: encoder 768; GRU 470,016; head 263,169; total 733,953.

## 5. Frozen YAML and training

Representation values are 15 mm, 1000 mm, and 512 points. Model values are embedding 96, hidden 128, two layers, 10 Hilbert bits, head 256, and dropout 0.10. Training uses batch 16, 50 epochs, learning rate $5\times10^{-4}$, weight decay $10^{-4}$, clip 1.0, patience 12, seed 42, and AMP auto.

The runner uses `BCEWithLogitsLoss`, AdamW, and cosine annealing. Best means maximum validation AUC; last is separate. Balanced loading alternates classes. The configured event buffer should not be claimed to execute in that balanced path.

## 6. Complexity and memory

Two sorts cost roughly $O(BN\log N)$. Two orders, directions, and layers have recurrent cost linear in $N$, approximately $O(BNLh(h+d))$. Training stores recurrent activations $O(BNLh)$. Packing lengths onto CPU introduces a small synchronization.

## 7. Method boundary

This is a **GRU-based serialized-point baseline**. The original GRU paper addresses machine translation and contains no Hilbert point cloud, bidirectional stack, voxel-energy feature, or this classifier. Hilbert ordering imposes an artificial 1-D structure; bidirectionality reduces but does not remove ordering bias. There is no attention, decoder, or pretraining.

## 8. tmux, paths, stop, and recovery

```bash
cd /home/wenyu/summer
RUN_ID=<timestamp-or-explicit-name>
tmux new-session -d -s "next-nontransformer-v2-${RUN_ID}" -n gpu-queue \
  "cd /home/wenyu/summer && bash 01_code/architectures/run_nontransformer_training_queue.sh --run-id ${RUN_ID}"
```

The queue calls `python 01_code/architectures/seq_001_bigru/train_classification.py <config.snapshot.yaml>`. Artifacts go to `02_models/checkpoints/<RUN_ID>/seq_001_bigru/attempt_001/` and `03_training_runs/campaigns/<RUN_ID>/seq_001_bigru/attempt_001/`. Ctrl-C stops the queue pane. `--resume-queue` skips `DONE`; a failure creates a new from-scratch attempt. `last.pt` is not checkpoint resume, and old attempts are immutable.

## 9. Training result (campaign fills this)

| Status | epochs / best epoch | best val AUC / loss | duration | early stop / retry | artifacts |
|---|---|---|---|---|---|
| `PRE-CAMPAIGN PLACEHOLDER` (see appended result) | — | — | — | — | — |

Only train/validation results belong here.

## 10. Limitations

- Hilbert ordering, per-event min-max quantization, and code ties can perturb locality.
- The x/y Trans-Hilbert convention is project-specific.
- The 512-node cap may discard low-energy branches; recurrent execution is not parallel scan.
- The representation is not rotation invariant/equivariant, and one seed has no uncertainty estimate.

## 11. References

1. Kyunghyun Cho, Bart van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio, “Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation,” EMNLP 2014, DOI: 10.3115/v1/D14-1179, [ACL Anthology](https://aclanthology.org/D14-1179/).
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
| Trainable parameters | 733,953 |
| Trees / tree nodes | N/A / N/A |
| Completed / best epoch | 32 / 20 |
| Best validation AUC | **0.936488** |
| Best validation loss | 0.335716 |
| Training time | 00:44:49 |
| Early stopped | `true` |
| Failed-attempt retry | `no` |
| Python / framework | `3.11.15` / `2.11.0+cu128` |
| Device | `NVIDIA GeForce RTX 5090` |
| Best / last checkpoint | `/home/wenyu/summer/02_models/checkpoints/20260803_200356/seq_001_bigru/attempt_001/best.pt` / `/home/wenyu/summer/02_models/checkpoints/20260803_200356/seq_001_bigru/attempt_001/last.pt` |
| Training history | `/home/wenyu/summer/03_training_runs/campaigns/20260803_200356/seq_001_bigru/attempt_001/history.json` |
<!-- campaign-result:20260803_200356:end -->
