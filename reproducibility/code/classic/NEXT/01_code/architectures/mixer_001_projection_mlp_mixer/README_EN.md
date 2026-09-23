# MIXER-001: Three-Projection MLP-Mixer

[中文](README.md) | English

## 1. Positioning and hypothesis

MIXER-001 asks whether alternating token-axis and channel-axis MLPs can learn long-range NEXT
topology from the fixed XY/XZ/YZ projections without convolutional residual backbones or
self-attention. Its only convolution is a non-overlapping patch embedding, equivalent to a
shared linear projection of each patch.

| Item | Definition |
|---|---|
| `architecture_id` | `mixer_001_projection_mlp_mixer` |
| checkpoint `model_name` | `ProjectionMLPMixerClassifier` |
| Python class | `next_alt.models.mixer_sparse.ProjectionMLPMixerClassifier` |
| registry `input_kind` | `projection2d` |
| Task | Binary `0nubb` signal (1) versus `Bi214` background (0) classification |
| Output | One uncalibrated signal logit per event, shape `(B,)` |
| Default trainable parameters | **539,073** |

## 2. Raw data, split, and preprocessing

The shared reader consumes `event_id,x,y,z,energy` from `/MC/hits/table` and groups contiguous
rows with the same event ID. `0nubb_part_*` maps to label 1 and `Bi_part_*` to label 0. The
complete relative HDF5 path is the group for a deterministic file-level split with seed 42 and
fractions `[0.8,0.1,0.1]`. This stage constructs only train and validation loaders; the third
reserved split is never read. Discovery selects at most 100 files per class for train/validation.

For total event energy $E=\sum_i e_i$, first translate every hit by the energy centroid, then
use symmetric origin $\mathbf o=(-1920,-1920,-1920)$ mm and 30 mm bins:


$$
\mathbf c=\sum_i e_i\mathbf r_i/E,\qquad
\mathbf b_i=\left\lfloor((\mathbf r_i-\mathbf c)-\mathbf o)/(30\;\mathrm{mm})\right\rfloor.
$$

Only hits whose three indices lie in `[0,127]` enter the maps:


$$
P_{xy}[b_y,b_x]{+}{=}100e_i/E,\quad P_{xz}[b_z,b_x]{+}{=}100e_i/E,\quad
P_{yz}[b_z,b_y]{+}{=}100e_i/E.
$$

The denominator remains the complete-event energy, so out-of-range energy is not hidden by
renormalisation. `center_projection=true` ensures that neither total energy nor absolute
detector position enters the model; topology relative to the event energy centroid remains.

| Batch field | dtype / shape | Meaning |
|---|---|---|
| `projections` | float32 `(B,3,128,128)` | XY, XZ, YZ energy-fraction maps, scaled by 100 |
| `label` | float32 `(B,)` | Target, not a forward input feature |
| output | floating `(B,)` | Signal logits consumed by BCEWithLogitsLoss |

No mask is needed for the fixed-size representation.

## 3. Layer-by-layer architecture

Patch size 16 gives $S=8\times8=64$ tokens of width $C=128$.

| Stage | Operation | Input → output |
|---|---|---|
| Patch embedding | `Conv2d(3,128,kernel=16,stride=16)` | `(B,3,128,128) → (B,128,8,8)` |
| Flatten | Flatten spatial axes and transpose | `(B,128,8,8) → (B,64,128)` |
| Token branch ×6 | `LayerNorm`; transpose; `Linear(64,32) → GELU → Dropout → Linear(32,64) → Dropout`; residual | `(B,64,128) → (B,64,128)` |
| Channel branch ×6 | `LayerNorm → Linear(128,256) → GELU → Dropout → Linear(256,128) → Dropout`; residual | `(B,64,128) → (B,64,128)` |
| Event pool | final `LayerNorm(128)`, mean over 64 tokens | `(B,64,128) → (B,128)` |
| Head | `Linear(128,128) → GELU → Dropout(0.1) → Linear(128,1)` | `(B,128) → (B,1)` |
| Output | Squeeze | `(B,1) → (B,)` |

For a block,


$$
U=X+T^{-1}(\operatorname{MLP}_{token}(T(\operatorname{LN}(X)))),\qquad
Y=U+\operatorname{MLP}_{channel}(\operatorname{LN}(U)).
$$

There are no attention scores, queries/keys/values, class token, or explicit positional
embedding. Parameter indices in the token MLP implicitly distinguish fixed patch positions.

## 4. Parameters, complexity, and memory

The default count is patch embedding 98,432; six blocks at 70,624 each = 423,744; final norm
256; head 16,641; total **539,073**. For batch (B), tokens (S), width (C), hidden widths
(D_t,D_c), and depth (L), the backbone costs (O(BL(CSD_t+SCD_c))) time and
$O(BSC)$ activation storage. It creates no $S\times S$ attention matrix. The six token
activations and `(B,64,256)` channel-MLP intermediates are the expected memory bottleneck.

## 5. Frozen configuration

[config.yaml](config.yaml) is authoritative: projection grid/bin/origin/scale are
128 / 30 mm / `[-1920,-1920,-1920]` / 100, with centroiding enabled. Model values are
channels 3, patch 16, width 128,
depth 6, token hidden 32, channel hidden 256, classifier 128, dropout 0.10. Data uses at most
100 files/class, seed 42, fractions `[0.8,0.1,0.1]`, workers 0, balanced training, buffer 512.
Training uses batch 16, 50 epochs, BCEWithLogitsLoss, AdamW at `1e-3`, weight decay `1e-4`,
CosineAnnealingLR, clip 1.0, patience 12, seed 42, and AMP auto. Best selection uses validation
AUC only.

## 6. Paper boundary

This is an **MLP-Mixer-inspired projection classifier**, not a reproduction. It uses three
detector projections instead of RGB photographs, a much smaller 128×128/16-patch/6-block
configuration, no paper pretraining or augmentation recipe, early view mixing, and
NEXT-specific centroid/energy-fraction preprocessing. No paper-scale accuracy or capability
is claimed.

## 7. Reference

- Ilya Tolstikhin et al., “MLP-Mixer: An all-MLP Architecture for Vision,” *NeurIPS 34*,
  2021, arXiv:2105.01601. [Official arXiv](https://arxiv.org/abs/2105.01601); no DOI.

## 8. tmux campaign and artifacts

It runs fifth in the single-GPU queue:

```bash
RUN_ID=<timestamp-or-explicit-name>
tmux new-session -d -s "next-nontransformer-v2-${RUN_ID}" -n gpu-queue \
  "cd /home/wenyu/summer && bash 01_code/architectures/run_nontransformer_training_queue.sh --run-id ${RUN_ID}"
tmux new-window -t "next-nontransformer-v2-${RUN_ID}" -n monitor \
  "cd /home/wenyu/summer && bash 01_code/architectures/monitor_nontransformer_training.sh --run-id ${RUN_ID}"
```

Best/last checkpoints belong under
`02_models/checkpoints/<RUN_ID>/mixer_001_projection_mlp_mixer/attempt_NNN/`; logs, snapshot,
CSV, JSON, and plot belong under
`03_training_runs/campaigns/<RUN_ID>/mixer_001_projection_mlp_mixer/attempt_NNN/`. Stop with
Ctrl-C in `gpu-queue` while preserving the session. `--resume-queue` skips DONE entries and
starts FAILED/PENDING in a new attempt from scratch; `last.pt` is not true checkpoint resume.
Existing attempts must never be overwritten.

## 9. Limitations

Projection loses 3-D correspondences; early view mixing can hide plane-specific statistics;
centroiding removes detector position that might contain useful boundary/drift information;
token weights are tied to exactly 64 positions; and an all-MLP model lacks convolutional
locality bias and may overfit limited data.

## 10. Training result (fill only after the campaign)

Pre-campaign placeholder: **PENDING** (see the appended result for actual status). After real training, record actual parameters/environment, completed and
best epochs, best validation AUC/loss, duration, checkpoint/log paths, early stop, and attempts.
No metric from the reserved split belongs in this stage.


<!-- campaign-result:20260803_200356:start -->
## Campaign `20260803_200356` training result

This section is written after successful serial training. It contains train/validation information only; the test split was not read in this stage.

| Item | Observed value |
|---|---|
| Status / attempt | `DONE` / `attempt_001` |
| Backend | `pytorch` |
| Trainable parameters | 539,073 |
| Trees / tree nodes | N/A / N/A |
| Completed / best epoch | 19 / 7 |
| Best validation AUC | **0.896199** |
| Best validation loss | 0.421741 |
| Training time | 00:09:02 |
| Early stopped | `true` |
| Failed-attempt retry | `no` |
| Python / framework | `3.11.15` / `2.11.0+cu128` |
| Device | `NVIDIA GeForce RTX 5090` |
| Best / last checkpoint | `/home/wenyu/summer/02_models/checkpoints/20260803_200356/mixer_001_projection_mlp_mixer/attempt_001/best.pt` / `/home/wenyu/summer/02_models/checkpoints/20260803_200356/mixer_001_projection_mlp_mixer/attempt_001/last.pt` |
| Training history | `/home/wenyu/summer/03_training_runs/campaigns/20260803_200356/mixer_001_projection_mlp_mixer/attempt_001/history.json` |
<!-- campaign-result:20260803_200356:end -->
