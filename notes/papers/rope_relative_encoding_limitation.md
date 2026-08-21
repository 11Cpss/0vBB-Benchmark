# RoPE Underperforms on Real NEXT Data: Relative-Only Encoding vs. an Absolute-Extent Signal

## Summary

`transformer_007_sampled_hits_rope` (true rotary attention, `vivek/RoPE`
branch) trains to essentially chance-level test AUC (~0.51) on the real NEXT
dataset, even after fixing a real `rope_base` scaling bug. A same-data,
same-config control run with `coordinate_mlp` reached val AUC 0.93 within 3
epochs. The root cause is not a code bug: this classification task's
dominant signal is event spatial extent (how spread out an event's hits are
from their own centroid), which is an **absolute**-position quantity. Pure
rotary attention only encodes **relative** offsets between token pairs by
construction, so it has no direct channel to represent it, unlike the
additive `coordinate_mlp`/`fourier_xyz` encodings.

## Timeline of the investigation

1. **Initial concern**: a short real-data probe of `transformer_007_sampled_hits_rope`
   (`rope_base=10.0`, the value validated only against synthetic data) showed
   val AUC stuck near chance across 5 epochs, with val loss climbing
   monotonically (0.79 → 1.03 → 1.03 → 1.51).
2. **First hypothesis, confirmed**: `rope_base=10.0` was tuned against a
   synthetic dataset whose coordinates never exceeded ~0.2 in magnitude.
   Real post-tokenization coordinates (`sampled_hits`, `coordinate_scale=1000`,
   centered) have std ≈ 0.18–0.20 per axis but a much heavier tail — max
   |coord| up to 2.79, p99 at 0.90. At `rope_base=10`, the fastest rotation
   channel wraps ~4.5 full turns for outlier hits (`27.9 rad`), well into
   aliasing territory — consistent with the same collapse pattern seen in
   the original `rope_base` sweep on synthetic data at `rope_base >= 200`.
   Lowering to `rope_base=2.0` (also validated on synthetic data, AUC=1.0)
   keeps even the real tail well-behaved (`~5.6 rad`, <1 turn).
3. **Re-tested with `rope_base=2.0`**: val loss stabilized (no more runaway
   growth, converged toward ln(2) ≈ 0.693 — the loss of a model predicting
   the class prior) but AUC still never left chance. A full run (not
   walltime-truncated) early-stopped at epoch 9, restoring epoch 4's
   weights: **best validation AUC 0.523, test AUC 0.5145**. The `rope_base`
   fix was real (it fixed a genuine numerical instability) but not
   sufficient on its own.
4. **Control experiment**: same tokenization (`sampled_hits`), same real
   data, same `TrainingConfig`, but `position_encoding="coordinate_mlp"`
   instead of `"rope"`. Val AUC hit **0.901 after epoch 1** and **0.933 by
   epoch 3**. This isolates the problem to the RoPE encoding path
   specifically — it rules out a broader pipeline, learning-rate, or
   task-difficulty explanation.
5. **Root-cause check**: computed each event's mean per-token RMS distance
   from its own centroid (an extent proxy) on an 810-event real sample and
   correlated it with the label:

   | | mean extent | median extent |
   |---|---|---|
   | Bi214 (background) | 0.254 | 0.152 |
   | 0nubb (signal) | 0.091 | 0.081 |

   Background events are ~2.8x more spread out than signal events. **Using
   this single scalar as a classifier score alone gives AUC ≈ 0.86** —
   already close to what `coordinate_mlp` achieves. (Padding was ruled out
   as a factor in the same check: `sampled_hits` events are on average
   99.6% full at `max_tokens=512`, so padding-mask handling isn't a
   meaningful confound here.)

## Why this breaks RoPE specifically

RoPE's defining mathematical property is that
`dot(rotate(Q, p_m), rotate(K, p_n))` depends only on the *relative* offset
`p_m - p_n`, never on either position's absolute value. That is exactly what
makes it translation-invariant and exactly what makes it structurally unable
to expose "how far is this token from the event centroid" to the network.
`coordinate_mlp` and `fourier_xyz`, by contrast, are additive absolute-
position embeddings — a token's raw centered coordinates flow straight into
`content_embedding + position_embedding` before attention even runs, so
"distance from centroid" is directly and immediately learnable.

For a task where extent-from-centroid alone is already a strong classifier,
RoPE would have to reconstruct that information indirectly — through
attention weight patterns that vary systematically with how spread out the
relative offsets between token pairs are — a much harder representational
target for a small (2-layer, 4-head, d_model=64) model to discover within a
handful of epochs, even before early stopping (`patience=5`) cuts training
short. This is a known, general limitation of pure relative positional
encodings, not specific to this implementation.

## What this is not

- Not the `rope_base` aliasing bug — that was real, is fixed, and is
  reflected in `next_energybench_train.ipynb`'s `MODEL_CONFIG` and
  `rotary_attention.py`'s docstrings.
- Not a masking/padding bug — real `sampled_hits` events are almost fully
  populated (padding fraction ≈ 0.4%), so the padded-key-masking path in
  `RotarySelfAttention` is barely exercised either way.
- Not an artifact of too few epochs — the `rope_base=2.0` run completed a
  full, non-truncated training cycle (early-stopped, evaluated on the full
  116,549-event test set) and still landed at chance.

## Options considered, not yet acted on

1. **Add extent as an explicit content feature** (e.g. per-token distance
   from centroid, alongside energy) so a RoPE-based model can access
   absolute-extent information through the content channel instead of the
   positional channel, while keeping RoPE's relative-attention mechanism
   for genuine geometric relationships. Would need a tokenization/model
   change (`feature_dim` +1 for the rope path, or more broadly).
2. **Accept the limitation and run the official 3 real jobs anyway**,
   documenting that RoPE is expected to underperform for `sampled_hits` and
   `voxel` (raw point-cloud tokenizations where extent is a strong feature)
   — `summary_features` tokenization may behave differently since it's a
   fundamentally different, pre-aggregated representation.
3. **Stop here and treat this document as the deliverable** for now,
   deciding on real-run compute spend later.

Option 3 was chosen for this session.

## Where the fix so far lives

- `next_detector/next_transformer/rotary_attention.py`: `rope_base`
  aliasing math and docstring already reflect the corrected reasoning.
- `next_detector/notebooks/next_energybench_train.ipynb`: `MODEL_CONFIG`
  now uses `rope_base=2.0` with an inline comment explaining why, plus a
  diagnostic-only `transformer_control_sampled_hits_coordinate_mlp` entry
  in `ALL_EXPERIMENTS` (not part of the official 3x3 matrix — remove once
  no longer needed for reference).
- Real per-model probe outputs used for this investigation live under
  `/pscratch/sd/v/vsharma2/0vbb_benchmark/results/{probe,probe_control}/`
  (throwaway, not part of the official run tree).
