# MJD RoPE Classification Benchmark

Signal-vs-background (clean-vs-non-clean waveform) classification on
Majorana Demonstrator (MJD) data, using rotary positional encoding (RoPE) as
a third positional encoding alongside the official `mjd_detector` matrix's
`coordinate_mlp` and `fourier_coordinates`.

This folder is a sibling extension, built the same way
[`cuore_detector/`](../cuore_detector/README.md) added RoPE for CUORE:
`mjd_detector/` and `next_detector/` are not modified. Reused unchanged:

- the three tokenizers (`raw_patches`, `segment_summary`, `pulse_entities`)
  and the two additive coordinate encoders, from
  [`mjd_transformer`](../mjd_detector/mjd_transformer/);
- rotary attention, from
  [`next_transformer.rotary_attention`](../next_detector/next_transformer/rotary_attention.py);
- the classification data contract, preprocessing, training loop, checkpoint
  rule, and metrics, from
  [`mjdbench`](../mjd_detector/mjdbench/) (`train_model`, `evaluate_model`,
  `MJDWaveformDataset`, the official seed=42 90/10 train/validation split).

## Task

Predict whether an event passes all four PSD cuts (`low_avse`, `high_avse`,
`dcr`, `lq`) -- the same `clean_vs_non_clean` target as the official MJD
classification matrix. In the full official data this label is 38.01%
positive (clean) / 61.99% negative, driven mostly by the `low_avse` cut
(44.8% pass rate; the other three pass 70-99%).

## Data protocol

`MJD_Train_*.hdf5` waveforms are `[N, 3800]`, not the 10,000 samples
`mjd_detector/README.md` assumes -- that stale figure sized the official
`token_count=500` protocol. Here all three tokenizers use **190 tokens**
(`3800 / patch_size(20) = 190`), keeping sequence length comparable across
tokenizations.

The HDF5 shards are unchunked: random-row `DataLoader` batches read at
~770 events/s vs. ~30,500 events/s contiguous (measured). To stay
I/O-bound-free and to make repeated runs cheap, [`mjdrope_bench.data`](mjdrope_bench/data.py)
draws a **fixed, seeded 50,000 / 5,000 / 50,000 train/validation/test
subsample** once, applies `mjdbench`'s baseline-subtraction and
classification amplitude-normalization exactly as-is, and caches the result
as float32 tensors (~1.6 GB) so all three tokenization runs share one load:

- train/validation subsamples are prefixes of the *official* seed=42 90/10
  permutation (`mjdbench.data._subset_indices`, reused), so no validation
  event can leak into training even at reduced scale;
- the test subsample is an explicit seeded random draw from the full
  390,000-event official test pool (not a contiguous prefix, which would
  effectively only cover one shard).

## `rope_base`

MJD tokenizer coordinates span `[-1, 1]` (Δ = 2) on all three channels
(normalized time, normalized amplitude, first difference). With
`d_model=64, nhead=4` (`head_dim=16` → 8 rotation pairs, split `[3, 3, 2]`
across the three axes), RoPE's `cos(theta * Δcoord)` stays unambiguous only
while `theta_slow * Δcoord <= pi` for the worst-case pairwise gap
`Δcoord = 2.0`:

- time / amplitude axes (`count=3`): `rope_base <= (pi/2)^3 ≈ 3.876`
- first-difference axis (`count=2`, the binding constraint):
  `rope_base <= (pi/2)^2 ≈ 2.467`

`DEFAULT_MJD_ROPE_BASE = 2.0` (~19% headroom on the binding axis). This does
**not** transfer from CUORE's `DEFAULT_ROPE_BASE=8.0`, which relaxed the
same bound using the largest Δt actually present in CUORE's regression
target -- a task-specific quantity with no analog for classification, where
every pairwise token interaction across the full event matters.

## Folder layout

```text
mjd_rope_classifier/
├── mjd_rope_transformer/   # MJDRopeTransformer: MJDTransformer's `task`
│                           # parameter + CuoreWaveformTransformer's rope branch
├── mjdrope_bench/          # seeded subsample cache + DataLoader wrapper
├── scripts/
│   ├── run_mjd_rope_classification.py           # one cell (env-var driven)
│   ├── run_mjd_rope_classification_array.sbatch  # array 0-2 over tokenizations
│   └── collate_results.py                        # merge run_summary.json -> CSV
├── tests/                  # synthetic-data, no GPU/real data required
└── results/                # generated; gitignored
```

## Run the matrix

```bash
module load anaconda3/2023.09-0-python_3.11.5
source activate /ix/pvenkata/vis224/envs/pyT

# One cell, locally:
MJD_ROPE_TOKENIZATION=raw_patches \
  python mjd_rope_classifier/scripts/run_mjd_rope_classification.py

# All three, on the cluster (build the cache with task 0 first to avoid a race):
mkdir -p mjd_rope_classifier/scripts/logs
sbatch --array=0 mjd_rope_classifier/scripts/run_mjd_rope_classification_array.sbatch
sbatch --array=1-2 mjd_rope_classifier/scripts/run_mjd_rope_classification_array.sbatch

# After all three complete:
python mjd_rope_classifier/scripts/collate_results.py \
  --output-root mjd_rope_classifier/results/mjd_rope_classification_v1
```

Useful environment variables (all optional, see the script header for the
full list): `MJD_ROPE_DATA_DIR` (default `/vast/pvenkata/vis224/MJD`),
`MJD_ROPE_CACHE_DIR` (default `<MJD_ROPE_DATA_DIR>/../mjd_rope_cache`),
`MJD_ROPE_N_TRAIN` / `MJD_ROPE_N_VAL` / `MJD_ROPE_N_TEST` (default
50000/5000/50000), `MJD_ROPE_BASE` (default 2.0), `MJD_ROPE_SEED` (default
42), `MJD_ROPE_EPOCHS` (default 50), `MJD_ROPE_DEVICE` (default `auto`).

## Outputs

Each run writes, directly under `results/mjd_rope_classification_v1/<run_id>/`
(matching the output contract in `mjd_detector/README.md`):

```text
run_config.json   best.pt   history.json   metrics.json   predictions.npz   run_summary.json
```

`run_id = classification__<tokenization>__rope`. Re-running is safe: a run
whose `run_config.json` and `run_summary.json` both exist is skipped unless
`MJD_ROPE_OVERWRITE=1`.

## Tests

```bash
python -m compileall mjd_rope_classifier
python -m unittest discover -s mjd_rope_classifier/tests -v
```

Synthetic data only, no GPU or real MJD files: covers all three
tokenizations under `rope`, parity with `MJDTransformer` for the two
additive encodings, RoPE config guards, the seeded cache build/reuse
(against a fabricated MJD-schema HDF5 fixture), and one full
train + evaluate round trip through the unchanged `mjdbench.training`
functions.

## Scientific comparison notes

- Only `rope` is run here; `coordinate_mlp` and `fourier_coordinates`
  controls (already covered by the official `mjd_detector` matrix at
  `token_count=500`) are a one-line change away (`MJDRopeTransformer`
  supports all three `position_encoding` values) if a directly comparable
  190-token control run is wanted.
- Parameter counts differ across the tokenization axis (`raw_patches` has
  `feature_dim = patch_size = 20`, vs. 2 for the other two) -- not a
  matched-capacity comparison.
- All runs are single-seed: treat as one benchmark realization, not an
  uncertainty estimate.
