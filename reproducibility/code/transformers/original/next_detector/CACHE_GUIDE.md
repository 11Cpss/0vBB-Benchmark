# NEXT Transformer Token Cache

This directory contains a disk-backed preprocessing cache for the NEXT
Transformer. It preserves the shared Simple EnergyBench split, event metadata,
slice-level epoch shuffle, training loop, checkpoints, and evaluation code.

## Files

```text
next_detector/
├── next_transformer/
│   ├── cache.py
│   ├── tokenization.py
│   ├── positional_encoding.py
│   └── model.py
├── scripts/
│   ├── build_token_cache.py
│   └── benchmark_token_cache.py
├── tests/
│   └── test_cache.py
├── notebooks/
│   └── next_energybench_cached.ipynb
├── token_cache/       # generated on the lab computer; ignored by Git
└── results/           # generated on the lab computer; ignored by Git
```

## What is cached

For every event, the cache stores padded float32 coordinates and features, a
Boolean valid-token mask, event energy, retained-energy coverage, source event
ID, and label. It does not store positional embeddings, Transformer
activations, pooled representations, logits, or predictions.

Each tokenizer has one reusable cache:

```text
sampled-hit cache -> coordinate MLP and Fourier XYZ
voxel cache       -> coordinate MLP and Fourier XYZ
```

## Remote verification

From the remote project root:

```bash
cd /home/klz/Data/zeronu_benchmark/Transformer_Approach

python -m compileall -q \
  evalutaions_workflow/energybench \
  evalutaions_workflow/simple_energybench \
  next_detector/next_transformer \
  next_detector/scripts

PYTHONPATH=evalutaions_workflow:next_detector \
python -m unittest discover -s next_detector/tests -v
```

## Build the deadline-priority cache

Run cache construction from the terminal, not inside Jupyter:

```bash
python next_detector/scripts/build_token_cache.py \
  --data-root /home/klz/Data/zeronu_benchmark/NEXT \
  --split-manifest /home/klz/Data/zeronu_benchmark/Transformer_Approach/next_detector/results/event_split.json \
  --cache-root /home/klz/Data/zeronu_benchmark/Transformer_Approach/next_detector/token_cache \
  --tokenization sampled_hits \
  --max-tokens 512 \
  --voxel-size 15 \
  --coordinate-scale 1000 \
  --center-coordinates \
  --voxel-truncation occupancy \
  --seed 42 \
  --workers 8 \
  --resume
```

The final directory is published only after all slices pass. If interrupted,
run the identical command again; completed slices recorded in
`build_state.json` are skipped, and unrecorded or partially written slices are
rewritten.

Do not use `--overwrite` unless intentionally rebuilding the complete cache.

## Benchmark the cache

```bash
python next_detector/scripts/benchmark_token_cache.py \
  --data-root /home/klz/Data/zeronu_benchmark/NEXT \
  --split-manifest /home/klz/Data/zeronu_benchmark/Transformer_Approach/next_detector/results/event_split.json \
  --cache-root /home/klz/Data/zeronu_benchmark/Transformer_Approach/next_detector/token_cache \
  --tokenization sampled_hits \
  --max-tokens 512 \
  --voxel-size 15 \
  --coordinate-scale 1000 \
  --center-coordinates \
  --voxel-truncation occupancy \
  --seed 42 \
  --workers 8 \
  --batch-size 64 \
  --warmup-batches 20 \
  --batches 500
```

Proceed with cached training only when all tests pass and the benchmark reports
at least a 1.5x event-throughput improvement. The important real-world check
is the duration of cached epoch 1 compared with the observed uncached
approximately 6.5 minutes.

## Train the priority model

Open:

```text
next_detector/notebooks/next_energybench_cached.ipynb
```

The notebook defaults to only:

```text
transformer_001_sampled_hits_coordinate_mlp
```

It validates the cache and official event counts before training. It writes to
`next_detector/results/final_cached_v1`, preserving the interrupted uncached
run. EnergyBench writes `best_model.pt` and `last_model.pt`; after training, the
best-validation-AUC model is evaluated once on the held-out test set.

## Build the voxel cache later

After the priority model has a complete test evaluation, reuse the build and
benchmark commands with:

```text
--tokenization voxel
```

Then add `transformer_002_voxel_coordinate_mlp` to `RUN_MODEL_IDS` in the
cached notebook. Enable Fourier variants only when sufficient deadline time
remains.

## Local-to-remote copy

From the Mac repository root:

```bash
scp \
  next_detector/next_transformer/cache.py \
  next_detector/next_transformer/__init__.py \
  liuser@pc1.lilab.ucsd.edu:/home/klz/Data/zeronu_benchmark/Transformer_Approach/next_detector/next_transformer/

scp \
  next_detector/scripts/build_token_cache.py \
  next_detector/scripts/benchmark_token_cache.py \
  liuser@pc1.lilab.ucsd.edu:/home/klz/Data/zeronu_benchmark/Transformer_Approach/next_detector/scripts/

scp \
  next_detector/tests/test_cache.py \
  liuser@pc1.lilab.ucsd.edu:/home/klz/Data/zeronu_benchmark/Transformer_Approach/next_detector/tests/

scp \
  next_detector/notebooks/next_energybench_cached.ipynb \
  liuser@pc1.lilab.ucsd.edu:/home/klz/Data/zeronu_benchmark/Transformer_Approach/next_detector/notebooks/
```

Create `scripts`, `tests`, `notebooks`, `token_cache`, and `results` on the
remote computer before these copy commands if they do not already exist.
