# SuperNEMO RoPE Classification Benchmark

Signal-vs-background classification of SuperNEMO Demonstrator events
(`2nubb` vs. `Bi214`) with a Transformer that encodes tracker-hit position by
**rotary position encoding (RoPE)** -- the third positional encoding next to the
published coordinate-MLP and Fourier-XYZ models.

This folder is a sibling extension, built the same way
[`mjd_rope_classifier/`](../mjd_rope_classifier/README.md) and
[`cuore_detector/`](../cuore_detector/) added RoPE for MJD and CUORE: no existing
package is modified, and shared pieces are imported. It deliberately does **not**
use the directory or package names the top-level README reserves for the
collaboration's SuperNEMO benchmark (`supernemo_detector/`, `supernemobench`,
`supernemo_transformer`), so the two can be merged in either order without
path collisions.

## What is reused and what is new

Reused unchanged (imported, not copied):

- the classifier and its rotary attention, from
  [`next_transformer`](../next_detector/next_transformer/) -- `NEXTTransformerClassifier`
  already implements the architecture the published SuperNEMO Transformers use
  (content MLP -> pre-norm encoder -> masked-mean pooling -> MLP head, one raw
  logit per event) for `coordinate_mlp`, `fourier_xyz` and `rope`;
- training and evaluation, from the shared EnergyBench
  ([`simple_energybench`](../evalutaions_workflow/)): `train_model` and
  `evaluate_classification`, with the collaboration's standard optimizer,
  scheduler, loss, checkpoint rule, and energy-matched metrics.

New here:

| Module | Purpose |
|---|---|
| [`supernemo_rope_transformer/tokenization.py`](supernemo_rope_transformer/tokenization.py) | Tracker-hit tokenizers (`sampled_hits`, `voxel`, `summary_features`). **Verbatim copy** of the published SuperNEMO tokenizer (branch `wenyu/energybench-reproducibility`, not on `main`), so token content is byte-identical and only the position encoding differs. Swap for an import once that lands on `main`. |
| [`supernemo_rope_transformer/model.py`](supernemo_rope_transformer/model.py) | `build_supernemo_transformer`, the `rope_base` default, and the RoPE frequency-table helpers. |
| [`supernemorope_bench/`](supernemorope_bench/) | Event index, the published split, streaming loaders in EnergyBench's batch format, and the energy-grid wrapper. |
| [`scripts/`](scripts/) | Env-var-driven runner, SLURM array, result collation, and the `rope_base` scale report. |
| [`tests/`](tests/) | 44 synthetic-data tests; no GPU or real data needed. |

## Task and data contract

Everything below matches the published SuperNEMO Transformer benchmark so RoPE
results are directly comparable with its coordinate-MLP / Fourier-XYZ models.

- **Signal / background:** `2nubb` (label 1) vs. `Bi214` (label 0). `0nubb` and
  `Tl208` are not classified, but stay in the source list because a source's
  block permutation is seeded by its position in it.
- **Event:** the consecutive rows sharing an `ev_no` in one source file (one row
  = one tracker hit). `ev_no` runs 0..N-1 without gaps in every file; anything
  else is rejected.
- **Split:** seed 42, 80/10/10 by whole blocks of 4096 events, dealt by a permutation
  seeded with `(seed, source position)`. Whole blocks keep neighbouring,
  correlated events in one split.

  | Split | `2nubb` | `Bi214` | Total |
  |---|---:|---:|---:|
  | train | 2,629,632 | 2,105,618 | 4,735,250 |
  | validation | 326,804 | 266,240 | 593,044 |
  | test | 327,680 | 262,144 | 589,824 |

  The runner asserts these counts on any uncapped run. Independently, the split
  boundaries were compared with the published split manifest (from
  `wenyu/energybench-reproducibility`) and are **identical slice for slice**
  (220 train / 124 validation / 130 test slices). `tests/` pins the counts and
  slice numbers.
- **Inputs:** tracker-hit topology only -- `tX, tY, tZ` and the cell radius `tR`
  (NaN for some hits, handled). The calorimeter energies `E1`, `E2` are never
  seen by the model; `E1 + E2` is only the conditioning energy of the
  energy-matched evaluation (tested: changing the energies leaves every token
  unchanged).

### Tokenizations

Coordinates are per-event-centered and divided by 1000 mm. Settings are the
published ones.

| Tokenization | `max_tokens` | `feature_dim` | Token = |
|---|---:|---:|---|
| `sampled_hits` | 128 | 4 | a tracker hit (deterministic sample if more than 128) |
| `voxel` | 128 | 4 | hits in one 60 mm voxel (most populated 128 kept) |
| `summary_features` | 16 | 6 | one of 16 balanced Morton-ordered spatial groups; every hit retained |

Events are short (median 22 hits, 95th percentile about 35; up to 213), so for the
first two tokenizations almost every event fits without truncation.

## Position encoding: RoPE and `rope_base`

RoPE rotates each token's query and key by an angle built from its position
inside every attention layer, so attention scores depend on the *relative*
geometry of two hits, not on where the event sits in the detector. Here each
head's channels are split over x, y and z (`head_dim = 64 / 4 = 16` gives 8
channel pairs split 3 / 3 / 2), and a pair on axis *a* rotates by
`theta * coordinate_a`, with `theta` from `rope_base` (fastest) down to
`rope_base ** (1 / pairs_on_axis)` (slowest). A test confirms the model's output
is invariant to translating all coordinates (and that the additive coordinate MLP
is not).

The coordinate scale drives the choice of `rope_base`. Measured per-event extents
(max pairwise separation, units of 1000 mm; 30,000 events per class):

| axis | median | p95 | p99 | max |
|---|---:|---:|---:|---:|
| x | 0.72 | 0.81 | 0.81 | 0.81 |
| y | 0.40 | 1.23 | 2.07 | 4.93 |
| z | 0.36 | 1.01 | 1.49 | 2.79 |

A separation stays unambiguous on a channel while `theta * separation <= pi`.
Unlike the MJD analysis there is **no `rope_base > 1` that is unambiguous for the
worst-case separation** (y reaches 4.93), so some large events necessarily wrap on
their coarsest channel. The trade-off is coverage vs. local resolution
(`python supernemo_rope_classifier/scripts/rope_scale_report.py`):

| `rope_base` | fastest wavelength | coarsest range x / y / z | events fully inside (all axes) |
|---:|---:|---|---:|
| 2 | 3.14 m | 2.49 / 2.49 / 2.22 | 99.4% |
| 4 | 1.57 m | 1.98 / 1.98 / 1.57 | 98.2% |
| **8 (default)** | 0.79 m | 1.57 / 1.57 / 1.11 | 94.5% |
| 16 | 0.39 m | 1.25 / 1.25 / 0.79 | 85.9% |
| 32 | 0.20 m | 0.99 / 0.99 / 0.56 | 69.6% |
| 64 | 0.10 m | 0.79 / 0.79 / 0.39 | 23.3% |

`DEFAULT_SUPERNEMO_ROPE_BASE = 8.0` keeps about 95% of events unambiguous on
every axis while its fastest channel still has a sub-metre wavelength. **This is a
principled starting point, not a tuned optimum**: the cell pitch is 44 mm, so
finer local resolution (larger base) may pay off despite more wrapping. Sweep it
(`SUPERNEMO_ROPE_BASE`, see below).

## Evaluation

Primary metric: EnergyBench **energy-matched AUC** on the held-out test split, with
inclusive AUC, common-support AUC, shortcut gap, and (worst-group)
energy-independence scores alongside -- as for NEXT.

**Energy grid.** EnergyBench's fixed grid is 0-3000 keV and it *raises* for any
event outside it. `Bi214` reaches 3182 keV (338 of 2,634,002 events = 0.013%; 40
of the 262,144 in the test split); `2nubb` tops out at 2830 keV. The runner
therefore evaluates the test events inside the grid and records how many it set
aside (`test_events_above_energy_grid` in `run_summary.json`). These events are
`Bi214` only, so they lie outside the classes' common energy support. Training and
validation are unaffected.

**Not the paper profile.** The published SuperNEMO Transformer results use a
paper-specific energy-matching profile (its own quantile-based support, no fixed
grid), not canonical EnergyBench, so absolute numbers here are **not**
interchangeable with the paper's even though the split and tokens are identical.
Compare RoPE against the `coordinate_mlp` / `fourier_xyz` controls run through
*this* code path (below), or re-score `evaluation/predictions.npz` with the
paper profile (it stores score, label, energy, category and event id).

## Folder layout

```text
supernemo_rope_classifier/
├── supernemo_rope_transformer/   # tokenizer (published, verbatim) + model factory
├── supernemorope_bench/          # split, streaming loaders, energy-grid wrapper
├── scripts/
│   ├── run_supernemo_rope_classification.py        # one cell (env-var driven)
│   ├── run_supernemo_rope_classification_array.sbatch  # array 0-2 over tokenizations
│   ├── collate_results.py                          # run_summary.json -> CSV + ranking
│   └── rope_scale_report.py                        # extents vs. rope_base coverage
├── tests/                        # synthetic-data, no GPU/real data required
└── results/                      # generated; gitignored
```

## Run

```bash
module load anaconda3/2023.09-0-python_3.11.5
source activate /ix/pvenkata/vis224/envs/pyT

# Tests (synthetic, ~15 s, CPU):
python -m unittest discover -s supernemo_rope_classifier/tests -v

# CPU smoke on a tiny slice of the real data (about 35 s):
SUPERNEMO_ROPE_TOKENIZATION=voxel SUPERNEMO_ROPE_MAX_TRAIN_EVENTS=8192 \
SUPERNEMO_ROPE_MAX_VALIDATION_EVENTS=4096 SUPERNEMO_ROPE_MAX_TEST_EVENTS=8192 \
SUPERNEMO_ROPE_EPOCHS=1 SUPERNEMO_ROPE_DEVICE=cpu SUPERNEMO_ROPE_USE_AMP=0 \
  python supernemo_rope_classifier/scripts/run_supernemo_rope_classification.py

# The full official matrix on the cluster (3 rope cells; log dir ships with the repo):
sbatch supernemo_rope_classifier/scripts/run_supernemo_rope_classification_array.sbatch

# After the runs complete:
python supernemo_rope_classifier/scripts/collate_results.py \
  --output-root supernemo_rope_classifier/results/supernemo_rope_classification_v1
```

**Size the walltime from a measured epoch first.** The sbatch default (16 h) is an
estimate: 4.74M training events is about 74k steps per epoch at batch 64, for up to
50 epochs, and no GPU run has been timed. The sbatch header has a capped one-epoch
probe command.

Controls and sweeps use the same code path, so they are like-for-like:

```bash
# rope_base sweep
sbatch --export=ALL,SUPERNEMO_ROPE_BASE=16,SUPERNEMO_ROPE_RUN_SUFFIX=_base16 \
  supernemo_rope_classifier/scripts/run_supernemo_rope_classification_array.sbatch
# position-encoding control (same tokens, split, trainer, evaluator)
sbatch --export=ALL,SUPERNEMO_ROPE_POSITION_ENCODING=coordinate_mlp \
  supernemo_rope_classifier/scripts/run_supernemo_rope_classification_array.sbatch
```

| Variable | Default | Purpose |
|---|---|---|
| `SUPERNEMO_ROPE_DATA_DIR` | `/vast/pvenkata/vis224/SuperNEMO` | Directory with `data_*_merged.h5` |
| `SUPERNEMO_ROPE_OUTPUT_ROOT` | `supernemo_rope_classifier/results/supernemo_rope_classification_v1` | Run-tree root |
| `SUPERNEMO_ROPE_TOKENIZATION` | `voxel` | `sampled_hits`, `voxel`, `summary_features` |
| `SUPERNEMO_ROPE_POSITION_ENCODING` | `rope` | `rope`, `coordinate_mlp`, `fourier_xyz` |
| `SUPERNEMO_ROPE_BASE` | `8.0` | `rope_base` (used by `rope`) |
| `SUPERNEMO_ROPE_RUN_SUFFIX` / `_RUN_ID` | empty | Name sweeps/controls apart |
| `SUPERNEMO_ROPE_SEED` | `42` | Training/tokenization seed; **any other value is no longer the published split** |
| `SUPERNEMO_ROPE_EPOCHS`, `_BATCH_SIZE`, `_PATIENCE` | `50`, `64`, `12` | Published SuperNEMO training settings (early-stopping patience is 12 there; EnergyBench's own default is 5) |
| `SUPERNEMO_ROPE_MAX_TRAIN_EVENTS`, `_MAX_VALIDATION_EVENTS`, `_MAX_TEST_EVENTS` | unset (all) | Reduced-scale runs (seeded whole blocks, class ratio kept) |
| `SUPERNEMO_ROPE_DEVICE`, `_NUM_WORKERS`, `_USE_AMP` | `auto`, `0`, `1` | Execution only |
| `SUPERNEMO_ROPE_OVERWRITE` | `0` | Redo a completed/partial run |

A run whose `run_config.json` and `run_summary.json` both exist is skipped.
Capped runs are flagged `published_split: false` and called out in the ranking.

## Outputs

```text
results/<output_root>/<run_id>/
├── run_config.json        # full config, RoPE frequency table, split counts
├── training/              # best_model.pt, last_model.pt, history.json, training_history.png
├── evaluation/            # metrics.json, results.csv, predictions.npz, *.png
└── run_summary.json       # one row for collate_results.py
```

## Measured, and not

Measured on this repo's data: split identity with the published manifest; event
extents and the `rope_base` coverage table; single-worker loader throughput
(`sampled_hits` ~19k, `voxel` ~8.7k, `summary_features` ~5.4k events/s); that
evaluation is `ok` on 60k real test events; a 1-epoch CPU smoke run.

**Not** measured: GPU step time, full-epoch wall time, or any converged result.
The smoke run's AUC (about 0.5) only shows the pipeline runs. Nothing here says
RoPE beats or trails the additive encodings; that needs the sweep and the
controls, and several seeds before differences are interpretable.

## License and data

The original software is available under the repository's
[MIT License](../LICENSE). The SuperNEMO dataset is not distributed by this
repository and remains subject to its owners' terms.
