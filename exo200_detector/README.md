# EXO-200 v1 Transformer Benchmark

This folder benchmarks sensor-aware Transformer representations for EXO-200
waveforms while retaining the collaborator-provided EXOBench data loading,
preprocessing, run split, optimization, checkpoint selection, and evaluation.
The tokenizer runs inside the model, so CNN and Transformer baselines receive
the same preprocessed `[226, 300]` waveform.

## Task and shared protocol

The benchmark uses the EXO-200 v1 open data release: 29 HDF5 shards containing
515,002 events. `Charge_cluster_number == 1` is signal (label 0), while values
greater than one are background (label 1 and the positive ROC class).
`Charge_Clusters_Pos` is never returned as input because its shape reveals the
cluster-count label.

EXOBench subtracts each channel's mean over the first 200 samples and divides
the complete event by its maximum absolute amplitude. Entire detector runs are
kept indivisible across the seeded split:

| Partition | Events | Runs |
|---|---:|---|
| Train | 336,632 | 8956, 8957, 8958, 8968, 8972, 8999, 9000, 9001, 9008, 9009 |
| Validation | 37,987 | 8969 |
| Test | 140,383 | 8967, 8970, 8971, 9010 |

The split contract, class counts, and run-level event inventory are frozen in
[`manifests/exo200_v1_split.json`](manifests/exo200_v1_split.json). The training
notebook refuses to run if EXOBench reconstructs a different partition.

## Sensor-aware tokenization

The 226 detector channels are preserved as six physical blocks:

```text
0:38     positive U wires
38:76    positive V wires
76:114   negative U wires
114:152  negative V wires
152:189  positive APDs
189:226  negative APDs
```

Every representation produces 504 tokens:

```text
6 sensor blocks × 7 channel regions × 12 time regions
```

The 300 time samples divide into twelve 25-sample regions. Each token receives
normalized time, normalized within-block channel position, detector side, and
a U/V/APD one-hot indicator.

| Tokenization | One token represents | Content features |
|---|---|---|
| `raw_patches` | One channel/time region | Up to 6 × 25 raw amplitudes, internally zero-padded to 150 values |
| `segment_summary` | The same physical region | Mean, RMS, maximum absolute amplitude, temporal half-difference |
| `pulse_entities` | One selected channel/time point | Amplitude and 9-sample local RMS |

Pulse entities allocate 84 tokens to every sensor block: 42 fixed uniform-grid
points and 42 points ranked by absolute amplitude plus absolute temporal
change. Selection never uses labels, reconstructed energy, or cluster fields.

The benchmark compares `coordinate_mlp` and `fourier_coordinates`. Content and
coordinates are projected independently to 64 dimensions, added, normalized,
processed by a two-layer four-head Transformer, and mean-pooled into one
background-classification logit.

## Run the benchmark

Install from the repository root, then point the notebook to the v1 shards:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
export EXO200_BENCH_DATA=/absolute/path/to/EXO-200-v1
jupyter lab
```

Open `exo200_detector/notebooks/exo_transformer_train.ipynb`. The default run
executes all six combinations. Assign specific runs with a comma-separated
variable:

```bash
export EXO200_RUN_IDS=classification__raw_patches__coordinate_mlp
```

Other supported variables are `EXO200_OUTPUT_ROOT` and
`EXO200_TRANSFORMER_PROJECT_ROOT`.

Each completed run writes `run_config.json`, `best.pt`, `history.json`,
`metrics.json`, `predictions.npz`, and `run_summary.json`. The output root also
contains `transformer_results.csv`. The results-only notebook reads these
artifacts without loading detector data:

```text
exo200_detector/notebooks/exo_transformer_results.ipynb
```

## Energy-aware evaluation

After test predictions exist, open:

```text
exo200_detector/notebooks/exo_transformer_energybench_results.ipynb
```

This notebook does not rerun inference. It reconstructs the unchanged
EXOBench test ordering, attaches `Rotated_energy` as evaluation-only metadata,
and evaluates all completed Transformer runs with the vendored, fingerprinted
EnergyBench snapshot under `frozen_energybench/`. It produces inclusive and energy-matched AUC,
common-support diagnostics, and overall/worst-group energy-independence scores.

Optional environment variables are:

```bash
# Override only to reproduce a separately frozen evaluator checkout.
export EXO200_ENERGYBENCH_SOURCE=/absolute/path/containing/energybench

# Optional structural comparison against a collaborator summary.
export EXO200_WING_ENERGYBENCH_SUMMARY=/absolute/path/to/summary.json
export EXO200_REQUIRE_ALL_EVALUATIONS=1
```

The evaluator refuses test-label ordering mismatches and stale output folders
whose checkpoint or prediction hashes changed. Combined results are written to
`energybench_transformer_results.csv` and
`energybench_transformer_summary.json` under `EXO200_OUTPUT_ROOT`.

## Tests and scientific boundaries

```bash
python -m unittest discover -s exo200_detector/exo_transformer/tests -v
python -m compileall -q exo200_detector
```

- Keep `exobench` unchanged for direct specialized-model comparisons.
- Select checkpoints with validation AUC and report the untouched test only
  after training.
- Report parameter counts because raw patches have a larger input projection.
- Treat these as single-seed benchmark results rather than uncertainty bounds.
- Energy fields are not model inputs. Energy-matched AUC and independence are
  computed afterward from saved test logits with `Rotated_energy` attached as
  evaluation-only metadata.

## Data citation and licensing

The detector data are not distributed by this repository. Download EXO-200 v1
from Zenodo and follow its CC BY 4.0 terms:

> EXO-200 Collaboration. (2026). *EXO-200 Open Source Data Release for AI/ML
> Applications* (v1) [Dataset]. Zenodo.
> https://doi.org/10.5281/zenodo.20419164

The release also asks users to cite the EXO-200 complete-dataset search,
https://doi.org/10.1103/PhysRevLett.123.161802. Repository software is covered
by the root license; collaborator provenance is described in
[`NOTICE.md`](NOTICE.md) and [`frozen_energybench/SOURCE.json`](frozen_energybench/SOURCE.json).
