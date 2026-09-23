# EnergyBench benchmark code — start here

This directory exposes the actual metric implementations used by the archived
paper results. It contains executable Python source, protocol configurations,
and numerical tests. The implementations are byte-identical copies of the
previously bundled sources; their source paths and SHA-256 hashes are recorded
in [source_manifest.json](source_manifest.json). No original experiment code or
manuscript file has been changed.

## Where the calculations are implemented

| File | What it implements |
| --- | --- |
| [core/unified_metrics.py](core/unified_metrics.py) | Strict-range EnergyBench: energy conversion and binning, weighted inclusive AUC, groupwise Jensen–Shannon independence I, common support, overlap matching, pooled weighted AUC, coverage gates, effective sample sizes and diagnostics |
| [protocol.json](protocol.json) | Exact strict600 settings and reporting thresholds: keV, 0–3000 keV inclusive, 600 fixed 5 keV bins |
| [evaluate.py](evaluate.py) | Command-line entry for one prediction archive or the published per-event result manifest; loads the metric cores in this directory |
| [tests/test_metrics.py](tests/test_metrics.py) | 14 numerical checks: units, all boundaries, out-of-range exclusions, ties, constant scores, sparse bins, missing values, coverage and independent weighted-AUC enumeration |
| [legacy_v2/core/unified_metrics.py](legacy_v2/core/unified_metrics.py) | Preserved evaluator for published records that use 600 regular bins plus one overflow bin |
| [legacy_v2/protocol.json](legacy_v2/protocol.json) | Exact retained overflow601 settings |
| [historical/next/energybench/metrics.py](historical/next/energybench/metrics.py) | Original NEXT Transformer benchmark metrics; its supporting package is copied alongside it |
| [historical/supernemo/energybench/roc.py](historical/supernemo/energybench/roc.py) and [dependence.py](historical/supernemo/energybench/dependence.py) | Original SuperNEMO Transformer matching/AUC and independence implementations; its supporting package is copied alongside them |

In the strict-range core, `energy_bin_indices` assigns the global grid,
`weighted_auc` handles weighted pairs and ties, and `evaluate` computes the
complete benchmark result. The class/category independence calculation and
energy matching calculation are separate stages inside `evaluate`.

## Run from the bundle root

Use the Python 3.11 environment described in the [main README](../README.md).
The commands below assume that environment already exists.

```bash
# Numerical validation; no event data or GPU is required.
.venv-reproduce/bin/python -B benchmark/tests/test_metrics.py

# Recompute all 61 audited records from the separately packaged event data.
.venv-reproduce/bin/python -B benchmark/evaluate.py \
  --manifest provenance/event_inputs.json --data-root event_data \
  --output outputs/benchmark

# Only MJD and EXO-200 (23 records evaluated with strict600).
.venv-reproduce/bin/python -B benchmark/evaluate.py \
  --manifest provenance/event_inputs.json --data-root event_data \
  --dataset MJD --dataset EXO-200 --output outputs/benchmark_mjd_exo

# Evaluate your own correctly standardized event predictions.
.venv-reproduce/bin/python -B benchmark/evaluate.py \
  --input /path/to/predictions.npz --profile strict600 \
  --output outputs/my_benchmark.json
```

The NPZ must contain aligned one-dimensional `score`, `label`, `energy_keV`
and unique, nonempty `event_id` arrays. Optional `group` and `weight` preserve
the physical categories and original base weights. A higher score must favor
`label=1`. For physical energy in MeV, pass `--energy-key energy
--energy-unit MeV`; do not clip energy before evaluation. Read the dataset
label/score conventions in the main README before standardizing new inputs.

Manifest mode checks each input hash, its recorded protocol fingerprint and
the full-precision expected results. It does not calculate metrics from table
summaries. Outputs include inclusive AUC, I and each group I, matched AUC,
reporting status, sample counts, eligible bins, class coverage, matching
effective sample sizes, and range/support/sparsity losses.

## Use the correct recorded protocol

- MJD and EXO-200 records in `provenance/event_inputs.json` use `strict600`.
- Retained NEXT and SuperNEMO classic/extended records and the independent
  SuperNEMO energy-only illustration use `overflow601`.
- Twelve historical NEXT/SuperNEMO Transformer runs use their original
  recorded implementations/configurations. Their replay entry remains:

  ```bash
  .venv-reproduce/bin/python -B code/transformers/replay_historical_transformers.py \
    --output-dir outputs/historical_transformer_metrics
  ```

  That entry loads the original package snapshots under `code/transformers/`;
  their visible copies under `benchmark/historical/` are identical. Per-run
  configurations and exact event inputs are linked in
  `provenance/transformer_historical_inputs.json`.

These historical copies document the actual published computations. They are
not alternate defaults for new strict-range evaluation. The nine missing
NEXT/MJD/SuperNEMO RoPE experiment sources remain explicitly unresolved.

## The rest of the experiment workflow

Training, architecture and checkpoint inference code are in
[`code/classic/`](../code/classic/README.md) and
[`code/transformers/`](../code/transformers/README.md). Event extraction and
alignment sources are documented in
[`code/input_preparation/`](../code/input_preparation/README.md).
To regenerate final tables and figures after inspecting the benchmark inputs,
use `reproduction/reproduce.py render` as described in the main README.

The existing `reproduction/evaluate.py` entry remains available and produces
the same results. It reads the identical frozen cores in the paper snapshot;
`benchmark/evaluate.py` explicitly reads the visible copies in this directory.
