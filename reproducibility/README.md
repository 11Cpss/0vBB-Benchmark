# EnergyBench paper reproduction

This directory contains the model code, metric implementation, configurations,
final numerical results, and table/figure generators for **Learning Physics,
Not Energy: A Multimodal Benchmark for Neutrino Detectors**.

## Install

Run all commands below from this directory. Saved-prediction evaluation and
plotting require Python 3.11 and no GPU.

```bash
python3.11 -m venv .venv-reproduce
.venv-reproduce/bin/python -m pip install -r environments/requirements-core.txt
.venv-reproduce/bin/python -B benchmark/tests/test_metrics.py
```

Model training and inference use separate environments documented in
[Classic models](code/classic/README.md) and [Transformers](code/transformers/README.md).

## Contents

| Directory | Purpose |
| --- | --- |
| `benchmark/` | EnergyBench implementation, explicit paper profiles, CLI and numerical tests |
| `code/classic/` | Published classic-model architectures, training, prediction export and selected configurations |
| `code/transformers/` | Published Transformer/tokenization implementations and run configurations |
| `code/input_preparation/` | Event alignment, physical-energy conversion and score/label standardization |
| `code/figure_sources/` | Figure inputs and extent-distribution reconstruction |
| `results/` | Final result/input manifests, expected metrics and checkpoint identifiers |
| `paper/` | Active manuscript source, figures, final table inputs and rendering functions |
| `reproduction/` | Table/figure rebuilding and manuscript compilation |
| `environments/` | Dependency lists and optional TeX font fallback |

## Tables and figures

```bash
.venv-reproduce/bin/python -B reproduction/reproduce.py render --output outputs/render
```

This rebuilds six numerical tables and the six figures for which generating
code is available. Generated tables are checked against the supplied final
LaTeX. These commands render the final display inputs; they do not reconstruct
per-event predictions from aggregate numbers.

For the figure-input calculations, see [Figure sources](code/figure_sources/README.md).

## Recompute the benchmark

The implementation is [benchmark/metrics.py](benchmark/metrics.py); its input
schema, statistics and profile scope are in [the benchmark guide](benchmark/README.md).
For a new standardized event file:

```bash
.venv-reproduce/bin/python -B benchmark/evaluate.py \
  --input /path/to/predictions.npz --profile strict600 \
  --output outputs/model_metrics.json
```

Exact replay of the published metrics requires the separate event-prediction
companion. Its `event_data/` directory must be available locally; it is not
included in Git and this release does not provide a download URL. With it:

```bash
.venv-reproduce/bin/python -B benchmark/evaluate.py \
  --manifest results/event_inputs.json --data-root /path/to/event_data \
  --output outputs/metrics

.venv-reproduce/bin/python -B code/transformers/replay_paper_metrics.py \
  --data-root /path/to/event_data --output-dir outputs/transformer_metrics
```

The first command covers 61 result records, including the NEXT appendix
comparisons and the separate SuperNEMO illustration. The second covers 12
NEXT/SuperNEMO Transformer records with their specified paper definitions.
Together these cover 43 of the 52 main-table dataset/model entries, in addition
to the appendix and illustration results. Each replay checks actual event
inputs and full-precision expected metrics.

For only the MJD/EXO-200 shared-profile records, add `--dataset MJD --dataset
EXO-200` to the first command. To generate new predictions, use the model
READMEs and the [input standardization guide](code/input_preparation/README.md).
Keep the exact checkpoint, physical energy and test event identities.

## Compile the manuscript

With TeX Live and `latexmk` installed:

```bash
.venv-reproduce/bin/python -B reproduction/reproduce.py compile \
  --engine latexmk --output outputs/compiled
```

Tectonic is also supported with `--engine tectonic`. First use may require its
TeX bundle download. Compilation copies the manuscript into the selected build
directory and produces `outputs/compiled/paper.pdf`; use a fresh output
directory for each build.

## Reproduction scope

- The 600 fixed 5 keV bins on 0–3000 keV apply to the corrected MJD/EXO-200
  records. The retained NEXT/SuperNEMO subsets have explicitly recorded profiles;
  see the benchmark guide. This release preserves the paper's reported values.
- The nine NEXT/MJD/SuperNEMO RoPE run sources have not been recovered. Their
  table values are available, but reproducing those runs is not claimed.
- The eight detector-example panels are supplied as PDFs; their generating
  scripts and selected event identities are unavailable.
- Raw detector datasets and checkpoint tensors are external. Training seeds
  and configurations do not imply bit-identical retraining across hardware.
- The manuscript compiles to 25 pages and contains two unresolved source
  references (`app` and `fig:energybench-auc`); this code release does not alter
  the paper's scientific content.
