# EnergyBench Usage and Maintenance Guide

[Chinese version](USAGE_GUIDE.md)

This guide focuses on how to run the software, how responsibilities are divided across the
project structure, and where to make changes. For scoring formulas and output interpretation,
see [EVALUATION_STANDARD_EN.md](EVALUATION_STANDARD_EN.md)
([Chinese](EVALUATION_STANDARD.md)).

## 1. Single Python Environment

The entire project uses `/home/wenyu/summer/.venv`. Run the following whenever you open a new
terminal:

```bash
cd /home/wenyu/summer
source .venv/bin/activate
```

After activation, use only `python`, `python -m pip`, and `energybench`. Do not mix them with the
system `python3`, a bare `pip`, or paths from older environments. Verify the environment with:

```bash
which python
python --version
python -m pip check
energybench --version

python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The expected Python version is 3.11, and `which python` should point to this project's
`.venv/bin/python`. Use `deactivate` to leave the environment.

### Rebuilding `.venv` from Scratch

Rebuilding is normally unnecessary. If it is required, first confirm that
`envs/python-builds/cpython-3.11-linux-x86_64-gnu/bin/python3.11` exists, then run:

```bash
cd /home/wenyu/summer

envs/python-builds/cpython-3.11-linux-x86_64-gnu/bin/python3.11 \
  -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements/next-cnn-cu128.txt
python -m pip install -e .
```

`requirements/next-cnn-cu128.txt` separately installs the CUDA 12.8 build of PyTorch for the
RTX 5090. `pyproject.toml` installs the evaluator, HDF5/Parquet I/O, plotting, and NEXT data
reader dependencies.

## 2. Common Workflows

### 2.1 Evaluate a NEXT Checkpoint with One Command

First inspect the resolved configuration without running inference or writing files:

```bash
energybench next \
  02_models/checkpoints/NEXTCNN_next_cnn_v1_run2_best.pt \
  --dry-run
```

Run the full evaluation with:

```bash
energybench next \
  02_models/checkpoints/NEXTCNN_next_cnn_v1_run2_best.pt
```

Format-version-3 alternative checkpoints use the same command, for example:

```bash
energybench next \
  02_models/checkpoints/NEXTALT_gnn_002_particlenet_edgeconv_classification_best.pt \
  --device cuda:0
```

The program reads the data root and preprocessing settings from the checkpoint and writes the
outputs to a new `04_evaluations/<model-id>/` directory. Common overrides are:

```bash
energybench next CHECKPOINT.pt \
  --data /path/to/NEXT \
  --device cuda:0 \
  --batch-size 32 \
  --output-dir 04_evaluations/my-run
```

Use `--max-files-per-class 1` for a small-scale workflow check. Use `--no-plots` to write only
the tables and audit data.

See [ALTERNATIVE_ARCHITECTURES.md](ALTERNATIVE_ARCHITECTURES.md) and
[TMUX_TRAINING.md](TMUX_TRAINING.md) for the ten non-Transformer models and the formal queue.

### 2.2 Evaluate an Existing Prediction Table

Inspect the table first, then run strict evaluation:

```bash
energybench inspect predictions_test.npz

energybench evaluate predictions_test.npz \
  --manifest manifests/next_0nubb_vs_bi214.yaml \
  --model-id model-v1-run2 \
  --output-dir 04_evaluations/model-v1-run2/evaluation_test \
  --strict
```

For a multitask table containing both `energy_target` and `energy_pred`, use:

```bash
--manifest manifests/next_0nubb_vs_bi214_multitask.yaml
```

Use `manifests/next_0nubb_vs_bi214_no_energy.yaml` only when no energy variable is available for
energy-spectrum matching. In that case, matched AUC is not applicable and only inclusive AUC is
reported.

As a rule, the output directory should be new. Add `--allow-existing` only when you explicitly
intend to replace known EnergyBench artifacts in that directory; unknown files are never deleted
automatically.

### 2.3 Export Predictions from Another Model

A model adapter accepts checkpoint and data paths and returns event-level columns:

```bash
energybench predict \
  --adapter examples/adapter_template.py:predict \
  --model /path/to/checkpoint \
  --data /path/to/data \
  --output predictions_test.npz
```

To implement a new adapter, copy `examples/adapter_template.py`, return at least the fields
required by the task, and ensure that every column has the same length along its first dimension.
Adapters are executable Python; run only trusted code.

### 2.4 Compare Models

Models may be formally ranked only when their event/truth data, evaluation protocol, and scoring
source fingerprint are identical, and every result comes from a successful strict run:

```bash
energybench compare \
  04_evaluations/model-a/evaluation_test \
  04_evaluations/model-b/evaluation_test \
  --output-dir 04_evaluations/comparison
```

`--allow-mixed-data` produces a non-rankable inventory only; it does not make the results
comparable.

### 2.5 Score Decorrelation (Optional)

Fit decorrelation only on an independent calibration-background sample, then apply it to the test
sample:

```bash
energybench decorrelate predictions_test.npz \
  --calibration predictions_calibration.npz \
  --output predictions_test_decorrelated.npz \
  --score-column score \
  --energy-column energy_condition \
  --label-column label \
  --event-id-column event_id \
  --split-column split
```

Do not use `--allow-overlap` for official results.

## 3. Train CNN-001 Classification and Energy Regression

The paired training programs require CUDA and do not automatically fall back to the CPU. Run the
smoke checks first:

```bash
python 01_code/architectures/cnn_001_two_conv_baseline/train_classification.py --smoke
python 01_code/architectures/cnn_001_two_conv_baseline/train_energy_regression.py --smoke
```

Then run either or both full training jobs:

```bash
python 01_code/architectures/cnn_001_two_conv_baseline/train_classification.py
python 01_code/architectures/cnn_001_two_conv_baseline/train_energy_regression.py
```

Common settings are defined near the top of each program and can also be overridden temporarily
with environment variables:

```bash
NEXT_DATA_ROOT=/path/to/NEXT \
NEXT_BATCH_SIZE=32 \
NEXT_NUM_EPOCHS=20 \
NEXT_MODEL_SUFFIX=_cnn_001_regression_experiment \
python 01_code/architectures/cnn_001_two_conv_baseline/train_energy_regression.py
```

By default, an existing checkpoint is not overwritten; the programs automatically append `_run2`,
`_run3`, and so on. Use `--allow-existing` only after confirming that an overwrite is intended.
The architecture, preprocessing difference between tasks, and exact parameter count are documented
in `01_code/architectures/cnn_001_two_conv_baseline/README.md`.

The current project does not contain the historical v2 training source or a v2 checkpoint.
`src/next_cnn/model.py` and the adapter can still load a compatible v2 checkpoint, but the
historical v2 evaluation directories are not sufficient to reproduce the complete training run.

## 4. Project Structure

```text
summer/
├── 01_code/
│   ├── architectures/
│   │   └── cnn_001_two_conv_baseline/
│   │       ├── train_classification.py
│   │       ├── train_energy_regression.py
│   │       └── README.md
│   └── src/project_paths.py  # Checkpoint, log, and figure output locations
├── 02_models/checkpoints/    # Best/last checkpoints
├── 03_training_runs/
│   ├── logs/                 # Epoch CSV files and history JSON files
│   └── history_plots/        # Training curves and validation-score plots
├── 04_evaluations/           # Prediction tables and evaluation results
├── src/energybench/
│   ├── cli.py                # Command entry points and arguments
│   ├── config.py             # Manifest defaults and validation
│   ├── data.py               # NPZ/CSV/HDF5/Parquet table readers
│   ├── evaluation.py         # Evaluation workflow, strict checks, and artifact writing
│   ├── roc.py                # Inclusive/matched ROC
│   ├── regression.py         # ERS-v1 and regression metrics
│   ├── dependence.py         # Score-energy dependence
│   ├── plotting.py           # Four evaluation figures
│   ├── reporting.py          # results.csv and leaderboard.csv
│   ├── decorrelation.py      # Background conditional ECDF
│   └── next_workflow.py      # `energybench next`
├── src/next_cnn/
│   ├── model.py              # CNN-001 and historical multitask network definitions
│   ├── data.py               # HDF5 discovery, split, three-view projection, and Dataset
│   └── adapter.py            # Checkpoint → canonical prediction table
├── manifests/                # Fixed tasks and scoring parameters
├── examples/                 # Synthetic-table and adapter examples
├── docs/                     # Standards and this guide
├── pyproject.toml            # Package, entry points, and default dependencies
└── requirements/             # CUDA PyTorch versions
```

## 5. Making Changes

| Goal | Where to modify | Required checks after modification |
|---|---|---|
| Change classes, positive/negative roles, or scoring parameters | `manifests/*.yaml` | Task/model/data semantics, energy units, and split are all explicit |
| Change canonical column names | Manifest `columns`; `src/energybench/data.py` if necessary | `energybench inspect` reports the correct mapping |
| Change matched AUC | `src/energybench/roc.py`; policy thresholds are in `evaluation.py` | Update the formula, NA rules, and protocol/schema versions together |
| Change ERS-v1 | `src/energybench/regression.py` | Update score components, range, bootstrap procedure, and standards documentation together |
| Change result tables | `src/energybench/reporting.py` | Update the CSV header, field definitions, and schema version together |
| Change evaluation figures | `src/energybench/plotting.py` | Ensure axes, units, and sampling/weight descriptions remain unambiguous |
| Change a legacy NEXT CNN | `src/next_cnn/model.py` | v2 checkpoint detection and the state dict remain consistent |
| Change an alternative classifier | `src/next_alt/models/` and `src/next_alt/registry.py` | v3 model config, batch keys, and strict state-dict reconstruction remain consistent |
| Change projections or data split | `src/next_cnn/data.py` | Training and inference use the same implementation; old-checkpoint provenance is not presented as a new protocol |
| Change training hyperparameters or loop | The applicable program under `01_code/architectures/` | New model suffix, validation selection rule, and log fields are explicit |
| Integrate a new model framework | New adapter + manifest | Per-event alignment, stable event IDs, trustworthy split, units, and score direction |

When scoring definitions or field semantics change, do not reuse an old protocol/schema name.
Increment the version and keep old and new results separate. Existing `04_evaluations/` directories
are result evidence; do not edit their values manually or use them as source-code templates.

## 6. Minimal Verification

There is no need to create a `tests/` directory after making changes. Use temporary output for
workflow verification:

```bash
PYTHONDONTWRITEBYTECODE=1 python -c "import energybench, next_cnn; print('imports ok')"
energybench --help
energybench next \
  02_models/checkpoints/NEXTCNN_next_cnn_v1_run2_best.pt \
  --dry-run
```

To run the complete synthetic workflow, write its output to `/tmp`:

```bash
tmp_dir="$(mktemp -d /tmp/energybench-check.XXXXXX)"

python examples/make_synthetic_predictions.py \
  --output "$tmp_dir/predictions.npz" \
  --events 2000

energybench evaluate "$tmp_dir/predictions.npz" \
  --manifest examples/manifest.yaml \
  --output-dir "$tmp_dir/evaluation" \
  --roc-bootstrap 10 \
  --regression-bootstrap 10 \
  --strict
```

If a scoring algorithm changes, 10 bootstrap iterations can confirm only that the workflow runs;
they are not valid for an official confidence interval.
