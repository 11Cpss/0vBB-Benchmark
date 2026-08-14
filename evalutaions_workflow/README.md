# Simple EnergyBench

Simple EnergyBench is a small, standalone PyTorch workflow for the NEXT
binary-classification and deposited-energy-regression tasks. It prepares the
HDF5 data, trains any compatible model, runs the established EnergyBench
metrics, and writes the standard plots with five public functions.

The package does **not** import the legacy project. It contains no CLI,
registry, adapter, YAML manifest, or checkpoint-format dispatcher.

## Start here

- Open
  [`../next_detector/notebooks/next_energybench_train.ipynb`](../next_detector/notebooks/next_energybench_train.ipynb)
  for the standardized Transformer training workflow.
- Open
  [`../next_detector/notebooks/next_energybench_results.ipynb`](../next_detector/notebooks/next_energybench_results.ipynb)
  after training for the read-only results dashboard.
- Read [`USAGE.md`](USAGE.md) for the exact dataset, split, configuration,
  model, DataLoader, energy-grid, and artifact contracts.
- Inspect [`energybench/config.py`](energybench/config.py) for the
  authoritative dataclass fields and standardized defaults exposed through
  the public `simple_energybench` package.

The notebook instantiates `TrainingConfig()` and `EvaluationConfig()`
unchanged. It is the full standardized workflow, including 50 training epochs
per applicable task, rather than a reduced demonstration configuration.

## Files

```text
evalutaions_workflow/
├── README.md
├── USAGE.md
├── simple_energybench/          # public, collision-free package entry
│   └── __init__.py
├── energybench/                 # bundled implementation reused by the entry
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── models.py
│   ├── training.py
│   ├── metrics.py
│   ├── plotting.py
│   └── evaluation.py
└── tests/
    └── test_workflow.py
```

## Install

Use Python 3.11. From the repository root, install the PyTorch build
appropriate for your CPU or CUDA system, then install the remaining
requirements and the local packages:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Run Jupyter from the repository root:

```bash
jupyter lab
```

Use `simple_energybench` for all imports. Its lightweight package entry loads
the bundled implementation without requiring collaborators to import the
generic `energybench` name, which may belong to another project.

## Dataset format

The loader expects the extracted NEXT layout:

```text
DATA_ROOT/
├── 0nubb_part_*/*.h5
└── Bi_part_*/*.h5
```

Every file must contain `/MC/hits/table`. Its rows are detector hits, not
events. The required fields are:

- `values_block_0[:, 0]`: event ID;
- `values_block_1[:, :4]`: `x`, `y`, `z`, and deposited energy;
- `values_block_2[:, 0]`: `Signal` or `Bkg`.

Rows belonging to one event must be contiguous. Event energy is the float64
sum of its hit energies and is interpreted as MeV.

## Event-count split

`prepare_dataset` scans event IDs in chunks and allocates events—not files—to
train, validation, and test partitions. Files are shuffled reproducibly within
each class, and only files crossing an 80/10/10 boundary are sliced. The split
therefore preserves class composition without copying event data or spreading
every HDF5 file across every partition.

The JSON manifest stores relative paths, per-file event counts, and half-open
event ranges. It is reused until the file inventory, seed, fractions, class
mapping, or optional file limit changes. Preparation prints total and per-split
event counts and reports the few boundary files shared by partitions.

## Minimal classification workflow

```python
from pathlib import Path

from simple_energybench import (
    SimpleClassifier,
    TrainingConfig,
    evaluate_classification,
    prepare_dataset,
    set_seed,
    train_model,
)

data_root = Path("/path/to/NEXT")
config = TrainingConfig()  # supplied 64 / 50 / 5e-4 standard
set_seed(config.seed, config.deterministic)

data = prepare_dataset(
    data_root,
    batch_size=config.batch_size,
    mode="classification",
    seed=config.seed,
    num_workers=config.num_workers,
)

# Replace this with any torch.nn.Module having the documented output shape.
model = SimpleClassifier(base_channels=8)
history = train_model(
    model,
    data.train_loader,
    data.validation_loader,
    config=config,
    task="classification",
    output_dir="results/classification/training",
)
metrics = evaluate_classification(
    model,
    data.test_loader,
    device=config.device,
    output_dir="results/classification",
)
print(metrics["auc"], metrics["matched_auc"])
```

Classification batches contain normalized three-view energy projections with
shape `[batch, 3, 128, 128]`. A classifier must return one raw positive-class
logit per event as `[batch]` or `[batch, 1]`.

## Regression workflow

```python
from simple_energybench import SimpleRegressor, evaluate_regression

regression_data = prepare_dataset(
    data_root,
    batch_size=config.batch_size,
    mode="regression_energy",       # or "regression_topology"
    seed=config.seed,
    manifest_path=data.manifest_path,
)
regressor = SimpleRegressor(base_channels=8)
train_model(
    regressor,
    regression_data.train_loader,
    regression_data.validation_loader,
    config=config,
    task="regression",
    output_dir="results/regression/training",
)
regression_metrics = evaluate_regression(
    regressor,
    regression_data.test_loader,
    device=config.device,
    output_dir="results/regression",
)
print(regression_metrics["ers"], regression_metrics["rmse"])
```

A regressor returns physical energy in MeV as `[batch]` or `[batch, 1]`.

`regression_energy` preserves deposited-energy amplitude. Because the target is
the sum of those same amplitudes, this is primarily a data-flow baseline.
`regression_topology` replaces hit amplitudes with binary occupancy and is the
stricter choice when studying whether topology predicts energy.

## Batch and model contract

The default loader yields:

```python
{
    "inputs": tensor,             # [B, 3, H, W]
    "label": tensor,              # [B], numeric 0/1
    "energy": tensor,             # [B], MeV
    "event_id": list[str],
    "category": list[str],
    "group_id": list[str],
    "split": list[str],
    "sample_weight": tensor,
    "projection_coverage": tensor,
}
```

Training and evaluation call only `model(batch["inputs"])`. A collaborator can
therefore replace the reference CNN directly. A custom DataLoader may provide
another fixed input representation while keeping the remaining keys. Inputs
may also be arbitrarily nested mappings, tuples, and lists whose tensor leaves
are moved to the configured device recursively.

## Metrics and 5 keV bins

Classification reports inclusive, common-support, diagnostic matched, and
formal energy-matched AUC; the target-TPR operating point; coverage; effective
sample sizes; matching balance; shortcut gap; and class-conditional energy
dependence. Dependence includes weighted Pearson/Spearman, distance
correlation, score-distribution JSD, independence, acceptance, and sculpting.

Regression reports ERS-v1 and its finite/event/histogram components together
with MAE, RMSE, bias, R², MAE skill, fractional bias/resolution, balanced
fractional MAE, JSD, overlap, and Wasserstein distance.

The canonical grid covers 0–3000 keV in exact 5 keV steps: 601 edges define
600 bins. Energies remain in MeV in the Python API, so the same grid spans
`[0.0, 3.0]` MeV with `0.005 MeV` spacing. Each metric retains the smallest
contiguous canonical slice covering the energies it evaluates: trimmed
class-common support for matching, and the full finite analysis range for
dependence and regression truth. Retained edges always remain aligned to
integer multiples of 5 keV. Bins are left-closed/right-open except for the
final inclusive edge. Matching, dependence, spectrum/JSD, and plotted
response curves use these fixed bins. ERS-v1 retains weighted truth-quantile
bins internally for its balanced event-error term because equal-population
performance bins are part of that definition. Both edge sets are saved in
`metrics.json`. See [USAGE.md](USAGE.md#9-the-canonical-5-kev-energy-grid) for
the precise cropping and edge contract.

This is a new evaluation protocol. Fixed-bin matched/dependence/histogram
values should not be compared directly with old quantile-bin results unless
the old prediction files are evaluated again with this package.

## Outputs

Classification writes:

```text
metrics.json
results.csv
predictions.npz
energy_matched_roc.png
score_energy_dependence.png
```

Regression writes:

```text
metrics.json
results.csv
predictions.npz
energy_regression.png
energy_histograms.png
```

`metrics.json` contains all scalar and binned diagnostics but omits very large
per-threshold ROC arrays. `predictions.npz` is the compact event-level evidence
needed to recompute them. Existing artifacts are not overwritten unless
`overwrite=True` is passed explicitly.

## Reproducibility and tests

`set_seed` covers Python, NumPy, PyTorch, CUDA, and cuDNN. AMP is used only on
CUDA; CPU execution works automatically.

```bash
python -m unittest discover -s tests -v
python -m compileall -q simple_energybench energybench
```

Event bootstrap confidence intervals are intentionally absent: events from
the same HDF5 file may be correlated, while the legacy event bootstrap did not
respect those groups.
