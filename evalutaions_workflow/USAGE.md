# Simple EnergyBench Usage Guide

This guide is the technical handoff for collaborators who want to train and
evaluate a PyTorch model without reading the legacy project. The companion
[`../next_detector/notebooks/next_energybench_train.ipynb`](../next_detector/notebooks/next_energybench_train.ipynb)
connects this workflow to the NEXT Transformer with the standardized
configuration unchanged.

## 1. Install and start

Use Python 3.11 and install the PyTorch build appropriate for the machine. From
the repository root, install the remaining dependencies, install the local
packages, and start Jupyter:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
jupyter lab
```

The package has five normal entry points:

```python
prepare_dataset(...)
train_model(...)
evaluate_classification(...)
evaluate_regression(...)
set_seed(...)
```

## 2. Source-data contract

`prepare_dataset` expects this extracted NEXT layout:

```text
DATA_ROOT/
├── 0nubb_part_*/*.h5
└── Bi_part_*/*.h5
```

Each HDF5 file must contain the one-dimensional structured dataset
`/MC/hits/table` with these blocks:

| HDF5 field | Meaning | Required shape per row |
|---|---|---|
| `values_block_0` | event ID | `(1,)` |
| `values_block_1` | `x`, `y`, `z`, hit energy | `(4,)` |
| `values_block_2` | `Signal` or `Bkg` | `(1,)` |

Rows are detector hits, not events. All rows for one event ID must be
contiguous. Repeated non-contiguous event IDs are rejected. Coordinates are
interpreted in millimetres, and the float64 sum of hit energies is the event's
physical energy in MeV. Hit energies must be finite and non-negative; every
event must have positive total energy.

The default class mapping is:

```text
0nubb -> label 1
Bi214 -> label 0
```

## 3. Standard configuration

Use the dataclasses directly so the settings saved with an experiment are
explicit and serializable:

```python
from simple_energybench import EvaluationConfig, TrainingConfig

training_config = TrainingConfig()
evaluation_config = EvaluationConfig()
```

The collaboration-standard training defaults are:

| Field | Default |
|---|---:|
| batch size | `64` |
| epochs | `50` |
| learning rate | `0.0005` |
| weight decay | `0.0001` |
| optimizer | AdamW |
| scheduler | cosine annealing |
| gradient clip norm | `1.0` |
| early-stopping patience | `5` epochs |
| early-stopping minimum delta | `0.0` |
| classification loss | binary cross-entropy with logits |
| regression loss | mean squared error in MeV |
| seed | `42` |
| deterministic algorithms | disabled |
| AMP | enabled on CUDA, automatic precision |
| device | automatic CUDA/CPU selection |
| DataLoader workers | `0` |

`EvaluationConfig()` uses 5 keV bins, overlap energy matching, a 90% target
signal-efficiency operating point, minimum per-class/bin counts of 20, at
least two valid bins, 0.5% support trimming, minimum matching coverage of 50%,
20 score bins, 10 ERS performance bins, and seed 42. The complete authoritative
field list is in [`energybench/config.py`](energybench/config.py), exposed
through the public `simple_energybench` package.

The grid fields are protocol-locked at 0 keV, 3000 keV, 5 keV, and 600 bins;
other values are rejected. If a study intentionally changes a non-grid
setting, construct a new dataclass with that field named and retain the
generated `history.json` and `metrics.json`. Do not hide experiment settings
in unrelated notebook cells.

## 4. Prepare the event split

```python
from pathlib import Path
from simple_energybench import prepare_dataset, set_seed

data_root = Path("/path/to/NEXT")
set_seed(training_config.seed, training_config.deterministic)

classification_data = prepare_dataset(
    data_root,
    batch_size=training_config.batch_size,
    mode="classification",
    seed=training_config.seed,
    num_workers=training_config.num_workers,
    manifest_path="results/event_split.json",
)

print(classification_data.counts)
```

`prepare_dataset` returns a `PreparedData` object with:

- `train_loader`, `validation_loader`, and `test_loader`;
- `counts`, including totals, fractions, per-class counts, and boundary files;
- `manifest_path`;
- the selected input `mode` and resolved `ProjectionConfig`.

The split is based on event counts, not file counts. The implementation:

1. inventories and validates the HDF5 files;
2. counts contiguous event-ID runs in bounded chunks;
3. shuffles sorted file paths independently within each class using the seed;
4. assigns the nearest class-stratified 80/10/10 cumulative event targets;
5. cuts only files that cross a split boundary, storing half-open event-ordinal
   ranges such as `[event_start, event_stop)`.

Every selected event belongs to exactly one split, no hit data are copied, and
the reported totals may differ from exactly 80/10/10 only by integer rounding.
The manifest records relative paths and file inventory fingerprints based on
path, size, and modification time. It is reused only when the inventory, seed,
fractions, class map, and file-selection settings still match.

### Input representations

The default geometry produces three `XY`, `XZ`, and `YZ` projections with
shape `[3, 128, 128]`, 30 mm pixels, and origin
`(-1920, -1920, -120)` mm.

| `mode` | Pixel values | Intended task |
|---|---|---|
| `classification` | energy-weighted, divided by total event energy, then scaled by 100 | classification |
| `regression_energy` | energy-weighted and scaled by 100 without event normalization | energy-flow regression baseline |
| `regression_topology` | binary occupancy, with no amplitude information | topology-only regression |

`projection_coverage` reports the fraction of hit energy retained inside the
projection grid for energy modes, or the fraction of hits retained for binary
occupancy.

`regression_energy` exposes amplitudes whose sum determines the target. Treat
it as a data-flow baseline, not evidence that topology predicts energy. Use
`regression_topology` for that scientific question.

## 5. Train and evaluate a classifier

```python
from simple_energybench import (
    SimpleClassifier,
    evaluate_classification,
    train_model,
)

classifier = SimpleClassifier()

classification_history = train_model(
    classifier,
    classification_data.train_loader,
    classification_data.validation_loader,
    config=training_config,
    task="classification",
    output_dir="results/classification_training",
)

classification_results = evaluate_classification(
    classifier,
    classification_data.test_loader,
    device=training_config.device,
    output_dir="results/classification",
    config=evaluation_config,
)

print(classification_results["auc"])
print(classification_results["matched_auc"])
print(classification_results["energy_independence_score"])
```

Training selects the best epoch by validation AUC and restores those weights
into `classifier`. Classification evaluation expects raw logits; it applies
no sigmoid before AUC calculations because ranking metrics are invariant under
that monotonic transformation.

## 6. Train and evaluate a regressor

Reuse the classification manifest so both tasks use identical events:

```python
from simple_energybench import SimpleRegressor, evaluate_regression

regression_data = prepare_dataset(
    data_root,
    batch_size=training_config.batch_size,
    mode="regression_energy",
    seed=training_config.seed,
    num_workers=training_config.num_workers,
    manifest_path=classification_data.manifest_path,
)

regressor = SimpleRegressor()
regression_history = train_model(
    regressor,
    regression_data.train_loader,
    regression_data.validation_loader,
    config=training_config,
    task="regression",
    output_dir="results/regression_training",
)

regression_results = evaluate_regression(
    regressor,
    regression_data.test_loader,
    device=training_config.device,
    output_dir="results/regression",
    config=evaluation_config,
)

print(regression_results["ers"])
print(regression_results["mae"])
print(regression_results["rmse"])
```

Training selects the best epoch by the lowest validation RMSE and restores
those weights into `regressor`. Targets and predictions are physical MeV.

## 7. Bring your own model

There is no model registry. Pass any `torch.nn.Module` that follows one of
these contracts:

```text
Classification: model(inputs) -> floating raw logits [B] or [B, 1]
Regression:     model(inputs) -> floating MeV predictions [B] or [B, 1]
```

For the default loader, `inputs` is a floating tensor with shape
`[B, 3, 128, 128]`. Training requires one finite scalar per input event. A
classifier must not return two-class softmax probabilities, and a regressor
must not return normalized or keV values unless they are converted back to MeV
inside the model. Regression evaluation deliberately accepts non-finite
predictions as recorded failures: `finite_fraction` decreases and therefore
penalizes ERS-v1.

Example:

```python
import torch
from torch import nn

class MyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(1)
```

## 8. Custom DataLoader contract

The default DataLoader yields this mapping:

```python
{
    "inputs": tensor,                 # [B, 3, H, W]
    "label": tensor,                  # [B], numeric 0 or 1
    "energy": tensor,                 # [B], finite physical MeV
    "event_id": list[str],            # unique within an evaluation
    "category": list[str],
    "group_id": list[str],            # source HDF5 relative path
    "split": list[str],
    "sample_weight": tensor,          # [B], finite and non-negative
    "projection_coverage": tensor,    # [B]
}
```

For training, classification requires `inputs` and `label`; regression
requires `inputs` and `energy`. `sample_weight` is optional and defaults to
one. For evaluation, both tasks require `inputs` and `energy`, and
classification additionally requires `label`. Metadata fields are optional,
but preserving unique `event_id`, `group_id`, and `split` values makes the
saved prediction evidence auditable.

`inputs` may be a tensor or an arbitrarily nested composition of mappings,
tuples, and lists with tensor leaves. Training and evaluation recursively move
those tensors to the selected device while preserving the container shape.

## 9. The canonical 5 keV energy grid

EnergyBench uses the canonical deposited-energy grid from 0 through 3000 keV:

```python
canonical_edges_kev = np.arange(0.0, 3000.0 + 5.0, 5.0)
```

This is 601 edges defining exactly 600 bins of width 5 keV. The Python API
stores physical energies in MeV, so the equivalent width is 0.005 MeV and the
canonical range is `[0.0, 3.0]` MeV.

For each metric, the framework crops this canonical grid to the smallest
contiguous canonical slice covering the energies used by that calculation.
Classification matching uses its trimmed class-common support; dependence
uses its full finite analysis range; regression histogram and diagnostic
curves use the finite truth range. Therefore matching may intentionally omit
populated energies outside common support. Cropping never shifts the grid:
retained edges stay aligned to integer multiples of 5 keV and never start at
an arbitrary observed value. Bins are left-closed and right-open, except that
the final retained right edge includes values exactly on that edge. Evaluated
truth/conditioning energies must lie within 0–3000 keV.

The range requirement applies to truth/conditioning energy, not to regression
predictions. Finite predictions outside the cropped truth interval—including
values outside 0–3000 keV—are preserved in explicit histogram underflow or
overflow mass and remain part of the regression diagnostics.

The cropped fixed grid is used for classification matching and dependence,
regression spectrum comparison, and energy-dependent plots. Sparse bins remain
visible in counts/masks but do not contribute estimates until the configured
minimum population is met.

ERS-v1 has one deliberate exception: its balanced event-error component uses
weighted truth-quantile performance bins. Equal-population bins are part of
that metric's definition. `metrics.json` records the fixed canonical edges,
the ERS performance-bin edges, and all protocol settings.

## 10. Metrics and artifacts

Classification reports inclusive AUC, common-support AUC, formal and
diagnostic energy-matched AUC, shortcut gap, target-TPR operating point,
coverage, effective sample sizes, matching-balance diagnostics, and
class-conditional score/energy dependence.

Regression reports ERS-v1 and its component scores, MAE, RMSE, bias, R², MAE
skill, fractional bias and resolution, balanced fractional MAE, JSD,
histogram overlap, and Wasserstein distance.

Training directories contain:

```text
best_model.pt
last_model.pt
history.json
training_history.png
```

Classification evaluation directories contain:

```text
metrics.json
results.csv
predictions.npz
energy_matched_roc.png
score_energy_dependence.png
```

Regression evaluation directories contain:

```text
metrics.json
results.csv
predictions.npz
energy_regression.png
energy_histograms.png
```

The returned Python dictionary matches `metrics.json`. `results.csv` is the
one-row scalar summary. `predictions.npz` stores the event-level evidence
needed to reproduce metrics without another model inference pass. Large ROC
threshold arrays are intentionally omitted from JSON.

Outputs are protected by default: an existing artifact raises
`FileExistsError`. Pass `overwrite=True` only when replacing that experiment's
outputs is intentional.

## 11. Reproducibility checklist

- Call `set_seed(config.seed, config.deterministic)` before preparing data or
  constructing a model.
- Retain `event_split.json`, `history.json`, `metrics.json`, and the prediction
  NPZ with the experiment.
- Reuse the same manifest for classification and regression comparisons.
- Record any non-default dataclass fields and custom model constructor values.
- Do not compare the fixed-grid protocol directly with legacy quantile-bin
  results unless the old predictions are reevaluated through this package.
- Use a grouped uncertainty method if adding confidence intervals; events from
  the same HDF5 source may be correlated.
