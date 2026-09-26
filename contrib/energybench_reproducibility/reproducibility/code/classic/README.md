# Classic models

Training, model definitions, preprocessing and prediction export for the paper's
classic-model results. Reported EnergyBench metrics are computed only by
[`../../benchmark/evaluate.py`](../../benchmark/evaluate.py); these model workflows
export predictions and ordinary training diagnostics.

| Paper model | Architecture ID |
|---|---|
| Multi-view CNN | `cnn_004_multiview_late_fusion` |
| Static GINE | `gnn_001_static_gine` |
| BiGRU | `seq_001_bigru` |
| PointMamba-lite | `ssm_001_pointmamba` |

The NEXT directory additionally contains the architecture sweep used in the
appendix. Effective settings and split definitions are in each dataset's
`configurations/`. Raw detector files and checkpoint weights must be obtained
separately. Saved event predictions, when supplied, allow metric reproduction
without retraining. See the [training environment](../../environments/classic/README.md).

## NEXT

Run from `code/classic/NEXT` in an activated environment:

```bash
mkdir -p outputs/reproduction
cp configurations/event_split.json outputs/reproduction/event_split.json
python 01_code/architectures/gnn_001_static_gine/train_classification.py \
  --data /path/data/NEXT --manifest outputs/reproduction/event_split.json \
  --output-dir outputs/reproduction/gnn_001_static_gine --device cuda
```

Substitute another architecture directory for the appendix sweep. The fixed
split uses 116,549 test events. The loader rejects an existing incompatible
manifest before event counting or split allocation when the source inventory
or settings change. A new experiment can create a split at a new manifest path. Architecture YAMLs and `configurations/<architecture>.json` contain
model, representation and training settings.

`src/next_training` supplies the original loader and optimizer/validation
selection loop. Post-training export writes logits, labels, event IDs, groups,
weights and physical `energy_keV` in `evaluation/predictions.npz`. It does not
compute energy-conditioned metrics. Conversion from the loader's MeV energy is
explicit and does not clip the physical values.

The paper's retained PointMamba-lite result uses the separate file-based split
(115,499 test events). Its format-version-3 training configuration is
`configurations/ssm_001_pointmamba.yaml`. Copy this YAML into a new output directory
and set `data.root` and the three `output` destinations before training:

```bash
PYTHONPATH=src python -c 'from next_alt.training import main_for_architecture; main_for_architecture("ssm_001_pointmamba", "outputs/pointmamba_config.yaml")'
```

For an externally supplied checkpoint with that earlier file-split format:

```bash
python scripts/predict_checkpoint.py --checkpoint /path/checkpoints/model.pt \
  --data /path/data/NEXT --output outputs/predictions.npz --device cuda:0
```

This checkpoint exporter is for the earlier adapter format, not the canonical
`best_model.pt` format. Do not exchange the two checkpoints or test populations.
Canonical training exports its selected model's predictions automatically.

## MJD

Run from `code/classic/MJD`:

```bash
python architectures/gnn_001_static_gine/train_classification.py \
  --data-root /path/data/MJD --output-dir outputs/reproduction/gnn_001_static_gine \
  --device cuda
```

Replace the architecture ID with one of the four IDs above. The official test
partition has 390,000 events. GINE learns four PSD pass labels and constructs
the clean score as the logit of their pass-probability product; the other three
models emit a binary clean logit. Label 1 is clean.

For checkpoint inference, pass the exact saved configuration and checkpoint:

```bash
python scripts/run_checkpoint_inference.py gnn_001_static_gine \
  --task classification --batch-size 16 \
  --checkpoint /path/checkpoints/gnn_001_static_gine/best.pt \
  --config configurations/gnn_001_static_gine.json --data-root /path/data/MJD \
  --output-dir outputs/reproduction/gnn_001_static_gine_inference
```

This exporter keeps the original model score construction and training targets.
It now carries raw float64 `energy_label` through a separate `energy_keV` column,
with unique `source_file:row` event IDs captured from the same inference batches.
No physical energy is clipped or recovered from rounded caches. `score`, `label`,
`group` and `weight` are ready for the central benchmark. The default is CUDA/AMP;
`--device cpu --no-amp` is available, but changing inference precision can change
scores. Do not use `--max-events` for a paper result.

## EXO-200

Run from `code/classic/EXO200`:

```bash
python architectures/gnn_001_static_gine/train_classification.py \
  --data-root /path/data/EXO-200 --output-dir outputs/reproduction/gnn_001_static_gine \
  --device cuda
```

The output destination must be inside this copied project. The deterministic
run-group split has 140,383 test events; compare run membership and class counts
with `configurations/<architecture>.json`. Signal is charge-cluster count 1,
native label 0. Native scores are background logits: standardization uses
`label = 1 - native_label` and `score = -native_logit`. Conditional energy is
raw float64 `Rotated_energy` in keV. Prediction export records `energy_keV` and
stable `EXO200::<source_file>::<event_number>` IDs from the same inference batches.
Pass these native scores through the EXO-200 standardization step before evaluation.

The checkpoint-inference API is `exobench.data.prepare_dataset`, the architecture's
`model.build_model`, strict `load_state_dict(checkpoint["model_state_dict"])`, and
`exobench.training.evaluate_model`. The training workflow shows their call order;
there is no separate EXO checkpoint CLI. Supply the saved effective configuration,
not another architecture's defaults. Saved standardized predictions are the
supported direct route to final-metric reproduction.

## SuperNEMO

Run from `code/classic/SuperNEMO`:

```bash
python -m supernemobench.workflow --task classification \
  --model gnn_001_static_gine --mode train --data-root /path/data/SuperNEMO --device cuda
python -m supernemobench.workflow --task classification \
  --model gnn_001_static_gine --mode test --data-root /path/data/SuperNEMO --device cuda \
  --checkpoint /path/checkpoints/gnn_001_static_gine/best.pt
```

The model task is 2nu (label 1) versus Bi214 (label 0), using tracker topology.
It is distinct from the paper's separate 0nu energy-only illustration. Conditional
energy is calorimeter `E1 + E2` in keV. The source inventory and block split are
in `configurations/split_manifest.json`; binary event-offset caches are regenerated
from raw data. Keep checkpoint/data-identity checks enabled.

The four classic checkpoints use ordinary validation AUC for early stopping
and best-epoch selection. This agrees with each supplied checkpoint's score
and selected epoch; the Transformer package has its own distinct selection
criterion. Test mode exports aligned predictions for the central benchmark.

## Reproduction scope

The release retains ordinary AUC/loss routines needed to reproduce model selection.
Model classes, optimization losses and input representations were not replaced.
CLI/import and synthetic prediction-export checks passed. Full retraining and
checkpoint inference require the separately distributed data and exact weights;
they were not rerun for release packaging.
