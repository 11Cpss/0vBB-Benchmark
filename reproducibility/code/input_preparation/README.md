# Prediction inputs and physical-event alignment

The final evaluator consumes scalar scores, true labels, original physical
energy and event IDs. Model inference exporters are documented under
[Classic models](../classic/README.md) and [Transformers](../transformers/README.md).

`standardize.py` renames native arrays, orients labels/scores, converts physical
MeV to keV, and optionally joins original energy metadata by unique event IDs.
It does not clip, filter, resample, fit a model, or calculate metrics.

```bash
# A model export with clean/signal-oriented scores and energy_kev in keV.
.venv-reproduce/bin/python -B code/input_preparation/standardize.py \
  --input /path/to/native_predictions.npz --energy-key energy_kev \
  --energy-unit keV --weight-key weight --group-key group \
  --output outputs/predictions.npz

# EXO-200 native binary logits favor label 1 (multiple charge clusters).
# The paper positive class is native label 0 (one charge cluster).
.venv-reproduce/bin/python -B code/input_preparation/standardize.py \
  --input /path/to/exo_native.npz --energy-key energy_keV --energy-unit keV \
  --positive-label 0 --score-label 1 --output outputs/exo_predictions.npz
```

Use actual field names with `--score-key`, `--label-key`, `--event-id-key`,
`--energy-key`, `--weight-key` and `--group-key`. Omit optional weight/group
arguments only when unit base weights and binary-label groups are appropriate.
Apply label/score conversion once: a model exporter that already produces
paper-oriented scores needs no EXO sign reversal.

If physical energy is in a separate NPZ, pass `--energy-metadata FILE`. Both
files must have physical event IDs; all prediction IDs must match the energy
metadata. If both contain native labels, those labels must also agree. Equal
array lengths alone do not establish event correspondence.

| Dataset | Required semantics |
| --- | --- |
| NEXT | Positive 0nubb; negative Bi214; scalar signal logit; physical summed hit energy in MeV converted to keV; preserve the recorded split |
| MJD | Positive means all four reference PSD flags pass; original float64 `energy_label` in keV; preserve raw shard and row identity |
| EXO-200 | Positive is native class 0, one charge cluster; flip native binary labels and negate a background-oriented scalar logit; original rotated energy in keV |
| SuperNEMO | Training is 2nu/Bi214; the separate 0nu/Bi214 energy-only illustration uses physical calorimeter energy as its score; keep the tasks separate |

For MJD GINE, use the model exporter's stable scalar construction from four PSD
logits: the logit of the product of the four sigmoid probabilities. Do not
substitute one output channel or average the logits. Physical energy cannot be
recovered by undoing a clip or by casting a float32 cache to float64; export
it from the original dataset or join verified raw-energy metadata.

The NPZ companion used by published-result replay is already standardized.
Do not standardize it a second time, change its test population, or replace a
missing checkpoint with another model.
