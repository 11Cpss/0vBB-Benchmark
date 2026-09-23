# EXO-200 binary classification

This workspace implements only the EXO-200 `signal` versus `background`
classification task. It follows the current `/home/wenyu/MJD` workflow: lazy
HDF5 loading, deterministic splitting, per-epoch validation, best-checkpoint
restoration, one final test evaluation, and architecture-specific MJD training
defaults with the same output artifacts.

## Fixed target contract

The only label source is `Charge_cluster_number`:

```python
nccl = f["Charge_cluster_number"][:]
labels = (nccl > 1).astype(...)
```

- `label = 0`: `signal`, exactly `nccl == 1`
- `label = 1`: `background`, exactly `nccl > 1`
- `class_names = ["signal", "background"]`

`Charge_cluster_number`, derived labels, and `Charge_Clusters_Pos` are never
model inputs. The last field is excluded because its number of columns equals
`nccl` for every inspected event and therefore directly leaks the target.

## Data and preprocessing

The default source is `/home/klz/Data/zeronu_benchmark/EXO-200`. Event `i` is
formed only from `Waveforms[str(i)]` and `Charge_cluster_number[i]` after exact
file-length and complete key-set validation. A waveform is `[226, 300] int16`.
It is converted to float32, baseline-subtracted per channel over the first 200
time samples, and divided by the event-wide maximum absolute amplitude, exactly
following the MJD classification preprocessing semantics.

The stable event identity is `(run_number, event_number)`. All files from one
run stay in the same split. The held-out test target is the current MJD official
test share (`6/22`); validation is 10% of the remaining pool. Group assignment
uses seed 42 and optimizes event and class-count proportions without splitting a
run. No resampling, class weights, augmentation, or training-set-derived
normalization statistics are added.

## Architectures

All five classifiers use the same Dataset, grouped split, DataLoaders,
single-logit BCE training loop, validation logic, and final test evaluation:

- `cnn_001_waveform`: the MJD 1D CNN with its first convolution adapted to the
  226 EXO waveform channels.
- `cnn_004_multiview_late_fusion`: three-resolution STFT views, shared 2D CNN,
  and late fusion.
- `gnn_001_static_gine`: adaptive waveform points and a static kNN GINE.
- `seq_001_bigru`: Hilbert and Trans-Hilbert point sequences with a BiGRU.
- `ssm_001_pointmamba`: the same dual point ordering with the pure-PyTorch
  PointMamba-lite selective SSM.

The latter four MJD models require one serialized waveform. Their only common
EXO representation adapter strictly reshapes the normalized canonical input
from `[226, 300]` to channel-major `[1, 67800]`. This deterministic reshape has
no parameters and discards no values. It does make the end of one detector
channel adjacent to the beginning of the next; that is the explicit minimal
format compromise used to keep the MJD STFT/tokenizer/backbones and model
parameters unchanged. It is not presented as a new physical channel topology.

MJD's GINE classification head is the one model-specific target exception: it
uses four MJD-only PSD labels. EXO-200 has only the fixed `nccl` binary target,
so the same GINE backbone ends in one background logit and uses the shared
binary loss. No `Charge_Clusters_Pos` data are used to build graphs.

## Current data status

The current 29 HDF5 files contain both fixed classes:

```text
signal:     244678
background: 270324
```

The workflow reports these counts and requires both classes in every grouped
split before constructing the model. Values below 1, non-integer values, and
non-finite values are rejected rather than assigned to a class.

## Read-only inspection

```bash
cd /home/wenyu/EXO200
source /home/wenyu/summer/.venv/bin/activate
python scripts/inspect_data.py
```

To repeat the slower metadata/cluster scan over every event:

```bash
python scripts/inspect_data.py --full-event-scan
```

## Formal workflow

Each MJD-style entry performs formal training, per-epoch validation, restores
`best.pt`, and then evaluates the test split once. These are the formal
commands; none is run by the inspection workflow:

```bash
cd /home/wenyu/EXO200
source /home/wenyu/summer/.venv/bin/activate
python architectures/cnn_001_waveform/train_classification.py --device auto
python architectures/cnn_004_multiview_late_fusion/train_classification.py --device auto
python architectures/gnn_001_static_gine/train_classification.py --device auto
python architectures/seq_001_bigru/train_classification.py --device auto
python architectures/ssm_001_pointmamba/train_classification.py --device auto
```

Outputs are written below `outputs/classification/<architecture>/` as:

- `run_config.json`
- `best.pt`
- `history.json`
- `metrics.json`
- `predictions.npz`

There is no regression, reconstruction, inference-only, generation, or
smoke-test workflow in this project.
