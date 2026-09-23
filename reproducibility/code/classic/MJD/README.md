# MJD waveform workflow

This is the MJD-only workspace. It is intentionally separate from
`/home/wenyu/summer`, which remains the NEXT workspace.

The project follows the same high-level workflow as
`/home/wenyu/summer/evalutaions_workflow`:

1. inspect and validate the HDF5 data;
2. prepare reproducible train/validation/test loaders;
3. train one model for one task;
4. restore the best validation checkpoint;
5. evaluate once on the official test split;
6. save metrics, predictions, history, and checkpoints below `outputs/`.

## Task contract

MJD is a raw-waveform dataset, so it must not reuse the NEXT projection data
loader.

- **Classification** predicts one `clean` versus `non-clean` label from a
  waveform with BCE-with-logits loss. An event is clean only when all four
  PSD reference labels (`low_avse`, `high_avse`, `dcr`, and `lq`) equal one.
- An event is **clean** only when all four reference PSD labels equal one.
- **Regression** predicts `energy_label` in keV from clean events only.

Within an architecture folder, classification and regression use the same
backbone definition and hyperparameters. They are independent models with
different output heads, losses, checkpoints, histories, and result folders;
this is not a multitask network.

## Data

The default data directory is:

```text
/home/klz/Data/zeronu_benchmark/MJD
```

Expected files are:

```text
MJD_Train_*.hdf5       # labels available; training + validation source
MJD_Test_*.hdf5        # labels available; held-out final evaluation
MJD_NPML_*.hdf5        # challenge inference only; labels are masked
```

Do not copy these multi-GB files into this repository. The current download can
be checked with:

```bash
cd /home/wenyu/MJD
python scripts/inspect_data.py
```

An HDF5 file that is still downloading is reported as `INCOMPLETE`; it is never
silently used for training.

## Setup

The existing Python environment can be reused:

```bash
cd /home/wenyu/MJD
source /home/wenyu/summer/.venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the paired baseline

First inspect the data, then run the two tasks independently:

```bash
python scripts/inspect_data.py

python architectures/cnn_001_waveform/train_classification.py --smoke
python architectures/cnn_001_waveform/train_regression.py --smoke
```

Remove `--smoke` for full training. Use `--data-root` to override the shared
dataset location and `--device cuda` or `--device cpu` to select a device.

Outputs are kept separate:

```text
outputs/
├── classification/
│   └── cnn_001_waveform/
└── regression/
    └── cnn_001_waveform/
```

## Add another architecture

Create a new folder below `architectures/` with both entry points:

```text
architectures/<architecture_name>/
├── model.py
├── train_classification.py
└── train_regression.py
```

Keep one shared backbone implementation in `model.py`. The classifier must
return `[batch]` clean logits; the regressor must return `[batch]` energy
predictions in keV.

## Important current status

At workspace creation time, the MJD files were still being downloaded and the
available directory did not yet contain the official `MJD_Train_*.hdf5`
files. Training intentionally stops with a clear error until complete training
files are present. The included synthetic tests do not depend on the download.

Dataset reference: I. J. Arnquist et al., *Majorana Demonstrator Data Release
for AI/ML Applications*, arXiv:2308.10856.
