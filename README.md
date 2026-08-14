# NEXT Transformer EnergyBench

This repository is the standardized, collaborator-facing workflow for
benchmarking Transformer representations on the NEXT detector binary
classification task:

- **signal:** `0nubb`;
- **background:** `Bi214`;
- **primary benchmark:** held-out energy-matched ROC AUC;
- **secondary diagnostics:** inclusive AUC, common-support AUC, shortcut gap,
  energy-independence score, and worst-group energy-independence score.

The workflow streams complete events from raw HDF5 files. It does not require
precomputed token caches. Every collaborator uses the same EnergyBench split,
training loop, checkpoint selection, evaluation code, and plotting code.

## What collaborators may change

Transformer experiments may change:

- tokenization;
- positional encoding;
- Transformer architecture.

For directly comparable benchmark runs, do **not** change:

- the provided event split;
- the `TrainingConfig` numerical defaults other than `num_workers`;
- the `EvaluationConfig`;
- `energybench/training.py`;
- `energybench/evaluation.py`;
- `energybench/metrics.py`;
- `energybench/plotting.py`.

## Repository layout

```text
0vBB-Benchmark/
├── README.md
├── requirements.txt
├── evalutaions_workflow/
│   ├── simple_energybench/       # stable public import
│   ├── energybench/              # shared implementation
│   └── tests/
└── next_detector/
    ├── manifests/
    │   └── event_split.json      # shared 80/10/10 reference split
    ├── next_transformer/
    │   ├── tokenization.py
    │   ├── positional_encoding.py
    │   └── model.py
    ├── notebooks/
    │   ├── next_energybench_train.ipynb
    │   └── next_energybench_results.ipynb
    └── tests/
        └── test_transformer_workflow.py
```

The inherited directory name `evalutaions_workflow` is intentionally retained
so this copy remains directly comparable with the shared EnergyBench version.

## Environment setup

Use Python 3.11 in a dedicated environment:

```bash
conda create -n next-transformer python=3.11 pip -y
conda activate next-transformer
python -m pip install --upgrade pip
```

### NVIDIA lab computer

Install the PyTorch wheel compatible with the installed NVIDIA driver before
the remaining dependencies. The lab configuration used during development
worked with the CUDA 12.8 wheel:

```bash
python -m pip install torch==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Confirm GPU access:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

On a CPU-only system, install the appropriate CPU PyTorch build, install the
requirements, and run `python -m pip install -e . --no-deps`. CPU is suitable
for tests, not the full 1.16-million-event benchmark. The editable install
makes `simple_energybench` and `next_transformer` importable from tests and
scripts while keeping the source files in this repository.

## Dataset

Raw data are not included. The loader expects extracted HDF5 files arranged as:

```text
NEXT/
├── 0nubb_part_*/*.h5
└── Bi_part_*/*.h5
```

Each file must contain `/MC/hits/table`. Rows are detector hits; EnergyBench
groups contiguous rows with the same event ID into one physics event.

Point the workflow to the dataset before starting Jupyter:

```bash
export SIMPLE_ENERGYBENCH_DATA=/absolute/path/to/NEXT
```

## Run the standardized benchmark

Start Jupyter from the repository root so relative package discovery works:

```bash
jupyter lab
```

Open:

```text
next_detector/notebooks/next_energybench_train.ipynb
```

Run all cells. The default run executes the complete 2 × 2 matrix:

| ID | Tokenization | Positional encoding |
|---|---|---|
| `transformer_001` | sampled hits | coordinate MLP |
| `transformer_002` | voxels | coordinate MLP |
| `transformer_003` | voxels | Fourier XYZ |
| `transformer_004` | sampled hits | Fourier XYZ |

To assign one or more models to a collaborator, set a comma-separated list
before starting Jupyter:

```bash
export NEXT_RUN_MODEL_IDS=transformer_003_voxel_fourier_xyz
```

Multiple selections are supported:

```bash
export NEXT_RUN_MODEL_IDS=transformer_001_sampled_hits_coordinate_mlp,transformer_004_sampled_hits_fourier_xyz
```

Useful environment variables:

| Variable | Default | Purpose |
|---|---:|---|
| `SIMPLE_ENERGYBENCH_DATA` | `data/NEXT` | Raw NEXT dataset directory |
| `NEXT_OUTPUT_ROOT` | `next_detector/results` | Generated result root |
| `NEXT_NUM_WORKERS` | `8` | DataLoader worker processes |
| `NEXT_RUN_MODEL_IDS` | all four | Models assigned to this run |
| `NEXT_OFFICIAL_RUN` | `1` | Enforce official counts and exact split |
| `NEXT_REQUIRE_CUDA` | official-run value | Fail early when CUDA is unavailable |
| `NEXT_TRANSFORMER_PROJECT_ROOT` | auto-detected | Repository root override |

Changing `num_workers` is an execution choice and does not change the split,
model, optimizer, or metrics. Avoid launching enough concurrent jobs to
saturate shared CPU, storage, or GPU resources.

## Standard data and model contract

The official split contains:

| Partition | Events |
|---|---:|
| Train | 932,391 |
| Validation | 116,549 |
| Test | 116,549 |
| Total | 1,165,489 |

`NEXTTokenBuilder` receives one complete event and returns:

```python
{
    "coords": float32[max_tokens, 3],
    "features": float32[max_tokens, 2],
    "mask": bool[max_tokens],
}
```

EnergyBench retains labels, total energy, event IDs, group IDs, split names,
weights, and representation coverage for standardized evaluation. The model
returns one raw classification logit per event.

## Outputs and restart behavior

Generated files are written beneath:

```text
next_detector/results/final/<model_id>/
├── representation_config.json
├── training/
│   ├── best_model.pt
│   ├── last_model.pt
│   ├── history.json
│   └── training_history.png
└── evaluation/
    ├── metrics.json
    ├── results.csv
    ├── predictions.npz
    ├── energy_matched_roc.png
    └── score_energy_dependence.png
```

`next_detector/results/final/transformer_results.csv` is updated after each
completed evaluation. On restart, completed CSV rows are skipped. A partial
run directory is never overwritten silently; archive or remove only that
specific partial directory after checking it.

## Inspect final results

After training finishes, run:

```text
next_detector/notebooks/next_energybench_results.ipynb
```

It verifies the summary against each `metrics.json`, checks required
artifacts, ranks models by energy-matched test AUC, builds the 2 × 2 comparison
tables, and displays the saved EnergyBench figures. It does not load the raw
dataset or use the GPU.

## Tests

Run both the unchanged EnergyBench tests and the Transformer integration test:

```bash
python -m unittest discover -s evalutaions_workflow/tests -v
python -m unittest discover -s next_detector/tests -v
python -m compileall -q \
  evalutaions_workflow/energybench \
  evalutaions_workflow/simple_energybench \
  next_detector/next_transformer \
  next_detector/tests
```

The integration test creates temporary synthetic HDF5 files, passes them
through the real EnergyBench event loader and Transformer tokenizer, executes
Transformer forward passes, and completes one CPU training epoch. No NEXT data
or GPU is needed for tests.

## Scientific interpretation

- Validation AUC selects the checkpoint; it is not the final result.
- Headline classification performance is held-out energy-matched test AUC.
- Compare energy-independence and worst-group energy-independence alongside
  AUC to detect energy-dependent shortcuts.
- Runtime measurements on a shared machine are descriptive and should not be
  treated as controlled architecture benchmarks.

## Publishing

Dataset files, checkpoints, predictions, generated results, executed
notebooks, and local environments are ignored by Git. Choose and add an
explicit software license before making the repository public.
