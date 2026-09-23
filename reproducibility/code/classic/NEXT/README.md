# 0νββ Benchmark: EnergyBench and NEXT Model Zoo

This repository is a reproducible benchmark and model-development workspace for
event-level studies with NEXT detector data. It combines a model-agnostic
evaluation package (`energybench`), reference CNNs, a broad non-Transformer
model zoo, frozen task manifests, experiment records, and research reports.

The benchmark focuses on three questions:

- **Classification:** how well a model separates neutrinoless double-beta decay
  signal from background after matching their energy spectra.
- **Energy regression:** how accurately a model predicts per-event energy and
  reproduces the aggregate energy spectrum.
- **Energy dependence:** whether a classifier score changes with energy within
  each class, which can reveal spectrum sculpting and shortcut learning.

## Highlights

- Energy-matched and inclusive ROC/AUC with explicit common-support checks.
- ERS-v1 energy-regression scoring, diagnostic plots, and auditable JSON/CSV
  outputs.
- Backward-compatible NEXT checkpoint adapters and canonical event-level
  prediction tables.
- Reference CNN classification and energy-regression pipelines.
- A non-Transformer model zoo spanning 3D/multiview CNNs, point-set models,
  graph neural networks, sequence models, sparse models, topology features,
  and hybrid architectures.
- Frozen manifests, campaign orchestration, bilingual documentation, formal
  evaluation summaries, and a complete group-meeting research package.

## Repository structure

```text
.
├── 01_code/
│   ├── architectures/       # Per-model training entry points and configs
│   └── src/                 # Shared project path helpers
├── 02_models/                # Runtime checkpoints (ignored by Git)
├── 03_training_runs/
│   └── campaigns/           # Curated campaign manifests and run records
├── 04_evaluations/           # Large generated evaluation outputs (ignored)
├── 05_reports/               # Research reports and reproducible analyses
├── 06_group_meeting_presentation_20260804/
│   ├── analysis/            # Compact tables, audits, and report metadata
│   ├── assets/              # Figures used by the presentation/report
│   └── source/              # Rebuild and validation scripts
├── docs/                     # Evaluation, usage, training, and results docs
├── examples/                 # Adapter and prediction-table examples
├── manifests/                # Frozen datasets, tasks, and scoring contracts
├── requirements/             # CUDA/CNN and non-Transformer dependencies
├── src/
│   ├── energybench/         # Model-agnostic evaluation package and CLI
│   ├── next_cnn/            # NEXT CNN models, data pipeline, and adapter
│   └── next_alt/            # Alternative models and shared trainer
└── tests/                    # Regression and architecture tests
```

Large generated artifacts are intentionally excluded from Git. See
[`.gitignore`](.gitignore) for the exact boundary.

## Model zoo

The repository contains the following maintained architecture families:

| Family | Implementations | Main documentation |
|---|---|---|
| Reference CNNs | CNN-001 two-convolution baseline, CNN-002 global-energy skip, CNN-003 residual spatial model | [Architecture directories](01_code/architectures/) |
| Alternative CNNs | Multiview late fusion, multiscale projection, dense 3D ResNet | [Alternative architectures](docs/ALTERNATIVE_ARCHITECTURES.md) |
| Point-set models | DeepSets, PointNet++, PointMLP, rigid KPConv | [Alternative architectures](docs/ALTERNATIVE_ARCHITECTURES.md) |
| Graph models | Static GINE, ParticleNet EdgeConv, EGNN, GravNet, DimeNet-lite | [Alternative architectures](docs/ALTERNATIVE_ARCHITECTURES.md) |
| Sequence and state-space models | BiGRU, dilated TCN, PointMamba | [Alternative architectures](docs/ALTERNATIVE_ARCHITECTURES.md) |
| Other inductive biases | Projection MLP-Mixer, sparse submanifold ResNet, persistence/PersLay, CNN+GNN hybrid, topology XGBoost | [Alternative architectures](docs/ALTERNATIVE_ARCHITECTURES.md) |

Every model directory under [`01_code/architectures`](01_code/architectures/)
contains its configuration, training entry point, and detailed model-specific
README where available. Shared implementations live in
[`src/next_alt`](src/next_alt/) and [`src/next_cnn`](src/next_cnn/).

## Documentation index

### Benchmark and usage

- [EnergyBench evaluation standard (English)](docs/EVALUATION_STANDARD_EN.md)
- [EnergyBench evaluation standard (Chinese)](docs/EVALUATION_STANDARD.md)
- [Usage and maintenance guide (English)](docs/USAGE_GUIDE_EN.md)
- [Usage and maintenance guide (Chinese)](docs/USAGE_GUIDE.md)
- [Alternative architecture design](docs/ALTERNATIVE_ARCHITECTURES.md)
- [Training with tmux](docs/TMUX_TRAINING.md)

### Training and results

- [Alternative architecture evaluation results](docs/ALTERNATIVE_EVALUATION_RESULTS.md)
- [Non-Transformer v2 training campaign](docs/NONTRANSFORMER_V2_TRAINING.md)
- [Non-Transformer v2 evaluation results](docs/NONTRANSFORMER_V2_EVALUATION_RESULTS.md)
- [Energy-regression dataset assessment](05_reports/NEXT_energy_regression_dataset_assessment.md)
- [CNN-001 regression diagnosis notebook](05_reports/cnn_001_regression_diagnosis_20260801/analysis.ipynb)

### Presentation package

- [Package overview](06_group_meeting_presentation_20260804/README.md)
- [English PowerPoint](06_group_meeting_presentation_20260804/NEXT_group_meeting_progress_20260804.pptx)
- [Rendered PDF](06_group_meeting_presentation_20260804/NEXT_group_meeting_progress_20260804.pdf)
- [English speaker notes](06_group_meeting_presentation_20260804/speaker_notes_en.md)
- [Chinese technical report](06_group_meeting_presentation_20260804/NEXT_technical_report_zh.html)
- [Analysis validation report](06_group_meeting_presentation_20260804/analysis/validation_report.md)
- [Figure guide](06_group_meeting_presentation_20260804/analysis/figure_guide.md)

## Installation

Python 3.11 is required.

```bash
git clone https://github.com/wenhaoquestion/0vbb-benchmark.git
cd 0vbb-benchmark

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For NEXT CNN training on the original CUDA environment, install the pinned
packages in [`requirements/next-cnn-cu128.txt`](requirements/next-cnn-cu128.txt).
The broader non-Transformer model zoo uses
[`requirements/nontransformer.txt`](requirements/nontransformer.txt).

## Quick start

Inspect a canonical event-level prediction table:

```bash
energybench inspect predictions_test.npz
```

Evaluate exported predictions with a frozen manifest:

```bash
energybench evaluate predictions_test.npz \
  --manifest manifests/next_0nubb_vs_bi214.yaml \
  --model-id my-model \
  --output-dir 04_evaluations/my-model \
  --strict
```

Evaluate a compatible NEXT checkpoint:

```bash
energybench next --dry-run
energybench next 02_models/checkpoints/MY_CHECKPOINT.pt
```

Discover all CLI options:

```bash
energybench --help
energybench evaluate --help
energybench next --help
```

## Training

Run one architecture with the standalone Simple EnergyBench protocol:

```bash
python 01_code/architectures/gnn_003_egnn/train_classification.py \
  --output-dir 03_training_runs/energybench_manual/gnn_003_egnn \
  --manifest 03_training_runs/energybench_manual/event_split.json
```

Run all 23 classifiers, plus the three maintained CNN regressors, with
resource-aware parallel scheduling:

```bash
python 01_code/architectures/run_energybench_campaign.py \
  --run-id MY_RUN_ID \
  --parallel-capacity 3 \
  --include-regression
```

The new runner uses the event-count split, standard 50-epoch training config,
canonical 5 keV evaluation, native point/graph/3-D representations, immutable
run directories, and a shared split manifest. See
[`01_code/architectures/ENERGYBENCH_WORKFLOW.md`](01_code/architectures/ENERGYBENCH_WORKFLOW.md)
for the exact contract and resume command. Historical queue documentation is
retained for reproducing the older file-level-split campaigns.

## Testing

```bash
python -m pytest -q tests
python -m compileall -q src 01_code/architectures
```

## Data and reproducibility boundary

This repository does **not** redistribute the original NEXT event archives.
Dataset selection, checksums, label semantics, split rules, and evaluation
settings are frozen in [`manifests`](manifests/). Model checkpoints and full
event-level evaluation payloads are also excluded because they are generated,
large, and environment-specific.

The tracked campaign records, compact result tables, reports, and presentation
artifacts document the experiments without committing hundreds of megabytes of
duplicated intermediate data. Reproduction of the full training results still
requires access to the corresponding NEXT data archives and a compatible CUDA
environment.

## Current limitations

- Published results are primarily single-seed point estimates.
- Some historical v2 figures do not have matching training checkpoints in this
  repository.
- Energy regression from deposited voxel energy can become a deterministic
  data-flow check when the target is the sum of the same input amplitudes.
- Dataset access and hardware-specific CUDA setup remain external prerequisites.

## License

This project is released under the [MIT License](LICENSE).
