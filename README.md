# Rare-Event Transformer Benchmark

This repository provides reproducible Transformer representation benchmarks
for rare-event detector data. Each detector has its own folder, input
representations, notebooks, and tests, while collaborator-provided data splits,
training loops, checkpoint selection, and evaluation remain fixed within that
detector workflow.

Raw detector data, checkpoints, predictions, and generated results are not
distributed.

## Benchmarks

| Detector | Input | Tasks | Transformer matrix |
|---|---|---|---|
| [NEXT](next_detector/README.md) | Variable-length 3D detector-hit events | `0nubb` signal vs `Bi214` background classification | 3 tokenizations × 2 positional encodings |
| [MJD](mjd_detector/README.md) | 10,000-sample 1D waveforms | Clean/non-clean classification and clean-event energy regression | 3 tokenizations × 2 positional encodings × 2 tasks |

## Repository layout

```text
0vBB-Benchmark/
├── evalutaions_workflow/       # Shared NEXT EnergyBench implementation
├── next_detector/              # NEXT Transformer benchmark
│   ├── next_transformer/
│   ├── manifests/
│   ├── notebooks/
│   └── tests/
└── mjd_detector/               # MJD waveform benchmark
    ├── mjdbench/               # Shared MJD data/training/evaluation workflow
    ├── mjd_transformer/
    ├── notebooks/
    └── README.md
```

The inherited directory name `evalutaions_workflow` is intentionally retained
to preserve compatibility with the shared NEXT workflow.

## Installation

Use Python 3.11. Install a PyTorch build compatible with the target CPU or CUDA
driver first, then install the repository:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

For the NVIDIA lab machine used during development, the CUDA 12.8 PyTorch wheel
was compatible with the installed driver:

```bash
python -m pip install torch==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

## Tests

The tests use synthetic data and do not require detector datasets or a GPU:

```bash
python -m unittest discover -s evalutaions_workflow/tests -v
python -m unittest discover -s next_detector/tests -v
python -m unittest discover -s mjd_detector/mjd_transformer/tests -v
```

See each detector README for dataset layout, experiment assignment, metrics,
and scientific interpretation.

## Collaboration boundary

For directly comparable runs, collaborators may change Transformer-specific
tokenization, positional encoding, or architecture. Do not silently change a
detector's shared split, preprocessing, optimizer defaults, checkpoint rule,
or evaluation implementation. Record any deliberate protocol change as a new
benchmark version.

## License

Original software in this repository is available under the
[MIT License](LICENSE). The license does not grant rights to detector data,
generated checkpoints, predictions, or third-party material. Confirm release
permission and attribution for collaborator-contributed code before public use.
