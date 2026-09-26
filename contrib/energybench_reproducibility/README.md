# EnergyBench: paper reproduction

Code and inputs for **Learning Physics, Not Energy: A Multimodal Benchmark for Neutrino Detectors**.

Start with the [English reproduction guide](reproducibility/README.md).

| Task | Entry point |
| --- | --- |
| Understand and run EnergyBench | [Benchmark guide](reproducibility/benchmark/README.md) |
| Read the I and AUC implementation | [benchmark/metrics.py](reproducibility/benchmark/metrics.py) |
| Train or run classic models | [Classic models](reproducibility/code/classic/README.md) |
| Train or run Transformers | [Transformers](reproducibility/code/transformers/README.md) |
| Rebuild paper tables and figures | [Reproduction commands](reproducibility/README.md#tables-and-figures) |
| Inspect final result inputs | [Results](reproducibility/results/README.md) |

The release contains the code needed by the reported computations, selected
run configurations, final result data, and active manuscript assets. Full raw
detector datasets, trained checkpoint tensors and the optional bulk prediction
companion are external. The guide states the available reproduction coverage
and the remaining missing sources.

Each reported result keeps its specified evaluation definition. In particular,
the 0–3000 keV, 600-bin profile is the corrected MJD/EXO-200 protocol; other
reported subsets have explicit configurations. The release does not silently
assign a different protocol to an existing paper value.

Original project software is covered by the [MIT License](LICENSE). Retained
third-party source and font notices apply to their respective material; this
license does not grant rights to detector data or checkpoint tensors.
