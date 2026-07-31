# 0vBB-Benchmark

This project studies how machine learning models should be evaluated on rare-event detector data.

The main question is:

> Can one unified model handle very different rare-event detector datasets as well as specialized models designed for each detector?

We are comparing a unified tokenized transformer with specialized models such as waveform CNNs, point-cloud models, boosted trees, and physics-feature baselines.

## Datasets

The current scope contains five datasets:

- **CUORE** — 1D detector waveforms. The task is to distinguish clean pulses from pile-up pulses.
- **NEXT** — sparse 3D energy-deposition hits. The task is to distinguish 0νββ-like two-electron topologies from radioactive backgrounds.
- **SuperNEMO** — reconstructed event-level features. The task is to classify different double-beta-decay and background processes.
- **MJD** — germanium-detector waveforms and pulse-shape information, including physics-inspired features such as AvsE-related quantities.
- **EXO-200** — raw detector waveforms and reconstructed event quantities from a public AI/ML data release.

These datasets deliberately cover different input structures: time series, sparse 3D events, tabular physics features, and mixed raw/reconstructed detector data.

## Benchmark design

We use one benchmark evaluation suite rather than separate benchmarks for every model. Each model sees the same event-level splits and is evaluated using the same primary metrics.

For classification, the standard metric is **energy-matched AUC**. Signal and background events are matched or reweighted in energy so that a model cannot obtain most of its performance simply from different energy spectra. Regular ROC-AUC can be reported as a secondary metric.

For energy regression, the standard metric is **binned energy regression**. The true-energy range is divided into bins, and the regression error is calculated within each bin. This prevents the densest part of the spectrum from dominating the result. Overall RMSE or MAE can be reported as secondary metrics.

The benchmark also includes controlled tests for likely shortcuts, such as hit count, occupancy, total charge, detector position, waveform amplitude, baseline noise, and reconstruction artifacts. A model is more convincing if its performance remains strong after these effects are controlled.

## Model comparison

The unified model will be tested using approximately eight selected combinations of tokenization and positional encoding. These may include waveform-patch tokens, detector-hit tokens, feature tokens, learned position embeddings, sinusoidal encodings, and physics-aware time or coordinate encodings.

The eight combinations are model variants, not separate benchmarks. They will be compared against specialized models using the same splits and evaluation protocol.

## Scientific goal

The goal is not only to produce a leaderboard. We want to understand when a general-purpose representation can transfer across detector types, when specialized models are still better, and whether strong performance reflects meaningful physics rather than detector-specific shortcuts.

The project documentation contains the current dataset descriptions, metric definitions, model experiment plans, and the physics-aware evaluation ideas motivated by the KLZNet dependence study.
