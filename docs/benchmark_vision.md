# 0vBB-Benchmark Vision

## Main question

Can one unified model handle very different rare-event detector datasets as well as specialized models designed for each detector?

## Scope

The benchmark covers five datasets:

- CUORE: 1D detector waveforms;
- NEXT: sparse 3D energy-deposition hits;
- SuperNEMO: reconstructed event-level features;
- MJD: germanium-detector waveforms and pulse-shape information;
- EXO-200: raw waveforms and reconstructed event quantities.

## Evaluation

We use one benchmark evaluation suite with two standard physics-aware evaluations:

1. Energy-matched AUC for classification. Signal and background are matched or reweighted in energy so the model cannot rely mainly on different energy spectra.
2. Binned energy regression. Regression error is computed separately across true-energy bins so dense parts of the spectrum do not dominate.

Regular ROC-AUC and global RMSE/MAE can be reported as secondary metrics.

## Models

We compare specialized models with a unified tokenized transformer. The unified model will have approximately eight selected combinations of tokenization and positional encoding. Examples include waveform patches, hit tokens, feature tokens, learned position embeddings, sinusoidal encodings, and physics-aware coordinate or time encodings.

The eight combinations are model variants, not separate benchmarks. Every variant uses the same data splits and evaluation protocol.

## Scientific goal

The benchmark is not only a leaderboard. It tests whether models retain useful performance when likely shortcuts—such as energy, amplitude, occupancy, hit count, geometry, or reconstruction artifacts—are controlled. A model is more convincing when it performs well on the standard physics-aware metrics and remains stable across the relevant energy range.
