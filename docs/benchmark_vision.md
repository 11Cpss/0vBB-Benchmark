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

These are benchmark-wide metric families, not a requirement that every
dataset run both tasks. The current NEXT campaign runs classification only.
Its direct energy target is the sum of the per-hit energies already present in
the input, so a NEXT regression result would primarily test energy transport
unless a separately justified topology-only target is adopted later.

For NEXT, headline reporting uses energy-matched AUC and the EnergyBench
energy-independence score. Inclusive AUC, common-support AUC, shortcut gap,
and the worst-energy independence score remain supporting diagnostics.

## Models

We compare specialized models with a unified tokenized transformer. The common
design uses three general tokenization strategies crossed with two positional
encodings:

1. fixed-size patch tokens;
2. entity/object tokens;
3. summary-feature tokens.

Their concrete implementations are adapted to each dataset: waveform patches,
detector-hit entities, and time-window summaries for MJD; spatial voxel
patches, hit entities, and voxel summaries for NEXT; and grouped feature
representations for SuperNEMO. The interface is shared, not the raw tokens.

The six combinations are model variants, not separate benchmarks. Every
variant uses the same data splits and evaluation protocol.

For the completed NEXT milestone, sampled-hit/entity, voxel/patch, and
adaptive summary-feature tokenization were each crossed with coordinate-MLP
and Fourier-XYZ positional encoding, producing six full-data runs. Specialized
models and the transformer both use copies of the same collaborator-provided Simple
EnergyBench workflow. Shared data, training, checkpoint, metric, evaluation,
and plotting code is held fixed; only architecture-specific representation
and model code differs.

The collaborator-facing implementation streams raw HDF5 events and tokenizes
them during loading. Precomputed token caches are not required by the standard
workflow, which keeps setup and storage requirements manageable across
multiple experiment owners.

## Completed NEXT result

On the shared 116,549-event test split, voxel tokens with coordinate-MLP
positions achieved the highest energy-matched AUC, 0.987718. Adaptive summary
features with coordinate-MLP positions reached 0.987424, only 0.000294 lower.
Summary features had the highest descriptive mean AUC across the two tested
positional encodings, while sampled-hit models had the strongest energy-
independence scores. Coordinate MLP exceeded Fourier XYZ in matched AUC for
all three tokenizations.

These are single-seed results without confidence intervals. The voxel-summary
difference must be described as near-parity, not a significant improvement.
Summary tokens also use four engineered content features versus two for the
other representations, so this is a comparison of complete representation
strategies rather than a grouping-only ablation. The public reference table
and interpretation boundaries are recorded in `next_detector/README.md`.

## Scientific goal

The benchmark is not only a leaderboard. It tests whether models retain useful performance when likely shortcuts—such as energy, amplitude, occupancy, hit count, geometry, or reconstruction artifacts—are controlled. A model is more convincing when it performs well on the standard physics-aware metrics and remains stable across the relevant energy range.
