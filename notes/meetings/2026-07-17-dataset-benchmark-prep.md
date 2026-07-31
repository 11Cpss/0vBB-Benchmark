# Dataset Benchmark Prep

## Current Active Scope

For tomorrow's meeting, focus on three active datasets:

- CUORE
- NEXT
- SuperNEMO

MJD is important prior work because Jaden already used `MJD_Train_0.hdf5` in the previous winning hackathon solution. KLZ is paused because access is unavailable. nEXO and Project 8 are dropped for now.

## Notebook Files

- `notebooks/01_cuore_quick_exploration.ipynb`
- `notebooks/02_next_quick_exploration.ipynb`
- `notebooks/03_supernemo_quick_exploration.ipynb`

## What Each Dataset Looks Like

## CUORE

- Data type: 1D detector waveform
- Local file: `data/raw/cuore/20721645/cuoreTraining.h5`
- Shape: 9,000 labeled waveforms, each with 10,000 samples
- Labels: binary, clean single pulse vs pile-up
- Natural models: waveform features, 1D CNN, waveform transformer
- Meeting sentence: CUORE is a balanced waveform classification problem where the model must distinguish clean pulses from pile-up events.

## NEXT

- Data type: sparse 3D detector hits with energy
- Local file: `data/raw/Next/0nubb_part_1.tar`
- Current download: 999 HDF5 files, apparently `0nubb` signal only
- Natural models: engineered event features, voxelized 3D CNN, point cloud model, graph model, tokenized hit transformer
- Meeting sentence: NEXT is spatial detector-event data, so it tests whether the benchmark can handle point-cloud or voxel-like representations rather than only waveforms.

## SuperNEMO

- Data type: tabular/event-level physics features
- Local file: `data/raw/SuperNemo/data_0nubb_merged.h5`
- Shape: about 49.7M rows in current file
- Current download: appears to be `0nubb` only
- Natural models: logistic regression, random forest / gradient boosting, MLP, feature-token transformer
- Meeting sentence: SuperNEMO is the most tabular dataset, so it is a good contrast case against waveform and spatial detector data.

## Important Caveat

The current NEXT and SuperNEMO files appear to be signal-only `0nubb` samples. They are good for understanding and visualization, but a full classification benchmark needs the matching background / alternate-class files.

## Research Framing

The benchmark question is:

Can rare-event detector datasets with very different structures be handled well by one unified tokenized model, or do dataset-specific architectures remain stronger?

Specialized baselines:

- CUORE: 1D CNN or waveform feature model
- NEXT: point-cloud / voxel / graph model
- SuperNEMO: boosted trees or MLP
- MJD: waveform features / 1D CNN / transformer, informed by prior AvsE work

Unified baseline:

- Convert each dataset into tokens
- Train transformer-style models over token sequences
- Compare with the specialized baselines using common metrics

## Questions To Ask

- Do we need the missing background files for NEXT and SuperNEMO before building baselines?
- What should the benchmark's main physics metric be: ROC-AUC, background rejection at fixed signal efficiency, or both?
- Should MJD be included immediately because Jaden already has prior work with it?
- Does the paper contribution lean more toward dataset benchmark design or unified tokenization/modeling?

