# Prior MJD Hackathon Connection

## Why This Matters

Jaden's prior hackathon work is directly relevant to the Rare AI Lab assignment because it already used the Majorana Demonstrator dataset (`MJD_Train_0.hdf5`) as external transfer data for detector waveform classification.

This is not just generic machine learning experience. It is experience with:

- high-purity germanium detector waveforms
- pulse-shape discrimination
- rare-event search data
- HDF5 scientific data files
- physics-inspired waveform features
- transfer learning from related detector labels
- conservative model ensembling for leaderboard/generalization performance

## What The Winning Approach Did

The winning model treated the task as a waveform pulse-shape problem rather than a raw black-box classification problem.

Feature families included:

- rise timing features such as `t10`, `t50`, and `t90`
- current/derivative peak strength and timing
- AvsE-style ratios, especially maximum current divided by energy
- current peak width and asymmetry
- tail slope and delayed-charge-recovery behavior
- shoulder and late-charge-region features
- baseline noise and drift
- pulse-spectral or ringing texture features

The first strong solution was a robust blend of helper models. Each helper was trained on a slightly different waveform feature family, then combined conservatively.

The biggest improvement came from using the Zenodo Majorana release as external data. The model did not assume the external labels were identical to the hackathon labels. Instead, it trained helper models on the Majorana `low_avse` label because low AvsE is physically related to pulse-shape discrimination.

The final blend was approximately:

- 95% previous robust / Zenodo anchor
- 3% stacked public-label helper
- 2% physics-lite low-AvsE helper

The key lesson was that a small, physically meaningful external-data signal helped more than an oversized model.

## How This Helps In Rare AI Lab

This prior work gives Jaden a strong starting point for the MJD dataset in the current lab assignment.

Useful talking points:

- "I realized I have already worked with the Majorana Demonstrator Zenodo release in my previous winning hackathon solution."
- "My approach used pulse-shape features and AvsE-inspired detector physics rather than only raw waveform models."
- "The useful transfer label was low AvsE, which is connected to current-over-energy pulse-shape discrimination."
- "One lesson was that external detector data should be blended conservatively because related physics labels are helpful but not identical."

## Questions To Ask Aobo

- Is the current MJD tutorial dataset the same Zenodo release as `MJD_Train_0.hdf5`?
- Are AvsE, low AvsE, DCR, or LQ-style pulse-shape labels relevant to the lab's current MJD direction?
- Would a clean reproduction of the hackathon feature pipeline be useful as a baseline for this summer?
- Is the ICLR direction more likely to use MJD waveform classification, cross-dataset transfer, or a broader rare-event benchmark?

## Possible Summer Project Angles

### Cross-Dataset Transfer

Study when waveform features learned from MJD transfer to other rare-event detector datasets such as CUORE or KLZ.

### Physics-Guided Feature Baselines

Build a clean baseline suite using interpretable detector waveform features before adding deep models.

### External-Label Robustness

Analyze how related but non-identical physics labels, such as low AvsE, can improve a target classifier when blended conservatively.

### Benchmark Reproduction Notebook

Convert the previous winning pipeline into a clean notebook that documents data loading, feature extraction, helper models, blending, and final export.

