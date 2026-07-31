# Dataset Overview

Source: https://indico.physics.ucsd.edu/event/2/page/3-datasets

Assignment from Aobo: look at each dataset and get familiar with the files before the next group meeting.

Current working scope for the first benchmark pass:

- Active now: CUORE, NEXT, SuperNEMO
- Prior work / known connection: MJD
- Paused for now: KLZ because access is not available yet
- Dropped for now: nEXO and Project 8

## Working Priority

1. CUORE
2. NEXT
3. SuperNEMO
4. MJD as prior work / later integration

## CUORE

- Experiment: CUORE, Cryogenic Underground Observatory for Rare Events
- Physics area: rare event search / neutrinoless double beta decay context
- Events: 10,000 triggered waveforms
- Input: 1D waveform with 10,000 samples plus `eventId`
- Sampling: 1 kHz, 10 seconds per event, including 3 seconds of pretrigger baseline
- Labels: binary classification
  - `0`: clean single-pulse event
  - `1`: pile-up event with two or more pulses
- Split: 90% labeled train, 10% unlabeled test
- Format: HDF5
- First exploration:
  - Inspect HDF5 keys, shapes, dtypes, and label counts
  - Plot several clean and pile-up waveforms
  - Check baseline statistics before normalization
  - Try simple baselines: logistic regression on summary features, 1D CNN on waveform

## KLZ

- Experiment: KamLAND-Zen
- Access: approved for the summer school, not public release
- Status: paused because access is not available yet
- First exploration once available:
  - Inspect file structure and metadata
  - Identify event representation and labels
  - Ask what public/private handling rules apply

## MJD

- Experiment: Majorana Demonstrator
- Physics area: high-purity germanium detector waveforms for neutrinoless double beta decay
- Events: over 3 million time-series data points
- Input: 1D NumPy vector with 4,000 samples
- Labels: 4 binary classification labels plus 1 energy regression label
- Format: HDF5
- First exploration:
  - Inspect HDF5 keys, shapes, dtypes, and label definitions
  - Plot representative waveforms by label
  - Check energy distribution and class imbalance
  - Try simple baselines: summary-feature classifiers, MLP, 1D CNN, multitask model

## NEXT

- Experiment: NEXT high-pressure gaseous xenon TPC
- Physics area: neutrinoless double beta decay signal/background discrimination
- Events: approximately 650k signal and 500k background simulated events
- Input: Pandas-style dataframe with `event_id`, `x`, `y`, `z`, `energy`
- Labels: binary classification, `signal` or `background`
- Format: HDF5
- First exploration:
  - Start with the recommended 10% subset
  - Inspect event sizes and voxel distributions
  - Plot 3D/2D projections of energy depositions
  - Try simple baselines: event-level engineered features, point-cloud model, voxelized CNN

## SuperNEMO

- Experiment: SuperNEMO Demonstrator
- Physics area: simulated double-beta and background processes in source foil
- Input columns: `ev_no`, `E1`, `E2`, `tX`, `tY`, `tZ`, `tR`, `dY`, `dZ`, `theta`, `phiR`, `phiS`
- Labels: multiclass classification
  - `0nubb`
  - `2nubb`
  - `Bi214`
  - `Tl208`
- Format: HDF5
- First exploration:
  - Inspect array/dataframe structure
  - Check class balance
  - Plot energy and angular distributions by class
  - Try simple baselines: random forest, gradient boosting, MLP

## Skipped For Now

## nEXO

Dropped for now.

## Project 8

Dropped for now.

## Group Meeting Questions

- Which dataset is most likely connected to the summer ICLR paper?
- Are we expected to reproduce existing baselines or look for a new modeling angle?
- Should exploration prioritize physics interpretability, benchmark performance, or data loading/cleaning?
- What compute/storage should be used for large downloads?
- Are there private data handling rules for KLZ or other datasets?
