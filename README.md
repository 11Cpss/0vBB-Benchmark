# DSC Research Workspace

Research workspace for work with the Rare AI Lab at UC San Diego.

## Lab Context

Rare AI Lab studies artificial intelligence for rare physics event searches. Current themes include:

- Neutrinoless double beta decay
- Dark matter detection
- Gravitational-wave and detector time-series analysis
- Surrogate models for expensive detector simulation
- AI agents for scientific workflows
- Fast machine learning for detector systems

## Current Assignment

Aobo asked Jaden to get familiar with the 0vbb AI Summer School datasets before the next group meeting.

Dataset page: https://indico.physics.ucsd.edu/event/2/page/3-datasets

Focus datasets:

- CUORE
- KLZ
- MJD
- NEXT
- SuperNEMO

## CUORE vs NEXT vs SuperNEMO

All three experiments are looking for neutrinoless double beta decay, usually
written as 0vbb or 0nu beta beta. The difference is mostly in how each detector
"sees" an event.

### CUORE

CUORE is a cryogenic bolometer experiment. Think of it as a detector that waits
for a tiny heat pulse in a cold crystal. The summer-school dataset gives each
event as a 10-second 1D waveform with 10,000 samples.

The ML task is waveform classification:

- input: one raw detector waveform
- label: clean single pulse vs pile-up pulse
- main question: does this waveform contain one real pulse, or multiple pulses
  overlapping in time?

This is closest to audio or sensor time-series work. The model needs to learn
pulse shape, timing, baseline behavior, and whether a second pulse is hidden
inside the waveform.

### NEXT

NEXT is a high-pressure xenon gas time projection chamber. Instead of only
recording a pulse over time, it reconstructs where energy was deposited inside
the detector. The dataset represents events as voxelized 3D points with energy:
x, y, z, and energy deposition.

The ML task is event topology classification:

- input: 3D spatial energy deposits
- label: signal 0vbb vs background Bi-214
- main question: does the event look like the two-electron topology expected
  from 0vbb, or like a one-electron/gamma background?

This is closer to 3D point-cloud or sparse-image classification. The important
information is the shape of the track: signal should have two electron-like
ends, while background tends to look like a single-electron event.

### SuperNEMO

SuperNEMO uses a tracking detector plus calorimeters around a thin source foil.
It is designed to reconstruct electron trajectories and energies more directly.
The dataset is not raw waveforms or voxel grids; it is already summarized into
event-level reconstructed features such as two electron energies, track
positions, angular variables, and geometry variables.

The ML task is multiclass event classification:

- input: reconstructed event features
- label: 0vbb, 2vbb, Bi-214, or Tl-208
- main question: which physical process most likely produced this reconstructed
  event?

This is closest to tabular ML. The model is not learning directly from detector
signals; it is learning from reconstructed physics quantities.

### Short Comparison

| Experiment | Core datatype | Detector idea | Dataset shape | ML problem | What the model learns |
| --- | --- | --- | --- | --- | --- |
| CUORE | Dense 1D time series | Cold crystals measuring heat pulses | 1D waveform over time | Binary classification | Pulse shape and pile-up |
| NEXT | Sparse 3D point cloud / voxel event | Xenon gas TPC reconstructing 3D energy deposits | 3D points/voxels with energy | Binary classification | Event topology: signal-like vs background-like tracks |
| SuperNEMO | Tabular feature vector | Tracker + calorimeter around a source foil | Reconstructed event-level features | Multiclass classification | Physics-process identity from energies, angles, and tracks |

My mental shortcut:

- CUORE asks: "What does the pulse look like over time?"
- NEXT asks: "What shape did the energy deposit make in 3D?"
- SuperNEMO asks: "Given reconstructed energies and tracks, what process was it?"

Core datatype shortcut:

- CUORE: array of samples, like `waveform[t]`
- NEXT: set of spatial hits, like rows of `(x, y, z, energy)` for one event
- SuperNEMO: one fixed-length row of engineered/reconstructed features per event

Skip for now:

- Project 8
- nEXO

Summer direction: prepare a paper for ICLR, with an approximate deadline around September 20.

## Benchmark Paper Direction

The current research direction is to build a benchmark paper for rare-event physics datasets. The main question is whether heterogeneous detector datasets should use dataset-specific architectures or whether they can be represented through a unified tokenized / transformer-style approach.

Active benchmark datasets for the first pass:

- CUORE
- NEXT
- SuperNEMO
- MJD, connected to prior hackathon work

Paused or dropped for now:

- KLZ, because access is not currently available
- nEXO
- Project 8

## KLZNet Collaboration Context

Jaden is expected to work with the author of a KLZNet dependence study while building the benchmark paper. The KLZNet note is stored at:

```text
papers/klznet/klznet_dependence_paper.pdf
```

Extracted text for local searching is stored at:

```text
papers/klznet/klznet_dependence_paper_extracted.txt
```

The KLZNet paper studies how energy, timing, charge, position, and topology dependences enter sparse KamLAND-Zen detector representations and propagate into classifier scores. Its main lesson for the benchmark paper is that high inclusive ROC/AUC is not enough. Rare-event ML models can learn nuisance shortcuts through detector response, such as energy-dependent light yield, hit multiplicity, total charge, timing profiles, radius, and occupancy. Benchmark reporting should therefore include physics-aware diagnostics such as energy-matched AUC, radius-matched AUC, score-vs-energy dependence, threshold acceptance curves, and class-conditional stability checks.

## What We Learned From The KLZNet Paper

The benchmark should have one shared primary task, plus multiple controlled evaluation tests. The standard headline evaluations are energy-matched AUC for classification and binned energy regression where regression is available. Regular ROC-AUC and overall regression error can be reported as secondary metrics. The controlled tests hold likely shortcut variables approximately constant, remove them, perturb them, or test under shifted conditions. They show whether a model's performance survives when an easy detector-specific proxy is no longer useful.

Examples include:

- energy-matched evaluation to test dependence on energy spectra;
- radius- or geometry-matched evaluation to test position dependence;
- hit-count, occupancy, or total-charge matching for sparse detector data;
- amplitude, baseline, or noise controls for waveform data;
- feature-ablation tests that remove selected inputs;
- shifted-data tests using different simulation or detector conditions.

These are not necessarily separate research benchmarks. They form one benchmark evaluation suite: a common primary benchmark with dataset-specific controlled tests. Energy-matched AUC is the standard classification metric because it tests performance after energy-based shortcuts are reduced. For regression, true energy is divided into bins and error is computed within each bin so dense energy regions do not dominate. High inclusive AUC or low global regression error alone is not sufficient. If performance remains strong under energy matching or across energy bins, that provides stronger evidence that the model learned meaningful structure. A large drop indicates reliance on a shortcut or failure in part of the energy range.

The tests should be chosen according to each detector's likely failure modes:

- CUORE: amplitude, baseline, noise, and pulse-timing controls;
- NEXT: energy, hit count, total charge, occupancy, and spatial-extent controls;
- SuperNEMO: energy, angular, geometry, and reconstructed-feature controls;
- MJD: energy dependence and pulse-shape/AvsE-related controls.

Dependence on a variable is not automatically a shortcut or a mistake. Energy and geometry can carry real physical information. The goal is to determine whether the model still discriminates after easy class-correlated proxies are controlled, and whether its remaining dependence is stable, interpretable, and acceptable for the physics analysis. We should report both inclusive and controlled results rather than selecting models by inclusive AUC alone.

This gives the paper a stronger contribution than a model leaderboard: a reproducible benchmark and physics-aware reporting framework for comparing specialized and unified models on heterogeneous rare-event detector data.

## Repo Layout

```text
data/
  raw/          # downloaded datasets; not committed
  processed/    # derived local files; not committed by default
notebooks/      # exploratory notebooks
notes/
  datasets/     # dataset read-through notes
  meetings/     # meeting notes and questions
  papers/       # paper summaries and idea notes
outputs/
  figures/      # generated plots
  reports/      # short writeups
scripts/        # utility scripts
src/            # reusable analysis code
```

## First-Week Goal

For each focus dataset, answer:

1. What is the physics task?
2. What are the inputs, labels, and file format?
3. What should the first exploratory notebook load and plot?
4. What are the likely ML baselines?
5. What questions should be brought to group meeting?
