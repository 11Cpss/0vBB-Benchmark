# KLZNet Dependence Paper Summary

Paper file: `papers/klznet/klznet_dependence_paper.pdf`

Extracted text: `papers/klznet/klznet_dependence_paper_extracted.txt`

Title: **Energy, Timing, Charge, Position, and Topology Dependences in KLZNet Event Classification**

Author: Wenyu Huang

Date: May 6, 2026

## What The Paper Is About

The paper studies KLZNet, a classifier for KamLAND-Zen-like rare-event data. KLZNet uses sparse detector event representations over time, longitudinal coordinate, and azimuthal coordinate, with optional charge information. The goal is to classify rare-event physics samples using spatial, temporal, charge, and topology information from the detector.

The central issue is **dependence**: even if reconstructed energy is not explicitly given to the neural network, the model can still infer energy from detector response.

Energy leaks into the input through:

- number of nonzero sparse entries
- hit multiplicity
- total charge
- occupied detector bins
- charge concentration
- topology entropy
- high-charge components
- timing profiles
- radius and detector acceptance effects

The paper shows that these raw dependencies can propagate into the classifier score. This matters because a model can look strong by exploiting energy, timing, or radius differences between classes rather than learning robust event topology.

## Main Technical Lesson

The paper argues that **inclusive ROC/AUC is not enough** for rare-event ML.

A high AUC can be inflated if signal and background samples have different energy spectra, radius distributions, timing profiles, or detector-response patterns. In that case, a classifier may be partly learning nuisance shortcuts instead of stable physics topology.

For example, higher-energy events create more scintillation photons. More photons create more PMT hits, more occupied sparse bins, more total charge, and sometimes different timing profiles. A neural network can learn those signals even when energy is not an explicit input.

## Model Evolution In The Paper

The paper compares KLZNet versions v6 through v13.

- v6 showed strong energy and hit-multiplicity sculpting.
- v7 and v8 attempted direct decorrelation but did not reliably fix the issue.
- v9 improved dependence control using mass planing and moment-decomposition constraints, but reduced some nominal discrimination.
- v10 through v12 tried to recover performance while preserving decorrelation.
- v13 gave the best balance among the reported versions, using AUC-oriented training, MoDe-style penalties, adversarial decorrelation, charge-aware inputs, and a stronger point-cloud architecture.

The paper's conclusion is not that dependence must be zero. Some dependence is physically real. The correct target is **stable, understood, and reported dependence**.

## What We Can Learn For The Benchmark Paper

### 1. Benchmark Metrics Need Physics-Aware Diagnostics

For our benchmark, reporting only accuracy or ROC-AUC would be weak. We should report:

- inclusive ROC-AUC
- energy-matched ROC-AUC
- background rejection at fixed signal efficiency
- score correlation with energy
- score correlation with hit count / total charge when available
- threshold acceptance as a function of energy
- score distribution shifts across energy bins

If spatial variables are available, also report:

- radius-matched ROC-AUC
- score-vs-radius dependence
- score or acceptance heatmaps in energy-radius space

### 2. Unified Transformers Need Shortcut Tests

A unified tokenized transformer may be powerful, but it may also learn easy detector shortcuts:

- waveform length / occupancy
- number of hits
- total charge
- event size
- missing or thresholded detector bins
- geometry-dependent timing

So the benchmark should test not only whether the transformer gets high AUC, but also whether it is learning robust physics structure.

### 3. Specialized Models Are Not Automatically Safer

Dataset-specific models can also learn shortcuts. A 1D CNN on waveforms can learn energy-like amplitude or baseline artifacts. A point-cloud model can learn hit count. A tabular model can learn class-specific simulation artifacts.

This means the benchmark should compare models on both:

- raw predictive performance
- stability / dependence diagnostics

### 4. The Benchmark Paper Can Be Stronger Than A Leaderboard

The KLZNet paper suggests a good benchmark paper should not just rank models. It should define a reporting standard for rare-event ML:

- performance
- robustness
- nuisance dependence
- physics interpretability
- acceptance behavior after score cuts

That is more publishable than a plain model comparison.

## Practices To Take Forward

- Always inspect raw feature dependence before training.
- Report energy-matched metrics alongside inclusive metrics.
- Check whether score cuts sculpt physics-relevant distributions.
- Use class-conditional diagnostics, not only global metrics.
- Treat decorrelation as a tradeoff: removing shortcuts may lower inclusive AUC but improve physics validity.
- Keep plots for score vs energy, score vs timing, and score vs detector occupancy.
- Use control samples or real-data closure checks when available.

## Practices To Avoid

- Do not select the best model only by inclusive AUC.
- Do not assume removing explicit energy input removes energy dependence.
- Do not call a learned feature "topology" until its energy, charge, timing, and radius dependence has been checked.
- Do not ignore threshold behavior; a fixed score cut can distort spectra.
- Do not assume a unified transformer is better just because it is more general.
- Do not over-decorrelate blindly; some dependence is real physics information.

## How This Connects To Our Four-Dataset Benchmark

CUORE:

- Check whether waveform models learn pulse height, baseline, or pile-up-specific amplitude shortcuts.
- Report waveform score dependence on simple summary features such as peak value, integral, and rise time.

MJD:

- Directly connected to Jaden's previous AvsE / low-AvsE work.
- AvsE-style features are physically meaningful but also need energy-dependence checks.

NEXT:

- Point-cloud / voxel models can learn hit count, total energy, event size, and spatial extent.
- Need energy-matched and topology-aware comparisons.

SuperNEMO:

- Tabular baselines can strongly exploit energy and angular distributions.
- Need to separate "good classification" from learning class-specific spectrum differences.

## Meeting Talking Point

"The KLZNet dependence paper shows that for rare-event ML, high AUC is not enough because models can learn energy, timing, charge, or position shortcuts through detector response. For our benchmark, I think we should report both performance and physics-aware dependence diagnostics, especially energy-matched AUC and score-vs-energy or threshold-acceptance curves."

