# MJD Transformer evaluation audit, 2026-09-17

The six entity/region/summary × MLP/Fourier classifiers use the original saved
predictions from `Transformer_Approach/mjd_detector/results/transformer_official_v1`.
No model was trained or re-inferred. The three MJD RoPE rows are outside this
campaign by the user's explicit instruction; their historical paper entries are
not claims of evaluation under the final protocol.

## Final results

`strict600/` contains the final evaluation requested by the user: 600 fixed 5 keV
bins, physical energy 0–3000 keV inclusive, with out-of-range events excluded from
I and matched AUC. Inclusive AUC retains all 390,000 original test events.
`strict600/summary.json` and `strict600/comparison.csv` contain full-precision old
and new values. `strict600/details/` contains class-level I, valid bins, support,
coverage, effective sample sizes, and loss counts; coverage uses the original
finite per-class population before energy-range filtering. `strict600/manifest.json`
records all input hashes and the frozen protocol fingerprint.

Reproduce the final metrics from the already audited per-event inputs:

```bash
/home/wenyu/summer/.venv/bin/python -B \
  '/home/wenyu/iclr final paper/unified_5kev_evaluation/transformer_exo_mjd_20260917/mjd/recompute_strict600.py'
```

The script imports the repository's shared evaluator, accepts `--evaluation` and
`--output` paths, and requires exactly 600 bins. Raw predictions and HDF5 inputs
remain local; this audit does not place them in the paper Git repository.

## Source and population verification

`manifest.json` records the six original saved prediction files, best checkpoints,
run configurations, exact parameter counts, source-code hashes, and historical
unrounded metrics. `event_identity_audit.json` records the scalar source-field
hashes, physical event map, and exact comparisons. `standardized/*.npz` stores
score, label, original float64 energy in keV, true-class group, event ID and weight.

- The positive class is clean: all four low-AvsE, high-AvsE, DCR and LQ flags pass.
  The model predicts a single raw clean logit. No product of PSD probabilities,
  sign change, sigmoid, or model-specific label reconstruction is applied.
- Each saved energy-aware score array is exactly equal to that run's original
  training-evaluation prediction array. Its label vector is also exactly equal.
- The actual Transformer loader uses sorted `MJD_Test_0.hdf5` through
  `MJD_Test_5.hdf5`, every row, and unshuffled test indices. Its cached float32
  energy and clean labels match the saved arrays for all 390,000 events.
- Physical event IDs are recovered as shard basename, within-shard row and stored
  event ID. The cache did not itself save event IDs. Identity recovery is based on
  the actual loader and full per-event energy/label checks, not an assumed match
  to classic models. Afterwards, recovered raw energies, physical IDs, stored IDs,
  run numbers, detectors and trigger positions also exactly match the earlier
  independently extracted classic-test metadata.
- `energy_label` is original float64 keV. The historical Transformer reader cast
  it to float32. It did not clip energies. Restoring float64 moves three events
  across internal energy edges (indices 104979, 220470 and 297451). There are 67
  events above 3000 keV: 66 nonclean and one clean. All energies are finite and
  nonnegative. Their final energy-metric exclusion does not affect inclusive AUC.
- All six exact checkpoints load strictly into the architecture specified by their
  saved run configurations. Parameter counts, best epoch and validation score
  match the saved run metadata. Counts are 111105/113409 for entity and summary
  MLP/Fourier, and 112257/114561 for region MLP/Fourier.

## Historical replay and intermediate results

The historical saved Transformer protocol used 836 regular 5 keV bins from zero
to 4180 keV. The helper extended the grid to the observed maximum. This differs
from the historical classic MJD clipping and from both subsequent shared
protocols. `legacy_replay/` reproduces the old I, matched AUC and inclusive AUC from
the original per-event scores and native MeV arrays; discrepancies are asserted
below 1e-10. Only the auxiliary distance-correlation subsample is reduced to four
to save time; the headline metric definitions and settings are unchanged. Large
ROC point arrays are omitted from this duplicate audit, while their AUC and all
support diagnostics are retained.

The root-level `summary.json`, `comparison.csv` and `details/` preserve an
intermediate v2 computation with an overflow bin, performed before the user's
final instruction to exclude overflow. They are not the final paper results.
`reference_v2/` is a byte-identical frozen evaluator/configuration copy supporting
that historical audit. `recompute_mjd.py` rebuilds its source audit and v2 metrics;
`finalize_artifacts.py` verifies additional checkpoint and classic-test metadata.
`strict600/details/` links both this intermediate calculation and the original
historical replay, and reports the final changes separately.

The old MeV grid also placed one exact 2385 keV event in the preceding bin because
its generated MeV edge was one floating-point step above 2.385. The shared
integer-keV grid assigns it to bin 477 correctly. The precision, boundary and
range handling all contribute to the final metric changes. Matching trims its
common support well below 3000 keV, so these 67 high-energy events were already
outside historical matched support. However, excluding them before the final
support-quantile calculation slightly changes the support: from
[18.15252145767212, 2613.793811035156] keV to
[18.150531811779366, 2613.79042737314] keV. All class coverage fractions retain
the original finite-event denominators. The six matched-AUC changes are between
-1.38e-6 and -4.94e-7; I increases by 1.97e-5 to 4.72e-5. All six models satisfy
the reporting requirements, and their displayed four-decimal metrics and model
orderings remain unchanged. These are single-run point estimates.
