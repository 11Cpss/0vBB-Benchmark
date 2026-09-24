# EnergyBench evaluation

[metrics.py](metrics.py) implements energy independence I, energy-matched AUC,
and inclusive AUC. [evaluate.py](evaluate.py) is the single entry point for
standardized per-event predictions. Settings are explicit JSON files in
[configs/](configs/).

## Inputs

Supply an NPZ containing one-dimensional, event-aligned arrays:

| Field | Meaning |
| --- | --- |
| `score` | Scalar score; larger values favor `label=1` |
| `label` | True binary label, 0 or 1 |
| `energy_keV` | Original physical energy in keV, without clipping |
| `event_id` | Unique nonempty physical event identifier |
| `group` (optional) | Physical category for I; defaults to true binary label |
| `weight` (optional) | Finite nonnegative base event weight; defaults to one |

For a MeV energy array named `energy`, explicitly pass `--energy-key energy
--energy-unit MeV`. To standardize native model outputs and align energy by
event identity, see [input preparation](../code/input_preparation/README.md).

```bash
# Run from the reproducibility directory with its Python environment.
.venv-reproduce/bin/python -B benchmark/evaluate.py \
  --input /path/to/predictions.npz --profile strict600 \
  --output outputs/metrics.json

.venv-reproduce/bin/python -B benchmark/tests/test_metrics.py
```

## Statistical definitions

I measures changes of the within-group score distribution with energy. It uses
weighted score-quantile histogram bins, natural-log Jensen–Shannon divergence,
an average over eligible energy bins before the square root, and then a
population-weighted average of group scores. An energy bin must contain at
least 20 positive-weight group events. Outputs include every group I,
the minimum group score, retained fractions and eligible-bin counts.

Matched AUC uses trimmed common support, at least 20 events of each class per
eligible energy bin, an overlap target and class-normalized weights. The pooled
weighted AUC retains cross-bin pairs and gives ties half credit. Reporting
requires at least 50% coverage in each original finite score/energy class
population and at least two valid bins, with the stated exact-constant-support
exception. Non-estimable formal results are null. Diagnostic AUC is separately
named and must not be substituted for a result failing the reporting gates.

Inclusive AUC uses the original finite-score, positive-weight population,
independently of energy range. Missing energies and range exclusions therefore
do not silently change its meaning.

## Which profile applies to each paper result

| Reported subset | Definition |
| --- | --- |
| MJD: four classic and six MLP/Fourier Transformers; EXO-200: four classic and nine Transformers | [strict600](configs/strict600.json): keV; exactly 600 fixed 5 keV bins; 0–3000 inclusive; left-closed/right-open intervals, with 3000 in the final bin; out-of-range energy excluded without clipping |
| NEXT/SuperNEMO classic and appendix comparisons; separate SuperNEMO energy-only illustration | [overflow601](configs/overflow601.json): the same regular grid plus a separate finite-energy bin above 3000; negative energy excluded |
| Six NEXT MLP/Fourier Transformer results | Named published Transformer profile in `published_profiles/next_transformer/`; per-run settings in `results/transformer_inputs.json` |
| Six SuperNEMO MLP/Fourier Transformer results | Named published Transformer profile in `published_profiles/supernemo_transformer/`; six matching energy-quantile bins and eight classwise energy-quantile bins for I, as specified by those runs |

The shared profiles use one implementation in `metrics.py`. The two additional
Transformer computations are retained only because those definitions produce
results still reported in the paper. They are invoked by
`code/transformers/replay_paper_metrics.py` with a fixed result manifest, not
used as alternative defaults for MJD/EXO-200. Removing those computations or
substituting strict600 would make the retained paper values unreproducible.
Training checkpoint-selection metrics are separate from final test metrics.

## Published-result replay

```bash
.venv-reproduce/bin/python -B benchmark/evaluate.py \
  --manifest results/event_inputs.json --data-root /path/to/event_data \
  --output outputs/paper_metrics
```

Manifest mode selects each recorded profile, checks input and configuration
hashes, and compares I, matched AUC, inclusive AUC and event counts against
full-precision expected results. It also reports classwise coverage, matching
effective sample sizes, and separate range, support and sparse-bin losses.
No input is inferred from a table or an older metric value.
