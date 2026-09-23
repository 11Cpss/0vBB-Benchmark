# Portable paper evaluation and rendering

This directory carries the validated EnergyBench evaluators and the profiles used by the paper. The current reevaluation covers exactly 23 records: four classic models and six MLP/Fourier Transformers for MJD, plus four classic models and nine Transformers for EXO-200. No training or replacement checkpoint is used.

The default `core/unified_metrics.py` and `protocol.json` implement `EnergyBench-unified-5keV-range-v3.0.0`: exactly 600 fixed 5 keV bins from 0 to 3000 keV, with 3000 included in the final bin. Energies outside this range are excluded before support estimation and score-histogram construction, never clipped. Inclusive AUC retains the original score/weight population. Reported coverage divides by the original finite score/energy class mass before range filtering.

NEXT and SuperNEMO classic results, NEXT exploratory results, and the separate SuperNEMO energy-only illustration retain their preceding v2 profile and numerical sources. Its frozen evaluator and configuration are in `legacy_v2/`: 600 regular bins through 3000 keV plus one bin for all higher energies. The three MJD RoPE entries, and the NEXT/SuperNEMO Transformer entries, remain historical reported results outside this reevaluation.

The paper therefore contains more than one evaluation profile. `figures/data/unified_results.json` (relative to `wing_contribution/`) is a registry whose individual records identify their protocol version, fingerprint, input source and status. Rebuilding validates the recorded profile for each evaluated record; a historical entry is not relabeled as a v3 result. Evaluator provenance is recorded in `data/evaluator_provenance.json`, and the selected Transformer campaign is recorded in `campaigns/transformer_exo_mjd_20260917.json`.

Run these commands from the repository root with Python 3.11 or later:

```bash
python3 -m venv .venv-energybench
.venv-energybench/bin/python -m pip install -r wing_contribution/evaluation/requirements.txt
```

## Rebuild the published tables and figures

```bash
.venv-energybench/bin/python wing_contribution/scripts/rebuild_unified_evaluation.py --output-dir /tmp/energybench-paper-render
```

The output contains five tables, the NEXT capacity figure, the separate SuperNEMO energy-only figure, and a reconstruction receipt. This operation reads the checked-in final CSV/JSON data and leaves those numerical sources unchanged. It preserves the applicable v3 or v2 profile rather than applying the default evaluator to every figure or table row. It requires neither the original account layout nor a sibling audit directory. `data/diagnostics.json` contains reviewed aggregate display data for the diagnostic tables; it is not used as event-level metric input.

To update generated figures and tables in the checkout, omit `--output-dir`. Add `--compile` to compile the paper with a separately installed `latexmk` and TeX distribution. The renderer calls the repository's current main-table renderer, preserving its current formatting.

The existing individual entry points also work:

```bash
.venv-energybench/bin/python wing_contribution/scripts/plot_next_capacity.py --output-dir /tmp/energybench-figures
.venv-energybench/bin/python wing_contribution/scripts/prepare_supernemo_matching.py
.venv-energybench/bin/python wing_contribution/scripts/plot_supernemo_matching.py --output-dir /tmp/energybench-figures
```

`prepare_supernemo_matching.py` without predictions validates the checked-in histograms, ROC curves, normalization, AUC and v2 protocol fingerprint. The SuperNEMO illustration entry points use the frozen `legacy_v2` evaluator; changing the default MJD/EXO-200 profile does not change this figure's population or metrics. No event result is reconstructed from aggregate values.

### Figure typography

All seven figures included by Wing's sections use `paper_style.py`: the bundled
Nimbus Roman family matches the manuscript's Times text; Computer Modern
matches its unchanged LaTeX mathematics.
Axes and panel headings are normally 9 pt; ticks, legends and notes are 8 pt
(7–7.5 pt in compact insets and the three-panel ROC legend). Figures are drawn at their final LaTeX width,
usually 5.5 inches, so inclusion does not unexpectedly shrink the type. Export
vector PDF without `bbox_inches='tight'`, which would change that physical width.

The two MJD figures are rebuilt with `mjd_style/render.py`; see its README.
The energy spectrum, threshold tradeoff and tracker-extent figures are rebuilt
with `../scripts/plot_appendix_figures.py`; see `appendix_figures/README.md` for
the frozen aggregate inputs and the separately supplied exact ECDF source.

## Evaluate external event predictions

Full benchmark test sets, classifier prediction archives, and model checkpoints are not included. The MJD waveform figures include only their frozen display data; see `mjd_style/README.md` to regenerate those figures. Obtain the original held-out predictions and their matching physical energies separately. MJD uses the original float64 `energy_label` in keV, before any classic-adapter clipping or Transformer-cache float32 conversion. Recover energies through verified event identities and the actual test-loader mapping. Never substitute a different checkpoint or test population.

A standard NPZ contains aligned one-dimensional arrays:

- `score`: the original classifier score; higher supports `label=1`.
- `label`: binary 0/1 values with the experiment's audited class semantics.
- `energy_keV`: original physical energy, retaining out-of-range values so their exclusions can be counted against the original population.
- `event_id`: a unique, nonempty identifier for each test event.
- `group` (optional): the physical category used for class-conditional independence; otherwise true binary labels are used.
- `weight` (optional): finite, nonnegative base weights; otherwise unit weights are used.

```bash
.venv-energybench/bin/python wing_contribution/evaluation/evaluate_npz.py /path/to/test_predictions.npz --output /tmp/model_metrics.json
```

This command evaluates one standardized archive under the default strict v3 profile. For an archive storing MeV in its `energy` field, add `--energy-unit MeV`. Use `--energy-key NAME` for a differently named energy array. Optional `--weights-output /tmp/matched_weights.npz` saves event IDs with the matching weights and masks. The result records full precision, protocol fingerprint, source-file hash, class coverage, separate range/support/sparse losses, and estimation status. A missing formal matched AUC stays `null`; no gate is relaxed. Inclusive AUC uses the original score/weight population independently of energy.

The CLI does not infer label or score semantics. EXO-200's evaluation-positive class is stored label 0 (one charge cluster): standardized inputs use `label=1-stored_label` and `score=-native_background_logit`. MJD's positive class passes all four reference PSD flags; the six reevaluated Transformers output one clean-oriented binary logit. The four-output PSD score construction applies to classic MJD GINE, not those Transformers. Preserve the audited scalar score representation without adding a sigmoid or other calibration.

For the separate SuperNEMO illustration, the classification task must be `0nubb=1` versus `Bi214=0` and the fixed score must equal physical energy in keV. This is different from the trained `2nu` versus `Bi214` task:

```bash
.venv-energybench/bin/python wing_contribution/scripts/prepare_supernemo_matching.py --predictions /path/to/energy_only_0nu_predictions.npz --output-dir /tmp/energybench-illustration-data
.venv-energybench/bin/python wing_contribution/scripts/plot_supernemo_matching.py --data-dir /tmp/energybench-illustration-data --output-dir /tmp/energybench-illustration-figures
```

Fresh event evaluation deliberately writes a separate result. Updating the published multi-model source data requires review of source identities, class/score direction, units, protocol and reportability; rendering alone does not claim a new metric calculation. Machine paths preserved in old provenance JSON are historical descriptions, not runtime dependencies.
