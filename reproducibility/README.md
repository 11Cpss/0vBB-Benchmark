# EnergyBench paper reproducibility bundle

This is an independent, local snapshot for **Learning Physics, Not Energy: A Multimodal Benchmark for Neutrino Detectors**, prepared on **23 September 2026**. It collects the located training, inference, preprocessing, evaluation, table-generation, and figure-generation code used by the current manuscript. It includes the current Overleaf working tree, including the latest uncommitted motivation-figure revision. Original source directories and historical code were not edited, moved, or deleted.

**Start here:** use the commands below, not the old absolute-path audit scripts. The package reproduces the numerical and figure outputs for which source inputs were recovered. It also documents specific unresolved provenance gaps: nine reported RoPE configurations and eight supplied dataset-example PDFs do not have recovered original generators/checkpoints. Their archived results/assets are retained; no substitute model, event, or reconstructed result is claimed.

**Looking for the benchmark implementation?** Open [`benchmark/README.md`](benchmark/README.md). The actual I, matched AUC and inclusive AUC calculations are in [`benchmark/core/unified_metrics.py`](benchmark/core/unified_metrics.py), with [`benchmark/protocol.json`](benchmark/protocol.json), numerical tests, historical implementations, and the runnable entry [`benchmark/evaluate.py`](benchmark/evaluate.py). These are visible, byte-identical copies of the previously included evaluator sources.

## Contents

| Location | Purpose |
| --- | --- |
| `benchmark/` | Actual EnergyBench metric source, protocol configurations, numerical tests, historical benchmark packages and a visible evaluation entry point |
| `paper/` | Frozen current manuscript, original supplied PDF, all figure assets, reviewed numerical data, original publication scripts, fonts and bibliography |
| `code/classic/` | Dataset-specific classic-model training, inference, architectures, preprocessing and selected run configurations |
| `code/transformers/` | Transformer/tokenization/position-encoding implementations, actual published run configurations, and source-to-table linkage |
| `code/figure_sources/` | Recovered external figure-preprocessing sources, complete extent inputs, and portable figure verification |
| `code/input_preparation/` | The extraction and metadata-alignment sources used to construct the audited event archives; original audit snapshots are clearly separated from runnable entry points |
| `reproduction/` | Portable entry points for metric evaluation, final table/figure rendering, integrity checks, and isolated paper compilation |
| `environments/` | Scientific-Python requirements with tested versions; separate model-training environment notes; optional TeX font fallback |
| `provenance/` | Original locations, SHA-256 manifests, publication/output inventory, source-paper Git state, and external input requirements |
| `event_data/` | Optional, separately packaged exact per-event inputs; **excluded from the code-only archive** |
| `validation/` | Executed test, rendering, metric replay, portability, and compilation evidence |

The model-specific instructions are [classic README](code/classic/README.md) and [Transformer README](code/transformers/README.md). See [figure provenance](code/figure_sources/README.md) for the exact active figures and [input semantics](code/input_preparation/README.md) before evaluating a new prediction archive.

## 1. Install the lightweight evaluation and rendering environment

Run commands from the extracted bundle root. Use Python **3.11**; do not use an unqualified legacy Python 2 interpreter.

```bash
python3.11 -m venv .venv-reproduce
.venv-reproduce/bin/python -m pip install -r environments/requirements-core.txt
.venv-reproduce/bin/python reproduction/reproduce.py check --output outputs/check
```

The pinned scientific packages are the versions used for the successful validations in this bundle. GPU libraries are not needed for saved-prediction evaluation or figure replay. Training and checkpoint inference have additional dataset-specific PyTorch dependencies; install them in separate environments described under `environments/` and the model READMEs. No training job is started by the quick-start commands.

## 2. Rebuild final tables and available figures

```bash
.venv-reproduce/bin/python reproduction/reproduce.py render --output outputs/render
```

This regenerates the six numerical tables from the reviewed full-precision results and the recovered figures from their exact plotting inputs. It verifies that all six generated table files match the current manuscript. The main table has three final presentation differences from its original generator: column spacing, row spacing, and the first block heading. The portable wrapper applies only these three documented substitutions to its output; it never changes numerical values or the frozen generator.

All six active Wing figure PDFs were independently regenerated byte for byte in the validated environment. The renderer also emits a few original secondary figure outputs, but does not substitute them for the active manuscript assets. The eight dataset-description PDFs are preserved as supplied because their original generating code/event selection was not recovered. Their exact source hashes and status are listed in `provenance/figure_inventory.json`.

This rendering command reads numerical/plotting outputs. It is **not** an event-level metric recomputation, model inference, or retraining command.

## 3. Recompute metrics from the optional per-event companion

Extract the event-data companion into the same bundle root so that `event_data/events/` is present. Its event archives contain the original scores, labels, physical energies, event IDs, categories and base weights; they are not fabricated from tables.

```bash
.venv-reproduce/bin/python reproduction/evaluate.py \
  --manifest provenance/event_inputs.json \
  --data-root event_data \
  --output outputs/event_metrics
```

The manifest contains **61 audited event-result records** including the extended NEXT campaigns and the separate SuperNEMO illustration. This number is not the number of main-table rows. Input hashes are checked before evaluation, each record selects its frozen publication profile, and the recomputed I, matched AUC, inclusive AUC and event count must match the full-precision reference. No missing model is silently skipped, and no failed reporting gate is relaxed.

To evaluate just the strict-range MJD/EXO-200 records:

```bash
.venv-reproduce/bin/python reproduction/evaluate.py \
  --manifest provenance/event_inputs.json --data-root event_data \
  --dataset MJD --dataset EXO-200 --output outputs/mjd_exo_metrics
```

The companion additionally contains the original native event predictions for **12 historical NEXT/SuperNEMO MLP/Fourier Transformer runs**. Replay them with their original recorded implementations, without substituting either newer profile:

```bash
.venv-reproduce/bin/python code/transformers/replay_historical_transformers.py \
  --output-dir outputs/historical_transformer_metrics
```

Those 12 native replays and the 61 profile-registry replays all passed. Together they cover the recoverable results for 43 of the 52 main-table dataset/model entries plus extended/illustrative records. The remaining nine main-table entries are the unrecovered NEXT/MJD/SuperNEMO RoPE runs. The complete optional per-event companion is approximately 262.5 MiB before packaging.

The companion does not contain raw detector waveforms or trained checkpoints. Those are needed for a new inference or training run. Original locations and immutable identifiers are recorded in the dataset manifests; no large training dataset is downloaded automatically.

### One new standardized prediction file

```bash
.venv-reproduce/bin/python reproduction/evaluate.py \
  --input /path/to/standardized_predictions.npz \
  --profile strict600 --output outputs/new_model_metrics.json
```

Required aligned one-dimensional fields are `score`, `label`, `energy_keV`, and unique nonempty `event_id`. Optional `group` and `weight` preserve the physical categories and original base weights. Higher scores must favor `label=1`. For energies stored in MeV, explicitly use `--energy-key energy --energy-unit MeV`. The CLI cannot infer the correct class orientation, checkpoint, test split, or the physical meaning of the energy array.

Use `--profile overflow601` only when reproducing a record originally evaluated with that retained profile. Do not relabel a historical Transformer result as strict-range merely because a current evaluator is available.

## 4. Evaluation definitions and published scope

The archived results use more than one recorded profile. The code and per-record fingerprints, rather than a blanket assumption about all rows, specify the exact reproduction target.

| Publication subset | Correct evaluator/profile |
| --- | --- |
| Four classic models and six MLP/Fourier Transformers on MJD; four classic models and nine Transformers on EXO-200 | `strict600`: exactly 600 fixed 5 keV bins on 0–3000 keV, internal intervals left-closed/right-open, final interval includes 3000; out-of-range events excluded without clipping |
| Retained NEXT and SuperNEMO classic/extended results; independent SuperNEMO 0nu/Bi214 illustration | `overflow601`: the same 600 regular bins plus one bin for finite energies strictly above 3000 keV |
| Other reported Transformer results | Their recorded historical dataset implementations/configurations; see the Transformer linkage manifest. Unrecovered RoPE runs are not claimed as newly evaluated |

Both energy metrics use the specified energy grid but retain distinct statistics. I uses within-true-class/physical-category score distributions, up to 20 weighted score-quantile intervals, natural-log Jensen–Shannon divergence, an in-group weighted average before the square root, then population-weighted group aggregation. Each eligible group/energy bin requires at least 20 positive-weight events.

Matched AUC uses trimmed common support, an overlap target, per-class normalized event weights and a pooled weighted AUC, including cross-bin pairs and half credit for ties. Each retained energy bin requires at least 20 events of each class. Reporting requires at least 50% of each original finite score/energy class mass and at least two eligible bins, with the documented exact-constant-support exception for bin count only. Coverage denominators precede range exclusions. Inclusive AUC always retains its original finite-score/positive-weight population independently of energy.

Important input distinctions:

- EXO-200: stored label **0** (one charge cluster) is positive. Standardization flips the binary labels and negates the background-oriented logit. Energy is original rotated energy in keV.
- MJD: positive means all four reference PSD flags pass. Binary Transformer logits and the classic GINE four-output construction are different. Original float64 calibrated energy is recovered before any historical clipping/casting.
- NEXT: convert physical summed hit energy from MeV to keV. The referenced NEXT PointMamba test population has 115,499 events; the main campaign has 116,549.
- SuperNEMO: trained **2nu/Bi214 classification** and the independent **0nu/Bi214 energy-only illustration** are different tasks. The latest motivation figure uses MJD in panel (a) and that SuperNEMO illustration in panels (b,c).

The frozen shared implementations remain byte-identical to their audited sources. No historical incorrect evaluator was edited or silently replaced in its original project.

## 5. Reproduce preprocessing, training and inference

Use the dataset-specific commands and saved run configurations in the model READMEs. They preserve the published architecture, tokenization, positional encoding, training hyperparameters and split definitions wherever these were recoverable. Some source programs still contain original absolute-path defaults; the READMEs state which arguments/configuration paths must be supplied in a new checkout. Checkpoint inference must use the exact archived checkpoint and test population; substituting a retrained checkpoint is a new experiment.

For the independent figure preprocessing available without large raw detector files:

```bash
.venv-reproduce/bin/python code/figure_sources/rebuild_extent_inputs.py
```

This reconstructs the complete extent CDF, conditional histograms and medians from the included event-derived descriptors. It does not claim to reconstruct hit-level detector data.

The `reference_snapshots/` and `archived_preprocessing/` directories preserve historical source for inspection. Their old audit-tree layouts and absolute output destinations are not portable runtime contracts. The recommended quick-start never executes those historical scripts. In particular, historical quantile-bin metrics must not be used as replacements for the frozen final profiles.

## 6. Compile the manuscript copy

With a TeX Live distribution and `latexmk` installed:

```bash
.venv-reproduce/bin/python reproduction/reproduce.py compile \
  --engine latexmk --output outputs/compiled
```

Or with Tectonic:

```bash
.venv-reproduce/bin/python reproduction/reproduce.py compile \
  --engine tectonic --output outputs/compiled-tectonic
```

Tectonic may download its normal TeX bundle on first use. A prepared offline cache can be used with `--only-cached`. The optional `environments/texfonts/` fallback was used for the audited offline compilation. The result is `outputs/compiled/paper.pdf` (or the selected output directory). Compilation first copies the frozen paper into a new build directory and does not overwrite the original manuscript or its supplied PDF.

The supplied `paper/iclr2027_conference.pdf` predates some current source edits. The fresh validation PDF in `validation/compiled/paper.pdf` is built from this bundle's current working-tree source snapshot. A new compile requires a fresh output directory so an earlier build is not silently destroyed. The current source snapshot compiles to 25 pages, but already contains two unresolved references (`app` and `fig:energybench-auc`). They were preserved rather than silently edited during code packaging; see `validation/compiled/compile_receipt.json`.

## 7. Validation and remaining limitations

Executed validation records are under `validation/`: numerical boundary/unit/tie/sparsity tests, 61 profile-registry plus 12 historical event-record replays, final table equality, the six active figure PDF checks, preprocessing reconstruction, source hashes and isolated compilation. Model packages are syntax/import checked where feasible; these checks do not claim retraining or checkpoint inference.

The unresolved provenance gaps are explicit:

1. Original implementations/checkpoints for the three RoPE tokenizations on **NEXT, MJD and SuperNEMO** were not recovered in the available source trees. The EXO-200 RoPE implementation is present, but is not substituted for those other models. Their reported values remain traceable display results.
2. The original generators and exact selected events of eight `dataset_description/Images/{mjd,next,supernemo,exo}_{signal,background}.pdf` panels were not recovered. Supplied PDFs are included intact.
3. Raw detector datasets and trained checkpoints are external. Preserving training code/configuration is not a promise of bit-identical retraining across hardware/library versions. Single-run point estimates are not significance tests.
4. Published numerical summaries have their recorded precision. Where a workbook retained only rounded digits, a source-to-table match establishes those digits, not additional nonexistent precision.

No files were uploaded or pushed by the packaging operation. Use the code-only archive for code distribution and the separate event-data archive when event-level replay is desired. Original filesystem paths in provenance records document the local audit; recommended commands use paths inside the extracted bundle.
