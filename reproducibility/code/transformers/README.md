# Transformer source archive and runbook

Read the bundle root README first for the final paper's evaluation scope and the optional event-data companion. This directory preserves training, inference, preprocessing and tokenization code for the four datasets. Original files are byte-for-byte copies, including historical metric implementations and existing notebook outputs. New convenience runners are separate files; no original experiment source was edited.

## Coverage and authoritative records

`../../provenance/transformer_model_mapping.json` (and CSV) maps all 36 final-paper Transformer rows to source configuration, checkpoint and native prediction paths/hashes, parameter counts, stored paper values, historical values and explicit limitations. There are 27 source-backed rows: EXO-200 9, MJD 6, NEXT 6 and SuperNEMO 6. The 9 NEXT/MJD/SuperNEMO RoPE rows have no identified matching training implementation, checkpoint or event predictions. Their reported values are preserved as provenance, not claimed reproducible. The EXO RoPE implementation is present and is not substituted for another detector's missing model.

`../../provenance/transformer_source_manifest.json` records 201 copied files (12,591,651 bytes) with source and bundle SHA-256. All copied originals were rechecked unchanged. Checkpoints, raw HDF5, training logs and token caches are external. Original README files and notebooks are retained as historical evidence; their old scope descriptions, absolute paths and historical evaluation instructions do not supersede this runbook or the root README.

| Dataset | Source package and actual saved run | Configurations and final evaluation |
|---|---|---|
| EXO-200 | `original/exo200_detector`; original `results/transformer_official_v1/classification__*` | `configs/exo200_detector/`: pulse entities / raw patches / segment summary, each with coordinate MLP / Fourier / RoPE. Final metrics use root `strict600`. |
| MJD | `original/mjd_detector`; original `results/transformer_official_v1/classification__*` | `configs/mjd_detector/`: same three tokenizations, MLP / Fourier only. Final metrics use root `strict600`. |
| NEXT | `original/next_detector` + `original/evalutaions_workflow` | Four entity/region runs from `final_cached_v1`; two summary runs from `final`. Preserve the original historical NEXT metric configuration. |
| SuperNEMO | `published_supernemo`, copied from `/home/wenyu/SuperNEMO` | The six actual published classification runs are `configs/published_supernemo/`. Preserve original quantile evaluation: 6 matching bins and 8 within-group energy bins for I. |

The canonical `original/supernemo_detector` package is also preserved because it belongs to the requested source tree, but it is **not** the source of the six published SuperNEMO runs. Its differently named model runs must not replace those results. SuperNEMO classification is 2nu versus Bi214; the independent 0nu energy illustration is a different task.

For NEXT, all recorded paper/workbook digits of I and matched AUC match the corresponding saved run after rounding to that field's actual stored precision, and exact parameter counts match. The workbook does not store full binary metric precision; the unrounded source values are retained in the mapping. For SuperNEMO all six saved source values match the paper's stored values exactly. Missing models are not linked by names alone.

## Environments

Use an isolated environment; see `../../environments/transformers/README.md`. The original requirements and pyproject are preserved there. `requirements.txt` provides dependency bounds for the copied code; `validation_environment.json` records the actual environment used for checks and historical replays, not the unknown environment of every original training run.

From the bundle root:

```bash
python3 -m venv .venv-transformers
.venv-transformers/bin/pip install -r environments/transformers/requirements.txt
```

Select a PyTorch build appropriate to the compute host when training. Run the two different packages named `energybench` in separate processes, as the historical replay runner does.

## Reproduce the final metrics without training

The root evaluator takes explicitly standardized, aligned event fields (`score`, `label`, `energy_keV`, `event_id`, `group`, `weight`). It does not infer score direction:

```bash
python -B reproduction/evaluate.py --input /path/standardized_model.npz --profile strict600 --output outputs/model_metrics.json
```

Use this entry point for EXO-200 9 and MJD 6 Transformer rows. EXO original label 0 means single-cluster signal, and the native single logit favors background. The final adapter uses `label = 1 - native_label` and `score = -native_logit`, with the physical `Rotated_energy` in keV; no sigmoid is inserted. MJD positive events pass all four PSD labels, and the native logit already favors the positive class. Its final adapter recovers original float64 physical energy rather than treating cached float32 energy as the authoritative raw value. Event alignment and source physical energy are necessary; a freshly generated native prediction file is not automatically a final standardized input.

Historical NEXT and SuperNEMO Transformer rows were not migrated to `strict600` or `overflow601`. Their native event files are optional companion data under `event_data/historical_transformers/` (12 files, 47,147,890 bytes), with hashes and original metric configuration in `provenance/transformer_historical_inputs.json`. Replay their original definitions:

```bash
python -B code/transformers/replay_historical_transformers.py --output-dir outputs/historical_transformer_replay
```

All 12 replays passed against full-precision source values: SuperNEMO errors are zero; NEXT I errors are zero and maximum matched-AUC error is 3.34e-16. NEXT native energy is MeV, label 1 is 0nubb; SuperNEMO energy is E1 + E2 in keV, label 1 is 2nu. The archived scores are used unchanged. Only the unrelated distance-correlation sample cap is reduced to 4 for replay speed; neither I nor AUC uses that computation, and no distance-correlation result is reported.

The SuperNEMO replay imports the generic frozen package stored at `original/exo200_detector/frozen_energybench/energybench`. This is the exact evaluator used by the actual published SuperNEMO runs, despite its directory name: its package fingerprint, the installed original `/home/wenyu/summer/src/energybench` fingerprint, and all six run-config fingerprints are `f3f0abfd7916b75cabf19c3e9cdfc360940af3453f908bdabb1c8ed165a5d37c`. See `validation/transformer_historical_validation.json` for the algorithm and per-file comparison.

## EXO-200 and MJD training or original-checkpoint inference

The new `waveform_run.py` selects one exact archived configuration and delegates loaders, tokenization, model construction and optimization to the original copied packages. It requires a new/empty output directory, checks parameter and split counts, and checks the original checkpoint hash in test mode. It rejects missing configurations. These commands are examples for external original data/checkpoints; training was not run while assembling this archive.

```bash
python -B code/transformers/waveform_run.py --dataset EXO-200 --model-key entity_rope --describe
python -B code/transformers/waveform_run.py --dataset EXO-200 --model-key entity_rope --mode train --data-root /path/EXO-200 --output-dir outputs/exo_entity_rope
python -B code/transformers/waveform_run.py --dataset EXO-200 --model-key entity_rope --mode test --data-root /path/EXO-200 --checkpoint /path/original/best.pt --output-dir outputs/exo_entity_rope_test
python -B code/transformers/waveform_run.py --dataset MJD --model-key region_fourier --mode train --data-root /path/MJD --output-dir outputs/mjd_region_fourier
python -B code/transformers/waveform_run.py --dataset MJD --model-key region_fourier --mode test --data-root /path/MJD --checkpoint /path/original/best.pt --output-dir outputs/mjd_region_fourier_test
```

Model keys are `entity_mlp`, `entity_fourier`, `entity_rope`, `region_mlp`, `region_fourier`, `region_rope`, `summary_mlp`, `summary_fourier`, `summary_rope`; only EXO has source-backed RoPE keys. External raw files must be the original release and satisfy the preserved manifests. Reconstructing test events from a different dataset version or arbitrary split is not authorized by these examples. Original notebooks remain available for the full original workflow, but contain hardcoded paths and run-selection logic; do not execute them blindly against original output directories.

## NEXT training or original-checkpoint inference

The new `next_run.py` reproduces the archived representation and training settings through original APIs. For the four cached runs, the saved representation JSON has no serialized training config; the corresponding archived notebook uses `TrainingConfig(num_workers=8)` and `EvaluationConfig()`, which the wrapper uses. Two summary configurations contain their serialized settings. Split settings, counts and event-index lists are checked against `configs/next/event_split.json`. Test mode checks the original checkpoint SHA.

Create a writable working copy of the split manifest. Original file inventory/path checks still apply; relocating raw data requires explicit path bookkeeping and must preserve the exact event lists. A cache is validated against source file paths, manifest hash and tokenization source, so another cache is not silently accepted.

```bash
mkdir -p outputs/next_work
cp code/transformers/configs/next/event_split.json outputs/next_work/event_split.json
python -B code/transformers/original/next_detector/scripts/build_token_cache.py --data-root /path/NEXT --split-manifest outputs/next_work/event_split.json --cache-root outputs/next_work/cache --tokenization sampled_hits --max-tokens 512 --voxel-size 15 --coordinate-scale 1000 --seed 42
python -B code/transformers/next_run.py --model-key entity_mlp --mode train --data-root /path/NEXT --split-manifest outputs/next_work/event_split.json --token-cache /path/cache-directory-reported-by-builder --output-dir outputs/next_entity_mlp
python -B code/transformers/next_run.py --model-key entity_mlp --mode test --data-root /path/NEXT --split-manifest outputs/next_work/event_split.json --token-cache /path/cache-directory-reported-by-builder --checkpoint /path/original/training/best_model.pt --output-dir outputs/next_entity_mlp_test
python -B code/transformers/next_run.py --model-key summary_fourier --mode train --data-root /path/NEXT --split-manifest outputs/next_work/event_split.json --output-dir outputs/next_summary_fourier
```

Use `--tokenization voxel` to build the region cache. The six IDs map to entity MLP (001), region MLP (002), region Fourier (003), entity Fourier (004), summary MLP (005), summary Fourier (006). Exact IDs and source directories are in the mapping. The cached entity/region path and uncached summary path are intentional. RoPE has no source-backed key.

## Actual published SuperNEMO training and inference

Use the original published package, not the canonical alternative. Its CLI writes under its own package root. Work from a separate writable copy so the archived source stays unchanged:

```bash
mkdir -p outputs
cp -a code/transformers/published_supernemo outputs/supernemo_work
BUNDLE="$(pwd)"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$BUNDLE/outputs/supernemo_work:$BUNDLE/code/transformers/original/exo200_detector/frozen_energybench" python -B -m supernemobench.workflow --task classification --model transformer_001_sampled_hits_coordinate_mlp --mode train --data-root /path/SuperNEMO
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$BUNDLE/outputs/supernemo_work:$BUNDLE/code/transformers/original/exo200_detector/frozen_energybench" python -B -m supernemobench.workflow --task classification --model transformer_001_sampled_hits_coordinate_mlp --mode test --data-root /path/SuperNEMO --checkpoint /path/original/best.pt
```

The six model IDs are the same numbered tokenization/position combinations listed for NEXT, but the actual architectures, token caps and voxel sizes are SuperNEMO-specific (128-token entity/region, 60 mm voxels in the archived configuration). This CLI has no `--manifest-path`; preserve the packaged split manifest and original raw-file identities. Its strict checkpoint/config provenance checks may reject relocated absolute paths or changed mtimes; repair path references only in a separate working copy with recorded provenance, never change event assignments to bypass those checks. Raw release files and checkpoints are not included. The original full registry has shared classic/energy code, retained because package imports depend on it; this runbook invokes only the six published classification Transformers.

## Checks and limits

Run `python -B code/transformers/validate_sources.py` to repeat Python and notebook-cell syntax checks, isolated package imports/model constructors, and bundled-file hashes. Add `--check-originals` only on the original workstation to also compare the original source files; this optional mode requires those local paths. The validation record is `validation/transformer_validation.json`; convenience entry-point checks are in `validation/transformer_entrypoints.json`. Re-running the checker writes its environment to `validation/transformer_validation_environment.json` and preserves the original `environments/transformers/validation_environment.json` snapshot. Notebook cells were transformed with IPython and compiled, not executed. The two new convenience runners were checked with `--help` and `--describe`, not with full training or checkpoint inference. No model forward pass or training was performed for archive assembly. These checks establish source completeness/importability and saved-prediction metric replay, not bitwise hardware-independent retraining.

The search for missing RoPE files is recorded in `provenance/transformer_search_paths.txt`; an unrelated MJD Optuna energy-regression notebook was inspected and excluded. No checkpoint/model/task was fabricated or replaced. Missing code or checkpoint availability remains an explicit reproducibility limitation for the nine paper-only entries.
