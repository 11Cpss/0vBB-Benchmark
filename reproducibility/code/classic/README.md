# Classic-model training and inference sources

This directory preserves the original training, preprocessing, representation, and inference code used by the four classic-model families and the additional NEXT models discussed in the paper. Files inside `NEXT/`, `MJD/`, `EXO200/`, and `SuperNEMO/` are byte-identical source snapshots collected on 2026-09-23. Original workspaces were not modified. These are snapshots of the available workspaces, not a claim that every file has remained unchanged since each historical training run.

The collection contains 23 distinct NEXT architecture IDs, the four selected classic architectures for MJD, EXO-200 and SuperNEMO, their shared package dependencies, original documentation, configuration, dependency declarations, and selected run metadata. Shared packages also contain regression support and a SuperNEMO Transformer module needed by its model registry; their presence does not add those tasks to the classic-model results.

Raw detector data, trained checkpoint tensors, prediction archives, binary dataset caches, and large logs are excluded from this source collection. The bundle's separate `event_data/` companion contains the standardized prediction inputs for reproducing the final metrics without training or loading detector data.

## Final metrics versus native training utilities

Historical evaluation code is intentionally preserved without fixes. Some original training or inference commands also write historical metrics. Those metric outputs are not the authority for the final paper. In particular, old EXO/SuperNEMO adaptive energy bins, old NEXT evaluation defaults, and MJD clipped/float32 energy evaluation must not be substituted for the final profiles.

From the **bundle root**, evaluate standardized events with:

```bash
python reproduction/evaluate.py --input /path/model.npz --profile strict600 --output outputs/model_metrics.json
python reproduction/evaluate.py --input /path/model.npz --profile overflow601 --output outputs/model_metrics.json
```

Use `strict600` for the final MJD/EXO-200 results and the frozen `overflow601` profile for the retained NEXT/SuperNEMO results. The bundle-level event-input manifest assigns the appropriate profile to each record. Follow the root README for the full batch and paper regeneration commands. Do not use a native `metrics.json` or rounded table as a metric recomputation input.

Score orientation matters: MJD's positive class is clean; its GINE model uses the original four-PSD-to-clean logit construction, while the other three selected models emit a binary clean logit. EXO-200's native positive class is background, so the standardized signal score is the negative native background logit and its label is complemented. NEXT uses the 0nubb direction. The trained SuperNEMO classification task is 2nu versus Bi214; the separate energy-only illustration is not its model-training task. Final standardization and physical-energy recovery belong to the bundle's audited evaluation adapters.

## Environment and paths

Use a Python 3.11 virtual environment; activate it before the examples below. See [`../../environments/classic/README.md`](../../environments/classic/README.md) from the bundle root, or `environments/classic/README.md`. Original dependency declarations are copied there. No environment or package was installed during collection.

All commands below run from the indicated **copied** project directory. Replace `/path/data/...` and `/path/checkpoints/...` with separately supplied local artifacts. Never run the examples from the original workspaces. Set `PYTHONDONTWRITEBYTECODE=1` when you want to leave snapshots free of bytecode caches.

The four main architecture IDs are:

| Paper model | Architecture ID |
|---|---|
| MV-CNN | `cnn_004_multiview_late_fusion` |
| GINE | `gnn_001_static_gine` |
| BiGRU | `seq_001_bigru` |
| PointMamba-lite | `ssm_001_pointmamba` |

Use the run-specific settings, not a different model's defaults. Exact model, representation and training settings are in `run_records/` and the checkpoint configuration index described below.

## NEXT

The canonical campaign uses 116,549 held-out events; the earlier file-split campaign uses 115,499. They are different event populations. The paper has 22 available canonical results plus earlier exploratory results, including the retained PointMamba-lite result. There is no canonical PointMamba-lite checkpoint/result to replace the earlier one.

Canonical training entry, with a fresh output directory:

```bash
cd code/classic/NEXT
mkdir -p outputs/reproduction
cp run_records/canonical_campaign/event_split.json outputs/reproduction/event_split.json
python 01_code/architectures/gnn_001_static_gine/train_classification.py \
  --data /path/data/NEXT \
  --manifest outputs/reproduction/event_split.json \
  --output-dir outputs/reproduction/gnn_001_static_gine --device cuda
```

The same entry pattern covers every directory under `01_code/architectures/`. Model and representation YAML files are beside the entries. Retained run summaries in `run_records/canonical_campaign/runs/` record effective settings and selected epochs. `run_energybench_campaign.py` schedules the canonical campaign; `--help` documents its explicit data and destination arguments.

The saved manifest contains relative HDF5 paths and event-ordinal ranges. The loader checks source inventory fingerprints; a changed inventory can cause it to regenerate a manifest. Always use a working copy as above and compare its event selection with the archived manifest before calling a rerun the same test split. The original manifest in `run_records/` must remain unchanged.

Canonical model restoration uses `01_code/architectures/workflow_runner.py` and `evalutaions_workflow/energybench/{data,training}.py`; `workflow_models.py` and `workflow_data.py` hold the architecture bridges. The canonical `best_model.pt` format differs from the earlier format-version-3 checkpoints. There is no dedicated general canonical checkpoint-only CLI in the original snapshot; `run_energybench_reevaluation.py` evaluates saved predictions and is not an inference command. Reproducing the final metrics from archived event predictions is the supported route without a new inference implementation.

For an earlier format-version-3 checkpoint, the original inference adapter is runnable independently:

```bash
PYTHONPATH=src python -m energybench predict \
  --adapter next_cnn.adapter:predict \
  --model /path/checkpoints/earlier_model.pt --data /path/data/NEXT \
  --output outputs/reproduction/earlier_predictions.npz \
  --adapter-arg split=test --adapter-arg device=cuda:0
```

The exact adapter recorded for each checkpoint is in `provenance/classic_source_audits/next_mjd/exploratory/manifest.json`. `next_cnn.adapter:predict` dispatches format-version-3 alternatives to `next_alt.adapter`. Do not run this on canonical checkpoints or merge the two test populations.

The earlier training implementation remains in `src/next_alt/training.py:main_for_architecture(architecture_id, config_path)`. An explicit call uses:

```bash
PYTHONPATH=src python -c 'from next_alt.training import main_for_architecture; main_for_architecture("ssm_001_pointmamba", "outputs/reproduction/pointmamba_config.yaml")'
```

Prepare that **new** YAML from `run_records/exploratory_pointmamba/config.snapshot.yaml`, changing only `data.root` and `output.checkpoint_dir`, `output.log_dir`, `output.plot_dir` to your data and fresh writable destinations. For other earlier models, use their extracted checkpoint settings. This training API requires CUDA. The old `nontransformer_campaign.py` is preserved as historical orchestration code, but its positional-config invocation is incompatible with the current canonical architecture CLI wrappers; do not use it as a replacement for this API call. No old script has been repaired or silently repurposed.

## MJD

Preprocessing, official shard discovery and loaders are in `mjdbench/data.py`; waveform-to-model representations are in the architecture modules and `mjdbench/waveform_points.py`. The official test partition has 390,000 events. Raw float64 `energy_label` values, source shard and row identity are required for final energy evaluation; native training-time float32/clipped energies are not a replacement.

```bash
cd code/classic/MJD
python architectures/gnn_001_static_gine/train_classification.py \
  --data-root /path/data/MJD \
  --output-dir outputs/reproduction/gnn_001_static_gine --device cuda
```

Replace the architecture ID with any of the four selected IDs. The saved effective configurations are under `run_records/classification/<architecture>/run_config.json`.

The original checkpoint-only CLI is `scripts/run_checkpoint_inference.py`. It expects `outputs/classification/<architecture>/{run_config.json,best.pt}`, reads the raw-data path from that JSON, and currently selects CUDA and AMP internally. To use it, stage a **new working copy** of the matching archived configuration and externally supplied checkpoint at those expected paths, change the working configuration's `data.data_root` for relocation, and keep the checkpoint hash unchanged:

```bash
python scripts/run_checkpoint_inference.py gnn_001_static_gine \
  --task classification --batch-size 16 \
  --output-dir outputs/reproduction/gnn_001_static_gine_inference
```

Do not pass `--max-events` for a paper result. This legacy CLI does not expose a CPU switch or per-run AMP override; reproduce the recorded inference precision/device if exact saved scores are required. Its output does not itself restore raw physical energy or event identity. The general package API is `mjdbench.training.evaluate_model` with an explicit loader/device/AMP configuration.

## EXO-200

The original package uses complete waveform events and a deterministic run-group split. Test membership must match the archived `split_runs`, `counts` and class counts; matching only event totals is insufficient. The selected models use the same 140,383-event held-out population. `exobench/data.py`, `config.py`, `waveform_points.py` and the four architecture modules contain the complete preprocessing and model code.

```bash
cd code/classic/EXO200
python architectures/gnn_001_static_gine/train_classification.py \
  --data-root /path/data/EXO-200 \
  --output-dir outputs/reproduction/gnn_001_static_gine --device cuda
```

The native workflow requires its output directory to remain inside the copied EXO200 project. Its saved effective configuration is in `run_records/classification/<architecture>/run_config.json`.

No standalone checkpoint-only CLI exists in this original snapshot. The original inference components are `exobench.data.prepare_dataset(data_config=DataConfig(**saved_data_config), ...)`, the selected architecture's `model.build_model()`, `model.load_state_dict(checkpoint["model_state_dict"], strict=True)`, and `exobench.training.evaluate_model(model, data.test_loader, device=..., output_dir=..., use_amp=..., amp_precision=...)`. Override only `saved_data_config["data_root"]` for relocation and verify the generated split against the archived run configuration before inference. The training workflow demonstrates this call sequence. `scripts/evaluate_final_models.py` aggregates existing artifacts; it does not rerun checkpoint inference. Use the supplied standardized event predictions for the tested final-metric reproduction path.

## SuperNEMO

This classifier's task is 2nu versus Bi214, with tracker topology as model input and calorimeter energy retained for conditional evaluation. The four HDF5 source files, block split rules and source metadata are specified by the original package and `run_records/split_manifest.json`. Binary event-offset caches are excluded and are regenerated by the loader.

The package imports the historical EnergyBench dependency shipped in `../NEXT/src`:

```bash
cd code/classic/SuperNEMO
PYTHONPATH=.:../NEXT/src python -m supernemobench.workflow \
  --task classification --model gnn_001_static_gine --mode train \
  --data-root /path/data/SuperNEMO --device cuda

PYTHONPATH=.:../NEXT/src python -m supernemobench.workflow \
  --task classification --model gnn_001_static_gine --mode test \
  --data-root /path/data/SuperNEMO --device cuda \
  --checkpoint /path/checkpoints/gnn_001_static_gine/best.pt
```

Outputs are placed under the copied project's `outputs/`. The native workflow may check recorded dataset provenance when loading a checkpoint; retain that check, and do not bypass it to force a different data inventory to load. Native evaluation/selection settings remain historical. Final energy-conditioned metrics use the bundle's frozen `overflow601` profile.

## Provenance and verification

From the bundle root:

- `provenance/classic_source_manifest.{json,csv}` maps each of the 331 copied files to its original path, byte count and SHA-256. Original license material is retained where present; the NEXT root `LICENSE` is included. No additional license grant is inferred for other source trees.
- `provenance/classic_run_index.json` maps 45 prediction/checkpoint provenances: 22 canonical NEXT, 11 earlier NEXT, four MJD, four EXO-200, and four SuperNEMO. Each checkpoint is identified by its exact original path and SHA-256; weights are not included.
- `provenance/classic_checkpoint_metadata.json` records configuration extracted from those exact checkpoints, excluding tensor weights, optimizer states and tree payloads. It resolves saved checkpoint settings separately from current architecture defaults.
- `provenance/classic_source_audits/` preserves the prior input audits, including event counts, test identity, score direction, raw-energy recovery and source prediction hashes. Its historical metric values and external paths are provenance, not final numerical authority.
- `validation/classic_validation.json` and `validation/classic_source_checks/` record source-hash checks, syntax parsing and CLI/import smoke checks. All 40 process checks passed. No training, detector-data loading or checkpoint inference was performed during collection. `environments/classic/validation_environment.json` records the frozen assembly interpreter and installed package versions. Re-running the checker writes the new environment to `validation/classic_validation_environment.json` so the frozen snapshot is not overwritten.

Rerun source-only verification with an activated compatible environment:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B code/classic/validate_snapshots.py
```

`collect_sources.py` and `extract_checkpoint_metadata.py` are new collection utilities, separate from the byte-identical source snapshots. They document how provenance was collected from the original local workspaces and are not required for portable final-metric reproduction. No historical metric code was corrected in this bundle.
