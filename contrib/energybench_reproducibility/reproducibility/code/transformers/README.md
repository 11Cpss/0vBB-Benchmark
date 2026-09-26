# Transformer reproduction

This directory contains the model, tokenization, data-loading and training code used by the located final-paper Transformer runs. Final metric computation is in [`../../benchmark/`](../../benchmark/), separate from model inference. Commands below run from `reproducibility/`.

| Dataset | Source | Located configurations |
|---|---|---:|
| EXO-200 | `detectors/exo200/` | 9: entity, region and summary × MLP, Fourier and RoPE |
| MJD | `detectors/mjd/` | 6: entity, region and summary × MLP and Fourier |
| NEXT | `detectors/next/` | 6: entity, region and summary × MLP and Fourier |
| SuperNEMO | `detectors/supernemo/` | 6: entity, region and summary × MLP and Fourier |

[`../../results/transformer_models.json`](../../results/transformer_models.json) selects each configuration and records the original checkpoint checksum. The 9 NEXT/MJD/SuperNEMO RoPE entries have no recovered matching implementation, checkpoint or saved predictions and are not covered by these runners. EXO's available RoPE implementation is not a substitute for them. Raw detector releases, trained checkpoints, token caches and SuperNEMO event-offset indexes are external inputs.

## Reproduce saved-prediction metrics

For MJD and EXO-200, use the public benchmark's `strict600` profile with the supplied standardized per-event inputs; see [`../../benchmark/README.md`](../../benchmark/README.md). EXO's native label 0 denotes the positive single-cluster class: the standardization is `label = 1 - native_label`, `score = -native_logit`, and physical `Rotated_energy` in keV. MJD's positive class passes all four PSD conditions; use its native logit and original float64 physical energy. The waveform exporter now carries original float64 `energy_keV` and unique physical `event_id` values from the same inference batches; model inputs and native score direction are unchanged.

The six NEXT and six SuperNEMO Transformer rows use their recorded paper profiles in `benchmark/published_profiles/`. Replay all twelve directly from the optional event-data companion:

```bash
python -B code/transformers/replay_paper_metrics.py \
  --data-root /path/to/event_data \
  --output-dir outputs/transformer_metrics
```

The input manifest records hashes, evaluation settings and full-precision expected results. Every replay checks all three metrics against those expected values. NEXT native physical energy is MeV and label 1 is 0nubb. SuperNEMO uses `E1 + E2` in keV and label 1 is 2nu; its classification task is 2nu versus Bi214. The separate 0nu energy illustration is not this task. These recorded profiles must not be relabelled as `strict600`: SuperNEMO uses six matching energy quantiles and eight within-class energy quantiles for I.

## Train or export predictions

Install [`../../environments/transformers/requirements.txt`](../../environments/transformers/requirements.txt) in an isolated environment. Each runner requires an explicit data directory, a new/empty output directory and, for `--mode test`, the checkpoint matching the selected row's checksum. `--describe` prints the selected configuration without loading data. Supported model keys are `entity_mlp`, `entity_fourier`, `region_mlp`, `region_fourier`, `summary_mlp`, `summary_fourier`, plus EXO-only `entity_rope`, `region_rope`, `summary_rope`.

### EXO-200 and MJD

```bash
python -B code/transformers/waveform_run.py --dataset EXO-200 --model-key entity_rope --describe
python -B code/transformers/waveform_run.py --dataset EXO-200 --model-key entity_rope \
  --mode train --data-root /path/EXO-200 --output-dir outputs/exo_entity_rope
python -B code/transformers/waveform_run.py --dataset MJD --model-key region_fourier \
  --mode test --data-root /path/MJD --checkpoint /path/original/best.pt \
  --output-dir outputs/mjd_region_fourier
```

The runners check recorded split counts and model parameter counts. EXO also validates the packaged split contract. A different raw release or test partition does not reproduce the paper.

Each `predictions.npz` contains native `score` and `label` (also retained as `prediction` and `target`), raw float64 `energy_keV`, `event_id`, `source_file` and `row`. EXO IDs include the source filename and native event number; MJD IDs are `source_file:row`, so repeated native IDs in different shards remain distinct. No energy clipping or float32 evaluation-energy conversion is applied. Standardize EXO's positive-class direction once, then evaluate; MJD already has the required direction:

```bash
python -B code/input_preparation/standardize.py \
  --input outputs/exo_entity_rope/predictions.npz \
  --output outputs/exo_entity_rope/energybench_input.npz \
  --energy-unit keV --positive-label 0 --score-label 1
python -B benchmark/evaluate.py \
  --input outputs/exo_entity_rope/energybench_input.npz --profile strict600 \
  --output outputs/exo_entity_rope/energybench.json
python -B benchmark/evaluate.py \
  --input outputs/mjd_region_fourier/predictions.npz --profile strict600 \
  --output outputs/mjd_region_fourier/energybench.json
```


### NEXT

The four entity/region runs use cached tokens; the two summary runs stream raw input. Supply a working split manifest that preserves the exact event lists in `configs/next/event_split.json`. The cache verifies raw-file identity, the split manifest and tokenization settings. Both streaming and cached entry points reject incompatible existing manifests; they never regenerate or overwrite those event assignments. The streaming loader creates a split only when the requested manifest does not exist, for a new experiment.

```bash
python -B code/transformers/detectors/next/scripts/build_token_cache.py \
  --data-root /path/NEXT --split-manifest /path/next_work/event_split.json \
  --cache-root outputs/next_cache --tokenization sampled_hits \
  --max-tokens 512 --voxel-size 15 --coordinate-scale 1000 --seed 42
python -B code/transformers/next_run.py --model-key entity_mlp --mode train \
  --data-root /path/NEXT --split-manifest /path/next_work/event_split.json \
  --token-cache /path/cache-directory-reported-by-builder --output-dir outputs/next_entity_mlp
python -B code/transformers/next_run.py --model-key summary_fourier --mode test \
  --data-root /path/NEXT --split-manifest /path/next_work/event_split.json \
  --checkpoint /path/original/best_model.pt --output-dir outputs/next_summary_fourier
```

Use `--tokenization voxel` for the region cache. The cache builder preserves its original filename/source checks, so relocated data need a correctly relocated manifest; never change event assignments to bypass validation. The cached run configurations use the recorded training defaults with eight loader workers. The summary configurations contain their serialized settings.

### SuperNEMO

```bash
python -B code/transformers/supernemo_run.py --model-key entity_mlp --describe
python -B code/transformers/supernemo_run.py --model-key entity_mlp --mode train \
  --data-root /path/SuperNEMO --manifest-path /path/manifests/split_manifest.json \
  --output-dir outputs/supernemo_entity_mlp
python -B code/transformers/supernemo_run.py --model-key entity_mlp --mode test \
  --data-root /path/SuperNEMO --manifest-path /path/manifests/split_manifest.json \
  --checkpoint /path/original/best.pt --output-dir outputs/supernemo_entity_mlp_test
```

The supplied manifest must preserve the packaged event slices, settings and counts, and its adjacent event-offset indexes must exist. The loader checks source files and rejects changed identities. These models use SuperNEMO-specific tokenization, including 128-token entity/region inputs and 60 mm voxels. Training selects checkpoints using the paper's matched AUC on validation events through `supernemobench/evaluation.py`; this small adapter only selects checkpoints and exports native predictions. Final test metrics use the public benchmark entry points.

The package was checked with all twelve saved-prediction replays, Python import/model-construction checks and runner help/configuration checks. Cross-shard waveform fixtures also verify unchanged model inputs/native predictions, exact physical energy around 5 and 3000 keV boundaries, and the EXO standardization direction. Full retraining and checkpoint inference were not run as part of publication cleanup. Reproducing the recorded point estimates requires the recorded checkpoints and test events; retraining is not guaranteed to reproduce those exact numbers.
