# SuperNEMO Transformer benchmark

This detector benchmark compares six frozen Transformer configurations for
the binary task `2nubb -> 1` versus `Bi214 -> 0`. It is classification-only;
SuperNEMO energy regression is outside this benchmark.

The repository owns the Transformer tokenizers, positional encodings, model,
tests, notebooks, and the thin `supernemobench` adapter. The lab environment
owns the HDF5 data, Wing's split manifest and event indexes, Wing's specialized
`architectures` package, EnergyBench, all generated run JSON, predictions,
metrics, and checkpoints. None of those external or generated artifacts should
be committed.

## Frozen runs

| ID | Tokens | Position encoding |
|---|---|---|
| `transformer_001_entity_coordinate_mlp` | tracker-hit entities, cap 512 | coordinate MLP |
| `transformer_002_patch_88mm_coordinate_mlp` | occupied 88 mm detector patches, cap 512 | coordinate MLP |
| `transformer_003_patch_88mm_fourier_xyz` | occupied 88 mm detector patches, cap 512 | Fourier XYZ |
| `transformer_004_entity_fourier_xyz` | tracker-hit entities, cap 512 | Fourier XYZ |
| `transformer_005_summary_16_coordinate_mlp` | at most 16 balanced Morton groups | coordinate MLP |
| `transformer_006_summary_16_fourier_xyz` | at most 16 balanced Morton groups | Fourier XYZ |

Every model uses `d_model=64`, four heads, two encoder layers, feed-forward
width 256, dropout 0.1, and masked-mean pooling. Fourier runs use six
frequencies. Coordinates are grouped in absolute detector space, centered on
the original event hit centroid, then divided by 1000 mm.

The frozen SuperNEMO training configuration uses batch size 64, 50 maximum
epochs, AdamW at `5e-4`, cosine scheduling, gradient clipping at 1.0, AMP,
early-stopping patience 5, seed 42, and eight data-loader workers. The worker
count is an execution setting; every run records it in `run_config.json` and
its checkpoints. A run is complete only when `training_complete.json` exists;
ordinary epoch checkpoints are resumable but are not completion markers.

## Lab setup

Use your own Python environment and install this checkout editable so
`supernemobench` and `supernemo_transformer` resolve locally:

```bash
cd /path/to/Transformer_Approach
python -m pip install -e . --no-deps
```

The canonical external paths are:

```text
data:       /home/klz/Data/zeronu_benchmark/SuperNEMO
split:      /home/wenyu/SuperNEMO/data/manifests/split_manifest.json
evaluation: /home/wenyu/SuperNEMO/evaluation/supernemo_2nu_vs_bi214.json
EnergyBench package: <repository>/exo200_detector/frozen_energybench/energybench
```

Set the evaluation manifest explicitly and pass the split manifest on every
workflow call:

```bash
export SUPERNEMO_EVALUATION_MANIFEST=/home/wenyu/SuperNEMO/evaluation/supernemo_2nu_vs_bi214.json
export SUPERNEMO_ENERGYBENCH_PACKAGE=/path/to/Transformer_Approach/exo200_detector/frozen_energybench/energybench

python -m supernemobench.workflow \
  --task classification \
  --model transformer_001_entity_coordinate_mlp \
  --mode train \
  --data-root /home/klz/Data/zeronu_benchmark/SuperNEMO \
  --manifest-path /home/wenyu/SuperNEMO/data/manifests/split_manifest.json

python -m supernemobench.workflow \
  --task classification \
  --model transformer_001_entity_coordinate_mlp \
  --mode test \
  --data-root /home/klz/Data/zeronu_benchmark/SuperNEMO \
  --manifest-path /home/wenyu/SuperNEMO/data/manifests/split_manifest.json
```

Repeat with each frozen ID, or use the clean training notebook in
`notebooks/supernemo_transformer_train.ipynb`. The results notebook refuses to
combine runs whose split, evaluation protocol, or prediction-order
fingerprints differ.

The adapter continues to use the benchmark's proportional source schedule,
BCE-with-logits loss, AdamW optimizer, cosine scheduler, early stopping,
schema-v2 checkpoints, and validation energy-matched AUC selection. The local
training copy adds only atomic completion bookkeeping and a device-safe RNG
resume repair; neither changes uninterrupted model optimization. The held-out
test split is used only for final EnergyBench evaluation. A higher model score
means more signal-like (`2nu`); the matching condition is unclipped `E1 + E2`
in keV.

For baseline-paper comparisons, all SuperNEMO models must reuse the same split
and evaluation-manifest fingerprints. The split, label mapping, energy
condition, optimizer, scheduler, checkpoint-selection metric, and stopping
rule are scientific protocol choices. Data-loader workers and hardware are
execution choices, but still must be reported. Do not claim exact equivalence
to an external specialized-model run until its recorded fingerprints and
training configuration have been compared with this workflow.

## Tests

From the repository root:

```bash
python -m unittest discover -s supernemo_detector/supernemo_transformer/tests -v
```

The tests cover token invariants, the six forward/backward configurations,
shared synthetic HDF5 splits and metadata, provenance rejection, and a CPU
epoch through Wing's actual training function while mocking only the external
EnergyBench score.

Generated files belong under `supernemo_detector/outputs/`, `results/`, or
`analysis/`; these paths and generated JSON/CSV/model artifacts are ignored.
Do not add external manifest copies to the repository.
