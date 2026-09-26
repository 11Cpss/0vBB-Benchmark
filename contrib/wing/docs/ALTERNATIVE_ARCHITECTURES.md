# NEXT alternative classification architectures

This document describes the independently implemented model suite under
`01_code/architectures/`.  These programs do not reproduce or replace
CNN-001, CNN-002, or CNN-003.  They share only the dataset contract, the
file-level split, output locations, and EnergyBench inference interface.

All models solve the same binary task:

- positive class (`1`): `0nubb`;
- negative class (`0`): `Bi214`;
- output: one uncalibrated signal logit per event.

Energy regression is deliberately out of scope for this suite.

## Included models

| ID | Model card | Family | Event representation | Main idea |
|---|---|---|---|---|
| `cnn_004_multiview_late_fusion` | [English](../01_code/architectures/cnn_004_multiview_late_fusion/README_EN.md) · [中文](../01_code/architectures/cnn_004_multiview_late_fusion/README.md) | 2D CNN | normalized XY/XZ/YZ projections | encode each physical view separately with a shared residual encoder, then fuse the three embeddings |
| `cnn_005_multiscale_projection` | [English](../01_code/architectures/cnn_005_multiscale_projection/README_EN.md) · [中文](../01_code/architectures/cnn_005_multiscale_projection/README.md) | 2D CNN | global 30 mm and centered 15 mm projections | combine full-track context with a finer view of blobs and forks |
| `cnn_006_dense_3d_resnet` | [English](../01_code/architectures/cnn_006_dense_3d_resnet/README_EN.md) · [中文](../01_code/architectures/cnn_006_dense_3d_resnet/README.md) | 3D CNN | centered 15 mm, 96-cube volume | retain three-dimensional geometry with residual 3D convolutions |
| `point_001_deepsets` | [English](../01_code/architectures/point_001_deepsets/README_EN.md) · [中文](../01_code/architectures/point_001_deepsets/README.md) | set network | centered 15 mm voxels | independent point encoding followed by permutation-invariant pooling |
| `point_002_pointnetpp` | [English](../01_code/architectures/point_002_pointnetpp/README_EN.md) · [中文](../01_code/architectures/point_002_pointnetpp/README.md) | hierarchical point network | centered 15 mm voxels | aggregate local neighborhoods at progressively coarser scales |
| `gnn_001_static_gine` | [English](../01_code/architectures/gnn_001_static_gine/README_EN.md) · [中文](../01_code/architectures/gnn_001_static_gine/README.md) | graph network | fixed geometric k-nearest-neighbor graph | residual edge-aware message passing on a fixed event graph |
| `gnn_002_particlenet_edgeconv` | [English](../01_code/architectures/gnn_002_particlenet_edgeconv/README_EN.md) · [中文](../01_code/architectures/gnn_002_particlenet_edgeconv/README.md) | dynamic graph network | dynamic feature-space graph | rebuild neighbors after every EdgeConv stage, following the ParticleNet idea |
| `gnn_003_egnn` | [English](../01_code/architectures/gnn_003_egnn/README_EN.md) · [中文](../01_code/architectures/gnn_003_egnn/README.md) | equivariant graph network | distance graph plus coordinates | use coordinate-aware messages with E(n)-equivariant coordinate updates |
| `gnn_004_gravnet` | [English](../01_code/architectures/gnn_004_gravnet/README_EN.md) · [中文](../01_code/architectures/gnn_004_gravnet/README.md) | learned-space graph network | learned-coordinate k-nearest-neighbor graph | learn the space in which voxel neighborhoods should be formed |
| `hybrid_001_cnn_gnn` | [English](../01_code/architectures/hybrid_001_cnn_gnn/README_EN.md) · [中文](../01_code/architectures/hybrid_001_cnn_gnn/README.md) | hybrid | global projections plus centered voxel graph | fuse a multi-view CNN embedding with an EdgeConv event embedding |

Each model has equivalent English and Chinese model cards. They record the
exact preprocessing, layer dimensions, configuration, implementation files,
method references, training artifacts, formal test result, and known
limitations.

Method references:

- Deep Sets: <https://arxiv.org/abs/1703.06114>
- PointNet++: <https://arxiv.org/abs/1706.02413>
- GIN/GINE family: <https://arxiv.org/abs/1810.00826>
- Dynamic Graph CNN / EdgeConv: <https://arxiv.org/abs/1801.07829>
- ParticleNet: <https://arxiv.org/abs/1902.08570>
- EGNN: <https://arxiv.org/abs/2102.09844>
- GravNet: <https://arxiv.org/abs/1902.07987>

## Shared representations

Point and graph models first merge the public 3 mm hits into 15 mm voxels.
Coordinates are translated by the event's energy-weighted centroid and divided
by 1000 mm.  The two node features are the voxel energy fraction and the
logarithm of its merged-hit count.  Total event energy and absolute detector
position are not model inputs, although total energy remains in the exported
prediction table as `energy_condition` for matched evaluation.

The multi-view CNN uses complete, energy-normalized 30 mm projections.  The
multi-scale CNN adds a centered 15 mm crop.  The dense 3D model uses a centered
96 x 96 x 96 grid at 15 mm and records the retained-energy coverage.

All representation settings are written into the checkpoint.  Inference never
guesses preprocessing from a model name.

## Configuration and outputs

Every model directory contains:

```text
config.yaml              complete default training configuration
train_classification.py  executable training entry point
README.md                Chinese architecture-specific model card
README_EN.md             equivalent English model card
```

The implementations themselves live in `src/next_alt/models/`; keeping them in
one importable package gives training and inference exactly the same classes.

The shared runner writes:

```text
02_models/checkpoints/NEXTALT_<architecture>_classification_best.pt
02_models/checkpoints/NEXTALT_<architecture>_classification_last.pt
03_training_runs/logs/NEXTALT_<architecture>_classification_epochs.csv
03_training_runs/logs/NEXTALT_<architecture>_classification_history.json
03_training_runs/history_plots/NEXTALT_<architecture>_classification_history.png
```

Existing files are not overwritten.  With the default `allow_overwrite: false`,
the runner refuses to start if any target artifact already exists; use a copied
YAML with separate output directories for a second independent run.

## Direct training

Activate the one project environment and run the model's full entry point:

```bash
cd /home/wenyu/summer
source .venv/bin/activate
python 01_code/architectures/gnn_002_particlenet_edgeconv/train_classification.py
```

There is no smoke-training flag.  Change an experiment by copying and editing
its YAML configuration, then pass the configuration path as documented in the
model README.

Long-running sequential training is described in
[`TMUX_TRAINING.md`](TMUX_TRAINING.md).

## Evaluation with the existing program

Format-version-3 checkpoints are supported by the same one-command workflow as
legacy NEXT CNN checkpoints:

```bash
cd /home/wenyu/summer
source .venv/bin/activate

energybench next \
  02_models/checkpoints/NEXTALT_gnn_002_particlenet_edgeconv_classification_best.pt \
  --device cuda:0 \
  --output-dir 04_evaluations/NEXTALT_gnn_002_particlenet_edgeconv_test_rerun
```

The adapter exports the existing canonical classification columns, so no new
manifest or scoring implementation is needed.  Do not pass
`--max-files-per-class` for a complete test evaluation.

### Evaluate all alternative architectures

The repository includes a serial full-test queue for all ten best checkpoints:

```bash
cd /home/wenyu/summer
tmux new-session -d -s next-alt-evaluations -n evaluation \
  "cd /home/wenyu/summer && bash 01_code/architectures/run_alternative_evaluation_queue.sh"
```

The queue uses the complete `test` split, strict evaluation, a unique model ID,
and a conservative architecture-specific batch size.  It attempts every model
even if one fails, and only creates the formal comparison when all ten strict
evaluations succeed.  It never overwrites an existing evaluation directory.

Monitor the running queue with:

```bash
tmux attach -t next-alt-evaluations
tail -f "$(ls -1t 04_evaluations/logs/alternative_evaluation_queue_*.log | head -n 1)"
```

Per-model outputs are written to
`04_evaluations/NEXTALT_<architecture_id>_test/`.  After a fully successful
queue, the unified leaderboard is
`04_evaluations/NEXTALT_all_models_comparison/leaderboard.csv`.

The completed formal ranking and interpretation are recorded in
[`ALTERNATIVE_EVALUATION_RESULTS.md`](ALTERNATIVE_EVALUATION_RESULTS.md).
