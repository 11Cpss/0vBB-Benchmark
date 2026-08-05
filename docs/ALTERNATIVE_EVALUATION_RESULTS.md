# Alternative Architecture Evaluation Results

## Evaluation contract

All ten alternative classification architectures were evaluated on 2026-08-03
with their `classification_best.pt` checkpoints and the repository's existing
EnergyBench NEXT workflow.

- Dataset: `zeronu-benchmark-next`, version
  `zenodo-18927784-v1.0-tarset-0b57f1a2c33c`
- Split: complete `test` split, 1,490 files and 115,499 events
- Manifest: `manifests/next_0nubb_vs_bi214.yaml`
- Mode: strict; no mixed-data comparison and no file limit
- Validation: all rows have identical evaluation, protocol, and code
  fingerprints; every row has zero warnings and zero errors

The formal classification ranking is:

| Rank | Architecture | Family | Matched AUC | Inclusive AUC | Energy independence |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | `gnn_001_static_gine` | static-graph GINE | 0.981035 | 0.981224 | 0.977111 |
| 2 | `gnn_002_particlenet_edgeconv` | dynamic-graph EdgeConv | 0.970363 | 0.970754 | 0.979700 |
| 3 | `cnn_004_multiview_late_fusion` | multi-view 2D CNN | 0.955819 | 0.955936 | 0.978224 |
| 4 | `gnn_004_gravnet` | learned-neighborhood GNN | 0.955680 | 0.955733 | 0.977351 |
| 5 | `point_002_pointnetpp` | hierarchical point cloud | 0.953342 | 0.953552 | 0.977201 |
| 6 | `cnn_005_multiscale_projection` | multiscale 2D CNN | 0.947787 | 0.948255 | 0.980444 |
| 7 | `cnn_006_dense_3d_resnet` | dense 3D CNN | 0.928210 | 0.928082 | 0.977320 |
| 8 | `point_001_deepsets` | permutation-invariant point set | 0.920475 | 0.920727 | 0.974339 |
| 9 | `hybrid_001_cnn_gnn` | projection CNN + graph | 0.912542 | 0.912792 | 0.976863 |
| 10 | `gnn_003_egnn` | equivariant graph network | 0.886558 | 0.887098 | 0.977462 |

Energy regression is not applicable for these classification-only checkpoints;
the corresponding NA status is expected and is not an evaluation failure.

## Main observations

- Static GINE is the strongest classifier, leading ParticleNet EdgeConv by
  0.010673 matched-AUC points.
- Graph models are not uniformly superior: GINE and ParticleNet lead, GravNet
  is competitive with the strongest CNN, while this EGNN configuration ranks
  last.
- Multi-view late fusion and GravNet are effectively tied on matched AUC; their
  absolute difference is approximately 0.000139.
- The multiscale projection CNN has the highest energy-independence score,
  while GINE has the best discrimination.
- Dense 3D convolution is substantially slower at inference and does not beat
  the lighter projection, graph, or hierarchical point-cloud alternatives in
  this configuration.

## Artifacts

The authoritative formal leaderboard is:

```text
04_evaluations/NEXTALT_all_models_comparison/leaderboard.csv
```

Each model has its own directory named
`04_evaluations/NEXTALT_<architecture_id>_test/`, containing:

- `predictions_test.npz`
- `evaluation_test/results.csv`
- `evaluation_test/.energybench/metrics.json`
- `evaluation_test/energy_matched_roc.png`
- `evaluation_test/score_energy_dependence.png`

The complete queue log is:

```text
04_evaluations/logs/alternative_evaluation_queue_20260803_100924.log
```

The reproducible queue entry point is
`01_code/architectures/run_alternative_evaluation_queue.sh`.
