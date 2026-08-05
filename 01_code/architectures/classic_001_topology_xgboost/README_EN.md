# CLASSIC-001: NEXT Topology Features with XGBoost

[中文](README.md) | [English](README_EN.md)

## 1. Identity and hypothesis

| Item | Definition |
|---|---|
| `architecture_id` | `classic_001_topology_xgboost` |
| Model | `TopologyBoostedTreeClassifier` |
| Feature class | `next_alt.models.classic_topology.TopologyFeatureExtractor` |
| Estimator class | `next_alt.models.classic_topology.TopologyBoostedTreeClassifier` |
| `input_kind` | `topology` (materialized from point-compatible tensors) |
| Task/output | `0nubb=1`, `Bi214=0`; one uncalibrated margin/logit per event |
| Frozen backend | **`xgboost`** (environment version 3.0.5) |
| Parameter count | Not fixed; tree/node counts are recorded after fitting |

The hypothesis is that physics-motivated endpoint blobs, track length/tortuosity, connectivity,
and energy distribution form a strong low-dimensional baseline. This checks whether deep-model
gains require learned representations. No Transformer, attention, total event energy, or absolute
detector position is used.

## 2. Raw data, labels, and file split

Contiguous rows sharing an `event_id` are read from HDF5 `/MC/hits/table`, using `x/y/z/energy`.
`0nubb_part_*` maps to 1 and `Bi_part_*` to 0. A stable hash of the complete relative file path
with seed 42 assigns whole files at fractions 0.8/0.1/0.1. The standalone runner calls training
loaders only, materialising train and validation with at most 100 files per class. Validation AUC
sets early stopping and the best tree limit. No code path asks for the test split.

## 3. Voxel representation and exact tensors

$$
E=\sum_i e_i,\quad \mathbf c=\sum_i e_i\mathbf r_i/E,\quad
\mathbf q_i=\lfloor(\mathbf r_i-\mathbf c)/(15\,\mathrm{mm})\rfloor.
$$

Hits in a cell are merged; `(q+0.5)*15 mm` centers are energy-recentered. At most 512 voxels are
retained by energy with deterministic coordinate tie breakers; fractions are not renormalized.

| Key | dtype / shape | Meaning |
|---|---|---|
| `coords` | float32 `(B,N,3)` | centered voxel center divided by 1000 mm |
| `features` | float32 `(B,N,2)` | energy fraction and log hit count |
| `mask` | bool `(B,N)` | point versus padding |
| `label` | float32 `(B,)` | trainer-only target |
| topology matrix | float32 `(events,32)` | fixed features |
| prediction | float64 `(events,)` | XGBoost output-margin logit |

The extractor takes at most 192 highest-energy voxels and recenters that subset. Total energy is
used only as the fraction denominator and is not a matrix column.

## 4. The 32 ordered features

- Size/energy: log voxel count, retained fraction, largest/top-2/top-5 fractions, normalized energy
  entropy.
- Hit multiplicity: mean, standard deviation, and maximum of `log1p(hit_count)`.
- Shape: x/y/z extents; square roots of the three energy-weighted covariance eigenvalues;
  linearity `(λ1-λ2)/λ1`, planarity `(λ2-λ3)/λ1`, and sphericity `λ3/λ1`.
- Radius: weighted mean/std and maximum radius; first-principal-axis length.
- Endpoints: energy-fraction sums in 30 mm balls at the two principal-axis extremes, their minimum,
  and normalized asymmetry.
- Connectivity: component count and mean degree in the 26 mm radius graph.
- Exact Euclidean MST: degree-at-least-three fraction, total edge length, maximum edge, and
  total-length/principal-length tortuosity.

All lengths are in coordinates already divided by 1000 mm. Complete-graph MST statistics use Prim.
The largest-magnitude Cartesian component of the first principal axis is forced positive to remove
the eigenvector sign ambiguity before ordering the two endpoints.

## 5. Learner and objective

The formal backend uses XGBoost's histogram tree method:

$$
\hat y(x)=b+\eta\sum_{t=1}^{T}f_t(x),\qquad
\mathcal L=\sum_i\mathrm{BCEWithLogits}(y_i,\hat y_i)+\Omega(f_t).
$$

Each logistic gradient/Hessian round fits a depth-at-most-4 tree. Validation `logloss` and `auc`
are monitored; 12 rounds without AUC improvement stop training. `best.json` embeds the same raw UBJ
Booster with `prediction_tree_limit=best_iteration+1`; `last.json` uses the completed-round limit.
Output margin, not probability, preserves the `(B,)` logit contract.

The module also contains an explicitly named `numpy_hist_gbdt` emergency fallback based on quantile
candidates and Newton leaves. It lacks XGBoost's sparsity-aware algorithm, weighted quantile sketch,
parallel kernels, and full regularisation and is **not an XGBoost reproduction**. Formal YAML sets
`backend: xgboost`, so a missing dependency fails instead of silently switching. Any explicitly
requested fallback records its true backend in checkpoints and the run summary.

## 6. Complexity and memory

Feature extraction is `O(M²)` time/memory per event for `M≤192`. Histogram boosting scales roughly
with rounds, depth, rows, and sampled features; its matrix has 32 columns and is small compared with
dense detector images. Tree/node count is data-dependent, so `parameter_count` is null and
`tree_node_count` is written after fitting. Formal `tree_method=hist` is CPU because no XGBoost CUDA
device is selected; the job remains in the serial tmux campaign for deterministic ordering.

## 7. Frozen YAML parameters

- Data: root `/home/klz/Data/zeronu_benchmark/NEXT`; 100 files/class; seed 42,
  `[0.8,0.1,0.1]`; workers 0; balanced true; buffer 512.
- Representation: 15 mm bin, 1000 mm scale, 512 shared points.
- Extractor: 192 points; connectivity/blob radius 0.026/0.030 scaled units.
- Estimator: backend xgboost; 500 rounds; depth 4; eta .04; 32 bins;
  `min_samples_leaf=24` mapped to XGBoost `min_child_weight=24`; L2 1; gamma 0; row/column sample
  .85/.90; seed 42; early stopping 12.
- Loader batch 128 controls feature materialisation only.

`training.epochs=50`, lr .001, decay, clipping, and AMP are compatibility fields required by the
shared config schema; tree learning is controlled only by `model.estimator`.

## 8. Boundary relative to the papers

The NEXT paper uses reconstructed tracks and endpoint energy. Here, PCA, a radius graph, and
complete-graph MST produce engineered voxel features; a blob is a 30 mm ball around a principal-axis
extreme and uses fractions, not absolute energy. The learner is XGBoost histogram boosting, but the
data/features/hyperparameters differ from Chen and Guestrin's benchmarks. The top-512/top-192 caps,
axis extents, and radii are project-specific. No paper performance is claimed as reproduced.

## 9. Run, artifacts, and recovery

```bash
cd /home/wenyu/summer
source .venv/bin/activate
python 01_code/architectures/classic_001_topology_xgboost/train_classification.py CONFIG_SNAPSHOT
```

The detached `next-nontransformer-v2-<RUN_ID>` session runs this in the serial `gpu-queue`; monitor
is read-only. Exact artifacts are:

```text
02_models/checkpoints/<RUN_ID>/classic_001_topology_xgboost/attempt_NNN/{best.json,last.json}
03_training_runs/campaigns/<RUN_ID>/classic_001_topology_xgboost/attempt_NNN/
  stdout.log  config.snapshot.yaml  epochs.csv  history.json  history.png  run_summary.json
```

Checkpoint JSON includes backend, ordered feature names, extractor/split/file provenance, and a
base64 UBJ Booster. Stop with `C-c`; queue resume skips DONE and creates a fresh attempt for FAILED,
re-materialising features and retraining from scratch without overwriting history.

## 10. Limitations and result placeholder

Axis extents are not rotation invariant; fixed radii depend on 15 mm voxels; top-192 can remove long
low-energy tails; MST is outlier-sensitive; XGBoost `min_child_weight` is a Hessian sum, not a literal
sample count; embedded trusted Booster JSON can be large. No test split, prediction table, test metric,
or leaderboard is part of this stage.

This pre-campaign placeholder is superseded by the appended completed result, which records backend/version, completed/best round, best validation
AUC/loss, duration, tree/node count, artifacts, early stop, and retries after the campaign. Do not add
test metrics.

## 11. References

1. Tianqi Chen, Carlos Guestrin, “XGBoost: A Scalable Tree Boosting System,” *KDD 2016*,
   pp. 785–794, DOI `10.1145/2939672.2939785`, arXiv:1603.02754.
   [DOI](https://doi.org/10.1145/2939672.2939785) · [arXiv](https://arxiv.org/abs/1603.02754)
2. NEXT Collaboration, P. Ferrario et al., “First proof of topological signature in the high
   pressure xenon gas TPC with electroluminescence amplification for the NEXT experiment,”
   *JHEP* 2016, 104, DOI `10.1007/JHEP01(2016)104`, arXiv:1507.05902.
   [Journal](https://link.springer.com/article/10.1007/JHEP01%282016%29104) ·
   [arXiv](https://arxiv.org/abs/1507.05902)


<!-- campaign-result:20260803_200356:start -->
## Campaign `20260803_200356` training result

This section is written after successful serial training. It contains train/validation information only; the test split was not read in this stage.

| Item | Observed value |
|---|---|
| Status / attempt | `DONE` / `attempt_001` |
| Backend | `xgboost` |
| Trainable parameters | N/A |
| Trees / tree nodes | 491 / 9769 |
| Completed / best epoch | 500 / 491 |
| Best validation AUC | **0.948048** |
| Best validation loss | 0.289180 |
| Training time | 00:01:23 |
| Early stopped | `false` |
| Failed-attempt retry | `no` |
| Python / framework | `3.11.15` / `3.0.5` |
| Device | `not used (XGBoost hist on CPU)` |
| Best / last checkpoint | `/home/wenyu/summer/02_models/checkpoints/20260803_200356/classic_001_topology_xgboost/attempt_001/best.json` / `/home/wenyu/summer/02_models/checkpoints/20260803_200356/classic_001_topology_xgboost/attempt_001/last.json` |
| Training history | `/home/wenyu/summer/03_training_runs/campaigns/20260803_200356/classic_001_topology_xgboost/attempt_001/history.json` |
<!-- campaign-result:20260803_200356:end -->
