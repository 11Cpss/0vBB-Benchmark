# EnergyBench Evaluation Standard, Table Specification, and Output Interpretation

> [Chinese version](EVALUATION_STANDARD.md)  
> Document version: 1.0  
> Corresponding implementation: EnergyBench 0.1.0  
> Audit baseline: the `/home/wenyu/summer` workspace as of 2026-08-01

## Abstract

This standard specifies the data contract, statistics, quality gates, missing-value semantics, table fields, and figure interpretation used by EnergyBench to evaluate event-level model outputs for classification, energy regression, and classification score–energy dependence. The formal primary classification metric is the macro-average **energy-matched AUC** over the complete fixed set of class pairs; the formal primary energy-regression metric is the project-specific, versioned composite score **ERS-v1**. Inclusive AUC, the energy-independence score, and all other error or distribution metrics are required diagnostics, but they neither replace the primary metrics nor combine into a single overall score.

The evaluator accepts only an event-level prediction table and does not inspect model architecture during statistical evaluation. Formal comparisons require identical event truth, the same protocol, the same evaluation code, successful strict-mode evaluation, and a unique model identifier. If common energy support, sample-size, coverage, or complete class-pair requirements cannot be met, matched AUC must be recorded as NA; inclusive AUC must not be used as a substitute.

## 1. Scope and Metric Hierarchy

This standard applies to ranking evaluations for binary classification or multiclass tasks decomposed into fixed signal/background pairs, as well as to scalar energy regression. In the current NEXT task, `0nubb` is signal, `Bi214` is background, and the energy unit is MeV. The numerical values of the classification matching energy `energy_condition` and the regression truth `energy_target` may be identical, but their statistical roles and provenance must be declared separately.

### 1.1 Notation

| Symbol | Definition |
|---|---|
| $i=1,\ldots,n$ | Event index |
| $Y_i\in\{0,1\}$ | Background/signal label, with 1 denoting the positive class |
| $C_i$ | Physical category, such as `0nubb` or `Bi214` |
| $s_i$ | Raw classification score from the model |
| $S_i$ | Oriented score; larger values are more signal-like |
| $E_i$ | Classification matching condition `energy_condition` |
| $T_i$ | Regression target `energy_target` |
| $\widehat T_i$ | Regression prediction `energy_pred` |
| $a_i\ge0$ | Base sample weight specified by the manifest; defaults to 1 |
| $\Delta_i=\widehat T_i-T_i$ | Signed residual |
| $r_i=\Delta_i/\max(|T_i|,\epsilon)$ | Fractional residual, where $\epsilon$ is the frozen energy floor |

If `score_direction=higher`, then $S_i=s_i$; if it is `lower`, then $S_i=-s_i$. Thresholds, ROC calculations, and energy-dependence calculations all use $S_i$. Score direction must be declared in advance and must not be reversed post hoc based on test results.

### 1.2 Primary and Diagnostic Metrics

| Tier | Metric | Direction | Purpose |
|---|---|---:|---|
| Primary classification metric | `matched_auc_macro` | Higher is better | Ranking performance after spectrum matching over the complete fixed pair set |
| Primary regression metric | `energy_regression_score` / ERS-v1 | Higher is better | Jointly constrains event-level error, the overall energy spectrum, and the finite-prediction rate |
| Classification diagnostics | Inclusive/common-support/diagnostic matched AUC | Higher is better | Distinguishes raw performance, common-support performance, and performance before policy gates |
| Regression diagnostics | MAE, RMSE, bias, $R^2$, JSD, $W_1$, etc. | As defined | Explains the sources of error in ERS-v1 |
| Decorrelation diagnostics | Independence, correlation coefficients, acceptance flatness, sculpting | As defined | Tests class-conditional dependence of the classification score on energy |
| Quality diagnostics | Coverage, ESS, status, warnings/errors | Higher coverage/ESS is generally more stable | Determines whether the primary metric has sufficient applicability and statistical support |

Classification and regression are ranked separately; no composite rank is defined. A high independence score alone does not demonstrate that a classifier is strong: a constant score can also be independent of energy.

## 2. Canonical Event-Level Prediction Table Standard

All columns must have the same number of events along their first dimension, and every column must be at least one-dimensional. The recommended container is compressed NPZ. CSV, HDF5, and Parquet are accepted as inputs, but formal reproducibility should preferentially use NPZ because it can preserve metadata. In an NPZ file, ordinary keys are event-level columns, while `__metadata__` is a scalar JSON string.

| Canonical column | Recommended type | Requirement | Exact definition |
|---|---|---|---|
| `event_id` | string | Required in strict mode | Globally stable and unique event identifier; NEXT uses `NEXT::<relative HDF5 path>::<event number within file>` |
| `label` | integer/string | Classification requires `label` or valid category roles | The positive-class label is frozen by the manifest's `positive_label` |
| `category` | string | Required when explicit signal/background categories are used | Physical-process category; explicit category roles take precedence over a generic label |
| `score` | float | Required for classification | Score for the manifest-defined positive class; the NEXT adapter exports the raw logit |
| `energy_condition` | float | Required for energy-matched classification and dependence analysis | Covariate used only for energy matching and conditional diagnostics |
| `energy_target` | float | Required for regression | Event-level regression reference truth; mapped to the internal role `energy_true` |
| `energy_pred` | float | Required for regression | Physical-space prediction aligned with `energy_target` and expressed in the same unit |
| `sample_weight` | float | Optional | Finite, nonnegative base weight; at least one event must have positive weight |
| `split` | string | Required in strict mode | Event split; must match the manifest's evaluation split |
| `group_id` | string | Strongly recommended | Correlated group sharing a file/run/campaign; included in the fingerprint but currently not used for group bootstrap |
| `projection_coverage` | float | NEXT auxiliary column | Voxel energy retained within the projection-coordinate range divided by total event voxel energy; does not directly enter EnergyBench scoring |

For NEXT, both `energy_condition` and `energy_target` are the float64 sum of voxel energies from the same event's `/MC/hits/table`. The CNN-001 classification input is normalized by total event energy, while the paired CNN-001 regression input and the historical v2 input preserve absolute energy amplitude. When `projection_coverage=1`, a raw projection's pixel sum is directly proportional to the target, so this regression is a data-flow baseline, not independent evidence of experimental energy reconstruction.

### 2.1 NPZ Metadata

`__metadata__` should store, at minimum, the adapter, model/checkpoint identifier and SHA-256, data root and split, data inventory, score space, energy unit and derivation, and projection configuration. For v2, it should additionally store the energy-target type, source, unit, training-set-fitted normalization parameters, and prediction space. Metadata records provenance; it does not replace the statistical semantics frozen in the manifest.

Implementation basis: `src/energybench/data.py`, `src/next_cnn/adapter.py`, and `src/next_cnn/data.py`.

## 3. Manifest Freezing Rules

Before evaluation, the following content must be frozen and saved in `.energybench/resolved_manifest.json`:

1. `task_id`, a unique `model_id`, and the manifest schema version;
2. experiment, dataset ID/version, evaluation split, separate checkpoint-selection split, energy kind, and energy unit;
3. explicit column mapping for every canonical role;
4. positive class, signal/background category sets, pair mode, and score direction/space;
5. energy ROI, support trim, number of bins, minimum event count per class, minimum valid-bin count, minimum coverage, matching target, and target TPR;
6. regression histogram/performance bins, explicit edges if applicable, and fractional energy floor;
7. dependence binning and the distance-correlation sample cap;
8. number of bootstrap replicates, confidence level, and random seed.

Strict mode requires explicit event ID, split, and task columns; data provenance; unique and nonempty event IDs; a matching split; and different selection and evaluation splits. Probability scores must also lie in $[0,1]$. The evaluator rejects `split=train`.

For formal comparisons, the following three summaries must agree:

- `evaluation_fingerprint`: a hash computed after sorting event ID, truth/category, matching/target energy, base weight, group, split, and related fields; deliberately excludes `score` and `energy_pred`;
- `protocol_fingerprint`: a hash of the data, classification, regression, dependence, and seed configurations;
- `code_fingerprint`: a hash of the contents of `src/energybench/*.py`.

Implementation basis: `src/energybench/config.py` and `src/energybench/evaluation.py`.

## 4. Classification Scoring: Inclusive and Energy-Matched ROC

### 4.1 Weighted Inclusive ROC/AUC

For a threshold $\tau$, define

$$
\operatorname{TPR}(\tau)=
\frac{\sum_i a_i\,\mathbb 1(Y_i=1,S_i\ge\tau)}
{\sum_i a_i\,\mathbb 1(Y_i=1)},
\qquad
\operatorname{FPR}(\tau)=
\frac{\sum_i a_i\,\mathbb 1(Y_i=0,S_i\ge\tau)}
{\sum_i a_i\,\mathbb 1(Y_i=0)}.
$$

Thresholds are scanned downward from $+\infty$; events with tied scores are processed as a single threshold group. AUC is calculated by trapezoidal integration of the empirical ROC:

$$
\operatorname{AUC}_{\rm inclusive}
=\int_0^1 \operatorname{TPR}(u)\,du.
$$

Inclusive AUC measures ranking performance under the original sample composition and may contain both topological information and shortcuts arising from differences between signal and background energy spectra. It must be reported, but it is not a substitute for matched AUC. The first empirical point reaching `target_tpr` defines the operating point, for which the threshold, realized TPR, FPR, and background rejection $1-\mathrm{FPR}$ are reported.

### 4.2 Common Support, Binning, and Matching Weights

Energy support is first determined separately for each class. If `support_trim_quantile=\alpha>0`, the base-weighted central interval of each class is used; otherwise the sample extrema are used. This support is then intersected with the prespecified physical ROI:

$$
L=\max\{Q_{1,\alpha},Q_{0,\alpha},E_{\rm ROI}^{\rm low}\},\qquad
U=\min\{Q_{1,1-\alpha},Q_{0,1-\alpha},E_{\rm ROI}^{\rm high}\}.
$$

If $U<L$, the matched estimand is undefined. Within common support, bin edges are derived from pooled weighted quantiles in which signal and background each contribute half of the total mass. If the number of distinct energy values does not exceed the requested number of bins, adjacent discrete energy levels are separated by their midpoints.

Let the base mass and within-class mass fraction for class $c$ in bin $k$ be

$$
A_{ck}=\sum_{i:Y_i=c,E_i\in B_k}a_i,
\qquad
p_{ck}=\frac{A_{ck}}{\sum_j A_{cj}}.
$$

A bin is valid only if each class has at least `min_per_class` events and $A_{ck}>0$. The default overlap target is

$$
t_k=
\frac{\mathbb 1(k\text{ valid})\min(p_{1k},p_{0k})}
{\sum_j\mathbb 1(j\text{ valid})\min(p_{1j},p_{0j})}.
$$

When `matching_target=uniform`, $t_k$ is equal across valid bins. The final event weight is

$$
w_i=a_i\frac{t_k}{A_{Y_i k}},\qquad E_i\in B_k.
$$

Consequently, the final mass of each class in every valid bin equals $t_k$, and each class has total final mass 1. The matched ROC is a single global-threshold ROC calculated from $(Y_i,S_i,w_i)$; it is neither exact event-by-event matching at identical energy nor an average of per-bin AUCs.

### 4.3 Formal Gates, Macro-Averaging, and NA

The core matching algorithm first produces `diagnostic_matched_auc`. Formal `matched_auc` must additionally pass both of the following gates:

- the smaller of the two classes' matched **base-mass coverage** values is at least `min_coverage`;
- the number of valid bins is at least `min_valid_bins`; the second requirement is waived if common support consists of a single exact energy level.

The denominator of coverage is the set of same-class events with a finite score, positive base weight, and finite energy; the numerator is the original base mass of events entering valid matched bins:

$$
C_c^{(a)}=
\frac{\sum_{i:Y_i=c,\,i\in\text{matched}}a_i}
{\sum_{i:Y_i=c,\,E_i\text{ finite}}a_i}.
$$

When explicit category pairs are defined, the formal macro-average is the unweighted arithmetic mean over all fixed category pairs; otherwise the same rule is applied to the pooled pair:

$$
\operatorname{AUC}_{\rm matched,macro}
=\frac1P\sum_{p=1}^{P}\operatorname{AUC}_{{\rm matched},p}.
$$

This value exists only if all $P$ pairs are evaluable. `matched_auc_macro_available` is only the diagnostic mean over currently evaluable pairs and must not be used for formal ranking.

The following cases must produce NA and a reason; neither 0.5 nor inclusive AUC may be substituted:

| Condition | Pair-level status |
|---|---|
| Missing signal or background events | `not_evaluable_missing_class` |
| Missing `energy_condition` | `not_applicable_no_energy_condition` |
| Insufficient finite-energy events in a class or no valid bins | `not_evaluable_insufficient_statistics` |
| No common energy support between the two classes | `not_evaluable_no_common_support` |
| Base-mass coverage below the threshold | `not_evaluable_low_coverage` |
| Number of valid bins below the threshold | `not_evaluable_too_few_valid_bins` |
| Incomplete fixed pair set | Aggregate status `not_evaluable_incomplete_pair_set` |

### 4.4 Matching Diagnostics and Uncertainty

`coverage` reports total, with-energy, common-support, and matched event counts, together with count fractions and base-weight fractions. Weighted effective sample size is

$$
\operatorname{ESS}(w)=\frac{(\sum_iw_i)^2}{\sum_iw_i^2}.
$$

`balance` reports one-dimensional Wasserstein-1 distance, weighted KS distance, bin total variation before and after matching, and the maximum post-matching bin-mass difference. `inclusive_common_support_auc` uses all of common support with base weights; `shortcut_gap` is defined as

$$
\operatorname{AUC}_{\rm common\ support}
-\operatorname{AUC}_{\rm diagnostic\ matched}.
$$

This difference is a diagnostic contrast between two estimands; it does not quantify the “fraction of information attributable to energy.”

Classification bootstrap uses event resampling stratified by class and re-estimates common support, edges, valid bins, targets, and matching weights in every replicate. Events in the same HDF5 file may be correlated in the current NEXT data, while the implementation does not provide group bootstrap. NEXT manifests therefore set bootstrap to 0 and do not report potentially misleading event-level confidence intervals.

Implementation basis: `src/energybench/roc.py` and `src/energybench/evaluation.py`.

## 5. Energy Regression: ERS-v1

ERS-v1 is a project-defined, versioned composite score, not a general standard of an experimental collaboration or the machine-learning community. It is comparable only when the physical target definition, unit, event set, base weights, histogram edges, truth-energy bins, and energy floor are all identical.

### 5.1 Finite-Prediction Rate and Event-Level Term

Truth values must be finite. Nonfinite predictions are not silently dropped; instead, they reduce the weighted finite-prediction rate:

$$
f_{\rm finite}=
\frac{\sum_i a_i\mathbb 1(\widehat T_i\text{ finite})}
{\sum_i a_i}.
$$

If the manifest does not specify $\epsilon$ explicitly, the implementation uses $10^{-6}$ times the weighted median of positive $|T|$, with a lower bound of $10^{-12}$. The $K$ performance bins are constructed from base-weighted quantiles of truth. In each nonempty bin containing finite predictions,

$$
m_k=
\frac{\sum_{i\in B_k}a_i|r_i|}{\sum_{i\in B_k}a_i},
\qquad
\operatorname{BFMAE}=\frac1{K'}\sum_{k\in\mathcal K_{\rm nonempty}}m_k,
$$

where $K'$ is the number of nonempty bins containing finite predictions. The event-level term is

$$
S_{\rm event}=\max(0,1-\operatorname{BFMAE}).
$$

Nonempty truth-energy bins receive equal weight so that high-density regions do not dominate the event-level term. If there are no finite predictions, this term is 0.

### 5.2 Spectrum Term and Overall Score

By default, shared histogram edges form equal-width bins over the positive-weight truth range. Probability arrays additionally include explicit underflow and overflow bins. The truth histogram uses all positive-weight truth values; the predicted histogram uses only finite predictions and is renormalized internally. Nonfinite predictions are penalized separately through $f_{\rm finite}$.

For normalized probabilities $p$ and $q$, with $m=(p+q)/2$, the base-2 Jensen–Shannon divergence is

$$
\operatorname{JSD}_2(p,q)=
\frac12\sum_jp_j\log_2\frac{p_j}{m_j}
+\frac12\sum_jq_j\log_2\frac{q_j}{m_j}.
$$

Define

$$
S_{\rm hist}=1-\sqrt{\operatorname{JSD}_2(p,q)},
\qquad
\boxed{\operatorname{ERS\text{-}v1}
=f_{\rm finite}\sqrt{S_{\rm event}S_{\rm hist}}}.
$$

All three factors lie in $[0,1]$, so ERS-v1 also lies in $[0,1]$. The geometric mean requires both event-level accuracy and similarity of the overall energy spectrum; permuting predictions can preserve the spectrum but cannot achieve a high event score.

### 5.3 Auxiliary Regression Metrics

| Field | Definition and interpretation | Direction/unit |
|---|---|---|
| `mae` | $\sum a_i|\Delta_i|/\sum a_i$ | Lower is better; energy unit |
| `rmse` | $\sqrt{\sum a_i\Delta_i^2/\sum a_i}$ | Lower is better; energy unit |
| `bias` | Weighted mean of $\Delta$ | Closer to 0 is better; energy unit |
| `r2` | $1-\sum a_i\Delta_i^2/\sum a_i(T_i-\bar T_w)^2$ | Higher is better; may be negative, and is NA for constant truth |
| `mae_skill` | $1-\mathrm{MAE}/\mathrm{MAE}_{\text{weighted-median baseline}}$ | Higher is better; NA when baseline error is 0 |
| `fractional_bias` | Weighted median of $r$ | Closer to 0 is better; dimensionless |
| `fractional_resolution_68` | $[Q_{0.84}(r)-Q_{0.16}(r)]/2$ | Lower is better; dimensionless |
| `balanced_fractional_mae` | Equal-truth-bin weighted absolute fractional error defined above | Lower is better; dimensionless |
| `jsd_bits` | $\operatorname{JSD}_2$ between truth and predicted flow histograms | Lower is better; bits |
| `histogram_overlap` | $\sum_j\min(p_j,q_j)$ | Higher is better; $[0,1]$ |
| `wasserstein_1` | $W_1$ between the weighted one-dimensional truth and predicted distributions | Lower is better; energy unit |

Each truth-energy bin also stores the weighted mean truth, event count/weight, median response, median residual, absolute/fractional $\sigma_{68}$, MAE, and RMSE. For positive energies sufficiently far above the floor, `response=1+r=\widehat T/T`.

Regression bootstrap is a paired event bootstrap: truth, prediction, and weight are resampled using the same event indices, and every metric is recomputed in each replicate, while histogram edges, truth-energy edges, and the fractional floor remain fixed. Current NEXT manifests likewise set it to 0.

Implementation basis: `src/energybench/regression.py`.

## 6. Class-Conditional Energy Dependence of the Classification Score

Dependence diagnostics are grouped by true `category`; when category is unavailable, signal/background groups are used. Each group excludes nonfinite scores or energies and nonpositive weights, then constructs bins using weighted energy quantiles and shared weighted score quantiles.

### 6.1 Correlation and Distributional Stability

Each group reports:

- `pearson_abs`: absolute weighted Pearson correlation, which detects linear dependence;
- `spearman_abs`: absolute weighted midrank Spearman correlation, which detects monotonic dependence;
- `distance_correlation`: general nonlinear dependence. Exact computation is $O(n^2)$. When the event count exceeds the cap, a deterministic subsample is drawn without replacement with probabilities proportional to base weight, and unweighted empirical dCor is then computed on the subsample;
- `conditional_score_jsd_nats`: weighted mean JSD between the score distribution in each energy bin and the pooled score distribution.

Let $p_k(s)$ be the score histogram in energy bin $k$, $p(s)$ be the pooled histogram for the group, and $\pi_k$ be the bin's base-weight fraction. Then

$$
J_E=\sum_k\pi_k\operatorname{JSD}_{\ln}(p_k,p),
\qquad
I_E=\operatorname{clip}\!\left(1-\sqrt{J_E/\ln2},0,1\right).
$$

A higher `energy_independence_score=I_E` indicates less variation of the score distribution with energy. `overall_energy_independence_score` is the base-mass-weighted mean over evaluable groups, and `worst_group_energy_independence_score` is the minimum group score. The formal summary populates the mean and worst values only when all expected groups are evaluable.

### 6.2 Fixed-Threshold Acceptance and Spectrum Sculpting

For a threshold $\tau$,

$$
A_k=\frac{\sum_{i\in B_k}a_i\mathbb1(S_i\ge\tau)}{\sum_{i\in B_k}a_i},
\qquad
A=\frac{\sum_i a_i\mathbb1(S_i\ge\tau)}{\sum_i a_i}.
$$

The flatness metrics are

$$
\operatorname{RMS}_A=\sqrt{\sum_k\pi_k(A_k-A)^2},
\qquad
D_{\max}=\max_k|A_k-A|.
$$

The natural-log JSD between the energy spectra before and after selection is recorded as `energy_sculpting_jsd_nats`; its normalized distance is $\sqrt{\mathrm{JSD}/\ln2}$.

The current end-to-end implementation obtains $\tau$ from the matched-ROC target-TPR point on the **same evaluation sample** and records `threshold_source=matched_roc_target_tpr`. Acceptance and sculpting are therefore exploratory diagnostics, not unbiased deployment evaluations using a threshold frozen on an independent calibration sample.

Implementation basis: `src/energybench/dependence.py` and `src/energybench/evaluation.py`.

## 7. One-Row `results.csv` Summary Standard

`results.csv` has a fixed schema of one row per evaluation and 64 columns. Python/NumPy `None` or nonfinite results are written as empty cells; an empty cell means NA, not 0. Field order is frozen by `src/energybench/reporting.py`.

### 7.1 Identification and Data Scope

| Field | Meaning |
|---|---|
| `results_schema_version` | CSV schema version, currently 1 |
| `model_id` | Unique model/checkpoint identifier for this run |
| `task_id` | Frozen task identifier |
| `experiment` | Experiment name |
| `dataset_id` | Immutable dataset identifier |
| `dataset_version` | Data release/validation version |
| `split` | Evaluation split |

### 7.2 Classification Fields

| Field | Meaning |
|---|---|
| `classification_status`, `classification_reason` | Whether the classification task as a whole is applicable; overall `ok` does not guarantee that the fixed pair set is complete |
| `matched_auc_status`, `matched_auc_reason` | Status of the formal matched macro-average and aggregated NA reason |
| `matched_auc_macro` | Formal primary classification metric, present only when the complete pair set passes the gates |
| `matched_auc_macro_available` | Diagnostic mean over currently evaluable pairs; not rankable |
| `inclusive_auc_macro` | Unweighted macro-average of inclusive AUC over the fixed pair set when that set is complete |
| `matched_pairs_evaluable` | Number of pairs with a formal matched AUC |
| `matched_pairs_expected` | Total number of pairs in the frozen pair set |
| `complete_pair_set` | Whether the preceding two counts are equal and the pair set is nonempty |

### 7.3 Energy-Regression Fields

| Field | Meaning |
|---|---|
| `energy_regression_status`, `energy_regression_reason` | `ok`, `no_finite_predictions`, or `not_applicable`, and the associated reason |
| `energy_regression_score_name` | Currently `ERS-v1` |
| `energy_regression_score` | Primary ERS-v1 metric; 0 when there are no finite predictions |
| `energy_regression_ci_low`, `energy_regression_ci_high` | ERS-v1 percentile-bootstrap interval |
| `energy_regression_ci_level` | Confidence level of the interval |
| `energy_regression_bootstrap_requested` | Requested number of replicates |
| `energy_regression_bootstrap_successful` | Number of replicates producing a finite ERS |
| `event_score` | $S_{\rm event}$ |
| `histogram_similarity` | $S_{\rm hist}$ |
| `histogram_overlap` | Mass intersection of the flow histograms |
| `jsd_bits` | Base-2 JSD between truth and predicted histograms |
| `wasserstein_1` | $W_1$ between truth and predicted energy spectra |
| `mae`, `rmse`, `bias`, `r2` | Weighted absolute error, root-mean-square error, signed mean bias, and coefficient of determination |
| `fractional_bias` | Weighted median fractional residual |
| `fractional_resolution_68` | Central 68% half-width of the fractional residual |
| `balanced_fractional_mae` | Equal-truth-bin weighted fractional MAE |
| `finite_fraction` | Base-weight fraction of finite predictions |

### 7.4 Energy-Dependence Fields

| Field | Meaning |
|---|---|
| `energy_dependence_status`, `energy_dependence_reason` | Overall computation status of the dependence task |
| `energy_independence_score_status`, `energy_independence_score_reason` | Whether all expected class-conditional groups are evaluable |
| `dependence_groups_evaluable` | Number of groups with `status=ok` |
| `dependence_groups_expected` | Total number of expected groups |
| `complete_dependence_group_set` | Whether all groups are present |
| `energy_independence_score_mean` | Base-mass-weighted $I_E$ over groups when the set is complete |
| `energy_independence_score_worst` | Minimum $I_E$ across groups when the set is complete |

### 7.5 Score and Energy Semantics

| Field | Meaning |
|---|---|
| `score_column` | Column actually resolved as the classification score |
| `score_space` | Explicit space such as probability, logit, rank, or conditional quantile |
| `score_direction` | `higher` or `lower` |
| `energy_condition_kind` | Physical definition of the classification matching covariate |
| `energy_target_kind` | Physical definition of the regression target; `not_applicable` when inapplicable |
| `energy_unit` | Unit of the energy columns and corresponding errors |

### 7.6 Quality and Provenance

| Field | Meaning |
|---|---|
| `n_events` | Number of events in the canonical table |
| `strict` | Whether the result was produced by strict evaluation |
| `warning_count`, `error_count` | Counts of warnings/errors in the complete machine-readable report |
| `evaluation_fingerprint` | Hash over event identity/truth/weights and related content, excluding predictions |
| `protocol_fingerprint` | Hash of the frozen evaluation protocol |
| `code_fingerprint` | Hash of the EnergyBench Python source code |
| `input_sha256` | SHA-256 of the input file when the evaluator loads an actual file; may be empty for an in-memory bundle |
| `created_at_utc` | Generation time in UTC ISO-8601 format |
| `source_schema_version` | Schema version of the complete `metrics.json`, currently 1 |

Classification bootstrap confidence intervals are not included in this 64-column summary table. When enabled, they should be read from pair-level `matching.*_auc_ci` in `metrics.json`.

## 8. Complete Output Files and Their Meaning

### 8.1 Standard Output of `energybench inspect`

`inspect` does not create a fixed file; it prints JSON to the terminal. `source` is the input path, `n_events` is the event count, `columns` gives the shape/dtype of each column, `metadata` is the input provenance, `inferred_roles` is the final inferred canonical mapping, and `duplicate_event_ids` is the number of duplicate primary keys or NA. It performs only structural inventory. It does not compute scores and cannot establish the physical definition or unit of energy, or the absence of split leakage. The caller may redirect the output to a `.json` file if it must be preserved.

### 8.2 `predictions_<split>.npz`

This is the model-independent, event-level evidence layer containing canonical columns and provenance metadata. It can be read repeatedly by `inspect`, `evaluate`, and `decorrelate`, and forms the boundary for rescoring without rerunning inference. `energybench next` first saves this file and then evaluates the same in-memory bundle. Consequently, `input.path/input.sha256` is empty in some existing v2 `metrics.json` files even though the NPZ file exists.

### 8.3 `results.csv`

This is the one-row summary for human reading and downstream tabular processing. Formal conclusions should prioritize status, primary metrics, completeness, and fingerprints rather than a single floating-point value.

### 8.4 `.energybench/metrics.json`

This is the complete machine-readable report. Its top level includes:

| Object | Content |
|---|---|
| `evaluator`, `created_at_utc` | Evaluator version, runtime version, code fingerprint, and time |
| `dataset`, `task_id`, `model_id` | Frozen task/data identifiers |
| `input`, `resolved_columns` | Input path/hash/metadata and final column roles |
| `quality` | Event count, duplicate IDs, split, strict mode, and warnings/errors |
| `classification.aggregates` | Complete-pair-set macro-averages and completeness |
| `classification.pairs[]` | Roles, event counts, formal/diagnostic AUC, common-support AUC, gap, and status for every pair |
| `pairs[].matching` | Support, edges, per-bin mass, coverage, ESS, balance, inclusive/matched ROC arrays, operating point, and bootstrap |
| `energy_regression` | All ERS-v1 components, flow histograms, energy-binned curve arrays, and bootstrap for each metric |
| `energy_dependence` | Per-class correlations, JSD, independence, acceptance, sculpting, and aggregates |
| `artifacts` | User-visible filenames actually generated by this report |

The first point in a ROC array is $(0,0)$, with an in-memory threshold of $+\infty$. Strict JSON serialization writes this nonfinite threshold as `null`. This single `null` is an initial sentinel, not a missing curve. Current large-sample `metrics.json` files are approximately 23 MB, primarily because they retain the full inclusive and matched FPR/TPR/threshold arrays. They are suitable for auditing, not manual reading.

The regression object retains the compatibility aliases `hist_similarity`/`histogram_similarity`, `truth_histogram`/`true_histogram_probability`, and `prediction_histogram`/`pred_histogram_probability`. Each pair of aliases repeats the same numerical value and must not be treated as distinct metrics.

### 8.5 `.energybench/resolved_manifest.json`

This is the effective protocol snapshot after merging defaults, input YAML/JSON, and CLI overrides. It answers “which parameters were actually used in this run,” not “which defaults were written in the template.” For reproducibility, it should be preserved together with the prediction table and metrics.

### 8.6 Decorrelate Outputs

`energybench decorrelate` produces:

- a new NPZ that preserves the original columns, adds `score_raw` on the first run, and adds `score_decorrelated`; its metadata records calibration/test hashes, splits, the background label, sample counts, number of bins, and disjointness status;
- `<output>.decorrelator.json` (or the path specified by `--artifact`), which stores the background conditional weighted ECDF's `energy_edges`, sorted scores in each bin, cumulative weights, score direction, method version, and provenance.

The transformation uses a mid-distribution ECDF:

$$
U(E,s)=\tfrac12\{F_{0,E}(s^-)+F_{0,E}(s)\}.
$$

By default, the event IDs and split roles of calibration and test data must be verifiably disjoint. `--allow-overlap` produces only `unverified_override` and cannot be used for strict formal conclusions. Implementation basis: `src/energybench/decorrelation.py` and `src/energybench/cli.py`.

### 8.7 Compare Outputs

`energybench compare` produces:

- `leaderboard.csv`: one row per evaluation, adding `classification_rank`, `energy_regression_rank`, `comparison_mode`, `comparable`, and `source_metrics`, while carrying the primary statuses, scores, completeness indicators, data identifiers, and fingerprints;
- `.energybench/comparison.json`: stores the comparison mode, number of inputs, fingerprint/code/protocol sets, indices of inputs lacking fingerprints or not produced in strict mode, and all leaderboard rows.

`--allow-mixed-data` produces only a `non_comparative_inventory`, preserves input order, and leaves both ranks empty.

### 8.8 Training Artifacts

The currently source-reproducible CNN-001 classification training outputs include:

| Artifact | Meaning |
|---|---|
| `02_models/checkpoints/*_best.pt` | Checkpoint with the highest validation AUC; falls back to minimum validation loss when AUC cannot be computed |
| `02_models/checkpoints/*_last.pt` | Checkpoint from the final epoch |
| `03_training_runs/logs/*_validation_metrics_*.csv` | Per-epoch train/validation BCE, accuracy, inclusive AUC, AMP skips, validation projection coverage, learning rate used for that epoch, and whether the best checkpoint was updated |
| `03_training_runs/logs/*_history_*.json` | Complete nested record of the same history, additionally including event counts and train/validation coverage |
| `03_training_runs/history_plots/*_history_*.png` | Epoch curves for training/validation BCE and inclusive AUC |
| `03_training_runs/history_plots/*_score_*.png` | Validation logit density by class for the final best checkpoint |

Checkpoints also store model/optimizer state, projection, split seed/fractions, training configuration, data inventory, actual train/validation groups, and history. Implementation basis: `01_code/architectures/cnn_001_two_conv_baseline/train_classification.py`.

The paired CNN-001 energy-regression program additionally writes best/last checkpoints, per-epoch CSV/JSON histories containing standardized Smooth-L1 loss and physical MAE/RMSE/bias/$R^2$, a loss/error history figure, and a validation prediction-versus-target/residual figure. It selects the best checkpoint by minimum validation Smooth-L1 loss and stores a training-only energy normalizer in the checkpoint. These training diagnostics are not a substitute for a formal locked-test ERS-v1 evaluation. Implementation basis: `01_code/architectures/cnn_001_two_conv_baseline/train_energy_regression.py`.

The directory also retains historical v2 CSV/JSON files, joint-loss/AUC/energy-error, score, and energy-validation PNG files, and v2 prediction/evaluation artifacts. These historical records show that v2 logged classification BCE, a normalized-energy Smooth-L1 weighted term, MAE/RMSE/bias/$R^2$, and selected the best checkpoint by minimum validation joint loss. However, the current workspace lacks `01_code/src/next_cnn_v2.py` and the corresponding v2 `.pt` checkpoint. Therefore, v2 **historical results can be interpreted and can be rescored from the surviving NPZ files, but cannot be fully retrained or reproduced through new inference from the current repository**. The sole remaining CPython 3.11 bytecode trace has been isolated at `/home/wenyu/summer_legacy/bytecode/` as recovery evidence only; it is not executable or reproducible source code.

### 8.9 Legacy Artifacts

The `report.md`, `pair_metrics.csv`, root-level `metrics.json`, and `resolved_manifest.json` in `04_evaluations/next_cnn_v1_smoke/evaluation_5files/` are outputs from an older version. The current `run_evaluation` no longer generates a Markdown report, pair table, `energy_regression_bins.csv`, or `matching_diagnostics/`. The current formal layout is `results.csv`, applicable PNG files, and `.energybench/{metrics,resolved_manifest}.json`. Historical outputs are retained only for traceability and must not be used as examples of the current directory or table formats.

This cleanup changed the source contents of the scoring package without changing the mathematical definitions of the primary metrics. New runs therefore have a different `code_fingerprint` from stored historical results. For formal cross-run comparison, all surviving prediction NPZ files should be rescored with the same current version; results with old and new fingerprints must not be mixed directly.

Implementation basis: `src/energybench/evaluation.py` and `src/energybench/reporting.py`.

## 9. Meaning of Standard Evaluation and Training Figures

### 9.1 `energy_matched_roc.png`

The horizontal axis is FPR/background acceptance and the vertical axis is TPR/signal efficiency; the black diagonal represents random ranking. For each pair, the solid curve is the energy-matched ROC and the light dashed curve is the inclusive ROC. The legend gives the AUC from the core matching algorithm and an available confidence interval. A solid curve substantially below the dashed curve suggests that the original ranking may benefit from between-class energy-spectrum differences. Similar curves indicate only that the two estimands are close; they do not establish score–energy independence.

Note: the plotting layer reads the core `result.matched.auc` directly rather than the formal, policy-gated `pair.matched_auc`. If a coverage/bin gate fails, the figure may still display a diagnostic curve. Formal conclusions must use the status and primary metric in `results.csv`. Implementation: `src/energybench/plotting.py`.

### 9.2 `score_energy_dependence.png`

The left panel plots the weighted mean classifier score in each weighted energy-quantile bin, grouped by true class. The legend also gives dCor and the independence score. A slope or curvature indicates that score response varies with energy, but a mean curve cannot replace the full-distribution JSD.

The right panel plots class-specific acceptance in each energy bin at the reported threshold. A flat curve indicates greater stability of the efficiency for that fixed cut. Differences in absolute height between classes are classification behavior itself and should not be mistaken for sculpting. Because this threshold is obtained from the same evaluation sample, the figure is an exploratory diagnostic and has no confidence band. Implementation: `src/energybench/plotting.py`.

### 9.3 `energy_regression.png`

The left panel is a true-versus-predicted hexbin plot using up to 30,000 finite events. Color represents logarithmic event density, and the red dashed line is the ideal $\widehat T=T$. Concentration near the diagonal indicates correct event-level response; contraction into a horizontal band indicates that the model is close to a constant predictor.

The right panel plots residual $\widehat T-T$ against true energy, the weighted median residual in each truth bin, and a shaded `median ± central-68%-half-width` band. Departure of the red line from 0 indicates energy-dependent bias; band width indicates resolution. For an asymmetric residual distribution, `median ± half-width` is not identical to the actual endpoints $[Q_{0.16},Q_{0.84}]$. The visualization subsample is drawn uniformly over events using a fixed seed, and plotted density does not use sample weights; numerical metrics remain weighted. Implementation: `src/energybench/plotting.py`.

### 9.4 `energy_histograms.png`

The upper panel overlays weighted, normalized truth/predicted **probability mass per bin** and annotates JS similarity, histogram overlap, and the underflow/overflow mass of each distribution. Flow bins do not appear in the visible horizontal-axis range, but they enter the score. The lower panel shows the predicted-to-truth mass ratio in visible bins; the horizontal line at 1 indicates matching spectral shape, and bins with zero truth mass have an NA ratio.

This figure tests only the overall energy spectrum and cannot establish correct event-by-event correspondence. It must be interpreted together with the event score and `energy_regression.png`. For readability, the ratio-axis upper limit is set using the 95th percentile and capped at 5; it is not an error bar or confidence interval. Implementation: `src/energybench/plotting.py`.

### 9.5 CNN-001 classification `*_history_*.png`

The left panel shows training/validation `BCEWithLogitsLoss` over epochs; the right panel shows the inclusive AUC over the same period without energy matching. Decreasing training loss accompanied by increasing validation loss suggests overfitting; the train/validation AUC gap reflects the generalization gap. This figure diagnoses checkpoint training and is not a formal performance figure for the locked test split; it also cannot replace energy-matched AUC. Implementation: `01_code/architectures/cnn_001_two_conv_baseline/train_classification.py`.

### 9.6 CNN-001 classification `*_score_*.png`

The filled histogram is the validation `0nubb` logit density and the outline histogram is the `Bi214` density; the title's AUC is validation inclusive AUC. Greater distributional separation generally indicates better ranking, but the appearance depends on logit scale and binning. The figure has no energy matching, base weights, coverage, confidence interval, or test-set interpretation; it is used only to inspect score shape during training/selection. Implementation: `01_code/architectures/cnn_001_two_conv_baseline/train_classification.py`.

### 9.7 CNN-001 regression `*_energy_*.png`

The left panel is validation predicted energy versus target energy with the identity line; the right panel is the residual distribution `prediction - target`. The titles report validation MAE, RMSE, bias, and $R^2$ in MeV where applicable. This is a validation training diagnostic without locked-test status, ERS-v1, uncertainty, or sample weights, and it must not be presented as the formal regression result. Implementation: `01_code/architectures/cnn_001_two_conv_baseline/train_energy_regression.py`.

## 10. Formal Comparison and Reporting Rules

A formal leaderboard must satisfy all of the following:

1. Every input comes from a successful strict evaluation, and `quality.errors` is empty;
2. all inputs have exactly the same evaluation, protocol, and code fingerprints;
3. every `model_id` is nonempty and unique;
4. classification is ranked in descending order only by `matched_auc_macro` when `matched_auc_status=ok`;
5. regression is ranked independently in descending order by ERS-v1; an ERS of 0 with `no_finite_predictions` may rank after valid models;
6. ties use competition ranking, for example 1, 1, 3;
7. classification and regression ranks are not tie-breakers for one another and are not combined into an overall rank;
8. `matched_auc_macro_available`, inclusive AUC, and the independence score must not be used to fill a formal classification rank;
9. `--allow-mixed-data` may create only a noncomparative inventory and must not assign performance ranks.

A paper-style result should report, at minimum, the primary metric and status, complete pair/group counts, inclusive baseline, coverage/ESS, key regression components, dependence diagnostics, units, manifest/fingerprints, and either confidence intervals or the reason no interval is provided. Every NA should be reported with its machine-readable reason.

Implementation basis: `src/energybench/reporting.py`.

## 11. Limitations, Robustness, and Missing Capabilities

1. **Within-group correlation is not represented in confidence intervals.** Current classification and regression implement only event bootstrap; `group_id` enters the fingerprint only. NEXT formal manifests therefore disable bootstrap. If confidence intervals are needed in the future, a group bootstrap stratified by HDF5 file/run should be implemented.
2. **The threshold is not independently calibrated.** Current acceptance/sculpting diagnostics use the matched-ROC threshold from the test/evaluation sample itself and can only be interpreted exploratorily.
3. **Energy matching is a binned approximation.** Matching equalizes bin-level mass, but residual energy differences may remain within a bin. Post-matching KS/$W_1$, coverage, and ESS should be interpreted jointly.
4. **Dependence diagnostics are neither causal evidence nor proof of complete independence.** Pearson/Spearman have shape-specific blind spots, dCor is a subsampling approximation for large samples, JSD depends on binning, and the current implementation provides no uncertainty intervals.
5. **Diagnostics are sensitive to score space.** ROC is invariant under strictly monotonic transformations; mean score, Pearson, JSD, slope, and threshold-based diagnostics generally lack this invariance. Comparisons across probability and logit spaces require caution.
6. **ERS-v1 is a project-specific score.** It is sensitive to histogram/performance bins, the floor, target range, and weights. ERS values from different tasks are not directly comparable even if both lie in $[0,1]$.
7. **Figures are diagnostics, not sufficient evidence.** Current figures have no statistical uncertainty bands; regression scatterplots also use an unweighted subsample. Formal numerical conclusions should use the machine-readable results.
8. **The historical v2 chain is incomplete.** Missing v2 training source and checkpoints prevent end-to-end reproduction of the surviving v2 results. Until they are restored and validated, those results can be used only as historical evaluation examples.

## 12. Implementation Index

| Topic | Current source |
|---|---|
| Canonical table, NPZ, and role inference | `src/energybench/data.py` |
| Manifest defaults and validation | `src/energybench/config.py` |
| Classification pairs, policy gates, aggregation, fingerprints, and output orchestration | `src/energybench/evaluation.py` |
| Inclusive/matched ROC, coverage, ESS, balance, and bootstrap | `src/energybench/roc.py` |
| ERS-v1 and regression diagnostics | `src/energybench/regression.py` |
| Class-conditional score–energy dependence | `src/energybench/dependence.py` |
| Held-out background ECDF | `src/energybench/decorrelation.py` |
| Evaluation/training figures | `src/energybench/plotting.py`, `01_code/architectures/cnn_001_two_conv_baseline/train_classification.py` |
| `results.csv` and leaderboard | `src/energybench/reporting.py` |
| NEXT adapter and event-level physical columns | `src/next_cnn/adapter.py`, `src/next_cnn/data.py` |

## 13. References

1. Fawcett, T. “An Introduction to ROC Analysis.” *Pattern Recognition Letters* 27 (2006): 861–874. <https://doi.org/10.1016/j.patrec.2005.10.010>
2. Janes, H., Longton, G., and Pepe, M. S. “Accommodating Covariates in ROC Analysis.” *The Stata Journal* 9 (2009). <https://pmc.ncbi.nlm.nih.gov/articles/PMC2758790/>
3. Efron, B. “Bootstrap Methods: Another Look at the Jackknife.” *The Annals of Statistics* 7 (1979): 1–26. <https://doi.org/10.1214/aos/1176344552>
4. Lin, J. “Divergence Measures Based on the Shannon Entropy.” *IEEE Transactions on Information Theory* 37 (1991): 145–151. <https://doi.org/10.1109/18.61115>
5. Székely, G. J., Rizzo, M. L., and Bakirov, N. K. “Measuring and Testing Dependence by Correlation of Distances.” *The Annals of Statistics* 35 (2007): 2769–2794. <https://doi.org/10.1214/009053607000000505>
6. SciPy Developers. “`scipy.stats.wasserstein_distance`.” <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html>

This document is the sole current scoring and table specification for the project. If the document, manifest, and source code conflict, formal comparison must stop. The effective `resolved_manifest.json` must be checked, and the protocol/result version must be incremented in synchrony; legacy semantics must not be carried forward silently.
