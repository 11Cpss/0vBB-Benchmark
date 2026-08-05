"""Classical topology features and dependency-light boosted trees for NEXT.

The public architecture ID remains ``classic_001_topology_xgboost`` because
XGBoost is the preferred backend.  Importing this module does not require
XGBoost: ``TopologyBoostedTreeClassifier(backend="auto")`` selects XGBoost
when it is installed and otherwise uses the explicitly named
``numpy_hist_gbdt`` fallback implemented here.  The fallback is a small
Newton-boosted histogram tree ensemble, not an XGBoost reimplementation.

Neither the feature extractor nor either tree backend consumes total event
energy or absolute detector position.  Inputs are the project's centered,
scaled point tensors and the two per-voxel features.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from next_alt.metrics import binary_auc


TOPOLOGY_FEATURE_NAMES: Tuple[str, ...] = (
    "log_num_voxels",
    "topology_retained_energy_fraction",
    "max_voxel_energy_fraction",
    "top2_energy_fraction",
    "top5_energy_fraction",
    "normalized_energy_entropy",
    "mean_log_hit_count",
    "std_log_hit_count",
    "max_log_hit_count",
    "extent_x",
    "extent_y",
    "extent_z",
    "rms_major",
    "rms_middle",
    "rms_minor",
    "linearity",
    "planarity",
    "sphericity",
    "radial_mean",
    "radial_std",
    "radial_max",
    "principal_length",
    "endpoint_blob_low",
    "endpoint_blob_high",
    "endpoint_blob_min",
    "endpoint_blob_asymmetry",
    "radius_graph_components",
    "radius_graph_mean_degree",
    "radius_graph_branch_fraction",
    "mst_total_length",
    "mst_max_edge",
    "mst_tortuosity",
)


def _as_numpy(value: Any, dtype: Any) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _fraction(name: str, value: float, *, include_one: bool = True) -> float:
    converted = float(value)
    upper_ok = converted <= 1.0 if include_one else converted < 1.0
    if not math.isfinite(converted) or converted <= 0.0 or not upper_ok:
        bound = "(0, 1]" if include_one else "(0, 1)"
        raise ValueError(f"{name} must be in {bound}")
    return converted


class TopologyFeatureExtractor:
    """Map padded centered point events to 32 fixed topology/shape features."""

    feature_names = TOPOLOGY_FEATURE_NAMES

    def __init__(
        self,
        max_topology_points: int = 192,
        connectivity_radius: float = 0.026,
        blob_radius: float = 0.030,
    ) -> None:
        self.max_topology_points = _positive_int(
            "max_topology_points", max_topology_points
        )
        self.connectivity_radius = float(connectivity_radius)
        self.blob_radius = float(blob_radius)
        if not math.isfinite(self.connectivity_radius) or self.connectivity_radius <= 0:
            raise ValueError("connectivity_radius must be finite and positive")
        if not math.isfinite(self.blob_radius) or self.blob_radius <= 0:
            raise ValueError("blob_radius must be finite and positive")

    def config_dict(self) -> Dict[str, Any]:
        return {
            "max_topology_points": self.max_topology_points,
            "connectivity_radius": self.connectivity_radius,
            "blob_radius": self.blob_radius,
            "feature_names": list(self.feature_names),
        }

    def extract_batch(
        self,
        coords: Any,
        features: Any,
        mask: Any,
    ) -> np.ndarray:
        coordinate_array = _as_numpy(coords, np.float64)
        feature_array = _as_numpy(features, np.float64)
        mask_array = _as_numpy(mask, np.bool_)
        if coordinate_array.ndim != 3 or coordinate_array.shape[-1] != 3:
            raise ValueError("coords must have shape (batch, nodes, 3)")
        if feature_array.shape != coordinate_array.shape[:2] + (2,):
            raise ValueError("features must have shape (batch, nodes, 2)")
        if mask_array.shape != coordinate_array.shape[:2]:
            raise ValueError("mask must have shape (batch, nodes)")
        if np.any(np.sum(mask_array, axis=1) < 1):
            raise ValueError("every event must contain at least one valid point")
        rows = [
            self.extract_event(
                coordinate_array[index, mask_array[index]],
                feature_array[index, mask_array[index]],
            )
            for index in range(len(coordinate_array))
        ]
        return np.stack(rows).astype(np.float32, copy=False)

    def extract_event(self, coords: Any, features: Any) -> np.ndarray:
        xyz = _as_numpy(coords, np.float64)
        node = _as_numpy(features, np.float64)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 1:
            raise ValueError("event coords must have shape (nodes, 3) with nodes >= 1")
        if node.shape != (len(xyz), 2):
            raise ValueError("event features must have shape (nodes, 2)")
        if np.any(~np.isfinite(xyz)) or np.any(~np.isfinite(node)):
            raise ValueError("topology inputs must be finite")
        if np.any(node[:, 0] < 0.0):
            raise ValueError("energy fractions must be non-negative")

        if len(xyz) > self.max_topology_points:
            order = np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0], -node[:, 0]))
            keep = order[: self.max_topology_points]
            xyz = xyz[keep]
            node = node[keep]

        count = len(xyz)
        energy = node[:, 0]
        retained = float(np.sum(energy, dtype=np.float64))
        if retained <= 0.0:
            weights = np.full(count, 1.0 / count, dtype=np.float64)
        else:
            weights = energy / retained
        center = np.sum(xyz * weights[:, None], axis=0)
        centered = xyz - center[None, :]

        covariance = (centered * weights[:, None]).T @ centered
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        principal_axis = eigenvectors[:, order[0]]
        # Eigenvectors are defined up to sign.  Fix the sign by requiring the
        # largest-magnitude Cartesian component to be positive, so the low/high
        # endpoint feature order is reproducible on one representation.
        pivot = int(np.argmax(np.abs(principal_axis)))
        if principal_axis[pivot] < 0.0:
            principal_axis = -principal_axis
        rms = np.sqrt(eigenvalues)
        major_variance = float(eigenvalues[0])
        eps = 1.0e-12
        linearity = float((eigenvalues[0] - eigenvalues[1]) / (major_variance + eps))
        planarity = float((eigenvalues[1] - eigenvalues[2]) / (major_variance + eps))
        sphericity = float(eigenvalues[2] / (major_variance + eps))

        radius = np.linalg.norm(centered, axis=1)
        radial_mean = float(np.sum(weights * radius))
        radial_std = float(np.sqrt(np.sum(weights * (radius - radial_mean) ** 2)))
        radial_max = float(np.max(radius))
        extent = np.ptp(centered, axis=0)

        projection = centered @ principal_axis
        low_index = int(np.argmin(projection))
        high_index = int(np.argmax(projection))
        principal_length = float(projection[high_index] - projection[low_index])
        low_distance = np.linalg.norm(centered - centered[low_index], axis=1)
        high_distance = np.linalg.norm(centered - centered[high_index], axis=1)
        blob_low = float(np.sum(energy[low_distance <= self.blob_radius]))
        blob_high = float(np.sum(energy[high_distance <= self.blob_radius]))
        blob_min = min(blob_low, blob_high)
        blob_asymmetry = abs(blob_high - blob_low) / (blob_high + blob_low + eps)

        energy_order = np.sort(energy)[::-1]
        positive_weight = weights[weights > 0.0]
        entropy = -float(np.sum(positive_weight * np.log(positive_weight)))
        normalized_entropy = entropy / math.log(count) if count > 1 else 0.0

        delta = centered[:, None, :] - centered[None, :, :]
        distance = np.sqrt(np.sum(delta * delta, axis=-1))
        adjacency = (distance <= self.connectivity_radius) & ~np.eye(count, dtype=bool)
        degree = np.sum(adjacency, axis=1)
        components = self._component_count(adjacency)
        mst_total, mst_max, mst_degree = self._mst_statistics(distance)
        branch_fraction = float(np.mean(mst_degree >= 3)) if count else 0.0
        tortuosity = mst_total / (principal_length + eps)

        log_hits = node[:, 1]
        result = np.asarray(
            (
                math.log1p(count),
                retained,
                float(energy_order[0]),
                float(np.sum(energy_order[:2])),
                float(np.sum(energy_order[:5])),
                normalized_entropy,
                float(np.mean(log_hits)),
                float(np.std(log_hits)),
                float(np.max(log_hits)),
                float(extent[0]),
                float(extent[1]),
                float(extent[2]),
                float(rms[0]),
                float(rms[1]),
                float(rms[2]),
                linearity,
                planarity,
                sphericity,
                radial_mean,
                radial_std,
                radial_max,
                principal_length,
                blob_low,
                blob_high,
                blob_min,
                blob_asymmetry,
                float(components),
                float(np.mean(degree)),
                branch_fraction,
                mst_total,
                mst_max,
                tortuosity,
            ),
            dtype=np.float32,
        )
        if result.shape != (len(self.feature_names),) or np.any(~np.isfinite(result)):
            raise FloatingPointError("topology feature extraction produced invalid output")
        return result

    @staticmethod
    def _component_count(adjacency: np.ndarray) -> int:
        count = len(adjacency)
        visited = np.zeros(count, dtype=bool)
        components = 0
        for start in range(count):
            if visited[start]:
                continue
            components += 1
            stack = [start]
            visited[start] = True
            while stack:
                current = stack.pop()
                neighbours = np.flatnonzero(adjacency[current] & ~visited)
                visited[neighbours] = True
                stack.extend(int(value) for value in neighbours)
        return components

    @staticmethod
    def _mst_statistics(distance: np.ndarray) -> Tuple[float, float, np.ndarray]:
        count = len(distance)
        degree = np.zeros(count, dtype=np.int64)
        if count <= 1:
            return 0.0, 0.0, degree
        selected = np.zeros(count, dtype=bool)
        selected[0] = True
        best = distance[0].copy()
        parent = np.zeros(count, dtype=np.int64)
        best[0] = np.inf
        edges: List[float] = []
        for _ in range(1, count):
            candidate = best.copy()
            candidate[selected] = np.inf
            child = int(np.argmin(candidate))
            edge_length = float(candidate[child])
            if not math.isfinite(edge_length):
                break
            source = int(parent[child])
            edges.append(edge_length)
            degree[source] += 1
            degree[child] += 1
            selected[child] = True
            improve = (~selected) & (distance[child] < best)
            best[improve] = distance[child, improve]
            parent[improve] = child
        return float(np.sum(edges)), (max(edges) if edges else 0.0), degree


@dataclass
class _TreeNode:
    value: float
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["_TreeNode"] = None
    right: Optional["_TreeNode"] = None

    def predict(self, matrix: np.ndarray, rows: np.ndarray, output: np.ndarray) -> None:
        if self.feature is None:
            output[rows] = self.value
            return
        if self.left is None or self.right is None or self.threshold is None:
            raise RuntimeError("malformed boosted-tree node")
        go_left = matrix[rows, self.feature] <= self.threshold
        self.left.predict(matrix, rows[go_left], output)
        self.right.predict(matrix, rows[~go_left], output)

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"value": float(self.value)}
        if self.feature is not None:
            result.update(
                {
                    "feature": int(self.feature),
                    "threshold": float(self.threshold),
                    "left": self.left.to_dict() if self.left is not None else None,
                    "right": self.right.to_dict() if self.right is not None else None,
                }
            )
        return result

    def node_count(self) -> int:
        if self.feature is None:
            return 1
        return 1 + (self.left.node_count() if self.left else 0) + (
            self.right.node_count() if self.right else 0
        )


class PureNumpyHistogramGBDT:
    """Small logistic Newton-boosted tree fallback with quantile split bins."""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 4,
        learning_rate: float = 0.04,
        max_bins: int = 32,
        min_samples_leaf: int = 24,
        l2_regularization: float = 1.0,
        min_split_gain: float = 0.0,
        subsample: float = 0.85,
        colsample: float = 0.90,
        random_state: int = 42,
        early_stopping_rounds: int = 40,
    ) -> None:
        self.n_estimators = _positive_int("n_estimators", n_estimators)
        self.max_depth = _positive_int("max_depth", max_depth)
        self.learning_rate = float(learning_rate)
        self.max_bins = _positive_int("max_bins", max_bins)
        self.min_samples_leaf = _positive_int("min_samples_leaf", min_samples_leaf)
        self.l2_regularization = float(l2_regularization)
        self.min_split_gain = float(min_split_gain)
        self.subsample = _fraction("subsample", subsample)
        self.colsample = _fraction("colsample", colsample)
        self.random_state = int(random_state)
        self.early_stopping_rounds = _positive_int(
            "early_stopping_rounds", early_stopping_rounds
        )
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.l2_regularization) or self.l2_regularization < 0.0:
            raise ValueError("l2_regularization must be finite and non-negative")
        if not math.isfinite(self.min_split_gain) or self.min_split_gain < 0.0:
            raise ValueError("min_split_gain must be finite and non-negative")
        self.base_score_: float = 0.0
        self.trees_: List[_TreeNode] = []
        self.best_iteration_: int = -1
        self.best_score_: float = -float("inf")
        self.history_: List[Dict[str, Any]] = []

    @staticmethod
    def _sigmoid(logits: np.ndarray) -> np.ndarray:
        clipped = np.clip(logits, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    @staticmethod
    def _loss(labels: np.ndarray, logits: np.ndarray) -> float:
        return float(np.mean(np.logaddexp(0.0, logits) - labels * logits))

    @staticmethod
    def _accuracy(labels: np.ndarray, logits: np.ndarray) -> float:
        return float(np.mean((logits >= 0.0) == (labels == 1.0)))

    def fit(
        self,
        matrix: np.ndarray,
        labels: np.ndarray,
        validation_data: Tuple[np.ndarray, np.ndarray],
    ) -> "PureNumpyHistogramGBDT":
        train_x, train_y = self._validate_data(matrix, labels)
        validation_x, validation_y = self._validate_data(*validation_data)
        if validation_x.shape[1] != train_x.shape[1]:
            raise ValueError("train and validation feature counts must match")
        validation_auc = binary_auc(validation_y.astype(np.int64), np.zeros(len(validation_y)))
        if validation_auc is None:
            raise ValueError("validation data must contain both classes")

        positive_rate = float(np.clip(np.mean(train_y), 1.0e-6, 1.0 - 1.0e-6))
        self.base_score_ = math.log(positive_rate / (1.0 - positive_rate))
        train_logits = np.full(len(train_y), self.base_score_, dtype=np.float64)
        validation_logits = np.full(len(validation_y), self.base_score_, dtype=np.float64)
        self.trees_ = []
        self.history_ = []
        self.best_iteration_ = -1
        self.best_score_ = -float("inf")
        random = np.random.default_rng(self.random_state)

        for iteration in range(self.n_estimators):
            probability = self._sigmoid(train_logits)
            gradient = probability - train_y
            hessian = np.maximum(probability * (1.0 - probability), 1.0e-6)
            row_count = max(2 * self.min_samples_leaf, int(math.ceil(self.subsample * len(train_y))))
            row_count = min(row_count, len(train_y))
            rows = np.sort(random.choice(len(train_y), size=row_count, replace=False))
            feature_count = max(1, int(math.ceil(self.colsample * train_x.shape[1])))
            feature_indices = np.sort(
                random.choice(train_x.shape[1], size=feature_count, replace=False)
            )
            tree = self._fit_node(
                train_x,
                gradient,
                hessian,
                rows,
                feature_indices,
                depth=0,
            )
            self.trees_.append(tree)
            train_logits += self.learning_rate * self._tree_predict(tree, train_x)
            validation_logits += self.learning_rate * self._tree_predict(tree, validation_x)
            train_auc = binary_auc(train_y.astype(np.int64), train_logits)
            current_auc = binary_auc(validation_y.astype(np.int64), validation_logits)
            if current_auc is None:
                raise RuntimeError("validation AUC became undefined")
            improved = float(current_auc) > self.best_score_
            if improved:
                self.best_score_ = float(current_auc)
                self.best_iteration_ = iteration
            self.history_.append(
                {
                    "epoch": iteration + 1,
                    "learning_rate": self.learning_rate,
                    "train": {
                        "loss": self._loss(train_y, train_logits),
                        "accuracy": self._accuracy(train_y, train_logits),
                        "auc": train_auc,
                        "events": int(len(train_y)),
                    },
                    "validation": {
                        "loss": self._loss(validation_y, validation_logits),
                        "accuracy": self._accuracy(validation_y, validation_logits),
                        "auc": current_auc,
                        "events": int(len(validation_y)),
                    },
                    "improved": improved,
                }
            )
            if iteration - self.best_iteration_ >= self.early_stopping_rounds:
                break
        return self

    @staticmethod
    def _validate_data(matrix: Any, labels: Any) -> Tuple[np.ndarray, np.ndarray]:
        x = np.asarray(matrix, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64).reshape(-1)
        if x.ndim != 2 or len(x) < 1 or len(x) != len(y):
            raise ValueError("matrix must be non-empty 2D with one label per row")
        if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
            raise ValueError("boosted-tree data must be finite")
        if np.any((y != 0.0) & (y != 1.0)):
            raise ValueError("labels must contain only 0 and 1")
        return x, y

    def _fit_node(
        self,
        matrix: np.ndarray,
        gradient: np.ndarray,
        hessian: np.ndarray,
        rows: np.ndarray,
        feature_indices: np.ndarray,
        depth: int,
    ) -> _TreeNode:
        total_gradient = float(np.sum(gradient[rows]))
        total_hessian = float(np.sum(hessian[rows]))
        value = -total_gradient / (total_hessian + self.l2_regularization)
        leaf = _TreeNode(value=float(np.clip(value, -10.0, 10.0)))
        if depth >= self.max_depth or len(rows) < 2 * self.min_samples_leaf:
            return leaf

        parent_score = total_gradient**2 / (total_hessian + self.l2_regularization)
        best_gain = self.min_split_gain
        best_feature: Optional[int] = None
        best_threshold: Optional[float] = None
        best_left: Optional[np.ndarray] = None
        quantiles = np.linspace(0.0, 1.0, self.max_bins + 1)[1:-1]
        for feature in feature_indices:
            values = matrix[rows, feature]
            thresholds = np.unique(np.quantile(values, quantiles, method="linear"))
            for threshold in thresholds:
                left_mask = values <= threshold
                left_count = int(np.sum(left_mask))
                if left_count < self.min_samples_leaf or len(rows) - left_count < self.min_samples_leaf:
                    continue
                left_rows = rows[left_mask]
                right_rows = rows[~left_mask]
                left_gradient = float(np.sum(gradient[left_rows]))
                left_hessian = float(np.sum(hessian[left_rows]))
                right_gradient = total_gradient - left_gradient
                right_hessian = total_hessian - left_hessian
                gain = 0.5 * (
                    left_gradient**2 / (left_hessian + self.l2_regularization)
                    + right_gradient**2 / (right_hessian + self.l2_regularization)
                    - parent_score
                )
                if gain > best_gain:
                    best_gain = gain
                    best_feature = int(feature)
                    best_threshold = float(threshold)
                    best_left = left_mask
        if best_feature is None or best_left is None or best_threshold is None:
            return leaf
        return _TreeNode(
            value=leaf.value,
            feature=best_feature,
            threshold=best_threshold,
            left=self._fit_node(
                matrix,
                gradient,
                hessian,
                rows[best_left],
                feature_indices,
                depth + 1,
            ),
            right=self._fit_node(
                matrix,
                gradient,
                hessian,
                rows[~best_left],
                feature_indices,
                depth + 1,
            ),
        )

    @staticmethod
    def _tree_predict(tree: _TreeNode, matrix: np.ndarray) -> np.ndarray:
        output = np.empty(len(matrix), dtype=np.float64)
        tree.predict(matrix, np.arange(len(matrix)), output)
        return output

    def predict_logits(self, matrix: Any, tree_limit: Optional[int] = None) -> np.ndarray:
        x = np.asarray(matrix, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("prediction matrix must be 2D")
        limit = len(self.trees_) if tree_limit is None else int(tree_limit)
        if limit < 0 or limit > len(self.trees_):
            raise ValueError("invalid tree_limit")
        logits = np.full(len(x), self.base_score_, dtype=np.float64)
        for tree in self.trees_[:limit]:
            logits += self.learning_rate * self._tree_predict(tree, x)
        return logits

    def state_dict(self, tree_limit: Optional[int] = None) -> Dict[str, Any]:
        limit = len(self.trees_) if tree_limit is None else int(tree_limit)
        return {
            "backend": "numpy_hist_gbdt",
            "base_score": self.base_score_,
            "learning_rate": self.learning_rate,
            "tree_limit": limit,
            "best_iteration": self.best_iteration_,
            "best_validation_auc": self.best_score_,
            "trees": [tree.to_dict() for tree in self.trees_[:limit]],
            "tree_node_count": int(sum(tree.node_count() for tree in self.trees_[:limit])),
        }


class TopologyBoostedTreeClassifier:
    """Select XGBoost lazily, with an honest pure-NumPy fallback."""

    def __init__(self, backend: str = "auto", **parameters: Any) -> None:
        requested = str(backend).strip().lower()
        if requested not in {"auto", "xgboost", "numpy_hist_gbdt"}:
            raise ValueError("backend must be auto, xgboost, or numpy_hist_gbdt")
        xgboost_available = importlib.util.find_spec("xgboost") is not None
        if requested == "xgboost" and not xgboost_available:
            raise ImportError("backend=xgboost was requested but xgboost is not installed")
        self.backend = (
            "xgboost" if requested == "xgboost" or (requested == "auto" and xgboost_available)
            else "numpy_hist_gbdt"
        )
        self.parameters = dict(parameters)
        self.model: Any = None
        self.history_: List[Dict[str, Any]] = []
        self.best_iteration_: int = -1
        self.best_score_: float = -float("inf")

    def fit(
        self,
        matrix: np.ndarray,
        labels: np.ndarray,
        validation_data: Tuple[np.ndarray, np.ndarray],
    ) -> "TopologyBoostedTreeClassifier":
        if self.backend == "numpy_hist_gbdt":
            model = PureNumpyHistogramGBDT(**self.parameters)
            model.fit(matrix, labels, validation_data)
            self.model = model
            self.history_ = list(model.history_)
            self.best_iteration_ = int(model.best_iteration_)
            self.best_score_ = float(model.best_score_)
            return self

        import xgboost as xgb

        train_x = np.asarray(matrix, dtype=np.float32)
        train_y = np.asarray(labels, dtype=np.float32)
        validation_x = np.asarray(validation_data[0], dtype=np.float32)
        validation_y = np.asarray(validation_data[1], dtype=np.float32)
        rounds = int(self.parameters.get("n_estimators", 500))
        early_stopping = int(self.parameters.get("early_stopping_rounds", 40))
        params = {
            "objective": "binary:logistic",
            "eval_metric": ["logloss", "auc"],
            "tree_method": "hist",
            "max_depth": int(self.parameters.get("max_depth", 4)),
            "eta": float(self.parameters.get("learning_rate", 0.04)),
            "max_bin": int(self.parameters.get("max_bins", 32)),
            "min_child_weight": float(self.parameters.get("min_samples_leaf", 24)),
            "lambda": float(self.parameters.get("l2_regularization", 1.0)),
            "gamma": float(self.parameters.get("min_split_gain", 0.0)),
            "subsample": float(self.parameters.get("subsample", 0.85)),
            "colsample_bytree": float(self.parameters.get("colsample", 0.90)),
            "seed": int(self.parameters.get("random_state", 42)),
        }
        train_matrix = xgb.DMatrix(train_x, label=train_y, feature_names=list(TOPOLOGY_FEATURE_NAMES))
        validation_matrix = xgb.DMatrix(
            validation_x, label=validation_y, feature_names=list(TOPOLOGY_FEATURE_NAMES)
        )
        evaluations: Dict[str, Dict[str, List[float]]] = {}
        self.model = xgb.train(
            params,
            train_matrix,
            num_boost_round=rounds,
            evals=[(train_matrix, "train"), (validation_matrix, "validation")],
            early_stopping_rounds=early_stopping,
            evals_result=evaluations,
            verbose_eval=True,
        )
        self.best_iteration_ = int(self.model.best_iteration)
        self.best_score_ = float(self.model.best_score)
        # XGBoost exposes per-round loss/AUC but not per-round accuracy without
        # a callback.  Null records that distinction honestly in the history.
        rounds_completed = len(evaluations["validation"]["auc"])
        self.history_ = []
        running_best = -float("inf")
        for index in range(rounds_completed):
            validation_auc = float(evaluations["validation"]["auc"][index])
            improved = validation_auc > running_best
            if improved:
                running_best = validation_auc
            self.history_.append(
                {
                    "epoch": index + 1,
                    "learning_rate": float(params["eta"]),
                    "train": {
                        "loss": float(evaluations["train"]["logloss"][index]),
                        "accuracy": None,
                        "auc": float(evaluations["train"]["auc"][index]),
                        "events": int(len(train_y)),
                    },
                    "validation": {
                        "loss": float(evaluations["validation"]["logloss"][index]),
                        "accuracy": None,
                        "auc": validation_auc,
                        "events": int(len(validation_y)),
                    },
                    "improved": improved,
                }
            )
        return self

    def checkpoint_state(self, tree_limit: Optional[int] = None) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("classifier has not been fitted")
        if self.backend == "numpy_hist_gbdt":
            return self.model.state_dict(tree_limit=tree_limit)
        limit = self.best_iteration_ + 1 if tree_limit is None else int(tree_limit)
        try:
            raw = bytes(self.model.save_raw(raw_format="ubj"))
            raw_format = "ubj"
        except TypeError:
            raw = bytes(self.model.save_raw())
            raw_format = "legacy-binary"
        tree_dumps = self.model.get_dump(dump_format="json")[:limit]

        def node_count(node: Mapping[str, Any]) -> int:
            children = node.get("children", [])
            return 1 + sum(node_count(child) for child in children)

        tree_node_count = sum(node_count(json.loads(tree)) for tree in tree_dumps)
        return {
            "backend": "xgboost",
            "booster_raw_base64": base64.b64encode(raw).decode("ascii"),
            "booster_raw_format": raw_format,
            "prediction_tree_limit": limit,
            "tree_count": len(tree_dumps),
            "tree_node_count": tree_node_count,
            "best_iteration": self.best_iteration_,
            "best_validation_auc": self.best_score_,
            "parameters": self.parameters,
        }


__all__ = [
    "PureNumpyHistogramGBDT",
    "TOPOLOGY_FEATURE_NAMES",
    "TopologyBoostedTreeClassifier",
    "TopologyFeatureExtractor",
]
