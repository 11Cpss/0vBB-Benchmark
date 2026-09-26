"""Model-zoo bridge for the standalone EnergyBench workflow.

The repository has two model families with deliberately different public
interfaces:

* CNN-001--003 live in :mod:`next_cnn` and consume one projection tensor.
* The twenty alternative architectures are registered by :mod:`next_alt` and
  consume the representation mappings produced by ``workflow_data.py``.

This module gives the workflow one small, stable registry without duplicating
the alternative model implementations.  It also provides a transparent
microbatch wrapper so the workflow can keep its standard loader batch size of
64 while individual architectures use their previously validated device batch
sizes.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Sequence, Tuple

import torch
from torch import nn


# Per-architecture entry points run this file with ``architectures`` on
# ``sys.path``.  Add the repository source tree explicitly so importing the
# bridge directly has the same behaviour.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


@dataclass(frozen=True)
class ArchitectureSpec:
    """Workflow-facing construction metadata for one architecture."""

    architecture_id: str
    model_name: str
    input_kind: str
    backend: str
    supports_regression: bool
    microbatch_size: int
    source: str


# These are the batch sizes used by the architecture-specific training
# programs before the models were moved behind this workflow.  They are kept
# as device microbatch defaults; the workflow's statistical/effective batch
# remains 64.
_LEGACY_DEFINITIONS = (
    (
        "cnn_001_two_conv_baseline",
        "SimpleNextCNN",
        "projection_tensor",
        16,
    ),
    (
        "cnn_002_global_energy_skip",
        "GlobalEnergySkipCNN",
        "projection_tensor",
        16,
    ),
    (
        "cnn_003_residual_spatial",
        "ResidualSpatialNextCNN",
        "projection_tensor",
        16,
    ),
)

_ALTERNATIVE_MICROBATCH_SIZES = {
    "cnn_004_multiview_late_fusion": 16,
    "cnn_005_multiscale_projection": 8,
    "cnn_006_dense_3d_resnet": 2,
    "point_001_deepsets": 64,
    "point_002_pointnetpp": 16,
    "gnn_001_static_gine": 16,
    "gnn_002_particlenet_edgeconv": 12,
    "gnn_003_egnn": 12,
    "gnn_004_gravnet": 12,
    "hybrid_001_cnn_gnn": 8,
    "classic_001_topology_xgboost": 128,
    "point_003_pointmlp": 12,
    "seq_001_bigru": 16,
    "seq_002_dilated_tcn": 12,
    "mixer_001_projection_mlp_mixer": 16,
    "gnn_005_dimenet_lite": 4,
    "point_004_rigid_kpconv": 8,
    "topo_001_persistence_perslay": 16,
    "ssm_001_pointmamba": 4,
    "sparse_001_submanifold_resnet": 64,
}


def _make_registry() -> Dict[str, ArchitectureSpec]:
    from next_alt.registry import get_model_spec, registered_architectures

    registry: Dict[str, ArchitectureSpec] = {}
    for architecture_id, model_name, input_kind, microbatch_size in (
        _LEGACY_DEFINITIONS
    ):
        registry[architecture_id] = ArchitectureSpec(
            architecture_id=architecture_id,
            model_name=model_name,
            input_kind=input_kind,
            backend="torch",
            supports_regression=True,
            microbatch_size=microbatch_size,
            source="next_cnn",
        )

    alternative_ids = registered_architectures()
    missing_sizes = set(alternative_ids) - set(_ALTERNATIVE_MICROBATCH_SIZES)
    stale_sizes = set(_ALTERNATIVE_MICROBATCH_SIZES) - set(alternative_ids)
    if missing_sizes or stale_sizes:
        raise RuntimeError(
            "workflow microbatch table is out of sync with next_alt.registry "
            "(missing=%s, stale=%s)"
            % (sorted(missing_sizes), sorted(stale_sizes))
        )
    for architecture_id in alternative_ids:
        alternative = get_model_spec(architecture_id)
        is_classic = architecture_id == "classic_001_topology_xgboost"
        registry[architecture_id] = ArchitectureSpec(
            architecture_id=architecture_id,
            model_name=alternative.model_name,
            input_kind=alternative.input_kind,
            backend="xgboost" if is_classic else "torch",
            supports_regression=False,
            microbatch_size=_ALTERNATIVE_MICROBATCH_SIZES[architecture_id],
            source="next_alt",
        )
    if len(registry) != 23:
        raise RuntimeError(
            "expected 23 NEXT architectures, found %d" % len(registry)
        )
    return registry


REGISTRY: Dict[str, ArchitectureSpec] = _make_registry()
# A descriptive alias is useful to callers without weakening the concise
# public name used by the workflow.
ARCHITECTURE_REGISTRY = REGISTRY


def architecture_ids() -> Tuple[str, ...]:
    """Return all workflow architecture IDs in stable registry order."""

    return tuple(REGISTRY)


def get_spec(identifier: str) -> ArchitectureSpec:
    """Resolve an architecture ID (or an unambiguous checkpoint model name)."""

    requested = str(identifier).strip()
    if requested in REGISTRY:
        return REGISTRY[requested]
    matches = [spec for spec in REGISTRY.values() if spec.model_name == requested]
    if len(matches) == 1:
        return matches[0]
    raise KeyError(
        "unknown NEXT architecture %r; expected one of: %s"
        % (requested, ", ".join(architecture_ids()))
    )


def _model_config(model_config: Mapping[str, Any] | None) -> Dict[str, Any]:
    if model_config is None:
        return {}
    if not isinstance(model_config, Mapping):
        raise TypeError("model_config must be a mapping or None")
    return dict(model_config)


def _annotate_model(model: nn.Module, spec: ArchitectureSpec, task: str) -> nn.Module:
    """Attach checkpoint-friendly metadata without changing model state keys."""

    model.architecture_id = spec.architecture_id
    model.model_name = spec.model_name
    model.input_kind = spec.input_kind
    model.backend = spec.backend
    model.supports_regression = spec.supports_regression
    model.task = str(task)
    return model


def build_classifier(
    identifier: str,
    model_config: Mapping[str, Any] | None = None,
) -> nn.Module:
    """Build one PyTorch binary classifier from workflow-safe model options.

    The classical boosted-tree entry is intentionally rejected here: it is
    neither an ``nn.Module`` nor trainable by the workflow's AdamW loop.  Use
    :func:`build_classic_classifier` for that architecture.
    """

    spec = get_spec(identifier)
    if spec.backend == "xgboost":
        raise TypeError(
            "%s uses the xgboost backend and cannot be built as a torch "
            "classifier; use build_classic_classifier()"
            % spec.architecture_id
        )
    config = _model_config(model_config)
    if spec.source == "next_alt":
        from next_alt.registry import build_model

        model = build_model(spec.architecture_id, config)
    else:
        from next_cnn.model import (
            GlobalEnergySkipCNN,
            ResidualSpatialNextCNN,
            SimpleNextCNN,
        )

        model_classes = {
            "cnn_001_two_conv_baseline": SimpleNextCNN,
            "cnn_002_global_energy_skip": GlobalEnergySkipCNN,
            "cnn_003_residual_spatial": ResidualSpatialNextCNN,
        }
        model = model_classes[spec.architecture_id](**config)
    if not isinstance(model, nn.Module):
        raise TypeError(
            "%s did not construct a torch.nn.Module" % spec.architecture_id
        )
    return _annotate_model(model, spec, "classification")


def _fixed_physical_value(
    config: Dict[str, Any], key: str, expected: float
) -> None:
    if key in config and float(config[key]) != float(expected):
        raise ValueError(
            "%s must be %s for physical-MeV regression"
            % (key, format(expected, "g"))
        )
    config[key] = float(expected)


def build_regressor(
    identifier: str,
    model_config: Mapping[str, Any] | None = None,
) -> nn.Module:
    """Build a physical-MeV regressor for CNN-001, CNN-002, or CNN-003."""

    spec = get_spec(identifier)
    if not spec.supports_regression:
        supported = [
            item.architecture_id
            for item in REGISTRY.values()
            if item.supports_regression
        ]
        raise ValueError(
            "%s has no workflow regression model; supported architectures: %s"
            % (spec.architecture_id, ", ".join(supported))
        )
    config = _model_config(model_config)
    from next_cnn.model import (
        GlobalEnergySkipCNN,
        ResidualSpatialEnergyRegressor,
        SimpleNextEnergyRegressor,
    )

    if spec.architecture_id == "cnn_001_two_conv_baseline":
        model: nn.Module = SimpleNextEnergyRegressor(**config)
    elif spec.architecture_id == "cnn_002_global_energy_skip":
        # The workflow's regression projections are raw deposited energy times
        # 100.  Dividing their sums by input_scale therefore produces MeV;
        # center=0 and scale=1 keep that physical baseline unstandardized.
        config.setdefault("input_scale", 100.0)
        _fixed_physical_value(config, "global_center", 0.0)
        _fixed_physical_value(config, "global_scale", 1.0)
        model = GlobalEnergySkipCNN(**config)
    elif spec.architecture_id == "cnn_003_residual_spatial":
        config.setdefault("input_scale", 100.0)
        _fixed_physical_value(config, "energy_mean", 0.0)
        _fixed_physical_value(config, "energy_std", 1.0)
        model = ResidualSpatialEnergyRegressor(**config)
    else:  # pragma: no cover - protected by the registry invariant above
        raise AssertionError("unhandled regression architecture")
    return _annotate_model(model, spec, "regression")


def _classic_parts(
    identifier: str,
    model_config: Mapping[str, Any] | None,
) -> tuple[ArchitectureSpec, str, Dict[str, Any], Dict[str, Any]]:
    spec = get_spec(identifier)
    if spec.backend != "xgboost":
        raise ValueError(
            "%s is a torch architecture; use build_classifier()"
            % spec.architecture_id
        )
    config = _model_config(model_config)
    backend = str(config.pop("backend", "xgboost")).strip().lower()
    extractor = config.pop("extractor", {})
    estimator = config.pop("estimator", None)
    if not isinstance(extractor, Mapping):
        raise TypeError("classic model_config.extractor must be a mapping")
    if estimator is None:
        # Also accept a flat estimator mapping for small programmatic uses.
        estimator_config = config
    else:
        if not isinstance(estimator, Mapping):
            raise TypeError("classic model_config.estimator must be a mapping")
        if config:
            raise ValueError(
                "unknown classic model keys: %s" % ", ".join(sorted(config))
            )
        estimator_config = dict(estimator)
    return spec, backend, dict(extractor), estimator_config


def build_classic_feature_extractor(
    identifier: str = "classic_001_topology_xgboost",
    model_config: Mapping[str, Any] | None = None,
) -> Any:
    """Build the topology feature extractor declared by the classic config."""

    _, _, extractor_config, _ = _classic_parts(identifier, model_config)
    from next_alt.models.classic_topology import TopologyFeatureExtractor

    return TopologyFeatureExtractor(**extractor_config)


def build_classic_classifier(
    identifier: str = "classic_001_topology_xgboost",
    model_config: Mapping[str, Any] | None = None,
) -> Any:
    """Build the non-PyTorch boosted-tree classifier and its feature extractor.

    The returned classifier has a ``feature_extractor`` attribute so callers
    can collect topology matrices without parsing ``model_config`` a second
    time.
    """

    spec, backend, extractor_config, estimator_config = _classic_parts(
        identifier, model_config
    )
    from next_alt.models.classic_topology import (
        TopologyBoostedTreeClassifier,
        TopologyFeatureExtractor,
    )

    classifier = TopologyBoostedTreeClassifier(
        backend=backend, **estimator_config
    )
    classifier.feature_extractor = TopologyFeatureExtractor(**extractor_config)
    classifier.architecture_id = spec.architecture_id
    classifier.model_name = spec.model_name
    classifier.input_kind = spec.input_kind
    classifier.task = "classification"
    return classifier


def _tensor_leaves(value: Any) -> Iterator[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _tensor_leaves(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _tensor_leaves(child)


def _batch_size(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> int:
    sizes = {
        int(tensor.shape[0])
        for root in (*args, kwargs)
        for tensor in _tensor_leaves(root)
        if tensor.ndim > 0
    }
    if not sizes:
        raise ValueError(
            "microbatch input must contain at least one non-scalar tensor"
        )
    if len(sizes) != 1:
        raise ValueError(
            "all non-scalar tensors in a microbatch input must share their "
            "leading batch dimension; received %s" % sorted(sizes)
        )
    size = sizes.pop()
    if size < 1:
        raise ValueError("microbatch input cannot have an empty batch dimension")
    return size


def _slice_batch(value: Any, start: int, stop: int, batch_size: int) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value
        if int(value.shape[0]) != batch_size:  # guarded by _batch_size
            raise ValueError("encountered inconsistent tensor batch dimension")
        return value[start:stop]
    if isinstance(value, Mapping):
        return {
            key: _slice_batch(child, start, stop, batch_size)
            for key, child in value.items()
        }
    if isinstance(value, tuple):
        children = tuple(
            _slice_batch(child, start, stop, batch_size) for child in value
        )
        if hasattr(value, "_fields"):
            return type(value)(*children)
        return children
    if isinstance(value, list):
        return [
            _slice_batch(child, start, stop, batch_size) for child in value
        ]
    return value


def _concatenate_outputs(outputs: Sequence[Any]) -> Any:
    if not outputs:
        raise ValueError("cannot concatenate an empty output sequence")
    first = outputs[0]
    if isinstance(first, torch.Tensor):
        if not all(isinstance(item, torch.Tensor) for item in outputs):
            raise TypeError("microbatch output tensor types are inconsistent")
        if first.ndim == 0:
            return torch.stack(list(outputs), dim=0)
        return torch.cat(list(outputs), dim=0)
    if isinstance(first, Mapping):
        keys = tuple(first)
        if any(
            not isinstance(item, Mapping) or tuple(item) != keys
            for item in outputs
        ):
            raise TypeError("microbatch output mapping keys are inconsistent")
        return {
            key: _concatenate_outputs([item[key] for item in outputs])
            for key in keys
        }
    if isinstance(first, tuple):
        length = len(first)
        if any(not isinstance(item, tuple) or len(item) != length for item in outputs):
            raise TypeError("microbatch output tuple structures are inconsistent")
        children = tuple(
            _concatenate_outputs([item[index] for item in outputs])
            for index in range(length)
        )
        if hasattr(first, "_fields"):
            return type(first)(*children)
        return children
    if isinstance(first, list):
        length = len(first)
        if any(not isinstance(item, list) or len(item) != length for item in outputs):
            raise TypeError("microbatch output list structures are inconsistent")
        return [
            _concatenate_outputs([item[index] for item in outputs])
            for index in range(length)
        ]
    if all(item == first for item in outputs):
        return first
    raise TypeError(
        "microbatch model outputs must be tensors or matching nested "
        "Mapping/tuple/list structures"
    )


class MicrobatchModel(nn.Module):
    """Run a model on leading-dimension chunks and concatenate its outputs.

    Inputs may be tensors or arbitrarily nested mappings, tuples, and lists of
    tensors.  Every non-scalar tensor is required to share the same leading
    batch dimension.  The structure is preserved for each model invocation,
    and tensor outputs (including nested outputs) are concatenated along that
    same dimension.  ``torch.cat`` keeps autograd connections intact.
    """

    def __init__(
        self,
        model: nn.Module,
        microbatch_size: int,
        architecture_id: str | None = None,
        task: str | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        if (
            isinstance(microbatch_size, bool)
            or int(microbatch_size) != microbatch_size
            or int(microbatch_size) < 1
        ):
            raise ValueError("microbatch_size must be a positive integer")
        self.model = model
        self.microbatch_size = int(microbatch_size)

        resolved_id = architecture_id or getattr(model, "architecture_id", None)
        self.architecture_id = None if resolved_id is None else str(resolved_id)
        spec = None
        if self.architecture_id is not None:
            try:
                spec = get_spec(self.architecture_id)
            except KeyError:
                # The wrapper remains useful for external/test modules.
                spec = None
        self.model_name = getattr(
            model,
            "model_name",
            spec.model_name if spec is not None else type(model).__name__,
        )
        self.input_kind = getattr(
            model,
            "input_kind",
            spec.input_kind if spec is not None else None,
        )
        self.backend = getattr(
            model,
            "backend",
            spec.backend if spec is not None else "torch",
        )
        self.supports_regression = getattr(
            model,
            "supports_regression",
            spec.supports_regression if spec is not None else None,
        )
        inherited_task = getattr(model, "task", None)
        self.task = str(task if task is not None else inherited_task or "")

    @property
    def wrapped_model(self) -> nn.Module:
        """Return the underlying model without registering a duplicate module."""

        return self.model

    @property
    def architecture_metadata(self) -> Dict[str, Any]:
        """Return metadata needed to describe/reconstruct a workflow model."""

        return {
            "architecture_id": self.architecture_id,
            "model_name": self.model_name,
            "input_kind": self.input_kind,
            "backend": self.backend,
            "supports_regression": self.supports_regression,
            "task": self.task,
            "microbatch_size": self.microbatch_size,
        }

    def config_dict(self) -> Dict[str, Any]:
        """Delegate constructor configuration to the wrapped implementation."""

        method = getattr(self.model, "config_dict", None)
        if method is None:
            return {}
        config = method()
        if not isinstance(config, Mapping):
            raise TypeError("wrapped model config_dict() must return a mapping")
        return dict(config)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        batch_size = _batch_size(args, kwargs)
        if batch_size <= self.microbatch_size:
            return self.model(*args, **kwargs)

        outputs = []
        for start in range(0, batch_size, self.microbatch_size):
            stop = min(start + self.microbatch_size, batch_size)
            chunk_args = tuple(
                _slice_batch(value, start, stop, batch_size) for value in args
            )
            chunk_kwargs = {
                key: _slice_batch(value, start, stop, batch_size)
                for key, value in kwargs.items()
            }
            outputs.append(self.model(*chunk_args, **chunk_kwargs))
        return _concatenate_outputs(outputs)


__all__ = [
    "ARCHITECTURE_REGISTRY",
    "ArchitectureSpec",
    "MicrobatchModel",
    "REGISTRY",
    "architecture_ids",
    "build_classic_classifier",
    "build_classic_feature_extractor",
    "build_classifier",
    "build_regressor",
    "get_spec",
]
