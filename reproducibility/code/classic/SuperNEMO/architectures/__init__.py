"""Local registry for every supported SuperNEMO model class."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

from .cnn_004_multiview_late_fusion import MultiViewLateFusionCNN
from .gnn_001_static_gine import StaticGINEClassifier
from .seq_001_bigru import HilbertBiGRUClassifier
from .ssm_001_pointmamba import PointMambaLiteClassifier
from .trf_001_tracker_transformer import TrackerHitTransformer


_MODEL_CLASSES = (
    MultiViewLateFusionCNN,
    StaticGINEClassifier,
    HilbertBiGRUClassifier,
    PointMambaLiteClassifier,
    TrackerHitTransformer,
)
MODEL_CLASSES = MappingProxyType({model.__name__: model for model in _MODEL_CLASSES})
if len(MODEL_CLASSES) != 5:
    raise RuntimeError("the local SuperNEMO registry must contain exactly five classes")


def get_model_class(model_name: str) -> type[Any]:
    """Resolve one exact local class name without aliases or fuzzy matching."""

    if not isinstance(model_name, str):
        raise TypeError("model_name must be an exact string")
    try:
        return MODEL_CLASSES[model_name]
    except KeyError as error:
        raise KeyError(
            f"unknown local model class {model_name!r}; expected one of: "
            + ", ".join(MODEL_CLASSES)
        ) from error


__all__ = ["MODEL_CLASSES", "get_model_class"]
