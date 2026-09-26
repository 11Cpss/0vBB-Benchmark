"""Alternative NEXT event-classification architectures.

The package is intentionally separate from :mod:`next_cnn`: the legacy CNN
implementations keep their existing checkpoint contract, while the models in
this package share a format-version-3 contract and a common training runner.
"""

from .registry import (
    ModelSpec,
    build_model,
    get_model_spec,
    registered_architectures,
)

__all__ = [
    "ModelSpec",
    "build_model",
    "get_model_spec",
    "registered_architectures",
]
