"""EnergyBench: model-independent evaluation for energy-aware classifiers.

The public API intentionally works on NumPy arrays.  A model implemented in
PyTorch, TensorFlow, JAX, scikit-learn, ROOT, or a private framework only needs
to export the canonical prediction columns described in :mod:`energybench.data`.
"""

from .data import PredictionBundle, load_bundle, save_bundle

__all__ = ["PredictionBundle", "load_bundle", "save_bundle"]
__version__ = "0.1.0"

