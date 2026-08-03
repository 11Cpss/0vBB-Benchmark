"""Template for connecting any new model to EnergyBench.

Copy this file into the model repository and replace only the three marked
sections.  The function may use PyTorch, TensorFlow, JAX, sklearn, ROOT, or any
private framework; EnergyBench only sees the returned NumPy-compatible arrays.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def predict(
    model_path: str,
    data_path: str,
    batch_size: int = 128,
    **kwargs: Any
) -> Dict[str, Any]:
    # 1. Load the framework-specific model.
    # model = YourModel.load(model_path)

    # 2. Load a FIXED test manifest, iterate without shuffling, and collect one
    #    prediction per globally unique event ID.
    # event_id, label, category = ...
    # energy_condition, energy_target, sample_weight = ...
    # score, energy_pred = model(...)

    # 3. Return canonical event-aligned columns.  Delete a task-specific column
    #    only when that task truly does not apply.
    raise NotImplementedError(
        "Fill in model/data loading and return the canonical columns below"
    )
    return {
        "event_id": np.asarray(event_id),
        "label": np.asarray(label),
        "category": np.asarray(category),
        "energy_condition": np.asarray(energy_condition, dtype=float),
        "score": np.asarray(score, dtype=float),
        "energy_target": np.asarray(energy_target, dtype=float),
        "energy_pred": np.asarray(energy_pred, dtype=float),
        "sample_weight": np.asarray(sample_weight, dtype=float),
        "__metadata__": {
            "model_path": model_path,
            "dataset_path": data_path,
            "score_space": "logit",  # or probability
            "energy_unit": "MeV",
        },
    }
