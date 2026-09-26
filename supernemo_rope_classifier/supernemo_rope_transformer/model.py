"""SuperNEMO tracker-hit Transformer with rotary position encoding.

The published SuperNEMO Transformers (coordinate MLP, Fourier XYZ) and the NEXT
RoPE model share one architecture: content MLP -> pre-norm encoder -> masked
mean pooling -> MLP head, with one raw logit per event.
``next_transformer.NEXTTransformerClassifier`` already implements it for all
three position encodings, so this module reuses it unchanged and only:

- fixes ``feature_dim`` from the tokenization (4 for sampled-hit and voxel
  tokens, 6 for summary tokens);
- supplies a SuperNEMO-specific default ``rope_base`` (see below);
- records the tokenization in the checkpoint metadata;
- exposes the RoPE frequency table so the scale choice is inspectable.

``rope_base`` and the SuperNEMO coordinate scale
------------------------------------------------
Token coordinates are per-event-centered tracker positions divided by 1000 mm.
Per-event extents are about 0.8 (x, both sides of the foil), 1.2 (y) and
1.0 (z) at the 95th percentile, and reach 4.9 (y) and 2.8 (z) at the maximum
(``scripts/rope_scale_report.py``).
RoPE rotates a channel pair by ``theta * coordinate``; ``theta`` runs from
``rope_base`` (fastest) down to ``rope_base ** (1 / pairs_on_that_axis)``
(slowest). For ``head_dim = 16`` the pair split is x:3, y:3, z:2, so the z axis
has the fewest channels and is the binding constraint. A pair separation
``d`` stays unambiguous on a channel while ``theta * d <= pi``; the coarsest
channel of each axis sets how far apart two tokens can be while that axis still
carries an unwrapped relative position (``coarsest_unambiguous_extents``).

``DEFAULT_SUPERNEMO_ROPE_BASE = 8.0`` keeps the slowest channel unambiguous up
to ``pi / 8**(1/2) = 1.11`` on z and ``pi / 8**(1/3) = 1.57`` on y and x -- the
bulk of events -- while its fastest channel (wavelength ~0.8 m) still
resolves structure at the 44 mm cell pitch only weakly. Unlike the MJD
analysis there is no ``rope_base > 1`` that is unambiguous for the worst-case
separation, so some large events necessarily wrap on their slowest channel.
The value is a principled starting point, not a tuned optimum: sweep it.
"""

from __future__ import annotations

import math
from typing import Any

from next_transformer import NEXTTransformerClassifier
from next_transformer.rotary_attention import RotaryPositionAngles

from .tokenization import SuperNEMOTrackerTokenizationConfig


DEFAULT_SUPERNEMO_ROPE_BASE = 8.0

POSITION_ENCODINGS = ("rope", "coordinate_mlp", "fourier_xyz")

# Architecture shared by every published SuperNEMO Transformer.
D_MODEL = 64
N_HEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 256
DROPOUT = 0.1
NUM_FREQUENCIES = 6


def build_supernemo_transformer(
    tokenization_config: SuperNEMOTrackerTokenizationConfig,
    position_encoding: str = "rope",
    *,
    rope_base: float = DEFAULT_SUPERNEMO_ROPE_BASE,
    d_model: int = D_MODEL,
    nhead: int = N_HEAD,
    num_layers: int = NUM_LAYERS,
    dim_feedforward: int = DIM_FEEDFORWARD,
    dropout: float = DROPOUT,
    num_frequencies: int = NUM_FREQUENCIES,
) -> NEXTTransformerClassifier:
    """Build the classifier for one tokenization and position encoding.

    ``rope_base`` is only used by ``position_encoding='rope'`` but is always
    validated and recorded, so a run's configuration is complete either way.
    """

    if not isinstance(tokenization_config, SuperNEMOTrackerTokenizationConfig):
        raise TypeError(
            "tokenization_config must be a SuperNEMOTrackerTokenizationConfig"
        )
    if position_encoding not in POSITION_ENCODINGS:
        raise ValueError(f"position_encoding must be one of {POSITION_ENCODINGS}")

    model = NEXTTransformerClassifier(
        position_encoding=position_encoding,
        feature_dim=tokenization_config.feature_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        num_frequencies=num_frequencies,
        rope_base=rope_base,
    )
    # Picked up by EnergyBench's checkpoint metadata.
    model.architecture_metadata = {
        "dataset": "SuperNEMO",
        "task": "classification: 2nubb (1) vs Bi214 (0)",
        "tokenization": tokenization_config.to_dict(),
    }
    return model


def rope_frequency_table(
    *,
    head_dim: int,
    rope_base: float,
) -> list[dict[str, Any]]:
    """List every RoPE channel pair's axis, rate, wavelength, and safe extent.

    ``unambiguous_extent`` is ``pi / theta``: the largest coordinate separation
    (in units of 1000 mm) whose rotation angle still lies within ``(-pi, pi]``.
    """

    angles = RotaryPositionAngles(head_dim=head_dim, rope_base=rope_base)
    return [
        {
            "axis": "xyz"[int(axis)],
            "theta": float(theta),
            "wavelength": 2.0 * math.pi / float(theta),
            "unambiguous_extent": math.pi / float(theta),
        }
        for axis, theta in zip(angles.axis_index.tolist(), angles.theta.tolist())
    ]


def coarsest_unambiguous_extents(
    *,
    head_dim: int,
    rope_base: float,
) -> dict[str, float]:
    """Return, per axis, the separation its coarsest channel resolves unwrapped.

    This is ``pi / theta_slowest`` for each axis: the largest coordinate
    separation (in units of 1000 mm) for which the axis's slowest rotation still
    lies within ``(-pi, pi]``. Faster channels wrap sooner by design -- they
    provide the fine scales -- so the coarsest channel sets the range over which
    an axis carries an unambiguous relative position.
    """

    limits: dict[str, float] = {}
    for row in rope_frequency_table(head_dim=head_dim, rope_base=rope_base):
        axis = str(row["axis"])
        limits[axis] = max(limits.get(axis, 0.0), float(row["unambiguous_extent"]))
    return limits


__all__ = [
    "DEFAULT_SUPERNEMO_ROPE_BASE",
    "POSITION_ENCODINGS",
    "build_supernemo_transformer",
    "rope_frequency_table",
    "coarsest_unambiguous_extents",
]
