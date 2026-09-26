"""How well does each ``rope_base`` cover the real SuperNEMO event extents?

RoPE rotates a channel pair by ``theta * coordinate``. On each axis the coarsest
channel (smallest ``theta``) resolves separations up to ``pi / theta`` without
wrapping. This report measures the per-event extent (max pairwise separation,
in units of 1000 mm) of the first ``--events`` events of each class, and for
each candidate ``rope_base`` prints the fraction of events whose extent fits
inside that coarsest range on every axis. It is the evidence behind
``DEFAULT_SUPERNEMO_ROPE_BASE`` and a guide to the sweep range.

Extents are measured on raw tracker hits (before tokenization), which bounds the
token extents from above: voxel and summary centroids span slightly less.

Usage::

    python supernemo_rope_classifier/scripts/rope_scale_report.py \\
        [--events 30000] [--bases 2 4 8 16 32 64] [--head-dim 16]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np


def _find_project_root() -> Path:
    for candidate in (Path(__file__).resolve(), *Path(__file__).resolve().parents):
        if (
            (candidate / "next_detector" / "next_transformer").is_dir()
            and (candidate / "supernemo_rope_classifier" / "supernemorope_bench").is_dir()
        ):
            return candidate
    raise FileNotFoundError("could not locate the project root")


PROJECT_ROOT = _find_project_root()
for _source_root in (
    PROJECT_ROOT / "next_detector",
    PROJECT_ROOT / "supernemo_rope_classifier",
):
    if str(_source_root) not in sys.path:
        sys.path.insert(0, str(_source_root))

from supernemo_rope_transformer import (  # noqa: E402
    coarsest_unambiguous_extents,
    rope_frequency_table,
)
from supernemorope_bench.config import CLASSIFICATION_SOURCES, COORDINATE_FIELDS  # noqa: E402
from supernemorope_bench.data import scan_event_offsets  # noqa: E402

PERCENTILES = (50, 95, 99)


def _event_extents(path: Path, events: int) -> np.ndarray:
    """Return ``[events, 3]`` per-event coordinate extents in units of 1000 mm."""

    offsets = scan_event_offsets(path)
    events = min(events, len(offsets) - 1)
    stop = int(offsets[events])
    with h5py.File(path, "r") as handle:
        coordinates = np.stack(
            [np.asarray(handle[name][:stop], dtype=np.float64) for name in COORDINATE_FIELDS],
            axis=1,
        )
    extents = np.empty((events, 3))
    for index in range(events):
        block = coordinates[offsets[index] : offsets[index + 1]]
        extents[index] = np.ptp(block, axis=0) / 1000.0
    return extents


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=30_000, help="events per class")
    parser.add_argument("--bases", type=float, nargs="+", default=[2, 4, 8, 16, 32, 64])
    parser.add_argument("--head-dim", type=int, default=16, help="d_model // nhead")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("SUPERNEMO_ROPE_DATA_DIR", "/vast/pvenkata/vis224/SuperNEMO")),
    )
    arguments = parser.parse_args()

    extents = np.concatenate(
        [
            _event_extents(arguments.data_dir / spec.file_name, arguments.events)
            for spec in CLASSIFICATION_SOURCES
        ]
    )
    print(f"Per-event extent (x1000 mm), {len(extents)} events (2nubb + Bi214):")
    for axis, name in enumerate("xyz"):
        values = np.percentile(extents[:, axis], PERCENTILES)
        print(
            f"  {name}: "
            + "  ".join(f"p{p}={v:.2f}" for p, v in zip(PERCENTILES, values))
            + f"  max={extents[:, axis].max():.2f}"
        )

    print(f"\nhead_dim={arguments.head_dim}: fraction of events inside the coarsest channel's range")
    print(f"{'base':>6} {'theta_max':>9} {'lambda_min(m)':>13}   coarsest range x/y/z   inside x     y     z    all")
    for base in arguments.bases:
        table = rope_frequency_table(head_dim=arguments.head_dim, rope_base=base)
        limits = coarsest_unambiguous_extents(head_dim=arguments.head_dim, rope_base=base)
        inside = [(extents[:, axis] <= limits[name]).mean() for axis, name in enumerate("xyz")]
        everything = np.all(
            extents <= np.array([limits["x"], limits["y"], limits["z"]]), axis=1
        ).mean()
        print(
            f"{base:6g} {max(row['theta'] for row in table):9.2f} "
            f"{min(row['wavelength'] for row in table):13.3f}   "
            f"{limits['x']:5.2f}/{limits['y']:5.2f}/{limits['z']:5.2f}        "
            f"{inside[0]:11.3f} {inside[1]:5.3f} {inside[2]:5.3f} {everything:6.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
