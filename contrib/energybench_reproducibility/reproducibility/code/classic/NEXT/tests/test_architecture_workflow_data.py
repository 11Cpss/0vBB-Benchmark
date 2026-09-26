"""Synthetic contract tests for the standalone/next_alt data bridge."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "01_code"))

from architectures.workflow_data import build_architecture_loaders  # noqa: E402


def _write_events(path: Path, *, label: int, first_event_id: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.dtype(
        [
            ("index", "<i8"),
            ("values_block_0", "<i8", (1,)),
            ("values_block_1", "<f4", (4,)),
            ("values_block_2", "S6", (1,)),
        ]
    )
    rows = np.zeros(12, dtype=dtype)
    raw_label = b"Signal" if label else b"Bkg"
    for event_ordinal in range(6):
        for hit_ordinal in range(2):
            row = 2 * event_ordinal + hit_ordinal
            rows[row]["index"] = row
            rows[row]["values_block_0"][0] = first_event_id + event_ordinal
            rows[row]["values_block_1"] = (
                float(event_ordinal + hit_ordinal),
                float(2 * hit_ordinal),
                float(label + 3 * hit_ordinal),
                np.float32(0.1 + 0.01 * event_ordinal + 0.02 * hit_ordinal),
            )
            rows[row]["values_block_2"][0] = raw_label
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset("MC/hits/table", data=rows)
        dataset.attrs["values_block_0_kind"] = np.bytes_(
            "(lp0\nVevent_id\np1\na."
        )
        dataset.attrs["values_block_1_kind"] = np.bytes_(
            "(lp0\nVx\np1\naVy\np2\naVz\np3\naVenergy\np4\na."
        )
        dataset.attrs["values_block_2_kind"] = np.bytes_(
            "(lp0\nVlabel\np1\na."
        )


def _prepared_data(root: Path) -> SimpleNamespace:
    files = (
        ("signal/events.h5", "0nubb", 1),
        ("background/events.h5", "Bi214", 0),
    )
    for index, (relative_path, _, label) in enumerate(files):
        _write_events(
            root / relative_path,
            label=label,
            first_event_id=100 * (index + 1),
        )

    ranges = {
        "train": (0, 3),
        "validation": (3, 4),
        "test": (4, 6),
    }
    loaders = {}
    for split, (start, stop) in ranges.items():
        slices = [
            SimpleNamespace(
                relative_path=relative_path,
                category=category,
                label=label,
                split=split,
                event_start=start,
                event_stop=stop,
            )
            for relative_path, category, label in files
        ]
        loaders[split] = SimpleNamespace(
            dataset=SimpleNamespace(file_slices=slices, seed=17)
        )
    return SimpleNamespace(
        train_loader=loaders["train"],
        validation_loader=loaders["validation"],
        test_loader=loaders["test"],
        counts={"train": 6, "validation": 2, "test": 4},
    )


REPRESENTATION = {
    "projection_grid_size": 8,
    "projection_bin_size": 10.0,
    "projection_origin": [-40.0, -40.0, -40.0],
    "projection_input_scale": 10.0,
    "fine_grid_size": 8,
    "fine_bin_size": 5.0,
    "point_bin_size": 2.0,
    "coordinate_scale": 10.0,
    "max_points": 8,
    "dense_grid_size": 8,
    "dense_bin_size": 2.0,
}

EXPECTED_INPUTS = {
    "projection2d": {"projections"},
    "multiscale2d": {"projections", "fine_projections"},
    "dense3d": {"volume"},
    "points": {"coords", "features", "mask"},
    "graph": {"coords", "features", "mask"},
    "hybrid": {"projections", "coords", "features", "mask"},
    "sequence": {"coords", "features", "mask"},
    "topology": {"coords", "features", "mask"},
    "sparse3d": {"voxel_coords", "voxel_features", "voxel_mask"},
}


@pytest.mark.parametrize("input_kind", tuple(EXPECTED_INPUTS))
def test_all_input_kinds_emit_only_model_tensors(tmp_path: Path, input_kind: str) -> None:
    prepared = _prepared_data(tmp_path)
    _, validation_loader, _ = build_architecture_loaders(
        prepared,
        tmp_path,
        input_kind,
        REPRESENTATION,
        batch_size=2,
        num_workers=0,
        shuffle_buffer_size=3,
    )
    batch = next(iter(validation_loader))
    assert set(batch) == {
        "inputs",
        "label",
        "energy",
        "event_id",
        "category",
        "group_id",
        "split",
        "sample_weight",
        "projection_coverage",
    }
    assert set(batch["inputs"]) == EXPECTED_INPUTS[input_kind]
    assert all(isinstance(value, torch.Tensor) for value in batch["inputs"].values())
    assert batch["label"].dtype == torch.float32
    assert batch["energy"].dtype == torch.float64
    assert batch["sample_weight"].tolist() == [1.0, 1.0]
    assert batch["split"] == ["validation", "validation"]


def _event_order(loader: object) -> list[str]:
    return [
        event_id
        for batch in loader
        for event_id in batch["event_id"]
    ]


def test_slice_shuffle_is_reproducible_and_workers_do_not_duplicate(
    tmp_path: Path,
) -> None:
    prepared = _prepared_data(tmp_path)
    train_loader, _, _ = build_architecture_loaders(
        prepared,
        tmp_path,
        "points",
        REPRESENTATION,
        batch_size=2,
        num_workers=0,
        shuffle_buffer_size=3,
    )
    train_loader.dataset.set_epoch(4)
    first = _event_order(train_loader)
    train_loader.dataset.set_epoch(4)
    assert _event_order(train_loader) == first
    train_loader.dataset.set_epoch(5)
    assert _event_order(train_loader) != first

    parallel_loaders = build_architecture_loaders(
        prepared,
        tmp_path,
        "points",
        REPRESENTATION,
        batch_size=2,
        num_workers=2,
        shuffle_buffer_size=3,
    )
    all_ids = [event_id for loader in parallel_loaders for event_id in _event_order(loader)]
    assert len(all_ids) == 12
    assert len(set(all_ids)) == 12

