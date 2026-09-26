"""Shared event representations and loaders for alternative NEXT models.

All raw-file discovery, file-level splitting, and HDF5 event parsing are
delegated to :mod:`next_cnn.data`.  This module only converts an
``EventRecord`` into the representation requested by a registered model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from next_cnn.data import (
    EventRecord,
    ProjectionConfig,
    SourceFile,
    discover_source_files,
    iter_file_events,
    project_event,
)

from .config import INPUT_KINDS


@dataclass(frozen=True)
class RepresentationConfig:
    """Numerical definition of every supported event representation."""

    projection_grid_size: int = 128
    projection_bin_size: float = 30.0
    projection_origin: tuple[float, float, float] = (
        -1920.0,
        -1920.0,
        -120.0,
    )
    projection_input_scale: float = 100.0
    fine_grid_size: int = 128
    fine_bin_size: float = 15.0
    point_bin_size: float = 15.0
    coordinate_scale: float = 1000.0
    max_points: Optional[int] = 512
    dense_grid_size: int = 96
    dense_bin_size: float = 15.0
    center_projection: bool = False

    def __post_init__(self) -> None:
        integer_fields = {
            "projection_grid_size": self.projection_grid_size,
            "fine_grid_size": self.fine_grid_size,
            "dense_grid_size": self.dense_grid_size,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or int(value) != value or value <= 0:
                raise ValueError("%s must be a positive integer" % name)
        positive_fields = {
            "projection_bin_size": self.projection_bin_size,
            "projection_input_scale": self.projection_input_scale,
            "fine_bin_size": self.fine_bin_size,
            "point_bin_size": self.point_bin_size,
            "coordinate_scale": self.coordinate_scale,
            "dense_bin_size": self.dense_bin_size,
        }
        for name, value in positive_fields.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError("%s must be finite and positive" % name)
        if len(self.projection_origin) != 3 or not np.all(
            np.isfinite(self.projection_origin)
        ):
            raise ValueError("projection_origin must contain three finite values")
        if self.max_points is not None and (
            isinstance(self.max_points, bool)
            or int(self.max_points) != self.max_points
            or self.max_points <= 0
        ):
            raise ValueError("max_points must be a positive integer or null")

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | "RepresentationConfig" | None,
    ) -> "RepresentationConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("representation_config must be a mapping")
        known = {
            name: value[name]
            for name in cls.__dataclass_fields__
            if name in value
        }
        if "projection_origin" in known:
            known["projection_origin"] = tuple(known["projection_origin"])
        return cls(**known)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "projection_grid_size": int(self.projection_grid_size),
            "projection_bin_size": float(self.projection_bin_size),
            "projection_origin": [float(value) for value in self.projection_origin],
            "projection_input_scale": float(self.projection_input_scale),
            "fine_grid_size": int(self.fine_grid_size),
            "fine_bin_size": float(self.fine_bin_size),
            "point_bin_size": float(self.point_bin_size),
            "coordinate_scale": float(self.coordinate_scale),
            "max_points": (
                None if self.max_points is None else int(self.max_points)
            ),
            "dense_grid_size": int(self.dense_grid_size),
            "dense_bin_size": float(self.dense_bin_size),
            "center_projection": bool(self.center_projection),
        }

    def coarse_projection(self) -> ProjectionConfig:
        return ProjectionConfig(
            grid_size=int(self.projection_grid_size),
            bin_size=float(self.projection_bin_size),
            origin=tuple(float(value) for value in self.projection_origin),
            normalize_energy=True,
            input_scale=float(self.projection_input_scale),
            representation="energy",
        )

    def fine_projection(self) -> ProjectionConfig:
        half_width = 0.5 * float(self.fine_grid_size) * float(self.fine_bin_size)
        return ProjectionConfig(
            grid_size=int(self.fine_grid_size),
            bin_size=float(self.fine_bin_size),
            origin=(-half_width, -half_width, -half_width),
            normalize_energy=True,
            input_scale=float(self.projection_input_scale),
            representation="energy",
        )


@dataclass(frozen=True)
class VoxelizedEvent:
    """One event aggregated into centered 15 mm voxels."""

    coordinates_mm: np.ndarray
    coordinates_scaled: np.ndarray
    features: np.ndarray
    energy_fraction: np.ndarray
    hit_count: np.ndarray
    retained_energy_fraction: float


def _validated_event_arrays(
    coordinates: np.ndarray,
    energies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    xyz = np.asarray(coordinates, dtype=np.float32)
    energy = np.asarray(energies, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("coordinates must have shape (n_hits, 3)")
    if energy.ndim != 1 or len(energy) != len(xyz) or len(energy) == 0:
        raise ValueError("energies must have shape (n_hits,) for a non-empty event")
    if np.any(~np.isfinite(xyz)) or np.any(~np.isfinite(energy)):
        raise ValueError("coordinates and energies must be finite")
    if np.any(energy < 0):
        raise ValueError("energies must be non-negative")
    total = float(np.sum(energy, dtype=np.float64))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("event total energy must be finite and positive")
    return xyz, energy, total


def energy_weighted_center(
    coordinates: np.ndarray,
    energies: np.ndarray,
) -> np.ndarray:
    """Return a float32 energy-weighted event centroid."""

    xyz, energy, total = _validated_event_arrays(coordinates, energies)
    center = np.sum(
        xyz.astype(np.float64) * energy.astype(np.float64)[:, None],
        axis=0,
    ) / total
    return center.astype(np.float32)


def voxelize_centered_event(
    coordinates: np.ndarray,
    energies: np.ndarray,
    bin_size: float = 15.0,
    coordinate_scale: float = 1000.0,
    max_points: Optional[int] = 512,
) -> VoxelizedEvent:
    """Aggregate hits into centered voxels and build two node features.

    The first feature is deposited energy divided by the complete event
    energy.  The second is ``log1p`` of the number of original rows assigned
    to that voxel.  When ``max_points`` truncates an event, nodes are selected
    by deposited energy and the first feature is deliberately *not*
    renormalized; ``retained_energy_fraction`` therefore records coverage.
    """

    xyz, energy, total = _validated_event_arrays(coordinates, energies)
    if not np.isfinite(bin_size) or bin_size <= 0:
        raise ValueError("bin_size must be finite and positive")
    if not np.isfinite(coordinate_scale) or coordinate_scale <= 0:
        raise ValueError("coordinate_scale must be finite and positive")
    if max_points is not None and (
        isinstance(max_points, bool)
        or int(max_points) != max_points
        or max_points <= 0
    ):
        raise ValueError("max_points must be a positive integer or null")

    raw_center = energy_weighted_center(xyz, energy)
    centered = xyz.astype(np.float64) - raw_center.astype(np.float64)
    cell = np.floor(centered / float(bin_size)).astype(np.int64)
    unique_cell, inverse = np.unique(cell, axis=0, return_inverse=True)
    voxel_energy = np.bincount(
        inverse,
        weights=energy.astype(np.float64),
        minlength=len(unique_cell),
    )
    hit_count = np.bincount(inverse, minlength=len(unique_cell)).astype(np.int64)
    centers = (unique_cell.astype(np.float64) + 0.5) * float(bin_size)

    # Recenter the quantized voxel centers as well.  This removes the
    # half-bin quantization offset while retaining all relative distances.
    quantized_center = np.sum(centers * voxel_energy[:, None], axis=0) / total
    centers -= quantized_center

    if max_points is not None and len(unique_cell) > int(max_points):
        # Lexicographic tie breakers make truncation deterministic.
        order = np.lexsort(
            (
                unique_cell[:, 2],
                unique_cell[:, 1],
                unique_cell[:, 0],
                -voxel_energy,
            )
        )[: int(max_points)]
        centers = centers[order]
        voxel_energy = voxel_energy[order]
        hit_count = hit_count[order]

    energy_fraction = (voxel_energy / total).astype(np.float32)
    features = np.column_stack(
        (energy_fraction, np.log1p(hit_count).astype(np.float32))
    ).astype(np.float32, copy=False)
    coordinates_mm = centers.astype(np.float32)
    coordinates_scaled = (centers / float(coordinate_scale)).astype(np.float32)
    return VoxelizedEvent(
        coordinates_mm=coordinates_mm,
        coordinates_scaled=coordinates_scaled,
        features=features,
        energy_fraction=energy_fraction,
        hit_count=hit_count,
        retained_energy_fraction=float(np.sum(voxel_energy) / total),
    )


def _centered_projection(
    event: EventRecord,
    projection: ProjectionConfig,
) -> tuple[np.ndarray, float]:
    center = energy_weighted_center(event.coordinates, event.energies)
    centered = np.asarray(event.coordinates, dtype=np.float32) - center[None, :]
    return project_event(centered, event.energies, projection)


def _dense_voxel(
    event: EventRecord,
    config: RepresentationConfig,
) -> tuple[np.ndarray, float]:
    voxelized = voxelize_centered_event(
        event.coordinates,
        event.energies,
        bin_size=config.dense_bin_size,
        coordinate_scale=config.coordinate_scale,
        max_points=None,
    )
    size = int(config.dense_grid_size)
    indices = np.floor(
        voxelized.coordinates_mm / float(config.dense_bin_size) + 0.5 * size
    ).astype(np.int64)
    valid = np.all((indices >= 0) & (indices < size), axis=1)
    volume = np.zeros((2, size, size, size), dtype=np.float32)
    if np.any(valid):
        x_index, y_index, z_index = indices[valid].T
        np.add.at(
            volume[0],
            (z_index, y_index, x_index),
            voxelized.energy_fraction[valid],
        )
        np.add.at(
            volume[1],
            (z_index, y_index, x_index),
            np.log1p(voxelized.hit_count[valid]).astype(np.float32),
        )
    coverage = float(np.sum(voxelized.energy_fraction[valid], dtype=np.float64))
    return volume, coverage


def represent_event(
    event: EventRecord,
    input_kind: str,
    representation_config: Mapping[str, Any] | RepresentationConfig | None = None,
) -> Dict[str, Any]:
    """Convert one ``EventRecord`` to a model-ready NumPy sample."""

    kind = str(input_kind)
    if kind not in INPUT_KINDS:
        raise ValueError(
            "input_kind must be one of: %s" % ", ".join(sorted(INPUT_KINDS))
        )
    config = RepresentationConfig.from_mapping(representation_config)
    row: Dict[str, Any] = {
        "event_id": event.event_id,
        "label": np.float32(event.label),
        "category": event.category,
        "energy_condition": np.float64(event.energy_sum),
        "energy_target": np.float64(event.energy_sum),
        "source_file_complete": np.bool_(event.is_last_in_file),
        "split": event.split,
        "group_id": event.group_id,
    }

    if kind in {"projection2d", "multiscale2d", "hybrid"}:
        if config.center_projection:
            half_width = (
                0.5
                * float(config.projection_grid_size)
                * float(config.projection_bin_size)
            )
            centered_config = ProjectionConfig(
                grid_size=int(config.projection_grid_size),
                bin_size=float(config.projection_bin_size),
                origin=(-half_width, -half_width, -half_width),
                normalize_energy=True,
                input_scale=float(config.projection_input_scale),
                representation="energy",
            )
            coarse, coarse_coverage = _centered_projection(
                event, centered_config
            )
        else:
            coarse, coarse_coverage = project_event(
                event.coordinates,
                event.energies,
                config.coarse_projection(),
            )
        row["projection_coverage"] = np.float32(coarse_coverage)
        row["representation_coverage"] = np.float32(coarse_coverage)
        if kind == "projection2d":
            row["projections"] = coarse
        elif kind == "multiscale2d":
            fine, fine_coverage = _centered_projection(
                event, config.fine_projection()
            )
            row["projections"] = coarse
            row["fine_projections"] = fine
            row["fine_projection_coverage"] = np.float32(fine_coverage)
        else:
            row["projections"] = coarse

    if kind in {"points", "graph", "hybrid", "sequence", "topology"}:
        points = voxelize_centered_event(
            event.coordinates,
            event.energies,
            bin_size=config.point_bin_size,
            coordinate_scale=config.coordinate_scale,
            max_points=config.max_points,
        )
        row["coords"] = points.coordinates_scaled
        row["features"] = points.features
        row["point_coverage"] = np.float32(points.retained_energy_fraction)
        row.setdefault(
            "projection_coverage", np.float32(points.retained_energy_fraction)
        )
        row.setdefault(
            "representation_coverage", np.float32(points.retained_energy_fraction)
        )

    if kind == "sparse3d":
        # Sparse models receive every occupied 15 mm voxel.  Integer cell
        # coordinates are computed after removing the event energy centroid,
        # so neither absolute detector position nor total event energy enters
        # the classifier.  Unlike the padded point representation, this path
        # deliberately performs no max-points truncation.
        xyz, energy, total = _validated_event_arrays(
            event.coordinates, event.energies
        )
        center = energy_weighted_center(xyz, energy)
        cells = np.floor(
            (xyz.astype(np.float64) - center.astype(np.float64)[None, :])
            / float(config.point_bin_size)
        ).astype(np.int64)
        unique_cells, inverse = np.unique(cells, axis=0, return_inverse=True)
        voxel_energy = np.bincount(
            inverse,
            weights=energy.astype(np.float64),
            minlength=len(unique_cells),
        )
        hit_count = np.bincount(
            inverse, minlength=len(unique_cells)
        ).astype(np.int64)
        # Translate by an integer reference cell to keep indices compact while
        # preserving the exact occupied-neighbour relation.
        reference = np.rint(
            np.sum(unique_cells * voxel_energy[:, None], axis=0) / total
        ).astype(np.int64)
        row["voxel_coords"] = (unique_cells - reference[None, :]).astype(
            np.int64, copy=False
        )
        row["voxel_features"] = np.column_stack(
            (
                (voxel_energy / total).astype(np.float32),
                np.log1p(hit_count).astype(np.float32),
            )
        ).astype(np.float32, copy=False)
        row["representation_coverage"] = np.float32(1.0)

    if kind == "dense3d":
        voxel, coverage = _dense_voxel(event, config)
        row["volume"] = voxel
        row["projection_coverage"] = np.float32(coverage)
        row["representation_coverage"] = np.float32(coverage)
    return row


def padded_event_collate(samples: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Collate fixed tensors and pad variable point sets with a boolean mask."""

    if not samples:
        raise ValueError("cannot collate an empty batch")
    batch: Dict[str, Any] = {}
    string_fields = ("event_id", "category", "split", "group_id")
    for field in string_fields:
        batch[field] = [str(sample[field]) for sample in samples]
    batch["label"] = torch.as_tensor(
        [sample["label"] for sample in samples], dtype=torch.float32
    )
    batch["energy_condition"] = torch.as_tensor(
        [sample["energy_condition"] for sample in samples], dtype=torch.float64
    )
    batch["energy_target"] = torch.as_tensor(
        [sample["energy_target"] for sample in samples], dtype=torch.float64
    )
    batch["source_file_complete"] = torch.as_tensor(
        [sample["source_file_complete"] for sample in samples], dtype=torch.bool
    )

    optional_scalar_fields = (
        "projection_coverage",
        "representation_coverage",
        "fine_projection_coverage",
        "point_coverage",
    )
    for field in optional_scalar_fields:
        if field in samples[0]:
            if not all(field in sample for sample in samples):
                raise ValueError("inconsistent batch field: %s" % field)
            batch[field] = torch.as_tensor(
                [sample[field] for sample in samples], dtype=torch.float32
            )

    fixed_fields = ("projections", "fine_projections", "volume")
    for field in fixed_fields:
        if field in samples[0]:
            if not all(field in sample for sample in samples):
                raise ValueError("inconsistent batch field: %s" % field)
            batch[field] = torch.from_numpy(
                np.stack([np.asarray(sample[field]) for sample in samples])
            ).to(dtype=torch.float32)

    if "coords" in samples[0]:
        if not all("coords" in sample and "features" in sample for sample in samples):
            raise ValueError("coords and features must be present in every sample")
        counts = [int(len(sample["coords"])) for sample in samples]
        if any(count <= 0 for count in counts):
            raise ValueError("every event must contain at least one point")
        maximum = max(counts)
        coords = torch.zeros((len(samples), maximum, 3), dtype=torch.float32)
        features = torch.zeros((len(samples), maximum, 2), dtype=torch.float32)
        mask = torch.zeros((len(samples), maximum), dtype=torch.bool)
        for index, (sample, count) in enumerate(zip(samples, counts)):
            sample_coords = torch.as_tensor(sample["coords"], dtype=torch.float32)
            sample_features = torch.as_tensor(
                sample["features"], dtype=torch.float32
            )
            if sample_coords.shape != (count, 3):
                raise ValueError("coords must have shape (n_points, 3)")
            if sample_features.shape != (count, 2):
                raise ValueError("features must have shape (n_points, 2)")
            coords[index, :count] = sample_coords
            features[index, :count] = sample_features
            mask[index, :count] = True
        batch["coords"] = coords
        batch["features"] = features
        batch["mask"] = mask
        batch["num_points"] = torch.as_tensor(counts, dtype=torch.int64)

    if "voxel_coords" in samples[0]:
        if not all(
            "voxel_coords" in sample and "voxel_features" in sample
            for sample in samples
        ):
            raise ValueError(
                "voxel_coords and voxel_features must be present in every sample"
            )
        counts = [int(len(sample["voxel_coords"])) for sample in samples]
        if any(count <= 0 for count in counts):
            raise ValueError("every sparse event must contain an occupied voxel")
        maximum = max(counts)
        voxel_coords = torch.zeros(
            (len(samples), maximum, 3), dtype=torch.int64
        )
        voxel_features = torch.zeros(
            (len(samples), maximum, 2), dtype=torch.float32
        )
        voxel_mask = torch.zeros((len(samples), maximum), dtype=torch.bool)
        for index, (sample, count) in enumerate(zip(samples, counts)):
            sample_coords = torch.as_tensor(
                sample["voxel_coords"], dtype=torch.int64
            )
            sample_features = torch.as_tensor(
                sample["voxel_features"], dtype=torch.float32
            )
            if sample_coords.shape != (count, 3):
                raise ValueError("voxel_coords must have shape (n_voxels, 3)")
            if sample_features.shape != (count, 2):
                raise ValueError(
                    "voxel_features must have shape (n_voxels, 2)"
                )
            voxel_coords[index, :count] = sample_coords
            voxel_features[index, :count] = sample_features
            voxel_mask[index, :count] = True
        batch["voxel_coords"] = voxel_coords
        batch["voxel_features"] = voxel_features
        batch["voxel_mask"] = voxel_mask
        batch["num_voxels"] = torch.as_tensor(counts, dtype=torch.int64)
    return batch


class AlternativeEventDataset(IterableDataset):
    """Stream selected HDF5 events into one registered input representation."""

    def __init__(
        self,
        files: Iterable[SourceFile],
        input_kind: str,
        representation_config: Mapping[str, Any] | RepresentationConfig | None = None,
        shuffle_files: bool = False,
        balance_classes: bool = False,
        seed: int = 42,
        event_shuffle_buffer_size: int = 0,
    ) -> None:
        super().__init__()
        self.files = list(files)
        if not self.files:
            raise ValueError("AlternativeEventDataset needs at least one source file")
        if input_kind not in INPUT_KINDS:
            raise ValueError("unsupported input_kind: %s" % input_kind)
        if (
            isinstance(event_shuffle_buffer_size, bool)
            or int(event_shuffle_buffer_size) != event_shuffle_buffer_size
            or event_shuffle_buffer_size < 0
        ):
            raise ValueError("event_shuffle_buffer_size must be non-negative")
        self.input_kind = str(input_kind)
        self.representation = RepresentationConfig.from_mapping(
            representation_config
        )
        self.shuffle_files = bool(shuffle_files)
        self.balance_classes = bool(balance_classes)
        self.seed = int(seed)
        self.event_shuffle_buffer_size = int(event_shuffle_buffer_size)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @staticmethod
    def _events(sources: Sequence[SourceFile]) -> Iterator[EventRecord]:
        for source in sources:
            yield from iter_file_events(source)

    def _shuffled_events(
        self,
        stream: Iterator[EventRecord],
        worker_id: int,
    ) -> Iterator[EventRecord]:
        size = self.event_shuffle_buffer_size
        if size == 0:
            yield from stream
            return
        generator = np.random.default_rng(
            np.random.SeedSequence([self.seed, self.epoch, worker_id])
        )
        buffer: List[EventRecord] = []
        for event in stream:
            if len(buffer) < size:
                buffer.append(event)
                continue
            selected = int(generator.integers(0, len(buffer)))
            yield buffer[selected]
            buffer[selected] = event
        generator.shuffle(buffer)
        yield from buffer

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        files = list(self.files)
        if self.shuffle_files:
            generator = np.random.default_rng(self.seed + self.epoch)
            generator.shuffle(files)
        worker = torch.utils.data.get_worker_info()
        worker_id = 0 if worker is None else int(worker.id)

        if self.balance_classes:
            by_label = {
                label: [source for source in files if source.label == label]
                for label in (1, 0)
            }
            if worker is not None:
                by_label = {
                    label: sources[worker.id :: worker.num_workers]
                    for label, sources in by_label.items()
                }
            iterators = {
                label: iter(self._events(by_label[label])) for label in (1, 0)
            }
            while True:
                try:
                    pair = (next(iterators[1]), next(iterators[0]))
                except StopIteration:
                    return
                for event in pair:
                    yield represent_event(event, self.input_kind, self.representation)
            return

        if worker is not None:
            files = files[worker.id :: worker.num_workers]
        stream = self._events(files)
        if self.shuffle_files:
            stream = self._shuffled_events(stream, worker_id)
        for event in stream:
            yield represent_event(event, self.input_kind, self.representation)


def build_inference_loader(
    files: Iterable[SourceFile],
    representation_config: Mapping[str, Any] | RepresentationConfig | None,
    input_kind: str,
    batch_size: int,
    num_workers: int,
    device_type: str = "cuda",
) -> DataLoader:
    """Build the deterministic loader used by adapters and full evaluation."""

    dataset = AlternativeEventDataset(
        files=files,
        input_kind=input_kind,
        representation_config=representation_config,
        shuffle_files=False,
        balance_classes=False,
    )
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        pin_memory=str(device_type).startswith("cuda"),
        collate_fn=padded_event_collate,
    )


def build_training_loaders(
    config: Mapping[str, Any],
    input_kind: str,
    device_type: str = "cuda",
) -> tuple[
    DataLoader,
    DataLoader,
    AlternativeEventDataset,
    List[SourceFile],
    List[SourceFile],
]:
    """Discover train/validation files and construct their streaming loaders."""

    data = dict(config["data"])
    training = dict(config["training"])
    common = {
        "root": data["root"],
        "split_seed": int(data["split_seed"]),
        "split_fractions": data["split_fractions"],
        "max_files_per_class": data["max_files_per_class"],
    }
    train_files = discover_source_files(split="train", **common)
    validation_files = discover_source_files(split="validation", **common)
    train_dataset = AlternativeEventDataset(
        train_files,
        input_kind=input_kind,
        representation_config=config["representation"],
        shuffle_files=True,
        balance_classes=bool(data["balance_training_classes"]),
        seed=int(training["seed"]),
        event_shuffle_buffer_size=int(data["event_shuffle_buffer_size"]),
    )
    validation_dataset = AlternativeEventDataset(
        validation_files,
        input_kind=input_kind,
        representation_config=config["representation"],
        shuffle_files=False,
        balance_classes=False,
        seed=int(training["seed"]),
    )
    loader_options = {
        "batch_size": int(training["batch_size"]),
        "num_workers": int(data["num_workers"]),
        "pin_memory": str(device_type).startswith("cuda"),
        "collate_fn": padded_event_collate,
    }
    train_loader = DataLoader(train_dataset, **loader_options)
    validation_loader = DataLoader(validation_dataset, **loader_options)
    return (
        train_loader,
        validation_loader,
        train_dataset,
        train_files,
        validation_files,
    )


__all__ = [
    "AlternativeEventDataset",
    "RepresentationConfig",
    "VoxelizedEvent",
    "build_inference_loader",
    "build_training_loaders",
    "energy_weighted_center",
    "padded_event_collate",
    "represent_event",
    "voxelize_centered_event",
]
