"""EnergyBench inference adapter for format-version-3 NEXT classifiers."""

from __future__ import annotations

import hashlib
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import numpy as np

from next_cnn.data import dataset_inventory, discover_source_files

from .checkpoint import load_v3_checkpoint, validate_v3_checkpoint
from .data import RepresentationConfig, build_inference_loader
from .registry import build_model, get_model_spec


def _device(requested: str, torch: Any) -> Any:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = torch.device(requested)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return selected


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _status(enabled: bool, message: str) -> None:
    if enabled:
        print("NEXT alternative predict: %s" % message, file=sys.stderr, flush=True)


class _Progress:
    def __init__(
        self,
        total_files: int,
        split: str,
        device: str,
        enabled: bool,
        interval_seconds: float,
    ) -> None:
        self.total_files = int(total_files)
        self.split = str(split)
        self.device = str(device)
        self.enabled = bool(enabled)
        self.interval_seconds = float(interval_seconds)
        self.completed_files = 0
        self.events = 0
        self.started_at = time.monotonic()
        self.last_output_at = self.started_at
        self.bar = None
        if not self.enabled:
            return
        if bool(getattr(sys.stderr, "isatty", lambda: False)()):
            try:
                from tqdm import tqdm
            except ImportError:
                pass
            else:
                self.bar = tqdm(
                    total=self.total_files,
                    desc="NEXT %s alternative inference" % self.split,
                    unit="file",
                    file=sys.stderr,
                    dynamic_ncols=True,
                    mininterval=0.5,
                )
        if self.bar is None:
            self._write("started")

    def _snapshot(self) -> tuple[float, float]:
        elapsed = max(time.monotonic() - self.started_at, 1.0e-12)
        return elapsed, self.events / elapsed

    def _write(self, state: str) -> None:
        elapsed, event_rate = self._snapshot()
        percentage = (
            100.0 * self.completed_files / self.total_files
            if self.total_files
            else 100.0
        )
        print(
            "NEXT %s alternative inference %s: files=%d/%d (%.1f%%), "
            "events=%d, elapsed=%.1fs, rate=%.1f event/s, device=%s"
            % (
                self.split,
                state,
                self.completed_files,
                self.total_files,
                percentage,
                self.events,
                elapsed,
                event_rate,
                self.device,
            ),
            file=sys.stderr,
            flush=True,
        )
        self.last_output_at = time.monotonic()

    def update(self, completed_files: int, events: int) -> None:
        if not self.enabled:
            return
        increment = int(completed_files)
        self.completed_files += increment
        self.events = int(events)
        _, event_rate = self._snapshot()
        if self.bar is not None:
            self.bar.set_postfix(
                events=self.events,
                **{"event/s": "%.1f" % event_rate, "device": self.device},
                refresh=False,
            )
            self.bar.update(increment)
        elif time.monotonic() - self.last_output_at >= self.interval_seconds:
            self._write("running")

    def close(self, success: bool) -> None:
        if not self.enabled:
            return
        state = "complete" if success else "stopped"
        if self.bar is not None:
            self.bar.set_postfix(status=state, refresh=True)
            self.bar.close()
        else:
            self._write(state)


def _split_contract(checkpoint: Mapping[str, Any]) -> tuple[int, list[float]]:
    split_config = checkpoint["split_config"]
    if not isinstance(split_config, Mapping):
        raise ValueError("checkpoint split_config must be a mapping")
    if "seed" not in split_config or "fractions" not in split_config:
        raise ValueError("checkpoint split_config requires seed and fractions")
    raw_seed = split_config["seed"]
    try:
        seed = int(raw_seed)
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint split seed must be an integer") from exc
    if isinstance(raw_seed, bool) or seed != raw_seed:
        raise ValueError("checkpoint split seed must be an integer")
    fractions = np.asarray(split_config["fractions"], dtype=float)
    if (
        fractions.shape != (3,)
        or np.any(~np.isfinite(fractions))
        or np.any(fractions <= 0.0)
        or not np.isclose(fractions.sum(), 1.0, rtol=0.0, atol=1.0e-9)
    ):
        raise ValueError(
            "checkpoint split fractions must be three positive values summing to one"
        )
    return seed, fractions.tolist()


def _model_batch(
    batch: Mapping[str, Any],
    input_kind: str,
    selected_device: Any,
    torch: Any,
) -> Dict[str, Any]:
    keys = {
        "projection2d": ("projections",),
        "multiscale2d": ("projections", "fine_projections"),
        "dense3d": ("volume",),
        "points": ("coords", "features", "mask"),
        "graph": ("coords", "features", "mask"),
        "sequence": ("coords", "features", "mask"),
        "topology": ("coords", "features", "mask"),
        "sparse3d": ("voxel_coords", "voxel_features", "voxel_mask"),
        "hybrid": ("projections", "coords", "features", "mask"),
    }[input_kind]
    moved: Dict[str, Any] = {}
    for key in keys:
        if key not in batch:
            raise KeyError("inference batch is missing model input %r" % key)
        tensor = batch[key]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("inference model input %r must be a tensor" % key)
        if key in {"mask", "voxel_mask"}:
            moved[key] = tensor.to(
                device=selected_device, dtype=torch.bool, non_blocking=True
            )
        elif key == "voxel_coords":
            moved[key] = tensor.to(
                device=selected_device, dtype=torch.int64, non_blocking=True
            )
        else:
            moved[key] = tensor.to(
                device=selected_device, dtype=torch.float32, non_blocking=True
            )
    return moved


def _validate_inventory_contract(
    selection: Mapping[str, Any],
    actual_inventory: Mapping[str, Any],
    data_root: Path,
) -> None:
    """Validate legacy full inventories and the v2 campaign selection contract."""

    expected = selection.get("inventory")
    if not isinstance(expected, Mapping):
        raise ValueError("checkpoint data_selection.inventory must be a mapping")
    if expected.get("scope") != "selected-train-validation-files-only":
        if dict(actual_inventory) != dict(expected):
            raise ValueError(
                "NEXT data inventory differs from the training checkpoint: expected %r, "
                "found %r" % (expected, actual_inventory)
            )
        return

    training_groups = [str(value) for value in selection.get("training_groups", ())]
    validation_groups = [
        str(value) for value in selection.get("validation_groups", ())
    ]
    expected_training_count = int(expected.get("training_file_count", -1))
    expected_validation_count = int(expected.get("validation_file_count", -1))
    if len(training_groups) != expected_training_count:
        raise ValueError(
            "checkpoint training selection count differs from its inventory contract"
        )
    if len(validation_groups) != expected_validation_count:
        raise ValueError(
            "checkpoint validation selection count differs from its inventory contract"
        )
    overlap = sorted(set(training_groups).intersection(validation_groups))
    if overlap:
        raise ValueError(
            "checkpoint training and validation selections overlap; examples: %s"
            % ", ".join(overlap[:5])
        )
    missing = [
        group_id
        for group_id in training_groups + validation_groups
        if not (data_root / group_id).is_file()
    ]
    if missing:
        raise ValueError(
            "checkpoint selection files are missing from the dataset; examples: %s"
            % ", ".join(missing[:5])
        )


def predict(
    model_path: str,
    data_path: str,
    batch_size: int = 32,
    device: str = "cuda:0",
    num_workers: int = 0,
    split: str = "test",
    max_files_per_class: int = 0,
    include_energy_condition: bool = True,
    show_progress: bool = True,
    progress_interval_seconds: float = 10.0,
    split_seed: Optional[int] = None,
    split_fractions: Optional[Sequence[float]] = None,
    _checkpoint_payload: Optional[Mapping[str, Any]] = None,
    **kwargs: Any,
) -> Iterator[Dict[str, Any]]:
    """Yield canonical event-aligned predictions from one trusted v3 checkpoint."""

    if kwargs:
        raise TypeError(
            "unsupported NEXT alternative adapter arguments: %s"
            % ", ".join(sorted(kwargs))
        )
    if isinstance(batch_size, bool) or int(batch_size) != batch_size or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if (
        isinstance(num_workers, bool)
        or int(num_workers) != num_workers
        or num_workers < 0
    ):
        raise ValueError("num_workers must be a non-negative integer")
    if (
        isinstance(max_files_per_class, bool)
        or int(max_files_per_class) != max_files_per_class
        or max_files_per_class < 0
    ):
        raise ValueError("max_files_per_class must be a non-negative integer")
    if not isinstance(include_energy_condition, (bool, np.bool_)):
        raise ValueError("include_energy_condition must be true or false")
    if not isinstance(show_progress, (bool, np.bool_)):
        raise ValueError("show_progress must be true or false")
    try:
        progress_interval_seconds = float(progress_interval_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "progress_interval_seconds must be finite and positive"
        ) from exc
    if (
        not np.isfinite(progress_interval_seconds)
        or progress_interval_seconds <= 0.0
    ):
        raise ValueError("progress_interval_seconds must be finite and positive")

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "NEXT alternative inference requires PyTorch; activate the project environment"
        ) from exc

    checkpoint_path = Path(model_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "NEXT alternative checkpoint does not exist: %s" % checkpoint_path
        )
    selected_device = _device(str(device), torch)
    _status(bool(show_progress), "loading checkpoint %s" % checkpoint_path)
    if _checkpoint_payload is None:
        checkpoint = load_v3_checkpoint(checkpoint_path, map_location="cpu")
    else:
        checkpoint = validate_v3_checkpoint(_checkpoint_payload)

    architecture_id = str(checkpoint["architecture_id"])
    spec = get_model_spec(architecture_id)
    if str(checkpoint["model_name"]) != spec.model_name:
        raise ValueError(
            "checkpoint model_name %r does not match registry model %r"
            % (checkpoint["model_name"], spec.model_name)
        )
    if str(checkpoint["input_kind"]) != spec.input_kind:
        raise ValueError(
            "checkpoint input_kind %r does not match registry input %r"
            % (checkpoint["input_kind"], spec.input_kind)
        )
    input_kind = spec.input_kind
    representation = RepresentationConfig.from_mapping(
        checkpoint["representation_config"]
    )
    checkpoint_seed, checkpoint_fractions = _split_contract(checkpoint)
    if split_seed is not None and int(split_seed) != checkpoint_seed:
        raise ValueError(
            "split_seed differs from the checkpoint; refusing a potentially leaky split"
        )
    if split_fractions is not None and not np.allclose(
        np.asarray(split_fractions, dtype=float),
        np.asarray(checkpoint_fractions, dtype=float),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(
            "split_fractions differ from the checkpoint; refusing a potentially leaky split"
        )

    _status(bool(show_progress), "validating dataset inventory under %s" % data_path)
    resolved_data_root = Path(data_path).expanduser().resolve()
    actual_inventory = dataset_inventory(resolved_data_root)
    selection = checkpoint["data_selection"]
    _validate_inventory_contract(selection, actual_inventory, resolved_data_root)
    maximum = None if int(max_files_per_class) == 0 else int(max_files_per_class)
    files = discover_source_files(
        root=data_path,
        split=str(split),
        split_seed=checkpoint_seed,
        split_fractions=checkpoint_fractions,
        max_files_per_class=maximum,
    )
    prediction_groups = {source.group_id for source in files}
    forbidden_groups = set(selection.get("training_groups", ()))
    if str(split) == "test":
        forbidden_groups.update(selection.get("validation_groups", ()))
    overlap = sorted(prediction_groups.intersection(forbidden_groups))
    if overlap:
        raise ValueError(
            "prediction split overlaps checkpoint selection data; examples: %s"
            % ", ".join(overlap[:5])
        )

    model = build_model(architecture_id, checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(selected_device).eval()
    loader = build_inference_loader(
        files=files,
        representation_config=representation,
        input_kind=input_kind,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        device_type=selected_device.type,
    )
    checkpoint_digest = _file_sha256(checkpoint_path)
    _status(
        bool(show_progress),
        "selected %d %s files; starting %s on %s"
        % (len(files), split, architecture_id, selected_device),
    )
    metadata = {
        "framework": "pytorch",
        "model_name": spec.model_name,
        "architecture_id": architecture_id,
        "input_kind": input_kind,
        "checkpoint_format_version": 3,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_digest,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "dataset_root": str(resolved_data_root),
        "data_split": str(split),
        "split_seed": checkpoint_seed,
        "split_fractions": list(checkpoint_fractions),
        "dataset_inventory": actual_inventory,
        "source_file_count": len(files),
        "representation": representation.to_dict(),
        "score_space": "logit",
        "positive_class": "0nubb",
        "energy_condition_derivation": (
            "float64 sum of MC/hits/table values_block_1 energy rows per event_id"
        ),
        "energy_unit": "MeV",
        "energy_unit_source": (
            "NEXT zeronu benchmark data paper arXiv:2605.07566; "
            "source HDF5 attributes do not declare a unit"
        ),
        "energy_unit_reference": "https://arxiv.org/abs/2605.07566",
        "representation_coverage_definition": (
            "fraction of complete event deposited energy retained by the model representation"
        ),
        "projection_coverage_compatibility_alias": True,
    }

    def batches() -> Iterator[Dict[str, Any]]:
        produced = 0
        completed_files = 0
        success = False
        progress = _Progress(
            total_files=len(files),
            split=str(split),
            device=str(selected_device),
            enabled=bool(show_progress),
            interval_seconds=progress_interval_seconds,
        )
        try:
            with torch.inference_mode():
                for batch in loader:
                    logits = model(
                        _model_batch(batch, input_kind, selected_device, torch)
                    )
                    if not isinstance(logits, torch.Tensor) or logits.ndim != 1:
                        raise ValueError(
                            "NEXT alternative model logits must have shape (batch,)"
                        )
                    score = logits.detach().cpu().numpy().astype(np.float32)
                    if not np.all(np.isfinite(score)):
                        raise ValueError("NEXT alternative logits contain non-finite values")
                    size = len(batch["event_id"])
                    if len(score) != size:
                        raise ValueError("model logits are not aligned with input events")
                    representation_coverage = (
                        batch["representation_coverage"]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
                    columns: Dict[str, Any] = {
                        "event_id": np.asarray(batch["event_id"], dtype=str),
                        "label": (
                            batch["label"]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.int8)
                        ),
                        "category": np.asarray(batch["category"], dtype=str),
                        "score": score,
                        "sample_weight": np.ones(size, dtype=np.float32),
                        "split": np.asarray(batch["split"], dtype=str),
                        "group_id": np.asarray(batch["group_id"], dtype=str),
                        "representation_coverage": representation_coverage,
                        "projection_coverage": representation_coverage,
                        "__metadata__": metadata,
                    }
                    if bool(include_energy_condition):
                        columns["energy_condition"] = (
                            batch["energy_condition"]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float64)
                        )
                    produced += size
                    completed_in_batch = int(
                        batch["source_file_complete"].sum().item()
                    )
                    completed_files += completed_in_batch
                    if completed_files > len(files):
                        raise RuntimeError(
                            "NEXT inference completed more source files than selected"
                        )
                    progress.update(completed_in_batch, produced)
                    yield columns
            if produced == 0:
                raise RuntimeError("NEXT alternative inference produced no events")
            if completed_files != len(files):
                raise RuntimeError(
                    "NEXT inference completed %d/%d selected source files"
                    % (completed_files, len(files))
                )
            success = True
        finally:
            progress.close(success)

    return batches()


__all__ = ["predict"]
