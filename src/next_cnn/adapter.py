"""EnergyBench adapter for legacy v2 CNNs and v3 alternative dispatch."""

from __future__ import annotations

import hashlib
import sys
import time
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence

import numpy as np

from .data import (
    NextIterableDataset,
    ProjectionConfig,
    dataset_inventory,
    discover_source_files,
)


def _read_checkpoint(path: Path, device: Any, torch: Any) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError("NEXT CNN checkpoint does not exist: %s" % path)
    try:
        payload = torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:  # PyTorch versions before the weights_only argument.
        payload = torch.load(str(path), map_location=device)
    if not isinstance(payload, dict):
        raise ValueError("NEXT checkpoint root must be a mapping: %s" % path)
    return payload


def _load_checkpoint(
    path: Path,
    device: Any,
    torch: Any,
    payload: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    if payload is None:
        checkpoint = _read_checkpoint(path, device, torch)
    elif isinstance(payload, Mapping):
        checkpoint = dict(payload)
    else:
        raise ValueError("NEXT checkpoint root must be a mapping: %s" % path)
    if checkpoint.get("format_version") != 2:
        raise ValueError("unsupported NEXT CNN checkpoint format: %s" % path)
    required = {
        "model_config",
        "model_state_dict",
        "projection_config",
        "split_config",
        "data_selection",
    }
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError("checkpoint is missing fields: %s" % ", ".join(missing))
    model_name = str(checkpoint.get("model_name", "SimpleNextCNN"))
    supported_models = {
        "SimpleNextCNN",
        "SimpleNextEnergyRegressor",
        "GlobalEnergySkipCNN",
        "MultiTaskNextCNN",
        "ResidualSpatialEnergyRegressor",
        "ResidualSpatialNextCNN",
    }
    if model_name not in supported_models:
        raise ValueError(
            "unsupported NEXT CNN model_name %r in %s" % (model_name, path)
        )
    task = str(checkpoint.get("task", "")).strip().lower()
    if model_name == "SimpleNextEnergyRegressor" and task != "energy_regression":
        raise ValueError("SimpleNextEnergyRegressor requires task=energy_regression")
    if model_name == "GlobalEnergySkipCNN" and task not in {
        "binary_classification",
        "energy_regression",
    }:
        raise ValueError(
            "GlobalEnergySkipCNN requires task=binary_classification or "
            "task=energy_regression"
        )
    if model_name == "ResidualSpatialNextCNN" and task not in {
        "binary_classification",
        "energy_regression",
    }:
        raise ValueError(
            "ResidualSpatialNextCNN requires task=binary_classification or "
            "task=energy_regression"
        )
    if (
        model_name == "ResidualSpatialEnergyRegressor"
        and task != "energy_regression"
    ):
        raise ValueError(
            "ResidualSpatialEnergyRegressor requires task=energy_regression"
        )
    if model_name in {"SimpleNextEnergyRegressor", "MultiTaskNextCNN"} or (
        model_name == "GlobalEnergySkipCNN" and task == "energy_regression"
    ) or model_name == "ResidualSpatialEnergyRegressor" or (
        model_name == "ResidualSpatialNextCNN" and task == "energy_regression"
    ):
        _energy_target_config(checkpoint)
    return checkpoint


def _energy_target_config(checkpoint: Mapping[str, Any]) -> Dict[str, Any]:
    """Return and validate the physical-energy contract of a regression run."""

    config = checkpoint.get("energy_target_config")
    if not isinstance(config, Mapping):
        raise ValueError(
            "energy regression checkpoint requires an energy_target_config mapping"
        )
    required = {"kind", "unit", "source", "derivation", "normalizer"}
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(
            "energy_target_config is missing fields: %s" % ", ".join(missing)
        )
    for key in ("kind", "unit", "source", "derivation"):
        if not str(config[key]).strip():
            raise ValueError("energy_target_config.%s cannot be empty" % key)
    if str(config["unit"]) != "MeV":
        raise ValueError(
            "energy_target_config.unit must be MeV because "
            "energy_pred is exported in physical MeV"
        )
    normalizer = config["normalizer"]
    if not isinstance(normalizer, Mapping):
        raise ValueError("energy_target_config.normalizer must be a mapping")
    normalizer_required = {"transform", "mean", "std", "fit_split"}
    normalizer_missing = sorted(normalizer_required.difference(normalizer))
    if normalizer_missing:
        raise ValueError(
            "energy_target_config.normalizer is missing fields: %s"
            % ", ".join(normalizer_missing)
        )
    if str(normalizer["transform"]) != "standardize":
        raise ValueError(
            "energy_target_config.normalizer.transform must be standardize"
        )
    if str(normalizer["fit_split"]) != "train":
        raise ValueError(
            "energy_target_config.normalizer.fit_split must be train"
        )
    try:
        normalizer_mean = float(normalizer["mean"])
        normalizer_std = float(normalizer["std"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "energy_target_config.normalizer mean/std must be numeric"
        ) from exc
    if not np.isfinite(normalizer_mean) or not np.isfinite(normalizer_std):
        raise ValueError(
            "energy_target_config.normalizer mean/std must be finite"
        )
    if normalizer_std <= 0.0:
        raise ValueError("energy_target_config.normalizer.std must be positive")

    model_name = str(checkpoint.get("model_name", ""))
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, Mapping):
        raise ValueError("energy regression model_config must be a mapping")
    model_normalizer_keys = None
    if model_name in {"MultiTaskNextCNN", "ResidualSpatialEnergyRegressor"}:
        model_normalizer_keys = ("energy_mean", "energy_std")
    elif model_name == "GlobalEnergySkipCNN":
        model_normalizer_keys = ("global_center", "global_scale")
    if model_normalizer_keys is not None:
        missing_model_keys = sorted(set(model_normalizer_keys).difference(model_config))
        if missing_model_keys:
            raise ValueError(
                "%s model_config is missing fields: %s"
                % (model_name, ", ".join(missing_model_keys))
            )
        try:
            model_mean = float(model_config[model_normalizer_keys[0]])
            model_std = float(model_config[model_normalizer_keys[1]])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "%s model normalizer values must be numeric" % model_name
            ) from exc
        if not np.isclose(
            normalizer_mean, model_mean, rtol=1.0e-6, atol=1.0e-8
        ) or not np.isclose(
            normalizer_std, model_std, rtol=1.0e-6, atol=1.0e-8
        ):
            raise ValueError(
                "energy_target_config.normalizer mean/std differ from the "
                "%s model configuration" % model_name
            )
    return deepcopy(dict(config))


def _build_model(checkpoint: Mapping[str, Any], model_module: Any) -> tuple:
    """Instantiate the checkpoint-declared model without changing v1 behavior."""

    model_name = str(checkpoint.get("model_name", "SimpleNextCNN"))
    if model_name == "SimpleNextCNN":
        model_class = model_module.SimpleNextCNN
        task = "classification"
    elif model_name == "SimpleNextEnergyRegressor":
        model_class = model_module.SimpleNextEnergyRegressor
        task = "regression"
    elif model_name == "GlobalEnergySkipCNN":
        model_class = model_module.GlobalEnergySkipCNN
        task = (
            "regression"
            if str(checkpoint.get("task", "")).lower() == "energy_regression"
            else "classification"
        )
    elif model_name == "MultiTaskNextCNN":
        try:
            model_class = model_module.MultiTaskNextCNN
        except AttributeError as exc:
            raise RuntimeError(
                "MultiTaskNextCNN checkpoint requires a next_cnn.model version "
                "that provides MultiTaskNextCNN"
            ) from exc
        task = "multitask"
    elif model_name == "ResidualSpatialNextCNN":
        try:
            model_class = model_module.ResidualSpatialNextCNN
        except AttributeError as exc:
            raise RuntimeError(
                "ResidualSpatialNextCNN checkpoint requires a next_cnn.model "
                "version that provides ResidualSpatialNextCNN"
            ) from exc
        task = (
            "regression"
            if str(checkpoint.get("task", "")).lower() == "energy_regression"
            else "classification"
        )
    elif model_name == "ResidualSpatialEnergyRegressor":
        try:
            model_class = model_module.ResidualSpatialEnergyRegressor
        except AttributeError as exc:
            raise RuntimeError(
                "ResidualSpatialEnergyRegressor checkpoint requires a "
                "next_cnn.model version that provides the regression model"
            ) from exc
        task = "regression"
    else:  # Guarded by _load_checkpoint; retained for direct helper use.
        raise ValueError("unsupported NEXT CNN model_name %r" % model_name)
    return model_class(**checkpoint["model_config"]), model_name, task


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


def _progress_status(enabled: bool, message: str) -> None:
    if enabled:
        print("NEXT predict: %s" % message, file=sys.stderr, flush=True)


class _InferenceProgress:
    """TTY progress bar with a line-oriented fallback for captured logs."""

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
        is_terminal = bool(getattr(sys.stderr, "isatty", lambda: False)())
        if is_terminal:
            try:
                from tqdm import tqdm
            except ImportError:
                pass
            else:
                self.bar = tqdm(
                    total=self.total_files,
                    desc="NEXT %s inference" % self.split,
                    unit="file",
                    file=sys.stderr,
                    dynamic_ncols=True,
                    mininterval=0.5,
                )
        if self.bar is None:
            self._write_line("started")

    def _snapshot(self) -> tuple:
        elapsed = max(time.monotonic() - self.started_at, 1e-12)
        return elapsed, self.events / elapsed

    def _write_line(self, status: str) -> None:
        elapsed, event_rate = self._snapshot()
        percentage = (
            100.0 * self.completed_files / self.total_files
            if self.total_files
            else 100.0
        )
        print(
            "NEXT %s inference %s: files=%d/%d (%.1f%%), events=%d, "
            "elapsed=%.1fs, rate=%.1f event/s, device=%s"
            % (
                self.split,
                status,
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
        elapsed, event_rate = self._snapshot()
        if self.bar is not None:
            self.bar.set_postfix(
                events=self.events,
                **{"event/s": "%.1f" % event_rate, "device": self.device},
                refresh=False,
            )
            self.bar.update(increment)
            return
        now = time.monotonic()
        if now - self.last_output_at >= self.interval_seconds:
            self._write_line("running")

    def close(self, success: bool) -> None:
        if not self.enabled:
            return
        status = "complete" if success else "stopped"
        if self.bar is not None:
            elapsed, event_rate = self._snapshot()
            self.bar.set_postfix(
                events=self.events,
                **{
                    "event/s": "%.1f" % event_rate,
                    "device": self.device,
                    "status": status,
                },
                refresh=True,
            )
            self.bar.close()
        else:
            self._write_line(status)


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
    **kwargs: Any,
) -> Iterator[Dict[str, Any]]:
    """Run inference and yield canonical, event-aligned prediction batches."""

    if kwargs:
        raise TypeError(
            "unsupported NEXT CNN adapter arguments: %s"
            % ", ".join(sorted(kwargs))
        )
    if not isinstance(show_progress, (bool, np.bool_)):
        raise ValueError("show_progress must be true or false")
    show_progress = bool(show_progress)
    try:
        progress_interval_seconds = float(progress_interval_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "progress_interval_seconds must be finite and positive"
        ) from exc
    if (
        not np.isfinite(progress_interval_seconds)
        or progress_interval_seconds <= 0
    ):
        raise ValueError("progress_interval_seconds must be finite and positive")
    checkpoint_path = Path(model_path).expanduser().resolve()
    _progress_status(
        show_progress,
        "loading PyTorch and checkpoint %s" % checkpoint_path,
    )
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "NEXT CNN inference requires PyTorch; "
            "install `requirements/next-cnn-cu128.txt` in the GPU environment"
        ) from exc
    if int(batch_size) != batch_size or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if int(num_workers) != num_workers or num_workers < 0:
        raise ValueError("num_workers must be a non-negative integer")
    raw_checkpoint = _read_checkpoint(checkpoint_path, "cpu", torch)
    if raw_checkpoint.get("format_version") == 3:
        from next_alt.adapter import predict as alternative_predict

        return alternative_predict(
            model_path=model_path,
            data_path=data_path,
            batch_size=batch_size,
            device=device,
            num_workers=num_workers,
            split=split,
            max_files_per_class=max_files_per_class,
            include_energy_condition=include_energy_condition,
            show_progress=show_progress,
            progress_interval_seconds=progress_interval_seconds,
            split_seed=split_seed,
            split_fractions=split_fractions,
            _checkpoint_payload=raw_checkpoint,
        )

    from . import model as model_module

    selected_device = _device(device, torch)
    checkpoint = _load_checkpoint(
        checkpoint_path,
        selected_device,
        torch,
        payload=raw_checkpoint,
    )
    checkpoint_split = checkpoint["split_config"]
    checkpoint_seed = int(checkpoint_split["seed"])
    checkpoint_fractions = np.asarray(checkpoint_split["fractions"], dtype=float)
    if split_seed is not None and int(split_seed) != checkpoint_seed:
        raise ValueError(
            "split_seed differs from the checkpoint; refusing a potentially leaky split"
        )
    if split_fractions is not None and not np.allclose(
        np.asarray(split_fractions, dtype=float),
        checkpoint_fractions,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "split_fractions differ from the checkpoint; "
            "refusing a potentially leaky split"
        )
    actual_seed = checkpoint_seed
    actual_fractions = checkpoint_fractions.tolist()
    selection = checkpoint["data_selection"]
    expected_inventory = selection.get("inventory")
    _progress_status(
        show_progress,
        "validating the extracted dataset inventory under %s" % data_path,
    )
    actual_inventory = dataset_inventory(data_path)
    if actual_inventory != expected_inventory:
        raise ValueError(
            "NEXT data inventory differs from the training checkpoint: "
            "expected %r, found %r" % (expected_inventory, actual_inventory)
        )
    maximum = None if int(max_files_per_class) <= 0 else int(max_files_per_class)
    files = discover_source_files(
        root=data_path,
        split=split,
        split_seed=actual_seed,
        split_fractions=actual_fractions,
        max_files_per_class=maximum,
    )
    prediction_groups = {item.group_id for item in files}
    training_groups = set(selection.get("training_groups", []))
    validation_groups = set(selection.get("validation_groups", []))
    forbidden_groups = training_groups
    if split == "test":
        forbidden_groups = forbidden_groups.union(validation_groups)
    overlap = sorted(prediction_groups.intersection(forbidden_groups))
    if overlap:
        raise ValueError(
            "prediction split overlaps checkpoint selection data; examples: %s"
            % ", ".join(overlap[:5])
        )
    projection = ProjectionConfig.from_dict(checkpoint["projection_config"])
    model, model_name, model_task = _build_model(checkpoint, model_module)
    classification = model_task in {"classification", "multitask"}
    regression = model_task in {"regression", "multitask"}
    dataset = NextIterableDataset(
        files,
        projection=projection,
        shuffle_files=False,
        include_classification_metadata=classification,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        pin_memory=selected_device.type == "cuda",
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(selected_device).eval()
    checkpoint_digest = _file_sha256(checkpoint_path)
    _progress_status(
        show_progress,
        "selected %d %s HDF5 files; starting inference on %s"
        % (len(files), split, selected_device),
    )
    metadata = {
        "framework": "pytorch",
        "model_name": model_name,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_digest,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "dataset_root": str(Path(data_path).expanduser().resolve()),
        "data_split": split,
        "split_seed": actual_seed,
        "split_fractions": list(actual_fractions),
        "dataset_inventory": actual_inventory,
        "source_file_count": len(files),
        "projection": projection.to_dict(),
        "energy_condition_derivation": (
            "float64 sum of MC/hits/table values_block_1 energy rows per event_id"
        ),
        "energy_unit": "MeV",
        "energy_unit_source": (
            "NEXT zeronu benchmark data paper arXiv:2605.07566; "
            "source HDF5 attributes do not declare a unit"
        ),
        "energy_unit_reference": "https://arxiv.org/abs/2605.07566",
    }
    if classification:
        metadata.update(
            {
                "score_space": "logit",
                "positive_class": "0nubb",
            }
        )
    target_config = None
    if regression:
        target_config = _energy_target_config(checkpoint)
        metadata.update(
            {
                "energy_target_kind": target_config["kind"],
                "energy_target_unit": target_config["unit"],
                "energy_target_source": target_config["source"],
                "energy_target_derivation": target_config["derivation"],
                "energy_target_normalizer": target_config["normalizer"],
                "energy_prediction_kind": target_config["kind"],
                "energy_prediction_unit": target_config["unit"],
                "energy_prediction_space": "physical",
                "energy_target_config": target_config,
            }
        )

    def batches() -> Iterator[Dict[str, Any]]:
        produced = 0
        completed_files = 0
        success = False
        progress = _InferenceProgress(
            total_files=len(files),
            split=split,
            device=str(selected_device),
            enabled=show_progress,
            interval_seconds=progress_interval_seconds,
        )
        try:
            with torch.inference_mode():
                for batch in loader:
                    images = batch["image"].to(
                        device=selected_device,
                        dtype=torch.float32,
                        non_blocking=True,
                    )
                    energy_prediction = None
                    logits = None
                    if model_task == "multitask":
                        logits, parts = model(images, return_parts=True)
                        if not isinstance(parts, Mapping):
                            raise TypeError(
                                "MultiTaskNextCNN return_parts must be a mapping"
                            )
                        if "energy_pred" not in parts:
                            raise KeyError(
                                "MultiTaskNextCNN return_parts is missing energy_pred"
                            )
                        energy_tensor = parts["energy_pred"]
                        if energy_tensor.ndim != 1:
                            raise ValueError(
                                "MultiTaskNextCNN energy_pred must have shape (batch,)"
                            )
                        energy_prediction = (
                            energy_tensor.detach().cpu().numpy().astype(np.float64)
                        )
                    elif model_task == "regression":
                        standardized_energy = model(images)
                        if standardized_energy.ndim != 1:
                            raise ValueError(
                                "NEXT CNN energy output must have shape (batch,)"
                            )
                        normalizer = target_config["normalizer"]
                        energy_tensor = (
                            float(normalizer["mean"])
                            + float(normalizer["std"]) * standardized_energy
                        )
                        energy_prediction = (
                            energy_tensor.detach().cpu().numpy().astype(np.float64)
                        )
                    else:
                        logits = model(images)
                    if logits is not None and logits.ndim != 1:
                        raise ValueError("NEXT CNN logits must have shape (batch,)")
                    score = None
                    if logits is not None:
                        score = logits.detach().cpu().numpy().astype(np.float32)
                    size = len(batch["event_id"])
                    if score is not None and len(score) != size:
                        raise ValueError(
                            "NEXT CNN logits are not aligned with the input batch"
                        )
                    if energy_prediction is not None and len(energy_prediction) != size:
                        raise ValueError(
                            "NEXT CNN energy_pred is not aligned with the input batch"
                        )
                    columns: Dict[str, Any] = {
                        "event_id": np.asarray(batch["event_id"], dtype=str),
                        "sample_weight": np.ones(size, dtype=np.float32),
                        "split": np.asarray(batch["split"], dtype=str),
                        "group_id": np.asarray(batch["group_id"], dtype=str),
                        "projection_coverage": (
                            batch["projection_coverage"]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                        ),
                        "__metadata__": metadata,
                    }
                    if classification:
                        if "label" not in batch or "category" not in batch:
                            raise KeyError(
                                "classification inference requires label/category"
                            )
                        columns["label"] = (
                            batch["label"]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.int8)
                        )
                        columns["category"] = np.asarray(
                            batch["category"], dtype=str
                        )
                    if score is not None:
                        columns["score"] = score
                    if include_energy_condition:
                        columns["energy_condition"] = (
                            batch["energy_condition"]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float64)
                        )
                    if regression:
                        if "energy_target" not in batch:
                            raise KeyError(
                                "energy regression inference batch is missing energy_target"
                            )
                        energy_target = (
                            batch["energy_target"]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float64)
                        )
                        if energy_target.ndim != 1 or len(energy_target) != size:
                            raise ValueError(
                                "energy_target must have shape (batch,) and align with events"
                            )
                        if not np.all(np.isfinite(energy_target)):
                            raise ValueError("energy_target must contain only finite values")
                        columns["energy_target"] = energy_target
                        columns["energy_pred"] = energy_prediction
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
                raise RuntimeError("NEXT CNN inference produced no events")
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
