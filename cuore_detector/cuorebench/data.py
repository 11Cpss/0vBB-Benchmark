"""CUORE waveform dataset for first-two-pulse Δt regression.

Pipeline for each event:

1. read ``/Waveform`` (raw ADC) plus the per-event normalization arrays and the
   ``/pulseFinder`` products from the CUORE HDF5 file;
2. normalize: ``norm = clip((Waveform - offset) / (scale + eps), 0, 1)``;
3. target: ``Δt_ms = pulseFinder/dtBetweenPeaks[:, 0]`` (gap between the first
   two detected pulses); single-pulse events (``nPulses < 2``) get ``Δt = 0``;
4. scale the target into the canonical EnergyBench ``[0, 3]`` range via
   ``y = Δt_ms / dt_scale`` so ``evaluate_regression`` accepts it.

Tokenization is NOT done here -- the MJD tokenizers live inside the model and
consume the raw waveform, so batches carry dense waveforms::

    {"inputs": FloatTensor[1, L],   # -> [B, 1, L] after collation
     "energy": <scaled Δt>,         # regression target key read by train_model
     "event_id": <unique str>,
     "sample_weight": <float>,      # always present -> weighted loss + metrics
     "category": "pileup" | "clean"}

Nothing in ``energybench``, ``mjd_detector``, or ``next_detector`` is modified.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import (
    CANONICAL_SCALED_MAX,
    CuoreDataConfig,
    scale_target,
)

_NORMALIZATION_KEYS = (
    "normalizationOffset",
    "normalizationScale",
    "normalizationMaximum",
)


def _read_cuore_file(path: Path) -> dict[str, Any]:
    """Read one CUORE HDF5 file. ``Labels`` is optional (test file has none)."""

    if not path.is_file():
        raise FileNotFoundError(f"CUORE file not found: {path}")

    with h5py.File(path, "r") as handle:
        if "Waveform" not in handle:
            raise KeyError(f"{path} has no /Waveform dataset")
        waveform = np.asarray(handle["Waveform"][()], dtype=np.float64)
        if waveform.ndim != 2:
            raise ValueError(f"/Waveform in {path} must be 2-D [events, samples]")
        n_events = int(waveform.shape[0])

        normalization: dict[str, np.ndarray] = {}
        for key in _NORMALIZATION_KEYS:
            if key not in handle:
                raise KeyError(f"{path} has no /{key} dataset")
            values = np.asarray(handle[key][()], dtype=np.float64).reshape(-1)
            if values.shape[0] != n_events:
                raise ValueError(f"/{key} in {path} has the wrong length")
            normalization[key] = values

        if "pulseFinder" not in handle:
            raise KeyError(f"{path} has no /pulseFinder group")
        finder = handle["pulseFinder"]
        n_pulses = np.asarray(finder["nPulses"][()], dtype=np.int64).reshape(-1)
        dt_between_peaks = np.asarray(
            finder["dtBetweenPeaks"][()], dtype=np.float64
        )
        if dt_between_peaks.ndim != 2 or dt_between_peaks.shape[1] < 1:
            raise ValueError(
                f"/pulseFinder/dtBetweenPeaks in {path} must be 2-D with at "
                "least one column"
            )
        tail_elevated = np.asarray(
            finder["tailElevated"][()], dtype=bool
        ).reshape(-1)

        event_id = np.asarray(handle["eventId"][()], dtype=np.int64).reshape(-1)

    for name, array in (
        ("nPulses", n_pulses),
        ("tailElevated", tail_elevated),
        ("eventId", event_id),
        ("dtBetweenPeaks", dt_between_peaks),
    ):
        if array.shape[0] != n_events:
            raise ValueError(f"/{name} length mismatch in {path}")

    return {
        "waveform": waveform,
        "offset": normalization["normalizationOffset"],
        "scale": normalization["normalizationScale"],
        "n_pulses": n_pulses,
        "dt_between_peaks": dt_between_peaks,
        "tail_elevated": tail_elevated,
        "n_events": n_events,
    }


def _normalize_waveforms(
    waveform: np.ndarray,
    offset: np.ndarray,
    scale: np.ndarray,
    config: CuoreDataConfig,
) -> np.ndarray:
    """Per-event normalization from the CUORE file attributes -> float32."""

    normalized = (waveform - offset[:, None]) / (scale[:, None] + config.norm_eps)
    if config.clip_normalized:
        normalized = np.clip(normalized, 0.0, 1.0)

    normalized = normalized.astype(np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError("normalized waveforms contain non-finite values")
    return normalized


def _extract_targets(
    n_pulses: np.ndarray,
    dt_between_peaks: np.ndarray,
    config: CuoreDataConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(dt_ms[float64], has_second_pulse[bool])``.

    Single-pulse events (or events whose first gap is not finite) get
    ``Δt = config.missing_dt_fill`` (0 by default).
    """

    first_gap = np.asarray(dt_between_peaks[:, 0], dtype=np.float64)
    has_second_pulse = np.isfinite(first_gap) & (n_pulses >= 2)

    dt_ms = np.where(
        has_second_pulse, first_gap, float(config.missing_dt_fill)
    ).astype(np.float64)

    if not np.isfinite(dt_ms).all():
        raise ValueError("extracted Δt targets contain non-finite values")
    if np.any(dt_ms < 0.0):
        raise ValueError("extracted Δt targets contain negative values")

    return dt_ms, has_second_pulse


def _compute_sample_weights(
    has_second_pulse: np.ndarray,
    tail_elevated: np.ndarray,
    config: CuoreDataConfig,
    *,
    is_test: bool,
) -> np.ndarray:
    """Per-event weights. Test weights are always ``base_weight`` (unweighted)."""

    weight = np.full(
        has_second_pulse.shape[0], float(config.base_weight), dtype=np.float64
    )

    if not is_test:
        weight *= np.where(tail_elevated, float(config.tailelevated_weight), 1.0)
        weight *= np.where(has_second_pulse, 1.0, float(config.zero_dt_weight))

    if config.weight_normalization == "mean_one":
        mean = float(weight.mean())
        if mean > 0.0:
            weight = weight / mean

    weight = weight.astype(np.float32)
    if (
        not np.isfinite(weight).all()
        or np.any(weight < 0.0)
        or float(weight.sum()) <= 0.0
    ):
        raise ValueError(
            "sample weights must be finite, non-negative, and sum to > 0"
        )
    return weight


class CuoreRegressionDataset(Dataset):
    """Map-style dataset of normalized waveforms + scaled Δt targets."""

    def __init__(
        self,
        *,
        waveforms: np.ndarray,
        scaled_dt: np.ndarray,
        event_ids: list[str],
        sample_weight: np.ndarray,
        has_second_pulse: np.ndarray,
    ) -> None:
        n_events = int(waveforms.shape[0])
        if waveforms.ndim != 2:
            raise ValueError("waveforms must have shape [events, samples]")
        if not (
            scaled_dt.shape[0] == n_events
            and len(event_ids) == n_events
            and sample_weight.shape[0] == n_events
            and has_second_pulse.shape[0] == n_events
        ):
            raise ValueError(
                "CuoreRegressionDataset arrays must share the first dimension"
            )

        self.waveforms = np.ascontiguousarray(waveforms, dtype=np.float32)
        self.scaled_dt = np.ascontiguousarray(scaled_dt, dtype=np.float64)
        self.event_ids = list(event_ids)
        self.sample_weight = np.ascontiguousarray(sample_weight, dtype=np.float32)
        self.has_second_pulse = np.ascontiguousarray(has_second_pulse, dtype=bool)

    def __len__(self) -> int:
        return int(self.waveforms.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        waveform = torch.from_numpy(self.waveforms[index]).unsqueeze(0)
        return {
            "inputs": waveform,
            "energy": np.float64(self.scaled_dt[index]),
            "event_id": self.event_ids[index],
            "sample_weight": np.float32(self.sample_weight[index]),
            "category": (
                "pileup" if bool(self.has_second_pulse[index]) else "clean"
            ),
        }


def _split_stats(
    dt_ms: np.ndarray,
    has_second_pulse: np.ndarray,
    sample_weight: np.ndarray,
) -> dict[str, Any]:
    count = int(dt_ms.shape[0])
    return {
        "n_events": count,
        "has_second_pulse_fraction": (
            float(np.mean(has_second_pulse)) if count else 0.0
        ),
        "dt_ms_stats": {
            "mean": float(np.mean(dt_ms)) if count else 0.0,
            "std": float(np.std(dt_ms)) if count else 0.0,
            "min": float(np.min(dt_ms)) if count else 0.0,
            "max": float(np.max(dt_ms)) if count else 0.0,
        },
        "sample_weight_stats": {
            "min": float(np.min(sample_weight)),
            "mean": float(np.mean(sample_weight)),
            "max": float(np.max(sample_weight)),
            "sum": float(np.sum(sample_weight)),
            "downweighted": int(np.sum(sample_weight < 1.0)),
        },
    }


def prepare_cuore_regression_data(
    config: CuoreDataConfig,
    training_config: Any,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, Any]]:
    """Build ``(train_loader, val_loader, test_loader, meta)``.

    ``training_config`` only needs ``batch_size`` and ``num_workers`` attributes
    (a ``simple_energybench.TrainingConfig`` satisfies this).
    """

    data_dir = Path(config.data_dir).expanduser().resolve()
    train_raw = _read_cuore_file(data_dir / config.train_file)
    test_raw = _read_cuore_file(data_dir / config.test_file)

    for name, raw in (("train", train_raw), ("test", test_raw)):
        samples = int(raw["waveform"].shape[1])
        if samples != config.waveform_length:
            raise ValueError(
                f"the {name} file has {samples}-sample waveforms but "
                f"waveform_length={config.waveform_length}"
            )

    train_norm = _normalize_waveforms(
        train_raw["waveform"], train_raw["offset"], train_raw["scale"], config
    )
    test_norm = _normalize_waveforms(
        test_raw["waveform"], test_raw["offset"], test_raw["scale"], config
    )
    # Free the raw float64 waveforms (~720 MB for the training file).
    train_raw["waveform"] = None
    test_raw["waveform"] = None

    train_dt, train_has2 = _extract_targets(
        train_raw["n_pulses"], train_raw["dt_between_peaks"], config
    )
    test_dt, test_has2 = _extract_targets(
        test_raw["n_pulses"], test_raw["dt_between_peaks"], config
    )

    # ---- shuffle before splitting (rows are clean-block then pileup-block) ----
    rng = np.random.default_rng(config.split_seed)
    permutation = rng.permutation(train_raw["n_events"])

    dropped_tailelevated = config.tailelevated_weight == 0.0
    if dropped_tailelevated:
        permutation = permutation[~train_raw["tail_elevated"][permutation]]
    if permutation.shape[0] < 2:
        raise ValueError("training pool has fewer than 2 events after filtering")

    n_val = int(round(config.val_fraction * permutation.shape[0]))
    n_val = max(1, min(permutation.shape[0] - 1, n_val))
    train_index = permutation[:-n_val]
    val_index = permutation[-n_val:]

    # The test file is also stored clean-block then pileup-block, so shuffle it
    # too -- otherwise a ``max_events`` cap would see only one regime.
    test_index = np.random.default_rng(config.split_seed + 1).permutation(
        test_raw["n_events"]
    )

    if config.max_events is not None:
        train_index = train_index[: config.max_events]
        val_cap = max(
            1,
            int(
                round(
                    config.max_events
                    * config.val_fraction
                    / (1.0 - config.val_fraction)
                )
            ),
        )
        val_index = val_index[:val_cap]
        test_index = test_index[: config.max_events]

    splits = {
        "train": (train_norm, train_dt, train_has2,
                  train_raw["tail_elevated"], train_index, False),
        "val": (train_norm, train_dt, train_has2,
                train_raw["tail_elevated"], val_index, False),
        "test": (test_norm, test_dt, test_has2,
                 test_raw["tail_elevated"], test_index, True),
    }

    # ---- guard the canonical [0, 3] target band across every split ----
    observed_lo = math.inf
    observed_hi = -math.inf
    for _, dt_values, _, _, index, _ in splits.values():
        scaled = scale_target(dt_values[index], config)
        observed_lo = min(observed_lo, float(scaled.min()))
        observed_hi = max(observed_hi, float(scaled.max()))
    if observed_lo < 0.0 or observed_hi > CANONICAL_SCALED_MAX:
        raise ValueError(
            "scaled Δt target falls outside the canonical EnergyBench range "
            f"[0, {CANONICAL_SCALED_MAX}]: observed "
            f"[{observed_lo:.6g}, {observed_hi:.6g}]. Increase dt_scale."
        )

    datasets: dict[str, CuoreRegressionDataset] = {}
    split_meta: dict[str, Any] = {}
    for name, (norm, dt_values, has2, tail, index, is_test) in splits.items():
        if index.shape[0] == 0:
            raise ValueError(f"the {name} split is empty")

        weights = _compute_sample_weights(
            has2[index], tail[index], config, is_test=is_test
        )
        datasets[name] = CuoreRegressionDataset(
            waveforms=norm[index],
            scaled_dt=scale_target(dt_values[index], config),
            event_ids=[f"cuore-{name}-{int(row)}" for row in index],
            sample_weight=weights,
            has_second_pulse=has2[index],
        )
        split_meta[name] = _split_stats(dt_values[index], has2[index], weights)

    batch_size = int(getattr(training_config, "batch_size", 64))
    num_workers = int(getattr(training_config, "num_workers", 0))
    loaders = {
        name: DataLoader(
            datasets[name],
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
            drop_last=False,
        )
        for name in ("train", "val", "test")
    }

    meta = {
        "dt_scale": float(config.dt_scale),
        "scaled_target_range": (observed_lo, observed_hi),
        "waveform_length": int(config.waveform_length),
        "n_train": split_meta["train"]["n_events"],
        "n_val": split_meta["val"]["n_events"],
        "n_test": split_meta["test"]["n_events"],
        "train_weight_sum": split_meta["train"]["sample_weight_stats"]["sum"],
        "train_dt_ms_stats": split_meta["train"]["dt_ms_stats"],
        "has_second_pulse_fraction": {
            name: split_meta[name]["has_second_pulse_fraction"]
            for name in ("train", "val", "test")
        },
        "dropped_tailelevated_from_train_val": bool(dropped_tailelevated),
        "splits": split_meta,
        "data_config": config.to_dict(),
    }
    return loaders["train"], loaders["val"], loaders["test"], meta


__all__ = [
    "CuoreRegressionDataset",
    "prepare_cuore_regression_data",
]
