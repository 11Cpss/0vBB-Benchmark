"""Validation helpers for the frozen EXO-200 v1 benchmark split."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_SPLIT_MANIFEST = (
    Path(__file__).resolve().parents[1] / "manifests" / "exo200_v1_split.json"
)


def load_split_manifest(path: str | Path | None = None) -> dict[str, Any]:
    """Load the versioned split contract committed with this benchmark."""

    manifest_path = Path(
        DEFAULT_SPLIT_MANIFEST if path is None else path
    ).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"EXO-200 split manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported EXO-200 split manifest schema")
    return payload


def validate_split_manifest(
    prepared_data: Any,
    path: str | Path | None = None,
    *,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    """Fail if prepared loaders do not match the official v1 split contract."""

    manifest = load_split_manifest(path)
    comparisons = {
        "counts": prepared_data.counts,
        "class_counts": prepared_data.class_counts,
        "runs": prepared_data.runs,
        "overlap_counts": prepared_data.overlap_counts,
    }
    for name, actual in comparisons.items():
        expected = manifest[name]
        if actual != expected:
            raise RuntimeError(
                f"EXO-200 v1 {name} differ from the frozen manifest: "
                f"expected {expected}, found {actual}"
            )
    if data_root is not None:
        root = Path(data_root).expanduser().resolve()
        actual_files = sorted(item.name for item in root.glob("*.h5"))
        expected_files = manifest["expected_files"]
        if actual_files != expected_files:
            missing = sorted(set(expected_files) - set(actual_files))
            unexpected = sorted(set(actual_files) - set(expected_files))
            raise RuntimeError(
                "EXO-200 v1 file inventory differs from the frozen manifest: "
                f"missing={missing}, unexpected={unexpected}"
            )
    return manifest


__all__ = [
    "DEFAULT_SPLIT_MANIFEST",
    "load_split_manifest",
    "validate_split_manifest",
]
