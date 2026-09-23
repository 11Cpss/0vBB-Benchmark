#!/usr/bin/env python3
"""Build and validate a persistent NEXT Transformer token cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute deterministic NEXT Transformer tokens into "
            "uncompressed NumPy memory maps."
        )
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--tokenization",
        choices=("sampled_hits", "voxel"),
        required=True,
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--voxel-size", type=float, default=15.0)
    parser.add_argument("--coordinate-scale", type=float, default=1000.0)
    parser.add_argument(
        "--center-coordinates",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--voxel-truncation",
        choices=("occupancy", "energy"),
        default="occupancy",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = _parse_arguments()

    # The script lives at next_detector/scripts/build_token_cache.py.  Add the
    # next_detector package root and the sibling workflow copy explicitly so
    # it works when launched from the project root on the lab computer.
    next_root = Path(__file__).resolve().parents[1]
    project_root = next_root.parent
    workflow_root = project_root / "evalutaions_workflow"
    for path in (workflow_root, next_root):
        if not path.is_dir():
            raise FileNotFoundError(f"required source directory is missing: {path}")
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)

    from next_transformer import (  # pylint: disable=import-outside-toplevel
        TokenizationConfig,
        build_token_cache,
        validate_token_cache,
    )

    configuration = TokenizationConfig(
        tokenization=arguments.tokenization,
        max_tokens=arguments.max_tokens,
        voxel_size=arguments.voxel_size,
        coordinate_scale=arguments.coordinate_scale,
        center_coordinates=arguments.center_coordinates,
        voxel_truncation=arguments.voxel_truncation,
        seed=arguments.seed,
    )
    cache_path = build_token_cache(
        arguments.data_root,
        arguments.split_manifest,
        arguments.cache_root,
        configuration,
        num_workers=arguments.workers,
        resume=arguments.resume,
        overwrite=arguments.overwrite,
    )
    report = validate_token_cache(
        cache_path,
        arguments.data_root,
        arguments.split_manifest,
        configuration,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
