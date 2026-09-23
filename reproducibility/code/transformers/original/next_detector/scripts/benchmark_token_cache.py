#!/usr/bin/env python3
"""Compare raw-tokenization and cached NEXT DataLoader throughput."""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import islice
from pathlib import Path


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup-batches", type=int, default=20)
    parser.add_argument("--batches", type=int, default=500)
    return parser.parse_args()


def _measure(loader, warmup_batches: int, measured_batches: int) -> dict:
    iterator = iter(loader)
    for _ in islice(iterator, warmup_batches):
        pass
    start = time.perf_counter()
    batches = 0
    events = 0
    for batch in islice(iterator, measured_batches):
        batches += 1
        events += int(len(batch["label"]))
    seconds = time.perf_counter() - start
    if batches != measured_batches:
        raise RuntimeError(
            f"loader produced only {batches} measured batches; "
            f"requested {measured_batches}"
        )
    return {
        "batches": batches,
        "events": events,
        "seconds": seconds,
        "batches_per_second": batches / seconds,
        "events_per_second": events / seconds,
    }


def main() -> int:
    arguments = _arguments()
    next_root = Path(__file__).resolve().parents[1]
    project_root = next_root.parent
    for path in (project_root / "evalutaions_workflow", next_root):
        if not path.is_dir():
            raise FileNotFoundError(f"required source directory is missing: {path}")
        sys.path.insert(0, str(path))

    from simple_energybench import prepare_dataset
    from next_transformer import (
        NEXTTokenBuilder,
        TokenizationConfig,
        find_token_cache,
        prepare_cached_dataset,
    )

    config = TokenizationConfig(
        tokenization=arguments.tokenization,
        max_tokens=arguments.max_tokens,
        voxel_size=arguments.voxel_size,
        coordinate_scale=arguments.coordinate_scale,
        center_coordinates=arguments.center_coordinates,
        voxel_truncation=arguments.voxel_truncation,
        seed=arguments.seed,
    )
    cache_dir = find_token_cache(
        arguments.cache_root,
        arguments.data_root,
        arguments.split_manifest,
        config,
    )
    raw = prepare_dataset(
        arguments.data_root,
        batch_size=arguments.batch_size,
        mode="classification",
        split_fractions=(0.8, 0.1, 0.1),
        seed=arguments.seed,
        num_workers=arguments.workers,
        manifest_path=arguments.split_manifest,
        max_files_per_class=None,
        verbose=False,
        input_builder=NEXTTokenBuilder(config),
    )
    cached = prepare_cached_dataset(
        cache_dir,
        batch_size=arguments.batch_size,
        num_workers=arguments.workers,
        seed=arguments.seed,
        pin_memory=True,
    )
    raw_result = _measure(
        raw.train_loader, arguments.warmup_batches, arguments.batches
    )
    cached_result = _measure(
        cached.train_loader, arguments.warmup_batches, arguments.batches
    )
    report = {
        "tokenization": arguments.tokenization,
        "cache_dir": str(cache_dir),
        "workers": arguments.workers,
        "batch_size": arguments.batch_size,
        "raw": raw_result,
        "cached": cached_result,
        "speedup": (
            cached_result["events_per_second"] / raw_result["events_per_second"]
        ),
        "passes_1_5x_gate": (
            cached_result["events_per_second"]
            >= 1.5 * raw_result["events_per_second"]
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
