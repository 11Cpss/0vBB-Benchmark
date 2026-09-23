#!/usr/bin/env python3
"""Measure dynamic voxel-padding reduction over one cached training epoch."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the cached voxel training loader and compare its dynamic "
            "batch lengths with a fixed maximum-token baseline."
        )
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--epoch",
        type=int,
        default=0,
        help="Zero-based epoch used for deterministic file-slice ordering.",
    )
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument(
        "--allow-tokenizer-source-mismatch",
        action="store_true",
        help=(
            "Measure the immutable stored masks even if tokenization.py was "
            "edited after cache construction. This bypass is measurement-only."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path at which to save the JSON report.",
    )
    return parser.parse_args()


def _positive(value: int, name: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or value < minimum:
        relation = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {relation} integer")
    return int(value)


def _core_forward_work(
    *,
    batch_size: int,
    tokens: int,
    layers: int,
    d_model: int,
    d_ff: int,
) -> int:
    """Return a transparent Transformer-core operation proxy.

    Per layer, this counts Q/K/V/output projections, two attention matrix
    products, and the two feed-forward projections. It intentionally excludes
    layer normalization, activation, positional encoding, the classifier,
    backward computation, and optimizer work.
    """

    projection_work = 4 * tokens * d_model * d_model
    attention_work = 2 * tokens * tokens * d_model
    feedforward_work = 2 * tokens * d_model * d_ff
    return batch_size * layers * (
        projection_work + attention_work + feedforward_work
    )


def main() -> int:
    arguments = _arguments()
    batch_size = _positive(arguments.batch_size, "batch-size")
    workers = _positive(arguments.workers, "workers", allow_zero=True)
    epoch = _positive(arguments.epoch, "epoch", allow_zero=True)
    heads = _positive(arguments.heads, "heads")
    layers = _positive(arguments.layers, "layers")
    d_model = _positive(arguments.d_model, "d-model")
    d_ff = _positive(arguments.d_ff, "d-ff")

    cache_dir = arguments.cache_dir.expanduser().resolve()
    manifest_path = cache_dir / "cache_manifest.json"
    success_path = cache_dir / "_SUCCESS"
    if not manifest_path.is_file() or not success_path.is_file():
        raise FileNotFoundError(
            f"cache must contain cache_manifest.json and _SUCCESS: {cache_dir}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tokenization_config = dict(manifest["tokenization_config"])
    if tokenization_config.get("tokenization") != "voxel":
        raise ValueError("padding measurement requires a voxel token cache")
    max_tokens = _positive(int(tokenization_config["max_tokens"]), "max-tokens")

    next_root = Path(__file__).resolve().parents[1]
    project_root = next_root.parent
    for path in (project_root / "evalutaions_workflow", next_root):
        if not path.is_dir():
            raise FileNotFoundError(f"required source directory is missing: {path}")
        sys.path.insert(0, str(path))

    from next_transformer import prepare_cached_dataset

    provenance_mode = "fully validated cache loader"
    try:
        prepared = prepare_cached_dataset(
            cache_dir,
            batch_size=batch_size,
            num_workers=workers,
            seed=arguments.seed,
            pin_memory=False,
            trim_padding=True,
            compact_training_batches=True,
        )
        train_loader = prepared.train_loader
        train_dataset = train_loader.dataset
    except ValueError as error:
        source_mismatch = (
            str(error) == "tokenization.py changed after this cache was built"
        )
        if not source_mismatch or not arguments.allow_tokenizer_source_mismatch:
            raise

        # This path intentionally does not rebuild or reinterpret tokens. It
        # replays the immutable mask arrays and original FileSlice layout from
        # the completed cache. It is appropriate for measuring the historical
        # run after unrelated tokenizer strategies changed the source-file hash.
        from torch.utils.data import DataLoader

        from next_transformer.cache import (
            CachedNEXTDataset,
            _CachedBatchCollator,
            _CachedSlice,
        )

        train_dataset = CachedNEXTDataset(
            cache_dir,
            "train",
            [
                _CachedSlice.from_dict(value)
                for value in manifest["slices"]["train"]
            ],
            seed=int(arguments.seed),
            shuffle_slices=True,
            array_specs=manifest["arrays"]["train"],
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=workers,
            pin_memory=False,
            persistent_workers=False,
            drop_last=False,
            collate_fn=_CachedBatchCollator(
                trim_padding=True,
                compact_metadata=True,
            ),
        )
        provenance_mode = (
            "stored-mask replay; tokenizer source hash mismatch explicitly allowed"
        )

    if hasattr(train_dataset, "set_epoch"):
        train_dataset.set_epoch(epoch)

    totals = {
        "batches": 0,
        "events": 0,
        "real_tokens": 0,
        "fixed_token_positions": 0,
        "trimmed_token_positions": 0,
        "fixed_attention_score_positions": 0,
        "trimmed_attention_score_positions": 0,
        "fixed_core_forward_work_proxy": 0,
        "trimmed_core_forward_work_proxy": 0,
    }
    minimum_batch_tokens = max_tokens
    maximum_batch_tokens = 0
    started = time.perf_counter()

    for batch in train_loader:
        mask = batch["inputs"]["mask"]
        current_batch_size = int(mask.shape[0])
        current_tokens = int(mask.shape[1])
        minimum_batch_tokens = min(minimum_batch_tokens, current_tokens)
        maximum_batch_tokens = max(maximum_batch_tokens, current_tokens)

        totals["batches"] += 1
        totals["events"] += current_batch_size
        totals["real_tokens"] += int(mask.sum().item())
        totals["fixed_token_positions"] += current_batch_size * max_tokens
        totals["trimmed_token_positions"] += current_batch_size * current_tokens
        totals["fixed_attention_score_positions"] += (
            current_batch_size * heads * layers * max_tokens * max_tokens
        )
        totals["trimmed_attention_score_positions"] += (
            current_batch_size * heads * layers * current_tokens * current_tokens
        )
        totals["fixed_core_forward_work_proxy"] += _core_forward_work(
            batch_size=current_batch_size,
            tokens=max_tokens,
            layers=layers,
            d_model=d_model,
            d_ff=d_ff,
        )
        totals["trimmed_core_forward_work_proxy"] += _core_forward_work(
            batch_size=current_batch_size,
            tokens=current_tokens,
            layers=layers,
            d_model=d_model,
            d_ff=d_ff,
        )

    elapsed = time.perf_counter() - started
    expected_events = int(manifest["counts"]["train"])
    if totals["events"] != expected_events:
        raise RuntimeError(
            f"measured {totals['events']} events; expected {expected_events}"
        )

    def reduction(fixed: int, trimmed: int) -> float:
        return 100.0 * (1.0 - trimmed / fixed)

    report = {
        "measurement": "one complete cached voxel training-loader epoch",
        "cache_dir": str(cache_dir),
        "provenance_mode": provenance_mode,
        "epoch_zero_based": epoch,
        "batch_size": batch_size,
        "workers": workers,
        "max_tokens": max_tokens,
        "model_shape": {
            "heads": heads,
            "layers": layers,
            "d_model": d_model,
            "d_ff": d_ff,
        },
        "batches": totals["batches"],
        "events": totals["events"],
        "minimum_batch_tokens": minimum_batch_tokens,
        "maximum_batch_tokens": maximum_batch_tokens,
        "mean_real_tokens_per_event": (
            totals["real_tokens"] / totals["events"]
        ),
        "mean_executed_tokens_per_event_after_batch_trimming": (
            totals["trimmed_token_positions"] / totals["events"]
        ),
        "token_position_reduction_percent": reduction(
            totals["fixed_token_positions"], totals["trimmed_token_positions"]
        ),
        "attention_score_position_reduction_percent": reduction(
            totals["fixed_attention_score_positions"],
            totals["trimmed_attention_score_positions"],
        ),
        "estimated_core_forward_work_reduction_percent": reduction(
            totals["fixed_core_forward_work_proxy"],
            totals["trimmed_core_forward_work_proxy"],
        ),
        "scan_seconds": elapsed,
        "interpretation": {
            "attention_score_positions": (
                "Exact N^2 tensor-position reduction for this loader epoch; "
                "not an equal wall-clock speedup."
            ),
            "core_forward_work_proxy": (
                "Architecture-aware estimate covering projections, attention "
                "matrix products, and feed-forward projections; excludes "
                "backward, optimizer, and non-core operations."
            ),
        },
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if arguments.output is not None:
        output = arguments.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Saved report: {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
