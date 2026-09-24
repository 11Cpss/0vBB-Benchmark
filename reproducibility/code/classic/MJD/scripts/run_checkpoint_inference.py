#!/usr/bin/env python3
"""Run test-only inference from a final MJD checkpoint."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mjdbench.data import MJDWaveformDataset, discover_files  # noqa: E402
from mjdbench.training import evaluate_model  # noqa: E402


class _ProgressLoader:
    def __init__(self, loader: DataLoader[Any], event_count: int) -> None:
        self.loader = loader
        self.event_count = event_count

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self) -> Iterator[Any]:
        started = time.monotonic()
        completed = 0
        next_report = 10_000
        for batch in self.loader:
            yield batch
            completed += int(batch["inputs"].shape[0])
            if completed >= next_report or completed == self.event_count:
                elapsed = time.monotonic() - started
                rate = completed / elapsed if elapsed else 0.0
                remaining = (self.event_count - completed) / rate if rate else 0.0
                print(
                    f"progress {completed}/{self.event_count} "
                    f"({100.0 * completed / self.event_count:.1f}%); "
                    f"{rate:.1f} events/s; ETA {remaining / 60.0:.1f} min",
                    flush=True,
                )
                next_report += 10_000


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("architecture")
    parser.add_argument("--task", choices=("classification", "regression"), required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="Exact trained checkpoint")
    parser.add_argument("--config", type=Path, help="Effective saved run configuration")
    parser.add_argument("--data-root", type=Path, help="Relocated raw MJD directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp-precision", choices=("auto", "float16", "bfloat16"), default="auto")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    model_dir = PROJECT_ROOT / "outputs" / args.task / args.architecture
    run_config = json.loads((args.config or (model_dir / "run_config.json")).read_text(encoding="utf-8"))
    data_config = dict(run_config["data"])
    if args.data_root is not None:
        data_config["data_root"] = str(args.data_root)
    if run_config.get("architecture", args.architecture) != args.architecture or run_config.get("task") != args.task:
        raise ValueError("configuration architecture/task does not match request")
    source = MJDWaveformDataset(
        discover_files(Path(data_config["data_root"]), "test"),
        task=args.task,
        baseline_samples=int(data_config["baseline_samples"]),
        classification_amplitude_normalization=bool(
            data_config["classification_amplitude_normalization"]
        ),
        regression_waveform_scale=float(data_config["regression_waveform_scale"]),
    )
    event_count = len(source) if args.max_events is None else min(len(source), args.max_events)
    dataset: Any = source if event_count == len(source) else Subset(source, range(event_count))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    module = importlib.import_module(f"architectures.{args.architecture}.model")
    if "model" in run_config and run_config["model"] != module.MODEL_CONFIG:
        raise ValueError("saved model settings differ from this architecture definition")
    model = module.build_model(args.task)
    checkpoint = torch.load(args.checkpoint or (model_dir / "best.pt"), map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    destination = args.output_dir or model_dir
    print(
        f"Restored epoch {checkpoint.get('epoch')} checkpoint; "
        f"evaluating {event_count} events with batch_size={args.batch_size}",
        flush=True,
    )
    try:
        metrics = evaluate_model(
            model,
            _ProgressLoader(loader, event_count),
            task=args.task,
            device=args.device,
            output_dir=destination,
            use_amp=not args.no_amp,
            amp_precision=args.amp_precision,
        )
    finally:
        source.close()
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Saved inference artifacts to {destination}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
