#!/usr/bin/env python3
"""Read-only schema, label, identity, and split inspection for EXO-200."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exobench.config import DEFAULT_DATA_ROOT, DataConfig
from exobench.data import inspect_data_root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--full-event-scan",
        action="store_true",
        help="Check every waveform/cluster child dataset metadata and cluster values.",
    )
    parser.add_argument(
        "--sample-events-per-file",
        type=int,
        default=0,
        help="When doing a full scan, also read this many evenly spaced waveforms per file.",
    )
    args = parser.parse_args()
    if args.sample_events_per_file < 0:
        parser.error("--sample-events-per-file must be non-negative")
    if args.sample_events_per_file and not args.full_event_scan:
        parser.error("--sample-events-per-file requires --full-event-scan")
    report = inspect_data_root(
        DataConfig(data_root=args.data_root),
        full_event_scan=args.full_event_scan,
        sample_events_per_file=args.sample_events_per_file,
    )
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
