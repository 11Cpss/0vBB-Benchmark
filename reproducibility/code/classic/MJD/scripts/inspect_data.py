#!/usr/bin/env python3
"""Report whether local MJD shards are complete and schema-compatible."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mjdbench import inspect_data_root
from mjdbench.config import DEFAULT_DATA_ROOT


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    rows = inspect_data_root(args.data_root)
    if args.json:
        print(json.dumps(rows, indent=2))
    elif not rows:
        print(f"No MJD_*.hdf5 files found in {args.data_root}")
    else:
        for row in rows:
            size_gb = row["bytes"] / 1_000_000_000
            details = ""
            if row["status"] == "ok":
                details = (
                    f" events={row['events']} waveform={row['waveform_shape']}"
                    f" dtype={row['waveform_dtype']}"
                )
            else:
                details = f" error={row.get('error', 'unknown')}"
            print(
                f"{row['status'].upper():10s} {size_gb:6.2f} GB "
                f"{Path(row['path']).name}{details}"
            )
    return 1 if any(row["status"] != "ok" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
