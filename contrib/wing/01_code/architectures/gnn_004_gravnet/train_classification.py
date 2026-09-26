#!/usr/bin/env python3
"""Train the gnn_004_gravnet classifier with the shared NEXT runner."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = next(
    parent for parent in (SCRIPT_DIR, *SCRIPT_DIR.parents) if (parent / "pyproject.toml").is_file()
)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from next_alt.training import main_for_architecture


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path, default=SCRIPT_DIR / "config.yaml")
    args = parser.parse_args()
    return main_for_architecture("gnn_004_gravnet", args.config)


if __name__ == "__main__":
    raise SystemExit(main())

