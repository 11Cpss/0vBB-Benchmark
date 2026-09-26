#!/usr/bin/env python3
"""Train POINT-004, the rigid KPConv-style NEXT classifier."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ARCHITECTURE_ID = "point_004_rigid_kpconv"
SCRIPT_DIR = Path(__file__).resolve().parent


def _project_root() -> Path:
    for candidate in SCRIPT_DIR.parents:
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "src" / "next_alt"
        ).is_dir():
            return candidate
    raise RuntimeError(f"cannot locate project root from {SCRIPT_DIR}")


PROJECT_ROOT = _project_root()
PACKAGE_SRC = PROJECT_ROOT / "src"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from next_alt.training import main_for_architecture


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=SCRIPT_DIR / "config.yaml",
        help="training YAML (default: config.yaml beside this script)",
    )
    args = parser.parse_args()
    return main_for_architecture(ARCHITECTURE_ID, args.config)


if __name__ == "__main__":
    raise SystemExit(main())
