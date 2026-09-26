#!/usr/bin/env python3
"""Run point_001_deepsets classification with Simple EnergyBench."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ARCHITECTURES_ROOT = SCRIPT_DIR.parent
if str(ARCHITECTURES_ROOT) not in sys.path:
    sys.path.insert(0, str(ARCHITECTURES_ROOT))

from workflow_runner import main_for_architecture


if __name__ == "__main__":
    raise SystemExit(
        main_for_architecture("point_001_deepsets", task="classification")
    )

