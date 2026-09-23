#!/usr/bin/env python3
"""Train GNN-001 independently for clean-event MJD energy regression."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARCHITECTURE_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, ARCHITECTURE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mjdbench.workflow import main_for_architecture
from model import ARCHITECTURE_CONFIG, build_model


if __name__ == "__main__":
    raise SystemExit(
        main_for_architecture(
            "gnn_001_static_gine",
            task="regression",
            model_factory=build_model,
            architecture_config=ARCHITECTURE_CONFIG,
        )
    )
