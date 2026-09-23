#!/usr/bin/env python3
"""Run EnergyBench using the visible benchmark/ metric sources and protocols.

The shared bundle CLI handles NPZ validation, manifests, output and result
checks. Metric calculations are loaded from the byte-identical cores here.
"""
from pathlib import Path
import importlib.util
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.dont_write_bytecode = True


def main():
    spec = importlib.util.spec_from_file_location(
        "energybench_bundle_cli", ROOT / "reproduction/evaluate.py"
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    cli.PROFILES = {
        "strict600": HERE / "core/unified_metrics.py",
        "overflow601": HERE / "legacy_v2/core/unified_metrics.py",
    }
    cli.main()


if __name__ == "__main__":
    main()
