#!/usr/bin/env python3
"""Print a read-only snapshot of one non-Transformer campaign."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def latest_epoch(attempt_dir: Any) -> tuple[str, str]:
    if not attempt_dir:
        return "—", "—"
    csv_path = Path(str(attempt_dir)) / "epochs.csv"
    if not csv_path.is_file():
        return "—", "—"
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return "—", "—"
    row = rows[-1]
    return row.get("epoch", "—"), row.get("validation_auc", "—")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    campaign_dir = PROJECT_ROOT / "03_training_runs" / "campaigns" / args.run_id
    manifest_path = campaign_dir / "manifest.json"
    if not manifest_path.is_file():
        print("Manifest not found: %s" % manifest_path)
        return 2
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    print("RUN_ID: %s" % manifest.get("run_id"))
    print("Updated: %s" % manifest.get("updated_at"))
    print("%-40s %-9s %7s %12s" % ("architecture", "status", "epoch", "val AUC"))
    print("-" * 72)
    for architecture_id in manifest.get("training_order", []):
        record = manifest["models"][architecture_id]
        attempts = record.get("attempts") or []
        attempt_dir = attempts[-1].get("attempt_dir") if attempts else None
        epoch, current_auc = latest_epoch(attempt_dir)
        best_auc = record.get("best_validation_auc")
        auc = "%.6f" % best_auc if best_auc is not None else current_auc
        print(
            "%-40s %-9s %7s %12s"
            % (architecture_id, record.get("status"), epoch, auc)
        )
    queue_log = campaign_dir / "queue.log"
    print("\nLatest queue log:")
    if queue_log.is_file():
        lines = queue_log.read_text(encoding="utf-8", errors="replace").splitlines()
        for line in lines[-16:]:
            print(line)
    return 0 if bool(manifest.get("complete")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
