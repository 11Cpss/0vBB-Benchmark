#!/usr/bin/env python3
"""Print a compact status table for a Simple EnergyBench model-zoo campaign."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = PROJECT_ROOT / "03_training_runs" / "energybench_campaigns"


def _metric(job: Mapping[str, Any], campaign_root: Path) -> str:
    attempts = job.get("attempts", [])
    if not attempts:
        return "—"
    command = attempts[-1].get("command", [])
    try:
        output_index = command.index("--output-dir") + 1
        run_root = Path(command[output_index])
    except (ValueError, IndexError, TypeError):
        run_root = (
            campaign_root
            / "runs"
            / str(job["architecture_id"])
            / str(job["task"])
        )
    summary = run_root / "run_summary.json"
    if not summary.is_file():
        return "—"
    try:
        values = json.loads(summary.read_text(encoding="utf-8"))["metrics"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return "invalid summary"
    if job["task"] == "classification":
        auc = values.get("auc")
        matched = values.get("matched_auc")
        return f"auc={auc!s} matched={matched!s}"
    return f"ers={values.get('ers')!s} rmse={values.get('rmse')!s}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--campaigns-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.campaigns_root.expanduser().resolve() / args.run_id
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        parser.error(f"manifest does not exist: {manifest_path}")
    campaign = json.loads(manifest_path.read_text(encoding="utf-8"))
    jobs = campaign["jobs"]
    print(
        f"run={campaign['run_id']} status={campaign['status']} "
        f"updated={campaign['updated_at']}"
    )
    supervisor = campaign.get("supervisor") or {}
    supervisor_pid = supervisor.get("pid")
    try:
        if supervisor_pid is None:
            supervisor_state = "unknown"
        else:
            os.kill(int(supervisor_pid), 0)
            supervisor_state = "alive"
    except (TypeError, ValueError, ProcessLookupError):
        supervisor_state = "not-running"
    except PermissionError:
        supervisor_state = "alive"
    print(
        f"supervisor={supervisor_state} pid={supervisor_pid!s} "
        f"heartbeat={supervisor.get('heartbeat_at', '—')}"
    )
    print(
        "summary "
        + " ".join(
            f"{status.lower()}={sum(job['status'] == status for job in jobs)}"
            for status in ("DONE", "RUNNING", "PENDING", "FAILED")
        )
    )
    print()
    print(f"{'job':58} {'status':8} {'attempt':7} result")
    print("-" * 100)
    for job in jobs:
        attempts = job.get("attempts", [])
        print(
            f"{job['job_id']:58} {job['status']:8} "
            f"{len(attempts):7d} {_metric(job, root)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
