#!/usr/bin/env python3
"""Retry failed campaign jobs after the active supervisor finishes.

Incomplete task directories are moved into ``interrupted_attempts`` before
each retry. Completed jobs and the event-count split are never rerun.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ARCHITECTURES_ROOT = Path(__file__).absolute().parent
PROJECT_ROOT = ARCHITECTURES_ROOT.parents[1]
DEFAULT_CAMPAIGNS_ROOT = PROJECT_ROOT / "03_training_runs" / "energybench_campaigns"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("jobs"), list):
        raise ValueError(f"invalid campaign manifest: {path}")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def process_alive(pid: object) -> bool:
    try:
        value = int(pid)
        if value <= 0:
            return False
        stat = Path(f"/proc/{value}/stat").read_text(encoding="utf-8")
        if stat.rsplit(")", 1)[1].strip().split()[0] == "Z":
            return False
        os.kill(value, 0)
    except (TypeError, ValueError, OSError, IndexError):
        return False
    return True


def active_processes(campaign: Mapping[str, Any]) -> list[int]:
    result: list[int] = []
    supervisor = campaign.get("supervisor") or {}
    if process_alive(supervisor.get("pid")):
        result.append(int(supervisor["pid"]))
    for job in campaign["jobs"]:
        attempts = job.get("attempts") or []
        if job.get("status") == "RUNNING" and attempts:
            pid = attempts[-1].get("pid")
            if process_alive(pid):
                result.append(int(pid))
    return result


def archive_failures(
    campaign_root: Path,
    manifest_path: Path,
    campaign: dict[str, Any],
    retry_round: int,
) -> list[str]:
    failed = [job for job in campaign["jobs"] if job.get("status") == "FAILED"]
    if not failed:
        return []
    timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    archive_root = (
        campaign_root
        / "interrupted_attempts"
        / f"auto_retry_{retry_round}_{timestamp}"
        / "runs"
    )
    identifiers: list[str] = []
    for job in failed:
        identifier = str(job["job_id"])
        identifiers.append(identifier)
        source = (
            campaign_root
            / "runs"
            / str(job["architecture_id"])
            / str(job["task"])
        )
        destination = (
            archive_root / str(job["architecture_id"]) / str(job["task"])
        )
        if source.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
            attempts = job.get("attempts") or []
            if attempts:
                attempts[-1]["output_archive"] = str(destination)
    campaign.setdefault("recoveries", []).append(
        {
            "created_at": utc_now(),
            "retry_round": retry_round,
            "failed_jobs": identifiers,
            "archive_root": str(archive_root),
        }
    )
    atomic_json(manifest_path, campaign)
    return identifiers


def recovery_command(
    *,
    campaign: Mapping[str, Any],
    campaigns_root: Path,
    python: Path,
) -> list[str]:
    command = [
        str(python),
        str(ARCHITECTURES_ROOT / "run_energybench_campaign.py"),
        "--run-id",
        str(campaign["run_id"]),
        "--campaigns-root",
        str(campaigns_root),
        "--data",
        str(campaign["data_root"]),
        "--parallel-capacity",
        str(campaign.get("parallel_gpu_capacity", 3)),
        "--resume",
    ]
    if campaign.get("include_regression"):
        command.append("--include-regression")
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--campaigns-root", type=Path, default=DEFAULT_CAMPAIGNS_ROOT)
    parser.add_argument("--max-retry-rounds", type=int, default=3)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument(
        "--python", type=Path, default=Path(sys.prefix) / "bin" / "python"
    )
    args = parser.parse_args()
    if args.max_retry_rounds <= 0:
        parser.error("--max-retry-rounds must be positive")
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    campaigns_root = args.campaigns_root.expanduser().absolute()
    campaign_root = campaigns_root / args.run_id
    manifest_path = campaign_root / "manifest.json"
    if not manifest_path.is_file():
        parser.error(f"manifest does not exist: {manifest_path}")
    python = args.python.expanduser().absolute()
    if not python.is_file():
        parser.error(f"Python interpreter does not exist: {python}")

    for retry_round in range(1, args.max_retry_rounds + 1):
        announced_wait = False
        while True:
            campaign = load_manifest(manifest_path)
            processes = active_processes(campaign)
            if not processes:
                break
            if not announced_wait:
                print(
                    f"Waiting for active campaign {args.run_id}; "
                    f"processes={processes}",
                    flush=True,
                )
                announced_wait = True
            time.sleep(args.poll_seconds)

        campaign = load_manifest(manifest_path)
        stale_running = [
            job for job in campaign["jobs"] if job.get("status") == "RUNNING"
        ]
        if stale_running:
            stopped_at = utc_now()
            for job in stale_running:
                job["status"] = "FAILED"
                attempts = job.get("attempts") or []
                if attempts and attempts[-1].get("return_code") is None:
                    attempts[-1].update(
                        {
                            "completed_at": stopped_at,
                            "return_code": -1,
                            "termination_reason": "stale process detected by autorecover",
                        }
                    )
            campaign["status"] = "FAILED"
            atomic_json(manifest_path, campaign)
        if campaign.get("status") == "DONE":
            print("Campaign completed without recovery.", flush=True)
            return 0
        failed_ids = archive_failures(
            campaign_root, manifest_path, campaign, retry_round
        )
        if not failed_ids:
            print("No failed jobs remain to recover.", flush=True)
            return 0
        print(
            f"Retry round {retry_round}: {len(failed_ids)} jobs: "
            + ", ".join(failed_ids),
            flush=True,
        )
        return_code = subprocess.run(
            recovery_command(
                campaign=campaign,
                campaigns_root=campaigns_root,
                python=python,
            ),
            cwd=PROJECT_ROOT,
            check=False,
        ).returncode
        if return_code == 0:
            print("Campaign recovery completed successfully.", flush=True)
            return 0
        print(f"Recovery supervisor exited rc={return_code}.", flush=True)

    print("Maximum automatic retry rounds exhausted.", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
