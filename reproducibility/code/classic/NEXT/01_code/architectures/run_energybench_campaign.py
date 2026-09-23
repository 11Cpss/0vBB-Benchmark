#!/usr/bin/env python3
"""Run the complete model zoo through Simple EnergyBench with safe parallelism."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ARCHITECTURES_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ARCHITECTURES_ROOT.parents[1]
WORKFLOW_ROOT = PROJECT_ROOT / "evalutaions_workflow"
for candidate in (WORKFLOW_ROOT, PROJECT_ROOT / "src", ARCHITECTURES_ROOT):
    value = str(candidate)
    if value not in sys.path:
        sys.path.insert(0, value)

from simple_energybench import TrainingConfig, prepare_dataset  # noqa: E402
from workflow_models import architecture_ids, get_spec  # noqa: E402


DEFAULT_DATA_ROOT = Path("/home/klz/Data/zeronu_benchmark/NEXT")
DEFAULT_CAMPAIGNS_ROOT = PROJECT_ROOT / "03_training_runs" / "energybench_campaigns"
HEARTBEAT_INTERVAL_SECONDS = 30.0
CAPACITY_CONTROL_FILENAME = "parallel_capacity.txt"
REGRESSION_ARCHITECTURES = (
    "cnn_001_two_conv_baseline",
    "cnn_002_global_energy_skip",
    "cnn_003_residual_spatial",
)
HEAVY_GPU_JOBS = {
    # Reservations approximate peak GiB under the host's per-user VRAM quota.
    # Values above a campaign's capacity are clamped to the full capacity so
    # the job runs alone rather than becoming unschedulable.
    "gnn_001_static_gine:classification": 6,
    "point_003_pointmlp:classification": 4,
    "seq_002_dilated_tcn:classification": 2,
    "cnn_006_dense_3d_resnet:classification": 3,
    "gnn_005_dimenet_lite:classification": 4,
    "cnn_003_residual_spatial:classification": 2,
    "cnn_003_residual_spatial:regression": 2,
    "cnn_004_multiview_late_fusion:classification": 2,
    "cnn_005_multiscale_projection:classification": 2,
    "point_002_pointnetpp:classification": 2,
    # Measured peaks from the first campaign retry under the host watchdog.
    "gnn_002_particlenet_edgeconv:classification": 3,
    "gnn_003_egnn:classification": 7,
    "gnn_004_gravnet:classification": 2,
    "hybrid_001_cnn_gnn:classification": 3,
    "ssm_001_pointmamba:classification": 12,
    "sparse_001_submanifold_resnet:classification": 1,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def read_capacity(path: Path, fallback: int) -> int:
    try:
        value = int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return fallback
    return value if value > 0 else fallback


def job_id(architecture_id: str, task: str) -> str:
    return f"{architecture_id}:{task}"


def expected_job_weight(architecture_id: str, task: str) -> int:
    identifier = job_id(architecture_id, task)
    if task == "classification" and get_spec(architecture_id).backend == "xgboost":
        return 0
    return HEAVY_GPU_JOBS.get(identifier, 1)


def make_jobs(include_regression: bool) -> list[dict[str, Any]]:
    jobs = [
        {
            "job_id": job_id(architecture_id, "classification"),
            "architecture_id": architecture_id,
            "task": "classification",
            "status": "PENDING",
            "weight": expected_job_weight(architecture_id, "classification"),
            "attempts": [],
        }
        for architecture_id in architecture_ids()
    ]
    if include_regression:
        for architecture_id in REGRESSION_ARCHITECTURES:
            identifier = job_id(architecture_id, "regression")
            jobs.append(
                {
                    "job_id": identifier,
                    "architecture_id": architecture_id,
                    "task": "regression",
                    "status": "PENDING",
                    "weight": expected_job_weight(architecture_id, "regression"),
                    "attempts": [],
                }
            )
    return jobs


def new_campaign(
    run_id: str,
    data_root: Path,
    include_regression: bool,
    capacity: int,
) -> dict[str, Any]:
    now = utc_now()
    return {
        "schema_version": 1,
        "run_id": run_id,
        "workflow": "Simple EnergyBench full model-zoo training and evaluation",
        "status": "PREPARING",
        "created_at": now,
        "updated_at": now,
        "supervisor": {
            "pid": os.getpid(),
            "status": "PREPARING",
            "started_at": now,
            "heartbeat_at": now,
            "active_jobs": [],
        },
        "data_root": str(data_root),
        "split_policy": {
            "kind": "event-count-stratified",
            "fractions": [0.8, 0.1, 0.1],
            "seed": 42,
            "max_files_per_class": None,
        },
        "training": TrainingConfig().to_dict(),
        "evaluation": "EvaluationConfig() canonical 5 keV protocol",
        "parallel_gpu_capacity": int(capacity),
        "include_regression": bool(include_regression),
        "jobs": make_jobs(include_regression),
    }


def load_campaign(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("jobs"), list):
        raise ValueError(f"invalid campaign manifest: {path}")
    return value


def update_campaign(path: Path, campaign: dict[str, Any]) -> None:
    campaign["updated_at"] = utc_now()
    atomic_json(path, campaign)


def prepare_split(data_root: Path, split_path: Path) -> dict[str, Any]:
    config = TrainingConfig()
    prepared = prepare_dataset(
        data_root,
        batch_size=config.batch_size,
        mode="classification",
        seed=config.seed,
        num_workers=config.num_workers,
        manifest_path=split_path,
        max_files_per_class=None,
        shuffle_buffer_size=512,
        verbose=True,
    )
    return prepared.counts


def entrypoint(architecture_id: str, task: str) -> Path:
    filename = (
        "train_classification.py"
        if task == "classification"
        else "train_energy_regression.py"
    )
    path = ARCHITECTURES_ROOT / architecture_id / filename
    if not path.is_file():
        raise FileNotFoundError(f"missing campaign entry point: {path}")
    return path


def command_for(
    python: Path,
    campaign_root: Path,
    split_path: Path,
    data_root: Path,
    job: Mapping[str, Any],
) -> list[str]:
    architecture_id = str(job["architecture_id"])
    task = str(job["task"])
    output_dir = campaign_root / "runs" / architecture_id / task
    return [
        str(python),
        str(entrypoint(architecture_id, task)),
        "--output-dir",
        str(output_dir),
        "--data",
        str(data_root),
        "--manifest",
        str(split_path),
    ]


def run_campaign(
    campaign_root: Path,
    campaign: dict[str, Any],
    manifest_path: Path,
    python: Path,
    capacity: int,
) -> int:
    data_root = Path(campaign["data_root"])
    split_path = campaign_root / "event_split.json"
    log_root = campaign_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    campaign_log = campaign_root / "campaign.log"
    capacity_path = campaign_root / CAPACITY_CONTROL_FILENAME
    if capacity_path.exists():
        capacity = read_capacity(capacity_path, capacity)
    else:
        atomic_text(capacity_path, f"{capacity}\n")
    campaign["parallel_gpu_capacity"] = int(capacity)
    running: dict[str, dict[str, Any]] = {}

    # A prior interrupted RUNNING process cannot still be owned by this new
    # orchestrator.  Mark it retryable while retaining the attempt record.
    for job in campaign["jobs"]:
        if job["status"] == "RUNNING":
            job["status"] = "PENDING"
    supervisor_started_at = utc_now()
    campaign["status"] = "RUNNING"
    campaign["supervisor"] = {
        "pid": os.getpid(),
        "status": "RUNNING",
        "started_at": supervisor_started_at,
        "heartbeat_at": supervisor_started_at,
        "active_jobs": [],
    }
    update_campaign(manifest_path, campaign)
    last_heartbeat_clock = time.monotonic()

    def record(message: str) -> None:
        line = f"{utc_now()} {message}"
        print(line, flush=True)
        with campaign_log.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    while True:
        completed_ids: list[str] = []
        for identifier, state in list(running.items()):
            return_code = state["process"].poll()
            if return_code is None:
                continue
            state["log_handle"].close()
            job = state["job"]
            attempt = job["attempts"][-1]
            attempt["completed_at"] = utc_now()
            attempt["return_code"] = int(return_code)
            attempt["duration_seconds"] = time.monotonic() - state["started_clock"]
            if return_code == 0:
                job["status"] = "DONE"
                record(f"DONE {identifier} duration={attempt['duration_seconds']:.1f}s")
            else:
                job["status"] = "FAILED"
                record(
                    f"FAILED {identifier} rc={return_code} "
                    f"log={attempt['log_path']}"
                )
            completed_ids.append(identifier)
        for identifier in completed_ids:
            del running[identifier]
        if completed_ids:
            update_campaign(manifest_path, campaign)

        now_clock = time.monotonic()
        if now_clock - last_heartbeat_clock >= HEARTBEAT_INTERVAL_SECONDS:
            campaign["supervisor"]["heartbeat_at"] = utc_now()
            campaign["supervisor"]["active_jobs"] = sorted(running)
            update_campaign(manifest_path, campaign)
            last_heartbeat_clock = now_clock

        configured_capacity = read_capacity(capacity_path, capacity)
        if configured_capacity != capacity:
            previous_capacity = capacity
            capacity = configured_capacity
            campaign["parallel_gpu_capacity"] = int(capacity)
            record(f"CAPACITY {previous_capacity} -> {capacity}")
            update_campaign(manifest_path, campaign)

        pending = [job for job in campaign["jobs"] if job["status"] == "PENDING"]
        if not pending and not running:
            break

        used_capacity = sum(
            min(int(state["job"]["weight"]), capacity)
            for state in running.values()
            if int(state["job"]["weight"]) > 0
        )
        started_any = False
        for job in pending:
            weight = int(job["weight"])
            reservation = min(weight, capacity) if weight > 0 else 0
            # CPU-only jobs do not consume GPU capacity. A reservation at or
            # above capacity runs alone among GPU jobs.
            if reservation > 0 and used_capacity + reservation > capacity:
                continue
            identifier = str(job["job_id"])
            attempt_number = len(job["attempts"]) + 1
            log_path = log_root / f"{identifier.replace(':', '__')}.log"
            mode = "a" if log_path.exists() else "x"
            log_handle = log_path.open(mode, encoding="utf-8")
            started_at = utc_now()
            log_handle.write(
                f"\n=== ENERGYBENCH ATTEMPT {attempt_number} START {started_at} ===\n"
            )
            log_handle.flush()
            command = command_for(
                python,
                campaign_root,
                split_path,
                data_root,
                job,
            )
            child_environment = {**os.environ, "PYTHONUNBUFFERED": "1"}
            if weight == 0:
                # The classic XGBoost path is CPU-only. Hiding CUDA prevents a
                # needless Torch context from consuming ~500 MiB of quota.
                child_environment["CUDA_VISIBLE_DEVICES"] = ""
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                env=child_environment,
            )
            attempt = {
                "attempt": attempt_number,
                "started_at": started_at,
                "command": command,
                "pid": process.pid,
                "log_path": str(log_path),
                "return_code": None,
            }
            job["attempts"].append(attempt)
            job["status"] = "RUNNING"
            running[identifier] = {
                "process": process,
                "log_handle": log_handle,
                "job": job,
                "started_clock": time.monotonic(),
            }
            used_capacity += reservation
            started_any = True
            record(
                f"START {identifier} pid={process.pid} weight={weight} "
                f"reservation={reservation}"
            )
            update_campaign(manifest_path, campaign)
        if not started_any and not running and pending:
            impossible = ", ".join(
                f"{job['job_id']}(weight={job['weight']})" for job in pending
            )
            raise RuntimeError(
                f"no pending job fits parallel capacity {capacity}: {impossible}"
            )
        time.sleep(5.0)

    failures = [job for job in campaign["jobs"] if job["status"] != "DONE"]
    campaign["status"] = "FAILED" if failures else "DONE"
    campaign["completed_at"] = utc_now()
    campaign["summary"] = {
        "done": sum(job["status"] == "DONE" for job in campaign["jobs"]),
        "failed": len(failures),
        "total": len(campaign["jobs"]),
    }
    campaign["supervisor"].update(
        {
            "status": campaign["status"],
            "heartbeat_at": utc_now(),
            "completed_at": utc_now(),
            "active_jobs": [],
        }
    )
    update_campaign(manifest_path, campaign)
    record(
        f"CAMPAIGN {campaign['status']} done={campaign['summary']['done']} "
        f"failed={campaign['summary']['failed']}"
    )
    return 1 if failures else 0


def _run_id(value: str) -> str:
    if not value or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for character in value):
        raise argparse.ArgumentTypeError(
            "run ID may contain only letters, digits, dot, underscore, and dash"
        )
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        type=_run_id,
        default=datetime.now().strftime("%Y%m%d_%H%M%S_energybench"),
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--campaigns-root", type=Path, default=DEFAULT_CAMPAIGNS_ROOT)
    # ``sys.executable`` resolves through this environment's symlink to the
    # bare CPython build, which loses the venv site-packages when launched as a
    # new process.  ``sys.prefix/bin/python`` preserves the active venv.
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.prefix) / "bin" / "python",
    )
    parser.add_argument("--parallel-capacity", type=int, default=3)
    parser.add_argument("--include-regression", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.parallel_capacity <= 0:
        parser.error("--parallel-capacity must be positive")
    data_root = args.data.expanduser().resolve()
    if not data_root.is_dir():
        parser.error(f"dataset directory does not exist: {data_root}")
    # Do not resolve this symlink: executing the venv's own ``bin/python`` is
    # what activates its site-packages for child processes.
    python = args.python.expanduser().absolute()
    if not python.is_file():
        parser.error(f"Python interpreter does not exist: {python}")
    campaign_root = args.campaigns_root.expanduser().resolve() / args.run_id
    manifest_path = campaign_root / "manifest.json"
    if args.resume:
        if not manifest_path.is_file():
            parser.error(f"--resume requires {manifest_path}")
        campaign = load_campaign(manifest_path)
        if campaign.get("run_id") != args.run_id:
            parser.error("campaign manifest run_id mismatch")
        if campaign.get("status") == "DONE":
            print(f"Campaign is already complete: {campaign_root}")
            return 0
        for job in campaign["jobs"]:
            job["weight"] = expected_job_weight(
                str(job["architecture_id"]), str(job["task"])
            )
            if job["status"] == "FAILED":
                job["status"] = "PENDING"
    else:
        if campaign_root.exists():
            parser.error(f"campaign output already exists: {campaign_root}")
        campaign_root.mkdir(parents=True)
        campaign = new_campaign(
            args.run_id,
            data_root,
            args.include_regression,
            args.parallel_capacity,
        )
        update_campaign(manifest_path, campaign)
        counts = prepare_split(data_root, campaign_root / "event_split.json")
        campaign["data_counts"] = counts
        campaign["status"] = "READY"
        update_campaign(manifest_path, campaign)
    return run_campaign(
        campaign_root,
        campaign,
        manifest_path,
        python,
        args.parallel_capacity,
    )


if __name__ == "__main__":
    raise SystemExit(main())
