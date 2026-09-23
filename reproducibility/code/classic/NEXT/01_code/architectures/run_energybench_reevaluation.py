#!/usr/bin/env python3
"""Re-run canonical EnergyBench metrics from saved held-out predictions.

Training already writes deterministic predictions for the held-out test split.
Replaying those arrays through the public EnergyBench evaluation entry points
recomputes every metric, CSV, prediction archive, and plot without retraining or
competing with unrelated GPU jobs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import torch
from torch import nn


ARCHITECTURES_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ARCHITECTURES_ROOT.parents[1]
WORKFLOW_ROOT = PROJECT_ROOT / "evalutaions_workflow"
for candidate in (WORKFLOW_ROOT, PROJECT_ROOT / "src", ARCHITECTURES_ROOT):
    value = str(candidate)
    if value not in sys.path:
        sys.path.insert(0, value)

from simple_energybench import (  # noqa: E402
    EvaluationConfig,
    evaluate_classification,
    evaluate_regression,
)


DEFAULT_CAMPAIGN_ROOT = (
    PROJECT_ROOT
    / "03_training_runs"
    / "energybench_campaigns"
    / "20260808_energybench_rewrite_v2"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


class ReplayModel(nn.Module):
    """Return the prediction tensor supplied as the model input."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs


def prediction_batches(
    arrays: Mapping[str, np.ndarray], task: str, batch_size: int
) -> Iterator[dict[str, Any]]:
    prediction_key = "score" if task == "classification" else "energy_pred"
    energy_key = "energy" if task == "classification" else "energy_true"
    count = int(len(arrays[prediction_key]))
    optional = (
        "sample_weight",
        "event_id",
        "category",
        "group_id",
        "split",
        "projection_coverage",
    )
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        batch: dict[str, Any] = {
            "inputs": torch.from_numpy(
                np.asarray(arrays[prediction_key][start:stop], dtype=np.float32)
            ),
            "energy": np.asarray(arrays[energy_key][start:stop]),
        }
        if task == "classification":
            batch["label"] = np.asarray(arrays["label"][start:stop])
        for key in optional:
            if key in arrays:
                batch[key] = np.asarray(arrays[key][start:stop])
        yield batch


def run_worker(source: Path, output: Path, batch_size: int) -> int:
    with np.load(source, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    task = "classification" if "score" in arrays else "regression"
    required = (
        {"score", "label", "energy"}
        if task == "classification"
        else {"energy_true", "energy_pred"}
    )
    missing = required.difference(arrays)
    if missing:
        raise ValueError(f"{source} is missing arrays: {sorted(missing)}")
    model = ReplayModel()
    batches = prediction_batches(arrays, task, batch_size)
    config = EvaluationConfig()
    if task == "classification":
        metrics = evaluate_classification(
            model, batches, device="cpu", output_dir=output, config=config
        )
    else:
        metrics = evaluate_regression(
            model, batches, device="cpu", output_dir=output, config=config
        )
    atomic_json(
        output / "reevaluation_record.json",
        {
            "status": "DONE",
            "task": task,
            "completed_at": utc_now(),
            "source_predictions": str(source),
            "protocol": config.to_dict(),
            "metrics": {
                key: metrics.get(key)
                for key in (
                    "n_events",
                    "auc",
                    "matched_auc",
                    "energy_independence_score",
                    "ers",
                    "rmse",
                )
                if key in metrics
            },
        },
    )
    print(f"DONE {task} {source} -> {output}", flush=True)
    return 0


def discover_jobs(campaign_root: Path, output_root: Path) -> list[dict[str, Any]]:
    manifest = json.loads((campaign_root / "manifest.json").read_text(encoding="utf-8"))
    jobs: list[dict[str, Any]] = []
    for source_job in manifest["jobs"]:
        architecture = str(source_job["architecture_id"])
        task = str(source_job["task"])
        source = campaign_root / "runs" / architecture / task / "evaluation" / "predictions.npz"
        output = output_root / architecture / task
        available = source.is_file()
        jobs.append(
            {
                "job_id": str(source_job["job_id"]),
                "architecture_id": architecture,
                "task": task,
                "source_training_status": source_job.get("status"),
                "source_predictions": str(source),
                "output_dir": str(output),
                "status": "PENDING" if available else "UNAVAILABLE",
                "reason": None if available else "held-out predictions.npz is missing",
                "attempts": [],
            }
        )
    return jobs


def run_queue(
    campaign_root: Path, output_root: Path, parallel: int, batch_size: int
) -> int:
    manifest_path = output_root / "manifest.json"
    log_root = output_root / "logs"
    output_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for job in manifest["jobs"]:
            record = Path(job["output_dir"]) / "reevaluation_record.json"
            if job["status"] == "RUNNING":
                job["status"] = "DONE" if record.is_file() else "PENDING"
            elif job["status"] == "FAILED" and not record.is_file():
                job["status"] = "PENDING"
    else:
        manifest = {
            "schema_version": 1,
            "kind": "EnergyBench canonical prediction replay evaluation",
            "created_at": utc_now(),
            "source_campaign": str(campaign_root),
            "output_root": str(output_root),
            "parallel": parallel,
            "protocol": EvaluationConfig().to_dict(),
            "jobs": discover_jobs(campaign_root, output_root),
        }
    manifest.update(
        {
            "status": "RUNNING",
            "updated_at": utc_now(),
            "supervisor_pid": os.getpid(),
            "parallel": parallel,
        }
    )
    atomic_json(manifest_path, manifest)
    running: dict[str, dict[str, Any]] = {}
    while True:
        for identifier, state in list(running.items()):
            code = state["process"].poll()
            if code is None:
                continue
            state["log"].close()
            job = state["job"]
            attempt = job["attempts"][-1]
            attempt["completed_at"] = utc_now()
            attempt["return_code"] = int(code)
            job["status"] = "DONE" if code == 0 else "FAILED"
            print(f"{job['status']} {identifier} rc={code}", flush=True)
            del running[identifier]
            manifest["updated_at"] = utc_now()
            atomic_json(manifest_path, manifest)
        pending = [job for job in manifest["jobs"] if job["status"] == "PENDING"]
        while pending and len(running) < parallel:
            job = pending.pop(0)
            identifier = str(job["job_id"])
            output = Path(job["output_dir"])
            if output.exists() and any(output.iterdir()):
                record = output / "reevaluation_record.json"
                if record.is_file():
                    job["status"] = "DONE"
                    continue
                raise FileExistsError(f"incomplete non-empty output directory: {output}")
            log_path = log_root / f"{identifier.replace(':', '__')}.log"
            log_handle = log_path.open("a", encoding="utf-8")
            command = [
                str(Path(sys.prefix) / "bin" / "python"),
                str(Path(__file__).resolve()),
                "--worker-source",
                str(job["source_predictions"]),
                "--worker-output",
                str(output),
                "--batch-size",
                str(batch_size),
            ]
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1", "CUDA_VISIBLE_DEVICES": ""},
            )
            job["status"] = "RUNNING"
            job["attempts"].append(
                {
                    "attempt": len(job["attempts"]) + 1,
                    "started_at": utc_now(),
                    "pid": process.pid,
                    "log_path": str(log_path),
                    "return_code": None,
                }
            )
            running[identifier] = {"process": process, "log": log_handle, "job": job}
            print(f"START {identifier} pid={process.pid}", flush=True)
            manifest["updated_at"] = utc_now()
            atomic_json(manifest_path, manifest)
        if not pending and not running:
            break
        time.sleep(2)
    failed = [job for job in manifest["jobs"] if job["status"] == "FAILED"]
    unavailable = [job for job in manifest["jobs"] if job["status"] == "UNAVAILABLE"]
    done = [job for job in manifest["jobs"] if job["status"] == "DONE"]
    manifest.update(
        {
            "status": "FAILED" if failed else "DONE_WITH_UNAVAILABLE" if unavailable else "DONE",
            "completed_at": utc_now(),
            "updated_at": utc_now(),
            "summary": {
                "done": len(done),
                "failed": len(failed),
                "unavailable": len(unavailable),
                "total": len(manifest["jobs"]),
            },
        }
    )
    atomic_json(manifest_path, manifest)
    print(f"COMPLETE done={len(done)} failed={len(failed)} unavailable={len(unavailable)}", flush=True)
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, default=DEFAULT_CAMPAIGN_ROOT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--worker-source", type=Path)
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args()
    if args.parallel <= 0 or args.batch_size <= 0:
        parser.error("--parallel and --batch-size must be positive")
    if args.worker_source is not None or args.worker_output is not None:
        if args.worker_source is None or args.worker_output is None:
            parser.error("worker mode requires both --worker-source and --worker-output")
        return run_worker(
            args.worker_source.expanduser().resolve(),
            args.worker_output.expanduser().resolve(),
            args.batch_size,
        )
    campaign_root = args.campaign_root.expanduser().resolve()
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else campaign_root / "reevaluation_20260818"
    )
    return run_queue(campaign_root, output_root, args.parallel, args.batch_size)


if __name__ == "__main__":
    raise SystemExit(main())
