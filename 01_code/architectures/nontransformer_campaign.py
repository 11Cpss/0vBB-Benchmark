#!/usr/bin/env python3
"""Run the ten non-Transformer NEXT training jobs as one serial campaign.

This is orchestration, not a model/data smoke test.  Preflight only validates
imports, YAML syntax, required paths, and immutable output ownership.  Each
training program is then executed once in the documented order and receives a
fresh attempt directory.  A failure is recorded and the queue continues.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARCHITECTURES_ROOT = PROJECT_ROOT / "01_code" / "architectures"
CAMPAIGNS_ROOT = PROJECT_ROOT / "03_training_runs" / "campaigns"
CHECKPOINTS_ROOT = PROJECT_ROOT / "02_models" / "checkpoints"
ARCHITECTURES = (
    "classic_001_topology_xgboost",
    "point_003_pointmlp",
    "seq_001_bigru",
    "seq_002_dilated_tcn",
    "mixer_001_projection_mlp_mixer",
    "gnn_005_dimenet_lite",
    "point_004_rigid_kpconv",
    "topo_001_persistence_perslay",
    "ssm_001_pointmamba",
    "sparse_001_submanifold_resnet",
)
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("JSON root must be a mapping: %s" % path)
    return value


def _required_files(architecture_id: str) -> Iterable[Path]:
    directory = ARCHITECTURES_ROOT / architecture_id
    yield directory / "config.yaml"
    yield directory / "train_classification.py"
    yield directory / "README.md"
    yield directory / "README_EN.md"


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("configuration root must be a mapping: %s" % path)
    return value


def preflight(run_id: str, resume_queue: bool) -> Dict[str, Dict[str, Any]]:
    """Perform only the non-executing startup checks allowed by the campaign."""

    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "run ID must match %s" % RUN_ID_PATTERN.pattern
        )
    required_imports = (
        "numpy",
        "yaml",
        "h5py",
        "torch",
        "matplotlib",
        "tqdm",
        "xgboost",
    )
    for module_name in required_imports:
        importlib.import_module(module_name)

    configurations: Dict[str, Dict[str, Any]] = {}
    data_roots: set[Path] = set()
    for architecture_id in ARCHITECTURES:
        for path in _required_files(architecture_id):
            if not path.is_file():
                raise FileNotFoundError("required campaign file is missing: %s" % path)
        config_path = ARCHITECTURES_ROOT / architecture_id / "config.yaml"
        config = _read_yaml(config_path)
        if config.get("architecture_id") != architecture_id:
            raise ValueError(
                "%s declares architecture_id=%r"
                % (config_path, config.get("architecture_id"))
            )
        data = config.get("data")
        if not isinstance(data, dict) or "root" not in data:
            raise ValueError("%s must declare data.root" % config_path)
        data_roots.add(Path(str(data["root"])).expanduser())
        configurations[architecture_id] = config
    for data_root in data_roots:
        if not data_root.is_dir():
            raise FileNotFoundError("NEXT data directory is missing: %s" % data_root)

    campaign_dir = CAMPAIGNS_ROOT / run_id
    checkpoint_dir = CHECKPOINTS_ROOT / run_id
    if resume_queue:
        if not (campaign_dir / "manifest.json").is_file():
            raise FileNotFoundError(
                "--resume-queue requires an existing manifest: %s"
                % (campaign_dir / "manifest.json")
            )
    elif campaign_dir.exists() or checkpoint_dir.exists():
        raise FileExistsError(
            "run ID already owns campaign/checkpoint output; choose a new run ID: %s"
            % run_id
        )
    return configurations


def new_manifest(run_id: str) -> Dict[str, Any]:
    created = utc_now()
    return {
        "schema_version": 1,
        "run_id": run_id,
        "campaign": "NEXT non-Transformer v2 training-only",
        "created_at": created,
        "updated_at": created,
        "project_root": str(PROJECT_ROOT),
        "split_policy": {
            "allowed": ["train", "validation"],
            "forbidden": ["test"],
            "seed": 42,
            "fractions": [0.8, 0.1, 0.1],
        },
        "training_order": list(ARCHITECTURES),
        "models": {
            architecture_id: {
                "status": "PENDING",
                "attempts": [],
                "latest_attempt": None,
                "best_validation_auc": None,
                "duration_seconds": None,
            }
            for architecture_id in ARCHITECTURES
        },
    }


def validate_manifest(manifest: Mapping[str, Any], run_id: str) -> None:
    if manifest.get("run_id") != run_id:
        raise ValueError("manifest run_id does not match --run-id")
    if tuple(manifest.get("training_order", ())) != ARCHITECTURES:
        raise ValueError("manifest training order does not match this campaign version")
    models = manifest.get("models")
    if not isinstance(models, dict) or set(models) != set(ARCHITECTURES):
        raise ValueError("manifest model set does not match this campaign version")


def next_attempt_number(run_id: str, architecture_id: str) -> int:
    roots = (
        CAMPAIGNS_ROOT / run_id / architecture_id,
        CHECKPOINTS_ROOT / run_id / architecture_id,
    )
    existing: set[int] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for child in root.iterdir():
            match = re.fullmatch(r"attempt_(\d{3})", child.name)
            if match:
                existing.add(int(match.group(1)))
    return max(existing, default=0) + 1


def snapshot_config(
    source: Mapping[str, Any],
    destination: Path,
    checkpoint_dir: Path,
    attempt_dir: Path,
) -> None:
    snapshot = json.loads(json.dumps(source))
    output = snapshot.setdefault("output", {})
    if not isinstance(output, dict):
        raise ValueError("output configuration must be a mapping")
    output.update(
        {
            "checkpoint_dir": str(checkpoint_dir),
            "log_dir": str(attempt_dir),
            "plot_dir": str(attempt_dir),
            "allow_overwrite": False,
            "campaign_layout": True,
        }
    )
    destination.parent.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    with destination.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(snapshot, handle, sort_keys=False, allow_unicode=True)


def append_log(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()


def run_process(
    command: list[str],
    environment: Mapping[str, str],
    stdout_path: Path,
    queue_log: Path,
) -> int:
    with stdout_path.open("x", encoding="utf-8") as model_log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=dict(environment),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if process.stdout is None:
            raise RuntimeError("training subprocess did not expose stdout")
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            model_log.write(line)
            model_log.flush()
            append_log(queue_log, line)
        return int(process.wait())


def _format_duration(seconds: Any) -> str:
    if seconds is None:
        return "—"
    value = int(round(float(seconds)))
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    return "%02d:%02d:%02d" % (hours, minutes, secs)


def update_model_cards(
    architecture_id: str,
    run_id: str,
    attempt: int,
    summary: Mapping[str, Any],
) -> None:
    directory = ARCHITECTURES_ROOT / architecture_id
    auc = float(summary["best_validation_auc"])
    loss = float(summary["best_validation_loss"])
    duration = _format_duration(summary.get("duration_seconds"))
    environment = summary.get("environment", {})
    artifacts = summary.get("artifacts", {})
    retries = "no" if attempt == 1 else "yes (attempt_%03d)" % attempt
    parameter_count = summary.get("parameter_count")
    parameter_text = "N/A" if parameter_count is None else format(int(parameter_count), ",")
    entries = {
        "README.md": f"""

<!-- campaign-result:{run_id}:start -->
## Campaign `{run_id}` 训练结果

本节由串行训练队列在训练成功后写入，只包含 train/validation 信息；本阶段没有读取 test split。

| 项目 | 实际值 |
|---|---|
| 状态 / attempt | `DONE` / `attempt_{attempt:03d}` |
| 后端 | `{summary.get('backend', 'unknown')}` |
| 实际参数量 | {parameter_text} |
| 树数量 / 树节点数 | {summary.get('tree_count', 'N/A')} / {summary.get('tree_node_count', 'N/A')} |
| 完成 epoch / best epoch | {summary.get('epochs_completed')} / {summary.get('best_epoch')} |
| best validation AUC | **{auc:.6f}** |
| best validation loss | {loss:.6f} |
| 总训练时间 | {duration} |
| early stop | `{str(bool(summary.get('early_stopped'))).lower()}` |
| 失败重试 | `{retries}` |
| Python / 框架 | `{environment.get('python', 'unknown')}` / `{environment.get('torch', environment.get('xgboost', 'unknown'))}` |
| 设备 | `{environment.get('gpu', environment.get('device', 'unknown'))}` |
| best / last checkpoint | `{artifacts.get('best', 'unknown')}` / `{artifacts.get('last', 'unknown')}` |
| 训练日志 | `{artifacts.get('json', artifacts.get('history', 'unknown'))}` |
<!-- campaign-result:{run_id}:end -->
""",
        "README_EN.md": f"""

<!-- campaign-result:{run_id}:start -->
## Campaign `{run_id}` training result

This section is written after successful serial training. It contains train/validation information only; the test split was not read in this stage.

| Item | Observed value |
|---|---|
| Status / attempt | `DONE` / `attempt_{attempt:03d}` |
| Backend | `{summary.get('backend', 'unknown')}` |
| Trainable parameters | {parameter_text} |
| Trees / tree nodes | {summary.get('tree_count', 'N/A')} / {summary.get('tree_node_count', 'N/A')} |
| Completed / best epoch | {summary.get('epochs_completed')} / {summary.get('best_epoch')} |
| Best validation AUC | **{auc:.6f}** |
| Best validation loss | {loss:.6f} |
| Training time | {duration} |
| Early stopped | `{str(bool(summary.get('early_stopped'))).lower()}` |
| Failed-attempt retry | `{retries}` |
| Python / framework | `{environment.get('python', 'unknown')}` / `{environment.get('torch', environment.get('xgboost', 'unknown'))}` |
| Device | `{environment.get('gpu', environment.get('device', 'unknown'))}` |
| Best / last checkpoint | `{artifacts.get('best', 'unknown')}` / `{artifacts.get('last', 'unknown')}` |
| Training history | `{artifacts.get('json', artifacts.get('history', 'unknown'))}` |
<!-- campaign-result:{run_id}:end -->
""",
    }
    for filename, block in entries.items():
        path = directory / filename
        content = path.read_text(encoding="utf-8")
        marker = "<!-- campaign-result:%s:start -->" % run_id
        if marker in content:
            continue
        # Turn the training-time placeholder into an explicitly superseded
        # record before appending the observed result.  Recovery instructions
        # that mention FAILED/PENDING are intentionally left unchanged.
        replacements = {
            "| `PENDING` | — | — | — | — | — |": (
                "| `SUPERSEDED` | see appended campaign result | — | — | — | — |"
            ),
            "状态：**PENDING**。": (
                "训练前占位状态：**PENDING**（已由文末 campaign 结果取代）。"
            ),
            "Status: **PENDING**.": (
                "Pre-campaign placeholder: **PENDING** (superseded by the appended result)."
            ),
            "正式训练尚未启动。": (
                "本段训练前占位说明已由文末追加的 campaign 结果取代。"
            ),
            "尚未启动正式训练。": (
                "本段训练前占位说明已由文末追加的 campaign 结果取代。"
            ),
            "Formal training has not started.": (
                "This pre-campaign placeholder is superseded by the appended completed result."
            ),
        }
        for old, new in replacements.items():
            content = content.replace(old, new)
        temporary = path.with_name(path.name + ".campaign-result.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(content)
            handle.write(block)
        os.replace(temporary, path)


def write_campaign_docs(manifest: Mapping[str, Any], campaign_dir: Path) -> None:
    run_id = str(manifest["run_id"])
    rows_cn = []
    rows_en = []
    for architecture_id in ARCHITECTURES:
        record = manifest["models"][architecture_id]
        status = record["status"]
        auc = record.get("best_validation_auc")
        auc_text = "—" if auc is None else "%.6f" % float(auc)
        duration = _format_duration(record.get("duration_seconds"))
        relative_card = "../../../01_code/architectures/%s/README.md" % architecture_id
        relative_card_en = "../../../01_code/architectures/%s/README_EN.md" % architecture_id
        rows_cn.append(
            "| [%s](%s) | `%s` | %s | %s |"
            % (architecture_id, relative_card, status, auc_text, duration)
        )
        rows_en.append(
            "| [%s](%s) | `%s` | %s | %s |"
            % (architecture_id, relative_card_en, status, auc_text, duration)
        )
    common_note_cn = (
        "本 campaign 只训练并使用 validation 做 early stopping/best checkpoint 选择；"
        "没有读取 test split，也没有生成正式测试排行榜。"
    )
    common_note_en = (
        "This campaign trains models and uses validation only for early stopping and "
        "best-checkpoint selection. It does not read the test split or produce a test leaderboard."
    )
    cn = "\n".join(
        [
            "# NEXT 非 Transformer 训练 campaign `%s`" % run_id,
            "",
            common_note_cn,
            "",
            "| 模型卡 | 状态 | best validation AUC | 耗时 |",
            "|---|---|---:|---:|",
            *rows_cn,
            "",
            "- 权威状态文件：[`manifest.json`](./manifest.json)",
            "- 完整队列日志：[`queue.log`](./queue.log)",
            "- `FAILED` 模型必须以同一 RUN_ID 执行 `--resume-queue`；新 attempt 从头训练。",
            "",
        ]
    )
    en = "\n".join(
        [
            "# NEXT non-Transformer training campaign `%s`" % run_id,
            "",
            common_note_en,
            "",
            "| Model card | Status | Best validation AUC | Duration |",
            "|---|---|---:|---:|",
            *rows_en,
            "",
            "- Authoritative state: [`manifest.json`](./manifest.json)",
            "- Complete queue log: [`queue.log`](./queue.log)",
            "- Resume a `FAILED` model under the same RUN_ID with `--resume-queue`; a new attempt starts from scratch.",
            "",
        ]
    )
    (campaign_dir / "README.md").write_text(cn, encoding="utf-8")
    (campaign_dir / "README_EN.md").write_text(en, encoding="utf-8")


def run_campaign(run_id: str, resume_queue: bool) -> int:
    configurations = preflight(run_id, resume_queue)
    campaign_dir = CAMPAIGNS_ROOT / run_id
    manifest_path = campaign_dir / "manifest.json"
    queue_log = campaign_dir / "queue.log"
    if resume_queue:
        manifest = load_json(manifest_path)
        validate_manifest(manifest, run_id)
    else:
        campaign_dir.mkdir(parents=True, exist_ok=False)
        (CHECKPOINTS_ROOT / run_id).mkdir(parents=True, exist_ok=False)
        queue_log.touch(exist_ok=False)
        manifest = new_manifest(run_id)
        atomic_json(manifest_path, manifest)
        write_campaign_docs(manifest, campaign_dir)

    header = "[%s] campaign %s %s\n" % (
        utc_now(), run_id, "RESUME" if resume_queue else "START"
    )
    print(header, end="", flush=True)
    append_log(queue_log, header)

    for architecture_id in ARCHITECTURES:
        model_record = manifest["models"][architecture_id]
        if model_record["status"] == "DONE":
            line = "[%s] SKIP %s (DONE)\n" % (utc_now(), architecture_id)
            print(line, end="", flush=True)
            append_log(queue_log, line)
            continue

        attempt = next_attempt_number(run_id, architecture_id)
        attempt_name = "attempt_%03d" % attempt
        attempt_dir = campaign_dir / architecture_id / attempt_name
        checkpoint_dir = CHECKPOINTS_ROOT / run_id / architecture_id / attempt_name
        snapshot_path = attempt_dir / "config.snapshot.yaml"
        snapshot_config(
            configurations[architecture_id],
            snapshot_path,
            checkpoint_dir,
            attempt_dir,
        )
        stdout_path = attempt_dir / "stdout.log"
        attempt_record = {
            "attempt": attempt,
            "status": "RUNNING",
            "started_at": utc_now(),
            "completed_at": None,
            "exit_code": None,
            "attempt_dir": str(attempt_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "config_snapshot": str(snapshot_path),
            "stdout_log": str(stdout_path),
        }
        model_record["status"] = "RUNNING"
        model_record["latest_attempt"] = attempt
        model_record.setdefault("attempts", []).append(attempt_record)
        manifest["updated_at"] = utc_now()
        atomic_json(manifest_path, manifest)
        write_campaign_docs(manifest, campaign_dir)

        start_line = "[%s] START %s %s\n" % (
            utc_now(), architecture_id, attempt_name
        )
        print(start_line, end="", flush=True)
        append_log(queue_log, start_line)
        environment = os.environ.copy()
        environment.update(
            {
                "NEXT_CAMPAIGN_RUN_ID": run_id,
                "NEXT_CAMPAIGN_ATTEMPT": str(attempt),
                "PYTHONUNBUFFERED": "1",
            }
        )
        entrypoint = (
            ARCHITECTURES_ROOT / architecture_id / "train_classification.py"
        )
        exit_code = run_process(
            [sys.executable, str(entrypoint), str(snapshot_path)],
            environment,
            stdout_path,
            queue_log,
        )
        attempt_record["completed_at"] = utc_now()
        attempt_record["exit_code"] = exit_code
        summary_path = attempt_dir / "run_summary.json"
        if exit_code == 0 and summary_path.is_file():
            summary = load_json(summary_path)
            attempt_record["status"] = "DONE"
            attempt_record["summary"] = str(summary_path)
            model_record["status"] = "DONE"
            model_record["best_validation_auc"] = summary.get(
                "best_validation_auc"
            )
            model_record["best_validation_loss"] = summary.get(
                "best_validation_loss"
            )
            model_record["best_epoch"] = summary.get("best_epoch")
            model_record["epochs_completed"] = summary.get("epochs_completed")
            model_record["duration_seconds"] = summary.get("duration_seconds")
            update_model_cards(
                architecture_id, run_id, attempt, summary
            )
            outcome = "DONE"
        else:
            attempt_record["status"] = "FAILED"
            if exit_code == 0:
                attempt_record["failure"] = "missing run_summary.json"
                exit_code = 86
                attempt_record["exit_code"] = exit_code
            model_record["status"] = "FAILED"
            outcome = "FAILED"
        manifest["updated_at"] = utc_now()
        atomic_json(manifest_path, manifest)
        write_campaign_docs(manifest, campaign_dir)
        end_line = "[%s] %s %s exit=%d\n" % (
            utc_now(), outcome, architecture_id, exit_code
        )
        print(end_line, end="", flush=True)
        append_log(queue_log, end_line)

    statuses = [manifest["models"][item]["status"] for item in ARCHITECTURES]
    complete = all(status == "DONE" for status in statuses)
    manifest["complete"] = complete
    manifest["completed_at"] = utc_now() if complete else None
    manifest["updated_at"] = utc_now()
    atomic_json(manifest_path, manifest)
    write_campaign_docs(manifest, campaign_dir)
    line = "[%s] campaign %s: %s\n" % (
        utc_now(), run_id, "COMPLETE" if complete else "INCOMPLETE"
    )
    print(line, end="", flush=True)
    append_log(queue_log, line)
    return 0 if complete else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume-queue", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_campaign(args.run_id, bool(args.resume_queue))


if __name__ == "__main__":
    raise SystemExit(main())
