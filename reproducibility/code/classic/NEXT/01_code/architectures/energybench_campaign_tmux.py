#!/usr/bin/env python3
"""Launch or observe an EnergyBench campaign in a persistent tmux session.

With ``--launch``, the campaign supervisor itself runs in the ``00-campaign``
window and owns all training workers. The remaining windows are read-only
views of the per-job logs, so attaching or detaching cannot interrupt a run.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


ARCHITECTURES_ROOT = Path(__file__).absolute().parent
PROJECT_ROOT = ARCHITECTURES_ROOT.parents[1]
DEFAULT_CAMPAIGNS_ROOT = PROJECT_ROOT / "03_training_runs" / "energybench_campaigns"
DEFAULT_DATA_ROOT = Path("/home/klz/Data/zeronu_benchmark/NEXT")
DEFAULT_REFRESH_SECONDS = 5.0
SESSION_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
ATTEMPT_MARKER = b"=== ENERGYBENCH ATTEMPT "
CAPACITY_CONTROL_FILENAME = "parallel_capacity.txt"
FORMAL_START_MARKERS = (
    b"Using cached event split manifest:",
    b"Created event split manifest:",
)


def load_campaign(manifest_path: Path) -> dict[str, Any]:
    campaign = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(campaign, dict) or not isinstance(campaign.get("jobs"), list):
        raise ValueError(f"invalid campaign manifest: {manifest_path}")
    return campaign


def find_job(campaign: Mapping[str, Any], identifier: str) -> Mapping[str, Any]:
    for job in campaign["jobs"]:
        if job.get("job_id") == identifier:
            return job
    raise KeyError(f"job is not present in campaign: {identifier}")


def job_log_path(campaign_root: Path, identifier: str) -> Path:
    return campaign_root / "logs" / f"{identifier.replace(':', '__')}.log"


def window_name(index: int, job: Mapping[str, Any]) -> str:
    task_suffix = "cls" if job["task"] == "classification" else "reg"
    return f"{index:02d}-{job['architecture_id']}-{task_suffix}"


def tmux(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["tmux", *arguments],
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def session_names() -> set[str]:
    result = tmux("list-sessions", "-F", "#{session_name}", check=False)
    if result.returncode != 0:
        message = result.stderr.strip()
        if "no server running" in message or "failed to connect" in message:
            return set()
        raise RuntimeError(message or "unable to list tmux sessions")
    return {line for line in result.stdout.splitlines() if line}


def window_names(session: str) -> set[str]:
    result = tmux("list-windows", "-t", session, "-F", "#{window_name}")
    return {line for line in result.stdout.splitlines() if line}


def window_ids(session: str) -> list[str]:
    result = tmux("list-windows", "-t", session, "-F", "#{window_id}")
    return [line for line in result.stdout.splitlines() if line]


def shell_command(arguments: Sequence[object]) -> str:
    return shlex.join([str(argument) for argument in arguments])


def update_capacity(campaign_root: Path, capacity: int) -> None:
    path = campaign_root / CAPACITY_CONTROL_FILENAME
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(f"{capacity}\n", encoding="utf-8")
    os.replace(temporary, path)
    print(f"requested parallel capacity {capacity}: {path}")


def python_executable() -> Path:
    # Preserve the virtual-environment launcher instead of resolving its
    # symlink to the bare CPython build.
    candidate = (Path(sys.prefix) / "bin" / "python").absolute()
    return candidate if candidate.is_file() else Path(sys.executable).absolute()


def process_alive(pid: object) -> bool:
    try:
        value = int(pid)
        if value <= 0:
            return False
        os.kill(value, 0)
    except (TypeError, ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True
    return True


def heartbeat_age_seconds(supervisor: Mapping[str, Any]) -> float | None:
    raw_value = supervisor.get("heartbeat_at")
    if not isinstance(raw_value, str):
        return None
    try:
        heartbeat = datetime.fromisoformat(raw_value)
    except ValueError:
        return None
    return max(0.0, (datetime.now().astimezone() - heartbeat).total_seconds())


def clear_screen() -> None:
    print("\033[2J\033[H", end="", flush=True)


def latest_attempt_offset(log_path: Path, attempt_count: int) -> int | None:
    """Return a byte offset that hides completed append-only attempts."""

    if attempt_count <= 1 or not log_path.is_file():
        return None
    contents = log_path.read_bytes()
    positions = [contents.rfind(ATTEMPT_MARKER)]
    positions.extend(contents.rfind(marker) for marker in FORMAL_START_MARKERS)
    start = max(positions)
    if start >= 0:
        return start

    # Compatibility for the first campaign, launched before explicit attempt
    # markers existed. Its failed bare-Python attempt ended at this traceback.
    stale_numpy_error = b"ModuleNotFoundError: No module named 'numpy'"
    error = contents.rfind(stale_numpy_error)
    if error >= 0:
        newline = contents.find(b"\n", error)
        return len(contents) if newline < 0 else newline + 1
    return None


def print_job_header(
    *,
    run_id: str,
    identifier: str,
    job: Mapping[str, Any],
    log_path: Path,
    note: str,
) -> None:
    clear_screen()
    print(f"EnergyBench run : {run_id}")
    print(f"job             : {identifier}")
    print(f"manifest status : {job.get('status', 'UNKNOWN')}")
    print(f"attempts        : {len(job.get('attempts', []))}")
    print(f"log             : {log_path}")
    print()
    print(note)
    print("-" * 88, flush=True)


def follow_job(
    *,
    campaign_root: Path,
    identifier: str,
    history_lines: int,
    refresh_seconds: float,
) -> int:
    manifest_path = campaign_root / "manifest.json"
    campaign = load_campaign(manifest_path)
    run_id = str(campaign["run_id"])
    initial_job = find_job(campaign, identifier)
    initial_status = str(initial_job.get("status", "UNKNOWN"))
    initial_attempts = len(initial_job.get("attempts", []))
    log_path = job_log_path(campaign_root, identifier)

    # PENDING jobs may already have an append-only log from an older failed
    # attempt.  Remember its byte boundary and show only the next attempt, so a
    # stale traceback cannot be mistaken for the current run.
    pending_offset = log_path.stat().st_size if log_path.exists() else 0
    if initial_status == "PENDING":
        while True:
            campaign = load_campaign(manifest_path)
            job = find_job(campaign, identifier)
            status = str(job.get("status", "UNKNOWN"))
            attempts = len(job.get("attempts", []))
            print_job_header(
                run_id=run_id,
                identifier=identifier,
                job=job,
                log_path=log_path,
                note="Waiting for the scheduler to start this task. Old attempts are hidden.",
            )
            if status != "PENDING" or attempts > initial_attempts:
                break
            time.sleep(refresh_seconds)
        print_job_header(
            run_id=run_id,
            identifier=identifier,
            job=job,
            log_path=log_path,
            note="Following output written by the newly started attempt.",
        )
        os.execvp(
            "tail",
            ["tail", "-c", f"+{pending_offset + 1}", "-F", str(log_path)],
        )

    job = initial_job
    while not log_path.exists():
        campaign = load_campaign(manifest_path)
        job = find_job(campaign, identifier)
        print_job_header(
            run_id=run_id,
            identifier=identifier,
            job=job,
            log_path=log_path,
            note="The task is active; waiting for its log file to appear.",
        )
        time.sleep(refresh_seconds)
    attempt_offset = latest_attempt_offset(log_path, initial_attempts)
    if attempt_offset is not None:
        print_job_header(
            run_id=run_id,
            identifier=identifier,
            job=job,
            log_path=log_path,
            note="Following only the latest manifest attempt; stale attempts are hidden.",
        )
        os.execvp(
            "tail",
            ["tail", "-c", f"+{attempt_offset + 1}", "-F", str(log_path)],
        )
    print_job_header(
        run_id=run_id,
        identifier=identifier,
        job=job,
        log_path=log_path,
        note=f"Following the latest {history_lines} lines and all new output.",
    )
    os.execvp(
        "tail",
        ["tail", "-n", str(history_lines), "-F", str(log_path)],
    )
    return 0


def dashboard(campaign_root: Path, refresh_seconds: float) -> int:
    manifest_path = campaign_root / "manifest.json"
    while True:
        campaign = load_campaign(manifest_path)
        jobs = campaign["jobs"]
        clear_screen()
        print(f"EnergyBench campaign {campaign['run_id']}")
        print(f"local time: {datetime.now().astimezone().isoformat(timespec='seconds')}")
        print(f"manifest  : {manifest_path}")
        supervisor = campaign.get("supervisor") or {}
        supervisor_pid = supervisor.get("pid", "—")
        supervisor_alive = process_alive(supervisor_pid)
        heartbeat_age = heartbeat_age_seconds(supervisor)
        heartbeat_text = "—" if heartbeat_age is None else f"{heartbeat_age:.0f}s ago"
        supervisor_state = "ALIVE" if supervisor_alive else "NOT RUNNING"
        if (
            campaign.get("status") == "RUNNING"
            and (not supervisor_alive or (heartbeat_age is not None and heartbeat_age > 120.0))
        ):
            supervisor_state = "STALE"
        print(
            f"supervisor: {supervisor_state} pid={supervisor_pid} "
            f"heartbeat={heartbeat_text}"
        )
        print()
        print(
            "  ".join(
                f"{status}={sum(job['status'] == status for job in jobs)}"
                for status in ("DONE", "RUNNING", "PENDING", "FAILED")
            )
        )
        print()
        print(f"{'window':42} {'status':8} {'try':>3}  job")
        print("-" * 110)
        for index, job in enumerate(jobs, start=1):
            print(
                f"{window_name(index, job):42} "
                f"{job['status']:8} {len(job.get('attempts', [])):3d}  {job['job_id']}"
            )
        print()
        print("Ctrl-b w: window list   Ctrl-b n/p: next/previous   Ctrl-b d: detach")
        sys.stdout.flush()
        time.sleep(refresh_seconds)


def setup_session(
    *,
    campaign_root: Path,
    session: str,
    history_lines: int,
    refresh_seconds: float,
) -> None:
    manifest_path = campaign_root / "manifest.json"
    campaign = load_campaign(manifest_path)
    run_id = str(campaign["run_id"])
    script_path = Path(__file__).absolute()
    python = python_executable()
    root_arguments = ["--campaigns-root", campaign_root.parent]
    dashboard_command = shell_command(
        [
            python,
            script_path,
            run_id,
            *root_arguments,
            "--dashboard",
            "--refresh-seconds",
            refresh_seconds,
        ]
    )

    existing_sessions = session_names()
    if session not in existing_sessions:
        tmux("new-session", "-d", "-s", session, "-n", "00-dashboard", dashboard_command)
        tmux("set-environment", "-t", session, "ENERGYBENCH_RUN_ID", run_id)
    else:
        environment = tmux(
            "show-environment", "-t", session, "ENERGYBENCH_RUN_ID", check=False
        )
        if environment.returncode == 0:
            configured_run = environment.stdout.strip().partition("=")[2]
            if configured_run and configured_run != run_id:
                raise RuntimeError(
                    f"tmux session {session!r} belongs to run {configured_run!r}"
                )
        else:
            tmux("set-environment", "-t", session, "ENERGYBENCH_RUN_ID", run_id)

    tmux("set-option", "-t", session, "renumber-windows", "off")
    existing_windows = window_names(session)

    if "00-dashboard" not in existing_windows:
        tmux("new-window", "-d", "-t", session, "-n", "00-dashboard", dashboard_command)
        existing_windows.add("00-dashboard")

    campaign_window = "00-campaign"
    if campaign_window not in existing_windows:
        campaign_log = campaign_root / "campaign.log"
        campaign_command = shell_command(
            ["tail", "-n", str(history_lines), "-F", campaign_log]
        )
        tmux(
            "new-window",
            "-d",
            "-t",
            session,
            "-n",
            campaign_window,
            campaign_command,
        )
        existing_windows.add(campaign_window)

    for index, job in enumerate(campaign["jobs"], start=1):
        name = window_name(index, job)
        if name in existing_windows:
            continue
        command = shell_command(
            [
                python,
                script_path,
                run_id,
                *root_arguments,
                "--watch-job",
                job["job_id"],
                "--history-lines",
                history_lines,
                "--refresh-seconds",
                refresh_seconds,
            ]
        )
        tmux("new-window", "-d", "-t", session, "-n", name, command)

    # Apply stable names and keep completed views visible to every window.
    # Window options are set by ID so no server-global tmux defaults belonging
    # to other users/sessions are changed.
    for identifier in window_ids(session):
        tmux("set-window-option", "-t", identifier, "automatic-rename", "off")
        tmux("set-window-option", "-t", identifier, "allow-rename", "off")
        tmux("set-window-option", "-t", identifier, "remain-on-exit", "on")
    tmux("select-window", "-t", f"{session}:00-dashboard")

    total_windows = len(window_names(session))
    print(f"tmux session ready: {session} ({total_windows} windows)")
    print(f"attach: tmux attach -t {shlex.quote(session)}")


def launch_session(
    *,
    run_id: str,
    campaign_root: Path,
    data_root: Path,
    session: str,
    capacity: int,
    include_regression: bool,
    history_lines: int,
    refresh_seconds: float,
) -> None:
    if campaign_root.exists():
        raise FileExistsError(f"campaign output already exists: {campaign_root}")
    if session in session_names():
        raise RuntimeError(f"tmux session already exists: {session}")
    python = python_executable()
    campaign_script = ARCHITECTURES_ROOT / "run_energybench_campaign.py"
    command: list[object] = [
        python,
        campaign_script,
        "--run-id",
        run_id,
        "--campaigns-root",
        campaign_root.parent,
        "--data",
        data_root,
        "--parallel-capacity",
        capacity,
    ]
    if include_regression:
        command.append("--include-regression")
    tmux(
        "new-session",
        "-d",
        "-s",
        session,
        "-n",
        "00-campaign",
        shell_command(command),
    )
    tmux("set-environment", "-t", session, "ENERGYBENCH_RUN_ID", run_id)
    tmux("set-window-option", "-t", f"{session}:00-campaign", "remain-on-exit", "on")

    manifest_path = campaign_root / "manifest.json"
    deadline = time.monotonic() + 60.0
    while not manifest_path.is_file() and time.monotonic() < deadline:
        if session not in session_names():
            raise RuntimeError("campaign tmux session exited before creating its manifest")
        time.sleep(0.5)
    if not manifest_path.is_file():
        pane = tmux(
            "capture-pane", "-p", "-t", f"{session}:00-campaign", check=False
        )
        detail = pane.stdout.strip() or pane.stderr.strip()
        raise RuntimeError(
            "campaign did not create its manifest within 60 seconds"
            + (f": {detail}" if detail else "")
        )
    setup_session(
        campaign_root=campaign_root,
        session=session,
        history_lines=history_lines,
        refresh_seconds=refresh_seconds,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--campaigns-root", type=Path, default=DEFAULT_CAMPAIGNS_ROOT)
    parser.add_argument("--session")
    parser.add_argument(
        "--launch",
        action="store_true",
        help="launch the campaign supervisor inside the tmux session",
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--parallel-capacity", type=int, default=3)
    parser.add_argument(
        "--update-capacity",
        type=int,
        help="change a running campaign's weighted parallel capacity",
    )
    parser.add_argument("--include-regression", action="store_true")
    parser.add_argument("--history-lines", type=int, default=60)
    parser.add_argument(
        "--refresh-seconds", type=float, default=DEFAULT_REFRESH_SECONDS
    )
    parser.add_argument("--watch-job", help=argparse.SUPPRESS)
    parser.add_argument("--dashboard", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.history_lines <= 0:
        parser.error("--history-lines must be positive")
    if args.refresh_seconds <= 0:
        parser.error("--refresh-seconds must be positive")
    if args.parallel_capacity <= 0:
        parser.error("--parallel-capacity must be positive")
    if args.update_capacity is not None and args.update_capacity <= 0:
        parser.error("--update-capacity must be positive")

    campaign_root = args.campaigns_root.expanduser().absolute() / args.run_id
    manifest_path = campaign_root / "manifest.json"
    session = args.session or f"energybench-{args.run_id[:8]}"
    if not SESSION_NAME_PATTERN.fullmatch(args.run_id):
        parser.error(
            "run ID may contain only letters, digits, dot, underscore, and dash"
        )
    if not SESSION_NAME_PATTERN.fullmatch(session):
        parser.error(
            "tmux session may contain only letters, digits, dot, underscore, and dash"
        )
    if args.launch:
        if args.watch_job or args.dashboard or args.update_capacity is not None:
            parser.error("--launch cannot be combined with internal viewer modes")
        data_root = args.data.expanduser().absolute()
        if not data_root.is_dir():
            parser.error(f"dataset directory does not exist: {data_root}")
        launch_session(
            run_id=args.run_id,
            campaign_root=campaign_root,
            data_root=data_root,
            session=session,
            capacity=args.parallel_capacity,
            include_regression=args.include_regression,
            history_lines=args.history_lines,
            refresh_seconds=args.refresh_seconds,
        )
        return 0
    if not manifest_path.is_file():
        parser.error(f"manifest does not exist: {manifest_path}")
    if args.update_capacity is not None:
        update_capacity(campaign_root, args.update_capacity)
        return 0
    if args.watch_job:
        return follow_job(
            campaign_root=campaign_root,
            identifier=args.watch_job,
            history_lines=args.history_lines,
            refresh_seconds=args.refresh_seconds,
        )
    if args.dashboard:
        return dashboard(campaign_root, args.refresh_seconds)

    setup_session(
        campaign_root=campaign_root,
        session=session,
        history_lines=args.history_lines,
        refresh_seconds=args.refresh_seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
