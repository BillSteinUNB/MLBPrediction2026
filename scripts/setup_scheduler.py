from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import textwrap
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASK_NAME = "MLBPrediction2026 Daily Pipeline"
DEFAULT_FIRST_PITCH = "19:05"
DEFAULT_HOURS_BEFORE = 3
DEFAULT_TIMEZONE = "America/New_York"
CRON_MARKER = "# MLBPrediction2026 daily pipeline"


@dataclass(frozen=True, slots=True)
class WindowsTaskPlan:
    task_name: str
    run_time: str
    runner_path: Path
    working_directory: Path
    registration_script: str
    query_command: list[str]


def compute_daily_run_time(
    first_pitch: str = DEFAULT_FIRST_PITCH,
    *,
    hours_before: int = DEFAULT_HOURS_BEFORE,
) -> str:
    """Return the daily run time in HH:MM based on a typical first pitch."""

    first_pitch_time = datetime.strptime(first_pitch, "%H:%M")
    run_time = first_pitch_time - timedelta(hours=hours_before)
    return run_time.strftime("%H:%M")


def build_windows_task_plan(
    *,
    repo_root: Path = REPO_ROOT,
    task_name: str = DEFAULT_TASK_NAME,
    run_time: str,
) -> WindowsTaskPlan:
    """Build the PowerShell registration plan for Windows Task Scheduler."""

    runner_path = repo_root / "scripts" / "run_daily.bat"
    task_name_literal = _powershell_single_quoted(task_name)
    runner_literal = _powershell_single_quoted(str(runner_path))
    working_directory_literal = _powershell_single_quoted(str(repo_root))
    run_time_literal = _powershell_single_quoted(run_time)
    registration_script = textwrap.dedent(
        f"""
        $taskName = '{task_name_literal}'
        $runnerPath = '{runner_literal}'
        $workingDirectory = '{working_directory_literal}'
        $triggerAt = [datetime]::ParseExact('{run_time_literal}', 'HH:mm', $null)
        $action = New-ScheduledTaskAction -Execute 'cmd.exe' -Argument ('/c ""' + $runnerPath + '""') -WorkingDirectory $workingDirectory
        $trigger = New-ScheduledTaskTrigger -Daily -At $triggerAt
        $settings = New-ScheduledTaskSettingsSet -WakeToRun -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew
        $principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType S4U
        Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force | Out-Null
        """
    ).strip()
    query_command = ["schtasks.exe", "/Query", "/TN", task_name, "/V", "/FO", "LIST"]
    return WindowsTaskPlan(
        task_name=task_name,
        run_time=run_time,
        runner_path=runner_path,
        working_directory=repo_root,
        registration_script=registration_script,
        query_command=query_command,
    )


def build_cron_entry(
    *,
    repo_root: Path = REPO_ROOT,
    run_time: str,
    timezone_name: str = DEFAULT_TIMEZONE,
) -> str:
    """Build a cron entry that executes the daily shell wrapper once per day."""

    hour, minute = run_time.split(":", maxsplit=1)
    runner_path = shlex.quote(str(repo_root / "scripts" / "run_daily.sh"))
    working_directory = shlex.quote(str(repo_root))
    log_path = shlex.quote(str(repo_root / "data" / "logs" / "run_daily.log"))
    schedule_line = (
        f"{int(minute)} {int(hour)} * * * "
        f"cd {working_directory} && /usr/bin/env bash {runner_path} >> {log_path} 2>&1"
    )
    return "\n".join((CRON_MARKER, f"CRON_TZ={timezone_name}", schedule_line))


def build_systemd_service(*, repo_root: Path = REPO_ROOT) -> str:
    """Build the systemd service unit contents for the daily runner."""

    runner_path = repo_root / "scripts" / "run_daily.sh"
    quoted_runner = shlex.quote(str(runner_path))
    return textwrap.dedent(
        f"""
        [Unit]
        Description=MLBPrediction2026 daily pipeline runner

        [Service]
        Type=oneshot
        WorkingDirectory={repo_root}
        ExecStart=/usr/bin/env bash {quoted_runner}
        """
    ).strip() + "\n"


def build_systemd_timer(*, run_time: str) -> str:
    """Build the systemd timer unit contents for the daily runner."""

    return textwrap.dedent(
        f"""
        [Unit]
        Description=Run MLBPrediction2026 daily pipeline 3 hours before first pitch

        [Timer]
        OnCalendar=*-*-* {run_time}:00
        Persistent=true
        WakeSystem=true

        [Install]
        WantedBy=timers.target
        """
    ).strip() + "\n"


def install_scheduler(
    *,
    platform_name: str,
    task_name: str,
    run_time: str,
    repo_root: Path = REPO_ROOT,
    timezone_name: str = DEFAULT_TIMEZONE,
    test_mode: bool = False,
) -> dict[str, object]:
    """Install the appropriate scheduler for the target platform."""

    if platform_name == "windows":
        return install_windows_task(
            repo_root=repo_root,
            task_name=task_name,
            run_time=run_time,
        )

    if platform_name == "cron":
        if test_mode:
            return preview_scheduler(
                platform_name=platform_name,
                task_name=task_name,
                run_time=run_time,
                repo_root=repo_root,
                timezone_name=timezone_name,
            )
        return install_cron_entry(
            repo_root=repo_root,
            run_time=run_time,
            timezone_name=timezone_name,
        )

    if platform_name == "systemd":
        if test_mode:
            return preview_scheduler(
                platform_name=platform_name,
                task_name=task_name,
                run_time=run_time,
                repo_root=repo_root,
                timezone_name=timezone_name,
            )
        return install_systemd_units(repo_root=repo_root, run_time=run_time)

    raise ValueError(f"Unsupported scheduler platform: {platform_name}")


def preview_scheduler(
    *,
    platform_name: str,
    task_name: str,
    run_time: str,
    repo_root: Path = REPO_ROOT,
    timezone_name: str = DEFAULT_TIMEZONE,
) -> dict[str, object]:
    """Return a non-mutating preview of the scheduler configuration."""

    if platform_name == "windows":
        plan = build_windows_task_plan(repo_root=repo_root, task_name=task_name, run_time=run_time)
        return {
            "platform": platform_name,
            "task_name": task_name,
            "run_time": run_time,
            "runner_path": str(plan.runner_path),
            "query_command": plan.query_command,
            "registration_script": plan.registration_script,
        }

    if platform_name == "cron":
        return {
            "platform": platform_name,
            "task_name": task_name,
            "run_time": run_time,
            "timezone": timezone_name,
            "cron_entry": build_cron_entry(
                repo_root=repo_root,
                run_time=run_time,
                timezone_name=timezone_name,
            ),
        }

    if platform_name == "systemd":
        return {
            "platform": platform_name,
            "task_name": task_name,
            "run_time": run_time,
            "service_unit": build_systemd_service(repo_root=repo_root),
            "timer_unit": build_systemd_timer(run_time=run_time),
        }

    raise ValueError(f"Unsupported scheduler platform: {platform_name}")


def install_windows_task(
    *,
    repo_root: Path = REPO_ROOT,
    task_name: str = DEFAULT_TASK_NAME,
    run_time: str,
) -> dict[str, object]:
    """Create or update a Windows scheduled task and return its query output."""

    plan = build_windows_task_plan(repo_root=repo_root, task_name=task_name, run_time=run_time)
    registration_mode = "powershell"
    try:
        _run_command(["powershell.exe", "-NoProfile", "-Command", plan.registration_script])
    except RuntimeError as exc:
        if not _is_access_denied_error(exc):
            raise
        registration_mode = "schtasks-fallback"
        _run_command(_build_schtasks_create_command(task_name=task_name, runner_path=plan.runner_path, run_time=run_time))

    query = _run_command(plan.query_command)
    return {
        "platform": "windows",
        "task_name": task_name,
        "run_time": run_time,
        "runner_path": str(plan.runner_path),
        "registration_mode": registration_mode,
        "query_command": plan.query_command,
        "query_output": query.stdout.strip(),
    }


def install_cron_entry(
    *,
    repo_root: Path = REPO_ROOT,
    run_time: str,
    timezone_name: str = DEFAULT_TIMEZONE,
) -> dict[str, object]:
    """Install or replace the user's cron entry for the daily pipeline."""

    existing = _run_command(["crontab", "-l"], allow_failure=True)
    existing_lines = []
    if existing.returncode == 0:
        existing_lines = _strip_managed_cron_entries(existing.stdout.splitlines())

    cron_entry = build_cron_entry(repo_root=repo_root, run_time=run_time, timezone_name=timezone_name)
    updated_content = "\n".join([*existing_lines, cron_entry]).strip() + "\n"
    _run_command(["crontab", "-"], input_text=updated_content)
    return {
        "platform": "cron",
        "run_time": run_time,
        "timezone": timezone_name,
        "cron_entry": cron_entry,
    }


def install_systemd_units(
    *,
    repo_root: Path = REPO_ROOT,
    run_time: str,
) -> dict[str, object]:
    """Write and enable systemd user units for the daily pipeline."""

    systemd_dir = Path.home() / ".config" / "systemd" / "user"
    systemd_dir.mkdir(parents=True, exist_ok=True)

    service_path = systemd_dir / "mlbprediction2026-daily.service"
    timer_path = systemd_dir / "mlbprediction2026-daily.timer"

    service_contents = build_systemd_service(repo_root=repo_root)
    timer_contents = build_systemd_timer(run_time=run_time)

    service_path.write_text(service_contents, encoding="utf-8")
    timer_path.write_text(timer_contents, encoding="utf-8")

    _run_command(["systemctl", "--user", "daemon-reload"])
    _run_command(["systemctl", "--user", "enable", "--now", timer_path.name])

    return {
        "platform": "systemd",
        "run_time": run_time,
        "service_path": str(service_path),
        "timer_path": str(timer_path),
    }


def resolve_platform(target: str) -> str:
    """Resolve auto/platform aliases into a supported scheduler backend."""

    normalized = target.casefold()
    if normalized in {"windows", "cron", "systemd"}:
        return normalized

    if normalized != "auto":
        raise ValueError(f"Unsupported platform option: {target}")

    if os.name == "nt":
        return "windows"
    if shutil.which("systemctl"):
        return "systemd"
    return "cron"


def _powershell_single_quoted(value: str) -> str:
    return value.replace("'", "''")


def _build_schtasks_create_command(*, task_name: str, runner_path: Path, run_time: str) -> list[str]:
    return [
        "schtasks.exe",
        "/Create",
        "/TN",
        task_name,
        "/SC",
        "DAILY",
        "/ST",
        run_time,
        "/TR",
        f'"{runner_path}"',
        "/F",
    ]


def _is_access_denied_error(exc: RuntimeError) -> bool:
    message = str(exc).casefold()
    return "access is denied" in message or "0x80070005" in message


def _strip_managed_cron_entries(lines: Sequence[str]) -> list[str]:
    """Remove prior managed cron entries in either legacy or block form."""

    cleaned_lines: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped_line = line.strip()

        if stripped_line == CRON_MARKER:
            index += 1
            if index < len(lines) and lines[index].startswith("CRON_TZ="):
                index += 1
            if index < len(lines):
                index += 1
            continue

        if CRON_MARKER in line:
            index += 1
            continue

        cleaned_lines.append(line)
        index += 1

    return cleaned_lines


def _run_command(
    command: list[str],
    *,
    input_text: str | None = None,
    allow_failure: bool = False,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        input=input_text,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0 and not allow_failure:
        stderr = completed.stderr.strip() or completed.stdout.strip() or "command failed"
        raise RuntimeError(f"{' '.join(command)} failed: {stderr}")
    return completed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Configure daily scheduler support for MLBPrediction2026")
    parser.add_argument("--platform", default="auto", help="windows, cron, systemd, or auto")
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME)
    parser.add_argument("--first-pitch", default=DEFAULT_FIRST_PITCH)
    parser.add_argument("--hours-before", type=int, default=DEFAULT_HOURS_BEFORE)
    parser.add_argument("--run-time", default=None, help="Override the computed daily run time (HH:MM)")
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    parser.add_argument("--print-only", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args(argv)

    platform_name = resolve_platform(args.platform)
    run_time = args.run_time or compute_daily_run_time(
        args.first_pitch,
        hours_before=args.hours_before,
    )
    task_name = args.task_name
    if args.test and platform_name == "windows":
        task_name = f"{task_name} (Test)"

    if args.print_only:
        payload = preview_scheduler(
            platform_name=platform_name,
            task_name=task_name,
            run_time=run_time,
            timezone_name=args.timezone,
        )
    else:
        payload = install_scheduler(
            platform_name=platform_name,
            task_name=task_name,
            run_time=run_time,
            timezone_name=args.timezone,
            test_mode=args.test,
        )

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CRON_MARKER",
    "DEFAULT_FIRST_PITCH",
    "DEFAULT_HOURS_BEFORE",
    "DEFAULT_TASK_NAME",
    "DEFAULT_TIMEZONE",
    "WindowsTaskPlan",
    "build_cron_entry",
    "build_systemd_service",
    "build_systemd_timer",
    "build_windows_task_plan",
    "compute_daily_run_time",
    "install_scheduler",
    "install_windows_task",
    "main",
    "preview_scheduler",
    "resolve_platform",
]
