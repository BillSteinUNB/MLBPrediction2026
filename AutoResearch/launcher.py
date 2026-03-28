from __future__ import annotations

import argparse
from collections import deque
from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

AUTORESEARCH_ROOT = Path(__file__).resolve().parent
REPO_ROOT = AUTORESEARCH_ROOT.parent
for path in (AUTORESEARCH_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import agent as autoresearch_agent  # noqa: E402

DEFAULT_DB_PATH = AUTORESEARCH_ROOT / "experiments.db"
DEFAULT_PROGRAM_PATH = AUTORESEARCH_ROOT / "program.md"
DEFAULT_TRAIN_PATH = AUTORESEARCH_ROOT / "train.py"
DEFAULT_MIN_FAST_WINDOW_MINUTES = 30
DEFAULT_POLL_INTERVAL_SECONDS = 15
_CONSOLE = Console()
_ESCAPE_KEY = "\x1b"

try:
    import msvcrt  # type: ignore
except ImportError:  # pragma: no cover - non-Windows fallback
    msvcrt = None


class _StopAfterCurrentRunController:
    def __init__(self) -> None:
        self.enabled = False
        self.supported = os.name == "nt" and msvcrt is not None

    def poll(self) -> bool:
        if not self.supported or msvcrt is None:
            return False
        while msvcrt.kbhit():
            key = msvcrt.getwch()
            if key == _ESCAPE_KEY:
                self.enabled = not self.enabled
                return True
        return False

    def status_text(self) -> str:
        if not self.supported:
            return "Esc toggle unavailable"
        return (
            "Esc: stop after current run ON"
            if self.enabled
            else "Esc: stop after current run OFF"
        )


def _trim_text(value: str | None, *, max_length: int = 220) -> str:
    normalized = " ".join(str(value or "").split())
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 3].rstrip() + "..."


def _print_event_panel(*, title: str, lines: list[str], border_style: str) -> None:
    _CONSOLE.print(Panel("\n".join(lines), title=title, border_style=border_style))


def _format_metrics(payload: dict[str, object]) -> list[str]:
    result = payload.get("result")
    if not isinstance(result, dict):
        return []
    metrics = result.get("metrics")
    model = result.get("model")
    if not isinstance(metrics, dict):
        return []
    lines = [
        (
            "Holdout: "
            f"R2={float(metrics['holdout_r2']) * 100:.2f}% "
            f"RMSE={float(metrics['holdout_rmse']):.4f} "
            f"Poisson={float(metrics['holdout_poisson_deviance']):.4f}"
        ),
        (
            "CV: "
            f"RMSE={float(metrics['cv_rmse']):.4f} "
            f"{metrics['cv_metric_name']}={float(metrics['cv_metric_value']):.4f}"
        ),
    ]
    if isinstance(model, dict):
        lines.append(f"Model: `{model.get('model_name', '<unknown>')}`")
    return lines


def _print_planned_run(
    *,
    run_number: int,
    kind_label: str,
    prepared_run: autoresearch_agent.PreparedExperimentRun,
) -> None:
    decision = prepared_run.planner_decision
    proposal = decision.proposal
    planner_label = decision.planner_type
    if decision.planner_model:
        planner_label = f"{planner_label} via {decision.planner_model}"
    _print_event_panel(
        title=f"{kind_label} Run {run_number} Plan",
        border_style="cyan",
        lines=[
            f"Planner: {planner_label}",
            f"Hypothesis: {_trim_text(decision.hypothesis, max_length=260)}",
            f"Why: {_trim_text(decision.reasoning, max_length=260)}",
            (
                "Config: "
                f"selector={proposal.selector_type} "
                f"max_features={proposal.max_features} "
                f"forced_delta_count={proposal.forced_delta_count} "
                f"trials={proposal.trials} folds={proposal.folds}"
            ),
            f"Buckets: {proposal.bucket_quotas}",
            f"Force include: {proposal.force_include_patterns or '[]'}",
        ],
    )


def _print_run_result(*, run_number: int, kind: str, payload: dict[str, object]) -> None:
    status = str(payload.get("status", "unknown"))
    lines = [
        f"Experiment: `{payload.get('experiment_name', '<unknown>')}`",
        f"Status: `{status}`",
    ]
    result_lines = _format_metrics(payload)
    if result_lines:
        lines.extend(result_lines)
    if status != "succeeded":
        error_message = _trim_text(str(payload.get("error_message") or ""), max_length=260)
        if error_message:
            lines.append(f"Error: {error_message}")
        lines.append(f"stdout: `{payload.get('stdout_path', '<missing>')}`")
        lines.append(f"stderr: `{payload.get('stderr_path', '<missing>')}`")
    note_ids = payload.get("note_ids")
    if note_ids:
        lines.append(f"Notes recorded: {note_ids}")
    _print_event_panel(
        title=f"{kind.title()} Run {run_number} Result",
        border_style="green" if status == "succeeded" else "red",
        lines=lines,
    )


def _render_live_output_panel(lines: list[str]) -> Panel:
    body = "\n".join(lines) if lines else "Waiting for trainer output..."
    return Panel(body, title="Trainer Output", border_style="blue")


def _render_run_table(rows: list[dict[str, object]], current_status: str) -> Table:
    table = Table(title="AutoResearch Overnight Progress")
    table.add_column("Run")
    table.add_column("Kind")
    table.add_column("Elapsed")
    table.add_column("Status")
    for row in rows:
        table.add_row(str(row["run_number"]), str(row["kind"]), str(row["elapsed"]), str(row["status"]))
    table.caption = current_status
    return table


def _render_live_view(
    rows: list[dict[str, object]],
    current_status: str,
    live_output_lines: list[str],
):
    return Group(
        _render_run_table(rows, current_status),
        _render_live_output_panel(live_output_lines),
    )


def _run_with_live_timer(
    *,
    rows: list[dict[str, object]],
    kind: str,
    action,
    live_output_lines: deque[str] | None = None,
    stop_controller: _StopAfterCurrentRunController | None = None,
):
    run_number = len(rows) + 1
    row = {"run_number": run_number, "kind": kind, "elapsed": "00:00", "status": "running"}
    rows.append(row)
    started = time.monotonic()
    if live_output_lines is not None:
        live_output_lines.clear()
    current_status = f"{kind} {run_number} running"
    if stop_controller is not None:
        current_status = f"{current_status} | {stop_controller.status_text()}"
    with Live(
        _render_live_view(rows, current_status, list(live_output_lines or [])),
        console=_CONSOLE,
        refresh_per_second=4,
    ) as live:
        while True:
            if hasattr(action, "done") and action.done():
                break
            if stop_controller is not None and stop_controller.poll():
                _print_event_panel(
                    title="Stop Toggle",
                    border_style="yellow",
                    lines=[stop_controller.status_text()],
                )
            elapsed_seconds = int(time.monotonic() - started)
            row["elapsed"] = f"{elapsed_seconds // 60:02d}:{elapsed_seconds % 60:02d}"
            current_status = f"{kind} {run_number} running"
            if stop_controller is not None:
                current_status = f"{current_status} | {stop_controller.status_text()}"
            live.update(_render_live_view(rows, current_status, list(live_output_lines or [])))
            time.sleep(0.5)
        payload = action.result()
        elapsed_seconds = int(time.monotonic() - started)
        row["elapsed"] = f"{elapsed_seconds // 60:02d}:{elapsed_seconds % 60:02d}"
        row["status"] = str(payload.get("status", "completed"))
        current_status = f"{kind} {run_number} finished"
        if stop_controller is not None:
            current_status = f"{current_status} | {stop_controller.status_text()}"
        live.update(_render_live_view(rows, current_status, list(live_output_lines or [])))
    return payload


def _run_git_command(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _git_branch_name(now: datetime | None = None) -> str:
    current = datetime.now(UTC) if now is None else now
    return f"AutoResearch-{current.date().isoformat()}"


def prepare_git_checkpoint(*, reports_dir: str | Path = autoresearch_agent.DEFAULT_REPORTS_DIR) -> dict[str, str]:
    status_before = _run_git_command("status", "--short")
    if status_before.returncode != 0:
        raise RuntimeError(status_before.stderr.strip() or "git status failed")
    branch_before = _run_git_command("rev-parse", "--abbrev-ref", "HEAD")
    if branch_before.returncode != 0:
        raise RuntimeError(branch_before.stderr.strip() or "git rev-parse failed")

    commit_message = f"Start of auto research {datetime.now(UTC).date().isoformat()}"
    add_result = _run_git_command("add", "-A")
    if add_result.returncode != 0:
        raise RuntimeError(add_result.stderr.strip() or "git add failed")
    commit_result = _run_git_command("commit", "-m", commit_message)
    if commit_result.returncode != 0 and "nothing to commit" not in (commit_result.stdout + commit_result.stderr).lower():
        raise RuntimeError(commit_result.stderr.strip() or commit_result.stdout.strip() or "git commit failed")
    push_main = _run_git_command("push", "origin", "main")
    if push_main.returncode != 0:
        raise RuntimeError(push_main.stderr.strip() or push_main.stdout.strip() or "git push origin main failed")

    branch_name = _git_branch_name()
    checkout_result = _run_git_command("checkout", "-B", branch_name)
    if checkout_result.returncode != 0:
        raise RuntimeError(checkout_result.stderr.strip() or checkout_result.stdout.strip() or "git checkout failed")
    push_branch = _run_git_command("push", "-u", "origin", branch_name)
    if push_branch.returncode != 0:
        raise RuntimeError(push_branch.stderr.strip() or push_branch.stdout.strip() or "git push branch failed")
    autoresearch_agent.append_debug_trace(
        event_type="git_checkpoint",
        payload={
            "branch_before": branch_before.stdout.strip(),
            "status_before": status_before.stdout.strip(),
            "commit_message": commit_message,
            "branch_name": branch_name,
        },
    )

    autoresearch_agent.append_nightly_log(
        event_type="git_checkpoint",
        heading="Startup git checkpoint",
        body_lines=[
            f"branch_before: `{branch_before.stdout.strip()}`",
            f"status_before: `{status_before.stdout.strip() or 'clean'}`",
            f"checkpoint_commit: `{commit_message}`",
            f"night_branch: `{branch_name}`",
        ],
        reports_dir=reports_dir,
    )
    return {
        "branch_before": branch_before.stdout.strip(),
        "status_before": status_before.stdout.strip(),
        "checkpoint_commit": commit_message,
        "branch_name": branch_name,
    }


def resolve_stop_at(*, started_at: datetime, duration_hours: int) -> datetime | None:
    if duration_hours <= 0:
        return None
    return started_at + timedelta(hours=duration_hours)


def should_start_fast_run(
    *,
    now: datetime,
    stop_at: datetime | None,
    min_fast_window_minutes: int,
) -> bool:
    if stop_at is None:
        return True
    if stop_at.tzinfo is None and now.tzinfo is not None:
        now = now.replace(tzinfo=None)
    elif stop_at.tzinfo is not None and now.tzinfo is None:
        stop_at = stop_at.replace(tzinfo=None)
    remaining_seconds = (stop_at - now).total_seconds()
    return remaining_seconds >= (min_fast_window_minutes * 60)


def _prompt_hours() -> int:
    while True:
        raw = input("How many hours should autoresearch run? Enter 0 for until interrupted (0-24): ").strip()
        try:
            value = int(raw)
        except ValueError:
            print("Please enter a whole number from 0 to 24.")
            continue
        if 0 <= value <= 24:
            return value
        print("Please enter a whole number from 0 to 24.")


def _prompt_yes_no(prompt: str, *, default: bool) -> bool:
    suffix = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{prompt} [{suffix}]: ").strip().lower()
        if not raw:
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Please answer yes or no.")


def collect_session_config(
    *,
    duration_hours: int | None,
    exploration_mode: str | None,
    run_full_at_end: bool | None,
) -> autoresearch_agent.AutoresearchSessionConfig:
    resolved_hours = _prompt_hours() if duration_hours is None else int(duration_hours)
    if not 0 <= resolved_hours <= 24:
        raise ValueError("duration_hours must be between 0 and 24")
    resolved_mode = "fast" if exploration_mode is None else exploration_mode
    if resolved_mode != "fast":
        raise ValueError("exploration_mode must be 'fast'")
    resolved_run_full = (
        _prompt_yes_no("Run a promoted full training job at the end?", default=True)
        if run_full_at_end is None
        else bool(run_full_at_end)
    )
    return autoresearch_agent.AutoresearchSessionConfig(
        exploration_mode=resolved_mode,
        duration_hours=resolved_hours,
        until_interrupted=resolved_hours == 0,
        run_full_at_end=resolved_run_full,
    )


def _session_summary(
    *,
    db_path: str | Path,
    session_id: int,
    status: str,
    run_full_at_end: bool,
) -> dict[str, object]:
    summary, summary_json_path, summary_md_path, review_prompt_path = autoresearch_agent.write_session_summary(
        session_id,
        db_path=db_path,
        status_override=status,
    )
    payload = {
        "status": status,
        "session_id": session_id,
        "run_full_at_end": run_full_at_end,
        "summary_json_path": str(summary_json_path),
        "summary_md_path": str(summary_md_path),
        "review_prompt_path": str(review_prompt_path),
        "best_experiment_name": None
        if summary["best_exploration"] is None
        else summary["best_exploration"]["experiment_name"],
        "best_holdout_r2": None
        if summary["best_exploration"] is None
        else summary["best_exploration"]["holdout_r2"],
        "best_cv_rmse": None
        if summary["best_exploration"] is None
        else summary["best_exploration"]["cv_rmse"],
        "note_count": len(summary["notes"]),
        "recommendations": summary["recommendations"],
    }
    return payload


def run_launcher(
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    program_path: str | Path = DEFAULT_PROGRAM_PATH,
    train_path: str | Path = DEFAULT_TRAIN_PATH,
    duration_hours: int | None = None,
    exploration_mode: str | None = None,
    run_full_at_end: bool | None = None,
    min_fast_window_minutes: int = DEFAULT_MIN_FAST_WINDOW_MINUTES,
    poll_interval_seconds: int = DEFAULT_POLL_INTERVAL_SECONDS,
    planner_self_check: bool = True,
) -> dict[str, object]:
    git_checkpoint = prepare_git_checkpoint()
    session_config = collect_session_config(
        duration_hours=duration_hours,
        exploration_mode=exploration_mode,
        run_full_at_end=run_full_at_end,
    )
    started_at = datetime.now(UTC)
    stop_at = resolve_stop_at(started_at=started_at, duration_hours=session_config.duration_hours)
    autoresearch_agent.ensure_experiment_db(db_path)
    session_id = autoresearch_agent.create_session(
        db_path=db_path,
        config=session_config,
        stop_at=stop_at,
    )
    autoresearch_agent.append_nightly_log(
        event_type="session_start",
        heading="AutoResearch session started",
        body_lines=[
            f"session_id: `{session_id}`",
            f"exploration_mode: `{session_config.exploration_mode}`",
            f"duration_hours: `{session_config.duration_hours}`",
            f"run_full_at_end: `{session_config.run_full_at_end}`",
            f"git_branch: `{git_checkpoint['branch_name']}`",
        ],
    )
    if planner_self_check:
        payload = autoresearch_agent.run_planner_self_check()
        autoresearch_agent.append_debug_trace(
            event_type="planner_self_check_passed",
            payload={"session_id": session_id, **payload},
        )
        autoresearch_agent.append_nightly_log(
            event_type="planner_self_check",
            heading="Planner self-check passed",
            body_lines=[
                f"session_id: `{session_id}`",
                f"provider: `{payload['provider']}`",
                f"model: `{payload['model']}`",
            ],
        )
        _print_event_panel(
            title="Planner Self-Check",
            border_style="green",
            lines=[
                f"Session: `{session_id}`",
                f"Provider: `{payload['provider']}`",
                f"Model: `{payload['model']}`",
                f"Response: `{payload.get('response_text', 'OK')}`",
            ],
        )

    interrupted = False
    graceful_stop_requested = False
    run_rows: list[dict[str, object]] = []
    live_output_lines: deque[str] = deque(maxlen=10)
    stop_controller = _StopAfterCurrentRunController()
    successful_fast_runs_since_review = 0
    active_improvement_anchor_id: int | None = None
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        while True:
            if stop_controller.poll():
                _print_event_panel(
                    title="Stop Toggle",
                    border_style="yellow",
                    lines=[stop_controller.status_text()],
                )
            if stop_controller.enabled and run_rows:
                graceful_stop_requested = True
                break
            pending_issues = autoresearch_agent.pending_fix_validation_issues(
                db_path=db_path,
                session_id=session_id,
            )
            if pending_issues:
                future = executor.submit(
                    autoresearch_agent._run_validation_retest,
                    issue=pending_issues[0],
                    db_path=db_path,
                    train_path=train_path,
                    session_id=session_id,
                    live_line_callback=live_output_lines.append,
                )
                validation_payload = _run_with_live_timer(
                    rows=run_rows,
                    kind="validation",
                    action=future,
                    live_output_lines=live_output_lines,
                    stop_controller=stop_controller,
                )
                print(
                    json.dumps(
                        {
                            "event": "issue_validation_completed",
                            "session_id": session_id,
                            "payload": validation_payload,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
                autoresearch_agent.append_debug_trace(
                    event_type="issue_validation_completed",
                    payload={"session_id": session_id, **validation_payload},
                )
                _print_run_result(
                    run_number=len(run_rows),
                    kind="validation-fix",
                    payload=validation_payload,
                )
                successful_fast_runs_since_review = 0
                active_improvement_anchor_id = None
                if validation_payload.get("status") != "succeeded":
                    time.sleep(poll_interval_seconds)
                continue
            prepared_validation_run = None
            if active_improvement_anchor_id is not None:
                with _CONSOLE.status("Planning improvement validation...", spinner="dots"):
                    prepared_validation_run = autoresearch_agent.prepare_improvement_validation(
                        anchor_experiment_id=active_improvement_anchor_id,
                        db_path=db_path,
                        train_path=train_path,
                        session_id=session_id,
                    )
                if prepared_validation_run is None:
                    active_improvement_anchor_id = None
                else:
                    _print_planned_run(
                        run_number=len(run_rows) + 1,
                        kind_label="Validation Improve",
                        prepared_run=prepared_validation_run,
                    )
            elif successful_fast_runs_since_review >= 3:
                candidate = autoresearch_agent.select_promising_fast_candidate(
                    db_path=db_path,
                    session_id=session_id,
                )
                successful_fast_runs_since_review = 0
                if candidate is not None:
                    active_improvement_anchor_id = int(candidate["id"])
                    with _CONSOLE.status("Planning improvement validation...", spinner="dots"):
                        prepared_validation_run = autoresearch_agent.prepare_improvement_validation(
                            anchor_experiment_id=active_improvement_anchor_id,
                            db_path=db_path,
                            train_path=train_path,
                            session_id=session_id,
                        )
                    if prepared_validation_run is not None:
                        _print_planned_run(
                            run_number=len(run_rows) + 1,
                            kind_label="Validation Improve",
                            prepared_run=prepared_validation_run,
                        )
                    else:
                        active_improvement_anchor_id = None
            if prepared_validation_run is not None:
                future = executor.submit(
                    autoresearch_agent.execute_prepared_fast_once,
                    prepared_run=prepared_validation_run,
                    db_path=db_path,
                    train_path=train_path,
                    live_line_callback=live_output_lines.append,
                )
                payload = _run_with_live_timer(
                    rows=run_rows,
                    kind="validation-improve",
                    action=future,
                    live_output_lines=live_output_lines,
                    stop_controller=stop_controller,
                )
                autoresearch_agent.append_debug_trace(
                    event_type="validation_improvement_completed",
                    payload={"session_id": session_id, **payload},
                )
                autoresearch_agent.append_nightly_log(
                    event_type="validation_improvement",
                    heading="Improvement validation completed",
                    body_lines=[
                        f"session_id: `{session_id}`",
                        f"experiment_id: `{payload['experiment_id']}`",
                        f"status: `{payload['status']}`",
                        f"experiment_name: `{payload['experiment_name']}`",
                        f"parent_experiment_id: `{active_improvement_anchor_id}`",
                    ],
                )
                _print_run_result(
                    run_number=len(run_rows),
                    kind="validation-improve",
                    payload=payload,
                )
                if payload["status"] != "succeeded":
                    active_improvement_anchor_id = None
                    time.sleep(poll_interval_seconds)
                continue
            now = datetime.now(UTC)
            if should_start_fast_run(
                now=now,
                stop_at=stop_at,
                min_fast_window_minutes=min_fast_window_minutes,
            ):
                with _CONSOLE.status("Planning next fast run...", spinner="dots"):
                    prepared_fast_run = autoresearch_agent.prepare_fast_once(
                        db_path=db_path,
                        program_path=program_path,
                        train_path=train_path,
                        exploration_mode=session_config.exploration_mode,
                        session_id=session_id,
                    )
                _print_planned_run(
                    run_number=len(run_rows) + 1,
                    kind_label="Fast",
                    prepared_run=prepared_fast_run,
                )
                future = executor.submit(
                    autoresearch_agent.execute_prepared_fast_once,
                    prepared_run=prepared_fast_run,
                    db_path=db_path,
                    train_path=train_path,
                    live_line_callback=live_output_lines.append,
                )
                payload = _run_with_live_timer(
                    rows=run_rows,
                    kind="fast",
                    action=future,
                    live_output_lines=live_output_lines,
                    stop_controller=stop_controller,
                )
                autoresearch_agent.append_debug_trace(
                    event_type="fast_experiment_completed",
                    payload={"session_id": session_id, **payload},
                )
                autoresearch_agent.append_nightly_log(
                    event_type="fast_experiment",
                    heading="Fast experiment completed",
                    body_lines=[
                        f"session_id: `{session_id}`",
                        f"experiment_id: `{payload['experiment_id']}`",
                        f"status: `{payload['status']}`",
                        f"experiment_name: `{payload['experiment_name']}`",
                        f"note_ids: `{payload.get('note_ids', [])}`",
                    ],
                )
                _print_run_result(
                    run_number=len(run_rows),
                    kind="fast",
                    payload=payload,
                )
                if payload["status"] == "succeeded":
                    successful_fast_runs_since_review += 1
                else:
                    successful_fast_runs_since_review = 0
                    active_improvement_anchor_id = None
                if payload["status"] != "succeeded":
                    time.sleep(poll_interval_seconds)
                continue
            break
    except KeyboardInterrupt:
        interrupted = True
        print(
            json.dumps(
                {
                    "event": "launcher_interrupted",
                    "session_id": session_id,
                },
                indent=2,
                sort_keys=True,
            )
        )
        autoresearch_agent.append_debug_trace(
            event_type="launcher_interrupted",
            payload={"session_id": session_id},
        )
        autoresearch_agent.append_nightly_log(
            event_type="launcher_interrupted",
            heading="Launcher interrupted",
            body_lines=[f"session_id: `{session_id}`"],
        )

    if session_config.run_full_at_end:
        future = executor.submit(
            autoresearch_agent.run_best_full,
            db_path=db_path,
            program_path=program_path,
            train_path=train_path,
            session_id=session_id,
            live_line_callback=live_output_lines.append,
        )
        payload = _run_with_live_timer(
            rows=run_rows,
            kind="full",
            action=future,
            live_output_lines=live_output_lines,
            stop_controller=stop_controller,
        )
        _print_run_result(
            run_number=len(run_rows),
            kind="full",
            payload=payload,
        )
        summary_payload = _session_summary(
            db_path=db_path,
            session_id=session_id,
            status=(
                "interrupted_then_full"
                if interrupted
                else "stopped_after_run_then_full"
                if graceful_stop_requested
                else "completed"
            ),
            run_full_at_end=session_config.run_full_at_end,
        )
        autoresearch_agent.append_debug_trace(
            event_type="full_experiment_completed",
            payload={"session_id": session_id, **payload, "session_summary": summary_payload},
        )
        autoresearch_agent.append_nightly_log(
            event_type="full_experiment",
            heading="Promoted full experiment completed",
            body_lines=[
                f"session_id: `{session_id}`",
                f"status: `{payload['status']}`",
                f"summary_md_path: `{summary_payload['summary_md_path']}`",
            ],
        )
        autoresearch_agent.finalize_session(
            session_id,
            db_path=db_path,
            status=(
                "interrupted_then_full"
                if interrupted
                else "stopped_after_run_then_full"
                if graceful_stop_requested
                else "completed"
            ),
            summary_json_path=summary_payload["summary_json_path"],
            summary_md_path=summary_payload["summary_md_path"],
        )
        return {
            **payload,
            "session_summary": summary_payload,
        }

    summary = _session_summary(
        db_path=db_path,
        session_id=session_id,
        status=(
            "interrupted"
            if interrupted
            else "stopped_after_run"
            if graceful_stop_requested
            else "completed"
        ),
        run_full_at_end=session_config.run_full_at_end,
    )
    autoresearch_agent.finalize_session(
        session_id,
        db_path=db_path,
        status=str(summary["status"]),
        summary_json_path=summary["summary_json_path"],
        summary_md_path=summary["summary_md_path"],
    )
    _print_event_panel(
        title="Session Complete",
        border_style="green",
        lines=[
            f"Session: `{summary['session_id']}`",
            f"Status: `{summary['status']}`",
            f"Best experiment: `{summary['best_experiment_name']}`",
            (
                "Best metrics: "
                f"holdout_r2={summary['best_holdout_r2']} "
                f"cv_rmse={summary['best_cv_rmse']}"
            ),
            f"Summary: `{summary['summary_md_path']}`",
        ],
    )
    autoresearch_agent.append_nightly_log(
        event_type="session_completed",
        heading="Session completed",
        body_lines=[
            f"session_id: `{session_id}`",
            f"status: `{summary['status']}`",
            f"summary_md_path: `{summary['summary_md_path']}`",
        ],
    )
    autoresearch_agent.append_debug_trace(
        event_type="session_completed",
        payload={"session_id": session_id, **summary},
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch overnight autoresearch experiments")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--program-path", default=str(DEFAULT_PROGRAM_PATH))
    parser.add_argument("--train-path", default=str(DEFAULT_TRAIN_PATH))
    parser.add_argument("--hours", type=int)
    parser.add_argument("--exploration-mode", choices=("fast",))
    parser.add_argument("--run-full-at-end", dest="run_full_at_end", action="store_true")
    parser.add_argument("--no-run-full-at-end", dest="run_full_at_end", action="store_false")
    parser.set_defaults(run_full_at_end=None)
    parser.add_argument("--min-fast-window-minutes", type=int, default=DEFAULT_MIN_FAST_WINDOW_MINUTES)
    parser.add_argument("--poll-interval-seconds", type=int, default=DEFAULT_POLL_INTERVAL_SECONDS)
    parser.add_argument("--skip-planner-self-check", action="store_true")
    args = parser.parse_args(argv)

    payload = run_launcher(
        db_path=args.db_path,
        program_path=args.program_path,
        train_path=args.train_path,
        duration_hours=args.hours,
        exploration_mode=args.exploration_mode,
        run_full_at_end=args.run_full_at_end,
        min_fast_window_minutes=args.min_fast_window_minutes,
        poll_interval_seconds=args.poll_interval_seconds,
        planner_self_check=not args.skip_planner_self_check,
    )
    return 0 if payload["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
