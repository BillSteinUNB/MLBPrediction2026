from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import subprocess
import sys
import time

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
        autoresearch_agent.append_nightly_log(
            event_type="planner_self_check",
            heading="Planner self-check passed",
            body_lines=[
                f"session_id: `{session_id}`",
                f"provider: `{payload['provider']}`",
                f"model: `{payload['model']}`",
            ],
        )
        print(
            json.dumps(
                {
                    "event": "planner_self_check_passed",
                    "session_id": session_id,
                    "payload": payload,
                },
                indent=2,
                sort_keys=True,
            )
        )

    interrupted = False
    try:
        while True:
            pending_issues = autoresearch_agent.pending_suspected_issues(
                db_path=db_path,
                session_id=session_id,
            )
            if pending_issues:
                validation_payload = autoresearch_agent._run_validation_retest(
                    issue=pending_issues[0],
                    db_path=db_path,
                    train_path=train_path,
                    session_id=session_id,
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
            now = datetime.now(UTC)
            if should_start_fast_run(
                now=now,
                stop_at=stop_at,
                min_fast_window_minutes=min_fast_window_minutes,
            ):
                payload = autoresearch_agent.run_fast_once(
                    db_path=db_path,
                    program_path=program_path,
                    train_path=train_path,
                    exploration_mode=session_config.exploration_mode,
                    session_id=session_id,
                )
                print(
                    json.dumps(
                        {
                            "event": "fast_experiment_completed",
                            "session_id": session_id,
                            "stop_at": None if stop_at is None else stop_at.isoformat(),
                            "payload": payload,
                        },
                        indent=2,
                        sort_keys=True,
                    )
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
        autoresearch_agent.append_nightly_log(
            event_type="launcher_interrupted",
            heading="Launcher interrupted",
            body_lines=[f"session_id: `{session_id}`"],
        )

    if session_config.run_full_at_end:
        payload = autoresearch_agent.run_best_full(
            db_path=db_path,
            program_path=program_path,
            train_path=train_path,
            session_id=session_id,
        )
        summary_payload = _session_summary(
            db_path=db_path,
            session_id=session_id,
            status="completed" if not interrupted else "interrupted_then_full",
            run_full_at_end=session_config.run_full_at_end,
        )
        print(
            json.dumps(
                {
                    "event": "full_experiment_completed",
                    "session_id": session_id,
                    "payload": payload,
                    "session_summary": summary_payload,
                },
                indent=2,
                sort_keys=True,
            )
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
            status="completed" if not interrupted else "interrupted_then_full",
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
        status="completed" if not interrupted else "interrupted",
        run_full_at_end=session_config.run_full_at_end,
    )
    autoresearch_agent.finalize_session(
        session_id,
        db_path=db_path,
        status=str(summary["status"]),
        summary_json_path=summary["summary_json_path"],
        summary_md_path=summary["summary_md_path"],
    )
    print(
        json.dumps(
            {
                "event": "session_completed",
                "payload": summary,
            },
            indent=2,
            sort_keys=True,
        )
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
