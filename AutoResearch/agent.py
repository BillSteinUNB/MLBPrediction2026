from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import importlib
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
from typing import Any, Sequence

AUTORESEARCH_ROOT = Path(__file__).resolve().parent
REPO_ROOT = AUTORESEARCH_ROOT.parent
for path in (AUTORESEARCH_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train as train_module  # noqa: E402
import llm_client  # noqa: E402

DEFAULT_DB_PATH = AUTORESEARCH_ROOT / "experiments.db"
DEFAULT_PROGRAM_PATH = AUTORESEARCH_ROOT / "program.md"
DEFAULT_TRAIN_PATH = AUTORESEARCH_ROOT / "train.py"
DEFAULT_LOG_DIR = AUTORESEARCH_ROOT / "logs" / "autoresearch"
DEFAULT_REPORTS_DIR = AUTORESEARCH_ROOT / "reports"
_FORCE_7G_PATTERNS = ["*_7g", "*_7s", "*_delta_7v30g", "*_delta_7v30s"]


@dataclass(frozen=True, slots=True)
class ExperimentProposal:
    max_features: int
    selector_type: str
    bucket_quotas: list[int]
    exclude_patterns: list[str]
    force_include_patterns: list[str]
    forced_delta_count: int
    trials: int
    folds: int
    rationale: str


@dataclass(frozen=True, slots=True)
class PlannerDecision:
    proposal: ExperimentProposal
    hypothesis: str
    planner_type: str
    planner_model: str | None
    reasoning: str
    prompt_text: str | None = None
    response_text: str | None = None


@dataclass(frozen=True, slots=True)
class AutoresearchSessionConfig:
    exploration_mode: str
    duration_hours: int
    until_interrupted: bool
    run_full_at_end: bool


@dataclass(frozen=True, slots=True)
class SuspectedIssue:
    note_id: int
    experiment_id: int | None
    title: str
    body: str
    metadata: dict[str, Any]
    created_at: str


def ensure_experiment_db(db_path: str | Path = DEFAULT_DB_PATH) -> Path:
    resolved_db_path = Path(db_path)
    resolved_db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(resolved_db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                session_id INTEGER,
                experiment_id INTEGER,
                note_type TEXT NOT NULL,
                importance TEXT NOT NULL,
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                metadata_json TEXT,
                FOREIGN KEY(session_id) REFERENCES sessions(id),
                FOREIGN KEY(experiment_id) REFERENCES experiments(id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS issue_validations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                session_id INTEGER,
                issue_note_id INTEGER NOT NULL,
                source_experiment_id INTEGER,
                validation_experiment_id INTEGER,
                status TEXT NOT NULL,
                outcome TEXT,
                details_json TEXT,
                FOREIGN KEY(session_id) REFERENCES sessions(id),
                FOREIGN KEY(issue_note_id) REFERENCES notes(id),
                FOREIGN KEY(source_experiment_id) REFERENCES experiments(id),
                FOREIGN KEY(validation_experiment_id) REFERENCES experiments(id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                status TEXT NOT NULL,
                exploration_mode TEXT NOT NULL,
                duration_hours INTEGER NOT NULL,
                until_interrupted INTEGER NOT NULL,
                run_full_at_end INTEGER NOT NULL,
                stop_at TEXT,
                summary_json_path TEXT,
                summary_md_path TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                mode TEXT NOT NULL,
                status TEXT NOT NULL,
                hypothesis TEXT NOT NULL,
                config_json TEXT NOT NULL,
                config_fingerprint TEXT NOT NULL,
                metrics_json TEXT,
                duration_seconds REAL,
                experiment_name TEXT NOT NULL,
                output_dir TEXT,
                summary_path TEXT,
                stdout_path TEXT,
                stderr_path TEXT,
                holdout_r2 REAL,
                holdout_rmse REAL,
                cv_rmse REAL,
                cv_metric_name TEXT,
                cv_metric_value REAL,
                returncode INTEGER,
                session_id INTEGER,
                parent_experiment_id INTEGER,
                error_message TEXT,
                planner_type TEXT,
                planner_model TEXT,
                planner_notes TEXT,
                planner_prompt_path TEXT,
                planner_response_path TEXT,
                FOREIGN KEY(session_id) REFERENCES sessions(id),
                FOREIGN KEY(parent_experiment_id) REFERENCES experiments(id)
            )
            """
        )
        _ensure_experiment_columns(
            connection,
            {
                "session_id": "INTEGER",
                "planner_type": "TEXT",
                "planner_model": "TEXT",
                "planner_notes": "TEXT",
                "planner_prompt_path": "TEXT",
                "planner_response_path": "TEXT",
            },
        )
        _ensure_session_columns(
            connection,
            {
                "summary_json_path": "TEXT",
                "summary_md_path": "TEXT",
            },
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status, started_at DESC)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_experiments_mode_status ON experiments(mode, status)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_experiments_score ON experiments(mode, holdout_r2 DESC, cv_rmse ASC)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_experiments_session ON experiments(session_id, started_at ASC)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_notes_session ON notes(session_id, created_at ASC)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_issue_validations_session ON issue_validations(session_id, created_at ASC)"
        )
        connection.commit()
    return resolved_db_path


def _ensure_experiment_columns(
    connection: sqlite3.Connection,
    expected_columns: dict[str, str],
) -> None:
    existing_columns = {
        str(row[1])
        for row in connection.execute("PRAGMA table_info(experiments)").fetchall()
    }
    for column_name, column_type in expected_columns.items():
        if column_name in existing_columns:
            continue
        connection.execute(f"ALTER TABLE experiments ADD COLUMN {column_name} {column_type}")


def _ensure_session_columns(
    connection: sqlite3.Connection,
    expected_columns: dict[str, str],
) -> None:
    existing_columns = {
        str(row[1])
        for row in connection.execute("PRAGMA table_info(sessions)").fetchall()
    }
    for column_name, column_type in expected_columns.items():
        if column_name in existing_columns:
            continue
        connection.execute(f"ALTER TABLE sessions ADD COLUMN {column_name} {column_type}")


def create_session(
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    config: AutoresearchSessionConfig,
    stop_at: datetime | None,
) -> int:
    ensure_experiment_db(db_path)
    with _connect(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO sessions (
                started_at,
                status,
                exploration_mode,
                duration_hours,
                until_interrupted,
                run_full_at_end,
                stop_at
            )
            VALUES (?, 'running', ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(UTC).isoformat(),
                config.exploration_mode,
                int(config.duration_hours),
                int(config.until_interrupted),
                int(config.run_full_at_end),
                None if stop_at is None else stop_at.isoformat(),
            ),
        )
        connection.commit()
        return int(cursor.lastrowid)


def load_session(
    session_id: int,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> sqlite3.Row:
    with _connect(db_path) as connection:
        row = connection.execute(
            "SELECT * FROM sessions WHERE id = ?",
            (int(session_id),),
        ).fetchone()
    if row is None:
        raise ValueError(f"Unknown autoresearch session id: {session_id}")
    return row


def finalize_session(
    session_id: int,
    *,
    status: str,
    summary_json_path: str | Path | None = None,
    summary_md_path: str | Path | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> None:
    with _connect(db_path) as connection:
        connection.execute(
            """
            UPDATE sessions
            SET ended_at = ?, status = ?, summary_json_path = ?, summary_md_path = ?
            WHERE id = ?
            """,
            (
                datetime.now(UTC).isoformat(),
                status,
                None if summary_json_path is None else str(summary_json_path),
                None if summary_md_path is None else str(summary_md_path),
                int(session_id),
            ),
        )
        connection.commit()


def load_experiment(
    experiment_id: int,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> sqlite3.Row:
    with _connect(db_path) as connection:
        row = connection.execute(
            "SELECT * FROM experiments WHERE id = ?",
            (int(experiment_id),),
        ).fetchone()
    if row is None:
        raise ValueError(f"Unknown experiment id: {experiment_id}")
    return row


def record_note(
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    session_id: int | None,
    experiment_id: int | None,
    note_type: str,
    importance: str,
    title: str,
    body: str,
    metadata: dict[str, Any] | None = None,
    reports_dir: str | Path = DEFAULT_REPORTS_DIR,
) -> int:
    payload = json.dumps(metadata or {}, sort_keys=True)
    created_at = datetime.now(UTC).isoformat()
    with _connect(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO notes (
                created_at,
                session_id,
                experiment_id,
                note_type,
                importance,
                title,
                body,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                session_id,
                experiment_id,
                note_type,
                importance,
                title,
                body,
                payload,
            ),
        )
        connection.commit()
        note_id = int(cursor.lastrowid)

    resolved_reports_dir = Path(reports_dir)
    resolved_reports_dir.mkdir(parents=True, exist_ok=True)
    note_log_path = resolved_reports_dir / "notes.jsonl"
    with note_log_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "id": note_id,
                    "created_at": created_at,
                    "session_id": session_id,
                    "experiment_id": experiment_id,
                    "note_type": note_type,
                    "importance": importance,
                    "title": title,
                    "body": body,
                    "metadata": metadata or {},
                },
                sort_keys=True,
            )
            + "\n"
        )
    return note_id


def append_nightly_log(
    *,
    event_type: str,
    heading: str,
    body_lines: Sequence[str],
    reports_dir: str | Path = DEFAULT_REPORTS_DIR,
) -> Path:
    resolved_reports_dir = Path(reports_dir)
    resolved_reports_dir.mkdir(parents=True, exist_ok=True)
    log_path = resolved_reports_dir / "nightly_log.md"
    timestamp = datetime.now(UTC).isoformat()
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"## {timestamp} - {heading}\n")
        handle.write(f"- event_type: `{event_type}`\n")
        for line in body_lines:
            handle.write(f"- {line}\n")
        handle.write("\n")
    return log_path


def load_notes(
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    session_id: int | None = None,
) -> list[sqlite3.Row]:
    query = "SELECT * FROM notes"
    parameters: list[Any] = []
    if session_id is not None:
        query += " WHERE session_id = ?"
        parameters.append(int(session_id))
    query += " ORDER BY created_at ASC, id ASC"
    with _connect(db_path) as connection:
        rows = connection.execute(query, parameters).fetchall()
    return rows


def identify_suspected_issues(
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    session_id: int | None = None,
) -> list[SuspectedIssue]:
    suspicious_types = {"failure", "warning", "suspicious_issue"}
    issues: list[SuspectedIssue] = []
    for row in load_notes(db_path=db_path, session_id=session_id):
        note_type = str(row["note_type"])
        importance = str(row["importance"])
        title = str(row["title"]).lower()
        metadata = json.loads(str(row["metadata_json"] or "{}"))
        if (
            note_type in suspicious_types
            or importance == "high"
            or bool(metadata.get("suspicious"))
            or "broken" in title
            or "suspicious" in title
        ):
            issues.append(
                SuspectedIssue(
                    note_id=int(row["id"]),
                    experiment_id=None if row["experiment_id"] is None else int(row["experiment_id"]),
                    title=str(row["title"]),
                    body=str(row["body"]),
                    metadata=metadata,
                    created_at=str(row["created_at"]),
                )
            )
    return issues


def pending_suspected_issues(
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    session_id: int,
) -> list[SuspectedIssue]:
    issues = identify_suspected_issues(db_path=db_path, session_id=session_id)
    if not issues:
        return []
    with _connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT issue_note_id
            FROM issue_validations
            WHERE session_id = ?
            """,
            (int(session_id),),
        ).fetchall()
    validated_note_ids = {int(row["issue_note_id"]) for row in rows}
    return [issue for issue in issues if issue.note_id not in validated_note_ids]


def record_issue_validation(
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    session_id: int | None,
    issue_note_id: int,
    source_experiment_id: int | None,
    validation_experiment_id: int | None,
    status: str,
    outcome: str,
    details: dict[str, Any] | None = None,
) -> int:
    created_at = datetime.now(UTC).isoformat()
    payload = json.dumps(details or {}, sort_keys=True)
    with _connect(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO issue_validations (
                created_at,
                session_id,
                issue_note_id,
                source_experiment_id,
                validation_experiment_id,
                status,
                outcome,
                details_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                session_id,
                int(issue_note_id),
                source_experiment_id,
                validation_experiment_id,
                status,
                outcome,
                payload,
            ),
        )
        connection.commit()
        return int(cursor.lastrowid)


def proposal_to_snapshot(proposal: ExperimentProposal) -> dict[str, Any]:
    return {
        "max_features": proposal.max_features,
        "selector_type": proposal.selector_type,
        "bucket_quotas": list(proposal.bucket_quotas),
        "exclude_patterns": list(proposal.exclude_patterns),
        "force_include_patterns": list(proposal.force_include_patterns),
        "forced_delta_count": proposal.forced_delta_count,
        "trials": proposal.trials,
        "folds": proposal.folds,
    }


def config_fingerprint(config_snapshot: dict[str, Any]) -> str:
    encoded = json.dumps(config_snapshot, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_program_text(program_path: str | Path = DEFAULT_PROGRAM_PATH) -> str:
    return Path(program_path).read_text(encoding="utf-8")


def _expected_trials_and_folds(exploration_mode: str) -> tuple[int, int]:
    normalized_mode = exploration_mode.strip().lower()
    if normalized_mode == "fast":
        return train_module.TRIALS, train_module.FOLDS
    raise ValueError(f"Unsupported exploration mode: {exploration_mode}")


def _proposal_for_exploration_mode(
    proposal: ExperimentProposal,
    exploration_mode: str,
) -> ExperimentProposal:
    trials, folds = _expected_trials_and_folds(exploration_mode)
    return ExperimentProposal(
        max_features=proposal.max_features,
        selector_type=proposal.selector_type,
        bucket_quotas=list(proposal.bucket_quotas),
        exclude_patterns=list(proposal.exclude_patterns),
        force_include_patterns=list(proposal.force_include_patterns),
        forced_delta_count=proposal.forced_delta_count,
        trials=trials,
        folds=folds,
        rationale=proposal.rationale,
    )


def _serialize_session_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "status": row["status"],
        "started_at": row["started_at"],
        "ended_at": row["ended_at"],
        "exploration_mode": row["exploration_mode"],
        "duration_hours": int(row["duration_hours"]),
        "until_interrupted": bool(row["until_interrupted"]),
        "run_full_at_end": bool(row["run_full_at_end"]),
        "stop_at": row["stop_at"],
    }


def build_session_context(
    session_id: int,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    session_row = load_session(session_id, db_path=db_path)
    session_history = load_history(
        db_path=db_path,
        mode=("fast", "full"),
        session_id=session_id,
    )
    successful_session_runs = _successful_history(session_history)
    best_row = None
    if successful_session_runs:
        best_row = max(
            successful_session_runs,
            key=lambda row: (
                float(row["holdout_r2"] if row["holdout_r2"] is not None else float("-inf")),
                -float(row["cv_rmse"] if row["cv_rmse"] is not None else float("inf")),
            ),
        )
    now = datetime.now(UTC)
    stop_at = None
    if session_row["stop_at"]:
        stop_at = datetime.fromisoformat(str(session_row["stop_at"]))
    remaining_minutes = None if stop_at is None else max(
        0.0,
        (stop_at - now).total_seconds() / 60.0,
    )
    session_notes = load_notes(db_path=db_path, session_id=session_id)
    artifact_reviews = []
    for row in successful_session_runs[-3:]:
        diagnostics = _extract_run_diagnostics(row)
        if diagnostics is None:
            continue
        artifact_reviews.append(
            {
                "experiment_id": int(row["id"]),
                "experiment_name": row["experiment_name"],
                "holdout_r2": row["holdout_r2"],
                "cv_rmse": row["cv_rmse"],
                "diagnostics": diagnostics,
            }
        )
    return {
        "session": _serialize_session_row(session_row),
        "progress": {
            "completed_runs": len(session_history),
            "successful_runs": len(successful_session_runs),
            "remaining_minutes": remaining_minutes,
        },
        "best_run": None
        if best_row is None
        else {
            "experiment_name": best_row["experiment_name"],
            "mode": best_row["mode"],
            "holdout_r2": best_row["holdout_r2"],
            "cv_rmse": best_row["cv_rmse"],
        },
        "recent_session_runs": [
            _serialize_history_row(row)
            for row in session_history[-6:]
        ],
        "recent_notes": [
            _serialize_note_row(row)
            for row in session_notes[-4:]
        ],
        "artifact_reviews": artifact_reviews,
    }


def _json_code_block_payload(text: str) -> str:
    fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if fenced_match:
        return fenced_match.group(1)

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Planner response did not contain a JSON object")
    return text[start : end + 1]


def _serialize_history_row(row: sqlite3.Row) -> dict[str, Any]:
    payload = {
        "id": int(row["id"]),
        "experiment_name": row["experiment_name"],
        "status": row["status"],
        "hypothesis": row["hypothesis"],
        "holdout_r2": row["holdout_r2"],
        "holdout_rmse": row["holdout_rmse"],
        "cv_rmse": row["cv_rmse"],
        "config": json.loads(row["config_json"]),
        "planner_type": row["planner_type"],
        "planner_model": row["planner_model"],
        "error_message": row["error_message"],
    }
    diagnostics = _extract_run_diagnostics(row)
    if diagnostics is not None:
        payload["diagnostics"] = diagnostics
    return payload


def _serialize_note_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "created_at": row["created_at"],
        "experiment_id": row["experiment_id"],
        "note_type": row["note_type"],
        "importance": row["importance"],
        "title": row["title"],
        "body": row["body"],
        "metadata": json.loads(row["metadata_json"] or "{}"),
    }


def _load_json_if_exists(path_value: str | None) -> dict[str, Any] | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    if not path.exists() or path.suffix.lower() != ".json":
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _extract_run_diagnostics(row: sqlite3.Row) -> dict[str, Any] | None:
    result = _load_json_if_exists(row["summary_path"])
    if result is None:
        output_dir = row["output_dir"]
        if output_dir:
            resolved_output_dir = Path(output_dir)
            if not resolved_output_dir.is_absolute():
                resolved_output_dir = (REPO_ROOT / resolved_output_dir).resolve()
            metadata_files = sorted(resolved_output_dir.glob("*.metadata.json"))
            for metadata_file in metadata_files:
                try:
                    result = json.loads(metadata_file.read_text(encoding="utf-8"))
                    break
                except (OSError, json.JSONDecodeError):
                    continue
    if result is None:
        return None

    holdout_metrics = dict(result.get("holdout_metrics") or {})
    feature_health = dict(result.get("feature_health_diagnostics") or {})
    importance = list(result.get("feature_importance_rankings") or [])[:10]
    selected_by_bucket = dict(result.get("selected_features_by_bucket") or {})
    omitted_by_bucket = dict(result.get("omitted_top_features_by_bucket") or {})
    return {
        "cv_metric_name": result.get("cv_metric_name"),
        "cv_best_score": result.get("cv_best_score"),
        "holdout_metrics": holdout_metrics,
        "selected_feature_counts": feature_health.get("selected_feature_counts"),
        "expected_delta_columns": feature_health.get("expected_delta_columns"),
        "selected_feature_fill_health": feature_health.get("selected_feature_fill_health"),
        "selected_feature_drift": feature_health.get("selected_feature_drift"),
        "selected_features_by_bucket": selected_by_bucket,
        "omitted_top_features_by_bucket": omitted_by_bucket,
        "top_feature_importance": importance,
        "feature_column_count": len(result.get("feature_columns") or []),
    }


def _list_output_artifact_references(row: sqlite3.Row) -> list[dict[str, str]]:
    output_dir = row["output_dir"]
    if not output_dir:
        return []
    resolved_output_dir = Path(output_dir)
    if not resolved_output_dir.is_absolute():
        resolved_output_dir = (REPO_ROOT / resolved_output_dir).resolve()
    if not resolved_output_dir.exists():
        return []
    references: list[dict[str, str]] = []
    for artifact_path in sorted(path for path in resolved_output_dir.iterdir() if path.is_file()):
        references.append(
            {
                "file_name": artifact_path.name,
                "path": str(artifact_path),
            }
        )
    return references


def _history_digest(history_rows: Sequence[sqlite3.Row]) -> dict[str, Any]:
    successful = _successful_history(history_rows)
    recent_rows = history_rows[-8:]
    top_rows = sorted(
        successful,
        key=lambda row: (
            float(row["holdout_r2"] if row["holdout_r2"] is not None else float("-inf")),
            -float(row["cv_rmse"] if row["cv_rmse"] is not None else float("inf")),
        ),
        reverse=True,
    )[:5]
    return {
        "recent_runs": [_serialize_history_row(row) for row in recent_rows],
        "top_runs": [_serialize_history_row(row) for row in top_rows],
        "tried_config_fingerprints": [
            row["config_fingerprint"]
            for row in history_rows
            if row["mode"] == "fast"
        ],
    }


def _planner_system_prompt() -> str:
    return (
        "You are an autonomous MLB run-count ML research planner inspired by Karpathy's "
        "autoresearch. Your job is to propose the single next fast experiment that is most "
        "likely to improve model accuracy overnight. Return only valid JSON."
    )


def _planner_user_prompt(
    *,
    program_text: str,
    history_rows: Sequence[sqlite3.Row],
    session_context: dict[str, Any] | None,
    exploration_mode: str,
) -> str:
    history_digest = _history_digest(history_rows)
    expected_trials, expected_folds = _expected_trials_and_folds(exploration_mode)
    return "\n".join(
        [
            "Research instructions:",
            program_text.strip(),
            "",
            "Current repo constraints:",
            "- Only edit the AGENT_CONFIG block in train.py.",
            f"- Exploration mode for this session is `{exploration_mode}`.",
            f"- This session must use trials={expected_trials} and folds={expected_folds}.",
            "- max_features must be one of [60, 80, 100, 120].",
            "- selector_type must be one of ['pearson', 'bucketed', 'ablation'].",
            "- bucket_quotas must contain 3 or 4 non-negative integers with total <= max_features.",
            "- exclude_patterns and force_include_patterns must be short lists of feature name patterns.",
            "- The trainer already strips most 60g/60s candidate windows before selection.",
            "- Avoid repeating any tried non-full configuration fingerprint.",
            "- Use artifact diagnostics when available to explain why runs were good or bad.",
            "- Inspect selected/omitted features, fill health, drift, and Poisson deviance before proposing the next test.",
            "",
            "Current session context:",
            json.dumps(session_context or {}, indent=2, sort_keys=True),
            "",
            "Experiment history digest:",
            json.dumps(history_digest, indent=2, sort_keys=True),
            "",
            "Return JSON with this exact schema:",
            "{",
            '  "hypothesis": "one concise sentence",',
            '  "reasoning": "brief explanation of why this experiment should improve accuracy",',
            '  "config": {',
            '    "max_features": 80,',
            '    "selector_type": "pearson",',
            '    "bucket_quotas": [24, 28, 12, 16],',
            '    "exclude_patterns": [],',
            '    "force_include_patterns": [],',
            '    "trials": 50,',
            '    "folds": 3',
            "  }",
            "}",
        ]
    )


def _coerce_pattern_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("Pattern lists must be arrays")
    patterns = [str(item).strip() for item in value if str(item).strip()]
    if len(patterns) > 8:
        raise ValueError("Pattern lists may contain at most 8 entries")
    return patterns


def _proposal_from_payload(payload: dict[str, Any]) -> ExperimentProposal:
    config = dict(payload["config"])
    max_features = int(config["max_features"])
    if max_features not in {60, 80, 100, 120}:
        raise ValueError("max_features must be one of 60, 80, 100, 120")

    selector_type = str(config["selector_type"]).strip().lower()
    if selector_type not in {"pearson", "bucketed", "ablation"}:
        raise ValueError("selector_type must be pearson, bucketed, or ablation")

    bucket_quotas = [int(item) for item in list(config["bucket_quotas"])]
    train_module.resolve_bucket_targets(max_features=max_features, bucket_quotas=bucket_quotas)

    exclude_patterns = _coerce_pattern_list(config.get("exclude_patterns"))
    force_include_patterns = _coerce_pattern_list(config.get("force_include_patterns"))
    trials = int(config.get("trials", train_module.TRIALS))
    folds = int(config.get("folds", train_module.FOLDS))

    rationale = str(payload.get("reasoning") or payload.get("hypothesis") or "").strip()
    if not rationale:
        raise ValueError("Planner payload must include reasoning or hypothesis")

    return ExperimentProposal(
        max_features=max_features,
        selector_type=selector_type,
        bucket_quotas=bucket_quotas,
        exclude_patterns=exclude_patterns,
        force_include_patterns=force_include_patterns,
        forced_delta_count=int(config.get("forced_delta_count", len(force_include_patterns))),
        trials=trials,
        folds=folds,
        rationale=rationale,
    )


def _plan_next_experiment_with_llm(
    history_rows: Sequence[sqlite3.Row],
    *,
    program_text: str,
    session_context: dict[str, Any] | None,
    exploration_mode: str,
) -> PlannerDecision:
    prompt_text = _planner_user_prompt(
        program_text=program_text,
        history_rows=history_rows,
        session_context=session_context,
        exploration_mode=exploration_mode,
    )
    response = llm_client.generate_text(
        system_prompt=_planner_system_prompt(),
        user_prompt=prompt_text,
        temperature=0.2,
        max_output_tokens=1400,
    )
    payload = json.loads(_json_code_block_payload(response.text))
    proposal = _proposal_for_exploration_mode(
        _proposal_from_payload(payload),
        exploration_mode,
    )
    expected_trials, expected_folds = _expected_trials_and_folds(exploration_mode)
    if proposal.trials != expected_trials or proposal.folds != expected_folds:
        raise ValueError(
            f"Planner must use exactly {expected_trials} trials and {expected_folds} folds "
            f"for exploration mode {exploration_mode}"
        )
    fingerprint = config_fingerprint(proposal_to_snapshot(proposal))
    tried_fingerprints = {
        row["config_fingerprint"]
        for row in history_rows
        if row["mode"] == "fast"
    }
    if fingerprint in tried_fingerprints:
        raise ValueError("Planner proposed a fast configuration that was already tried")

    hypothesis = str(payload["hypothesis"]).strip()
    if not hypothesis:
        raise ValueError("Planner must provide a non-empty hypothesis")

    return PlannerDecision(
        proposal=proposal,
        hypothesis=hypothesis,
        planner_type="llm",
        planner_model=f"{response.provider}:{response.model}",
        reasoning=str(payload.get("reasoning") or proposal.rationale),
        prompt_text=prompt_text,
        response_text=response.text,
    )


def build_proposal_catalog(program_text: str) -> list[ExperimentProposal]:
    _ = program_text
    proposals: list[ExperimentProposal] = []

    proposals.append(
        ExperimentProposal(
            max_features=80,
            selector_type="pearson",
            bucket_quotas=[80, 0, 0, 0],
            exclude_patterns=[],
            force_include_patterns=_FORCE_7G_PATTERNS,
            forced_delta_count=8,
            trials=120,
            folds=3,
            rationale="Baseline tonight's best known manual region: flat, 80 features, forced_delta_count=8.",
        )
    )
    for forced_delta_count in (4, 6, 8, 10, 12, 14, 16):
        force_patterns = _FORCE_7G_PATTERNS[: min(len(_FORCE_7G_PATTERNS), max(1, forced_delta_count // 4))]
        proposals.append(
            ExperimentProposal(
                max_features=80,
                selector_type="pearson",
                bucket_quotas=[80, 0, 0, 0],
                exclude_patterns=[],
                force_include_patterns=force_patterns,
                forced_delta_count=forced_delta_count,
                trials=120,
                folds=3,
                rationale=f"Map the forced-delta region at max_features=80 with forced_delta_count={forced_delta_count}.",
            )
        )
    for forced_delta_count in (8, 10, 12, 14):
        force_patterns = _FORCE_7G_PATTERNS[: min(len(_FORCE_7G_PATTERNS), max(1, forced_delta_count // 4))]
        for max_features in (72, 80, 88):
            proposals.append(
                ExperimentProposal(
                    max_features=max_features,
                    selector_type="pearson",
                    bucket_quotas=[max_features, 0, 0, 0],
                    exclude_patterns=[],
                    force_include_patterns=force_patterns,
                    forced_delta_count=forced_delta_count,
                    trials=120,
                    folds=3,
                    rationale=(
                        f"Local refinement around the forced-delta region with forced_delta_count={forced_delta_count} "
                        f"and max_features={max_features}."
                    ),
                )
            )

    deduped: list[ExperimentProposal] = []
    seen: set[str] = set()
    for proposal in proposals:
        fingerprint = config_fingerprint(proposal_to_snapshot(proposal))
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        deduped.append(proposal)
    return deduped


def update_train_config(
    *,
    train_path: str | Path,
    proposal: ExperimentProposal,
) -> None:
    resolved_train_path = Path(train_path)
    source = resolved_train_path.read_text(encoding="utf-8")
    replacement = "\n".join(
        [
            "# AGENT_CONFIG_START",
            f"MAX_FEATURES = {proposal.max_features}",
            f'SELECTOR_TYPE = "{proposal.selector_type}"',
            f"BUCKET_QUOTAS = {json.dumps(proposal.bucket_quotas)}",
            f"EXCLUDE_PATTERNS: list[str] = {json.dumps(proposal.exclude_patterns)}",
            f"FORCE_INCLUDE_PATTERNS: list[str] = {json.dumps(proposal.force_include_patterns)}",
            f"FORCED_DELTA_COUNT = {proposal.forced_delta_count}",
            f"TRIALS = {proposal.trials}",
            f"FOLDS = {proposal.folds}",
            "# AGENT_CONFIG_END",
        ]
    )
    updated = re.sub(
        r"# AGENT_CONFIG_START.*?# AGENT_CONFIG_END",
        replacement,
        source,
        count=1,
        flags=re.DOTALL,
    )
    if updated == source:
        raise ValueError("Could not locate AGENT_CONFIG block in train.py")
    resolved_train_path.write_text(updated, encoding="utf-8")


def _connect(db_path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(Path(db_path))
    connection.row_factory = sqlite3.Row
    return connection


def load_history(
    db_path: str | Path = DEFAULT_DB_PATH,
    *,
    mode: str | Sequence[str] = "fast",
    session_id: int | None = None,
) -> list[sqlite3.Row]:
    modes = [str(mode)] if isinstance(mode, str) else [str(item) for item in mode]
    placeholders = ", ".join("?" for _ in modes)
    parameters: list[Any] = list(modes)
    session_clause = ""
    if session_id is not None:
        session_clause = "AND session_id = ?"
        parameters.append(int(session_id))
    with _connect(db_path) as connection:
        rows = connection.execute(
            f"""
            SELECT *
            FROM experiments
            WHERE mode IN ({placeholders})
            {session_clause}
            ORDER BY started_at ASC, id ASC
            """,
            parameters,
        ).fetchall()
    return rows


def best_fast_experiment(
    db_path: str | Path = DEFAULT_DB_PATH,
    *,
    session_id: int | None = None,
) -> sqlite3.Row | None:
    parameters: list[Any] = []
    session_clause = ""
    if session_id is not None:
        session_clause = "AND session_id = ?"
        parameters.append(int(session_id))
    with _connect(db_path) as connection:
        row = connection.execute(
            f"""
            SELECT *
            FROM experiments
            WHERE mode = 'fast' AND status = 'succeeded'
            {session_clause}
            ORDER BY holdout_r2 DESC, cv_rmse ASC, started_at ASC
            LIMIT 1
            """,
            parameters,
        ).fetchone()
    return row


def _successful_history(rows: Sequence[sqlite3.Row]) -> list[sqlite3.Row]:
    return [row for row in rows if row["status"] == "succeeded"]


def _selector_types_seen(rows: Sequence[sqlite3.Row]) -> set[str]:
    seen: set[str] = set()
    for row in rows:
        config = json.loads(row["config_json"])
        selector_type = config.get("selector_type")
        if selector_type:
            seen.add(str(selector_type))
    return seen


def _config_distance(best_config: dict[str, Any], proposal_config: dict[str, Any]) -> int:
    keys = (
        "max_features",
        "selector_type",
        "bucket_quotas",
        "exclude_patterns",
        "force_include_patterns",
    )
    return sum(best_config.get(key) != proposal_config.get(key) for key in keys)


def plan_next_experiment_heuristic(
    history_rows: Sequence[sqlite3.Row],
    *,
    program_text: str,
    exploration_mode: str,
) -> PlannerDecision:
    proposals = build_proposal_catalog(program_text)
    tried_fingerprints = {
        row["config_fingerprint"]
        for row in history_rows
        if row["mode"] == "fast"
    }
    remaining = [
        _proposal_for_exploration_mode(proposal, exploration_mode)
        for proposal in proposals
        if config_fingerprint(
            proposal_to_snapshot(_proposal_for_exploration_mode(proposal, exploration_mode))
        )
        not in tried_fingerprints
    ]
    if not remaining:
        raise RuntimeError("No untried fast proposals remain in the catalog")

    successful = _successful_history(history_rows)
    if not successful:
        proposal = remaining[0]
        return PlannerDecision(
            proposal=proposal,
            hypothesis=proposal.rationale,
            planner_type="heuristic",
            planner_model=None,
            reasoning=proposal.rationale,
        )

    best_row = max(
        successful,
        key=lambda row: (
            float(row["holdout_r2"] if row["holdout_r2"] is not None else float("-inf")),
            -float(row["cv_rmse"] if row["cv_rmse"] is not None else float("inf")),
        ),
    )
    best_config = json.loads(best_row["config_json"])
    seen_selector_types = _selector_types_seen(successful)

    def _score(proposal: ExperimentProposal) -> float:
        proposal_config = proposal_to_snapshot(proposal)
        distance = _config_distance(best_config, proposal_config)
        score = 100.0 - (distance * 15.0)
        score -= abs(int(proposal_config["max_features"]) - int(best_config["max_features"])) / 5.0
        if proposal_config["selector_type"] == best_config["selector_type"]:
            score += 12.0
        if proposal_config["selector_type"] not in seen_selector_types:
            score += 8.0
        if proposal_config["force_include_patterns"] == best_config.get("force_include_patterns", []):
            score += 5.0
        if proposal_config["exclude_patterns"] == best_config.get("exclude_patterns", []):
            score += 5.0
        return score

    proposal = max(remaining, key=_score)
    best_r2 = float(best_row["holdout_r2"] or 0.0) * 100.0
    best_cv_rmse = float(best_row["cv_rmse"] or 0.0)
    hypothesis = (
        f"Best fast run so far is `{best_row['experiment_name']}` at {best_r2:.2f}% holdout R² "
        f"and {best_cv_rmse:.4f} cv_rmse. Next: {proposal.rationale}"
    )
    return PlannerDecision(
        proposal=proposal,
        hypothesis=hypothesis,
        planner_type="heuristic",
        planner_model=None,
        reasoning=proposal.rationale,
    )


def plan_next_experiment(
    history_rows: Sequence[sqlite3.Row],
    *,
    program_text: str,
    session_context: dict[str, Any] | None = None,
    exploration_mode: str = "fast",
) -> PlannerDecision:
    configured_llm = llm_client.load_llm_config()
    if configured_llm is not None:
        try:
            return _plan_next_experiment_with_llm(
                history_rows,
                program_text=program_text,
                session_context=session_context,
                exploration_mode=exploration_mode,
            )
        except Exception as exc:
            if llm_client.require_llm():
                raise RuntimeError(f"LLM planning failed and AUTORESEARCH_REQUIRE_LLM is enabled: {exc}") from exc
            heuristic_decision = plan_next_experiment_heuristic(
                history_rows,
                program_text=program_text,
                exploration_mode=exploration_mode,
            )
            return PlannerDecision(
                proposal=heuristic_decision.proposal,
                hypothesis=heuristic_decision.hypothesis,
                planner_type="heuristic_fallback",
                planner_model=(
                    f"{configured_llm.provider}:{configured_llm.model}"
                    if configured_llm is not None
                    else None
                ),
                reasoning=f"LLM planning failed, so the heuristic fallback was used. {exc}",
            )
    if llm_client.require_llm():
        raise RuntimeError(
            "AUTORESEARCH_REQUIRE_LLM is enabled but no LLM provider is configured. "
            "Set OPENAI_API_KEY or ANTHROPIC_API_KEY."
        )
    return plan_next_experiment_heuristic(
        history_rows,
        program_text=program_text,
        exploration_mode=exploration_mode,
    )


def _insert_started_experiment(
    *,
    db_path: str | Path,
    mode: str,
    planner_decision: PlannerDecision,
    proposal: ExperimentProposal,
    experiment_name: str,
    session_id: int | None = None,
    parent_experiment_id: int | None = None,
) -> int:
    config_snapshot = proposal_to_snapshot(proposal)
    fingerprint = config_fingerprint(config_snapshot)
    with _connect(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO experiments (
                started_at,
                mode,
                status,
                hypothesis,
                config_json,
                config_fingerprint,
                experiment_name,
                session_id,
                parent_experiment_id,
                planner_type,
                planner_model,
                planner_notes
            )
            VALUES (?, ?, 'running', ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(UTC).isoformat(),
                mode,
                planner_decision.hypothesis,
                json.dumps(config_snapshot, sort_keys=True),
                fingerprint,
                experiment_name,
                session_id,
                parent_experiment_id,
                planner_decision.planner_type,
                planner_decision.planner_model,
                planner_decision.reasoning,
            ),
        )
        connection.commit()
        return int(cursor.lastrowid)


def _finalize_experiment(
    *,
    db_path: str | Path,
    experiment_id: int,
    planner_decision: PlannerDecision,
    status: str,
    result_payload: dict[str, Any] | None,
    stdout_path: Path,
    stderr_path: Path,
    returncode: int,
    planner_prompt_path: Path | None,
    planner_response_path: Path | None,
    error_message: str | None = None,
) -> None:
    completed_at = datetime.now(UTC).isoformat()
    duration_seconds = None
    config_json = None
    metrics_json = None
    output_dir = None
    summary_path = None
    holdout_r2 = None
    holdout_rmse = None
    cv_rmse = None
    cv_metric_name = None
    cv_metric_value = None

    if result_payload is not None:
        config_json = json.dumps(result_payload["config"], sort_keys=True)
        metrics_json = json.dumps(result_payload["metrics"], sort_keys=True)
        duration_seconds = float(result_payload["duration_seconds"])
        output_dir = result_payload["output_dir"]
        summary_path = result_payload["summary_path"]
        holdout_r2 = float(result_payload["metrics"]["holdout_r2"])
        holdout_rmse = float(result_payload["metrics"]["holdout_rmse"])
        cv_rmse = float(result_payload["metrics"]["cv_rmse"])
        cv_metric_name = str(result_payload["metrics"]["cv_metric_name"])
        cv_metric_value = float(result_payload["metrics"]["cv_metric_value"])

    with _connect(db_path) as connection:
        connection.execute(
            """
            UPDATE experiments
            SET completed_at = ?,
                status = ?,
                config_json = COALESCE(?, config_json),
                metrics_json = ?,
                duration_seconds = ?,
                output_dir = ?,
                summary_path = ?,
                stdout_path = ?,
                stderr_path = ?,
                holdout_r2 = ?,
                holdout_rmse = ?,
                cv_rmse = ?,
                cv_metric_name = ?,
                cv_metric_value = ?,
                returncode = ?,
                error_message = ?,
                planner_type = ?,
                planner_model = ?,
                planner_notes = ?,
                planner_prompt_path = ?,
                planner_response_path = ?
            WHERE id = ?
            """,
            (
                completed_at,
                status,
                config_json,
                metrics_json,
                duration_seconds,
                output_dir,
                summary_path,
                str(stdout_path),
                str(stderr_path),
                holdout_r2,
                holdout_rmse,
                cv_rmse,
                cv_metric_name,
                cv_metric_value,
                returncode,
                error_message,
                planner_decision.planner_type,
                planner_decision.planner_model,
                planner_decision.reasoning,
                None if planner_prompt_path is None else str(planner_prompt_path),
                None if planner_response_path is None else str(planner_response_path),
                experiment_id,
            ),
        )
        connection.commit()


def _experiment_beats_baseline(
    *,
    holdout_r2: float,
    cv_rmse: float,
    baseline_row: sqlite3.Row | None,
) -> bool:
    if baseline_row is None:
        return True
    baseline_r2 = float(baseline_row["holdout_r2"] or float("-inf"))
    baseline_cv_rmse = float(baseline_row["cv_rmse"] or float("inf"))
    if holdout_r2 > baseline_r2:
        return True
    if holdout_r2 == baseline_r2 and cv_rmse < baseline_cv_rmse:
        return True
    return False


def maybe_record_experiment_notes(
    *,
    db_path: str | Path,
    experiment_id: int,
    session_id: int | None,
    proposal: ExperimentProposal,
    exploration_mode: str,
    prior_session_best: dict[str, Any] | None,
    result_payload: dict[str, Any] | None,
    status: str,
    error_message: str | None,
    reports_dir: str | Path = DEFAULT_REPORTS_DIR,
) -> list[int]:
    note_ids: list[int] = []
    if status != "succeeded" or result_payload is None:
        failure_metadata = {
            "exploration_mode": exploration_mode,
            "selector_type": proposal.selector_type,
            "max_features": proposal.max_features,
            "config": proposal_to_snapshot(proposal),
            "suspicious": True,
        }
        note_id = record_note(
            db_path=db_path,
            session_id=session_id,
            experiment_id=experiment_id,
            note_type="failure",
            importance="high",
            title="Experiment failed",
            body=(
                f"{exploration_mode} experiment failed. "
                f"Error: {(error_message or 'Unknown error')[:500]}"
            ),
            metadata=failure_metadata,
            reports_dir=reports_dir,
        )
        note_ids.append(note_id)
        append_nightly_log(
            event_type="suspected_issue",
            heading="Suspicious experiment failure",
            body_lines=[
                f"session_id: `{session_id}`",
                f"experiment_id: `{experiment_id}`",
                f"note_id: `{note_id}`",
                f"reason: `{error_message or 'Unknown error'}`",
            ],
            reports_dir=reports_dir,
        )
        return note_ids

    metrics = dict(result_payload["metrics"])
    holdout_r2 = float(metrics["holdout_r2"])
    cv_rmse = float(metrics["cv_rmse"])
    experiment_row = load_experiment(experiment_id, db_path=db_path)
    diagnostics = _extract_run_diagnostics(experiment_row)
    diagnostic_good: list[str] = []
    diagnostic_bad: list[str] = []
    diagnostic_suspicious: list[str] = []
    previous_best_r2 = None if prior_session_best is None else prior_session_best.get("holdout_r2")
    previous_best_cv_rmse = None if prior_session_best is None else prior_session_best.get("cv_rmse")
    improvement_text = None
    if previous_best_r2 is not None:
        improvement_text = (
            f"Previous best holdout R² was {float(previous_best_r2) * 100:.2f}% "
            f"with cv_rmse {float(previous_best_cv_rmse):.4f}."
        )

    if previous_best_r2 is None or (
        holdout_r2 > float(previous_best_r2)
        or (
            holdout_r2 == float(previous_best_r2)
            and cv_rmse < float(previous_best_cv_rmse)
        )
    ):
        note_ids.append(
            record_note(
                db_path=db_path,
                session_id=session_id,
                experiment_id=experiment_id,
                note_type="new_best",
                importance="high",
                title="New session best configuration",
                body=(
                    f"{exploration_mode} run `{result_payload['experiment_name']}` became the new session leader "
                    f"at {holdout_r2 * 100:.2f}% holdout R² and {cv_rmse:.4f} cv_rmse. "
                    + (improvement_text or "This is the first successful run in the session.")
                ),
                metadata={
                    "exploration_mode": exploration_mode,
                    "selector_type": proposal.selector_type,
                    "max_features": proposal.max_features,
                    "exclude_patterns": proposal.exclude_patterns,
                    "force_include_patterns": proposal.force_include_patterns,
                },
                reports_dir=reports_dir,
            )
        )
        diagnostic_good.append("became the new session best on holdout_r2/cv_rmse")

    if proposal.exclude_patterns:
        note_ids.append(
            record_note(
                db_path=db_path,
                session_id=session_id,
                experiment_id=experiment_id,
                note_type="ablation",
                importance="medium",
                title="Ablation was tested",
                body=(
                    f"Excluded patterns {proposal.exclude_patterns} under `{proposal.selector_type}` "
                    f"with {proposal.max_features} features. Holdout R² was {holdout_r2 * 100:.2f}%."
                ),
                metadata={"exclude_patterns": proposal.exclude_patterns},
                reports_dir=reports_dir,
            )
        )
        diagnostic_bad.append(f"ablation excluded {proposal.exclude_patterns}")

    if proposal.force_include_patterns:
        note_ids.append(
            record_note(
                db_path=db_path,
                session_id=session_id,
                experiment_id=experiment_id,
                note_type="feature_bias",
                importance="medium",
                title="Forced feature family test",
                body=(
                    f"Forced include patterns {proposal.force_include_patterns}. "
                    f"Use this result to judge whether short-window features deserve more quota."
                ),
                metadata={"force_include_patterns": proposal.force_include_patterns},
                reports_dir=reports_dir,
            )
        )
        diagnostic_good.append(f"tested forced feature family {proposal.force_include_patterns}")

    if holdout_r2 < 0:
        warning_metadata = {
            "holdout_r2": holdout_r2,
            "cv_rmse": cv_rmse,
            "config": proposal_to_snapshot(proposal),
            "suspicious": True,
        }
        note_id = record_note(
            db_path=db_path,
            session_id=session_id,
            experiment_id=experiment_id,
            note_type="warning",
            importance="medium",
            title="Negative holdout R²",
            body=(
                f"Run `{result_payload['experiment_name']}` produced negative holdout R² "
                f"({holdout_r2 * 100:.2f}%). This config likely moved away from the useful region."
            ),
            metadata=warning_metadata,
            reports_dir=reports_dir,
        )
        note_ids.append(note_id)
        append_nightly_log(
            event_type="suspected_issue",
            heading="Suspicious metric regression",
            body_lines=[
                f"session_id: `{session_id}`",
                f"experiment_id: `{experiment_id}`",
                f"note_id: `{note_id}`",
                f"holdout_r2: `{holdout_r2:.6f}`",
                f"cv_rmse: `{cv_rmse:.6f}`",
            ],
            reports_dir=reports_dir,
        )
        diagnostic_suspicious.append("negative holdout_r2 pushed the run into a likely bad region")
    if diagnostics is not None:
        expected_delta = dict(diagnostics.get("expected_delta_columns") or {})
        fill_health = dict(diagnostics.get("selected_feature_fill_health") or {})
        holdout_fill = dict(fill_health.get("holdout") or {})
        top_default_fill = list(holdout_fill.get("top_default_fill_share") or [])
        holdout_metrics = dict(diagnostics.get("holdout_metrics") or {})
        selected_counts = dict(diagnostics.get("selected_feature_counts") or {})
        family_counts = dict(selected_counts.get("family") or {})
        omitted_deltas = list((diagnostics.get("omitted_top_features_by_bucket") or {}).get("delta") or [])
        top_importance = list(diagnostics.get("top_feature_importance") or [])
        if holdout_metrics.get("poisson_deviance") is not None:
            diagnostic_good.append(
                f"poisson_deviance={float(holdout_metrics['poisson_deviance']):.4f}"
            )
        if family_counts:
            top_family = max(family_counts.items(), key=lambda item: item[1])
            diagnostic_bad.append(f"selected-feature family skew led by {top_family[0]}={top_family[1]}")
        if omitted_deltas:
            diagnostic_bad.append(
                f"top omitted delta features included {[item.get('feature') for item in omitted_deltas[:3]]}"
            )
        if top_importance:
            diagnostic_good.append(
                f"top importance features were {[item.get('feature') for item in top_importance[:3]]}"
            )
        if not expected_delta.get("all_present", True):
            note_ids.append(
                record_note(
                    db_path=db_path,
                    session_id=session_id,
                    experiment_id=experiment_id,
                    note_type="suspicious_issue",
                    importance="high",
                    title="Expected delta columns missing",
                    body=(
                        f"Run `{result_payload['experiment_name']}` is missing expected delta columns: "
                        f"{expected_delta.get('missing_columns', [])}."
                    ),
                    metadata={"suspicious": True, "expected_delta_columns": expected_delta},
                    reports_dir=reports_dir,
                )
            )
            diagnostic_suspicious.append(
                f"missing expected delta columns {expected_delta.get('missing_columns', [])}"
            )
        if top_default_fill and float(top_default_fill[0].get("default_fill_share", 0.0)) >= 0.8:
            note_ids.append(
                record_note(
                    db_path=db_path,
                    session_id=session_id,
                    experiment_id=experiment_id,
                    note_type="suspicious_issue",
                    importance="medium",
                    title="Default-heavy selected features",
                    body=(
                        f"Run `{result_payload['experiment_name']}` selected features with very high "
                        "default-fill share on holdout. Review data quality before trusting the result."
                    ),
                    metadata={"suspicious": True, "top_default_fill_share": top_default_fill[:5]},
                    reports_dir=reports_dir,
                )
            )
            diagnostic_suspicious.append("selected features were extremely default-heavy on holdout")
    if not diagnostic_good:
        diagnostic_good.append("no clear positive artifact signal beyond the top-line metrics")
    if not diagnostic_bad:
        diagnostic_bad.append("no obvious negative artifact pattern was detected")
    if not diagnostic_suspicious:
        diagnostic_suspicious.append("no reproducible suspicious artifact signal was detected")
    summary_note_id = record_note(
        db_path=db_path,
        session_id=session_id,
        experiment_id=experiment_id,
        note_type="diagnostic_summary",
        importance="medium",
        title="Run diagnostic summary",
        body=(
            f"Good because: {'; '.join(diagnostic_good)}. "
            f"Bad because: {'; '.join(diagnostic_bad)}. "
            f"Suspicious because: {'; '.join(diagnostic_suspicious)}."
        ),
        metadata={
            "good": diagnostic_good,
            "bad": diagnostic_bad,
            "suspicious": diagnostic_suspicious,
        },
        reports_dir=reports_dir,
    )
    note_ids.append(summary_note_id)
    append_nightly_log(
        event_type="diagnostic_summary",
        heading="Run diagnostic summary",
        body_lines=[
            f"session_id: `{session_id}`",
            f"experiment_id: `{experiment_id}`",
            f"note_id: `{summary_note_id}`",
            f"good: `{' | '.join(diagnostic_good)}`",
            f"bad: `{' | '.join(diagnostic_bad)}`",
            f"suspicious: `{' | '.join(diagnostic_suspicious)}`",
        ],
        reports_dir=reports_dir,
    )
    return note_ids


def build_session_summary(
    session_id: int,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    session_row = load_session(session_id, db_path=db_path)
    experiments = load_history(
        db_path=db_path,
        mode=("fast", "full"),
        session_id=session_id,
    )
    notes = load_notes(db_path=db_path, session_id=session_id)
    status_counts = Counter(str(row["status"]) for row in experiments)
    mode_counts = Counter(str(row["mode"]) for row in experiments)
    successful = _successful_history(experiments)
    best_exploration = best_fast_experiment(db_path=db_path, session_id=session_id)
    best_overall = None
    if successful:
        best_overall = max(
            successful,
            key=lambda row: (
                float(row["holdout_r2"] if row["holdout_r2"] is not None else float("-inf")),
                -float(row["cv_rmse"] if row["cv_rmse"] is not None else float("inf")),
            ),
        )
    previous_sessions = []
    with _connect(db_path) as connection:
        previous_sessions = connection.execute(
            """
            SELECT *
            FROM sessions
            WHERE id <> ?
            ORDER BY started_at ASC, id ASC
            """,
            (int(session_id),),
        ).fetchall()
    previous_best_r2 = None
    for row in previous_sessions:
        candidate = best_fast_experiment(db_path=db_path, session_id=int(row["id"]))
        if candidate is None:
            continue
        candidate_r2 = float(candidate["holdout_r2"] or float("-inf"))
        if previous_best_r2 is None or candidate_r2 > previous_best_r2:
            previous_best_r2 = candidate_r2

    delta_vs_prior_best = None
    if best_exploration is not None and previous_best_r2 is not None:
        delta_vs_prior_best = float(best_exploration["holdout_r2"]) - previous_best_r2

    leaderboard = [
        {
            "experiment_name": row["experiment_name"],
            "mode": row["mode"],
            "holdout_r2": row["holdout_r2"],
            "cv_rmse": row["cv_rmse"],
            "status": row["status"],
            "artifact_references": _list_output_artifact_references(row),
        }
        for row in sorted(
            successful,
            key=lambda row: (
                float(row["holdout_r2"] if row["holdout_r2"] is not None else float("-inf")),
                -float(row["cv_rmse"] if row["cv_rmse"] is not None else float("inf")),
            ),
            reverse=True,
        )[:5]
    ]
    top_notes = [_serialize_note_row(row) for row in notes[-10:]]
    recommendations: list[str] = []
    if best_exploration is not None:
        best_config = json.loads(best_exploration["config_json"])
        recommendations.append(
            f"Start the next session near selector `{best_config['selector_type']}` "
            f"with {best_config['max_features']} features."
        )
        if best_config.get("exclude_patterns"):
            recommendations.append(
                f"Retest the ablation {best_config['exclude_patterns']} around neighboring feature counts."
            )
        if best_config.get("force_include_patterns"):
            recommendations.append(
                f"Investigate whether forced patterns {best_config['force_include_patterns']} deserve permanent quota."
            )
    failure_count = status_counts.get("failed", 0)
    if failure_count:
        recommendations.append(
            f"Review {failure_count} failed run(s) in the logs before reusing the same search region."
        )
    summary = {
        "session": _serialize_session_row(session_row),
        "counts": {
            "experiments": len(experiments),
            "by_status": dict(status_counts),
            "by_mode": dict(mode_counts),
        },
        "best_exploration": None
        if best_exploration is None
        else {
            "experiment_name": best_exploration["experiment_name"],
            "mode": best_exploration["mode"],
            "holdout_r2": best_exploration["holdout_r2"],
            "cv_rmse": best_exploration["cv_rmse"],
            "config": json.loads(best_exploration["config_json"]),
        },
        "best_overall": None
        if best_overall is None
        else {
            "experiment_name": best_overall["experiment_name"],
            "mode": best_overall["mode"],
            "holdout_r2": best_overall["holdout_r2"],
            "cv_rmse": best_overall["cv_rmse"],
        },
        "delta_vs_prior_best_exploration_r2": delta_vs_prior_best,
        "leaderboard": leaderboard,
        "notes": top_notes,
        "artifact_references": [
            {
                "experiment_id": int(row["id"]),
                "experiment_name": row["experiment_name"],
                "output_dir": row["output_dir"],
                "summary_path": row["summary_path"],
                "files": _list_output_artifact_references(row),
            }
            for row in experiments
            if row["output_dir"]
        ],
        "recommendations": recommendations,
    }
    return summary


def _session_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"# AutoResearch Session {summary['session']['id']}",
        "",
        f"- status: `{summary['session']['status']}`",
        f"- started_at: `{summary['session']['started_at']}`",
        f"- ended_at: `{summary['session'].get('ended_at')}`",
        f"- exploration_mode: `{summary['session']['exploration_mode']}`",
        f"- experiments: `{summary['counts']['experiments']}`",
        "",
        "## Best exploration run",
    ]
    best = summary["best_exploration"]
    if best is None:
        lines.append("- none")
    else:
        lines.extend(
            [
                f"- experiment: `{best['experiment_name']}`",
                f"- holdout_r2: `{float(best['holdout_r2']) * 100:.2f}%`",
                f"- cv_rmse: `{float(best['cv_rmse']):.4f}`",
                f"- selector: `{best['config']['selector_type']}`",
                f"- max_features: `{best['config']['max_features']}`",
            ]
        )
    lines.extend(["", "## Notes"])
    if not summary["notes"]:
        lines.append("- none")
    else:
        for note in summary["notes"]:
            lines.append(f"- [{note['importance']}] {note['title']}: {note['body']}")
    lines.extend(["", "## Recommendations"])
    if not summary["recommendations"]:
        lines.append("- none")
    else:
        for item in summary["recommendations"]:
            lines.append(f"- {item}")
    lines.extend(["", "## AutoResearch-created artifact references"])
    if not summary["artifact_references"]:
        lines.append("- none")
    else:
        for artifact_group in summary["artifact_references"]:
            lines.append(
                f"- `{artifact_group['experiment_name']}` -> `{artifact_group['output_dir']}`"
            )
            for artifact in artifact_group["files"]:
                lines.append(f"  - `{artifact['file_name']}`")
    return "\n".join(lines) + "\n"


def _morning_review_prompt(summary: dict[str, Any]) -> str:
    best = summary["best_exploration"]
    best_line = (
        "No successful exploration run was recorded."
        if best is None
        else (
            f"Best exploration run was `{best['experiment_name']}` with holdout_r2="
            f"{float(best['holdout_r2']) * 100:.2f}% and cv_rmse={float(best['cv_rmse']):.4f}."
        )
    )
    suspicious_notes = [
        note for note in summary["notes"] if note["note_type"] in {"failure", "warning", "suspicious_issue"}
    ]
    artifact_lines = []
    for artifact_group in summary["artifact_references"]:
        names = ", ".join(file["file_name"] for file in artifact_group["files"]) or "no files found"
        artifact_lines.append(
            f"- {artifact_group['experiment_name']}: output_dir={artifact_group['output_dir']} files=[{names}]"
        )
    return "\n".join(
        [
            "Review this AutoResearch overnight session and tell me what mattered.",
            "",
            "Session summary:",
            f"- session_id: {summary['session']['id']}",
            f"- status: {summary['session']['status']}",
            f"- experiments: {summary['counts']['experiments']}",
            f"- {best_line}",
            "",
            "Suspicious issues to review:",
            *(
                [f"- {note['title']}: {note['body']}" for note in suspicious_notes]
                if suspicious_notes
                else ["- none"]
            ),
            "",
            "Artifacts created/referenced by AutoResearch:",
            *(artifact_lines if artifact_lines else ["- none"]),
            "",
            "Please answer:",
            "1. What made the best run good?",
            "2. What made the worst or suspicious runs fail or look unreliable?",
            "3. Which variables or feature-selection pressures should be tried next?",
            "4. Are there any signals that point to a real code/data issue versus mere hyperparameter noise?",
            "5. What should be promoted to the main project, and what should stay isolated pending more review?",
        ]
    ) + "\n"


def write_session_summary(
    session_id: int,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    reports_dir: str | Path = DEFAULT_REPORTS_DIR,
    status_override: str | None = None,
) -> tuple[dict[str, Any], Path, Path, Path]:
    summary = build_session_summary(session_id, db_path=db_path)
    if status_override is not None:
        summary["session"]["status"] = status_override
        if summary["session"].get("ended_at") is None:
            summary["session"]["ended_at"] = datetime.now(UTC).isoformat()
    resolved_reports_dir = Path(reports_dir)
    session_reports_dir = resolved_reports_dir / "sessions"
    session_reports_dir.mkdir(parents=True, exist_ok=True)
    started_tag = str(summary["session"]["started_at"]).replace(":", "").replace("+00:00", "Z")
    file_stem = f"session_{session_id}_{started_tag}"
    json_path = session_reports_dir / f"{file_stem}.json"
    md_path = session_reports_dir / f"{file_stem}.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_session_summary_markdown(summary), encoding="utf-8")
    review_prompt_path = session_reports_dir / f"{file_stem}_morning_review_prompt.md"
    review_prompt_path.write_text(_morning_review_prompt(summary), encoding="utf-8")

    history_log_path = resolved_reports_dir / "session_history.jsonl"
    with history_log_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "session_id": session_id,
                    "started_at": summary["session"]["started_at"],
                    "ended_at": summary["session"]["ended_at"],
                    "status": summary["session"]["status"],
                    "exploration_mode": summary["session"]["exploration_mode"],
                    "experiment_count": summary["counts"]["experiments"],
                    "best_experiment_name": None
                    if summary["best_exploration"] is None
                    else summary["best_exploration"]["experiment_name"],
                    "best_holdout_r2": None
                    if summary["best_exploration"] is None
                    else summary["best_exploration"]["holdout_r2"],
                    "best_cv_rmse": None
                    if summary["best_exploration"] is None
                    else summary["best_exploration"]["cv_rmse"],
                },
                sort_keys=True,
            )
            + "\n"
        )
    return summary, json_path, md_path, review_prompt_path


def _reload_train_module():
    return importlib.reload(train_module)


def _run_train_process(
    *,
    mode: str,
    experiment_name: str,
    train_path: str | Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(Path(train_path)),
            "--mode",
            mode,
            "--experiment-name",
            experiment_name,
            "--json-output",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _write_process_logs(
    *,
    experiment_name: str,
    stdout: str,
    stderr: str,
    log_dir: str | Path = DEFAULT_LOG_DIR,
) -> tuple[Path, Path]:
    resolved_log_dir = Path(log_dir)
    resolved_log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = resolved_log_dir / f"{experiment_name}.stdout.log"
    stderr_path = resolved_log_dir / f"{experiment_name}.stderr.log"
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    return stdout_path, stderr_path


def _write_planner_logs(
    *,
    experiment_name: str,
    planner_decision: PlannerDecision,
    log_dir: str | Path = DEFAULT_LOG_DIR,
) -> tuple[Path | None, Path | None]:
    if planner_decision.prompt_text is None and planner_decision.response_text is None:
        return None, None
    resolved_log_dir = Path(log_dir)
    resolved_log_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = resolved_log_dir / f"{experiment_name}.planner_prompt.md"
    response_path = resolved_log_dir / f"{experiment_name}.planner_response.txt"
    prompt_path.write_text(planner_decision.prompt_text or "", encoding="utf-8")
    response_path.write_text(planner_decision.response_text or "", encoding="utf-8")
    return prompt_path, response_path


def _run_validation_retest(
    *,
    issue: SuspectedIssue,
    db_path: str | Path,
    train_path: str | Path,
    session_id: int,
    reports_dir: str | Path = DEFAULT_REPORTS_DIR,
) -> dict[str, Any]:
    config = dict(issue.metadata.get("config") or {})
    if not config:
        record_issue_validation(
            db_path=db_path,
            session_id=session_id,
            issue_note_id=issue.note_id,
            source_experiment_id=issue.experiment_id,
            validation_experiment_id=None,
            status="skipped",
            outcome="missing_config",
            details={"reason": "Issue note did not include a replayable config snapshot."},
        )
        append_nightly_log(
            event_type="issue_validation_skipped",
            heading="Issue validation skipped",
            body_lines=[
                f"session_id: `{session_id}`",
                f"issue_note_id: `{issue.note_id}`",
                "reason: `missing replayable config snapshot`",
            ],
            reports_dir=reports_dir,
        )
        return {"status": "skipped", "outcome": "missing_config", "issue_note_id": issue.note_id}

    proposal = ExperimentProposal(
        max_features=int(config["max_features"]),
        selector_type=str(config["selector_type"]),
        bucket_quotas=[int(value) for value in config["bucket_quotas"]],
        exclude_patterns=[str(value) for value in config.get("exclude_patterns", [])],
        force_include_patterns=[str(value) for value in config.get("force_include_patterns", [])],
        trials=int(config["trials"]),
        folds=int(config["folds"]),
        rationale=f"Validation retest for suspicious issue note {issue.note_id}",
    )
    update_train_config(train_path=train_path, proposal=proposal)
    train_api = _reload_train_module()
    effective_config = train_api.resolve_effective_config("fast")
    experiment_name = train_api.build_experiment_name("fast", effective_config) + f"-retest-{issue.note_id}"
    planner_decision = PlannerDecision(
        proposal=proposal,
        hypothesis=f"Retest suspicious issue note {issue.note_id}: {issue.title}",
        planner_type="issue_validation",
        planner_model=None,
        reasoning=issue.body,
    )
    experiment_id = _insert_started_experiment(
        db_path=db_path,
        mode="fast",
        planner_decision=planner_decision,
        proposal=proposal,
        experiment_name=experiment_name,
        session_id=session_id,
        parent_experiment_id=issue.experiment_id,
    )
    completed = _run_train_process(
        mode="fast",
        experiment_name=experiment_name,
        train_path=train_path,
    )
    stdout_path, stderr_path = _write_process_logs(
        experiment_name=experiment_name,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
    result_payload = None
    status = "failed"
    error_message = None
    if completed.returncode == 0:
        result_payload = _parse_result_payload(completed.stdout)
        merged_config = dict(result_payload["config"])
        merged_config["bucket_quotas"] = list(proposal.bucket_quotas)
        result_payload["config"] = merged_config
        status = "succeeded"
    else:
        error_message = completed.stderr.strip() or completed.stdout.strip() or "train.py failed"
    _finalize_experiment(
        db_path=db_path,
        experiment_id=experiment_id,
        planner_decision=planner_decision,
        status=status,
        result_payload=result_payload,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        returncode=completed.returncode,
        planner_prompt_path=None,
        planner_response_path=None,
        error_message=error_message,
    )
    outcome = "reproduced" if status != "succeeded" else "not_reproduced"
    validation_id = record_issue_validation(
        db_path=db_path,
        session_id=session_id,
        issue_note_id=issue.note_id,
        source_experiment_id=issue.experiment_id,
        validation_experiment_id=experiment_id,
        status=status,
        outcome=outcome,
        details={
            "experiment_name": experiment_name,
            "error_message": error_message,
            "result": result_payload,
        },
    )
    append_nightly_log(
        event_type="issue_validation",
        heading="Issue validation rerun completed",
        body_lines=[
            f"session_id: `{session_id}`",
            f"issue_note_id: `{issue.note_id}`",
            f"validation_id: `{validation_id}`",
            f"validation_experiment_id: `{experiment_id}`",
            f"status: `{status}`",
            f"outcome: `{outcome}`",
        ],
        reports_dir=reports_dir,
    )
    return {
        "status": status,
        "outcome": outcome,
        "issue_note_id": issue.note_id,
        "validation_id": validation_id,
        "validation_experiment_id": experiment_id,
        "result": result_payload,
        "error_message": error_message,
    }


def _parse_result_payload(stdout: str) -> dict[str, Any]:
    return json.loads(stdout)


def run_planner_self_check() -> dict[str, Any]:
    result = llm_client.planner_self_check()
    return {
        "provider": result.provider,
        "model": result.model,
        "command": result.command,
        "response_text": result.response_text,
        "session_id": result.session_id,
    }


def run_fast_once(
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    program_path: str | Path = DEFAULT_PROGRAM_PATH,
    train_path: str | Path = DEFAULT_TRAIN_PATH,
    exploration_mode: str = "fast",
    session_id: int | None = None,
) -> dict[str, Any]:
    ensure_experiment_db(db_path)
    program_text = load_program_text(program_path)
    history = load_history(db_path, mode="fast")
    session_context = None if session_id is None else build_session_context(session_id, db_path=db_path)
    prior_session_best = None if session_context is None else session_context.get("best_run")
    planner_decision = plan_next_experiment(
        history,
        program_text=program_text,
        session_context=session_context,
        exploration_mode=exploration_mode,
    )
    proposal = planner_decision.proposal
    update_train_config(train_path=train_path, proposal=proposal)
    train_api = _reload_train_module()
    effective_config = train_api.resolve_effective_config(exploration_mode)
    experiment_name = train_api.build_experiment_name(exploration_mode, effective_config)
    planner_prompt_path, planner_response_path = _write_planner_logs(
        experiment_name=experiment_name,
        planner_decision=planner_decision,
    )

    experiment_id = _insert_started_experiment(
        db_path=db_path,
        mode=exploration_mode,
        planner_decision=planner_decision,
        proposal=proposal,
        experiment_name=experiment_name,
        session_id=session_id,
    )
    completed = _run_train_process(
        mode=exploration_mode,
        experiment_name=experiment_name,
        train_path=train_path,
    )
    stdout_path, stderr_path = _write_process_logs(
        experiment_name=experiment_name,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )

    result_payload = None
    status = "failed"
    error_message = None
    if completed.returncode == 0:
        result_payload = _parse_result_payload(completed.stdout)
        merged_config = dict(result_payload["config"])
        merged_config["bucket_quotas"] = list(proposal.bucket_quotas)
        result_payload["config"] = merged_config
        status = "succeeded"
    else:
        error_message = completed.stderr.strip() or completed.stdout.strip() or "train.py failed"

    _finalize_experiment(
        db_path=db_path,
        experiment_id=experiment_id,
        planner_decision=planner_decision,
        status=status,
        result_payload=result_payload,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        returncode=completed.returncode,
        planner_prompt_path=planner_prompt_path,
        planner_response_path=planner_response_path,
        error_message=error_message,
    )
    note_ids = maybe_record_experiment_notes(
        db_path=db_path,
        experiment_id=experiment_id,
        session_id=session_id,
        proposal=proposal,
        exploration_mode=exploration_mode,
        prior_session_best=prior_session_best,
        result_payload=result_payload,
        status=status,
        error_message=error_message,
    )
    return {
        "experiment_id": experiment_id,
        "session_id": session_id,
        "experiment_name": experiment_name,
        "status": status,
        "hypothesis": planner_decision.hypothesis,
        "planner_type": planner_decision.planner_type,
        "planner_model": planner_decision.planner_model,
        "planner_reasoning": planner_decision.reasoning,
        "exploration_mode": exploration_mode,
        "proposal": proposal_to_snapshot(proposal),
        "result": result_payload,
        "note_ids": note_ids,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "planner_prompt_path": None if planner_prompt_path is None else str(planner_prompt_path),
        "planner_response_path": None if planner_response_path is None else str(planner_response_path),
    }


def run_best_full(
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    program_path: str | Path = DEFAULT_PROGRAM_PATH,
    train_path: str | Path = DEFAULT_TRAIN_PATH,
    session_id: int | None = None,
) -> dict[str, Any]:
    ensure_experiment_db(db_path)
    best_row = best_fast_experiment(db_path, session_id=session_id)
    parent_experiment_id = None
    if best_row is None:
        proposal = _proposal_for_exploration_mode(
            build_proposal_catalog(load_program_text(program_path))[0],
            "fast",
        )
        planner_decision = PlannerDecision(
            proposal=proposal,
            hypothesis=(
                "No successful fast experiment was available at cutoff, so run the baseline "
                "configuration in full mode."
            ),
            planner_type="promotion_baseline",
            planner_model=None,
            reasoning="No successful fast winner existed, so the baseline config was promoted.",
        )
    else:
        best_config = json.loads(best_row["config_json"])
        proposal = ExperimentProposal(
            max_features=int(best_config["max_features"]),
            selector_type=str(best_config["selector_type"]),
            bucket_quotas=list(best_config.get("bucket_quotas") or [
                best_config["bucket_targets"]["short_form"],
                best_config["bucket_targets"]["medium_form"],
                best_config["bucket_targets"]["delta"],
                best_config["bucket_targets"]["context"],
            ]),
            exclude_patterns=list(best_config.get("exclude_patterns", [])),
            force_include_patterns=list(best_config.get("force_include_patterns", [])),
            trials=int(best_config.get("search_iterations", train_module.TRIALS)),
            folds=int(best_config.get("time_series_splits", train_module.FOLDS)),
            rationale="Promote the best fast configuration into a full overnight run.",
        )
        parent_experiment_id = int(best_row["id"])
        planner_decision = PlannerDecision(
            proposal=proposal,
            hypothesis=(
                f"Promote fast winner `{best_row['experiment_name']}` into full mode with the "
                "same feature-selection config."
            ),
            planner_type="promotion_winner",
            planner_model=None,
            reasoning="Use the best fast experiment as the ratcheted winner for the full run.",
        )

    update_train_config(train_path=train_path, proposal=proposal)
    train_api = _reload_train_module()
    effective_config = train_api.resolve_effective_config("full")
    experiment_name = train_api.build_experiment_name("full", effective_config)
    planner_decision = PlannerDecision(
        proposal=proposal,
        hypothesis=(
            f"{planner_decision.hypothesis} Full mode upgrades to "
            f"{effective_config.search_iterations} trials and {effective_config.time_series_splits} folds."
        ),
        planner_type=planner_decision.planner_type,
        planner_model=planner_decision.planner_model,
        reasoning=planner_decision.reasoning,
    )

    experiment_id = _insert_started_experiment(
        db_path=db_path,
        mode="full",
        planner_decision=planner_decision,
        proposal=proposal,
        experiment_name=experiment_name,
        session_id=session_id,
        parent_experiment_id=parent_experiment_id,
    )
    completed = _run_train_process(
        mode="full",
        experiment_name=experiment_name,
        train_path=train_path,
    )
    stdout_path, stderr_path = _write_process_logs(
        experiment_name=experiment_name,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )

    result_payload = None
    status = "failed"
    error_message = None
    if completed.returncode == 0:
        result_payload = _parse_result_payload(completed.stdout)
        if result_payload is not None:
            merged_config = dict(result_payload["config"])
            merged_config["bucket_quotas"] = list(proposal.bucket_quotas)
            result_payload["config"] = merged_config
        status = "succeeded"
    else:
        error_message = completed.stderr.strip() or completed.stdout.strip() or "train.py failed"

    _finalize_experiment(
        db_path=db_path,
        experiment_id=experiment_id,
        planner_decision=planner_decision,
        status=status,
        result_payload=result_payload,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        returncode=completed.returncode,
        planner_prompt_path=None,
        planner_response_path=None,
        error_message=error_message,
    )
    note_ids = maybe_record_experiment_notes(
        db_path=db_path,
        experiment_id=experiment_id,
        session_id=session_id,
        proposal=proposal,
        exploration_mode="full",
        prior_session_best=None if best_row is None else {
            "holdout_r2": best_row["holdout_r2"],
            "cv_rmse": best_row["cv_rmse"],
        },
        result_payload=result_payload,
        status=status,
        error_message=error_message,
    )
    return {
        "experiment_id": experiment_id,
        "session_id": session_id,
        "experiment_name": experiment_name,
        "source_fast_experiment_id": None if best_row is None else int(best_row["id"]),
        "status": status,
        "hypothesis": planner_decision.hypothesis,
        "planner_type": planner_decision.planner_type,
        "planner_model": planner_decision.planner_model,
        "result": result_payload,
        "note_ids": note_ids,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Autoresearch agent for MLB run-count experiments")
    parser.add_argument(
        "--action",
        choices=("planner-self-check", "fast-once", "full-best"),
        default="fast-once",
    )
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--program-path", default=str(DEFAULT_PROGRAM_PATH))
    parser.add_argument("--train-path", default=str(DEFAULT_TRAIN_PATH))
    parser.add_argument("--exploration-mode", choices=("fast",), default="fast")
    parser.add_argument("--session-id", type=int)
    args = parser.parse_args(argv)

    if args.action == "planner-self-check":
        payload = run_planner_self_check()
    elif args.action == "fast-once":
        payload = run_fast_once(
            db_path=args.db_path,
            program_path=args.program_path,
            train_path=args.train_path,
            exploration_mode=args.exploration_mode,
            session_id=args.session_id,
        )
    else:
        payload = run_best_full(
            db_path=args.db_path,
            program_path=args.program_path,
            train_path=args.train_path,
            session_id=args.session_id,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
