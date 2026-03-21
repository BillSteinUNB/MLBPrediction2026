from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml

from src.ops.experiment_report import write_experiment_report


DEFAULT_AUTORESEARCH_DIR = Path("data") / "autoresearch"
DEFAULT_MODELS_DIR = Path("data") / "models"
DEFAULT_EXPERIMENTS_DIR = Path("data") / "experiments"


@dataclass(frozen=True, slots=True)
class ObjectiveConfig:
    metric: str
    direction: Literal["max", "min"]


@dataclass(frozen=True, slots=True)
class ConstraintConfig:
    metric: str
    operator: Literal["<=", ">=", "<", ">", "=="]
    value: float


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run local autoresearch experiment sweeps")
    parser.add_argument("--config", required=True)
    parser.add_argument("--session-name")
    parser.add_argument("--trials", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    payload = _load_yaml(args.config)
    session_name = args.session_name or str(payload.get("session_name") or "autoresearch")
    max_trials = int(args.trials or payload.get("trials", 10))
    session_dir = _resolve_session_dir(session_name, resume=args.resume)
    session_dir.mkdir(parents=True, exist_ok=True)

    git_snapshot = _capture_git_snapshot(session_dir)
    trial_log_path = session_dir / "trial_log.jsonl"
    leaderboard_path = session_dir / "leaderboard.csv"
    session_manifest_path = session_dir / "session_manifest.json"

    manifest = {
        "session_name": session_name,
        "created_at": datetime.now(UTC).isoformat(),
        "config_path": str(Path(args.config).resolve()),
        "max_trials": max_trials,
        "git": git_snapshot,
        "payload": payload,
    }
    if not session_manifest_path.exists():
        session_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    existing_trials = _read_existing_trials(trial_log_path)
    completed_trials = len(existing_trials)
    rng = random.Random(int(payload.get("random_seed", 20260321)))
    objective = _parse_objective(payload)
    constraints = _parse_constraints(payload)

    for trial_index in range(completed_trials + 1, max_trials + 1):
        base_params = _select_base_params(existing_trials, payload, objective, constraints)
        trial_params = _mutate_params(
            base_params=base_params,
            search_space=dict(payload.get("search_space", {})),
            mutation_count=int(payload.get("max_param_changes_per_trial", 2)),
            rng=rng,
        )
        trial_name = f"trial-{trial_index:03d}"
        experiment_name = f"{session_name}-{trial_name}"
        output_dir = Path(payload.get("output_root", DEFAULT_MODELS_DIR)) / experiment_name

        command = _build_command(
            module=str(payload["module"]),
            fixed_args=dict(payload.get("fixed_args", {})),
            trial_params=trial_params,
            output_dir=output_dir,
        )

        trial_started_at = datetime.now(UTC)
        stdout_path = session_dir / f"{trial_name}.stdout.log"
        stderr_path = session_dir / f"{trial_name}.stderr.log"
        result_payload = _run_trial(command, stdout_path=stdout_path, stderr_path=stderr_path)
        duration_seconds = float(result_payload["duration_seconds"])

        record = {
            "trial_index": trial_index,
            "trial_name": trial_name,
            "experiment_name": experiment_name,
            "started_at": trial_started_at.isoformat(),
            "duration_seconds": duration_seconds,
            "params": trial_params,
            "base_params": base_params,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "returncode": int(result_payload["returncode"]),
            "status": result_payload["status"],
            "summary": result_payload.get("summary"),
            "git_head": git_snapshot.get("head"),
        }
        record["metrics"] = _extract_metrics(record["summary"])
        record["objective_metric"] = objective.metric
        record["objective_value"] = _objective_value(record["metrics"], objective)
        record["constraints_passed"] = _constraints_passed(record["metrics"], constraints)

        _append_jsonl(trial_log_path, record)
        existing_trials.append(record)
        _write_leaderboard(existing_trials, leaderboard_path, objective, constraints)

        if payload.get("refresh_reports", True):
            write_experiment_report(
                models_dir=payload.get("output_root", DEFAULT_MODELS_DIR),
                output_dir=payload.get("experiments_dir", DEFAULT_EXPERIMENTS_DIR),
            )

    final_payload = {
        "session_dir": str(session_dir),
        "leaderboard_path": str(leaderboard_path),
        "trial_log_path": str(trial_log_path),
        "completed_trials": len(existing_trials),
    }
    print(json.dumps(final_payload, indent=2, sort_keys=True))
    return 0


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = _json_ready(yaml.safe_load(handle) or {})
    if "module" not in payload:
        raise ValueError("Autoresearch config must include 'module'")
    return dict(payload)


def _resolve_session_dir(session_name: str, *, resume: bool) -> Path:
    root = DEFAULT_AUTORESEARCH_DIR / _slugify(session_name)
    if resume:
        return root
    if not root.exists():
        return root
    suffix = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return root.parent / f"{root.name}-{suffix}"


def _capture_git_snapshot(session_dir: Path) -> dict[str, Any]:
    head = _run_git(["rev-parse", "HEAD"])
    branch = _run_git(["branch", "--show-current"])
    status = _run_git(["status", "--short"])
    diff = _run_git(["diff", "--binary", "HEAD"])

    snapshot = {
        "head": head.strip(),
        "branch": branch.strip(),
        "status": status.splitlines(),
        "dirty": bool(status.strip()),
    }
    (session_dir / "git_snapshot.json").write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
    if diff.strip():
        (session_dir / "git_diff.patch").write_text(diff, encoding="utf-8")
    return snapshot


def _run_git(args: list[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout or ""


def _parse_objective(payload: dict[str, Any]) -> ObjectiveConfig:
    objective_payload = dict(payload.get("objective", {}))
    metric = str(objective_payload.get("metric", "bankroll_return_pct"))
    direction = str(objective_payload.get("direction", "max")).lower()
    if direction not in {"max", "min"}:
        raise ValueError("objective.direction must be 'max' or 'min'")
    return ObjectiveConfig(metric=metric, direction=direction)


def _parse_constraints(payload: dict[str, Any]) -> list[ConstraintConfig]:
    constraints: list[ConstraintConfig] = []
    for raw in payload.get("constraints", []):
        node = dict(raw)
        constraints.append(
            ConstraintConfig(
                metric=str(node["metric"]),
                operator=str(node.get("operator", "<=")),
                value=float(node["value"]),
            )
        )
    return constraints


def _read_existing_trials(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _select_base_params(
    trial_records: list[dict[str, Any]],
    payload: dict[str, Any],
    objective: ObjectiveConfig,
    constraints: list[ConstraintConfig],
) -> dict[str, Any]:
    baseline = dict(payload.get("baseline_params", {}))
    if not trial_records:
        return baseline

    eligible = [record for record in trial_records if record.get("status") == "success" and record.get("constraints_passed")]
    if not eligible:
        return baseline

    reverse = objective.direction == "max"
    sorted_records = sorted(
        eligible,
        key=lambda record: record.get("objective_value", float("-inf") if reverse else float("inf")),
        reverse=reverse,
    )
    return dict(sorted_records[0].get("params", baseline))


def _mutate_params(
    *,
    base_params: dict[str, Any],
    search_space: dict[str, Any],
    mutation_count: int,
    rng: random.Random,
) -> dict[str, Any]:
    resolved = dict(base_params)
    if not search_space:
        return resolved

    keys = list(search_space.keys())
    mutate_count = min(max(1, mutation_count), len(keys))
    chosen_keys = rng.sample(keys, k=mutate_count)
    for key in chosen_keys:
        space_node = dict(search_space[key])
        values = list(space_node.get("values", []))
        if not values:
            continue
        current_value = resolved.get(key, space_node.get("default"))
        candidates = [value for value in values if value != current_value]
        if not candidates:
            continue
        resolved[key] = rng.choice(candidates)
    return resolved


def _build_command(
    *,
    module: str,
    fixed_args: dict[str, Any],
    trial_params: dict[str, Any],
    output_dir: Path,
) -> list[str]:
    command = [sys.executable, "-m", module]
    merged = {**fixed_args, **trial_params, "output-dir": str(output_dir)}
    if module in {"src.model.calibration", "src.model.xgboost_trainer", "src.model.stacking"} and "experiment-name" not in merged:
        merged["experiment-name"] = output_dir.name
    for key, value in merged.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                command.append(flag)
            continue
        if value is None:
            continue
        command.extend([flag, str(value)])
    return command


def _run_trial(
    command: list[str],
    *,
    stdout_path: Path,
    stderr_path: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True)
    duration_seconds = time.perf_counter() - started
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")

    summary = None
    if completed.returncode == 0:
        try:
            summary = json.loads(completed.stdout)
        except json.JSONDecodeError:
            summary = None

    return {
        "status": "success" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "duration_seconds": duration_seconds,
        "summary": summary,
    }


def _extract_metrics(summary: dict[str, Any] | None) -> dict[str, Any]:
    if not summary:
        return {}
    keys = (
        "aggregate_brier_score",
        "aggregate_roi",
        "bankroll_return_pct",
        "ending_bankroll_units",
        "peak_bankroll_units",
        "max_drawdown_pct",
        "longest_losing_streak",
        "total_bets",
        "window_count",
    )
    return {key: summary.get(key) for key in keys}


def _objective_value(metrics: dict[str, Any], objective: ObjectiveConfig) -> float | None:
    value = metrics.get(objective.metric)
    if value is None:
        return None
    return float(value)


def _constraints_passed(metrics: dict[str, Any], constraints: list[ConstraintConfig]) -> bool:
    for constraint in constraints:
        value = metrics.get(constraint.metric)
        if value is None:
            return False
        numeric = float(value)
        target = float(constraint.value)
        if constraint.operator == "<=" and not (numeric <= target):
            return False
        if constraint.operator == ">=" and not (numeric >= target):
            return False
        if constraint.operator == "<" and not (numeric < target):
            return False
        if constraint.operator == ">" and not (numeric > target):
            return False
        if constraint.operator == "==" and not (numeric == target):
            return False
    return True


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_ready(payload), sort_keys=True) + "\n")


def _write_leaderboard(
    records: list[dict[str, Any]],
    path: Path,
    objective: ObjectiveConfig,
    constraints: list[ConstraintConfig],
) -> None:
    rows: list[dict[str, Any]] = []
    for record in records:
        row = {
            "trial_index": record.get("trial_index"),
            "trial_name": record.get("trial_name"),
            "experiment_name": record.get("experiment_name"),
            "status": record.get("status"),
            "duration_seconds": record.get("duration_seconds"),
            "constraints_passed": record.get("constraints_passed"),
            "objective_metric": objective.metric,
            "objective_value": record.get("objective_value"),
            "git_head": record.get("git_head"),
        }
        row.update({f"metric_{key}": value for key, value in dict(record.get("metrics", {})).items()})
        row.update({f"param_{key}": value for key, value in dict(record.get("params", {})).items()})
        rows.append(row)

    frame = pd.DataFrame(rows)
    if not frame.empty and "objective_value" in frame.columns:
        reverse = objective.direction == "max"
        frame = frame.sort_values(
            by=["constraints_passed", "objective_value", "trial_index"],
            ascending=[False, not reverse, True],
            kind="stable",
        )
    frame.to_csv(path, index=False)


def _slugify(value: str) -> str:
    cleaned = "".join(character.lower() if character.isalnum() else "-" for character in value.strip())
    cleaned = "-".join(segment for segment in cleaned.split("-") if segment)
    if not cleaned:
        raise ValueError("Session name must contain at least one alphanumeric character")
    return cleaned


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(nested) for key, nested in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if hasattr(value, "isoformat") and callable(value.isoformat):
        try:
            return value.isoformat()
        except TypeError:
            pass
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
