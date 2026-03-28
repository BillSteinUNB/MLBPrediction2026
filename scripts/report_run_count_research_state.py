from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REGISTRY_COLUMNS = [
    "experiment_dir",
    "artifact_path",
    "artifact_modified_time_utc",
    "model_name",
    "target_column",
    "model_version",
    "data_version_hash",
    "holdout_season",
    "cv_metric_name",
    "cv_best_score",
    "cv_best_rmse",
    "holdout_r2",
    "holdout_rmse",
    "holdout_mae",
    "holdout_poisson_deviance",
    "selected_feature_count",
    "feature_selection_mode",
    "blend_mode",
    "cv_aggregation_mode",
    "forced_delta_count",
    "forced_delta_feature_count",
    "weather_feature_count",
    "log5_feature_count",
    "plate_umpire_feature_count",
    "framing_feature_count",
    "delta_feature_count",
    "top_5_features",
    "metadata_warnings",
]


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _read_metadata(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed metadata JSON: {path}") from exc


def _relative_posix(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_feature_columns(metadata: dict[str, Any]) -> list[str] | None:
    feature_columns = metadata.get("feature_columns")
    if not isinstance(feature_columns, list):
        return None
    if any(not isinstance(feature, str) for feature in feature_columns):
        return None
    return feature_columns


def _count_feature_family(
    feature_columns: list[str] | None,
    *,
    prefix: str | None = None,
    token: str | None = None,
) -> int | None:
    if feature_columns is None:
        return None
    return sum(
        1
        for feature in feature_columns
        if (prefix is not None and feature.startswith(prefix))
        or (token is not None and token in feature)
    )


def _extract_top_features(metadata: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    rankings = metadata.get("feature_importance_rankings")
    if not isinstance(rankings, list):
        return []

    top_features: list[dict[str, Any]] = []
    for item in rankings:
        if not isinstance(item, dict):
            continue
        feature = item.get("feature")
        importance = _safe_float(item.get("importance"))
        if not isinstance(feature, str) or importance is None:
            continue
        top_features.append({"feature": feature, "importance": importance})
        if len(top_features) == limit:
            break
    return top_features


def _extract_selected_feature_count(metadata: dict[str, Any]) -> int | None:
    feature_columns = _safe_feature_columns(metadata)
    if feature_columns is None:
        return None
    return len(feature_columns)


def _extract_forced_delta_feature_count(
    metadata: dict[str, Any],
    metadata_warnings: list[str],
) -> int | None:
    explicit_count = _safe_int(metadata.get("forced_delta_feature_count"))
    forced_delta_features = metadata.get("forced_delta_features")
    counted_from_list: int | None = None
    if isinstance(forced_delta_features, list) and all(
        isinstance(feature, str) for feature in forced_delta_features
    ):
        counted_from_list = len(forced_delta_features)

    if explicit_count is not None and counted_from_list is not None and explicit_count != counted_from_list:
        metadata_warnings.append(
            "forced_delta_feature_count does not match forced_delta_features length"
        )
    if explicit_count is not None:
        return explicit_count
    return counted_from_list


def extract_registry_row(metadata_path: Path, *, repo_root: Path) -> dict[str, Any]:
    metadata = _read_metadata(metadata_path)
    holdout_metrics = metadata.get("holdout_metrics") or {}
    feature_columns = _safe_feature_columns(metadata)
    metadata_warnings: list[str] = []

    row = {
        "experiment_dir": _relative_posix(metadata_path.parent, repo_root),
        "artifact_path": _relative_posix(metadata_path, repo_root),
        "artifact_modified_time_utc": datetime.fromtimestamp(
            metadata_path.stat().st_mtime,
            tz=timezone.utc,
        ).isoformat(),
        "model_name": metadata.get("model_name"),
        "target_column": metadata.get("target_column"),
        "model_version": metadata.get("model_version"),
        "data_version_hash": metadata.get("data_version_hash"),
        "holdout_season": _safe_int(metadata.get("holdout_season")),
        "cv_metric_name": metadata.get("cv_metric_name"),
        "cv_best_score": _safe_float(metadata.get("cv_best_score")),
        "cv_best_rmse": _safe_float(metadata.get("cv_best_rmse")),
        "holdout_r2": _safe_float(holdout_metrics.get("r2")),
        "holdout_rmse": _safe_float(holdout_metrics.get("rmse")),
        "holdout_mae": _safe_float(holdout_metrics.get("mae")),
        "holdout_poisson_deviance": _safe_float(holdout_metrics.get("poisson_deviance")),
        "selected_feature_count": _extract_selected_feature_count(metadata),
        "feature_selection_mode": metadata.get("feature_selection_mode"),
        "blend_mode": metadata.get("blend_mode"),
        "cv_aggregation_mode": metadata.get("cv_aggregation_mode"),
        "forced_delta_count": _safe_int(metadata.get("forced_delta_count")),
        "forced_delta_feature_count": _extract_forced_delta_feature_count(
            metadata,
            metadata_warnings,
        ),
        "weather_feature_count": _count_feature_family(feature_columns, prefix="weather_"),
        "log5_feature_count": _count_feature_family(feature_columns, token="_log5_"),
        "plate_umpire_feature_count": _count_feature_family(
            feature_columns,
            prefix="plate_umpire_",
        ),
        "framing_feature_count": _count_feature_family(
            feature_columns,
            token="adjusted_framing",
        ),
        "delta_feature_count": _count_feature_family(feature_columns, token="_delta_7v30"),
        "top_5_features": _extract_top_features(metadata),
        "metadata_warnings": metadata_warnings,
        "_artifact_mtime_epoch": metadata_path.stat().st_mtime,
    }
    return row


def _registry_sort_key(row: dict[str, Any]) -> tuple[float, float, float]:
    holdout_r2 = row.get("holdout_r2")
    holdout_rmse = row.get("holdout_rmse")
    return (
        -float(row["_artifact_mtime_epoch"]),
        -(holdout_r2 if holdout_r2 is not None else float("-inf")),
        holdout_rmse if holdout_rmse is not None else float("inf"),
    )


def _serialize_csv_value(value: Any) -> str | int | float | None:
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":"), default=_json_default)
    return value


def _build_selection_reason(applied_filters: list[str]) -> str:
    if not applied_filters:
        return "selected top registry row after fallback"
    return "matched " + " + ".join(applied_filters)


def select_current_control(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "selected_artifact_path": None,
            "experiment_dir": None,
            "selection_reason": "no registry rows available",
            "warning": "No matching metadata artifacts were found.",
            "holdout_r2": None,
            "holdout_rmse": None,
            "holdout_poisson_deviance": None,
            "data_version_hash": None,
            "blend_mode": None,
            "feature_selection_mode": None,
        }

    candidates = list(rows)
    applied_filters: list[str] = []
    fallback_messages: list[str] = []
    preference_order = [
        (
            "forceddelta8",
            lambda row: "forceddelta8" in str(row.get("experiment_dir", "")).lower(),
        ),
        ("xgb_only", lambda row: row.get("blend_mode") == "xgb_only"),
        ("flat", lambda row: row.get("feature_selection_mode") == "flat"),
        ("holdout 2025", lambda row: row.get("holdout_season") == 2025),
    ]

    for label, predicate in preference_order:
        filtered = [row for row in candidates if predicate(row)]
        if filtered:
            candidates = filtered
            applied_filters.append(label)
        else:
            fallback_messages.append(f"no candidates matched {label}")

    candidates_with_r2 = [row for row in candidates if row.get("holdout_r2") is not None]
    finalists = candidates
    if candidates_with_r2:
        best_holdout_r2 = max(row["holdout_r2"] for row in candidates_with_r2)
        finalists = [row for row in candidates_with_r2 if row["holdout_r2"] == best_holdout_r2]
        applied_filters.append("highest holdout_r2")
    else:
        fallback_messages.append("holdout_r2 unavailable for remaining candidates")

    warning: str | None = None
    if len(finalists) > 1:
        warning_message = (
            f"{len(finalists)} candidates tied after control selection; "
            "selected the top registry-sorted artifact."
        )
        fallback_messages.append(warning_message)

    selected = finalists[0]
    if fallback_messages:
        warning = "; ".join(fallback_messages)

    return {
        "selected_artifact_path": selected.get("artifact_path"),
        "experiment_dir": selected.get("experiment_dir"),
        "selection_reason": _build_selection_reason(applied_filters),
        "warning": warning,
        "holdout_r2": selected.get("holdout_r2"),
        "holdout_rmse": selected.get("holdout_rmse"),
        "holdout_poisson_deviance": selected.get("holdout_poisson_deviance"),
        "data_version_hash": selected.get("data_version_hash"),
        "blend_mode": selected.get("blend_mode"),
        "feature_selection_mode": selected.get("feature_selection_mode"),
    }


def write_registry_outputs(
    rows: list[dict[str, Any]],
    *,
    models_root: Path,
    output_dir: Path,
    target_model: str,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    registry_basename = (
        target_model[: -len("_model")] if target_model.endswith("_model") else target_model
    )
    csv_path = output_dir / f"{registry_basename}_registry.csv"
    json_path = output_dir / f"{registry_basename}_registry.json"
    control_path = output_dir / "current_control.json"

    public_rows = [{key: row[key] for key in REGISTRY_COLUMNS} for row in rows]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_COLUMNS)
        writer.writeheader()
        for row in public_rows:
            writer.writerow({key: _serialize_csv_value(value) for key, value in row.items()})

    json_payload = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "models_root": models_root.resolve().as_posix(),
        "target_model": target_model,
        "row_count": len(public_rows),
        "rows": public_rows,
    }
    json_path.write_text(
        json.dumps(json_payload, indent=2, default=_json_default),
        encoding="utf-8",
    )

    control_payload = select_current_control(rows)
    control_path.write_text(
        json.dumps(control_payload, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return {
        "csv_path": csv_path,
        "json_path": json_path,
        "current_control_path": control_path,
    }


def generate_registry(
    *,
    models_root: Path,
    output_dir: Path,
    target_model: str,
    repo_root: Path | None = None,
) -> list[dict[str, Any]]:
    resolved_repo_root = repo_root or Path.cwd()
    pattern = f"{target_model}_*.metadata.json"
    rows = [
        extract_registry_row(metadata_path, repo_root=resolved_repo_root)
        for metadata_path in models_root.rglob(pattern)
    ]
    rows.sort(key=_registry_sort_key)
    write_registry_outputs(
        rows,
        models_root=models_root,
        output_dir=output_dir,
        target_model=target_model,
    )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Report the current full-game away-runs research state from metadata files.",
    )
    parser.add_argument(
        "--models-root",
        default=r"data\models",
        help="Root directory to scan for model metadata files.",
    )
    parser.add_argument(
        "--output-dir",
        default=r"data\reports\run_count\registry",
        help="Directory where registry artifacts will be written.",
    )
    parser.add_argument(
        "--target-model",
        default="full_game_away_runs_model",
        help="Model name prefix used to locate metadata artifacts.",
    )
    args = parser.parse_args(argv)

    models_root = Path(args.models_root)
    output_dir = Path(args.output_dir)
    rows = generate_registry(
        models_root=models_root,
        output_dir=output_dir,
        target_model=args.target_model,
        repo_root=Path.cwd(),
    )
    print(f"Wrote {len(rows)} registry rows to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
