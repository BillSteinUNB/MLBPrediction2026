from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.artifact_runtime import validate_runtime_versions  # noqa: E402
from src.model.data_builder import inspect_run_count_training_data, validate_run_count_training_data  # noqa: E402
from src.model.run_distribution_metrics import (  # noqa: E402
    DEFAULT_CALIBRATION_BIN_COUNT,
    DEFAULT_SUPPORT_TAIL_PROBABILITY,
    dataclass_to_dict,
    event_probability,
    fit_negative_binomial_dispersion,
    fit_zero_adjustment,
    negative_binomial_pmf_matrix,
    poisson_pmf_matrix,
    resolve_support_max,
    summarize_distribution_metrics,
    zero_adjusted_negative_binomial_pmf_matrix,
)


DEFAULT_CURRENT_CONTROL_PATH = Path("data/reports/run_count/registry/current_control.json")
DEFAULT_REGISTRY_PATH = Path("data/reports/run_count/registry/full_game_away_runs_registry.json")
DEFAULT_TRAINING_ROOT = Path("data/training")
DEFAULT_OUTPUT_DIR = Path("data/reports/run_count/distribution_eval")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate historical away-run artifacts as probabilistic run distributions.",
    )
    parser.add_argument("--current-control", default=str(DEFAULT_CURRENT_CONTROL_PATH))
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--training-data", default=None)
    parser.add_argument("--training-root", default=str(DEFAULT_TRAINING_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument("--weaker-artifact", default=None)
    parser.add_argument("--tail-probability", type=float, default=DEFAULT_SUPPORT_TAIL_PROBABILITY)
    parser.add_argument("--calibration-bins", type=int, default=DEFAULT_CALIBRATION_BIN_COUNT)
    args = parser.parse_args(argv)

    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    registry_payload = _read_json((PROJECT_ROOT / args.registry).resolve())
    current_control_payload = _read_json((PROJECT_ROOT / args.current_control).resolve())

    control_metadata_path = (PROJECT_ROOT / str(current_control_payload["selected_artifact_path"])).resolve()
    weaker_metadata_path = (
        (PROJECT_ROOT / args.weaker_artifact).resolve()
        if args.weaker_artifact
        else _select_weaker_comparable_path(
            registry_payload,
            control_payload=current_control_payload,
        )
    )

    requested_paths = [control_metadata_path, weaker_metadata_path]
    requested_paths.extend((PROJECT_ROOT / path).resolve() for path in args.artifact)

    artifact_paths: list[Path] = []
    seen: set[Path] = set()
    for path in requested_paths:
        if path not in seen:
            artifact_paths.append(path)
            seen.add(path)

    parquet_resolution_cache: dict[str, Path] = {}
    evaluations: list[dict[str, Any]] = []
    flat_summary_rows: list[dict[str, Any]] = []
    explicit_training_data = Path(args.training_data).resolve() if args.training_data else None
    training_root = (PROJECT_ROOT / args.training_root).resolve()

    for metadata_path in artifact_paths:
        evaluation = evaluate_artifact(
            metadata_path=metadata_path,
            explicit_training_data=explicit_training_data,
            training_root=training_root,
            output_dir=output_dir,
            tail_probability=float(args.tail_probability),
            calibration_bins=int(args.calibration_bins),
            parquet_resolution_cache=parquet_resolution_cache,
        )
        evaluations.append(evaluation)
        flat_summary_rows.extend(evaluation["csv_rows"])

    summary_payload = build_summary_payload(
        evaluations=evaluations,
        control_metadata_path=control_metadata_path,
        weaker_metadata_path=weaker_metadata_path,
    )
    summary_json_path = output_dir / "distribution_eval_summary.json"
    summary_csv_path = output_dir / "distribution_eval_summary.csv"
    summary_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    pd.DataFrame(flat_summary_rows).to_csv(summary_csv_path, index=False)

    print(f"Evaluated control: {summary_payload['comparison']['control_model_version']}")
    print(f"Evaluated weaker comparable: {summary_payload['comparison']['weaker_model_version']}")
    print(f"Summary JSON: {summary_json_path}")
    print(f"Summary CSV: {summary_csv_path}")
    return 0


def evaluate_artifact(
    *,
    metadata_path: Path,
    explicit_training_data: Path | None,
    training_root: Path,
    output_dir: Path,
    tail_probability: float,
    calibration_bins: int,
    parquet_resolution_cache: dict[str, Path],
) -> dict[str, Any]:
    metadata = _read_json(metadata_path)
    metadata["_metadata_path"] = str(metadata_path)
    model_path = _metadata_path_to_model_path(metadata_path)
    runtime_report = validate_runtime_versions(metadata, artifact_path=model_path, strict=False)
    estimator = joblib.load(model_path)

    training_data_path = resolve_training_data_path(
        metadata=metadata,
        explicit_training_data=explicit_training_data,
        training_root=training_root,
        parquet_resolution_cache=parquet_resolution_cache,
    )
    dataset = validate_run_count_training_data(training_data_path)
    inspection = inspect_run_count_training_data(training_data_path)
    filtered_dataset, window_details = filter_dataset_for_artifact(dataset=dataset, metadata=metadata)

    target_column = str(metadata["target_column"])
    feature_columns = [str(column) for column in metadata["feature_columns"]]
    holdout_season = int(metadata["holdout_season"])
    train_frame = filtered_dataset.loc[filtered_dataset["season"] < holdout_season].copy()
    holdout_frame = filtered_dataset.loc[filtered_dataset["season"] == holdout_season].copy()

    train_actual = train_frame[target_column].astype(int).to_numpy()
    holdout_actual = holdout_frame[target_column].astype(int).to_numpy()
    train_mean = estimator.predict(train_frame[feature_columns]).astype(float)
    holdout_mean = estimator.predict(holdout_frame[feature_columns]).astype(float)

    negative_binomial_fit = fit_negative_binomial_dispersion(train_actual, train_mean)
    baseline_train_support = range(
        resolve_support_max(
            train_actual,
            train_mean,
            family="negative_binomial",
            dispersion_size=negative_binomial_fit.dispersion_size,
            tail_probability=tail_probability,
        )
        + 1
    )
    baseline_train_pmf = negative_binomial_pmf_matrix(
        train_mean,
        baseline_train_support,
        dispersion_size=negative_binomial_fit.dispersion_size,
    )
    zero_adjustment_fit = fit_zero_adjustment(
        train_actual,
        event_probability(baseline_train_pmf, baseline_train_support, kind="eq", threshold=0),
    )

    family_payloads: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    for family_name in ("poisson", "negative_binomial", "zero_adjusted_negative_binomial"):
        family_payload = _evaluate_distribution_family(
            family_name=family_name,
            holdout_actual=holdout_actual,
            holdout_mean=holdout_mean,
            negative_binomial_fit=negative_binomial_fit,
            zero_adjustment_fit=zero_adjustment_fit,
            tail_probability=tail_probability,
            calibration_bins=calibration_bins,
        )
        family_payloads.append(family_payload)
        csv_rows.append(
            flatten_family_metrics_row(
                metadata=metadata,
                training_data_path=training_data_path,
                family_payload=family_payload,
            )
        )

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "artifact_path": _relative_to_project(metadata_path),
        "model_path": _relative_to_project(model_path),
        "training_data_path": _relative_to_project(training_data_path),
        "experiment_dir": _relative_to_project(metadata_path.parent),
        "model_name": metadata.get("model_name"),
        "model_version": metadata.get("model_version"),
        "target_column": target_column,
        "data_version_hash": metadata.get("data_version_hash"),
        "holdout_season": holdout_season,
        "window_details": window_details,
        "artifact_holdout_metrics": metadata.get("holdout_metrics", {}),
        "runtime_compatibility": {
            "compatible": runtime_report.compatible,
            "warnings": runtime_report.warnings,
            "errors": runtime_report.errors,
        },
        "training_data_inspection": {
            "parquet_path": _relative_to_project(inspection.parquet_path) if inspection.parquet_path else None,
            "data_version_hash": inspection.data_version_hash,
            "schema_name": inspection.schema_name,
            "schema_version": inspection.schema_version,
            "row_count": inspection.row_count,
        },
        "regenerated_prediction_summary": {
            "train_row_count": int(len(train_frame)),
            "holdout_row_count": int(len(holdout_frame)),
            "train_predicted_mean": float(train_mean.mean()),
            "holdout_predicted_mean": float(holdout_mean.mean()),
            "holdout_actual_mean": float(holdout_actual.mean()),
        },
        "distribution_families": family_payloads,
    }

    json_output_path = output_dir / f"{metadata['model_version']}.distribution_eval.json"
    csv_output_path = output_dir / f"{metadata['model_version']}.distribution_eval.csv"
    json_output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(csv_rows).to_csv(csv_output_path, index=False)
    payload["output_paths"] = {
        "json": _relative_to_project(json_output_path),
        "csv": _relative_to_project(csv_output_path),
    }
    payload["csv_rows"] = csv_rows
    return payload


def _evaluate_distribution_family(
    *,
    family_name: str,
    holdout_actual: Any,
    holdout_mean: Any,
    negative_binomial_fit: Any,
    zero_adjustment_fit: Any,
    tail_probability: float,
    calibration_bins: int,
) -> dict[str, Any]:
    if family_name == "poisson":
        support = range(
            resolve_support_max(
                holdout_actual,
                holdout_mean,
                family="poisson",
                tail_probability=tail_probability,
            )
            + 1
        )
        pmf = poisson_pmf_matrix(holdout_mean, support)
        fitted_parameters: dict[str, Any] = {}
    elif family_name == "negative_binomial":
        support = range(
            resolve_support_max(
                holdout_actual,
                holdout_mean,
                family="negative_binomial",
                dispersion_size=negative_binomial_fit.dispersion_size,
                tail_probability=tail_probability,
            )
            + 1
        )
        pmf = negative_binomial_pmf_matrix(
            holdout_mean,
            support,
            dispersion_size=negative_binomial_fit.dispersion_size,
        )
        fitted_parameters = {
            "negative_binomial_fit": dataclass_to_dict(negative_binomial_fit),
        }
    else:
        support = range(
            resolve_support_max(
                holdout_actual,
                holdout_mean,
                family="zero_adjusted_negative_binomial",
                dispersion_size=negative_binomial_fit.dispersion_size,
                tail_probability=tail_probability,
            )
            + 1
        )
        pmf = zero_adjusted_negative_binomial_pmf_matrix(
            holdout_mean,
            support,
            dispersion_size=negative_binomial_fit.dispersion_size,
            zero_adjustment_delta=zero_adjustment_fit.delta,
        )
        fitted_parameters = {
            "negative_binomial_fit": dataclass_to_dict(negative_binomial_fit),
            "zero_adjustment_fit": dataclass_to_dict(zero_adjustment_fit),
        }

    return {
        "family": family_name,
        "support_max": int(max(support)),
        "fitted_parameters": fitted_parameters,
        "holdout_metrics": summarize_distribution_metrics(
            holdout_actual,
            pmf,
            support,
            calibration_bin_count=calibration_bins,
        ),
    }


def resolve_training_data_path(
    *,
    metadata: dict[str, Any],
    explicit_training_data: Path | None,
    training_root: Path,
    parquet_resolution_cache: dict[str, Path],
) -> Path:
    expected_hash = str(metadata.get("data_version_hash") or "").strip()
    if not expected_hash:
        raise ValueError("artifact metadata is missing data_version_hash")

    if explicit_training_data is not None:
        inspection = inspect_run_count_training_data(explicit_training_data)
        if inspection.data_version_hash != expected_hash:
            raise ValueError(
                f"Explicit training parquet {explicit_training_data} has hash {inspection.data_version_hash}, "
                f"expected {expected_hash}."
            )
        return explicit_training_data

    cached = parquet_resolution_cache.get(expected_hash)
    if cached is not None:
        return cached

    candidates = discover_training_data_candidates(training_root, expected_hash=expected_hash)
    if not candidates:
        available_hashes = sorted(
            {
                value
                for value in (
                    _read_data_version_hash_from_sidecar(parquet_path)
                    for parquet_path in training_root.glob("*.parquet")
                )
                if value
            }
        )
        raise FileNotFoundError(
            f"No training parquet under {training_root} matches data_version_hash {expected_hash}. "
            f"Available hashes: {available_hashes}"
        )

    preferred_name = "ParquetDefault.parquet"
    selected = next(
        (candidate for candidate in candidates if candidate.name == preferred_name),
        max(candidates, key=lambda path: path.stat().st_mtime),
    )
    parquet_resolution_cache[expected_hash] = selected
    return selected


def discover_training_data_candidates(training_root: Path, *, expected_hash: str) -> list[Path]:
    candidates: list[Path] = []
    for parquet_path in training_root.glob("*.parquet"):
        sidecar_hash = _read_data_version_hash_from_sidecar(parquet_path)
        if sidecar_hash == expected_hash:
            candidates.append(parquet_path.resolve())
    if candidates:
        return candidates

    for parquet_path in training_root.glob("*.parquet"):
        inspection = inspect_run_count_training_data(parquet_path)
        if inspection.data_version_hash == expected_hash:
            candidates.append(parquet_path.resolve())
    return candidates


def filter_dataset_for_artifact(
    *,
    dataset: pd.DataFrame,
    metadata: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    holdout_season = int(metadata["holdout_season"])
    holdout_row_count = metadata.get("holdout_row_count")
    train_row_count = metadata.get("train_row_count")
    season_counts = dataset.groupby(dataset["season"].astype(int)).size().sort_index()
    if holdout_season not in season_counts.index:
        raise ValueError(f"Holdout season {holdout_season} is not present in the resolved training parquet")

    if holdout_row_count is not None and int(holdout_row_count) != int(season_counts.loc[holdout_season]):
        raise ValueError(
            f"Holdout row-count mismatch for season {holdout_season}: parquet has {int(season_counts.loc[holdout_season])}, "
            f"artifact metadata expects {holdout_row_count}."
        )

    if train_row_count is None:
        filtered = dataset.loc[dataset["season"].astype(int) <= holdout_season].copy()
        return filtered, {
            "start_season": int(filtered["season"].min()),
            "end_season": int(filtered["season"].max()),
            "selection_reason": "metadata missing train_row_count; used all seasons through holdout",
        }

    season_index = [int(season) for season in season_counts.index.tolist() if int(season) <= holdout_season]
    candidate_starts: list[int] = []
    for start_season in season_index:
        included_training_rows = sum(
            int(season_counts.loc[season])
            for season in season_index
            if start_season <= int(season) < holdout_season
        )
        if included_training_rows == int(train_row_count):
            candidate_starts.append(int(start_season))

    if len(candidate_starts) != 1:
        raise ValueError(
            f"Could not infer a unique season window for artifact {metadata.get('model_version')}. "
            f"Expected train_row_count={train_row_count}, holdout_row_count={holdout_row_count}, "
            f"candidate_starts={candidate_starts}"
        )

    start_season = candidate_starts[0]
    filtered = dataset.loc[
        dataset["season"].astype(int).between(start_season, holdout_season, inclusive="both")
    ].copy()
    return filtered, {
        "start_season": start_season,
        "end_season": holdout_season,
        "selection_reason": "inferred contiguous season window from artifact train_row_count/holdout_row_count",
    }


def flatten_family_metrics_row(
    *,
    metadata: dict[str, Any],
    training_data_path: Path,
    family_payload: dict[str, Any],
) -> dict[str, Any]:
    holdout_metrics = family_payload["holdout_metrics"]
    zero_calibration = holdout_metrics["zero_calibration"]
    tail_calibration = holdout_metrics["tail_calibration"]
    interval_coverage = holdout_metrics["interval_coverage"]
    prediction_summary = holdout_metrics["prediction_summary"]
    return {
        "model_version": metadata.get("model_version"),
        "artifact_path": _relative_to_project(Path(str(metadata["_metadata_path"]))),
        "training_data_path": _relative_to_project(training_data_path),
        "family": family_payload["family"],
        "support_max": family_payload["support_max"],
        "mean_crps": holdout_metrics["mean_crps"],
        "mean_log_score": holdout_metrics["mean_log_score"],
        "mean_negative_log_score": holdout_metrics["mean_negative_log_score"],
        "zero_abs_error": zero_calibration["p_0"]["absolute_error"],
        "ge_1_abs_error": zero_calibration["p_ge_1"]["absolute_error"],
        "ge_3_abs_error": tail_calibration["p_ge_3"]["absolute_error"],
        "ge_5_abs_error": tail_calibration["p_ge_5"]["absolute_error"],
        "ge_10_abs_error": tail_calibration["p_ge_10"]["absolute_error"],
        "interval_50_coverage": interval_coverage["central_50"]["empirical_coverage"],
        "interval_50_mean_width": interval_coverage["central_50"]["mean_width"],
        "interval_80_coverage": interval_coverage["central_80"]["empirical_coverage"],
        "interval_80_mean_width": interval_coverage["central_80"]["mean_width"],
        "interval_95_coverage": interval_coverage["central_95"]["empirical_coverage"],
        "interval_95_mean_width": interval_coverage["central_95"]["mean_width"],
        "predicted_mean_runs": prediction_summary["mean_predicted_runs"],
        "predicted_p_0": prediction_summary["mean_predicted_p_0"],
        "predicted_p_ge_3": prediction_summary["mean_predicted_p_ge_3"],
        "predicted_p_ge_5": prediction_summary["mean_predicted_p_ge_5"],
        "predicted_p_ge_10": prediction_summary["mean_predicted_p_ge_10"],
        "dispersion_size": family_payload["fitted_parameters"].get("negative_binomial_fit", {}).get("dispersion_size"),
        "zero_adjustment_delta": family_payload["fitted_parameters"].get("zero_adjustment_fit", {}).get("delta"),
    }


def build_summary_payload(
    *,
    evaluations: list[dict[str, Any]],
    control_metadata_path: Path,
    weaker_metadata_path: Path,
) -> dict[str, Any]:
    evaluation_lookup = {
        (PROJECT_ROOT / evaluation["artifact_path"]).resolve(): evaluation
        for evaluation in evaluations
    }
    control = evaluation_lookup[control_metadata_path.resolve()]
    weaker = evaluation_lookup[weaker_metadata_path.resolve()]

    control_family_lookup = {item["family"]: item for item in control["distribution_families"]}
    weaker_family_lookup = {item["family"]: item for item in weaker["distribution_families"]}
    family_comparison: dict[str, Any] = {}
    for family_name in sorted(control_family_lookup):
        control_metrics = control_family_lookup[family_name]["holdout_metrics"]
        weaker_metrics = weaker_family_lookup[family_name]["holdout_metrics"]
        family_comparison[family_name] = {
            "control_mean_crps": control_metrics["mean_crps"],
            "weaker_mean_crps": weaker_metrics["mean_crps"],
            "control_mean_negative_log_score": control_metrics["mean_negative_log_score"],
            "weaker_mean_negative_log_score": weaker_metrics["mean_negative_log_score"],
            "control_beats_weaker_on_crps": control_metrics["mean_crps"] < weaker_metrics["mean_crps"],
            "control_beats_weaker_on_negative_log_score": (
                control_metrics["mean_negative_log_score"] < weaker_metrics["mean_negative_log_score"]
            ),
        }

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "evaluated_artifacts": [
            {
                "model_version": evaluation["model_version"],
                "artifact_path": evaluation["artifact_path"],
                "training_data_path": evaluation["training_data_path"],
                "output_paths": evaluation["output_paths"],
            }
            for evaluation in evaluations
        ],
        "comparison": {
            "control_model_version": control["model_version"],
            "control_artifact_path": control["artifact_path"],
            "weaker_model_version": weaker["model_version"],
            "weaker_artifact_path": weaker["artifact_path"],
            "family_comparison": family_comparison,
        },
    }


def _select_weaker_comparable_path(
    registry_payload: dict[str, Any],
    *,
    control_payload: dict[str, Any],
) -> Path:
    rows = list(registry_payload.get("rows", []))
    control_artifact_path = str(control_payload["selected_artifact_path"]).replace("\\", "/")
    control_row = next(
        (row for row in rows if str(row.get("artifact_path", "")).replace("\\", "/") == control_artifact_path),
        None,
    )
    if control_row is None:
        raise ValueError(f"Could not locate control artifact {control_artifact_path} inside registry")

    candidates = [
        row
        for row in rows
        if str(row.get("artifact_path", "")).replace("\\", "/") != control_artifact_path
        and row.get("target_column") == control_row.get("target_column")
        and row.get("holdout_season") == control_row.get("holdout_season")
        and row.get("data_version_hash") == control_row.get("data_version_hash")
        and row.get("feature_selection_mode") == control_row.get("feature_selection_mode")
        and row.get("blend_mode") == control_row.get("blend_mode")
        and row.get("holdout_r2") is not None
        and float(row["holdout_r2"]) < float(control_row["holdout_r2"])
    ]
    if not candidates:
        candidates = [
            row
            for row in rows
            if str(row.get("artifact_path", "")).replace("\\", "/") != control_artifact_path
            and row.get("target_column") == control_row.get("target_column")
            and row.get("holdout_season") == control_row.get("holdout_season")
            and row.get("holdout_r2") is not None
            and float(row["holdout_r2"]) < float(control_row["holdout_r2"])
        ]
    if not candidates:
        raise ValueError("Could not auto-select a weaker comparable artifact from the Stage 1 registry")

    selected = min(
        candidates,
        key=lambda row: (
            float(row.get("holdout_r2") or 999.0),
            float(row.get("holdout_rmse") or 999.0),
            str(row.get("artifact_modified_time_utc") or ""),
        ),
    )
    return (PROJECT_ROOT / str(selected["artifact_path"])).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _read_data_version_hash_from_sidecar(parquet_path: Path) -> str | None:
    sidecar_path = parquet_path.with_suffix(".metadata.json")
    if not sidecar_path.exists():
        return None
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    value = payload.get("data_version_hash")
    return str(value) if value is not None else None


def _metadata_path_to_model_path(metadata_path: Path) -> Path:
    suffix = ".metadata.json"
    if not metadata_path.name.endswith(suffix):
        raise ValueError(f"Expected metadata artifact path, got {metadata_path}")
    return metadata_path.with_name(metadata_path.name[: -len(suffix)] + ".joblib")


def _relative_to_project(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
