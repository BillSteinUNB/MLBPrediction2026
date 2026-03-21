from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import pandas as pd

if TYPE_CHECKING:
    from src.backtest.walk_forward import WalkForwardBacktestResult
    from src.model.calibration import CalibrationRunResult
    from src.model.xgboost_trainer import TrainingRunResult


DEFAULT_EXPERIMENT_TRACKING_DIR = Path("data") / "experiments"
DEFAULT_EXPERIMENT_LOG_PATH = DEFAULT_EXPERIMENT_TRACKING_DIR / "experiment_log.jsonl"
DEFAULT_EXPERIMENT_CSV_PATH = DEFAULT_EXPERIMENT_TRACKING_DIR / "experiment_log.csv"


def log_training_run(
    result: TrainingRunResult,
    *,
    experiment_name: str | None,
    output_dir: str | Path,
    training_data: str | Path,
    start_year: int,
    end_year: int,
    refresh_training_data: bool,
    allow_backfill_years: bool,
    search_iterations: int,
    time_series_splits: int,
    tracking_dir: str | Path = DEFAULT_EXPERIMENT_TRACKING_DIR,
) -> Path:
    record = {
        "run_type": "training",
        "experiment_name": experiment_name,
        "output_dir": str(Path(output_dir)),
        "training_data": str(Path(training_data)),
        "start_year": int(start_year),
        "end_year": int(end_year),
        "refresh_training_data": bool(refresh_training_data),
        "allow_backfill_years": bool(allow_backfill_years),
        "search_iterations": int(search_iterations),
        "time_series_splits": int(time_series_splits),
        "model_version": result.model_version,
        "data_version_hash": result.data_version_hash,
        "holdout_season": int(result.holdout_season),
        "feature_column_count": int(len(result.feature_columns)),
        "summary_path": str(result.summary_path),
        "models": {
            name: {
                "target_column": artifact.target_column,
                "best_params": artifact.best_params,
                "cv_best_log_loss": artifact.cv_best_log_loss,
                "holdout_metrics": artifact.holdout_metrics,
                "train_row_count": artifact.train_row_count,
                "holdout_row_count": artifact.holdout_row_count,
            }
            for name, artifact in result.models.items()
        },
    }
    return append_experiment_record(record, tracking_dir=tracking_dir)


def log_calibration_run(
    result: CalibrationRunResult,
    *,
    experiment_name: str | None,
    output_dir: str | Path,
    training_data: str | Path,
    start_year: int,
    end_year: int,
    refresh_training_data: bool,
    allow_backfill_years: bool,
    search_iterations: int,
    time_series_splits: int,
    tracking_dir: str | Path = DEFAULT_EXPERIMENT_TRACKING_DIR,
) -> Path:
    record = {
        "run_type": "calibration",
        "experiment_name": experiment_name,
        "output_dir": str(Path(output_dir)),
        "training_data": str(Path(training_data)),
        "start_year": int(start_year),
        "end_year": int(end_year),
        "refresh_training_data": bool(refresh_training_data),
        "allow_backfill_years": bool(allow_backfill_years),
        "search_iterations": int(search_iterations),
        "time_series_splits": int(time_series_splits),
        "model_version": result.model_version,
        "data_version_hash": result.data_version_hash,
        "holdout_season": int(result.holdout_season),
        "calibration_method": result.calibration_method,
        "calibration_fraction": float(result.calibration_fraction),
        "model_training_row_count": int(result.model_training_row_count),
        "calibration_row_count": int(result.calibration_row_count),
        "holdout_row_count": int(result.holdout_row_count),
        "summary_path": str(result.summary_path),
        "models": {
            name: {
                "target_column": artifact.target_column,
                "calibration_method": artifact.calibration_method,
                "holdout_metrics": artifact.holdout_metrics,
                "train_row_count": artifact.train_row_count,
                "calibration_row_count": artifact.calibration_row_count,
                "holdout_row_count": artifact.holdout_row_count,
            }
            for name, artifact in result.models.items()
        },
    }
    return append_experiment_record(record, tracking_dir=tracking_dir)


def log_walk_forward_run(
    result: WalkForwardBacktestResult,
    *,
    output_dir: str | Path,
    training_data: str | Path | None,
    start_date: str,
    end_date: str,
    train_window_months: int,
    test_window_months: int,
    window_mode: str,
    anchored_train_start: str | None,
    calibration_method: str,
    calibration_fraction: float,
    edge_threshold: float,
    market_vig: float,
    historical_odds_db_path: str | Path | None,
    historical_odds_book_name: str | None,
    starting_bankroll_units: float,
    staking_mode: str,
    flat_bet_size_units: float,
    kelly_fraction: float,
    max_bet_fraction: float,
    max_drawdown: float,
    tracking_dir: str | Path = DEFAULT_EXPERIMENT_TRACKING_DIR,
) -> Path:
    record = {
        "run_type": "walk_forward",
        "experiment_name": Path(output_dir).name,
        "output_dir": str(Path(output_dir)),
        "training_data": None if training_data is None else str(Path(training_data)),
        "start_date": str(start_date),
        "end_date": str(end_date),
        "train_window_months": int(train_window_months),
        "test_window_months": int(test_window_months),
        "window_mode": str(window_mode),
        "anchored_train_start": anchored_train_start,
        "calibration_method": str(calibration_method),
        "calibration_fraction": float(calibration_fraction),
        "edge_threshold": float(edge_threshold),
        "market_vig": float(market_vig),
        "starting_bankroll_units": float(starting_bankroll_units),
        "staking_mode": str(staking_mode),
        "flat_bet_size_units": float(flat_bet_size_units),
        "kelly_fraction": float(kelly_fraction),
        "max_bet_fraction": float(max_bet_fraction),
        "max_drawdown": float(max_drawdown),
        "historical_odds_db_path": (
            None if historical_odds_db_path is None else str(Path(historical_odds_db_path))
        ),
        "historical_odds_book_name": historical_odds_book_name,
        "aggregate_brier_score": float(result.aggregate_brier_score),
        "aggregate_roi": float(result.aggregate_roi),
        "bankroll_return_pct": float(result.bankroll_return_pct),
        "ending_bankroll_units": float(result.ending_bankroll_units),
        "peak_bankroll_units": float(result.peak_bankroll_units),
        "max_drawdown_pct": float(result.max_drawdown_pct),
        "longest_losing_streak": int(result.longest_losing_streak),
        "total_bets": int(result.total_bets),
        "window_count": int(result.window_count),
        "data_version_hash": result.data_version_hash,
        "code_version_hash": result.code_version_hash,
        "summary_path": str(result.summary_path),
        "predictions_path": str(result.predictions_path),
        "window_metrics_path": str(result.window_metrics_path),
        "window_builds": [
            {
                "window_index": build.window_index,
                "data_source": build.data_source,
                "build_action": build.build_action,
                "row_count": build.row_count,
                "train_row_count": build.train_row_count,
                "test_row_count": build.test_row_count,
                "cache_path": build.cache_path,
            }
            for build in result.window_builds
        ],
    }
    return append_experiment_record(record, tracking_dir=tracking_dir)


def append_experiment_record(
    record: Mapping[str, Any],
    *,
    tracking_dir: str | Path = DEFAULT_EXPERIMENT_TRACKING_DIR,
) -> Path:
    resolved_tracking_dir = Path(tracking_dir)
    resolved_tracking_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = resolved_tracking_dir / DEFAULT_EXPERIMENT_LOG_PATH.name
    csv_path = resolved_tracking_dir / DEFAULT_EXPERIMENT_CSV_PATH.name

    payload = dict(record)
    payload["tracked_at"] = datetime.now(UTC).isoformat()

    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")

    flattened_row = _flatten_record(payload)
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        combined = pd.concat([existing, pd.DataFrame([flattened_row])], ignore_index=True)
    else:
        combined = pd.DataFrame([flattened_row])
    combined.to_csv(csv_path, index=False)
    return jsonl_path


def _flatten_record(record: Mapping[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    _flatten_into(flattened, prefix="", value=record)
    return flattened


def _flatten_into(target: dict[str, Any], *, prefix: str, value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested_value in value.items():
            next_prefix = f"{prefix}_{key}" if prefix else str(key)
            _flatten_into(target, prefix=next_prefix, value=nested_value)
        return
    if isinstance(value, list):
        target[prefix] = json.dumps(value, sort_keys=True)
        return
    target[prefix] = value
