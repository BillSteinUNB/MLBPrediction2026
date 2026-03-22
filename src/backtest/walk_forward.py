from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from dataclasses import asdict, dataclass
from math import ceil
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from xgboost import XGBClassifier

from src.clients.historical_odds_client import load_historical_odds_for_games
from src.clients.odds_client import devig_probabilities
from src.clients.weather_client import fetch_game_weather
from src.engine.bankroll import (
    DEFAULT_KELLY_FRACTION,
    DEFAULT_MAX_DRAWDOWN,
    MAX_BET_FRACTION,
    calculate_kelly_stake,
)
from src.model.calibration import (
    CalibratedStackingModel,
    DEFAULT_CALIBRATION_METHOD,
    SUPPORTED_CALIBRATION_METHODS,
    _fit_probability_calibrator,
)
from src.model.data_builder import (
    WeatherFetcher,
    _compute_data_version_hash,
    assert_training_data_is_leakage_free,
    build_training_dataset,
)
from src.model.stacking import (
    DEFAULT_META_LEARNER_MAX_ITER,
    DEFAULT_RAW_META_FEATURE_COLUMNS,
    StackingEnsembleModel,
    _generate_temporal_oof_probabilities,
    _resolve_raw_meta_feature_columns,
)
from src.model.xgboost_trainer import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_XGBOOST_N_JOBS,
    _load_training_dataframe,
    _resolve_numeric_feature_columns,
)
from src.ops.experiment_tracker import log_walk_forward_run


logger = logging.getLogger(__name__)


DEFAULT_BACKTEST_OUTPUT_DIR = Path("data") / "backtest"
DEFAULT_BACKTEST_CACHE_DIR = Path("data") / "training"
DEFAULT_TRAIN_WINDOW_MONTHS = 6
DEFAULT_TEST_WINDOW_MONTHS = 1
DEFAULT_WINDOW_MODE = "rolling"
SUPPORTED_WINDOW_MODES = ("rolling", "anchored_expanding")
DEFAULT_STAKING_MODE = "flat"
SUPPORTED_STAKING_MODES = ("flat", "kelly", "edge_scaled", "edge_bucketed")
DEFAULT_CALIBRATION_FRACTION = 0.15
DEFAULT_WALK_FORWARD_CALIBRATION_METHOD = DEFAULT_CALIBRATION_METHOD
DEFAULT_EDGE_THRESHOLD = 0.03
DEFAULT_MAX_EDGE_TO_BET: float | None = None
DEFAULT_MARKET_VIG = 0.04
DEFAULT_TIME_SERIES_SPLITS = 3
DEFAULT_FLOAT_FORMAT = "%.10f"
DEFAULT_STARTING_BANKROLL_UNITS = 100.0
DEFAULT_FLAT_BET_SIZE_UNITS = 1.0
DEFAULT_MIN_BET_SIZE_UNITS = 0.5
DEFAULT_MAX_BET_SIZE_UNITS = 3.0
DEFAULT_EDGE_SCALE_CAP = 0.10
DEFAULT_ESTIMATOR_KWARGS: dict[str, Any] = {
    "max_depth": 3,
    "n_estimators": 120,
    "learning_rate": 0.05,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
}
_VERSION_HASH_PATHS = (
    Path("config") / "settings.yaml",
    Path("src") / "model" / "data_builder.py",
    Path("src") / "model" / "xgboost_trainer.py",
    Path("src") / "model" / "stacking.py",
    Path("src") / "model" / "calibration.py",
    Path("src") / "backtest" / "walk_forward.py",
)


@dataclass(frozen=True, slots=True)
class WalkForwardWindow:
    window_index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass(frozen=True, slots=True)
class _BankrollState:
    current_bankroll: float
    peak_bankroll: float
    max_drawdown_pct: float
    longest_losing_streak: int


@dataclass(frozen=True, slots=True)
class WalkForwardBacktestResult:
    predictions: pd.DataFrame
    window_metrics: pd.DataFrame
    predictions_path: Path
    window_metrics_path: Path
    summary_path: Path
    output_fingerprint: str
    aggregate_brier_score: float
    aggregate_roi: float
    bankroll_return_pct: float
    ending_bankroll_units: float
    peak_bankroll_units: float
    max_drawdown_pct: float
    longest_losing_streak: int
    total_bets: int
    window_count: int
    data_version_hash: str
    code_version_hash: str
    window_builds: tuple["WalkForwardWindowBuild", ...]


@dataclass(frozen=True, slots=True)
class WalkForwardWindowBuild:
    window_index: int
    data_source: str
    build_action: str
    scheduled_start_before: str
    cache_path: str | None
    row_count: int
    train_row_count: int
    test_row_count: int
    data_version_hash: str


@dataclass(frozen=True, slots=True)
class _WindowModelBundle:
    calibrated_model: CalibratedStackingModel
    model_training_row_count: int
    calibration_row_count: int
    calibration_method: str


def create_walk_forward_windows(
    training_data: pd.DataFrame | str | Path,
    *,
    start_date: str,
    end_date: str,
    train_window_months: int = DEFAULT_TRAIN_WINDOW_MONTHS,
    test_window_months: int = DEFAULT_TEST_WINDOW_MONTHS,
    window_mode: Literal["rolling", "anchored_expanding"] = DEFAULT_WINDOW_MODE,
    anchored_train_start: str | pd.Timestamp | None = None,
) -> list[WalkForwardWindow]:
    """Create monthly walk-forward windows with a 6-month train and 1-month test stride."""

    if train_window_months < 1:
        raise ValueError("train_window_months must be at least 1")
    if test_window_months < 1:
        raise ValueError("test_window_months must be at least 1")
    if window_mode not in SUPPORTED_WINDOW_MODES:
        raise ValueError(f"window_mode must be one of {SUPPORTED_WINDOW_MODES}")

    dataframe = _load_backtest_dataframe(training_data)
    requested_start = _month_floor(start_date)
    requested_end = _month_floor(end_date)
    anchored_start = (
        _month_floor(anchored_train_start)
        if anchored_train_start is not None
        else _month_floor(pd.to_datetime(dataframe["scheduled_start"], utc=True).min())
    )

    windows: list[WalkForwardWindow] = []
    test_start = requested_start
    window_index = 1
    while test_start <= requested_end:
        test_end = test_start + pd.DateOffset(months=test_window_months)
        train_start = (
            anchored_start
            if window_mode == "anchored_expanding"
            else test_start - pd.DateOffset(months=train_window_months)
        )
        train_end = test_start

        train_frame = _slice_time_range(dataframe, start=train_start, end=train_end)
        test_frame = _slice_time_range(dataframe, start=test_start, end=test_end)
        if not train_frame.empty and not test_frame.empty:
            windows.append(
                WalkForwardWindow(
                    window_index=window_index,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                )
            )
            window_index += 1

        test_start = test_start + pd.DateOffset(months=test_window_months)

    return windows


def run_walk_forward_backtest(
    *,
    training_data: pd.DataFrame | str | Path | None = None,
    start_date: str,
    end_date: str,
    output_dir: str | Path = DEFAULT_BACKTEST_OUTPUT_DIR,
    cache_dir: str | Path = DEFAULT_BACKTEST_CACHE_DIR,
    refresh_data: bool = False,
    seed: int = DEFAULT_RANDOM_STATE,
    train_window_months: int = DEFAULT_TRAIN_WINDOW_MONTHS,
    test_window_months: int = DEFAULT_TEST_WINDOW_MONTHS,
    window_mode: Literal["rolling", "anchored_expanding"] = DEFAULT_WINDOW_MODE,
    anchored_train_start: str | pd.Timestamp | None = None,
    calibration_fraction: float = DEFAULT_CALIBRATION_FRACTION,
    calibration_method: str = DEFAULT_WALK_FORWARD_CALIBRATION_METHOD,
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
    max_edge_to_bet: float | None = DEFAULT_MAX_EDGE_TO_BET,
    market_vig: float = DEFAULT_MARKET_VIG,
    time_series_splits: int = DEFAULT_TIME_SERIES_SPLITS,
    raw_meta_feature_columns: Sequence[str] = DEFAULT_RAW_META_FEATURE_COLUMNS,
    estimator_kwargs: Mapping[str, Any] | None = None,
    meta_learner_max_iter: int = DEFAULT_META_LEARNER_MAX_ITER,
    weather_fetcher: WeatherFetcher = fetch_game_weather,
    historical_odds_db_path: str | Path | None = None,
    historical_odds_book_name: str | None = None,
    starting_bankroll_units: float = DEFAULT_STARTING_BANKROLL_UNITS,
    staking_mode: Literal["flat", "kelly", "edge_scaled", "edge_bucketed"] = DEFAULT_STAKING_MODE,
    flat_bet_size_units: float = DEFAULT_FLAT_BET_SIZE_UNITS,
    min_bet_size_units: float = DEFAULT_MIN_BET_SIZE_UNITS,
    max_bet_size_units: float = DEFAULT_MAX_BET_SIZE_UNITS,
    edge_scale_cap: float = DEFAULT_EDGE_SCALE_CAP,
    kelly_fraction: float = DEFAULT_KELLY_FRACTION,
    max_bet_fraction: float = MAX_BET_FRACTION,
    max_drawdown: float = DEFAULT_MAX_DRAWDOWN,
) -> WalkForwardBacktestResult:
    """Run a deterministic walk-forward backtest and persist predictions plus window metrics."""

    _set_random_seed(seed)
    logger.info(
        "[backtest] Starting walk-forward from %s to %s",
        start_date,
        end_date,
    )
    resolved_estimator_kwargs = {**DEFAULT_ESTIMATOR_KWARGS, **dict(estimator_kwargs or {})}
    code_version_hash = _compute_code_version_hash()

    prediction_frames: list[pd.DataFrame] = []
    window_metric_rows: list[dict[str, Any]] = []
    window_builds: list[WalkForwardWindowBuild] = []

    if training_data is not None:
        dataframe = _load_backtest_dataframe(training_data)
        assert_training_data_is_leakage_free(dataframe)
        windows = create_walk_forward_windows(
            dataframe,
            start_date=start_date,
            end_date=end_date,
            train_window_months=train_window_months,
            test_window_months=test_window_months,
            window_mode=window_mode,
            anchored_train_start=anchored_train_start,
        )
        if not windows:
            raise ValueError("No walk-forward windows matched the requested date range")
        logger.info("[backtest] Prepared %s windows from explicit training data", len(windows))

        feature_columns = _resolve_numeric_feature_columns(dataframe)
        resolved_raw_meta_feature_columns = _resolve_raw_meta_feature_columns(
            dataframe,
            requested_columns=raw_meta_feature_columns,
        )
        data_version_hash = _resolve_data_version_hash(dataframe)
        cache_path = str(Path(training_data)) if isinstance(training_data, str | Path) else None
        bankroll_state = _BankrollState(
            current_bankroll=float(starting_bankroll_units),
            peak_bankroll=float(starting_bankroll_units),
            max_drawdown_pct=0.0,
            longest_losing_streak=0,
        )

        for window in windows:
            build = WalkForwardWindowBuild(
                window_index=window.window_index,
                data_source="explicit_training_data",
                build_action="provided",
                scheduled_start_before=window.test_end.isoformat(),
                cache_path=cache_path,
                row_count=int(len(dataframe)),
                train_row_count=int(len(_slice_time_range(dataframe, start=window.train_start, end=window.train_end))),
                test_row_count=int(len(_slice_time_range(dataframe, start=window.test_start, end=window.test_end))),
                data_version_hash=data_version_hash,
            )
            predictions, metrics = _evaluate_window(
                dataframe=dataframe,
                window=window,
                feature_columns=feature_columns,
                raw_meta_feature_columns=resolved_raw_meta_feature_columns,
                calibration_fraction=calibration_fraction,
                calibration_method=calibration_method,
                edge_threshold=edge_threshold,
                max_edge_to_bet=max_edge_to_bet,
                market_vig=market_vig,
                time_series_splits=time_series_splits,
                estimator_kwargs=resolved_estimator_kwargs,
                meta_learner_max_iter=meta_learner_max_iter,
                seed=seed,
                data_version_hash=data_version_hash,
                code_version_hash=code_version_hash,
                historical_odds_db_path=historical_odds_db_path,
                historical_odds_book_name=historical_odds_book_name,
                bankroll_state=bankroll_state,
                staking_mode=staking_mode,
                flat_bet_size_units=flat_bet_size_units,
                min_bet_size_units=min_bet_size_units,
                max_bet_size_units=max_bet_size_units,
                edge_scale_cap=edge_scale_cap,
                kelly_fraction=kelly_fraction,
                max_bet_fraction=max_bet_fraction,
                max_drawdown=max_drawdown,
            )
            metrics.update(_window_build_to_metric_fields(build))
            prediction_frames.append(predictions)
            window_metric_rows.append(metrics)
            window_builds.append(build)
            bankroll_state = _BankrollState(
                current_bankroll=float(metrics["ending_bankroll_units"]),
                peak_bankroll=float(metrics["peak_bankroll_units"]),
                max_drawdown_pct=float(metrics["max_drawdown_pct"]),
                longest_losing_streak=int(metrics["longest_losing_streak"]),
            )
    else:
        windows = _create_requested_windows(
            start_date=start_date,
            end_date=end_date,
            train_window_months=train_window_months,
            test_window_months=test_window_months,
            window_mode=window_mode,
            anchored_train_start=anchored_train_start,
        )
        logger.info("[backtest] Prepared %s requested windows", len(windows))
        bankroll_state = _BankrollState(
            current_bankroll=float(starting_bankroll_units),
            peak_bankroll=float(starting_bankroll_units),
            max_drawdown_pct=0.0,
            longest_losing_streak=0,
        )

        for window in windows:
            logger.info(
                "[backtest %s/%s] building data train=%s..%s test=%s..%s",
                window.window_index,
                len(windows),
                window.train_start.date(),
                (window.train_end - pd.Timedelta(days=1)).date(),
                window.test_start.date(),
                (window.test_end - pd.Timedelta(days=1)).date(),
            )
            dataframe, build = _resolve_window_training_data(
                window=window,
                cache_dir=cache_dir,
                refresh_data=refresh_data,
                weather_fetcher=weather_fetcher,
            )
            if build.train_row_count == 0 or build.test_row_count == 0:
                logger.info(
                    "[backtest %s/%s] skipping because feature data yielded %s train rows and %s test rows",
                    window.window_index,
                    len(windows),
                    build.train_row_count,
                    build.test_row_count,
                )
                continue

            feature_columns = _resolve_numeric_feature_columns(dataframe)
            resolved_raw_meta_feature_columns = _resolve_raw_meta_feature_columns(
                dataframe,
                requested_columns=raw_meta_feature_columns,
            )
            predictions, metrics = _evaluate_window(
                dataframe=dataframe,
                window=window,
                feature_columns=feature_columns,
                raw_meta_feature_columns=resolved_raw_meta_feature_columns,
                calibration_fraction=calibration_fraction,
                calibration_method=calibration_method,
                edge_threshold=edge_threshold,
                max_edge_to_bet=max_edge_to_bet,
                market_vig=market_vig,
                time_series_splits=time_series_splits,
                estimator_kwargs=resolved_estimator_kwargs,
                meta_learner_max_iter=meta_learner_max_iter,
                seed=seed,
                data_version_hash=build.data_version_hash,
                code_version_hash=code_version_hash,
                historical_odds_db_path=historical_odds_db_path,
                historical_odds_book_name=historical_odds_book_name,
                bankroll_state=bankroll_state,
                staking_mode=staking_mode,
                flat_bet_size_units=flat_bet_size_units,
                min_bet_size_units=min_bet_size_units,
                max_bet_size_units=max_bet_size_units,
                edge_scale_cap=edge_scale_cap,
                kelly_fraction=kelly_fraction,
                max_bet_fraction=max_bet_fraction,
                max_drawdown=max_drawdown,
            )
            metrics.update(_window_build_to_metric_fields(build))
            prediction_frames.append(predictions)
            window_metric_rows.append(metrics)
            window_builds.append(build)
            bankroll_state = _BankrollState(
                current_bankroll=float(metrics["ending_bankroll_units"]),
                peak_bankroll=float(metrics["peak_bankroll_units"]),
                max_drawdown_pct=float(metrics["max_drawdown_pct"]),
                longest_losing_streak=int(metrics["longest_losing_streak"]),
            )
            logger.info(
                "[backtest %s/%s] complete brier=%.4f roi=%.4f bankroll=%.2f bets=%s",
                window.window_index,
                len(windows),
                metrics["brier_score"],
                metrics["roi"],
                metrics["ending_bankroll_units"],
                metrics["bet_count"],
            )

        if not prediction_frames:
            raise ValueError("No walk-forward windows matched the requested date range")

    data_version_hash = _combine_window_data_version_hashes(window_builds)

    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["window_index", "scheduled_start", "game_pk"]
    )
    window_metrics = pd.DataFrame(window_metric_rows).sort_values("window_index").reset_index(
        drop=True
    )
    scored_mask = predictions["is_push"] == 0
    aggregate_brier = _safe_brier_score(
        predictions.loc[scored_mask, "actual_home_win"],
        predictions.loc[scored_mask, "model_home_prob"],
    )
    total_staked = float(predictions["bet_stake_units"].sum())
    total_profit = float(predictions["bet_profit_units"].sum())
    aggregate_roi = float(total_profit / total_staked) if total_staked else 0.0
    ending_bankroll_units = float(predictions["bankroll_after_units"].iloc[-1])
    peak_bankroll_units = float(predictions["peak_bankroll_units"].max())
    max_drawdown_pct = float(predictions["bankroll_drawdown_pct"].max())
    bankroll_return_pct = (
        float((ending_bankroll_units - float(starting_bankroll_units)) / float(starting_bankroll_units))
        if starting_bankroll_units
        else 0.0
    )
    longest_losing_streak = int(predictions["losing_streak"].max())

    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{_month_floor(start_date).strftime('%Y%m%d')}_{_month_floor(end_date).strftime('%Y%m%d')}"
    predictions_path = resolved_output_dir / f"walk_forward_predictions_{suffix}.csv"
    window_metrics_path = resolved_output_dir / f"walk_forward_windows_{suffix}.csv"
    summary_path = resolved_output_dir / f"walk_forward_summary_{suffix}.json"

    _write_csv(predictions, predictions_path)
    _write_csv(window_metrics, window_metrics_path)

    output_fingerprint = hashlib.sha256(
        predictions_path.read_bytes() + window_metrics_path.read_bytes()
    ).hexdigest()
    summary_payload = {
        "start_date": _month_floor(start_date).isoformat(),
        "end_date": _month_floor(end_date).isoformat(),
        "seed": seed,
        "train_window_months": train_window_months,
        "test_window_months": test_window_months,
        "window_mode": window_mode,
        "anchored_train_start": (
            None if anchored_train_start is None else _month_floor(anchored_train_start).isoformat()
        ),
        "calibration_fraction": calibration_fraction,
        "calibration_method": calibration_method,
        "edge_threshold": edge_threshold,
        "max_edge_to_bet": max_edge_to_bet,
        "market_vig": market_vig,
        "time_series_splits": time_series_splits,
        "starting_bankroll_units": float(starting_bankroll_units),
        "staking_mode": staking_mode,
        "flat_bet_size_units": float(flat_bet_size_units),
        "min_bet_size_units": float(min_bet_size_units),
        "max_bet_size_units": float(max_bet_size_units),
        "edge_scale_cap": float(edge_scale_cap),
        "kelly_fraction": float(kelly_fraction),
        "max_bet_fraction": float(max_bet_fraction),
        "max_drawdown": float(max_drawdown),
        "estimator_kwargs": resolved_estimator_kwargs,
        "raw_meta_feature_columns": list(raw_meta_feature_columns),
        "window_count": int(len(window_metrics)),
        "aggregate_brier_score": aggregate_brier,
        "aggregate_roi": aggregate_roi,
        "bankroll_return_pct": bankroll_return_pct,
        "ending_bankroll_units": ending_bankroll_units,
        "peak_bankroll_units": peak_bankroll_units,
        "max_drawdown_pct": max_drawdown_pct,
        "longest_losing_streak": longest_losing_streak,
        "total_bets": int(predictions["is_bet"].sum()),
        "total_profit_units": total_profit,
        "total_staked_units": total_staked,
        "data_version_hash": data_version_hash,
        "code_version_hash": code_version_hash,
        "predictions_path": str(predictions_path),
        "window_metrics_path": str(window_metrics_path),
        "window_builds": [asdict(window_build) for window_build in window_builds],
        "output_fingerprint": output_fingerprint,
    }
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return WalkForwardBacktestResult(
        predictions=predictions.reset_index(drop=True),
        window_metrics=window_metrics,
        predictions_path=predictions_path,
        window_metrics_path=window_metrics_path,
        summary_path=summary_path,
        output_fingerprint=output_fingerprint,
        aggregate_brier_score=aggregate_brier,
        aggregate_roi=aggregate_roi,
        bankroll_return_pct=bankroll_return_pct,
        ending_bankroll_units=ending_bankroll_units,
        peak_bankroll_units=peak_bankroll_units,
        max_drawdown_pct=max_drawdown_pct,
        longest_losing_streak=longest_losing_streak,
        total_bets=int(predictions["is_bet"].sum()),
        window_count=len(window_metrics),
        data_version_hash=data_version_hash,
        code_version_hash=code_version_hash,
        window_builds=tuple(window_builds),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for the monthly walk-forward backtest."""

    parser = argparse.ArgumentParser(description="Run walk-forward MLB F5 backtests")
    parser.add_argument("--start", dest="start_date", required=True)
    parser.add_argument("--end", dest="end_date", required=True)
    parser.add_argument("--training-data")
    parser.add_argument("--output-dir", default=str(DEFAULT_BACKTEST_OUTPUT_DIR))
    parser.add_argument("--cache-dir", default=str(DEFAULT_BACKTEST_CACHE_DIR))
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--train-window-months", type=int, default=DEFAULT_TRAIN_WINDOW_MONTHS)
    parser.add_argument("--test-window-months", type=int, default=DEFAULT_TEST_WINDOW_MONTHS)
    parser.add_argument("--window-mode", default=DEFAULT_WINDOW_MODE, choices=SUPPORTED_WINDOW_MODES)
    parser.add_argument("--anchored-train-start")
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=DEFAULT_CALIBRATION_FRACTION,
    )
    parser.add_argument(
        "--calibration-method",
        default=DEFAULT_WALK_FORWARD_CALIBRATION_METHOD,
        choices=SUPPORTED_CALIBRATION_METHODS,
    )
    parser.add_argument("--edge-threshold", type=float, default=DEFAULT_EDGE_THRESHOLD)
    parser.add_argument("--max-edge-to-bet", type=float)
    parser.add_argument("--market-vig", type=float, default=DEFAULT_MARKET_VIG)
    parser.add_argument("--time-series-splits", type=int, default=DEFAULT_TIME_SERIES_SPLITS)
    parser.add_argument("--max-depth", type=int, default=DEFAULT_ESTIMATOR_KWARGS["max_depth"])
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=DEFAULT_ESTIMATOR_KWARGS["n_estimators"],
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_ESTIMATOR_KWARGS["learning_rate"],
    )
    parser.add_argument("--subsample", type=float, default=DEFAULT_ESTIMATOR_KWARGS["subsample"])
    parser.add_argument(
        "--colsample-bytree",
        type=float,
        default=DEFAULT_ESTIMATOR_KWARGS["colsample_bytree"],
    )
    parser.add_argument("--min-child-weight", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--historical-odds-db")
    parser.add_argument("--historical-odds-book")
    parser.add_argument(
        "--starting-bankroll-units",
        type=float,
        default=DEFAULT_STARTING_BANKROLL_UNITS,
    )
    parser.add_argument("--staking-mode", default=DEFAULT_STAKING_MODE, choices=SUPPORTED_STAKING_MODES)
    parser.add_argument("--flat-bet-size-units", type=float, default=DEFAULT_FLAT_BET_SIZE_UNITS)
    parser.add_argument("--min-bet-size-units", type=float, default=DEFAULT_MIN_BET_SIZE_UNITS)
    parser.add_argument("--max-bet-size-units", type=float, default=DEFAULT_MAX_BET_SIZE_UNITS)
    parser.add_argument("--edge-scale-cap", type=float, default=DEFAULT_EDGE_SCALE_CAP)
    parser.add_argument("--kelly-fraction", type=float, default=DEFAULT_KELLY_FRACTION)
    parser.add_argument("--max-bet-fraction", type=float, default=MAX_BET_FRACTION)
    parser.add_argument("--max-drawdown", type=float, default=DEFAULT_MAX_DRAWDOWN)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _configure_cli_logging()

    result = run_walk_forward_backtest(
        training_data=args.training_data,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        refresh_data=args.refresh_data,
        seed=args.seed,
        train_window_months=args.train_window_months,
        test_window_months=args.test_window_months,
        window_mode=args.window_mode,
        anchored_train_start=args.anchored_train_start,
        calibration_fraction=args.calibration_fraction,
        calibration_method=args.calibration_method,
        edge_threshold=args.edge_threshold,
        max_edge_to_bet=args.max_edge_to_bet,
        market_vig=args.market_vig,
        time_series_splits=args.time_series_splits,
        estimator_kwargs={
            "max_depth": args.max_depth,
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "min_child_weight": args.min_child_weight,
            "gamma": args.gamma,
            "reg_alpha": args.reg_alpha,
            "reg_lambda": args.reg_lambda,
        },
        historical_odds_db_path=args.historical_odds_db,
        historical_odds_book_name=args.historical_odds_book,
        starting_bankroll_units=args.starting_bankroll_units,
        staking_mode=args.staking_mode,
        flat_bet_size_units=args.flat_bet_size_units,
        min_bet_size_units=args.min_bet_size_units,
        max_bet_size_units=args.max_bet_size_units,
        edge_scale_cap=args.edge_scale_cap,
        kelly_fraction=args.kelly_fraction,
        max_bet_fraction=args.max_bet_fraction,
        max_drawdown=args.max_drawdown,
    )
    log_walk_forward_run(
        result,
        output_dir=args.output_dir,
        training_data=args.training_data,
        start_date=args.start_date,
        end_date=args.end_date,
        train_window_months=args.train_window_months,
        test_window_months=args.test_window_months,
        window_mode=args.window_mode,
        anchored_train_start=args.anchored_train_start,
        calibration_method=args.calibration_method,
        calibration_fraction=args.calibration_fraction,
        edge_threshold=args.edge_threshold,
        max_edge_to_bet=args.max_edge_to_bet,
        market_vig=args.market_vig,
        historical_odds_db_path=args.historical_odds_db,
        historical_odds_book_name=args.historical_odds_book,
        starting_bankroll_units=args.starting_bankroll_units,
        staking_mode=args.staking_mode,
        flat_bet_size_units=args.flat_bet_size_units,
        min_bet_size_units=args.min_bet_size_units,
        max_bet_size_units=args.max_bet_size_units,
        edge_scale_cap=args.edge_scale_cap,
        kelly_fraction=args.kelly_fraction,
        max_bet_fraction=args.max_bet_fraction,
        max_drawdown=args.max_drawdown,
    )
    print(
        json.dumps(
            {
                "predictions_path": str(result.predictions_path),
                "window_metrics_path": str(result.window_metrics_path),
                "summary_path": str(result.summary_path),
                "aggregate_brier_score": result.aggregate_brier_score,
                "aggregate_roi": result.aggregate_roi,
                "bankroll_return_pct": result.bankroll_return_pct,
                "ending_bankroll_units": result.ending_bankroll_units,
                "peak_bankroll_units": result.peak_bankroll_units,
                "max_drawdown_pct": result.max_drawdown_pct,
                "longest_losing_streak": result.longest_losing_streak,
                "total_bets": result.total_bets,
                "window_count": result.window_count,
                "data_version_hash": result.data_version_hash,
                "code_version_hash": result.code_version_hash,
                "output_fingerprint": result.output_fingerprint,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _configure_cli_logging() -> None:
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def _evaluate_window(
    *,
    dataframe: pd.DataFrame,
    window: WalkForwardWindow,
    feature_columns: Sequence[str],
    raw_meta_feature_columns: Sequence[str],
    calibration_fraction: float,
    calibration_method: str,
    edge_threshold: float,
    max_edge_to_bet: float | None,
    market_vig: float,
    time_series_splits: int,
    estimator_kwargs: Mapping[str, Any],
    meta_learner_max_iter: int,
    seed: int,
    data_version_hash: str,
    code_version_hash: str,
    historical_odds_db_path: str | Path | None,
    historical_odds_book_name: str | None,
    bankroll_state: _BankrollState,
    staking_mode: Literal["flat", "kelly", "edge_scaled", "edge_bucketed"],
    flat_bet_size_units: float,
    min_bet_size_units: float,
    max_bet_size_units: float,
    edge_scale_cap: float,
    kelly_fraction: float,
    max_bet_fraction: float,
    max_drawdown: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    train_frame = _slice_time_range(dataframe, start=window.train_start, end=window.train_end)
    test_frame = _slice_time_range(dataframe, start=window.test_start, end=window.test_end)
    if train_frame.empty or test_frame.empty:
        raise ValueError(f"Window {window.window_index} does not contain both train and test rows")

    if pd.to_datetime(train_frame["scheduled_start"], utc=True).max() >= pd.to_datetime(
        test_frame["scheduled_start"],
        utc=True,
    ).min():
        raise AssertionError("Walk-forward split leaked test rows into the training period")

    model_bundle = _train_window_model(
        train_frame=train_frame,
        feature_columns=feature_columns,
        raw_meta_feature_columns=raw_meta_feature_columns,
        calibration_fraction=calibration_fraction,
        calibration_method=calibration_method,
        estimator_kwargs=estimator_kwargs,
        meta_learner_max_iter=meta_learner_max_iter,
        time_series_splits=time_series_splits,
        seed=seed,
    )

    model_home_prob = np.asarray(
        model_bundle.calibrated_model.predict_calibrated(test_frame),
        dtype=float,
    )
    (
        market_home_fair_prob,
        market_away_fair_prob,
        home_implied_prob,
        away_implied_prob,
        home_odds,
        away_odds,
        market_source,
    ) = _resolve_market_pricing(
        test_frame=test_frame,
        market_vig=market_vig,
        historical_odds_db_path=historical_odds_db_path,
        historical_odds_book_name=historical_odds_book_name,
    )
    edge_home = model_home_prob - market_home_fair_prob
    edge_away = (1.0 - model_home_prob) - market_away_fair_prob
    actual_home_win = pd.to_numeric(test_frame["f5_ml_result"], errors="raise").astype(int).to_numpy()
    is_push = pd.to_numeric(test_frame["f5_tied_after_5"], errors="coerce").fillna(0).astype(int).to_numpy().astype(bool)

    bet_side = np.where(edge_home >= edge_threshold, "home", "none")
    bet_side = np.where(edge_away >= edge_threshold, "away", bet_side)
    if max_edge_to_bet is not None:
        bet_side = np.where((bet_side == "home") & (edge_home > max_edge_to_bet), "none", bet_side)
        bet_side = np.where((bet_side == "away") & (edge_away > max_edge_to_bet), "none", bet_side)
    bet_edge = np.where(
        bet_side == "home",
        edge_home,
        np.where(bet_side == "away", edge_away, 0.0),
    )
    bet_odds = np.where(
        bet_side == "home",
        home_odds,
        np.where(bet_side == "away", away_odds, 0),
    )
    bet_model_prob = np.where(
        bet_side == "home",
        model_home_prob,
        np.where(bet_side == "away", 1.0 - model_home_prob, 0.0),
    )
    bet_expected_value = np.array(
        [
            0.0 if side == "none" else _expected_value(prob, odds)
            for side, prob, odds in zip(bet_side, bet_model_prob, bet_odds, strict=False)
        ],
        dtype=float,
    )

    bet_result: list[str] = []
    bet_profit_units: list[float] = []
    for side, push, outcome, odds in zip(bet_side, is_push, actual_home_win, bet_odds, strict=False):
        if side == "none":
            bet_result.append("NO_BET")
            bet_profit_units.append(0.0)
            continue
        if push:
            bet_result.append("PUSH")
            bet_profit_units.append(0.0)
            continue

        is_win = (side == "home" and outcome == 1) or (side == "away" and outcome == 0)
        if is_win:
            bet_result.append("WIN")
            bet_profit_units.append(_payout_for_american_odds(int(odds)))
        else:
            bet_result.append("LOSS")
            bet_profit_units.append(-1.0)

    predictions = pd.DataFrame(
        {
            "window_index": window.window_index,
            "train_start": window.train_start.isoformat(),
            "train_end": window.train_end.isoformat(),
            "test_start": window.test_start.isoformat(),
            "test_end": window.test_end.isoformat(),
            "game_pk": test_frame["game_pk"].to_numpy(),
            "scheduled_start": pd.to_datetime(test_frame["scheduled_start"], utc=True).astype(str),
            "as_of_timestamp": pd.to_datetime(test_frame["as_of_timestamp"], utc=True).astype(str),
            "home_team": test_frame["home_team"].astype(str).to_numpy(),
            "away_team": test_frame["away_team"].astype(str).to_numpy(),
            "actual_home_win": actual_home_win,
            "is_push": is_push.astype(int),
            "model_home_prob": model_home_prob,
            "market_home_fair_prob": market_home_fair_prob,
            "market_away_fair_prob": market_away_fair_prob,
            "market_home_implied_prob": home_implied_prob,
            "market_away_implied_prob": away_implied_prob,
            "home_odds": home_odds,
            "away_odds": away_odds,
            "market_source": market_source,
            "edge_home": edge_home,
            "edge_away": edge_away,
            "bet_side": bet_side,
            "bet_edge": bet_edge,
            "bet_odds": bet_odds,
            "bet_model_prob": bet_model_prob,
            "bet_expected_value": bet_expected_value,
            "bet_result": bet_result,
            "is_bet": (bet_side != "none").astype(int),
            "calibration_method": model_bundle.calibration_method,
            "data_version_hash": data_version_hash,
            "code_version_hash": code_version_hash,
        }
    )
    predictions = _apply_bankroll_strategy(
        predictions=predictions,
        starting_bankroll=bankroll_state.current_bankroll,
        peak_bankroll=bankroll_state.peak_bankroll,
        prior_longest_losing_streak=bankroll_state.longest_losing_streak,
        edge_threshold=edge_threshold,
        staking_mode=staking_mode,
        flat_bet_size_units=flat_bet_size_units,
        min_bet_size_units=min_bet_size_units,
        max_bet_size_units=max_bet_size_units,
        edge_scale_cap=edge_scale_cap,
        kelly_fraction=kelly_fraction,
        max_bet_fraction=max_bet_fraction,
        max_drawdown=max_drawdown,
    )

    scored_mask = predictions["is_push"] == 0
    brier_score = _safe_brier_score(
        predictions.loc[scored_mask, "actual_home_win"],
        predictions.loc[scored_mask, "model_home_prob"],
    )
    total_staked = float(predictions["bet_stake_units"].sum())
    total_profit = float(predictions["bet_profit_units"].sum())
    roi = float(total_profit / total_staked) if total_staked else 0.0

    metrics = {
        "window_index": window.window_index,
        "train_start": window.train_start.isoformat(),
        "train_end": window.train_end.isoformat(),
        "test_start": window.test_start.isoformat(),
        "test_end": window.test_end.isoformat(),
        "train_row_count": int(len(train_frame)),
        "model_training_row_count": int(model_bundle.model_training_row_count),
        "calibration_row_count": int(model_bundle.calibration_row_count),
        "test_row_count": int(len(test_frame)),
        "scored_test_row_count": int(scored_mask.sum()),
        "bet_count": int(predictions["is_bet"].sum()),
        "push_count": int(predictions["bet_result"].eq("PUSH").sum()),
        "win_count": int(predictions["bet_result"].eq("WIN").sum()),
        "loss_count": int(predictions["bet_result"].eq("LOSS").sum()),
        "brier_score": brier_score,
        "roi": roi,
        "total_profit_units": total_profit,
        "total_staked_units": total_staked,
        "max_edge_to_bet": max_edge_to_bet,
        "starting_bankroll_units": float(predictions["bankroll_before_units"].iloc[0]),
        "ending_bankroll_units": float(predictions["bankroll_after_units"].iloc[-1]),
        "peak_bankroll_units": float(predictions["peak_bankroll_units"].max()),
        "max_drawdown_pct": float(predictions["bankroll_drawdown_pct"].max()),
        "win_rate": float(
            predictions["bet_result"].eq("WIN").sum()
            / max(1, int(predictions["bet_result"].isin(["WIN", "LOSS"]).sum()))
        ),
        "average_stake_units": float(predictions.loc[predictions["is_bet"] == 1, "bet_stake_units"].mean())
        if int(predictions["is_bet"].sum()) > 0
        else 0.0,
        "longest_losing_streak": int(predictions["losing_streak"].max()),
        "staking_mode": staking_mode,
        "mean_model_home_prob": float(predictions["model_home_prob"].mean()),
        "mean_market_home_fair_prob": float(predictions["market_home_fair_prob"].mean()),
        "historical_odds_coverage": float((predictions["market_source"] == "historical").mean()),
        "calibration_method": model_bundle.calibration_method,
        "window_version_hash": _compute_window_version_hash(
            predictions["game_pk"].tolist(),
            data_version_hash=data_version_hash,
            code_version_hash=code_version_hash,
            seed=seed,
        ),
        "data_version_hash": data_version_hash,
        "code_version_hash": code_version_hash,
    }
    return predictions, metrics


def _resolve_market_pricing(
    *,
    test_frame: pd.DataFrame,
    market_vig: float,
    historical_odds_db_path: str | Path | None,
    historical_odds_book_name: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    default_home_fair = np.clip(
        pd.to_numeric(test_frame["home_team_log5_30g"], errors="coerce").to_numpy(dtype=float),
        0.02,
        0.98,
    )
    default_away_fair = 1.0 - default_home_fair
    default_home_implied = np.clip(default_home_fair * (1.0 + market_vig), 0.02, 0.99)
    default_away_implied = np.clip(default_away_fair * (1.0 + market_vig), 0.02, 0.99)
    default_home_odds = np.array(
        [_implied_probability_to_american(value) for value in default_home_implied],
        dtype=int,
    )
    default_away_odds = np.array(
        [_implied_probability_to_american(value) for value in default_away_implied],
        dtype=int,
    )
    market_source = np.full(len(test_frame), "synthetic", dtype=object)

    if historical_odds_db_path is None:
        return (
            default_home_fair,
            default_away_fair,
            default_home_implied,
            default_away_implied,
            default_home_odds,
            default_away_odds,
            market_source,
        )

    historical = load_historical_odds_for_games(
        db_path=historical_odds_db_path,
        game_pks=test_frame["game_pk"].astype(int).tolist(),
        market_type="f5_ml",
        book_name=historical_odds_book_name,
    )
    if historical.empty:
        return (
            default_home_fair,
            default_away_fair,
            default_home_implied,
            default_away_implied,
            default_home_odds,
            default_away_odds,
            market_source,
        )

    historical = historical.drop_duplicates(subset=["game_pk"], keep="last").set_index("game_pk")

    home_fair = default_home_fair.copy()
    away_fair = default_away_fair.copy()
    home_implied = default_home_implied.copy()
    away_implied = default_away_implied.copy()
    home_odds = default_home_odds.copy()
    away_odds = default_away_odds.copy()

    for index, game_pk in enumerate(test_frame["game_pk"].astype(int).tolist()):
        if game_pk not in historical.index:
            continue
        odds_row = historical.loc[game_pk]
        resolved_home_odds = int(odds_row["home_odds"])
        resolved_away_odds = int(odds_row["away_odds"])
        resolved_home_fair, resolved_away_fair = devig_probabilities(
            resolved_home_odds,
            resolved_away_odds,
        )
        home_odds[index] = resolved_home_odds
        away_odds[index] = resolved_away_odds
        home_implied[index] = float(
            abs(resolved_home_odds) / (abs(resolved_home_odds) + 100)
            if resolved_home_odds < 0
            else 100 / (resolved_home_odds + 100)
        )
        away_implied[index] = float(
            abs(resolved_away_odds) / (abs(resolved_away_odds) + 100)
            if resolved_away_odds < 0
            else 100 / (resolved_away_odds + 100)
        )
        home_fair[index] = resolved_home_fair
        away_fair[index] = resolved_away_fair
        market_source[index] = "historical"

    return home_fair, away_fair, home_implied, away_implied, home_odds, away_odds, market_source


def _train_window_model(
    *,
    train_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    raw_meta_feature_columns: Sequence[str],
    calibration_fraction: float,
    calibration_method: str,
    estimator_kwargs: Mapping[str, Any],
    meta_learner_max_iter: int,
    time_series_splits: int,
    seed: int,
) -> _WindowModelBundle:
    training_frame = train_frame.loc[
        pd.to_numeric(train_frame["f5_tied_after_5"], errors="coerce").fillna(0).astype(int) == 0
    ].copy()
    training_frame = training_frame.sort_values(["scheduled_start", "game_pk"]).reset_index(
        drop=True
    )
    target = pd.to_numeric(training_frame["f5_ml_result"], errors="raise").astype(int)

    if len(training_frame) < 30 or target.nunique() < 2:
        raise ValueError("Need at least 30 non-push rows and both classes to train a window")

    resolved_calibration_method = str(calibration_method).strip().lower()
    if resolved_calibration_method == "identity" or calibration_fraction <= 0:
        model_training_frame = training_frame.copy().reset_index(drop=True)
        calibration_frame = training_frame.iloc[0:0].copy()
    else:
        split_index = _resolve_calibration_split_index(len(training_frame), calibration_fraction)
        model_training_frame = training_frame.iloc[:split_index].copy().reset_index(drop=True)
        calibration_frame = training_frame.iloc[split_index:].copy().reset_index(drop=True)
    if model_training_frame.empty or pd.to_numeric(
        model_training_frame["f5_ml_result"],
        errors="raise",
    ).nunique() < 2:
        raise ValueError("Model-training segment needs both classes for walk-forward training")

    estimator = _build_estimator(seed=seed, estimator_kwargs=estimator_kwargs)
    oof_result = _generate_temporal_oof_probabilities(
        estimator=estimator,
        feature_frame=model_training_frame[list(feature_columns)],
        target=pd.to_numeric(model_training_frame["f5_ml_result"], errors="raise").astype(int),
        requested_splits=time_series_splits,
    )
    fitted_estimator = clone(estimator)
    fitted_estimator.fit(
        model_training_frame[list(feature_columns)],
        pd.to_numeric(model_training_frame["f5_ml_result"], errors="raise").astype(int),
    )

    oof_frame = model_training_frame.iloc[oof_result.warmup_row_count :].copy().reset_index(drop=True)
    oof_frame["xgb_probability"] = oof_result.probabilities.reset_index(drop=True)
    meta_feature_columns = ["xgb_probability", *raw_meta_feature_columns]
    meta_learner = LogisticRegression(max_iter=meta_learner_max_iter, random_state=seed)
    meta_learner.fit(
        oof_frame[meta_feature_columns],
        pd.to_numeric(oof_frame["f5_ml_result"], errors="raise").astype(int),
    )

    stacking_model = StackingEnsembleModel(
        model_name="walk_forward_f5_ml",
        target_column="f5_ml_result",
        base_estimator=fitted_estimator,
        meta_learner=meta_learner,
        base_feature_columns=list(feature_columns),
        raw_meta_feature_columns=list(raw_meta_feature_columns),
        meta_feature_columns=meta_feature_columns,
    )
    if calibration_frame.empty:
        calibrator = _fit_probability_calibrator(method="identity", probabilities=[], y_true=[])
    else:
        calibration_probabilities = np.asarray(
            stacking_model.predict_proba(calibration_frame)[:, 1],
            dtype=float,
        )
        calibrator = _fit_probability_calibrator(
            method=calibration_method,
            probabilities=calibration_probabilities,
            y_true=pd.to_numeric(calibration_frame["f5_ml_result"], errors="raise").astype(int),
        )
    calibrated_model = CalibratedStackingModel(
        model_name="walk_forward_f5_ml",
        target_column="f5_ml_result",
        stacking_model=stacking_model,
        calibrator=calibrator,
    )
    return _WindowModelBundle(
        calibrated_model=calibrated_model,
        model_training_row_count=len(model_training_frame),
        calibration_row_count=len(calibration_frame),
        calibration_method=str(getattr(calibrator, "method", calibration_method)),
    )


def _create_requested_windows(
    *,
    start_date: str,
    end_date: str,
    train_window_months: int,
    test_window_months: int,
    window_mode: Literal["rolling", "anchored_expanding"],
    anchored_train_start: str | pd.Timestamp | None,
) -> list[WalkForwardWindow]:
    start_timestamp = _month_floor(start_date)
    end_timestamp = _month_floor(end_date)
    anchored_start = (
        _month_floor(anchored_train_start)
        if anchored_train_start is not None
        else start_timestamp - pd.DateOffset(months=train_window_months)
    )

    windows: list[WalkForwardWindow] = []
    window_index = 1
    test_start = start_timestamp
    while test_start <= end_timestamp:
        test_end = test_start + pd.DateOffset(months=test_window_months)
        windows.append(
            WalkForwardWindow(
                window_index=window_index,
                train_start=(
                    anchored_start
                    if window_mode == "anchored_expanding"
                    else test_start - pd.DateOffset(months=train_window_months)
                ),
                train_end=test_start,
                test_start=test_start,
                test_end=test_end,
            )
        )
        test_start = test_end
        window_index += 1

    return windows


def _resolve_window_training_data(
    *,
    window: WalkForwardWindow,
    cache_dir: str | Path,
    refresh_data: bool,
    weather_fetcher: WeatherFetcher,
) -> tuple[pd.DataFrame, WalkForwardWindowBuild]:
    cache_path = Path(cache_dir) / (
        f"walk_forward_cache_{window.train_start.strftime('%Y%m%d')}_{window.test_end.strftime('%Y%m%d')}.parquet"
    )
    cutoff_timestamp = pd.Timestamp(window.test_end).tz_convert("UTC")
    build_action = "rebuilt" if refresh_data or not cache_path.exists() else "cached"

    if build_action == "rebuilt":
        build_training_dataset(
            start_year=window.train_start.year,
            end_year=(window.test_end - pd.Timedelta(microseconds=1)).year,
            output_path=cache_path,
            full_regular_seasons_target=(window.test_end.year - window.train_start.year + 1),
            scheduled_start_before=cutoff_timestamp.isoformat(),
            refresh=refresh_data,
            weather_fetcher=weather_fetcher,
        )

    dataframe = _load_backtest_dataframe(cache_path)
    assert_training_data_is_leakage_free(dataframe)
    train_row_count = int(len(_slice_time_range(dataframe, start=window.train_start, end=window.train_end)))
    test_row_count = int(len(_slice_time_range(dataframe, start=window.test_start, end=window.test_end)))
    build = WalkForwardWindowBuild(
        window_index=window.window_index,
        data_source="window_rebuild",
        build_action=build_action,
        scheduled_start_before=cutoff_timestamp.isoformat(),
        cache_path=str(cache_path),
        row_count=int(len(dataframe)),
        train_row_count=train_row_count,
        test_row_count=test_row_count,
        data_version_hash=_resolve_data_version_hash(dataframe),
    )
    logger.info(
        "%s feature data for walk-forward window %s with cutoff < %s (%s total rows, %s train, %s test)",
        "Rebuilt" if build_action == "rebuilt" else "Loaded cached",
        window.window_index,
        build.scheduled_start_before,
        build.row_count,
        build.train_row_count,
        build.test_row_count,
    )
    return dataframe, build


def _load_backtest_dataframe(training_data: pd.DataFrame | str | Path) -> pd.DataFrame:
    dataframe = _load_training_dataframe(training_data)
    dataframe = dataframe.sort_values(["scheduled_start", "game_pk"]).reset_index(drop=True)
    return dataframe


def _month_floor(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.normalize().replace(day=1)


def _slice_time_range(
    dataframe: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    scheduled_start = pd.to_datetime(dataframe["scheduled_start"], utc=True)
    mask = (scheduled_start >= start) & (scheduled_start < end)
    return dataframe.loc[mask].copy().reset_index(drop=True)


def _resolve_data_version_hash(dataframe: pd.DataFrame) -> str:
    if "data_version_hash" in dataframe.columns:
        hashes = dataframe["data_version_hash"].dropna().astype(str).unique().tolist()
        if len(hashes) == 1:
            return hashes[0]
    return _compute_data_version_hash(dataframe)


def _combine_window_data_version_hashes(window_builds: Sequence[WalkForwardWindowBuild]) -> str:
    unique_hashes = {window_build.data_version_hash for window_build in window_builds}
    if len(unique_hashes) == 1:
        return next(iter(unique_hashes))

    payload = json.dumps(
        [
            {
                "window_index": window_build.window_index,
                "scheduled_start_before": window_build.scheduled_start_before,
                "data_version_hash": window_build.data_version_hash,
            }
            for window_build in window_builds
        ],
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _window_build_to_metric_fields(window_build: WalkForwardWindowBuild) -> dict[str, Any]:
    return {
        "feature_data_source": window_build.data_source,
        "feature_data_action": window_build.build_action,
        "feature_data_cutoff": window_build.scheduled_start_before,
        "feature_data_path": window_build.cache_path,
        "feature_data_row_count": window_build.row_count,
        "feature_data_train_row_count": window_build.train_row_count,
        "feature_data_test_row_count": window_build.test_row_count,
        "feature_data_version_hash": window_build.data_version_hash,
    }


def _compute_code_version_hash() -> str:
    digest = hashlib.sha256()
    for file_path in _VERSION_HASH_PATHS:
        digest.update(file_path.as_posix().encode("utf-8"))
        digest.update(file_path.read_bytes())
    return digest.hexdigest()


def _compute_window_version_hash(
    game_pks: Sequence[int],
    *,
    data_version_hash: str,
    code_version_hash: str,
    seed: int,
) -> str:
    payload = json.dumps(
        {
            "game_pks": list(game_pks),
            "data_version_hash": data_version_hash,
            "code_version_hash": code_version_hash,
            "seed": seed,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _apply_bankroll_strategy(
    *,
    predictions: pd.DataFrame,
    starting_bankroll: float,
    peak_bankroll: float,
    prior_longest_losing_streak: int,
    edge_threshold: float,
    staking_mode: Literal["flat", "kelly", "edge_scaled", "edge_bucketed"],
    flat_bet_size_units: float,
    min_bet_size_units: float,
    max_bet_size_units: float,
    edge_scale_cap: float,
    kelly_fraction: float,
    max_bet_fraction: float,
    max_drawdown: float,
) -> pd.DataFrame:
    if staking_mode not in SUPPORTED_STAKING_MODES:
        raise ValueError(f"staking_mode must be one of {SUPPORTED_STAKING_MODES}")
    if flat_bet_size_units < 0:
        raise ValueError("flat_bet_size_units must be non-negative")
    if min_bet_size_units < 0 or max_bet_size_units < 0:
        raise ValueError("min_bet_size_units and max_bet_size_units must be non-negative")
    if min_bet_size_units > max_bet_size_units:
        raise ValueError("min_bet_size_units cannot exceed max_bet_size_units")
    if edge_scale_cap <= 0:
        raise ValueError("edge_scale_cap must be positive")

    enriched = predictions.copy()
    current_bankroll = float(starting_bankroll)
    running_peak = max(float(peak_bankroll), current_bankroll)
    current_losing_streak = 0
    longest_losing_streak = int(prior_longest_losing_streak)

    bankroll_before: list[float] = []
    bankroll_after: list[float] = []
    peak_bankrolls: list[float] = []
    drawdown_pcts: list[float] = []
    stake_units: list[float] = []
    stake_fractions: list[float] = []
    full_kelly_fractions: list[float] = []
    kill_switch_flags: list[int] = []
    losing_streaks: list[int] = []
    profit_units: list[float] = []

    for row in enriched.itertuples(index=False):
        bankroll_before.append(current_bankroll)
        kill_switch_active = int(_calculate_drawdown_pct(current_bankroll, running_peak) >= max_drawdown)
        stake = 0.0
        stake_fraction = 0.0
        full_kelly_fraction = 0.0

        if row.bet_side != "none" and current_bankroll > 0 and not kill_switch_active:
            if staking_mode == "flat":
                stake = min(float(flat_bet_size_units), current_bankroll)
                stake_fraction = float(stake / current_bankroll) if current_bankroll else 0.0
            elif staking_mode == "edge_scaled":
                scaled_units = _edge_scaled_units(
                    edge=float(row.bet_edge),
                    edge_threshold=edge_threshold,
                    min_units=min_bet_size_units,
                    max_units=max_bet_size_units,
                    edge_scale_cap=edge_scale_cap,
                )
                stake = min(float(scaled_units), current_bankroll)
                stake_fraction = float(stake / current_bankroll) if current_bankroll else 0.0
            elif staking_mode == "edge_bucketed":
                bucketed_units = _edge_bucketed_units(float(row.bet_edge))
                stake = min(float(bucketed_units), current_bankroll)
                stake_fraction = float(stake / current_bankroll) if current_bankroll else 0.0
            else:
                sizing = calculate_kelly_stake(
                    bankroll=current_bankroll,
                    model_probability=float(row.bet_model_prob),
                    odds=int(row.bet_odds),
                    peak_bankroll=running_peak,
                    fraction=kelly_fraction,
                    max_bet_fraction=max_bet_fraction,
                    max_drawdown=max_drawdown,
                )
                stake = min(float(sizing.stake), current_bankroll)
                stake_fraction = float(sizing.stake_fraction)
                full_kelly_fraction = float(sizing.full_kelly_fraction)
                kill_switch_active = int(sizing.kill_switch_active)

        if row.bet_result == "WIN" and stake > 0:
            profit = _payout_for_american_odds(int(row.bet_odds)) * stake
        elif row.bet_result == "LOSS" and stake > 0:
            profit = -stake
        else:
            profit = 0.0

        current_bankroll = max(0.0, current_bankroll + profit)
        running_peak = max(running_peak, current_bankroll)

        if row.bet_result == "LOSS" and stake > 0:
            current_losing_streak += 1
            longest_losing_streak = max(longest_losing_streak, current_losing_streak)
        elif row.bet_result in {"WIN", "PUSH"} or row.bet_side == "none":
            current_losing_streak = 0

        stake_units.append(stake)
        stake_fractions.append(stake_fraction)
        full_kelly_fractions.append(full_kelly_fraction)
        profit_units.append(profit)
        bankroll_after.append(current_bankroll)
        peak_bankrolls.append(running_peak)
        drawdown_pcts.append(_calculate_drawdown_pct(current_bankroll, running_peak))
        kill_switch_flags.append(kill_switch_active)
        losing_streaks.append(current_losing_streak)

    enriched["bet_stake_units"] = np.asarray(stake_units, dtype=float)
    enriched["bet_profit_units"] = np.asarray(profit_units, dtype=float)
    enriched["bankroll_before_units"] = np.asarray(bankroll_before, dtype=float)
    enriched["bankroll_after_units"] = np.asarray(bankroll_after, dtype=float)
    enriched["peak_bankroll_units"] = np.asarray(peak_bankrolls, dtype=float)
    enriched["bankroll_drawdown_pct"] = np.asarray(drawdown_pcts, dtype=float)
    enriched["bet_stake_fraction"] = np.asarray(stake_fractions, dtype=float)
    enriched["bet_full_kelly_fraction"] = np.asarray(full_kelly_fractions, dtype=float)
    enriched["kill_switch_active"] = np.asarray(kill_switch_flags, dtype=int)
    enriched["losing_streak"] = np.asarray(losing_streaks, dtype=int)
    enriched["staking_mode"] = staking_mode
    enriched["longest_losing_streak"] = int(longest_losing_streak)
    return enriched


def _calculate_drawdown_pct(current_bankroll: float, peak_bankroll: float) -> float:
    if peak_bankroll <= 0:
        return 0.0
    return float((peak_bankroll - current_bankroll) / peak_bankroll)


def _edge_scaled_units(
    *,
    edge: float,
    edge_threshold: float,
    min_units: float,
    max_units: float,
    edge_scale_cap: float,
) -> float:
    effective_edge = max(0.0, float(edge) - float(edge_threshold))
    scale = min(1.0, effective_edge / float(edge_scale_cap))
    return float(min_units + ((max_units - min_units) * scale))


def _edge_bucketed_units(edge: float) -> float:
    resolved_edge = float(edge)
    if resolved_edge >= 0.30:
        return 0.5
    if resolved_edge >= 0.20:
        return 2.0
    if resolved_edge >= 0.15:
        return 1.5
    if resolved_edge >= 0.12:
        return 1.0
    return 0.5


def _resolve_calibration_split_index(row_count: int, calibration_fraction: float) -> int:
    if calibration_fraction <= 0 or calibration_fraction >= 1:
        raise ValueError("calibration_fraction must be between 0 and 1")

    calibration_row_count = min(max(1, ceil(row_count * calibration_fraction)), row_count - 1)
    return row_count - calibration_row_count


def _build_estimator(*, seed: int, estimator_kwargs: Mapping[str, Any]) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        tree_method="hist",
        n_jobs=DEFAULT_XGBOOST_N_JOBS,
        verbosity=0,
        **dict(estimator_kwargs),
    )


def _implied_probability_to_american(probability: float) -> int:
    clipped_probability = min(max(float(probability), 1e-6), 1.0 - 1e-6)
    if clipped_probability >= 0.5:
        return int(round(-100.0 * clipped_probability / (1.0 - clipped_probability)))
    return int(round(100.0 * (1.0 - clipped_probability) / clipped_probability))


def _payout_for_american_odds(odds: int) -> float:
    if odds >= 100:
        return float(odds) / 100.0
    if odds <= -100:
        return 100.0 / abs(float(odds))
    raise ValueError("American odds must be <= -100 or >= 100")


def _expected_value(model_probability: float, odds: int) -> float:
    profit = _payout_for_american_odds(int(odds))
    return float((model_probability * profit) - ((1.0 - model_probability) * 1.0))


def _safe_brier_score(
    y_true: Sequence[int] | pd.Series,
    probabilities: Sequence[float] | pd.Series,
) -> float:
    if len(y_true) == 0:
        return 0.0
    return float(brier_score_loss(y_true, probabilities))


def _write_csv(dataframe: pd.DataFrame, output_path: Path) -> None:
    dataframe.to_csv(
        output_path,
        index=False,
        float_format=DEFAULT_FLOAT_FORMAT,
        lineterminator="\n",
    )


def _set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    raise SystemExit(main())
