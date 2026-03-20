from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from dataclasses import asdict, dataclass
from math import ceil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from xgboost import XGBClassifier

from src.clients.weather_client import fetch_game_weather
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
    _load_training_dataframe,
    _resolve_numeric_feature_columns,
)


logger = logging.getLogger(__name__)


DEFAULT_BACKTEST_OUTPUT_DIR = Path("data") / "backtest"
DEFAULT_BACKTEST_CACHE_DIR = Path("data") / "training"
DEFAULT_TRAIN_WINDOW_MONTHS = 6
DEFAULT_TEST_WINDOW_MONTHS = 1
DEFAULT_CALIBRATION_FRACTION = 0.15
DEFAULT_WALK_FORWARD_CALIBRATION_METHOD = DEFAULT_CALIBRATION_METHOD
DEFAULT_EDGE_THRESHOLD = 0.03
DEFAULT_MARKET_VIG = 0.04
DEFAULT_TIME_SERIES_SPLITS = 3
DEFAULT_FLOAT_FORMAT = "%.10f"
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
class WalkForwardBacktestResult:
    predictions: pd.DataFrame
    window_metrics: pd.DataFrame
    predictions_path: Path
    window_metrics_path: Path
    summary_path: Path
    output_fingerprint: str
    aggregate_brier_score: float
    aggregate_roi: float
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
) -> list[WalkForwardWindow]:
    """Create monthly walk-forward windows with a 6-month train and 1-month test stride."""

    if train_window_months < 1:
        raise ValueError("train_window_months must be at least 1")
    if test_window_months < 1:
        raise ValueError("test_window_months must be at least 1")

    dataframe = _load_backtest_dataframe(training_data)
    requested_start = _month_floor(start_date)
    requested_end = _month_floor(end_date)

    windows: list[WalkForwardWindow] = []
    test_start = requested_start
    window_index = 1
    while test_start <= requested_end:
        test_end = test_start + pd.DateOffset(months=test_window_months)
        train_start = test_start - pd.DateOffset(months=train_window_months)
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
    calibration_fraction: float = DEFAULT_CALIBRATION_FRACTION,
    calibration_method: str = DEFAULT_WALK_FORWARD_CALIBRATION_METHOD,
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
    market_vig: float = DEFAULT_MARKET_VIG,
    time_series_splits: int = DEFAULT_TIME_SERIES_SPLITS,
    raw_meta_feature_columns: Sequence[str] = DEFAULT_RAW_META_FEATURE_COLUMNS,
    estimator_kwargs: Mapping[str, Any] | None = None,
    meta_learner_max_iter: int = DEFAULT_META_LEARNER_MAX_ITER,
    weather_fetcher: WeatherFetcher = fetch_game_weather,
) -> WalkForwardBacktestResult:
    """Run a deterministic walk-forward backtest and persist predictions plus window metrics."""

    _set_random_seed(seed)
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
        )
        if not windows:
            raise ValueError("No walk-forward windows matched the requested date range")

        feature_columns = _resolve_numeric_feature_columns(dataframe)
        resolved_raw_meta_feature_columns = _resolve_raw_meta_feature_columns(
            dataframe,
            requested_columns=raw_meta_feature_columns,
        )
        data_version_hash = _resolve_data_version_hash(dataframe)
        cache_path = str(Path(training_data)) if isinstance(training_data, str | Path) else None

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
                market_vig=market_vig,
                time_series_splits=time_series_splits,
                estimator_kwargs=resolved_estimator_kwargs,
                meta_learner_max_iter=meta_learner_max_iter,
                seed=seed,
                data_version_hash=data_version_hash,
                code_version_hash=code_version_hash,
            )
            metrics.update(_window_build_to_metric_fields(build))
            prediction_frames.append(predictions)
            window_metric_rows.append(metrics)
            window_builds.append(build)
    else:
        windows = _create_requested_windows(
            start_date=start_date,
            end_date=end_date,
            train_window_months=train_window_months,
            test_window_months=test_window_months,
        )

        for window in windows:
            dataframe, build = _resolve_window_training_data(
                window=window,
                cache_dir=cache_dir,
                refresh_data=refresh_data,
                weather_fetcher=weather_fetcher,
            )
            if build.train_row_count == 0 or build.test_row_count == 0:
                logger.info(
                    "Skipping walk-forward window %s because rebuilt feature data yielded %s train rows and %s test rows",
                    window.window_index,
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
                market_vig=market_vig,
                time_series_splits=time_series_splits,
                estimator_kwargs=resolved_estimator_kwargs,
                meta_learner_max_iter=meta_learner_max_iter,
                seed=seed,
                data_version_hash=build.data_version_hash,
                code_version_hash=code_version_hash,
            )
            metrics.update(_window_build_to_metric_fields(build))
            prediction_frames.append(predictions)
            window_metric_rows.append(metrics)
            window_builds.append(build)

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
        "calibration_fraction": calibration_fraction,
        "calibration_method": calibration_method,
        "edge_threshold": edge_threshold,
        "market_vig": market_vig,
        "time_series_splits": time_series_splits,
        "estimator_kwargs": resolved_estimator_kwargs,
        "raw_meta_feature_columns": list(raw_meta_feature_columns),
        "window_count": int(len(window_metrics)),
        "aggregate_brier_score": aggregate_brier,
        "aggregate_roi": aggregate_roi,
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
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

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
        calibration_fraction=args.calibration_fraction,
        calibration_method=args.calibration_method,
        edge_threshold=args.edge_threshold,
        market_vig=args.market_vig,
        time_series_splits=args.time_series_splits,
        estimator_kwargs={
            "max_depth": args.max_depth,
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
        },
    )
    print(
        json.dumps(
            {
                "predictions_path": str(result.predictions_path),
                "window_metrics_path": str(result.window_metrics_path),
                "summary_path": str(result.summary_path),
                "aggregate_brier_score": result.aggregate_brier_score,
                "aggregate_roi": result.aggregate_roi,
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


def _evaluate_window(
    *,
    dataframe: pd.DataFrame,
    window: WalkForwardWindow,
    feature_columns: Sequence[str],
    raw_meta_feature_columns: Sequence[str],
    calibration_fraction: float,
    calibration_method: str,
    edge_threshold: float,
    market_vig: float,
    time_series_splits: int,
    estimator_kwargs: Mapping[str, Any],
    meta_learner_max_iter: int,
    seed: int,
    data_version_hash: str,
    code_version_hash: str,
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
    market_home_fair_prob = np.clip(
        pd.to_numeric(test_frame["home_team_log5_30g"], errors="coerce").to_numpy(dtype=float),
        0.02,
        0.98,
    )
    market_away_fair_prob = 1.0 - market_home_fair_prob
    home_implied_prob = np.clip(market_home_fair_prob * (1.0 + market_vig), 0.02, 0.99)
    away_implied_prob = np.clip(market_away_fair_prob * (1.0 + market_vig), 0.02, 0.99)
    home_odds = np.array([_implied_probability_to_american(value) for value in home_implied_prob])
    away_odds = np.array([_implied_probability_to_american(value) for value in away_implied_prob])
    edge_home = model_home_prob - market_home_fair_prob
    edge_away = (1.0 - model_home_prob) - market_away_fair_prob
    actual_home_win = pd.to_numeric(test_frame["f5_ml_result"], errors="raise").astype(int).to_numpy()
    is_push = pd.to_numeric(test_frame["f5_tied_after_5"], errors="coerce").fillna(0).astype(int).to_numpy().astype(bool)

    bet_side = np.where(edge_home >= edge_threshold, "home", "none")
    bet_side = np.where(edge_away >= edge_threshold, "away", bet_side)
    bet_stake_units = np.where(bet_side == "none", 0.0, 1.0)
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
            "edge_home": edge_home,
            "edge_away": edge_away,
            "bet_side": bet_side,
            "bet_edge": bet_edge,
            "bet_odds": bet_odds,
            "bet_model_prob": bet_model_prob,
            "bet_expected_value": bet_expected_value,
            "bet_stake_units": bet_stake_units,
            "bet_profit_units": bet_profit_units,
            "bet_result": bet_result,
            "is_bet": (bet_side != "none").astype(int),
            "calibration_method": model_bundle.calibration_method,
            "data_version_hash": data_version_hash,
            "code_version_hash": code_version_hash,
        }
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
        "mean_model_home_prob": float(predictions["model_home_prob"].mean()),
        "mean_market_home_fair_prob": float(predictions["market_home_fair_prob"].mean()),
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
) -> list[WalkForwardWindow]:
    start_timestamp = _month_floor(start_date)
    end_timestamp = _month_floor(end_date)

    windows: list[WalkForwardWindow] = []
    window_index = 1
    test_start = start_timestamp
    while test_start <= end_timestamp:
        test_end = test_start + pd.DateOffset(months=test_window_months)
        windows.append(
            WalkForwardWindow(
                window_index=window_index,
                train_start=test_start - pd.DateOffset(months=train_window_months),
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
        n_jobs=1,
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
