from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

from src.clients.weather_client import fetch_game_weather
from src.model.data_builder import build_training_dataset
from src.model.direct_rl_trainer import (
    DEFAULT_DIRECT_RL_MARKET_COLUMNS,
    _augment_direct_rl_training_frame,
)
from src.model.margin_pricing import margin_to_cover_probability
from src.model.xgboost_trainer import (
    DEFAULT_MODEL_OUTPUT_DIR,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_RANDOM_STATE,
    DEFAULT_SEARCH_ITERATIONS,
    DEFAULT_SEARCH_SPACE,
    DEFAULT_TIME_SERIES_SPLITS,
    DEFAULT_TOP_FEATURE_COUNT,
    DEFAULT_XGBOOST_N_JOBS,
    _build_model_version,
    _extract_feature_importance_rankings,
    _load_training_dataframe,
    _normalize_best_params,
    _resolve_data_version_hash,
    _resolve_experiment_output_dir,
    _resolve_holdout_season,
    _resolve_numeric_feature_columns,
    _resolve_search_iterations,
    _safe_roc_auc,
    create_time_series_split,
)


logger = logging.getLogger(__name__)

DEFAULT_MARGIN_MODEL_NAME = "f5_margin_v2_model"
DEFAULT_MARGIN_TARGET_COLUMN = "f5_margin"


@dataclass(frozen=True, slots=True)
class MarginTrainingArtifact:
    model_name: str
    target_column: str
    model_version: str
    model_path: Path
    metadata_path: Path
    best_params: dict[str, float | int]
    cv_best_rmse: float
    holdout_metrics: dict[str, float | None]
    feature_columns: list[str]
    feature_importance_rankings: list[dict[str, float | str]]
    train_row_count: int
    holdout_row_count: int
    holdout_priced_row_count: int
    holdout_season: int
    residual_std: float
    include_market_features: bool
    market_book_name: str | None


@dataclass(frozen=True, slots=True)
class MarginTrainingResult:
    model_version: str
    data_version_hash: str
    holdout_season: int
    summary_path: Path
    artifact: MarginTrainingArtifact


def train_margin_model(
    *,
    training_data: pd.DataFrame | str | Path,
    output_dir: str | Path = DEFAULT_MODEL_OUTPUT_DIR,
    holdout_season: int | None = None,
    search_space: Mapping[str, Sequence[float | int]] = DEFAULT_SEARCH_SPACE,
    time_series_splits: int = DEFAULT_TIME_SERIES_SPLITS,
    search_iterations: int = DEFAULT_SEARCH_ITERATIONS,
    random_state: int = DEFAULT_RANDOM_STATE,
    top_feature_count: int = DEFAULT_TOP_FEATURE_COUNT,
    target_column: str = DEFAULT_MARGIN_TARGET_COLUMN,
    include_market_features: bool = False,
    market_book_name: str | None = None,
) -> MarginTrainingResult:
    dataset = _augment_direct_rl_training_frame(_load_training_dataframe(training_data))
    frame = _prepare_margin_frame(dataset, target_column=target_column)
    if frame.empty:
        raise ValueError("No margin rows are available for margin training")

    effective_holdout_season = _resolve_holdout_season(frame, holdout_season)
    feature_columns = _resolve_margin_feature_columns(
        frame,
        include_market_features=include_market_features,
        target_column=target_column,
    )
    if not feature_columns:
        raise ValueError("Margin training data does not contain any usable numeric features")

    train_frame = frame.loc[frame["season"] < effective_holdout_season].copy()
    holdout_frame = frame.loc[frame["season"] == effective_holdout_season].copy()
    if train_frame.empty:
        raise ValueError(f"No margin training rows found before holdout season {effective_holdout_season}")
    if holdout_frame.empty:
        raise ValueError(f"No margin holdout rows found for season {effective_holdout_season}")

    data_version_hash = _resolve_data_version_hash(dataset)
    model_version = _build_model_version(data_version_hash)
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    search = RandomizedSearchCV(
        estimator=_build_estimator(random_state=random_state),
        param_distributions={key: list(values) for key, values in search_space.items()},
        n_iter=_resolve_search_iterations(search_space, search_iterations),
        scoring="neg_root_mean_squared_error",
        cv=create_time_series_split(
            row_count=len(train_frame),
            requested_splits=time_series_splits,
        ),
        random_state=random_state,
        refit=True,
        n_jobs=1,
    )
    search.fit(train_frame[feature_columns], train_frame[target_column])
    best_estimator: XGBRegressor = search.best_estimator_

    train_predictions = best_estimator.predict(train_frame[feature_columns])
    residual_std = float((train_frame[target_column] - train_predictions).std(ddof=1) or 0.0)
    holdout_predictions = best_estimator.predict(holdout_frame[feature_columns])
    holdout_metrics = _compute_margin_holdout_metrics(
        holdout_frame=holdout_frame,
        holdout_predictions=holdout_predictions,
        residual_std=residual_std,
    )

    best_params = _normalize_best_params(search.best_params_)
    cv_best_rmse = float(-search.best_score_)
    feature_importance_rankings = _extract_feature_importance_rankings(
        best_estimator,
        feature_columns,
        top_feature_count=top_feature_count,
    )

    model_path = resolved_output_dir / f"{DEFAULT_MARGIN_MODEL_NAME}_{model_version}.joblib"
    joblib.dump(best_estimator, model_path)
    metadata_path = model_path.with_suffix(".metadata.json")
    metadata_payload = {
        "model_name": DEFAULT_MARGIN_MODEL_NAME,
        "target_column": target_column,
        "model_version": model_version,
        "data_version_hash": data_version_hash,
        "holdout_season": effective_holdout_season,
        "market_book_name": market_book_name,
        "include_market_features": include_market_features,
        "train_row_count": int(len(train_frame)),
        "holdout_row_count": int(len(holdout_frame)),
        "holdout_priced_row_count": int(holdout_metrics["priced_row_count"]),
        "feature_columns": feature_columns,
        "best_params": best_params,
        "cv_best_rmse": cv_best_rmse,
        "holdout_metrics": holdout_metrics,
        "feature_importance_rankings": feature_importance_rankings,
        "search_space": {key: list(values) for key, values in search_space.items()},
        "time_series_splits": int(search.cv.n_splits),
        "residual_std": residual_std,
        "trained_at": datetime.now(UTC).isoformat(),
    }
    metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")

    artifact = MarginTrainingArtifact(
        model_name=DEFAULT_MARGIN_MODEL_NAME,
        target_column=target_column,
        model_version=model_version,
        model_path=model_path,
        metadata_path=metadata_path,
        best_params=best_params,
        cv_best_rmse=cv_best_rmse,
        holdout_metrics=holdout_metrics,
        feature_columns=feature_columns,
        feature_importance_rankings=feature_importance_rankings,
        train_row_count=len(train_frame),
        holdout_row_count=len(holdout_frame),
        holdout_priced_row_count=int(holdout_metrics["priced_row_count"]),
        holdout_season=effective_holdout_season,
        residual_std=residual_std,
        include_market_features=include_market_features,
        market_book_name=market_book_name,
    )

    summary_path = resolved_output_dir / f"margin_training_run_{model_version}.json"
    summary_path.write_text(
        json.dumps(
            {
                "model_version": model_version,
                "data_version_hash": data_version_hash,
                "holdout_season": effective_holdout_season,
                "artifact": _artifact_to_json_ready(artifact),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return MarginTrainingResult(
        model_version=model_version,
        data_version_hash=data_version_hash,
        holdout_season=effective_holdout_season,
        summary_path=summary_path,
        artifact=artifact,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train an F5 margin model for RL v2 pricing")
    parser.add_argument("--training-data", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_MODEL_OUTPUT_DIR))
    parser.add_argument("--experiment-name")
    parser.add_argument("--holdout-season", type=int, default=2025)
    parser.add_argument("--start-year", type=int, default=2018)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--refresh-training-data", action="store_true")
    parser.add_argument("--allow-backfill-years", action="store_true")
    parser.add_argument("--historical-odds-db")
    parser.add_argument("--historical-rl-book")
    parser.add_argument("--include-market-features", action="store_true")
    parser.add_argument("--time-series-splits", type=int, default=DEFAULT_TIME_SERIES_SPLITS)
    parser.add_argument("--search-iterations", type=int, default=DEFAULT_SEARCH_ITERATIONS)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    training_path = Path(args.training_data)
    if args.refresh_training_data or not training_path.exists():
        if not args.historical_odds_db:
            raise ValueError("--historical-odds-db is required when rebuilding margin training data")
        build_training_dataset(
            start_year=args.start_year,
            end_year=args.end_year,
            output_path=training_path,
            allow_backfill_years=args.allow_backfill_years,
            refresh=args.refresh_training_data,
            weather_fetcher=fetch_game_weather,
            historical_odds_db_path=args.historical_odds_db,
            historical_rl_book_name=args.historical_rl_book,
        )

    resolved_output_dir = _resolve_experiment_output_dir(args.output_dir, args.experiment_name)
    result = train_margin_model(
        training_data=training_path,
        output_dir=resolved_output_dir,
        holdout_season=args.holdout_season,
        time_series_splits=args.time_series_splits,
        search_iterations=args.search_iterations,
        random_state=args.random_state,
        include_market_features=args.include_market_features,
        market_book_name=args.historical_rl_book,
    )
    print(json.dumps(_run_result_to_json_ready(result), indent=2))
    return 0


def _prepare_margin_frame(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
) -> pd.DataFrame:
    frame = dataframe.copy()
    frame[target_column] = pd.to_numeric(frame[target_column], errors="coerce")
    frame = frame.loc[frame[target_column].notna()].copy()
    return frame.sort_values(["scheduled_start", "game_pk"]).reset_index(drop=True)


def _resolve_margin_feature_columns(
    dataframe: pd.DataFrame,
    *,
    include_market_features: bool,
    target_column: str,
) -> list[str]:
    feature_columns = _resolve_numeric_feature_columns(dataframe)
    blocked = {
        target_column,
        "home_cover_at_posted_line",
        "away_cover_at_posted_line",
        "push_at_posted_line",
        "f5_ml_result",
        "f5_rl_result",
        "f5_home_score",
        "f5_away_score",
        "f5_tied_after_5",
    }
    if not include_market_features:
        blocked.update(DEFAULT_DIRECT_RL_MARKET_COLUMNS)
    return [column for column in feature_columns if column not in blocked]


def _compute_margin_holdout_metrics(
    *,
    holdout_frame: pd.DataFrame,
    holdout_predictions: Sequence[float],
    residual_std: float,
) -> dict[str, float | None]:
    actual_margin = pd.to_numeric(holdout_frame[DEFAULT_MARGIN_TARGET_COLUMN], errors="coerce")
    predicted_margin = pd.Series(holdout_predictions, index=holdout_frame.index, dtype=float)

    metrics: dict[str, float | None] = {
        "mae": float(mean_absolute_error(actual_margin, predicted_margin)),
        "rmse": float(mean_squared_error(actual_margin, predicted_margin) ** 0.5),
        "margin_correlation": float(actual_margin.corr(predicted_margin)),
        "priced_row_count": 0,
        "cover_accuracy": None,
        "cover_log_loss": None,
        "cover_roc_auc": None,
        "cover_brier": None,
    }

    if (
        residual_std > 0
        and "posted_f5_rl_home_point" in holdout_frame.columns
        and "home_cover_at_posted_line" in holdout_frame.columns
    ):
        priced = holdout_frame.copy()
        priced["predicted_margin"] = predicted_margin
        priced["cover_probability"] = priced.apply(
            lambda row: margin_to_cover_probability(
                predicted_margin=float(row["predicted_margin"]),
                home_point=(
                    None
                    if pd.isna(row.get("posted_f5_rl_home_point"))
                    else float(row["posted_f5_rl_home_point"])
                ),
                residual_std=residual_std,
            ),
            axis=1,
        )
        priced["target"] = pd.to_numeric(priced["home_cover_at_posted_line"], errors="coerce")
        priced = priced.loc[priced["cover_probability"].notna() & priced["target"].notna()].copy()
        if not priced.empty:
            probabilities = priced["cover_probability"].astype(float)
            outcomes = priced["target"].astype(int)
            metrics["priced_row_count"] = int(len(priced))
            metrics["cover_accuracy"] = float(accuracy_score(outcomes, probabilities >= 0.5))
            metrics["cover_log_loss"] = float(log_loss(outcomes, probabilities, labels=[0, 1]))
            metrics["cover_brier"] = float(brier_score_loss(outcomes, probabilities))
            metrics["cover_roc_auc"] = _safe_roc_auc(outcomes, probabilities)

    return metrics


def _build_estimator(*, random_state: int) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        random_state=random_state,
        tree_method="hist",
        n_jobs=DEFAULT_XGBOOST_N_JOBS,
        verbosity=0,
    )


def _artifact_to_json_ready(artifact: MarginTrainingArtifact) -> dict[str, Any]:
    payload = asdict(artifact)
    payload["model_path"] = str(artifact.model_path)
    payload["metadata_path"] = str(artifact.metadata_path)
    return payload


def _run_result_to_json_ready(result: MarginTrainingResult) -> dict[str, Any]:
    return {
        "model_version": result.model_version,
        "data_version_hash": result.data_version_hash,
        "holdout_season": result.holdout_season,
        "summary_path": str(result.summary_path),
        "artifact": _artifact_to_json_ready(result.artifact),
    }


if __name__ == "__main__":
    raise SystemExit(main())
