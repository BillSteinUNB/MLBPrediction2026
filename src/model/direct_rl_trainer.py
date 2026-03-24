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
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

from src.clients.weather_client import fetch_game_weather
from src.model.data_builder import DEFAULT_OUTPUT_PATH, build_training_dataset
from src.model.xgboost_trainer import (
    DEFAULT_MODEL_OUTPUT_DIR,
    DEFAULT_RANDOM_STATE,
    DEFAULT_SEARCH_ITERATIONS,
    DEFAULT_SEARCH_SPACE,
    DEFAULT_TIME_SERIES_SPLITS,
    DEFAULT_TOP_FEATURE_COUNT,
    DEFAULT_XGBOOST_N_JOBS,
    _build_model_version,
    _compute_data_version_hash,
    _extract_feature_importance_rankings,
    _load_training_dataframe,
    _normalize_best_params,
    _resolve_data_version_hash,
    _resolve_experiment_output_dir,
    _resolve_holdout_season,
    _resolve_search_iterations,
    _resolve_numeric_feature_columns,
    _safe_roc_auc,
    create_time_series_split,
)


logger = logging.getLogger(__name__)

DEFAULT_DIRECT_RL_MODEL_NAME = "f5_rl_direct_model"
DEFAULT_DIRECT_RL_TARGET_COLUMN = "home_cover_at_posted_line"
DEFAULT_DIRECT_RL_MARKET_COLUMNS: tuple[str, ...] = (
    "posted_f5_rl_home_point",
    "posted_f5_rl_away_point",
    "posted_f5_rl_home_odds",
    "posted_f5_rl_away_odds",
    "posted_f5_rl_home_implied_prob",
    "posted_f5_rl_away_implied_prob",
    "posted_f5_rl_point_abs",
    "posted_f5_rl_home_is_favorite",
)


@dataclass(frozen=True, slots=True)
class DirectRLTrainingArtifact:
    model_name: str
    target_column: str
    model_version: str
    model_path: Path
    metadata_path: Path
    best_params: dict[str, float | int]
    cv_best_log_loss: float
    holdout_metrics: dict[str, float | None]
    feature_columns: list[str]
    feature_importance_rankings: list[dict[str, float | str]]
    train_row_count: int
    holdout_row_count: int
    dropped_push_row_count: int
    holdout_season: int
    market_book_name: str | None


@dataclass(frozen=True, slots=True)
class DirectRLTrainingResult:
    model_version: str
    data_version_hash: str
    holdout_season: int
    summary_path: Path
    artifact: DirectRLTrainingArtifact


def train_direct_rl_model(
    *,
    training_data: pd.DataFrame | str | Path,
    output_dir: str | Path = DEFAULT_MODEL_OUTPUT_DIR,
    holdout_season: int | None = None,
    search_space: Mapping[str, Sequence[float | int]] = DEFAULT_SEARCH_SPACE,
    time_series_splits: int = DEFAULT_TIME_SERIES_SPLITS,
    search_iterations: int = DEFAULT_SEARCH_ITERATIONS,
    random_state: int = DEFAULT_RANDOM_STATE,
    top_feature_count: int = DEFAULT_TOP_FEATURE_COUNT,
    target_column: str = DEFAULT_DIRECT_RL_TARGET_COLUMN,
    market_book_name: str | None = None,
) -> DirectRLTrainingResult:
    dataset = _augment_direct_rl_training_frame(_load_training_dataframe(training_data))
    frame, dropped_push_row_count = _prepare_direct_rl_frame(dataset, target_column=target_column)
    if frame.empty:
        raise ValueError("No direct RL rows with posted runline targets are available")

    effective_holdout_season = _resolve_holdout_season(frame, holdout_season)
    feature_columns = _resolve_direct_rl_feature_columns(frame)
    if not feature_columns:
        raise ValueError("Direct RL training data does not contain any usable numeric features")

    train_frame = frame.loc[frame["season"] < effective_holdout_season].copy()
    holdout_frame = frame.loc[frame["season"] == effective_holdout_season].copy()
    if train_frame.empty:
        raise ValueError(f"No direct RL training rows found before holdout season {effective_holdout_season}")
    if holdout_frame.empty:
        raise ValueError(f"No direct RL holdout rows found for season {effective_holdout_season}")

    data_version_hash = _resolve_data_version_hash(dataset)
    model_version = _build_model_version(data_version_hash)
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    search = RandomizedSearchCV(
        estimator=_build_estimator(random_state=random_state),
        param_distributions={key: list(values) for key, values in search_space.items()},
        n_iter=_resolve_search_iterations(search_space, search_iterations),
        scoring="neg_log_loss",
        cv=create_time_series_split(
            row_count=len(train_frame),
            requested_splits=time_series_splits,
        ),
        random_state=random_state,
        refit=True,
        n_jobs=1,
    )
    search.fit(train_frame[feature_columns], train_frame[target_column])
    best_estimator: XGBClassifier = search.best_estimator_
    holdout_probabilities = best_estimator.predict_proba(holdout_frame[feature_columns])[:, 1]
    holdout_predictions = best_estimator.predict(holdout_frame[feature_columns])

    holdout_metrics = {
        "accuracy": float(accuracy_score(holdout_frame[target_column], holdout_predictions)),
        "log_loss": float(log_loss(holdout_frame[target_column], holdout_probabilities, labels=[0, 1])),
        "roc_auc": _safe_roc_auc(holdout_frame[target_column], holdout_probabilities),
        "brier": float(brier_score_loss(holdout_frame[target_column], holdout_probabilities)),
    }
    best_params = _normalize_best_params(search.best_params_)
    cv_best_log_loss = float(-search.best_score_)
    feature_importance_rankings = _extract_feature_importance_rankings(
        best_estimator,
        feature_columns,
        top_feature_count=top_feature_count,
    )

    model_path = resolved_output_dir / f"{DEFAULT_DIRECT_RL_MODEL_NAME}_{model_version}.joblib"
    joblib.dump(best_estimator, model_path)
    metadata_path = model_path.with_suffix(".metadata.json")
    metadata_payload = {
        "model_name": DEFAULT_DIRECT_RL_MODEL_NAME,
        "target_column": target_column,
        "model_version": model_version,
        "data_version_hash": data_version_hash,
        "holdout_season": effective_holdout_season,
        "market_book_name": market_book_name,
        "train_row_count": int(len(train_frame)),
        "holdout_row_count": int(len(holdout_frame)),
        "dropped_push_row_count": int(dropped_push_row_count),
        "feature_columns": feature_columns,
        "best_params": best_params,
        "cv_best_log_loss": cv_best_log_loss,
        "holdout_metrics": holdout_metrics,
        "feature_importance_rankings": feature_importance_rankings,
        "search_space": {key: list(values) for key, values in search_space.items()},
        "time_series_splits": int(search.cv.n_splits),
        "trained_at": datetime.now(UTC).isoformat(),
    }
    metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")

    artifact = DirectRLTrainingArtifact(
        model_name=DEFAULT_DIRECT_RL_MODEL_NAME,
        target_column=target_column,
        model_version=model_version,
        model_path=model_path,
        metadata_path=metadata_path,
        best_params=best_params,
        cv_best_log_loss=cv_best_log_loss,
        holdout_metrics=holdout_metrics,
        feature_columns=feature_columns,
        feature_importance_rankings=feature_importance_rankings,
        train_row_count=len(train_frame),
        holdout_row_count=len(holdout_frame),
        dropped_push_row_count=dropped_push_row_count,
        holdout_season=effective_holdout_season,
        market_book_name=market_book_name,
    )

    summary_path = resolved_output_dir / f"direct_rl_training_run_{model_version}.json"
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
    return DirectRLTrainingResult(
        model_version=model_version,
        data_version_hash=data_version_hash,
        holdout_season=effective_holdout_season,
        summary_path=summary_path,
        artifact=artifact,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a direct posted-line F5 RL cover model")
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
    parser.add_argument("--time-series-splits", type=int, default=DEFAULT_TIME_SERIES_SPLITS)
    parser.add_argument("--search-iterations", type=int, default=DEFAULT_SEARCH_ITERATIONS)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    training_path = Path(args.training_data)
    if args.refresh_training_data or not training_path.exists():
        if not args.historical_odds_db:
            raise ValueError("--historical-odds-db is required when rebuilding direct RL training data")
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
    result = train_direct_rl_model(
        training_data=training_path,
        output_dir=resolved_output_dir,
        holdout_season=args.holdout_season,
        time_series_splits=args.time_series_splits,
        search_iterations=args.search_iterations,
        random_state=args.random_state,
        market_book_name=args.historical_rl_book,
    )
    print(json.dumps(_run_result_to_json_ready(result), indent=2))
    return 0


def _augment_direct_rl_training_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    frame = dataframe.copy()
    if "posted_f5_rl_home_odds" in frame.columns:
        frame["posted_f5_rl_home_implied_prob"] = frame["posted_f5_rl_home_odds"].map(
            _american_to_implied_probability
        )
    if "posted_f5_rl_away_odds" in frame.columns:
        frame["posted_f5_rl_away_implied_prob"] = frame["posted_f5_rl_away_odds"].map(
            _american_to_implied_probability
        )
    if "posted_f5_rl_home_point" in frame.columns:
        frame["posted_f5_rl_point_abs"] = pd.to_numeric(
            frame["posted_f5_rl_home_point"],
            errors="coerce",
        ).abs()
    if "posted_f5_rl_home_odds" in frame.columns and "posted_f5_rl_away_odds" in frame.columns:
        home_odds = pd.to_numeric(frame["posted_f5_rl_home_odds"], errors="coerce")
        away_odds = pd.to_numeric(frame["posted_f5_rl_away_odds"], errors="coerce")
        frame["posted_f5_rl_home_is_favorite"] = (
            home_odds.lt(away_odds).astype(float).where(home_odds.notna() & away_odds.notna())
        )
    return frame


def _prepare_direct_rl_frame(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
) -> tuple[pd.DataFrame, int]:
    required_columns = {
        target_column,
        "push_at_posted_line",
        "posted_f5_rl_home_point",
        "posted_f5_rl_home_odds",
        "posted_f5_rl_away_odds",
    }
    missing = sorted(column for column in required_columns if column not in dataframe.columns)
    if missing:
        raise ValueError(
            "Direct RL training data is missing required posted-line columns: " + ", ".join(missing)
        )

    frame = dataframe.copy()
    frame[target_column] = pd.to_numeric(frame[target_column], errors="coerce")
    frame["push_at_posted_line"] = pd.to_numeric(frame["push_at_posted_line"], errors="coerce")
    frame = frame.loc[frame[target_column].notna()].copy()
    dropped_push_row_count = int(frame["push_at_posted_line"].fillna(0).eq(1).sum())
    frame = frame.loc[frame["push_at_posted_line"].fillna(0) == 0].copy()
    frame[target_column] = frame[target_column].astype(int)
    return frame.sort_values(["scheduled_start", "game_pk"]).reset_index(drop=True), dropped_push_row_count


def _resolve_direct_rl_feature_columns(dataframe: pd.DataFrame) -> list[str]:
    feature_columns = _resolve_numeric_feature_columns(dataframe)
    resolved = list(feature_columns)
    for column in DEFAULT_DIRECT_RL_MARKET_COLUMNS:
        if (
            column in dataframe.columns
            and pd.api.types.is_numeric_dtype(dataframe[column])
            and dataframe[column].nunique(dropna=False) > 1
            and column not in resolved
        ):
            resolved.append(column)
    return resolved


def _american_to_implied_probability(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    odds = float(value)
    if odds == 0:
        return None
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def _build_estimator(*, random_state: int) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        tree_method="hist",
        n_jobs=DEFAULT_XGBOOST_N_JOBS,
        verbosity=0,
    )


def _artifact_to_json_ready(artifact: DirectRLTrainingArtifact) -> dict[str, Any]:
    payload = asdict(artifact)
    payload["model_path"] = str(artifact.model_path)
    payload["metadata_path"] = str(artifact.metadata_path)
    return payload


def _run_result_to_json_ready(result: DirectRLTrainingResult) -> dict[str, Any]:
    return {
        "model_version": result.model_version,
        "data_version_hash": result.data_version_hash,
        "holdout_season": result.holdout_season,
        "summary_path": str(result.summary_path),
        "artifact": _artifact_to_json_ready(result.artifact),
    }


if __name__ == "__main__":
    raise SystemExit(main())
