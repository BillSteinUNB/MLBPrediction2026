from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import joblib
import pandas as pd

from src.model.direct_rl_trainer import (
    _prepare_direct_rl_frame,
    _resolve_direct_rl_feature_columns,
    train_direct_rl_model,
)


def _training_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    start = datetime(2024, 4, 1, 19, 5, tzinfo=UTC)

    for index in range(24):
        scheduled_start = start + timedelta(days=index)
        season = 2024 if index < 14 else 2025
        signal = 1.0 if index % 2 == 0 else 0.0
        push = 1 if index in {5, 17} else 0
        target = 1 if index % 4 in {0, 1} else 0
        if push:
            target = 0

        home_point = -0.5 if index % 3 else -1.5
        away_point = abs(home_point)
        home_odds = -130 if target else 110
        away_odds = 110 if target else -130
        rows.append(
            {
                "game_pk": 20_000 + index,
                "season": season,
                "game_date": scheduled_start.date().isoformat(),
                "scheduled_start": scheduled_start.isoformat(),
                "as_of_timestamp": (scheduled_start - timedelta(days=1)).isoformat(),
                "home_team": "NYY",
                "away_team": "BOS",
                "venue": "Yankee Stadium",
                "game_type": "R",
                "build_timestamp": scheduled_start.isoformat(),
                "data_version_hash": "synthetic-direct-rl-hash",
                "home_team_log5": 0.35 + (0.45 * signal),
                "away_team_log5": 0.65 - (0.45 * signal),
                "park_runs_factor": 1.0 + (0.02 * (index % 4)),
                "weather_composite": 1.0,
                "f5_margin": 1.0 if target else -1.0,
                "f5_tied_after_5": 0,
                "f5_ml_result": int(signal),
                "f5_rl_result": int(signal),
                "posted_f5_rl_book_name": "sbr:caesars",
                "posted_f5_rl_home_point": home_point,
                "posted_f5_rl_away_point": away_point,
                "posted_f5_rl_home_odds": home_odds,
                "posted_f5_rl_away_odds": away_odds,
                "home_cover_at_posted_line": target,
                "away_cover_at_posted_line": 1 - target if not push else 0,
                "push_at_posted_line": push,
            }
        )

    return pd.DataFrame(rows)


def test_prepare_direct_rl_frame_drops_push_rows() -> None:
    frame, dropped_pushes = _prepare_direct_rl_frame(
        _training_frame(),
        target_column="home_cover_at_posted_line",
    )

    assert dropped_pushes == 2
    assert frame["push_at_posted_line"].sum() == 0
    assert frame["home_cover_at_posted_line"].isin([0, 1]).all()


def test_resolve_direct_rl_feature_columns_includes_market_context() -> None:
    frame = _training_frame()
    frame["posted_f5_rl_home_implied_prob"] = 0.5
    frame["posted_f5_rl_away_implied_prob"] = 0.5
    frame["posted_f5_rl_point_abs"] = frame["posted_f5_rl_home_point"].abs()
    frame["posted_f5_rl_home_is_favorite"] = (frame["posted_f5_rl_home_odds"] < 0).astype(float)

    feature_columns = _resolve_direct_rl_feature_columns(frame)

    assert "posted_f5_rl_home_point" in feature_columns
    assert "posted_f5_rl_home_odds" in feature_columns
    assert "posted_f5_rl_home_is_favorite" in feature_columns


def test_train_direct_rl_model_trains_and_saves_versioned_model(tmp_path) -> None:
    input_path = tmp_path / "direct_rl_training.parquet"
    _training_frame().to_parquet(input_path, index=False)

    result = train_direct_rl_model(
        training_data=input_path,
        output_dir=tmp_path / "models",
        holdout_season=2025,
        search_space={
            "max_depth": [3],
            "n_estimators": [20, 30],
            "learning_rate": [0.1],
        },
        time_series_splits=3,
        search_iterations=2,
        random_state=7,
        market_book_name="sbr:caesars",
    )

    assert result.artifact.model_path.exists()
    assert result.artifact.holdout_metrics["accuracy"] >= 0.5
    loaded_model = joblib.load(result.artifact.model_path)
    assert loaded_model.__class__.__name__ == "XGBClassifier"

    metadata = json.loads(result.artifact.metadata_path.read_text(encoding="utf-8"))
    assert metadata["market_book_name"] == "sbr:caesars"
    assert metadata["dropped_push_row_count"] == 2
    assert "posted_f5_rl_home_point" in metadata["feature_columns"]
