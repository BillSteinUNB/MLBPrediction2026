from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.model.margin_pricing import margin_to_cover_probability
from src.model.margin_trainer import train_margin_model


def _training_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for season in (2022, 2023, 2024, 2025):
        for game_index in range(1, 9):
            margin = float((game_index % 5) - 2)
            rows.append(
                {
                    "season": season,
                    "game_pk": season * 100 + game_index,
                    "scheduled_start": f"{season}-04-{game_index:02d}T17:05:00+00:00",
                    "f5_margin": margin,
                    "home_cover_at_posted_line": int(margin > 0.5),
                    "push_at_posted_line": 0,
                    "posted_f5_rl_home_point": -0.5,
                    "posted_f5_rl_away_point": 0.5,
                    "posted_f5_rl_home_odds": -110,
                    "posted_f5_rl_away_odds": 100,
                    "posted_f5_rl_home_implied_prob": 0.5238,
                    "posted_f5_rl_away_implied_prob": 0.5,
                    "posted_f5_rl_point_abs": 0.5,
                    "posted_f5_rl_home_is_favorite": 1.0,
                    "home_team_wrc_plus_14g": 100 + game_index + (season - 2022),
                    "away_team_wrc_plus_14g": 95 + game_index,
                    "home_team_bullpen_xfip": 4.0 + (game_index * 0.05),
                    "away_team_bullpen_xfip": 4.3 - (game_index * 0.04),
                }
            )
    return pd.DataFrame(rows)


def test_margin_to_cover_probability_behaves_reasonably() -> None:
    assert margin_to_cover_probability(predicted_margin=1.0, home_point=-0.5, residual_std=1.0) > 0.5
    assert margin_to_cover_probability(predicted_margin=-1.0, home_point=-0.5, residual_std=1.0) < 0.5


def test_train_margin_model_writes_artifacts(tmp_path: Path) -> None:
    result = train_margin_model(
        training_data=_training_frame(),
        output_dir=tmp_path,
        holdout_season=2025,
        include_market_features=False,
        search_space={
            "max_depth": [3],
            "n_estimators": [50],
            "learning_rate": [0.05],
            "subsample": [0.8],
            "colsample_bytree": [0.8],
            "min_child_weight": [1],
            "gamma": [0.0],
            "reg_alpha": [0.0],
            "reg_lambda": [1.0],
        },
        search_iterations=1,
        time_series_splits=2,
    )

    assert result.artifact.model_path.exists()
    assert result.artifact.metadata_path.exists()
    assert result.summary_path.exists()
    assert result.artifact.holdout_metrics["mae"] is not None
    assert result.artifact.holdout_metrics["cover_log_loss"] is not None

    payload = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert payload["artifact"]["model_name"] == "f5_margin_v2_model"
