from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.model.direct_rl_trainer import train_direct_rl_model
from src.model.margin_trainer import train_margin_model
from src.ops.rl_v2_evaluator import evaluate_rl_v2_models


def _training_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for season in (2022, 2023, 2024, 2025):
        for game_index in range(1, 10):
            margin = float((game_index % 5) - 2)
            rows.append(
                {
                    "season": season,
                    "game_pk": season * 100 + game_index,
                    "game_date": f"{season}-04-{game_index:02d}",
                    "scheduled_start": f"{season}-04-{game_index:02d}T17:05:00+00:00",
                    "f5_margin": margin,
                    "home_cover_at_posted_line": int(margin > 0.5),
                    "away_cover_at_posted_line": int(margin < 0.5),
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


def test_evaluate_rl_v2_models_returns_direct_margin_and_blend(tmp_path: Path) -> None:
    training_path = tmp_path / "direct_rl_eval.parquet"
    _training_frame().to_parquet(training_path, index=False)

    direct_dir = tmp_path / "direct"
    margin_dir = tmp_path / "margin"
    direct_result = train_direct_rl_model(
        training_data=training_path,
        output_dir=direct_dir,
        holdout_season=2025,
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
    margin_result = train_margin_model(
        training_data=training_path,
        output_dir=margin_dir,
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

    payload = evaluate_rl_v2_models(
        training_data=training_path,
        direct_model_path=direct_result.artifact.model_path,
        direct_metadata_path=direct_result.artifact.metadata_path,
        margin_model_path=margin_result.artifact.model_path,
        margin_metadata_path=margin_result.artifact.metadata_path,
        holdout_season=2025,
        edge_threshold=0.01,
        starting_bankroll=100.0,
        blend_weights=(0.0, 0.5, 1.0),
    )

    assert payload["row_count"] > 0
    assert payload["direct"]["metrics"]["log_loss"] is not None
    assert payload["margin"]["strategy"]["picks"] >= 0
    assert len(payload["blend_weights"]) == 3
    assert payload["best_blend"] is not None
