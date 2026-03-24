from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.ops.model_regime_scan import (
    build_edge_behavior_dataframe,
    build_feature_relationship_dataframe,
    build_season_trend_dataframe,
    write_model_regime_scan,
)


def test_build_season_trend_dataframe_and_relationships() -> None:
    frame = pd.DataFrame(
        {
            "season": [2021] * 80 + [2024] * 80,
            "home_team_bullpen_xfip": [5.2] * 80 + [3.4] * 80,
            "home_starter_pitch_mix_entropy_30s": [1.70] * 80 + [1.85] * 80,
            "f5_ml_result": [0] * 40 + [1] * 40 + [0] * 20 + [1] * 60,
        }
    )
    trend = build_season_trend_dataframe(
        frame,
        trend_columns=["home_team_bullpen_xfip", "home_starter_pitch_mix_entropy_30s", "f5_ml_result"],
    )
    assert not trend.empty
    assert "season_2021" in trend.columns
    rel = build_feature_relationship_dataframe(
        frame,
        columns=["home_team_bullpen_xfip", "home_starter_pitch_mix_entropy_30s"],
        target_column="f5_ml_result",
    )
    assert not rel.empty


def test_write_model_regime_scan_outputs_files(tmp_path: Path) -> None:
    training = pd.DataFrame(
        {
            "season": [2021] * 100 + [2024] * 100,
            "home_team_bullpen_xfip": [5.2] * 100 + [3.4] * 100,
            "away_team_bullpen_xfip": [5.3] * 100 + [3.5] * 100,
            "home_starter_pitch_mix_entropy_30s": [1.72] * 100 + [1.86] * 100,
            "away_starter_pitch_mix_entropy_30s": [1.73] * 100 + [1.87] * 100,
            "home_starter_xfip_30s": [4.8] * 100 + [4.5] * 100,
            "away_starter_xfip_30s": [4.8] * 100 + [4.5] * 100,
            "home_team_woba_30g": [0.315] * 100 + [0.323] * 100,
            "away_team_woba_30g": [0.314] * 100 + [0.321] * 100,
            "f5_ml_result": [0] * 50 + [1] * 50 + [0] * 40 + [1] * 60,
            "f5_rl_result": [0] * 65 + [1] * 35 + [0] * 55 + [1] * 45,
        }
    )
    training_path = tmp_path / "training.parquet"
    training.to_parquet(training_path, index=False)

    def _predictions(period: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "scheduled_start": ["2021-04-01T00:00:00Z"] * 4 if period == "early" else ["2024-04-01T00:00:00Z"] * 4,
                "is_bet": [1, 1, 1, 1],
                "bet_edge": [0.05, 0.09, 0.16, 0.22],
                "bet_result": ["WIN", "LOSS", "WIN", "LOSS"] if period == "early" else ["LOSS", "LOSS", "WIN", "LOSS"],
                "bet_stake_units": [1.0, 1.0, 1.5, 2.0],
                "bet_profit_units": [0.9, -1.0, 1.2, -2.0] if period == "early" else [-1.0, -1.0, 1.2, -2.0],
            }
        )

    early_path = tmp_path / "early.csv"
    recent_path = tmp_path / "recent.csv"
    _predictions("early").to_csv(early_path, index=False)
    _predictions("recent").to_csv(recent_path, index=False)

    result = write_model_regime_scan(
        training_data_path=training_path,
        early_predictions_path=early_path,
        recent_predictions_path=recent_path,
        output_dir=tmp_path,
    )
    assert result["trend_csv"].exists()
    assert result["relationship_csv"].exists()
    assert result["edge_csv"].exists()
    assert result["html_path"].exists()
    assert result["json_path"].exists()

    edge_frame = build_edge_behavior_dataframe(early_path, label="early", edge_buckets=(0.05, 0.10, 0.20, 1.0))
    assert not edge_frame.empty
