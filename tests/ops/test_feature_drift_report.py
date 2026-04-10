from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.ops.feature_drift_report import build_family_summary, build_feature_drift_dataframe, write_feature_drift_report


def test_build_feature_drift_dataframe_flags_shift(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "season": [2021] * 120 + [2024] * 120,
            "game_pk": list(range(240)),
            "home_team_woba_30g": [0.310] * 120 + [0.370] * 120,
            "weather_composite": [1.0] * 120 + [1.0] * 120,
            "f5_ml_result": [1] * 60 + [0] * 60 + [1] * 90 + [0] * 30,
        }
    )
    parquet_path = tmp_path / "training.parquet"
    frame.to_parquet(parquet_path, index=False)

    drift_frame = build_feature_drift_dataframe(
        parquet_path,
        baseline_start=2021,
        baseline_end=2021,
        recent_start=2024,
        recent_end=2024,
        min_non_null_count=50,
    )
    assert not drift_frame.empty
    top_columns = set(drift_frame.head(3)["column"])
    assert "home_team_woba_30g" in top_columns
    assert "f5_ml_result" in set(drift_frame["column"])
    assert drift_frame.loc[drift_frame["column"] == "f5_ml_result", "data_role"].iloc[0] == "response"


def test_write_feature_drift_report_outputs_files(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "season": [2021] * 100 + [2024] * 100,
            "game_pk": list(range(200)),
            "away_starter_xfip_30s": [4.1] * 100 + [4.8] * 100,
            "plate_umpire_known": [1] * 100 + [0] * 100,
            "f5_rl_result": [0] * 65 + [1] * 35 + [0] * 50 + [1] * 50,
        }
    )
    parquet_path = tmp_path / "training.parquet"
    frame.to_parquet(parquet_path, index=False)

    paths = write_feature_drift_report(
        parquet_path,
        output_dir=tmp_path,
        baseline_start=2021,
        baseline_end=2021,
        recent_start=2024,
        recent_end=2024,
        min_non_null_count=50,
    )
    assert paths["csv_path"].exists()
    assert paths["family_csv_path"].exists()
    assert paths["html_path"].exists()
    assert paths["json_path"].exists()

    family_frame = build_family_summary(pd.read_csv(paths["csv_path"]))
    assert not family_frame.empty
