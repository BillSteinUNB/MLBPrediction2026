from __future__ import annotations

import json

import pandas as pd

from src.features.baselines import calculate_pythagorean_win_percentage
from src.model.data_builder import (
    assert_training_data_is_leakage_free,
    build_training_dataset,
    resolve_training_years,
)


def _schedule_row(
    game_pk: int,
    scheduled_start: str,
    home_team: str,
    away_team: str,
    venue: str,
    *,
    game_type: str = "R",
    status: str = "Final",
    f5_home_score: int = 3,
    f5_away_score: int = 1,
    final_home_score: int = 5,
    final_away_score: int = 2,
) -> dict[str, object]:
    return {
        "game_pk": game_pk,
        "scheduled_start": scheduled_start,
        "home_team": home_team,
        "away_team": away_team,
        "venue": venue,
        "game_type": game_type,
        "status": status,
        "f5_home_score": f5_home_score,
        "f5_away_score": f5_away_score,
        "final_home_score": final_home_score,
        "final_away_score": final_away_score,
    }


def _batting_snapshot() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Season": [2024, 2024, 2024, 2024],
            "Team": ["NYY", "NYY", "BOS", "BOS"],
            "PA": [600, 400, 500, 500],
            "wRC+": [120.0, 90.0, 105.0, 95.0],
            "wOBA": [0.350, 0.320, 0.330, 0.310],
            "ISO": [0.210, 0.170, 0.190, 0.150],
            "BABIP": [0.320, 0.300, 0.310, 0.295],
            "K%": [0.220, 0.260, 0.210, 0.230],
            "BB%": [0.110, 0.090, 0.095, 0.080],
        }
    )


def _pitching_snapshot() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Season": [2024, 2024],
            "Team": ["NYY", "BOS"],
            "IP": [1450.0, 1460.0],
            "TBF": [6200, 6180],
            "xFIP": [3.90, 4.05],
            "xERA": [3.85, 4.02],
            "K%": [0.245, 0.238],
            "BB%": [0.081, 0.085],
            "GB%": [0.430, 0.418],
            "HR/FB": [0.101, 0.109],
            "FBv": [94.7, 94.1],
        }
    )


def _fielding_snapshot() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Team": ["NYY", "NYY", "BOS", "BOS"],
            "DRS": [8.0, 2.0, 4.0, -1.0],
            "OAA": [6.0, 1.0, 3.0, 0.0],
        }
    )


def _framing_snapshot() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "team": ["NYY", "BOS"],
            "runs_extra_strikes": [12.0, 4.0],
        }
    )


def test_build_training_dataset_filters_to_regular_games_and_persists_reproducibility(tmp_path) -> None:
    output_path = tmp_path / "training_sample.parquet"
    schedules = {
        2025: pd.DataFrame(
            [
                _schedule_row(1001, "2025-04-02T23:05:00Z", "NYY", "BOS", "Yankee Stadium"),
                _schedule_row(
                    1002,
                    "2025-04-03T23:05:00Z",
                    "BOS",
                    "NYY",
                    "Fenway Park",
                    f5_home_score=2,
                    f5_away_score=2,
                    final_home_score=4,
                    final_away_score=4,
                ),
                _schedule_row(
                    1998,
                    "2025-03-10T17:05:00Z",
                    "NYY",
                    "BOS",
                    "Spring Park",
                    game_type="S",
                ),
                _schedule_row(
                    1999,
                    "2025-10-10T23:05:00Z",
                    "NYY",
                    "BOS",
                    "Yankee Stadium",
                    game_type="F",
                ),
            ]
        )
    }

    result = build_training_dataset(
        start_year=2025,
        end_year=2025,
        output_path=output_path,
        full_regular_seasons_target=1,
        shortened_season_game_threshold=0,
        schedule_fetcher=lambda year: schedules[year],
        batting_stats_fetcher=lambda season, **_: _batting_snapshot(),
        pitching_stats_fetcher=lambda season, **_: _pitching_snapshot(),
        fielding_stats_fetcher=lambda season, **_: _fielding_snapshot(),
        framing_stats_fetcher=lambda season, **_: _framing_snapshot(),
    )

    dataset = result.dataframe
    assert dataset["game_pk"].tolist() == [1001, 1002]
    assert set(dataset["game_type"]) == {"R"}
    assert dataset["f5_ml_result"].tolist() == [1, 0]
    assert dataset["f5_rl_result"].tolist() == [1, 0]
    assert dataset["f5_tied_after_5"].tolist() == [0, 1]
    assert_training_data_is_leakage_free(dataset)
    assert (pd.to_datetime(dataset["as_of_timestamp"], utc=True) < pd.to_datetime(dataset["scheduled_start"], utc=True)).all()

    feature_columns = [
        column
        for column in dataset.columns
        if column
        not in {
            "game_pk",
            "season",
            "game_date",
            "scheduled_start",
            "as_of_timestamp",
            "build_timestamp",
            "data_version_hash",
            "home_team",
            "away_team",
            "venue",
            "game_type",
        }
    ]
    assert not dataset[feature_columns].isna().any().any()
    assert output_path.exists()
    assert result.metadata_path.exists()

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["row_count"] == 2
    assert metadata["data_version_hash"] == dataset["data_version_hash"].iat[0]
    assert dataset["data_version_hash"].nunique() == 1


def test_build_training_dataset_uses_only_prior_games_in_rolling_features(tmp_path) -> None:
    output_path = tmp_path / "rolling_sample.parquet"
    schedules = {
        2025: pd.DataFrame(
            [
                _schedule_row(
                    2001,
                    "2025-04-02T23:05:00Z",
                    "NYY",
                    "BOS",
                    "Yankee Stadium",
                    f5_home_score=3,
                    f5_away_score=1,
                    final_home_score=5,
                    final_away_score=2,
                ),
                _schedule_row(
                    2002,
                    "2025-04-05T23:05:00Z",
                    "NYY",
                    "BOS",
                    "Yankee Stadium",
                    f5_home_score=0,
                    f5_away_score=4,
                    final_home_score=1,
                    final_away_score=9,
                ),
            ]
        )
    }

    dataset = build_training_dataset(
        start_year=2025,
        end_year=2025,
        output_path=output_path,
        full_regular_seasons_target=1,
        shortened_season_game_threshold=0,
        schedule_fetcher=lambda year: schedules[year],
        batting_stats_fetcher=lambda season, **_: _batting_snapshot(),
        pitching_stats_fetcher=lambda season, **_: _pitching_snapshot(),
        fielding_stats_fetcher=lambda season, **_: _fielding_snapshot(),
        framing_stats_fetcher=lambda season, **_: _framing_snapshot(),
    ).dataframe

    second_game = dataset.loc[dataset["game_pk"] == 2002].iloc[0]
    assert second_game["home_offense_runs_scored_7g"] == 5.0
    assert second_game["home_pitching_runs_allowed_7g"] == 2.0
    assert second_game["away_offense_runs_scored_7g"] == 2.0
    assert second_game["away_pitching_runs_allowed_7g"] == 5.0

    expected_home_pythagorean = calculate_pythagorean_win_percentage(5.0, 2.0)
    assert second_game["home_team_pythagorean_wp_30g"] == expected_home_pythagorean


def test_resolve_training_years_backfills_shortened_season_with_previous_full_year() -> None:
    resolved_years = resolve_training_years(
        start_year=2019,
        end_year=2025,
        full_regular_seasons_target=7,
        season_row_counts={2019: 2430, 2020: 898, 2021: 2430, 2022: 2430, 2023: 2430, 2024: 2430, 2025: 2430},
        shortened_season_game_threshold=2000,
    )

    assert resolved_years == [2018, 2019, 2021, 2022, 2023, 2024, 2025]
