from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.model.data_builder import (
    _prepare_schedule_frame,
    _schedule_adjustment_features,
    build_training_dataset,
)
from tests.test_data_builder import (
    _fake_bullpen_metrics_fetcher,
    _fake_fielding_fetcher,
    _fake_framing_fetcher,
    _fake_start_metrics_fetcher,
    _fake_team_logs_fetcher,
    _fake_weather_fetcher,
    _bullpen_metrics_by_season,
    _fielding_by_season,
    _framing_by_season,
    _schedule_row,
    _start_metrics_by_season,
    _team_logs_by_season,
)


@pytest.fixture()
def engineered_training_fixture(tmp_path: Path) -> pd.DataFrame:
    schedule = pd.DataFrame(
        [
            _schedule_row(
                4001,
                "2025-04-08T23:05:00Z",
                "NYY",
                "BOS",
                "Yankee Stadium",
                f5_home_score=2,
                f5_away_score=1,
                final_home_score=5,
                final_away_score=3,
            ),
            _schedule_row(
                4002,
                "2025-04-10T23:05:00Z",
                "NYY",
                "BOS",
                "Yankee Stadium",
                f5_home_score=3,
                f5_away_score=1,
                final_home_score=4,
                final_away_score=2,
            ),
        ]
    )

    result = build_training_dataset(
        start_year=2025,
        end_year=2025,
        output_path=tmp_path / "feature_engineering_fixture.parquet",
        full_regular_seasons_target=1,
        shortened_season_game_threshold=0,
        schedule_fetcher=lambda _year: schedule.copy(),
        batting_stats_fetcher=lambda *_args, **_kwargs: pd.DataFrame(),
        team_logs_fetcher=_fake_team_logs_fetcher(_team_logs_by_season()),
        fielding_stats_fetcher=_fake_fielding_fetcher(_fielding_by_season()),
        framing_stats_fetcher=_fake_framing_fetcher(_framing_by_season()),
        start_metrics_fetcher=_fake_start_metrics_fetcher(_start_metrics_by_season()),
        bullpen_metrics_fetcher=_fake_bullpen_metrics_fetcher(_bullpen_metrics_by_season()),
        lineup_fetcher=lambda *_args, **_kwargs: [],
        weather_fetcher=_fake_weather_fetcher,
    )
    return result.dataframe
def test_feature_engineering_fixture_keeps_wrc_plus_and_xfip_in_reasonable_ranges(
    engineered_training_fixture: pd.DataFrame,
) -> None:
    target_row = engineered_training_fixture.loc[
        engineered_training_fixture["game_pk"] == 4002
    ].iloc[0]
    wrc_columns = sorted(column for column in engineered_training_fixture.columns if "wrc_plus" in column)
    xfip_columns = sorted(column for column in engineered_training_fixture.columns if "xfip" in column)

    assert wrc_columns
    assert xfip_columns
    for column in wrc_columns:
        assert 50.0 <= float(target_row[column]) <= 200.0, column
    for column in xfip_columns:
        assert 2.0 <= float(target_row[column]) <= 6.0, column


def test_feature_engineering_applies_sutter_health_park_factors() -> None:
    schedule = pd.DataFrame(
        [
            _schedule_row(
                5101,
                "2025-04-10T23:05:00Z",
                "OAK",
                "SEA",
                "Sutter Health Park",
                home_starter_id=500,
                away_starter_id=600,
            )
        ]
    )

    row = _prepare_schedule_frame(schedule, require_final_scores=False).iloc[0]

    assert row["park_runs_factor"] == pytest.approx(1.25)
    assert row["park_hr_factor"] == pytest.approx(1.30)


def test_feature_engineering_disables_abs_adjustments_for_exception_venue() -> None:
    schedule = pd.DataFrame(
        [
            _schedule_row(
                5201,
                "2025-04-10T18:05:00Z",
                "ARI",
                "COL",
                "Alfredo Harp Helu Stadium - Mexico City Series",
                home_starter_id=701,
                away_starter_id=702,
            )
        ]
    )

    row = _prepare_schedule_frame(schedule, require_final_scores=False).iloc[0].to_dict()
    schedule_adjustments = _schedule_adjustment_features(row)

    assert float(row["is_abs_active"]) == pytest.approx(0.0)
    assert schedule_adjustments["abs_active"] == pytest.approx(0.0)
    assert schedule_adjustments["abs_walk_rate_delta"] == pytest.approx(0.0)
    assert schedule_adjustments["abs_strikeout_rate_delta"] == pytest.approx(0.0)
