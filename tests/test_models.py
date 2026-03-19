from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from src.models.bet import BetDecision, BetResult
from src.models.features import GameFeatures
from src.models.game import Game
from src.models.lineup import Lineup, LineupPlayer
from src.models.odds import OddsSnapshot
from src.models.prediction import Prediction
from src.models.weather import WeatherData


UTC = timezone.utc
EASTERN_OFFSET = timezone(timedelta(hours=-4))


def test_prediction_rejects_probability_above_one() -> None:
    with pytest.raises(ValidationError):
        Prediction(
            game_pk=12345,
            model_version="ensemble-v1",
            f5_ml_home_prob=1.01,
            f5_ml_away_prob=0.49,
            f5_rl_home_prob=0.52,
            f5_rl_away_prob=0.48,
            predicted_at=datetime(2026, 4, 15, 16, 0, tzinfo=UTC),
        )


def test_game_features_requires_timezone_aware_as_of_timestamp() -> None:
    with pytest.raises(ValidationError):
        GameFeatures(
            game_pk=12345,
            feature_name="home_team_wrc_plus_7g",
            feature_value=117.4,
            window_size=7,
            as_of_timestamp=datetime(2026, 4, 14, 0, 0),
        )


def test_game_features_normalizes_as_of_timestamp_to_utc() -> None:
    feature_row = GameFeatures(
        game_pk=12345,
        feature_name="home_team_wrc_plus_7g",
        feature_value=117.4,
        window_size=7,
        as_of_timestamp=datetime(2026, 4, 14, 8, 30, tzinfo=EASTERN_OFFSET),
    )

    assert feature_row.as_of_timestamp.tzinfo == UTC
    assert feature_row.as_of_timestamp.hour == 12


def test_odds_snapshot_rejects_non_american_odds() -> None:
    with pytest.raises(ValidationError):
        OddsSnapshot(
            game_pk=12345,
            book_name="DraftKings",
            market_type="f5_ml",
            home_odds=90,
            away_odds=-110,
            fetched_at=datetime(2026, 4, 15, 16, 0, tzinfo=UTC),
        )


def test_bet_result_enum_matches_expected_contract() -> None:
    assert [result.value for result in BetResult] == [
        "WIN",
        "LOSS",
        "PUSH",
        "NO_ACTION",
        "PENDING",
    ]


def test_bet_decision_allows_negative_edge_and_round_trips_to_json() -> None:
    decision = BetDecision(
        game_pk=12345,
        market_type="f5_ml",
        side="away",
        model_probability=0.48,
        fair_probability=0.5,
        edge_pct=-0.02,
        ev=-0.03,
        is_positive_ev=False,
        kelly_stake=0.0,
        odds_at_bet=120,
    )

    restored = BetDecision.model_validate_json(decision.model_dump_json())

    assert restored.edge_pct == pytest.approx(-0.02)
    assert restored.result is BetResult.PENDING


def test_all_models_serialize_and_deserialize_to_json() -> None:
    scheduled_start = datetime(2026, 4, 15, 20, 5, tzinfo=UTC)

    game = Game(
        game_pk=12345,
        scheduled_start=scheduled_start,
        home_team="NYY",
        away_team="BOS",
        home_starter_id=54321,
        away_starter_id=98765,
        venue="Yankee Stadium",
        is_dome=False,
        is_abs_active=True,
        status="scheduled",
    )
    prediction = Prediction(
        game_pk=12345,
        model_version="ensemble-v1",
        f5_ml_home_prob=0.58,
        f5_ml_away_prob=0.42,
        f5_rl_home_prob=0.51,
        f5_rl_away_prob=0.49,
        predicted_at=scheduled_start,
    )
    odds_snapshot = OddsSnapshot(
        game_pk=12345,
        book_name="DraftKings",
        market_type="f5_ml",
        home_odds=-135,
        away_odds=120,
        fetched_at=scheduled_start,
    )
    lineup = Lineup(
        game_pk=12345,
        team="NYY",
        source="mlb-api",
        confirmed=True,
        as_of_timestamp=scheduled_start,
        starting_pitcher_id=54321,
        projected_starting_pitcher_id=54321,
        starter_avg_innings_pitched=5.8,
        players=[
            LineupPlayer(
                batting_order=1,
                player_id=1001,
                player_name="Leadoff Hitter",
                position="CF",
            ),
            LineupPlayer(
                batting_order=2,
                player_id=1002,
                player_name="No. 2 Hitter",
                position="RF",
            ),
        ],
    )
    weather = WeatherData(
        temperature_f=72.0,
        humidity_pct=55.0,
        wind_speed_mph=8.0,
        wind_direction_deg=180.0,
        pressure_hpa=1013.0,
        air_density=1.21,
        wind_factor=5.5,
        is_dome_default=False,
        fetched_at=scheduled_start,
    )

    assert Game.model_validate_json(game.model_dump_json()) == game
    assert Prediction.model_validate_json(prediction.model_dump_json()) == prediction
    assert OddsSnapshot.model_validate_json(odds_snapshot.model_dump_json()) == odds_snapshot
    assert Lineup.model_validate_json(lineup.model_dump_json()) == lineup
    assert WeatherData.model_validate_json(weather.model_dump_json()) == weather
