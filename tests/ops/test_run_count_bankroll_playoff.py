from __future__ import annotations

import pandas as pd

from src.ops.run_count_bankroll_playoff import normalize_runline_points
from src.ops.run_count_bankroll_playoff import _settle_market_bet
from src.ops.run_count_bankroll_playoff import _load_all_market_odds
from src.ops.run_count_bankroll_playoff import TRUSTED_BANKROLL_MARKET_TYPES
from src.ops.run_count_bankroll_playoff import default_bankroll_playoff_candidates


def test_normalize_runline_points_assigns_signs_from_favorite_odds() -> None:
    home_point, away_point = normalize_runline_points(
        home_point=1.5,
        away_point=1.5,
        home_odds=-145,
        away_odds=125,
    )

    assert home_point == -1.5
    assert away_point == 1.5


def test_settle_market_bet_handles_runline_and_totals() -> None:
    profit_units, result = _settle_market_bet(
        market_type="full_game_rl",
        side="away",
        line_at_bet=1.5,
        odds_at_bet=-110,
        flat_bet_size_units=1.0,
        full_game_home_score=5,
        full_game_away_score=4,
        f5_home_score=0,
        f5_away_score=0,
    )

    assert result == "WIN"
    assert profit_units > 0.0

    profit_units, result = _settle_market_bet(
        market_type="f5_total",
        side="under",
        line_at_bet=4.0,
        odds_at_bet=-105,
        flat_bet_size_units=1.0,
        full_game_home_score=0,
        full_game_away_score=0,
        f5_home_score=2,
        f5_away_score=2,
    )

    assert result == "PUSH"
    assert profit_units == 0.0


def test_load_all_market_odds_filters_oddsportal_sources(monkeypatch) -> None:
    sample = pd.DataFrame(
        [
            {"game_pk": 1, "source_origin": "oddsportal"},
            {"game_pk": 2, "source_origin": "legacy_old_scraper"},
            {"game_pk": 3, "source_origin": "canonical"},
        ]
    )

    def _fake_loader(**_: object) -> pd.DataFrame:
        return sample.copy()

    monkeypatch.setattr(
        "src.ops.run_count_bankroll_playoff.load_historical_odds_for_games",
        _fake_loader,
    )

    candidate = default_bankroll_playoff_candidates()["reset_benchmark_v1"]
    market_frames = _load_all_market_odds(candidate=candidate, games_frame=pd.DataFrame({"game_pk": [1, 2, 3]}))

    assert set(market_frames) == set(TRUSTED_BANKROLL_MARKET_TYPES)
    for market_frame in market_frames.values():
        assert set(market_frame["source_origin"]) == {"legacy_old_scraper", "canonical"}
