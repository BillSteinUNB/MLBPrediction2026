from __future__ import annotations

import pandas as pd

from src.clients.historical_odds_client import _match_requested_games_to_old_scraper_snapshots
from src.clients.historical_odds_client import _pair_old_scraper_market_rows


def test_pair_old_scraper_market_rows_keeps_alternate_lines_separate_for_runlines() -> None:
    raw = pd.DataFrame(
        [
            {
                "event_id": "e1",
                "game_date": "2024-09-02",
                "commence_time_utc": "2024-09-02T19:05:00Z",
                "away_team": "CHW",
                "home_team": "BAL",
                "book_name": "OddsPortal:provider_417",
                "market_type": "full_game_rl",
                "side": "home",
                "point": 1.5,
                "price": 460,
                "fetched_at": "2024-09-02T18:00:00Z",
                "is_opening": 1,
            },
            {
                "event_id": "e1",
                "game_date": "2024-09-02",
                "commence_time_utc": "2024-09-02T19:05:00Z",
                "away_team": "CHW",
                "home_team": "BAL",
                "book_name": "OddsPortal:provider_417",
                "market_type": "full_game_rl",
                "side": "away",
                "point": 1.5,
                "price": -556,
                "fetched_at": "2024-09-02T18:00:00Z",
                "is_opening": 1,
            },
            {
                "event_id": "e1",
                "game_date": "2024-09-02",
                "commence_time_utc": "2024-09-02T19:05:00Z",
                "away_team": "CHW",
                "home_team": "BAL",
                "book_name": "OddsPortal:provider_417",
                "market_type": "full_game_rl",
                "side": "home",
                "point": 2.5,
                "price": 750,
                "fetched_at": "2024-09-02T18:00:00Z",
                "is_opening": 1,
            },
            {
                "event_id": "e1",
                "game_date": "2024-09-02",
                "commence_time_utc": "2024-09-02T19:05:00Z",
                "away_team": "CHW",
                "home_team": "BAL",
                "book_name": "OddsPortal:provider_417",
                "market_type": "full_game_rl",
                "side": "away",
                "point": 2.5,
                "price": -1000,
                "fetched_at": "2024-09-02T18:00:00Z",
                "is_opening": 1,
            },
        ]
    )

    paired = _pair_old_scraper_market_rows(
        raw,
        market_type="full_game_rl",
        max_abs_odds=10000,
        source_db_path="data/mlb_odds_oddsportal.db",
    )

    assert len(paired) == 2
    assert set(pd.to_numeric(paired["home_point"], errors="coerce").tolist()) == {1.5, 2.5}
    assert set(pd.to_numeric(paired["away_point"], errors="coerce").tolist()) == {1.5, 2.5}


def test_match_requested_games_prefers_line_bearing_opening_totals() -> None:
    requested = pd.DataFrame(
        [
            {
                "_request_id": 1,
                "game_pk": 745444,
                "scheduled_start": pd.Timestamp("2024-03-28T19:05:00Z"),
                "pregame_cutoff": pd.Timestamp("2024-03-28T19:05:00Z"),
                "request_game_date": pd.Timestamp("2024-03-28").date(),
                "home_team_norm": "BAL",
                "away_team_norm": "LAA",
            }
        ]
    )
    snapshots = pd.DataFrame(
        [
            {
                "event_id": "e1",
                "game_date": "2024-03-28",
                "commence_time_utc": "2024-03-28T19:05:00Z",
                "away_team": "LAA",
                "home_team": "BAL",
                "away_team_norm": "LAA",
                "home_team_norm": "BAL",
                "book_name": "Opener",
                "market_type": "f5_total",
                "home_odds": None,
                "away_odds": None,
                "home_point": None,
                "away_point": None,
                "total_point": None,
                "over_odds": -135,
                "under_odds": 100,
                "fetched_at": "2024-03-28T14:00:00Z",
                "opening_rank": 0,
                "line_present": False,
                "source_schema": "old_scraper",
                "source_origin": "legacy_old_scraper",
                "source_db_path": "OddsScraper/data/mlb_odds.db",
            },
            {
                "event_id": "e1",
                "game_date": "2024-03-28",
                "commence_time_utc": "2024-03-28T19:05:00Z",
                "away_team": "LAA",
                "home_team": "BAL",
                "away_team_norm": "LAA",
                "home_team_norm": "BAL",
                "book_name": "BetMGM",
                "market_type": "f5_total",
                "home_odds": None,
                "away_odds": None,
                "home_point": None,
                "away_point": None,
                "total_point": 4.0,
                "over_odds": -130,
                "under_odds": -105,
                "fetched_at": "2024-03-28T15:00:00Z",
                "opening_rank": 1,
                "line_present": True,
                "source_schema": "old_scraper",
                "source_origin": "legacy_old_scraper",
                "source_db_path": "OddsScraper/data/mlb_odds.db",
            },
        ]
    )

    matched = _match_requested_games_to_old_scraper_snapshots(
        requested=requested,
        snapshots=snapshots,
        snapshot_selection="opening",
        aggregate_consensus=False,
    )

    assert len(matched) == 1
    assert float(matched.iloc[0]["total_point"]) == 4.0
