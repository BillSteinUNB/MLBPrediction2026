from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from src.db import init_db


UTC = timezone.utc


def _seed_game(db_path: Path, game_pk: int = 12345) -> None:
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO games (game_pk, date, home_team, away_team, venue, status)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (game_pk, "2026-04-15T20:05:00+00:00", "NYY", "BOS", "Yankee Stadium", "scheduled"),
        )
        connection.commit()


def test_american_to_implied_handles_negative_positive_and_even_odds() -> None:
    from src.clients.odds_client import american_to_implied

    assert american_to_implied(-150) == pytest.approx(0.6)
    assert american_to_implied(130) == pytest.approx(100 / 230)
    assert american_to_implied(100) == pytest.approx(0.5)
    assert american_to_implied(-100) == pytest.approx(0.5)


def test_devig_probabilities_sum_to_one() -> None:
    from src.clients.odds_client import devig_probabilities

    home_fair, away_fair = devig_probabilities(-120, 110)

    assert home_fair + away_fair == pytest.approx(1.0)


def test_fetch_mlb_odds_returns_snapshots_and_updates_usage_tracking(tmp_path: Path) -> None:
    from src.clients.odds_client import fetch_mlb_odds

    db_path = tmp_path / "odds.db"
    init_db(db_path)
    _seed_game(db_path)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/events"):
            return httpx.Response(
                200,
                json=[
                    {
                        "id": "evt-123",
                        "sport_key": "baseball_mlb",
                        "sport_title": "MLB",
                        "commence_time": "2026-04-15T20:05:00Z",
                        "home_team": "New York Yankees",
                        "away_team": "Boston Red Sox",
                    }
                ],
                headers={
                    "x-requests-last": "0",
                    "x-requests-used": "0",
                    "x-requests-remaining": "500",
                },
            )

        return httpx.Response(
            200,
            json={
                "id": "evt-123",
                "sport_key": "baseball_mlb",
                "sport_title": "MLB",
                "commence_time": "2026-04-15T20:05:00Z",
                "home_team": "New York Yankees",
                "away_team": "Boston Red Sox",
                "bookmakers": [
                    {
                        "key": "draftkings",
                        "title": "DraftKings",
                        "last_update": "2026-04-15T16:00:00Z",
                        "markets": [
                            {
                                "key": "h2h_1st_5_innings",
                                "last_update": "2026-04-15T16:01:00Z",
                                "outcomes": [
                                    {"name": "New York Yankees", "price": -150},
                                    {"name": "Boston Red Sox", "price": 130},
                                ],
                            },
                            {
                                "key": "spreads_1st_5_innings",
                                "last_update": "2026-04-15T16:01:00Z",
                                "outcomes": [
                                    {"name": "New York Yankees", "price": -105, "point": -0.5},
                                    {"name": "Boston Red Sox", "price": -115, "point": 0.5},
                                ],
                            },
                        ],
                    }
                ],
            },
            headers={
                "x-requests-last": "2",
                "x-requests-used": "2",
                "x-requests-remaining": "498",
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.the-odds-api.com")

    snapshots = fetch_mlb_odds(
        api_key="test-key",
        db_path=db_path,
        client=client,
        commence_time_from=datetime(2026, 4, 15, 0, 0, tzinfo=UTC),
        commence_time_to=datetime(2026, 4, 16, 0, 0, tzinfo=UTC),
    )

    assert len(snapshots) == 2
    assert {snapshot.market_type for snapshot in snapshots} == {"f5_ml", "f5_rl"}
    assert {snapshot.game_pk for snapshot in snapshots} == {12345}

    with sqlite3.connect(db_path) as connection:
        stored_snapshots = connection.execute(
            "SELECT COUNT(*) FROM odds_snapshots WHERE game_pk = ?",
            (12345,),
        ).fetchone()
        usage_row = connection.execute(
            "SELECT api_usage_count FROM odds_api_usage"
        ).fetchone()

    assert stored_snapshots == (2,)
    assert usage_row == (2,)


def test_fetch_mlb_odds_returns_empty_list_when_f5_markets_missing(tmp_path: Path) -> None:
    from src.clients.odds_client import fetch_mlb_odds

    db_path = tmp_path / "odds.db"
    init_db(db_path)
    _seed_game(db_path)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/events"):
            return httpx.Response(
                200,
                json=[
                    {
                        "id": "evt-123",
                        "sport_key": "baseball_mlb",
                        "sport_title": "MLB",
                        "commence_time": "2026-04-15T20:05:00Z",
                        "home_team": "New York Yankees",
                        "away_team": "Boston Red Sox",
                    }
                ],
            )

        return httpx.Response(
            200,
            json={
                "id": "evt-123",
                "sport_key": "baseball_mlb",
                "sport_title": "MLB",
                "commence_time": "2026-04-15T20:05:00Z",
                "home_team": "New York Yankees",
                "away_team": "Boston Red Sox",
                "bookmakers": [
                    {
                        "key": "draftkings",
                        "title": "DraftKings",
                        "last_update": "2026-04-15T16:00:00Z",
                        "markets": [],
                    }
                ],
            },
            headers={
                "x-requests-last": "0",
                "x-requests-used": "0",
                "x-requests-remaining": "500",
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.the-odds-api.com")

    snapshots = fetch_mlb_odds(
        api_key="test-key",
        db_path=db_path,
        client=client,
        commence_time_from=datetime(2026, 4, 15, 0, 0, tzinfo=UTC),
        commence_time_to=datetime(2026, 4, 16, 0, 0, tzinfo=UTC),
    )

    assert snapshots == []


def test_fetch_mlb_odds_blocks_when_monthly_limit_would_be_exceeded(tmp_path: Path) -> None:
    from src.clients.odds_client import OddsApiRateLimitError, fetch_mlb_odds

    db_path = tmp_path / "odds.db"
    init_db(db_path)
    _seed_game(db_path)

    current_month = datetime.now(UTC).strftime("%Y-%m")
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS odds_api_usage (
                usage_month TEXT PRIMARY KEY,
                api_usage_count INTEGER NOT NULL,
                quota_limit INTEGER NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            "INSERT INTO odds_api_usage (usage_month, api_usage_count, quota_limit) VALUES (?, ?, ?)",
            (current_month, 499, 500),
        )
        connection.commit()

    calls = {"event_odds": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/events"):
            return httpx.Response(
                200,
                json=[
                    {
                        "id": "evt-123",
                        "sport_key": "baseball_mlb",
                        "sport_title": "MLB",
                        "commence_time": "2026-04-15T20:05:00Z",
                        "home_team": "New York Yankees",
                        "away_team": "Boston Red Sox",
                    }
                ],
            )

        calls["event_odds"] += 1
        return httpx.Response(500)

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.the-odds-api.com")

    with pytest.raises(OddsApiRateLimitError):
        fetch_mlb_odds(
            api_key="test-key",
            db_path=db_path,
            client=client,
            commence_time_from=datetime(2026, 4, 15, 0, 0, tzinfo=UTC),
            commence_time_to=datetime(2026, 4, 16, 0, 0, tzinfo=UTC),
        )

    assert calls["event_odds"] == 0


def test_freeze_odds_marks_snapshots_as_frozen(tmp_path: Path) -> None:
    from src.clients.odds_client import freeze_odds

    db_path = tmp_path / "odds.db"
    init_db(db_path)
    _seed_game(db_path)

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO odds_snapshots (
                game_pk,
                book_name,
                market_type,
                home_odds,
                away_odds,
                fetched_at,
                is_frozen
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (12345, "DraftKings", "f5_ml", -135, 115, "2026-04-15T16:00:00+00:00", 0),
        )
        connection.commit()

    updated_rows = freeze_odds(12345, db_path=db_path)

    with sqlite3.connect(db_path) as connection:
        frozen_row = connection.execute(
            "SELECT is_frozen FROM odds_snapshots WHERE game_pk = ?",
            (12345,),
        ).fetchone()

    assert updated_rows == 1
    assert frozen_row == (1,)
