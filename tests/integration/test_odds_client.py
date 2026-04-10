from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from src.db import init_db


UTC = timezone.utc


def _seed_game(
    db_path: Path,
    game_pk: int = 12345,
    *,
    scheduled_start: str = "2026-04-15T20:05:00+00:00",
    home_team: str = "NYY",
    away_team: str = "BOS",
    venue: str = "Yankee Stadium",
) -> None:
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO games (game_pk, date, home_team, away_team, venue, status)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (game_pk, scheduled_start, home_team, away_team, venue, "scheduled"),
        )
        connection.commit()


def _build_event(
    event_id: str,
    *,
    commence_time: str,
    home_team: str,
    away_team: str,
    venue: str | None = None,
) -> dict[str, str]:
    payload = {
        "id": event_id,
        "sport_key": "baseball_mlb",
        "sport_title": "MLB",
        "commence_time": commence_time,
        "home_team": home_team,
        "away_team": away_team,
    }
    if venue is not None:
        payload["venue"] = venue

    return payload


def _build_event_odds_payload(
    event_id: str,
    *,
    commence_time: str,
    home_team: str,
    away_team: str,
) -> dict[str, object]:
    return {
        "id": event_id,
        "sport_key": "baseball_mlb",
        "sport_title": "MLB",
        "commence_time": commence_time,
        "home_team": home_team,
        "away_team": away_team,
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
                            {"name": home_team, "price": -150},
                            {"name": away_team, "price": 130},
                        ],
                    },
                    {
                        "key": "spreads_1st_5_innings",
                        "last_update": "2026-04-15T16:01:00Z",
                        "outcomes": [
                            {"name": home_team, "price": -105, "point": -0.5},
                            {"name": away_team, "price": -115, "point": 0.5},
                        ],
                    },
                ],
            }
        ],
    }


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


def test_build_estimated_f5_ml_snapshots_creates_labeled_preview_odds() -> None:
    from src.clients.odds_client import build_estimated_f5_ml_snapshots

    snapshots = build_estimated_f5_ml_snapshots(
        {
            12345: {
                "full_game_ml_pairs": [
                    {"book_name": "DraftKings", "home_odds": -150, "away_odds": 130}
                ]
            }
        },
        fetched_at=datetime(2026, 4, 15, 16, 0, tzinfo=UTC),
    )

    assert len(snapshots) == 1
    snapshot = snapshots[0]
    assert snapshot.game_pk == 12345
    assert snapshot.market_type == "f5_ml"
    assert snapshot.book_name == "estimate:full-game:DraftKings"
    assert snapshot.home_odds > -150
    assert snapshot.away_odds < 130


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
                    _build_event(
                        "evt-123",
                        commence_time="2026-04-15T20:05:00Z",
                        home_team="New York Yankees",
                        away_team="Boston Red Sox",
                    )
                ],
                headers={
                    "x-requests-last": "0",
                    "x-requests-used": "0",
                    "x-requests-remaining": "500",
                },
            )

        return httpx.Response(
            200,
            json=_build_event_odds_payload(
                "evt-123",
                commence_time="2026-04-15T20:05:00Z",
                home_team="New York Yankees",
                away_team="Boston Red Sox",
            ),
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
    rl_snapshot = next(snapshot for snapshot in snapshots if snapshot.market_type == "f5_rl")
    assert rl_snapshot.home_point == pytest.approx(-0.5)
    assert rl_snapshot.away_point == pytest.approx(0.5)

    with sqlite3.connect(db_path) as connection:
        stored_snapshots = connection.execute(
            "SELECT COUNT(*) FROM odds_snapshots WHERE game_pk = ?",
            (12345,),
        ).fetchone()
        stored_points = connection.execute(
            "SELECT home_point, away_point FROM odds_snapshots WHERE game_pk = ? AND market_type = 'f5_rl'",
            (12345,),
        ).fetchone()
        usage_row = connection.execute(
            "SELECT api_usage_count FROM odds_api_usage"
        ).fetchone()

    assert stored_snapshots == (2,)
    assert stored_points == (-0.5, 0.5)
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
                    _build_event(
                        "evt-123",
                        commence_time="2026-04-15T20:05:00Z",
                        home_team="New York Yankees",
                        away_team="Boston Red Sox",
                    )
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
                    _build_event(
                        "evt-123",
                        commence_time="2026-04-15T20:05:00Z",
                        home_team="New York Yankees",
                        away_team="Boston Red Sox",
                    )
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


def test_fetch_mlb_odds_creates_missing_game_row_and_defers_abs_when_venue_unknown(
    tmp_path: Path,
) -> None:
    from src.clients.odds_client import fetch_mlb_odds

    db_path = tmp_path / "odds.db"
    init_db(db_path)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/events"):
            return httpx.Response(
                200,
                json=[
                    _build_event(
                        "evt-123",
                        commence_time="2026-04-15T20:05:00Z",
                        home_team="New York Yankees",
                        away_team="Boston Red Sox",
                    )
                ],
            )

        return httpx.Response(
            200,
            json=_build_event_odds_payload(
                "evt-123",
                commence_time="2026-04-15T20:05:00Z",
                home_team="New York Yankees",
                away_team="Boston Red Sox",
            ),
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

    with sqlite3.connect(db_path) as connection:
        game_row = connection.execute(
            "SELECT game_pk, date, home_team, away_team, venue, is_abs_active, status FROM games"
        ).fetchone()

    assert len(snapshots) == 2
    assert game_row is not None
    assert game_row[0] == snapshots[0].game_pk
    assert game_row[1] == "2026-04-15T20:05:00+00:00"
    assert game_row[2:] == ("NYY", "BOS", "Yankee Stadium", 0, "scheduled")


def test_fetch_mlb_odds_uses_event_venue_metadata_for_abs_exception_games(tmp_path: Path) -> None:
    from src.clients.odds_client import fetch_mlb_odds

    db_path = tmp_path / "odds.db"
    init_db(db_path)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/events"):
            return httpx.Response(
                200,
                json=[
                    _build_event(
                        "evt-mexico-city",
                        commence_time="2026-04-25T23:05:00Z",
                        home_team="Houston Astros",
                        away_team="Colorado Rockies",
                        venue="Alfredo Harp Helu Stadium - Mexico City Series",
                    )
                ],
            )

        return httpx.Response(
            200,
            json=_build_event_odds_payload(
                "evt-mexico-city",
                commence_time="2026-04-25T23:05:00Z",
                home_team="Houston Astros",
                away_team="Colorado Rockies",
            ),
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
        commence_time_from=datetime(2026, 4, 25, 0, 0, tzinfo=UTC),
        commence_time_to=datetime(2026, 4, 26, 0, 0, tzinfo=UTC),
    )

    with sqlite3.connect(db_path) as connection:
        game_row = connection.execute(
            "SELECT venue, is_abs_active FROM games WHERE game_pk = ?",
            (snapshots[0].game_pk,),
        ).fetchone()

    assert len(snapshots) == 2
    assert game_row == ("Alfredo Harp Helu Stadium - Mexico City Series", 0)


def test_fetch_mlb_odds_commits_each_event_before_later_failure(tmp_path: Path) -> None:
    from src.clients.odds_client import fetch_mlb_odds

    db_path = tmp_path / "odds.db"
    init_db(db_path)
    _seed_game(db_path, 12345)
    _seed_game(
        db_path,
        67890,
        scheduled_start="2026-04-15T23:05:00+00:00",
        home_team="LAD",
        away_team="SF",
        venue="Dodger Stadium",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/events"):
            return httpx.Response(
                200,
                json=[
                    _build_event(
                        "evt-123",
                        commence_time="2026-04-15T20:05:00Z",
                        home_team="New York Yankees",
                        away_team="Boston Red Sox",
                    ),
                    _build_event(
                        "evt-456",
                        commence_time="2026-04-15T23:05:00Z",
                        home_team="Los Angeles Dodgers",
                        away_team="San Francisco Giants",
                    ),
                ],
            )

        if request.url.path.endswith("/evt-123/odds"):
            return httpx.Response(
                200,
                json=_build_event_odds_payload(
                    "evt-123",
                    commence_time="2026-04-15T20:05:00Z",
                    home_team="New York Yankees",
                    away_team="Boston Red Sox",
                ),
                headers={
                    "x-requests-last": "2",
                    "x-requests-used": "2",
                    "x-requests-remaining": "498",
                },
            )

        return httpx.Response(500, json={"message": "boom"})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.the-odds-api.com")

    with pytest.raises(httpx.HTTPStatusError):
        fetch_mlb_odds(
            api_key="test-key",
            db_path=db_path,
            client=client,
            commence_time_from=datetime(2026, 4, 15, 0, 0, tzinfo=UTC),
            commence_time_to=datetime(2026, 4, 16, 0, 0, tzinfo=UTC),
        )

    with sqlite3.connect(db_path) as connection:
        stored_counts = connection.execute(
            "SELECT game_pk, COUNT(*) FROM odds_snapshots GROUP BY game_pk ORDER BY game_pk"
        ).fetchall()
        usage_row = connection.execute(
            "SELECT api_usage_count FROM odds_api_usage"
        ).fetchone()

    assert stored_counts == [(12345, 2)]
    assert usage_row == (2,)
