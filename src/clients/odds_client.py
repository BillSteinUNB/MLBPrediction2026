from __future__ import annotations

import os
import sqlite3
import unicodedata
from contextlib import nullcontext
from datetime import datetime, timezone
from hashlib import blake2b
from math import ceil
from pathlib import Path
from typing import Any, Mapping, Sequence

import httpx
from dotenv import load_dotenv

from src.config import DEFAULT_ENV_FILE, _load_settings_yaml
from src.db import DEFAULT_DB_PATH, init_db
from src.features.adjustments.abs_adjustment import is_abs_active
from src.models.odds import OddsSnapshot


ODDS_API_BASE_URL = "https://api.the-odds-api.com"
ODDS_API_SPORT_KEY = "baseball_mlb"
ODDS_API_MONTHLY_LIMIT = 500
ODDS_API_EVENTS_PATH = f"/v4/sports/{ODDS_API_SPORT_KEY}/events"
ODDS_API_F5_MARKETS = {
    "h2h_1st_5_innings": "f5_ml",
    "spreads_1st_5_innings": "f5_rl",
}
_SETTINGS_PAYLOAD = _load_settings_yaml()


class OddsApiError(RuntimeError):
    """Base exception for odds client failures."""


class OddsApiRateLimitError(OddsApiError):
    """Raised when the configured monthly quota would be exceeded."""


def _build_team_name_index() -> dict[str, str]:
    teams = _SETTINGS_PAYLOAD["teams"]
    team_name_to_code = {
        team_payload["full_name"]: team_code
        for team_code, team_payload in teams.items()
    }
    team_name_to_code.setdefault("Oakland Athletics", "OAK")
    return team_name_to_code


TEAM_NAME_TO_CODE = _build_team_name_index()
STADIUMS_BY_TEAM_CODE = _SETTINGS_PAYLOAD["stadiums"]


def _parse_iso_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _normalize_datetime(value: datetime) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError("datetime values must be timezone-aware")

    return value.astimezone(timezone.utc)


def _to_iso_z(value: datetime) -> str:
    return _normalize_datetime(value).isoformat().replace("+00:00", "Z")


def _resolve_api_key(api_key: str | None) -> str:
    if api_key:
        return api_key

    load_dotenv(DEFAULT_ENV_FILE)
    resolved_api_key = os.getenv("ODDS_API_KEY")
    if not resolved_api_key:
        raise OddsApiError("ODDS_API_KEY is required to fetch odds")

    return resolved_api_key


def _usage_month(now: datetime | None = None) -> str:
    current_time = now or datetime.now(timezone.utc)
    return current_time.astimezone(timezone.utc).strftime("%Y-%m")


def _ensure_usage_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS odds_api_usage (
            usage_month TEXT PRIMARY KEY,
            api_usage_count INTEGER NOT NULL DEFAULT 0 CHECK (api_usage_count >= 0),
            quota_limit INTEGER NOT NULL DEFAULT 500 CHECK (quota_limit > 0),
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


def _get_usage_count(connection: sqlite3.Connection, usage_month: str) -> int:
    _ensure_usage_table(connection)
    row = connection.execute(
        "SELECT api_usage_count FROM odds_api_usage WHERE usage_month = ?",
        (usage_month,),
    ).fetchone()
    return int(row[0]) if row else 0


def _set_usage_count(
    connection: sqlite3.Connection,
    usage_month: str,
    usage_count: int,
    quota_limit: int = ODDS_API_MONTHLY_LIMIT,
) -> None:
    _ensure_usage_table(connection)
    connection.execute(
        """
        INSERT INTO odds_api_usage (usage_month, api_usage_count, quota_limit)
        VALUES (?, ?, ?)
        ON CONFLICT(usage_month) DO UPDATE SET
            api_usage_count = excluded.api_usage_count,
            quota_limit = excluded.quota_limit,
            updated_at = CURRENT_TIMESTAMP
        """,
        (usage_month, usage_count, quota_limit),
    )


def _record_usage(
    connection: sqlite3.Connection,
    *,
    usage_month: str,
    headers: Mapping[str, str],
    estimated_cost: int,
    quota_limit: int,
) -> None:
    current_usage = _get_usage_count(connection, usage_month)
    header_usage = headers.get("x-requests-used")
    header_last = headers.get("x-requests-last")

    if header_usage is not None:
        next_usage = max(current_usage, int(header_usage))
    elif header_last is not None:
        next_usage = current_usage + int(header_last)
    else:
        next_usage = current_usage + estimated_cost

    _set_usage_count(connection, usage_month, next_usage, quota_limit)


def _quota_cost(regions: str, bookmakers: Sequence[str] | None) -> int:
    if bookmakers:
        return max(1, ceil(len(bookmakers) / 10))

    region_tokens = [region.strip() for region in regions.split(",") if region.strip()]
    return max(1, len(region_tokens))


def _ensure_quota_available(
    connection: sqlite3.Connection,
    *,
    usage_month: str,
    additional_cost: int,
    quota_limit: int,
) -> None:
    current_usage = _get_usage_count(connection, usage_month)
    if current_usage + additional_cost > quota_limit:
        raise OddsApiRateLimitError(
            f"Odds API monthly limit would be exceeded: {current_usage} used + {additional_cost} requested > {quota_limit}"
        )


def _resolve_game_pk(
    connection: sqlite3.Connection,
    *,
    home_team_name: str,
    away_team_name: str,
    commence_time: datetime,
) -> int | None:
    home_team_code = TEAM_NAME_TO_CODE.get(home_team_name)
    away_team_code = TEAM_NAME_TO_CODE.get(away_team_name)
    if home_team_code is None or away_team_code is None:
        return None

    game_date = commence_time.date().isoformat()
    row = connection.execute(
        """
        SELECT game_pk
        FROM games
        WHERE home_team = ?
          AND away_team = ?
          AND date LIKE ?
        ORDER BY CASE WHEN game_pk > 0 THEN 0 ELSE 1 END ASC,
                 ABS(julianday(date) - julianday(?)) ASC,
                 date ASC
        LIMIT 1
        """,
        (home_team_code, away_team_code, f"{game_date}%", commence_time.isoformat()),
    ).fetchone()
    return int(row[0]) if row else None


def _fallback_game_pk(event_id: str) -> int:
    digest = blake2b(event_id.encode("utf-8"), digest_size=8).digest()
    return -((int.from_bytes(digest, byteorder="big") & ((1 << 63) - 1)) or 1)


def _normalize_text(value: str | None) -> str:
    if value is None:
        return ""

    normalized = unicodedata.normalize("NFKD", str(value))
    without_marks = "".join(character for character in normalized if not unicodedata.combining(character))
    return " ".join(without_marks.casefold().split())


def _extract_event_venue(event: Mapping[str, Any] | None) -> str | None:
    if not isinstance(event, Mapping):
        return None

    for field_name in ("venue", "stadium", "site"):
        candidate = event.get(field_name)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()

        if isinstance(candidate, Mapping):
            for nested_field_name in ("name", "full_name", "display_name"):
                nested_candidate = candidate.get(nested_field_name)
                if isinstance(nested_candidate, str) and nested_candidate.strip():
                    return nested_candidate.strip()

    return None


def _resolve_is_dome_for_venue(venue: str, *, fallback_stadium: Mapping[str, Any] | None) -> bool:
    normalized_venue = _normalize_text(venue)
    if not normalized_venue:
        return False

    candidate_stadiums: list[Mapping[str, Any]] = []
    if isinstance(fallback_stadium, Mapping):
        candidate_stadiums.append(fallback_stadium)

    candidate_stadiums.extend(
        stadium for stadium in STADIUMS_BY_TEAM_CODE.values() if isinstance(stadium, Mapping)
    )

    for stadium in candidate_stadiums:
        park_name = stadium.get("park_name")
        if isinstance(park_name, str) and _normalize_text(park_name) == normalized_venue:
            return bool(stadium.get("is_dome"))

    return False


def _resolve_backfill_game_context(
    *,
    home_team_name: str,
    home_team_code: str,
    event_venue: str | None,
) -> tuple[str, bool, bool]:
    stadium = STADIUMS_BY_TEAM_CODE.get(home_team_code, {})
    fallback_venue = stadium.get("park_name") if isinstance(stadium, Mapping) else None
    fallback_is_dome = bool(stadium.get("is_dome")) if isinstance(stadium, Mapping) else False

    if isinstance(event_venue, str) and event_venue.strip():
        resolved_venue = event_venue.strip()
        return (
            resolved_venue,
            _resolve_is_dome_for_venue(resolved_venue, fallback_stadium=stadium),
            is_abs_active(resolved_venue),
        )

    if isinstance(fallback_venue, str) and fallback_venue:
        return fallback_venue, fallback_is_dome, False

    return home_team_name, False, False


def _refresh_synthetic_game_row(
    connection: sqlite3.Connection,
    *,
    game_pk: int,
    home_team_name: str,
    home_team_code: str,
    event_venue: str | None,
) -> None:
    venue, is_dome, abs_active = _resolve_backfill_game_context(
        home_team_name=home_team_name,
        home_team_code=home_team_code,
        event_venue=event_venue,
    )
    connection.execute(
        """
        UPDATE games
        SET venue = ?,
            is_dome = ?,
            is_abs_active = ?
        WHERE game_pk = ?
        """,
        (venue, int(is_dome), int(abs_active), game_pk),
    )


def _ensure_game_row(
    connection: sqlite3.Connection,
    *,
    event_id: str,
    home_team_name: str,
    away_team_name: str,
    commence_time: datetime,
    event_venue: str | None = None,
) -> int | None:
    existing_game_pk = _resolve_game_pk(
        connection,
        home_team_name=home_team_name,
        away_team_name=away_team_name,
        commence_time=commence_time,
    )
    home_team_code = TEAM_NAME_TO_CODE.get(home_team_name)
    away_team_code = TEAM_NAME_TO_CODE.get(away_team_name)
    if existing_game_pk is not None:
        if existing_game_pk < 0 and home_team_code is not None:
            _refresh_synthetic_game_row(
                connection,
                game_pk=existing_game_pk,
                home_team_name=home_team_name,
                home_team_code=home_team_code,
                event_venue=event_venue,
            )
        return existing_game_pk

    if home_team_code is None or away_team_code is None:
        return None

    venue, is_dome, abs_active = _resolve_backfill_game_context(
        home_team_name=home_team_name,
        home_team_code=home_team_code,
        event_venue=event_venue,
    )
    game_pk = _fallback_game_pk(event_id)

    existing_row = connection.execute(
        "SELECT date, home_team, away_team FROM games WHERE game_pk = ?",
        (game_pk,),
    ).fetchone()
    if existing_row is not None:
        if existing_row == (commence_time.isoformat(), home_team_code, away_team_code):
            return game_pk
        raise OddsApiError(f"Synthetic game_pk collision while backfilling event {event_id}")

    connection.execute(
        """
        INSERT INTO games (
            game_pk,
            date,
            home_team,
            away_team,
            venue,
            is_dome,
            is_abs_active,
            status
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            game_pk,
            commence_time.isoformat(),
            home_team_code,
            away_team_code,
            venue,
            int(is_dome),
            int(abs_active),
            "scheduled",
        ),
    )
    return game_pk


def _extract_market_prices(
    outcomes: list[dict[str, Any]],
    *,
    home_team_name: str,
    away_team_name: str,
) -> tuple[int, int] | None:
    prices_by_team: dict[str, int] = {}
    for outcome in outcomes:
        team_name = outcome.get("name")
        price = outcome.get("price")
        if team_name in {home_team_name, away_team_name} and isinstance(price, (int, float)):
            prices_by_team[team_name] = int(price)

    if home_team_name not in prices_by_team or away_team_name not in prices_by_team:
        return None

    return prices_by_team[home_team_name], prices_by_team[away_team_name]


def _persist_snapshots(connection: sqlite3.Connection, snapshots: Sequence[OddsSnapshot]) -> None:
    connection.executemany(
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
        [
            (
                snapshot.game_pk,
                snapshot.book_name,
                snapshot.market_type,
                snapshot.home_odds,
                snapshot.away_odds,
                snapshot.fetched_at.isoformat(),
                int(snapshot.is_frozen),
            )
            for snapshot in snapshots
        ],
    )


def american_to_implied(odds: int) -> float:
    """Convert American odds to implied probability."""

    if odds > 0:
        return 100 / (odds + 100)

    return abs(odds) / (abs(odds) + 100)


def devig_probabilities(home_odds: int, away_odds: int) -> tuple[float, float]:
    """Apply proportional de-vig so fair probabilities sum to 1.0."""

    home_implied = american_to_implied(home_odds)
    away_implied = american_to_implied(away_odds)
    total_implied = home_implied + away_implied

    if total_implied == 0:
        raise ValueError("Cannot de-vig odds with zero implied probability")

    return home_implied / total_implied, away_implied / total_implied


def get_monthly_usage(
    db_path: str | Path = DEFAULT_DB_PATH,
    *,
    month: str | None = None,
) -> int:
    """Return the tracked Odds API usage for the requested month."""

    init_db(db_path)
    usage_month = month or _usage_month()
    with sqlite3.connect(db_path) as connection:
        return _get_usage_count(connection, usage_month)


def fetch_mlb_odds(
    *,
    api_key: str | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
    client: httpx.Client | None = None,
    regions: str = "us",
    bookmakers: Sequence[str] | None = None,
    commence_time_from: datetime | None = None,
    commence_time_to: datetime | None = None,
    quota_limit: int = ODDS_API_MONTHLY_LIMIT,
) -> list[OddsSnapshot]:
    """Fetch MLB F5 odds, persist snapshots, and track monthly API usage."""

    resolved_api_key = _resolve_api_key(api_key)
    database_path = init_db(db_path)
    usage_month = _usage_month()

    events_params: dict[str, str] = {
        "apiKey": resolved_api_key,
        "dateFormat": "iso",
    }
    if commence_time_from is not None:
        events_params["commenceTimeFrom"] = _to_iso_z(commence_time_from)
    if commence_time_to is not None:
        events_params["commenceTimeTo"] = _to_iso_z(commence_time_to)

    client_context = nullcontext(client) if client is not None else httpx.Client(base_url=ODDS_API_BASE_URL, timeout=30.0)
    snapshots: list[OddsSnapshot] = []
    per_event_cost = len(ODDS_API_F5_MARKETS) * _quota_cost(regions, bookmakers)

    with client_context as http_client, sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        _ensure_usage_table(connection)

        events_response = http_client.get(ODDS_API_EVENTS_PATH, params=events_params)
        events_response.raise_for_status()

        events_payload = events_response.json()
        if not isinstance(events_payload, list):
            raise OddsApiError("Unexpected events response payload")

        for event in events_payload:
            commence_time = _parse_iso_datetime(event["commence_time"])
            try:
                game_pk = _ensure_game_row(
                    connection,
                    event_id=event["id"],
                    home_team_name=event["home_team"],
                    away_team_name=event["away_team"],
                    commence_time=commence_time,
                    event_venue=_extract_event_venue(event),
                )
                if game_pk is None:
                    continue

                _ensure_quota_available(
                    connection,
                    usage_month=usage_month,
                    additional_cost=per_event_cost,
                    quota_limit=quota_limit,
                )

                event_params: dict[str, str] = {
                    "apiKey": resolved_api_key,
                    "regions": regions,
                    "markets": ",".join(ODDS_API_F5_MARKETS),
                    "oddsFormat": "american",
                    "dateFormat": "iso",
                }
                if bookmakers:
                    event_params["bookmakers"] = ",".join(bookmakers)

                event_response = http_client.get(
                    f"/v4/sports/{ODDS_API_SPORT_KEY}/events/{event['id']}/odds",
                    params=event_params,
                )
                event_response.raise_for_status()
                event_payload = event_response.json()
                if not isinstance(event_payload, dict):
                    raise OddsApiError("Unexpected event odds response payload")

                event_snapshots: list[OddsSnapshot] = []
                for bookmaker in event_payload.get("bookmakers", []):
                    book_name = bookmaker.get("title") or bookmaker.get("key")
                    if not isinstance(book_name, str) or not book_name:
                        continue

                    for market in bookmaker.get("markets", []):
                        market_key = market.get("key")
                        market_type = ODDS_API_F5_MARKETS.get(market_key)
                        if market_type is None:
                            continue

                        prices = _extract_market_prices(
                            market.get("outcomes", []),
                            home_team_name=event["home_team"],
                            away_team_name=event["away_team"],
                        )
                        if prices is None:
                            continue

                        fetched_at_raw = (
                            market.get("last_update")
                            or bookmaker.get("last_update")
                            or event.get("commence_time")
                        )
                        if not isinstance(fetched_at_raw, str):
                            continue

                        event_snapshots.append(
                            OddsSnapshot(
                                game_pk=game_pk,
                                book_name=book_name,
                                market_type=market_type,
                                home_odds=prices[0],
                                away_odds=prices[1],
                                fetched_at=_parse_iso_datetime(fetched_at_raw),
                            )
                        )

                _record_usage(
                    connection,
                    usage_month=usage_month,
                    headers=dict(event_response.headers),
                    estimated_cost=per_event_cost,
                    quota_limit=quota_limit,
                )

                if event_snapshots:
                    _persist_snapshots(connection, event_snapshots)
                    snapshots.extend(event_snapshots)

                connection.commit()
            except Exception:
                connection.rollback()
                raise

    return sorted(snapshots, key=lambda snapshot: (snapshot.game_pk, snapshot.book_name, snapshot.market_type))


def freeze_odds(
    game_pk: int,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    connection: sqlite3.Connection | None = None,
    market_type: str | None = None,
    commit: bool = True,
) -> int:
    """Mark persisted odds snapshots as frozen after notification."""

    database_path = init_db(db_path) if connection is None else Path(db_path)
    owns_connection = connection is None
    resolved_connection = connection or sqlite3.connect(database_path)

    try:
        resolved_connection.execute("PRAGMA foreign_keys = ON")
        if market_type is None:
            cursor = resolved_connection.execute(
                "UPDATE odds_snapshots SET is_frozen = 1 WHERE game_pk = ?",
                (game_pk,),
            )
        else:
            cursor = resolved_connection.execute(
                "UPDATE odds_snapshots SET is_frozen = 1 WHERE game_pk = ? AND market_type = ?",
                (game_pk, market_type),
            )
        if commit:
            resolved_connection.commit()
        return int(cursor.rowcount)
    except Exception:
        if owns_connection:
            resolved_connection.rollback()
        raise
    finally:
        if owns_connection:
            resolved_connection.close()
