from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from pydantic import TypeAdapter

from src.clients.odds_client import american_to_implied, devig_probabilities
from src.config import _load_settings_yaml
from src.db import DEFAULT_DB_PATH, init_db
from src.models._base import AmericanOdds, Probability
from src.models.bet import BetDecision


_SETTINGS_PAYLOAD = _load_settings_yaml()
DEFAULT_EDGE_THRESHOLD = float(_SETTINGS_PAYLOAD["thresholds"]["edge_min"])

_AMERICAN_ODDS_ADAPTER = TypeAdapter(AmericanOdds)
_PROBABILITY_ADAPTER = TypeAdapter(Probability)
logger = logging.getLogger(__name__)


def _ensure_edge_calculations_table(db_path: str | Path) -> Path:
    """Create the SQLite edge calculation audit table if it does not exist."""

    database_path = init_db(db_path)
    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS edge_calculations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_pk INTEGER NOT NULL,
                market_type TEXT NOT NULL CHECK (market_type IN ('f5_ml', 'f5_rl')),
                side TEXT NOT NULL CHECK (side IN ('home', 'away')),
                book_name TEXT NOT NULL,
                model_probability REAL NOT NULL CHECK (model_probability BETWEEN 0 AND 1),
                fair_probability REAL NOT NULL CHECK (fair_probability BETWEEN 0 AND 1),
                home_implied_probability REAL NOT NULL CHECK (
                    home_implied_probability BETWEEN 0 AND 1
                ),
                away_implied_probability REAL NOT NULL CHECK (
                    away_implied_probability BETWEEN 0 AND 1
                ),
                home_fair_probability REAL NOT NULL CHECK (
                    home_fair_probability BETWEEN 0 AND 1
                ),
                away_fair_probability REAL NOT NULL CHECK (
                    away_fair_probability BETWEEN 0 AND 1
                ),
                edge_pct REAL NOT NULL,
                ev REAL NOT NULL,
                edge_threshold REAL NOT NULL CHECK (edge_threshold BETWEEN 0 AND 1),
                is_positive_ev INTEGER NOT NULL CHECK (is_positive_ev IN (0, 1)),
                odds_at_bet INTEGER NOT NULL,
                home_odds INTEGER NOT NULL,
                away_odds INTEGER NOT NULL,
                calculated_at TEXT NOT NULL,
                FOREIGN KEY (game_pk) REFERENCES games (game_pk)
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_edge_calculations_game_market
            ON edge_calculations (game_pk, market_type)
            """
        )
        connection.commit()

    return database_path


def _normalize_timestamp(value: datetime | None) -> datetime:
    """Return a timezone-aware UTC timestamp for persistence."""

    resolved = value or datetime.now(timezone.utc)
    if resolved.tzinfo is None or resolved.tzinfo.utcoffset(resolved) is None:
        raise ValueError("calculated_at must be timezone-aware")

    return resolved.astimezone(timezone.utc)


def payout_for_american_odds(odds: int) -> float:
    """Return net profit on a 1-unit stake for the provided American odds."""

    validated_odds = _AMERICAN_ODDS_ADAPTER.validate_python(odds)
    if validated_odds >= 100:
        return float(validated_odds) / 100.0

    return 100.0 / abs(validated_odds)


def expected_value(model_probability: float, odds: int, *, stake: float = 1.0) -> float:
    """Compute expected value using the American-odds payout structure."""

    validated_probability = _PROBABILITY_ADAPTER.validate_python(model_probability)
    if stake <= 0:
        raise ValueError("stake must be positive")

    profit = payout_for_american_odds(odds) * stake
    return float((validated_probability * profit) - ((1.0 - validated_probability) * stake))


def _resolve_threshold(edge_threshold: float | None) -> float:
    resolved_threshold = DEFAULT_EDGE_THRESHOLD if edge_threshold is None else float(edge_threshold)
    if not 0 <= resolved_threshold <= 1:
        raise ValueError("edge_threshold must be between 0 and 1")
    return resolved_threshold


def _log_edge_calculation(
    *,
    db_path: str | Path,
    decision: BetDecision,
    book_name: str,
    home_odds: int,
    away_odds: int,
    home_implied_probability: float,
    away_implied_probability: float,
    home_fair_probability: float,
    away_fair_probability: float,
    edge_threshold: float,
    calculated_at: datetime,
) -> None:
    if decision.market_type not in {"f5_ml", "f5_rl"}:
        return
    if decision.side not in {"home", "away"}:
        return
    try:
        database_path = _ensure_edge_calculations_table(db_path)

        with sqlite3.connect(database_path) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(
                """
                INSERT INTO edge_calculations (
                    game_pk,
                    market_type,
                    side,
                    book_name,
                    model_probability,
                    fair_probability,
                    home_implied_probability,
                    away_implied_probability,
                    home_fair_probability,
                    away_fair_probability,
                    edge_pct,
                    ev,
                    edge_threshold,
                    is_positive_ev,
                    odds_at_bet,
                    home_odds,
                    away_odds,
                    calculated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision.game_pk,
                    decision.market_type,
                    decision.side,
                    book_name,
                    decision.model_probability,
                    decision.fair_probability,
                    home_implied_probability,
                    away_implied_probability,
                    home_fair_probability,
                    away_fair_probability,
                    decision.edge_pct,
                    decision.ev,
                    edge_threshold,
                    int(decision.is_positive_ev),
                    decision.odds_at_bet,
                    home_odds,
                    away_odds,
                    calculated_at.isoformat(),
                ),
            )
            connection.commit()
    except sqlite3.DatabaseError:
        logger.warning(
            "Skipping edge_calculations audit logging due to SQLite database error",
            exc_info=True,
        )


def calculate_edge(
    *,
    game_pk: int,
    market_type: str,
    side: str,
    model_probability: float,
    home_odds: int,
    away_odds: int,
    home_point: float | None = None,
    away_point: float | None = None,
    book_name: str = "manual",
    db_path: str | Path = DEFAULT_DB_PATH,
    edge_threshold: float | None = None,
    calculated_at: datetime | None = None,
) -> BetDecision:
    """Calculate edge, expected value, recommendation status, and log the result."""

    validated_model_probability = _PROBABILITY_ADAPTER.validate_python(model_probability)
    validated_home_odds = _AMERICAN_ODDS_ADAPTER.validate_python(home_odds)
    validated_away_odds = _AMERICAN_ODDS_ADAPTER.validate_python(away_odds)
    resolved_threshold = _resolve_threshold(edge_threshold)
    normalized_timestamp = _normalize_timestamp(calculated_at)
    resolved_book_name = book_name.strip() or "manual"

    home_implied_probability = american_to_implied(validated_home_odds)
    away_implied_probability = american_to_implied(validated_away_odds)
    home_fair_probability, away_fair_probability = devig_probabilities(
        validated_home_odds,
        validated_away_odds,
    )

    if side in {"home", "over"}:
        fair_probability = home_fair_probability
        odds_at_bet = validated_home_odds
        line_at_bet = home_point
    else:
        fair_probability = away_fair_probability
        odds_at_bet = validated_away_odds
        line_at_bet = away_point

    edge_pct = float(validated_model_probability - fair_probability)
    ev = expected_value(validated_model_probability, odds_at_bet)
    is_positive_ev = bool(edge_pct >= resolved_threshold)

    decision = BetDecision(
        game_pk=game_pk,
        market_type=market_type,
        side=side,
        book_name=resolved_book_name,
        model_probability=validated_model_probability,
        fair_probability=fair_probability,
        edge_pct=edge_pct,
        ev=ev,
        is_positive_ev=is_positive_ev,
        odds_at_bet=odds_at_bet,
        line_at_bet=line_at_bet,
    )

    _log_edge_calculation(
        db_path=db_path,
        decision=decision,
        book_name=resolved_book_name,
        home_odds=validated_home_odds,
        away_odds=validated_away_odds,
        home_implied_probability=home_implied_probability,
        away_implied_probability=away_implied_probability,
        home_fair_probability=home_fair_probability,
        away_fair_probability=away_fair_probability,
        edge_threshold=resolved_threshold,
        calculated_at=normalized_timestamp,
    )

    return decision


__all__ = [
    "DEFAULT_EDGE_THRESHOLD",
    "american_to_implied",
    "calculate_edge",
    "devig_probabilities",
    "expected_value",
    "payout_for_american_odds",
]
