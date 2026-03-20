from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from src.db import init_db
from src.engine.edge_calculator import (
    DEFAULT_EDGE_THRESHOLD,
    american_to_implied,
    calculate_edge,
    devig_probabilities,
    expected_value,
)


UTC = timezone.utc
GAME_PK = 12345


def seed_game(db_path: Path) -> None:
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO games (game_pk, date, home_team, away_team, venue, status)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                GAME_PK,
                "2026-04-15T20:05:00+00:00",
                "NYY",
                "BOS",
                "Yankee Stadium",
                "scheduled",
            ),
        )
        connection.commit()


def build_probe(output_path: Path, db_path: Path) -> dict[str, object]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if db_path.exists():
        db_path.unlink()

    init_db(db_path)
    seed_game(db_path)

    minus_150 = american_to_implied(-150)
    plus_130 = american_to_implied(130)
    plus_100 = american_to_implied(100)
    minus_100 = american_to_implied(-100)

    fair_home, fair_away = devig_probabilities(-150, 130)
    even_home, even_away = devig_probabilities(100, -100)

    known_edge = calculate_edge(
        game_pk=GAME_PK,
        market_type="f5_ml",
        side="home",
        model_probability=0.58,
        home_odds=-150,
        away_odds=100,
        book_name="DraftKings",
        db_path=db_path,
        calculated_at=datetime(2026, 4, 15, 16, 0, tzinfo=UTC),
    )
    threshold_edge = calculate_edge(
        game_pk=GAME_PK,
        market_type="f5_ml",
        side="home",
        model_probability=0.53,
        home_odds=100,
        away_odds=-100,
        book_name="DraftKings",
        db_path=db_path,
        calculated_at=datetime(2026, 4, 15, 16, 5, tzinfo=UTC),
    )
    subthreshold_edge = calculate_edge(
        game_pk=GAME_PK,
        market_type="f5_ml",
        side="home",
        model_probability=0.529,
        home_odds=100,
        away_odds=-100,
        book_name="DraftKings",
        db_path=db_path,
        calculated_at=datetime(2026, 4, 15, 16, 10, tzinfo=UTC),
    )
    heavy_away = calculate_edge(
        game_pk=GAME_PK,
        market_type="f5_ml",
        side="away",
        model_probability=0.27,
        home_odds=-360,
        away_odds=330,
        book_name="FanDuel",
        db_path=db_path,
        calculated_at=datetime(2026, 4, 15, 16, 15, tzinfo=UTC),
    )

    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT side, edge_pct, ev, is_positive_ev, calculated_at
            FROM edge_calculations
            ORDER BY id
            """
        ).fetchall()

    payload = {
        "threshold": DEFAULT_EDGE_THRESHOLD,
        "assertions": {
            "VAL-ODDS-001": {
                "negative_minus_150": round(minus_150, 6),
                "positive_plus_130": round(plus_130, 6),
                "formula_negative": "abs(odds) / (abs(odds) + 100)",
                "formula_positive": "100 / (odds + 100)",
            },
            "VAL-ODDS-002": {
                "home_implied": round(minus_150, 6),
                "away_implied": round(plus_130, 6),
                "fair_home": round(fair_home, 6),
                "fair_away": round(fair_away, 6),
                "fair_sum": round(fair_home + fair_away, 6),
            },
            "VAL-ODDS-003": {
                "plus_100": round(plus_100, 6),
                "minus_100": round(minus_100, 6),
                "devig_plus_100_minus_100": [round(even_home, 6), round(even_away, 6)],
                "fair_sum": round(even_home + even_away, 6),
            },
            "VAL-ODDS-004": {
                "known_home_edge": known_edge.model_dump(mode="json"),
                "expected_edge": round(0.58 - (0.6 / 1.1), 6),
                "heavy_away_edge": heavy_away.model_dump(mode="json"),
            },
            "VAL-ODDS-005": {
                "threshold_edge": threshold_edge.model_dump(mode="json"),
                "subthreshold_edge": subthreshold_edge.model_dump(mode="json"),
                "threshold_enforced": threshold_edge.is_positive_ev and not subthreshold_edge.is_positive_ev,
            },
            "VAL-ODDS-006": {
                "ev_minus_150_model_0_58": round(expected_value(0.58, -150), 6),
                "ev_plus_130_model_0_58": round(expected_value(0.58, 130), 6),
                "formula": "(model_prob * profit) - ((1 - model_prob) * stake)",
            },
        },
        "edge_calculation_audit_rows": [
            {
                "side": row[0],
                "edge_pct": row[1],
                "ev": row[2],
                "is_positive_ev": bool(row[3]),
                "calculated_at": row[4],
            }
            for row in rows
        ],
    }

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--db-path", type=Path, required=True)
    args = parser.parse_args()

    payload = build_probe(args.output, args.db_path)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
