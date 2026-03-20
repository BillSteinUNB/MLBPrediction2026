from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

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


def test_american_to_implied_handles_even_and_plus_minus_three_hundred_edges() -> None:
    from src.engine.edge_calculator import american_to_implied

    assert american_to_implied(100) == pytest.approx(0.5)
    assert american_to_implied(-100) == pytest.approx(0.5)
    assert american_to_implied(-330) == pytest.approx(330 / 430)
    assert american_to_implied(330) == pytest.approx(100 / 430)


def test_devig_probabilities_sum_to_one_for_even_and_heavy_lines() -> None:
    from src.engine.edge_calculator import devig_probabilities

    even_home, even_away = devig_probabilities(100, -100)
    heavy_home, heavy_away = devig_probabilities(-330, 300)

    assert even_home == pytest.approx(0.5)
    assert even_away == pytest.approx(0.5)
    assert even_home + even_away == pytest.approx(1.0)
    assert heavy_home + heavy_away == pytest.approx(1.0)
    assert heavy_home > heavy_away


def test_calculate_edge_matches_known_home_edge_example(tmp_path: Path) -> None:
    from src.engine.edge_calculator import calculate_edge

    db_path = tmp_path / "edge.db"
    init_db(db_path)
    _seed_game(db_path)

    decision = calculate_edge(
        game_pk=12345,
        market_type="f5_ml",
        side="home",
        model_probability=0.58,
        home_odds=-150,
        away_odds=100,
        book_name="DraftKings",
        db_path=db_path,
        calculated_at=datetime(2026, 4, 15, 16, 0, tzinfo=UTC),
    )

    assert decision.edge_pct == pytest.approx(0.58 - (0.6 / 1.1))
    assert decision.edge_pct == pytest.approx(0.0345454545)
    assert decision.ev == pytest.approx((0.58 * (100 / 150)) - (0.42 * 1.0))
    assert decision.is_positive_ev is False


def test_calculate_edge_recommends_bets_at_or_above_three_percent_threshold(tmp_path: Path) -> None:
    from src.engine.edge_calculator import calculate_edge

    db_path = tmp_path / "edge.db"
    init_db(db_path)
    _seed_game(db_path)

    decision = calculate_edge(
        game_pk=12345,
        market_type="f5_ml",
        side="home",
        model_probability=0.53,
        home_odds=100,
        away_odds=-100,
        book_name="DraftKings",
        db_path=db_path,
    )

    assert decision.edge_pct == pytest.approx(0.03)
    assert decision.ev == pytest.approx(0.06)
    assert decision.is_positive_ev is True


def test_calculate_edge_filters_subthreshold_edges(tmp_path: Path) -> None:
    from src.engine.edge_calculator import calculate_edge

    db_path = tmp_path / "edge.db"
    init_db(db_path)
    _seed_game(db_path)

    decision = calculate_edge(
        game_pk=12345,
        market_type="f5_ml",
        side="home",
        model_probability=0.529,
        home_odds=100,
        away_odds=-100,
        book_name="DraftKings",
        db_path=db_path,
    )

    assert decision.edge_pct == pytest.approx(0.029)
    assert decision.ev == pytest.approx(0.058)
    assert decision.is_positive_ev is False


def test_calculate_edge_handles_heavy_underdog_side(tmp_path: Path) -> None:
    from src.engine.edge_calculator import calculate_edge

    db_path = tmp_path / "edge.db"
    init_db(db_path)
    _seed_game(db_path)

    decision = calculate_edge(
        game_pk=12345,
        market_type="f5_ml",
        side="away",
        model_probability=0.27,
        home_odds=-360,
        away_odds=330,
        book_name="FanDuel",
        db_path=db_path,
    )

    assert decision.side == "away"
    assert decision.fair_probability == pytest.approx((100 / 430) / ((360 / 460) + (100 / 430)))
    assert decision.edge_pct == pytest.approx(
        0.27 - ((100 / 430) / ((360 / 460) + (100 / 430)))
    )
    assert decision.ev == pytest.approx((0.27 * 3.3) - (0.73 * 1.0))
    assert decision.is_positive_ev is True


def test_calculate_edge_logs_each_calculation_to_sqlite(tmp_path: Path) -> None:
    from src.engine.edge_calculator import calculate_edge

    db_path = tmp_path / "edge.db"
    init_db(db_path)
    _seed_game(db_path)

    calculate_edge(
        game_pk=12345,
        market_type="f5_ml",
        side="home",
        model_probability=0.53,
        home_odds=100,
        away_odds=-100,
        book_name="DraftKings",
        db_path=db_path,
        calculated_at=datetime(2026, 4, 15, 16, 0, tzinfo=UTC),
    )
    calculate_edge(
        game_pk=12345,
        market_type="f5_ml",
        side="away",
        model_probability=0.47,
        home_odds=100,
        away_odds=-100,
        book_name="DraftKings",
        db_path=db_path,
        calculated_at=datetime(2026, 4, 15, 16, 5, tzinfo=UTC),
    )

    with sqlite3.connect(db_path) as connection:
        row_count = connection.execute("SELECT COUNT(*) FROM edge_calculations").fetchone()
        latest_row = connection.execute(
            """
            SELECT side, edge_pct, ev, is_positive_ev, calculated_at
            FROM edge_calculations
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()

    assert row_count == (2,)
    assert latest_row is not None
    assert latest_row[0] == "away"
    assert latest_row[1] == pytest.approx(-0.03)
    assert latest_row[2] == pytest.approx(-0.06)
    assert latest_row[3] == 0
    assert latest_row[4] == "2026-04-15T16:05:00+00:00"
