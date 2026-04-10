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


@pytest.mark.parametrize(
    ("odds", "expected_probability"),
    [
        pytest.param(100, 0.5, id="plus-100"),
        pytest.param(-100, 0.5, id="minus-100"),
        pytest.param(-300, 0.75, id="minus-300"),
        pytest.param(300, 0.25, id="plus-300"),
    ],
)
def test_american_to_implied_handles_even_and_exact_plus_minus_three_hundred_edges(
    odds: int,
    expected_probability: float,
) -> None:
    from src.engine.edge_calculator import american_to_implied

    assert american_to_implied(odds) == pytest.approx(expected_probability)


@pytest.mark.parametrize(
    ("home_odds", "away_odds", "expected_home_probability", "expected_away_probability"),
    [
        pytest.param(100, -100, 0.5, 0.5, id="even"),
        pytest.param(-300, 300, 0.75, 0.25, id="exact-three-hundred"),
    ],
)
def test_devig_probabilities_sum_to_one_for_even_and_heavy_lines(
    home_odds: int,
    away_odds: int,
    expected_home_probability: float,
    expected_away_probability: float,
) -> None:
    from src.engine.edge_calculator import devig_probabilities

    home_probability, away_probability = devig_probabilities(home_odds, away_odds)

    assert home_probability == pytest.approx(expected_home_probability)
    assert away_probability == pytest.approx(expected_away_probability)
    assert home_probability + away_probability == pytest.approx(1.0)


def test_calculate_edge_supports_f5_rl_market_type_with_exact_three_hundred_lines(
    tmp_path: Path,
) -> None:
    from src.engine.edge_calculator import calculate_edge

    db_path = tmp_path / "edge.db"
    init_db(db_path)
    _seed_game(db_path)

    decision = calculate_edge(
        game_pk=12345,
        market_type="f5_rl",
        side="away",
        model_probability=0.32,
        home_odds=-300,
        away_odds=300,
        book_name="Caesars",
        db_path=db_path,
        calculated_at=datetime(2026, 4, 15, 16, 10, tzinfo=UTC),
    )

    assert decision.market_type == "f5_rl"
    assert decision.side == "away"
    assert decision.fair_probability == pytest.approx(0.25)
    assert decision.edge_pct == pytest.approx(0.07)
    assert decision.ev == pytest.approx((0.32 * 3.0) - (0.68 * 1.0))
    assert decision.is_positive_ev is True

    with sqlite3.connect(db_path) as connection:
        logged_row = connection.execute(
            "SELECT market_type, side, book_name, odds_at_bet FROM edge_calculations ORDER BY id DESC LIMIT 1"
        ).fetchone()

    assert logged_row == ("f5_rl", "away", "Caesars", 300)


def test_calculate_edge_keeps_ev_informational_even_when_edge_is_below_live_threshold(
    tmp_path: Path,
) -> None:
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
    assert decision.ev < 0
    assert decision.is_positive_ev is False


def test_calculate_edge_recommends_bets_at_or_above_six_percent_threshold(tmp_path: Path) -> None:
    from src.engine.edge_calculator import calculate_edge

    db_path = tmp_path / "edge.db"
    init_db(db_path)
    _seed_game(db_path)

    decision = calculate_edge(
        game_pk=12345,
        market_type="f5_ml",
        side="home",
        model_probability=0.56,
        home_odds=100,
        away_odds=-100,
        book_name="DraftKings",
        db_path=db_path,
    )

    assert decision.edge_pct == pytest.approx(0.06)
    assert decision.ev == pytest.approx(0.12)
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
        model_probability=0.559,
        home_odds=100,
        away_odds=-100,
        book_name="DraftKings",
        db_path=db_path,
    )

    assert decision.edge_pct == pytest.approx(0.059)
    assert decision.ev == pytest.approx(0.118)
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
        model_probability=0.30,
        home_odds=-360,
        away_odds=330,
        book_name="FanDuel",
        db_path=db_path,
    )

    assert decision.side == "away"
    assert decision.fair_probability == pytest.approx((100 / 430) / ((360 / 460) + (100 / 430)))
    assert decision.edge_pct == pytest.approx(
        0.30 - ((100 / 430) / ((360 / 460) + (100 / 430)))
    )
    assert decision.ev == pytest.approx((0.30 * 3.3) - (0.70 * 1.0))
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


def test_calculate_edge_accounts_for_push_probability_on_whole_number_markets(tmp_path: Path) -> None:
    from src.engine.edge_calculator import calculate_edge, payout_for_american_odds

    db_path = tmp_path / "edge.db"
    init_db(db_path)
    _seed_game(db_path)

    decision = calculate_edge(
        game_pk=12345,
        market_type="full_game_total",
        side="under",
        model_probability=0.46,
        home_odds=-110,
        away_odds=-110,
        home_point=10.0,
        away_point=10.0,
        push_probability=0.08,
        book_name="bet365",
        db_path=db_path,
    )

    assert decision.fair_probability == pytest.approx(0.46)
    assert decision.edge_pct == pytest.approx(0.0)
    expected_ev = (0.46 * payout_for_american_odds(-110)) - (0.46 * 1.0)
    assert decision.ev == pytest.approx(expected_ev)
