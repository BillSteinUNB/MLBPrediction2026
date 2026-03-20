from __future__ import annotations

import csv
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.db import init_db
from src.models.bet import BetDecision, BetResult


UTC = timezone.utc


def _seed_game(
    db_path: Path,
    game_pk: int,
    *,
    scheduled_start: str = "2026-04-15T20:05:00+00:00",
    home_team: str = "NYY",
    away_team: str = "BOS",
) -> None:
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO games (game_pk, date, home_team, away_team, venue, status)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (game_pk, scheduled_start, home_team, away_team, "Yankee Stadium", "scheduled"),
        )
        connection.commit()


def _seed_snapshot(
    db_path: Path,
    *,
    game_pk: int,
    market_type: str,
    book_name: str = "DraftKings",
    home_odds: int,
    away_odds: int,
    fetched_at: datetime,
) -> None:
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
            (
                game_pk,
                book_name,
                market_type,
                home_odds,
                away_odds,
                fetched_at.isoformat(),
                0,
            ),
        )
        connection.commit()


def _decision(
    *,
    game_pk: int,
    market_type: str = "f5_ml",
    side: str = "home",
    book_name: str = "DraftKings",
    model_probability: float,
    edge_pct: float,
    ev: float,
    kelly_stake: float,
    odds_at_bet: int,
    result: BetResult = BetResult.PENDING,
    settled_at: datetime | None = None,
) -> BetDecision:
    return BetDecision(
        game_pk=game_pk,
        market_type=market_type,
        side=side,
        book_name=book_name,
        model_probability=model_probability,
        fair_probability=model_probability - edge_pct,
        edge_pct=edge_pct,
        ev=ev,
        is_positive_ev=edge_pct >= 0.03,
        kelly_stake=kelly_stake,
        odds_at_bet=odds_at_bet,
        result=result,
        settled_at=settled_at,
    )


def test_update_bankroll_records_per_bet_performance_metrics_in_sqlite(tmp_path: Path) -> None:
    from src.engine.bankroll import update_bankroll

    db_path = tmp_path / "performance.db"
    init_db(db_path)
    _seed_game(db_path, 12345)

    placed_decision = _decision(
        game_pk=12345,
        model_probability=0.59,
        edge_pct=0.04,
        ev=0.08,
        kelly_stake=50.0,
        odds_at_bet=-130,
    )
    update_bankroll(
        action="place",
        decision=placed_decision,
        db_path=db_path,
        starting_bankroll=1000.0,
        timestamp=datetime(2026, 4, 15, 16, 0, tzinfo=UTC),
    )

    update_bankroll(
        action="settle",
        decision=placed_decision.model_copy(
            update={
                "result": BetResult.WIN,
                "settled_at": datetime(2026, 4, 15, 21, 0, tzinfo=UTC),
            }
        ),
        db_path=db_path,
        starting_bankroll=1000.0,
        timestamp=datetime(2026, 4, 15, 21, 0, tzinfo=UTC),
    )

    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT
                game_pk,
                market_type,
                side,
                book_name,
                model_probability,
                market_probability,
                edge_pct,
                result,
                profit_loss,
                stake,
                placed_at,
                settled_at
            FROM bet_performance
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()

    assert row is not None
    assert row[0] == 12345
    assert row[1] == "f5_ml"
    assert row[2] == "home"
    assert row[3] == "DraftKings"
    assert row[4] == pytest.approx(0.59)
    assert row[5] == pytest.approx(130 / 230)
    assert row[6] == pytest.approx(0.04)
    assert row[7] == "WIN"
    assert row[8] == pytest.approx(5000 / 130)
    assert row[9] == pytest.approx(50.0)
    assert row[10] == "2026-04-15T16:00:00+00:00"
    assert row[11] == "2026-04-15T21:00:00+00:00"


def test_sync_closing_lines_uses_latest_snapshot_and_persists_clv(tmp_path: Path) -> None:
    from src.engine.bankroll import update_bankroll
    from src.ops.performance_tracker import sync_closing_lines_from_snapshots

    db_path = tmp_path / "performance.db"
    init_db(db_path)
    _seed_game(db_path, 12345)
    _seed_snapshot(
        db_path,
        game_pk=12345,
        market_type="f5_ml",
        home_odds=-130,
        away_odds=110,
        fetched_at=datetime(2026, 4, 15, 16, 0, tzinfo=UTC),
    )
    _seed_snapshot(
        db_path,
        game_pk=12345,
        market_type="f5_ml",
        home_odds=-140,
        away_odds=120,
        fetched_at=datetime(2026, 4, 15, 19, 55, tzinfo=UTC),
    )

    update_bankroll(
        action="place",
        decision=_decision(
            game_pk=12345,
            model_probability=0.59,
            edge_pct=0.04,
            ev=0.08,
            kelly_stake=40.0,
            odds_at_bet=-130,
        ),
        db_path=db_path,
        starting_bankroll=1000.0,
        timestamp=datetime(2026, 4, 15, 16, 0, tzinfo=UTC),
    )

    updated_count = sync_closing_lines_from_snapshots(
        game_pk=12345,
        market_type="f5_ml",
        db_path=db_path,
    )

    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            "SELECT closing_odds, closing_probability, clv FROM bet_performance"
        ).fetchone()

    assert updated_count == 1
    assert row is not None
    assert row[0] == -140
    assert row[1] == pytest.approx(140 / 240)
    assert row[2] == pytest.approx((140 / 240) - (130 / 230), abs=1e-6)


def test_sync_closing_lines_uses_matching_bookmaker_snapshot(tmp_path: Path) -> None:
    from src.engine.bankroll import update_bankroll
    from src.ops.performance_tracker import sync_closing_lines_from_snapshots

    db_path = tmp_path / "performance.db"
    init_db(db_path)
    _seed_game(db_path, 22222)
    _seed_snapshot(
        db_path,
        game_pk=22222,
        market_type="f5_ml",
        book_name="DraftKings",
        home_odds=-150,
        away_odds=130,
        fetched_at=datetime(2026, 4, 15, 19, 50, tzinfo=UTC),
    )
    _seed_snapshot(
        db_path,
        game_pk=22222,
        market_type="f5_ml",
        book_name="FanDuel",
        home_odds=-125,
        away_odds=105,
        fetched_at=datetime(2026, 4, 15, 19, 55, tzinfo=UTC),
    )

    update_bankroll(
        action="place",
        decision=_decision(
            game_pk=22222,
            book_name="FanDuel",
            model_probability=0.57,
            edge_pct=0.03,
            ev=0.04,
            kelly_stake=25.0,
            odds_at_bet=-110,
        ),
        db_path=db_path,
        starting_bankroll=1000.0,
        timestamp=datetime(2026, 4, 15, 16, 0, tzinfo=UTC),
    )

    updated_count = sync_closing_lines_from_snapshots(
        game_pk=22222,
        market_type="f5_ml",
        db_path=db_path,
    )

    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            "SELECT book_name, closing_odds, closing_probability, clv FROM bet_performance"
        ).fetchone()

    assert updated_count == 1
    assert row == (
        "FanDuel",
        -125,
        pytest.approx(125 / 225),
        pytest.approx((125 / 225) - (110 / 210), abs=1e-6),
    )


def test_generate_periodic_reports_computes_roi_brier_log_loss_and_clv(tmp_path: Path) -> None:
    from src.engine.bankroll import update_bankroll
    from src.ops.performance_tracker import generate_periodic_reports, sync_closing_lines_from_snapshots

    db_path = tmp_path / "performance.db"
    init_db(db_path)

    _seed_game(db_path, 101)
    _seed_game(db_path, 102, scheduled_start="2026-04-12T20:05:00+00:00")
    _seed_game(db_path, 103, scheduled_start="2026-03-26T20:05:00+00:00")

    scenarios = [
        {
            "game_pk": 101,
            "placed_at": datetime(2026, 4, 15, 10, 0, tzinfo=UTC),
            "settled_at": datetime(2026, 4, 15, 15, 0, tzinfo=UTC),
            "decision": _decision(
                game_pk=101,
                model_probability=0.60,
                edge_pct=0.10,
                ev=0.20,
                kelly_stake=50.0,
                odds_at_bet=100,
            ),
            "result": BetResult.WIN,
            "closing_home_odds": -110,
            "closing_away_odds": 100,
        },
        {
            "game_pk": 102,
            "placed_at": datetime(2026, 4, 12, 10, 0, tzinfo=UTC),
            "settled_at": datetime(2026, 4, 12, 15, 0, tzinfo=UTC),
            "decision": _decision(
                game_pk=102,
                model_probability=0.55,
                edge_pct=0.03,
                ev=0.05,
                kelly_stake=40.0,
                odds_at_bet=-110,
            ),
            "result": BetResult.LOSS,
            "closing_home_odds": -120,
            "closing_away_odds": 100,
        },
        {
            "game_pk": 103,
            "placed_at": datetime(2026, 3, 26, 10, 0, tzinfo=UTC),
            "settled_at": datetime(2026, 3, 26, 15, 0, tzinfo=UTC),
            "decision": _decision(
                game_pk=103,
                model_probability=0.52,
                edge_pct=0.04,
                ev=0.144,
                kelly_stake=30.0,
                odds_at_bet=120,
            ),
            "result": BetResult.WIN,
            "closing_home_odds": 130,
            "closing_away_odds": -150,
        },
    ]

    for scenario in scenarios:
        update_bankroll(
            action="place",
            decision=scenario["decision"],
            db_path=db_path,
            starting_bankroll=1000.0,
            timestamp=scenario["placed_at"],
        )
        update_bankroll(
            action="settle",
            decision=scenario["decision"].model_copy(
                update={
                    "result": scenario["result"],
                    "settled_at": scenario["settled_at"],
                }
            ),
            db_path=db_path,
            starting_bankroll=1000.0,
            timestamp=scenario["settled_at"],
        )
        _seed_snapshot(
            db_path,
            game_pk=scenario["game_pk"],
            market_type="f5_ml",
            home_odds=scenario["closing_home_odds"],
            away_odds=scenario["closing_away_odds"],
            fetched_at=scenario["settled_at"],
        )
        sync_closing_lines_from_snapshots(
            game_pk=scenario["game_pk"],
            market_type="f5_ml",
            db_path=db_path,
        )

    reports = generate_periodic_reports(
        db_path=db_path,
        as_of=datetime(2026, 4, 15, 23, 59, tzinfo=UTC),
    )

    daily = reports["daily"]
    weekly = reports["weekly"]
    monthly = reports["monthly"]

    assert daily.total_bets == 1
    assert daily.win_rate == pytest.approx(1.0)
    assert daily.roi == pytest.approx(1.0)
    assert daily.brier_score == pytest.approx((1.0 - 0.60) ** 2)
    assert daily.log_loss == pytest.approx(-math.log(0.60))
    assert daily.average_clv == pytest.approx((110 / 210) - 0.5)

    assert weekly.total_bets == 2
    assert weekly.win_rate == pytest.approx(0.5)
    assert weekly.roi == pytest.approx(10.0 / 90.0)
    assert weekly.brier_score == pytest.approx((((1.0 - 0.60) ** 2) + ((0.0 - 0.55) ** 2)) / 2)
    assert weekly.log_loss == pytest.approx((-math.log(0.60) - math.log(0.45)) / 2)

    assert monthly.total_bets == 3
    assert monthly.win_rate == pytest.approx(2.0 / 3.0)
    assert monthly.roi == pytest.approx(46.0 / 120.0)
    assert monthly.average_clv == pytest.approx(
        (((110 / 210) - 0.5) + ((120 / 220) - (110 / 210)) + ((100 / 230) - (100 / 220))) / 3
    )


def test_export_performance_csv_writes_tracked_rows(tmp_path: Path) -> None:
    from src.engine.bankroll import update_bankroll
    from src.ops.performance_tracker import export_performance_csv, sync_closing_lines_from_snapshots

    db_path = tmp_path / "performance.db"
    init_db(db_path)
    _seed_game(db_path, 12345)
    _seed_snapshot(
        db_path,
        game_pk=12345,
        market_type="f5_ml",
        home_odds=-120,
        away_odds=100,
        fetched_at=datetime(2026, 4, 15, 19, 30, tzinfo=UTC),
    )

    decision = _decision(
        game_pk=12345,
        model_probability=0.57,
        edge_pct=0.05,
        ev=0.09,
        kelly_stake=25.0,
        odds_at_bet=-110,
    )
    update_bankroll(
        action="place",
        decision=decision,
        db_path=db_path,
        starting_bankroll=1000.0,
        timestamp=datetime(2026, 4, 15, 16, 0, tzinfo=UTC),
    )
    update_bankroll(
        action="settle",
        decision=decision.model_copy(
            update={
                "result": BetResult.LOSS,
                "settled_at": datetime(2026, 4, 15, 21, 0, tzinfo=UTC),
            }
        ),
        db_path=db_path,
        starting_bankroll=1000.0,
        timestamp=datetime(2026, 4, 15, 21, 0, tzinfo=UTC),
    )
    sync_closing_lines_from_snapshots(game_pk=12345, market_type="f5_ml", db_path=db_path)

    csv_path = export_performance_csv(output_path=tmp_path / "performance.csv", db_path=db_path)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["game_pk"] == "12345"
    assert rows[0]["result"] == "LOSS"
    assert rows[0]["book_name"] == "DraftKings"
    assert rows[0]["profit_loss"] == "-25.0"
    assert rows[0]["closing_odds"] == "-120"
