from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.db import init_db
from src.models.bet import BetDecision, BetResult


UTC = timezone.utc


def _seed_game(db_path: Path, game_pk: int) -> None:
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO games (game_pk, date, home_team, away_team, venue, status)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (game_pk, "2026-04-15T20:05:00+00:00", "NYY", "BOS", "Yankee Stadium", "scheduled"),
        )
        connection.commit()


def _decision(
    *,
    game_pk: int = 12345,
    market_type: str = "f5_ml",
    side: str = "home",
    model_probability: float = 0.55,
    fair_probability: float = 0.5,
    edge_pct: float = 0.05,
    ev: float = 0.1,
    kelly_stake: float = 0.0,
    odds_at_bet: int = -110,
    result: BetResult = BetResult.PENDING,
    settled_at: datetime | None = None,
    profit_loss: float | None = None,
) -> BetDecision:
    return BetDecision(
        game_pk=game_pk,
        market_type=market_type,
        side=side,
        model_probability=model_probability,
        fair_probability=fair_probability,
        edge_pct=edge_pct,
        ev=ev,
        is_positive_ev=edge_pct >= 0.03 and ev > 0,
        kelly_stake=kelly_stake,
        odds_at_bet=odds_at_bet,
        result=result,
        settled_at=settled_at,
        profit_loss=profit_loss,
    )


def test_calculate_kelly_stake_applies_quarter_kelly_formula() -> None:
    from src.engine.bankroll import calculate_kelly_stake

    result = calculate_kelly_stake(bankroll=1000.0, model_probability=0.55, odds=-110)

    payout_multiple = 100 / 110
    full_kelly_fraction = ((payout_multiple * 0.55) - 0.45) / payout_multiple

    assert result.full_kelly_fraction == pytest.approx(full_kelly_fraction)
    assert result.stake_fraction == pytest.approx(full_kelly_fraction * 0.25)
    assert result.stake == pytest.approx(1000.0 * full_kelly_fraction * 0.25)
    assert result.kill_switch_active is False


def test_calculate_kelly_stake_enforces_five_percent_cap() -> None:
    from src.engine.bankroll import calculate_kelly_stake

    result = calculate_kelly_stake(bankroll=1000.0, model_probability=0.75, odds=130)

    assert result.stake == pytest.approx(50.0)
    assert result.stake_fraction == pytest.approx(0.05)
    assert result.is_capped is True


def test_calculate_kelly_stake_returns_zero_when_drawdown_hits_killswitch() -> None:
    from src.engine.bankroll import calculate_kelly_stake

    result = calculate_kelly_stake(
        bankroll=690.0,
        peak_bankroll=1000.0,
        model_probability=0.60,
        odds=120,
    )

    assert result.drawdown_pct == pytest.approx(0.31)
    assert result.kill_switch_active is True
    assert result.stake == 0.0
    assert result.stake_fraction == 0.0


def test_calculate_kelly_stake_treats_same_team_ml_and_rl_as_single_bet() -> None:
    from src.engine.bankroll import calculate_kelly_stake

    ml_decision = _decision(
        market_type="f5_ml",
        side="home",
        model_probability=0.55,
        fair_probability=0.5,
        edge_pct=0.05,
        ev=0.05,
        odds_at_bet=-110,
    )
    rl_decision = _decision(
        market_type="f5_rl",
        side="home",
        model_probability=0.47,
        fair_probability=0.40,
        edge_pct=0.07,
        ev=0.081,
        odds_at_bet=130,
    )

    result = calculate_kelly_stake(
        bankroll=1000.0,
        correlated_decisions=[ml_decision, rl_decision],
    )

    payout_multiple = 1.3
    full_kelly_fraction = ((payout_multiple * 0.47) - 0.53) / payout_multiple

    assert result.stake == pytest.approx(1000.0 * full_kelly_fraction * 0.25)
    assert result.selected_market_type == "f5_rl"
    assert result.suppressed_market_types == ("f5_ml",)


def test_update_bankroll_records_bet_placement_and_win_in_sqlite(tmp_path: Path) -> None:
    from src.engine.bankroll import update_bankroll

    db_path = tmp_path / "bankroll.db"
    init_db(db_path)
    _seed_game(db_path, 12345)

    placed_decision = _decision(kelly_stake=50.0, odds_at_bet=120)
    placement_summary = update_bankroll(
        action="place",
        decision=placed_decision,
        db_path=db_path,
        starting_bankroll=1000.0,
        timestamp=datetime(2026, 4, 15, 16, 0, tzinfo=UTC),
    )

    winning_decision = placed_decision.model_copy(
        update={
            "result": BetResult.WIN,
            "settled_at": datetime(2026, 4, 15, 21, 0, tzinfo=UTC),
        }
    )
    settlement_summary = update_bankroll(
        action="settle",
        decision=winning_decision,
        db_path=db_path,
        starting_bankroll=1000.0,
        timestamp=datetime(2026, 4, 15, 21, 0, tzinfo=UTC),
    )

    assert placement_summary.current_bankroll == pytest.approx(950.0)
    assert placement_summary.peak_bankroll == pytest.approx(1000.0)
    assert placement_summary.total_bets == 1
    assert placement_summary.win_rate == 0.0
    assert placement_summary.roi == 0.0

    assert settlement_summary.current_bankroll == pytest.approx(1060.0)
    assert settlement_summary.peak_bankroll == pytest.approx(1060.0)
    assert settlement_summary.total_bets == 1
    assert settlement_summary.win_rate == pytest.approx(1.0)
    assert settlement_summary.roi == pytest.approx(1.2)

    with sqlite3.connect(db_path) as connection:
        ledger_rows = connection.execute(
            "SELECT event_type, amount, running_balance FROM bankroll_ledger ORDER BY id"
        ).fetchall()
        bet_row = connection.execute(
            "SELECT result, profit_loss FROM bets ORDER BY id DESC LIMIT 1"
        ).fetchone()

    assert ledger_rows == [
        ("bet_placed", -50.0, 950.0),
        ("bet_settled", 110.0, 1060.0),
    ]
    assert bet_row == ("WIN", 60.0)


def test_update_bankroll_logs_drawdown_alert_and_killswitch_once(tmp_path: Path) -> None:
    from src.engine.bankroll import calculate_kelly_stake, update_bankroll

    db_path = tmp_path / "bankroll.db"
    init_db(db_path)

    summary = None
    current_bankroll = 1000.0

    for game_pk in range(1, 10):
        _seed_game(db_path, game_pk)
        kelly = calculate_kelly_stake(
            bankroll=current_bankroll,
            peak_bankroll=1000.0,
            model_probability=0.75,
            odds=130,
        )
        losing_bet = _decision(
            game_pk=game_pk,
            market_type="f5_ml",
            side="home",
            model_probability=0.75,
            fair_probability=0.5,
            edge_pct=0.25,
            ev=0.725,
            kelly_stake=kelly.stake,
            odds_at_bet=130,
        )

        summary = update_bankroll(
            action="place",
            decision=losing_bet,
            db_path=db_path,
            starting_bankroll=1000.0,
            timestamp=datetime(2026, 4, 15, 16, game_pk, tzinfo=UTC),
        )
        summary = update_bankroll(
            action="settle",
            decision=losing_bet.model_copy(
                update={
                    "result": BetResult.LOSS,
                    "settled_at": datetime(2026, 4, 15, 21, game_pk, tzinfo=UTC),
                }
            ),
            db_path=db_path,
            starting_bankroll=1000.0,
            timestamp=datetime(2026, 4, 15, 21, game_pk, tzinfo=UTC),
        )
        current_bankroll = summary.current_bankroll

        if summary.kill_switch_active:
            break

    assert summary is not None
    assert summary.kill_switch_active is True
    assert summary.drawdown_pct > 0.30
    assert summary.current_bankroll < 700.0

    with sqlite3.connect(db_path) as connection:
        event_counts = dict(
            connection.execute(
                "SELECT event_type, COUNT(*) FROM bankroll_ledger GROUP BY event_type"
            ).fetchall()
        )

    assert event_counts["drawdown_alert"] == 1
    assert event_counts["kill_switch"] == 1
