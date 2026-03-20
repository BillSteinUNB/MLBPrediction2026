from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.db import init_db
from src.engine.bankroll import update_bankroll
from src.models.bet import BetDecision, BetResult


UTC = timezone.utc


def _seed_game(
    db_path: Path,
    game_pk: int,
    *,
    home_team: str = "NYY",
    away_team: str = "BOS",
    scheduled_start: str = "2026-04-15T20:05:00+00:00",
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


def _decision(
    *,
    game_pk: int = 12345,
    market_type: str = "f5_ml",
    side: str = "home",
    kelly_stake: float = 50.0,
    odds_at_bet: int = 120,
) -> BetDecision:
    return BetDecision(
        game_pk=game_pk,
        market_type=market_type,
        side=side,
        model_probability=0.55,
        fair_probability=0.50,
        edge_pct=0.05,
        ev=0.10,
        is_positive_ev=True,
        kelly_stake=kelly_stake,
        odds_at_bet=odds_at_bet,
        result=BetResult.PENDING,
    )


@pytest.mark.parametrize(
    ("market_type", "side", "home_score", "away_score", "innings_completed", "starter_scratched", "expected"),
    [
        pytest.param("f5_ml", "home", 3, 1, 5.0, False, BetResult.WIN, id="ml-home-lead-home-win"),
        pytest.param("f5_ml", "away", 1, 3, 5.0, False, BetResult.WIN, id="ml-away-lead-away-win"),
        pytest.param("f5_ml", "home", 2, 2, 5.0, False, BetResult.PUSH, id="ml-tie-push"),
        pytest.param("f5_rl", "home", 4, 2, 5.0, False, BetResult.WIN, id="rl-home-minus-wins-by-two"),
        pytest.param("f5_rl", "home", 3, 2, 5.0, False, BetResult.LOSS, id="rl-home-minus-loses-by-one"),
        pytest.param("f5_rl", "away", 2, 2, 5.0, False, BetResult.WIN, id="rl-away-plus-wins-on-tie"),
        pytest.param("f5_rl", "home", 1, 3, 5.0, False, BetResult.LOSS, id="rl-home-minus-loses-when-trailing"),
        pytest.param("f5_ml", "home", 3, 1, 4.5, False, BetResult.NO_ACTION, id="short-game-no-action"),
        pytest.param("f5_ml", "home", 3, 1, 5.0, True, BetResult.NO_ACTION, id="starter-scratch-no-action"),
    ],
)
def test_settle_bet_handles_all_contract_scenarios(
    market_type: str,
    side: str,
    home_score: int,
    away_score: int,
    innings_completed: float,
    starter_scratched: bool,
    expected: BetResult,
) -> None:
    from src.engine.settlement import settle_bet

    decision = _decision(market_type=market_type, side=side)

    result = settle_bet(
        decision,
        home_score=home_score,
        away_score=away_score,
        innings_completed=innings_completed,
        starter_scratched=starter_scratched,
    )

    assert result is expected


@pytest.mark.parametrize(
    ("market_type", "side", "home_score", "away_score", "expected"),
    [
        pytest.param("f5_ml", "away", 3, 1, BetResult.LOSS, id="ml-away-bet-loses-when-home-leads"),
        pytest.param("f5_ml", "home", 1, 3, BetResult.LOSS, id="ml-home-bet-loses-when-away-leads"),
        pytest.param("f5_rl", "away", 4, 2, BetResult.LOSS, id="rl-away-plus-loses-on-two-run-deficit"),
        pytest.param("f5_rl", "away", 3, 2, BetResult.WIN, id="rl-away-plus-wins-on-one-run-deficit"),
    ],
)
def test_settle_bet_handles_opposite_side_outcomes(
    market_type: str,
    side: str,
    home_score: int,
    away_score: int,
    expected: BetResult,
) -> None:
    from src.engine.settlement import settle_bet

    decision = _decision(market_type=market_type, side=side)

    result = settle_bet(
        decision,
        home_score=home_score,
        away_score=away_score,
        innings_completed=5.0,
    )

    assert result is expected


@pytest.mark.parametrize(
    ("market_type", "side", "home_score", "away_score", "innings_completed", "starter_scratched", "expected_result", "expected_profit_loss", "expected_settlement_amount", "expected_balance"),
    [
        pytest.param("f5_ml", "home", 3, 1, 5.0, False, "WIN", 60.0, 110.0, 1060.0, id="win"),
        pytest.param("f5_ml", "home", 1, 3, 5.0, False, "LOSS", -50.0, 0.0, 950.0, id="loss"),
        pytest.param("f5_ml", "home", 2, 2, 5.0, False, "PUSH", 0.0, 50.0, 1000.0, id="push"),
        pytest.param("f5_ml", "home", 3, 1, 4.0, False, "NO_ACTION", 0.0, 50.0, 1000.0, id="no-action"),
    ],
)
def test_settle_game_bets_updates_bets_and_bankroll_ledger(
    tmp_path: Path,
    market_type: str,
    side: str,
    home_score: int,
    away_score: int,
    innings_completed: float,
    starter_scratched: bool,
    expected_result: str,
    expected_profit_loss: float,
    expected_settlement_amount: float,
    expected_balance: float,
) -> None:
    from src.engine.settlement import settle_game_bets

    db_path = tmp_path / "settlement.db"
    init_db(db_path)
    _seed_game(db_path, 12345)

    placed_decision = _decision(game_pk=12345, market_type=market_type, side=side)
    update_bankroll(
        action="place",
        decision=placed_decision,
        db_path=db_path,
        starting_bankroll=1000.0,
        timestamp=datetime(2026, 4, 15, 16, 0, tzinfo=UTC),
    )

    settled_bets = settle_game_bets(
        12345,
        home_score=home_score,
        away_score=away_score,
        innings_completed=innings_completed,
        starter_scratched=starter_scratched,
        db_path=db_path,
        starting_bankroll=1000.0,
        settled_at=datetime(2026, 4, 15, 21, 0, tzinfo=UTC),
    )

    assert [bet.result.value for bet in settled_bets] == [expected_result]
    assert settled_bets[0].profit_loss == pytest.approx(expected_profit_loss)

    with sqlite3.connect(db_path) as connection:
        bet_row = connection.execute(
            "SELECT result, settled_at, profit_loss FROM bets ORDER BY id DESC LIMIT 1"
        ).fetchone()
        ledger_rows = connection.execute(
            "SELECT event_type, amount, running_balance FROM bankroll_ledger ORDER BY id"
        ).fetchall()

    assert bet_row == (
        expected_result,
        "2026-04-15T21:00:00+00:00",
        pytest.approx(expected_profit_loss),
    )
    assert ledger_rows[-1] == ("bet_settled", expected_settlement_amount, expected_balance)


def test_settle_game_bets_uses_game_pk_to_disambiguate_doubleheaders(tmp_path: Path) -> None:
    from src.engine.settlement import settle_game_bets

    db_path = tmp_path / "doubleheader.db"
    init_db(db_path)
    _seed_game(db_path, 2001, scheduled_start="2026-04-15T17:05:00+00:00")
    _seed_game(db_path, 2002, scheduled_start="2026-04-15T23:05:00+00:00")

    for game_pk in (2001, 2002):
        update_bankroll(
            action="place",
            decision=_decision(game_pk=game_pk, market_type="f5_ml", side="home"),
            db_path=db_path,
            starting_bankroll=1000.0,
            timestamp=datetime(2026, 4, 15, 15 + game_pk - 2001, 0, tzinfo=UTC),
        )

    settled_bets = settle_game_bets(
        2002,
        home_score=4,
        away_score=2,
        innings_completed=5.0,
        db_path=db_path,
        starting_bankroll=1000.0,
        settled_at=datetime(2026, 4, 15, 23, 30, tzinfo=UTC),
    )

    assert [bet.game_pk for bet in settled_bets] == [2002]
    assert [bet.result.value for bet in settled_bets] == ["WIN"]

    with sqlite3.connect(db_path) as connection:
        bet_rows = connection.execute(
            "SELECT game_pk, result FROM bets ORDER BY game_pk"
        ).fetchall()
        settlement_rows = connection.execute(
            "SELECT notes FROM bankroll_ledger WHERE event_type = 'bet_settled' ORDER BY id"
        ).fetchall()

    assert bet_rows == [(2001, "PENDING"), (2002, "WIN")]
    assert settlement_rows == [
        ("bet_settled:WIN: game_pk=2002, market_type=f5_ml, side=home, stake=50.00",),
    ]
