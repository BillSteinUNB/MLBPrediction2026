from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from src.models.bet import BetDecision
from src.models.prediction import Prediction
from src.ops.live_season_tracker import (
    build_live_season_summary,
    capture_daily_result,
    list_tracked_games,
    settle_tracked_games,
    sync_live_game_state,
)
from src.pipeline.daily import DailyPipelineResult, GameProcessingResult


def _prediction(game_pk: int, *, ml_home: float, rl_home: float) -> Prediction:
    return Prediction(
        game_pk=game_pk,
        model_version="live-test-v1",
        f5_ml_home_prob=ml_home,
        f5_ml_away_prob=1 - ml_home,
        f5_rl_home_prob=rl_home,
        f5_rl_away_prob=1 - rl_home,
        predicted_at=datetime(2026, 3, 25, 12, 0, tzinfo=UTC),
    )


def _decision(
    game_pk: int,
    *,
    market_type: str,
    side: str,
    odds: int,
    line_at_bet: float | None = None,
    model_probability: float = 0.6,
) -> BetDecision:
    return BetDecision(
        game_pk=game_pk,
        market_type=market_type,
        side=side,
        book_name="TestBook",
        model_probability=model_probability,
        fair_probability=0.5,
        edge_pct=model_probability - 0.5,
        ev=0.1,
        is_positive_ev=True,
        kelly_stake=1.0,
        odds_at_bet=odds,
        line_at_bet=line_at_bet,
    )


def test_capture_and_settle_live_season_tracking(tmp_path: Path) -> None:
    db_path = tmp_path / "live_tracking.db"

    result = DailyPipelineResult(
        run_id="run-20260325",
        pipeline_date="2026-03-25",
        mode="prod",
        dry_run=True,
        model_version="live-test-v1",
        pick_count=1,
        no_pick_count=1,
        error_count=0,
        notification_type="picks",
        notification_payload={},
        games=[
            GameProcessingResult(
                game_pk=1001,
                matchup="NYY @ SF",
                status="pick",
                prediction=_prediction(1001, ml_home=0.44, rl_home=0.30),
                selected_decision=_decision(
                    1001,
                    market_type="f5_rl",
                    side="away",
                    odds=120,
                    line_at_bet=-0.5,
                    model_probability=0.70,
                ),
                forced_decision=_decision(
                    1001,
                    market_type="f5_rl",
                    side="away",
                    odds=120,
                    line_at_bet=-0.5,
                    model_probability=0.70,
                ),
                paper_fallback=False,
                input_status={"home_lineup_available": True},
            ),
            GameProcessingResult(
                game_pk=1002,
                matchup="PIT @ NYM",
                status="no_pick",
                prediction=_prediction(1002, ml_home=0.51, rl_home=0.52),
                forced_decision=_decision(
                    1002,
                    market_type="f5_ml",
                    side="home",
                    odds=-110,
                    model_probability=0.51,
                ),
                no_pick_reason="lineup unavailable",
                input_status={"home_lineup_available": False},
            ),
        ],
    )

    inserted = capture_daily_result(result=result, db_path=db_path)
    assert inserted == 2

    tracked = list_tracked_games(season=2026, db_path=db_path)
    assert len(tracked) == 2
    assert tracked[0]["is_play_of_day"] == 1
    assert tracked[0]["selected_market_type"] == "f5_rl"
    assert tracked[0]["forced_market_type"] == "f5_rl"
    assert tracked[0]["line_at_bet"] == -0.5

    # Populate actual final game results in the shared games table.
    from src.db import init_db
    import sqlite3

    init_db(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.executemany(
            """
            INSERT INTO games (
                game_pk, date, home_team, away_team, home_starter_id, away_starter_id,
                venue, is_dome, is_abs_active, f5_home_score, f5_away_score,
                final_home_score, final_away_score, status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(game_pk) DO UPDATE SET
                f5_home_score = excluded.f5_home_score,
                f5_away_score = excluded.f5_away_score,
                final_home_score = excluded.final_home_score,
                final_away_score = excluded.final_away_score,
                status = excluded.status
            """,
            [
                (1001, "2026-03-25", "SF", "NYY", None, None, "Oracle", 0, 1, 0, 2, 1, 4, "final"),
                (1002, "2026-03-25", "NYM", "PIT", None, None, "Citi", 0, 1, 3, 1, 5, 2, "final"),
            ],
        )
        connection.commit()

    updated = settle_tracked_games(pipeline_date="2026-03-25", db_path=db_path)
    assert updated == 2

    summary = build_live_season_summary(season=2026, db_path=db_path)
    assert summary.tracked_games == 2
    assert summary.settled_games == 2
    assert summary.picks == 1
    assert summary.graded_picks == 1
    assert summary.wins == 1
    assert summary.losses == 0
    assert summary.play_of_day_count == 1
    assert summary.play_of_day_wins == 1
    assert summary.forced_picks == 2
    assert summary.forced_graded_picks == 2
    assert summary.f5_ml_accuracy is not None
    assert summary.f5_rl_accuracy is not None
    assert summary.flat_profit_units > 0


def test_sync_live_game_state_updates_games_and_settles(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "live_tracking_sync.db"

    result = DailyPipelineResult(
        run_id="run-20260326",
        pipeline_date="2026-03-26",
        mode="prod",
        dry_run=True,
        model_version="live-test-v1",
        pick_count=1,
        no_pick_count=0,
        error_count=0,
        notification_type="picks",
        notification_payload={},
        games=[
            GameProcessingResult(
                game_pk=2001,
                matchup="NYY @ SF",
                status="pick",
                prediction=_prediction(2001, ml_home=0.40, rl_home=0.35),
                selected_decision=_decision(
                    2001,
                    market_type="f5_ml",
                    side="away",
                    odds=110,
                    model_probability=0.60,
                ),
                input_status={"home_lineup_available": True},
            )
        ],
    )
    capture_daily_result(result=result, db_path=db_path)

    from src.db import init_db
    import sqlite3
    import src.ops.live_season_tracker as tracker_module

    init_db(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO games (
                game_pk, date, home_team, away_team, home_starter_id, away_starter_id,
                venue, is_dome, is_abs_active, f5_home_score, f5_away_score,
                final_home_score, final_away_score, status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (2001, "2026-03-26", "SF", "NYY", None, None, "Oracle", 0, 1, 0, 2, None, None, "scheduled"),
        )
        connection.commit()

    monkeypatch.setattr(
        tracker_module,
        "_default_schedule_fetcher",
        lambda target_date, mode: __import__("pandas").DataFrame(),
    )

    game_state_path = tmp_path / "live_game_state.json"
    game_state_path.write_text(
        __import__("json").dumps(
            [
                {
                    "event_id": "game-2001",
                    "game_date": "2026-03-26",
                    "away_team": "NYY",
                    "home_team": "SF",
                    "away_team_score": 4,
                    "home_team_score": 1,
                    "game_status_text": "Final",
                    "status": "1",
                    "inning": 9,
                    "outs": 3,
                    "fetched_at": "2026-03-27T00:00:00+00:00",
                }
            ]
        ),
        encoding="utf-8",
    )

    summary = sync_live_game_state(input_path=game_state_path, db_path=db_path, settle=True)
    assert summary["imported_rows"] == 1
    assert summary["settled_rows"] == 1

    tracked = list_tracked_games(season=2026, db_path=db_path)
    assert tracked[0]["actual_status"] == "final"
    assert tracked[0]["actual_f5_home_score"] == 0
    assert tracked[0]["actual_f5_away_score"] == 2
    assert tracked[0]["actual_final_home_score"] == 1
    assert tracked[0]["actual_final_away_score"] == 4
    assert tracked[0]["settled_result"] == "WIN"
