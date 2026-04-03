from __future__ import annotations

import json
import sqlite3
from datetime import UTC, date, datetime
from pathlib import Path

from src.models.bet import BetDecision
from src.models.prediction import Prediction
from src.ops.live_season_tracker import (
    _settle_outstanding_tracking_dates,
    _refresh_frozen_slate_market_context,
    build_live_season_summary,
    build_manual_tracking_summary,
    capture_daily_result,
    list_tracked_games,
    list_manual_tracked_bets,
    settle_tracked_games,
    settle_manual_tracked_bets,
    submit_manual_tracked_bet,
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


def test_refresh_frozen_slate_market_context_skips_started_games(
    tmp_path: Path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "live_tracking_refresh.db"

    from src.db import init_db
    import src.ops.live_season_tracker as tracker_module

    init_db(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS cached_slate_responses (
                pipeline_date TEXT NOT NULL,
                mode TEXT NOT NULL CHECK (mode IN ('prod', 'backtest')),
                dry_run INTEGER NOT NULL CHECK (dry_run IN (0, 1)),
                run_id TEXT NOT NULL,
                model_version TEXT,
                payload_json TEXT NOT NULL,
                refreshed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (pipeline_date, mode, dry_run)
            )
            """
        )
        connection.executemany(
            """
            INSERT INTO games (
                game_pk, date, home_team, away_team, home_starter_id, away_starter_id,
                venue, is_dome, is_abs_active, status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (3001, "2026-04-02", "KC", "MIN", None, None, "Kauffman", 0, 1, "final"),
                (3002, "2026-04-02", "ARI", "ATL", None, None, "Chase", 1, 1, "scheduled"),
            ],
        )
        payload = {
            "games": [
                {
                    "game_pk": 3001,
                    "matchup": "MIN @ KC",
                    "status": "pick",
                    "input_status": {
                        "full_game_total": 9.5,
                        "full_game_total_over_odds": -115,
                        "full_game_total_under_odds": -115,
                    },
                },
                {
                    "game_pk": 3002,
                    "matchup": "ATL @ ARI",
                    "status": "no_pick",
                    "input_status": {
                        "full_game_total": 8.5,
                        "full_game_total_over_odds": -110,
                        "full_game_total_under_odds": -110,
                    },
                },
            ]
        }
        connection.execute(
            """
            INSERT INTO cached_slate_responses (
                pipeline_date, mode, dry_run, run_id, model_version, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "2026-04-02",
                "backtest",
                1,
                "run-20260402",
                "test-model",
                json.dumps(payload),
            ),
        )
        connection.commit()

    monkeypatch.setattr(
        tracker_module,
        "_default_full_game_odds_context_fetcher",
        lambda target_date, mode, db_path: {
            3001: {
                "full_game_total": 7.5,
                "full_game_total_over_odds": 120,
                "full_game_total_under_odds": -140,
            },
            3002: {
                "full_game_total": 9.0,
                "full_game_total_over_odds": -106,
                "full_game_total_under_odds": -115,
            },
        },
    )
    monkeypatch.setattr(
        tracker_module,
        "fetch_mlb_full_game_odds_context",
        lambda **_: {},
    )

    refreshed_games = _refresh_frozen_slate_market_context(
        target_day=date(2026, 4, 2),
        db_path=db_path,
    )
    assert refreshed_games == 1

    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT payload_json
            FROM cached_slate_responses
            WHERE pipeline_date = '2026-04-02' AND mode = 'backtest' AND dry_run = 1
            """
        ).fetchone()

    updated_payload = json.loads(row[0])
    by_game_pk = {int(game["game_pk"]): game for game in updated_payload["games"]}
    assert by_game_pk[3001]["input_status"]["full_game_total"] == 9.5
    assert by_game_pk[3001]["input_status"]["full_game_total_over_odds"] == -115
    assert by_game_pk[3002]["input_status"]["full_game_total"] == 9.0
    assert by_game_pk[3002]["input_status"]["full_game_total_over_odds"] == -106


def test_settle_live_season_full_game_markets(tmp_path: Path) -> None:
    db_path = tmp_path / "live_tracking_full_game.db"

    result = DailyPipelineResult(
        run_id="run-20260403",
        pipeline_date="2026-04-03",
        mode="prod",
        dry_run=True,
        model_version="live-test-v1",
        pick_count=2,
        no_pick_count=0,
        error_count=0,
        notification_type="picks",
        notification_payload={},
        games=[
            GameProcessingResult(
                game_pk=4001,
                matchup="PHI @ COL",
                status="pick",
                prediction=_prediction(4001, ml_home=0.45, rl_home=0.47),
                selected_decision=_decision(
                    4001,
                    market_type="full_game_total",
                    side="under",
                    odds=100,
                    line_at_bet=10.0,
                    model_probability=0.61,
                ),
                input_status={"full_game_total": 10.0},
            ),
            GameProcessingResult(
                game_pk=4002,
                matchup="HOU @ OAK",
                status="pick",
                prediction=_prediction(4002, ml_home=0.40, rl_home=0.42),
                selected_decision=_decision(
                    4002,
                    market_type="full_game_ml",
                    side="away",
                    odds=-120,
                    model_probability=0.58,
                ),
                input_status={"full_game_home_ml": 110, "full_game_away_ml": -120},
            ),
        ],
    )

    capture_daily_result(result=result, db_path=db_path)

    from src.db import init_db

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
                final_home_score = excluded.final_home_score,
                final_away_score = excluded.final_away_score,
                status = excluded.status
            """,
            [
                (4001, "2026-04-03", "COL", "PHI", None, None, "Coors", 0, 1, 3, 2, 4, 5, "final"),
                (4002, "2026-04-03", "OAK", "HOU", None, None, "Oakland", 0, 1, 1, 2, 3, 6, "final"),
            ],
        )
        connection.commit()

    settle_tracked_games(pipeline_date="2026-04-03", db_path=db_path)
    tracked = list_tracked_games(season=2026, pipeline_date="2026-04-03", db_path=db_path)
    by_game = {row["game_pk"]: row for row in tracked}
    assert by_game[4001]["settled_result"] == "WIN"
    assert by_game[4002]["settled_result"] == "WIN"


def test_submit_manual_bet_and_settle(tmp_path: Path) -> None:
    db_path = tmp_path / "manual_tracking.db"

    from src.db import init_db

    init_db(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO games (
                game_pk, date, home_team, away_team, home_starter_id, away_starter_id,
                venue, is_dome, is_abs_active, final_home_score, final_away_score, status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (5001, "2026-04-03", "WSH", "LAD", None, None, "Nationals Park", 0, 1, 2, 6, "scheduled"),
        )
        connection.commit()

    row = submit_manual_tracked_bet(
        season=2026,
        pipeline_date="2026-04-03",
        game_pk=5001,
        matchup="LAD @ WSH",
        market_type="full_game_ml",
        side="away",
        odds_at_bet=-110,
        bet_units=1.7,
        model_probability=0.58,
        fair_probability=0.51,
        edge_pct=0.07,
        db_path=db_path,
    )
    assert row["selected_market_type"] == "full_game_ml"
    assert float(row["bet_units"]) == 1.5

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE games SET status = 'final', final_home_score = 2, final_away_score = 6 WHERE game_pk = 5001"
        )
        connection.commit()

    updated = settle_manual_tracked_bets(pipeline_date="2026-04-03", db_path=db_path)
    assert updated == 1

    manual_rows = list_manual_tracked_bets(season=2026, db_path=db_path)
    assert len(manual_rows) == 1
    assert manual_rows[0]["settled_result"] == "WIN"

    summary = build_manual_tracking_summary(season=2026, db_path=db_path)
    assert summary.picks == 1
    assert summary.graded_picks == 1
    assert summary.wins == 1


def test_settle_outstanding_tracking_dates_settles_prior_machine_and_manual_rows(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "outstanding_tracking.db"

    result = DailyPipelineResult(
        run_id="run-20260401",
        pipeline_date="2026-04-01",
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
                game_pk=6001,
                matchup="NYY @ BOS",
                status="pick",
                prediction=_prediction(6001, ml_home=0.42, rl_home=0.40),
                selected_decision=_decision(
                    6001,
                    market_type="full_game_total",
                    side="under",
                    odds=-110,
                    line_at_bet=8.5,
                    model_probability=0.61,
                ),
                input_status={"full_game_total": 8.5},
            )
        ],
    )
    capture_daily_result(result=result, db_path=db_path)

    from src.db import init_db

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
                final_home_score = excluded.final_home_score,
                final_away_score = excluded.final_away_score,
                status = excluded.status
            """,
            [
                (6001, "2026-04-01", "BOS", "NYY", None, None, "Fenway", 0, 1, 1, 0, 5, 3, "final"),
                (6002, "2026-04-02", "LAD", "SD", None, None, "Dodger", 0, 1, 0, 0, 2, 4, "scheduled"),
            ],
        )
        connection.commit()

    submit_manual_tracked_bet(
        season=2026,
        pipeline_date="2026-04-02",
        game_pk=6002,
        matchup="SD @ LAD",
        market_type="full_game_ml",
        side="away",
        odds_at_bet=125,
        fair_probability=0.50,
        model_probability=0.56,
        edge_pct=0.06,
        ev=0.1,
        kelly_stake=1.0,
        bet_units=1.5,
        book_name="Bet365",
        db_path=db_path,
    )

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE games SET status = 'final', final_home_score = 2, final_away_score = 4 WHERE game_pk = 6002"
        )
        connection.commit()

    updated = _settle_outstanding_tracking_dates(
        season=2026,
        through_date=date(2026, 4, 3),
        db_path=db_path,
    )
    assert updated == 2

    tracked = list_tracked_games(season=2026, db_path=db_path)
    manual_rows = list_manual_tracked_bets(season=2026, db_path=db_path)
    assert tracked[0]["settled_result"] in {"WIN", "LOSS", "PUSH"}
    assert manual_rows[0]["settled_result"] in {"WIN", "LOSS", "PUSH"}
