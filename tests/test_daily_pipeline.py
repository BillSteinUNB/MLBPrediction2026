from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

import pandas as pd

from src.db import init_db
from src.models.lineup import Lineup, LineupPlayer
from src.models.odds import OddsSnapshot
from src.models.prediction import Prediction
from src.pipeline.daily import PipelineDependencies, run_daily_pipeline


class _RecordingPredictionEngine:
    model_version = "test-model-v1"

    def __init__(self, predictions: dict[int, Prediction], errors: dict[int, Exception] | None = None) -> None:
        self._predictions = predictions
        self._errors = errors or {}

    def predict(self, inference_frame: pd.DataFrame) -> Prediction:
        game_pk = int(inference_frame.iloc[0]["game_pk"])
        if game_pk in self._errors:
            raise self._errors[game_pk]
        return self._predictions[game_pk]


class _RecordingNotifier:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def send_picks(self, **payload: object) -> dict[str, object]:
        self.calls.append(("picks", payload))
        return {"type": "picks", **payload}

    def send_no_picks(self, **payload: object) -> dict[str, object]:
        self.calls.append(("no_picks", payload))
        return {"type": "no_picks", **payload}

    def send_failure_alert(self, **payload: object) -> dict[str, object]:
        self.calls.append(("failure_alert", payload))
        return {"type": "failure_alert", **payload}

    def send_drawdown_alert(self, **payload: object) -> dict[str, object]:
        self.calls.append(("drawdown_alert", payload))
        return {"type": "drawdown_alert", **payload}


def _schedule_frame(*, game_pks: list[int], final_scores: dict[int, tuple[int, int]] | None = None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, game_pk in enumerate(game_pks, start=1):
        home_team = "NYY" if index % 2 else "BOS"
        away_team = "BOS" if index % 2 else "NYY"
        f5_home_score, f5_away_score = (final_scores or {}).get(game_pk, (None, None))
        rows.append(
            {
                "game_pk": game_pk,
                "season": 2025,
                "game_date": "2025-09-15",
                "scheduled_start": f"2025-09-15T{17 + index:02d}:05:00+00:00",
                "home_team": home_team,
                "away_team": away_team,
                "home_starter_id": 1000 + game_pk,
                "away_starter_id": 2000 + game_pk,
                "venue": f"Stadium {index}",
                "is_dome": False,
                "is_abs_active": True,
                "park_runs_factor": 1.02,
                "park_hr_factor": 1.01,
                "game_type": "R",
                "status": "final" if f5_home_score is not None else "scheduled",
                "f5_home_score": f5_home_score,
                "f5_away_score": f5_away_score,
                "final_home_score": f5_home_score,
                "final_away_score": f5_away_score,
            }
        )
    return pd.DataFrame(rows)


def _make_lineups(game_pks: list[int]) -> list[Lineup]:
    players = [
        LineupPlayer(
            batting_order=slot,
            player_id=10000 + slot,
            player_name=f"Player {slot}",
            position="1B",
        )
        for slot in range(1, 10)
    ]
    lineups: list[Lineup] = []
    for index, game_pk in enumerate(game_pks, start=1):
        teams = (("NYY", 1000 + game_pk), ("BOS", 2000 + game_pk)) if index % 2 else (("BOS", 1000 + game_pk), ("NYY", 2000 + game_pk))
        for team, starter_id in teams:
            lineups.append(
                Lineup(
                    game_pk=game_pk,
                    team=team,
                    source="test",
                    confirmed=True,
                    as_of_timestamp=datetime(2025, 9, 15, tzinfo=UTC),
                    starting_pitcher_id=starter_id,
                    projected_starting_pitcher_id=starter_id,
                    starter_avg_innings_pitched=5.4,
                    is_opener=False,
                    is_bullpen_game=False,
                    players=players,
                )
            )
    return lineups


def _prediction(game_pk: int, *, ml_home: float, rl_home: float) -> Prediction:
    return Prediction(
        game_pk=game_pk,
        model_version="test-model-v1",
        f5_ml_home_prob=ml_home,
        f5_ml_away_prob=1 - ml_home,
        f5_rl_home_prob=rl_home,
        f5_rl_away_prob=1 - rl_home,
        predicted_at=datetime(2025, 9, 15, tzinfo=UTC),
    )


def _odds_snapshot(game_pk: int, *, market_type: str, home_odds: int, away_odds: int) -> OddsSnapshot:
    return OddsSnapshot(
        game_pk=game_pk,
        book_name="TestBook",
        market_type=market_type,
        home_odds=home_odds,
        away_odds=away_odds,
        fetched_at=datetime(2025, 9, 15, tzinfo=UTC),
    )


def _feature_frame_builder(**kwargs: object) -> pd.DataFrame:
    schedule = kwargs["schedule"]
    return pd.DataFrame(
        [
            {
                "game_pk": int(row["game_pk"]),
                "scheduled_start": row["scheduled_start"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "venue": row["venue"],
                "park_runs_factor": 1.02,
                "park_hr_factor": 1.01,
                "home_team_f5_pythagorean_wp_30g": 0.56,
                "home_team_log5_30g": 0.57,
                "weather_composite": 1.0,
                "weather_data_missing": 0.0,
            }
            for row in schedule.to_dict(orient="records")
        ]
    )


def _dependencies(
    *,
    schedule: pd.DataFrame,
    notifier: _RecordingNotifier,
    prediction_engine: _RecordingPredictionEngine,
    odds: list[OddsSnapshot],
    feature_frame_builder: Callable[..., pd.DataFrame] = _feature_frame_builder,
) -> PipelineDependencies:
    return PipelineDependencies(
        schedule_fetcher=lambda target_date, mode: schedule.copy(),
        history_fetcher=lambda season, before_date: pd.DataFrame(),
        lineups_fetcher=lambda target_date: _make_lineups(schedule["game_pk"].astype(int).tolist()),
        odds_fetcher=lambda target_date, mode, db_path: list(odds),
        feature_frame_builder=feature_frame_builder,
        prediction_engine=prediction_engine,
        notifier=notifier,
    )


def test_run_daily_pipeline_dry_run_persists_predictions_and_returns_pick_payload(tmp_path: Path) -> None:
    db_path = tmp_path / "dry_run.db"
    schedule = _schedule_frame(game_pks=[1001])
    notifier = _RecordingNotifier()
    prediction_engine = _RecordingPredictionEngine({1001: _prediction(1001, ml_home=0.62, rl_home=0.56)})
    dependencies = _dependencies(
        schedule=schedule,
        notifier=notifier,
        prediction_engine=prediction_engine,
        odds=[_odds_snapshot(1001, market_type="f5_ml", home_odds=-110, away_odds=100)],
    )

    result = run_daily_pipeline(
        target_date="2025-09-15",
        mode="backtest",
        dry_run=True,
        db_path=db_path,
        dependencies=dependencies,
    )

    assert result.pick_count == 1
    assert result.no_pick_count == 0
    assert result.error_count == 0
    assert result.notification_type == "picks"
    assert notifier.calls[0][0] == "picks"
    assert notifier.calls[0][1]["dry_run"] is True

    with sqlite3.connect(db_path) as connection:
        prediction_count = connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        stored_result = connection.execute(
            "SELECT status, selected_market_type, selected_side, no_pick_reason FROM daily_pipeline_results"
        ).fetchone()
        frozen_count = connection.execute(
            "SELECT COUNT(*) FROM odds_snapshots WHERE is_frozen = 1"
        ).fetchone()[0]

    assert prediction_count == 1
    assert stored_result == ("pick", "f5_ml", "home", None)
    assert frozen_count == 0


def test_run_daily_pipeline_records_explicit_no_pick_when_odds_are_missing(tmp_path: Path) -> None:
    db_path = tmp_path / "no_pick.db"
    schedule = _schedule_frame(game_pks=[1001])
    notifier = _RecordingNotifier()
    prediction_engine = _RecordingPredictionEngine({1001: _prediction(1001, ml_home=0.58, rl_home=0.53)})
    dependencies = _dependencies(
        schedule=schedule,
        notifier=notifier,
        prediction_engine=prediction_engine,
        odds=[],
    )

    result = run_daily_pipeline(
        target_date="2025-09-15",
        mode="backtest",
        dry_run=True,
        db_path=db_path,
        dependencies=dependencies,
    )

    assert result.pick_count == 0
    assert result.no_pick_count == 1
    assert result.notification_type == "no_picks"
    assert notifier.calls[0][0] == "no_picks"

    with sqlite3.connect(db_path) as connection:
        prediction_count = connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        stored_reason = connection.execute(
            "SELECT no_pick_reason FROM daily_pipeline_results"
        ).fetchone()[0]

    assert prediction_count == 1
    assert stored_reason == "odds unavailable"


def test_run_daily_pipeline_records_explicit_no_pick_when_weather_is_missing(tmp_path: Path) -> None:
    db_path = tmp_path / "weather_no_pick.db"
    schedule = _schedule_frame(game_pks=[1001])
    notifier = _RecordingNotifier()
    prediction_engine = _RecordingPredictionEngine({1001: _prediction(1001, ml_home=0.58, rl_home=0.53)})

    def _weather_missing_feature_frame_builder(**kwargs: object) -> pd.DataFrame:
        frame = _feature_frame_builder(**kwargs)
        frame["weather_data_missing"] = 1.0
        return frame

    dependencies = _dependencies(
        schedule=schedule,
        notifier=notifier,
        prediction_engine=prediction_engine,
        odds=[_odds_snapshot(1001, market_type="f5_ml", home_odds=-110, away_odds=100)],
        feature_frame_builder=_weather_missing_feature_frame_builder,
    )

    result = run_daily_pipeline(
        target_date="2025-09-15",
        mode="backtest",
        dry_run=True,
        db_path=db_path,
        dependencies=dependencies,
    )

    assert result.pick_count == 0
    assert result.no_pick_count == 1
    assert result.notification_type == "no_picks"

    with sqlite3.connect(db_path) as connection:
        stored_reason = connection.execute(
            "SELECT no_pick_reason FROM daily_pipeline_results"
        ).fetchone()[0]

    assert stored_reason == "weather unavailable"


def test_run_daily_pipeline_continues_after_per_game_prediction_error(tmp_path: Path) -> None:
    db_path = tmp_path / "partial_success.db"
    schedule = _schedule_frame(game_pks=[1001, 1002])
    notifier = _RecordingNotifier()
    prediction_engine = _RecordingPredictionEngine(
        {1002: _prediction(1002, ml_home=0.63, rl_home=0.57)},
        errors={1001: RuntimeError("boom")},
    )
    dependencies = _dependencies(
        schedule=schedule,
        notifier=notifier,
        prediction_engine=prediction_engine,
        odds=[
            _odds_snapshot(1001, market_type="f5_ml", home_odds=-110, away_odds=100),
            _odds_snapshot(1002, market_type="f5_ml", home_odds=-110, away_odds=100),
        ],
    )

    result = run_daily_pipeline(
        target_date="2025-09-15",
        mode="backtest",
        dry_run=True,
        db_path=db_path,
        dependencies=dependencies,
    )

    assert result.pick_count == 1
    assert result.error_count == 1
    assert result.no_pick_count == 0

    with sqlite3.connect(db_path) as connection:
        statuses = connection.execute(
            "SELECT game_pk, status, error_message FROM daily_pipeline_results ORDER BY game_pk"
        ).fetchall()

    assert statuses == [
        (1001, "error", "boom"),
        (1002, "pick", None),
    ]


def test_run_daily_pipeline_non_dry_run_backtest_freezes_odds_and_settles_bets(tmp_path: Path) -> None:
    db_path = tmp_path / "backtest_settle.db"
    schedule = _schedule_frame(game_pks=[1001], final_scores={1001: (3, 1)})
    notifier = _RecordingNotifier()
    prediction_engine = _RecordingPredictionEngine({1001: _prediction(1001, ml_home=0.64, rl_home=0.58)})
    dependencies = _dependencies(
        schedule=schedule,
        notifier=notifier,
        prediction_engine=prediction_engine,
        odds=[_odds_snapshot(1001, market_type="f5_ml", home_odds=-110, away_odds=100)],
    )

    result = run_daily_pipeline(
        target_date="2025-09-15",
        mode="backtest",
        dry_run=False,
        db_path=db_path,
        dependencies=dependencies,
    )

    assert result.pick_count == 1
    assert result.notification_type == "picks"

    with sqlite3.connect(db_path) as connection:
        frozen_count = connection.execute(
            "SELECT COUNT(*) FROM odds_snapshots WHERE is_frozen = 1"
        ).fetchone()[0]
        bet_row = connection.execute(
            "SELECT result FROM bets WHERE game_pk = 1001 ORDER BY id DESC LIMIT 1"
        ).fetchone()
        ledger_rows = connection.execute(
            "SELECT COUNT(*) FROM bankroll_ledger"
        ).fetchone()[0]

    assert frozen_count == 1
    assert bet_row == ("WIN",)
    assert ledger_rows >= 2


def test_run_daily_pipeline_sends_drawdown_alert_when_kill_switch_is_active(tmp_path: Path) -> None:
    db_path = tmp_path / "killswitch.db"
    init_db(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "INSERT INTO bankroll_ledger (timestamp, event_type, amount, running_balance, notes) VALUES (?, ?, ?, ?, ?)",
            ("2025-09-14T00:00:00+00:00", "bet_settled", 0.0, 1000.0, "seed peak"),
        )
        connection.execute(
            "INSERT INTO bankroll_ledger (timestamp, event_type, amount, running_balance, notes) VALUES (?, ?, ?, ?, ?)",
            ("2025-09-14T01:00:00+00:00", "bet_settled", -310.0, 690.0, "seed drawdown"),
        )
        connection.commit()

    schedule = _schedule_frame(game_pks=[1001])
    notifier = _RecordingNotifier()
    prediction_engine = _RecordingPredictionEngine({1001: _prediction(1001, ml_home=0.66, rl_home=0.59)})
    dependencies = _dependencies(
        schedule=schedule,
        notifier=notifier,
        prediction_engine=prediction_engine,
        odds=[_odds_snapshot(1001, market_type="f5_ml", home_odds=-110, away_odds=100)],
    )

    result = run_daily_pipeline(
        target_date="2025-09-15",
        mode="backtest",
        dry_run=False,
        db_path=db_path,
        dependencies=dependencies,
    )

    assert result.pick_count == 0
    assert result.no_pick_count == 1
    assert result.notification_type == "drawdown_alert"
    assert notifier.calls[0][0] == "drawdown_alert"

    with sqlite3.connect(db_path) as connection:
        stored_reason = connection.execute(
            "SELECT no_pick_reason FROM daily_pipeline_results WHERE game_pk = 1001"
        ).fetchone()[0]
        bet_count = connection.execute("SELECT COUNT(*) FROM bets").fetchone()[0]

    assert stored_reason == "kill-switch active"
    assert bet_count == 0
