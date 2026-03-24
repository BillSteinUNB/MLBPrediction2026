from __future__ import annotations

import sqlite3
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import joblib
import pandas as pd
import pytest

from src.db import init_db
from src.model.artifact_runtime import collect_runtime_versions
from src.model.calibration import CalibratedStackingModel, IdentityProbabilityCalibrator
from src.models.bet import BetDecision
from src.models.lineup import Lineup, LineupPlayer
from src.models.odds import OddsSnapshot
from src.models.prediction import Prediction
from src.pipeline.daily import (
    ArtifactOrFallbackPredictionEngine,
    PipelineDependencies,
    _parse_schedule_game,
    _select_game_decision,
    _default_feature_frame_builder,
    run_daily_pipeline,
)


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


def _dummy_calibrated_model() -> CalibratedStackingModel:
    stacking_model = SimpleNamespace(
        base_feature_columns=["park_runs_factor"],
        raw_meta_feature_columns=["home_team_log5_30g"],
        predict_proba=lambda dataframe: [[0.4, 0.6] for _ in range(len(dataframe))],
    )
    return CalibratedStackingModel(
        model_name="dummy",
        target_column="dummy_target",
        stacking_model=stacking_model,
        calibrator=IdentityProbabilityCalibrator(),
    )


def _dummy_base_model() -> SimpleNamespace:
    return SimpleNamespace(
        predict_proba=lambda dataframe: [[0.4, 0.6] for _ in range(len(dataframe))],
    )


def _write_legacy_metadata(
    path: Path,
    *,
    variant: str,
    version: str,
    holdout_season: int = 2025,
    log_loss: float,
    roc_auc: float,
    accuracy: float,
    brier: float | None = None,
    runtime_versions: dict[str, str] | None = None,
) -> None:
    resolved_runtime_versions = runtime_versions or collect_runtime_versions()
    if variant == "base":
        payload = {
            "model_version": version,
            "holdout_season": holdout_season,
            "runtime_versions": resolved_runtime_versions,
            "feature_columns": ["park_runs_factor"],
            "holdout_metrics": {
                "log_loss": log_loss,
                "roc_auc": roc_auc,
                "accuracy": accuracy,
                **({"brier": brier} if brier is not None else {}),
            },
        }
    elif variant == "stacking":
        payload = {
            "model_version": version,
            "holdout_season": holdout_season,
            "runtime_versions": resolved_runtime_versions,
            "feature_columns": ["park_runs_factor"],
            "raw_meta_feature_columns": ["home_team_log5_30g"],
            "holdout_metrics": {
                "stacked_log_loss": log_loss,
                "stacked_roc_auc": roc_auc,
                "stacked_accuracy": accuracy,
                "stacked_brier": brier,
            },
            "persisted": True,
        }
    else:
        payload = {
            "model_version": version,
            "holdout_season": holdout_season,
            "runtime_versions": resolved_runtime_versions,
            "holdout_metrics": {
                "calibrated_log_loss": log_loss,
                "calibrated_roc_auc": roc_auc,
                "calibrated_accuracy": accuracy,
                "calibrated_brier": brier,
            },
        }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_artifact_engine_prefers_best_holdout_variant_over_calibrated_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_dir = tmp_path / "exp"
    experiment_dir.mkdir()
    version = "20260321T010000Z_bbbb2222"

    ml_base = experiment_dir / f"f5_ml_model_{version}.joblib"
    ml_stack = experiment_dir / f"f5_ml_stacking_model_{version}.joblib"
    ml_cal = experiment_dir / f"f5_ml_calibrated_model_{version}.joblib"
    rl_base = experiment_dir / f"f5_rl_model_{version}.joblib"
    rl_stack = experiment_dir / f"f5_rl_stacking_model_{version}.joblib"
    rl_cal = experiment_dir / f"f5_rl_calibrated_model_{version}.joblib"
    for path in (ml_base, ml_stack, ml_cal, rl_base, rl_stack, rl_cal):
        path.write_text("placeholder", encoding="utf-8")

    _write_legacy_metadata(
        ml_base.with_suffix(".metadata.json"),
        variant="base",
        version=version,
        log_loss=0.60,
        roc_auc=0.61,
        accuracy=0.58,
    )
    _write_legacy_metadata(
        ml_stack.with_suffix(".metadata.json"),
        variant="stacking",
        version=version,
        log_loss=0.62,
        roc_auc=0.60,
        accuracy=0.57,
        brier=0.24,
    )
    _write_legacy_metadata(
        ml_cal.with_suffix(".metadata.json"),
        variant="calibrated",
        version=version,
        log_loss=0.62,
        roc_auc=0.60,
        accuracy=0.57,
        brier=0.24,
    )
    _write_legacy_metadata(
        rl_base.with_suffix(".metadata.json"),
        variant="base",
        version=version,
        log_loss=0.55,
        roc_auc=0.63,
        accuracy=0.68,
    )
    _write_legacy_metadata(
        rl_stack.with_suffix(".metadata.json"),
        variant="stacking",
        version=version,
        log_loss=0.58,
        roc_auc=0.61,
        accuracy=0.67,
        brier=0.21,
    )
    _write_legacy_metadata(
        rl_cal.with_suffix(".metadata.json"),
        variant="calibrated",
        version=version,
        log_loss=0.58,
        roc_auc=0.61,
        accuracy=0.67,
        brier=0.21,
    )

    def _fake_load(path: Path) -> object:
        if "calibrated" in path.name:
            return _dummy_calibrated_model()
        return _dummy_base_model()

    monkeypatch.setattr(joblib, "load", _fake_load)

    engine = ArtifactOrFallbackPredictionEngine(model_dir=tmp_path)

    assert engine.ml_model_path == ml_base
    assert engine.rl_model_path == rl_base
    assert "ml=" in engine.model_version
    assert "base" in engine.model_version
    assert engine.ml_model is not None
    assert engine.rl_model is not None


def test_artifact_engine_falls_back_without_complete_recursive_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    incomplete_dir = tmp_path / "incomplete-exp"
    incomplete_dir.mkdir()
    version = "20260321T010000Z_onlyml"
    ml_path = incomplete_dir / f"f5_ml_model_{version}.joblib"
    ml_path.write_text(
        "placeholder",
        encoding="utf-8",
    )
    _write_legacy_metadata(
        ml_path.with_suffix(".metadata.json"),
        variant="base",
        version=version,
        log_loss=0.6,
        roc_auc=0.6,
        accuracy=0.57,
    )

    monkeypatch.setattr(joblib, "load", lambda path: _dummy_base_model())

    engine = ArtifactOrFallbackPredictionEngine(model_dir=tmp_path)

    assert engine.ml_model_path is None
    assert engine.rl_model_path is None
    assert engine.model_version == "baseline-fallback"


def test_artifact_engine_skips_incompatible_runtime_versions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_dir = tmp_path / "exp"
    experiment_dir.mkdir()
    version = "20260321T010000Z_badv"

    ml_base = experiment_dir / f"f5_ml_model_{version}.joblib"
    rl_base = experiment_dir / f"f5_rl_model_{version}.joblib"
    for path in (ml_base, rl_base):
        path.write_text("placeholder", encoding="utf-8")

    bad_versions = {**collect_runtime_versions(), "xgboost": "999.0.0"}
    _write_legacy_metadata(
        ml_base.with_suffix(".metadata.json"),
        variant="base",
        version=version,
        log_loss=0.60,
        roc_auc=0.61,
        accuracy=0.58,
        runtime_versions=bad_versions,
    )
    _write_legacy_metadata(
        rl_base.with_suffix(".metadata.json"),
        variant="base",
        version=version,
        log_loss=0.55,
        roc_auc=0.63,
        accuracy=0.68,
        runtime_versions=bad_versions,
    )

    monkeypatch.setattr(joblib, "load", lambda path: _dummy_base_model())

    engine = ArtifactOrFallbackPredictionEngine(model_dir=tmp_path)

    assert engine.model_version == "baseline-fallback"
    assert engine.ml_model_path is None
    assert engine.rl_model_path is None


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
    assert len(notifier.calls) == 1
    assert notifier.calls[0][0] == "picks"
    assert notifier.calls[0][1]["dry_run"] is True
    assert notifier.calls[0][1]["bankroll_summary"] == {
        "current_bankroll": 1000.0,
        "peak_bankroll": 1000.0,
        "drawdown_pct": 0.0,
        "total_bets": 0,
        "win_rate": 0.0,
        "roi": 0.0,
        "kill_switch_active": False,
    }

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
    assert stored_reason == "f5 odds unavailable"


def test_run_daily_pipeline_uses_preview_estimated_f5_odds_in_dry_run(tmp_path: Path) -> None:
    db_path = tmp_path / "preview_estimate.db"
    schedule = _schedule_frame(game_pks=[1001])
    notifier = _RecordingNotifier()
    prediction_engine = _RecordingPredictionEngine({1001: _prediction(1001, ml_home=0.72, rl_home=0.53)})
    dependencies = PipelineDependencies(
        schedule_fetcher=lambda target_date, mode: schedule.copy(),
        history_fetcher=lambda season, before_date: pd.DataFrame(),
        lineups_fetcher=lambda target_date: [],
        odds_fetcher=lambda target_date, mode, db_path: [],
        full_game_odds_fetcher=lambda target_date, mode, db_path: {
            1001: {
                "full_game_odds_available": True,
                "full_game_odds_books": ["DraftKings"],
                "full_game_home_ml": -150,
                "full_game_home_ml_book": "DraftKings",
                "full_game_away_ml": 130,
                "full_game_away_ml_book": "DraftKings",
                "full_game_home_spread": -1.5,
                "full_game_home_spread_odds": 160,
                "full_game_home_spread_book": "DraftKings",
                "full_game_away_spread": 1.5,
                "full_game_away_spread_odds": -180,
                "full_game_away_spread_book": "DraftKings",
                "full_game_ml_pairs": [
                    {"book_name": "DraftKings", "home_odds": -150, "away_odds": 130}
                ],
            }
        },
        feature_frame_builder=_feature_frame_builder,
        prediction_engine=prediction_engine,
        notifier=notifier,
    )

    result = run_daily_pipeline(
        target_date="2025-09-15",
        mode="prod",
        dry_run=True,
        db_path=db_path,
        dependencies=dependencies,
    )

    assert result.pick_count == 1
    assert result.games[0].paper_fallback is True
    assert result.games[0].input_status is not None
    assert result.games[0].input_status["f5_odds_estimated"] is True
    assert "estimated from full-game market" in result.games[0].input_status["f5_odds_sources"]


def test_parse_schedule_game_prefers_official_date_over_utc_rollover() -> None:
    game = {
        "gamePk": 823244,
        "gameType": "R",
        "officialDate": "2026-03-25",
        "gameDate": "2026-03-26T00:05:00Z",
        "status": {"detailedState": "Scheduled"},
        "teams": {
            "home": {"team": {"abbreviation": "SF"}, "probablePitcher": {"id": 657277}},
            "away": {"team": {"abbreviation": "NYY"}, "probablePitcher": {"id": 608331}},
        },
        "venue": {"name": "Oracle Park"},
        "linescore": {"innings": []},
    }

    row = _parse_schedule_game(game)

    assert row is not None
    assert row["game_date"] == "2026-03-25"


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


def test_run_daily_pipeline_sends_only_failure_alert_when_pick_side_effect_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "post_send_failure.db"
    schedule = _schedule_frame(game_pks=[1001])
    notifier = _RecordingNotifier()
    prediction_engine = _RecordingPredictionEngine({1001: _prediction(1001, ml_home=0.64, rl_home=0.58)})
    dependencies = _dependencies(
        schedule=schedule,
        notifier=notifier,
        prediction_engine=prediction_engine,
        odds=[_odds_snapshot(1001, market_type="f5_ml", home_odds=-110, away_odds=100)],
    )

    def _raise_update_bankroll(**_kwargs: object) -> None:
        raise RuntimeError("bankroll unavailable")

    monkeypatch.setattr("src.pipeline.daily.update_bankroll", _raise_update_bankroll)

    result = run_daily_pipeline(
        target_date="2025-09-15",
        mode="backtest",
        dry_run=False,
        db_path=db_path,
        dependencies=dependencies,
    )

    assert result.pick_count == 0
    assert result.error_count == 1
    assert result.notification_type == "failure_alert"
    assert [call[0] for call in notifier.calls] == ["failure_alert"]

    with sqlite3.connect(db_path) as connection:
        stored_row = connection.execute(
            "SELECT status, error_message, notified FROM daily_pipeline_results"
        ).fetchone()
        bet_count = connection.execute("SELECT COUNT(*) FROM bets").fetchone()[0]
        frozen_count = connection.execute(
            "SELECT COUNT(*) FROM odds_snapshots WHERE is_frozen = 1"
        ).fetchone()[0]

    assert stored_row == ("error", "bankroll unavailable", 0)
    assert bet_count == 0
    assert frozen_count == 0


def test_run_daily_pipeline_rolls_back_pick_side_effects_when_settlement_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "post_place_failure.db"
    schedule = _schedule_frame(game_pks=[1001], final_scores={1001: (3, 1)})
    notifier = _RecordingNotifier()
    prediction_engine = _RecordingPredictionEngine({1001: _prediction(1001, ml_home=0.64, rl_home=0.58)})
    dependencies = _dependencies(
        schedule=schedule,
        notifier=notifier,
        prediction_engine=prediction_engine,
        odds=[_odds_snapshot(1001, market_type="f5_ml", home_odds=-110, away_odds=100)],
    )

    def _raise_settlement_failure(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("settlement unavailable")

    monkeypatch.setattr("src.pipeline.daily.settle_game_bets", _raise_settlement_failure)

    result = run_daily_pipeline(
        target_date="2025-09-15",
        mode="backtest",
        dry_run=False,
        db_path=db_path,
        dependencies=dependencies,
    )

    assert result.pick_count == 0
    assert result.error_count == 1
    assert result.notification_type == "failure_alert"
    assert [call[0] for call in notifier.calls] == ["failure_alert"]

    with sqlite3.connect(db_path) as connection:
        stored_row = connection.execute(
            "SELECT status, error_message, notified FROM daily_pipeline_results"
        ).fetchone()
        bet_count = connection.execute("SELECT COUNT(*) FROM bets").fetchone()[0]
        frozen_count = connection.execute(
            "SELECT COUNT(*) FROM odds_snapshots WHERE is_frozen = 1"
        ).fetchone()[0]
        ledger_count = connection.execute("SELECT COUNT(*) FROM bankroll_ledger").fetchone()[0]

    assert stored_row == ("error", "settlement unavailable", 0)
    assert bet_count == 0
    assert frozen_count == 0
    assert ledger_count == 0


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
    recommendations = notifier.calls[0][1]["recommendations"]
    assert len(recommendations) == 1
    assert recommendations[0]["matchup"] == "BOS @ NYY"
    assert recommendations[0]["scheduled_start"] == "2025-09-15T18:05:00+00:00"
    assert recommendations[0]["market"] == "f5_ml home"
    assert recommendations[0]["odds"] == "-110"
    assert recommendations[0]["model_probability"] == 0.66
    assert recommendations[0]["edge_pct"] == pytest.approx(result.games[0].selected_decision.edge_pct)
    assert recommendations[0]["kelly_stake"] == 0.0
    assert recommendations[0]["venue"] == "Stadium 1"
    assert recommendations[0]["weather"] == "composite=1.00, wind=0.00"
    assert result.games[0].selected_decision is not None
    assert result.games[0].selected_decision.kelly_stake == 0.0

    with sqlite3.connect(db_path) as connection:
        stored_reason = connection.execute(
            "SELECT status, selected_market_type, selected_side, kelly_stake, no_pick_reason FROM daily_pipeline_results WHERE game_pk = 1001"
        ).fetchone()
        bet_count = connection.execute("SELECT COUNT(*) FROM bets").fetchone()[0]

    assert stored_reason == ("no_pick", "f5_ml", "home", 0.0, "kill-switch active")
    assert bet_count == 0


def test_half_runline_candidates_use_ml_equivalent_probability(tmp_path: Path) -> None:
    engine = ArtifactOrFallbackPredictionEngine(model_dir=tmp_path)
    frame = pd.DataFrame([{"game_pk": 1001}])
    prediction = _prediction(1001, ml_home=0.62, rl_home=0.30)
    db_path = init_db(tmp_path / "edge.db")
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO games (game_pk, date, home_team, away_team, venue, status)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (1001, "2025-09-15", "NYY", "BOS", "Stadium 1", "scheduled"),
        )
        connection.commit()
    snapshot = OddsSnapshot(
        game_pk=1001,
        book_name="TestBook",
        market_type="f5_rl",
        home_odds=-110,
        away_odds=100,
        home_point=-0.5,
        away_point=0.5,
        fetched_at=datetime(2025, 9, 15, tzinfo=UTC),
    )

    candidates = engine.build_candidate_decisions(
        inference_frame=frame,
        prediction=prediction,
        snapshots=[snapshot],
        db_path=db_path,
    )

    legacy_home = next(
        candidate
        for candidate in candidates
        if candidate.source_model == "legacy_f5_ml_equiv" and candidate.side == "home"
    )
    legacy_away = next(
        candidate
        for candidate in candidates
        if candidate.source_model == "legacy_f5_ml_equiv" and candidate.side == "away"
    )

    assert legacy_home.model_probability == pytest.approx(0.62)
    assert legacy_away.model_probability == pytest.approx(0.38)


def test_official_selector_prefers_promoted_direct_runline_candidates() -> None:
    ml_candidate = BetDecision(
        game_pk=1001,
        market_type="f5_ml",
        side="away",
        source_model="legacy_f5_ml",
        source_model_version="test-model-v1",
        book_name="TestBook",
        model_probability=0.58,
        fair_probability=0.48,
        edge_pct=0.10,
        ev=0.15,
        is_positive_ev=True,
        odds_at_bet=110,
    )
    experimental_rl_candidate = BetDecision(
        game_pk=1001,
        market_type="f5_rl",
        side="away",
        source_model="rlv2_direct",
        source_model_version="rlv2-test",
        book_name="TestBook",
        model_probability=0.67,
        fair_probability=0.45,
        edge_pct=0.13,
        ev=0.20,
        is_positive_ev=True,
        odds_at_bet=115,
        line_at_bet=-0.5,
    )

    selected, kill_switch = _select_game_decision(
        candidates=[ml_candidate, experimental_rl_candidate],
        current_bankroll=100.0,
        peak_bankroll=100.0,
    )

    assert kill_switch is False
    assert selected is not None
    assert selected.market_type == "f5_rl"
    assert selected.source_model == "rlv2_direct"
    assert selected.kelly_stake > 0.0


def test_official_selector_rejects_out_of_range_odds() -> None:
    wide_dog_candidate = BetDecision(
        game_pk=1001,
        market_type="f5_ml",
        side="away",
        source_model="legacy_f5_ml",
        source_model_version="test-model-v1",
        book_name="TestBook",
        model_probability=0.51,
        fair_probability=0.40,
        edge_pct=0.11,
        ev=0.20,
        is_positive_ev=True,
        odds_at_bet=200,
    )
    short_favorite_candidate = BetDecision(
        game_pk=1001,
        market_type="f5_ml",
        side="home",
        source_model="legacy_f5_ml",
        source_model_version="test-model-v1",
        book_name="TestBook",
        model_probability=0.56,
        fair_probability=0.48,
        edge_pct=0.08,
        ev=0.10,
        is_positive_ev=True,
        odds_at_bet=-120,
    )

    selected, kill_switch = _select_game_decision(
        candidates=[wide_dog_candidate, short_favorite_candidate],
        current_bankroll=100.0,
        peak_bankroll=100.0,
    )

    assert kill_switch is False
    assert selected is not None
    assert selected.side == "home"
    assert selected.odds_at_bet == -120


def test_official_selector_rejects_untrusted_edge_spikes() -> None:
    suspicious_candidate = BetDecision(
        game_pk=1001,
        market_type="f5_ml",
        side="away",
        source_model="legacy_f5_ml",
        source_model_version="test-model-v1",
        book_name="TestBook",
        model_probability=0.72,
        fair_probability=0.44,
        edge_pct=0.28,
        ev=0.55,
        is_positive_ev=True,
        odds_at_bet=115,
    )
    trusted_candidate = BetDecision(
        game_pk=1001,
        market_type="f5_ml",
        side="home",
        source_model="legacy_f5_ml",
        source_model_version="test-model-v1",
        book_name="TestBook",
        model_probability=0.55,
        fair_probability=0.47,
        edge_pct=0.08,
        ev=0.09,
        is_positive_ev=True,
        odds_at_bet=-105,
    )

    selected, kill_switch = _select_game_decision(
        candidates=[suspicious_candidate, trusted_candidate],
        current_bankroll=100.0,
        peak_bankroll=100.0,
    )

    assert kill_switch is False
    assert selected is not None
    assert selected.side == "home"
    assert selected.edge_pct == pytest.approx(0.08)


def test_default_feature_frame_builder_uses_live_feature_assembly_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    db_path = init_db(tmp_path / "live_builder.db")
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO features (game_pk, feature_name, feature_value, window_size, as_of_timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            (1001, "home_team_woba_7g", -999.0, 7, "2025-09-14T00:00:00+00:00"),
        )
        connection.commit()

    schedule = _schedule_frame(game_pks=[1001])
    historical_games = pd.DataFrame(
        [
            {
                "game_pk": 999,
                "season": 2025,
                "game_date": "2025-09-14",
                "scheduled_start": "2025-09-14T23:05:00+00:00",
                "home_team": "NYY",
                "away_team": "BOS",
                "home_starter_id": 1999,
                "away_starter_id": 2999,
                "venue": "Stadium 0",
                "is_dome": False,
                "is_abs_active": True,
                "park_runs_factor": 1.02,
                "park_hr_factor": 1.01,
                "game_type": "R",
                "status": "final",
                "f5_home_score": 2,
                "f5_away_score": 1,
                "final_home_score": 4,
                "final_away_score": 3,
            }
        ]
    )
    fresh_frame = pd.DataFrame(
        [
            {
                "game_pk": 1001,
                "season": 2025,
                "game_date": "2025-09-15",
                "scheduled_start": "2025-09-15T18:05:00+00:00",
                "as_of_timestamp": "2025-09-14T00:00:00+00:00",
                "home_team": "NYY",
                "away_team": "BOS",
                "venue": "Stadium 1",
                "game_type": "R",
                "status": "scheduled",
                "home_team_woba_7g": 0.612,
                "weather_data_missing": 0.0,
            }
        ]
    )
    captured: dict[str, object] = {}

    def _fake_build_live_feature_frame(**kwargs: object) -> pd.DataFrame:
        captured.update(kwargs)
        return fresh_frame.copy()

    monkeypatch.setattr("src.pipeline.daily.build_live_feature_frame", _fake_build_live_feature_frame)

    frame = _default_feature_frame_builder(
        target_date=datetime(2025, 9, 15, tzinfo=UTC).date(),
        schedule=schedule,
        historical_games=historical_games,
        lineups=_make_lineups([1001]),
        db_path=db_path,
        weather_fetcher=lambda *_args, **_kwargs: None,
    )

    assert frame["game_pk"].tolist() == [1001]
    assert frame.iloc[0]["home_team_woba_7g"] == 0.612
    assert frame.iloc[0]["home_team_woba_7g"] != -999.0
    assert captured["historical_games"].equals(historical_games)
    assert captured["lineups"] == _make_lineups([1001])
