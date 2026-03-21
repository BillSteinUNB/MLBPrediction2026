from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Callable, Literal, Protocol, Sequence

import httpx
import joblib
import pandas as pd

from src.clients.lineup_client import fetch_confirmed_lineups
from src.clients.odds_client import fetch_mlb_odds, freeze_odds
from src.clients.weather_client import fetch_game_weather
from src.config import _load_settings_yaml
from src.db import DEFAULT_DB_PATH, init_db
from src.engine.bankroll import calculate_kelly_stake, get_bankroll_summary, update_bankroll
from src.engine.edge_calculator import calculate_edge
from src.engine.settlement import settle_game_bets
from src.features.adjustments.abs_adjustment import is_abs_active
from src.features.adjustments.park_factors import get_park_factors
from src.model.calibration import CalibratedStackingModel
from src.model.data_builder import (
    _default_feature_fill_value,
    _fetch_regular_season_schedule,
    _normalize_game_status,
    _normalize_team_code,
    _prepare_schedule_frame,
    build_live_feature_frame,
)
from src.model.xgboost_trainer import DEFAULT_MODEL_OUTPUT_DIR
from src.models.bet import BetDecision
from src.models.lineup import Lineup
from src.models.odds import OddsSnapshot
from src.models.prediction import Prediction
from src.notifications.discord import (
    send_drawdown_alert,
    send_failure_alert,
    send_no_picks,
    send_picks,
)
from src.ops.error_handler import (
    CircuitBreaker,
    call_with_graceful_degradation,
    notify_fatal_error,
    retry,
)
from src.ops.logging_config import configure_logging
from src.ops.performance_tracker import sync_closing_lines_from_snapshots


logger = logging.getLogger(__name__)

_SCHEDULE_CIRCUIT = CircuitBreaker(name="schedule")
_HISTORY_CIRCUIT = CircuitBreaker(name="history")
_LINEUPS_CIRCUIT = CircuitBreaker(name="lineups")
_ODDS_CIRCUIT = CircuitBreaker(name="odds")

Mode = Literal["prod", "backtest"]
SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
_SETTINGS = _load_settings_yaml()


ScheduleFetcher = Callable[[date, Mode], pd.DataFrame]
HistoryFetcher = Callable[[int, date], pd.DataFrame]
LineupsFetcher = Callable[[str], list[Lineup]]
OddsFetcher = Callable[[date, Mode, str | Path], list[OddsSnapshot]]
FeatureFrameBuilder = Callable[..., pd.DataFrame]


class PredictionEngine(Protocol):
    model_version: str

    def predict(self, inference_frame: pd.DataFrame) -> Prediction: ...


class PipelineNotifier(Protocol):
    def send_picks(self, **payload: Any) -> dict[str, Any]: ...

    def send_no_picks(self, **payload: Any) -> dict[str, Any]: ...

    def send_failure_alert(self, **payload: Any) -> dict[str, Any]: ...

    def send_drawdown_alert(self, **payload: Any) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class PipelineDependencies:
    schedule_fetcher: ScheduleFetcher | None = None
    history_fetcher: HistoryFetcher | None = None
    lineups_fetcher: LineupsFetcher | None = None
    odds_fetcher: OddsFetcher | None = None
    feature_frame_builder: FeatureFrameBuilder | None = None
    prediction_engine: PredictionEngine | None = None
    notifier: PipelineNotifier | None = None
    weather_fetcher: Callable[..., Any] | None = fetch_game_weather


@dataclass(slots=True)
class GameProcessingResult:
    game_pk: int
    matchup: str
    status: Literal["pick", "no_pick", "error"]
    prediction: Prediction | None = None
    selected_decision: BetDecision | None = None
    no_pick_reason: str | None = None
    error_message: str | None = None
    notified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_pk": self.game_pk,
            "matchup": self.matchup,
            "status": self.status,
            "prediction": self.prediction.model_dump(mode="json") if self.prediction else None,
            "selected_decision": (
                self.selected_decision.model_dump(mode="json") if self.selected_decision else None
            ),
            "no_pick_reason": self.no_pick_reason,
            "error_message": self.error_message,
            "notified": self.notified,
        }


@dataclass(frozen=True, slots=True)
class DailyPipelineResult:
    run_id: str
    pipeline_date: str
    mode: Mode
    dry_run: bool
    model_version: str
    pick_count: int
    no_pick_count: int
    error_count: int
    notification_type: str
    notification_payload: dict[str, Any]
    games: list[GameProcessingResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "pipeline_date": self.pipeline_date,
            "mode": self.mode,
            "dry_run": self.dry_run,
            "model_version": self.model_version,
            "pick_count": self.pick_count,
            "no_pick_count": self.no_pick_count,
            "error_count": self.error_count,
            "notification_type": self.notification_type,
            "notification_payload": self.notification_payload,
            "games": [game.to_dict() for game in self.games],
        }


class DiscordNotifier:
    def send_picks(self, **payload: Any) -> dict[str, Any]:
        return send_picks(**payload)

    def send_no_picks(self, **payload: Any) -> dict[str, Any]:
        return send_no_picks(**payload)

    def send_failure_alert(self, **payload: Any) -> dict[str, Any]:
        return send_failure_alert(**payload)

    def send_drawdown_alert(self, **payload: Any) -> dict[str, Any]:
        return send_drawdown_alert(**payload)


class ArtifactOrFallbackPredictionEngine:
    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_OUTPUT_DIR) -> None:
        self.model_dir = Path(model_dir)
        self.ml_model_path, self.rl_model_path, self.model_version = self._resolve_model_bundle()
        self.ml_model = self._load_model(self.ml_model_path)
        self.rl_model = self._load_model(self.rl_model_path)

    def predict(self, inference_frame: pd.DataFrame) -> Prediction:
        resolved_frame = inference_frame.copy().reset_index(drop=True)
        if resolved_frame.empty:
            raise ValueError("Inference frame must contain exactly one game row")

        game_pk = int(resolved_frame.iloc[0]["game_pk"])
        prediction_time = datetime.now(UTC)
        resolved_frame = self._ensure_required_columns(resolved_frame)

        if self.ml_model is not None and self.rl_model is not None:
            ml_home_probability = float(self.ml_model.predict_calibrated(resolved_frame)[0])
            rl_home_probability = float(self.rl_model.predict_calibrated(resolved_frame)[0])
        else:
            ml_home_probability = _fallback_ml_home_probability(resolved_frame.iloc[0])
            rl_home_probability = _fallback_rl_home_probability(
                resolved_frame.iloc[0],
                ml_home_probability=ml_home_probability,
            )

        return Prediction(
            game_pk=game_pk,
            model_version=self.model_version,
            f5_ml_home_prob=ml_home_probability,
            f5_ml_away_prob=1.0 - ml_home_probability,
            f5_rl_home_prob=rl_home_probability,
            f5_rl_away_prob=1.0 - rl_home_probability,
            predicted_at=prediction_time,
        )

    def _ensure_required_columns(self, inference_frame: pd.DataFrame) -> pd.DataFrame:
        resolved = inference_frame.copy()
        required_columns: set[str] = set()
        for model in (self.ml_model, self.rl_model):
            if model is None:
                continue
            required_columns.update(model.stacking_model.base_feature_columns)
            required_columns.update(model.stacking_model.raw_meta_feature_columns)

        for column in required_columns:
            if column not in resolved.columns:
                resolved[column] = _default_feature_fill_value(column)

        return resolved

    def _resolve_model_bundle(self) -> tuple[Path | None, Path | None, str]:
        ml_candidates = self._collect_model_candidates("f5_ml_calibrated_model")
        rl_candidates = self._collect_model_candidates("f5_rl_calibrated_model")
        shared_versions = set(ml_candidates) & set(rl_candidates)

        if not shared_versions:
            logger.warning(
                "No complete calibrated model bundle found under %s; daily pipeline will use baseline fallback",
                self.model_dir,
            )
            return None, None, "baseline-fallback"

        selected_version = max(
            shared_versions,
            key=lambda version: (
                max(
                    ml_candidates[version].stat().st_mtime,
                    rl_candidates[version].stat().st_mtime,
                ),
                version,
            ),
        )
        ml_model_path = ml_candidates[selected_version]
        rl_model_path = rl_candidates[selected_version]
        logger.info(
            "Loaded calibrated model bundle version=%s ml=%s rl=%s",
            selected_version,
            ml_model_path,
            rl_model_path,
        )
        return ml_model_path, rl_model_path, selected_version

    def _collect_model_candidates(self, prefix: str) -> dict[str, Path]:
        candidates: dict[str, Path] = {}
        pattern = f"{prefix}_*.joblib"
        for path in self.model_dir.rglob(pattern):
            version = path.name.removeprefix(f"{prefix}_").removesuffix(".joblib")
            current = candidates.get(version)
            if current is None or path.stat().st_mtime > current.stat().st_mtime:
                candidates[version] = path
        return candidates

    def _load_model(self, path: Path | None) -> CalibratedStackingModel | None:
        if path is None:
            return None

        loaded = joblib.load(path)
        return loaded if isinstance(loaded, CalibratedStackingModel) else None

def run_daily_pipeline(
    *,
    target_date: str | date | datetime,
    mode: Mode = "prod",
    dry_run: bool = False,
    db_path: str | Path = DEFAULT_DB_PATH,
    dependencies: PipelineDependencies | None = None,
    starting_bankroll: float = 1000.0,
) -> DailyPipelineResult:
    """Run the daily MLB F5 pipeline for a target date."""

    resolved_dependencies = dependencies or PipelineDependencies()
    pipeline_day = _coerce_date(target_date)
    database_path = init_db(db_path)
    _ensure_pipeline_tables(database_path)

    schedule_fetcher = resolved_dependencies.schedule_fetcher or _default_schedule_fetcher
    history_fetcher = resolved_dependencies.history_fetcher or _default_history_fetcher
    lineups_fetcher = resolved_dependencies.lineups_fetcher or _default_lineups_fetcher
    odds_fetcher = resolved_dependencies.odds_fetcher or _default_odds_fetcher
    feature_frame_builder = (
        resolved_dependencies.feature_frame_builder or _default_feature_frame_builder
    )
    prediction_engine = (
        resolved_dependencies.prediction_engine or ArtifactOrFallbackPredictionEngine()
    )
    notifier = resolved_dependencies.notifier or DiscordNotifier()
    weather_fetcher = resolved_dependencies.weather_fetcher

    schedule = schedule_fetcher(pipeline_day, mode)
    if schedule.empty:
        payload = notifier.send_no_picks(
            pipeline_date=pipeline_day.isoformat(),
            reasons=["no scheduled games"],
            dry_run=dry_run,
        )
        result = DailyPipelineResult(
            run_id=_build_run_id(pipeline_day),
            pipeline_date=pipeline_day.isoformat(),
            mode=mode,
            dry_run=dry_run,
            model_version=prediction_engine.model_version,
            pick_count=0,
            no_pick_count=0,
            error_count=0,
            notification_type="no_picks",
            notification_payload=payload,
            games=[],
        )
        return result

    historical_games = history_fetcher(pipeline_day.year, pipeline_day)
    if not historical_games.empty:
        _upsert_games(database_path, historical_games)
    _upsert_games(database_path, schedule)

    lineups = lineups_fetcher(pipeline_day.isoformat())
    odds = odds_fetcher(pipeline_day, mode, database_path)
    _persist_odds_snapshots(database_path, odds)

    inference_frame = feature_frame_builder(
        target_date=pipeline_day,
        schedule=schedule,
        historical_games=historical_games,
        lineups=lineups,
        db_path=database_path,
        weather_fetcher=weather_fetcher,
    )

    run_id = _build_run_id(pipeline_day)
    lineups_by_game_team = {(lineup.game_pk, lineup.team): lineup for lineup in lineups}
    odds_by_game = _group_odds_by_game(odds)
    schedule_lookup = {
        int(row["game_pk"]): row for row in schedule.to_dict(orient="records")
    }

    current_bankroll, peak_bankroll, drawdown_pct = _load_bankroll_state(
        database_path,
        starting_bankroll=starting_bankroll,
    )
    kill_switch_triggered = drawdown_pct >= 0.30
    virtual_bankroll = current_bankroll
    results: list[GameProcessingResult] = []

    for game_pk in schedule["game_pk"].astype(int).tolist():
        game = schedule_lookup[game_pk]
        matchup = f"{game['away_team']} @ {game['home_team']}"
        row_frame = inference_frame.loc[inference_frame["game_pk"] == game_pk].reset_index(drop=True)

        try:
            if row_frame.empty:
                raise ValueError(f"No feature row available for game_pk={game_pk}")
            inference_row = row_frame.iloc[0]

            prediction = prediction_engine.predict(row_frame)
            _upsert_prediction(database_path, prediction)

            validation_reason = _validation_no_pick_reason(
                game=game,
                inference_row=inference_row,
                lineups_by_game_team=lineups_by_game_team,
                odds_by_game=odds_by_game,
            )
            if validation_reason is not None:
                results.append(
                    GameProcessingResult(
                        game_pk=game_pk,
                        matchup=matchup,
                        status="no_pick",
                        prediction=prediction,
                        no_pick_reason=validation_reason,
                    )
                )
                continue

            decision, kill_switch_active = _select_game_decision(
                prediction=prediction,
                snapshots=odds_by_game.get(game_pk, []),
                db_path=database_path,
                current_bankroll=virtual_bankroll,
                peak_bankroll=peak_bankroll,
            )
            if kill_switch_active:
                kill_switch_triggered = True
                results.append(
                    GameProcessingResult(
                        game_pk=game_pk,
                        matchup=matchup,
                        status="no_pick",
                        prediction=prediction,
                        selected_decision=decision,
                        no_pick_reason="kill-switch active",
                    )
                )
                continue

            if decision is None:
                results.append(
                    GameProcessingResult(
                        game_pk=game_pk,
                        matchup=matchup,
                        status="no_pick",
                        prediction=prediction,
                        no_pick_reason="edge below threshold",
                    )
                )
                continue

            virtual_bankroll = max(0.0, virtual_bankroll - float(decision.kelly_stake))
            results.append(
                GameProcessingResult(
                    game_pk=game_pk,
                    matchup=matchup,
                    status="pick",
                    prediction=prediction,
                    selected_decision=decision,
                )
            )
        except Exception as exc:
            logger.warning("Game %s failed during daily pipeline", game_pk, exc_info=True)
            results.append(
                GameProcessingResult(
                    game_pk=game_pk,
                    matchup=matchup,
                    status="error",
                    error_message=str(exc),
                )
            )

    _apply_pick_side_effects(
        database_path,
        mode=mode,
        dry_run=dry_run,
        results=results,
        schedule_lookup=schedule_lookup,
        starting_bankroll=starting_bankroll,
    )

    _persist_game_results(
        database_path,
        run_id=run_id,
        pipeline_date=pipeline_day.isoformat(),
        mode=mode,
        dry_run=dry_run,
        results=results,
    )

    notification_type, notification_payload = _send_daily_notification(
        notifier=notifier,
        pipeline_date=pipeline_day.isoformat(),
        dry_run=dry_run,
        results=results,
        schedule_lookup=schedule_lookup,
        inference_frame=inference_frame,
        bankroll_summary=get_bankroll_summary(
            db_path=database_path,
            starting_bankroll=starting_bankroll,
        ).model_dump(mode="json"),
        drawdown_pct=drawdown_pct,
        kill_switch_triggered=kill_switch_triggered,
    )

    if notification_type == "picks" and not dry_run:
        for result in results:
            if result.status == "pick" and result.selected_decision is not None:
                result.notified = True
        try:
            _mark_pick_results_notified(
                database_path,
                run_id=run_id,
                results=results,
            )
        except sqlite3.Error:
            logger.warning("Failed to update daily pipeline notified flags", exc_info=True)

    return DailyPipelineResult(
        run_id=run_id,
        pipeline_date=pipeline_day.isoformat(),
        mode=mode,
        dry_run=dry_run,
        model_version=prediction_engine.model_version,
        pick_count=sum(result.status == "pick" for result in results),
        no_pick_count=sum(result.status == "no_pick" for result in results),
        error_count=sum(result.status == "error" for result in results),
        notification_type=notification_type,
        notification_payload=notification_payload,
        games=results,
    )


def _fetch_schedule_payload(target_date: date) -> dict[str, Any]:
    with httpx.Client(timeout=60.0) as client:
        response = client.get(
            SCHEDULE_URL,
            params={
                "sportId": 1,
                "date": target_date.isoformat(),
                "hydrate": "linescore,probablePitcher,venue,team",
            },
        )
        response.raise_for_status()

    return response.json()


def _fetch_historical_schedule(season: int) -> pd.DataFrame:
    return _prepare_schedule_frame(_fetch_regular_season_schedule(season))


def _fetch_lineups_with_retry(target_date: str) -> list[Lineup]:
    return fetch_confirmed_lineups(target_date)


def _fetch_live_odds_with_retry(
    *,
    db_path: str | Path,
    commence_time_from: datetime,
    commence_time_to: datetime,
) -> list[OddsSnapshot]:
    return fetch_mlb_odds(
        db_path=db_path,
        commence_time_from=commence_time_from,
        commence_time_to=commence_time_to,
    )


def _retry_with_circuit_breaker(
    operation: Callable[..., Any],
    breaker: CircuitBreaker,
    *args: Any,
    **kwargs: Any,
) -> Any:
    @retry(logger_=logger)
    def _wrapped() -> Any:
        return breaker.call(operation, *args, **kwargs)

    return _wrapped()


def _default_schedule_fetcher(target_date: date, mode: Mode) -> pd.DataFrame:
    del mode
    payload = _retry_with_circuit_breaker(_fetch_schedule_payload, _SCHEDULE_CIRCUIT, target_date)
    rows: list[dict[str, Any]] = []
    for date_entry in payload.get("dates", []):
        for game in date_entry.get("games", []):
            row = _parse_schedule_game(game)
            if row is not None:
                rows.append(row)

    return pd.DataFrame(rows)


def _default_history_fetcher(season: int, before_date: date) -> pd.DataFrame:
    schedule = call_with_graceful_degradation(
        lambda: _retry_with_circuit_breaker(_fetch_historical_schedule, _HISTORY_CIRCUIT, season),
        operation_name=f"historical schedule fetch for {season}",
        fallback=pd.DataFrame(),
        logger_=logger,
    )
    if schedule.empty:
        return schedule
    return schedule.loc[
        pd.to_datetime(schedule["scheduled_start"], utc=True).dt.date < before_date
    ].reset_index(drop=True)


def _default_lineups_fetcher(target_date: str) -> list[Lineup]:
    return call_with_graceful_degradation(
        lambda: _retry_with_circuit_breaker(_fetch_lineups_with_retry, _LINEUPS_CIRCUIT, target_date),
        operation_name=f"lineup fetch for {target_date}",
        fallback=[],
        logger_=logger,
    )


def _default_odds_fetcher(target_date: date, mode: Mode, db_path: str | Path) -> list[OddsSnapshot]:
    if mode == "prod":
        start = datetime.combine(target_date, time.min, tzinfo=UTC)
        end = start + timedelta(days=1)
        return call_with_graceful_degradation(
            lambda: _retry_with_circuit_breaker(
                _fetch_live_odds_with_retry,
                _ODDS_CIRCUIT,
                db_path=db_path,
                commence_time_from=start,
                commence_time_to=end,
            ),
            operation_name=f"live odds fetch for {target_date.isoformat()}",
            fallback=lambda _exc: _load_odds_from_db_for_date(db_path, target_date),
            logger_=logger,
        )

    return _load_odds_from_db_for_date(db_path, target_date)


def _default_feature_frame_builder(
    *,
    target_date: date,
    schedule: pd.DataFrame,
    historical_games: pd.DataFrame,
    lineups: Sequence[Lineup],
    db_path: str | Path,
    weather_fetcher: Callable[..., Any] | None,
) -> pd.DataFrame:
    database_path = Path(db_path)
    return build_live_feature_frame(
        target_date=target_date,
        schedule=schedule,
        historical_games=historical_games,
        db_path=database_path,
        lineups=lineups,
        weather_fetcher=weather_fetcher,
    )


def _parse_schedule_game(game: dict[str, Any]) -> dict[str, Any] | None:
    game_type = str(game.get("gameType") or "").upper()
    if game_type != "R":
        return None

    scheduled_start = _coerce_timestamp(game.get("gameDate"))
    home_payload = game.get("teams", {}).get("home", {})
    away_payload = game.get("teams", {}).get("away", {})
    home_team = _normalize_team_code(
        home_payload.get("team", {}).get("abbreviation")
        or home_payload.get("team", {}).get("name")
    )
    away_team = _normalize_team_code(
        away_payload.get("team", {}).get("abbreviation")
        or away_payload.get("team", {}).get("name")
    )
    if home_team is None or away_team is None:
        return None

    venue = str(
        game.get("venue", {}).get("name")
        or game.get("venue", {}).get("locationName")
        or home_team
    )
    park = get_park_factors(team_code=home_team, venue=venue)
    linescore = game.get("linescore", {})
    innings = linescore.get("innings") or []

    return {
        "game_pk": int(game["gamePk"]),
        "season": int(scheduled_start.year),
        "game_date": scheduled_start.date().isoformat(),
        "scheduled_start": scheduled_start.isoformat(),
        "home_team": home_team,
        "away_team": away_team,
        "home_starter_id": _optional_int(home_payload.get("probablePitcher", {}).get("id")),
        "away_starter_id": _optional_int(away_payload.get("probablePitcher", {}).get("id")),
        "venue": venue,
        "is_dome": bool(_SETTINGS["stadiums"].get(home_team, {}).get("is_dome", False)),
        "is_abs_active": bool(is_abs_active(venue)),
        "park_runs_factor": float(park.runs),
        "park_hr_factor": float(park.hr),
        "game_type": game_type,
        "status": _normalize_game_status(game.get("status", {}).get("detailedState")),
        "f5_home_score": sum(_inning_runs(inning, "home") for inning in innings[:5]) if innings else None,
        "f5_away_score": sum(_inning_runs(inning, "away") for inning in innings[:5]) if innings else None,
        "final_home_score": _optional_int(home_payload.get("score")),
        "final_away_score": _optional_int(away_payload.get("score")),
    }


def _ensure_pipeline_tables(db_path: str | Path) -> None:
    database_path = init_db(db_path)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_pipeline_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                pipeline_date TEXT NOT NULL,
                game_pk INTEGER NOT NULL,
                mode TEXT NOT NULL CHECK (mode IN ('prod', 'backtest')),
                dry_run INTEGER NOT NULL CHECK (dry_run IN (0, 1)),
                status TEXT NOT NULL CHECK (status IN ('pick', 'no_pick', 'error')),
                model_version TEXT,
                f5_ml_home_prob REAL,
                f5_ml_away_prob REAL,
                f5_rl_home_prob REAL,
                f5_rl_away_prob REAL,
                selected_market_type TEXT,
                selected_side TEXT,
                odds_at_bet INTEGER,
                fair_probability REAL,
                model_probability REAL,
                edge_pct REAL,
                ev REAL,
                kelly_stake REAL,
                no_pick_reason TEXT,
                error_message TEXT,
                notified INTEGER NOT NULL DEFAULT 0 CHECK (notified IN (0, 1)),
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (game_pk) REFERENCES games (game_pk)
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_daily_pipeline_results_game ON daily_pipeline_results (game_pk)"
        )
        connection.commit()


def _upsert_games(db_path: str | Path, schedule: pd.DataFrame) -> None:
    if schedule.empty:
        return

    rows = [
        (
            int(game["game_pk"]),
            str(game.get("game_date") or _coerce_timestamp(game["scheduled_start"]).date().isoformat()),
            str(game["home_team"]),
            str(game["away_team"]),
            _optional_int(game.get("home_starter_id")),
            _optional_int(game.get("away_starter_id")),
            str(game["venue"]),
            int(bool(game.get("is_dome", False))),
            int(bool(game.get("is_abs_active", True))),
            _optional_int(game.get("f5_home_score")),
            _optional_int(game.get("f5_away_score")),
            _optional_int(game.get("final_home_score")),
            _optional_int(game.get("final_away_score")),
            str(game.get("status", "scheduled")),
        )
        for game in schedule.to_dict(orient="records")
    ]

    with sqlite3.connect(db_path) as connection:
        connection.executemany(
            """
            INSERT OR REPLACE INTO games (
                game_pk,
                date,
                home_team,
                away_team,
                home_starter_id,
                away_starter_id,
                venue,
                is_dome,
                is_abs_active,
                f5_home_score,
                f5_away_score,
                final_home_score,
                final_away_score,
                status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        connection.commit()


def _persist_odds_snapshots(db_path: str | Path, snapshots: Sequence[OddsSnapshot]) -> None:
    if not snapshots:
        return

    with sqlite3.connect(db_path) as connection:
        connection.executemany(
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
            [
                (
                    snapshot.game_pk,
                    snapshot.book_name,
                    snapshot.market_type,
                    snapshot.home_odds,
                    snapshot.away_odds,
                    snapshot.fetched_at.isoformat(),
                    int(snapshot.is_frozen),
                )
                for snapshot in snapshots
            ],
        )
        for game_pk, market_type in {(snapshot.game_pk, snapshot.market_type) for snapshot in snapshots}:
            sync_closing_lines_from_snapshots(
                game_pk=game_pk,
                market_type=market_type,
                connection=connection,
                commit=False,
            )
        connection.commit()


def _load_odds_from_db_for_date(db_path: str | Path, target_date: date) -> list[OddsSnapshot]:
    database_path = Path(db_path)
    if not database_path.exists():
        return []

    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            """
            SELECT o.game_pk, o.book_name, o.market_type, o.home_odds, o.away_odds, o.fetched_at, o.is_frozen
            FROM odds_snapshots AS o
            INNER JOIN games AS g ON g.game_pk = o.game_pk
            WHERE substr(g.date, 1, 10) = ?
            ORDER BY o.game_pk, o.fetched_at DESC, o.id DESC
            """,
            (target_date.isoformat(),),
        ).fetchall()

    return [
        OddsSnapshot(
            game_pk=row[0],
            book_name=row[1],
            market_type=row[2],
            home_odds=row[3],
            away_odds=row[4],
            fetched_at=_coerce_timestamp(row[5]),
            is_frozen=bool(row[6]),
        )
        for row in rows
    ]


def _upsert_prediction(db_path: str | Path, prediction: Prediction) -> None:
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO predictions (
                game_pk,
                model_version,
                f5_ml_home_prob,
                f5_ml_away_prob,
                f5_rl_home_prob,
                f5_rl_away_prob,
                predicted_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(game_pk, model_version) DO UPDATE SET
                f5_ml_home_prob = excluded.f5_ml_home_prob,
                f5_ml_away_prob = excluded.f5_ml_away_prob,
                f5_rl_home_prob = excluded.f5_rl_home_prob,
                f5_rl_away_prob = excluded.f5_rl_away_prob,
                predicted_at = excluded.predicted_at
            """,
            (
                prediction.game_pk,
                prediction.model_version,
                prediction.f5_ml_home_prob,
                prediction.f5_ml_away_prob,
                prediction.f5_rl_home_prob,
                prediction.f5_rl_away_prob,
                prediction.predicted_at.isoformat(),
            ),
        )
        connection.commit()


def _group_odds_by_game(snapshots: Sequence[OddsSnapshot]) -> dict[int, list[OddsSnapshot]]:
    grouped: dict[int, list[OddsSnapshot]] = {}
    for snapshot in snapshots:
        grouped.setdefault(snapshot.game_pk, []).append(snapshot)
    return grouped


def _validation_no_pick_reason(
    *,
    game: dict[str, Any],
    inference_row: pd.Series,
    lineups_by_game_team: dict[tuple[int, str], Lineup],
    odds_by_game: dict[int, list[OddsSnapshot]],
) -> str | None:
    game_pk = int(game["game_pk"])
    reasons: list[str] = []

    for side, starter_key in (("home", "home_starter_id"), ("away", "away_starter_id")):
        team = str(game[f"{side}_team"])
        lineup = lineups_by_game_team.get((game_pk, team))
        if lineup is None or not lineup.players:
            reasons.append("lineup unavailable")
            break
        if (lineup.starting_pitcher_id or lineup.projected_starting_pitcher_id or _optional_int(game.get(starter_key))) is None:
            reasons.append("starter unavailable")
            break

    if not odds_by_game.get(game_pk):
        reasons.append("odds unavailable")

    if (not bool(game.get("is_dome", False))) and float(
        inference_row.get("weather_data_missing", 0.0) or 0.0
    ) >= 1.0:
        reasons.append("weather unavailable")

    return "; ".join(reasons) if reasons else None


def _select_game_decision(
    *,
    prediction: Prediction,
    snapshots: Sequence[OddsSnapshot],
    db_path: str | Path,
    current_bankroll: float,
    peak_bankroll: float,
) -> tuple[BetDecision | None, bool]:
    candidates: list[BetDecision] = []
    for snapshot in snapshots:
        if snapshot.market_type == "f5_ml":
            home_probability = prediction.f5_ml_home_prob
            away_probability = prediction.f5_ml_away_prob
        else:
            home_probability = prediction.f5_rl_home_prob
            away_probability = prediction.f5_rl_away_prob

        candidates.extend(
            [
                calculate_edge(
                    game_pk=prediction.game_pk,
                    market_type=snapshot.market_type,
                    side="home",
                    model_probability=home_probability,
                    home_odds=snapshot.home_odds,
                    away_odds=snapshot.away_odds,
                    book_name=snapshot.book_name,
                    db_path=db_path,
                ),
                calculate_edge(
                    game_pk=prediction.game_pk,
                    market_type=snapshot.market_type,
                    side="away",
                    model_probability=away_probability,
                    home_odds=snapshot.home_odds,
                    away_odds=snapshot.away_odds,
                    book_name=snapshot.book_name,
                    db_path=db_path,
                ),
            ]
        )

    positive_candidates = [candidate for candidate in candidates if candidate.is_positive_ev]
    if not positive_candidates:
        return None, False

    grouped_by_side: dict[str, list[BetDecision]] = {}
    for candidate in positive_candidates:
        grouped_by_side.setdefault(candidate.side, []).append(candidate)

    ranked_groups: list[tuple[float, float, BetDecision]] = []
    kill_switch_recommendations: list[tuple[float, BetDecision]] = []
    kill_switch_active = False
    for decisions in grouped_by_side.values():
        kelly = calculate_kelly_stake(
            current_bankroll,
            correlated_decisions=decisions,
            peak_bankroll=peak_bankroll,
        )
        selected_market_type = kelly.selected_market_type or decisions[0].market_type
        selected_side = kelly.selected_side or decisions[0].side
        selected_decision = next(
            decision
            for decision in decisions
            if decision.market_type == selected_market_type and decision.side == selected_side
        )
        if kelly.kill_switch_active:
            kill_switch_active = True
            kill_switch_recommendations.append(
                (
                    float(selected_decision.edge_pct),
                    selected_decision.model_copy(update={"kelly_stake": 0.0}),
                )
            )
            continue
        if kelly.stake <= 0:
            continue

        ranked_groups.append(
            (
                float(kelly.stake),
                float(selected_decision.edge_pct),
                selected_decision.model_copy(update={"kelly_stake": float(kelly.stake)}),
            )
        )

    if not ranked_groups:
        if kill_switch_recommendations:
            kill_switch_recommendations.sort(key=lambda item: item[0], reverse=True)
            return kill_switch_recommendations[0][1], True
        return None, kill_switch_active

    ranked_groups.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return ranked_groups[0][2], False


def _load_bankroll_state(
    db_path: str | Path,
    *,
    starting_bankroll: float,
) -> tuple[float, float, float]:
    with sqlite3.connect(db_path) as connection:
        current_row = connection.execute(
            "SELECT running_balance FROM bankroll_ledger ORDER BY id DESC LIMIT 1"
        ).fetchone()
        peak_row = connection.execute(
            "SELECT MAX(running_balance) FROM bankroll_ledger"
        ).fetchone()

    current_bankroll = float(current_row[0]) if current_row else float(starting_bankroll)
    peak_bankroll = max(float(starting_bankroll), float(peak_row[0] or 0.0))
    drawdown_pct = ((peak_bankroll - current_bankroll) / peak_bankroll) if peak_bankroll else 0.0
    return current_bankroll, peak_bankroll, drawdown_pct


def _send_daily_notification(
    *,
    notifier: PipelineNotifier,
    pipeline_date: str,
    dry_run: bool,
    results: Sequence[GameProcessingResult],
    schedule_lookup: dict[int, dict[str, Any]],
    inference_frame: pd.DataFrame,
    bankroll_summary: dict[str, Any],
    drawdown_pct: float,
    kill_switch_triggered: bool,
) -> tuple[str, dict[str, Any]]:
    picks = [result for result in results if result.status == "pick" and result.selected_decision is not None]
    if picks:
        payload = notifier.send_picks(
            pipeline_date=pipeline_date,
            picks=[
                _build_pick_payload_item(
                    result,
                    schedule_lookup=schedule_lookup,
                    inference_frame=inference_frame,
                )
                for result in picks
            ],
            bankroll_summary=bankroll_summary,
            dry_run=dry_run,
        )
        return "picks", payload

    if kill_switch_triggered:
        recommendations = [
            _build_pick_payload_item(
                result,
                schedule_lookup=schedule_lookup,
                inference_frame=inference_frame,
            )
            for result in results
            if result.no_pick_reason == "kill-switch active" and result.selected_decision is not None
        ]
        payload = notifier.send_drawdown_alert(
            pipeline_date=pipeline_date,
            drawdown_pct=drawdown_pct,
            recommendations=recommendations,
            dry_run=dry_run,
        )
        return "drawdown_alert", payload

    if results and all(result.status == "error" for result in results):
        message = "; ".join(result.error_message or "unknown error" for result in results)
        payload = notifier.send_failure_alert(
            pipeline_date=pipeline_date,
            error_message=message,
            dry_run=dry_run,
        )
        return "failure_alert", payload

    reasons = [result.no_pick_reason for result in results if result.no_pick_reason]
    payload = notifier.send_no_picks(
        pipeline_date=pipeline_date,
        reasons=reasons,
        dry_run=dry_run,
    )
    return "no_picks", payload


def _build_pick_payload_item(
    result: GameProcessingResult,
    *,
    schedule_lookup: dict[int, dict[str, Any]],
    inference_frame: pd.DataFrame,
) -> dict[str, Any]:
    if result.selected_decision is None or result.prediction is None:
        raise ValueError("Pick payload requires both prediction and decision")

    game = schedule_lookup[result.game_pk]
    frame_row = inference_frame.loc[inference_frame["game_pk"] == result.game_pk].iloc[0]
    decision = result.selected_decision

    return {
        "matchup": result.matchup,
        "scheduled_start": str(game["scheduled_start"]),
        "market": f"{decision.market_type} {decision.side}",
        "odds": str(decision.odds_at_bet),
        "model_probability": float(decision.model_probability),
        "edge_pct": float(decision.edge_pct),
        "kelly_stake": float(decision.kelly_stake),
        "venue": str(game["venue"]),
        "weather": _weather_summary(frame_row),
    }


def _weather_summary(row: pd.Series) -> str:
    if float(row.get("weather_data_missing", 0.0) or 0.0) >= 1.0:
        return "neutral default"

    return (
        f"composite={float(row.get('weather_composite', 1.0)):.2f}, "
        f"wind={float(row.get('weather_wind_factor', 0.0)):.2f}"
    )


def _apply_pick_side_effects(
    db_path: str | Path,
    *,
    mode: Mode,
    dry_run: bool,
    results: Sequence[GameProcessingResult],
    schedule_lookup: dict[int, dict[str, Any]],
    starting_bankroll: float,
) -> None:
    if dry_run:
        return

    for result in results:
        if result.status != "pick" or result.selected_decision is None:
            continue

        try:
            with sqlite3.connect(db_path) as connection:
                connection.execute("PRAGMA foreign_keys = ON")
                update_bankroll(
                    action="place",
                    decision=result.selected_decision,
                    db_path=db_path,
                    connection=connection,
                    starting_bankroll=starting_bankroll,
                    timestamp=datetime.now(UTC),
                    commit=False,
                )
                freeze_odds(
                    result.game_pk,
                    db_path=db_path,
                    connection=connection,
                    market_type=result.selected_decision.market_type,
                    commit=False,
                )
                _maybe_settle_backtest_pick(
                    result.game_pk,
                    schedule_lookup[result.game_pk],
                    mode=mode,
                    db_path=db_path,
                    connection=connection,
                    starting_bankroll=starting_bankroll,
                    commit=False,
                )
        except Exception as exc:
            logger.warning("Game %s failed during pick finalization", result.game_pk, exc_info=True)
            result.status = "error"
            result.selected_decision = None
            result.no_pick_reason = None
            result.error_message = str(exc)
            result.notified = False


def _maybe_settle_backtest_pick(
    game_pk: int,
    game: dict[str, Any],
    *,
    mode: Mode,
    db_path: str | Path,
    connection: sqlite3.Connection | None = None,
    starting_bankroll: float,
    commit: bool = True,
) -> None:
    if mode != "backtest":
        return
    if str(game.get("status")) != "final":
        return

    settle_game_bets(
        game_pk,
        home_score=_optional_int(game.get("f5_home_score")),
        away_score=_optional_int(game.get("f5_away_score")),
        innings_completed=5.0,
        db_path=db_path,
        connection=connection,
        starting_bankroll=starting_bankroll,
        commit=commit,
    )


def _persist_game_results(
    db_path: str | Path,
    *,
    run_id: str,
    pipeline_date: str,
    mode: Mode,
    dry_run: bool,
    results: Sequence[GameProcessingResult],
) -> None:
    with sqlite3.connect(db_path) as connection:
        connection.executemany(
            """
            INSERT INTO daily_pipeline_results (
                run_id,
                pipeline_date,
                game_pk,
                mode,
                dry_run,
                status,
                model_version,
                f5_ml_home_prob,
                f5_ml_away_prob,
                f5_rl_home_prob,
                f5_rl_away_prob,
                selected_market_type,
                selected_side,
                odds_at_bet,
                fair_probability,
                model_probability,
                edge_pct,
                ev,
                kelly_stake,
                no_pick_reason,
                error_message,
                notified
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    pipeline_date,
                    result.game_pk,
                    mode,
                    int(dry_run),
                    result.status,
                    result.prediction.model_version if result.prediction else None,
                    result.prediction.f5_ml_home_prob if result.prediction else None,
                    result.prediction.f5_ml_away_prob if result.prediction else None,
                    result.prediction.f5_rl_home_prob if result.prediction else None,
                    result.prediction.f5_rl_away_prob if result.prediction else None,
                    result.selected_decision.market_type if result.selected_decision else None,
                    result.selected_decision.side if result.selected_decision else None,
                    result.selected_decision.odds_at_bet if result.selected_decision else None,
                    result.selected_decision.fair_probability if result.selected_decision else None,
                    result.selected_decision.model_probability if result.selected_decision else None,
                    result.selected_decision.edge_pct if result.selected_decision else None,
                    result.selected_decision.ev if result.selected_decision else None,
                    result.selected_decision.kelly_stake if result.selected_decision else None,
                    result.no_pick_reason,
                    result.error_message,
                    int(result.notified),
                )
                for result in results
            ],
        )
        connection.commit()


def _mark_pick_results_notified(
    db_path: str | Path,
    *,
    run_id: str,
    results: Sequence[GameProcessingResult],
) -> None:
    notified_rows = [
        (run_id, result.game_pk)
        for result in results
        if result.status == "pick" and result.selected_decision is not None
    ]
    if not notified_rows:
        return

    with sqlite3.connect(db_path) as connection:
        connection.executemany(
            "UPDATE daily_pipeline_results SET notified = 1 WHERE run_id = ? AND game_pk = ?",
            notified_rows,
        )
        connection.commit()


def _coerce_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if value == "today":
        return datetime.now(UTC).date()
    return date.fromisoformat(value)


def _coerce_timestamp(value: Any) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.to_pydatetime().astimezone(UTC)


def _optional_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _fallback_ml_home_probability(row: pd.Series) -> float:
    candidates = [
        row.get("home_team_log5_30g"),
        row.get("home_team_f5_pythagorean_wp_30g"),
        row.get("home_team_pythagorean_wp_30g"),
    ]
    values = [float(value) for value in candidates if value is not None and pd.notna(value)]
    if not values:
        return 0.5
    return min(0.99, max(0.01, float(sum(values) / len(values))))


def _fallback_rl_home_probability(row: pd.Series, *, ml_home_probability: float) -> float:
    home_f5 = row.get("home_team_f5_pythagorean_wp_30g")
    away_f5 = row.get("away_team_f5_pythagorean_wp_30g")
    differential = ml_home_probability - 0.5
    if home_f5 is not None and away_f5 is not None and pd.notna(home_f5) and pd.notna(away_f5):
        differential += (float(home_f5) - float(away_f5)) / 2.0
    return min(0.99, max(0.01, 0.5 + (differential * 1.2) - 0.05))


def _inning_runs(inning: dict[str, Any], side: str) -> int:
    return int(inning.get(side, {}).get("runs") or 0)


def _build_run_id(pipeline_day: date) -> str:
    return f"daily-{pipeline_day.isoformat()}-{datetime.now(UTC).strftime('%H%M%S')}"


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for the daily pipeline."""

    parser = argparse.ArgumentParser(description="Run the MLB F5 daily pipeline")
    parser.add_argument("--date", default="today")
    parser.add_argument("--mode", choices=["prod", "backtest"], default="prod")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--starting-bankroll", type=float, default=1000.0)
    args = parser.parse_args(argv)

    configure_logging(log_dir=Path(args.db_path).resolve().parent / "logs", log_name="pipeline")

    try:
        result = run_daily_pipeline(
            target_date=args.date,
            mode=args.mode,
            dry_run=args.dry_run,
            db_path=args.db_path,
            starting_bankroll=args.starting_bankroll,
        )
    except Exception as exc:
        payload = notify_fatal_error(
            pipeline_date=_coerce_date(args.date).isoformat(),
            error=exc,
            dry_run=args.dry_run,
            logger_=logger,
        )
        print(
            json.dumps(
                {
                    "pipeline_date": _coerce_date(args.date).isoformat(),
                    "mode": args.mode,
                    "dry_run": args.dry_run,
                    "error": str(exc),
                    "notification_payload": payload,
                },
                indent=2,
            )
        )
        return 1

    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
