from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Callable, Collection, Literal, Mapping, Protocol, Sequence

import httpx
import joblib
import pandas as pd

from src.clients.lineup_client import fetch_confirmed_lineups
from src.clients.odds_client import (
    build_estimated_f5_ml_snapshots,
    devig_probabilities,
    fetch_mlb_full_game_odds_context,
    fetch_mlb_odds,
    fetch_sbr_f5_odds,
    freeze_odds,
)
from src.clients.weather_client import fetch_game_weather
from src.config import _load_settings_yaml
from src.db import DEFAULT_DB_PATH, init_db
from src.engine.bankroll import get_bankroll_summary, update_bankroll
from src.engine.edge_calculator import calculate_edge
from src.engine.settlement import settle_game_bets
from src.features.adjustments.abs_adjustment import is_abs_active
from src.features.adjustments.park_factors import get_park_factors
from src.model.calibration import CalibratedStackingModel
from src.model.artifact_runtime import validate_runtime_versions
from src.model.data_builder import (
    _default_feature_fill_value,
    _fetch_regular_season_schedule,
    _normalize_game_status,
    _normalize_team_code,
    _prepare_schedule_frame,
    build_live_feature_frame,
)
from src.model.market_recalibration import shrink_probability_toward_market
from src.model.margin_pricing import margin_to_cover_probability
from src.model.run_count_trainer import BlendedRunCountRegressor
from src.model.score_pricing import (
    moneyline_probabilities,
    spread_outcome_probabilities,
    spread_cover_probabilities,
    totals_outcome_probabilities,
    totals_probabilities,
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
from src.pipeline.narrative import generate_game_narrative


logger = logging.getLogger(__name__)

DEFAULT_PROJECTED_F5_TOTAL_RUNS = 4.6
DEFAULT_OFFICIAL_MIN_BET_ODDS = -250
DEFAULT_OFFICIAL_MAX_BET_ODDS = 180
DEFAULT_OFFICIAL_MAX_TRUSTED_EDGE = 0.225
DEFAULT_OFFICIAL_VALUE_PLAY_MIN_EDGE = 0.15
DEFAULT_OFFICIAL_RL_SELECTION_PENALTY = 0.02
DEFAULT_OFFICIAL_EDGE_SCALE_CAP = 0.075
DEFAULT_OFFICIAL_MIN_UNITS = 0.5
DEFAULT_OFFICIAL_MAX_UNITS = 5.0
DEFAULT_OFFICIAL_RL_MIN_BET_ODDS = -140
DEFAULT_OFFICIAL_RL_MAX_BET_ODDS = 105
DEFAULT_OFFICIAL_ML_MARKET_BASE_MULTIPLIER = 0.70
DEFAULT_OFFICIAL_ML_MARKET_PLUS_MONEY_MULTIPLIER = 1.0
DEFAULT_OFFICIAL_ML_MARKET_HIGH_EDGE_THRESHOLD = 0.10
DEFAULT_OFFICIAL_ML_MARKET_HIGH_EDGE_MULTIPLIER = 0.60
DEFAULT_RLV2_DIRECT_EXPERIMENTS: tuple[str, ...] = ("rlv2-direct-longhorizon", "rlv2-direct-tuned")
DEFAULT_RLV2_MARGIN_EXPERIMENTS: tuple[str, ...] = ("rlv2-margin-longhorizon", "rlv2-margin-tuned")
DEFAULT_RLV2_BLEND_EVAL_PATHS: tuple[tuple[str, str], ...] = (
    ("rlv2-blend-longhorizon", "rl_v2_blend_eval_2021_2025.json"),
    ("rlv2-blend-tuned", "rl_v2_blend_eval_2025.json"),
)
DEFAULT_RUN_COUNT_EXPERIMENT = os.getenv(
    "MLB_RUN_COUNT_EXPERIMENT",
    "2026-run-count-newfeatures-150x5",
)
LIVE_ODDS_CACHE_TTL_MINUTES = 15
FULL_GAME_ODDS_CACHE_TTL_MINUTES = 30
LIVE_ODDS_CACHE_RETENTION_DAYS = 7
LINEUP_STALE_TTL_MINUTES = 90
SCRAPER_ODDS_DB_PATH = Path("OddsScraper") / "data" / "mlb_odds.db"
SCRAPER_MARKET_STATE_PATH = Path("OddsScraper") / "data" / "live_market_state.json"
WINDOWS_MARKET_SYNC_STATE_PATH = Path("OddsScraper") / "data" / "windows_market_sync_state.json"

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
FullGameOddsContextFetcher = Callable[[date, Mode, str | Path], dict[int, dict[str, Any]]]
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
    full_game_odds_fetcher: FullGameOddsContextFetcher | None = None
    feature_frame_builder: FeatureFrameBuilder | None = None
    prediction_engine: PredictionEngine | None = None
    notifier: PipelineNotifier | None = None
    weather_fetcher: Callable[..., Any] | None = fetch_game_weather


@dataclass(frozen=True, slots=True)
class StructuredModelBundle:
    model: Any
    feature_columns: list[str]
    model_version: str
    metadata_path: Path
    extra_metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class LegacyModelBundle:
    model: Any
    model_path: Path
    metadata_path: Path
    model_version: str
    target_key: Literal["ml", "rl"]
    variant: Literal["base", "stacking", "calibrated"]
    feature_columns: list[str]
    raw_meta_feature_columns: list[str]
    holdout_season: int
    holdout_log_loss: float
    holdout_roc_auc: float | None
    holdout_accuracy: float | None
    holdout_brier: float | None


@dataclass(slots=True)
class GameProcessingResult:
    game_pk: int
    matchup: str
    status: Literal["pick", "no_pick", "error"]
    game_status: str | None = None
    is_completed: bool = False
    prediction: Prediction | None = None
    selected_decision: BetDecision | None = None
    forced_decision: BetDecision | None = None
    no_pick_reason: str | None = None
    error_message: str | None = None
    notified: bool = False
    paper_fallback: bool = False
    input_status: dict[str, Any] | None = None
    narrative: str | None = None
    candidate_decisions: list[BetDecision] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_pk": self.game_pk,
            "matchup": self.matchup,
            "status": self.status,
            "game_status": self.game_status,
            "is_completed": self.is_completed,
            "prediction": self.prediction.model_dump(mode="json") if self.prediction else None,
            "selected_decision": (
                self.selected_decision.model_dump(mode="json") if self.selected_decision else None
            ),
            "forced_decision": (
                self.forced_decision.model_dump(mode="json") if self.forced_decision else None
            ),
            "no_pick_reason": self.no_pick_reason,
            "error_message": self.error_message,
            "notified": self.notified,
            "paper_fallback": self.paper_fallback,
            "input_status": self.input_status,
            "narrative": self.narrative,
            "candidate_decisions": (
                [decision.model_dump(mode="json") for decision in self.candidate_decisions]
                if self.candidate_decisions
                else []
            ),
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


@dataclass(frozen=True, slots=True)
class SlateRuntimeInputs:
    schedule: pd.DataFrame
    lineups: list[Lineup]
    odds: list[OddsSnapshot]
    full_game_odds_context: dict[int, dict[str, Any]]
    inference_frame: pd.DataFrame
    refresh: bool = False


def _slate_response_payload(result: DailyPipelineResult) -> dict[str, Any]:
    payload = result.to_dict()
    payload.pop("notification_payload", None)
    return payload


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
        self.ml_bundle, self.rl_bundle, self.model_version = self._resolve_model_bundle()
        self.ml_model_path = self.ml_bundle.model_path if self.ml_bundle is not None else None
        self.rl_model_path = self.rl_bundle.model_path if self.rl_bundle is not None else None
        self.ml_model = self.ml_bundle.model if self.ml_bundle is not None else None
        self.rl_model = self.rl_bundle.model if self.rl_bundle is not None else None
        self.rlv2_direct = self._load_structured_model_bundle(
            experiment_names=DEFAULT_RLV2_DIRECT_EXPERIMENTS,
            model_prefix="f5_rl_direct_model",
        )
        self.rlv2_margin = self._load_structured_model_bundle(
            experiment_names=DEFAULT_RLV2_MARGIN_EXPERIMENTS,
            model_prefix="f5_margin_v2_model",
        )
        self.rlv2_blend_weight = self._resolve_rlv2_blend_weight()
        self.run_count_f5_home = self._load_structured_model_bundle(
            experiment_names=(DEFAULT_RUN_COUNT_EXPERIMENT,),
            model_prefix="f5_home_runs_model",
        )
        self.run_count_f5_away = self._load_structured_model_bundle(
            experiment_names=(DEFAULT_RUN_COUNT_EXPERIMENT,),
            model_prefix="f5_away_runs_model",
        )
        self.run_count_full_home = self._load_structured_model_bundle(
            experiment_names=(DEFAULT_RUN_COUNT_EXPERIMENT,),
            model_prefix="full_game_home_runs_model",
        )
        self.run_count_full_away = self._load_structured_model_bundle(
            experiment_names=(DEFAULT_RUN_COUNT_EXPERIMENT,),
            model_prefix="full_game_away_runs_model",
        )

    def predict(self, inference_frame: pd.DataFrame) -> Prediction:
        resolved_frame = inference_frame.copy().reset_index(drop=True)
        if resolved_frame.empty:
            raise ValueError("Inference frame must contain exactly one game row")

        game_pk = int(resolved_frame.iloc[0]["game_pk"])
        prediction_time = datetime.now(UTC)
        resolved_frame = self._ensure_required_columns(resolved_frame)

        if self.ml_bundle is not None and self.rl_bundle is not None:
            ml_home_probability = float(
                self._predict_legacy_home_probability(self.ml_bundle, resolved_frame)
            )
            rl_home_probability = float(
                self._predict_legacy_home_probability(self.rl_bundle, resolved_frame)
            )
        else:
            ml_home_probability = _fallback_ml_home_probability(resolved_frame.iloc[0])
            rl_home_probability = _fallback_rl_home_probability(
                resolved_frame.iloc[0],
                ml_home_probability=ml_home_probability,
            )

        projected_home_margin = self._predict_rlv2_margin_value(resolved_frame)
        if projected_home_margin is None:
            projected_home_margin = float((ml_home_probability - 0.5) * 2.0)
        run_count_result = self._predict_f5_run_counts(resolved_frame)
        if run_count_result is not None:
            projected_home_runs, projected_away_runs = run_count_result
            projected_home_margin = projected_home_runs - projected_away_runs
            projected_total_runs = projected_home_runs + projected_away_runs
        else:
            projected_total_runs = self._estimate_projected_f5_total_runs(resolved_frame.iloc[0])
            projected_home_runs, projected_away_runs = _margin_total_to_team_runs(
                home_margin=projected_home_margin,
                total_runs=projected_total_runs,
            )

        if run_count_result is not None:
            f5_home_runs_std = self._resolve_run_count_std(self.run_count_f5_home)
            f5_away_runs_std = self._resolve_run_count_std(self.run_count_f5_away)
            if f5_home_runs_std is not None and f5_away_runs_std is not None:
                run_count_ml_home, run_count_ml_away = moneyline_probabilities(
                    home_runs_mean=projected_home_runs,
                    away_runs_mean=projected_away_runs,
                    home_runs_std=f5_home_runs_std,
                    away_runs_std=f5_away_runs_std,
                )
                if run_count_ml_home is not None and run_count_ml_away is not None:
                    ml_home_probability = run_count_ml_home

        full_game_ml_home_probability: float | None = None
        full_game_ml_away_probability: float | None = None
        projected_full_game_home_runs: float | None = None
        projected_full_game_away_runs: float | None = None
        projected_full_game_total_runs: float | None = None
        projected_full_game_home_margin: float | None = None
        full_game_run_count_result = self._predict_full_game_run_counts(resolved_frame)
        if full_game_run_count_result is not None:
            projected_full_game_home_runs, projected_full_game_away_runs = full_game_run_count_result
            projected_full_game_home_margin = (
                projected_full_game_home_runs - projected_full_game_away_runs
            )
            projected_full_game_total_runs = (
                projected_full_game_home_runs + projected_full_game_away_runs
            )
            home_runs_std = self._resolve_run_count_std(self.run_count_full_home)
            away_runs_std = self._resolve_run_count_std(self.run_count_full_away)
            if home_runs_std is not None and away_runs_std is not None:
                (
                    full_game_ml_home_probability,
                    full_game_ml_away_probability,
                ) = moneyline_probabilities(
                    home_runs_mean=projected_full_game_home_runs,
                    away_runs_mean=projected_full_game_away_runs,
                    home_runs_std=home_runs_std,
                    away_runs_std=away_runs_std,
                )

        return Prediction(
            game_pk=game_pk,
            model_version=self.model_version,
            full_game_ml_home_prob=full_game_ml_home_probability,
            full_game_ml_away_prob=full_game_ml_away_probability,
            f5_ml_home_prob=ml_home_probability,
            f5_ml_away_prob=1.0 - ml_home_probability,
            f5_rl_home_prob=rl_home_probability,
            f5_rl_away_prob=1.0 - rl_home_probability,
            projected_full_game_home_runs=projected_full_game_home_runs,
            projected_full_game_away_runs=projected_full_game_away_runs,
            projected_full_game_total_runs=projected_full_game_total_runs,
            projected_full_game_home_margin=projected_full_game_home_margin,
            projected_f5_home_runs=projected_home_runs,
            projected_f5_away_runs=projected_away_runs,
            projected_f5_total_runs=projected_total_runs,
            projected_f5_home_margin=projected_home_margin,
            predicted_at=prediction_time,
        )

    def build_candidate_decisions(
        self,
        *,
        inference_frame: pd.DataFrame,
        prediction: Prediction,
        snapshots: Sequence[OddsSnapshot],
        db_path: str | Path,
        full_game_context: Mapping[str, Any] | None = None,
    ) -> list[BetDecision]:
        candidates: list[BetDecision] = []
        resolved_frame = inference_frame.copy().reset_index(drop=True)
        legacy_frame = self._ensure_required_columns(resolved_frame)

        for snapshot in snapshots:
            if snapshot.market_type == "f5_ml":
                ml_source_model = "legacy_f5_ml"
                if (
                    self.run_count_f5_home is not None
                    and self.run_count_f5_away is not None
                    and prediction.projected_f5_home_runs is not None
                    and prediction.projected_f5_away_runs is not None
                    and self._resolve_run_count_std(self.run_count_f5_home) is not None
                    and self._resolve_run_count_std(self.run_count_f5_away) is not None
                ):
                    ml_source_model = "run_count_f5_ml"
                adjusted_home_probability, adjusted_away_probability = _recalibrate_ml_market_pair(
                    home_probability=float(prediction.f5_ml_home_prob),
                    away_probability=float(prediction.f5_ml_away_prob),
                    home_odds=snapshot.home_odds,
                    away_odds=snapshot.away_odds,
                )
                candidates.extend(
                    self._build_two_way_candidates(
                        game_pk=prediction.game_pk,
                        market_type=snapshot.market_type,
                        first_side="home",
                        first_probability=adjusted_home_probability,
                        first_odds=snapshot.home_odds,
                        first_point=snapshot.home_point,
                        second_side="away",
                        second_probability=adjusted_away_probability,
                        second_odds=snapshot.away_odds,
                        second_point=snapshot.away_point,
                        book_name=snapshot.book_name,
                        db_path=db_path,
                        source_model=ml_source_model,
                        source_model_version=self.model_version,
                    )
                )
                continue

            if snapshot.market_type == "f5_total":
                if (
                    prediction.projected_f5_home_runs is None
                    or prediction.projected_f5_away_runs is None
                    or snapshot.home_point is None
                ):
                    continue
                total_over_probability, total_under_probability = totals_probabilities(
                    home_runs_mean=prediction.projected_f5_home_runs,
                    away_runs_mean=prediction.projected_f5_away_runs,
                    home_runs_std=self._resolve_run_count_std(self.run_count_f5_home) or 2.35,
                    away_runs_std=self._resolve_run_count_std(self.run_count_f5_away) or 2.29,
                    total_point=float(snapshot.home_point),
                )
                if total_over_probability is None or total_under_probability is None:
                    continue
                candidates.extend(
                    self._build_two_way_candidates(
                        game_pk=prediction.game_pk,
                        market_type=snapshot.market_type,
                        first_side="over",
                        first_probability=total_over_probability,
                        first_odds=snapshot.home_odds,
                        first_point=snapshot.home_point,
                        second_side="under",
                        second_probability=total_under_probability,
                        second_odds=snapshot.away_odds,
                        second_point=snapshot.away_point,
                        book_name=snapshot.book_name,
                        db_path=db_path,
                        source_model="run_count_f5_total",
                        source_model_version=self.model_version,
                    )
                )
                continue

            if (
                not _is_ml_equivalent_f5_runline(snapshot)
                and prediction.projected_f5_home_runs is not None
                and prediction.projected_f5_away_runs is not None
                and snapshot.home_point is not None
            ):
                rc_rl_home, rc_rl_away = spread_cover_probabilities(
                    home_runs_mean=prediction.projected_f5_home_runs,
                    away_runs_mean=prediction.projected_f5_away_runs,
                    home_runs_std=self._resolve_run_count_std(self.run_count_f5_home) or 2.35,
                    away_runs_std=self._resolve_run_count_std(self.run_count_f5_away) or 2.29,
                    home_point=float(snapshot.home_point),
                )
                legacy_home_probability = (
                    float(rc_rl_home)
                    if rc_rl_home is not None
                    else float(prediction.f5_rl_home_prob)
                )
                legacy_source_model = "run_count_f5_rl"
            else:
                legacy_home_probability = float(prediction.f5_rl_home_prob)
                legacy_source_model = "legacy_f5_rl"
                if _is_ml_equivalent_f5_runline(snapshot):
                    legacy_home_probability = float(prediction.f5_ml_home_prob)
                    legacy_source_model = "legacy_f5_ml_equiv"

            candidates.extend(
                self._build_two_way_candidates(
                    game_pk=prediction.game_pk,
                    market_type=snapshot.market_type,
                    first_side="home",
                    first_probability=legacy_home_probability,
                    first_odds=snapshot.home_odds,
                    first_point=snapshot.home_point,
                    second_side="away",
                    second_probability=1.0 - legacy_home_probability,
                    second_odds=snapshot.away_odds,
                    second_point=snapshot.away_point,
                    book_name=snapshot.book_name,
                    db_path=db_path,
                    source_model=legacy_source_model,
                    source_model_version=self.model_version,
                )
            )

            direct_probability = self._predict_rlv2_direct_home_probability(
                inference_frame=legacy_frame,
                snapshot=snapshot,
            )
            margin_probability = self._predict_rlv2_margin_home_probability(
                inference_frame=legacy_frame,
                snapshot=snapshot,
            )
            if direct_probability is not None:
                candidates.extend(
                    self._build_two_way_candidates(
                        game_pk=prediction.game_pk,
                        market_type=snapshot.market_type,
                        first_side="home",
                        first_probability=direct_probability,
                        first_odds=snapshot.home_odds,
                        first_point=snapshot.home_point,
                        second_side="away",
                        second_probability=1.0 - direct_probability,
                        second_odds=snapshot.away_odds,
                        second_point=snapshot.away_point,
                        book_name=snapshot.book_name,
                        db_path=db_path,
                        source_model="rlv2_direct",
                        source_model_version=self.rlv2_direct.model_version
                        if self.rlv2_direct
                        else None,
                    )
                )
            if margin_probability is not None:
                candidates.extend(
                    self._build_two_way_candidates(
                        game_pk=prediction.game_pk,
                        market_type=snapshot.market_type,
                        first_side="home",
                        first_probability=margin_probability,
                        first_odds=snapshot.home_odds,
                        first_point=snapshot.home_point,
                        second_side="away",
                        second_probability=1.0 - margin_probability,
                        second_odds=snapshot.away_odds,
                        second_point=snapshot.away_point,
                        book_name=snapshot.book_name,
                        db_path=db_path,
                        source_model="rlv2_margin",
                        source_model_version=self.rlv2_margin.model_version
                        if self.rlv2_margin
                        else None,
                    )
                )
            if direct_probability is not None and margin_probability is not None:
                blend_probability = (
                    self.rlv2_blend_weight * direct_probability
                    + (1.0 - self.rlv2_blend_weight) * margin_probability
                )
                candidates.extend(
                    self._build_two_way_candidates(
                        game_pk=prediction.game_pk,
                        market_type=snapshot.market_type,
                        first_side="home",
                        first_probability=blend_probability,
                        first_odds=snapshot.home_odds,
                        first_point=snapshot.home_point,
                        second_side="away",
                        second_probability=1.0 - blend_probability,
                        second_odds=snapshot.away_odds,
                        second_point=snapshot.away_point,
                        book_name=snapshot.book_name,
                        db_path=db_path,
                        source_model="rlv2_blend",
                        source_model_version=(
                            f"direct={self.rlv2_direct.model_version if self.rlv2_direct else 'na'};"
                            f"margin={self.rlv2_margin.model_version if self.rlv2_margin else 'na'};"
                            f"weight={self.rlv2_blend_weight:.2f}"
                        ),
                    )
                )

        resolved_full_game_context = dict(full_game_context or {})
        full_game_ml_home_probability = prediction.full_game_ml_home_prob
        if full_game_ml_home_probability is not None:
            for pair in resolved_full_game_context.get("full_game_ml_pairs") or []:
                home_odds = pair.get("home_odds")
                away_odds = pair.get("away_odds")
                book_name = pair.get("book_name")
                if home_odds is None or away_odds is None or not isinstance(book_name, str):
                    continue
                adjusted_home_probability, adjusted_away_probability = _recalibrate_ml_market_pair(
                    home_probability=float(full_game_ml_home_probability),
                    away_probability=float(1.0 - full_game_ml_home_probability),
                    home_odds=int(home_odds),
                    away_odds=int(away_odds),
                )
                candidates.extend(
                    self._build_two_way_candidates(
                        game_pk=prediction.game_pk,
                        market_type="full_game_ml",
                        first_side="home",
                        first_probability=adjusted_home_probability,
                        first_odds=int(home_odds),
                        first_point=None,
                        second_side="away",
                        second_probability=adjusted_away_probability,
                        second_odds=int(away_odds),
                        second_point=None,
                        book_name=book_name,
                        db_path=db_path,
                        source_model="run_count_full_game_ml",
                        source_model_version=self.model_version,
                    )
                )

        if (
            prediction.projected_full_game_home_runs is not None
            and prediction.projected_full_game_away_runs is not None
        ):
            for pair in resolved_full_game_context.get("full_game_rl_pairs") or []:
                home_odds = pair.get("home_odds")
                away_odds = pair.get("away_odds")
                home_point = pair.get("home_point")
                away_point = pair.get("away_point")
                book_name = pair.get("book_name")
                if (
                    home_odds is None
                    or away_odds is None
                    or home_point is None
                    or away_point is None
                    or not isinstance(book_name, str)
                ):
                    continue
                (
                    full_game_rl_home_probability,
                    full_game_rl_away_probability,
                    full_game_rl_push_probability,
                ) = spread_outcome_probabilities(
                    home_runs_mean=prediction.projected_full_game_home_runs,
                    away_runs_mean=prediction.projected_full_game_away_runs,
                    home_runs_std=self._resolve_run_count_std(self.run_count_full_home) or 3.13,
                    away_runs_std=self._resolve_run_count_std(self.run_count_full_away) or 3.36,
                    home_point=float(home_point),
                )
                if full_game_rl_home_probability is None or full_game_rl_away_probability is None:
                    continue
                candidates.extend(
                    self._build_two_way_candidates(
                        game_pk=prediction.game_pk,
                        market_type="full_game_rl",
                        first_side="home",
                        first_probability=full_game_rl_home_probability,
                        first_odds=int(home_odds),
                        first_point=float(home_point),
                        second_side="away",
                        second_probability=full_game_rl_away_probability,
                        second_odds=int(away_odds),
                        second_point=float(away_point),
                        book_name=book_name,
                        db_path=db_path,
                        source_model="run_count_full_game_rl",
                        source_model_version=self.model_version,
                        push_probability=full_game_rl_push_probability or 0.0,
                    )
                )

            for pair in resolved_full_game_context.get("full_game_total_pairs") or []:
                over_odds = pair.get("over_odds")
                under_odds = pair.get("under_odds")
                total_point = pair.get("total_point")
                book_name = pair.get("book_name")
                if (
                    over_odds is None
                    or under_odds is None
                    or total_point is None
                    or not isinstance(book_name, str)
                ):
                    continue
                (
                    total_over_probability,
                    total_under_probability,
                    total_push_probability,
                ) = totals_outcome_probabilities(
                    home_runs_mean=prediction.projected_full_game_home_runs,
                    away_runs_mean=prediction.projected_full_game_away_runs,
                    home_runs_std=self._resolve_run_count_std(self.run_count_full_home) or 3.13,
                    away_runs_std=self._resolve_run_count_std(self.run_count_full_away) or 3.36,
                    total_point=float(total_point),
                )
                if total_over_probability is None or total_under_probability is None:
                    continue
                candidates.extend(
                    self._build_two_way_candidates(
                        game_pk=prediction.game_pk,
                        market_type="full_game_total",
                        first_side="over",
                        first_probability=total_over_probability,
                        first_odds=int(over_odds),
                        first_point=float(total_point),
                        second_side="under",
                        second_probability=total_under_probability,
                        second_odds=int(under_odds),
                        second_point=float(total_point),
                        book_name=book_name,
                        db_path=db_path,
                        source_model="run_count_full_game_total",
                        source_model_version=self.model_version,
                        push_probability=total_push_probability or 0.0,
                    )
                )

        return candidates

    def _ensure_required_columns(self, inference_frame: pd.DataFrame) -> pd.DataFrame:
        resolved = inference_frame.copy()
        required_columns: set[str] = set()
        for bundle in (self.ml_bundle, self.rl_bundle):
            if bundle is None:
                continue
            required_columns.update(bundle.feature_columns)
            required_columns.update(bundle.raw_meta_feature_columns)

        missing_columns = {
            column: _default_feature_fill_value(column)
            for column in required_columns
            if column not in resolved.columns
        }
        if missing_columns:
            resolved = pd.concat(
                [resolved, pd.DataFrame([missing_columns], index=resolved.index)],
                axis=1,
            )

        return resolved

    def _resolve_model_bundle(
        self,
    ) -> tuple[LegacyModelBundle | None, LegacyModelBundle | None, str]:
        ml_bundle = self._resolve_best_legacy_bundle("ml")
        rl_bundle = self._resolve_best_legacy_bundle("rl")

        if ml_bundle is None or rl_bundle is None:
            logger.warning(
                "No complete legacy model bundle found under %s; daily pipeline will use baseline fallback",
                self.model_dir,
            )
            return None, None, "baseline-fallback"

        resolved_version = (
            f"ml={ml_bundle.model_version}:{ml_bundle.variant};"
            f"rl={rl_bundle.model_version}:{rl_bundle.variant}"
        )
        logger.info(
            "Loaded legacy model bundles version=%s ml=%s (%s) rl=%s (%s)",
            resolved_version,
            ml_bundle.model_path,
            ml_bundle.variant,
            rl_bundle.model_path,
            rl_bundle.variant,
        )
        return ml_bundle, rl_bundle, resolved_version

    def _resolve_best_legacy_bundle(
        self, target_key: Literal["ml", "rl"]
    ) -> LegacyModelBundle | None:
        resolved_candidates: list[LegacyModelBundle] = []
        target_prefix = f"f5_{target_key}"
        variant_patterns: tuple[tuple[str, Literal["base", "stacking", "calibrated"]], ...] = (
            (f"{target_prefix}_model_*.joblib", "base"),
            (f"{target_prefix}_stacking_model_*.joblib", "stacking"),
            (f"{target_prefix}_calibrated_model_*.joblib", "calibrated"),
        )

        for pattern, variant in variant_patterns:
            for model_path in self.model_dir.rglob(pattern):
                candidate = self._load_legacy_model_bundle(
                    model_path=model_path,
                    target_key=target_key,
                    variant=variant,
                )
                if candidate is not None:
                    resolved_candidates.append(candidate)

        if not resolved_candidates:
            return None

        latest_holdout_season = max(candidate.holdout_season for candidate in resolved_candidates)
        season_candidates = [
            candidate
            for candidate in resolved_candidates
            if candidate.holdout_season == latest_holdout_season
        ]
        return min(
            season_candidates,
            key=lambda candidate: (
                candidate.holdout_log_loss,
                float("inf") if candidate.holdout_brier is None else candidate.holdout_brier,
                -(
                    candidate.holdout_roc_auc
                    if candidate.holdout_roc_auc is not None
                    else float("-inf")
                ),
                -(
                    candidate.holdout_accuracy
                    if candidate.holdout_accuracy is not None
                    else float("-inf")
                ),
                _legacy_variant_priority(candidate.variant),
                -candidate.model_path.stat().st_mtime,
            ),
        )

    def _load_legacy_model_bundle(
        self,
        *,
        model_path: Path,
        target_key: Literal["ml", "rl"],
        variant: Literal["base", "stacking", "calibrated"],
    ) -> LegacyModelBundle | None:
        metadata_path = model_path.with_suffix(".metadata.json")
        if not metadata_path.exists():
            return None

        try:
            metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        try:
            validate_runtime_versions(metadata_payload, artifact_path=model_path)
        except RuntimeError:
            logger.warning("Skipping incompatible artifact %s", model_path, exc_info=True)
            return None

        holdout_metrics = metadata_payload.get("holdout_metrics", {})
        resolved_holdout_season = self._resolve_legacy_holdout_season(
            metadata_payload=metadata_payload,
        )
        if variant == "base":
            holdout_log_loss = holdout_metrics.get("log_loss")
            holdout_roc_auc = holdout_metrics.get("roc_auc")
            holdout_accuracy = holdout_metrics.get("accuracy")
            holdout_brier = holdout_metrics.get("brier")
            feature_columns = list(metadata_payload.get("feature_columns", []))
            raw_meta_feature_columns: list[str] = []
        elif variant == "stacking":
            holdout_log_loss = holdout_metrics.get("stacked_log_loss")
            holdout_roc_auc = holdout_metrics.get("stacked_roc_auc")
            holdout_accuracy = holdout_metrics.get("stacked_accuracy")
            holdout_brier = holdout_metrics.get("stacked_brier")
            feature_columns = list(metadata_payload.get("feature_columns", []))
            raw_meta_feature_columns = list(metadata_payload.get("raw_meta_feature_columns", []))
        else:
            holdout_log_loss = holdout_metrics.get("calibrated_log_loss")
            holdout_roc_auc = holdout_metrics.get("calibrated_roc_auc")
            holdout_accuracy = holdout_metrics.get("calibrated_accuracy")
            holdout_brier = holdout_metrics.get("calibrated_brier")
            loaded_model = joblib.load(model_path)
            if not isinstance(loaded_model, CalibratedStackingModel):
                return None
            return LegacyModelBundle(
                model=loaded_model,
                model_path=model_path,
                metadata_path=metadata_path,
                model_version=str(metadata_payload.get("model_version", model_path.stem)),
                target_key=target_key,
                variant=variant,
                feature_columns=list(loaded_model.stacking_model.base_feature_columns),
                raw_meta_feature_columns=list(loaded_model.stacking_model.raw_meta_feature_columns),
                holdout_season=resolved_holdout_season,
                holdout_log_loss=float(holdout_log_loss),
                holdout_roc_auc=(None if holdout_roc_auc is None else float(holdout_roc_auc)),
                holdout_accuracy=(None if holdout_accuracy is None else float(holdout_accuracy)),
                holdout_brier=None if holdout_brier is None else float(holdout_brier),
            )

        if holdout_log_loss is None:
            return None

        loaded_model = joblib.load(model_path)
        return LegacyModelBundle(
            model=loaded_model,
            model_path=model_path,
            metadata_path=metadata_path,
            model_version=str(metadata_payload.get("model_version", model_path.stem)),
            target_key=target_key,
            variant=variant,
            feature_columns=feature_columns,
            raw_meta_feature_columns=raw_meta_feature_columns,
            holdout_season=resolved_holdout_season,
            holdout_log_loss=float(holdout_log_loss),
            holdout_roc_auc=None if holdout_roc_auc is None else float(holdout_roc_auc),
            holdout_accuracy=None if holdout_accuracy is None else float(holdout_accuracy),
            holdout_brier=None if holdout_brier is None else float(holdout_brier),
        )

    def _resolve_legacy_holdout_season(self, *, metadata_payload: Mapping[str, Any]) -> int:
        direct_holdout = metadata_payload.get("holdout_season")
        if direct_holdout is not None:
            try:
                return int(direct_holdout)
            except (TypeError, ValueError):
                pass

        for related_key in ("base_model_path", "stacking_model_path"):
            related_value = metadata_payload.get(related_key)
            if not isinstance(related_value, str) or not related_value.strip():
                continue
            related_model_path = Path(related_value)
            if not related_model_path.is_absolute():
                related_model_path = Path.cwd() / related_model_path
            related_metadata_path = related_model_path.with_suffix(".metadata.json")
            if not related_metadata_path.exists():
                continue
            try:
                related_payload = json.loads(related_metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            related_holdout = related_payload.get("holdout_season")
            if related_holdout is None:
                continue
            try:
                return int(related_holdout)
            except (TypeError, ValueError):
                continue

        return 0

    def _predict_legacy_home_probability(
        self,
        bundle: LegacyModelBundle,
        inference_frame: pd.DataFrame,
    ) -> float:
        if bundle.variant == "calibrated":
            return float(bundle.model.predict_calibrated(inference_frame)[0])
        probability = bundle.model.predict_proba(inference_frame[bundle.feature_columns])[:, 1][0]
        return float(probability)

    def _load_structured_model_bundle(
        self,
        *,
        experiment_names: Sequence[str],
        model_prefix: str,
    ) -> StructuredModelBundle | None:
        for experiment_name in experiment_names:
            experiment_dir = self.model_dir / experiment_name
            if not experiment_dir.exists():
                continue

            model_candidates = sorted(experiment_dir.glob(f"{model_prefix}_*.joblib"))
            if not model_candidates:
                continue

            model_path = max(model_candidates, key=lambda path: path.stat().st_mtime)
            metadata_path = model_path.with_suffix(".metadata.json")
            if not metadata_path.exists():
                continue

            metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            try:
                validate_runtime_versions(metadata_payload, artifact_path=model_path)
            except RuntimeError:
                logger.warning("Skipping incompatible artifact %s", model_path, exc_info=True)
                continue
            model = self._load_structured_joblib_model(model_path)
            return StructuredModelBundle(
                model=model,
                feature_columns=list(metadata_payload.get("feature_columns", [])),
                model_version=str(metadata_payload.get("model_version", model_path.stem)),
                metadata_path=metadata_path,
                extra_metadata=metadata_payload,
            )
        return None

    def _load_structured_joblib_model(self, model_path: Path) -> Any:
        try:
            return joblib.load(model_path)
        except AttributeError as exc:
            if "BlendedRunCountRegressor" not in str(exc):
                raise
            main_module = sys.modules.get("__main__")
            if main_module is not None and not hasattr(main_module, "BlendedRunCountRegressor"):
                setattr(main_module, "BlendedRunCountRegressor", BlendedRunCountRegressor)
            return joblib.load(model_path)

    def _resolve_rlv2_blend_weight(self) -> float:
        for experiment_name, filename in DEFAULT_RLV2_BLEND_EVAL_PATHS:
            payload_path = self.model_dir / experiment_name / filename
            if not payload_path.exists():
                continue
            try:
                payload = json.loads(payload_path.read_text(encoding="utf-8"))
                best_blend = payload.get("best_blend") or {}
                weight = float(best_blend.get("direct_weight", 0.8))
                return max(0.0, min(1.0, weight))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
        return 0.8

    def _predict_rlv2_direct_home_probability(
        self,
        *,
        inference_frame: pd.DataFrame,
        snapshot: OddsSnapshot,
    ) -> float | None:
        if self.rlv2_direct is None or snapshot.market_type != "f5_rl":
            return None
        frame = self._prepare_rlv2_frame(
            inference_frame=inference_frame,
            snapshot=snapshot,
            feature_columns=self.rlv2_direct.feature_columns,
        )
        probability = self.rlv2_direct.model.predict_proba(frame[self.rlv2_direct.feature_columns])[
            :, 1
        ][0]
        return max(0.0, min(1.0, float(probability)))

    def _predict_rlv2_margin_home_probability(
        self,
        *,
        inference_frame: pd.DataFrame,
        snapshot: OddsSnapshot,
    ) -> float | None:
        if self.rlv2_margin is None or snapshot.market_type != "f5_rl":
            return None
        residual_std = float(self.rlv2_margin.extra_metadata.get("residual_std", 0.0) or 0.0)
        if residual_std <= 0:
            return None
        frame = self._prepare_rlv2_frame(
            inference_frame=inference_frame,
            snapshot=snapshot,
            feature_columns=self.rlv2_margin.feature_columns,
        )
        predicted_margin = float(
            self.rlv2_margin.model.predict(frame[self.rlv2_margin.feature_columns])[0]
        )
        return margin_to_cover_probability(
            predicted_margin=predicted_margin,
            home_point=snapshot.home_point,
            residual_std=residual_std,
        )

    def _predict_rlv2_margin_value(self, inference_frame: pd.DataFrame) -> float | None:
        if self.rlv2_margin is None:
            return None
        frame = inference_frame.copy().reset_index(drop=True)
        missing_columns = {
            column: _default_feature_fill_value(column)
            for column in self.rlv2_margin.feature_columns
            if column not in frame.columns
        }
        if missing_columns:
            frame = pd.concat(
                [frame, pd.DataFrame([missing_columns], index=frame.index)],
                axis=1,
            )
        return float(self.rlv2_margin.model.predict(frame[self.rlv2_margin.feature_columns])[0])

    def _estimate_projected_f5_total_runs(self, inference_row: pd.Series) -> float:
        park_runs_factor = float(inference_row.get("park_runs_factor", 1.0) or 1.0)
        weather_composite = float(inference_row.get("weather_composite", 1.0) or 1.0)
        adjusted_total = DEFAULT_PROJECTED_F5_TOTAL_RUNS * park_runs_factor * weather_composite
        return float(min(max(adjusted_total, 2.5), 7.5))

    def _predict_f5_run_counts(
        self, inference_frame: pd.DataFrame
    ) -> tuple[float, float] | None:
        return self._predict_run_counts(
            inference_frame=inference_frame,
            home_bundle=self.run_count_f5_home,
            away_bundle=self.run_count_f5_away,
            failure_message="F5 run count prediction failed, falling back to formula",
        )

    def _predict_full_game_run_counts(
        self, inference_frame: pd.DataFrame
    ) -> tuple[float, float] | None:
        return self._predict_run_counts(
            inference_frame=inference_frame,
            home_bundle=self.run_count_full_home,
            away_bundle=self.run_count_full_away,
            failure_message="Full-game run count prediction failed; omitting full-game projections",
        )

    def _predict_run_counts(
        self,
        *,
        inference_frame: pd.DataFrame,
        home_bundle: StructuredModelBundle | None,
        away_bundle: StructuredModelBundle | None,
        failure_message: str,
    ) -> tuple[float, float] | None:
        if home_bundle is None or away_bundle is None:
            return None
        try:
            def _fill_frame(bundle: StructuredModelBundle) -> pd.DataFrame:
                missing = {
                    col: _default_feature_fill_value(col)
                    for col in bundle.feature_columns
                    if col not in inference_frame.columns
                }
                frame = inference_frame.copy()
                if missing:
                    frame = pd.concat(
                        [frame, pd.DataFrame([missing], index=frame.index)], axis=1
                    )
                return frame[bundle.feature_columns]

            home_pred = float(home_bundle.model.predict(_fill_frame(home_bundle))[0])
            away_pred = float(away_bundle.model.predict(_fill_frame(away_bundle))[0])
            return max(0.0, home_pred), max(0.0, away_pred)
        except Exception:
            logger.warning(failure_message, exc_info=True)
            return None

    def _resolve_run_count_std(self, bundle: StructuredModelBundle | None) -> float | None:
        if bundle is None:
            return None
        holdout_metrics = bundle.extra_metadata.get("holdout_metrics", {})
        holdout_rmse = holdout_metrics.get("rmse")
        if holdout_rmse is not None:
            try:
                resolved_rmse = float(holdout_rmse)
                if resolved_rmse > 0:
                    return resolved_rmse
            except (TypeError, ValueError):
                pass
        cv_best_rmse = bundle.extra_metadata.get("cv_best_rmse")
        if cv_best_rmse is not None:
            try:
                resolved_rmse = float(cv_best_rmse)
                if resolved_rmse > 0:
                    return resolved_rmse
            except (TypeError, ValueError):
                pass
        return None

    def _prepare_rlv2_frame(
        self,
        *,
        inference_frame: pd.DataFrame,
        snapshot: OddsSnapshot,
        feature_columns: Sequence[str],
    ) -> pd.DataFrame:
        frame = inference_frame.copy().reset_index(drop=True)
        frame["posted_f5_rl_home_point"] = snapshot.home_point
        frame["posted_f5_rl_away_point"] = snapshot.away_point
        frame["posted_f5_rl_home_odds"] = snapshot.home_odds
        frame["posted_f5_rl_away_odds"] = snapshot.away_odds
        frame["posted_f5_rl_home_implied_prob"] = _american_to_implied_probability(
            snapshot.home_odds
        )
        frame["posted_f5_rl_away_implied_prob"] = _american_to_implied_probability(
            snapshot.away_odds
        )
        frame["posted_f5_rl_point_abs"] = abs(float(snapshot.home_point or 0.0))
        frame["posted_f5_rl_home_is_favorite"] = (
            1.0 if int(snapshot.home_odds) < int(snapshot.away_odds) else 0.0
        )
        missing_columns = {
            column: _default_feature_fill_value(column)
            for column in feature_columns
            if column not in frame.columns
        }
        if missing_columns:
            frame = pd.concat(
                [frame, pd.DataFrame([missing_columns], index=frame.index)],
                axis=1,
            )
        return frame

    def _build_two_way_candidates(
        self,
        *,
        game_pk: int,
        market_type: str,
        first_side: str,
        first_probability: float,
        first_odds: int,
        first_point: float | None,
        second_side: str,
        second_probability: float,
        second_odds: int,
        second_point: float | None,
        book_name: str,
        db_path: str | Path,
        source_model: str,
        source_model_version: str | None,
        push_probability: float = 0.0,
    ) -> list[BetDecision]:
        decisions = [
            calculate_edge(
                game_pk=game_pk,
                market_type=market_type,
                side=first_side,
                model_probability=first_probability,
                home_odds=first_odds,
                away_odds=second_odds,
                home_point=first_point,
                away_point=second_point,
                book_name=book_name,
                db_path=db_path,
                push_probability=push_probability,
            ),
            calculate_edge(
                game_pk=game_pk,
                market_type=market_type,
                side=second_side,
                model_probability=second_probability,
                home_odds=first_odds,
                away_odds=second_odds,
                home_point=first_point,
                away_point=second_point,
                book_name=book_name,
                db_path=db_path,
                push_probability=push_probability,
            ),
        ]
        return [
            decision.model_copy(
                update={
                    "source_model": source_model,
                    "source_model_version": source_model_version,
                }
            )
            for decision in decisions
        ]


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
    full_game_odds_fetcher = (
        resolved_dependencies.full_game_odds_fetcher or _default_full_game_odds_context_fetcher
    )
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
        _persist_cached_slate_response(
            database_path,
            pipeline_date=pipeline_day.isoformat(),
            mode=mode,
            dry_run=dry_run,
            result=result,
        )
        return result

    historical_games = history_fetcher(pipeline_day.year, pipeline_day)
    if not historical_games.empty:
        _upsert_games(database_path, historical_games)
    _upsert_games(database_path, schedule)
    initial_inputs = _build_slate_runtime_inputs(
        pipeline_day=pipeline_day,
        mode=mode,
        dry_run=dry_run,
        database_path=database_path,
        schedule=schedule,
        historical_games=historical_games,
        lineups_fetcher=lineups_fetcher,
        odds_fetcher=odds_fetcher,
        full_game_odds_fetcher=full_game_odds_fetcher,
        feature_frame_builder=feature_frame_builder,
        weather_fetcher=weather_fetcher,
        refresh=False,
    )
    baseline_inputs: SlateRuntimeInputs | None = None
    active_inputs = initial_inputs
    if mode == "prod":
        refreshed_schedule = schedule_fetcher(pipeline_day, mode)
        if not refreshed_schedule.empty:
            _upsert_games(database_path, refreshed_schedule)
            baseline_inputs = initial_inputs
            active_inputs = _build_slate_runtime_inputs(
                pipeline_day=pipeline_day,
                mode=mode,
                dry_run=dry_run,
                database_path=database_path,
                schedule=refreshed_schedule,
                historical_games=historical_games,
                lineups_fetcher=lineups_fetcher,
                odds_fetcher=odds_fetcher,
                full_game_odds_fetcher=full_game_odds_fetcher,
                feature_frame_builder=feature_frame_builder,
                weather_fetcher=weather_fetcher,
                refresh=True,
            )

    schedule = active_inputs.schedule
    lineups = active_inputs.lineups
    odds = active_inputs.odds
    full_game_odds_context = active_inputs.full_game_odds_context
    inference_frame = active_inputs.inference_frame

    run_id = _build_run_id(pipeline_day)
    lineups_by_game_team = {(lineup.game_pk, lineup.team): lineup for lineup in lineups}
    odds_by_game = _group_odds_by_game(odds)
    schedule_lookup = {int(row["game_pk"]): row for row in schedule.to_dict(orient="records")}
    baseline_schedule_lookup = (
        {
            int(row["game_pk"]): row
            for row in baseline_inputs.schedule.to_dict(orient="records")
        }
        if baseline_inputs is not None
        else {}
    )
    baseline_lineups_by_game_team = (
        {(lineup.game_pk, lineup.team): lineup for lineup in baseline_inputs.lineups}
        if baseline_inputs is not None
        else {}
    )
    baseline_odds_by_game = (
        _group_odds_by_game(baseline_inputs.odds) if baseline_inputs is not None else {}
    )
    baseline_inference_lookup = (
        _build_inference_row_lookup(baseline_inputs.inference_frame)
        if baseline_inputs is not None
        else {}
    )

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
        game_status = _normalize_game_status(game.get("status"))
        is_completed = game_status == "final"
        row_frame = inference_frame.loc[inference_frame["game_pk"] == game_pk].reset_index(
            drop=True
        )
        input_status: dict[str, Any] | None = None

        try:
            if row_frame.empty:
                raise ValueError(f"No feature row available for game_pk={game_pk}")
            inference_row = row_frame.iloc[0]
            input_status = _build_input_status(
                game=game,
                inference_row=inference_row,
                lineups_by_game_team=lineups_by_game_team,
                odds_by_game=odds_by_game,
                full_game_odds_context_by_game=full_game_odds_context,
                baseline_schedule_lookup=baseline_schedule_lookup,
                baseline_lineups_by_game_team=baseline_lineups_by_game_team,
                baseline_odds_by_game=baseline_odds_by_game,
                baseline_inference_lookup=baseline_inference_lookup,
                recheck_performed=baseline_inputs is not None,
            )
            row_frame, umpire_neutralized = _neutralize_missing_umpire_features(row_frame)
            if input_status is not None:
                input_status["umpire_neutralized"] = bool(umpire_neutralized)
                input_status["umpire_fallback_mode"] = (
                    "neutralized_defaults" if umpire_neutralized else None
                )

            prediction = prediction_engine.predict(row_frame)
            _upsert_prediction(database_path, prediction)
            candidate_decisions = _build_candidate_decisions(
                inference_frame=row_frame,
                prediction=prediction,
                snapshots=odds_by_game.get(game_pk, []),
                db_path=database_path,
                prediction_engine=prediction_engine,
                full_game_context=full_game_odds_context.get(game_pk, {}),
            )
            forced_decision = _select_forced_game_decision(candidate_decisions)

            validation_reasons = _collect_validation_reasons(
                game=game,
                inference_row=inference_row,
                lineups_by_game_team=lineups_by_game_team,
                odds_by_game=odds_by_game,
                full_game_odds_context_by_game=full_game_odds_context,
                after_refresh=baseline_inputs is not None,
            )
            lineup_only_block = bool(validation_reasons) and {
                _base_validation_reason(reason) for reason in validation_reasons
            } <= {
                "lineup unavailable"
            }
            allow_paper_fallback = (
                dry_run and lineup_only_block and bool(input_status["odds_available"])
            )
            if validation_reasons and not allow_paper_fallback:
                results.append(
                    GameProcessingResult(
                        game_pk=game_pk,
                        matchup=matchup,
                        status="no_pick",
                        game_status=game_status,
                        is_completed=is_completed,
                        prediction=prediction,
                        forced_decision=forced_decision,
                        no_pick_reason="; ".join(validation_reasons),
                        input_status=input_status,
                        candidate_decisions=candidate_decisions,
                    )
                )
                continue

            decision, kill_switch_active = _select_game_decision(
                candidates=candidate_decisions,
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
                        game_status=game_status,
                        is_completed=is_completed,
                        prediction=prediction,
                        selected_decision=decision,
                        forced_decision=forced_decision,
                        no_pick_reason="kill-switch active",
                        input_status=input_status,
                        candidate_decisions=candidate_decisions,
                    )
                )
                continue

            if decision is None:
                results.append(
                    GameProcessingResult(
                        game_pk=game_pk,
                        matchup=matchup,
                        status="no_pick",
                        game_status=game_status,
                        is_completed=is_completed,
                        prediction=prediction,
                        forced_decision=forced_decision,
                        no_pick_reason="edge below threshold",
                        input_status=input_status,
                        candidate_decisions=candidate_decisions,
                    )
                )
                continue

            virtual_bankroll = max(0.0, virtual_bankroll - float(decision.kelly_stake))
            results.append(
                GameProcessingResult(
                    game_pk=game_pk,
                    matchup=matchup,
                    status="pick",
                    game_status=game_status,
                    is_completed=is_completed,
                    prediction=prediction,
                    selected_decision=decision,
                    forced_decision=forced_decision,
                    paper_fallback=allow_paper_fallback,
                    input_status=input_status,
                    candidate_decisions=candidate_decisions,
                )
            )
        except Exception as exc:
            logger.warning("Game %s failed during daily pipeline", game_pk, exc_info=True)
            results.append(
                GameProcessingResult(
                    game_pk=game_pk,
                    matchup=matchup,
                    status="error",
                    game_status=game_status,
                    is_completed=is_completed,
                    error_message=str(exc),
                    input_status=input_status if "input_status" in locals() else None,
                )
            )

    # --- Generate narratives for all processed games ---
    _POTD_MIN_EDGE = 0.05
    potd_game_pk: int | None = None
    best_potd_score = 0.0
    for result in results:
        if result.status == "pick" and result.selected_decision is not None:
            edge = float(result.selected_decision.edge_pct)
            if edge >= _POTD_MIN_EDGE:
                score = edge * float(result.selected_decision.model_probability)
                if score > best_potd_score:
                    best_potd_score = score
                    potd_game_pk = result.game_pk

    for result in results:
        if result.selected_decision is None and result.status != "error":
            continue
        if result.status == "error":
            continue
        try:
            game_features: dict[str, Any] = {}
            row = inference_frame.loc[inference_frame["game_pk"] == result.game_pk]
            if not row.empty:
                for col_name in row.columns:
                    val = row.iloc[0][col_name]
                    if pd.notna(val):
                        game_features[str(col_name)] = val

            decision_dict: dict[str, Any] | None = None
            if result.selected_decision is not None:
                decision_dict = result.selected_decision.model_dump(mode="json")

            prediction_dict: dict[str, Any] | None = None
            if result.prediction is not None:
                prediction_dict = result.prediction.model_dump(mode="json")

            result.narrative = generate_game_narrative(
                matchup=result.matchup,
                prediction=prediction_dict,
                decision=decision_dict,
                features=game_features,
                is_play_of_day=(result.game_pk == potd_game_pk),
                no_pick_reason=result.no_pick_reason,
            )
        except Exception:
            logger.debug("Narrative generation failed for game %s", result.game_pk, exc_info=True)

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

    result = DailyPipelineResult(
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
    _persist_cached_slate_response(
        database_path,
        pipeline_date=pipeline_day.isoformat(),
        mode=mode,
        dry_run=dry_run,
        result=result,
    )
    return result


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
    game_pk_allowlist: Collection[int] | None = None,
) -> list[OddsSnapshot]:
    primary_snapshots = fetch_mlb_odds(
        db_path=db_path,
        commence_time_from=commence_time_from,
        commence_time_to=commence_time_to,
        game_pk_allowlist=game_pk_allowlist,
    )
    primary_f5_games = {
        snapshot.game_pk for snapshot in primary_snapshots if snapshot.market_type == "f5_ml"
    }
    if primary_f5_games:
        return primary_snapshots

    sbr_snapshots = fetch_sbr_f5_odds(
        target_date=commence_time_from,
        db_path=db_path,
    )
    if not sbr_snapshots:
        return primary_snapshots

    merged = list(primary_snapshots)
    merged.extend(
        snapshot for snapshot in sbr_snapshots if snapshot.game_pk not in primary_f5_games
    )
    return sorted(
        merged,
        key=lambda snapshot: (snapshot.game_pk, snapshot.book_name, snapshot.market_type),
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
        lambda: _retry_with_circuit_breaker(
            _fetch_lineups_with_retry, _LINEUPS_CIRCUIT, target_date
        ),
        operation_name=f"lineup fetch for {target_date}",
        fallback=[],
        logger_=logger,
    )


def _load_scraper_market_state_payload(scraper_market_state_path: str | Path) -> list[dict[str, Any]]:
    path = Path(scraper_market_state_path)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Failed to load scraper market state from %s", path, exc_info=True)
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        games = payload.get("games")
        if isinstance(games, list):
            return [item for item in games if isinstance(item, dict)]
        return [item for item in payload.values() if isinstance(item, dict)]
    return []


def _load_windows_market_sync_state(sync_state_path: str | Path) -> dict[str, Any]:
    path = Path(sync_state_path)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Failed to load Windows market sync state from %s", path, exc_info=True)
        return {}
    return payload if isinstance(payload, dict) else {}


def _persist_windows_market_sync_state(
    sync_state_path: str | Path,
    payload: Mapping[str, Any],
) -> None:
    path = Path(sync_state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")


def _resolve_market_state_game_pk(
    *,
    state: Mapping[str, Any],
    game_pk_lookup: Mapping[tuple[str, str], int],
) -> int | None:
    raw_game_pk = state.get("game_pk")
    if raw_game_pk is not None:
        try:
            return int(raw_game_pk)
        except (TypeError, ValueError):
            pass
    away_team = _resolve_scraper_team_code(state.get("away_team"))
    home_team = _resolve_scraper_team_code(state.get("home_team"))
    if away_team is None or home_team is None:
        return None
    return game_pk_lookup.get((away_team, home_team))


def _eligible_f5_refresh_game_pks(
    *,
    target_date: date,
    repo_db_path: str | Path,
    scraper_market_state_path: str | Path,
    sync_state_path: str | Path,
) -> set[int] | None:
    states = _load_scraper_market_state_payload(scraper_market_state_path)
    if not states:
        return None

    game_pk_lookup = _load_repo_game_pk_lookup(repo_db_path=repo_db_path, target_date=target_date)
    if not game_pk_lookup:
        return None

    sync_state = _load_windows_market_sync_state(sync_state_path)
    fingerprint_state = sync_state.setdefault("full_game_fingerprints", {})
    eligible: set[int] = set()
    target_date_iso = target_date.isoformat()
    seen_game_pks: set[int] = set()
    required_f5_markets = {"f5_ml", "f5_rl"}

    for state in states:
        state_date = str(
            state.get("game_date")
            or state.get("pipeline_date")
            or state.get("date")
            or ""
        )
        if state_date and state_date != target_date_iso:
            continue
        game_pk = _resolve_market_state_game_pk(state=state, game_pk_lookup=game_pk_lookup)
        if game_pk is None:
            continue
        seen_game_pks.add(int(game_pk))
        markets_present = {
            str(market)
            for market in (state.get("markets_present") or [])
            if market is not None
        }
        if required_f5_markets.issubset(markets_present):
            continue
        fingerprint = state.get("full_game_market_fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint:
            eligible.add(int(game_pk))
            continue
        if fingerprint_state.get(str(game_pk)) != fingerprint:
            eligible.add(int(game_pk))

    if not seen_game_pks:
        return None
    return eligible


def _mark_games_synced_to_fingerprint(
    *,
    target_date: date,
    game_pks: Collection[int],
    repo_db_path: str | Path,
    scraper_market_state_path: str | Path,
    sync_state_path: str | Path,
) -> None:
    if not game_pks:
        return
    normalized_game_pks = {int(value) for value in game_pks}
    states = _load_scraper_market_state_payload(scraper_market_state_path)
    if not states:
        return
    game_pk_lookup = _load_repo_game_pk_lookup(repo_db_path=repo_db_path, target_date=target_date)
    if not game_pk_lookup:
        return
    sync_state = _load_windows_market_sync_state(sync_state_path)
    fingerprint_state = sync_state.setdefault("full_game_fingerprints", {})
    for state in states:
        game_pk = _resolve_market_state_game_pk(state=state, game_pk_lookup=game_pk_lookup)
        if game_pk is None or int(game_pk) not in normalized_game_pks:
            continue
        fingerprint = state.get("full_game_market_fingerprint")
        if isinstance(fingerprint, str) and fingerprint:
            fingerprint_state[str(int(game_pk))] = fingerprint
    sync_state["updated_at"] = datetime.now(UTC).isoformat()
    _persist_windows_market_sync_state(sync_state_path, sync_state)


def _default_odds_fetcher(target_date: date, mode: Mode, db_path: str | Path) -> list[OddsSnapshot]:
    if mode == "prod":
        _prune_live_odds_cache(db_path, anchor_date=target_date)
        scraper_snapshots = _load_scraper_f5_odds_for_date(
            target_date=target_date,
            scraper_db_path=SCRAPER_ODDS_DB_PATH,
            repo_db_path=db_path,
        )
        cached_snapshots = _load_fresh_odds_from_db_for_date(
            db_path,
            target_date,
            max_age=timedelta(minutes=LIVE_ODDS_CACHE_TTL_MINUTES),
        )
        eligible_game_pks = _eligible_f5_refresh_game_pks(
            target_date=target_date,
            repo_db_path=db_path,
            scraper_market_state_path=SCRAPER_MARKET_STATE_PATH,
            sync_state_path=WINDOWS_MARKET_SYNC_STATE_PATH,
        )
        if eligible_game_pks == set():
            return sorted(
                [*scraper_snapshots, *cached_snapshots],
                key=lambda snapshot: (snapshot.game_pk, snapshot.book_name, snapshot.market_type),
            )
        cached_by_key = {
            (
                int(snapshot.game_pk),
                str(snapshot.book_name),
                str(snapshot.market_type),
                float(snapshot.home_point) if snapshot.home_point is not None else None,
                float(snapshot.away_point) if snapshot.away_point is not None else None,
            ): snapshot
            for snapshot in cached_snapshots
        }
        for snapshot in scraper_snapshots:
            cached_by_key[
                (
                    int(snapshot.game_pk),
                    str(snapshot.book_name),
                    str(snapshot.market_type),
                    float(snapshot.home_point) if snapshot.home_point is not None else None,
                    float(snapshot.away_point) if snapshot.away_point is not None else None,
                )
            ] = snapshot
        if cached_by_key and eligible_game_pks is not None and not eligible_game_pks:
            return sorted(
                cached_by_key.values(),
                key=lambda snapshot: (snapshot.game_pk, snapshot.book_name, snapshot.market_type),
            )
        start = datetime.combine(target_date, time.min, tzinfo=UTC)
        # MLB official game dates routinely spill past midnight UTC, especially
        # for West Coast starts and overseas openers. Extend the fetch window so
        # same official-date games still pick up live odds.
        end = start + timedelta(days=1, hours=8)
        live_snapshots = call_with_graceful_degradation(
            lambda: _retry_with_circuit_breaker(
                _fetch_live_odds_with_retry,
                _ODDS_CIRCUIT,
                db_path=db_path,
                commence_time_from=start,
                commence_time_to=end,
                game_pk_allowlist=eligible_game_pks,
            ),
            operation_name=f"live odds fetch for {target_date.isoformat()}",
            fallback=lambda _exc: _load_odds_from_db_for_date(db_path, target_date),
            logger_=logger,
        )
        combined_by_key = dict(cached_by_key)
        for snapshot in live_snapshots:
            combined_by_key[
                (
                    int(snapshot.game_pk),
                    str(snapshot.book_name),
                    str(snapshot.market_type),
                    float(snapshot.home_point) if snapshot.home_point is not None else None,
                    float(snapshot.away_point) if snapshot.away_point is not None else None,
                )
            ] = snapshot
        _mark_games_synced_to_fingerprint(
            target_date=target_date,
            game_pks={
                int(snapshot.game_pk)
                for snapshot in live_snapshots
                if snapshot.market_type in {"f5_ml", "f5_rl"}
            },
            repo_db_path=db_path,
            scraper_market_state_path=SCRAPER_MARKET_STATE_PATH,
            sync_state_path=WINDOWS_MARKET_SYNC_STATE_PATH,
        )
        return sorted(
            combined_by_key.values(),
            key=lambda snapshot: (snapshot.game_pk, snapshot.book_name, snapshot.market_type),
        )

    return _load_odds_from_db_for_date(db_path, target_date)


def _default_full_game_odds_context_fetcher(
    target_date: date, mode: Mode, db_path: str | Path
) -> dict[int, dict[str, Any]]:
    if mode != "prod":
        return {}

    _prune_live_odds_cache(db_path, anchor_date=target_date)
    scraper_context = _load_scraper_full_game_odds_context(
        target_date=target_date,
        scraper_db_path=SCRAPER_ODDS_DB_PATH,
        repo_db_path=db_path,
    )
    if scraper_context:
        return scraper_context
    cached_context = _load_cached_full_game_odds_context(
        db_path,
        target_date,
        max_age=timedelta(minutes=FULL_GAME_ODDS_CACHE_TTL_MINUTES),
    )
    if cached_context:
        return cached_context

    start = datetime.combine(target_date, time.min, tzinfo=UTC)
    end = start + timedelta(days=1, hours=8)
    context = call_with_graceful_degradation(
        lambda: fetch_mlb_full_game_odds_context(
            db_path=db_path,
            commence_time_from=start,
            commence_time_to=end,
        ),
        operation_name=f"full-game odds context fetch for {target_date.isoformat()}",
        fallback=lambda _exc: _load_cached_full_game_odds_context(
            db_path,
            target_date,
            max_age=None,
        ),
        logger_=logger,
    )
    if context:
        _persist_full_game_odds_context(db_path, target_date, context)
    return context


def _default_feature_frame_builder(
    *,
    target_date: date,
    schedule: pd.DataFrame,
    historical_games: pd.DataFrame,
    lineups: Sequence[Lineup],
    db_path: str | Path,
    weather_fetcher: Callable[..., Any] | None,
    refresh: bool = False,
) -> pd.DataFrame:
    database_path = Path(db_path)
    return build_live_feature_frame(
        target_date=target_date,
        schedule=schedule,
        historical_games=historical_games,
        db_path=database_path,
        lineups=lineups,
        refresh=refresh,
        weather_fetcher=weather_fetcher,
    )


def _build_slate_runtime_inputs(
    *,
    pipeline_day: date,
    mode: Mode,
    dry_run: bool,
    database_path: str | Path,
    schedule: pd.DataFrame,
    historical_games: pd.DataFrame,
    lineups_fetcher: LineupsFetcher,
    odds_fetcher: OddsFetcher,
    full_game_odds_fetcher: FullGameOddsContextFetcher,
    feature_frame_builder: FeatureFrameBuilder,
    weather_fetcher: Callable[..., Any] | None,
    refresh: bool,
) -> SlateRuntimeInputs:
    lineups = lineups_fetcher(pipeline_day.isoformat())
    odds = odds_fetcher(pipeline_day, mode, database_path)
    full_game_odds_context = full_game_odds_fetcher(pipeline_day, mode, database_path)
    _persist_odds_snapshots(database_path, odds)

    resolved_odds = list(odds)
    if dry_run:
        existing_f5_games = {
            snapshot.game_pk for snapshot in resolved_odds if snapshot.market_type == "f5_ml"
        }
        estimated_odds = [
            snapshot
            for snapshot in build_estimated_f5_ml_snapshots(full_game_odds_context)
            if snapshot.game_pk not in existing_f5_games
        ]
        if estimated_odds:
            logger.info(
                "Added %s preview-only estimated F5 odds snapshots from full-game markets",
                len(estimated_odds),
            )
            resolved_odds = sorted(
                [*resolved_odds, *estimated_odds],
                key=lambda snapshot: (snapshot.game_pk, snapshot.book_name, snapshot.market_type),
            )

    inference_frame = _invoke_feature_frame_builder(
        feature_frame_builder=feature_frame_builder,
        target_date=pipeline_day,
        schedule=schedule,
        historical_games=historical_games,
        lineups=lineups,
        db_path=database_path,
        weather_fetcher=weather_fetcher,
        refresh=refresh,
    )
    return SlateRuntimeInputs(
        schedule=schedule.copy().reset_index(drop=True),
        lineups=list(lineups),
        odds=resolved_odds,
        full_game_odds_context=dict(full_game_odds_context),
        inference_frame=inference_frame,
        refresh=refresh,
    )


def _invoke_feature_frame_builder(
    *,
    feature_frame_builder: FeatureFrameBuilder,
    target_date: date,
    schedule: pd.DataFrame,
    historical_games: pd.DataFrame,
    lineups: Sequence[Lineup],
    db_path: str | Path,
    weather_fetcher: Callable[..., Any] | None,
    refresh: bool,
) -> pd.DataFrame:
    kwargs = {
        "target_date": target_date,
        "schedule": schedule,
        "historical_games": historical_games,
        "lineups": lineups,
        "db_path": db_path,
        "weather_fetcher": weather_fetcher,
        "refresh": refresh,
    }
    try:
        return feature_frame_builder(**kwargs)
    except TypeError as exc:
        if "refresh" not in str(exc):
            raise
        kwargs.pop("refresh", None)
        return feature_frame_builder(**kwargs)


def _parse_schedule_game(game: dict[str, Any]) -> dict[str, Any] | None:
    game_type = str(game.get("gameType") or "").upper()
    if game_type != "R":
        return None

    scheduled_start = _coerce_timestamp(game.get("gameDate"))
    official_date = str(game.get("officialDate") or scheduled_start.date().isoformat())
    home_payload = game.get("teams", {}).get("home", {})
    away_payload = game.get("teams", {}).get("away", {})
    home_team = _normalize_team_code(
        home_payload.get("team", {}).get("abbreviation") or home_payload.get("team", {}).get("name")
    )
    away_team = _normalize_team_code(
        away_payload.get("team", {}).get("abbreviation") or away_payload.get("team", {}).get("name")
    )
    if home_team is None or away_team is None:
        return None

    venue = str(
        game.get("venue", {}).get("name") or game.get("venue", {}).get("locationName") or home_team
    )
    park = get_park_factors(team_code=home_team, venue=venue)
    linescore = game.get("linescore", {})
    innings = linescore.get("innings") or []

    return {
        "game_pk": int(game["gamePk"]),
        "season": int(scheduled_start.year),
        "game_date": official_date,
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
        "f5_home_score": sum(_inning_runs(inning, "home") for inning in innings[:5])
        if innings
        else None,
        "f5_away_score": sum(_inning_runs(inning, "away") for inning in innings[:5])
        if innings
        else None,
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
                game_status TEXT,
                is_completed INTEGER NOT NULL DEFAULT 0 CHECK (is_completed IN (0, 1)),
                notified INTEGER NOT NULL DEFAULT 0 CHECK (notified IN (0, 1)),
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (game_pk) REFERENCES games (game_pk)
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_daily_pipeline_results_game ON daily_pipeline_results (game_pk)"
        )
        _ensure_sqlite_column(
            connection,
            table_name="daily_pipeline_results",
            column_name="game_status",
            column_definition="TEXT",
        )
        _ensure_sqlite_column(
            connection,
            table_name="daily_pipeline_results",
            column_name="is_completed",
            column_definition="INTEGER NOT NULL DEFAULT 0 CHECK (is_completed IN (0, 1))",
        )
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
        connection.commit()


def _ensure_sqlite_column(
    connection: sqlite3.Connection,
    *,
    table_name: str,
    column_name: str,
    column_definition: str,
) -> None:
    existing_columns = {
        str(row[1]) for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    }
    if column_name in existing_columns:
        return
    connection.execute(
        f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}"
    )


def _upsert_games(db_path: str | Path, schedule: pd.DataFrame) -> None:
    if schedule.empty:
        return

    rows = [
        (
            int(game["game_pk"]),
            str(
                game.get("game_date")
                or _coerce_timestamp(game["scheduled_start"]).date().isoformat()
            ),
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
                home_point,
                away_point,
                fetched_at,
                is_frozen
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    snapshot.game_pk,
                    snapshot.book_name,
                    snapshot.market_type,
                    snapshot.home_odds,
                    snapshot.away_odds,
                    snapshot.home_point,
                    snapshot.away_point,
                    snapshot.fetched_at.isoformat(),
                    int(snapshot.is_frozen),
                )
                for snapshot in snapshots
            ],
        )
        for game_pk, market_type in {
            (snapshot.game_pk, snapshot.market_type) for snapshot in snapshots
        }:
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
            SELECT o.game_pk, o.book_name, o.market_type, o.home_odds, o.away_odds, o.home_point, o.away_point, o.fetched_at, o.is_frozen
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
            home_point=row[5],
            away_point=row[6],
            fetched_at=_coerce_timestamp(row[7]),
            is_frozen=bool(row[8]),
        )
        for row in rows
    ]


def _resolve_scraper_team_code(value: Any) -> str | None:
    raw_value = str(value or "").strip()
    if not raw_value:
        return None
    alias_map = {
        "AZ": "ARI",
        "ARI": "ARI",
        "ATH": "OAK",
        "OAK": "OAK",
        "CHW": "CWS",
        "CWS": "CWS",
        "WAS": "WSH",
        "WSH": "WSH",
        "SDP": "SD",
        "SD": "SD",
        "SFG": "SF",
        "SF": "SF",
        "KCR": "KC",
        "KC": "KC",
        "TBR": "TB",
        "TB": "TB",
        "NYY": "NYY",
        "NYM": "NYM",
        "LAD": "LAD",
        "LAA": "LAA",
    }
    if raw_value in alias_map:
        return alias_map[raw_value]
    return _normalize_team_code(raw_value)


def _load_repo_game_pk_lookup(
    *,
    repo_db_path: str | Path,
    target_date: date,
) -> dict[tuple[str, str], int]:
    database_path = Path(repo_db_path)
    if not database_path.exists():
        return {}

    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            """
            SELECT game_pk, home_team, away_team
            FROM games
            WHERE substr(date, 1, 10) = ?
            """,
            (target_date.isoformat(),),
        ).fetchall()

    lookup: dict[tuple[str, str], int] = {}
    for game_pk, home_team, away_team in rows:
        resolved_home = _resolve_scraper_team_code(home_team)
        resolved_away = _resolve_scraper_team_code(away_team)
        if resolved_home is None or resolved_away is None:
            continue
        lookup[(resolved_away, resolved_home)] = int(game_pk)
    return lookup


def _load_scraper_rows_for_date(
    *,
    scraper_db_path: str | Path,
    target_date: date,
    market_types: Sequence[str],
) -> list[sqlite3.Row]:
    database_path = Path(scraper_db_path)
    if not database_path.exists():
        return []

    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        placeholders = ",".join("?" for _ in market_types)
        rows = connection.execute(
            f"""
            SELECT event_id, game_date, commence_time_utc, away_team, home_team,
                   fetched_at, bookmaker, market_type, side, point, price, is_opening
            FROM odds
            WHERE game_date = ?
              AND market_type IN ({placeholders})
            ORDER BY fetched_at DESC, id DESC
            """,
            (target_date.isoformat(), *market_types),
        ).fetchall()
    return list(rows)


def _latest_scraper_market_rows(
    rows: Sequence[sqlite3.Row],
    *,
    game_pk_lookup: Mapping[tuple[str, str], int],
) -> dict[tuple[int, str, str, float | None, str], sqlite3.Row]:
    latest: dict[tuple[int, str, str, float | None, str], sqlite3.Row] = {}
    for row in rows:
        away_team = _resolve_scraper_team_code(row["away_team"])
        home_team = _resolve_scraper_team_code(row["home_team"])
        if away_team is None or home_team is None:
            continue
        game_pk = game_pk_lookup.get((away_team, home_team))
        if game_pk is None:
            continue
        point = float(row["point"]) if row["point"] is not None else None
        market_type = str(row["market_type"])
        point_key = abs(point) if market_type in {"f5_rl", "full_game_rl"} and point is not None else point
        key = (
            int(game_pk),
            str(row["bookmaker"]),
            market_type,
            point_key,
            str(row["side"]),
        )
        if key not in latest:
            latest[key] = row
    return latest


def _load_scraper_f5_odds_for_date(
    *,
    target_date: date,
    scraper_db_path: str | Path,
    repo_db_path: str | Path,
) -> list[OddsSnapshot]:
    game_pk_lookup = _load_repo_game_pk_lookup(repo_db_path=repo_db_path, target_date=target_date)
    if not game_pk_lookup:
        return []

    rows = _load_scraper_rows_for_date(
        scraper_db_path=scraper_db_path,
        target_date=target_date,
        market_types=("f5_ml", "f5_rl"),
    )
    if not rows:
        return []

    latest_rows = _latest_scraper_market_rows(rows, game_pk_lookup=game_pk_lookup)
    paired: dict[tuple[int, str, str, float | None], dict[str, sqlite3.Row]] = {}
    for (game_pk, bookmaker, market_type, point, side), row in latest_rows.items():
        paired.setdefault((game_pk, bookmaker, market_type, point), {})[side] = row

    snapshots: list[OddsSnapshot] = []
    for (game_pk, bookmaker, market_type, _point), sides in paired.items():
        home_row = sides.get("home")
        away_row = sides.get("away")
        if home_row is None or away_row is None:
            continue
        home_point = float(home_row["point"]) if home_row["point"] is not None else None
        away_point = float(away_row["point"]) if away_row["point"] is not None else None
        fetched_at = max(
            _coerce_timestamp(home_row["fetched_at"]),
            _coerce_timestamp(away_row["fetched_at"]),
        )
        snapshots.append(
            OddsSnapshot(
                game_pk=int(game_pk),
                book_name=f"scraper:{bookmaker}",
                market_type=str(market_type),
                home_odds=int(home_row["price"]),
                away_odds=int(away_row["price"]),
                home_point=home_point,
                away_point=away_point,
                fetched_at=fetched_at,
                is_frozen=False,
            )
        )

    return sorted(snapshots, key=lambda snapshot: (snapshot.game_pk, snapshot.book_name, snapshot.market_type))


def _is_better_price(candidate: int, current: int | None) -> bool:
    if current is None:
        return True
    return int(candidate) > int(current)


def _load_scraper_full_game_odds_context(
    *,
    target_date: date,
    scraper_db_path: str | Path,
    repo_db_path: str | Path,
) -> dict[int, dict[str, Any]]:
    game_pk_lookup = _load_repo_game_pk_lookup(repo_db_path=repo_db_path, target_date=target_date)
    if not game_pk_lookup:
        return {}

    rows = _load_scraper_rows_for_date(
        scraper_db_path=scraper_db_path,
        target_date=target_date,
        market_types=("full_game_ml", "full_game_rl", "full_game_total"),
    )
    if not rows:
        return {}

    latest_rows = _latest_scraper_market_rows(rows, game_pk_lookup=game_pk_lookup)
    paired: dict[tuple[int, str, str, float | None], dict[str, sqlite3.Row]] = {}
    for (game_pk, bookmaker, market_type, point, side), row in latest_rows.items():
        paired.setdefault((game_pk, bookmaker, market_type, point), {})[side] = row

    result: dict[int, dict[str, Any]] = {}
    for (game_pk, bookmaker, market_type, point), sides in paired.items():
        game_context = result.setdefault(
            int(game_pk),
            {
                "full_game_odds_available": False,
                "full_game_odds_books": [],
                "full_game_home_ml": None,
                "full_game_home_ml_book": None,
                "full_game_away_ml": None,
                "full_game_away_ml_book": None,
                "bet365_full_game_home_ml": None,
                "bet365_full_game_away_ml": None,
                "consensus_full_game_home_ml": None,
                "consensus_full_game_away_ml": None,
                "full_game_home_spread": None,
                "full_game_home_spread_odds": None,
                "full_game_home_spread_book": None,
                "full_game_away_spread": None,
                "full_game_away_spread_odds": None,
                "full_game_away_spread_book": None,
                "bet365_full_game_home_spread": None,
                "bet365_full_game_home_spread_odds": None,
                "bet365_full_game_away_spread": None,
                "bet365_full_game_away_spread_odds": None,
                "consensus_full_game_home_spread": None,
                "consensus_full_game_home_spread_odds": None,
                "consensus_full_game_away_spread": None,
                "consensus_full_game_away_spread_odds": None,
                "full_game_total": None,
                "full_game_total_over_odds": None,
                "full_game_total_under_odds": None,
                "full_game_total_book": None,
                "bet365_full_game_total": None,
                "bet365_full_game_total_over_odds": None,
                "bet365_full_game_total_under_odds": None,
                "consensus_full_game_total": None,
                "consensus_full_game_total_over_odds": None,
                "consensus_full_game_total_under_odds": None,
                "full_game_ml_pairs": [],
                "full_game_rl_pairs": [],
                "full_game_total_pairs": [],
            },
        )
        if bookmaker not in game_context["full_game_odds_books"]:
            game_context["full_game_odds_books"].append(bookmaker)

        if market_type == "full_game_ml":
            home_row = sides.get("home")
            away_row = sides.get("away")
            if home_row is None or away_row is None:
                continue
            home_price = int(home_row["price"])
            away_price = int(away_row["price"])
            game_context["full_game_odds_available"] = True
            game_context["full_game_ml_pairs"].append(
                {
                    "book_name": bookmaker,
                    "home_odds": home_price,
                    "away_odds": away_price,
                }
            )
            if _book_name_key(bookmaker) == "bet365":
                game_context["bet365_full_game_home_ml"] = home_price
                game_context["bet365_full_game_away_ml"] = away_price
            if _is_better_price(home_price, game_context["full_game_home_ml"]):
                game_context["full_game_home_ml"] = home_price
                game_context["full_game_home_ml_book"] = bookmaker
            if _is_better_price(away_price, game_context["full_game_away_ml"]):
                game_context["full_game_away_ml"] = away_price
                game_context["full_game_away_ml_book"] = bookmaker
            continue

        if market_type == "full_game_rl":
            home_row = sides.get("home")
            away_row = sides.get("away")
            if home_row is None or away_row is None:
                continue
            home_price = int(home_row["price"])
            away_price = int(away_row["price"])
            home_point = float(home_row["point"]) if home_row["point"] is not None else None
            away_point = float(away_row["point"]) if away_row["point"] is not None else None
            game_context["full_game_odds_available"] = True
            game_context["full_game_rl_pairs"].append(
                {
                    "book_name": bookmaker,
                    "home_point": home_point,
                    "home_odds": home_price,
                    "away_point": away_point,
                    "away_odds": away_price,
                }
            )
            if _book_name_key(bookmaker) == "bet365":
                game_context["bet365_full_game_home_spread"] = home_point
                game_context["bet365_full_game_home_spread_odds"] = home_price
                game_context["bet365_full_game_away_spread"] = away_point
                game_context["bet365_full_game_away_spread_odds"] = away_price
            if _is_better_price(home_price, game_context["full_game_home_spread_odds"]):
                game_context["full_game_home_spread"] = home_point
                game_context["full_game_home_spread_odds"] = home_price
                game_context["full_game_home_spread_book"] = bookmaker
            if _is_better_price(away_price, game_context["full_game_away_spread_odds"]):
                game_context["full_game_away_spread"] = away_point
                game_context["full_game_away_spread_odds"] = away_price
                game_context["full_game_away_spread_book"] = bookmaker
            continue

        if market_type == "full_game_total":
            over_row = sides.get("over")
            under_row = sides.get("under")
            if over_row is None or under_row is None:
                continue
            over_price = int(over_row["price"])
            under_price = int(under_row["price"])
            total_point = (
                float(over_row["point"])
                if over_row["point"] is not None
                else float(under_row["point"])
                if under_row["point"] is not None
                else None
            )
            game_context["full_game_odds_available"] = True
            game_context["full_game_total_pairs"].append(
                {
                    "book_name": bookmaker,
                    "total_point": total_point,
                    "over_odds": over_price,
                    "under_odds": under_price,
                }
            )
            if _book_name_key(bookmaker) == "bet365":
                game_context["bet365_full_game_total"] = total_point
                game_context["bet365_full_game_total_over_odds"] = over_price
                game_context["bet365_full_game_total_under_odds"] = under_price
            if _is_better_price(over_price, game_context["full_game_total_over_odds"]):
                game_context["full_game_total"] = total_point
                game_context["full_game_total_over_odds"] = over_price
                game_context["full_game_total_under_odds"] = under_price
                game_context["full_game_total_book"] = bookmaker

    for game_context in result.values():
        game_context["full_game_odds_books"] = sorted(game_context["full_game_odds_books"])
        ml_pairs = list(game_context.get("full_game_ml_pairs") or [])
        if ml_pairs:
            game_context["consensus_full_game_home_ml"] = _average_int(
                [pair["home_odds"] for pair in ml_pairs if pair.get("home_odds") is not None]
            )
            game_context["consensus_full_game_away_ml"] = _average_int(
                [pair["away_odds"] for pair in ml_pairs if pair.get("away_odds") is not None]
            )
        rl_pairs = list(game_context.get("full_game_rl_pairs") or [])
        if rl_pairs:
            game_context["consensus_full_game_home_spread"] = _average_float(
                [pair["home_point"] for pair in rl_pairs if pair.get("home_point") is not None]
            )
            game_context["consensus_full_game_home_spread_odds"] = _average_int(
                [pair["home_odds"] for pair in rl_pairs if pair.get("home_odds") is not None]
            )
            game_context["consensus_full_game_away_spread"] = _average_float(
                [pair["away_point"] for pair in rl_pairs if pair.get("away_point") is not None]
            )
            game_context["consensus_full_game_away_spread_odds"] = _average_int(
                [pair["away_odds"] for pair in rl_pairs if pair.get("away_odds") is not None]
            )
        total_pairs = list(game_context.get("full_game_total_pairs") or [])
        if total_pairs:
            game_context["consensus_full_game_total"] = _average_float(
                [pair["total_point"] for pair in total_pairs if pair.get("total_point") is not None]
            )
            game_context["consensus_full_game_total_over_odds"] = _average_int(
                [pair["over_odds"] for pair in total_pairs if pair.get("over_odds") is not None]
            )
            game_context["consensus_full_game_total_under_odds"] = _average_int(
                [pair["under_odds"] for pair in total_pairs if pair.get("under_odds") is not None]
            )
    return result


def _load_fresh_odds_from_db_for_date(
    db_path: str | Path,
    target_date: date,
    *,
    max_age: timedelta | None,
) -> list[OddsSnapshot]:
    snapshots = _load_odds_from_db_for_date(db_path, target_date)
    if not snapshots or max_age is None:
        return snapshots

    newest_fetched_at = max(snapshot.fetched_at for snapshot in snapshots)
    if datetime.now(UTC) - newest_fetched_at > max_age:
        return []
    return snapshots


def _ensure_full_game_odds_cache_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS full_game_odds_context_cache (
            game_pk INTEGER NOT NULL,
            target_date TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            fetched_at TEXT NOT NULL,
            PRIMARY KEY (game_pk, target_date),
            FOREIGN KEY (game_pk) REFERENCES games (game_pk)
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_full_game_odds_context_cache_target_date
        ON full_game_odds_context_cache (target_date, fetched_at)
        """
    )


def _persist_full_game_odds_context(
    db_path: str | Path,
    target_date: date,
    context_by_game: Mapping[int, Mapping[str, Any]],
) -> None:
    if not context_by_game:
        return

    fetched_at = datetime.now(UTC).isoformat()
    with sqlite3.connect(db_path) as connection:
        _ensure_full_game_odds_cache_table(connection)
        connection.executemany(
            """
            INSERT INTO full_game_odds_context_cache (
                game_pk,
                target_date,
                payload_json,
                fetched_at
            )
            VALUES (?, ?, ?, ?)
            ON CONFLICT(game_pk, target_date) DO UPDATE SET
                payload_json = excluded.payload_json,
                fetched_at = excluded.fetched_at
            """,
            [
                (
                    int(game_pk),
                    target_date.isoformat(),
                    json.dumps(dict(payload)),
                    fetched_at,
                )
                for game_pk, payload in context_by_game.items()
            ],
        )
        connection.commit()


def _load_cached_full_game_odds_context(
    db_path: str | Path,
    target_date: date,
    *,
    max_age: timedelta | None,
) -> dict[int, dict[str, Any]]:
    database_path = Path(db_path)
    if not database_path.exists():
        return {}

    with sqlite3.connect(database_path) as connection:
        _ensure_full_game_odds_cache_table(connection)
        rows = connection.execute(
            """
            SELECT game_pk, payload_json, fetched_at
            FROM full_game_odds_context_cache
            WHERE target_date = ?
            """,
            (target_date.isoformat(),),
        ).fetchall()

    if not rows:
        return {}

    newest_fetched_at = max(_coerce_timestamp(row[2]) for row in rows)
    if max_age is not None and datetime.now(UTC) - newest_fetched_at > max_age:
        return {}

    result: dict[int, dict[str, Any]] = {}
    for game_pk, payload_json, _fetched_at in rows:
        payload = json.loads(str(payload_json))
        if isinstance(payload, dict):
            result[int(game_pk)] = payload
    return result


def _prune_live_odds_cache(db_path: str | Path, *, anchor_date: date) -> None:
    database_path = Path(db_path)
    if not database_path.exists():
        return

    min_date = (anchor_date - timedelta(days=LIVE_ODDS_CACHE_RETENTION_DAYS)).isoformat()
    max_date = (anchor_date + timedelta(days=LIVE_ODDS_CACHE_RETENTION_DAYS)).isoformat()
    with sqlite3.connect(database_path) as connection:
        _ensure_full_game_odds_cache_table(connection)
        connection.execute(
            """
            DELETE FROM odds_snapshots
            WHERE game_pk IN (
                SELECT game_pk
                FROM games
                WHERE substr(date, 1, 10) < ? OR substr(date, 1, 10) > ?
            )
            """,
            (min_date, max_date),
        )
        connection.execute(
            """
            DELETE FROM full_game_odds_context_cache
            WHERE target_date < ? OR target_date > ?
            """,
            (min_date, max_date),
        )
        connection.commit()


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


def _build_inference_row_lookup(inference_frame: pd.DataFrame) -> dict[int, pd.Series]:
    if inference_frame.empty or "game_pk" not in inference_frame.columns:
        return {}
    lookup: dict[int, pd.Series] = {}
    for _, row in inference_frame.drop_duplicates(subset=["game_pk"], keep="last").iterrows():
        lookup[int(row["game_pk"])] = row.copy()
    return lookup


def _validation_no_pick_reason(
    *,
    game: dict[str, Any],
    inference_row: pd.Series,
    lineups_by_game_team: dict[tuple[int, str], Lineup],
    odds_by_game: dict[int, list[OddsSnapshot]],
    full_game_odds_context_by_game: dict[int, dict[str, Any]],
    after_refresh: bool = False,
) -> str | None:
    reasons = _collect_validation_reasons(
        game=game,
        inference_row=inference_row,
        lineups_by_game_team=lineups_by_game_team,
        odds_by_game=odds_by_game,
        full_game_odds_context_by_game=full_game_odds_context_by_game,
        after_refresh=after_refresh,
    )
    return "; ".join(reasons) if reasons else None


def _collect_validation_reasons(
    *,
    game: dict[str, Any],
    inference_row: pd.Series,
    lineups_by_game_team: dict[tuple[int, str], Lineup],
    odds_by_game: dict[int, list[OddsSnapshot]],
    full_game_odds_context_by_game: dict[int, dict[str, Any]],
    after_refresh: bool = False,
) -> list[str]:
    game_pk = int(game["game_pk"])
    reasons: list[str] = []
    full_game_context = full_game_odds_context_by_game.get(game_pk, {})
    if not bool(full_game_context.get("full_game_odds_available")):
        reasons.append("full-game odds unavailable")

    if (not bool(game.get("is_dome", False))) and float(
        inference_row.get("weather_data_missing", 0.0) or 0.0
    ) >= 1.0:
        reasons.append("weather unavailable")

    if not after_refresh:
        return reasons
    return [f"{reason} after refresh" for reason in reasons]


def _build_input_status(
    *,
    game: dict[str, Any],
    inference_row: pd.Series,
    lineups_by_game_team: dict[tuple[int, str], Lineup],
    odds_by_game: dict[int, list[OddsSnapshot]],
    full_game_odds_context_by_game: dict[int, dict[str, Any]],
    baseline_schedule_lookup: Mapping[int, dict[str, Any]] | None = None,
    baseline_lineups_by_game_team: Mapping[tuple[int, str], Lineup] | None = None,
    baseline_odds_by_game: Mapping[int, list[OddsSnapshot]] | None = None,
    baseline_inference_lookup: Mapping[int, pd.Series] | None = None,
    recheck_performed: bool = False,
) -> dict[str, Any]:
    game_pk = int(game["game_pk"])
    home_lineup = lineups_by_game_team.get((game_pk, str(game["home_team"])))
    away_lineup = lineups_by_game_team.get((game_pk, str(game["away_team"])))
    snapshots = odds_by_game.get(game_pk, [])
    full_game_context = full_game_odds_context_by_game.get(game_pk, {})
    odds_sources = sorted({_odds_source_label(snapshot.book_name) for snapshot in snapshots})
    f5_ml_summary = _summarize_snapshot_market(snapshots, market_type="f5_ml")
    f5_rl_summary = _summarize_snapshot_market(snapshots, market_type="f5_rl")
    baseline_game = (baseline_schedule_lookup or {}).get(game_pk)
    baseline_home_lineup = (baseline_lineups_by_game_team or {}).get((game_pk, str(game["home_team"])))
    baseline_away_lineup = (baseline_lineups_by_game_team or {}).get((game_pk, str(game["away_team"])))
    baseline_snapshots = (baseline_odds_by_game or {}).get(game_pk, [])
    baseline_inference_row = (baseline_inference_lookup or {}).get(game_pk)
    lineup_age_minutes = _max_age_minutes(
        [
            home_lineup.as_of_timestamp if home_lineup is not None else None,
            away_lineup.as_of_timestamp if away_lineup is not None else None,
        ]
    )
    odds_age_minutes = _max_age_minutes([snapshot.fetched_at for snapshot in snapshots])
    feature_frame_age_minutes = _age_minutes(inference_row.get("as_of_timestamp"))
    weather_missing = (
        (not bool(game.get("is_dome", False)))
        and float(inference_row.get("weather_data_missing", 0.0) or 0.0) >= 1.0
    )
    status = {
        "home_lineup_available": bool(home_lineup and home_lineup.players),
        "home_lineup_confirmed": bool(home_lineup and home_lineup.confirmed),
        "home_lineup_source": home_lineup.source if home_lineup is not None else None,
        "away_lineup_available": bool(away_lineup and away_lineup.players),
        "away_lineup_confirmed": bool(away_lineup and away_lineup.confirmed),
        "away_lineup_source": away_lineup.source if away_lineup is not None else None,
        "odds_available": bool(snapshots),
        "odds_books": sorted({_display_odds_book_name(snapshot.book_name) for snapshot in snapshots}),
        "f5_odds_estimated": any(
            snapshot.book_name.startswith("estimate:") for snapshot in snapshots
        ),
        "f5_odds_sources": odds_sources,
        "bet365_f5_ml_home_odds": f5_ml_summary.get("bet365_home_odds"),
        "bet365_f5_ml_away_odds": f5_ml_summary.get("bet365_away_odds"),
        "consensus_f5_ml_home_odds": f5_ml_summary.get("consensus_home_odds"),
        "consensus_f5_ml_away_odds": f5_ml_summary.get("consensus_away_odds"),
        "bet365_f5_rl_home_point": f5_rl_summary.get("bet365_home_point"),
        "bet365_f5_rl_home_odds": f5_rl_summary.get("bet365_home_odds"),
        "bet365_f5_rl_away_point": f5_rl_summary.get("bet365_away_point"),
        "bet365_f5_rl_away_odds": f5_rl_summary.get("bet365_away_odds"),
        "consensus_f5_rl_home_point": f5_rl_summary.get("consensus_home_point"),
        "consensus_f5_rl_home_odds": f5_rl_summary.get("consensus_home_odds"),
        "consensus_f5_rl_away_point": f5_rl_summary.get("consensus_away_point"),
        "consensus_f5_rl_away_odds": f5_rl_summary.get("consensus_away_odds"),
        "full_game_odds_available": bool(full_game_context.get("full_game_odds_available")),
        "full_game_odds_books": list(full_game_context.get("full_game_odds_books") or []),
        "full_game_home_ml": full_game_context.get("full_game_home_ml"),
        "full_game_home_ml_book": full_game_context.get("full_game_home_ml_book"),
        "full_game_away_ml": full_game_context.get("full_game_away_ml"),
        "full_game_away_ml_book": full_game_context.get("full_game_away_ml_book"),
        "bet365_full_game_home_ml": full_game_context.get("bet365_full_game_home_ml"),
        "bet365_full_game_away_ml": full_game_context.get("bet365_full_game_away_ml"),
        "consensus_full_game_home_ml": full_game_context.get("consensus_full_game_home_ml"),
        "consensus_full_game_away_ml": full_game_context.get("consensus_full_game_away_ml"),
        "full_game_home_spread": full_game_context.get("full_game_home_spread"),
        "full_game_home_spread_odds": full_game_context.get("full_game_home_spread_odds"),
        "full_game_home_spread_book": full_game_context.get("full_game_home_spread_book"),
        "full_game_away_spread": full_game_context.get("full_game_away_spread"),
        "full_game_away_spread_odds": full_game_context.get("full_game_away_spread_odds"),
        "full_game_away_spread_book": full_game_context.get("full_game_away_spread_book"),
        "bet365_full_game_home_spread": full_game_context.get("bet365_full_game_home_spread"),
        "bet365_full_game_home_spread_odds": full_game_context.get("bet365_full_game_home_spread_odds"),
        "bet365_full_game_away_spread": full_game_context.get("bet365_full_game_away_spread"),
        "bet365_full_game_away_spread_odds": full_game_context.get("bet365_full_game_away_spread_odds"),
        "consensus_full_game_home_spread": full_game_context.get("consensus_full_game_home_spread"),
        "consensus_full_game_home_spread_odds": full_game_context.get("consensus_full_game_home_spread_odds"),
        "consensus_full_game_away_spread": full_game_context.get("consensus_full_game_away_spread"),
        "consensus_full_game_away_spread_odds": full_game_context.get("consensus_full_game_away_spread_odds"),
        "weather_available": not weather_missing,
        "lineup_age_minutes": lineup_age_minutes,
        "lineups_stale": lineup_age_minutes is None or lineup_age_minutes > LINEUP_STALE_TTL_MINUTES,
        "odds_age_minutes": odds_age_minutes,
        "odds_stale": odds_age_minutes is None or odds_age_minutes > LIVE_ODDS_CACHE_TTL_MINUTES,
        "feature_frame_age_minutes": feature_frame_age_minutes,
        "feature_frame_stale": (
            feature_frame_age_minutes is None
            or feature_frame_age_minutes > FULL_GAME_ODDS_CACHE_TTL_MINUTES
        ),
        "weather_refresh_required": bool(
            (not bool(game.get("is_dome", False)))
            and (
                weather_missing
                or feature_frame_age_minutes is None
                or feature_frame_age_minutes > FULL_GAME_ODDS_CACHE_TTL_MINUTES
            )
        ),
        "umpire_available": float(inference_row.get("plate_umpire_known", 0.0) or 0.0) >= 1.0,
        "umpire_neutralized": False,
        "umpire_fallback_mode": None,
        "prelock_recheck_performed": bool(recheck_performed),
        "lineup_changed_during_recheck": bool(
            recheck_performed
            and (
                _lineup_player_ids(home_lineup) != _lineup_player_ids(baseline_home_lineup)
                or _lineup_player_ids(away_lineup) != _lineup_player_ids(baseline_away_lineup)
            )
        ),
        "starter_changed_during_recheck": bool(
            recheck_performed
            and (
                _resolve_lineup_or_schedule_starter_id(game, home_lineup, "home")
                != _resolve_lineup_or_schedule_starter_id(baseline_game, baseline_home_lineup, "home")
                or _resolve_lineup_or_schedule_starter_id(game, away_lineup, "away")
                != _resolve_lineup_or_schedule_starter_id(baseline_game, baseline_away_lineup, "away")
            )
        ),
        "weather_changed_during_recheck": bool(
            recheck_performed and _weather_signature(inference_row) != _weather_signature(baseline_inference_row)
        ),
        "odds_changed_during_recheck": bool(
            recheck_performed and _odds_market_signature(snapshots) != _odds_market_signature(baseline_snapshots)
        ),
        "scheduled_start_changed_during_recheck": bool(
            recheck_performed
            and _schedule_value(game, "scheduled_start") != _schedule_value(baseline_game, "scheduled_start")
        ),
        "game_status_changed_during_recheck": bool(
            recheck_performed
            and _normalize_game_status(_schedule_value(game, "status"))
            != _normalize_game_status(_schedule_value(baseline_game, "status"))
        ),
    }
    return status


def _lineup_player_ids(lineup: Lineup | None) -> tuple[int, ...]:
    if lineup is None:
        return ()
    players = sorted(lineup.players, key=lambda player: player.batting_order)
    return tuple(int(player.player_id) for player in players)


def _resolve_lineup_or_schedule_starter_id(
    game: Mapping[str, Any] | None,
    lineup: Lineup | None,
    side: Literal["home", "away"],
) -> int | None:
    if lineup is not None:
        lineup_starter = lineup.starting_pitcher_id or lineup.projected_starting_pitcher_id
        if lineup_starter is not None:
            return int(lineup_starter)
    if game is None:
        return None
    return _optional_int(game.get(f"{side}_starter_id"))


def _schedule_value(game: Mapping[str, Any] | None, key: str) -> Any:
    if game is None:
        return None
    return game.get(key)


def _weather_signature(row: pd.Series | None) -> tuple[float | None, float | None, float | None]:
    if row is None:
        return (None, None, None)
    return (
        _coerce_optional_float(row.get("weather_composite")),
        _coerce_optional_float(row.get("weather_wind_factor")),
        _coerce_optional_float(row.get("weather_data_missing")),
    )


def _odds_market_signature(snapshots: Sequence[OddsSnapshot]) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        sorted(
            (
                snapshot.market_type,
                str(snapshot.book_name),
                int(snapshot.home_odds),
                int(snapshot.away_odds),
                None if snapshot.home_point is None else float(snapshot.home_point),
                None if snapshot.away_point is None else float(snapshot.away_point),
            )
            for snapshot in snapshots
        )
    )


def _coerce_optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _age_minutes(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(timestamp):
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return max(
        0.0,
        float((pd.Timestamp(datetime.now(UTC)) - timestamp).total_seconds() / 60.0),
    )


def _max_age_minutes(values: Sequence[Any]) -> float | None:
    ages = [_age_minutes(value) for value in values]
    resolved = [age for age in ages if age is not None]
    if not resolved:
        return None
    return float(max(resolved))


def _neutralize_missing_umpire_features(row_frame: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    if row_frame.empty:
        return row_frame, False
    resolved = row_frame.copy()
    plate_umpire_known = float(resolved.iloc[0].get("plate_umpire_known", 0.0) or 0.0)
    if plate_umpire_known >= 1.0:
        return resolved, False
    umpire_columns = [column for column in resolved.columns if column.startswith("plate_umpire_")]
    if not umpire_columns:
        return resolved, False
    for column in umpire_columns:
        resolved.loc[:, column] = float(_default_feature_fill_value(column))
    return resolved, True


def _base_validation_reason(reason: str) -> str:
    return str(reason).removesuffix(" after refresh")


def _odds_source_label(book_name: str) -> str:
    if book_name.startswith("estimate:"):
        return "estimated from full-game market"
    if book_name.startswith("sbr:"):
        return "sportsbookreview fallback"
    if book_name.startswith("scraper:"):
        return "local scraper"
    return "odds api"


def _display_odds_book_name(book_name: str) -> str:
    if book_name.startswith("scraper:"):
        return book_name.split(":", 1)[1]
    if book_name.startswith("sbr:"):
        return book_name.split(":", 1)[1]
    return book_name


def _book_name_key(book_name: str | None) -> str:
    return _display_odds_book_name(str(book_name or "")).strip().casefold()


def _average_int(values: Sequence[int]) -> int | None:
    resolved = [int(value) for value in values]
    if not resolved:
        return None
    return int(round(sum(resolved) / len(resolved)))


def _average_float(values: Sequence[float]) -> float | None:
    resolved = [float(value) for value in values]
    if not resolved:
        return None
    return float(sum(resolved) / len(resolved))


def _summarize_snapshot_market(
    snapshots: Sequence[OddsSnapshot],
    *,
    market_type: str,
) -> dict[str, Any]:
    relevant = [snapshot for snapshot in snapshots if snapshot.market_type == market_type]
    if not relevant:
        return {}

    summary: dict[str, Any] = {}
    if market_type == "f5_ml":
        bet365 = next(
            (snapshot for snapshot in relevant if _book_name_key(snapshot.book_name) == "bet365"),
            None,
        )
        if bet365 is not None:
            summary["bet365_home_odds"] = int(bet365.home_odds)
            summary["bet365_away_odds"] = int(bet365.away_odds)
        summary["consensus_home_odds"] = _average_int([snapshot.home_odds for snapshot in relevant])
        summary["consensus_away_odds"] = _average_int([snapshot.away_odds for snapshot in relevant])
        return summary

    if market_type == "f5_rl":
        bet365 = next(
            (snapshot for snapshot in relevant if _book_name_key(snapshot.book_name) == "bet365"),
            None,
        )
        if bet365 is not None:
            summary["bet365_home_point"] = bet365.home_point
            summary["bet365_home_odds"] = int(bet365.home_odds)
            summary["bet365_away_point"] = bet365.away_point
            summary["bet365_away_odds"] = int(bet365.away_odds)
        home_points = [snapshot.home_point for snapshot in relevant if snapshot.home_point is not None]
        away_points = [snapshot.away_point for snapshot in relevant if snapshot.away_point is not None]
        summary["consensus_home_point"] = _average_float(home_points) if home_points else None
        summary["consensus_home_odds"] = _average_int([snapshot.home_odds for snapshot in relevant])
        summary["consensus_away_point"] = _average_float(away_points) if away_points else None
        summary["consensus_away_odds"] = _average_int([snapshot.away_odds for snapshot in relevant])
        return summary

    return summary


def _build_candidate_decisions(
    *,
    inference_frame: pd.DataFrame,
    prediction: Prediction,
    snapshots: Sequence[OddsSnapshot],
    db_path: str | Path,
    prediction_engine: PredictionEngine | Any,
    full_game_context: Mapping[str, Any] | None = None,
) -> list[BetDecision]:
    if hasattr(prediction_engine, "build_candidate_decisions"):
        return prediction_engine.build_candidate_decisions(
            inference_frame=inference_frame,
            prediction=prediction,
            snapshots=snapshots,
            db_path=db_path,
            full_game_context=full_game_context,
        )

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
                    home_point=snapshot.home_point,
                    away_point=snapshot.away_point,
                    book_name=snapshot.book_name,
                    db_path=db_path,
                ).model_copy(
                    update={
                        "source_model": (
                            "legacy_f5_ml" if snapshot.market_type == "f5_ml" else "legacy_f5_rl"
                        ),
                        "source_model_version": prediction.model_version,
                    }
                ),
                calculate_edge(
                    game_pk=prediction.game_pk,
                    market_type=snapshot.market_type,
                    side="away",
                    model_probability=away_probability,
                    home_odds=snapshot.home_odds,
                    away_odds=snapshot.away_odds,
                    home_point=snapshot.home_point,
                    away_point=snapshot.away_point,
                    book_name=snapshot.book_name,
                    db_path=db_path,
                ).model_copy(
                    update={
                        "source_model": (
                            "legacy_f5_ml" if snapshot.market_type == "f5_ml" else "legacy_f5_rl"
                        ),
                        "source_model_version": prediction.model_version,
                    }
                ),
            ]
        )
    return candidates


def _select_game_decision(
    *,
    candidates: Sequence[BetDecision],
    current_bankroll: float,
    peak_bankroll: float,
) -> tuple[BetDecision | None, bool]:
    positive_candidates = [
        candidate
        for candidate in candidates
        if candidate.is_positive_ev and _is_official_live_candidate(candidate)
    ]
    if not positive_candidates:
        return None, False

    drawdown_pct = (
        ((peak_bankroll - current_bankroll) / peak_bankroll) if peak_bankroll > 0 else 0.0
    )
    kill_switch_active = drawdown_pct >= float(_SETTINGS["thresholds"]["max_drawdown"])

    ranked_candidates: list[tuple[int, float, float, BetDecision]] = []
    for candidate in positive_candidates:
        threshold = _official_edge_threshold(candidate)
        passes_threshold = float(candidate.edge_pct) >= threshold
        selection_score = _official_selection_score(candidate)
        stake_units = (
            0.0
            if kill_switch_active
            else _official_stake_units(candidate, bankroll=current_bankroll)
        )
        ranked_candidates.append(
            (
                int(passes_threshold),
                float(selection_score),
                float(candidate.model_probability),
                candidate.model_copy(update={"kelly_stake": float(stake_units)}),
            )
        )

    ranked_candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    if not ranked_candidates:
        return None, kill_switch_active

    selected = ranked_candidates[0][3]
    if ranked_candidates[0][0] == 0:
        return None, kill_switch_active
    if kill_switch_active:
        return selected.model_copy(update={"kelly_stake": 0.0}), True
    return selected, False


def _select_forced_game_decision(candidates: Sequence[BetDecision]) -> BetDecision | None:
    bet365_candidates = [
        candidate
        for candidate in candidates
        if _book_name_key(candidate.book_name) == "bet365"
    ]
    target_candidates = bet365_candidates or []
    if not target_candidates:
        return None
    return max(
        target_candidates,
        key=lambda decision: (
            float(decision.edge_pct),
            float(decision.model_probability),
            float(decision.ev),
        ),
    )


def _is_official_live_candidate(candidate: BetDecision) -> bool:
    if _book_name_key(candidate.book_name) != "bet365":
        return False
    if str(candidate.book_name or "").startswith("estimate:"):
        return False
    if candidate.odds_at_bet < DEFAULT_OFFICIAL_MIN_BET_ODDS:
        return False
    if candidate.odds_at_bet > DEFAULT_OFFICIAL_MAX_BET_ODDS:
        return False
    if float(candidate.edge_pct) >= DEFAULT_OFFICIAL_MAX_TRUSTED_EDGE:
        return False
    if candidate.market_type == "full_game_ml":
        return candidate.source_model == "run_count_full_game_ml"
    if candidate.market_type == "full_game_rl":
        if candidate.odds_at_bet < DEFAULT_OFFICIAL_RL_MIN_BET_ODDS:
            return False
        if candidate.odds_at_bet > DEFAULT_OFFICIAL_RL_MAX_BET_ODDS:
            return False
        return candidate.source_model == "run_count_full_game_rl"
    if candidate.market_type == "full_game_total":
        return candidate.source_model == "run_count_full_game_total"
    return False


def _official_edge_threshold(candidate: BetDecision) -> float:
    _ = candidate
    return DEFAULT_OFFICIAL_VALUE_PLAY_MIN_EDGE


def _official_selection_score(candidate: BetDecision) -> float:
    score = float(candidate.edge_pct)
    if candidate.market_type in {"f5_rl", "full_game_rl"}:
        score -= DEFAULT_OFFICIAL_RL_SELECTION_PENALTY
    return score


def _official_stake_units(candidate: BetDecision, *, bankroll: float) -> float:
    threshold = _official_edge_threshold(candidate)
    effective_edge = max(0.0, float(candidate.edge_pct) - threshold)
    scale = min(1.0, effective_edge / DEFAULT_OFFICIAL_EDGE_SCALE_CAP)
    units = DEFAULT_OFFICIAL_MIN_UNITS + (
        (DEFAULT_OFFICIAL_MAX_UNITS - DEFAULT_OFFICIAL_MIN_UNITS) * scale
    )
    return min(float(units), float(bankroll))


def _recalibrate_ml_market_pair(
    *,
    home_probability: float,
    away_probability: float,
    home_odds: int,
    away_odds: int,
) -> tuple[float, float]:
    home_fair, away_fair = devig_probabilities(home_odds, away_odds)
    adjusted_home = shrink_probability_toward_market(
        model_probability=float(home_probability),
        fair_probability=float(home_fair),
        odds=int(home_odds),
        base_multiplier=DEFAULT_OFFICIAL_ML_MARKET_BASE_MULTIPLIER,
        plus_money_multiplier=DEFAULT_OFFICIAL_ML_MARKET_PLUS_MONEY_MULTIPLIER,
        high_edge_threshold=DEFAULT_OFFICIAL_ML_MARKET_HIGH_EDGE_THRESHOLD,
        high_edge_multiplier=DEFAULT_OFFICIAL_ML_MARKET_HIGH_EDGE_MULTIPLIER,
    )
    adjusted_away = shrink_probability_toward_market(
        model_probability=float(away_probability),
        fair_probability=float(away_fair),
        odds=int(away_odds),
        base_multiplier=DEFAULT_OFFICIAL_ML_MARKET_BASE_MULTIPLIER,
        plus_money_multiplier=DEFAULT_OFFICIAL_ML_MARKET_PLUS_MONEY_MULTIPLIER,
        high_edge_threshold=DEFAULT_OFFICIAL_ML_MARKET_HIGH_EDGE_THRESHOLD,
        high_edge_multiplier=DEFAULT_OFFICIAL_ML_MARKET_HIGH_EDGE_MULTIPLIER,
    )
    return adjusted_home, adjusted_away


def _load_bankroll_state(
    db_path: str | Path,
    *,
    starting_bankroll: float,
) -> tuple[float, float, float]:
    with sqlite3.connect(db_path) as connection:
        current_row = connection.execute(
            "SELECT running_balance FROM bankroll_ledger ORDER BY id DESC LIMIT 1"
        ).fetchone()
        peak_row = connection.execute("SELECT MAX(running_balance) FROM bankroll_ledger").fetchone()

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
    picks = [
        result
        for result in results
        if result.status == "pick" and result.selected_decision is not None
    ]
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
            if result.no_pick_reason == "kill-switch active"
            and result.selected_decision is not None
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
        "source_model": decision.source_model,
        "source_model_version": decision.source_model_version,
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
                game_status,
                is_completed,
                notified
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    result.selected_decision.model_probability
                    if result.selected_decision
                    else None,
                    result.selected_decision.edge_pct if result.selected_decision else None,
                    result.selected_decision.ev if result.selected_decision else None,
                    result.selected_decision.kelly_stake if result.selected_decision else None,
                    result.no_pick_reason,
                    result.error_message,
                    result.game_status,
                    int(result.is_completed),
                    int(result.notified),
                )
                for result in results
            ],
        )
        connection.commit()


def _persist_cached_slate_response(
    db_path: str | Path,
    *,
    pipeline_date: str,
    mode: Mode,
    dry_run: bool,
    result: DailyPipelineResult,
) -> None:
    payload_json = json.dumps(result.to_dict())
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO cached_slate_responses (
                pipeline_date,
                mode,
                dry_run,
                run_id,
                model_version,
                payload_json,
                refreshed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(pipeline_date, mode, dry_run)
            DO UPDATE SET
                run_id = excluded.run_id,
                model_version = excluded.model_version,
                payload_json = excluded.payload_json,
                refreshed_at = excluded.refreshed_at
            """,
            (
                pipeline_date,
                mode,
                int(dry_run),
                result.run_id,
                result.model_version,
                payload_json,
                datetime.now(UTC).isoformat(),
            ),
        )
        connection.commit()


def load_cached_slate_response(
    *,
    pipeline_date: str | date | datetime,
    mode: Mode,
    dry_run: bool,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict[str, Any] | None:
    database_path = init_db(db_path)
    _ensure_pipeline_tables(database_path)
    resolved_date = _coerce_date(pipeline_date).isoformat()
    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            """
            SELECT payload_json
            FROM cached_slate_responses
            WHERE pipeline_date = ? AND mode = ? AND dry_run = ?
            """,
            (resolved_date, mode, int(dry_run)),
        ).fetchone()
    if row is None or not row[0]:
        return None
    payload = json.loads(str(row[0]))
    if "games" not in payload:
        return None
    payload.pop("notification_payload", None)
    return payload


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


def _american_to_implied_probability(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    odds = float(value)
    if odds == 0:
        return None
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def _margin_total_to_team_runs(*, home_margin: float, total_runs: float) -> tuple[float, float]:
    home_runs = (float(total_runs) + float(home_margin)) / 2.0
    away_runs = float(total_runs) - home_runs
    if home_runs < 0.1:
        home_runs = 0.1
        away_runs = max(0.1, float(total_runs) - home_runs)
    if away_runs < 0.1:
        away_runs = 0.1
        home_runs = max(0.1, float(total_runs) - away_runs)
    return round(home_runs, 2), round(away_runs, 2)


def _is_ml_equivalent_f5_runline(snapshot: OddsSnapshot) -> bool:
    if snapshot.market_type != "f5_rl":
        return False
    if snapshot.home_point is None or snapshot.away_point is None:
        return False
    try:
        home_point = float(snapshot.home_point)
        away_point = float(snapshot.away_point)
    except (TypeError, ValueError):
        return False
    return abs(home_point) == 0.5 and abs(away_point) == 0.5 and home_point == -away_point


def _legacy_variant_priority(variant: Literal["base", "stacking", "calibrated"]) -> int:
    if variant == "base":
        return 0
    if variant == "calibrated":
        return 1
    return 2


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
