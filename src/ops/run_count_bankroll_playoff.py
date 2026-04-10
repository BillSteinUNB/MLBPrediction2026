from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from rich.console import Console

from src.clients.historical_odds_client import load_historical_odds_for_games
from src.clients.odds_client import devig_probabilities
from src.engine.edge_calculator import payout_for_american_odds
from src.model.data_builder import validate_run_count_training_data
from src.model.run_count_trainer import (
    DEFAULT_RUN_COUNT_EARLY_STOPPING_ROUNDS,
    DEFAULT_RUN_COUNT_MODEL_SPECS,
    DEFAULT_RUN_COUNT_SEARCH_ITERATIONS,
    RunCountTrainingResult,
    train_run_count_models,
)
from src.model.run_distribution_trainer import (
    RunDistributionTrainingArtifact,
    train_run_distribution_model,
)
from src.model.score_pricing import (
    moneyline_probabilities,
    spread_cover_probabilities,
    totals_probabilities,
)
from src.model.xgboost_trainer import _load_training_dataframe


PROJECT_ROOT = Path(__file__).resolve().parents[2]
console = Console()
DEFAULT_BANKROLL_PLAYOFF_DIR = Path("data/reports/run_count/bankroll_playoff")
DEFAULT_STARTING_BANKROLL_UNITS = 100.0
DEFAULT_FLAT_BET_SIZE_UNITS = 1.0
DEFAULT_HISTORICAL_ODDS_DB = "data/mlb_odds_oddsportal_working_2021_2025.db"
DEFAULT_BANKROLL_HOLDOUT_SEASONS = (2024, 2025)
DEFAULT_BANKROLL_CANDIDATE_LABELS = (
    "reset_benchmark_v1",
    "window2018_test_19_stage4_starter_6",
    "window2018_test_20_stage4_sim_2000",
)
TRUSTED_BANKROLL_MARKET_TYPES = (
    "full_game_ml",
    "full_game_rl",
    "full_game_total",
    "f5_ml",
    "f5_total",
)


@dataclass(frozen=True, slots=True)
class BankrollPlayoffCandidateConfig:
    label: str
    training_data_path: str
    start_year: int
    folds: int
    feature_selection_mode: str
    forced_delta_count: int
    mu_delta_mode: str
    simulations: int
    starter_innings: int
    enable_market_priors: bool = True
    historical_odds_db_path: str | None = DEFAULT_HISTORICAL_ODDS_DB
    historical_market_book_name: str | None = None
    xgb_workers: int = 4


@dataclass(frozen=True, slots=True)
class _CompanionModelBundle:
    model: Any
    feature_columns: list[str]
    rmse: float
    metadata_path: Path
    model_version: str


@dataclass(frozen=True, slots=True)
class _CandidateSeasonArtifacts:
    companion_training_result: RunCountTrainingResult
    stage3_artifact: RunDistributionTrainingArtifact
    stage4_metadata_path: Path
    stage4_metadata: dict[str, Any]
    stage4_predictions_csv_path: Path


@dataclass(frozen=True, slots=True)
class BankrollPlayoffSeasonResult:
    candidate_label: str
    holdout_season: int
    starting_bankroll_units: float
    ending_bankroll_units: float
    peak_bankroll_units: float
    bankroll_return_pct: float
    max_drawdown_pct: float
    total_bets: int
    win_count: int
    loss_count: int
    push_count: int
    win_rate: float | None
    roi: float | None
    net_units: float
    average_bet_size_units: float | None
    median_bet_size_units: float | None
    report_json_path: Path
    bets_csv_path: Path
    by_market_type: dict[str, dict[str, float | int | None]]
    stage4_model_version: str
    stage4_mean_crps: float
    stage4_mean_negative_log_score: float


@dataclass(frozen=True, slots=True)
class BankrollPlayoffCandidateResult:
    candidate_label: str
    seasons: tuple[BankrollPlayoffSeasonResult, ...]
    combined_report_json_path: Path
    combined_bets_csv_path: Path
    combined_summary: dict[str, Any]


@dataclass(frozen=True, slots=True)
class BankrollPlayoffResult:
    generated_at: str
    output_dir: Path
    starting_bankroll_units: float
    flat_bet_size_units: float
    holdout_seasons: tuple[int, ...]
    candidate_results: tuple[BankrollPlayoffCandidateResult, ...]
    summary_json_path: Path


def _log_progress(message: str) -> None:
    console.print(f"[bold cyan][bankroll-playoff][/bold cyan] {message}")


def _load_existing_season_result(
    *,
    candidate_label: str,
    holdout_season: int,
    output_dir: Path,
) -> tuple[BankrollPlayoffSeasonResult, pd.DataFrame] | None:
    candidate_dir = output_dir / candidate_label / str(holdout_season)
    report_json_path = candidate_dir / "bankroll_summary.json"
    bets_csv_path = candidate_dir / "bankroll_bets.csv"
    if not report_json_path.exists():
        return None
    payload = json.loads(report_json_path.read_text(encoding="utf-8"))
    bets_frame = pd.read_csv(bets_csv_path) if bets_csv_path.exists() else pd.DataFrame()
    return (
        BankrollPlayoffSeasonResult(
            candidate_label=str(payload["candidate_label"]),
            holdout_season=int(payload["holdout_season"]),
            starting_bankroll_units=float(payload["starting_bankroll_units"]),
            ending_bankroll_units=float(payload["ending_bankroll_units"]),
            peak_bankroll_units=float(payload["peak_bankroll_units"]),
            bankroll_return_pct=float(payload["bankroll_return_pct"]),
            max_drawdown_pct=float(payload["max_drawdown_pct"]),
            total_bets=int(payload["total_bets"]),
            win_count=int(payload["win_count"]),
            loss_count=int(payload["loss_count"]),
            push_count=int(payload["push_count"]),
            win_rate=payload["win_rate"],
            roi=payload["roi"],
            net_units=float(payload["net_units"]),
            average_bet_size_units=payload["average_bet_size_units"],
            median_bet_size_units=payload["median_bet_size_units"],
            report_json_path=report_json_path,
            bets_csv_path=bets_csv_path,
            by_market_type=payload.get("by_market_type", {}),
            stage4_model_version=str(payload["stage4_model_version"]),
            stage4_mean_crps=float(payload["stage4_mean_crps"]),
            stage4_mean_negative_log_score=float(payload["stage4_mean_negative_log_score"]),
        ),
        bets_frame,
    )


def default_bankroll_playoff_candidates() -> dict[str, BankrollPlayoffCandidateConfig]:
    return {
        "reset_benchmark_v1": BankrollPlayoffCandidateConfig(
            label="reset_benchmark_v1",
            training_data_path="data/training/ParquetDefault.parquet",
            start_year=2021,
            folds=3,
            feature_selection_mode="flat",
            forced_delta_count=0,
            mu_delta_mode="off",
            simulations=1000,
            starter_innings=5,
        ),
        "window2018_test_19_stage4_starter_6": BankrollPlayoffCandidateConfig(
            label="window2018_test_19_stage4_starter_6",
            training_data_path="data/training/ParquetDefault_2018_no2020.parquet",
            start_year=2018,
            folds=3,
            feature_selection_mode="flat",
            forced_delta_count=0,
            mu_delta_mode="off",
            simulations=1000,
            starter_innings=6,
        ),
        "window2018_test_20_stage4_sim_2000": BankrollPlayoffCandidateConfig(
            label="window2018_test_20_stage4_sim_2000",
            training_data_path="data/training/ParquetDefault_2018_no2020.parquet",
            start_year=2018,
            folds=3,
            feature_selection_mode="flat",
            forced_delta_count=0,
            mu_delta_mode="off",
            simulations=2000,
            starter_innings=5,
        ),
        "workingdb_2021_2025_flat_baseline": BankrollPlayoffCandidateConfig(
            label="workingdb_2021_2025_flat_baseline",
            training_data_path="data/training/ParquetDefault.parquet",
            start_year=2021,
            folds=3,
            feature_selection_mode="flat",
            forced_delta_count=0,
            mu_delta_mode="off",
            simulations=1000,
            starter_innings=5,
        ),
    }


def _resolve_project_relative_path(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_and_filter_training_data(
    *,
    training_data_path: str | Path,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    validated = validate_run_count_training_data(Path(training_data_path))
    dataset = _load_training_dataframe(validated).copy()
    if "season" in dataset.columns:
        season_values = pd.to_numeric(dataset["season"], errors="coerce")
    else:
        season_values = pd.to_datetime(dataset["game_date"], errors="coerce").dt.year
    filtered = dataset.loc[(season_values >= int(start_year)) & (season_values <= int(end_year))].copy()
    filtered.attrs.update(dataset.attrs)
    return filtered


def _calculate_drawdown_pct(current_bankroll: float, peak_bankroll: float) -> float:
    if peak_bankroll <= 0:
        return 0.0
    return float((peak_bankroll - current_bankroll) / peak_bankroll)


def normalize_runline_points(
    *,
    home_point: float | int | None,
    away_point: float | int | None,
    home_odds: float | int | None,
    away_odds: float | int | None,
) -> tuple[float | None, float | None]:
    home_point_value = _coerce_finite_float(home_point)
    away_point_value = _coerce_finite_float(away_point)

    if home_point_value is None and away_point_value is None:
        return None, None
    if home_point_value is None or away_point_value is None:
        return (
            home_point_value,
            away_point_value,
        )

    resolved_home = float(home_point_value)
    resolved_away = float(away_point_value)
    if resolved_home == 0.0 and resolved_away == 0.0:
        return 0.0, 0.0
    if resolved_home * resolved_away < 0:
        return resolved_home, resolved_away

    if abs(resolved_home) == abs(resolved_away):
        abs_point = abs(resolved_home)
        if home_odds is None or away_odds is None:
            return abs_point, -abs_point
        def _implied_probability(odds: int) -> float:
            return (-odds / (-odds + 100.0)) if odds < 0 else (100.0 / (odds + 100.0))

        home_is_favorite = _implied_probability(int(home_odds)) > _implied_probability(int(away_odds))
        return (-abs_point, abs_point) if home_is_favorite else (abs_point, -abs_point)

    if abs(resolved_home) > abs(resolved_away):
        return resolved_home, -abs(resolved_away)
    return -abs(resolved_home), resolved_away


def _build_candidate_decision(
    *,
    game_pk: int,
    market_type: str,
    side: str,
    model_probability: float,
    fair_probability: float,
    odds_at_bet: int,
    line_at_bet: float | None,
    book_name: str,
) -> dict[str, Any]:
    edge_pct = float(model_probability - fair_probability)
    ev = float((model_probability * payout_for_american_odds(int(odds_at_bet))) - (1.0 - model_probability))
    return {
        "game_pk": int(game_pk),
        "market_type": market_type,
        "side": side,
        "book_name": book_name,
        "model_probability": float(model_probability),
        "fair_probability": float(fair_probability),
        "edge_pct": edge_pct,
        "ev": ev,
        "odds_at_bet": int(odds_at_bet),
        "line_at_bet": None if line_at_bet is None else float(line_at_bet),
        "is_positive_ev": bool(edge_pct > 0.0 and ev > 0.0),
    }


def _positive_ev_two_way_bets(
    *,
    game_pk: int,
    market_type: str,
    home_probability: float,
    away_probability: float,
    home_odds: int,
    away_odds: int,
    home_point: float | None,
    away_point: float | None,
    book_name: str,
) -> list[dict[str, Any]]:
    home_fair, away_fair = devig_probabilities(int(home_odds), int(away_odds))
    candidates = [
        _build_candidate_decision(
            game_pk=game_pk,
            market_type=market_type,
            side="home",
            model_probability=float(home_probability),
            fair_probability=float(home_fair),
            odds_at_bet=int(home_odds),
            line_at_bet=home_point,
            book_name=book_name,
        ),
        _build_candidate_decision(
            game_pk=game_pk,
            market_type=market_type,
            side="away",
            model_probability=float(away_probability),
            fair_probability=float(away_fair),
            odds_at_bet=int(away_odds),
            line_at_bet=away_point,
            book_name=book_name,
        ),
    ]
    return [candidate for candidate in candidates if candidate["is_positive_ev"]]


def _positive_ev_total_bets(
    *,
    game_pk: int,
    market_type: str,
    over_probability: float,
    under_probability: float,
    total_point: float,
    over_odds: int,
    under_odds: int,
    book_name: str,
) -> list[dict[str, Any]]:
    over_fair, under_fair = devig_probabilities(int(over_odds), int(under_odds))
    candidates = [
        _build_candidate_decision(
            game_pk=game_pk,
            market_type=market_type,
            side="over",
            model_probability=float(over_probability),
            fair_probability=float(over_fair),
            odds_at_bet=int(over_odds),
            line_at_bet=float(total_point),
            book_name=book_name,
        ),
        _build_candidate_decision(
            game_pk=game_pk,
            market_type=market_type,
            side="under",
            model_probability=float(under_probability),
            fair_probability=float(under_fair),
            odds_at_bet=int(under_odds),
            line_at_bet=float(total_point),
            book_name=book_name,
        ),
    ]
    return [candidate for candidate in candidates if candidate["is_positive_ev"]]


def _settle_market_bet(
    *,
    market_type: str,
    side: str,
    line_at_bet: float | None,
    odds_at_bet: int,
    flat_bet_size_units: float,
    full_game_home_score: int,
    full_game_away_score: int,
    f5_home_score: int,
    f5_away_score: int,
) -> tuple[float, str]:
    if market_type.startswith("f5_"):
        home_score = int(f5_home_score)
        away_score = int(f5_away_score)
    else:
        home_score = int(full_game_home_score)
        away_score = int(full_game_away_score)

    if market_type.endswith("_ml"):
        result = "WIN" if ((side == "home" and home_score > away_score) or (side == "away" and away_score > home_score)) else "LOSS"
    elif market_type.endswith("_rl"):
        if line_at_bet is None:
            raise ValueError(f"Runline bet missing line_at_bet for {market_type}")
        if side == "home":
            adjusted_home = float(home_score) + float(line_at_bet)
            adjusted_away = float(away_score)
        else:
            adjusted_home = float(home_score)
            adjusted_away = float(away_score) + float(line_at_bet)
        if adjusted_home == adjusted_away:
            result = "PUSH"
        elif (side == "home" and adjusted_home > adjusted_away) or (
            side == "away" and adjusted_away > adjusted_home
        ):
            result = "WIN"
        else:
            result = "LOSS"
    elif market_type.endswith("_total"):
        if line_at_bet is None:
            raise ValueError(f"Total bet missing line_at_bet for {market_type}")
        total_runs = home_score + away_score
        if side == "over":
            result = "WIN" if total_runs > float(line_at_bet) else "LOSS" if total_runs < float(line_at_bet) else "PUSH"
        else:
            result = "WIN" if total_runs < float(line_at_bet) else "LOSS" if total_runs > float(line_at_bet) else "PUSH"
    else:
        raise ValueError(f"Unsupported market_type: {market_type}")

    if result == "WIN":
        return float(payout_for_american_odds(int(odds_at_bet)) * float(flat_bet_size_units)), result
    if result == "LOSS":
        return float(-float(flat_bet_size_units)), result
    return 0.0, result


def _load_companion_models(
    training_result: RunCountTrainingResult,
) -> dict[str, _CompanionModelBundle]:
    bundles: dict[str, _CompanionModelBundle] = {}
    for model_name, artifact in training_result.models.items():
        bundles[model_name] = _CompanionModelBundle(
            model=joblib.load(artifact.model_path),
            feature_columns=list(artifact.feature_columns),
            rmse=float(artifact.holdout_metrics["rmse"]),
            metadata_path=artifact.metadata_path,
            model_version=artifact.model_version,
        )
    return bundles


def _predict_bundle(bundle: _CompanionModelBundle, frame: pd.DataFrame) -> pd.Series:
    missing_columns = {
        column: 0.0
        for column in bundle.feature_columns
        if column not in frame.columns
    }
    resolved = frame.copy()
    if missing_columns:
        resolved = pd.concat([resolved, pd.DataFrame(missing_columns, index=resolved.index)], axis=1)
    predictions = bundle.model.predict(resolved.loc[:, bundle.feature_columns])
    return pd.Series(np.maximum(np.asarray(predictions, dtype=float), 0.0), index=resolved.index)


def _load_all_market_odds(
    *,
    candidate: BankrollPlayoffCandidateConfig,
    games_frame: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    market_frames: dict[str, pd.DataFrame] = {}
    for market_type in TRUSTED_BANKROLL_MARKET_TYPES:
        market_frames[market_type] = load_historical_odds_for_games(
            db_path=str(candidate.historical_odds_db_path),
            games_frame=games_frame,
            market_type=market_type,
            book_name=candidate.historical_market_book_name,
            snapshot_selection="opening",
        )
        if "source_origin" in market_frames[market_type].columns:
            market_frames[market_type] = market_frames[market_type].loc[
                market_frames[market_type]["source_origin"].isin({"canonical", "legacy_old_scraper"})
            ].copy()
    return market_frames


def _coerce_finite_float(value: Any) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    resolved = float(numeric)
    return resolved if np.isfinite(resolved) else None


def _coerce_valid_odds(value: Any) -> int | None:
    numeric = _coerce_finite_float(value)
    if numeric is None:
        return None
    if numeric == 0.0:
        return None
    return int(round(numeric))


def _build_market_bets_for_row(
    *,
    row: pd.Series,
    market_row: dict[str, Any],
    companion_models: dict[str, _CompanionModelBundle],
) -> list[dict[str, Any]]:
    market_type = str(market_row["market_type"])
    if market_type == "full_game_ml":
        home_odds = _coerce_valid_odds(market_row.get("home_odds"))
        away_odds = _coerce_valid_odds(market_row.get("away_odds"))
        if home_odds is None or away_odds is None:
            return []
        probs = moneyline_probabilities(
            home_runs_mean=float(row["full_game_home_pred"]),
            away_runs_mean=float(row["mcmc_expected_away_runs"]),
            home_runs_std=float(companion_models["full_game_home_runs_model"].rmse),
            away_runs_std=float(row["full_game_away_std"]),
        )
        if probs[0] is None or probs[1] is None:
            return []
        return _positive_ev_two_way_bets(
            game_pk=int(row["game_pk"]),
            market_type=market_type,
            home_probability=float(probs[0]),
            away_probability=float(probs[1]),
            home_odds=home_odds,
            away_odds=away_odds,
            home_point=None,
            away_point=None,
            book_name=str(market_row.get("book_name") or "consensus"),
        )

    if market_type == "full_game_total":
        total_point = _coerce_finite_float(market_row.get("total_point"))
        over_odds = _coerce_valid_odds(market_row.get("over_odds"))
        under_odds = _coerce_valid_odds(market_row.get("under_odds"))
        if total_point is None or over_odds is None or under_odds is None:
            return []
        total_probs = totals_probabilities(
            home_runs_mean=float(row["full_game_home_pred"]),
            away_runs_mean=float(row["mcmc_expected_away_runs"]),
            home_runs_std=float(companion_models["full_game_home_runs_model"].rmse),
            away_runs_std=float(row["full_game_away_std"]),
            total_point=total_point,
        )
        if total_probs[0] is None or total_probs[1] is None:
            return []
        return _positive_ev_total_bets(
            game_pk=int(row["game_pk"]),
            market_type=market_type,
            over_probability=float(total_probs[0]),
            under_probability=float(total_probs[1]),
            total_point=total_point,
            over_odds=over_odds,
            under_odds=under_odds,
            book_name=str(market_row.get("book_name") or "consensus"),
        )

    if market_type == "full_game_rl":
        home_odds = _coerce_valid_odds(market_row.get("home_odds"))
        away_odds = _coerce_valid_odds(market_row.get("away_odds"))
        if home_odds is None or away_odds is None:
            return []
        home_point, away_point = normalize_runline_points(
            home_point=market_row.get("home_point"),
            away_point=market_row.get("away_point"),
            home_odds=home_odds,
            away_odds=away_odds,
        )
        if home_point is None or away_point is None:
            return []
        probs = spread_cover_probabilities(
            home_runs_mean=float(row["full_game_home_pred"]),
            away_runs_mean=float(row["mcmc_expected_away_runs"]),
            home_runs_std=float(companion_models["full_game_home_runs_model"].rmse),
            away_runs_std=float(row["full_game_away_std"]),
            home_point=float(home_point),
        )
        if probs[0] is None or probs[1] is None:
            return []
        return _positive_ev_two_way_bets(
            game_pk=int(row["game_pk"]),
            market_type=market_type,
            home_probability=float(probs[0]),
            away_probability=float(probs[1]),
            home_odds=home_odds,
            away_odds=away_odds,
            home_point=float(home_point),
            away_point=float(away_point),
            book_name=str(market_row.get("book_name") or "consensus"),
        )

    if market_type == "f5_ml":
        home_odds = _coerce_valid_odds(market_row.get("home_odds"))
        away_odds = _coerce_valid_odds(market_row.get("away_odds"))
        if home_odds is None or away_odds is None:
            return []
        probs = moneyline_probabilities(
            home_runs_mean=float(row["f5_home_pred"]),
            away_runs_mean=float(row["f5_away_pred"]),
            home_runs_std=float(companion_models["f5_home_runs_model"].rmse),
            away_runs_std=float(companion_models["f5_away_runs_model"].rmse),
        )
        if probs[0] is None or probs[1] is None:
            return []
        return _positive_ev_two_way_bets(
            game_pk=int(row["game_pk"]),
            market_type=market_type,
            home_probability=float(probs[0]),
            away_probability=float(probs[1]),
            home_odds=home_odds,
            away_odds=away_odds,
            home_point=None,
            away_point=None,
            book_name=str(market_row.get("book_name") or "consensus"),
        )

    if market_type == "f5_total":
        total_point = _coerce_finite_float(market_row.get("total_point"))
        over_odds = _coerce_valid_odds(market_row.get("over_odds"))
        under_odds = _coerce_valid_odds(market_row.get("under_odds"))
        if total_point is None or over_odds is None or under_odds is None:
            return []
        total_probs = totals_probabilities(
            home_runs_mean=float(row["f5_home_pred"]),
            away_runs_mean=float(row["f5_away_pred"]),
            home_runs_std=float(companion_models["f5_home_runs_model"].rmse),
            away_runs_std=float(companion_models["f5_away_runs_model"].rmse),
            total_point=total_point,
        )
        if total_probs[0] is None or total_probs[1] is None:
            return []
        return _positive_ev_total_bets(
            game_pk=int(row["game_pk"]),
            market_type=market_type,
            over_probability=float(total_probs[0]),
            under_probability=float(total_probs[1]),
            total_point=total_point,
            over_odds=over_odds,
            under_odds=under_odds,
            book_name=str(market_row.get("book_name") or "consensus"),
        )

    if market_type == "f5_rl":
        home_odds = _coerce_valid_odds(market_row.get("home_odds"))
        away_odds = _coerce_valid_odds(market_row.get("away_odds"))
        if home_odds is None or away_odds is None:
            return []
        home_point, away_point = normalize_runline_points(
            home_point=market_row.get("home_point"),
            away_point=market_row.get("away_point"),
            home_odds=home_odds,
            away_odds=away_odds,
        )
        if home_point is None or away_point is None:
            return []
        probs = spread_cover_probabilities(
            home_runs_mean=float(row["f5_home_pred"]),
            away_runs_mean=float(row["f5_away_pred"]),
            home_runs_std=float(companion_models["f5_home_runs_model"].rmse),
            away_runs_std=float(companion_models["f5_away_runs_model"].rmse),
            home_point=float(home_point),
        )
        if probs[0] is None or probs[1] is None:
            return []
        return _positive_ev_two_way_bets(
            game_pk=int(row["game_pk"]),
            market_type=market_type,
            home_probability=float(probs[0]),
            away_probability=float(probs[1]),
            home_odds=home_odds,
            away_odds=away_odds,
            home_point=float(home_point),
            away_point=float(away_point),
            book_name=str(market_row.get("book_name") or "consensus"),
        )

    return []


def _summarize_market_type_breakdown(bets_frame: pd.DataFrame) -> dict[str, dict[str, float | int | None]]:
    grouped: dict[str, dict[str, float | int | None]] = {}
    for market_type, group in bets_frame.groupby("market_type", sort=True):
        wins = int((group["bet_result"] == "WIN").sum())
        losses = int((group["bet_result"] == "LOSS").sum())
        pushes = int((group["bet_result"] == "PUSH").sum())
        graded = wins + losses
        total_staked = float(group["bet_stake_units"].sum())
        net_units = float(group["profit_units"].sum())
        grouped[str(market_type)] = {
            "bet_count": int(len(group)),
            "win_count": wins,
            "loss_count": losses,
            "push_count": pushes,
            "win_rate": float(wins / graded) if graded else None,
            "roi": float(net_units / total_staked) if total_staked else None,
            "net_units": net_units,
        }
    return grouped


def _summarize_bankroll_bets(
    *,
    bets_frame: pd.DataFrame,
    candidate_label: str,
    holdout_season: int,
    starting_bankroll_units: float,
    stage4_metadata: dict[str, Any],
) -> dict[str, Any]:
    if bets_frame.empty:
        return {
            "candidate_label": candidate_label,
            "holdout_season": int(holdout_season),
            "starting_bankroll_units": float(starting_bankroll_units),
            "ending_bankroll_units": float(starting_bankroll_units),
            "peak_bankroll_units": float(starting_bankroll_units),
            "bankroll_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "total_bets": 0,
            "win_count": 0,
            "loss_count": 0,
            "push_count": 0,
            "win_rate": None,
            "roi": None,
            "net_units": 0.0,
            "average_bet_size_units": None,
            "median_bet_size_units": None,
            "stage4_model_version": str(stage4_metadata["model_version"]),
            "stage4_mean_crps": float(stage4_metadata["distribution_metrics"]["mean_crps"]),
            "stage4_mean_negative_log_score": float(
                stage4_metadata["distribution_metrics"]["mean_negative_log_score"]
            ),
            "by_market_type": {},
        }

    total_bets = int(len(bets_frame))
    net_units = float(bets_frame["profit_units"].sum())
    total_staked = float(bets_frame["bet_stake_units"].sum())
    ending_bankroll_units = float(bets_frame["bankroll_after_units"].iloc[-1])
    peak_bankroll_units = float(bets_frame["peak_bankroll_units"].max())
    wins = int((bets_frame["bet_result"] == "WIN").sum())
    losses = int((bets_frame["bet_result"] == "LOSS").sum())
    pushes = int((bets_frame["bet_result"] == "PUSH").sum())
    graded_bets = wins + losses
    return {
        "candidate_label": candidate_label,
        "holdout_season": int(holdout_season),
        "starting_bankroll_units": float(starting_bankroll_units),
        "ending_bankroll_units": ending_bankroll_units,
        "peak_bankroll_units": peak_bankroll_units,
        "bankroll_return_pct": (
            float((ending_bankroll_units - float(starting_bankroll_units)) / float(starting_bankroll_units))
            if float(starting_bankroll_units) > 0
            else 0.0
        ),
        "max_drawdown_pct": float(bets_frame["bankroll_drawdown_pct"].max()),
        "total_bets": total_bets,
        "win_count": wins,
        "loss_count": losses,
        "push_count": pushes,
        "win_rate": float(wins / graded_bets) if graded_bets else None,
        "roi": float(net_units / total_staked) if total_staked else None,
        "net_units": net_units,
        "average_bet_size_units": float(bets_frame["bet_stake_units"].mean()),
        "median_bet_size_units": float(bets_frame["bet_stake_units"].median()),
        "stage4_model_version": str(stage4_metadata["model_version"]),
        "stage4_mean_crps": float(stage4_metadata["distribution_metrics"]["mean_crps"]),
        "stage4_mean_negative_log_score": float(
            stage4_metadata["distribution_metrics"]["mean_negative_log_score"]
        ),
        "by_market_type": _summarize_market_type_breakdown(bets_frame),
    }


def _build_combined_candidate_summary(
    *,
    candidate_label: str,
    season_results: Sequence[BankrollPlayoffSeasonResult],
    combined_bets_frame: pd.DataFrame,
    starting_bankroll_units: float,
) -> dict[str, Any]:
    if not season_results:
        raise ValueError("Candidate summary requires at least one season result")
    final_bankroll = float(season_results[-1].ending_bankroll_units)
    total_staked = float(combined_bets_frame["bet_stake_units"].sum()) if not combined_bets_frame.empty else 0.0
    net_units = float(combined_bets_frame["profit_units"].sum()) if not combined_bets_frame.empty else 0.0
    wins = int((combined_bets_frame["bet_result"] == "WIN").sum()) if not combined_bets_frame.empty else 0
    losses = int((combined_bets_frame["bet_result"] == "LOSS").sum()) if not combined_bets_frame.empty else 0
    pushes = int((combined_bets_frame["bet_result"] == "PUSH").sum()) if not combined_bets_frame.empty else 0
    graded = wins + losses
    serialized_season_results = []
    for result in season_results:
        payload = asdict(result)
        payload["report_json_path"] = str(result.report_json_path)
        payload["bets_csv_path"] = str(result.bets_csv_path)
        serialized_season_results.append(payload)
    return {
        "candidate_label": candidate_label,
        "holdout_seasons": [int(result.holdout_season) for result in season_results],
        "starting_bankroll_units": float(starting_bankroll_units),
        "ending_bankroll_units": final_bankroll,
        "bankroll_return_pct": (
            float((final_bankroll - float(starting_bankroll_units)) / float(starting_bankroll_units))
            if float(starting_bankroll_units) > 0
            else 0.0
        ),
        "peak_bankroll_units": max(float(result.peak_bankroll_units) for result in season_results),
        "max_drawdown_pct": (
            float(combined_bets_frame["bankroll_drawdown_pct"].max()) if not combined_bets_frame.empty else 0.0
        ),
        "total_bets": int(len(combined_bets_frame)),
        "win_count": wins,
        "loss_count": losses,
        "push_count": pushes,
        "win_rate": float(wins / graded) if graded else None,
        "roi": float(net_units / total_staked) if total_staked else None,
        "net_units": net_units,
        "average_bet_size_units": (
            float(combined_bets_frame["bet_stake_units"].mean()) if not combined_bets_frame.empty else None
        ),
        "median_bet_size_units": (
            float(combined_bets_frame["bet_stake_units"].median()) if not combined_bets_frame.empty else None
        ),
        "by_market_type": _summarize_market_type_breakdown(combined_bets_frame)
        if not combined_bets_frame.empty
        else {},
        "season_results": serialized_season_results,
    }


def _resolve_candidate_season_artifacts(
    *,
    candidate: BankrollPlayoffCandidateConfig,
    holdout_season: int,
    output_dir: Path,
    companion_cache: dict[tuple[Any, ...], RunCountTrainingResult],
    stage3_cache: dict[tuple[Any, ...], RunDistributionTrainingArtifact],
    companion_search_iterations: int,
    companion_optuna_workers: int,
    companion_early_stopping_rounds: int,
) -> _CandidateSeasonArtifacts:
    _log_progress(f"{candidate.label} {holdout_season}: resolving shared companion models")
    companion_key = (
        candidate.training_data_path,
        candidate.start_year,
        holdout_season,
        candidate.feature_selection_mode,
        candidate.forced_delta_count,
        candidate.xgb_workers,
        companion_search_iterations,
        companion_optuna_workers,
        companion_early_stopping_rounds,
    )
    companion_result = companion_cache.get(companion_key)
    if companion_result is None:
        _log_progress(f"{candidate.label} {holdout_season}: training shared companion models")
        training_dataset = _load_and_filter_training_data(
            training_data_path=candidate.training_data_path,
            start_year=candidate.start_year,
            end_year=holdout_season,
        )
        companion_output_dir = (
            output_dir
            / "_shared"
            / f"companions_{candidate.start_year}_{holdout_season}_{candidate.feature_selection_mode}_delta{candidate.forced_delta_count}"
        )
        companion_result = train_run_count_models(
            training_data=training_dataset,
            output_dir=companion_output_dir,
            holdout_season=holdout_season,
            search_iterations=int(companion_search_iterations),
            time_series_splits=int(candidate.folds),
            optuna_workers=int(companion_optuna_workers),
            early_stopping_rounds=int(companion_early_stopping_rounds),
            feature_selection_mode=str(candidate.feature_selection_mode),
            forced_delta_feature_count=int(candidate.forced_delta_count),
            model_specs=DEFAULT_RUN_COUNT_MODEL_SPECS,
        )
        companion_cache[companion_key] = companion_result

    _log_progress(f"{candidate.label} {holdout_season}: resolving shared Stage 3 artifact")
    stage3_key = (
        candidate.training_data_path,
        candidate.start_year,
        holdout_season,
        candidate.feature_selection_mode,
        candidate.forced_delta_count,
        candidate.mu_delta_mode,
        candidate.enable_market_priors,
        candidate.historical_odds_db_path,
        candidate.historical_market_book_name,
        candidate.xgb_workers,
    )
    stage3_artifact = stage3_cache.get(stage3_key)
    if stage3_artifact is None:
        _log_progress(f"{candidate.label} {holdout_season}: training shared Stage 3")
        training_dataset = _load_and_filter_training_data(
            training_data_path=candidate.training_data_path,
            start_year=candidate.start_year,
            end_year=holdout_season,
        )
        stage3_output_dir = (
            output_dir
            / "_shared"
            / f"stage3_{candidate.start_year}_{holdout_season}_{candidate.feature_selection_mode}_delta{candidate.forced_delta_count}_{candidate.mu_delta_mode}"
        )
        stage3_report_dir = output_dir / "_shared" / "stage3_reports"
        stage3_report_dir.mkdir(parents=True, exist_ok=True)
        stage3_artifact = train_run_distribution_model(
            training_data=training_dataset,
            output_dir=stage3_output_dir,
            mean_artifact_metadata_path=companion_result.models["full_game_away_runs_model"].metadata_path,
            holdout_season=holdout_season,
            feature_selection_mode=str(candidate.feature_selection_mode),
            forced_delta_feature_count=int(candidate.forced_delta_count),
            time_series_splits=int(candidate.folds),
            xgb_n_jobs=int(candidate.xgb_workers),
            distribution_report_dir=stage3_report_dir,
            enable_market_priors=bool(candidate.enable_market_priors),
            historical_odds_db_path=candidate.historical_odds_db_path,
            historical_market_book_name=candidate.historical_market_book_name,
            research_lane_name=f"{candidate.label}_bankroll_stage3",
            mu_delta_mode=str(candidate.mu_delta_mode),
        )
        stage3_cache[stage3_key] = stage3_artifact

    stage4_experiment = f"bankroll-playoff-{candidate.label}-{holdout_season}"
    stage4_model_dir = PROJECT_ROOT / "data" / "models" / stage4_experiment
    stage4_model_dir.mkdir(parents=True, exist_ok=True)
    stage4_report_dir = output_dir / candidate.label / str(holdout_season) / "mcmc"
    stage4_report_dir.mkdir(parents=True, exist_ok=True)
    existing_stage4_metadata = [
        path.resolve()
        for path in stage4_model_dir.glob("*.metadata.json")
        if path.is_file()
    ]
    if existing_stage4_metadata:
        stage4_metadata_path = max(existing_stage4_metadata, key=lambda path: path.stat().st_mtime_ns)
        stage4_metadata = json.loads(stage4_metadata_path.read_text(encoding="utf-8"))
        stage4_predictions_csv_path = _resolve_project_relative_path(
            Path(stage4_metadata["output_paths"]["predictions_csv"])
        )
        if stage4_predictions_csv_path.exists():
            _log_progress(f"{candidate.label} {holdout_season}: reusing existing Stage 4 artifact")
            return _CandidateSeasonArtifacts(
                companion_training_result=companion_result,
                stage3_artifact=stage3_artifact,
                stage4_metadata_path=stage4_metadata_path,
                stage4_metadata=stage4_metadata,
                stage4_predictions_csv_path=stage4_predictions_csv_path,
            )

    stage4_before = {path.resolve() for path in stage4_model_dir.glob("*.metadata.json")}
    stage4_command = [
        sys.executable,
        "scripts/run_mcmc_distribution.py",
        "--experiment",
        stage4_experiment,
        "--training-data",
        str(candidate.training_data_path),
        "--start",
        str(candidate.start_year),
        "--end",
        str(holdout_season),
        "--holdout",
        str(holdout_season),
        "--mean-artifact-metadata",
        str(companion_result.models["full_game_away_runs_model"].metadata_path),
        "--stage3-report-json",
        str(stage3_artifact.distribution_report_json_path),
        "--mcmc-report-dir",
        str(stage4_report_dir),
        "--distribution-report-dir",
        str(stage3_artifact.distribution_report_json_path.parent),
        "--simulations",
        str(candidate.simulations),
        "--starter-innings",
        str(candidate.starter_innings),
        "--seed",
        "20260328",
        "--research-lane-name",
        f"{candidate.label}_bankroll_stage4",
    ]
    if candidate.enable_market_priors:
        stage4_command.append("--enable-market-priors")
    if candidate.historical_odds_db_path:
        stage4_command.extend(["--historical-odds-db", str(candidate.historical_odds_db_path)])
    if candidate.historical_market_book_name:
        stage4_command.extend(["--historical-market-book", str(candidate.historical_market_book_name)])
    _log_progress(f"{candidate.label} {holdout_season}: starting Stage 4")
    subprocess.run(stage4_command, cwd=PROJECT_ROOT, check=True)
    stage4_metadata_candidates = [
        path.resolve()
        for path in stage4_model_dir.glob("*.metadata.json")
        if path.resolve() not in stage4_before
    ]
    if not stage4_metadata_candidates:
        stage4_metadata_candidates = [path.resolve() for path in stage4_model_dir.glob("*.metadata.json")]
    if not stage4_metadata_candidates:
        raise FileNotFoundError(f"No Stage 4 metadata found in {stage4_model_dir}")
    stage4_metadata_path = max(stage4_metadata_candidates, key=lambda path: path.stat().st_mtime_ns)
    stage4_metadata = json.loads(stage4_metadata_path.read_text(encoding="utf-8"))
    stage4_predictions_csv_path = _resolve_project_relative_path(
        Path(stage4_metadata["output_paths"]["predictions_csv"])
    )
    return _CandidateSeasonArtifacts(
        companion_training_result=companion_result,
        stage3_artifact=stage3_artifact,
        stage4_metadata_path=stage4_metadata_path,
        stage4_metadata=stage4_metadata,
        stage4_predictions_csv_path=stage4_predictions_csv_path,
    )


def _simulate_candidate_holdout_bankroll(
    *,
    candidate: BankrollPlayoffCandidateConfig,
    holdout_season: int,
    artifacts: _CandidateSeasonArtifacts,
    output_dir: Path,
    starting_bankroll_units: float,
    flat_bet_size_units: float,
    max_games: int | None,
) -> tuple[BankrollPlayoffSeasonResult, pd.DataFrame]:
    _log_progress(f"{candidate.label} {holdout_season}: loading holdout frame")
    holdout_frame = _load_and_filter_training_data(
        training_data_path=candidate.training_data_path,
        start_year=candidate.start_year,
        end_year=holdout_season,
    )
    holdout_frame = holdout_frame.loc[
        pd.to_numeric(holdout_frame["season"], errors="coerce") == int(holdout_season)
    ].copy()
    if holdout_frame.empty:
        raise ValueError(f"No holdout rows available for season {holdout_season}")
    holdout_frame = holdout_frame.sort_values(["game_date", "game_pk"]).reset_index(drop=True)
    if max_games is not None:
        holdout_frame = holdout_frame.head(int(max_games)).copy()

    companion_models = _load_companion_models(artifacts.companion_training_result)
    stage4_predictions = pd.read_csv(artifacts.stage4_predictions_csv_path)
    stage4_predictions = stage4_predictions.loc[
        stage4_predictions["game_pk"].isin(holdout_frame["game_pk"].tolist())
    ].copy()
    if stage4_predictions.empty:
        raise ValueError(
            f"No Stage 4 holdout predictions matched the selected holdout games for season {holdout_season}"
        )

    merged_frame = holdout_frame.merge(
        stage4_predictions[
            [
                "game_pk",
                "mcmc_expected_away_runs",
                "distribution_stddev",
            ]
        ],
        on="game_pk",
        how="inner",
    ).copy()
    if merged_frame.empty:
        raise ValueError(f"No bankroll rows matched Stage 4 predictions for season {holdout_season}")

    _log_progress(f"{candidate.label} {holdout_season}: generating companion predictions")
    merged_frame["full_game_home_pred"] = _predict_bundle(companion_models["full_game_home_runs_model"], merged_frame)
    merged_frame["f5_home_pred"] = _predict_bundle(companion_models["f5_home_runs_model"], merged_frame)
    merged_frame["f5_away_pred"] = _predict_bundle(companion_models["f5_away_runs_model"], merged_frame)
    merged_frame["full_game_away_std"] = pd.to_numeric(
        merged_frame["distribution_stddev"],
        errors="coerce",
    ).fillna(companion_models["full_game_away_runs_model"].rmse)

    historical_odds = _load_all_market_odds(candidate=candidate, games_frame=merged_frame)
    _log_progress(f"{candidate.label} {holdout_season}: settling bankroll across {len(merged_frame)} games")
    bets: list[dict[str, Any]] = []
    bankroll = float(starting_bankroll_units)
    peak_bankroll = float(starting_bankroll_units)

    for _, row in merged_frame.iterrows():
        game_market_rows = [
            market_row
            for market_frame in historical_odds.values()
            for market_row in market_frame.loc[market_frame["game_pk"] == int(row["game_pk"])].to_dict(orient="records")
        ]
        game_market_rows.sort(key=lambda payload: (str(payload.get("market_type")), str(payload.get("book_name") or "")))
        for market_row in game_market_rows:
            decisions = _build_market_bets_for_row(
                row=row,
                market_row=market_row,
                companion_models=companion_models,
            )
            for decision in decisions:
                bankroll_before = bankroll
                profit_units, result = _settle_market_bet(
                    market_type=str(decision["market_type"]),
                    side=str(decision["side"]),
                    line_at_bet=decision.get("line_at_bet"),
                    odds_at_bet=int(decision["odds_at_bet"]),
                    flat_bet_size_units=float(flat_bet_size_units),
                    full_game_home_score=int(row["final_home_score"]),
                    full_game_away_score=int(row["final_away_score"]),
                    f5_home_score=int(row["f5_home_score"]),
                    f5_away_score=int(row["f5_away_score"]),
                )
                bankroll = max(0.0, bankroll + profit_units)
                peak_bankroll = max(peak_bankroll, bankroll)
                decision.update(
                    {
                        "candidate_label": candidate.label,
                        "holdout_season": int(holdout_season),
                        "game_date": str(row["game_date"]),
                        "home_team": str(row["home_team"]),
                        "away_team": str(row["away_team"]),
                        "profit_units": float(profit_units),
                        "bet_result": result,
                        "bet_stake_units": float(flat_bet_size_units),
                        "bankroll_before_units": float(bankroll_before),
                        "bankroll_after_units": float(bankroll),
                        "peak_bankroll_units": float(peak_bankroll),
                        "bankroll_drawdown_pct": _calculate_drawdown_pct(bankroll, peak_bankroll),
                        "stage4_model_version": str(artifacts.stage4_metadata["model_version"]),
                    }
                )
                bets.append(decision)

    bets_frame = pd.DataFrame(bets)
    candidate_dir = output_dir / candidate.label / str(holdout_season)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    bets_csv_path = candidate_dir / "bankroll_bets.csv"
    report_json_path = candidate_dir / "bankroll_summary.json"
    if not bets_frame.empty:
        bets_frame = bets_frame.sort_values(["game_date", "game_pk", "market_type", "side"]).reset_index(drop=True)
        bets_frame.to_csv(bets_csv_path, index=False)

    season_summary = _summarize_bankroll_bets(
        bets_frame=bets_frame,
        candidate_label=candidate.label,
        holdout_season=holdout_season,
        starting_bankroll_units=float(starting_bankroll_units),
        stage4_metadata=artifacts.stage4_metadata,
    )
    report_json_path.write_text(
        json.dumps(season_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _log_progress(
        f"{candidate.label} {holdout_season}: complete "
        f"(bets={season_summary['total_bets']}, roi={season_summary['roi']}, "
        f"ending_bankroll={season_summary['ending_bankroll_units']:.3f})"
    )
    return (
        BankrollPlayoffSeasonResult(
            candidate_label=candidate.label,
            holdout_season=int(holdout_season),
            starting_bankroll_units=float(season_summary["starting_bankroll_units"]),
            ending_bankroll_units=float(season_summary["ending_bankroll_units"]),
            peak_bankroll_units=float(season_summary["peak_bankroll_units"]),
            bankroll_return_pct=float(season_summary["bankroll_return_pct"]),
            max_drawdown_pct=float(season_summary["max_drawdown_pct"]),
            total_bets=int(season_summary["total_bets"]),
            win_count=int(season_summary["win_count"]),
            loss_count=int(season_summary["loss_count"]),
            push_count=int(season_summary["push_count"]),
            win_rate=season_summary["win_rate"],
            roi=season_summary["roi"],
            net_units=float(season_summary["net_units"]),
            average_bet_size_units=season_summary["average_bet_size_units"],
            median_bet_size_units=season_summary["median_bet_size_units"],
            report_json_path=report_json_path,
            bets_csv_path=bets_csv_path,
            by_market_type=season_summary["by_market_type"],
            stage4_model_version=str(artifacts.stage4_metadata["model_version"]),
            stage4_mean_crps=float(artifacts.stage4_metadata["distribution_metrics"]["mean_crps"]),
            stage4_mean_negative_log_score=float(
                artifacts.stage4_metadata["distribution_metrics"]["mean_negative_log_score"]
            ),
        ),
        bets_frame,
    )


def run_bankroll_playoff(
    *,
    candidate_labels: Sequence[str] = DEFAULT_BANKROLL_CANDIDATE_LABELS,
    holdout_seasons: Sequence[int] = DEFAULT_BANKROLL_HOLDOUT_SEASONS,
    output_dir: str | Path = DEFAULT_BANKROLL_PLAYOFF_DIR,
    starting_bankroll_units: float = DEFAULT_STARTING_BANKROLL_UNITS,
    flat_bet_size_units: float = DEFAULT_FLAT_BET_SIZE_UNITS,
    max_games: int | None = None,
    companion_search_iterations: int = DEFAULT_RUN_COUNT_SEARCH_ITERATIONS,
    companion_optuna_workers: int = 3,
    companion_early_stopping_rounds: int = DEFAULT_RUN_COUNT_EARLY_STOPPING_ROUNDS,
) -> BankrollPlayoffResult:
    candidate_presets = default_bankroll_playoff_candidates()
    selected_candidates = [candidate_presets[label] for label in candidate_labels if label in candidate_presets]
    missing_labels = [label for label in candidate_labels if label not in candidate_presets]
    if missing_labels:
        raise ValueError(f"Unknown bankroll playoff candidates: {', '.join(sorted(missing_labels))}")
    if not selected_candidates:
        raise ValueError("At least one bankroll playoff candidate is required")

    resolved_output_dir = Path(output_dir)
    if not resolved_output_dir.is_absolute():
        resolved_output_dir = (PROJECT_ROOT / resolved_output_dir).resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    _log_progress(
        "starting playoff "
        f"(candidates={', '.join(candidate.label for candidate in selected_candidates)}; "
        f"holdout_seasons={', '.join(str(season) for season in holdout_seasons)})"
    )

    companion_cache: dict[tuple[Any, ...], RunCountTrainingResult] = {}
    stage3_cache: dict[tuple[Any, ...], RunDistributionTrainingArtifact] = {}
    candidate_results: list[BankrollPlayoffCandidateResult] = []

    for candidate in selected_candidates:
        _log_progress(f"candidate {candidate.label}: starting")
        season_results: list[BankrollPlayoffSeasonResult] = []
        combined_bets: list[pd.DataFrame] = []
        bankroll_before = float(starting_bankroll_units)
        for holdout_season in holdout_seasons:
            existing_result = _load_existing_season_result(
                candidate_label=candidate.label,
                holdout_season=int(holdout_season),
                output_dir=resolved_output_dir,
            )
            if existing_result is not None:
                season_result, bets_frame = existing_result
                _log_progress(
                    f"{candidate.label} {holdout_season}: reusing existing bankroll summary "
                    f"(bets={season_result.total_bets}, roi={season_result.roi})"
                )
                season_results.append(season_result)
                combined_bets.append(bets_frame)
                bankroll_before = season_result.ending_bankroll_units
                continue

            artifacts = _resolve_candidate_season_artifacts(
                candidate=candidate,
                holdout_season=int(holdout_season),
                output_dir=resolved_output_dir,
                companion_cache=companion_cache,
                stage3_cache=stage3_cache,
                companion_search_iterations=companion_search_iterations,
                companion_optuna_workers=companion_optuna_workers,
                companion_early_stopping_rounds=companion_early_stopping_rounds,
            )
            season_result, bets_frame = _simulate_candidate_holdout_bankroll(
                candidate=candidate,
                holdout_season=int(holdout_season),
                artifacts=artifacts,
                output_dir=resolved_output_dir,
                starting_bankroll_units=bankroll_before,
                flat_bet_size_units=flat_bet_size_units,
                max_games=max_games,
            )
            season_results.append(season_result)
            combined_bets.append(bets_frame)
            bankroll_before = season_result.ending_bankroll_units

        candidate_dir = resolved_output_dir / candidate.label
        candidate_dir.mkdir(parents=True, exist_ok=True)
        combined_frame = pd.concat(combined_bets, ignore_index=True) if combined_bets else pd.DataFrame()
        combined_bets_path = candidate_dir / "combined_bankroll_bets.csv"
        if not combined_frame.empty:
            combined_frame.to_csv(combined_bets_path, index=False)
        combined_summary = _build_combined_candidate_summary(
            candidate_label=candidate.label,
            season_results=season_results,
            combined_bets_frame=combined_frame,
            starting_bankroll_units=float(starting_bankroll_units),
        )
        combined_summary_path = candidate_dir / "combined_bankroll_summary.json"
        combined_summary_path.write_text(
            json.dumps(combined_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        _log_progress(
            f"candidate {candidate.label}: combined complete "
            f"(roi={combined_summary['roi']}, ending_bankroll={combined_summary['ending_bankroll_units']:.3f})"
        )
        candidate_results.append(
            BankrollPlayoffCandidateResult(
                candidate_label=candidate.label,
                seasons=tuple(season_results),
                combined_report_json_path=combined_summary_path,
                combined_bets_csv_path=combined_bets_path,
                combined_summary=combined_summary,
            )
        )

    summary_payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(resolved_output_dir),
        "starting_bankroll_units": float(starting_bankroll_units),
        "flat_bet_size_units": float(flat_bet_size_units),
        "holdout_seasons": [int(season) for season in holdout_seasons],
        "candidate_summaries": [result.combined_summary for result in candidate_results],
    }
    summary_path = resolved_output_dir / "bankroll_playoff_summary.json"
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _log_progress(f"playoff complete -> {summary_path}")
    return BankrollPlayoffResult(
        generated_at=summary_payload["generated_at"],
        output_dir=resolved_output_dir,
        starting_bankroll_units=float(starting_bankroll_units),
        flat_bet_size_units=float(flat_bet_size_units),
        holdout_seasons=tuple(int(season) for season in holdout_seasons),
        candidate_results=tuple(candidate_results),
        summary_json_path=summary_path,
    )


__all__ = [
    "BankrollPlayoffCandidateConfig",
    "BankrollPlayoffCandidateResult",
    "BankrollPlayoffResult",
    "BankrollPlayoffSeasonResult",
    "DEFAULT_BANKROLL_CANDIDATE_LABELS",
    "DEFAULT_BANKROLL_HOLDOUT_SEASONS",
    "DEFAULT_BANKROLL_PLAYOFF_DIR",
    "DEFAULT_FLAT_BET_SIZE_UNITS",
    "DEFAULT_HISTORICAL_ODDS_DB",
    "DEFAULT_STARTING_BANKROLL_UNITS",
    "default_bankroll_playoff_candidates",
    "normalize_runline_points",
    "run_bankroll_playoff",
]
