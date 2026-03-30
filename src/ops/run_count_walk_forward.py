from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from src.clients.historical_odds_client import load_historical_odds_for_games
from src.clients.odds_client import american_to_implied
from src.engine.edge_calculator import DEFAULT_EDGE_THRESHOLD, calculate_edge, payout_for_american_odds
from src.model.data_builder import validate_run_count_training_data
from src.model.run_count_trainer import _prepare_run_count_frame
from src.model.run_distribution_metrics import summarize_distribution_metrics
from src.model.run_distribution_trainer import (
    RunDistributionModel,
    build_variable_nb_support,
)
from src.model.run_research_features import augment_run_research_features
from src.model.xgboost_trainer import _load_training_dataframe


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_COUNT_WALK_FORWARD_DIR = Path("data/reports/run_count/walk_forward")
DEFAULT_TARGET_COLUMN = "final_away_score"
DEFAULT_WALK_FORWARD_BANKROLL_UNITS = 100.0


@dataclass(frozen=True, slots=True)
class RunCountWalkForwardReportPaths:
    json_path: Path
    csv_path: Path


def evaluate_stage3_holdout_walk_forward(
    *,
    training_data: pd.DataFrame | str | Path,
    model_metadata_path: str | Path,
    output_dir: str | Path = DEFAULT_RUN_COUNT_WALK_FORWARD_DIR,
    holdout_season: int = 2025,
    enable_market_priors: bool = False,
    historical_odds_db_path: str | Path | None = None,
    historical_market_book_name: str | None = None,
) -> tuple[dict[str, Any], RunCountWalkForwardReportPaths]:
    resolved_metadata_path = _resolve_path(model_metadata_path)
    metadata = json.loads(resolved_metadata_path.read_text(encoding="utf-8"))
    model = joblib.load(_metadata_to_model_path(resolved_metadata_path))
    if not isinstance(model, RunDistributionModel):
        raise TypeError(f"Expected RunDistributionModel at {resolved_metadata_path}")

    frame = _prepare_stage_frame(
        training_data=training_data,
        holdout_season=holdout_season,
        enable_market_priors=enable_market_priors,
        historical_odds_db_path=historical_odds_db_path,
        historical_market_book_name=historical_market_book_name,
    )
    actual = frame[DEFAULT_TARGET_COLUMN].astype(int).to_numpy()
    components = model.predict_components(frame)
    support = build_variable_nb_support(
        actual_counts=actual,
        mean_predictions=components["mu"],
        dispersion_size=components["dispersion"],
        tail_probability=1e-8,
    )
    pmf, _ = model.predict_pmf(frame, support=support)
    report = _build_report_payload(
        lane_key="best_distribution_lane",
        lane_kind="distribution",
        artifact_path=_relative_to_project(resolved_metadata_path),
        holdout_frame=frame,
        actual=actual,
        mean_predictions=np.asarray(components["mu"], dtype=float),
        pmf_matrix=pmf,
        support=np.asarray(support, dtype=int),
        market_anchor_coverage=float(frame.get("market_priors_available", pd.Series(dtype=float)).mean())
        if "market_priors_available" in frame.columns and not frame.empty
        else 0.0,
        historical_odds_db_path=historical_odds_db_path,
        historical_market_book_name=historical_market_book_name,
    )
    output_paths = _write_report(
        report,
        output_dir=output_dir,
        output_stem=f"{metadata['model_version']}.stage3_walk_forward",
    )
    return report, output_paths


def evaluate_mcmc_holdout_walk_forward(
    *,
    mcmc_metadata_path: str | Path,
    output_dir: str | Path = DEFAULT_RUN_COUNT_WALK_FORWARD_DIR,
    historical_odds_db_path: str | Path | None = None,
    historical_market_book_name: str | None = None,
) -> tuple[dict[str, Any], RunCountWalkForwardReportPaths]:
    resolved_metadata_path = _resolve_path(mcmc_metadata_path)
    metadata = json.loads(resolved_metadata_path.read_text(encoding="utf-8"))
    predictions_csv = _resolve_path(metadata["output_paths"]["predictions_csv"])
    predictions = pd.read_csv(predictions_csv)
    if predictions.empty:
        raise ValueError(f"MCMC holdout predictions are empty at {predictions_csv}")

    pmf_entries = predictions["away_run_pmf_json"].map(json.loads)
    support_max = max(_max_run_in_pmf(pmf) for pmf in pmf_entries)
    support = np.arange(support_max + 1, dtype=int)
    pmf_matrix = np.vstack([_pmf_list_to_vector(pmf, support_max=support_max) for pmf in pmf_entries])
    mean_predictions = np.asarray(
        [sum(float(item["probability"]) * int(item["runs"]) for item in pmf) for pmf in pmf_entries],
        dtype=float,
    )
    actual = predictions["actual_away_runs"].astype(int).to_numpy()
    holdout_frame = pd.DataFrame(
        {
            "game_pk": predictions["game_pk"] if "game_pk" in predictions.columns else pd.Series(dtype=int),
            "game_date": predictions["game_date"],
            "scheduled_start": predictions["game_date"],
            "home_team": predictions["home_team"] if "home_team" in predictions.columns else pd.Series(dtype=str),
            "away_team": predictions["away_team"] if "away_team" in predictions.columns else pd.Series(dtype=str),
            "market_priors_available": (
                predictions["market_priors_available"]
                if "market_priors_available" in predictions.columns
                else pd.Series(np.zeros(len(predictions), dtype=float))
            ),
            "market_anchor_confidence": (
                predictions["market_anchor_confidence"]
                if "market_anchor_confidence" in predictions.columns
                else pd.Series(np.zeros(len(predictions), dtype=float))
            ),
            "market_implied_full_game_away_runs": (
                predictions["market_implied_away_runs"]
                if "market_implied_away_runs" in predictions.columns
                else pd.Series(np.full(len(predictions), np.nan, dtype=float))
            ),
            "fallback_applied": (
                predictions["fallback_applied"]
                if "fallback_applied" in predictions.columns
                else pd.Series(np.zeros(len(predictions), dtype=bool))
            ),
            "mean_anchor_applied": (
                predictions["mean_anchor_applied"]
                if "mean_anchor_applied" in predictions.columns
                else pd.Series(np.zeros(len(predictions), dtype=bool))
            ),
            "mean_drift_vs_control": (
                predictions["mean_drift_vs_control"]
                if "mean_drift_vs_control" in predictions.columns
                else pd.Series(np.zeros(len(predictions), dtype=float))
            ),
            "post_anchor_implied_mean_runs": (
                predictions["post_anchor_implied_mean_runs"]
                if "post_anchor_implied_mean_runs" in predictions.columns
                else pd.Series(np.full(len(predictions), np.nan, dtype=float))
            ),
        }
    )
    report = _build_report_payload(
        lane_key="best_mcmc_lane",
        lane_kind="mcmc",
        artifact_path=_relative_to_project(resolved_metadata_path),
        holdout_frame=holdout_frame,
        actual=actual,
        mean_predictions=mean_predictions,
        pmf_matrix=pmf_matrix,
        support=support,
        market_anchor_coverage=0.0,
        historical_odds_db_path=historical_odds_db_path,
        historical_market_book_name=historical_market_book_name,
    )
    output_paths = _write_report(
        report,
        output_dir=output_dir,
        output_stem=f"{metadata['model_version']}.mcmc_walk_forward",
    )
    return report, output_paths


def _prepare_stage_frame(
    *,
    training_data: pd.DataFrame | str | Path,
    holdout_season: int,
    enable_market_priors: bool,
    historical_odds_db_path: str | Path | None,
    historical_market_book_name: str | None,
) -> pd.DataFrame:
    validated = validate_run_count_training_data(training_data)
    dataset = _load_training_dataframe(validated)
    frame = _prepare_run_count_frame(dataset, target_column=DEFAULT_TARGET_COLUMN)
    frame = augment_run_research_features(
        frame,
        enable_market_priors=enable_market_priors,
        historical_odds_db_path=historical_odds_db_path,
        historical_market_book_name=historical_market_book_name,
    ).dataframe
    holdout = frame.loc[frame["season"] == int(holdout_season)].copy()
    if holdout.empty:
        raise ValueError(f"No holdout rows found for season {holdout_season}")
    return holdout


def _build_report_payload(
    *,
    lane_key: str,
    lane_kind: str,
    artifact_path: str,
    holdout_frame: pd.DataFrame,
    actual: np.ndarray,
    mean_predictions: np.ndarray,
    pmf_matrix: np.ndarray,
    support: np.ndarray,
    market_anchor_coverage: float,
    historical_odds_db_path: str | Path | None,
    historical_market_book_name: str | None,
) -> dict[str, Any]:
    scheduled = _resolve_scheduled_series(holdout_frame)
    month_labels = scheduled.dt.to_period("M").astype(str)
    month_rows: list[dict[str, Any]] = []
    for month_label in sorted(month_labels.unique()):
        month_mask = month_labels == month_label
        month_actual = actual[month_mask.to_numpy()]
        month_predictions = mean_predictions[month_mask.to_numpy()]
        month_pmf = pmf_matrix[month_mask.to_numpy()]
        metrics = summarize_distribution_metrics(month_actual, month_pmf, support)
        month_rows.append(
            {
                "period": month_label,
                "game_count": int(month_mask.sum()),
                "mean_crps": float(metrics["mean_crps"]),
                "mean_negative_log_score": float(metrics["mean_negative_log_score"]),
                "rmse": float(np.sqrt(np.mean((month_predictions - month_actual) ** 2))),
                "actual_mean_runs": float(np.mean(month_actual)),
                "predicted_mean_runs": float(np.mean(month_predictions)),
                "shutout_abs_error": float(metrics["zero_calibration"]["p_0"]["absolute_error"]),
                "p_ge_5_abs_error": float(metrics["tail_calibration"]["p_ge_5"]["absolute_error"]),
            }
        )

    aggregate_metrics = summarize_distribution_metrics(actual, pmf_matrix, support)
    betting_evidence = _build_betting_evidence(
        holdout_frame=holdout_frame,
        actual=actual,
        pmf_matrix=pmf_matrix,
        support=support,
        market_anchor_coverage=market_anchor_coverage,
        historical_odds_db_path=historical_odds_db_path,
        historical_market_book_name=historical_market_book_name,
    )
    proxy_market_decision_evidence = _build_proxy_market_decision_evidence(
        holdout_frame=holdout_frame,
        actual=actual,
        mean_predictions=mean_predictions,
        pmf_matrix=pmf_matrix,
        support=support,
    )
    operational_diagnostics = _build_operational_diagnostics(
        holdout_frame=holdout_frame,
        mean_predictions=mean_predictions,
    )
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "lane_key": lane_key,
        "lane_kind": lane_kind,
        "artifact_path": artifact_path,
        "evaluation_mode": "holdout_month_walk_forward",
        "holdout_season": int(pd.to_datetime(scheduled, errors="coerce").dt.year.mode().iloc[0]),
        "month_summaries": month_rows,
        "aggregate_mean_metrics": {
            "rmse": float(np.sqrt(np.mean((mean_predictions - actual) ** 2))),
            "actual_mean_runs": float(np.mean(actual)),
            "predicted_mean_runs": float(np.mean(mean_predictions)),
        },
        "aggregate_distribution_metrics": aggregate_metrics,
        "betting_evidence": betting_evidence,
        "proxy_market_decision_evidence": proxy_market_decision_evidence,
        "operational_diagnostics": operational_diagnostics,
    }
    return report


def _build_betting_evidence(
    *,
    holdout_frame: pd.DataFrame,
    actual: np.ndarray,
    pmf_matrix: np.ndarray,
    support: np.ndarray,
    market_anchor_coverage: float,
    historical_odds_db_path: str | Path | None,
    historical_market_book_name: str | None,
) -> dict[str, Any]:
    default_payload = {
        "available": False,
        "passed": None,
        "bet_count": 0,
        "roi": None,
        "net_units": None,
        "max_drawdown": None,
        "market_anchor_coverage": float(market_anchor_coverage),
        "market_type": "full_game_team_total_away",
        "away_team_total_coverage": 0.0,
        "away_team_total_closing_coverage": 0.0,
        "full_game_total_coverage": 0.0,
        "f5_total_coverage": 0.0,
        "full_game_total_closing_coverage": 0.0,
        "f5_total_closing_coverage": 0.0,
        "clv_supported": False,
        "clv_basis": None,
        "clv_coverage": 0.0,
        "clv_bet_count": 0,
        "average_clv": None,
        "positive_clv_rate": None,
        "bet_side_counts": {"over": 0, "under": 0},
        "bet_win_rate": None,
        "historical_market_source": None,
        "reason": "Historical away-run totals or away team totals are not available in the current repo-local data.",
    }
    if historical_odds_db_path is None or holdout_frame.empty:
        return default_payload

    opening_away_team_totals = load_historical_odds_for_games(
        db_path=historical_odds_db_path,
        games_frame=holdout_frame,
        market_type="full_game_team_total_away",
        book_name=historical_market_book_name,
        snapshot_selection="opening",
    )
    closing_away_team_totals = load_historical_odds_for_games(
        db_path=historical_odds_db_path,
        games_frame=holdout_frame,
        market_type="full_game_team_total_away",
        book_name=historical_market_book_name,
        snapshot_selection="latest",
    )
    full_game_totals = load_historical_odds_for_games(
        db_path=historical_odds_db_path,
        games_frame=holdout_frame,
        market_type="full_game_total",
        book_name=historical_market_book_name,
        snapshot_selection="opening",
    )
    f5_totals = load_historical_odds_for_games(
        db_path=historical_odds_db_path,
        games_frame=holdout_frame,
        market_type="f5_total",
        book_name=historical_market_book_name,
        snapshot_selection="opening",
    )
    full_game_totals_closing = load_historical_odds_for_games(
        db_path=historical_odds_db_path,
        games_frame=holdout_frame,
        market_type="full_game_total",
        book_name=historical_market_book_name,
        snapshot_selection="latest",
    )
    f5_totals_closing = load_historical_odds_for_games(
        db_path=historical_odds_db_path,
        games_frame=holdout_frame,
        market_type="f5_total",
        book_name=historical_market_book_name,
        snapshot_selection="latest",
    )
    if opening_away_team_totals.empty and full_game_totals.empty and f5_totals.empty:
        default_payload["reason"] = (
            "Historical market archive was supplied but no away-team-total or companion total-market rows matched the holdout frame."
        )
        return default_payload

    denominator = max(1, len(holdout_frame))
    source_schemas = sorted(
        {
            str(frame["source_schema"].iloc[0])
            for frame in (opening_away_team_totals, closing_away_team_totals, full_game_totals, f5_totals)
            if not frame.empty and "source_schema" in frame.columns
        }
    )
    payload = {
        **default_payload,
        "away_team_total_coverage": float(
            len(opening_away_team_totals["game_pk"].unique()) / denominator
            if "game_pk" in opening_away_team_totals.columns
            else 0.0
        ),
        "away_team_total_closing_coverage": float(
            len(closing_away_team_totals["game_pk"].unique()) / denominator
            if "game_pk" in closing_away_team_totals.columns
            else 0.0
        ),
        "full_game_total_coverage": float(
            len(full_game_totals["game_pk"].unique()) / denominator if "game_pk" in full_game_totals.columns else 0.0
        ),
        "f5_total_coverage": float(
            len(f5_totals["game_pk"].unique()) / denominator if "game_pk" in f5_totals.columns else 0.0
        ),
        "full_game_total_closing_coverage": float(
            len(full_game_totals_closing["game_pk"].unique()) / denominator
            if "game_pk" in full_game_totals_closing.columns
            else 0.0
        ),
        "f5_total_closing_coverage": float(
            len(f5_totals_closing["game_pk"].unique()) / denominator
            if "game_pk" in f5_totals_closing.columns
            else 0.0
        ),
        "clv_basis": "opening_vs_closing",
        "historical_market_source": _market_source_name_from_schemas(source_schemas),
    }
    if opening_away_team_totals.empty:
        payload["reason"] = (
            "Historical totals matched the holdout frame, but direct away-team-total prices are still unavailable; "
            "promotion betting evidence and per-bet CLV remain blocked."
        )
        return payload
    return {
        **payload,
        **_summarize_away_team_total_bets(
            holdout_frame=holdout_frame,
            actual=actual,
            pmf_matrix=pmf_matrix,
            support=support,
            opening_market=opening_away_team_totals,
            closing_market=closing_away_team_totals,
            historical_market_book_name=historical_market_book_name,
        ),
    }


def _build_proxy_market_decision_evidence(
    *,
    holdout_frame: pd.DataFrame,
    actual: np.ndarray,
    mean_predictions: np.ndarray,
    pmf_matrix: np.ndarray,
    support: np.ndarray,
) -> dict[str, Any]:
    if "market_implied_full_game_away_runs" not in holdout_frame.columns or holdout_frame.empty:
        return {
            "available": False,
            "proxy_only": True,
            "coverage": 0.0,
            "reason": "Holdout frame does not include a market-implied away-run anchor.",
        }

    market_implied = pd.to_numeric(
        holdout_frame["market_implied_full_game_away_runs"],
        errors="coerce",
    )
    available_mask = market_implied.notna().to_numpy(dtype=bool)
    if "market_priors_available" in holdout_frame.columns:
        market_available = (
            pd.to_numeric(holdout_frame["market_priors_available"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
            > 0.5
        )
        available_mask = available_mask & market_available
    coverage = float(np.mean(available_mask)) if len(holdout_frame) else 0.0
    if not np.any(available_mask):
        return {
            "available": False,
            "proxy_only": True,
            "coverage": coverage,
            "reason": "No holdout rows contain a market-implied away-run anchor.",
        }

    market_values = market_implied.fillna(0.0).to_numpy(dtype=float)
    model_gap = mean_predictions - market_values
    actual_gap = actual.astype(float) - market_values
    threshold_summaries: dict[str, Any] = {}
    for threshold in (0.5, 0.75, 1.0):
        threshold_key = f"abs_gap_ge_{str(threshold).replace('.', '_')}"
        over_mask = available_mask & (model_gap >= threshold)
        under_mask = available_mask & (model_gap <= -threshold)
        threshold_summaries[threshold_key] = {
            "over": _summarize_proxy_market_slice(
                mask=over_mask,
                direction="over",
                actual=actual,
                mean_predictions=mean_predictions,
                pmf_matrix=pmf_matrix,
                support=support,
                model_gap=model_gap,
                actual_gap=actual_gap,
            ),
            "under": _summarize_proxy_market_slice(
                mask=under_mask,
                direction="under",
                actual=actual,
                mean_predictions=mean_predictions,
                pmf_matrix=pmf_matrix,
                support=support,
                model_gap=model_gap,
                actual_gap=actual_gap,
            ),
        }

    high_gap_mask = available_mask & (np.abs(model_gap) >= 0.75)
    direct_away_team_total_available = (
        "market_full_game_away_team_total_available" in holdout_frame.columns
        and pd.to_numeric(
            holdout_frame["market_full_game_away_team_total_available"],
            errors="coerce",
        )
        .fillna(0.0)
        .gt(0.5)
        .any()
    )
    return {
        "available": True,
        "proxy_only": not direct_away_team_total_available,
        "coverage": coverage,
        "market_anchor_type": (
            "direct_away_team_total_line"
            if direct_away_team_total_available
            else "derived_away_runs_from_full_game_total_and_side_markets"
        ),
        "mean_model_minus_market": float(np.mean(model_gap[available_mask])),
        "mean_abs_model_minus_market": float(np.mean(np.abs(model_gap[available_mask]))),
        "mean_actual_minus_market": float(np.mean(actual_gap[available_mask])),
        "high_gap_sign_agreement": (
            float(np.mean(np.sign(model_gap[high_gap_mask]) == np.sign(actual_gap[high_gap_mask])))
            if np.any(high_gap_mask)
            else None
        ),
        "threshold_summaries": threshold_summaries,
        "reason": (
            "This is proxy decision evidence against the direct away-team-total anchor."
            if direct_away_team_total_available
            else "This is proxy decision evidence against the derived away-run anchor. Direct away-team-total pricing is still missing."
        ),
    }


def _summarize_proxy_market_slice(
    *,
    mask: np.ndarray,
    direction: str,
    actual: np.ndarray,
    mean_predictions: np.ndarray,
    pmf_matrix: np.ndarray,
    support: np.ndarray,
    model_gap: np.ndarray,
    actual_gap: np.ndarray,
) -> dict[str, float | int | None]:
    if not np.any(mask):
        return {
            "row_count": 0,
            "mean_model_minus_market": None,
            "mean_actual_minus_market": None,
            "hit_rate": None,
            "rmse": None,
            "mean_crps": None,
            "mean_negative_log_score": None,
        }
    metrics = summarize_distribution_metrics(actual[mask], pmf_matrix[mask], support)
    hit_rate = (
        float(np.mean(actual_gap[mask] > 0.0))
        if direction == "over"
        else float(np.mean(actual_gap[mask] < 0.0))
    )
    return {
        "row_count": int(np.count_nonzero(mask)),
        "mean_model_minus_market": float(np.mean(model_gap[mask])),
        "mean_actual_minus_market": float(np.mean(actual_gap[mask])),
        "hit_rate": hit_rate,
        "rmse": float(np.sqrt(np.mean((mean_predictions[mask] - actual[mask]) ** 2))),
        "mean_crps": float(metrics["mean_crps"]),
        "mean_negative_log_score": float(metrics["mean_negative_log_score"]),
    }


def _build_operational_diagnostics(
    *,
    holdout_frame: pd.DataFrame,
    mean_predictions: np.ndarray,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    if "fallback_applied" in holdout_frame.columns:
        fallback = holdout_frame["fallback_applied"].astype(bool).to_numpy(dtype=bool)
        diagnostics["fallback_rate"] = float(np.mean(fallback)) if len(fallback) else 0.0
    if "mean_anchor_applied" in holdout_frame.columns:
        anchored = holdout_frame["mean_anchor_applied"].astype(bool).to_numpy(dtype=bool)
        diagnostics["mean_anchor_rate"] = float(np.mean(anchored)) if len(anchored) else 0.0
    if "mean_drift_vs_control" in holdout_frame.columns:
        drift = pd.to_numeric(holdout_frame["mean_drift_vs_control"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        diagnostics["mean_drift_vs_control"] = {
            "mean": float(np.mean(drift)),
            "mean_abs": float(np.mean(np.abs(drift))),
            "max_abs": float(np.max(np.abs(drift))) if len(drift) else 0.0,
        }
    if "post_anchor_implied_mean_runs" in holdout_frame.columns:
        post_anchor = pd.to_numeric(
            holdout_frame["post_anchor_implied_mean_runs"],
            errors="coerce",
        )
        diagnostics["post_anchor_implied_mean_runs"] = {
            "non_null_rows": int(post_anchor.notna().sum()),
            "mean": float(post_anchor.dropna().mean()) if post_anchor.notna().any() else None,
        }
    diagnostics["predicted_mean_runs"] = {
        "mean": float(np.mean(mean_predictions)),
        "std": float(np.std(mean_predictions)),
    }
    return diagnostics


def _summarize_away_team_total_bets(
    *,
    holdout_frame: pd.DataFrame,
    actual: np.ndarray,
    pmf_matrix: np.ndarray,
    support: np.ndarray,
    opening_market: pd.DataFrame,
    closing_market: pd.DataFrame,
    historical_market_book_name: str | None,
) -> dict[str, Any]:
    holdout = holdout_frame.reset_index(drop=True).copy()
    holdout["row_index"] = np.arange(len(holdout), dtype=int)
    opening = opening_market.rename(
        columns={
            "book_name": "opening_book_name",
            "total_point": "opening_line",
            "over_odds": "opening_over_odds",
            "under_odds": "opening_under_odds",
        }
    )
    closing = closing_market.rename(
        columns={
            "book_name": "closing_book_name",
            "total_point": "closing_line",
            "over_odds": "closing_over_odds",
            "under_odds": "closing_under_odds",
        }
    )
    merged = holdout.merge(
        opening[
            [
                "game_pk",
                "opening_book_name",
                "opening_line",
                "opening_over_odds",
                "opening_under_odds",
            ]
        ],
        on="game_pk",
        how="inner",
    )
    if merged.empty:
        return {
            "available": False,
            "clv_supported": False,
            "reason": "Direct away-team-total rows were found historically, but none matched the holdout rows during scoring.",
        }
    merged = merged.merge(
        closing[
            [
                "game_pk",
                "closing_book_name",
                "closing_line",
                "closing_over_odds",
                "closing_under_odds",
            ]
        ],
        on="game_pk",
        how="left",
    )

    bets: list[dict[str, Any]] = []
    for row in merged.itertuples(index=False):
        line = _safe_float(row.opening_line)
        over_odds = _safe_int(row.opening_over_odds)
        under_odds = _safe_int(row.opening_under_odds)
        if line is None or over_odds is None or under_odds is None:
            continue

        row_index = int(row.row_index)
        over_decision = calculate_edge(
            game_pk=int(row.game_pk),
            market_type="full_game_team_total_away",
            side="over",
            model_probability=_total_side_model_probability(
                pmf_row=pmf_matrix[row_index],
                support=support,
                line=line,
                side="over",
            ),
            home_odds=over_odds,
            away_odds=under_odds,
            home_point=line,
            away_point=line,
            book_name=str(row.opening_book_name or historical_market_book_name or "historical"),
        )
        under_decision = calculate_edge(
            game_pk=int(row.game_pk),
            market_type="full_game_team_total_away",
            side="under",
            model_probability=_total_side_model_probability(
                pmf_row=pmf_matrix[row_index],
                support=support,
                line=line,
                side="under",
            ),
            home_odds=over_odds,
            away_odds=under_odds,
            home_point=line,
            away_point=line,
            book_name=str(row.opening_book_name or historical_market_book_name or "historical"),
        )
        candidate = max((over_decision, under_decision), key=lambda decision: float(decision.edge_pct))
        if not candidate.is_positive_ev:
            continue

        result = _settle_total_side(
            actual_runs=int(actual[row_index]),
            line=line,
            side=str(candidate.side),
        )
        opening_implied_probability = float(american_to_implied(int(candidate.odds_at_bet)))
        closing_odds = _closing_odds_for_side(
            closing_over_odds=row.closing_over_odds,
            closing_under_odds=row.closing_under_odds,
            side=str(candidate.side),
        )
        bets.append(
            {
                "game_pk": int(row.game_pk),
                "scheduled_start": row.scheduled_start if hasattr(row, "scheduled_start") else row.game_date,
                "side": str(candidate.side),
                "edge_pct": float(candidate.edge_pct),
                "profit_units": _profit_units_for_total_bet(result=result, odds=int(candidate.odds_at_bet)),
                "result": result,
                "clv": (
                    float(american_to_implied(int(closing_odds)) - opening_implied_probability)
                    if closing_odds is not None
                    else None
                ),
            }
        )

    if not bets:
        return {
            "available": True,
            "bet_count": 0,
            "roi": None,
            "net_units": 0.0,
            "max_drawdown": 0.0,
            "clv_supported": False,
            "clv_coverage": 0.0,
            "clv_bet_count": 0,
            "average_clv": None,
            "positive_clv_rate": None,
            "bet_side_counts": {"over": 0, "under": 0},
            "bet_win_rate": None,
            "reason": (
                f"Direct away-team-total opening markets matched the holdout, but no side cleared the "
                f"{DEFAULT_EDGE_THRESHOLD:.2%} edge threshold."
            ),
        }

    bets_frame = pd.DataFrame(bets).sort_values(["scheduled_start", "game_pk"]).reset_index(drop=True)
    bankroll = DEFAULT_WALK_FORWARD_BANKROLL_UNITS + bets_frame["profit_units"].cumsum()
    peak_bankroll = bankroll.cummax()
    drawdown = ((bankroll - peak_bankroll) / peak_bankroll).fillna(0.0)
    clv_mask = bets_frame["clv"].notna()
    bet_count = int(len(bets_frame))
    net_units = float(bets_frame["profit_units"].sum())
    return {
        "available": True,
        "bet_count": bet_count,
        "roi": float(net_units / bet_count) if bet_count else None,
        "net_units": net_units,
        "max_drawdown": float(drawdown.min()) if not drawdown.empty else 0.0,
        "clv_supported": bool(clv_mask.any()),
        "clv_coverage": float(clv_mask.mean()) if bet_count else 0.0,
        "clv_bet_count": int(clv_mask.sum()),
        "average_clv": float(bets_frame.loc[clv_mask, "clv"].mean()) if clv_mask.any() else None,
        "positive_clv_rate": float((bets_frame.loc[clv_mask, "clv"] > 0.0).mean()) if clv_mask.any() else None,
        "bet_side_counts": {
            "over": int((bets_frame["side"] == "over").sum()),
            "under": int((bets_frame["side"] == "under").sum()),
        },
        "bet_win_rate": float((bets_frame["result"] == "WIN").mean()),
        "reason": "Walk-forward betting evidence uses direct away-team-total opening prices and same-market closing CLV.",
    }


def _total_side_model_probability(
    *,
    pmf_row: np.ndarray,
    support: np.ndarray,
    line: float,
    side: str,
) -> float:
    if side == "over":
        return float(pmf_row[support > float(line)].sum())
    if side == "under":
        return float(pmf_row[support < float(line)].sum())
    raise ValueError(f"Unsupported total side: {side}")


def _settle_total_side(*, actual_runs: int, line: float, side: str) -> str:
    if side == "over":
        if actual_runs > line:
            return "WIN"
        if actual_runs < line:
            return "LOSS"
        return "PUSH"
    if side == "under":
        if actual_runs < line:
            return "WIN"
        if actual_runs > line:
            return "LOSS"
        return "PUSH"
    raise ValueError(f"Unsupported total side: {side}")


def _profit_units_for_total_bet(*, result: str, odds: int) -> float:
    if result == "WIN":
        return float(payout_for_american_odds(odds))
    if result == "LOSS":
        return -1.0
    return 0.0


def _closing_odds_for_side(
    *,
    closing_over_odds: Any,
    closing_under_odds: Any,
    side: str,
) -> int | None:
    value = closing_over_odds if side == "over" else closing_under_odds
    return _safe_int(value)


def _write_report(
    report: Mapping[str, Any],
    *,
    output_dir: str | Path,
    output_stem: str,
) -> RunCountWalkForwardReportPaths:
    resolved_output_dir = _resolve_path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    json_path = resolved_output_dir / f"{output_stem}.json"
    csv_path = resolved_output_dir / f"{output_stem}.csv"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "period",
                "game_count",
                "mean_crps",
                "mean_negative_log_score",
                "rmse",
                "actual_mean_runs",
                "predicted_mean_runs",
                "shutout_abs_error",
                "p_ge_5_abs_error",
            ],
        )
        writer.writeheader()
        for row in report.get("month_summaries", []):
            writer.writerow(row)
    return RunCountWalkForwardReportPaths(json_path=json_path, csv_path=csv_path)


def _resolve_scheduled_series(holdout_frame: pd.DataFrame) -> pd.Series:
    if "scheduled_start" in holdout_frame.columns:
        resolved = pd.to_datetime(holdout_frame["scheduled_start"], utc=True, errors="coerce")
    else:
        resolved = pd.to_datetime(holdout_frame["game_date"], utc=True, errors="coerce")
    return resolved.dt.tz_convert(None)


def _pmf_list_to_vector(pmf: Sequence[Mapping[str, Any]], *, support_max: int) -> np.ndarray:
    vector = np.zeros(support_max + 1, dtype=float)
    for item in pmf:
        run_count = int(item["runs"])
        if run_count <= support_max:
            vector[run_count] = float(item["probability"])
    total = float(vector.sum())
    if total <= 0.0:
        vector[0] = 1.0
        return vector
    return vector / total


def _max_run_in_pmf(pmf: Sequence[Mapping[str, Any]]) -> int:
    if not pmf:
        return 0
    return max(int(item["runs"]) for item in pmf)


def _metadata_to_model_path(metadata_path: Path) -> Path:
    suffix = ".metadata.json"
    if not metadata_path.name.endswith(suffix):
        raise ValueError(f"Expected metadata path, got {metadata_path}")
    return metadata_path.with_name(metadata_path.name[: -len(suffix)] + ".joblib")


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _relative_to_project(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def _market_source_name_from_schemas(source_schemas: Sequence[str]) -> str | None:
    resolved = tuple(sorted({str(schema).strip() for schema in source_schemas if str(schema).strip()}))
    if not resolved:
        return None
    if resolved == ("canonical",):
        return "historical_market_archive_canonical"
    if resolved == ("old_scraper",):
        return "historical_market_archive_old_scraper"
    return "historical_market_archive_hybrid"


def _safe_float(value: Any) -> float | None:
    if value is None or bool(pd.isna(value)):
        return None
    return float(value)


def _safe_int(value: Any) -> int | None:
    resolved = _safe_float(value)
    return None if resolved is None else int(resolved)


__all__ = [
    "DEFAULT_RUN_COUNT_WALK_FORWARD_DIR",
    "RunCountWalkForwardReportPaths",
    "evaluate_mcmc_holdout_walk_forward",
    "evaluate_stage3_holdout_walk_forward",
]
