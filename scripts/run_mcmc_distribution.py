from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from rich.console import Console


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.data_builder import inspect_run_count_training_data, validate_run_count_training_data  # noqa: E402
from src.model.mcmc_engine import DEFAULT_SIMULATION_COUNT, pad_probability_vector, simulate_away_game_distribution  # noqa: E402
from src.model.mcmc_feature_builder import build_mcmc_feature_bundle  # noqa: E402
from src.model.mcmc_pricing import (  # noqa: E402
    build_distribution_comparison,
    flatten_mcmc_report_row,
    summarize_away_run_distribution,
)
from src.model.run_research_features import (  # noqa: E402
    augment_run_research_features,
    metadata_to_dict as research_metadata_to_dict,
)
from src.model.run_count_trainer import _compute_holdout_metrics, _prepare_run_count_frame  # noqa: E402
from src.model.run_distribution_metrics import summarize_distribution_metrics  # noqa: E402
from src.model.run_distribution_trainer import (  # noqa: E402
    DEFAULT_CURRENT_CONTROL_PATH,
    DEFAULT_DISTRIBUTION_REPORT_DIR,
    DEFAULT_TARGET_COLUMN,
    evaluate_control_distribution_baseline,
    load_mean_head_reference,
    resolve_mean_artifact_metadata_path,
)
from src.model.xgboost_trainer import _build_model_version, _load_training_dataframe, _resolve_data_version_hash  # noqa: E402


console = Console()
DEFAULT_TRAINING_DATA = Path("data/training/ParquetDefault.parquet")
DEFAULT_MCMC_REPORT_DIR = Path("data/reports/run_count/mcmc")
DEFAULT_MODEL_NAME = "full_game_away_runs_mcmc_model"


def _filter_training_data_by_season(
    dataset: pd.DataFrame,
    *,
    start_year: int | None,
    end_year: int | None,
) -> pd.DataFrame:
    if start_year is None and end_year is None:
        return dataset

    filtered = dataset.copy()
    if "season" in filtered.columns:
        season_values = pd.to_numeric(filtered["season"], errors="coerce")
    else:
        season_values = pd.to_datetime(filtered["game_date"], errors="coerce").dt.year

    if start_year is not None:
        filtered = filtered.loc[season_values >= int(start_year)].copy()
        season_values = season_values.loc[filtered.index]
    if end_year is not None:
        filtered = filtered.loc[season_values <= int(end_year)].copy()

    filtered.attrs.update(dataset.attrs)
    return filtered


def _default_experiment_name() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"2026-away-mcmc-markov-v1-controlmu-{timestamp}"


def _resolve_stage3_distribution_report_path(
    *,
    explicit_path: str | Path | None,
    report_dir: str | Path,
) -> Path | None:
    if explicit_path is not None:
        path = Path(explicit_path)
        return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()

    report_dir_path = Path(report_dir)
    if not report_dir_path.is_absolute():
        report_dir_path = (PROJECT_ROOT / report_dir_path).resolve()
    candidates = sorted(report_dir_path.glob("*.distribution_eval.json"), reverse=True)
    for candidate in candidates:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        artifact_path = str(payload.get("artifact_path") or "")
        if "away_runs_distribution_model" in artifact_path:
            return candidate
    return None


def _relative_to_project(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def _row_seed(base_seed: int, row: pd.Series) -> int:
    game_pk = pd.to_numeric(pd.Series([row.get("game_pk")]), errors="coerce").iloc[0]
    if pd.notna(game_pk):
        return int(base_seed + int(game_pk))
    return int(base_seed + (int(row.name) * 17))


def _build_slice_summaries(
    *,
    summary_frame: pd.DataFrame,
    actual: np.ndarray,
    pmf_matrix: np.ndarray,
    support: np.ndarray,
) -> dict[str, dict[str, Any]]:
    if summary_frame.empty:
        return {}

    slices = {
        "actual_zero": summary_frame["actual_away_runs"].astype(int) == 0,
        "actual_ge_10": summary_frame["actual_away_runs"].astype(int) >= 10,
        "col_home": summary_frame["home_team"].astype(str) == "COL",
        "non_col_home": summary_frame["home_team"].astype(str) != "COL",
        "high_market_gap": summary_frame["market_gap_abs"].astype(float) >= 0.75,
        "market_available": summary_frame["market_priors_available"].astype(float) > 0.5,
        "fallback_applied": summary_frame["fallback_applied"].astype(bool),
    }
    payload: dict[str, dict[str, Any]] = {}
    predicted_mean = np.sum(pmf_matrix * support[None, :], axis=1)
    for slice_name, mask_series in slices.items():
        mask = mask_series.to_numpy(dtype=bool)
        if not np.any(mask):
            continue
        subset_metrics = summarize_distribution_metrics(actual[mask], pmf_matrix[mask], support)
        payload[slice_name] = {
            "row_count": int(np.count_nonzero(mask)),
            "rmse": float(np.sqrt(np.mean((predicted_mean[mask] - actual[mask]) ** 2))),
            "mean_crps": float(subset_metrics["mean_crps"]),
            "mean_negative_log_score": float(subset_metrics["mean_negative_log_score"]),
            "predicted_mean": float(np.mean(predicted_mean[mask])),
            "actual_mean": float(np.mean(actual[mask])),
        }
    return payload


def _build_sharpness_diagnostics(
    summary_frame: pd.DataFrame,
) -> dict[str, dict[str, float | int]]:
    if summary_frame.empty:
        return {}

    slices = {
        "overall": pd.Series(np.ones(len(summary_frame), dtype=bool), index=summary_frame.index),
        "fallback_applied": summary_frame["fallback_applied"].astype(bool),
        "non_fallback": ~summary_frame["fallback_applied"].astype(bool),
        "high_market_gap": summary_frame["market_gap_abs"].astype(float) >= 0.75,
    }
    payload: dict[str, dict[str, float | int]] = {}
    for slice_name, mask_series in slices.items():
        mask = mask_series.to_numpy(dtype=bool)
        if not np.any(mask):
            continue
        subset = summary_frame.loc[mask]
        payload[slice_name] = {
            "row_count": int(len(subset)),
            "mean_stddev": float(subset["distribution_stddev"].mean()),
            "mean_variance": float(subset["distribution_variance"].mean()),
            "mean_entropy": float(subset["distribution_entropy"].mean()),
            "mean_iqr": float(subset["distribution_iqr"].mean()),
            "mean_p50_away_runs": float(subset["p50_away_runs"].mean()),
            "mean_p75_away_runs": float(subset["p75_away_runs"].mean()),
        }
    return payload


def _build_market_decision_proxy_summary(
    *,
    summary_frame: pd.DataFrame,
    actual: np.ndarray,
    pmf_matrix: np.ndarray,
    support: np.ndarray,
) -> dict[str, Any]:
    if summary_frame.empty or "market_implied_away_runs" not in summary_frame.columns:
        return {
            "available": False,
            "proxy_only": True,
            "coverage": 0.0,
            "reason": "Market-implied away-run anchor is unavailable in the MCMC holdout predictions.",
        }

    market_implied = pd.to_numeric(summary_frame["market_implied_away_runs"], errors="coerce")
    available_mask = market_implied.notna().to_numpy(dtype=bool)
    if "market_priors_available" in summary_frame.columns:
        market_available = (
            pd.to_numeric(summary_frame["market_priors_available"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
            > 0.5
        )
        available_mask = available_mask & market_available
    coverage = float(np.mean(available_mask)) if len(summary_frame) else 0.0
    if not np.any(available_mask):
        return {
            "available": False,
            "proxy_only": True,
            "coverage": coverage,
            "reason": "No holdout rows contain a market-implied away-run anchor.",
        }

    predicted_mean = summary_frame["mcmc_expected_away_runs"].to_numpy(dtype=float)
    model_gap = predicted_mean - market_implied.fillna(0.0).to_numpy(dtype=float)
    actual_gap = actual.astype(float) - market_implied.fillna(0.0).to_numpy(dtype=float)
    threshold_payload: dict[str, Any] = {}
    for threshold in (0.5, 0.75, 1.0):
        threshold_key = f"abs_gap_ge_{str(threshold).replace('.', '_')}"
        over_mask = available_mask & (model_gap >= threshold)
        under_mask = available_mask & (model_gap <= -threshold)
        threshold_payload[threshold_key] = {
            "over": _summarize_proxy_direction_slice(
                mask=over_mask,
                actual=actual,
                pmf_matrix=pmf_matrix,
                support=support,
                predicted_mean=predicted_mean,
                model_gap=model_gap,
                actual_gap=actual_gap,
                direction="over",
            ),
            "under": _summarize_proxy_direction_slice(
                mask=under_mask,
                actual=actual,
                pmf_matrix=pmf_matrix,
                support=support,
                predicted_mean=predicted_mean,
                model_gap=model_gap,
                actual_gap=actual_gap,
                direction="under",
            ),
        }

    high_gap_mask = available_mask & (np.abs(model_gap) >= 0.75)
    sign_agreement = (
        float(np.mean(np.sign(model_gap[high_gap_mask]) == np.sign(actual_gap[high_gap_mask])))
        if np.any(high_gap_mask)
        else None
    )
    return {
        "available": True,
        "proxy_only": True,
        "coverage": coverage,
        "market_anchor_type": "derived_away_runs_from_full_game_total_and_side_markets",
        "mean_model_minus_market": float(np.mean(model_gap[available_mask])),
        "mean_abs_model_minus_market": float(np.mean(np.abs(model_gap[available_mask]))),
        "mean_actual_minus_market": float(np.mean(actual_gap[available_mask])),
        "high_gap_sign_agreement": sign_agreement,
        "threshold_summaries": threshold_payload,
        "reason": (
            "Proxy market-decision diagnostics are available, but direct away-team-total pricing is still absent."
        ),
    }


def _summarize_proxy_direction_slice(
    *,
    mask: np.ndarray,
    actual: np.ndarray,
    pmf_matrix: np.ndarray,
    support: np.ndarray,
    predicted_mean: np.ndarray,
    model_gap: np.ndarray,
    actual_gap: np.ndarray,
    direction: str,
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
    subset_metrics = summarize_distribution_metrics(actual[mask], pmf_matrix[mask], support)
    if direction == "over":
        hit_rate = float(np.mean(actual_gap[mask] > 0.0))
    else:
        hit_rate = float(np.mean(actual_gap[mask] < 0.0))
    return {
        "row_count": int(np.count_nonzero(mask)),
        "mean_model_minus_market": float(np.mean(model_gap[mask])),
        "mean_actual_minus_market": float(np.mean(actual_gap[mask])),
        "hit_rate": hit_rate,
        "rmse": float(np.sqrt(np.mean((predicted_mean[mask] - actual[mask]) ** 2))),
        "mean_crps": float(subset_metrics["mean_crps"]),
        "mean_negative_log_score": float(subset_metrics["mean_negative_log_score"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the Stage 4 away-run Markov / Monte Carlo distribution lane.",
    )
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--training-data", default=str(DEFAULT_TRAINING_DATA))
    parser.add_argument("--start", "--start-year", dest="start_year", type=int, default=2018)
    parser.add_argument("--end", "--end-year", dest="end_year", type=int, default=2025)
    parser.add_argument("--holdout", "--holdout-season", dest="holdout_season", type=int, default=2025)
    parser.add_argument(
        "--current-control",
        default=str(DEFAULT_CURRENT_CONTROL_PATH),
        help="Path to Stage 1 current_control.json. Ignored when --mean-artifact-metadata is supplied.",
    )
    parser.add_argument("--mean-artifact-metadata", default=None)
    parser.add_argument("--stage3-report-json", default=None)
    parser.add_argument("--distribution-report-dir", default=str(DEFAULT_DISTRIBUTION_REPORT_DIR))
    parser.add_argument("--mcmc-report-dir", default=str(DEFAULT_MCMC_REPORT_DIR))
    parser.add_argument("--simulations", type=int, default=DEFAULT_SIMULATION_COUNT)
    parser.add_argument("--starter-innings", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260328)
    parser.add_argument("--enable-market-priors", action="store_true")
    parser.add_argument("--historical-odds-db", default=None)
    parser.add_argument("--historical-market-book", default=None)
    parser.add_argument("--research-lane-name", default="mcmc_markov_v1")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    experiment_name = args.experiment or _default_experiment_name()
    output_dir = PROJECT_ROOT / "data" / "models" / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    stage3_report_path = _resolve_stage3_distribution_report_path(
        explicit_path=args.stage3_report_json,
        report_dir=args.distribution_report_dir,
    )
    mean_artifact_metadata_path = resolve_mean_artifact_metadata_path(
        current_control_path=args.current_control,
        explicit_mean_artifact_metadata_path=args.mean_artifact_metadata,
    )

    if args.dry_run:
        console.print(
            f"experiment={experiment_name} training_data={args.training_data} "
            f"season_range={args.start_year}-{args.end_year} holdout={args.holdout_season} "
            f"simulations={args.simulations} starter_innings={args.starter_innings} "
            f"mean_artifact_metadata={mean_artifact_metadata_path} stage3_report={stage3_report_path} "
            f"enable_market_priors={args.enable_market_priors} historical_odds_db={args.historical_odds_db}"
        )
        return 0

    validated_training_data = validate_run_count_training_data(Path(args.training_data))
    filtered_training_data = _filter_training_data_by_season(
        validated_training_data,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    if filtered_training_data.empty:
        raise ValueError(
            f"No training rows remain after applying season filter start={args.start_year} end={args.end_year}"
        )

    inspection = inspect_run_count_training_data(filtered_training_data)
    dataset = _load_training_dataframe(filtered_training_data)
    frame = _prepare_run_count_frame(dataset, target_column=DEFAULT_TARGET_COLUMN)
    research_feature_result = augment_run_research_features(
        frame,
        enable_market_priors=args.enable_market_priors,
        historical_odds_db_path=args.historical_odds_db,
        historical_market_book_name=args.historical_market_book,
    )
    frame = research_feature_result.dataframe
    train_frame = frame.loc[frame["season"] < int(args.holdout_season)].copy()
    holdout_frame = frame.loc[frame["season"] == int(args.holdout_season)].copy()
    if train_frame.empty:
        raise ValueError(f"No training rows found before holdout season {args.holdout_season}")
    if holdout_frame.empty:
        raise ValueError(f"No holdout rows found for season {args.holdout_season}")

    mean_head = load_mean_head_reference(mean_artifact_metadata_path)
    train_mean = np.clip(
        np.asarray(mean_head.estimator.predict(train_frame.loc[:, mean_head.feature_columns]), dtype=float),
        1e-9,
        None,
    )
    holdout_control_mean = np.clip(
        np.asarray(mean_head.estimator.predict(holdout_frame.loc[:, mean_head.feature_columns]), dtype=float),
        1e-9,
        None,
    )
    train_actual = train_frame[DEFAULT_TARGET_COLUMN].astype(int).to_numpy()
    holdout_actual = holdout_frame[DEFAULT_TARGET_COLUMN].astype(int).to_numpy()

    console.print("[bold green]Stage 4 MCMC lane[/bold green]")
    console.print(f"  parquet_path={inspection.parquet_path or '<in-memory>'}")
    console.print(f"  data_version_hash={inspection.data_version_hash}")
    console.print(f"  row_count={inspection.row_count}")
    console.print(f"  holdout_rows={len(holdout_frame)}")
    console.print(f"  mean_head_artifact={mean_artifact_metadata_path}")
    console.print(f"  stage3_report={stage3_report_path or '<none found>'}")
    console.print(
        f"  simulations={int(args.simulations)} starter_innings={int(args.starter_innings)} seed={int(args.seed)}"
    )

    per_row_probabilities: list[np.ndarray] = []
    per_row_summaries: list[dict[str, Any]] = []
    support_max = int(np.max(holdout_actual, initial=0))
    total_event_counts: dict[str, int] | None = None
    mean_runs_by_inning_accumulator: np.ndarray | None = None
    mean_plate_appearances_by_inning_accumulator: np.ndarray | None = None
    truncated_half_innings = 0
    mean_anchor_applied_count = 0
    fallback_applied_count = 0
    profile_out_of_bounds_count = 0
    fallback_reason_counts: dict[str, int] = {}
    pre_anchor_drift: list[float] = []
    post_anchor_drift: list[float] = []

    for position, (_, row) in enumerate(holdout_frame.iterrows(), start=1):
        feature_bundle = build_mcmc_feature_bundle(
            row,
            target_mean_runs=float(holdout_control_mean[position - 1]),
            starter_innings=int(args.starter_innings),
        )
        simulation = simulate_away_game_distribution(
            starter_profile=feature_bundle.starter_profile,
            bullpen_profile=feature_bundle.bullpen_profile,
            simulations=args.simulations,
            starter_innings=args.starter_innings,
            seed=_row_seed(int(args.seed), row),
        )
        support_max = max(support_max, int(simulation.support[-1]) if len(simulation.support) > 0 else 0)
        per_row_probabilities.append(simulation.pmf)
        summary = summarize_away_run_distribution(
            support=simulation.support,
            probabilities=simulation.pmf,
        )
        market_priors_available = pd.to_numeric(
            pd.Series([row.get("market_priors_available", 0.0)]),
            errors="coerce",
        ).fillna(0.0).iloc[0]
        market_anchor_confidence = pd.to_numeric(
            pd.Series([row.get("market_anchor_confidence", 0.0)]),
            errors="coerce",
        ).fillna(0.0).iloc[0]
        market_implied_away_runs = pd.to_numeric(
            pd.Series([row.get("market_implied_full_game_away_runs")]),
            errors="coerce",
        ).iloc[0]
        resolved_market_implied_away_runs = (
            float(market_implied_away_runs) if pd.notna(market_implied_away_runs) else float("nan")
        )
        per_row_summaries.append(
            {
                "game_pk": int(row["game_pk"]),
                "game_date": str(row["game_date"]),
                "away_team": str(row["away_team"]),
                "home_team": str(row["home_team"]),
                "actual_away_runs": int(row[DEFAULT_TARGET_COLUMN]),
                "control_mean_runs": float(holdout_control_mean[position - 1]),
                "market_priors_available": float(market_priors_available),
                "market_anchor_confidence": float(market_anchor_confidence),
                "market_implied_away_runs": resolved_market_implied_away_runs,
                "target_mean_runs": float(feature_bundle.target_mean_runs),
                "mcmc_expected_away_runs": float(summary.expected_away_runs),
                "mean_drift_vs_control": float(summary.expected_away_runs - holdout_control_mean[position - 1]),
                "pre_anchor_implied_mean_runs": float(feature_bundle.pre_anchor_implied_mean_runs),
                "post_anchor_implied_mean_runs": float(feature_bundle.post_anchor_implied_mean_runs),
                "mean_anchor_applied": bool(feature_bundle.mean_anchor_applied),
                "fallback_applied": bool(feature_bundle.fallback_applied),
                "fallback_reason": feature_bundle.fallback_reason,
                "profile_out_of_bounds": bool(feature_bundle.profile_out_of_bounds),
                "market_gap_abs": float(
                    abs(
                        float(feature_bundle.raw_feature_snapshot["market_implied_full_game_away_runs"])
                        - float(holdout_control_mean[position - 1])
                    )
                ),
                "model_minus_market_runs": float(
                    summary.expected_away_runs - float(feature_bundle.raw_feature_snapshot["market_implied_full_game_away_runs"])
                ),
                "control_minus_market_runs": float(
                    float(holdout_control_mean[position - 1])
                    - float(feature_bundle.raw_feature_snapshot["market_implied_full_game_away_runs"])
                ),
                "actual_minus_market_runs": float(
                    float(row[DEFAULT_TARGET_COLUMN])
                    - float(feature_bundle.raw_feature_snapshot["market_implied_full_game_away_runs"])
                ),
                "shutout_probability": float(summary.shutout_probability),
                "p_ge_3": float(summary.tail_probabilities["p_ge_3"]),
                "p_ge_5": float(summary.tail_probabilities["p_ge_5"]),
                "p_ge_10": float(summary.tail_probabilities["p_ge_10"]),
                "p25_away_runs": float(summary.quantiles["p25"]),
                "p50_away_runs": float(summary.quantiles["p50"]),
                "p75_away_runs": float(summary.quantiles["p75"]),
                "distribution_variance": float(summary.shape_summary["variance"]),
                "distribution_stddev": float(summary.shape_summary["stddev"]),
                "distribution_entropy": float(summary.shape_summary["entropy"]),
                "distribution_iqr": float(summary.shape_summary["iqr"]),
                "away_run_pmf_json": json.dumps(summary.away_run_pmf, separators=(",", ":")),
                "feature_bundle_diagnostics_json": json.dumps(feature_bundle.diagnostics, separators=(",", ":")),
                "diagnostics_json": json.dumps(simulation.diagnostics, separators=(",", ":")),
            }
        )

        mean_anchor_applied_count += int(feature_bundle.mean_anchor_applied)
        fallback_applied_count += int(feature_bundle.fallback_applied)
        profile_out_of_bounds_count += int(feature_bundle.profile_out_of_bounds)
        if feature_bundle.fallback_reason:
            fallback_reason_counts[feature_bundle.fallback_reason] = (
                fallback_reason_counts.get(feature_bundle.fallback_reason, 0) + 1
            )
        pre_anchor_drift.append(
            float(feature_bundle.pre_anchor_implied_mean_runs - feature_bundle.target_mean_runs)
        )
        post_anchor_drift.append(
            float(feature_bundle.post_anchor_implied_mean_runs - feature_bundle.target_mean_runs)
        )

        diagnostics = simulation.diagnostics
        if total_event_counts is None:
            total_event_counts = {name: int(count) for name, count in diagnostics["event_counts"].items()}
        else:
            for event_name, count in diagnostics["event_counts"].items():
                total_event_counts[event_name] += int(count)

        inning_run_values = np.asarray(diagnostics["mean_runs_by_inning"], dtype=float)
        inning_pa_values = np.asarray(diagnostics["mean_plate_appearances_by_inning"], dtype=float)
        if mean_runs_by_inning_accumulator is None:
            mean_runs_by_inning_accumulator = inning_run_values
            mean_plate_appearances_by_inning_accumulator = inning_pa_values
        else:
            mean_runs_by_inning_accumulator = mean_runs_by_inning_accumulator + inning_run_values
            mean_plate_appearances_by_inning_accumulator = (
                mean_plate_appearances_by_inning_accumulator + inning_pa_values
            )
        truncated_half_innings += int(diagnostics["truncated_half_innings"])

        if position % 250 == 0 or position == len(holdout_frame):
            console.print(f"  simulated {position}/{len(holdout_frame)} holdout games")

    pmf_matrix = np.vstack(
        [pad_probability_vector(probabilities, support_max=support_max) for probabilities in per_row_probabilities]
    )
    support = np.arange(support_max + 1, dtype=int)
    distribution_metrics = summarize_distribution_metrics(
        holdout_actual,
        pmf_matrix,
        support,
    )
    pmf_expected_runs = np.sum(pmf_matrix * support[None, :], axis=1)
    mean_metrics = _compute_holdout_metrics(
        train_frame=train_frame,
        holdout_frame=holdout_frame,
        target_column=DEFAULT_TARGET_COLUMN,
        holdout_predictions=pmf_expected_runs,
    )
    control_baseline = evaluate_control_distribution_baseline(
        train_actual=train_actual,
        train_mean=train_mean,
        holdout_actual=holdout_actual,
        holdout_mean=holdout_control_mean,
        calibration_bin_count=10,
        tail_probability=1e-8,
    )
    control_mean_metrics = _compute_holdout_metrics(
        train_frame=train_frame,
        holdout_frame=holdout_frame,
        target_column=DEFAULT_TARGET_COLUMN,
        holdout_predictions=holdout_control_mean,
    )
    comparison_to_control = build_distribution_comparison(
        challenger_label="stage4_mcmc",
        challenger_mean_metrics=mean_metrics,
        challenger_distribution_metrics=distribution_metrics,
        baseline_label="control",
        baseline_mean_metrics=control_mean_metrics,
        baseline_distribution_metrics=control_baseline["holdout_metrics"],
    )

    stage3_report_payload: dict[str, Any] | None = None
    comparison_to_stage3: dict[str, Any] | None = None
    if stage3_report_path is not None and stage3_report_path.exists():
        stage3_report_payload = json.loads(stage3_report_path.read_text(encoding="utf-8"))
        comparison_to_stage3 = build_distribution_comparison(
            challenger_label="stage4_mcmc",
            challenger_mean_metrics=mean_metrics,
            challenger_distribution_metrics=distribution_metrics,
            baseline_label="stage3_distribution",
            baseline_mean_metrics=stage3_report_payload["mean_metrics"],
            baseline_distribution_metrics=stage3_report_payload["holdout_metrics"],
        )

    model_version = _build_model_version(_resolve_data_version_hash(dataset))
    report_dir = Path(args.mcmc_report_dir)
    if not report_dir.is_absolute():
        report_dir = (PROJECT_ROOT / report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"{DEFAULT_MODEL_NAME}_{model_version}.joblib"
    metadata_path = output_dir / f"{DEFAULT_MODEL_NAME}_{model_version}.metadata.json"
    summary_path = output_dir / f"{DEFAULT_MODEL_NAME}_run_{model_version}.json"
    report_json_path = report_dir / f"{model_version}.mcmc_eval.json"
    report_csv_path = report_dir / f"{model_version}.mcmc_eval.csv"
    control_comparison_path = report_dir / f"{model_version}.vs_control.json"
    stage3_comparison_path = report_dir / f"{model_version}.vs_stage3.json"
    predictions_csv_path = report_dir / f"{model_version}.holdout_predictions.csv"

    aggregate_event_counts = total_event_counts or {}
    total_plate_appearances = sum(aggregate_event_counts.values())
    simulation_diagnostics = {
        "holdout_game_count": int(len(holdout_frame)),
        "total_truncated_half_innings": int(truncated_half_innings),
        "mean_runs_by_inning": (
            (mean_runs_by_inning_accumulator / len(holdout_frame)).tolist()
            if mean_runs_by_inning_accumulator is not None
            else []
        ),
        "mean_plate_appearances_by_inning": (
            (mean_plate_appearances_by_inning_accumulator / len(holdout_frame)).tolist()
            if mean_plate_appearances_by_inning_accumulator is not None
            else []
        ),
        "aggregate_event_counts": aggregate_event_counts,
        "aggregate_event_share_by_plate_appearance": {
            name: (float(count) / float(total_plate_appearances) if total_plate_appearances > 0 else 0.0)
            for name, count in aggregate_event_counts.items()
        },
        "mean_control_expected_runs": float(np.mean(holdout_control_mean)),
        "mean_mcmc_expected_runs": float(np.mean(pmf_expected_runs)),
        "mean_anchor_applied_count": int(mean_anchor_applied_count),
        "fallback_applied_count": int(fallback_applied_count),
        "profile_out_of_bounds_count": int(profile_out_of_bounds_count),
        "fallback_reason_counts": fallback_reason_counts,
        "pre_anchor_mean_drift_summary": {
            "mean": float(np.mean(pre_anchor_drift)) if pre_anchor_drift else 0.0,
            "mean_abs": float(np.mean(np.abs(pre_anchor_drift))) if pre_anchor_drift else 0.0,
            "max_abs": float(np.max(np.abs(pre_anchor_drift))) if pre_anchor_drift else 0.0,
        },
        "post_anchor_mean_drift_summary": {
            "mean": float(np.mean(post_anchor_drift)) if post_anchor_drift else 0.0,
            "mean_abs": float(np.mean(np.abs(post_anchor_drift))) if post_anchor_drift else 0.0,
            "max_abs": float(np.max(np.abs(post_anchor_drift))) if post_anchor_drift else 0.0,
        },
    }
    summary_feature_columns = (
        "p25_away_runs",
        "p50_away_runs",
        "p75_away_runs",
        "distribution_variance",
        "distribution_stddev",
        "distribution_entropy",
        "distribution_iqr",
    )
    summary_feature_aggregates = {
        f"mean_{column}": float(np.mean([float(item[column]) for item in per_row_summaries]))
        for column in summary_feature_columns
    }
    summary_frame = pd.DataFrame(per_row_summaries)
    slice_summaries = _build_slice_summaries(
        summary_frame=summary_frame,
        actual=holdout_actual,
        pmf_matrix=pmf_matrix,
        support=support,
    )
    sharpness_diagnostics = _build_sharpness_diagnostics(summary_frame)
    market_decision_proxy_summary = _build_market_decision_proxy_summary(
        summary_frame=summary_frame,
        actual=holdout_actual,
        pmf_matrix=pmf_matrix,
        support=support,
    )
    metadata_payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model_name": DEFAULT_MODEL_NAME,
        "model_version": model_version,
        "target_column": DEFAULT_TARGET_COLUMN,
        "holdout_season": int(args.holdout_season),
        "distribution_family": "markov_monte_carlo",
        "distribution_lane_stage": 4,
        "research_lane_name": args.research_lane_name,
        "simulation_count": int(args.simulations),
        "starter_innings": int(args.starter_innings),
        "seed": int(args.seed),
        "data_version_hash": _resolve_data_version_hash(dataset),
        "training_data_path": str(args.training_data),
        "mean_head_source_artifact_path": _relative_to_project(Path(mean_artifact_metadata_path)),
        "stage3_report_path": None if stage3_report_path is None else _relative_to_project(stage3_report_path),
        "research_feature_metadata": research_metadata_to_dict(research_feature_result.metadata),
        "mean_metrics": mean_metrics,
        "distribution_metrics": distribution_metrics,
        "prediction_summary": distribution_metrics["prediction_summary"],
        "simulation_diagnostics": simulation_diagnostics,
        "slice_summaries": slice_summaries,
        "sharpness_diagnostics": sharpness_diagnostics,
        "market_decision_proxy_summary": market_decision_proxy_summary,
        "distribution_summary_features": {
            "usage": "report_only",
            "feature_columns": list(summary_feature_columns),
            "aggregate_means": summary_feature_aggregates,
            "notes": [
                "Stage 4 quantile and shape summaries are persisted for pricing and reporting.",
                "These summaries are not currently looped back into upstream training features.",
            ],
        },
        "comparison_to_control": comparison_to_control,
        "comparison_to_stage3": comparison_to_stage3,
        "output_paths": {
            "report_json": _relative_to_project(report_json_path),
            "report_csv": _relative_to_project(report_csv_path),
            "predictions_csv": _relative_to_project(predictions_csv_path),
            "vs_control_json": _relative_to_project(control_comparison_path),
            "vs_stage3_json": _relative_to_project(stage3_comparison_path),
        },
    }

    joblib.dump(
        {
            "model_name": DEFAULT_MODEL_NAME,
            "model_version": model_version,
            "simulation_count": int(args.simulations),
            "starter_innings": int(args.starter_innings),
            "seed": int(args.seed),
            "mean_head_source_artifact_path": str(mean_artifact_metadata_path),
        },
        model_path,
    )
    metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")

    report_payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "artifact_path": _relative_to_project(metadata_path),
        "model_path": _relative_to_project(model_path),
        "model_version": model_version,
        "model_name": DEFAULT_MODEL_NAME,
        "research_lane_name": args.research_lane_name,
        "research_feature_metadata": research_metadata_to_dict(research_feature_result.metadata),
        "mean_metrics": mean_metrics,
        "holdout_metrics": distribution_metrics,
        "simulation_diagnostics": simulation_diagnostics,
        "slice_summaries": slice_summaries,
        "sharpness_diagnostics": sharpness_diagnostics,
        "market_decision_proxy_summary": market_decision_proxy_summary,
        "distribution_summary_features": metadata_payload["distribution_summary_features"],
        "comparison_to_control": comparison_to_control,
        "comparison_to_stage3": comparison_to_stage3,
    }
    report_json_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    pd.DataFrame(
        [
            flatten_mcmc_report_row(
                model_version=model_version,
                metadata_path=metadata_path,
                mean_metrics=mean_metrics,
                distribution_metrics=distribution_metrics,
                control_comparison=comparison_to_control,
                stage3_comparison=comparison_to_stage3,
                summary_feature_aggregates=summary_feature_aggregates,
            )
        ]
    ).to_csv(report_csv_path, index=False)
    summary_frame.to_csv(predictions_csv_path, index=False)
    control_comparison_path.write_text(json.dumps(comparison_to_control, indent=2), encoding="utf-8")
    stage3_comparison_path.write_text(
        json.dumps(comparison_to_stage3, indent=2) if comparison_to_stage3 is not None else "null",
        encoding="utf-8",
    )

    console.print(f"[bold green]Done[/bold green] metadata={metadata_path}")
    console.print(
        f"RMSE={mean_metrics['rmse']:.4f} "
        f"CRPS={distribution_metrics['mean_crps']:.6f} "
        f"NegLogScore={distribution_metrics['mean_negative_log_score']:.6f}"
    )
    console.print(
        f"beats_control_on_crps={comparison_to_control['improvement_flags']['beats_baseline_on_crps']} "
        f"beats_control_on_negative_log_score={comparison_to_control['improvement_flags']['beats_baseline_on_negative_log_score']} "
        f"catastrophic_regression={comparison_to_control['catastrophic_regression']}"
    )
    if comparison_to_stage3 is not None:
        console.print(
            f"beats_stage3_on_crps={comparison_to_stage3['improvement_flags']['beats_baseline_on_crps']} "
            f"beats_stage3_on_negative_log_score={comparison_to_stage3['improvement_flags']['beats_baseline_on_negative_log_score']}"
        )
    console.print(
        json.dumps(
            {
                "model_version": model_version,
                "model_path": str(model_path),
                "metadata_path": str(metadata_path),
                "report_json_path": str(report_json_path),
                "report_csv_path": str(report_csv_path),
                "predictions_csv_path": str(predictions_csv_path),
                "vs_control_json_path": str(control_comparison_path),
                "vs_stage3_json_path": str(stage3_comparison_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
