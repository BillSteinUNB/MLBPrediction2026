from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import joblib
import pandas as pd

from src.clients.historical_odds_client import load_historical_odds_for_games
from src.clients.odds_client import devig_probabilities
from src.engine.bankroll import calculate_kelly_stake
from src.engine.edge_calculator import payout_for_american_odds
from src.model.calibration import CalibratedStackingModel
from src.model.direct_rl_trainer import _augment_direct_rl_training_frame
from src.model.market_recalibration import shrink_probability_toward_market
from src.model.margin_pricing import margin_to_cover_probability
from src.model.xgboost_trainer import _load_training_dataframe
from src.pipeline.daily import ArtifactOrFallbackPredictionEngine


StrategyMode = Literal["ml_only", "rl_only", "mixed"]
RlSource = Literal["direct", "blend"]
SelectorScoreMode = Literal["edge", "edge_ev", "edge_prob"]
StakingMode = Literal["flat", "kelly", "edge_scaled"]
EvaluationMode = Literal["nested_walk_forward", "in_sample_research"]


@dataclass(frozen=True, slots=True)
class StrategyParams:
    strategy_mode: StrategyMode
    rl_source: RlSource
    ml_market_base_multiplier: float
    ml_market_plus_money_multiplier: float
    ml_market_high_edge_threshold: float
    ml_market_high_edge_multiplier: float
    ml_edge_threshold: float
    rl_edge_threshold: float
    rl_selection_penalty: float
    selector_score_mode: SelectorScoreMode
    staking_mode: StakingMode
    flat_bet_size_units: float
    min_bet_size_units: float
    max_bet_size_units: float
    edge_scale_cap: float
    max_bet_odds: int | None
    min_bet_odds: int | None
    play_of_day_min_edge: float


@dataclass(frozen=True, slots=True)
class StrategyMetrics:
    starting_bankroll_units: float
    ending_bankroll_units: float
    peak_bankroll_units: float
    bankroll_return_pct: float
    roi: float | None
    max_drawdown_pct: float
    total_bets: int
    total_profit_units: float
    total_staked_units: float
    wins: int
    losses: int
    pushes: int
    average_stake_units: float
    play_of_day_picks: int
    play_of_day_profit_units: float
    play_of_day_roi: float | None
    forced_picks: int
    forced_profit_units: float
    forced_roi: float | None


@dataclass(frozen=True, slots=True)
class SweepResult:
    trial_index: int
    params: StrategyParams
    metrics: StrategyMetrics


def run_strategy_sweep(
    *,
    ml_training_data: str | Path,
    rl_training_data: str | Path,
    model_dir: str | Path,
    db_path: str | Path,
    book_name: str = "sbr:caesars",
    start_year: int = 2021,
    end_year: int = 2025,
    trial_count: int = 60,
    output_dir: str | Path | None = None,
    seed: int = 42,
    strategy_mode_filter: Sequence[StrategyMode] | None = None,
    include_baselines: bool = True,
    evaluation_mode: EvaluationMode = "nested_walk_forward",
    min_training_seasons: int = 1,
) -> dict[str, Any]:
    random.seed(seed)

    engine = ArtifactOrFallbackPredictionEngine(model_dir=model_dir)
    if engine.ml_model is None:
        raise ValueError("No calibrated ML model bundle is available for strategy sweep")
    if engine.rlv2_direct is None or engine.rlv2_margin is None:
        raise ValueError("RL v2 direct and margin artifacts are required for strategy sweep")

    ml_frame = _load_training_dataframe(ml_training_data)
    ml_frame = ml_frame.loc[ml_frame["season"].between(start_year, end_year)].copy()
    rl_frame = _load_training_dataframe(rl_training_data)
    rl_frame = rl_frame.loc[rl_frame["season"].between(start_year, end_year)].copy()
    rl_frame = _augment_direct_rl_training_frame(rl_frame)

    if ml_frame.empty or rl_frame.empty:
        raise ValueError("Training data does not contain rows for the requested evaluation years")

    evaluation_frame = _build_evaluation_frame(
        engine=engine,
        ml_frame=ml_frame,
        rl_frame=rl_frame,
        db_path=db_path,
        book_name=book_name,
    )
    if evaluation_frame.empty:
        raise ValueError("No games had both model inputs and historical odds for the requested sweep")

    allowed_modes = tuple(strategy_mode_filter or ("ml_only", "rl_only", "mixed"))

    baseline_params = [
        StrategyParams(
            strategy_mode="ml_only",
            rl_source="blend",
            ml_market_base_multiplier=1.0,
            ml_market_plus_money_multiplier=1.0,
            ml_market_high_edge_threshold=0.10,
            ml_market_high_edge_multiplier=1.0,
            ml_edge_threshold=0.05,
            rl_edge_threshold=0.05,
            rl_selection_penalty=0.0,
            selector_score_mode="edge",
            staking_mode="edge_scaled",
            flat_bet_size_units=1.0,
            min_bet_size_units=0.25,
            max_bet_size_units=3.0,
            edge_scale_cap=0.06,
            max_bet_odds=150,
            min_bet_odds=-175,
            play_of_day_min_edge=0.08,
        ),
        StrategyParams(
            strategy_mode="rl_only",
            rl_source="blend",
            ml_market_base_multiplier=1.0,
            ml_market_plus_money_multiplier=1.0,
            ml_market_high_edge_threshold=0.10,
            ml_market_high_edge_multiplier=1.0,
            ml_edge_threshold=0.05,
            rl_edge_threshold=0.05,
            rl_selection_penalty=0.0,
            selector_score_mode="edge",
            staking_mode="edge_scaled",
            flat_bet_size_units=1.0,
            min_bet_size_units=0.25,
            max_bet_size_units=3.0,
            edge_scale_cap=0.06,
            max_bet_odds=150,
            min_bet_odds=-175,
            play_of_day_min_edge=0.08,
        ),
        StrategyParams(
            strategy_mode="mixed",
            rl_source="blend",
            ml_market_base_multiplier=1.0,
            ml_market_plus_money_multiplier=1.0,
            ml_market_high_edge_threshold=0.10,
            ml_market_high_edge_multiplier=1.0,
            ml_edge_threshold=0.05,
            rl_edge_threshold=0.05,
            rl_selection_penalty=0.0,
            selector_score_mode="edge",
            staking_mode="edge_scaled",
            flat_bet_size_units=1.0,
            min_bet_size_units=0.25,
            max_bet_size_units=3.0,
            edge_scale_cap=0.06,
            max_bet_odds=150,
            min_bet_odds=-175,
            play_of_day_min_edge=0.08,
        ),
    ]

    if evaluation_mode == "nested_walk_forward":
        leaderboard, payload = _run_nested_walk_forward_sweep(
            evaluation_frame=evaluation_frame,
            trial_count=trial_count,
            allowed_modes=allowed_modes,
            include_baselines=include_baselines,
            baseline_params=baseline_params,
            min_training_seasons=min_training_seasons,
        )
    else:
        leaderboard, payload = _run_in_sample_sweep(
            evaluation_frame=evaluation_frame,
            trial_count=trial_count,
            allowed_modes=allowed_modes,
            include_baselines=include_baselines,
            baseline_params=baseline_params,
        )

    payload["evaluation"] = {
        "book_name": book_name,
        "start_year": start_year,
        "end_year": end_year,
        "row_count": int(len(evaluation_frame)),
        "game_count": int(evaluation_frame["game_pk"].nunique()),
        "model_version": engine.model_version,
        "rlv2_direct_version": engine.rlv2_direct.model_version,
        "rlv2_margin_version": engine.rlv2_margin.model_version,
        "rlv2_blend_weight": float(engine.rlv2_blend_weight),
        "evaluation_mode": evaluation_mode,
        "min_training_seasons": int(min_training_seasons),
    }

    if output_dir is not None:
        resolved_output_dir = Path(output_dir)
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
        leaderboard.to_csv(resolved_output_dir / "leaderboard.csv", index=False)
        (resolved_output_dir / "summary.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    return {
        "leaderboard": leaderboard,
        "summary": payload,
    }


def _run_in_sample_sweep(
    *,
    evaluation_frame: pd.DataFrame,
    trial_count: int,
    allowed_modes: Sequence[StrategyMode],
    include_baselines: bool,
    baseline_params: Sequence[StrategyParams],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    results: list[SweepResult] = []
    if include_baselines:
        for params in baseline_params:
            if params.strategy_mode not in allowed_modes:
                continue
            metrics = _simulate_strategy(evaluation_frame=evaluation_frame, params=params)
            results.append(SweepResult(trial_index=len(results) + 1, params=params, metrics=metrics))

    for _offset in range(trial_count):
        params = _sample_params(allowed_modes, include_kelly=True)
        metrics = _simulate_strategy(evaluation_frame=evaluation_frame, params=params)
        results.append(SweepResult(trial_index=len(results) + 1, params=params, metrics=metrics))

    leaderboard = _results_to_leaderboard(results)
    payload = {
        "best_overall": leaderboard.iloc[0].to_dict(),
        "best_by_mode": {
            mode: leaderboard.loc[leaderboard["strategy_mode"] == mode].iloc[0].to_dict()
            for mode in sorted(leaderboard["strategy_mode"].unique())
        },
    }
    return leaderboard, payload


def _run_nested_walk_forward_sweep(
    *,
    evaluation_frame: pd.DataFrame,
    trial_count: int,
    allowed_modes: Sequence[StrategyMode],
    include_baselines: bool,
    baseline_params: Sequence[StrategyParams],
    min_training_seasons: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    seasons = sorted(int(value) for value in evaluation_frame["season"].dropna().astype(int).unique())
    outer_test_seasons = seasons[int(min_training_seasons) :]
    if not outer_test_seasons:
        raise ValueError("Nested strategy evaluation requires at least one outer test season")

    current_bankroll = 100.0
    current_peak_bankroll = 100.0
    outer_rows: list[dict[str, Any]] = []
    total_profit = 0.0
    total_staked = 0.0
    total_bets = 0
    wins = losses = pushes = 0
    play_of_day_picks = 0
    play_of_day_profit = 0.0
    forced_picks = 0
    forced_profit = 0.0

    for outer_index, test_season in enumerate(outer_test_seasons, start=1):
        train_seasons = [season for season in seasons if season < test_season]
        if len(train_seasons) < int(min_training_seasons):
            continue
        train_frame = evaluation_frame.loc[evaluation_frame["season"].isin(train_seasons)].copy()
        test_frame = evaluation_frame.loc[evaluation_frame["season"] == test_season].copy()
        if train_frame.empty or test_frame.empty:
            continue

        candidate_results: list[SweepResult] = []
        if include_baselines:
            for params in baseline_params:
                if params.strategy_mode not in allowed_modes:
                    continue
                metrics = _simulate_strategy(evaluation_frame=train_frame, params=params)
                candidate_results.append(
                    SweepResult(trial_index=len(candidate_results) + 1, params=params, metrics=metrics)
                )

        for _offset in range(trial_count):
            params = _sample_params(allowed_modes, include_kelly=False)
            metrics = _simulate_strategy(evaluation_frame=train_frame, params=params)
            candidate_results.append(
                SweepResult(trial_index=len(candidate_results) + 1, params=params, metrics=metrics)
            )

        train_leaderboard = _results_to_leaderboard(candidate_results)
        selected_row = train_leaderboard.iloc[0].to_dict()
        selected_params = StrategyParams(
            strategy_mode=selected_row["strategy_mode"],
            rl_source=selected_row["rl_source"],
            ml_market_base_multiplier=float(selected_row["ml_market_base_multiplier"]),
            ml_market_plus_money_multiplier=float(selected_row["ml_market_plus_money_multiplier"]),
            ml_market_high_edge_threshold=float(selected_row["ml_market_high_edge_threshold"]),
            ml_market_high_edge_multiplier=float(selected_row["ml_market_high_edge_multiplier"]),
            ml_edge_threshold=float(selected_row["ml_edge_threshold"]),
            rl_edge_threshold=float(selected_row["rl_edge_threshold"]),
            rl_selection_penalty=float(selected_row["rl_selection_penalty"]),
            selector_score_mode=selected_row["selector_score_mode"],
            staking_mode=selected_row["staking_mode"],
            flat_bet_size_units=float(selected_row["flat_bet_size_units"]),
            min_bet_size_units=float(selected_row["min_bet_size_units"]),
            max_bet_size_units=float(selected_row["max_bet_size_units"]),
            edge_scale_cap=float(selected_row["edge_scale_cap"]),
            max_bet_odds=None if pd.isna(selected_row["max_bet_odds"]) else int(selected_row["max_bet_odds"]),
            min_bet_odds=None if pd.isna(selected_row["min_bet_odds"]) else int(selected_row["min_bet_odds"]),
            play_of_day_min_edge=float(selected_row["play_of_day_min_edge"]),
        )
        test_metrics = _simulate_strategy(
            evaluation_frame=test_frame,
            params=selected_params,
            starting_bankroll=current_bankroll,
            starting_peak_bankroll=current_peak_bankroll,
        )

        current_bankroll = float(test_metrics.ending_bankroll_units)
        current_peak_bankroll = float(test_metrics.peak_bankroll_units)
        total_profit += float(test_metrics.total_profit_units)
        total_staked += float(test_metrics.total_staked_units)
        total_bets += int(test_metrics.total_bets)
        wins += int(test_metrics.wins)
        losses += int(test_metrics.losses)
        pushes += int(test_metrics.pushes)
        play_of_day_picks += int(test_metrics.play_of_day_picks)
        play_of_day_profit += float(test_metrics.play_of_day_profit_units)
        forced_picks += int(test_metrics.forced_picks)
        forced_profit += float(test_metrics.forced_profit_units)

        outer_rows.append(
            {
                "outer_index": outer_index,
                "train_start_season": min(train_seasons),
                "train_end_season": max(train_seasons),
                "test_season": int(test_season),
                **asdict(selected_params),
                **{f"train_{key}": value for key, value in asdict(candidate_results[int(selected_row["trial_index"]) - 1].metrics).items()},
                **{f"test_{key}": value for key, value in asdict(test_metrics).items()},
            }
        )

    if not outer_rows:
        raise ValueError("Nested strategy evaluation produced no valid outer windows")

    leaderboard = pd.DataFrame(outer_rows).sort_values("outer_index").reset_index(drop=True)
    aggregate_metrics = {
        "starting_bankroll_units": 100.0,
        "ending_bankroll_units": current_bankroll,
        "peak_bankroll_units": current_peak_bankroll,
        "bankroll_return_pct": float((current_bankroll - 100.0) / 100.0),
        "roi": float(total_profit / total_staked) if total_staked else None,
        "max_drawdown_pct": float(leaderboard["test_max_drawdown_pct"].max()),
        "total_bets": int(total_bets),
        "total_profit_units": float(total_profit),
        "total_staked_units": float(total_staked),
        "wins": int(wins),
        "losses": int(losses),
        "pushes": int(pushes),
        "average_stake_units": float(total_staked / total_bets) if total_bets else 0.0,
        "play_of_day_picks": int(play_of_day_picks),
        "play_of_day_profit_units": float(play_of_day_profit),
        "play_of_day_roi": float(play_of_day_profit / play_of_day_picks) if play_of_day_picks else None,
        "forced_picks": int(forced_picks),
        "forced_profit_units": float(forced_profit),
        "forced_roi": float(forced_profit / forced_picks) if forced_picks else None,
        "outer_window_count": int(len(leaderboard)),
    }
    payload = {
        "aggregate_test_metrics": aggregate_metrics,
        "outer_windows": leaderboard.to_dict(orient="records"),
    }
    return leaderboard, payload


def _results_to_leaderboard(results: Sequence[SweepResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trial_index": result.trial_index,
                **asdict(result.params),
                **asdict(result.metrics),
            }
            for result in results
        ]
    ).sort_values(
        ["bankroll_return_pct", "roi", "play_of_day_profit_units", "forced_profit_units"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def _build_evaluation_frame(
    *,
    engine: ArtifactOrFallbackPredictionEngine,
    ml_frame: pd.DataFrame,
    rl_frame: pd.DataFrame,
    db_path: str | Path,
    book_name: str,
) -> pd.DataFrame:
    game_pks = sorted({int(value) for value in ml_frame["game_pk"].tolist()} | {int(value) for value in rl_frame["game_pk"].tolist()})
    ml_odds = load_historical_odds_for_games(
        db_path=db_path,
        game_pks=game_pks,
        market_type="f5_ml",
        book_name=book_name,
        snapshot_selection="opening",
    ).rename(
        columns={
            "home_odds": "ml_home_odds",
            "away_odds": "ml_away_odds",
            "fetched_at": "ml_fetched_at",
        }
    )
    rl_odds = load_historical_odds_for_games(
        db_path=db_path,
        game_pks=game_pks,
        market_type="f5_rl",
        book_name=book_name,
        snapshot_selection="opening",
    ).rename(
        columns={
            "home_odds": "rl_home_odds",
            "away_odds": "rl_away_odds",
            "home_point": "rl_home_point",
            "away_point": "rl_away_point",
            "fetched_at": "rl_fetched_at",
        }
    )

    ml_base = _ensure_ml_features(engine.ml_model, ml_frame.copy())
    ml_base["ml_home_probability"] = engine.ml_model.predict_calibrated(ml_base)
    ml_base["ml_away_probability"] = 1.0 - ml_base["ml_home_probability"]
    ml_base = ml_base[
        [
            "game_pk",
            "season",
            "game_date",
            "scheduled_start",
            "home_team",
            "away_team",
            "f5_ml_result",
            "f5_tied_after_5",
            "ml_home_probability",
            "ml_away_probability",
        ]
    ].drop_duplicates(subset=["game_pk"])

    rl_base = rl_frame.copy()
    direct_prepared = engine._prepare_rlv2_frame(
        inference_frame=rl_base.copy(),
        snapshot=_row_to_snapshot_placeholder(),
        feature_columns=engine.rlv2_direct.feature_columns,
    )
    # overwrite market columns from each row after placeholder fill
    direct_prepared["posted_f5_rl_home_point"] = rl_base["posted_f5_rl_home_point"]
    direct_prepared["posted_f5_rl_away_point"] = rl_base["posted_f5_rl_away_point"]
    direct_prepared["posted_f5_rl_home_odds"] = rl_base["posted_f5_rl_home_odds"]
    direct_prepared["posted_f5_rl_away_odds"] = rl_base["posted_f5_rl_away_odds"]
    direct_prepared["posted_f5_rl_home_implied_prob"] = rl_base["posted_f5_rl_home_implied_prob"]
    direct_prepared["posted_f5_rl_away_implied_prob"] = rl_base["posted_f5_rl_away_implied_prob"]
    direct_prepared["posted_f5_rl_point_abs"] = rl_base["posted_f5_rl_point_abs"]
    direct_prepared["posted_f5_rl_home_is_favorite"] = rl_base["posted_f5_rl_home_is_favorite"]

    rl_base["rl_direct_home_probability"] = engine.rlv2_direct.model.predict_proba(
        direct_prepared[engine.rlv2_direct.feature_columns]
    )[:, 1]

    margin_prepared = rl_base.copy()
    missing_margin_columns = {
        column: 0.0
        for column in engine.rlv2_margin.feature_columns
        if column not in margin_prepared.columns
    }
    if missing_margin_columns:
        margin_prepared = pd.concat(
            [margin_prepared, pd.DataFrame(missing_margin_columns, index=margin_prepared.index)],
            axis=1,
        )
    predicted_margin = engine.rlv2_margin.model.predict(margin_prepared[engine.rlv2_margin.feature_columns])
    residual_std = float(engine.rlv2_margin.extra_metadata.get("residual_std", 0.0) or 0.0)
    rl_base["rl_margin_home_probability"] = [
        margin_to_cover_probability(
            predicted_margin=float(margin),
            home_point=float(home_point),
            residual_std=residual_std,
        )
        for margin, home_point in zip(predicted_margin, rl_base["posted_f5_rl_home_point"], strict=False)
    ]
    rl_base["rl_blend_home_probability"] = (
        engine.rlv2_blend_weight * rl_base["rl_direct_home_probability"]
        + (1.0 - engine.rlv2_blend_weight) * rl_base["rl_margin_home_probability"]
    )
    rl_base = rl_base[
        [
            "game_pk",
            "home_cover_at_posted_line",
            "away_cover_at_posted_line",
            "push_at_posted_line",
            "posted_f5_rl_home_point",
            "posted_f5_rl_away_point",
            "rl_direct_home_probability",
            "rl_margin_home_probability",
            "rl_blend_home_probability",
        ]
    ].drop_duplicates(subset=["game_pk"])

    evaluation = ml_base.merge(ml_odds, on=["game_pk"], how="inner")
    evaluation = evaluation.merge(rl_base, on=["game_pk"], how="left")
    evaluation = evaluation.merge(rl_odds, on=["game_pk"], how="left")
    evaluation = evaluation.sort_values(["scheduled_start", "game_pk"]).reset_index(drop=True)
    return evaluation


def _ensure_ml_features(model: CalibratedStackingModel, frame: pd.DataFrame) -> pd.DataFrame:
    required_columns = set(model.stacking_model.base_feature_columns) | set(
        model.stacking_model.raw_meta_feature_columns
    )
    missing_columns = {column: 0.0 for column in required_columns if column not in frame.columns}
    if missing_columns:
        frame = pd.concat([frame, pd.DataFrame(missing_columns, index=frame.index)], axis=1)
    return frame


def _row_to_snapshot_placeholder():
    from datetime import UTC, datetime
    from src.models.odds import OddsSnapshot

    return OddsSnapshot(
        game_pk=0,
        book_name="placeholder",
        market_type="f5_rl",
        home_odds=-110,
        away_odds=100,
        home_point=-0.5,
        away_point=0.5,
        fetched_at=datetime.now(UTC),
        is_frozen=True,
    )


def _sample_params(allowed_modes: Sequence[StrategyMode], *, include_kelly: bool) -> StrategyParams:
    staking_modes: list[StakingMode] = ["flat", "edge_scaled"]
    if include_kelly:
        staking_modes.append("kelly")
    return StrategyParams(
        strategy_mode=random.choice(list(allowed_modes)),
        rl_source=random.choice(["direct", "blend"]),
        ml_market_base_multiplier=random.choice([0.55, 0.7, 0.85, 1.0]),
        ml_market_plus_money_multiplier=random.choice([0.35, 0.5, 0.7, 0.85, 1.0]),
        ml_market_high_edge_threshold=random.choice([0.08, 0.10, 0.12, 0.15]),
        ml_market_high_edge_multiplier=random.choice([0.4, 0.6, 0.8, 1.0]),
        ml_edge_threshold=random.choice([0.03, 0.04, 0.05, 0.06, 0.07]),
        rl_edge_threshold=random.choice([0.03, 0.04, 0.05, 0.06, 0.07, 0.08]),
        rl_selection_penalty=random.choice([0.0, 0.01, 0.015, 0.02, 0.03]),
        selector_score_mode=random.choice(["edge", "edge_ev", "edge_prob"]),
        staking_mode=random.choice(staking_modes),
        flat_bet_size_units=random.choice([0.5, 1.0]),
        min_bet_size_units=random.choice([0.25, 0.5, 0.75]),
        max_bet_size_units=random.choice([1.5, 2.0, 3.0]),
        edge_scale_cap=random.choice([0.05, 0.06, 0.08, 0.10]),
        max_bet_odds=random.choice([125, 150]),
        min_bet_odds=random.choice([-175, -150, -125]),
        play_of_day_min_edge=random.choice([0.06, 0.08, 0.10, 0.12]),
    )


def _simulate_strategy(
    *,
    evaluation_frame: pd.DataFrame,
    params: StrategyParams,
    starting_bankroll: float = 100.0,
    starting_peak_bankroll: float | None = None,
) -> StrategyMetrics:
    bankroll = float(starting_bankroll)
    peak_bankroll = float(starting_peak_bankroll if starting_peak_bankroll is not None else starting_bankroll)
    max_drawdown_pct = 0.0
    total_profit = 0.0
    total_staked = 0.0
    total_bets = wins = losses = pushes = 0
    stake_total = 0.0

    forced_profit = 0.0
    forced_picks = 0

    value_rows: list[dict[str, Any]] = []
    forced_rows: list[dict[str, Any]] = []

    for row in evaluation_frame.to_dict(orient="records"):
        decision = _best_decision_for_game(row, params=params, forced=False)
        forced = _best_decision_for_game(row, params=params, forced=True)
        if forced is not None:
            forced_result, forced_profit_units = _settle(forced)
            forced_profit += forced_profit_units
            forced_picks += 1
            forced_rows.append({"game_date": row["game_date"], **forced, "result": forced_result})

        if decision is None:
            continue

        stake = _resolve_stake(bankroll=bankroll, decision=decision, params=params, peak_bankroll=peak_bankroll)
        if stake <= 0:
            continue
        result, profit_per_unit = _settle(decision)
        realized_profit = float(profit_per_unit * stake)
        bankroll += realized_profit
        peak_bankroll = max(peak_bankroll, bankroll)
        drawdown = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll else 0.0
        max_drawdown_pct = max(max_drawdown_pct, drawdown)
        total_profit += realized_profit
        total_staked += stake
        stake_total += stake
        total_bets += 1
        wins += int(result == "WIN")
        losses += int(result == "LOSS")
        pushes += int(result == "PUSH")
        value_rows.append(
            {
                "game_date": row["game_date"],
                "stake": stake,
                "profit": realized_profit,
                "result": result,
                **decision,
            }
        )

    play_of_day_rows = []
    for game_date, items in pd.DataFrame(value_rows).groupby("game_date") if value_rows else []:
        eligible = items.loc[items["edge"] >= float(params.play_of_day_min_edge)].copy()
        if eligible.empty:
            continue
        eligible["selection_score"] = eligible.apply(
            lambda current: _selection_score(
                edge=float(current["edge"]),
                ev=float(current["ev"]),
                model_probability=float(current["model_probability"]),
                mode=params.selector_score_mode,
                market_type=str(current["market_type"]),
                rl_penalty=params.rl_selection_penalty,
            ),
            axis=1,
        )
        play_of_day_rows.append(eligible.sort_values("selection_score", ascending=False).iloc[0].to_dict())

    pod_bankroll = float(starting_bankroll)
    pod_profit = 0.0
    pod_staked = 0.0
    for row in play_of_day_rows:
        stake = _resolve_stake(bankroll=pod_bankroll, decision=row, params=params, peak_bankroll=pod_bankroll)
        if stake <= 0:
            continue
        realized_profit = float(row["profit"] / row["stake"] * stake) if float(row["stake"]) > 0 else 0.0
        pod_bankroll += realized_profit
        pod_profit += realized_profit
        pod_staked += stake

    roi = (total_profit / total_staked) if total_staked else None
    forced_roi = (forced_profit / forced_picks) if forced_picks else None
    pod_roi = (pod_profit / pod_staked) if pod_staked else None
    average_stake = (stake_total / total_bets) if total_bets else 0.0

    return StrategyMetrics(
        starting_bankroll_units=float(starting_bankroll),
        ending_bankroll_units=float(bankroll),
        peak_bankroll_units=float(peak_bankroll),
        bankroll_return_pct=float((bankroll - float(starting_bankroll)) / float(starting_bankroll)),
        roi=float(roi) if roi is not None else None,
        max_drawdown_pct=float(max_drawdown_pct),
        total_bets=int(total_bets),
        total_profit_units=float(total_profit),
        total_staked_units=float(total_staked),
        wins=int(wins),
        losses=int(losses),
        pushes=int(pushes),
        average_stake_units=float(average_stake),
        play_of_day_picks=int(len(play_of_day_rows)),
        play_of_day_profit_units=float(pod_profit),
        play_of_day_roi=float(pod_roi) if pod_roi is not None else None,
        forced_picks=int(forced_picks),
        forced_profit_units=float(forced_profit),
        forced_roi=float(forced_roi) if forced_roi is not None else None,
    )


def _best_decision_for_game(
    row: dict[str, Any],
    *,
    params: StrategyParams,
    forced: bool,
) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    ml_candidates = _build_ml_candidates(row, params=params)
    rl_candidates = _build_rl_candidates(row, rl_source=params.rl_source)
    if params.strategy_mode == "ml_only":
        candidates.extend(ml_candidates)
    elif params.strategy_mode == "rl_only":
        candidates.extend(rl_candidates)
    else:
        candidates.extend(ml_candidates)
        candidates.extend(rl_candidates)

    if params.min_bet_odds is not None:
        candidates = [candidate for candidate in candidates if int(candidate["odds"]) >= int(params.min_bet_odds)]
    if params.max_bet_odds is not None:
        candidates = [candidate for candidate in candidates if int(candidate["odds"]) <= int(params.max_bet_odds)]
    if not candidates:
        return None

    for candidate in candidates:
        threshold = params.ml_edge_threshold if candidate["market_type"] == "f5_ml" else params.rl_edge_threshold
        candidate["passes_threshold"] = float(candidate["edge"]) >= float(threshold)
        candidate["selection_score"] = _selection_score(
            edge=float(candidate["edge"]),
            ev=float(candidate["ev"]),
            model_probability=float(candidate["model_probability"]),
            mode=params.selector_score_mode,
            market_type=str(candidate["market_type"]),
            rl_penalty=params.rl_selection_penalty,
        )

    candidates.sort(
        key=lambda candidate: (
            int(candidate["passes_threshold"]),
            float(candidate["selection_score"]),
            float(candidate["model_probability"]),
        ),
        reverse=True,
    )
    if forced:
        return candidates[0]
    if not candidates[0]["passes_threshold"]:
        return None
    return candidates[0]


def _build_ml_candidates(row: dict[str, Any], *, params: StrategyParams) -> list[dict[str, Any]]:
    raw_home_prob = float(row["ml_home_probability"])
    raw_away_prob = float(row["ml_away_probability"])
    home_odds = int(row["ml_home_odds"])
    away_odds = int(row["ml_away_odds"])
    home_fair, away_fair = devig_probabilities(home_odds, away_odds)
    home_prob = shrink_probability_toward_market(
        model_probability=raw_home_prob,
        fair_probability=float(home_fair),
        odds=home_odds,
        base_multiplier=params.ml_market_base_multiplier,
        plus_money_multiplier=params.ml_market_plus_money_multiplier,
        high_edge_threshold=params.ml_market_high_edge_threshold,
        high_edge_multiplier=params.ml_market_high_edge_multiplier,
    )
    away_prob = shrink_probability_toward_market(
        model_probability=raw_away_prob,
        fair_probability=float(away_fair),
        odds=away_odds,
        base_multiplier=params.ml_market_base_multiplier,
        plus_money_multiplier=params.ml_market_plus_money_multiplier,
        high_edge_threshold=params.ml_market_high_edge_threshold,
        high_edge_multiplier=params.ml_market_high_edge_multiplier,
    )
    return [
        {
            "market_type": "f5_ml",
            "side": "home",
            "model_probability": home_prob,
            "fair_probability": float(home_fair),
            "edge": float(home_prob - home_fair),
            "ev": float((home_prob * payout_for_american_odds(home_odds)) - (1.0 - home_prob)),
            "odds": home_odds,
            "line": None,
            "actual_cover": int(row["f5_ml_result"]),
        },
        {
            "market_type": "f5_ml",
            "side": "away",
            "model_probability": away_prob,
            "fair_probability": float(away_fair),
            "edge": float(away_prob - away_fair),
            "ev": float((away_prob * payout_for_american_odds(away_odds)) - (1.0 - away_prob)),
            "odds": away_odds,
            "line": None,
            "actual_cover": 1 - int(row["f5_ml_result"]) if int(row["f5_tied_after_5"]) == 0 else 0,
            "is_push": int(row["f5_tied_after_5"]),
        },
    ]


def _build_rl_candidates(row: dict[str, Any], *, rl_source: RlSource) -> list[dict[str, Any]]:
    if pd.isna(row.get("rl_home_odds")) or pd.isna(row.get("rl_away_odds")):
        return []
    probability_column = "rl_blend_home_probability" if rl_source == "blend" else "rl_direct_home_probability"
    home_prob = float(row[probability_column])
    away_prob = 1.0 - home_prob
    home_odds = int(row["rl_home_odds"])
    away_odds = int(row["rl_away_odds"])
    home_fair, away_fair = devig_probabilities(home_odds, away_odds)
    home_cover = int(row["home_cover_at_posted_line"]) if not pd.isna(row["home_cover_at_posted_line"]) else 0
    away_cover = int(row["away_cover_at_posted_line"]) if not pd.isna(row["away_cover_at_posted_line"]) else 0
    push = int(row["push_at_posted_line"]) if not pd.isna(row["push_at_posted_line"]) else 0
    return [
        {
            "market_type": "f5_rl",
            "side": "home",
            "model_probability": home_prob,
            "fair_probability": float(home_fair),
            "edge": float(home_prob - home_fair),
            "ev": float((home_prob * payout_for_american_odds(home_odds)) - (1.0 - home_prob)),
            "odds": home_odds,
            "line": None if pd.isna(row["rl_home_point"]) else float(row["rl_home_point"]),
            "actual_cover": home_cover,
            "is_push": push,
        },
        {
            "market_type": "f5_rl",
            "side": "away",
            "model_probability": away_prob,
            "fair_probability": float(away_fair),
            "edge": float(away_prob - away_fair),
            "ev": float((away_prob * payout_for_american_odds(away_odds)) - (1.0 - away_prob)),
            "odds": away_odds,
            "line": None if pd.isna(row["rl_away_point"]) else float(row["rl_away_point"]),
            "actual_cover": away_cover,
            "is_push": push,
        },
    ]


def _selection_score(
    *,
    edge: float,
    ev: float,
    model_probability: float,
    mode: SelectorScoreMode,
    market_type: str,
    rl_penalty: float,
) -> float:
    if mode == "edge_ev":
        score = float(edge + (0.25 * ev))
    elif mode == "edge_prob":
        score = float(edge * model_probability)
    else:
        score = float(edge)
    if market_type == "f5_rl":
        score -= float(rl_penalty)
    return score


def _resolve_stake(
    *,
    bankroll: float,
    decision: dict[str, Any],
    params: StrategyParams,
    peak_bankroll: float,
) -> float:
    if bankroll <= 0:
        return 0.0
    if params.staking_mode == "flat":
        return min(float(params.flat_bet_size_units), bankroll)
    if params.staking_mode == "edge_scaled":
        effective_edge = max(
            0.0,
            float(decision["edge"])
            - (
                float(params.ml_edge_threshold)
                if decision["market_type"] == "f5_ml"
                else float(params.rl_edge_threshold)
            ),
        )
        scale = min(1.0, effective_edge / float(params.edge_scale_cap))
        units = float(params.min_bet_size_units + ((params.max_bet_size_units - params.min_bet_size_units) * scale))
        return min(units, bankroll)
    return float(
        calculate_kelly_stake(
            bankroll=bankroll,
            model_probability=float(decision["model_probability"]),
            odds=int(decision["odds"]),
            peak_bankroll=peak_bankroll,
        ).stake
    )


def _settle(decision: dict[str, Any]) -> tuple[str, float]:
    if int(decision.get("is_push", 0)) == 1:
        return "PUSH", 0.0
    if int(decision["actual_cover"]) == 1:
        return "WIN", float(payout_for_american_odds(int(decision["odds"])))
    return "LOSS", -1.0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sweep ML-only, RL-only, and mixed strategy selectors")
    parser.add_argument("--ml-training-data", default="data/training/training_data_2018_2025.parquet")
    parser.add_argument("--rl-training-data", default="data/training/training_data_2018_2025_direct_rl.parquet")
    parser.add_argument("--model-dir", default="data/models")
    parser.add_argument("--db-path", default="data/mlb.db")
    parser.add_argument("--book-name", default="sbr:caesars")
    parser.add_argument("--start-year", type=int, default=2021)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--trial-count", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--evaluation-mode",
        choices=["nested_walk_forward", "in_sample_research"],
        default="nested_walk_forward",
    )
    parser.add_argument("--min-training-seasons", type=int, default=1)
    parser.add_argument(
        "--strategy-mode-filter",
        nargs="+",
        choices=["ml_only", "rl_only", "mixed"],
    )
    parser.add_argument("--skip-baselines", action="store_true")
    args = parser.parse_args(argv)

    result = run_strategy_sweep(
        ml_training_data=args.ml_training_data,
        rl_training_data=args.rl_training_data,
        model_dir=args.model_dir,
        db_path=args.db_path,
        book_name=args.book_name,
        start_year=args.start_year,
        end_year=args.end_year,
        trial_count=args.trial_count,
        output_dir=args.output_dir,
        seed=args.seed,
        strategy_mode_filter=args.strategy_mode_filter,
        include_baselines=not args.skip_baselines,
        evaluation_mode=args.evaluation_mode,
        min_training_seasons=args.min_training_seasons,
    )
    summary = result["summary"]
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
