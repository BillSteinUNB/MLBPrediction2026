from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from src.clients.odds_client import devig_probabilities
from src.engine.bankroll import calculate_kelly_stake
from src.engine.edge_calculator import payout_for_american_odds
from src.model.direct_rl_trainer import _augment_direct_rl_training_frame, _prepare_direct_rl_frame
from src.model.margin_pricing import margin_to_cover_probability
from src.model.xgboost_trainer import _load_training_dataframe


DEFAULT_BLEND_WEIGHTS = tuple(round(weight, 2) for weight in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])


@dataclass(frozen=True, slots=True)
class StrategyResult:
    total_games: int
    picks: int
    wins: int
    losses: int
    pushes: int
    roi: float | None
    profit_units: float
    ending_bankroll: float
    play_of_day_picks: int
    play_of_day_roi: float | None
    play_of_day_profit_units: float
    forced_picks: int
    forced_roi: float | None
    forced_profit_units: float
    max_drawdown_pct: float


def evaluate_rl_v2_models(
    *,
    training_data: str | Path,
    direct_model_path: str | Path,
    direct_metadata_path: str | Path,
    margin_model_path: str | Path,
    margin_metadata_path: str | Path,
    holdout_season: int = 2025,
    edge_threshold: float = 0.05,
    starting_bankroll: float = 100.0,
    blend_weights: Sequence[float] = DEFAULT_BLEND_WEIGHTS,
) -> dict[str, Any]:
    dataframe = _augment_direct_rl_training_frame(_load_training_dataframe(training_data))
    frame, _ = _prepare_direct_rl_frame(dataframe, target_column="home_cover_at_posted_line")
    holdout_frame = frame.loc[frame["season"] == int(holdout_season)].copy()
    if holdout_frame.empty:
        raise ValueError(f"No RL holdout rows found for season {holdout_season}")

    direct_model = joblib.load(direct_model_path)
    margin_model = joblib.load(margin_model_path)
    direct_metadata = json.loads(Path(direct_metadata_path).read_text(encoding="utf-8"))
    margin_metadata = json.loads(Path(margin_metadata_path).read_text(encoding="utf-8"))

    direct_features = list(direct_metadata["feature_columns"])
    margin_features = list(margin_metadata["feature_columns"])
    residual_std = float(margin_metadata["residual_std"])

    direct_probabilities = direct_model.predict_proba(holdout_frame[direct_features])[:, 1]
    predicted_margin = margin_model.predict(holdout_frame[margin_features])
    margin_probabilities = [
        margin_to_cover_probability(
            predicted_margin=float(margin),
            home_point=None if pd.isna(home_point) else float(home_point),
            residual_std=residual_std,
        )
        for margin, home_point in zip(
            predicted_margin,
            holdout_frame["posted_f5_rl_home_point"],
            strict=False,
        )
    ]
    margin_series = pd.Series(margin_probabilities, index=holdout_frame.index, dtype=float).fillna(0.5)
    outcomes = holdout_frame["home_cover_at_posted_line"].astype(int)

    evaluations: dict[str, Any] = {
        "holdout_season": int(holdout_season),
        "row_count": int(len(holdout_frame)),
        "direct": _evaluate_probability_series(
            holdout_frame=holdout_frame,
            probabilities=pd.Series(direct_probabilities, index=holdout_frame.index, dtype=float),
            outcomes=outcomes,
            edge_threshold=edge_threshold,
            starting_bankroll=starting_bankroll,
        ),
        "margin": _evaluate_probability_series(
            holdout_frame=holdout_frame,
            probabilities=margin_series,
            outcomes=outcomes,
            edge_threshold=edge_threshold,
            starting_bankroll=starting_bankroll,
        ),
        "blend_weights": [],
    }

    blend_results: list[dict[str, Any]] = []
    best_weight = None
    best_score = None
    for weight in blend_weights:
        probabilities = (
            float(weight) * pd.Series(direct_probabilities, index=holdout_frame.index, dtype=float)
            + (1.0 - float(weight)) * margin_series
        )
        evaluated = _evaluate_probability_series(
            holdout_frame=holdout_frame,
            probabilities=probabilities,
            outcomes=outcomes,
            edge_threshold=edge_threshold,
            starting_bankroll=starting_bankroll,
        )
        row = {"direct_weight": float(weight), **evaluated}
        blend_results.append(row)
        score = (
            float(row["strategy"]["profit_units"]),
            float(row["metrics"]["roc_auc"] or 0.0),
            -float(row["strategy"]["max_drawdown_pct"]),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_weight = row

    evaluations["blend_weights"] = blend_results
    evaluations["best_blend"] = best_weight
    return evaluations


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate RL v2 models with bankroll-aware holdout simulation")
    parser.add_argument("--training-data", required=True)
    parser.add_argument("--direct-model-path", required=True)
    parser.add_argument("--direct-metadata-path", required=True)
    parser.add_argument("--margin-model-path", required=True)
    parser.add_argument("--margin-metadata-path", required=True)
    parser.add_argument("--output-path")
    parser.add_argument("--holdout-season", type=int, default=2025)
    parser.add_argument("--edge-threshold", type=float, default=0.05)
    parser.add_argument("--starting-bankroll", type=float, default=100.0)
    args = parser.parse_args(argv)

    payload = evaluate_rl_v2_models(
        training_data=args.training_data,
        direct_model_path=args.direct_model_path,
        direct_metadata_path=args.direct_metadata_path,
        margin_model_path=args.margin_model_path,
        margin_metadata_path=args.margin_metadata_path,
        holdout_season=args.holdout_season,
        edge_threshold=args.edge_threshold,
        starting_bankroll=args.starting_bankroll,
    )
    if args.output_path:
        Path(args.output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


def _evaluate_probability_series(
    *,
    holdout_frame: pd.DataFrame,
    probabilities: pd.Series,
    outcomes: pd.Series,
    edge_threshold: float,
    starting_bankroll: float,
) -> dict[str, Any]:
    metrics = {
        "accuracy": float(accuracy_score(outcomes, probabilities >= 0.5)),
        "log_loss": float(log_loss(outcomes, probabilities, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(outcomes, probabilities)) if outcomes.nunique() > 1 else None,
        "brier": float(brier_score_loss(outcomes, probabilities)),
    }
    strategy = _simulate_strategies(
        holdout_frame=holdout_frame,
        probabilities=probabilities,
        edge_threshold=edge_threshold,
        starting_bankroll=starting_bankroll,
    )
    return {"metrics": metrics, "strategy": asdict(strategy)}


def _simulate_strategies(
    *,
    holdout_frame: pd.DataFrame,
    probabilities: pd.Series,
    edge_threshold: float,
    starting_bankroll: float,
) -> StrategyResult:
    frame = holdout_frame.copy()
    frame["home_probability"] = probabilities.astype(float)
    frame = frame.sort_values(["scheduled_start", "game_pk"]).reset_index(drop=True)

    bankroll = float(starting_bankroll)
    peak_bankroll = bankroll
    max_drawdown_pct = 0.0
    total_profit = 0.0
    picks = wins = losses = pushes = 0

    forced_profit = 0.0
    forced_picks = 0

    daily_value_candidates: dict[str, list[dict[str, Any]]] = {}
    prepared_rows: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        decision = _best_decision_for_game(row, edge_threshold=edge_threshold, forced=False)
        forced = _best_decision_for_game(row, edge_threshold=edge_threshold, forced=True)
        prepared = {"row": row, "decision": decision, "forced": forced}
        prepared_rows.append(prepared)
        if decision is not None:
            daily_value_candidates.setdefault(str(row["game_date"]), []).append(prepared)

    play_of_day_keys: set[tuple[str, int]] = set()
    for game_date, items in daily_value_candidates.items():
        best = max(
            items,
            key=lambda item: (
                float(item["decision"]["edge"]),
                float(item["decision"]["model_probability"]),
                float(item["decision"]["ev"]),
            ),
        )
        play_of_day_keys.add((game_date, int(best["row"]["game_pk"])))

    play_of_day_profit = 0.0
    play_of_day_picks = 0
    play_of_day_wins = play_of_day_losses = play_of_day_pushes = 0
    play_of_day_bankroll = float(starting_bankroll)
    play_of_day_peak = play_of_day_bankroll

    for item in prepared_rows:
        row = item["row"]
        decision = item["decision"]
        forced = item["forced"]

        if decision is not None:
            stake = calculate_kelly_stake(
                bankroll,
                model_probability=float(decision["model_probability"]),
                odds=int(decision["odds"]),
                peak_bankroll=peak_bankroll,
            ).stake
            result, profit = _settle_decision(decision)
            if stake > 0:
                realized_profit = float(profit * stake)
                bankroll += realized_profit
                peak_bankroll = max(peak_bankroll, bankroll)
                drawdown = ((peak_bankroll - bankroll) / peak_bankroll) if peak_bankroll else 0.0
                max_drawdown_pct = max(max_drawdown_pct, drawdown)
                total_profit += realized_profit
                picks += 1
                wins += int(result == "WIN")
                losses += int(result == "LOSS")
                pushes += int(result == "PUSH")

                if (str(row["game_date"]), int(row["game_pk"])) in play_of_day_keys:
                    pod_stake = calculate_kelly_stake(
                        play_of_day_bankroll,
                        model_probability=float(decision["model_probability"]),
                        odds=int(decision["odds"]),
                        peak_bankroll=play_of_day_peak,
                    ).stake
                    if pod_stake > 0:
                        play_of_day_profit += float(profit * pod_stake)
                        play_of_day_bankroll += float(profit * pod_stake)
                        play_of_day_peak = max(play_of_day_peak, play_of_day_bankroll)
                        play_of_day_picks += 1
                        play_of_day_wins += int(result == "WIN")
                        play_of_day_losses += int(result == "LOSS")
                        play_of_day_pushes += int(result == "PUSH")

        if forced is not None:
            forced_profit += float(_settle_decision(forced)[1])
            forced_picks += 1

    roi = (total_profit / picks) if picks else None
    play_of_day_roi = (play_of_day_profit / play_of_day_picks) if play_of_day_picks else None
    forced_roi = (forced_profit / forced_picks) if forced_picks else None
    return StrategyResult(
        total_games=int(len(frame)),
        picks=int(picks),
        wins=int(wins),
        losses=int(losses),
        pushes=int(pushes),
        roi=float(roi) if roi is not None else None,
        profit_units=float(total_profit),
        ending_bankroll=float(bankroll),
        play_of_day_picks=int(play_of_day_picks),
        play_of_day_roi=float(play_of_day_roi) if play_of_day_roi is not None else None,
        play_of_day_profit_units=float(play_of_day_profit),
        forced_picks=int(forced_picks),
        forced_roi=float(forced_roi) if forced_roi is not None else None,
        forced_profit_units=float(forced_profit),
        max_drawdown_pct=float(max_drawdown_pct),
    )


def _best_decision_for_game(
    row: dict[str, Any],
    *,
    edge_threshold: float,
    forced: bool,
) -> dict[str, Any] | None:
    home_probability = float(row["home_probability"])
    away_probability = 1.0 - home_probability
    home_odds = int(row["posted_f5_rl_home_odds"])
    away_odds = int(row["posted_f5_rl_away_odds"])
    home_fair, away_fair = devig_probabilities(home_odds, away_odds)
    home_ev = (home_probability * payout_for_american_odds(home_odds)) - (1.0 - home_probability)
    away_ev = (away_probability * payout_for_american_odds(away_odds)) - (1.0 - away_probability)
    candidates = [
        {
            "side": "home",
            "model_probability": home_probability,
            "fair_probability": float(home_fair),
            "edge": float(home_probability - home_fair),
            "ev": float(home_ev),
            "odds": home_odds,
            "line": None if pd.isna(row["posted_f5_rl_home_point"]) else float(row["posted_f5_rl_home_point"]),
            "actual_cover": int(row["home_cover_at_posted_line"]),
        },
        {
            "side": "away",
            "model_probability": away_probability,
            "fair_probability": float(away_fair),
            "edge": float(away_probability - away_fair),
            "ev": float(away_ev),
            "odds": away_odds,
            "line": None if pd.isna(row["posted_f5_rl_away_point"]) else float(row["posted_f5_rl_away_point"]),
            "actual_cover": int(1 - int(row["home_cover_at_posted_line"])),
        },
    ]
    candidates.sort(key=lambda candidate: (candidate["edge"], candidate["model_probability"], candidate["ev"]), reverse=True)
    if forced:
        return candidates[0]
    if candidates[0]["edge"] < float(edge_threshold):
        return None
    return candidates[0]


def _settle_decision(decision: dict[str, Any]) -> tuple[str, float]:
    if int(decision["actual_cover"]) == 1:
        return "WIN", float(payout_for_american_odds(int(decision["odds"])))
    return "LOSS", -1.0


if __name__ == "__main__":
    raise SystemExit(main())
