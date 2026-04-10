from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import joblib
import pandas as pd

from src.clients.historical_odds_client import load_historical_odds_for_games
from src.model.data_builder import validate_run_count_training_data
from src.model.run_count_trainer import BlendedRunCountRegressor
from src.model.xgboost_trainer import _load_training_dataframe
from src.ops.run_count_bankroll_playoff import (
    DEFAULT_FLAT_BET_SIZE_UNITS,
    DEFAULT_STARTING_BANKROLL_UNITS,
    TRUSTED_BANKROLL_MARKET_TYPES,
    _build_market_bets_for_row,
    _calculate_drawdown_pct,
    _coerce_finite_float,
    _predict_bundle,
    _settle_market_bet,
    _summarize_bankroll_bets,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FAST_BANKROLL_OUTPUT_DIR = Path("data/reports/run_count/fast_bankroll")
DEFAULT_TRACKER_LATEST_JSON = Path("data/reports/run_count/tracker/latest_run.json")
_COMPANION_MODEL_NAMES = (
    "full_game_home_runs_model",
    "f5_home_runs_model",
    "f5_away_runs_model",
)
_MARKETS_REQUIRING_COMPANIONS = {
    "full_game_ml": ("full_game_home_runs_model",),
    "full_game_rl": ("full_game_home_runs_model",),
    "full_game_total": ("full_game_home_runs_model",),
    "f5_ml": ("f5_home_runs_model", "f5_away_runs_model"),
    "f5_total": ("f5_home_runs_model", "f5_away_runs_model"),
}


@dataclass(frozen=True, slots=True)
class _CompanionBundle:
    model_name: str
    model: Any
    feature_columns: list[str]
    rmse: float
    metadata_path: Path
    model_version: str


@dataclass(frozen=True, slots=True)
class FastBankrollCheckResult:
    summary_json_path: Path
    bets_csv_path: Path
    summary: dict[str, Any]


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(_resolve_path(path).read_text(encoding="utf-8"))


def _load_tracker_context(latest_tracker_json: str | Path) -> dict[str, Any]:
    return _read_json(latest_tracker_json)


def _stage4_metadata_from_tracker_payload(payload: dict[str, Any]) -> dict[str, Any]:
    stage4_report_path = payload["artifacts"]["stage4_report_path"]
    stage4_report = _read_json(stage4_report_path)
    stage4_metadata_path = stage4_report["artifact_path"]
    stage4_metadata = _read_json(stage4_metadata_path)
    stage4_metadata["_resolved_metadata_path"] = str(_resolve_path(stage4_metadata_path))
    return stage4_metadata


def _infer_joblib_path(metadata_path: Path, metadata_payload: dict[str, Any]) -> Path:
    artifact = metadata_payload.get("artifact")
    if isinstance(artifact, dict) and artifact.get("model_path"):
        return _resolve_path(artifact["model_path"])
    if metadata_payload.get("model_path"):
        return _resolve_path(metadata_payload["model_path"])
    return metadata_path.with_name(metadata_path.name.replace(".metadata.json", ".joblib"))


def _load_structured_joblib_model(joblib_path: Path) -> Any:
    try:
        return joblib.load(joblib_path)
    except AttributeError as exc:
        if "BlendedRunCountRegressor" not in str(exc):
            raise
        main_module = sys.modules.get("__main__")
        if main_module is not None and not hasattr(main_module, "BlendedRunCountRegressor"):
            setattr(main_module, "BlendedRunCountRegressor", BlendedRunCountRegressor)
        return joblib.load(joblib_path)


def _load_companion_bundle(metadata_path: str | Path) -> _CompanionBundle:
    resolved_metadata_path = _resolve_path(metadata_path)
    payload = json.loads(resolved_metadata_path.read_text(encoding="utf-8"))
    artifact_payload = payload.get("artifact", payload)
    joblib_path = _infer_joblib_path(resolved_metadata_path, payload)
    return _CompanionBundle(
        model_name=str(artifact_payload["model_name"]),
        model=_load_structured_joblib_model(joblib_path),
        feature_columns=list(artifact_payload["feature_columns"]),
        rmse=float(artifact_payload["holdout_metrics"]["rmse"]),
        metadata_path=resolved_metadata_path,
        model_version=str(artifact_payload["model_version"]),
    )


def _resolve_latest_companion_metadata(*, holdout_season: int, model_name: str) -> Path | None:
    latest_path: Path | None = None
    latest_mtime = -1.0
    for path in (PROJECT_ROOT / "data" / "models").rglob("*.metadata.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("model_name") != model_name:
            continue
        if int(payload.get("holdout_season", -1)) != int(holdout_season):
            continue
        mtime = path.stat().st_mtime
        if mtime > latest_mtime:
            latest_mtime = mtime
            latest_path = path.resolve()
    return latest_path


def _resolve_companion_bundles(
    *,
    holdout_season: int,
    full_game_home_metadata: str | Path | None,
    f5_home_metadata: str | Path | None,
    f5_away_metadata: str | Path | None,
) -> dict[str, _CompanionBundle]:
    resolved_paths = {
        "full_game_home_runs_model": full_game_home_metadata,
        "f5_home_runs_model": f5_home_metadata,
        "f5_away_runs_model": f5_away_metadata,
    }
    bundles: dict[str, _CompanionBundle] = {}
    for model_name, explicit_path in resolved_paths.items():
        candidate_path = (
            _resolve_path(explicit_path)
            if explicit_path is not None
            else _resolve_latest_companion_metadata(holdout_season=holdout_season, model_name=model_name)
        )
        if candidate_path is None or not candidate_path.exists():
            continue
        bundle = _load_companion_bundle(candidate_path)
        bundles[bundle.model_name] = bundle
    return bundles


def _auto_supported_markets(companion_bundles: dict[str, _CompanionBundle]) -> list[str]:
    selected: list[str] = []
    for market_type in TRUSTED_BANKROLL_MARKET_TYPES:
        required = _MARKETS_REQUIRING_COMPANIONS[market_type]
        if all(model_name in companion_bundles for model_name in required):
            selected.append(market_type)
    return selected


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


def _load_market_frames(
    *,
    historical_odds_db: str | Path,
    games_frame: pd.DataFrame,
    market_types: Sequence[str],
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for market_type in market_types:
        frames[market_type] = load_historical_odds_for_games(
            db_path=str(_resolve_path(historical_odds_db)),
            games_frame=games_frame,
            market_type=str(market_type),
            snapshot_selection="opening",
        )
    return frames


def run_fast_bankroll_check(
    *,
    latest_tracker_json: str | Path = DEFAULT_TRACKER_LATEST_JSON,
    historical_odds_db: str | Path | None = None,
    full_game_home_metadata: str | Path | None = None,
    f5_home_metadata: str | Path | None = None,
    f5_away_metadata: str | Path | None = None,
    markets: Sequence[str] | None = None,
    output_dir: str | Path = DEFAULT_FAST_BANKROLL_OUTPUT_DIR,
    starting_bankroll_units: float = DEFAULT_STARTING_BANKROLL_UNITS,
    flat_bet_size_units: float = DEFAULT_FLAT_BET_SIZE_UNITS,
    max_games: int | None = None,
) -> FastBankrollCheckResult:
    tracker_payload = _load_tracker_context(latest_tracker_json)
    workflow_config = tracker_payload["workflow_config"]
    stage4_metadata = _stage4_metadata_from_tracker_payload(tracker_payload)
    holdout_season = int(workflow_config["holdout_season"])
    start_year = int(workflow_config["start_year"])
    run_label = str(tracker_payload["run_label"])

    companion_bundles = _resolve_companion_bundles(
        holdout_season=holdout_season,
        full_game_home_metadata=full_game_home_metadata,
        f5_home_metadata=f5_home_metadata,
        f5_away_metadata=f5_away_metadata,
    )
    if not companion_bundles:
        raise ValueError("No companion model metadata could be resolved for the requested holdout season.")

    selected_markets = list(markets) if markets else _auto_supported_markets(companion_bundles)
    if not selected_markets:
        raise ValueError("No bankroll markets are supported by the resolved companion models.")

    training_data_path = workflow_config["training_data_path"]
    resolved_historical_odds_db = historical_odds_db or workflow_config.get("historical_odds_db")
    if not resolved_historical_odds_db:
        raise ValueError("A historical odds DB path is required.")

    holdout_frame = _load_and_filter_training_data(
        training_data_path=training_data_path,
        start_year=start_year,
        end_year=holdout_season,
    )
    holdout_frame = holdout_frame.loc[
        pd.to_numeric(holdout_frame["season"], errors="coerce") == int(holdout_season)
    ].copy()
    if holdout_frame.empty:
        raise ValueError(f"No holdout rows found for season {holdout_season}.")
    holdout_frame = holdout_frame.sort_values(["game_date", "game_pk"]).reset_index(drop=True)
    if max_games is not None:
        holdout_frame = holdout_frame.head(int(max_games)).copy()

    predictions_csv = _resolve_path(stage4_metadata["output_paths"]["predictions_csv"])
    stage4_predictions = pd.read_csv(predictions_csv)
    stage4_predictions = stage4_predictions.loc[
        stage4_predictions["game_pk"].isin(holdout_frame["game_pk"].tolist())
    ].copy()
    if stage4_predictions.empty:
        raise ValueError("No Stage 4 predictions matched the requested holdout frame.")

    merged_frame = holdout_frame.merge(
        stage4_predictions[["game_pk", "mcmc_expected_away_runs", "distribution_stddev"]],
        on="game_pk",
        how="inner",
    ).copy()
    if merged_frame.empty:
        raise ValueError("No merged bankroll rows matched the Stage 4 predictions.")

    if "full_game_home_runs_model" in companion_bundles:
        merged_frame["full_game_home_pred"] = _predict_bundle(
            companion_bundles["full_game_home_runs_model"],
            merged_frame,
        )
    if "f5_home_runs_model" in companion_bundles:
        merged_frame["f5_home_pred"] = _predict_bundle(companion_bundles["f5_home_runs_model"], merged_frame)
    if "f5_away_runs_model" in companion_bundles:
        merged_frame["f5_away_pred"] = _predict_bundle(companion_bundles["f5_away_runs_model"], merged_frame)
    merged_frame["full_game_away_std"] = pd.to_numeric(
        merged_frame["distribution_stddev"],
        errors="coerce",
    ).fillna(_coerce_finite_float(stage4_metadata["mean_metrics"]["rmse"]) or 0.0)

    market_frames = _load_market_frames(
        historical_odds_db=resolved_historical_odds_db,
        games_frame=merged_frame,
        market_types=selected_markets,
    )

    bankroll = float(starting_bankroll_units)
    peak_bankroll = float(starting_bankroll_units)
    bets: list[dict[str, Any]] = []
    for _, row in merged_frame.iterrows():
        game_market_rows = [
            market_row
            for market_type, market_frame in market_frames.items()
            for market_row in market_frame.loc[market_frame["game_pk"] == int(row["game_pk"])].to_dict(orient="records")
            if market_type in selected_markets
        ]
        game_market_rows.sort(key=lambda payload: (str(payload.get("market_type")), str(payload.get("book_name") or "")))
        for market_row in game_market_rows:
            decisions = _build_market_bets_for_row(
                row=row,
                market_row=market_row,
                companion_models=companion_bundles,  # type: ignore[arg-type]
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
                        "candidate_label": run_label,
                        "holdout_season": holdout_season,
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
                        "stage4_model_version": str(stage4_metadata["model_version"]),
                    }
                )
                bets.append(decision)

    bets_frame = pd.DataFrame(bets)
    resolved_output_dir = _resolve_path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    slug = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in run_label).strip("_")
    candidate_dir = resolved_output_dir / f"{slug}_{holdout_season}_{stamp}"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    bets_csv_path = candidate_dir / "bankroll_bets.csv"
    summary_json_path = candidate_dir / "bankroll_summary.json"
    if not bets_frame.empty:
        bets_frame = bets_frame.sort_values(["game_date", "game_pk", "market_type", "side"]).reset_index(drop=True)
        bets_frame.to_csv(bets_csv_path, index=False)

    summary = _summarize_bankroll_bets(
        bets_frame=bets_frame,
        candidate_label=run_label,
        holdout_season=holdout_season,
        starting_bankroll_units=float(starting_bankroll_units),
        stage4_metadata=stage4_metadata,
    )
    summary["historical_odds_db"] = str(_resolve_path(resolved_historical_odds_db))
    summary["markets"] = list(selected_markets)
    summary["companion_model_versions"] = {
        name: bundle.model_version for name, bundle in sorted(companion_bundles.items())
    }
    summary_json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return FastBankrollCheckResult(
        summary_json_path=summary_json_path,
        bets_csv_path=bets_csv_path,
        summary=summary,
    )


__all__ = [
    "DEFAULT_FAST_BANKROLL_OUTPUT_DIR",
    "FastBankrollCheckResult",
    "run_fast_bankroll_check",
]
