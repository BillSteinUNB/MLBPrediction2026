from __future__ import annotations

import os
import subprocess
from datetime import date as date_type
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from src.dashboard.schemas import MacSyncResponse, SlateResponse
from src.engine.bankroll import calculate_kelly_stake
from src.engine.edge_calculator import calculate_edge
from src.model.score_pricing import spread_outcome_probabilities, totals_outcome_probabilities
from src.ops.live_release import live_release_metadata
from src.ops.live_season_tracker import list_tracked_games
from src.pipeline.daily import (
    SCRAPER_ODDS_DB_PATH,
    _load_scraper_full_game_odds_context,
    load_cached_slate_response,
    run_daily_pipeline,
)

router = APIRouter(prefix="/api/slate", tags=["slate"])

DEFAULT_FULL_GAME_HOME_STD = 3.13
DEFAULT_FULL_GAME_AWAY_STD = 3.36


def _recompute_slate_summary(payload: dict) -> None:
    games = list(payload.get("games") or [])
    pick_count = sum(1 for game in games if game.get("selected_decision"))
    error_count = sum(1 for game in games if game.get("status") == "error")
    no_pick_count = sum(1 for game in games if game.get("status") == "no_pick")
    if pick_count > 0:
        notification_type = "picks"
    elif error_count > 0:
        notification_type = "failure_alert"
    else:
        notification_type = "no_picks"
    payload["pick_count"] = pick_count
    payload["error_count"] = error_count
    payload["no_pick_count"] = no_pick_count
    payload["notification_type"] = notification_type


def _mac_sync_enabled() -> bool:
    return os.getenv("MAC_SYNC_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}


def _merge_missing_input_status_fields(
    payload: dict,
    *,
    pipeline_date: str,
    db_path: str,
) -> None:
    try:
        target_date = date_type.fromisoformat(pipeline_date)
    except ValueError:
        return

    full_game_context = _load_scraper_full_game_odds_context(
        target_date=target_date,
        scraper_db_path=SCRAPER_ODDS_DB_PATH,
        repo_db_path=db_path,
    )
    if not full_game_context:
        return

    for game in payload.get("games") or []:
        game_pk = int(game.get("game_pk") or 0)
        context = full_game_context.get(game_pk)
        if not context:
            continue
        input_status = dict(game.get("input_status") or {})
        for key, value in context.items():
            if input_status.get(key) is None and value is not None:
                input_status[key] = value
        if input_status:
            game["input_status"] = input_status


def _candidate_exists(
    game: dict,
    *,
    market_type: str,
    book_name: str,
) -> bool:
    for decision in game.get("candidate_decisions") or []:
        if decision.get("market_type") == market_type and decision.get("book_name") == book_name:
            return True
    return False


def _serialize_decision(decision, *, kelly_stake: float) -> dict:
    return {
        "game_pk": int(decision.game_pk),
        "market_type": decision.market_type,
        "side": decision.side,
        "source_model": decision.source_model,
        "source_model_version": decision.source_model_version,
        "book_name": decision.book_name,
        "model_probability": float(decision.model_probability),
        "fair_probability": float(decision.fair_probability),
        "edge_pct": float(decision.edge_pct),
        "ev": float(decision.ev),
        "is_positive_ev": bool(decision.is_positive_ev),
        "kelly_stake": float(kelly_stake),
        "odds_at_bet": decision.odds_at_bet,
        "line_at_bet": decision.line_at_bet,
        "result": "PENDING",
        "settled_at": None,
        "profit_loss": None,
    }


def _append_missing_bet365_candidates(payload: dict) -> None:
    for game in payload.get("games") or []:
        prediction = game.get("prediction") or {}
        input_status = game.get("input_status") or {}
        game_pk = int(game.get("game_pk") or 0)
        if game_pk <= 0:
            continue

        candidates = list(game.get("candidate_decisions") or [])

        home_runs = prediction.get("projected_full_game_home_runs")
        away_runs = prediction.get("projected_full_game_away_runs")
        home_ml_prob = prediction.get("full_game_ml_home_prob")
        away_ml_prob = prediction.get("full_game_ml_away_prob")

        if (
            not _candidate_exists(game, market_type="full_game_ml", book_name="bet365")
            and home_ml_prob is not None
            and away_ml_prob is not None
            and input_status.get("bet365_full_game_home_ml") is not None
            and input_status.get("bet365_full_game_away_ml") is not None
        ):
            home_decision = calculate_edge(
                game_pk=game_pk,
                market_type="full_game_ml",
                side="home",
                model_probability=float(home_ml_prob),
                home_odds=int(input_status["bet365_full_game_home_ml"]),
                away_odds=int(input_status["bet365_full_game_away_ml"]),
                book_name="bet365",
            )
            away_decision = calculate_edge(
                game_pk=game_pk,
                market_type="full_game_ml",
                side="away",
                model_probability=float(away_ml_prob),
                home_odds=int(input_status["bet365_full_game_home_ml"]),
                away_odds=int(input_status["bet365_full_game_away_ml"]),
                book_name="bet365",
            )
            home_kelly = calculate_kelly_stake(100.0, decision=home_decision).stake
            away_kelly = calculate_kelly_stake(100.0, decision=away_decision).stake
            candidates.append(_serialize_decision(home_decision, kelly_stake=home_kelly))
            candidates.append(_serialize_decision(away_decision, kelly_stake=away_kelly))

        if (
            not _candidate_exists(game, market_type="full_game_rl", book_name="bet365")
            and home_runs is not None
            and away_runs is not None
            and input_status.get("bet365_full_game_home_spread") is not None
            and input_status.get("bet365_full_game_home_spread_odds") is not None
            and input_status.get("bet365_full_game_away_spread") is not None
            and input_status.get("bet365_full_game_away_spread_odds") is not None
        ):
            home_prob, away_prob, push_prob = spread_outcome_probabilities(
                home_runs_mean=float(home_runs),
                away_runs_mean=float(away_runs),
                home_runs_std=DEFAULT_FULL_GAME_HOME_STD,
                away_runs_std=DEFAULT_FULL_GAME_AWAY_STD,
                home_point=float(input_status["bet365_full_game_home_spread"]),
            )
            if home_prob is not None and away_prob is not None:
                home_decision = calculate_edge(
                    game_pk=game_pk,
                    market_type="full_game_rl",
                    side="home",
                    model_probability=float(home_prob),
                    home_odds=int(input_status["bet365_full_game_home_spread_odds"]),
                    away_odds=int(input_status["bet365_full_game_away_spread_odds"]),
                    home_point=float(input_status["bet365_full_game_home_spread"]),
                    away_point=float(input_status["bet365_full_game_away_spread"]),
                    book_name="bet365",
                    push_probability=float(push_prob or 0.0),
                )
                away_decision = calculate_edge(
                    game_pk=game_pk,
                    market_type="full_game_rl",
                    side="away",
                    model_probability=float(away_prob),
                    home_odds=int(input_status["bet365_full_game_home_spread_odds"]),
                    away_odds=int(input_status["bet365_full_game_away_spread_odds"]),
                    home_point=float(input_status["bet365_full_game_home_spread"]),
                    away_point=float(input_status["bet365_full_game_away_spread"]),
                    book_name="bet365",
                    push_probability=float(push_prob or 0.0),
                )
                home_kelly = calculate_kelly_stake(100.0, decision=home_decision).stake
                away_kelly = calculate_kelly_stake(100.0, decision=away_decision).stake
                candidates.append(_serialize_decision(home_decision, kelly_stake=home_kelly))
                candidates.append(_serialize_decision(away_decision, kelly_stake=away_kelly))

        if (
            not _candidate_exists(game, market_type="full_game_total", book_name="bet365")
            and home_runs is not None
            and away_runs is not None
            and input_status.get("bet365_full_game_total") is not None
            and input_status.get("bet365_full_game_total_over_odds") is not None
            and input_status.get("bet365_full_game_total_under_odds") is not None
        ):
            over_prob, under_prob, push_prob = totals_outcome_probabilities(
                home_runs_mean=float(home_runs),
                away_runs_mean=float(away_runs),
                home_runs_std=DEFAULT_FULL_GAME_HOME_STD,
                away_runs_std=DEFAULT_FULL_GAME_AWAY_STD,
                total_point=float(input_status["bet365_full_game_total"]),
            )
            if over_prob is not None and under_prob is not None:
                over_decision = calculate_edge(
                    game_pk=game_pk,
                    market_type="full_game_total",
                    side="over",
                    model_probability=float(over_prob),
                    home_odds=int(input_status["bet365_full_game_total_over_odds"]),
                    away_odds=int(input_status["bet365_full_game_total_under_odds"]),
                    home_point=float(input_status["bet365_full_game_total"]),
                    away_point=float(input_status["bet365_full_game_total"]),
                    book_name="bet365",
                    push_probability=float(push_prob or 0.0),
                )
                under_decision = calculate_edge(
                    game_pk=game_pk,
                    market_type="full_game_total",
                    side="under",
                    model_probability=float(under_prob),
                    home_odds=int(input_status["bet365_full_game_total_over_odds"]),
                    away_odds=int(input_status["bet365_full_game_total_under_odds"]),
                    home_point=float(input_status["bet365_full_game_total"]),
                    away_point=float(input_status["bet365_full_game_total"]),
                    book_name="bet365",
                    push_probability=float(push_prob or 0.0),
                )
                over_kelly = calculate_kelly_stake(100.0, decision=over_decision).stake
                under_kelly = calculate_kelly_stake(100.0, decision=under_decision).stake
                candidates.append(_serialize_decision(over_decision, kelly_stake=over_kelly))
                candidates.append(_serialize_decision(under_decision, kelly_stake=under_kelly))

        if candidates:
            game["candidate_decisions"] = candidates


def _overlay_tracked_pick_state(
    payload: dict,
    *,
    pipeline_date: str,
    db_path: str,
) -> None:
    season = date_type.fromisoformat(pipeline_date).year
    tracked_rows = list_tracked_games(
        season=season,
        pipeline_date=pipeline_date,
        db_path=db_path,
    )
    if not tracked_rows:
        return

    tracked_by_game_pk = {int(row["game_pk"]): row for row in tracked_rows}
    for game in payload.get("games") or []:
        row = tracked_by_game_pk.get(int(game.get("game_pk") or 0))
        if row is None:
            continue
        game["status"] = row.get("status") or game.get("status")
        game["no_pick_reason"] = row.get("no_pick_reason")
        if row.get("selected_market_type"):
            game["selected_decision"] = {
                "game_pk": int(row["game_pk"]),
                "market_type": row.get("selected_market_type"),
                "side": row.get("selected_side"),
                "source_model": row.get("source_model"),
                "source_model_version": row.get("source_model_version"),
                "book_name": row.get("book_name"),
                "model_probability": row.get("model_probability"),
                "fair_probability": row.get("fair_probability"),
                "edge_pct": row.get("edge_pct"),
                "ev": row.get("ev"),
                "is_positive_ev": bool((row.get("edge_pct") or 0.0) >= 0.03),
                "kelly_stake": row.get("kelly_stake"),
                "odds_at_bet": row.get("odds_at_bet"),
                "line_at_bet": row.get("line_at_bet"),
                "result": row.get("settled_result") or "PENDING",
                "settled_at": row.get("settled_at"),
                "profit_loss": row.get("flat_profit_loss"),
            }
        else:
            game["selected_decision"] = None
        if row.get("forced_market_type"):
            game["forced_decision"] = {
                "game_pk": int(row["game_pk"]),
                "market_type": row.get("forced_market_type"),
                "side": row.get("forced_side"),
                "source_model": row.get("forced_source_model"),
                "source_model_version": row.get("forced_source_model_version"),
                "book_name": row.get("forced_book_name"),
                "model_probability": row.get("forced_model_probability"),
                "fair_probability": row.get("forced_fair_probability"),
                "edge_pct": row.get("forced_edge_pct"),
                "ev": row.get("forced_ev"),
                "is_positive_ev": bool((row.get("forced_edge_pct") or 0.0) >= 0.03),
                "kelly_stake": row.get("forced_kelly_stake"),
                "odds_at_bet": row.get("forced_odds_at_bet"),
                "line_at_bet": row.get("forced_line_at_bet"),
                "result": row.get("forced_settled_result") or "PENDING",
                "settled_at": row.get("settled_at"),
                "profit_loss": row.get("forced_flat_profit_loss"),
            }
        elif row.get("selected_market_type") is None:
            game["forced_decision"] = None


@router.get("", response_model=SlateResponse)
async def get_slate(
    pipeline_date: str | None = Query(
        default=None,
        description="Target slate date in YYYY-MM-DD format. Defaults to today.",
    ),
    db_path: str = Query(
        default=str(Path("data") / "mlb.db"),
        description="SQLite path used by the daily pipeline.",
    ),
    refresh: bool = Query(
        default=False,
        description="When true, rebuild the slate instead of using the cached response.",
    ),
) -> SlateResponse:
    resolved_date = pipeline_date or date_type.today().isoformat()
    payload = None
    if not refresh:
        payload = load_cached_slate_response(
            pipeline_date=resolved_date,
            mode="prod",
            dry_run=True,
            db_path=db_path,
        )
        if payload is None:
            payload = load_cached_slate_response(
                pipeline_date=resolved_date,
                mode="backtest",
                dry_run=True,
                db_path=db_path,
            )
    if payload is None and not refresh:
        raise HTTPException(
            status_code=404,
            detail="No cached slate available. Pull today's slate first.",
        )
    if payload is None:
        result = run_daily_pipeline(
            target_date=resolved_date,
            mode="prod",
            dry_run=True,
            db_path=db_path,
        )
        payload = result.to_dict()
        payload.pop("notification_payload", None)
    _merge_missing_input_status_fields(
        payload,
        pipeline_date=resolved_date,
        db_path=db_path,
    )
    _append_missing_bet365_candidates(payload)
    _overlay_tracked_pick_state(
        payload,
        pipeline_date=resolved_date,
        db_path=db_path,
    )
    _recompute_slate_summary(payload)
    payload.update(live_release_metadata())
    return SlateResponse.model_validate(payload)


@router.post("/pull-from-mac", response_model=MacSyncResponse)
async def pull_from_mac(
    pipeline_date: str | None = Query(
        default=None,
        description="Target slate date in YYYY-MM-DD format. Defaults to today.",
    ),
) -> MacSyncResponse:
    if not _mac_sync_enabled():
        raise HTTPException(
            status_code=503,
            detail="Mac sync is currently disabled on this host.",
        )

    sync_script = Path("scripts") / "sync_mac_odds_and_refresh.ps1"
    if not sync_script.exists():
        raise HTTPException(status_code=500, detail=f"Sync script not found: {sync_script}")

    mac_host = os.getenv("MAC_SYNC_HOST")
    mac_user = os.getenv("MAC_SYNC_USER", "bill")
    remote_repo_path = os.getenv("MAC_SYNC_REMOTE_REPO_PATH", "/Users/bill/Code/MLBTracker")
    if not mac_host:
        raise HTTPException(
            status_code=400,
            detail="MAC_SYNC_HOST is not configured on the backend host.",
        )

    resolved_date = pipeline_date or date_type.today().isoformat()
    command = [
        "powershell",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(sync_script),
        "-MacHost",
        mac_host,
        "-MacUser",
        mac_user,
        "-RemoteRepoPath",
        remote_repo_path,
        "-PipelineDate",
        resolved_date,
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=900,
        cwd=Path.cwd(),
    )
    output = "\n".join(
        part.strip()
        for part in (completed.stdout, completed.stderr)
        if part and part.strip()
    ).strip()
    if completed.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=output or "Mac sync failed.",
        )
    return MacSyncResponse(
        ok=True,
        message="Pulled latest Mac data and refreshed local slate.",
        output=output or None,
    )
