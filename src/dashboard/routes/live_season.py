from __future__ import annotations

from datetime import date as date_type
from datetime import datetime, time, timedelta
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from playwright.async_api import async_playwright

from OddsScraper.scraper import MLBOddsScraper, SQLiteStore

from src.dashboard.schemas import (
    LiveSeasonCaptureResponse,
    LiveSeasonDashboardResponse,
    LiveSeasonGameResponse,
    LiveSeasonManualBetDeleteRequest,
    LiveSeasonManualBetRequest,
    LiveSeasonSummaryResponse,
)
from src.ops.live_release import live_release_metadata
from src.ops.live_season_tracker import (
    build_live_season_summary,
    build_manual_tracking_summary,
    capture_live_slate_once,
    delete_manual_tracked_bet,
    list_tracked_games,
    list_manual_tracked_bets,
    submit_manual_tracked_bet,
)

router = APIRouter(prefix="/api/live-season", tags=["live-season"])
_SCRAPER_DB_PATH = Path("OddsScraper") / "data" / "mlb_odds.db"


def _display_bet_units(raw_units: object) -> float | None:
    if raw_units is None:
        return None
    try:
        resolved = float(raw_units)
    except (TypeError, ValueError):
        return None
    clamped = min(5.0, max(0.5, resolved))
    return round(clamped * 2.0) / 2.0


def _normalize_rows(rows: list[dict]) -> list[dict]:
    normalized_rows = []
    for row in rows:
        normalized = dict(row)
        normalized["paper_fallback"] = bool(normalized.get("paper_fallback"))
        normalized["is_play_of_day"] = bool(normalized.get("is_play_of_day"))
        normalized["bet_units"] = (
            _display_bet_units(normalized.get("bet_units"))
            if normalized.get("bet_units") is not None
            else _display_bet_units(normalized.get("kelly_stake"))
        )
        normalized_rows.append(normalized)
    return normalized_rows


def _summary_with_units(summary_payload: dict, rows: list[dict]) -> LiveSeasonSummaryResponse:
    graded_rows = [
        row
        for row in rows
        if row.get("selected_market_type") and row.get("settled_result") in {"WIN", "LOSS", "PUSH"}
    ]
    units_risked = (
        float(sum(float(row.get("bet_units") or 0.0) for row in graded_rows))
        if graded_rows
        else 0.0
    )
    profit_units = (
        float(
            sum(
                float(row.get("flat_profit_loss") or 0.0) * float(row.get("bet_units") or 0.0)
                for row in graded_rows
            )
        )
        if graded_rows
        else 0.0
    )
    summary_payload = dict(summary_payload)
    summary_payload["official_units_risked"] = units_risked
    summary_payload["official_profit_units"] = profit_units
    summary_payload["official_roi"] = (
        float(profit_units / units_risked) if units_risked > 0 else None
    )
    return LiveSeasonSummaryResponse.model_validate(summary_payload)


def _forced_rows_from_model_rows(rows: list[dict]) -> list[dict]:
    forced_rows: list[dict] = []
    for row in rows:
        if not row.get("forced_market_type"):
            continue
        forced = dict(row)
        forced["selected_market_type"] = row.get("forced_market_type")
        forced["selected_side"] = row.get("forced_side")
        forced["source_model"] = row.get("forced_source_model")
        forced["source_model_version"] = row.get("forced_source_model_version")
        forced["book_name"] = row.get("forced_book_name")
        forced["odds_at_bet"] = row.get("forced_odds_at_bet")
        forced["line_at_bet"] = row.get("forced_line_at_bet")
        forced["fair_probability"] = row.get("forced_fair_probability")
        forced["model_probability"] = row.get("forced_model_probability")
        forced["edge_pct"] = row.get("forced_edge_pct")
        forced["ev"] = row.get("forced_ev")
        forced["kelly_stake"] = row.get("forced_kelly_stake")
        forced["bet_units"] = row.get("forced_kelly_stake")
        forced["settled_result"] = row.get("forced_settled_result")
        forced["flat_profit_loss"] = row.get("forced_flat_profit_loss")
        forced_rows.append(forced)
    return _normalize_rows(forced_rows)


def _build_dashboard_response(
    *,
    season: int,
    pipeline_date: str,
    db_path: str,
    capture_payload: dict | None = None,
) -> LiveSeasonDashboardResponse:
    release = live_release_metadata()
    all_rows = _normalize_rows(list_tracked_games(season=season, db_path=db_path))
    manual_rows = _normalize_rows(list_manual_tracked_bets(season=season, db_path=db_path))
    forced_rows = _forced_rows_from_model_rows(all_rows)
    summary = _summary_with_units(
        build_live_season_summary(season=season, db_path=db_path).to_dict(),
        all_rows,
    )
    manual_summary = _summary_with_units(
        build_manual_tracking_summary(season=season, db_path=db_path).to_dict(),
        manual_rows,
    )
    forced_summary = _summary_with_units(
        {
            **build_live_season_summary(season=season, db_path=db_path).to_dict(),
            "tracked_games": len(forced_rows),
            "settled_games": sum(1 for row in forced_rows if row.get("actual_status") == "final"),
            "picks": len(forced_rows),
            "graded_picks": sum(
                1 for row in forced_rows if row.get("settled_result") in {"WIN", "LOSS", "PUSH"}
            ),
            "wins": sum(1 for row in forced_rows if row.get("settled_result") == "WIN"),
            "losses": sum(1 for row in forced_rows if row.get("settled_result") == "LOSS"),
            "pushes": sum(1 for row in forced_rows if row.get("settled_result") == "PUSH"),
            "no_picks": 0,
            "errors": 0,
            "paper_fallback_picks": 0,
            "flat_profit_units": 0.0,
            "flat_roi": None,
            "play_of_day_count": 0,
            "play_of_day_graded_picks": 0,
            "play_of_day_wins": 0,
            "play_of_day_losses": 0,
            "play_of_day_pushes": 0,
            "play_of_day_profit_units": 0.0,
            "play_of_day_roi": None,
            "forced_picks": 0,
            "forced_graded_picks": 0,
            "forced_wins": 0,
            "forced_losses": 0,
            "forced_pushes": 0,
            "forced_profit_units": 0.0,
            "forced_roi": None,
            "f5_ml_accuracy": None,
            "f5_ml_brier": None,
            "f5_ml_log_loss": None,
            "f5_rl_accuracy": None,
            "f5_rl_brier": None,
            "f5_rl_log_loss": None,
        },
        forced_rows,
    )
    today_rows = [row for row in all_rows if str(row.get("pipeline_date")) == pipeline_date]
    historical_rows = [
        row for row in all_rows if str(row.get("pipeline_date")) != pipeline_date and row.get("selected_market_type")
    ]
    manual_today_rows = [row for row in manual_rows if str(row.get("pipeline_date")) == pipeline_date]
    manual_historical_rows = [
        row
        for row in manual_rows
        if str(row.get("pipeline_date")) != pipeline_date and row.get("selected_market_type")
    ]
    forced_today_rows = [row for row in forced_rows if str(row.get("pipeline_date")) == pipeline_date]
    forced_historical_rows = [
        row
        for row in forced_rows
        if str(row.get("pipeline_date")) != pipeline_date and row.get("selected_market_type")
    ]
    return LiveSeasonDashboardResponse(
        season=season,
        pipeline_date=pipeline_date,
        release_name=release["release_name"],
        release_version=release["release_version"],
        model_display_name=release["model_display_name"],
        strategy_name=release["strategy_name"],
        strategy_version=release["strategy_version"],
        technical_model_version=release["technical_model_version"],
        research_baseline_label=release["research_baseline_label"],
        policy_summary=release["policy_summary"],
        summary=summary,
        manual_summary=manual_summary,
        forced_summary=forced_summary,
        today_games=[LiveSeasonGameResponse.model_validate(row) for row in today_rows],
        historical_games=[LiveSeasonGameResponse.model_validate(row) for row in historical_rows],
        manual_today_games=[
            LiveSeasonGameResponse.model_validate(row) for row in manual_today_rows
        ],
        manual_historical_games=[
            LiveSeasonGameResponse.model_validate(row) for row in manual_historical_rows
        ],
        forced_today_games=[
            LiveSeasonGameResponse.model_validate(row) for row in forced_today_rows
        ],
        forced_historical_games=[
            LiveSeasonGameResponse.model_validate(row) for row in forced_historical_rows
        ],
        capture=(
            None
            if capture_payload is None
            else LiveSeasonCaptureResponse.model_validate(capture_payload)
        ),
    )


async def _refresh_recent_bet365_scraper_window(
    *,
    target_day: date_type,
    lookback_days: int = 4,
) -> dict[str, object]:
    scraper_db_path = _SCRAPER_DB_PATH
    scraper_db_path.parent.mkdir(parents=True, exist_ok=True)
    db = SQLiteStore(scraper_db_path)
    before_rows = db.count()
    before_range = db.date_range()
    scraper = MLBOddsScraper(output_dir=str(scraper_db_path.parent), base_delay=1.5, max_parallel=2)
    start_date = datetime.combine(target_day, time.min)
    end_date = datetime.combine(target_day - timedelta(days=lookback_days), time.min)
    inserted = 0

    try:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            )
            try:
                inserted = await scraper.scrape_date_range_backwards_db(
                    context,
                    start_date=start_date,
                    end_date=end_date,
                    db=db,
                    resume=False,
                )
            finally:
                await browser.close()
    finally:
        after_rows = db.count()
        after_range = db.date_range()
        db.close()

    return {
        "scraper_refreshed": True,
        "scraper_inserted_rows": int(inserted),
        "scraper_total_rows_before": int(before_rows),
        "scraper_total_rows_after": int(after_rows),
        "scraper_range_before": before_range,
        "scraper_range_after": after_range,
        "scraper_window_start": target_day.isoformat(),
        "scraper_window_end": (target_day - timedelta(days=lookback_days)).isoformat(),
    }


@router.get("/summary", response_model=LiveSeasonSummaryResponse)
async def get_live_season_summary(
    season: int = Query(default=2026, ge=2000, le=2100),
    db_path: str = Query(
        default=str(Path("data") / "mlb.db"),
        description="SQLite path used by live season tracking.",
    ),
) -> LiveSeasonSummaryResponse:
    summary = build_live_season_summary(season=season, db_path=db_path)
    return LiveSeasonSummaryResponse.model_validate(summary.to_dict())


@router.get("/games", response_model=list[LiveSeasonGameResponse])
async def get_live_season_games(
    season: int = Query(default=2026, ge=2000, le=2100),
    pipeline_date: str | None = Query(default=None),
    db_path: str = Query(
        default=str(Path("data") / "mlb.db"),
        description="SQLite path used by live season tracking.",
    ),
) -> list[LiveSeasonGameResponse]:
    rows = _normalize_rows(
        list_tracked_games(season=season, pipeline_date=pipeline_date, db_path=db_path)
    )
    return [LiveSeasonGameResponse.model_validate(row) for row in rows]


@router.get("/dashboard", response_model=LiveSeasonDashboardResponse)
async def get_live_season_dashboard(
    season: int = Query(default=2026, ge=2000, le=2100),
    pipeline_date: str | None = Query(default=None),
    db_path: str = Query(
        default=str(Path("data") / "mlb.db"),
        description="SQLite path used by live season tracking.",
    ),
) -> LiveSeasonDashboardResponse:
    resolved_date = pipeline_date or date_type.today().isoformat()
    return _build_dashboard_response(
        season=season,
        pipeline_date=resolved_date,
        db_path=db_path,
    )


@router.post("/capture-today", response_model=LiveSeasonDashboardResponse)
async def capture_live_season_today(
    season: int = Query(default=2026, ge=2000, le=2100),
    pipeline_date: str | None = Query(default=None),
    db_path: str = Query(
        default=str(Path("data") / "mlb.db"),
        description="SQLite path used by live season tracking.",
    ),
) -> LiveSeasonDashboardResponse:
    resolved_date = pipeline_date or date_type.today().isoformat()
    resolved_day = date_type.fromisoformat(resolved_date)
    scraper_payload: dict[str, object]
    try:
        scraper_payload = await _refresh_recent_bet365_scraper_window(target_day=resolved_day)
    except Exception as exc:
        scraper_payload = {
            "scraper_refreshed": False,
            "scraper_error": str(exc),
            "scraper_window_start": resolved_day.isoformat(),
            "scraper_window_end": (resolved_day - timedelta(days=4)).isoformat(),
        }
    capture_payload = capture_live_slate_once(
        target_date=resolved_date,
        db_path=db_path,
        fast=True,
    )
    capture_payload.update(scraper_payload)
    return _build_dashboard_response(
        season=season,
        pipeline_date=resolved_date,
        db_path=db_path,
        capture_payload=capture_payload,
    )


@router.post("/manual-bets", response_model=LiveSeasonDashboardResponse)
async def submit_live_season_manual_bet(
    request: LiveSeasonManualBetRequest,
    db_path: str = Query(
        default=str(Path("data") / "mlb.db"),
        description="SQLite path used by live season tracking.",
    ),
) -> LiveSeasonDashboardResponse:
    submit_manual_tracked_bet(
        season=request.season,
        pipeline_date=request.pipeline_date,
        game_pk=request.game_pk,
        matchup=request.matchup,
        market_type=request.market_type,
        side=request.side,
        odds_at_bet=request.odds_at_bet,
        line_at_bet=request.line_at_bet,
        fair_probability=request.fair_probability,
        model_probability=request.model_probability,
        edge_pct=request.edge_pct,
        ev=request.ev,
        kelly_stake=request.kelly_stake,
        bet_units=request.bet_units,
        book_name=request.book_name,
        model_version=request.model_version,
        source_model=request.source_model,
        source_model_version=request.source_model_version,
        input_status=request.input_status,
        narrative=request.narrative,
        db_path=db_path,
    )
    return _build_dashboard_response(
        season=request.season,
        pipeline_date=request.pipeline_date,
        db_path=db_path,
    )


@router.delete("/manual-bets", response_model=LiveSeasonDashboardResponse)
async def delete_live_season_manual_bet(
    request: LiveSeasonManualBetDeleteRequest,
    db_path: str = Query(
        default=str(Path("data") / "mlb.db"),
        description="SQLite path used by live season tracking.",
    ),
) -> LiveSeasonDashboardResponse:
    try:
        delete_manual_tracked_bet(
            manual_bet_id=request.manual_bet_id,
            season=request.season,
            allow_locked_delete=True,
            db_path=db_path,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _build_dashboard_response(
        season=request.season,
        pipeline_date=request.pipeline_date,
        db_path=db_path,
    )
