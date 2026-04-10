from __future__ import annotations

from src.ops.live_season_tracker import LiveSeasonSummary


def test_live_season_summary_route(client, monkeypatch) -> None:
    from src.dashboard.routes import live_season as live_season_route

    monkeypatch.setattr(
        live_season_route,
        "build_live_season_summary",
        lambda season, db_path: LiveSeasonSummary(
            season=season,
            tracked_games=3,
            settled_games=2,
            picks=2,
            graded_picks=2,
            wins=1,
            losses=1,
            pushes=0,
            no_picks=1,
            errors=0,
            paper_fallback_picks=1,
            flat_profit_units=0.2,
            flat_roi=0.1,
            play_of_day_count=1,
            play_of_day_graded_picks=1,
            play_of_day_wins=1,
            play_of_day_losses=0,
            play_of_day_pushes=0,
            play_of_day_profit_units=1.2,
            play_of_day_roi=1.2,
            forced_picks=3,
            forced_graded_picks=2,
            forced_wins=1,
            forced_losses=1,
            forced_pushes=0,
            forced_profit_units=0.2,
            forced_roi=0.1,
            f5_ml_accuracy=0.5,
            f5_ml_brier=0.24,
            f5_ml_log_loss=0.68,
            f5_rl_accuracy=0.5,
            f5_rl_brier=0.22,
            f5_rl_log_loss=0.61,
            latest_capture_at="2026-03-25T12:00:00+00:00",
        ),
    )

    response = client.get("/api/live-season/summary?season=2026")
    assert response.status_code == 200
    payload = response.json()
    assert payload["season"] == 2026
    assert payload["tracked_games"] == 3
    assert payload["flat_roi"] == 0.1


def test_live_season_games_route(client, monkeypatch) -> None:
    from src.dashboard.routes import live_season as live_season_route

    monkeypatch.setattr(
        live_season_route,
        "list_tracked_games",
        lambda season, pipeline_date, db_path: [
            {
                "season": season,
                "pipeline_date": pipeline_date or "2026-03-25",
                "game_pk": 1001,
                "matchup": "NYY @ SF",
                "run_id": "run-1",
                "captured_at": "2026-03-25T12:00:00+00:00",
                "status": "pick",
                "paper_fallback": 1,
                "is_play_of_day": 1,
                "selected_market_type": "f5_ml",
                "selected_side": "away",
                "odds_at_bet": 125,
                "forced_market_type": "f5_ml",
                "forced_side": "away",
                "forced_odds_at_bet": 125,
            }
        ],
    )

    response = client.get("/api/live-season/games?season=2026&pipeline_date=2026-03-25")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 1
    assert payload[0]["game_pk"] == 1001
    assert payload[0]["paper_fallback"] is True
    assert payload[0]["is_play_of_day"] is True
