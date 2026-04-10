#!/usr/bin/env python3
"""Export daily picks data from SQLite to JSON for dashboard consumption.

This standalone script extracts daily picks data from the SQLite database
and exports it as JSON conforming to the DailyPicsData TypeScript interface.

Usage:
    python -m scripts.export_pics_json --date today --db data/mlb.db --output data/reports/pics/
    python -m scripts.export_pics_json --date 2026-04-15 --db data/mlb.db --output data/reports/pics/ --dry-run
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any


def american_to_implied_probability(odds: int) -> float:
    """Convert American odds to implied probability (0.0-1.0)."""
    if odds == 0:
        return 0.5
    if odds >= 100:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def devig_probabilities(home_odds: int, away_odds: int) -> tuple[float, float]:
    """De-vig home and away odds to fair probabilities."""
    home_implied = american_to_implied_probability(home_odds)
    away_implied = american_to_implied_probability(away_odds)
    total = home_implied + away_implied

    if total == 0:
        return 0.5, 0.5

    return home_implied / total, away_implied / total


def coerce_date(value: str | date) -> date:
    """Coerce date string to date object."""
    if isinstance(value, date):
        return value
    if value == "today":
        return datetime.now(UTC).date()
    return date.fromisoformat(value)


def build_market_view(
    *,
    prediction: dict[str, Any],
    edge_map: dict[str, dict[str, dict[str, Any]]],
    odds_map: dict[str, dict[str, Any]],
    market_prefix: str,
) -> dict[str, Any | None]:
    """Build a MarketView object conforming to TypeScript interface.

    Args:
        prediction: Prediction object from cached_slate_responses
        edge_map: Map of market_type -> side -> edge_calculation fields
        odds_map: Map of market_type -> odds_snapshot fields
        market_prefix: "f5" or "full_game"

    Returns:
        MarketView dict with all fields nullable
    """
    ml_market = f"{market_prefix}_ml"
    rl_market = f"{market_prefix}_rl"

    ml_odds = odds_map.get(ml_market, {})
    rl_odds = odds_map.get(rl_market, {})

    ml_edge_home = edge_map.get(ml_market, {}).get("home", {})
    ml_edge_away = edge_map.get(ml_market, {}).get("away", {})
    rl_edge_home = edge_map.get(rl_market, {}).get("home", {})
    rl_edge_away = edge_map.get(rl_market, {}).get("away", {})

    return {
        "ml_home_prob": prediction.get(f"{market_prefix}_ml_home_prob"),
        "ml_away_prob": prediction.get(f"{market_prefix}_ml_away_prob"),
        "rl_home_prob": prediction.get(f"{market_prefix}_rl_home_prob"),
        "rl_away_prob": prediction.get(f"{market_prefix}_rl_away_prob"),
        "home_odds": ml_odds.get("home_odds"),
        "away_odds": ml_odds.get("away_odds"),
        "home_implied_prob": ml_edge_home.get("home_implied_probability"),
        "away_implied_prob": ml_edge_away.get("away_implied_probability"),
        "home_edge_pct": ml_edge_home.get("edge_pct"),
        "away_edge_pct": ml_edge_away.get("edge_pct"),
        "home_ev": ml_edge_home.get("ev"),
        "away_ev": ml_edge_away.get("ev"),
        "home_signal": rl_edge_home.get("model_probability"),
        "away_signal": rl_edge_away.get("model_probability"),
    }


def export_daily_picks_json(
    *,
    pipeline_date: str | date,
    db_path: str | Path,
    output_dir: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Export daily picks data from SQLite to JSON conforming to DailyPicsData interface.

    Args:
        pipeline_date: Date to export picks for (ISO format or 'today')
        db_path: Path to SQLite database
        output_dir: Directory to write JSON files (if None, dry-run prints to stdout)
        dry_run: If True, print to stdout instead of writing files

    Returns:
        Dictionary containing the exported data conforming to DailyPicsData
    """
    resolved_date = coerce_date(pipeline_date)
    date_str = resolved_date.isoformat()

    database_path = Path(db_path)
    if not database_path.exists():
        print(f"Error: Database not found at {database_path}", file=sys.stderr)
        sys.exit(1)

    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row

        # Load cached slate response
        cached_row = connection.execute(
            """
            SELECT run_id, model_version, payload_json, refreshed_at
            FROM cached_slate_responses
            WHERE pipeline_date = ?
            ORDER BY refreshed_at DESC
            LIMIT 1
            """,
            (date_str,),
        ).fetchone()

        if cached_row is None:
            # No data for this date - return empty structure
            result = {
                "pipeline_date": date_str,
                "generated_date": datetime.now(UTC).date().isoformat(),
                "model_version": "unknown",
                "total_games": 0,
                "picks_count": 0,
                "games": [],
                "play_of_the_day": None,
            }

            if dry_run or output_dir is None:
                print(json.dumps(result, indent=2))
                return result

            # Write to file
            output_path = Path(output_dir) / "daily.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"Exported empty picks data to {output_path}")
            return result

        # Parse cached payload
        payload = json.loads(cached_row["payload_json"])
        model_version = cached_row["model_version"] or "unknown"

        # Extract games from payload
        games_data = payload.get("games", [])

        # Build games list with enriched data
        games = []
        best_pick = None
        best_edge = 0.0

        for game_data in games_data:
            game_pk = game_data.get("game_pk")
            if game_pk is None:
                continue

            # Parse matchup to extract teams
            matchup = game_data.get("matchup", "")
            parts = matchup.split(" @ ")
            away_team = parts[0] if len(parts) == 2 else "UNK"
            home_team = parts[1] if len(parts) == 2 else "UNK"

            prediction = game_data.get("prediction")
            if prediction is None:
                continue

            # Get edge calculations for this game
            edge_rows = connection.execute(
                """
                SELECT 
                    market_type,
                    side,
                    model_probability,
                    fair_probability,
                    home_implied_probability,
                    away_implied_probability,
                    home_fair_probability,
                    away_fair_probability,
                    edge_pct,
                    ev,
                    odds_at_bet
                FROM edge_calculations
                WHERE game_pk = ?
                ORDER BY calculated_at DESC
                """,
                (game_pk,),
            ).fetchall()

            # Build edge lookup
            edge_map: dict[str, dict[str, dict[str, Any]]] = {}
            for edge_row in edge_rows:
                market_type = edge_row["market_type"]
                side = edge_row["side"]
                if market_type not in edge_map:
                    edge_map[market_type] = {}
                if side not in edge_map[market_type]:
                    edge_map[market_type][side] = dict(edge_row)

            # Get latest odds snapshots for this game
            odds_rows = connection.execute(
                """
                SELECT 
                    market_type,
                    home_odds,
                    away_odds,
                    home_point,
                    away_point
                FROM odds_snapshots
                WHERE game_pk = ? AND is_frozen = 1
                ORDER BY fetched_at DESC
                """,
                (game_pk,),
            ).fetchall()

            # Build odds lookup
            odds_map: dict[str, dict[str, Any]] = {}
            for odds_row in odds_rows:
                market_type = odds_row["market_type"]
                if market_type not in odds_map:
                    odds_map[market_type] = dict(odds_row)

            # Build market views
            f5_market_view = build_market_view(
                prediction=prediction,
                edge_map=edge_map,
                odds_map=odds_map,
                market_prefix="f5",
            )

            full_game_market_view = build_market_view(
                prediction=prediction,
                edge_map=edge_map,
                odds_map=odds_map,
                market_prefix="full_game",
            )

            # Extract selected decision
            selected_decision = game_data.get("selected_decision")
            selected_market_type = None
            selected_side = None
            selected_odds = None
            selected_edge_pct = None
            selected_kelly_stake = None

            if selected_decision is not None:
                selected_market_type = selected_decision.get("market_type")
                selected_side = selected_decision.get("side")
                selected_odds = selected_decision.get("odds_at_bet")
                selected_edge_pct = selected_decision.get("edge_pct")
                selected_kelly_stake = selected_decision.get("kelly_stake")

                # Track best pick for play of the day
                if selected_edge_pct is not None and selected_edge_pct > best_edge:
                    best_edge = selected_edge_pct
                    best_pick = {
                        "pick_date": date_str,
                        "game_pk": game_pk,
                        "game_date": date_str,
                        "home_team": home_team,
                        "away_team": away_team,
                        "market_type": selected_market_type,
                        "side": selected_side,
                        "odds": selected_odds,
                        "model_probability": selected_decision.get("model_probability", 0.0),
                        "implied_probability": (
                            american_to_implied_probability(selected_odds) if selected_odds else 0.0
                        ),
                        "edge_pct": selected_edge_pct,
                        "kelly_stake_pct": selected_kelly_stake,
                        "narrative": game_data.get("narrative"),
                    }

            # Build GamePick object
            game_pick = {
                "game_pk": game_pk,
                "game_date": date_str,
                "home_team": home_team,
                "away_team": away_team,
                "game_status": game_data.get("game_status", "Scheduled"),
                "projected_home_runs": prediction.get("projected_f5_home_runs"),
                "projected_away_runs": prediction.get("projected_f5_away_runs"),
                "projected_total_runs": prediction.get("projected_f5_total_runs"),
                "cover_home_prob": prediction.get("f5_rl_home_prob"),
                "cover_away_prob": prediction.get("f5_rl_away_prob"),
                "f5": f5_market_view,
                "full_game": full_game_market_view,
                "selected_market_type": selected_market_type,
                "selected_side": selected_side,
                "selected_odds": selected_odds,
                "selected_edge_pct": selected_edge_pct,
                "selected_kelly_stake": selected_kelly_stake,
            }

            games.append(game_pick)

        # Count picks
        picks_count = sum(1 for game in games if game.get("selected_market_type") is not None)

        # Build final result conforming to DailyPicsData interface
        result = {
            "pipeline_date": date_str,
            "generated_date": datetime.now(UTC).date().isoformat(),
            "model_version": model_version,
            "total_games": len(games),
            "picks_count": picks_count,
            "games": games,
            "play_of_the_day": best_pick,
        }

    # Output
    if dry_run or output_dir is None:
        print(json.dumps(result, indent=2))
        return result

    # Write to files
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    daily_file = output_path / "daily.json"
    daily_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"✓ Wrote {len(games)} game(s) to {daily_file}")

    if best_pick is not None:
        play_file = output_path / "play_of_the_day.json"
        play_file.write_text(json.dumps(best_pick, indent=2), encoding="utf-8")
        print(f"✓ Wrote play of the day to {play_file}")

    return result


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Export daily picks data from SQLite to JSON (DailyPicsData format)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m scripts.export_pics_json --date today --db data/mlb.db --output data/reports/pics/
  python -m scripts.export_pics_json --date 2026-04-15 --db data/mlb.db --dry-run
        """,
    )
    parser.add_argument(
        "--date",
        required=True,
        help="Date to export picks for (ISO format YYYY-MM-DD or 'today')",
    )
    parser.add_argument(
        "--db",
        default="data/mlb.db",
        help="Path to SQLite database (default: data/mlb.db)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for JSON files (default: data/reports/pics/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print JSON to stdout instead of writing files",
    )

    args = parser.parse_args()

    output_dir = args.output if not args.dry_run else None
    if output_dir is None and not args.dry_run:
        output_dir = "data/reports/pics/"

    try:
        export_daily_picks_json(
            pipeline_date=args.date,
            db_path=args.db,
            output_dir=output_dir,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
