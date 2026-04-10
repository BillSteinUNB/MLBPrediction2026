from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ops.live_season_tracker import capture_live_slate_once, settle_tracked_games  # noqa: E402


console = Console()
DEFAULT_SOURCE_DB = Path("data/mlb.db")
DEFAULT_GAMES_DB = Path("data/2026MLB.db")
DEFAULT_RESULTS_DB = Path("data/2026Results.db")
DEFAULT_OUTPUT_DIR = Path("data/reports/2026_tracker")


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def _run_script(script_name: str, *args: str) -> None:
    command = [sys.executable, str((PROJECT_ROOT / "scripts" / script_name).resolve()), *args]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _load_results(results_db: Path) -> pd.DataFrame:
    with sqlite3.connect(results_db) as connection:
        return pd.read_sql_query("SELECT * FROM model_results ORDER BY game_date, game_pk", connection)


def _scaled_units(raw_units: float | object) -> float | None:
    if pd.isna(raw_units):
        return None
    resolved = float(raw_units)
    clamped = min(5.0, max(0.5, resolved))
    return round(clamped * 2.0) / 2.0


def _format_pick(row: pd.Series) -> str:
    market = str(row.get("selected_market_type") or "")
    side = str(row.get("selected_side") or "")
    line_at_bet = row.get("line_at_bet")
    if market.endswith("_ml"):
        team = str(row["home_team"]) if side == "home" else str(row["away_team"])
        return f"ML {team}"
    if market.endswith("_rl"):
        team = str(row["home_team"]) if side == "home" else str(row["away_team"])
        if pd.isna(line_at_bet):
            return f"RL {team}"
        sign = "+" if float(line_at_bet) > 0 else ""
        return f"RL {team} {sign}{float(line_at_bet):g}"
    if market.endswith("_total"):
        direction = "Over" if side == "over" else "Under"
        if pd.isna(line_at_bet):
            return f"Total {direction}"
        return f"Total {direction} {float(line_at_bet):g}"
    return side.upper() if side else ""


def _format_edge_pct(value: float | object) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value) * 100.0:.2f}%"


def _format_american_odds(value: float | object) -> str:
    if pd.isna(value):
        return ""
    odds = int(float(value))
    return f"{odds:+d}"


def _format_result(row: pd.Series) -> str:
    market = str(row.get("selected_market_type") or "")
    if market == "f5_ml":
        winner = row.get("f5_winner")
        if pd.isna(winner):
            return ""
        if winner == "push":
            return "PUSH"
        return str(row["home_team"]) if winner == "home" else str(row["away_team"])
    if market == "full_game_ml":
        winner = row.get("final_winner")
        if pd.isna(winner):
            return ""
        if winner == "push":
            return "PUSH"
        return str(row["home_team"]) if winner == "home" else str(row["away_team"])
    if market.startswith("f5_"):
        away_score = row.get("f5_away_score")
        home_score = row.get("f5_home_score")
    else:
        away_score = row.get("final_away_score")
        home_score = row.get("final_home_score")
    if pd.isna(away_score) or pd.isna(home_score):
        return ""
    return f"{row['away_team']} {int(away_score)}-{int(home_score)} {row['home_team']}"


def _format_outcome(row: pd.Series) -> str:
    selected_pick_result = row.get("selected_pick_result")
    if pd.isna(selected_pick_result):
        return ""
    outcome = str(selected_pick_result).upper()
    profit = row.get("official_profit_units")
    if pd.isna(profit):
        return outcome
    if outcome == "PUSH":
        return "PUSH 0.00u"
    return f"{outcome} {float(profit):+.2f}u"


def _render_history_table(frame: pd.DataFrame) -> None:
    table = Table(title="2026 Tracked Picks Before Today")
    for column in ("Game", "Pick", "Edge %", "Value", "Unit", "Result", "Won/Loss", "Date"):
        table.add_column(column)

    for row in frame.to_dict(orient="records"):
        series = pd.Series(row)
        table.add_row(
            f"{row['away_team']} v {row['home_team']}",
            _format_pick(series),
            _format_edge_pct(row.get("edge_pct")),
            _format_american_odds(row.get("odds_at_bet")),
            "" if pd.isna(row.get("bet_units")) else f"{float(row['bet_units']):.1f}u",
            _format_result(series),
            _format_outcome(series),
            str(row["game_date"]),
        )
    console.print(table)


def _render_today_table(frame: pd.DataFrame) -> None:
    table = Table(title="Today's Picks")
    for column in ("Game", "Pick", "Edge %", "Value", "Unit", "Date"):
        table.add_column(column)

    for row in frame.to_dict(orient="records"):
        series = pd.Series(row)
        table.add_row(
            f"{row['away_team']} v {row['home_team']}",
            _format_pick(series),
            _format_edge_pct(row.get("edge_pct")),
            _format_american_odds(row.get("odds_at_bet")),
            "" if pd.isna(row.get("bet_units")) else f"{float(row['bet_units']):.1f}u",
            str(row["game_date"]),
        )
    console.print(table)


def _season_summary(frame: pd.DataFrame, *, today_date: str) -> dict[str, object]:
    tracked = frame.loc[frame["has_exact_tracking"] == 1].copy()
    picks = tracked.loc[tracked["has_pick"] == 1].copy()
    graded = picks.loc[picks["selected_pick_result"].isin(["WIN", "LOSS", "PUSH"])].copy()

    graded["flat_profit_loss"] = pd.to_numeric(graded["flat_profit_loss"], errors="coerce").fillna(0.0)
    graded["bet_units"] = pd.to_numeric(graded["kelly_stake"], errors="coerce").apply(_scaled_units)
    graded["official_profit_units"] = graded["flat_profit_loss"] * graded["bet_units"].fillna(0.0)
    graded["edge_pct"] = pd.to_numeric(graded["edge_pct"], errors="coerce")

    wins = int((graded["selected_pick_result"] == "WIN").sum())
    losses = int((graded["selected_pick_result"] == "LOSS").sum())
    pushes = int((graded["selected_pick_result"] == "PUSH").sum())
    decisions = wins + losses
    graded_count = len(graded)

    official_units_risked = float(graded["bet_units"].fillna(0.0).sum()) if graded_count else 0.0
    official_profit_units = float(graded["official_profit_units"].sum()) if graded_count else 0.0
    official_roi = (
        float(official_profit_units / official_units_risked) if official_units_risked > 0 else None
    )
    flat_profit_units = float(graded["flat_profit_loss"].sum()) if graded_count else 0.0
    flat_roi = float(flat_profit_units / graded_count) if graded_count else None

    accuracy = float(wins / decisions) if decisions else None
    accuracy_with_pushes = float(wins / graded_count) if graded_count else None
    avg_edge_pct = float(graded["edge_pct"].mean()) if graded_count else None
    avg_units = float(graded["bet_units"].mean()) if graded_count else None

    today_rows = tracked.loc[(tracked["game_date"] == today_date) & (tracked["has_pick"] == 1)].copy()
    pending_rows = picks.loc[picks["selected_pick_result"] == "pending"].copy()

    return {
        "today_date": today_date,
        "tracked_games": int(len(tracked)),
        "tracked_picks": int(len(picks)),
        "graded_picks": int(graded_count),
        "pending_picks": int(len(pending_rows)),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "accuracy_excluding_pushes": accuracy,
        "accuracy_including_pushes": accuracy_with_pushes,
        "flat_profit_units": flat_profit_units,
        "flat_roi": flat_roi,
        "official_units_risked": official_units_risked,
        "official_profit_units": official_profit_units,
        "official_roi": official_roi,
        "average_edge_pct": avg_edge_pct,
        "average_kelly_units": avg_units,
        "today_pick_count": int(len(today_rows)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Refresh 2026 tracking DBs and emit season-to-date performance plus today's picks."
    )
    parser.add_argument("--source-db", default=str(DEFAULT_SOURCE_DB))
    parser.add_argument("--games-db", default=str(DEFAULT_GAMES_DB))
    parser.add_argument("--results-db", default=str(DEFAULT_RESULTS_DB))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--today-date", default=date.today().isoformat())
    parser.add_argument("--capture-today", action="store_true")
    args = parser.parse_args(argv)

    source_db = _resolve_path(args.source_db)
    games_db = _resolve_path(args.games_db)
    results_db = _resolve_path(args.results_db)
    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.capture_today:
        capture_payload = capture_live_slate_once(
            target_date=args.today_date,
            db_path=source_db,
            fast=True,
        )
        status_message = "already captured" if capture_payload["already_captured"] else "captured"
        console.print(
            f"[cyan]{args.today_date} {status_message}[/cyan] "
            f"tracked_games={capture_payload['tracked_games']} settled_rows={capture_payload['settled_rows']}"
        )

    settle_tracked_games(season=2026, db_path=source_db)

    _run_script(
        "build_2026_mlb_db.py",
        "--source-db",
        str(source_db),
        "--target-db",
        str(games_db),
        "--fill-through-today",
    )
    _run_script(
        "build_2026_results_db.py",
        "--source-db",
        str(source_db),
        "--games-db",
        str(games_db),
        "--target-db",
        str(results_db),
    )

    results = _load_results(results_db)
    tracked = results.loc[results["has_exact_tracking"] == 1].copy()
    picks = tracked.loc[tracked["has_pick"] == 1].copy()
    graded = picks.loc[picks["selected_pick_result"].isin(["WIN", "LOSS", "PUSH"])].copy()
    today_picks = tracked.loc[(tracked["game_date"] == args.today_date) & (tracked["has_pick"] == 1)].copy()
    historical_picks = picks.loc[picks["game_date"] != args.today_date].copy()

    graded["official_profit_units"] = (
        pd.to_numeric(graded["flat_profit_loss"], errors="coerce").fillna(0.0)
        * pd.to_numeric(graded["kelly_stake"], errors="coerce").apply(_scaled_units).fillna(0.0)
    )
    graded["bet_units"] = pd.to_numeric(graded["kelly_stake"], errors="coerce").apply(_scaled_units)
    historical_picks["bet_units"] = pd.to_numeric(
        historical_picks["kelly_stake"], errors="coerce"
    ).apply(_scaled_units)
    today_picks["bet_units"] = pd.to_numeric(today_picks["kelly_stake"], errors="coerce").apply(
        _scaled_units
    )
    historical_picks["official_profit_units"] = (
        pd.to_numeric(historical_picks["flat_profit_loss"], errors="coerce").fillna(0.0)
        * historical_picks["bet_units"].fillna(0.0)
    )

    summary = _season_summary(results, today_date=args.today_date)
    summary_path = output_dir / "season_summary.json"
    tracked_picks_path = output_dir / "tracked_picks.csv"
    graded_picks_path = output_dir / "graded_picks.csv"
    today_picks_path = output_dir / "today_picks.csv"

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    picks.to_csv(tracked_picks_path, index=False)
    graded.to_csv(graded_picks_path, index=False)
    today_picks.to_csv(today_picks_path, index=False)

    if not historical_picks.empty:
        _render_history_table(historical_picks.sort_values(["game_date", "game_pk"]).reset_index(drop=True))
    else:
        console.print("[yellow]No tracked picks before today.[/yellow]")

    console.print(
        "[bold green]Season Summary[/bold green] "
        f"tracked_picks={summary['tracked_picks']} graded_picks={summary['graded_picks']}"
    )
    console.print(
        f"Accuracy(ex pushes)={summary['accuracy_excluding_pushes']:.2%} "
        if summary["accuracy_excluding_pushes"] is not None
        else "Accuracy(ex pushes)=N/A"
    )
    console.print(
        f"Official Units Risked={summary['official_units_risked']:.2f}u "
        f"Official Profit={summary['official_profit_units']:+.2f}u "
        f"Official ROI={summary['official_roi']:.2%}"
        if summary["official_roi"] is not None
        else f"Official Units Risked={summary['official_units_risked']:.2f}u Official Profit={summary['official_profit_units']:+.2f}u Official ROI=N/A"
    )
    console.print(
        f"Flat Profit={summary['flat_profit_units']:+.2f}u Flat ROI={summary['flat_roi']:.2%}"
        if summary["flat_roi"] is not None
        else f"Flat Profit={summary['flat_profit_units']:+.2f}u Flat ROI=N/A"
    )

    if not today_picks.empty:
        _render_today_table(today_picks.sort_values(["game_date", "game_pk"]).reset_index(drop=True))
    else:
        console.print("[yellow]No tracked picks for today.[/yellow]")

    console.print(f"summary={summary_path}")
    console.print(f"tracked_picks={tracked_picks_path}")
    console.print(f"graded_picks={graded_picks_path}")
    console.print(f"today_picks={today_picks_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
