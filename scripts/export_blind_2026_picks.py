from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from rich.console import Console


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


console = Console()
DEFAULT_DB_PATH = Path("data/mlb.db")
DEFAULT_OUTPUT_DIR = Path("data/reports/blind_2026")


LATEST_PIPELINE_ROWS_SQL = """
WITH ranked AS (
  SELECT
    d.game_pk,
    g.date AS game_date,
    g.away_team,
    g.home_team,
    g.status AS game_status_snapshot,
    d.mode,
    d.status AS pipeline_status,
    d.game_status,
    d.selected_market_type,
    d.selected_side,
    d.odds_at_bet,
    d.fair_probability,
    d.model_probability,
    d.edge_pct,
    d.ev,
    d.kelly_stake,
    d.no_pick_reason,
    d.created_at,
    ROW_NUMBER() OVER (PARTITION BY d.game_pk ORDER BY d.created_at DESC, d.id DESC) AS rn
  FROM daily_pipeline_results d
  JOIN games g ON g.game_pk = d.game_pk
  WHERE g.date LIKE '2026-%'
)
SELECT
  game_pk,
  game_date,
  away_team,
  home_team,
  mode,
  pipeline_status,
  game_status,
  game_status_snapshot,
  selected_market_type,
  selected_side,
  odds_at_bet,
  fair_probability,
  model_probability,
  edge_pct,
  ev,
  kelly_stake,
  no_pick_reason,
  created_at
FROM ranked
WHERE rn = 1
ORDER BY game_date, game_pk
"""


LATEST_PREDICTIONS_SQL = """
WITH ranked AS (
  SELECT
    p.game_pk,
    g.date AS game_date,
    g.away_team,
    g.home_team,
    p.f5_ml_home_prob,
    p.f5_ml_away_prob,
    p.f5_rl_home_prob,
    p.f5_rl_away_prob,
    p.predicted_at,
    ROW_NUMBER() OVER (PARTITION BY p.game_pk ORDER BY p.predicted_at DESC, p.id DESC) AS rn
  FROM predictions p
  JOIN games g ON g.game_pk = p.game_pk
  WHERE g.date LIKE '2026-%'
)
SELECT
  game_pk,
  game_date,
  away_team,
  home_team,
  f5_ml_home_prob,
  f5_ml_away_prob,
  f5_rl_home_prob,
  f5_rl_away_prob,
  predicted_at
FROM ranked
WHERE rn = 1
ORDER BY game_date, game_pk
"""


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def _load_frame(conn: sqlite3.Connection, sql: str) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export blind 2026 picks and latest probabilities without outcomes.",
    )
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--picks-only", action="store_true")
    args = parser.parse_args(argv)

    db_path = _resolve_path(args.db_path)
    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    pipeline_df = _load_frame(conn, LATEST_PIPELINE_ROWS_SQL)
    predictions_df = _load_frame(conn, LATEST_PREDICTIONS_SQL)

    merged = pipeline_df.merge(
        predictions_df[
            [
                "game_pk",
                "f5_ml_home_prob",
                "f5_ml_away_prob",
                "f5_rl_home_prob",
                "f5_rl_away_prob",
                "predicted_at",
            ]
        ],
        on="game_pk",
        how="outer",
    )
    merged["has_pick"] = merged["selected_market_type"].notna()
    merged["has_prediction"] = merged["predicted_at"].notna()

    blind_df = merged[
        [
            "game_pk",
            "game_date",
            "away_team",
            "home_team",
            "mode",
            "pipeline_status",
            "game_status",
            "selected_market_type",
            "selected_side",
            "odds_at_bet",
            "fair_probability",
            "model_probability",
            "edge_pct",
            "ev",
            "kelly_stake",
            "no_pick_reason",
            "f5_ml_home_prob",
            "f5_ml_away_prob",
            "f5_rl_home_prob",
            "f5_rl_away_prob",
            "created_at",
            "predicted_at",
            "has_pick",
            "has_prediction",
        ]
    ].sort_values(["game_date", "game_pk"]).reset_index(drop=True)

    if args.picks_only:
        blind_df = blind_df.loc[blind_df["has_pick"]].copy()

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    csv_path = output_dir / f"blind_2026_picks_{stamp}.csv"
    json_path = output_dir / f"blind_2026_summary_{stamp}.json"
    blind_df.to_csv(csv_path, index=False)

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "db_path": str(db_path),
        "csv_path": str(csv_path),
        "row_count": int(len(blind_df)),
        "pick_count": int(blind_df["has_pick"].sum()) if not blind_df.empty else 0,
        "prediction_count": int(blind_df["has_prediction"].sum()) if not blind_df.empty else 0,
        "game_date_min": None if blind_df.empty else str(blind_df["game_date"].min()),
        "game_date_max": None if blind_df.empty else str(blind_df["game_date"].max()),
        "picks_only": bool(args.picks_only),
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    console.print(
        "[bold green]Blind 2026 export complete[/bold green] "
        f"rows={summary['row_count']} picks={summary['pick_count']} predictions={summary['prediction_count']}"
    )
    console.print(f"csv={csv_path}")
    console.print(f"summary={json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
