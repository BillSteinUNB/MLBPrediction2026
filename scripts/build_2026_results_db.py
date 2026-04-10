from __future__ import annotations

import argparse
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
DEFAULT_SOURCE_DB = Path("data/mlb.db")
DEFAULT_GAMES_DB = Path("data/2026MLB.db")
DEFAULT_TARGET_DB = Path("data/2026Results.db")


LATEST_PIPELINE_SQL = """
WITH ranked AS (
  SELECT
    d.game_pk,
    g.date AS game_date,
    g.away_team,
    g.home_team,
    d.mode,
    d.status AS pipeline_status,
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
SELECT *
FROM ranked
WHERE rn = 1
"""


LATEST_PREDICTIONS_SQL = """
WITH ranked AS (
  SELECT
    p.game_pk,
    p.model_version,
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
SELECT *
FROM ranked
WHERE rn = 1
"""


LIVE_TRACKER_SQL = """
SELECT
  game_pk,
  pipeline_date,
  status AS tracker_status,
  model_version AS tracker_model_version,
  f5_ml_home_prob AS tracker_f5_ml_home_prob,
  f5_ml_away_prob AS tracker_f5_ml_away_prob,
  f5_rl_home_prob AS tracker_f5_rl_home_prob,
  f5_rl_away_prob AS tracker_f5_rl_away_prob,
  selected_market_type AS tracker_selected_market_type,
  selected_side AS tracker_selected_side,
  source_model,
  source_model_version,
  book_name,
  odds_at_bet AS tracker_odds_at_bet,
  line_at_bet,
  fair_probability AS tracker_fair_probability,
  model_probability AS tracker_model_probability,
  edge_pct AS tracker_edge_pct,
  ev AS tracker_ev,
  kelly_stake AS tracker_kelly_stake,
  no_pick_reason AS tracker_no_pick_reason,
  actual_status AS tracker_actual_status,
  actual_f5_home_score AS tracker_actual_f5_home_score,
  actual_f5_away_score AS tracker_actual_f5_away_score,
  actual_final_home_score AS tracker_actual_final_home_score,
  actual_final_away_score AS tracker_actual_final_away_score,
  settled_result,
  flat_profit_loss,
  created_at AS tracker_created_at,
  updated_at AS tracker_updated_at
FROM live_season_tracking
WHERE season = 2026
"""


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def _load_games(games_db: Path) -> pd.DataFrame:
    with sqlite3.connect(games_db) as connection:
        return pd.read_sql_query(
            """
            SELECT
                game_pk,
                date AS game_date,
                away_team,
                home_team,
                status AS game_status,
                f5_away_score,
                f5_home_score,
                final_away_score,
                final_home_score
            FROM games
            WHERE date LIKE '2026-%'
            ORDER BY date, game_pk
            """,
            connection,
        )


def _derive_result_columns(frame: pd.DataFrame) -> pd.DataFrame:
    derived = frame.copy()

    for score_col in ("f5_away_score", "f5_home_score", "final_away_score", "final_home_score"):
        derived[score_col] = pd.to_numeric(derived[score_col], errors="coerce")

    derived["f5_total_runs"] = derived["f5_away_score"] + derived["f5_home_score"]
    derived["final_total_runs"] = derived["final_away_score"] + derived["final_home_score"]
    derived["f5_run_margin"] = derived["f5_home_score"] - derived["f5_away_score"]
    derived["final_run_margin"] = derived["final_home_score"] - derived["final_away_score"]

    def _winner(home_score: float, away_score: float) -> str | None:
        if pd.isna(home_score) or pd.isna(away_score):
            return None
        if home_score > away_score:
            return "home"
        if away_score > home_score:
            return "away"
        return "push"

    derived["f5_winner"] = [
        _winner(home_score, away_score)
        for home_score, away_score in zip(derived["f5_home_score"], derived["f5_away_score"], strict=False)
    ]
    derived["final_winner"] = [
        _winner(home_score, away_score)
        for home_score, away_score in zip(
            derived["final_home_score"], derived["final_away_score"], strict=False
        )
    ]

    def _selected_result(row: pd.Series) -> str | None:
        settled_result = row.get("settled_result")
        if not pd.isna(settled_result):
            return str(settled_result)
        market = row.get("selected_market_type")
        side = row.get("selected_side")
        if pd.isna(market) or pd.isna(side):
            return None
        if str(row.get("game_status", "")).lower() != "final":
            return "pending"

        if market == "full_game_ml":
            winner = row.get("final_winner")
            return None if pd.isna(winner) else ("win" if side == winner else "loss" if winner != "push" else "push")
        if market == "f5_ml":
            winner = row.get("f5_winner")
            return None if pd.isna(winner) else ("win" if side == winner else "loss" if winner != "push" else "push")
        return "needs_line"

    derived["selected_pick_result"] = derived.apply(_selected_result, axis=1)
    return derived


def _prefer(merged: pd.DataFrame, primary: str, secondary: str) -> pd.Series:
    primary_series = merged[primary] if primary in merged.columns else pd.Series(index=merged.index, dtype=object)
    secondary_series = merged[secondary] if secondary in merged.columns else pd.Series(index=merged.index, dtype=object)
    return primary_series.combine_first(secondary_series)


def _prefer_authoritative(
    merged: pd.DataFrame,
    *,
    authority_mask: pd.Series,
    primary: str,
    secondary: str,
) -> pd.Series:
    secondary_series = merged[secondary] if secondary in merged.columns else pd.Series(index=merged.index, dtype=object)
    result = secondary_series.copy()
    if primary in merged.columns:
        result = result.where(~authority_mask, merged[primary])
    return result


def _load_latest_outputs(source_db: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with sqlite3.connect(source_db) as connection:
        pipeline_df = pd.read_sql_query(LATEST_PIPELINE_SQL, connection)
        predictions_df = pd.read_sql_query(LATEST_PREDICTIONS_SQL, connection)
        live_tracker_df = pd.read_sql_query(LIVE_TRACKER_SQL, connection)
    return pipeline_df, predictions_df, live_tracker_df


def _ensure_results_schema(connection: sqlite3.Connection) -> None:
    connection.execute("DROP TABLE IF EXISTS model_results")
    connection.execute(
        """
        CREATE TABLE model_results (
            game_pk INTEGER PRIMARY KEY,
            game_date TEXT NOT NULL,
            away_team TEXT NOT NULL,
            home_team TEXT NOT NULL,
            result_source TEXT,
            pipeline_date TEXT,
            tracker_status TEXT,
            game_status TEXT,
            f5_away_score INTEGER,
            f5_home_score INTEGER,
            final_away_score INTEGER,
            final_home_score INTEGER,
            f5_total_runs INTEGER,
            final_total_runs INTEGER,
            f5_run_margin INTEGER,
            final_run_margin INTEGER,
            f5_winner TEXT,
            final_winner TEXT,
            model_version TEXT,
            mode TEXT,
            pipeline_status TEXT,
            selected_market_type TEXT,
            selected_side TEXT,
            source_model TEXT,
            source_model_version TEXT,
            book_name TEXT,
            odds_at_bet INTEGER,
            line_at_bet REAL,
            fair_probability REAL,
            model_probability REAL,
            edge_pct REAL,
            ev REAL,
            kelly_stake REAL,
            no_pick_reason TEXT,
            f5_ml_home_prob REAL,
            f5_ml_away_prob REAL,
            f5_rl_home_prob REAL,
            f5_rl_away_prob REAL,
            predicted_at TEXT,
            created_at TEXT,
            updated_at TEXT,
            settled_result TEXT,
            flat_profit_loss REAL,
            selected_pick_result TEXT,
            has_exact_tracking INTEGER NOT NULL DEFAULT 0,
            has_pick INTEGER NOT NULL DEFAULT 0,
            has_prediction INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS build_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )


def _write_results(target_db: Path, frame: pd.DataFrame, metadata: dict[str, str]) -> None:
    target_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(target_db) as connection:
        _ensure_results_schema(connection)
        connection.execute("DELETE FROM model_results")
        rows = [
            (
                int(row["game_pk"]),
                str(row["game_date"]),
                str(row["away_team"]),
                str(row["home_team"]),
                None if pd.isna(row.get("result_source")) else str(row.get("result_source")),
                None if pd.isna(row.get("pipeline_date")) else str(row.get("pipeline_date")),
                None if pd.isna(row.get("tracker_status")) else str(row.get("tracker_status")),
                None if pd.isna(row.get("game_status")) else str(row.get("game_status")),
                None if pd.isna(row.get("f5_away_score")) else int(row.get("f5_away_score")),
                None if pd.isna(row.get("f5_home_score")) else int(row.get("f5_home_score")),
                None if pd.isna(row.get("final_away_score")) else int(row.get("final_away_score")),
                None if pd.isna(row.get("final_home_score")) else int(row.get("final_home_score")),
                None if pd.isna(row.get("f5_total_runs")) else int(row.get("f5_total_runs")),
                None if pd.isna(row.get("final_total_runs")) else int(row.get("final_total_runs")),
                None if pd.isna(row.get("f5_run_margin")) else int(row.get("f5_run_margin")),
                None if pd.isna(row.get("final_run_margin")) else int(row.get("final_run_margin")),
                None if pd.isna(row.get("f5_winner")) else str(row.get("f5_winner")),
                None if pd.isna(row.get("final_winner")) else str(row.get("final_winner")),
                None if pd.isna(row.get("model_version")) else str(row.get("model_version")),
                None if pd.isna(row.get("mode")) else str(row.get("mode")),
                None if pd.isna(row.get("pipeline_status")) else str(row.get("pipeline_status")),
                None if pd.isna(row.get("selected_market_type")) else str(row.get("selected_market_type")),
                None if pd.isna(row.get("selected_side")) else str(row.get("selected_side")),
                None if pd.isna(row.get("source_model")) else str(row.get("source_model")),
                None if pd.isna(row.get("source_model_version")) else str(row.get("source_model_version")),
                None if pd.isna(row.get("book_name")) else str(row.get("book_name")),
                None if pd.isna(row.get("odds_at_bet")) else int(row.get("odds_at_bet")),
                None if pd.isna(row.get("line_at_bet")) else float(row.get("line_at_bet")),
                None if pd.isna(row.get("fair_probability")) else float(row.get("fair_probability")),
                None if pd.isna(row.get("model_probability")) else float(row.get("model_probability")),
                None if pd.isna(row.get("edge_pct")) else float(row.get("edge_pct")),
                None if pd.isna(row.get("ev")) else float(row.get("ev")),
                None if pd.isna(row.get("kelly_stake")) else float(row.get("kelly_stake")),
                None if pd.isna(row.get("no_pick_reason")) else str(row.get("no_pick_reason")),
                None if pd.isna(row.get("f5_ml_home_prob")) else float(row.get("f5_ml_home_prob")),
                None if pd.isna(row.get("f5_ml_away_prob")) else float(row.get("f5_ml_away_prob")),
                None if pd.isna(row.get("f5_rl_home_prob")) else float(row.get("f5_rl_home_prob")),
                None if pd.isna(row.get("f5_rl_away_prob")) else float(row.get("f5_rl_away_prob")),
                None if pd.isna(row.get("predicted_at")) else str(row.get("predicted_at")),
                None if pd.isna(row.get("created_at")) else str(row.get("created_at")),
                None if pd.isna(row.get("updated_at")) else str(row.get("updated_at")),
                None if pd.isna(row.get("settled_result")) else str(row.get("settled_result")),
                None if pd.isna(row.get("flat_profit_loss")) else float(row.get("flat_profit_loss")),
                None if pd.isna(row.get("selected_pick_result")) else str(row.get("selected_pick_result")),
                int(bool(row.get("has_exact_tracking"))),
                int(bool(row.get("has_pick"))),
                int(bool(row.get("has_prediction"))),
            )
            for row in frame.to_dict(orient="records")
        ]
        connection.executemany(
            """
            INSERT INTO model_results (
                game_pk, game_date, away_team, home_team, result_source, pipeline_date, tracker_status,
                game_status, f5_away_score, f5_home_score,
                final_away_score, final_home_score, f5_total_runs, final_total_runs, f5_run_margin,
                final_run_margin, f5_winner, final_winner, model_version, mode, pipeline_status,
                selected_market_type, selected_side, source_model, source_model_version, book_name,
                odds_at_bet, line_at_bet, fair_probability, model_probability,
                edge_pct, ev, kelly_stake, no_pick_reason, f5_ml_home_prob, f5_ml_away_prob,
                f5_rl_home_prob, f5_rl_away_prob, predicted_at, created_at, updated_at,
                settled_result, flat_profit_loss, selected_pick_result, has_exact_tracking,
                has_pick, has_prediction
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        connection.execute("DELETE FROM build_metadata")
        connection.executemany(
            "INSERT INTO build_metadata (key, value) VALUES (?, ?)",
            list(metadata.items()),
        )
        connection.commit()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a dedicated blind 2026 model-results DB.")
    parser.add_argument("--source-db", default=str(DEFAULT_SOURCE_DB))
    parser.add_argument("--games-db", default=str(DEFAULT_GAMES_DB))
    parser.add_argument("--target-db", default=str(DEFAULT_TARGET_DB))
    parser.add_argument("--picks-only", action="store_true")
    args = parser.parse_args(argv)

    source_db = _resolve_path(args.source_db)
    games_db = _resolve_path(args.games_db)
    target_db = _resolve_path(args.target_db)

    games_df = _load_games(games_db)
    pipeline_df, predictions_df, live_tracker_df = _load_latest_outputs(source_db)
    merged = games_df.merge(
        live_tracker_df,
        on="game_pk",
        how="left",
    ).merge(
        pipeline_df[
            [
                "game_pk",
                "mode",
                "pipeline_status",
                "selected_market_type",
                "selected_side",
                "odds_at_bet",
                "fair_probability",
                "model_probability",
                "edge_pct",
                "ev",
                "kelly_stake",
                "no_pick_reason",
                "created_at",
            ]
        ],
        on="game_pk",
        how="left",
    ).merge(
        predictions_df[
            [
                "game_pk",
                "model_version",
                "f5_ml_home_prob",
                "f5_ml_away_prob",
                "f5_rl_home_prob",
                "f5_rl_away_prob",
                "predicted_at",
            ]
        ],
        on="game_pk",
        how="left",
    )
    tracker_present = merged["tracker_status"].notna()
    merged["result_source"] = tracker_present.map(
        lambda has_tracker: "live_tracker" if has_tracker else "pipeline_fallback"
    )
    merged["pipeline_date"] = merged.get("pipeline_date")
    merged["game_status"] = _prefer(merged, "tracker_actual_status", "game_status")
    merged["f5_home_score"] = _prefer(merged, "tracker_actual_f5_home_score", "f5_home_score")
    merged["f5_away_score"] = _prefer(merged, "tracker_actual_f5_away_score", "f5_away_score")
    merged["final_home_score"] = _prefer(merged, "tracker_actual_final_home_score", "final_home_score")
    merged["final_away_score"] = _prefer(merged, "tracker_actual_final_away_score", "final_away_score")
    merged["model_version"] = _prefer(merged, "tracker_model_version", "model_version")
    merged["selected_market_type"] = _prefer_authoritative(
        merged, authority_mask=tracker_present, primary="tracker_selected_market_type", secondary="selected_market_type"
    )
    merged["selected_side"] = _prefer_authoritative(
        merged, authority_mask=tracker_present, primary="tracker_selected_side", secondary="selected_side"
    )
    merged["source_model"] = _prefer_authoritative(
        merged, authority_mask=tracker_present, primary="source_model", secondary="source_model"
    )
    merged["source_model_version"] = _prefer_authoritative(
        merged, authority_mask=tracker_present, primary="source_model_version", secondary="source_model_version"
    )
    merged["book_name"] = _prefer_authoritative(
        merged, authority_mask=tracker_present, primary="book_name", secondary="book_name"
    )
    merged["odds_at_bet"] = _prefer_authoritative(
        merged, authority_mask=tracker_present, primary="tracker_odds_at_bet", secondary="odds_at_bet"
    )
    merged["line_at_bet"] = _prefer_authoritative(
        merged, authority_mask=tracker_present, primary="line_at_bet", secondary="line_at_bet"
    )
    merged["fair_probability"] = _prefer_authoritative(
        merged, authority_mask=tracker_present, primary="tracker_fair_probability", secondary="fair_probability"
    )
    merged["model_probability"] = _prefer_authoritative(
        merged, authority_mask=tracker_present, primary="tracker_model_probability", secondary="model_probability"
    )
    merged["edge_pct"] = _prefer_authoritative(
        merged, authority_mask=tracker_present, primary="tracker_edge_pct", secondary="edge_pct"
    )
    merged["ev"] = _prefer_authoritative(
        merged, authority_mask=tracker_present, primary="tracker_ev", secondary="ev"
    )
    merged["kelly_stake"] = _prefer_authoritative(
        merged, authority_mask=tracker_present, primary="tracker_kelly_stake", secondary="kelly_stake"
    )
    merged["no_pick_reason"] = _prefer_authoritative(
        merged, authority_mask=tracker_present, primary="tracker_no_pick_reason", secondary="no_pick_reason"
    )
    merged["f5_ml_home_prob"] = _prefer(merged, "tracker_f5_ml_home_prob", "f5_ml_home_prob")
    merged["f5_ml_away_prob"] = _prefer(merged, "tracker_f5_ml_away_prob", "f5_ml_away_prob")
    merged["f5_rl_home_prob"] = _prefer(merged, "tracker_f5_rl_home_prob", "f5_rl_home_prob")
    merged["f5_rl_away_prob"] = _prefer(merged, "tracker_f5_rl_away_prob", "f5_rl_away_prob")
    merged["created_at"] = _prefer_authoritative(
        merged, authority_mask=tracker_present, primary="tracker_created_at", secondary="created_at"
    )
    merged["updated_at"] = merged.get("tracker_updated_at")
    merged["has_exact_tracking"] = tracker_present
    merged["has_pick"] = merged["selected_market_type"].notna()
    merged["has_prediction"] = (
        merged["predicted_at"].notna()
        | merged["f5_ml_home_prob"].notna()
        | merged["f5_rl_home_prob"].notna()
    )
    merged = _derive_result_columns(merged)
    if args.picks_only:
        merged = merged.loc[merged["has_pick"]].copy()
    merged = merged.sort_values(["game_date", "game_pk"]).reset_index(drop=True)

    metadata = {
        "generated_at": datetime.now(UTC).isoformat(),
        "source_db": str(source_db),
        "games_db": str(games_db),
        "row_count": str(len(merged)),
        "pick_count": str(int(merged["has_pick"].sum()) if not merged.empty else 0),
        "prediction_count": str(int(merged["has_prediction"].sum()) if not merged.empty else 0),
        "exact_tracking_count": str(int(merged["has_exact_tracking"].sum()) if not merged.empty else 0),
        "picks_only": str(bool(args.picks_only)),
    }
    _write_results(target_db, merged, metadata)

    console.print(
        "[bold green]Built 2026 results DB[/bold green] "
        f"rows={len(merged)} picks={metadata['pick_count']} predictions={metadata['prediction_count']}"
    )
    console.print(f"db={target_db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
