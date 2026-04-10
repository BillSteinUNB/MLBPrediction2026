from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.db import DEFAULT_DB_PATH, sqlite_connection


DEFAULT_FANGRAPHS_BASERUNS_AUDIT_DIR = Path("data/reports/run_count/fangraphs_baseruns")


def build_repo_team_season_results_frame(
    *,
    season: int,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> pd.DataFrame:
    """Aggregate team-level actual season results from the repo game table."""

    with sqlite_connection(db_path, builder_optimized=True) as connection:
        games = pd.read_sql_query(
            """
            SELECT
                game_pk,
                date,
                home_team,
                away_team,
                final_home_score,
                final_away_score
            FROM games
            WHERE status = 'final'
              AND substr(date, 1, 4) = ?
            ORDER BY substr(date, 1, 10), game_pk
            """,
            connection,
            params=(str(int(season)),),
        )

    if games.empty:
        return pd.DataFrame(
            columns=[
                "season",
                "team",
                "repo_games",
                "repo_wins",
                "repo_losses",
                "repo_win_pct",
                "repo_runs_scored",
                "repo_runs_allowed",
                "repo_run_diff",
                "repo_runs_scored_per_game",
                "repo_runs_allowed_per_game",
                "repo_total_runs_per_game",
            ]
        )

    numeric_columns = ["final_home_score", "final_away_score"]
    for column in numeric_columns:
        games[column] = pd.to_numeric(games[column], errors="coerce")
    games = games.dropna(subset=numeric_columns).copy()
    if games.empty:
        return pd.DataFrame(
            columns=[
                "season",
                "team",
                "repo_games",
                "repo_wins",
                "repo_losses",
                "repo_win_pct",
                "repo_runs_scored",
                "repo_runs_allowed",
                "repo_run_diff",
                "repo_runs_scored_per_game",
                "repo_runs_allowed_per_game",
                "repo_total_runs_per_game",
            ]
        )

    home_rows = pd.DataFrame(
        {
            "team": games["home_team"].astype(str).str.strip().str.upper(),
            "runs_scored": games["final_home_score"].astype(float),
            "runs_allowed": games["final_away_score"].astype(float),
            "wins": (games["final_home_score"] > games["final_away_score"]).astype(int),
            "losses": (games["final_home_score"] < games["final_away_score"]).astype(int),
        }
    )
    away_rows = pd.DataFrame(
        {
            "team": games["away_team"].astype(str).str.strip().str.upper(),
            "runs_scored": games["final_away_score"].astype(float),
            "runs_allowed": games["final_home_score"].astype(float),
            "wins": (games["final_away_score"] > games["final_home_score"]).astype(int),
            "losses": (games["final_away_score"] < games["final_home_score"]).astype(int),
        }
    )
    team_games = pd.concat([home_rows, away_rows], ignore_index=True)

    summary = (
        team_games.groupby("team", as_index=False)
        .agg(
            repo_games=("team", "size"),
            repo_wins=("wins", "sum"),
            repo_losses=("losses", "sum"),
            repo_runs_scored=("runs_scored", "sum"),
            repo_runs_allowed=("runs_allowed", "sum"),
        )
        .sort_values(["repo_wins", "team"], ascending=[False, True])
        .reset_index(drop=True)
    )
    summary.insert(0, "season", int(season))
    summary["repo_win_pct"] = summary["repo_wins"] / summary["repo_games"].clip(lower=1)
    summary["repo_run_diff"] = summary["repo_runs_scored"] - summary["repo_runs_allowed"]
    summary["repo_runs_scored_per_game"] = (
        summary["repo_runs_scored"] / summary["repo_games"].clip(lower=1)
    )
    summary["repo_runs_allowed_per_game"] = (
        summary["repo_runs_allowed"] / summary["repo_games"].clip(lower=1)
    )
    summary["repo_total_runs_per_game"] = (
        (summary["repo_runs_scored"] + summary["repo_runs_allowed"])
        / summary["repo_games"].clip(lower=1)
    )
    return summary


def build_fangraphs_baseruns_audit_frame(
    fangraphs_frame: pd.DataFrame,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> pd.DataFrame:
    """Join FanGraphs BaseRuns standings to repo results and compute sanity metrics."""

    if fangraphs_frame.empty:
        raise ValueError("fangraphs_frame must not be empty")
    if "season" not in fangraphs_frame.columns:
        raise ValueError("fangraphs_frame must include a season column")

    seasons = sorted({int(value) for value in fangraphs_frame["season"].dropna().tolist()})
    if len(seasons) != 1:
        raise ValueError("fangraphs_frame must contain exactly one season for auditing")

    season = seasons[0]
    repo_frame = build_repo_team_season_results_frame(season=season, db_path=db_path)
    merged = repo_frame.merge(
        fangraphs_frame.copy(),
        on=["season", "team"],
        how="outer",
        suffixes=("", "_fangraphs"),
        indicator=True,
    )

    merged["repo_vs_fangraphs_games_delta"] = (
        merged["repo_games"].fillna(0.0) - merged["games"].fillna(0.0)
    )
    merged["repo_vs_fangraphs_wins_delta"] = (
        merged["repo_wins"].fillna(0.0) - merged["actual_wins"].fillna(0.0)
    )
    merged["repo_vs_fangraphs_runs_scored_per_game_delta"] = (
        merged["repo_runs_scored_per_game"].fillna(0.0)
        - merged["actual_runs_scored_per_game"].fillna(0.0)
    )
    merged["repo_vs_fangraphs_runs_allowed_per_game_delta"] = (
        merged["repo_runs_allowed_per_game"].fillna(0.0)
        - merged["actual_runs_allowed_per_game"].fillna(0.0)
    )

    merged["wins_minus_baseruns_wins"] = merged["repo_wins"] - merged["baseruns_wins"]
    merged["wins_minus_pythagorean_wins"] = merged["repo_wins"] - merged["pythagorean_wins"]
    merged["run_diff_minus_baseruns_run_diff"] = merged["repo_run_diff"] - merged["baseruns_run_diff"]
    merged["offensive_sequencing_delta_runs_per_game"] = (
        merged["repo_runs_scored_per_game"] - merged["baseruns_runs_scored_per_game"]
    )
    merged["preventive_sequencing_delta_runs_per_game"] = (
        merged["baseruns_runs_allowed_per_game"] - merged["repo_runs_allowed_per_game"]
    )
    merged["total_run_environment_delta_runs_per_game"] = (
        merged["repo_total_runs_per_game"]
        - (
            merged["baseruns_runs_scored_per_game"]
            + merged["baseruns_runs_allowed_per_game"]
        )
    )
    merged["baseruns_minus_pythagorean_wins"] = (
        merged["baseruns_wins"] - merged["pythagorean_wins"]
    )

    return merged.sort_values(
        ["wins_minus_baseruns_wins", "team"],
        ascending=[False, True],
    ).reset_index(drop=True)


def build_fangraphs_baseruns_audit_summary(
    audit_frame: pd.DataFrame,
    *,
    top_n: int = 10,
) -> dict[str, Any]:
    """Build a compact JSON-serializable summary for the markdown and JSON reports."""

    if audit_frame.empty:
        raise ValueError("audit_frame must not be empty")

    return {
        "season": int(pd.to_numeric(audit_frame["season"], errors="coerce").dropna().iloc[0]),
        "team_count": int(len(audit_frame)),
        "missing_repo_teams": sorted(
            audit_frame.loc[audit_frame["_merge"] == "right_only", "team"].dropna().astype(str).tolist()
        ),
        "missing_fangraphs_teams": sorted(
            audit_frame.loc[audit_frame["_merge"] == "left_only", "team"].dropna().astype(str).tolist()
        ),
        "max_repo_vs_fangraphs_games_delta": float(
            audit_frame["repo_vs_fangraphs_games_delta"].abs().max()
        ),
        "max_repo_vs_fangraphs_wins_delta": float(
            audit_frame["repo_vs_fangraphs_wins_delta"].abs().max()
        ),
        "top_overperformers_vs_baseruns": _select_team_rows(
            audit_frame,
            column="wins_minus_baseruns_wins",
            ascending=False,
            top_n=top_n,
        ),
        "top_underperformers_vs_baseruns": _select_team_rows(
            audit_frame,
            column="wins_minus_baseruns_wins",
            ascending=True,
            top_n=top_n,
        ),
        "top_positive_offensive_sequencing": _select_team_rows(
            audit_frame,
            column="offensive_sequencing_delta_runs_per_game",
            ascending=False,
            top_n=top_n,
        ),
        "top_positive_prevention_sequencing": _select_team_rows(
            audit_frame,
            column="preventive_sequencing_delta_runs_per_game",
            ascending=False,
            top_n=top_n,
        ),
        "top_negative_run_environment_gaps": _select_team_rows(
            audit_frame,
            column="total_run_environment_delta_runs_per_game",
            ascending=True,
            top_n=top_n,
        ),
        "top_positive_run_environment_gaps": _select_team_rows(
            audit_frame,
            column="total_run_environment_delta_runs_per_game",
            ascending=False,
            top_n=top_n,
        ),
    }


def write_fangraphs_baseruns_audit_outputs(
    *,
    audit_frame: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: str | Path = DEFAULT_FANGRAPHS_BASERUNS_AUDIT_DIR,
) -> dict[str, Path]:
    """Write CSV, JSON, and Markdown outputs for the Fangraphs sanity audit."""

    if audit_frame.empty:
        raise ValueError("audit_frame must not be empty")

    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    season = int(summary["season"])

    csv_path = resolved_output_dir / f"fangraphs_baseruns_audit_{season}.csv"
    json_path = resolved_output_dir / f"fangraphs_baseruns_audit_{season}.summary.json"
    markdown_path = resolved_output_dir / f"fangraphs_baseruns_audit_{season}.md"

    audit_frame.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(tz=timezone.utc).isoformat(),
                "summary": summary,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    markdown_path.write_text(
        build_fangraphs_baseruns_markdown_report(audit_frame, summary=summary),
        encoding="utf-8",
    )
    return {
        "csv": csv_path,
        "summary_json": json_path,
        "markdown": markdown_path,
    }


def build_fangraphs_baseruns_markdown_report(
    audit_frame: pd.DataFrame,
    *,
    summary: dict[str, Any],
) -> str:
    """Render a short markdown report for the FanGraphs sanity audit."""

    season = int(summary["season"])
    overperformers = _as_table_frame(summary["top_overperformers_vs_baseruns"])
    underperformers = _as_table_frame(summary["top_underperformers_vs_baseruns"])
    offense = _as_table_frame(summary["top_positive_offensive_sequencing"])
    prevention = _as_table_frame(summary["top_positive_prevention_sequencing"])

    lines = [
        f"# FanGraphs BaseRuns Sanity Audit ({season})",
        "",
        "This report joins FanGraphs season standings to repo-local game results and treats BaseRuns as an external regression sanity check.",
        "",
        "## Coverage",
        "",
        f"- Teams in audit: {summary['team_count']}",
        f"- Missing repo teams: {', '.join(summary['missing_repo_teams']) or 'none'}",
        f"- Missing FanGraphs teams: {', '.join(summary['missing_fangraphs_teams']) or 'none'}",
        f"- Max repo vs FanGraphs games delta: {summary['max_repo_vs_fangraphs_games_delta']:.3f}",
        f"- Max repo vs FanGraphs wins delta: {summary['max_repo_vs_fangraphs_wins_delta']:.3f}",
        "",
        "## Biggest Overperformers Vs BaseRuns",
        "",
        _markdown_table(
            overperformers,
            [
                "team",
                "wins_minus_baseruns_wins",
                "offensive_sequencing_delta_runs_per_game",
                "preventive_sequencing_delta_runs_per_game",
            ],
        ),
        "",
        "## Biggest Underperformers Vs BaseRuns",
        "",
        _markdown_table(
            underperformers,
            [
                "team",
                "wins_minus_baseruns_wins",
                "offensive_sequencing_delta_runs_per_game",
                "preventive_sequencing_delta_runs_per_game",
            ],
        ),
        "",
        "## Offensive Sequencing Gaps",
        "",
        _markdown_table(
            offense,
            [
                "team",
                "offensive_sequencing_delta_runs_per_game",
                "repo_runs_scored_per_game",
                "baseruns_runs_scored_per_game",
            ],
        ),
        "",
        "## Prevention Sequencing Gaps",
        "",
        _markdown_table(
            prevention,
            [
                "team",
                "preventive_sequencing_delta_runs_per_game",
                "repo_runs_allowed_per_game",
                "baseruns_runs_allowed_per_game",
            ],
        ),
        "",
        "## Notes",
        "",
        "- Positive `wins_minus_baseruns_wins` means the team won more games than BaseRuns expected.",
        "- Positive `offensive_sequencing_delta_runs_per_game` means actual RS/G beat BaseRuns RS/G.",
        "- Positive `preventive_sequencing_delta_runs_per_game` means actual RA/G was lower than BaseRuns RA/G.",
    ]
    return "\n".join(lines).strip() + "\n"


def _select_team_rows(
    frame: pd.DataFrame,
    *,
    column: str,
    ascending: bool,
    top_n: int,
) -> list[dict[str, Any]]:
    requested_columns = [
        "team",
        column,
        "repo_runs_scored_per_game",
        "repo_runs_allowed_per_game",
        "baseruns_runs_scored_per_game",
        "baseruns_runs_allowed_per_game",
        "wins_minus_baseruns_wins",
        "offensive_sequencing_delta_runs_per_game",
        "preventive_sequencing_delta_runs_per_game",
        "total_run_environment_delta_runs_per_game",
    ]
    available_columns = list(
        dict.fromkeys(column_name for column_name in requested_columns if column_name in frame.columns)
    )
    ordered = frame.loc[frame["_merge"] == "both", available_columns].sort_values(
        column,
        ascending=ascending,
    )
    return ordered.head(top_n).to_dict(orient="records")


def _as_table_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows available._"

    available_columns = [column for column in columns if column in frame.columns]
    working = frame.loc[:, available_columns].copy()
    for column in available_columns:
        if pd.api.types.is_numeric_dtype(working[column]):
            working[column] = working[column].map(lambda value: f"{float(value):.3f}")
        else:
            working[column] = working[column].astype(str)

    header = "| " + " | ".join(available_columns) + " |"
    divider = "| " + " | ".join(["---"] * len(available_columns)) + " |"
    rows = [
        "| " + " | ".join(working.iloc[index][available_columns].tolist()) + " |"
        for index in range(len(working))
    ]
    return "\n".join([header, divider, *rows])


__all__ = [
    "DEFAULT_FANGRAPHS_BASERUNS_AUDIT_DIR",
    "build_fangraphs_baseruns_audit_frame",
    "build_fangraphs_baseruns_audit_summary",
    "build_fangraphs_baseruns_markdown_report",
    "build_repo_team_season_results_frame",
    "write_fangraphs_baseruns_audit_outputs",
]
