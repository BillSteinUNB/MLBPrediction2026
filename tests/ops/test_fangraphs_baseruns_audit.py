from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from src.db import init_db
from src.ops.fangraphs_baseruns_audit import (
    build_fangraphs_baseruns_audit_frame,
    build_fangraphs_baseruns_audit_summary,
    build_repo_team_season_results_frame,
)


def _seed_games(db_path: Path) -> Path:
    with sqlite3.connect(init_db(db_path)) as connection:
        connection.executemany(
            """
            INSERT INTO games (
                game_pk,
                date,
                home_team,
                away_team,
                venue,
                is_dome,
                is_abs_active,
                f5_home_score,
                f5_away_score,
                final_home_score,
                final_away_score,
                status
            )
            VALUES (?, ?, ?, ?, ?, 0, 1, ?, ?, ?, ?, 'final')
            """,
            [
                (
                    1001,
                    "2025-04-01T19:05:00+00:00",
                    "NYY",
                    "BOS",
                    "Yankee Stadium",
                    3,
                    2,
                    5,
                    3,
                ),
                (
                    1002,
                    "2025-04-02T19:05:00+00:00",
                    "BOS",
                    "NYY",
                    "Fenway Park",
                    2,
                    4,
                    4,
                    6,
                ),
            ],
        )
        connection.commit()
    return db_path


def _fangraphs_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": 2025,
                "team": "NYY",
                "fangraphs_team_name": "Yankees",
                "fangraphs_abbreviation": "NYY",
                "games": 2,
                "actual_wins": 2,
                "actual_losses": 0,
                "actual_win_pct": 1.0,
                "actual_run_diff": 4.0,
                "actual_runs_scored_per_game": 5.5,
                "actual_runs_allowed_per_game": 3.5,
                "pythagorean_wins": 1.7,
                "pythagorean_losses": 0.3,
                "pythagorean_win_pct": 0.85,
                "pythagorean_win_delta": 0.3,
                "baseruns_wins": 1.5,
                "baseruns_losses": 0.5,
                "baseruns_win_pct": 0.75,
                "baseruns_win_delta": 0.5,
                "baseruns_run_diff": 2.0,
                "baseruns_runs_scored_per_game": 5.1,
                "baseruns_runs_allowed_per_game": 4.1,
            },
            {
                "season": 2025,
                "team": "BOS",
                "fangraphs_team_name": "Red Sox",
                "fangraphs_abbreviation": "BOS",
                "games": 2,
                "actual_wins": 0,
                "actual_losses": 2,
                "actual_win_pct": 0.0,
                "actual_run_diff": -4.0,
                "actual_runs_scored_per_game": 3.5,
                "actual_runs_allowed_per_game": 5.5,
                "pythagorean_wins": 0.3,
                "pythagorean_losses": 1.7,
                "pythagorean_win_pct": 0.15,
                "pythagorean_win_delta": -0.3,
                "baseruns_wins": 0.5,
                "baseruns_losses": 1.5,
                "baseruns_win_pct": 0.25,
                "baseruns_win_delta": -0.5,
                "baseruns_run_diff": -2.0,
                "baseruns_runs_scored_per_game": 4.1,
                "baseruns_runs_allowed_per_game": 5.1,
            },
        ]
    )


def test_build_repo_team_season_results_frame_aggregates_home_and_away_games(tmp_path: Path) -> None:
    db_path = _seed_games(tmp_path / "mlb.db")

    frame = build_repo_team_season_results_frame(season=2025, db_path=db_path)

    nyy = frame.loc[frame["team"] == "NYY"].iloc[0]
    bos = frame.loc[frame["team"] == "BOS"].iloc[0]
    assert int(nyy["repo_games"]) == 2
    assert int(nyy["repo_wins"]) == 2
    assert float(nyy["repo_runs_scored_per_game"]) == pytest.approx(5.5)
    assert float(nyy["repo_runs_allowed_per_game"]) == pytest.approx(3.5)
    assert int(bos["repo_losses"]) == 2
    assert float(bos["repo_total_runs_per_game"]) == pytest.approx(9.0)


def test_build_fangraphs_baseruns_audit_frame_computes_sequencing_deltas(tmp_path: Path) -> None:
    db_path = _seed_games(tmp_path / "mlb.db")

    audit = build_fangraphs_baseruns_audit_frame(_fangraphs_frame(), db_path=db_path)

    nyy = audit.loc[audit["team"] == "NYY"].iloc[0]
    bos = audit.loc[audit["team"] == "BOS"].iloc[0]
    assert float(nyy["wins_minus_baseruns_wins"]) == pytest.approx(0.5)
    assert float(nyy["offensive_sequencing_delta_runs_per_game"]) == pytest.approx(0.4)
    assert float(nyy["preventive_sequencing_delta_runs_per_game"]) == pytest.approx(0.6)
    assert float(bos["wins_minus_baseruns_wins"]) == pytest.approx(-0.5)
    assert float(bos["total_run_environment_delta_runs_per_game"]) == pytest.approx(-0.2)


def test_build_fangraphs_baseruns_audit_summary_returns_ranked_sections(tmp_path: Path) -> None:
    db_path = _seed_games(tmp_path / "mlb.db")
    audit = build_fangraphs_baseruns_audit_frame(_fangraphs_frame(), db_path=db_path)

    summary = build_fangraphs_baseruns_audit_summary(audit, top_n=1)

    assert summary["season"] == 2025
    assert summary["team_count"] == 2
    assert summary["top_overperformers_vs_baseruns"][0]["team"] == "NYY"
    assert summary["top_underperformers_vs_baseruns"][0]["team"] == "BOS"
