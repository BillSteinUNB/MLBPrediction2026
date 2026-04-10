from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from src.db import init_db
from src.ops.eva_statcast_audit import (
    build_eva_statcast_audit_summary,
    build_eva_statcast_metric_summary,
    build_eva_statcast_starter_audit_summary,
    build_eva_statcast_starter_metric_summary,
    build_eva_statcast_starter_start_audit_frame,
    build_eva_statcast_team_game_audit_frame,
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
                home_starter_id,
                away_starter_id,
                venue,
                is_dome,
                is_abs_active,
                f5_home_score,
                f5_away_score,
                final_home_score,
                final_away_score,
                status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, 0, 1, ?, ?, ?, ?, ?)
            """,
            [
                (1001, "2025-04-01T23:05:00+00:00", "NYY", "BOS", 700, 800, "Yankee Stadium", 1, 0, 2, 0, "final"),
                (1002, "2025-04-02T23:05:00+00:00", "NYY", "BOS", 700, 800, "Yankee Stadium", 0, 2, 1, 5, "final"),
                (1003, "2025-04-03T23:05:00+00:00", "NYY", "BOS", 700, 800, "Yankee Stadium", 2, 1, 4, 3, "final"),
            ],
        )
        connection.commit()
    return db_path


def _statcast_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_pk": 1001,
                "game_date": "2025-04-01",
                "at_bat_number": 1,
                "pitch_number": 1,
                "inning_topbot": "Top",
                "home_team": "NYY",
                "away_team": "BOS",
                "pitcher": 800,
                "events": "single",
                "estimated_woba_using_speedangle": 0.70,
                "launch_speed": 100.0,
                "launch_angle": 25.0,
                "bb_type": "fly_ball",
            },
            {
                "game_pk": 1001,
                "game_date": "2025-04-01",
                "at_bat_number": 2,
                "pitch_number": 1,
                "inning_topbot": "Top",
                "home_team": "NYY",
                "away_team": "BOS",
                "pitcher": 800,
                "events": "single",
                "estimated_woba_using_speedangle": 0.40,
                "launch_speed": 90.0,
                "launch_angle": 12.0,
                "bb_type": "line_drive",
            },
            {
                "game_pk": 1001,
                "game_date": "2025-04-01",
                "at_bat_number": 3,
                "pitch_number": 1,
                "inning_topbot": "Bot",
                "home_team": "NYY",
                "away_team": "BOS",
                "pitcher": 700,
                "events": "groundout",
                "estimated_woba_using_speedangle": 0.10,
                "launch_speed": 80.0,
                "launch_angle": -5.0,
                "bb_type": "ground_ball",
            },
            {
                "game_pk": 1002,
                "game_date": "2025-04-02",
                "at_bat_number": 1,
                "pitch_number": 1,
                "inning_topbot": "Top",
                "home_team": "NYY",
                "away_team": "BOS",
                "pitcher": 800,
                "events": "double",
                "estimated_woba_using_speedangle": 0.65,
                "launch_speed": 98.0,
                "launch_angle": 24.0,
                "bb_type": "fly_ball",
            },
            {
                "game_pk": 1002,
                "game_date": "2025-04-02",
                "at_bat_number": 2,
                "pitch_number": 1,
                "inning_topbot": "Top",
                "home_team": "NYY",
                "away_team": "BOS",
                "pitcher": 800,
                "events": "single",
                "estimated_woba_using_speedangle": 0.50,
                "launch_speed": 96.0,
                "launch_angle": 18.0,
                "bb_type": "line_drive",
            },
            {
                "game_pk": 1002,
                "game_date": "2025-04-02",
                "at_bat_number": 3,
                "pitch_number": 1,
                "inning_topbot": "Bot",
                "home_team": "NYY",
                "away_team": "BOS",
                "pitcher": 700,
                "events": "single",
                "estimated_woba_using_speedangle": 0.20,
                "launch_speed": 82.0,
                "launch_angle": 2.0,
                "bb_type": "ground_ball",
            },
            {
                "game_pk": 1003,
                "game_date": "2025-04-03",
                "at_bat_number": 1,
                "pitch_number": 1,
                "inning_topbot": "Top",
                "home_team": "NYY",
                "away_team": "BOS",
                "pitcher": 800,
                "events": "single",
                "estimated_woba_using_speedangle": 0.30,
                "launch_speed": 88.0,
                "launch_angle": 5.0,
                "bb_type": "ground_ball",
            },
            {
                "game_pk": 1003,
                "game_date": "2025-04-03",
                "at_bat_number": 2,
                "pitch_number": 1,
                "inning_topbot": "Bot",
                "home_team": "NYY",
                "away_team": "BOS",
                "pitcher": 700,
                "events": "home_run",
                "estimated_woba_using_speedangle": 0.92,
                "launch_speed": 104.0,
                "launch_angle": 27.0,
                "bb_type": "fly_ball",
            },
        ]
    )


def _eva_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "stat": "EXIT VELOCITY (AVERAGE)",
                "type": "predictive",
                "woba_corr": 0.45,
                "hr_pct_corr": 0.57,
                "babip_corr": -0.05,
                "iso_corr": 0.54,
                "ba_corr": 0.11,
            },
            {
                "stat": "BARREL%",
                "type": "predictive",
                "woba_corr": 0.38,
                "hr_pct_corr": 0.70,
                "babip_corr": -0.16,
                "iso_corr": 0.61,
                "ba_corr": -0.09,
            },
            {
                "stat": "LAUNCH ANGLE (-4 TO 26%)",
                "type": "predictive",
                "woba_corr": 0.07,
                "hr_pct_corr": -0.15,
                "babip_corr": 0.24,
                "iso_corr": -0.11,
                "ba_corr": 0.18,
            },
        ]
    )


def test_build_eva_statcast_team_game_audit_frame_uses_prior_games_only(tmp_path: Path) -> None:
    db_path = _seed_games(tmp_path / "mlb.db")

    audit = build_eva_statcast_team_game_audit_frame(
        season=2025,
        db_path=db_path,
        rolling_window=2,
        statcast_frame=_statcast_frame(),
    )

    bos_second = audit.loc[(audit["team"] == "BOS") & (audit["game_pk"] == 1002)].iloc[0]
    assert bos_second["rolling_2g_games_with_history"] == 1
    assert bos_second["rolling_2g_avg_exit_velocity"] == pytest.approx(95.0)
    assert bos_second["rolling_2g_barrel_pct"] == pytest.approx(50.0)
    assert bos_second["rolling_2g_xwoba"] == pytest.approx(0.55)
    assert bos_second["rolling_2g_launch_angle_23_34_pct"] == pytest.approx(50.0)
    assert bos_second["rolling_2g_launch_angle_neg4_26_pct"] == pytest.approx(100.0)

    bos_first = audit.loc[(audit["team"] == "BOS") & (audit["game_pk"] == 1001)].iloc[0]
    assert pd.isna(bos_first["rolling_2g_avg_exit_velocity"])


def test_build_eva_statcast_metric_summary_and_audit_summary_rank_targets(tmp_path: Path) -> None:
    db_path = _seed_games(tmp_path / "mlb.db")
    audit = build_eva_statcast_team_game_audit_frame(
        season=2025,
        db_path=db_path,
        rolling_window=2,
        statcast_frame=_statcast_frame(),
    )

    metric_summary = build_eva_statcast_metric_summary(
        audit,
        rolling_window=2,
        eva_correlations=_eva_frame(),
    )

    avg_exit_velocity = metric_summary.loc[
        metric_summary["metric_key"] == "avg_exit_velocity"
    ].iloc[0]
    assert bool(avg_exit_velocity["existing_repo_feature"]) is False
    assert avg_exit_velocity["eva_predictive_woba_corr"] == 0.45

    xwoba = metric_summary.loc[metric_summary["metric_key"] == "xwoba"].iloc[0]
    assert bool(xwoba["existing_repo_feature"]) is True
    assert pd.isna(xwoba["eva_predictive_woba_corr"])

    summary = build_eva_statcast_audit_summary(
        audit,
        metric_summary=metric_summary,
        rolling_window=2,
        eva_correlations=_eva_frame(),
        top_n=3,
    )

    assert summary["eva_recommended_usage"] == "research_snapshot"
    assert summary["recommended_feature_targets"]
    recommended_metric_keys = {
        row["metric_key"] for row in summary["recommended_feature_targets"]
    }
    assert "avg_exit_velocity" in recommended_metric_keys


def test_build_eva_statcast_starter_audit_ranks_allowed_contact_targets(tmp_path: Path) -> None:
    db_path = _seed_games(tmp_path / "mlb.db")

    audit = build_eva_statcast_starter_start_audit_frame(
        season=2025,
        db_path=db_path,
        rolling_window=2,
        statcast_frame=_statcast_frame(),
    )

    away_starter_second = audit.loc[
        (audit["pitcher_id"] == 800) & (audit["game_pk"] == 1002)
    ].iloc[0]
    assert away_starter_second["rolling_2s_starts_with_history"] == 1
    assert away_starter_second["rolling_2s_avg_exit_velocity"] == pytest.approx(95.0)
    assert away_starter_second["rolling_2s_barrel_pct"] == pytest.approx(50.0)
    assert away_starter_second["runs_allowed"] == pytest.approx(1.0)
    assert away_starter_second["f5_runs_allowed"] == pytest.approx(0.0)

    metric_summary = build_eva_statcast_starter_metric_summary(
        audit,
        rolling_window=2,
        eva_correlations=_eva_frame(),
    )
    avg_exit_velocity = metric_summary.loc[
        metric_summary["metric_key"] == "avg_exit_velocity"
    ].iloc[0]
    assert avg_exit_velocity["eva_predictive_woba_corr"] == 0.45

    summary = build_eva_statcast_starter_audit_summary(
        audit,
        metric_summary=metric_summary,
        rolling_window=2,
        eva_correlations=_eva_frame(),
        top_n=3,
    )
    assert summary["recommended_feature_targets"]
    starter_metric_keys = {row["metric_key"] for row in summary["recommended_feature_targets"]}
    assert "avg_exit_velocity" in starter_metric_keys
