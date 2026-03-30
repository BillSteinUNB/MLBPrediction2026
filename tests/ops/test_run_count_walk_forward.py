from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import PoissonRegressor

from src.model.data_builder import (
    RUN_COUNT_REQUIRED_TEMPORAL_DELTA_COLUMNS,
    _json_bytes,
    _run_count_training_schema_metadata,
    _write_parquet_with_metadata,
)
from src.model.run_distribution_trainer import train_run_distribution_model
from src.ops.run_count_walk_forward import (
    evaluate_mcmc_holdout_walk_forward,
    evaluate_stage3_holdout_walk_forward,
)


def _training_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    start = datetime(2024, 4, 1, 19, 5, tzinfo=UTC)
    for index in range(20):
        scheduled_start = start + timedelta(days=index * 7)
        season = 2024 if index < 12 else 2025
        rows.append(
            {
                "game_pk": 60_000 + index,
                "season": season,
                "game_date": scheduled_start.date().isoformat(),
                "scheduled_start": scheduled_start.isoformat(),
                "as_of_timestamp": (scheduled_start - timedelta(days=1)).isoformat(),
                "home_team": "NYY",
                "away_team": "BOS",
                "venue": "Yankee Stadium",
                "game_type": "R",
                "build_timestamp": scheduled_start.isoformat(),
                "data_version_hash": "synthetic-run-count-wf",
                "home_team_log5_30g": 0.52 + (0.01 * (index % 3)),
                "away_team_log5_30g": 0.48 - (0.01 * (index % 3)),
                "home_starter_k_pct_30s": 22.0 + (index % 4),
                "home_starter_k_pct_7s": 21.0 + (index % 3),
                "home_starter_bb_pct_30s": 7.4 + (0.2 * (index % 3)),
                "home_starter_gb_pct_30s": 42.0 + (index % 4),
                "home_starter_csw_pct_30s": 28.0 + (index % 3),
                "home_starter_avg_fastball_velocity_30s": 93.0 + (0.4 * (index % 4)),
                "home_starter_pitch_mix_entropy_30s": 2.20 + (0.05 * (index % 4)),
                "home_starter_last_start_pitch_count": 90.0 + (index % 5),
                "home_starter_cumulative_pitch_load_5s": 430.0 + (index % 6) * 8.0,
                "home_team_bullpen_pitch_count_3d": 42.0 + (index % 5),
                "away_lineup_woba_30g": 0.315 + (0.004 * (index % 4)),
                "away_lineup_woba_delta_7v30g": -0.01 + (0.005 * (index % 4)),
                "away_lineup_bb_pct_30g": 8.0 + (0.3 * (index % 4)),
                "away_lineup_k_pct_30g": 22.0 - (0.5 * (index % 4)),
                "away_lineup_iso_30g": 0.155 + (0.006 * (index % 4)),
                "away_lineup_barrel_pct_30g": 7.0 + (0.4 * (index % 4)),
                "away_lineup_babip_30g": 0.292 + (0.003 * (index % 4)),
                "away_lineup_xwoba_30g": 0.316 + (0.004 * (index % 4)),
                "away_lineup_woba_minus_xwoba_30g": -0.006 + (0.003 * (index % 4)),
                "away_lineup_lhb_pct": 0.33,
                "away_lineup_rhb_pct": 0.56,
                "away_lineup_platoon_advantage_pct": 0.55,
                "home_team_oaa_30g": 1.0 + (index % 3),
                "home_team_oaa_season": 0.4,
                "home_team_drs_30g": 2.0 + (index % 3),
                "home_team_drs_season": 1.0,
                "home_team_defensive_efficiency_30g": 0.705 + (0.003 * (index % 3)),
                "home_team_defensive_efficiency_season": 0.702,
                "home_team_adjusted_framing_30g": 1.0 + (0.2 * (index % 3)),
                "home_team_adjusted_framing_60g": 0.8 + (0.1 * (index % 3)),
                "home_team_adjusted_framing_season": 0.6,
                "plate_umpire_sample_size_30g": 20.0 + index,
                "plate_umpire_total_runs_avg_30g": 8.6 - (0.05 * (index % 3)),
                "plate_umpire_total_runs_avg_90g": 8.8,
                "plate_umpire_f5_total_runs_avg_30g": 4.3 - (0.04 * (index % 3)),
                "plate_umpire_f5_total_runs_avg_90g": 4.5,
                "plate_umpire_home_win_pct_30g": 0.53,
                "plate_umpire_home_win_pct_90g": 0.54,
                "weather_air_density_factor": 1.0 + (0.01 * (index % 3)),
                "weather_temp_factor": 1.0 + (0.01 * (index % 2)),
                "weather_composite": 1.0 + (0.02 * (index % 3)),
                "weather_wind_factor": 4.0 + (index % 4),
                "park_runs_factor": 0.98 + (0.01 * (index % 3)),
                "park_hr_factor": 0.99 + (0.02 * (index % 3)),
                "home_team_bullpen_xfip": 4.0 + (0.1 * (index % 3)),
                "f5_home_score": 2,
                "f5_away_score": 1,
                "final_home_score": 4,
                "final_away_score": 3 + (index % 4),
            }
        )
    frame = pd.DataFrame(rows)
    for column_index, column_name in enumerate(RUN_COUNT_REQUIRED_TEMPORAL_DELTA_COLUMNS, start=1):
        if column_name not in frame.columns:
            frame[column_name] = (frame.index + column_index) * 0.01
    metadata = _run_count_training_schema_metadata()
    frame.attrs.update(metadata)
    frame.attrs["run_count_training_schema"] = metadata
    return frame


def _write_training_parquet(dataframe: pd.DataFrame, output_path: Path) -> None:
    metadata_payload = _run_count_training_schema_metadata()
    _write_parquet_with_metadata(
        dataframe,
        output_path,
        parquet_metadata={
            b"mlbprediction2026.run_count_training_schema": _json_bytes(metadata_payload)
        },
    )


def _write_mean_artifact(frame: pd.DataFrame, output_dir: Path) -> Path:
    feature_columns = [
        "home_team_log5_30g",
        "away_team_log5_30g",
        "home_starter_k_pct_30s",
        "park_runs_factor",
        "weather_air_density_factor",
        "away_lineup_woba_30g",
    ]
    model = PoissonRegressor(alpha=0.0, max_iter=500)
    train_frame = frame.loc[frame["season"] < 2025].copy()
    model.fit(train_frame.loc[:, feature_columns], train_frame["final_away_score"])

    output_dir.mkdir(parents=True, exist_ok=True)
    version = "20260328T000000Z_walkforward"
    model_path = output_dir / f"full_game_away_runs_model_{version}.joblib"
    metadata_path = output_dir / f"full_game_away_runs_model_{version}.metadata.json"
    joblib.dump(model, model_path)
    metadata_path.write_text(
        json.dumps(
            {
                "model_name": "full_game_away_runs_model",
                "target_column": "final_away_score",
                "model_version": version,
                "data_version_hash": "synthetic-run-count-wf",
                "holdout_season": 2025,
                "feature_columns": feature_columns,
                "best_params": {"max_depth": 2, "n_estimators": 10, "learning_rate": 0.1},
                "model_family": "xgboost_lightgbm_blend",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return metadata_path


def _seed_old_scraper_totals_db(frame: pd.DataFrame, db_path: Path) -> Path:
    holdout = frame.loc[frame["season"] == 2025].copy().reset_index(drop=True)
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE games (
                game_id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT,
                game_date TEXT NOT NULL,
                commence_time_utc TEXT,
                away_team TEXT NOT NULL,
                home_team TEXT NOT NULL,
                game_type TEXT,
                away_pitcher TEXT,
                home_pitcher TEXT
            );
            CREATE TABLE odds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT,
                game_date TEXT NOT NULL,
                commence_time TEXT,
                away_team TEXT NOT NULL,
                home_team TEXT NOT NULL,
                game_type TEXT,
                away_pitcher TEXT,
                home_pitcher TEXT,
                fetched_at TEXT,
                bookmaker TEXT NOT NULL,
                market_type TEXT NOT NULL,
                side TEXT NOT NULL,
                point TEXT,
                price TEXT NOT NULL,
                commence_time_utc TEXT,
                is_opening INTEGER DEFAULT 0,
                game_id INTEGER
            );
            """
        )
        for row in holdout.itertuples(index=False):
            event_id = f"evt-{row.game_pk}"
            connection.execute(
                """
                INSERT INTO games (event_id, game_date, commence_time_utc, away_team, home_team, game_type)
                VALUES (?, ?, ?, ?, ?, 'R')
                """,
                (
                    event_id,
                    str(row.game_date),
                    str(row.scheduled_start),
                    str(row.away_team),
                    str(row.home_team),
                ),
            )
            total_line = 8.5 + (0.5 * (int(row.game_pk) % 2))
            f5_line = 4.5 + (0.5 * (int(row.game_pk) % 2))
            connection.executemany(
                """
                INSERT INTO odds (
                    event_id, game_date, commence_time, away_team, home_team, fetched_at,
                    bookmaker, market_type, side, point, price, commence_time_utc, is_opening
                )
                VALUES (?, ?, ?, ?, ?, ?, 'DraftKings', ?, ?, ?, ?, ?, 1)
                """,
                [
                    (
                        event_id,
                        str(row.game_date),
                        str(row.scheduled_start),
                        str(row.away_team),
                        str(row.home_team),
                        str(row.as_of_timestamp),
                        "full_game_total",
                        "over",
                        str(total_line),
                        "-110",
                        str(row.scheduled_start),
                    ),
                    (
                        event_id,
                        str(row.game_date),
                        str(row.scheduled_start),
                        str(row.away_team),
                        str(row.home_team),
                        str(row.as_of_timestamp),
                        "full_game_total",
                        "under",
                        str(total_line),
                        "-110",
                        str(row.scheduled_start),
                    ),
                    (
                        event_id,
                        str(row.game_date),
                        str(row.scheduled_start),
                        str(row.away_team),
                        str(row.home_team),
                        str(row.as_of_timestamp),
                        "f5_total",
                        "over",
                        str(f5_line),
                        "-110",
                        str(row.scheduled_start),
                    ),
                    (
                        event_id,
                        str(row.game_date),
                        str(row.scheduled_start),
                        str(row.away_team),
                        str(row.home_team),
                        str(row.as_of_timestamp),
                        "f5_total",
                        "under",
                        str(f5_line),
                        "-110",
                        str(row.scheduled_start),
                    ),
                ],
            )
        connection.commit()
    return db_path


def test_stage3_holdout_walk_forward_writes_machine_readable_report(tmp_path: Path) -> None:
    frame = _training_frame()
    training_path = tmp_path / "training.parquet"
    _write_training_parquet(frame, training_path)
    mean_metadata_path = _write_mean_artifact(frame, tmp_path / "mean_head")
    artifact = train_run_distribution_model(
        training_data=training_path,
        output_dir=tmp_path / "dist_models",
        mean_artifact_metadata_path=mean_metadata_path,
        holdout_season=2025,
        xgb_n_jobs=1,
        time_series_splits=3,
        head_param_overrides={
            "max_depth": 2,
            "n_estimators": 12,
            "learning_rate": 0.1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "min_child_weight": 1,
            "reg_lambda": 1.0,
        },
        distribution_report_dir=tmp_path / "distribution_reports",
    )

    report, paths = evaluate_stage3_holdout_walk_forward(
        training_data=training_path,
        model_metadata_path=artifact.metadata_path,
        output_dir=tmp_path / "walk_forward",
        holdout_season=2025,
    )

    assert paths.json_path.exists()
    assert paths.csv_path.exists()
    assert report["lane_key"] == "best_distribution_lane"
    assert report["betting_evidence"]["available"] is False
    assert report["betting_evidence"]["clv_supported"] is False
    assert report["betting_evidence"]["clv_coverage"] == 0.0
    assert report["proxy_market_decision_evidence"]["available"] is False
    assert "predicted_mean_runs" in report["operational_diagnostics"]
    assert report["month_summaries"]


def test_stage3_holdout_walk_forward_reports_old_scraper_total_market_coverage(tmp_path: Path) -> None:
    frame = _training_frame()
    training_path = tmp_path / "training.parquet"
    _write_training_parquet(frame, training_path)
    mean_metadata_path = _write_mean_artifact(frame, tmp_path / "mean_head")
    artifact = train_run_distribution_model(
        training_data=training_path,
        output_dir=tmp_path / "dist_models",
        mean_artifact_metadata_path=mean_metadata_path,
        holdout_season=2025,
        xgb_n_jobs=1,
        time_series_splits=3,
        head_param_overrides={
            "max_depth": 2,
            "n_estimators": 12,
            "learning_rate": 0.1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "min_child_weight": 1,
            "reg_lambda": 1.0,
        },
        distribution_report_dir=tmp_path / "distribution_reports",
    )
    old_scraper_db = _seed_old_scraper_totals_db(frame, tmp_path / "old_scraper.db")

    report, _ = evaluate_stage3_holdout_walk_forward(
        training_data=training_path,
        model_metadata_path=artifact.metadata_path,
        output_dir=tmp_path / "walk_forward",
        holdout_season=2025,
        enable_market_priors=True,
        historical_odds_db_path=old_scraper_db,
        historical_market_book_name="DraftKings",
    )

    assert report["betting_evidence"]["available"] is False
    assert report["betting_evidence"]["clv_supported"] is False
    assert report["betting_evidence"]["clv_basis"] == "opening_vs_closing"
    assert report["betting_evidence"]["historical_market_source"] == "historical_market_archive_old_scraper"
    assert report["betting_evidence"]["full_game_total_coverage"] == 1.0
    assert report["betting_evidence"]["f5_total_coverage"] == 1.0
    assert report["betting_evidence"]["full_game_total_closing_coverage"] == 1.0
    assert report["betting_evidence"]["f5_total_closing_coverage"] == 1.0
    assert report["proxy_market_decision_evidence"]["available"] is True
    assert report["proxy_market_decision_evidence"]["coverage"] == 1.0
    assert report["proxy_market_decision_evidence"]["proxy_only"] is True


def test_mcmc_holdout_walk_forward_reads_existing_predictions_csv(tmp_path: Path) -> None:
    predictions_csv = tmp_path / "predictions.csv"
    pd.DataFrame(
        [
            {
                "game_date": "2025-04-01",
                "actual_away_runs": 3,
                "market_priors_available": 1.0,
                "market_anchor_confidence": 0.9,
                "market_implied_away_runs": 2.5,
                "fallback_applied": False,
                "mean_anchor_applied": True,
                "mean_drift_vs_control": 0.1,
                "post_anchor_implied_mean_runs": 2.8,
                "away_run_pmf_json": json.dumps(
                    [
                        {"runs": 0, "probability": 0.10},
                        {"runs": 1, "probability": 0.20},
                        {"runs": 2, "probability": 0.30},
                        {"runs": 3, "probability": 0.40},
                    ]
                ),
            },
            {
                "game_date": "2025-05-01",
                "actual_away_runs": 1,
                "market_priors_available": 1.0,
                "market_anchor_confidence": 0.8,
                "market_implied_away_runs": 1.8,
                "fallback_applied": False,
                "mean_anchor_applied": True,
                "mean_drift_vs_control": -0.05,
                "post_anchor_implied_mean_runs": 1.6,
                "away_run_pmf_json": json.dumps(
                    [
                        {"runs": 0, "probability": 0.25},
                        {"runs": 1, "probability": 0.35},
                        {"runs": 2, "probability": 0.25},
                        {"runs": 3, "probability": 0.15},
                    ]
                ),
            },
        ]
    ).to_csv(predictions_csv, index=False)

    metadata_path = tmp_path / "mcmc.metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "model_version": "20260328T000000Z_mcmc",
                "output_paths": {"predictions_csv": str(predictions_csv)},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    report, paths = evaluate_mcmc_holdout_walk_forward(
        mcmc_metadata_path=metadata_path,
        output_dir=tmp_path / "walk_forward",
    )

    assert paths.json_path.exists()
    assert paths.csv_path.exists()
    assert report["lane_key"] == "best_mcmc_lane"
    assert len(report["month_summaries"]) == 2
    assert report["proxy_market_decision_evidence"]["available"] is True
    assert report["operational_diagnostics"]["fallback_rate"] == 0.0
    assert report["operational_diagnostics"]["mean_anchor_rate"] == 1.0
