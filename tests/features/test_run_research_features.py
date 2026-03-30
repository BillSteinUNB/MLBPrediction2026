from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from src.db import init_db
from src.model.run_research_features import augment_run_research_features


def _base_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_pk": 1001,
                "game_date": "2025-04-01",
                "scheduled_start": "2025-04-01T23:05:00Z",
                "abs_active": 1.0,
                "abs_walk_rate_delta": 0.04,
                "abs_strikeout_rate_delta": -0.03,
                "home_team": "NYY",
                "away_team": "BOS",
                "home_starter_last_start_pitch_count": 101.0,
                "home_starter_cumulative_pitch_load_5s": 430.0,
                "home_team_bullpen_pitch_count_3d": 36.0,
                "away_lineup_bb_pct_30g": 9.4,
                "away_lineup_k_pct_30g": 20.2,
                "away_lineup_woba_30g": 0.332,
                "away_lineup_woba_delta_7v30g": 0.018,
                "away_lineup_platoon_advantage_pct": 0.64,
                "home_starter_k_pct_30s": 24.8,
                "home_starter_bb_pct_30s": 7.2,
                "home_starter_gb_pct_30s": 38.0,
                "home_starter_csw_pct_30s": 31.2,
                "home_starter_avg_fastball_velocity_30s": 95.2,
                "home_starter_pitch_mix_entropy_30s": 2.55,
                "away_lineup_iso_30g": 0.188,
                "away_lineup_barrel_pct_30g": 9.1,
                "away_lineup_woba_minus_xwoba_30g": -0.012,
                "home_team_oaa_30g": 4.0,
                "home_team_oaa_season": 1.0,
                "home_team_drs_30g": 6.0,
                "home_team_drs_season": 2.0,
                "home_team_defensive_efficiency_30g": 0.721,
                "home_team_defensive_efficiency_season": 0.709,
                "home_team_adjusted_framing_30g": 3.2,
                "home_team_adjusted_framing_60g": 2.4,
                "home_team_adjusted_framing_season": 1.3,
                "home_team_framing_retention_proxy_30g": 0.79,
                "home_team_framing_retention_proxy_season": 0.76,
                "away_lineup_babip_30g": 0.309,
                "away_lineup_xwoba_30g": 0.327,
                "weather_wind_factor": 10.0,
                "weather_air_density_factor": 1.03,
                "weather_composite": 1.05,
                "away_lineup_lhb_pct": 0.44,
                "away_lineup_rhb_pct": 0.44,
                "plate_umpire_sample_size_30g": 26.0,
                "plate_umpire_total_runs_avg_30g": 8.1,
                "plate_umpire_total_runs_avg_90g": 8.7,
                "plate_umpire_f5_total_runs_avg_30g": 3.8,
                "plate_umpire_f5_total_runs_avg_90g": 4.3,
                "plate_umpire_home_win_pct_30g": 0.51,
                "plate_umpire_home_win_pct_90g": 0.53,
                "plate_umpire_abs_active_share_30g": 0.62,
                "plate_umpire_abs_total_runs_avg_30g": 7.7,
                "plate_umpire_abs_f5_total_runs_avg_30g": 3.6,
            }
        ]
    )


def _seed_historical_odds_db(db_path: Path) -> Path:
    with sqlite3.connect(init_db(db_path)) as connection:
        connection.execute(
            """
            INSERT INTO games (
                game_pk, date, home_team, away_team, venue, is_dome, is_abs_active,
                f5_home_score, f5_away_score, final_home_score, final_away_score, status
            )
            VALUES (?, ?, ?, ?, ?, 0, 1, 3, 2, 5, 4, 'final')
            """,
            (1001, "2025-04-01T23:05:00+00:00", "NYY", "BOS", "Yankee Stadium"),
        )
        connection.execute(
            """
            INSERT INTO odds_snapshots (
                game_pk, book_name, market_type, home_odds, away_odds, fetched_at, is_frozen
            )
            VALUES (?, 'hist-book', 'f5_ml', ?, ?, ?, 1)
            """,
            (1001, -130, 120, "2025-04-01T18:00:00+00:00"),
        )
        connection.execute(
            """
            INSERT INTO odds_snapshots (
                game_pk, book_name, market_type, home_odds, away_odds, home_point, away_point, fetched_at, is_frozen
            )
            VALUES (?, 'hist-book', 'f5_rl', ?, ?, ?, ?, ?, 1)
            """,
            (1001, -105, -115, 1.5, -1.5, "2025-04-01T18:00:00+00:00"),
        )
        connection.execute(
            """
            INSERT INTO odds_snapshots (
                game_pk, book_name, market_type, home_odds, away_odds, home_point, away_point, fetched_at, is_frozen
            )
            VALUES (?, 'hist-book', 'full_game_team_total_away', ?, ?, ?, ?, ?, 1)
            """,
            (1001, -110, -110, 4.5, 4.5, "2025-04-01T18:00:00+00:00"),
        )
        connection.commit()
    return db_path


def _seed_old_scraper_historical_odds_db(db_path: Path) -> Path:
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
            INSERT INTO games (event_id, game_date, commence_time_utc, away_team, home_team, game_type)
            VALUES ('evt-1001', '2025-04-01', '2025-04-01T23:05:00Z', 'BOS', 'NYY', 'R');
            """
        )
        connection.executemany(
            """
            INSERT INTO odds (
                event_id, game_date, commence_time, away_team, home_team, fetched_at,
                bookmaker, market_type, side, point, price, commence_time_utc, is_opening
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("evt-1001", "2025-04-01", "2025-04-01T23:05:00Z", "BOS", "NYY", "2025-04-01T16:00:00Z", "DraftKings", "f5_ml", "away", "", "120", "2025-04-01T23:05:00Z", 1),
                ("evt-1001", "2025-04-01", "2025-04-01T23:05:00Z", "BOS", "NYY", "2025-04-01T16:00:00Z", "DraftKings", "f5_ml", "home", "", "-130", "2025-04-01T23:05:00Z", 1),
                ("evt-1001", "2025-04-01", "2025-04-01T23:05:00Z", "BOS", "NYY", "2025-04-01T16:00:00Z", "DraftKings", "f5_rl", "away", "-0.5", "-115", "2025-04-01T23:05:00Z", 1),
                ("evt-1001", "2025-04-01", "2025-04-01T23:05:00Z", "BOS", "NYY", "2025-04-01T16:00:00Z", "DraftKings", "f5_rl", "home", "0.5", "-105", "2025-04-01T23:05:00Z", 1),
                ("evt-1001", "2025-04-01", "2025-04-01T23:05:00Z", "BOS", "NYY", "2025-04-01T16:00:00Z", "DraftKings", "f5_total", "over", "4.5", "-110", "2025-04-01T23:05:00Z", 1),
                ("evt-1001", "2025-04-01", "2025-04-01T23:05:00Z", "BOS", "NYY", "2025-04-01T16:00:00Z", "DraftKings", "f5_total", "under", "4.5", "-110", "2025-04-01T23:05:00Z", 1),
                ("evt-1001", "2025-04-01", "2025-04-01T23:05:00Z", "BOS", "NYY", "2025-04-01T16:00:00Z", "DraftKings", "full_game_ml", "away", "", "135", "2025-04-01T23:05:00Z", 1),
                ("evt-1001", "2025-04-01", "2025-04-01T23:05:00Z", "BOS", "NYY", "2025-04-01T16:00:00Z", "DraftKings", "full_game_ml", "home", "", "-145", "2025-04-01T23:05:00Z", 1),
                ("evt-1001", "2025-04-01", "2025-04-01T23:05:00Z", "BOS", "NYY", "2025-04-01T16:00:00Z", "DraftKings", "full_game_rl", "away", "-1.5", "145", "2025-04-01T23:05:00Z", 1),
                ("evt-1001", "2025-04-01", "2025-04-01T23:05:00Z", "BOS", "NYY", "2025-04-01T16:00:00Z", "DraftKings", "full_game_rl", "home", "1.5", "-175", "2025-04-01T23:05:00Z", 1),
                ("evt-1001", "2025-04-01", "2025-04-01T23:05:00Z", "BOS", "NYY", "2025-04-01T16:00:00Z", "DraftKings", "full_game_total", "over", "8.5", "-110", "2025-04-01T23:05:00Z", 1),
                ("evt-1001", "2025-04-01", "2025-04-01T23:05:00Z", "BOS", "NYY", "2025-04-01T16:00:00Z", "DraftKings", "full_game_total", "under", "8.5", "-110", "2025-04-01T23:05:00Z", 1),
                ("evt-1001", "2025-04-01", "2025-04-01T23:05:00Z", "BOS", "NYY", "2025-04-01T16:00:00Z", "DraftKings", "full_game_team_total_away", "over", "4.5", "-110", "2025-04-01T23:05:00Z", 1),
                ("evt-1001", "2025-04-01", "2025-04-01T23:05:00Z", "BOS", "NYY", "2025-04-01T16:00:00Z", "DraftKings", "full_game_team_total_away", "under", "4.5", "-110", "2025-04-01T23:05:00Z", 1),
            ],
        )
        connection.commit()
    return db_path


def _seed_old_scraper_backfill_opening_odds_db(db_path: Path) -> Path:
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
            INSERT INTO games (event_id, game_date, commence_time_utc, away_team, home_team, game_type)
            VALUES ('evt-1001', '2025-04-01', '2025-04-01T23:05:00Z', 'BOS', 'NYY', 'R');
            """
        )
        connection.executemany(
            """
            INSERT INTO odds (
                event_id, game_date, commence_time, away_team, home_team, fetched_at,
                bookmaker, market_type, side, point, price, commence_time_utc, is_opening
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("evt-1001", "2025-04-01", "2025-04-01T23:05:00Z", "BOS", "NYY", "2026-03-28T21:22:02.208406Z", "Opener", "f5_ml", "away", "", "104", "2025-04-01T23:05:00Z", 1),
                ("evt-1001", "2025-04-01", "2025-04-01T23:05:00Z", "BOS", "NYY", "2026-03-28T21:22:02.208406Z", "Opener", "f5_ml", "home", "", "-125", "2025-04-01T23:05:00Z", 1),
                ("evt-1001", "2025-04-01", "2025-04-01T23:05:00Z", "BOS", "NYY", "2026-03-28T21:22:02.208406Z", "Opener", "f5_rl", "away", "-0.5", "-112", "2025-04-01T23:05:00Z", 1),
                ("evt-1001", "2025-04-01", "2025-04-01T23:05:00Z", "BOS", "NYY", "2026-03-28T21:22:02.208406Z", "Opener", "f5_rl", "home", "0.5", "-108", "2025-04-01T23:05:00Z", 1),
                ("evt-1001", "2025-04-01", "2025-04-01T23:05:00Z", "BOS", "NYY", "2026-03-28T21:22:02.208406Z", "Opener", "f5_total", "over", "4.5", "-110", "2025-04-01T23:05:00Z", 1),
                ("evt-1001", "2025-04-01", "2025-04-01T23:05:00Z", "BOS", "NYY", "2026-03-28T21:22:02.208406Z", "Opener", "f5_total", "under", "4.5", "-110", "2025-04-01T23:05:00Z", 1),
                ("evt-1001", "2025-04-01", "2025-04-01T23:05:00Z", "BOS", "NYY", "2026-03-28T21:22:02.208406Z", "Opener", "full_game_ml", "away", "", "135", "2025-04-01T23:05:00Z", 1),
                ("evt-1001", "2025-04-01", "2025-04-01T23:05:00Z", "BOS", "NYY", "2026-03-28T21:22:02.208406Z", "Opener", "full_game_ml", "home", "", "-145", "2025-04-01T23:05:00Z", 1),
                ("evt-1001", "2025-04-01", "2025-04-01T23:05:00Z", "BOS", "NYY", "2026-03-28T21:22:02.208406Z", "Opener", "full_game_rl", "away", "-1.5", "145", "2025-04-01T23:05:00Z", 1),
                ("evt-1001", "2025-04-01", "2025-04-01T23:05:00Z", "BOS", "NYY", "2026-03-28T21:22:02.208406Z", "Opener", "full_game_rl", "home", "1.5", "-175", "2025-04-01T23:05:00Z", 1),
                ("evt-1001", "2025-04-01", "2025-04-01T23:05:00Z", "BOS", "NYY", "2026-03-28T21:22:02.208406Z", "Opener", "full_game_total", "over", "8.5", "-110", "2025-04-01T23:05:00Z", 1),
                ("evt-1001", "2025-04-01", "2025-04-01T23:05:00Z", "BOS", "NYY", "2026-03-28T21:22:02.208406Z", "Opener", "full_game_total", "under", "8.5", "-110", "2025-04-01T23:05:00Z", 1),
                ("evt-1001", "2025-04-01", "2025-04-01T23:05:00Z", "BOS", "NYY", "2026-03-28T21:22:02.208406Z", "Opener", "full_game_team_total_away", "over", "4.5", "-110", "2025-04-01T23:05:00Z", 1),
                ("evt-1001", "2025-04-01", "2025-04-01T23:05:00Z", "BOS", "NYY", "2026-03-28T21:22:02.208406Z", "Opener", "full_game_team_total_away", "under", "4.5", "-110", "2025-04-01T23:05:00Z", 1),
                ("evt-1001", "2025-04-01", "2025-04-01T23:05:00Z", "BOS", "NYY", "2026-03-28T21:22:02.208406Z", "DraftKings", "f5_ml", "away", "", "120", "2025-04-01T23:05:00Z", 0),
                ("evt-1001", "2025-04-01", "2025-04-01T23:05:00Z", "BOS", "NYY", "2026-03-28T21:22:02.208406Z", "DraftKings", "f5_ml", "home", "", "-130", "2025-04-01T23:05:00Z", 0),
            ],
        )
        connection.commit()
    return db_path


def test_market_priors_attach_when_historical_f5_rows_exist(tmp_path: Path) -> None:
    db_path = _seed_historical_odds_db(tmp_path / "historical_odds.db")

    result = augment_run_research_features(
        _base_frame(),
        enable_market_priors=True,
        historical_odds_db_path=db_path,
        historical_market_book_name="hist-book",
    )

    row = result.dataframe.iloc[0]
    assert result.metadata.market_priors.coverage_pct == 1.0
    assert row["market_priors_available"] == 1.0
    assert row["market_f5_home_fair_prob"] > row["market_f5_away_fair_prob"]
    assert row["market_full_game_away_team_total_line"] == 4.5
    assert row["market_full_game_away_team_total_available"] == 1.0
    assert row["market_implied_full_game_away_runs"] == 4.5
    assert row["market_anchor_confidence"] > 0.0


def test_market_priors_attach_from_old_scraper_market_archive(tmp_path: Path) -> None:
    db_path = _seed_old_scraper_historical_odds_db(tmp_path / "old_scraper_odds.db")

    result = augment_run_research_features(
        _base_frame(),
        enable_market_priors=True,
        historical_odds_db_path=db_path,
        historical_market_book_name="DraftKings",
    )

    row = result.dataframe.iloc[0]
    assert result.metadata.market_priors.source_name == "historical_market_archive_old_scraper"
    assert result.metadata.market_priors.coverage_pct == 1.0
    assert row["market_priors_available"] == 1.0
    assert row["market_f5_total_line"] == 4.5
    assert row["market_full_game_total_line"] == 8.5
    assert row["market_full_game_away_team_total_line"] == 4.5
    assert row["market_full_game_away_team_total_available"] == 1.0
    assert row["market_full_game_away_fair_prob"] > 0.0
    assert row["market_implied_full_game_away_runs"] == 4.5


def test_market_priors_attach_from_old_scraper_backfill_opener_rows(tmp_path: Path) -> None:
    db_path = _seed_old_scraper_backfill_opening_odds_db(tmp_path / "old_scraper_backfill_odds.db")

    result = augment_run_research_features(
        _base_frame(),
        enable_market_priors=True,
        historical_odds_db_path=db_path,
    )

    row = result.dataframe.iloc[0]
    assert result.metadata.market_priors.source_name == "historical_market_archive_old_scraper"
    assert result.metadata.market_priors.coverage_pct == 1.0
    assert row["market_priors_available"] == 1.0
    assert row["market_f5_home_fair_prob"] > row["market_f5_away_fair_prob"]
    assert row["market_full_game_total_line"] == 8.5
    assert row["market_full_game_away_team_total_line"] == 4.5
    assert row["market_implied_full_game_away_runs"] == 4.5


def test_market_priors_fall_back_cleanly_when_local_historical_data_is_missing() -> None:
    result = augment_run_research_features(
        _base_frame(),
        enable_market_priors=True,
        historical_odds_db_path=None,
    )

    row = result.dataframe.iloc[0]
    assert result.metadata.market_priors.coverage_pct == 0.0
    assert result.metadata.market_priors.fallback_reason is not None
    assert row["market_priors_available"] == 0.0
    assert row["market_f5_away_fair_prob"] == 0.5


def test_ttop_and_archetype_features_respond_to_matchup_shape() -> None:
    aggressive = _base_frame()
    passive = _base_frame()
    passive.loc[0, "home_starter_last_start_pitch_count"] = 82.0
    passive.loc[0, "away_lineup_bb_pct_30g"] = 7.4
    passive.loc[0, "away_lineup_iso_30g"] = 0.145
    passive.loc[0, "away_lineup_barrel_pct_30g"] = 6.2

    aggressive_result = augment_run_research_features(aggressive).dataframe.iloc[0]
    passive_result = augment_run_research_features(passive).dataframe.iloc[0]

    assert (
        aggressive_result["away_matchup_ttop_exposure_index"]
        > passive_result["away_matchup_ttop_exposure_index"]
    )
    assert (
        aggressive_result["away_matchup_power_archetype_mismatch"]
        > passive_result["away_matchup_power_archetype_mismatch"]
    )


def test_handedness_weather_and_umpire_framing_micro_features_are_emitted() -> None:
    result = augment_run_research_features(_base_frame())
    row = result.dataframe.iloc[0]

    assert row["away_weather_wind_lhb_interaction"] == 4.4
    assert row["away_weather_wind_rhb_interaction"] == 4.4
    assert row["plate_umpire_zone_suppression_index"] > 0.0
    assert row["home_team_framing_zone_support_index"] > 0.0
    assert 0.0 <= row["home_team_framing_stability_index"] <= 1.0


def test_abs_regime_features_are_emitted_as_proxy_features_and_follow_context() -> None:
    aggressive = _base_frame()
    conservative = _base_frame()
    conservative.loc[0, "away_lineup_bb_pct_30g"] = 7.4
    conservative.loc[0, "away_lineup_k_pct_30g"] = 24.9
    conservative.loc[0, "away_lineup_woba_30g"] = 0.302
    conservative.loc[0, "home_team_framing_retention_proxy_30g"] = 0.68
    conservative.loc[0, "plate_umpire_abs_active_share_30g"] = 0.24
    conservative.loc[0, "plate_umpire_abs_total_runs_avg_30g"] = 8.7
    conservative.loc[0, "plate_umpire_abs_f5_total_runs_avg_30g"] = 4.5

    aggressive_row = augment_run_research_features(aggressive).dataframe.iloc[0]
    conservative_row = augment_run_research_features(conservative).dataframe.iloc[0]

    assert aggressive_row["abs_challenge_opportunity_proxy"] > conservative_row["abs_challenge_opportunity_proxy"]
    assert (
        aggressive_row["abs_expected_challenge_pressure_proxy"]
        > conservative_row["abs_expected_challenge_pressure_proxy"]
    )
    assert (
        aggressive_row["abs_leverage_framing_retention_proxy"]
        > conservative_row["abs_leverage_framing_retention_proxy"]
    )
    assert (
        aggressive_row["abs_umpire_zone_suppression_proxy"]
        > conservative_row["abs_umpire_zone_suppression_proxy"]
    )
