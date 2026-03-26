from __future__ import annotations

import sqlite3
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from src.db import init_db


PYTHAGOREAN_EXPONENT = 1.83


def _seed_scheduled_game(
    db_path: Path,
    *,
    game_pk: int,
    game_date: str,
    home_team: str,
    away_team: str,
) -> None:
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO games (game_pk, date, home_team, away_team, venue, status)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (game_pk, game_date, home_team, away_team, "Test Park", "scheduled"),
        )
        connection.commit()


def _seed_result(
    db_path: Path,
    *,
    game_pk: int,
    game_date: str,
    team: str,
    opponent: str,
    team_runs: int,
    opponent_runs: int,
    team_f5_runs: int,
    opponent_f5_runs: int,
    team_is_home: bool,
) -> None:
    if team_is_home:
        home_team = team
        away_team = opponent
        final_home_score = team_runs
        final_away_score = opponent_runs
        f5_home_score = team_f5_runs
        f5_away_score = opponent_f5_runs
    else:
        home_team = opponent
        away_team = team
        final_home_score = opponent_runs
        final_away_score = team_runs
        f5_home_score = opponent_f5_runs
        f5_away_score = team_f5_runs

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO games (
                game_pk,
                date,
                home_team,
                away_team,
                venue,
                status,
                f5_home_score,
                f5_away_score,
                final_home_score,
                final_away_score
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                game_pk,
                game_date,
                home_team,
                away_team,
                "Test Park",
                "final",
                f5_home_score,
                f5_away_score,
                final_home_score,
                final_away_score,
            ),
        )
        connection.commit()


def _seed_team_history(
    db_path: Path,
    *,
    start_game_pk: int,
    team: str,
    opponent_prefix: str,
    older_full: tuple[int, int],
    recent_full: tuple[int, int],
    older_f5: tuple[int, int],
    recent_f5: tuple[int, int],
) -> None:
    for index in range(60):
        is_recent = index >= 30
        full_runs = recent_full if is_recent else older_full
        f5_runs = recent_f5 if is_recent else older_f5
        game_day = date(2025, 1, 1) + timedelta(days=index)
        _seed_result(
            db_path,
            game_pk=start_game_pk + index,
            game_date=f"{game_day.isoformat()}T19:05:00+00:00",
            team=team,
            opponent=f"{opponent_prefix}{index:02d}".upper()[-3:],
            team_runs=full_runs[0],
            opponent_runs=full_runs[1],
            team_f5_runs=f5_runs[0],
            opponent_f5_runs=f5_runs[1],
            team_is_home=index % 2 == 0,
        )


def _pythagorean_win_pct(runs_scored: int, runs_allowed: int) -> float:
    numerator = runs_scored**PYTHAGOREAN_EXPONENT
    denominator = numerator + runs_allowed**PYTHAGOREAN_EXPONENT
    return numerator / denominator


def _seed_round_robin_results(
    db_path: Path,
    *,
    start_day: date,
    total_days: int,
    start_game_pk: int = 10_000,
) -> list[dict[str, object]]:
    matchup_cycles = (
        (("NYY", "BOS"), ("SEA", "TB")),
        (("NYY", "SEA"), ("BOS", "TB")),
        (("NYY", "TB"), ("BOS", "SEA")),
    )
    seeded_games: list[dict[str, object]] = []
    game_pk = start_game_pk
    for day_index in range(total_days):
        game_day = start_day + timedelta(days=day_index)
        pairings = matchup_cycles[day_index % len(matchup_cycles)]
        for pairing_index, (home_team, away_team) in enumerate(pairings):
            home_runs = 3 + ((day_index + pairing_index) % 6)
            away_runs = 2 + ((day_index + (pairing_index * 2)) % 5)
            home_f5_runs = min(home_runs, 1 + ((day_index + pairing_index) % 4))
            away_f5_runs = min(away_runs, (day_index + (pairing_index * 3)) % 4)
            _seed_result(
                db_path,
                game_pk=game_pk,
                game_date=f"{game_day.isoformat()}T19:05:00+00:00",
                team=home_team,
                opponent=away_team,
                team_runs=home_runs,
                opponent_runs=away_runs,
                team_f5_runs=home_f5_runs,
                opponent_f5_runs=away_f5_runs,
                team_is_home=True,
            )
            seeded_games.append(
                {
                    "game_pk": game_pk,
                    "game_date": game_day.isoformat(),
                    "home_team": home_team,
                    "away_team": away_team,
                }
            )
            game_pk += 1
    return seeded_games


def _baseline_payload(rows: list[object]) -> dict[tuple[int, str, int | None, str], float]:
    return {
        (
            int(row.game_pk),
            str(row.feature_name),
            row.window_size,
            row.as_of_timestamp.isoformat(),
        ): float(row.feature_value)
        for row in rows
    }


def test_calculate_pythagorean_win_percentage_matches_contract_example() -> None:
    from src.features.baselines import calculate_pythagorean_win_percentage

    win_pct = calculate_pythagorean_win_percentage(500, 450)

    assert win_pct == pytest.approx(_pythagorean_win_pct(500, 450))
    assert win_pct == pytest.approx(0.549, abs=0.002)


def test_calculate_log5_probability_is_team_only_and_sums_to_one() -> None:
    from src.features.baselines import calculate_log5_probability

    team_a = calculate_log5_probability(0.62, 0.54)
    team_b = calculate_log5_probability(0.54, 0.62)

    assert 0.0 <= team_a <= 1.0
    assert 0.0 <= team_b <= 1.0
    assert team_a + team_b == pytest.approx(1.0)

    with pytest.raises(ValueError, match="team-only"):
        calculate_log5_probability(0.62, 0.54, entity_level="batter-pitcher")


def test_compute_baseline_features_persists_full_and_f5_pythagorean_with_log5(
    tmp_path: Path,
) -> None:
    from src.features.baselines import compute_baseline_features

    db_path = tmp_path / "baselines.db"
    init_db(db_path)
    _seed_scheduled_game(
        db_path,
        game_pk=9999,
        game_date="2025-05-01T19:05:00+00:00",
        home_team="NYY",
        away_team="BOS",
    )
    _seed_team_history(
        db_path,
        start_game_pk=1000,
        team="NYY",
        opponent_prefix="A",
        older_full=(3, 2),
        recent_full=(5, 4),
        older_f5=(1, 2),
        recent_f5=(3, 1),
    )
    _seed_team_history(
        db_path,
        start_game_pk=2000,
        team="BOS",
        opponent_prefix="B",
        older_full=(2, 3),
        recent_full=(4, 5),
        older_f5=(2, 1),
        recent_f5=(1, 3),
    )

    rows = compute_baseline_features("2025-05-01", db_path=db_path)

    by_name = {row.feature_name: row.feature_value for row in rows}
    expected_home_30 = _pythagorean_win_pct(150, 120)
    expected_home_60 = _pythagorean_win_pct(240, 180)
    expected_home_f5_30 = _pythagorean_win_pct(90, 30)
    expected_home_f5_60 = _pythagorean_win_pct(120, 90)
    expected_away_30 = _pythagorean_win_pct(120, 150)
    expected_home_runs_scored_7 = 5.0
    expected_home_runs_allowed_7 = 4.0
    expected_away_runs_scored_7 = 4.0
    expected_away_runs_allowed_7 = 5.0
    expected_home_log5_30 = (
        expected_home_30 * (1 - expected_away_30)
    ) / (
        expected_home_30 * (1 - expected_away_30)
        + expected_away_30 * (1 - expected_home_30)
    )

    assert by_name["home_team_pythagorean_wp_30g"] == pytest.approx(expected_home_30)
    assert by_name["home_team_pythagorean_wp_60g"] == pytest.approx(expected_home_60)
    assert by_name["home_team_f5_pythagorean_wp_30g"] == pytest.approx(expected_home_f5_30)
    assert by_name["home_team_f5_pythagorean_wp_60g"] == pytest.approx(expected_home_f5_60)
    assert by_name["home_team_f5_pythagorean_wp_30g"] != pytest.approx(
        by_name["home_team_pythagorean_wp_30g"]
    )
    assert by_name["home_team_log5_30g"] == pytest.approx(expected_home_log5_30)
    assert by_name["home_team_log5_30g"] + by_name["away_team_log5_30g"] == pytest.approx(1.0)
    assert by_name["home_team_runs_scored_7g"] == pytest.approx(expected_home_runs_scored_7)
    assert by_name["home_team_runs_allowed_7g"] == pytest.approx(expected_home_runs_allowed_7)
    assert by_name["away_team_runs_scored_7g"] == pytest.approx(expected_away_runs_scored_7)
    assert by_name["away_team_runs_allowed_7g"] == pytest.approx(expected_away_runs_allowed_7)
    assert by_name["home_team_runs_scored_14g"] == pytest.approx(expected_home_runs_scored_7)
    assert by_name["home_team_runs_allowed_14g"] == pytest.approx(expected_home_runs_allowed_7)

    with sqlite3.connect(db_path) as connection:
        stored_rows = connection.execute(
            "SELECT COUNT(*) FROM features WHERE game_pk = ?",
            (9999,),
        ).fetchone()[0]
        as_of_timestamp = connection.execute(
            "SELECT as_of_timestamp FROM features WHERE game_pk = ? AND feature_name = ?",
            (9999, "home_team_log5_30g"),
        ).fetchone()[0]

    assert stored_rows == 20
    assert as_of_timestamp == "2025-04-30T00:00:00+00:00"


def test_bulk_baseline_features_match_per_day(tmp_path: Path) -> None:
    from src.features.baselines import (
        compute_baseline_features,
        compute_baseline_features_for_schedule,
    )

    day_db_path = tmp_path / "baseline_day.db"
    bulk_db_path = tmp_path / "baseline_bulk.db"
    init_db(day_db_path)
    init_db(bulk_db_path)
    seeded_games = _seed_round_robin_results(day_db_path, start_day=date(2025, 1, 1), total_days=120)
    _seed_round_robin_results(bulk_db_path, start_day=date(2025, 1, 1), total_days=120)

    schedule = pd.DataFrame(seeded_games[-60:])
    target_dates = sorted(schedule["game_date"].unique().tolist())

    expected_rows = []
    for target_day in target_dates:
        expected_rows.extend(compute_baseline_features(target_day, db_path=day_db_path))

    actual_rows = compute_baseline_features_for_schedule(schedule, db_path=bulk_db_path)

    expected_payload = _baseline_payload(expected_rows)
    actual_payload = _baseline_payload(actual_rows)

    assert actual_payload.keys() == expected_payload.keys()
    for key, expected_value in expected_payload.items():
        assert actual_payload[key] == pytest.approx(expected_value, abs=1e-9)


def test_bulk_baselines_no_leakage(tmp_path: Path) -> None:
    from src.features.baselines import compute_baseline_features_for_schedule

    db_path = tmp_path / "baseline_leakage.db"
    init_db(db_path)
    _seed_result(
        db_path,
        game_pk=4101,
        game_date="2024-04-14T19:05:00+00:00",
        team="NYY",
        opponent="SEA",
        team_runs=4,
        opponent_runs=2,
        team_f5_runs=2,
        opponent_f5_runs=1,
        team_is_home=True,
    )
    _seed_result(
        db_path,
        game_pk=4102,
        game_date="2024-04-15T19:05:00+00:00",
        team="NYY",
        opponent="BOS",
        team_runs=10,
        opponent_runs=0,
        team_f5_runs=5,
        opponent_f5_runs=0,
        team_is_home=True,
    )
    _seed_scheduled_game(
        db_path,
        game_pk=5101,
        game_date="2024-04-15T23:05:00+00:00",
        home_team="NYY",
        away_team="TB",
    )
    _seed_scheduled_game(
        db_path,
        game_pk=5102,
        game_date="2024-04-16T23:05:00+00:00",
        home_team="NYY",
        away_team="TB",
    )

    schedule = pd.DataFrame(
        [
            {"game_pk": 5101, "game_date": "2024-04-15", "home_team": "NYY", "away_team": "TB"},
            {"game_pk": 5102, "game_date": "2024-04-16", "home_team": "NYY", "away_team": "TB"},
        ]
    )

    rows = compute_baseline_features_for_schedule(schedule, db_path=db_path)
    payload = _baseline_payload(rows)

    assert payload[(5101, "home_team_runs_scored_7g", 7, "2024-04-14T00:00:00+00:00")] == pytest.approx(4.0)
    assert payload[(5102, "home_team_runs_scored_7g", 7, "2024-04-15T00:00:00+00:00")] == pytest.approx(7.0)


def test_bulk_baselines_respects_season_boundary(tmp_path: Path) -> None:
    from src.features.baselines import compute_baseline_features_for_schedule

    db_path = tmp_path / "baseline_season_boundary.db"
    init_db(db_path)
    _seed_result(
        db_path,
        game_pk=6101,
        game_date="2024-09-30T19:05:00+00:00",
        team="NYY",
        opponent="BOS",
        team_runs=12,
        opponent_runs=1,
        team_f5_runs=6,
        opponent_f5_runs=0,
        team_is_home=True,
    )
    _seed_scheduled_game(
        db_path,
        game_pk=7101,
        game_date="2025-04-01T19:05:00+00:00",
        home_team="NYY",
        away_team="TB",
    )

    schedule = pd.DataFrame(
        [{"game_pk": 7101, "game_date": "2025-04-01", "home_team": "NYY", "away_team": "TB"}]
    )

    rows = compute_baseline_features_for_schedule(schedule, db_path=db_path)
    payload = _baseline_payload(rows)

    assert payload[(7101, "home_team_pythagorean_wp_30g", 30, "2025-03-31T00:00:00+00:00")] == pytest.approx(0.5)
    assert payload[(7101, "home_team_runs_scored_7g", 7, "2025-03-31T00:00:00+00:00")] == pytest.approx(4.5)
