from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.backtest.walk_forward import create_walk_forward_windows, run_walk_forward_backtest


def _synthetic_training_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    start = datetime(2021, 7, 1, 19, 5, tzinfo=UTC)

    for index in range(180):
        scheduled_start = start + timedelta(days=index * 2)
        season = scheduled_start.year
        base_market = 0.45 + (0.10 * ((index % 12) / 11))
        hidden_signal = 0.18 if index % 6 in {0, 1, 2} else -0.18
        model_signal = max(0.05, min(0.95, base_market + hidden_signal))
        actual_result = int(model_signal >= 0.5)
        tied_after_5 = 1 if index % 23 == 0 else 0

        rows.append(
            {
                "game_pk": 40_000 + index,
                "season": season,
                "game_date": scheduled_start.date().isoformat(),
                "scheduled_start": scheduled_start.isoformat(),
                "as_of_timestamp": (scheduled_start - timedelta(days=1)).isoformat(),
                "home_team": "NYY" if index % 2 == 0 else "BOS",
                "away_team": "BOS" if index % 2 == 0 else "NYY",
                "venue": "Yankee Stadium" if index % 2 == 0 else "Fenway Park",
                "game_type": "R",
                "build_timestamp": "2026-03-19T00:00:00+00:00",
                "data_version_hash": "synthetic-walk-forward-hash",
                "home_team_log5_30g": base_market,
                "away_team_log5_30g": 1.0 - base_market,
                "home_team_f5_pythagorean_wp_30g": model_signal,
                "away_team_f5_pythagorean_wp_30g": 1.0 - model_signal,
                "park_runs_factor": 0.96 + (0.04 * (index % 5) / 4),
                "park_hr_factor": 0.94 + (0.06 * ((index + 2) % 5) / 4),
                "weather_composite": 0.99 + (0.03 * ((index * 3) % 5) / 4),
                "bullpen_pitch_count_3d": 10.0 + (index % 9),
                "model_signal_feature": model_signal,
                "market_gap_feature": model_signal - base_market,
                "feature_noise_a": float((index * 13) % 7),
                "feature_noise_b": float((index * 17) % 11),
                "f5_tied_after_5": tied_after_5,
                "f5_ml_result": 0 if tied_after_5 else actual_result,
                "f5_rl_result": 1 if (not tied_after_5 and model_signal >= 0.57) else 0,
            }
        )

    return pd.DataFrame(rows)


def test_create_walk_forward_windows_uses_six_month_train_and_one_month_test_stride() -> None:
    frame = _synthetic_training_frame()

    windows = create_walk_forward_windows(
        frame,
        start_date="2022-01-01",
        end_date="2022-03-31",
    )

    assert [window.test_start.strftime("%Y-%m-%d") for window in windows] == [
        "2022-01-01",
        "2022-02-01",
        "2022-03-01",
    ]
    assert [window.train_start.strftime("%Y-%m-%d") for window in windows] == [
        "2021-07-01",
        "2021-08-01",
        "2021-09-01",
    ]
    assert all(window.train_end == window.test_start for window in windows)
    assert all(window.test_end > window.test_start for window in windows)


def test_run_walk_forward_backtest_writes_metrics_and_is_byte_reproducible(tmp_path) -> None:
    frame = _synthetic_training_frame()

    first_result = run_walk_forward_backtest(
        training_data=frame,
        start_date="2022-01-01",
        end_date="2022-03-31",
        output_dir=tmp_path / "run_a",
        calibration_fraction=0.15,
        estimator_kwargs={"max_depth": 1, "n_estimators": 8, "learning_rate": 0.2},
    )
    second_result = run_walk_forward_backtest(
        training_data=frame,
        start_date="2022-01-01",
        end_date="2022-03-31",
        output_dir=tmp_path / "run_b",
        calibration_fraction=0.15,
        estimator_kwargs={"max_depth": 1, "n_estimators": 8, "learning_rate": 0.2},
    )

    assert first_result.window_count == 3
    assert first_result.aggregate_brier_score < 0.25
    assert first_result.aggregate_roi > 0.0
    assert first_result.output_fingerprint == second_result.output_fingerprint
    assert first_result.predictions_path.read_bytes() == second_result.predictions_path.read_bytes()
    assert first_result.window_metrics_path.read_bytes() == second_result.window_metrics_path.read_bytes()

    predictions = first_result.predictions
    window_metrics = first_result.window_metrics

    assert {"window_index", "brier_score", "roi", "bet_count"}.issubset(window_metrics.columns)
    assert (pd.to_datetime(predictions["as_of_timestamp"], utc=True) < pd.to_datetime(predictions["scheduled_start"], utc=True)).all()
    assert predictions["bet_side"].isin(["home", "away", "none"]).all()
    assert predictions["market_home_fair_prob"].between(0.0, 1.0).all()
    assert predictions["model_home_prob"].between(0.0, 1.0).all()


def test_run_walk_forward_backtest_rebuilds_window_data_and_records_build_log(
    tmp_path,
    monkeypatch,
) -> None:
    frame = _synthetic_training_frame()
    build_calls: list[dict[str, str]] = []

    def _fake_build_training_dataset(**kwargs):
        cutoff = pd.Timestamp(kwargs["scheduled_start_before"])
        output_path = Path(kwargs["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)

        filtered = frame.loc[
            pd.to_datetime(frame["scheduled_start"], utc=True) < cutoff
        ].copy()
        filtered.to_parquet(output_path, index=False)

        metadata_path = output_path.with_suffix(".metadata.json")
        metadata_path.write_text(
            json.dumps({"scheduled_start_before": cutoff.isoformat()}),
            encoding="utf-8",
        )
        build_calls.append(
            {
                "cutoff": cutoff.isoformat(),
                "output_path": str(output_path),
            }
        )
        return SimpleNamespace(
            dataframe=filtered,
            output_path=output_path,
            metadata_path=metadata_path,
            data_version_hash=f"window-build-{len(build_calls)}",
            build_timestamp=datetime(2026, 3, 19, tzinfo=UTC),
            requested_years=(2021, 2022),
            effective_years=(2021, 2022),
        )

    monkeypatch.setattr(
        "src.backtest.walk_forward.build_training_dataset",
        _fake_build_training_dataset,
    )

    result = run_walk_forward_backtest(
        start_date="2022-01-01",
        end_date="2022-03-31",
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "output",
        refresh_data=True,
        calibration_fraction=0.15,
        estimator_kwargs={"max_depth": 1, "n_estimators": 8, "learning_rate": 0.2},
    )

    assert [call["cutoff"] for call in build_calls] == [
        "2022-02-01T00:00:00+00:00",
        "2022-03-01T00:00:00+00:00",
        "2022-04-01T00:00:00+00:00",
    ]
    assert result.window_metrics["feature_data_cutoff"].tolist() == [
        "2022-02-01T00:00:00+00:00",
        "2022-03-01T00:00:00+00:00",
        "2022-04-01T00:00:00+00:00",
    ]
    assert result.window_metrics["feature_data_action"].tolist() == ["rebuilt", "rebuilt", "rebuilt"]

    summary_payload = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert [entry["scheduled_start_before"] for entry in summary_payload["window_builds"]] == [
        "2022-02-01T00:00:00+00:00",
        "2022-03-01T00:00:00+00:00",
        "2022-04-01T00:00:00+00:00",
    ]
