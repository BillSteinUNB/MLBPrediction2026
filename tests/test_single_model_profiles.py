from __future__ import annotations

import importlib.util
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.model.data_builder import (
    RUN_COUNT_REQUIRED_TEMPORAL_DELTA_COLUMNS,
    _json_bytes,
    _run_count_training_schema_metadata,
    _write_parquet_with_metadata,
)
from src.model.single_model_profiles import resolve_single_model_experiment_profile


def test_resolve_single_model_experiment_profile_returns_expected_fast_defaults() -> None:
    profile = resolve_single_model_experiment_profile("fast")

    assert profile.profile_name == "fast"
    assert profile.experiment_name == "2026-away-fast-deltas-poisson-parallel-120x3"
    assert profile.search_iterations == 120
    assert profile.time_series_splits == 3
    assert profile.early_stopping_rounds == 30
    assert profile.search_space["n_estimators"][0] == 500


def test_resolve_single_model_experiment_profile_supports_experiment_override() -> None:
    profile = resolve_single_model_experiment_profile(
        "smoke",
        experiment_name_override="custom-smoke-run",
    )

    assert profile.profile_name == "smoke"
    assert profile.experiment_name == "custom-smoke-run"
    assert profile.search_iterations == 12
    assert profile.time_series_splits == 2
    assert profile.search_space["n_estimators"] == [200, 300, 400]


def test_resolve_single_model_experiment_profile_rejects_unknown_profile() -> None:
    with pytest.raises(ValueError, match="Unknown single-model experiment profile"):
        resolve_single_model_experiment_profile("turbo")


def _load_script_module(script_name: str):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(f"test_{script_name.replace('.', '_')}", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _base_training_row() -> dict[str, object]:
    return {
        "game_pk": 1001,
        "season": 2025,
        "scheduled_start": "2025-04-10T23:05:00+00:00",
        "game_date": "2025-04-10",
        "as_of_timestamp": "2025-04-09T00:00:00+00:00",
        "home_team": "NYY",
        "away_team": "BOS",
        "venue": "Yankee Stadium",
        "game_type": "R",
        "status": "final",
        "f5_home_score": 3,
        "f5_away_score": 2,
        "final_home_score": 5,
        "final_away_score": 4,
        "home_team_log5": 0.55,
        "away_team_log5": 0.45,
        "data_version_hash": "synthetic-data-hash",
    }


def _write_training_parquet(
    path: Path,
    *,
    include_delta_features: bool,
) -> None:
    row = _base_training_row()
    if include_delta_features:
        for index, column in enumerate(RUN_COUNT_REQUIRED_TEMPORAL_DELTA_COLUMNS):
            row[column] = float(index) / 100.0
    dataframe = pd.DataFrame([row])
    schema_metadata = _run_count_training_schema_metadata()
    dataframe.attrs.update(schema_metadata)
    dataframe.attrs["run_count_training_schema"] = schema_metadata
    _write_parquet_with_metadata(
        dataframe,
        path,
        parquet_metadata={
            b"mlbprediction2026.run_count_training_schema": _json_bytes(schema_metadata)
        },
    )


def test_train_single_model_refuses_delta_experiment_without_delta_features(tmp_path: Path) -> None:
    module = _load_script_module("train_run_count.py")
    training_path = tmp_path / "stale_training_data.parquet"
    _write_training_parquet(training_path, include_delta_features=False)
    module.rct.train_run_count_models = lambda **_kwargs: pytest.fail("trainer should not run")

    with pytest.raises(ValueError, match="missing required columns|schema mismatch|stale"):
        module.main(
            [
                "--profile",
                "fast",
                "--training-data",
                str(training_path),
            ]
        )


def test_train_single_model_force_rebuild_reports_rebuilt_data(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script_module("build_parquet.py")
    training_path = tmp_path / "rebuilt_training_data.parquet"
    captured: dict[str, object] = {}

    def _fake_build_training_dataset(**kwargs):
        captured["build_kwargs"] = kwargs
        output_path = Path(kwargs["output_path"])
        _write_training_parquet(output_path, include_delta_features=True)
        return SimpleNamespace(
            dataframe=pd.DataFrame([_base_training_row()]),
            output_path=output_path,
            metadata_path=output_path.with_suffix(".metadata.json"),
            data_version_hash="synthetic-data-hash",
        )

    module.build_training_dataset = _fake_build_training_dataset

    exit_code = module.main(
        [
            "--refresh-data",
            "--FeatureWorker",
            "15",
            "--training-data",
            str(training_path),
        ]
    )

    assert exit_code == 0
    assert captured["build_kwargs"]["output_path"] == training_path
    assert captured["build_kwargs"]["refresh"] is True
    assert captured["build_kwargs"]["refresh_raw_data"] is True
    assert Path(captured["build_kwargs"]["output_path"]) == training_path

    stdout = capsys.readouterr().out
    assert "Building parquet" in stdout
    assert training_path.name in stdout
    assert "Done" in stdout


def test_build_parquet_supports_start_end_aliases(tmp_path: Path) -> None:
    module = _load_script_module("build_parquet.py")
    training_path = tmp_path / "ParquetDefault.parquet"
    captured: dict[str, object] = {}

    def _fake_build_training_dataset(**kwargs):
        captured["build_kwargs"] = kwargs
        return SimpleNamespace(
            dataframe=pd.DataFrame([_base_training_row()]),
            output_path=Path(kwargs["output_path"]),
            metadata_path=Path(kwargs["output_path"]).with_suffix(".metadata.json"),
            data_version_hash="synthetic-data-hash",
        )

    module.build_training_dataset = _fake_build_training_dataset

    exit_code = module.main(
        [
            "--training-data",
            str(training_path),
            "--start",
            "2018",
            "--end",
            "2025",
        ]
    )

    assert exit_code == 0
    assert captured["build_kwargs"]["start_year"] == 2018
    assert captured["build_kwargs"]["end_year"] == 2025
    weather_fetcher = captured["build_kwargs"]["weather_fetcher"]
    umpire_fetcher = captured["build_kwargs"]["umpire_fetcher"]
    assert isinstance(weather_fetcher, partial)
    assert weather_fetcher.func is module.fetch_game_weather_local_only
    assert isinstance(umpire_fetcher, partial)
    assert umpire_fetcher.func is module.fetch_retrosheet_umpires


def test_build_parquet_uses_live_weather_fetcher_when_refreshing(tmp_path: Path) -> None:
    module = _load_script_module("build_parquet.py")
    training_path = tmp_path / "ParquetDefault.parquet"
    captured: dict[str, object] = {}

    def _fake_build_training_dataset(**kwargs):
        captured["build_kwargs"] = kwargs
        return SimpleNamespace(
            dataframe=pd.DataFrame([_base_training_row()]),
            output_path=Path(kwargs["output_path"]),
            metadata_path=Path(kwargs["output_path"]).with_suffix(".metadata.json"),
            data_version_hash="synthetic-data-hash",
        )

    module.build_training_dataset = _fake_build_training_dataset

    exit_code = module.main(
        [
            "--training-data",
            str(training_path),
            "--refresh-data",
        ]
    )

    assert exit_code == 0
    assert captured["build_kwargs"]["weather_fetcher"] is module.fetch_game_weather
    umpire_fetcher = captured["build_kwargs"]["umpire_fetcher"]
    assert isinstance(umpire_fetcher, partial)
    assert umpire_fetcher.func is module.fetch_retrosheet_umpires


def test_rebuild_and_train_single_model_uses_research_defaults(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script_module("train_run_count.py")
    training_path = tmp_path / "rebuilt_training_data.parquet"
    captured: dict[str, object] = {}
    _write_training_parquet(training_path, include_delta_features=True)

    def _fake_train_run_count_models(**kwargs):
        captured["train_kwargs"] = kwargs
        return SimpleNamespace(
            summary_path=tmp_path / "summary.json",
            models={
                module.MODEL_NAME: SimpleNamespace(
                    holdout_metrics={
                        "r2": 0.12,
                        "rmse": 1.9,
                        "poisson_deviance": 0.98,
                        "rmse_improvement_vs_naive_pct": 4.5,
                    },
                    final_n_estimators=42,
                    optuna_parallel_workers=2,
                    best_params={"max_depth": 4},
                )
            },
        )

    module.rct.train_run_count_models = _fake_train_run_count_models

    exit_code = module.main(
        [
            "--profile",
            "fast",
            "--start",
            "2018",
            "--end",
            "2025",
            "--holdout",
            "2025",
            "--XGBWork",
            "4",
            "--OptunaWork",
            "3",
            "--Iterations",
            "55",
            "--Folds",
            "4",
            "--training-data",
            str(training_path),
        ]
    )

    assert exit_code == 0
    validated_training_data = captured["train_kwargs"]["training_data"]
    assert isinstance(validated_training_data, pd.DataFrame)
    assert validated_training_data.loc[0, "data_version_hash"] == "synthetic-data-hash"
    assert set(RUN_COUNT_REQUIRED_TEMPORAL_DELTA_COLUMNS).issubset(validated_training_data.columns)
    assert validated_training_data["season"].min() == 2025
    assert validated_training_data["season"].max() == 2025
    assert captured["train_kwargs"]["optuna_workers"] == 3
    assert captured["train_kwargs"]["search_iterations"] == 55
    assert captured["train_kwargs"]["time_series_splits"] == 4
    assert captured["train_kwargs"]["feature_selection_mode"] == "grouped"
    assert captured["train_kwargs"]["forced_delta_feature_count"] == 0
    assert captured["train_kwargs"]["cv_aggregation_mode"] == "mean"
    assert captured["train_kwargs"]["lightgbm_param_mode"] == "independent"
    assert captured["train_kwargs"]["blend_mode"] == "learned"

    stdout = capsys.readouterr().out
    assert "Training parquet" in stdout
    assert "parquet_path=<in-memory>" in stdout
    assert "data_version_hash=synthetic-data-hash" in stdout
    assert "xgb_workers=4" in stdout
    assert "optuna_workers=3" in stdout
    assert "iterations=55" in stdout
    assert "folds=4" in stdout
    assert "forced_delta_count=0" in stdout


def test_train_run_count_passes_forced_delta_count(tmp_path: Path) -> None:
    module = _load_script_module("train_run_count.py")
    training_path = tmp_path / "forced_delta_training_data.parquet"
    captured: dict[str, object] = {}
    _write_training_parquet(training_path, include_delta_features=True)

    def _fake_train_run_count_models(**kwargs):
        captured["train_kwargs"] = kwargs
        return SimpleNamespace(
            summary_path=tmp_path / "summary.json",
            models={
                module.MODEL_NAME: SimpleNamespace(
                    holdout_metrics={
                        "r2": 0.12,
                        "rmse": 1.9,
                        "poisson_deviance": 0.98,
                        "rmse_improvement_vs_naive_pct": 4.5,
                    },
                    final_n_estimators=42,
                    optuna_parallel_workers=2,
                    best_params={"max_depth": 4},
                )
            },
        )

    module.rct.train_run_count_models = _fake_train_run_count_models

    exit_code = module.main(
        [
            "--profile",
            "flat-fast",
            "--training-data",
            str(training_path),
            "--forced-delta-count",
            "12",
        ]
    )

    assert exit_code == 0
    assert captured["train_kwargs"]["feature_selection_mode"] == "flat"
    assert captured["train_kwargs"]["forced_delta_feature_count"] == 12
