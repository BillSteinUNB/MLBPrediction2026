from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.model.data_builder import DEFAULT_OUTPUT_PATH
from src.model.data_builder import assert_training_data_is_leakage_free
from src.model.stacking import _PartitionedTemporalCV, _build_oof_fold_sizes
from src.model.xgboost_trainer import create_time_series_split


_CACHED_TRAINING_PARQUET_CANDIDATES = (
    DEFAULT_OUTPUT_PATH,
    Path(".factory")
    / "validation"
    / "ml-pipeline"
    / "user-testing"
    / "work"
    / "data-completeness-rerun"
    / "training_data_validation_fixture.parquet",
)


@pytest.fixture()
def cached_training_frame() -> pd.DataFrame:
    for parquet_path in _CACHED_TRAINING_PARQUET_CANDIDATES:
        if parquet_path.exists():
            return pd.read_parquet(parquet_path)
    raise AssertionError(
        "No cached training parquet found. Expected one of: "
        + ", ".join(str(path) for path in _CACHED_TRAINING_PARQUET_CANDIDATES)
    )


def test_antileak_random_sample_of_100_games_uses_prior_day_snapshot(
    cached_training_frame: pd.DataFrame,
) -> None:
    sampled_games = cached_training_frame.sample(n=100, random_state=2026).sort_values("game_pk")
    as_of_timestamp = pd.to_datetime(sampled_games["as_of_timestamp"], utc=True)
    scheduled_start = pd.to_datetime(sampled_games["scheduled_start"], utc=True)

    assert len(sampled_games) == 100
    assert (as_of_timestamp < scheduled_start).all()
    assert (as_of_timestamp.dt.normalize() == (scheduled_start.dt.normalize() - pd.Timedelta(days=1))).all()

    assert_training_data_is_leakage_free(sampled_games)

    if set(
        {
            "home_team_wrc_plus_7g",
            "away_team_wrc_plus_7g",
            "home_starter_xfip_7s",
            "away_starter_xfip_7s",
        }
    ).issubset(cached_training_frame.columns):
        april_ten_games = cached_training_frame.loc[
            (cached_training_frame["season"] == 2025)
            & (cached_training_frame["game_date"] == "2025-04-10")
        ]
        assert not april_ten_games.empty
        assert april_ten_games["home_team_wrc_plus_7g"].between(50.0, 200.0).all()
        assert april_ten_games["away_team_wrc_plus_7g"].between(50.0, 200.0).all()
        assert april_ten_games["home_starter_xfip_7s"].between(2.0, 6.0).all()
        assert april_ten_games["away_starter_xfip_7s"].between(2.0, 6.0).all()
        assert_training_data_is_leakage_free(april_ten_games)


def test_antileak_assertion_rejects_same_day_or_future_timestamp(
    cached_training_frame: pd.DataFrame,
) -> None:
    sampled_games = cached_training_frame.sample(n=100, random_state=7).copy()
    leaked_index = sampled_games.index[0]
    sampled_games.loc[leaked_index, "as_of_timestamp"] = sampled_games.loc[
        leaked_index, "scheduled_start"
    ]

    with pytest.raises(AssertionError, match="Anti-leakage assertion failed"):
        assert_training_data_is_leakage_free(sampled_games)


def test_antileak_time_series_split_never_trains_on_future_rows() -> None:
    splitter = create_time_series_split(row_count=24, requested_splits=4)

    for train_indices, test_indices in splitter.split(range(24)):
        assert len(train_indices) > 0
        assert len(test_indices) > 0
        assert max(train_indices) < min(test_indices)


def test_antileak_stacking_partitioned_temporal_cv_preserves_contiguous_future_test_blocks() -> None:
    fold_sizes = _build_oof_fold_sizes(row_count=15, requested_splits=4)
    splitter = _PartitionedTemporalCV(fold_sizes)

    running_start = 0
    for train_indices, test_indices in splitter.split(pd.DataFrame({"x": range(15)})):
        assert len(test_indices) > 0
        if len(train_indices) > 0:
            assert max(train_indices) < min(test_indices)
            assert train_indices[-1] == running_start - 1
        else:
            assert running_start == 0
        assert test_indices[0] == running_start
        assert test_indices[-1] == running_start + len(test_indices) - 1
        running_start += len(test_indices)
