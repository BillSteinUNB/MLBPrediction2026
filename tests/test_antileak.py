from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.model.data_builder import assert_training_data_is_leakage_free
from tests.test_data_builder import _write_cached_training_validation_fixture


@pytest.fixture()
def cached_training_frame(tmp_path: Path) -> pd.DataFrame:
    parquet_path = tmp_path / "training_data_antileak_fixture.parquet"
    return _write_cached_training_validation_fixture(parquet_path)


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
