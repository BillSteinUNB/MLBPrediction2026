from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.model.data_builder import assert_training_data_is_complete
from tests.test_data_builder import (
    _VALIDATION_FIXTURE_SEASONS,
    _write_cached_training_validation_fixture,
)


@pytest.fixture()
def cached_training_data(tmp_path: Path) -> tuple[Path, pd.DataFrame]:
    parquet_path = tmp_path / "training_data_integrity_fixture.parquet"
    dataframe = _write_cached_training_validation_fixture(parquet_path)
    return parquet_path, dataframe


def _assert_basic_training_data_integrity(dataframe: pd.DataFrame) -> None:
    duplicate_game_pks = dataframe.loc[dataframe["game_pk"].duplicated(), "game_pk"].tolist()
    if duplicate_game_pks:
        raise AssertionError(f"Found duplicate game_pk values: {duplicate_game_pks[:5]}")

    if dataframe[["f5_ml_result", "f5_rl_result"]].isna().any().any():
        raise AssertionError("Training targets must not contain NaN values")

    non_regular_games = sorted({str(game_type) for game_type in dataframe["game_type"] if game_type != "R"})
    if non_regular_games:
        raise AssertionError(f"Found non-regular game types: {non_regular_games}")


def test_training_data_fixture_has_unique_regular_season_rows_and_non_null_targets(
    cached_training_data: tuple[Path, pd.DataFrame],
) -> None:
    parquet_path, dataframe = cached_training_data

    summary = assert_training_data_is_complete(parquet_path)
    _assert_basic_training_data_integrity(dataframe)

    assert summary.row_count == len(dataframe) == 17_010
    assert dataframe["game_pk"].is_unique
    assert summary.target_null_counts == {"f5_ml_result": 0, "f5_rl_result": 0}
    assert summary.game_type_counts == {"R": 17_010}
    assert summary.seasons == _VALIDATION_FIXTURE_SEASONS


def test_training_data_integrity_flags_duplicate_game_pk(
    cached_training_data: tuple[Path, pd.DataFrame],
) -> None:
    _, dataframe = cached_training_data
    duplicated = pd.concat([dataframe, dataframe.iloc[[0]]], ignore_index=True)

    with pytest.raises(AssertionError, match="Found duplicate game_pk values"):
        _assert_basic_training_data_integrity(duplicated)


def test_training_data_integrity_flags_spring_training_or_postseason_games(
    cached_training_data: tuple[Path, pd.DataFrame],
) -> None:
    _, dataframe = cached_training_data
    mutated = dataframe.copy()
    mutated.loc[0, "game_type"] = "S"
    mutated.loc[1, "game_type"] = "F"

    with pytest.raises(AssertionError, match="Found non-regular game types"):
        _assert_basic_training_data_integrity(mutated)
