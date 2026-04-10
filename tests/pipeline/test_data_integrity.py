from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.model.data_builder import DEFAULT_OUTPUT_PATH
from src.model.data_builder import assert_training_data_is_complete
from tests.test_data_builder import _VALIDATION_FIXTURE_SEASONS


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
def cached_training_data() -> tuple[Path, pd.DataFrame]:
    for parquet_path in _CACHED_TRAINING_PARQUET_CANDIDATES:
        if parquet_path.exists():
            return parquet_path, pd.read_parquet(parquet_path)
    raise AssertionError(
        "No cached training parquet found. Expected one of: "
        + ", ".join(str(path) for path in _CACHED_TRAINING_PARQUET_CANDIDATES)
    )


def _assert_basic_training_data_integrity(dataframe: pd.DataFrame) -> None:
    duplicate_game_pks = dataframe.loc[dataframe["game_pk"].duplicated(), "game_pk"].tolist()
    if duplicate_game_pks:
        raise AssertionError(f"Found duplicate game_pk values: {duplicate_game_pks[:5]}")

    if dataframe[["f5_ml_result", "f5_rl_result"]].isna().any().any():
        raise AssertionError("Training targets must not contain NaN values")

    non_regular_games = sorted({str(game_type) for game_type in dataframe["game_type"] if game_type != "R"})
    if non_regular_games:
        raise AssertionError(f"Found non-regular game types: {non_regular_games}")


def _assert_real_training_features_present(dataframe: pd.DataFrame) -> None:
    expected_columns = {
        "home_team_wrc_plus_7g",
        "away_team_wrc_plus_7g",
        "home_starter_xfip_7s",
        "away_starter_xfip_7s",
        "home_team_drs_season",
        "home_team_bullpen_xfip",
        "home_team_log5_30g",
        "weather_composite",
    }
    missing_columns = sorted(expected_columns.difference(dataframe.columns))
    if missing_columns:
        raise AssertionError(
            "Expected builder-backed training features to be present: "
            + ", ".join(missing_columns)
        )


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
    if set(
        {
            "home_team_wrc_plus_7g",
            "away_team_wrc_plus_7g",
            "home_starter_xfip_7s",
            "away_starter_xfip_7s",
        }
    ).issubset(dataframe.columns):
        _assert_real_training_features_present(dataframe)
        april_ten_rows = dataframe.loc[
            (dataframe["season"] == 2025) & (dataframe["game_date"] == "2025-04-10")
        ]
        assert not april_ten_rows.empty
        assert april_ten_rows["home_team_wrc_plus_7g"].between(50.0, 200.0).all()
        assert april_ten_rows["away_team_wrc_plus_7g"].between(50.0, 200.0).all()
        assert april_ten_rows["home_starter_xfip_7s"].between(2.0, 6.0).all()
        assert april_ten_rows["away_starter_xfip_7s"].between(2.0, 6.0).all()


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
