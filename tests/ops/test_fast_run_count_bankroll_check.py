from __future__ import annotations

from pathlib import Path

from src.ops.fast_run_count_bankroll_check import _auto_supported_markets
from src.ops.fast_run_count_bankroll_check import _infer_joblib_path
from src.ops.fast_run_count_bankroll_check import _CompanionBundle


def test_auto_supported_markets_requires_matching_companions() -> None:
    full_game_bundle = _CompanionBundle(
        model_name="full_game_home_runs_model",
        model=object(),
        feature_columns=[],
        rmse=1.0,
        metadata_path=Path("full_game.metadata.json"),
        model_version="fg",
    )
    f5_home_bundle = _CompanionBundle(
        model_name="f5_home_runs_model",
        model=object(),
        feature_columns=[],
        rmse=1.0,
        metadata_path=Path("f5_home.metadata.json"),
        model_version="f5h",
    )
    f5_away_bundle = _CompanionBundle(
        model_name="f5_away_runs_model",
        model=object(),
        feature_columns=[],
        rmse=1.0,
        metadata_path=Path("f5_away.metadata.json"),
        model_version="f5a",
    )

    assert _auto_supported_markets({"full_game_home_runs_model": full_game_bundle}) == [
        "full_game_ml",
        "full_game_rl",
        "full_game_total",
    ]
    assert _auto_supported_markets(
        {
            "full_game_home_runs_model": full_game_bundle,
            "f5_home_runs_model": f5_home_bundle,
            "f5_away_runs_model": f5_away_bundle,
        }
    ) == [
        "full_game_ml",
        "full_game_rl",
        "full_game_total",
        "f5_ml",
        "f5_total",
    ]


def test_infer_joblib_path_prefers_explicit_artifact_path() -> None:
    metadata_path = Path("C:/tmp/model.metadata.json")
    payload = {
        "artifact": {
            "model_path": "C:/tmp/custom_model.joblib",
        }
    }

    assert _infer_joblib_path(metadata_path, payload) == Path("C:/tmp/custom_model.joblib")
