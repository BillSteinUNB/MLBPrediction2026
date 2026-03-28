from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SingleModelExperimentProfile:
    profile_name: str
    experiment_name: str
    search_iterations: int
    time_series_splits: int
    early_stopping_rounds: int
    search_space: dict[str, list[float | int]]


_FULL_SEARCH_SPACE: dict[str, list[float | int]] = {
    "max_depth": [3, 4, 5],
    "n_estimators": [500, 600, 700, 800, 900, 1000],
    "learning_rate": [0.005, 0.0075, 0.01, 0.015, 0.02],
    "subsample": [0.65, 0.75, 0.85, 0.95],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8],
    "min_child_weight": [3, 4, 5, 6, 7, 8],
    "gamma": [0.0, 0.05, 0.1, 0.2, 0.3],
    "reg_alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
    "reg_lambda": [0.1, 0.25, 0.5, 1.0, 2.0, 4.0],
}

_SMOKE_SEARCH_SPACE: dict[str, list[float | int]] = {
    "max_depth": [3, 4],
    "n_estimators": [200, 300, 400],
    "learning_rate": [0.01, 0.015, 0.02],
    "subsample": [0.7, 0.85],
    "colsample_bytree": [0.5, 0.7],
    "min_child_weight": [4, 6],
    "gamma": [0.0, 0.1, 0.3],
    "reg_alpha": [1e-4, 1e-2, 1e-1],
    "reg_lambda": [0.25, 1.0, 4.0],
}

_PROFILE_DEFAULTS: dict[str, dict[str, object]] = {
    "smoke": {
        "experiment_name": "2026-away-smoke-deltas-poisson-parallel-12x2",
        "search_iterations": 12,
        "time_series_splits": 2,
        "early_stopping_rounds": 20,
        "search_space": _SMOKE_SEARCH_SPACE,
    },
    "fast": {
        "experiment_name": "2026-away-fast-deltas-poisson-parallel-120x3",
        "search_iterations": 120,
        "time_series_splits": 3,
        "early_stopping_rounds": 30,
        "search_space": _FULL_SEARCH_SPACE,
    },
    "full": {
        "experiment_name": "2026-run12-away-deltas-poisson-parallel-500x5",
        "search_iterations": 500,
        "time_series_splits": 5,
        "early_stopping_rounds": 40,
        "search_space": _FULL_SEARCH_SPACE,
    },
    "flat-fast": {
        "experiment_name": "2026-away-flat-selector-fast-120x3",
        "search_iterations": 120,
        "time_series_splits": 3,
        "early_stopping_rounds": 30,
        "search_space": _FULL_SEARCH_SPACE,
    },
    "flat-full": {
        "experiment_name": "2026-away-flat-selector-full-500x5",
        "search_iterations": 500,
        "time_series_splits": 5,
        "early_stopping_rounds": 40,
        "search_space": _FULL_SEARCH_SPACE,
    },
}


def resolve_single_model_experiment_profile(
    profile_name: str,
    *,
    experiment_name_override: str | None = None,
) -> SingleModelExperimentProfile:
    normalized_name = profile_name.strip().lower()
    if normalized_name not in _PROFILE_DEFAULTS:
        raise ValueError(f"Unknown single-model experiment profile: {profile_name}")

    defaults = _PROFILE_DEFAULTS[normalized_name]
    return SingleModelExperimentProfile(
        profile_name=normalized_name,
        experiment_name=(
            str(experiment_name_override).strip()
            if experiment_name_override is not None
            else str(defaults["experiment_name"])
        ),
        search_iterations=int(defaults["search_iterations"]),
        time_series_splits=int(defaults["time_series_splits"]),
        early_stopping_rounds=int(defaults["early_stopping_rounds"]),
        search_space={key: list(values) for key, values in dict(defaults["search_space"]).items()},
    )
