from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping


MetricType = Literal["offense", "pitching", "defense"]
DEFAULT_REGRESSION_WEIGHTS: Mapping[MetricType, float] = MappingProxyType(
    {
        "offense": 30.0,
        "pitching": 15.0,
        "defense": 50.0,
    }
)
DEFAULT_TURNOVER_THRESHOLD = 0.40
DEFAULT_TURNOVER_PRIOR_WEIGHT_MULTIPLIER = 0.50


@dataclass(frozen=True, slots=True)
class MarcelBlendResult:
    blended_value: float
    resolved_current_value: float
    resolved_prior_value: float
    games_played: float
    regression_weight: float
    prior_weight: float
    used_league_average: bool
    turnover_adjusted: bool


def get_regression_weight(
    metric_type: MetricType,
    *,
    regression_weights: Mapping[MetricType, float] | None = None,
) -> float:
    """Return the configured Marcel regression weight for a metric family."""

    if regression_weights is not None and metric_type in regression_weights:
        return float(regression_weights[metric_type])
    return float(DEFAULT_REGRESSION_WEIGHTS[metric_type])


def calculate_marcel_blend(
    current_value: float,
    *,
    games_played: int | float,
    metric_type: MetricType,
    prior_value: float | None = None,
    league_average: float | None = None,
    regression_weight: float | None = None,
    regression_weights: Mapping[MetricType, float] | None = None,
    roster_turnover_pct: float | None = None,
    turnover_threshold: float = DEFAULT_TURNOVER_THRESHOLD,
    turnover_prior_weight_multiplier: float = DEFAULT_TURNOVER_PRIOR_WEIGHT_MULTIPLIER,
    is_first_year: bool = False,
) -> MarcelBlendResult:
    """Blend current and prior values using Marcel-style regression."""

    resolved_games_played = max(_coerce_float(games_played, default=0.0), 0.0)
    base_regression_weight = max(
        float(regression_weight)
        if regression_weight is not None
        else get_regression_weight(metric_type, regression_weights=regression_weights),
        0.0,
    )

    resolved_prior_value, used_league_average = _resolve_prior_value(
        prior_value=prior_value,
        league_average=league_average,
        current_value=current_value,
        is_first_year=is_first_year,
    )
    resolved_current_value = _coerce_float(current_value, default=resolved_prior_value)

    turnover_adjusted = False
    prior_weight = base_regression_weight
    if (
        not _is_missing(roster_turnover_pct)
        and float(roster_turnover_pct) > turnover_threshold
        and prior_weight > 0
    ):
        prior_weight *= turnover_prior_weight_multiplier
        turnover_adjusted = True

    denominator = resolved_games_played + prior_weight
    blended_value = resolved_current_value
    if denominator > 0:
        blended_value = (
            (resolved_current_value * resolved_games_played)
            + (resolved_prior_value * prior_weight)
        ) / denominator

    return MarcelBlendResult(
        blended_value=float(blended_value),
        resolved_current_value=float(resolved_current_value),
        resolved_prior_value=float(resolved_prior_value),
        games_played=float(resolved_games_played),
        regression_weight=float(base_regression_weight),
        prior_weight=float(prior_weight),
        used_league_average=used_league_average,
        turnover_adjusted=turnover_adjusted,
    )


def blend_value(current_value: float, **kwargs: object) -> float:
    """Return only the blended Marcel value."""

    return calculate_marcel_blend(current_value, **kwargs).blended_value


def _resolve_prior_value(
    *,
    prior_value: float | None,
    league_average: float | None,
    current_value: float,
    is_first_year: bool,
) -> tuple[float, bool]:
    if not is_first_year and not _is_missing(prior_value):
        return float(prior_value), False

    if not _is_missing(league_average):
        return float(league_average), True

    return _coerce_float(current_value, default=0.0), False


def _coerce_float(value: float | int | None, *, default: float) -> float:
    if _is_missing(value):
        return float(default)
    return float(value)


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return False
