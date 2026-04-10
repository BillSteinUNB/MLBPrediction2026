from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass

from src.config import _load_settings_yaml


_SETTINGS_PAYLOAD = _load_settings_yaml()


def _normalize_text(value: str | None) -> str:
    if value is None:
        return ""

    normalized = unicodedata.normalize("NFKD", str(value))
    without_marks = "".join(character for character in normalized if not unicodedata.combining(character))
    return " ".join(without_marks.casefold().split())


DEFAULT_WALK_RATE_DELTA = float(_SETTINGS_PAYLOAD["abs_adjustments"]["walk_rate_delta"])
DEFAULT_STRIKEOUT_RATE_DELTA = float(_SETTINGS_PAYLOAD["abs_adjustments"]["strikeout_rate_delta"])
DEFAULT_ABS_RETENTION_FACTOR = float(_SETTINGS_PAYLOAD["abs_retention_factor"])
ABS_EXCEPTION_IDENTIFIERS = tuple(
    _normalize_text(identifier) for identifier in _SETTINGS_PAYLOAD.get("abs_exceptions", [])
)


@dataclass(frozen=True, slots=True)
class AbsAdjustmentResult:
    """Resolved ABS adjustment state for a venue and pair of rates."""

    abs_active: bool
    walk_rate_delta: float
    strikeout_rate_delta: float
    adjusted_walk_rate: float
    adjusted_strikeout_rate: float


@dataclass(frozen=True, slots=True)
class AbsChallengeProxyResult:
    """Proxy estimates for challenge-era ABS context without event-level challenge logs."""

    abs_active: bool
    challenge_opportunity_proxy: float
    challenge_pressure_proxy: float
    challenge_conservation_proxy: float
    leverage_framing_retention_proxy: float
    umpire_zone_suppression_proxy: float


def is_abs_exception_venue(venue: str | None) -> bool:
    """Return whether a venue should bypass ABS adjustments."""

    normalized_venue = _normalize_text(venue)
    if not normalized_venue:
        return False

    return any(
        identifier and (identifier in normalized_venue or normalized_venue in identifier)
        for identifier in ABS_EXCEPTION_IDENTIFIERS
    )


def is_abs_active(venue: str | None) -> bool:
    """Return whether ABS adjustments should be active for the venue."""

    return not is_abs_exception_venue(venue)


def apply_abs_adjustments(
    walk_rate: float,
    strikeout_rate: float,
    *,
    venue: str | None = None,
    abs_active: bool | None = None,
    walk_rate_delta: float = DEFAULT_WALK_RATE_DELTA,
    strikeout_rate_delta: float = DEFAULT_STRIKEOUT_RATE_DELTA,
    decay_factor: float = 1.0,
) -> AbsAdjustmentResult:
    """Apply league-wide ABS walk and strikeout deltas unless the venue is exempt."""

    resolved_abs_active = is_abs_active(venue) if abs_active is None else bool(abs_active)
    resolved_walk_rate = max(float(walk_rate), 0.0)
    resolved_strikeout_rate = max(float(strikeout_rate), 0.0)

    if not resolved_abs_active:
        return AbsAdjustmentResult(
            abs_active=False,
            walk_rate_delta=0.0,
            strikeout_rate_delta=0.0,
            adjusted_walk_rate=resolved_walk_rate,
            adjusted_strikeout_rate=resolved_strikeout_rate,
        )

    resolved_decay_factor = max(float(decay_factor), 0.0)
    resolved_walk_delta = float(walk_rate_delta) * resolved_decay_factor
    resolved_strikeout_delta = float(strikeout_rate_delta) * resolved_decay_factor

    return AbsAdjustmentResult(
        abs_active=True,
        walk_rate_delta=resolved_walk_delta,
        strikeout_rate_delta=resolved_strikeout_delta,
        adjusted_walk_rate=max(resolved_walk_rate * (1 + resolved_walk_delta), 0.0),
        adjusted_strikeout_rate=max(
            resolved_strikeout_rate * (1 + resolved_strikeout_delta),
            0.0,
        ),
    )


def adjust_framing_for_abs(
    raw_framing_runs: float,
    *,
    abs_active: bool = True,
    retention_factor: float = DEFAULT_ABS_RETENTION_FACTOR,
) -> float:
    """Apply the configured framing retention factor in ABS-active environments."""

    return float(raw_framing_runs) * (float(retention_factor) if abs_active else 1.0)


def estimate_framing_retention_proxy(
    raw_framing_runs: float,
    *,
    adjusted_framing_runs: float | None = None,
    abs_active: bool = True,
    retention_factor: float = DEFAULT_ABS_RETENTION_FACTOR,
) -> float:
    """Estimate how much catcher framing value remains after ABS filtering."""

    if not abs_active:
        return 1.0

    resolved_raw_framing = abs(float(raw_framing_runs))
    if adjusted_framing_runs is None:
        adjusted_framing_runs = adjust_framing_for_abs(
            raw_framing_runs,
            abs_active=abs_active,
            retention_factor=retention_factor,
        )
    resolved_adjusted_framing = abs(float(adjusted_framing_runs))
    if resolved_raw_framing <= 1e-9:
        return _clip(float(retention_factor), 0.0, 1.0)
    return _clip(resolved_adjusted_framing / resolved_raw_framing, 0.0, 1.0)


def build_abs_challenge_proxy_context(
    *,
    abs_active: bool,
    lineup_walk_rate: float,
    lineup_strikeout_rate: float,
    lineup_quality: float,
    abs_walk_rate_delta: float = DEFAULT_WALK_RATE_DELTA,
    abs_strikeout_rate_delta: float = DEFAULT_STRIKEOUT_RATE_DELTA,
    umpire_zone_suppression: float = 0.0,
    umpire_zone_volatility: float = 0.0,
    umpire_abs_active_share: float = 0.5,
    framing_retention_proxy: float = DEFAULT_ABS_RETENTION_FACTOR,
    framing_stability: float = 0.75,
    framing_zone_support: float = 0.0,
    run_environment_anchor: float = 1.0,
    market_anchor_confidence: float = 0.0,
) -> AbsChallengeProxyResult:
    """Build honest ABS challenge proxies from available lineup, framing, and umpire context.

    These features are bounded proxy signals. They intentionally do not infer
    unavailable event-level challenge counts or pitch-by-pitch challenge usage.
    """

    if not abs_active:
        return AbsChallengeProxyResult(
            abs_active=False,
            challenge_opportunity_proxy=0.0,
            challenge_pressure_proxy=0.0,
            challenge_conservation_proxy=1.0,
            leverage_framing_retention_proxy=1.0,
            umpire_zone_suppression_proxy=0.0,
        )

    resolved_walk_rate = max(float(lineup_walk_rate), 0.0)
    resolved_strikeout_rate = max(float(lineup_strikeout_rate), 0.0)
    resolved_quality = max(float(lineup_quality), 0.0)
    resolved_walk_delta = abs(float(abs_walk_rate_delta))
    resolved_strikeout_delta = abs(float(abs_strikeout_rate_delta))
    resolved_zone_suppression = float(umpire_zone_suppression)
    resolved_zone_volatility = max(float(umpire_zone_volatility), 0.0)
    resolved_abs_active_share = _clip(float(umpire_abs_active_share), 0.0, 1.0)
    resolved_framing_retention = _clip(float(framing_retention_proxy), 0.0, 1.0)
    resolved_framing_stability = _clip(float(framing_stability), 0.0, 1.0)
    resolved_framing_support = float(framing_zone_support)
    resolved_environment_anchor = max(float(run_environment_anchor), 0.0)
    resolved_market_confidence = _clip(float(market_anchor_confidence), 0.0, 1.0)

    discipline_signal = (
        0.35 * ((resolved_walk_rate - 8.2) / 2.0)
        + 0.30 * ((22.8 - resolved_strikeout_rate) / 4.0)
        + 0.20 * ((resolved_quality - 0.315) / 0.020)
        + 0.15 * resolved_zone_volatility
    )
    delta_signal = (resolved_walk_delta * 5.0) + (resolved_strikeout_delta * 4.0)
    challenge_opportunity = _sigmoid(discipline_signal + delta_signal - 0.20)

    leverage_signal = (
        0.45 * ((resolved_quality - 0.315) / 0.020)
        + 0.35 * ((resolved_environment_anchor - 1.0) / 0.15) * resolved_market_confidence
        + 0.20 * resolved_zone_suppression
    )
    challenge_pressure = _clip(
        challenge_opportunity * (0.85 + 0.18 * leverage_signal + 0.06 * resolved_zone_volatility),
        0.0,
        1.0,
    )
    challenge_conservation = _clip(
        0.55
        + 0.25 * ((resolved_abs_active_share - 0.50) / 0.50)
        + 0.20 * ((resolved_framing_stability - 0.75) / 0.25)
        - 0.30 * (challenge_pressure - 0.50),
        0.0,
        1.0,
    )
    leverage_framing_retention = _clip(
        resolved_framing_retention
        * (
            1.0
            + 0.20 * (challenge_conservation - 0.50)
            + 0.15 * ((resolved_framing_stability - 0.75) / 0.25)
            + 0.12 * (resolved_framing_support / 2.0)
            - 0.18 * (challenge_pressure - 0.50)
        ),
        0.15,
        1.10,
    )
    umpire_zone_suppression_proxy = _signed_clip(
        resolved_zone_suppression,
        lower=0.0,
        upper=2.0,
    ) * (0.50 + 0.50 * resolved_abs_active_share) * (0.70 + 0.30 * challenge_conservation)
    if resolved_zone_suppression >= 0.0:
        umpire_zone_suppression_proxy = max(
            float(umpire_zone_suppression_proxy) - (resolved_zone_volatility * 0.10),
            0.0,
        )
    else:
        umpire_zone_suppression_proxy = min(
            float(umpire_zone_suppression_proxy) + (resolved_zone_volatility * 0.10),
            0.0,
        )

    return AbsChallengeProxyResult(
        abs_active=True,
        challenge_opportunity_proxy=float(challenge_opportunity),
        challenge_pressure_proxy=float(challenge_pressure),
        challenge_conservation_proxy=float(challenge_conservation),
        leverage_framing_retention_proxy=float(leverage_framing_retention),
        umpire_zone_suppression_proxy=float(_clip(umpire_zone_suppression_proxy, -2.0, 2.0)),
    )


def _clip(value: float, lower: float, upper: float) -> float:
    return max(float(lower), min(float(value), float(upper)))


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(value)))


def _signed_clip(value: float, *, lower: float, upper: float) -> float:
    magnitude = _clip(abs(float(value)), lower, upper)
    if float(value) < 0.0:
        return -magnitude
    return magnitude
