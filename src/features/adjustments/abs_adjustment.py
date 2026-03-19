from __future__ import annotations

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
