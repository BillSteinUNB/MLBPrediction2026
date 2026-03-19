from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Any

from src.config import _load_settings_yaml


LEAGUE_AVERAGE_PARK_FACTOR = 1.0
_SETTINGS_PAYLOAD = _load_settings_yaml()


def _normalize_text(value: str | None) -> str:
    if value is None:
        return ""

    normalized = unicodedata.normalize("NFKD", str(value))
    without_marks = "".join(character for character in normalized if not unicodedata.combining(character))
    return " ".join(without_marks.casefold().split())


def _coerce_factor(payload: Any, key: str) -> float:
    if isinstance(payload, dict) and key in payload:
        return float(payload[key])
    return LEAGUE_AVERAGE_PARK_FACTOR


@dataclass(frozen=True, slots=True)
class ParkFactor:
    """Resolved park-factor context for a team or venue."""

    team_code: str | None
    park_name: str
    runs: float
    hr: float


def load_park_factors() -> dict[str, ParkFactor]:
    """Load configured park factors for all MLB teams."""

    stadiums = _SETTINGS_PAYLOAD.get("stadiums", {})
    factors: dict[str, ParkFactor] = {}

    for team_code, stadium_payload in stadiums.items():
        if not isinstance(stadium_payload, dict):
            continue

        park_payload = stadium_payload.get("park_factors", {})
        factors[str(team_code).upper()] = ParkFactor(
            team_code=str(team_code).upper(),
            park_name=str(stadium_payload.get("park_name") or team_code),
            runs=float(_coerce_factor(park_payload, "runs")),
            hr=float(_coerce_factor(park_payload, "hr")),
        )

    return factors


_PARK_FACTORS_BY_TEAM = load_park_factors()
_PARK_FACTORS_BY_VENUE = {
    _normalize_text(park_factor.park_name): park_factor
    for park_factor in _PARK_FACTORS_BY_TEAM.values()
}


def get_park_factors(
    *,
    team_code: str | None = None,
    venue: str | None = None,
) -> ParkFactor:
    """Resolve park factors by team code or venue, defaulting to league average."""

    if team_code:
        normalized_team_code = team_code.strip().upper()
        resolved = _PARK_FACTORS_BY_TEAM.get(normalized_team_code)
        if resolved is not None:
            return resolved

    if venue:
        normalized_venue = _normalize_text(venue)
        if normalized_venue:
            resolved = _PARK_FACTORS_BY_VENUE.get(normalized_venue)
            if resolved is not None:
                return resolved

            for known_venue, known_factors in _PARK_FACTORS_BY_VENUE.items():
                if known_venue in normalized_venue or normalized_venue in known_venue:
                    return known_factors

    return ParkFactor(
        team_code=None,
        park_name=(venue or team_code or "League Average Park").strip() or "League Average Park",
        runs=LEAGUE_AVERAGE_PARK_FACTOR,
        hr=LEAGUE_AVERAGE_PARK_FACTOR,
    )


def adjust_for_park(metric: float, park_factor: float | ParkFactor) -> float:
    """Scale a metric by the resolved park factor."""

    resolved_factor = park_factor.runs if isinstance(park_factor, ParkFactor) else float(park_factor)
    if resolved_factor <= 0:
        raise ValueError("park_factor must be positive")

    return float(metric) * resolved_factor


