"""Generate human-readable narratives for MLB F5 betting recommendations.

This module translates raw model features and predictions into shareable
pick write-ups.  Play-of-the-Day picks receive a multi-sentence narrative;
regular value bets get a concise one-liner.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


# ---------------------------------------------------------------------------
# Baselines / thresholds used when comparing a feature to "normal"
# ---------------------------------------------------------------------------

LEAGUE_WRC_PLUS = 100.0
LEAGUE_XFIP = 4.20
LEAGUE_K_PCT = 22.0
LEAGUE_BB_PCT = 8.0
LEAGUE_WOBA = 0.320
LEAGUE_ISO = 0.150

_HOT_WRC_PLUS_THRESHOLD = 115.0  # top ~25 %
_COLD_WRC_PLUS_THRESHOLD = 85.0  # bottom ~25 %
_STRONG_XFIP_THRESHOLD = 3.60  # ace-calibre F5 starter
_WEAK_XFIP_THRESHOLD = 4.80  # struggling starter
_HIGH_K_PCT_THRESHOLD = 27.0  # strikeout pitcher
_HIGH_BB_PCT_THRESHOLD = 10.5  # control issues
_NOTABLE_EDGE_THRESHOLD = 0.06  # edge worth calling out
_DOMINANT_EDGE_THRESHOLD = 0.10  # very strong


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _fmt_odds(value: int | float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    v = int(value)
    return f"+{v}" if v > 0 else str(v)


def _fmt_line(value: float | int | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    numeric = float(value)
    return f"{numeric:.0f}" if numeric.is_integer() else f"{numeric:.1f}"


def _safe_float(mapping: Mapping[str, Any], key: str, default: float | None = None) -> float | None:
    """Extract a float from *mapping* returning *default* on miss or NaN."""
    raw = mapping.get(key)
    if raw is None:
        return default
    try:
        val = float(raw)
        return default if math.isnan(val) else val
    except (TypeError, ValueError):
        return default


def _team_label(side: str, matchup: str) -> str:
    """Return the team abbreviation for *side* extracted from 'AWAY @ HOME'."""
    parts = matchup.split(" @ ")
    if len(parts) != 2:
        return side.upper()
    return parts[1].strip() if side == "home" else parts[0].strip()


def _opponent_side(side: str) -> str:
    return "away" if side == "home" else "home"


def _market_label(market_type: str) -> str:
    labels = {
        "f5_ml": "First 5 Moneyline",
        "f5_rl": "First 5 Run Line",
        "f5_total": "First 5 Total",
        "full_game_ml": "Full Game Moneyline",
        "full_game_rl": "Full Game Run Line",
        "full_game_total": "Full Game Total",
    }
    return labels.get(market_type, market_type.upper())


def _selection_label(
    *,
    matchup: str,
    side: str,
    market_type: str,
    line_at_bet: float | None,
) -> str:
    if market_type in {"f5_total", "full_game_total"}:
        direction = "Under" if side == "under" else "Over"
        return f"{direction} {_fmt_line(line_at_bet)}" if line_at_bet is not None else direction
    if side in {"home", "away"}:
        team_name = _team_label(side, matchup)
        if market_type in {"f5_rl", "full_game_rl"} and line_at_bet is not None:
            sign = "+" if line_at_bet > 0 else ""
            return f"{team_name} {sign}{line_at_bet:g}"
        return team_name
    return side.upper()


# ---------------------------------------------------------------------------
# Feature-based insight extractors
# ---------------------------------------------------------------------------


def _offense_insight(
    features: Mapping[str, Any],
    side: str,
    team_name: str,
    window: int = 14,
) -> str | None:
    """Describe notable recent offensive performance."""
    wrc = _safe_float(features, f"{side}_team_wrc_plus_{window}g")
    if wrc is None:
        return None
    if wrc >= _HOT_WRC_PLUS_THRESHOLD:
        return f"{team_name} offense has been scorching hot (wRC+ {wrc:.0f} over the last {window} games, league avg is 100)"
    if wrc <= _COLD_WRC_PLUS_THRESHOLD:
        return f"{team_name} bats have gone cold (wRC+ {wrc:.0f} over the last {window} games)"
    return None


def _pitching_insight(
    features: Mapping[str, Any],
    side: str,
    team_name: str,
    window: int = 30,
) -> str | None:
    """Describe the starting pitcher quality for one side."""
    xfip = _safe_float(features, f"{side}_starter_xfip_{window}g")
    k_pct = _safe_float(features, f"{side}_starter_k_pct_{window}g")
    if xfip is None:
        return None
    parts: list[str] = []
    if xfip <= _STRONG_XFIP_THRESHOLD:
        parts.append(f"{team_name} starter has been dominant ({xfip:.2f} xFIP over {window} games)")
    elif xfip >= _WEAK_XFIP_THRESHOLD:
        parts.append(
            f"{team_name} starter has been struggling ({xfip:.2f} xFIP over {window} games)"
        )
    if k_pct is not None and k_pct >= _HIGH_K_PCT_THRESHOLD and not parts:
        parts.append(f"{team_name} starter generating strikeouts ({k_pct:.1f}% K rate)")
    return parts[0] if parts else None


def _pitching_matchup_insight(
    features: Mapping[str, Any],
    pick_side: str,
    pick_team: str,
    opp_team: str,
    window: int = 30,
) -> str | None:
    """Compare the two starters and describe the mismatch if notable."""
    pick_xfip = _safe_float(features, f"{pick_side}_starter_xfip_{window}g")
    opp_side = _opponent_side(pick_side)
    opp_xfip = _safe_float(features, f"{opp_side}_starter_xfip_{window}g")
    if pick_xfip is None or opp_xfip is None:
        return None
    diff = opp_xfip - pick_xfip
    if diff >= 0.80:
        return (
            f"Clear pitching mismatch — {pick_team} starter ({pick_xfip:.2f} xFIP) "
            f"vs {opp_team} starter ({opp_xfip:.2f} xFIP)"
        )
    if diff >= 0.45:
        return f"Pitching edge favors {pick_team} ({pick_xfip:.2f} xFIP vs {opp_xfip:.2f})"
    return None


def _bullpen_insight(
    features: Mapping[str, Any],
    side: str,
    team_name: str,
) -> str | None:
    """Call out bullpen fatigue or strength."""
    rest = _safe_float(features, f"{side}_bullpen_avg_rest_days_5d")
    xfip = _safe_float(features, f"{side}_bullpen_top5_xfip_3d")
    if rest is not None and rest <= 1.2:
        return f"{team_name} bullpen looks taxed (avg {rest:.1f} rest days over last 5 games)"
    if xfip is not None and xfip <= 3.0:
        return f"{team_name} bullpen has been elite recently ({xfip:.2f} xFIP over 3 games)"
    return None


def _baseline_insight(
    features: Mapping[str, Any],
    pick_side: str,
    pick_team: str,
    opp_team: str,
) -> str | None:
    """Highlight Pythagorean win-rate gap if large."""
    pick_pyth = _safe_float(features, f"{pick_side}_pyth_win_pct_60g")
    opp_pyth = _safe_float(features, f"{_opponent_side(pick_side)}_pyth_win_pct_60g")
    if pick_pyth is not None and opp_pyth is not None:
        gap = pick_pyth - opp_pyth
        if gap >= 0.08:
            return (
                f"{pick_team} running a {pick_pyth:.0%} Pythagorean win rate over 60 games "
                f"vs {opp_team}'s {opp_pyth:.0%}"
            )
    return None


def _defense_insight(
    features: Mapping[str, Any],
    side: str,
    team_name: str,
    window: int = 30,
) -> str | None:
    """Mention standout defensive metrics (OAA / DRS)."""
    oaa = _safe_float(features, f"{side}_oaa_{window}g")
    drs = _safe_float(features, f"{side}_drs_{window}g")
    if oaa is not None and oaa >= 8:
        return f"{team_name} defense ranked among the best (OAA {oaa:.0f} over {window} games)"
    if drs is not None and drs >= 10:
        return f"{team_name} defense has been strong (DRS +{drs:.0f} over {window} games)"
    return None


def _weather_park_insight(features: Mapping[str, Any]) -> str | None:
    """Call out notable park or weather effects."""
    park = _safe_float(features, "park_factor")
    weather = _safe_float(features, "weather_adjustment")
    parts: list[str] = []
    if park is not None and park >= 1.08:
        parts.append("hitter-friendly park")
    elif park is not None and park <= 0.92:
        parts.append("pitcher-friendly park")
    if weather is not None and abs(weather - 1.0) > 0.04:
        if weather > 1.0:
            parts.append("warm conditions boosting run environment")
        else:
            parts.append("cold/wind conditions suppressing runs")
    if parts:
        return "Environment: " + ", ".join(parts) + "."
    return None


# ---------------------------------------------------------------------------
# Core narrative builders
# ---------------------------------------------------------------------------


def _edge_sentence(
    edge_pct: float,
    model_prob: float,
    market_prob: float | None,
    matchup: str,
    side: str,
    market_type: str,
    line_at_bet: float | None,
    odds: int | float | None,
) -> str:
    """Opening sentence describing the edge and recommended bet."""
    selection_label = _selection_label(
        matchup=matchup,
        side=side,
        market_type=market_type,
        line_at_bet=line_at_bet,
    )
    market_label = _market_label(market_type)
    edge_descriptor = "strong" if edge_pct >= _DOMINANT_EDGE_THRESHOLD else "solid"
    sentence = (
        f"{edge_descriptor.capitalize()} {_pct(edge_pct)} edge on {selection_label} "
        f"in the {market_label} at {_fmt_odds(odds)}."
    )
    if market_prob is not None:
        sentence += (
            f" Model sees a {_pct(model_prob)} probability "
            f"vs the market's implied {_pct(market_prob)}."
        )
    return sentence


def _projection_sentence(
    prediction: Mapping[str, Any],
    matchup: str,
    market_type: str,
    line_at_bet: float | None,
) -> str | None:
    """Sentence about projected score or total for the current market."""
    parts = matchup.split(" @ ")
    if len(parts) == 2:
        away, home = parts[0].strip(), parts[1].strip()
    else:
        away, home = "Away", "Home"

    if market_type in {"f5_ml", "f5_rl"}:
        home_runs = _safe_float(prediction, "projected_f5_home_runs")
        away_runs = _safe_float(prediction, "projected_f5_away_runs")
        if home_runs is None or away_runs is None:
            return None
        return f"Model projects a {away_runs:.1f}–{home_runs:.1f} F5 score ({away}–{home})."

    if market_type in {"full_game_ml", "full_game_rl"}:
        home_runs = _safe_float(prediction, "projected_full_game_home_runs")
        away_runs = _safe_float(prediction, "projected_full_game_away_runs")
        if home_runs is None or away_runs is None:
            return None
        return f"Model projects a {away_runs:.1f}–{home_runs:.1f} full-game score ({away}–{home})."

    if market_type == "f5_total":
        total_runs = _safe_float(prediction, "projected_f5_total_runs")
        if total_runs is None:
            return None
        if line_at_bet is None:
            return f"Model projects {total_runs:.1f} total F5 runs."
        direction = "above" if total_runs >= line_at_bet else "below"
        return f"Model projects {total_runs:.1f} total F5 runs, {direction} the posted {_fmt_line(line_at_bet)}."

    if market_type == "full_game_total":
        total_runs = _safe_float(prediction, "projected_full_game_total_runs")
        if total_runs is None:
            return None
        if line_at_bet is None:
            return f"Model projects {total_runs:.1f} full-game runs."
        direction = "above" if total_runs >= line_at_bet else "below"
        return f"Model projects {total_runs:.1f} full-game runs, {direction} the posted {_fmt_line(line_at_bet)}."

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_pick_narrative(
    *,
    matchup: str,
    side: str,
    market_type: str,
    odds: int | float | None,
    line_at_bet: float | None,
    edge_pct: float,
    model_probability: float,
    fair_probability: float | None,
    prediction: Mapping[str, Any],
    features: Mapping[str, Any],
    is_play_of_day: bool = False,
) -> str:
    """Return a human-readable narrative for a pick.

    *is_play_of_day* controls verbosity — True yields a multi-sentence
    write-up, False yields a concise one-liner.
    """
    side_is_team = side in {"home", "away"}
    pick_team = _team_label(side, matchup) if side_is_team else _team_label("home", matchup)
    opp_team = _team_label(_opponent_side(side), matchup) if side_is_team else _team_label("away", matchup)

    # Always build the edge sentence first
    edge_sent = _edge_sentence(
        edge_pct=edge_pct,
        model_prob=model_probability,
        market_prob=fair_probability,
        matchup=matchup,
        side=side,
        market_type=market_type,
        line_at_bet=line_at_bet,
        odds=odds,
    )

    if not is_play_of_day:
        # --- Value bet one-liner ---
        # Grab the single most impactful insight for colour
        if market_type in {"f5_total", "full_game_total"}:
            insight_functions = (
                lambda: _weather_park_insight(features),
                lambda: _offense_insight(features, "home", _team_label("home", matchup)),
                lambda: _offense_insight(features, "away", _team_label("away", matchup)),
                lambda: _pitching_insight(features, "home", _team_label("home", matchup)),
                lambda: _pitching_insight(features, "away", _team_label("away", matchup)),
            )
        else:
            insight_functions = (
                lambda: _pitching_matchup_insight(features, side, pick_team, opp_team),
                lambda: _offense_insight(features, side, pick_team),
                lambda: _pitching_insight(features, _opponent_side(side), opp_team),
                lambda: _baseline_insight(features, side, pick_team, opp_team),
            )
        for insight_fn in insight_functions:
            insight = insight_fn()
            if insight:
                return f"{edge_sent} {insight}."
        return edge_sent

    # --- POTD detailed narrative ---
    sentences: list[str] = [edge_sent]

    score_sent = _projection_sentence(prediction, matchup, market_type, line_at_bet)
    if score_sent:
        sentences.append(score_sent)

    # Collect up to 4 supporting insights, ordered by relevance
    if market_type in {"f5_total", "full_game_total"}:
        insight_candidates: list[str | None] = [
            _offense_insight(features, "home", _team_label("home", matchup)),
            _offense_insight(features, "away", _team_label("away", matchup)),
            _pitching_insight(features, "home", _team_label("home", matchup)),
            _pitching_insight(features, "away", _team_label("away", matchup)),
            _bullpen_insight(features, "home", _team_label("home", matchup)),
            _bullpen_insight(features, "away", _team_label("away", matchup)),
            _weather_park_insight(features),
        ]
    else:
        insight_candidates = [
            _pitching_matchup_insight(features, side, pick_team, opp_team),
            _offense_insight(features, side, pick_team),
            _offense_insight(features, _opponent_side(side), opp_team),
            _pitching_insight(features, side, pick_team),
            _pitching_insight(features, _opponent_side(side), opp_team),
            _bullpen_insight(features, _opponent_side(side), opp_team),
            _defense_insight(features, side, pick_team),
            _baseline_insight(features, side, pick_team, opp_team),
            _weather_park_insight(features),
        ]
    added = 0
    for candidate in insight_candidates:
        if candidate and added < 4:
            sentences.append(candidate + ("." if not candidate.endswith(".") else ""))
            added += 1

    return " ".join(sentences)


def generate_no_pick_narrative(
    *,
    matchup: str,
    no_pick_reason: str | None,
) -> str | None:
    """Optional brief note for skipped games — returns ``None`` for most."""
    if not no_pick_reason:
        return None
    return f"No play on {matchup}: {no_pick_reason}."


def generate_game_narrative(
    *,
    matchup: str,
    prediction: Mapping[str, Any] | None,
    decision: Mapping[str, Any] | None,
    features: Mapping[str, Any],
    is_play_of_day: bool = False,
    no_pick_reason: str | None = None,
) -> str | None:
    """Top-level helper: returns a narrative or ``None`` when no pick exists."""
    if decision is None:
        return generate_no_pick_narrative(matchup=matchup, no_pick_reason=no_pick_reason)

    return generate_pick_narrative(
        matchup=matchup,
        side=str(decision.get("side", "")),
        market_type=str(decision.get("market_type", "")),
        odds=decision.get("odds_at_bet"),
        line_at_bet=_safe_float(decision, "line_at_bet"),
        edge_pct=float(decision.get("edge_pct", 0)),
        model_probability=float(decision.get("model_probability", 0.5)),
        fair_probability=_safe_float(decision, "fair_probability"),
        prediction=prediction or {},
        features=features,
        is_play_of_day=is_play_of_day,
    )
