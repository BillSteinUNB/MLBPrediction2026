from __future__ import annotations


def shrink_probability_toward_market(
    *,
    model_probability: float,
    fair_probability: float,
    odds: int,
    base_multiplier: float = 1.0,
    plus_money_multiplier: float = 1.0,
    high_edge_threshold: float = 0.10,
    high_edge_multiplier: float = 1.0,
) -> float:
    """Shrink a candidate probability toward the de-vigged market probability.

    Multipliers are applied to the model-vs-market deviation. Values below 1.0
    reduce model conviction; values above 1.0 are not allowed.
    """

    resolved_base = _clamp_multiplier(base_multiplier)
    resolved_plus_money = _clamp_multiplier(plus_money_multiplier)
    resolved_high_edge = _clamp_multiplier(high_edge_multiplier)
    resolved_threshold = max(0.0, float(high_edge_threshold))

    model_probability = float(model_probability)
    fair_probability = float(fair_probability)
    edge = model_probability - fair_probability

    multiplier = resolved_base
    if int(odds) > 0:
        multiplier *= resolved_plus_money
    if abs(edge) >= resolved_threshold:
        multiplier *= resolved_high_edge

    adjusted = fair_probability + (edge * multiplier)
    return max(0.0, min(1.0, float(adjusted)))


def _clamp_multiplier(value: float) -> float:
    resolved = float(value)
    if resolved < 0.0:
        return 0.0
    if resolved > 1.0:
        return 1.0
    return resolved

