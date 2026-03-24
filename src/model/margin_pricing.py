from __future__ import annotations

import math


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0)))


def margin_to_cover_probability(
    *,
    predicted_margin: float,
    home_point: float | None,
    residual_std: float,
) -> float | None:
    if home_point is None or residual_std <= 0:
        return None

    threshold = -float(home_point)
    z_score = (threshold - float(predicted_margin)) / float(residual_std)
    probability = 1.0 - normal_cdf(z_score)
    return max(0.0, min(1.0, float(probability)))
