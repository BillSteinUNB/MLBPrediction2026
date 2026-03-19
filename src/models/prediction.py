from __future__ import annotations

from src.models._base import ModelBase, Probability, UtcDatetime


class Prediction(ModelBase):
    """Model output probabilities for a single MLB game."""

    game_pk: int
    model_version: str
    f5_ml_home_prob: Probability
    f5_ml_away_prob: Probability
    f5_rl_home_prob: Probability
    f5_rl_away_prob: Probability
    predicted_at: UtcDatetime
