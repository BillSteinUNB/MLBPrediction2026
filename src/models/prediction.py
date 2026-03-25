from __future__ import annotations

from src.models._base import ModelBase, Probability, UtcDatetime


class Prediction(ModelBase):
    """Model output probabilities for a single MLB game."""

    game_pk: int
    model_version: str
    full_game_ml_home_prob: Probability | None = None
    full_game_ml_away_prob: Probability | None = None
    full_game_rl_home_prob: Probability | None = None
    full_game_rl_away_prob: Probability | None = None
    full_game_total_over_prob: Probability | None = None
    full_game_total_under_prob: Probability | None = None
    f5_ml_home_prob: Probability
    f5_ml_away_prob: Probability
    f5_rl_home_prob: Probability
    f5_rl_away_prob: Probability
    f5_total_over_prob: Probability | None = None
    f5_total_under_prob: Probability | None = None
    projected_full_game_home_runs: float | None = None
    projected_full_game_away_runs: float | None = None
    projected_full_game_total_runs: float | None = None
    projected_full_game_home_margin: float | None = None
    projected_f5_home_runs: float | None = None
    projected_f5_away_runs: float | None = None
    projected_f5_total_runs: float | None = None
    projected_f5_home_margin: float | None = None
    predicted_at: UtcDatetime
