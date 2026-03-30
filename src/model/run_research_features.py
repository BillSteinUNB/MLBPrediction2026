from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.clients.historical_odds_client import load_historical_odds_for_games
from src.clients.odds_client import devig_probabilities
from src.features.adjustments.abs_adjustment import build_abs_challenge_proxy_context
from src.features.marcel_blend import blend_value


DEFAULT_MARKET_FULL_GAME_TOTAL_BASELINE = 8.8
DEFAULT_MARKET_F5_TOTAL_BASELINE = 4.6
DEFAULT_MARKET_AWAY_RUN_BASELINE = 4.4
DEFAULT_MARKET_PRIOR_BOOK_NAME: str | None = None

DEFAULT_UMPIRE_HOME_WIN_BASELINE = 0.54
DEFAULT_UMPIRE_TOTAL_RUNS_BASELINE = 8.8
DEFAULT_UMPIRE_F5_TOTAL_RUNS_BASELINE = 4.5

DEFAULT_HOME_DEFENSIVE_EFFICIENCY = 0.700
DEFAULT_HOME_OAA = 0.0
DEFAULT_HOME_DRS = 0.0
DEFAULT_HOME_FRAMING = 0.0


@dataclass(frozen=True, slots=True)
class MarketPriorMetadata:
    enabled: bool
    source_name: str
    feature_columns: tuple[str, ...]
    rows_with_market_data: int
    coverage_pct: float
    fallback_reason: str | None
    notes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ResearchFeatureMetadata:
    feature_families: Mapping[str, tuple[str, ...]]
    market_priors: MarketPriorMetadata


@dataclass(frozen=True, slots=True)
class ResearchFeatureAugmentationResult:
    dataframe: pd.DataFrame
    metadata: ResearchFeatureMetadata


def augment_run_research_features(
    dataframe: pd.DataFrame,
    *,
    enable_market_priors: bool = False,
    historical_odds_db_path: str | Path | None = None,
    historical_market_book_name: str | None = DEFAULT_MARKET_PRIOR_BOOK_NAME,
) -> ResearchFeatureAugmentationResult:
    """Attach advanced research-lane features without mutating the control lane inputs."""

    augmented = dataframe.copy()
    market_columns, market_metadata = _attach_market_prior_features(
        augmented,
        enable_market_priors=enable_market_priors,
        historical_odds_db_path=historical_odds_db_path,
        historical_market_book_name=historical_market_book_name,
    )
    family_columns: dict[str, tuple[str, ...]] = {
        "market_priors": market_columns,
        "ttop": _add_ttop_features(augmented),
        "pitch_archetype": _add_pitch_archetype_mismatch_features(augmented),
        "defense_extension": _add_defense_extension_features(augmented),
        "bip_baserunning": _add_bip_baserunning_features(augmented),
        "weather_interactions": _add_handedness_aware_weather_features(augmented),
        "umpire_micro": _add_umpire_micro_features(augmented),
        "framing_micro": _add_framing_micro_features(augmented),
        "abs_regime": _add_abs_regime_features(augmented),
    }
    metadata = ResearchFeatureMetadata(
        feature_families=dict(family_columns),
        market_priors=market_metadata,
    )
    return ResearchFeatureAugmentationResult(dataframe=augmented, metadata=metadata)


def metadata_to_dict(metadata: ResearchFeatureMetadata) -> dict[str, Any]:
    payload = asdict(metadata)
    payload["market_priors"]["feature_columns"] = list(metadata.market_priors.feature_columns)
    payload["market_priors"]["notes"] = list(metadata.market_priors.notes)
    payload["feature_families"] = {
        key: list(value) for key, value in metadata.feature_families.items()
    }
    return payload


def _attach_market_prior_features(
    dataframe: pd.DataFrame,
    *,
    enable_market_priors: bool,
    historical_odds_db_path: str | Path | None,
    historical_market_book_name: str | None,
) -> tuple[tuple[str, ...], MarketPriorMetadata]:
    feature_columns = (
        "market_priors_available",
        "market_f5_home_fair_prob",
        "market_f5_away_fair_prob",
        "market_f5_home_implied_prob",
        "market_f5_away_implied_prob",
        "market_f5_rl_home_point",
        "market_f5_rl_away_point",
        "market_f5_total_line",
        "market_f5_total_over_fair_prob",
        "market_f5_total_under_fair_prob",
        "market_f5_total_over_implied_prob",
        "market_f5_total_under_implied_prob",
        "market_full_game_home_fair_prob",
        "market_full_game_away_fair_prob",
        "market_full_game_home_implied_prob",
        "market_full_game_away_implied_prob",
        "market_full_game_rl_home_point",
        "market_full_game_rl_away_point",
        "market_full_game_total_line",
        "market_full_game_total_over_fair_prob",
        "market_full_game_total_under_fair_prob",
        "market_full_game_total_over_implied_prob",
        "market_full_game_total_under_implied_prob",
        "market_full_game_away_team_total_line",
        "market_full_game_away_team_total_over_fair_prob",
        "market_full_game_away_team_total_under_fair_prob",
        "market_full_game_away_team_total_over_implied_prob",
        "market_full_game_away_team_total_under_implied_prob",
        "market_full_game_away_team_total_available",
        "market_implied_f5_away_runs",
        "market_implied_full_game_away_runs",
        "market_away_run_share_anchor",
        "market_run_environment_anchor",
        "market_anchor_confidence",
    )
    for column, default in (
        ("market_priors_available", 0.0),
        ("market_f5_home_fair_prob", 0.5),
        ("market_f5_away_fair_prob", 0.5),
        ("market_f5_home_implied_prob", 0.5),
        ("market_f5_away_implied_prob", 0.5),
        ("market_f5_rl_home_point", 0.0),
        ("market_f5_rl_away_point", 0.0),
        ("market_f5_total_line", DEFAULT_MARKET_F5_TOTAL_BASELINE),
        ("market_f5_total_over_fair_prob", 0.5),
        ("market_f5_total_under_fair_prob", 0.5),
        ("market_f5_total_over_implied_prob", 0.5),
        ("market_f5_total_under_implied_prob", 0.5),
        ("market_full_game_home_fair_prob", 0.5),
        ("market_full_game_away_fair_prob", 0.5),
        ("market_full_game_home_implied_prob", 0.5),
        ("market_full_game_away_implied_prob", 0.5),
        ("market_full_game_rl_home_point", 0.0),
        ("market_full_game_rl_away_point", 0.0),
        ("market_full_game_total_line", DEFAULT_MARKET_FULL_GAME_TOTAL_BASELINE),
        ("market_full_game_total_over_fair_prob", 0.5),
        ("market_full_game_total_under_fair_prob", 0.5),
        ("market_full_game_total_over_implied_prob", 0.5),
        ("market_full_game_total_under_implied_prob", 0.5),
        ("market_full_game_away_team_total_line", DEFAULT_MARKET_AWAY_RUN_BASELINE),
        ("market_full_game_away_team_total_over_fair_prob", 0.5),
        ("market_full_game_away_team_total_under_fair_prob", 0.5),
        ("market_full_game_away_team_total_over_implied_prob", 0.5),
        ("market_full_game_away_team_total_under_implied_prob", 0.5),
        ("market_full_game_away_team_total_available", 0.0),
        ("market_implied_f5_away_runs", DEFAULT_MARKET_F5_TOTAL_BASELINE / 2.0),
        ("market_implied_full_game_away_runs", DEFAULT_MARKET_AWAY_RUN_BASELINE),
        ("market_away_run_share_anchor", 0.5),
        ("market_run_environment_anchor", 1.0),
        ("market_anchor_confidence", 0.0),
    ):
        dataframe[column] = float(default)

    if not enable_market_priors:
        return feature_columns, MarketPriorMetadata(
            enabled=False,
            source_name="historical_f5_market_fallback",
            feature_columns=feature_columns,
            rows_with_market_data=0,
            coverage_pct=0.0,
            fallback_reason="market priors disabled for this lane",
            notes=(
                "Stage 3 and Stage 4 can opt into market priors without touching the control lane.",
            ),
        )

    if historical_odds_db_path is None or dataframe.empty or "game_pk" not in dataframe.columns:
        return feature_columns, MarketPriorMetadata(
            enabled=True,
            source_name="historical_f5_market_fallback",
            feature_columns=feature_columns,
            rows_with_market_data=0,
            coverage_pct=0.0,
            fallback_reason="historical odds database path was not supplied",
            notes=(
                "Historical away team totals are not available in the current repo-local data.",
                "The fallback uses neutral market anchors so the lane wiring stays auditable.",
            ),
        )

    market_lookup_kwargs = {
        "db_path": historical_odds_db_path,
        "games_frame": dataframe,
        "book_name": historical_market_book_name,
        "snapshot_selection": "opening",
    }
    f5_ml_market = load_historical_odds_for_games(market_type="f5_ml", **market_lookup_kwargs)
    f5_rl_market = load_historical_odds_for_games(market_type="f5_rl", **market_lookup_kwargs)
    f5_total_market = load_historical_odds_for_games(market_type="f5_total", **market_lookup_kwargs)
    full_ml_market = load_historical_odds_for_games(
        market_type="full_game_ml",
        **market_lookup_kwargs,
    )
    full_rl_market = load_historical_odds_for_games(
        market_type="full_game_rl",
        **market_lookup_kwargs,
    )
    full_total_market = load_historical_odds_for_games(
        market_type="full_game_total",
        **market_lookup_kwargs,
    )
    full_away_team_total_market = load_historical_odds_for_games(
        market_type="full_game_team_total_away",
        **market_lookup_kwargs,
    )

    loaded_markets = (
        f5_ml_market,
        f5_rl_market,
        f5_total_market,
        full_ml_market,
        full_rl_market,
        full_total_market,
        full_away_team_total_market,
    )
    if all(frame.empty for frame in loaded_markets):
        return feature_columns, MarketPriorMetadata(
            enabled=True,
            source_name="historical_market_fallback",
            feature_columns=feature_columns,
            rows_with_market_data=0,
            coverage_pct=0.0,
            fallback_reason="no matching historical market rows were found for the supplied game ids",
            notes=(
                "Current local odds storage contains no matching historical market rows for the supplied away-run frame.",
                "The fallback keeps the market-prior lane plumbed while honestly marking zero coverage.",
            ),
        )

    merged = dataframe[["game_pk"]].copy()
    market_sources = sorted(
        {
            str(frame["source_schema"].iloc[0])
            for frame in loaded_markets
            if not frame.empty and "source_schema" in frame.columns
        }
    )

    if not f5_ml_market.empty:
        f5_ml_market = f5_ml_market.rename(
            columns={
                "home_odds": "market_f5_home_odds",
                "away_odds": "market_f5_away_odds",
            }
        )
        merged = merged.merge(
            f5_ml_market[["game_pk", "market_f5_home_odds", "market_f5_away_odds"]],
            on="game_pk",
            how="left",
        )
    else:
        merged["market_f5_home_odds"] = np.nan
        merged["market_f5_away_odds"] = np.nan

    if not f5_rl_market.empty:
        f5_rl_market = f5_rl_market.rename(
            columns={
                "home_point": "market_f5_rl_home_point",
                "away_point": "market_f5_rl_away_point",
            }
        )
        merged = merged.merge(
            f5_rl_market[["game_pk", "market_f5_rl_home_point", "market_f5_rl_away_point"]],
            on="game_pk",
            how="left",
        )
    else:
        merged["market_f5_rl_home_point"] = np.nan
        merged["market_f5_rl_away_point"] = np.nan

    if not f5_total_market.empty:
        f5_total_market = f5_total_market.rename(
            columns={
                "total_point": "market_f5_total_line",
                "over_odds": "market_f5_total_over_odds",
                "under_odds": "market_f5_total_under_odds",
            }
        )
        merged = merged.merge(
            f5_total_market[
                [
                    "game_pk",
                    "market_f5_total_line",
                    "market_f5_total_over_odds",
                    "market_f5_total_under_odds",
                ]
            ],
            on="game_pk",
            how="left",
        )
    else:
        merged["market_f5_total_line"] = np.nan
        merged["market_f5_total_over_odds"] = np.nan
        merged["market_f5_total_under_odds"] = np.nan

    if not full_ml_market.empty:
        full_ml_market = full_ml_market.rename(
            columns={
                "home_odds": "market_full_game_home_odds",
                "away_odds": "market_full_game_away_odds",
            }
        )
        merged = merged.merge(
            full_ml_market[
                [
                    "game_pk",
                    "market_full_game_home_odds",
                    "market_full_game_away_odds",
                ]
            ],
            on="game_pk",
            how="left",
        )
    else:
        merged["market_full_game_home_odds"] = np.nan
        merged["market_full_game_away_odds"] = np.nan

    if not full_rl_market.empty:
        full_rl_market = full_rl_market.rename(
            columns={
                "home_point": "market_full_game_rl_home_point",
                "away_point": "market_full_game_rl_away_point",
            }
        )
        merged = merged.merge(
            full_rl_market[
                [
                    "game_pk",
                    "market_full_game_rl_home_point",
                    "market_full_game_rl_away_point",
                ]
            ],
            on="game_pk",
            how="left",
        )
    else:
        merged["market_full_game_rl_home_point"] = np.nan
        merged["market_full_game_rl_away_point"] = np.nan

    if not full_total_market.empty:
        full_total_market = full_total_market.rename(
            columns={
                "total_point": "market_full_game_total_line",
                "over_odds": "market_full_game_total_over_odds",
                "under_odds": "market_full_game_total_under_odds",
            }
        )
        merged = merged.merge(
            full_total_market[
                [
                    "game_pk",
                    "market_full_game_total_line",
                    "market_full_game_total_over_odds",
                    "market_full_game_total_under_odds",
                ]
            ],
            on="game_pk",
            how="left",
        )
    else:
        merged["market_full_game_total_line"] = np.nan
        merged["market_full_game_total_over_odds"] = np.nan
        merged["market_full_game_total_under_odds"] = np.nan

    if not full_away_team_total_market.empty:
        full_away_team_total_market = full_away_team_total_market.rename(
            columns={
                "total_point": "market_full_game_away_team_total_line",
                "over_odds": "market_full_game_away_team_total_over_odds",
                "under_odds": "market_full_game_away_team_total_under_odds",
            }
        )
        merged = merged.merge(
            full_away_team_total_market[
                [
                    "game_pk",
                    "market_full_game_away_team_total_line",
                    "market_full_game_away_team_total_over_odds",
                    "market_full_game_away_team_total_under_odds",
                ]
            ],
            on="game_pk",
            how="left",
        )
    else:
        merged["market_full_game_away_team_total_line"] = np.nan
        merged["market_full_game_away_team_total_over_odds"] = np.nan
        merged["market_full_game_away_team_total_under_odds"] = np.nan

    f5_fair_probabilities = merged.apply(_safe_devig_row, axis=1, result_type="expand")
    f5_fair_probabilities.columns = [
        "market_f5_home_fair_prob",
        "market_f5_away_fair_prob",
        "market_f5_ml_available",
    ]
    merged = pd.concat([merged, f5_fair_probabilities], axis=1)
    full_game_fair_probabilities = merged.apply(
        lambda row: _safe_devig_pair(
            row.get("market_full_game_home_odds"),
            row.get("market_full_game_away_odds"),
        ),
        axis=1,
        result_type="expand",
    )
    full_game_fair_probabilities.columns = [
        "market_full_game_home_fair_prob",
        "market_full_game_away_fair_prob",
        "market_full_game_ml_available",
    ]
    merged = pd.concat([merged, full_game_fair_probabilities], axis=1)
    f5_total_fair_probabilities = merged.apply(
        lambda row: _safe_devig_pair(
            row.get("market_f5_total_over_odds"),
            row.get("market_f5_total_under_odds"),
        ),
        axis=1,
        result_type="expand",
    )
    f5_total_fair_probabilities.columns = [
        "market_f5_total_over_fair_prob",
        "market_f5_total_under_fair_prob",
        "market_f5_total_available",
    ]
    merged = pd.concat([merged, f5_total_fair_probabilities], axis=1)
    full_total_fair_probabilities = merged.apply(
        lambda row: _safe_devig_pair(
            row.get("market_full_game_total_over_odds"),
            row.get("market_full_game_total_under_odds"),
        ),
        axis=1,
        result_type="expand",
    )
    full_total_fair_probabilities.columns = [
        "market_full_game_total_over_fair_prob",
        "market_full_game_total_under_fair_prob",
        "market_full_game_total_available",
    ]
    merged = pd.concat([merged, full_total_fair_probabilities], axis=1)
    away_team_total_fair_probabilities = merged.apply(
        lambda row: _safe_devig_pair(
            row.get("market_full_game_away_team_total_over_odds"),
            row.get("market_full_game_away_team_total_under_odds"),
        ),
        axis=1,
        result_type="expand",
    )
    away_team_total_fair_probabilities.columns = [
        "market_full_game_away_team_total_over_fair_prob",
        "market_full_game_away_team_total_under_fair_prob",
        "market_full_game_away_team_total_available",
    ]
    merged = pd.concat([merged, away_team_total_fair_probabilities], axis=1)

    merged["market_f5_home_implied_prob"] = merged["market_f5_home_odds"].map(_american_to_implied_or_default)
    merged["market_f5_away_implied_prob"] = merged["market_f5_away_odds"].map(_american_to_implied_or_default)
    merged["market_f5_total_over_implied_prob"] = merged["market_f5_total_over_odds"].map(
        _american_to_implied_or_default
    )
    merged["market_f5_total_under_implied_prob"] = merged["market_f5_total_under_odds"].map(
        _american_to_implied_or_default
    )
    merged["market_full_game_home_implied_prob"] = merged["market_full_game_home_odds"].map(
        _american_to_implied_or_default
    )
    merged["market_full_game_away_implied_prob"] = merged["market_full_game_away_odds"].map(
        _american_to_implied_or_default
    )
    merged["market_full_game_total_over_implied_prob"] = merged[
        "market_full_game_total_over_odds"
    ].map(_american_to_implied_or_default)
    merged["market_full_game_total_under_implied_prob"] = merged[
        "market_full_game_total_under_odds"
    ].map(_american_to_implied_or_default)
    merged["market_full_game_away_team_total_over_implied_prob"] = merged[
        "market_full_game_away_team_total_over_odds"
    ].map(_american_to_implied_or_default)
    merged["market_full_game_away_team_total_under_implied_prob"] = merged[
        "market_full_game_away_team_total_under_odds"
    ].map(_american_to_implied_or_default)
    merged["market_f5_rl_home_point"] = pd.to_numeric(merged["market_f5_rl_home_point"], errors="coerce").fillna(0.0)
    merged["market_f5_rl_away_point"] = pd.to_numeric(merged["market_f5_rl_away_point"], errors="coerce").fillna(0.0)
    merged["market_full_game_rl_home_point"] = pd.to_numeric(
        merged["market_full_game_rl_home_point"],
        errors="coerce",
    ).fillna(0.0)
    merged["market_full_game_rl_away_point"] = pd.to_numeric(
        merged["market_full_game_rl_away_point"],
        errors="coerce",
    ).fillna(0.0)
    merged["market_f5_total_line"] = pd.to_numeric(merged["market_f5_total_line"], errors="coerce").fillna(
        DEFAULT_MARKET_F5_TOTAL_BASELINE
    )
    merged["market_full_game_total_line"] = pd.to_numeric(
        merged["market_full_game_total_line"],
        errors="coerce",
    ).fillna(DEFAULT_MARKET_FULL_GAME_TOTAL_BASELINE)
    merged["market_full_game_away_team_total_line"] = pd.to_numeric(
        merged["market_full_game_away_team_total_line"],
        errors="coerce",
    ).fillna(DEFAULT_MARKET_AWAY_RUN_BASELINE)
    merged["market_priors_available"] = (
        merged[
            [
                "market_f5_ml_available",
                "market_full_game_ml_available",
                "market_f5_total_available",
                "market_full_game_total_available",
                "market_full_game_away_team_total_available",
            ]
        ]
        .max(axis=1)
        .astype(float)
    )
    merged["market_anchor_confidence"] = (
        0.24 * merged["market_f5_ml_available"]
        + 0.12 * merged["market_f5_rl_home_point"].abs().gt(0).astype(float)
        + 0.20 * merged["market_f5_total_available"]
        + 0.16 * merged["market_full_game_ml_available"]
        + 0.08 * merged["market_full_game_rl_home_point"].abs().gt(0).astype(float)
        + 0.20 * merged["market_full_game_total_available"]
        + 0.24 * merged["market_full_game_away_team_total_available"]
    ).clip(0.0, 1.0)
    merged["market_away_run_share_anchor"] = (
        0.50
        + (merged["market_f5_away_fair_prob"] - 0.50) * 0.28
        + (merged["market_full_game_away_fair_prob"] - 0.50) * 0.22
        + (-merged["market_f5_rl_away_point"].clip(-1.5, 1.5)) * 0.05
        + (-merged["market_full_game_rl_away_point"].clip(-2.5, 2.5)) * 0.03
    ).clip(0.28, 0.72)
    f5_total_price_pressure = (
        merged["market_f5_total_over_fair_prob"] - merged["market_f5_total_under_fair_prob"]
    ).clip(-0.18, 0.18)
    full_total_price_pressure = (
        merged["market_full_game_total_over_fair_prob"]
        - merged["market_full_game_total_under_fair_prob"]
    ).clip(-0.18, 0.18)
    away_team_total_price_pressure = (
        merged["market_full_game_away_team_total_over_fair_prob"]
        - merged["market_full_game_away_team_total_under_fair_prob"]
    ).clip(-0.18, 0.18)
    merged["market_implied_f5_away_runs"] = (
        merged["market_f5_total_line"] * merged["market_away_run_share_anchor"]
        + ((-merged["market_f5_rl_away_point"]).clip(-1.5, 1.5) * 0.18)
        + (f5_total_price_pressure * 0.60)
    ).clip(0.8, 4.8)
    merged["market_implied_full_game_away_runs"] = np.where(
        merged["market_full_game_away_team_total_available"] > 0.0,
        (
            merged["market_full_game_away_team_total_line"]
            + (away_team_total_price_pressure * 0.60)
        ),
        np.where(
            merged["market_full_game_total_available"] > 0.0,
        (
            merged["market_full_game_total_line"] * merged["market_away_run_share_anchor"]
            + ((-merged["market_full_game_rl_away_point"]).clip(-2.5, 2.5) * 0.24)
            + (full_total_price_pressure * 0.90)
        ),
            merged["market_implied_f5_away_runs"] * 1.82,
        ),
    ).clip(1.2, 8.6)
    merged["market_run_environment_anchor"] = (
        merged["market_implied_full_game_away_runs"] / DEFAULT_MARKET_AWAY_RUN_BASELINE
    ).clip(0.65, 1.45)

    coverage = float(merged["market_priors_available"].mean()) if len(merged) > 0 else 0.0
    for column in feature_columns:
        dataframe[column] = merged[column].to_numpy(dtype=float)

    return feature_columns, MarketPriorMetadata(
        enabled=True,
        source_name=_market_source_name_from_schemas(market_sources),
        feature_columns=feature_columns,
        rows_with_market_data=int(merged["market_priors_available"].sum()),
        coverage_pct=coverage,
        fallback_reason=(
            None
            if coverage > 0.0
            else "historical market lookup resolved but did not produce usable rows"
        ),
        notes=(
            (
                "Market priors use F5 and full-game totals plus ML/RL, and prefer direct away-team totals when present."
            ),
            (
                "Away-run anchors fall back to derived totals-and-side-market estimates only when direct away-team-total history is unavailable."
            ),
            (
                f"Historical market book selection: {historical_market_book_name or 'consensus across available books'}."
            ),
        ),
    )


def _add_ttop_features(dataframe: pd.DataFrame) -> tuple[str, ...]:
    last_start_pitch_count = _series(dataframe, "home_starter_last_start_pitch_count", 92.0)
    cumulative_pitch_load = _series(dataframe, "home_starter_cumulative_pitch_load_5s", 450.0)
    bullpen_pitch_count = _series(dataframe, "home_team_bullpen_pitch_count_3d", 45.0)
    away_lineup_bb_pct = _series(dataframe, "away_lineup_bb_pct_30g", 8.2)
    away_lineup_k_pct = _series(dataframe, "away_lineup_k_pct_30g", 22.8)
    away_lineup_woba = _series(dataframe, "away_lineup_woba_30g", 0.315)
    away_lineup_delta = _series(dataframe, "away_lineup_woba_delta_7v30g", 0.0)
    platoon_advantage = _series(dataframe, "away_lineup_platoon_advantage_pct", 0.50)

    starter_stamina = (
        0.45 * ((last_start_pitch_count - 92.0) / 12.0)
        - 0.25 * ((cumulative_pitch_load - 450.0) / 90.0)
        - 0.20 * ((bullpen_pitch_count - 45.0) / 18.0)
    )
    lineup_pressure = (
        0.35 * ((away_lineup_bb_pct - 8.2) / 2.0)
        + 0.25 * ((22.8 - away_lineup_k_pct) / 4.0)
        + 0.25 * ((away_lineup_woba - 0.315) / 0.020)
        + 0.15 * ((platoon_advantage - 0.50) / 0.20)
    )
    exposure = _sigmoid(starter_stamina + lineup_pressure)
    penalty = (
        exposure
        * (1.0 + ((away_lineup_delta.clip(-0.04, 0.04) / 0.04) * 0.25))
        * (1.0 + ((away_lineup_woba - 0.315).clip(-0.030, 0.030) / 0.030) * 0.20)
    ).clip(0.15, 1.35)
    fatigue_escape_hatch = (
        1.0
        - ((bullpen_pitch_count - 45.0).clip(-25.0, 35.0) / 70.0)
    ).clip(0.55, 1.35)

    dataframe["away_matchup_ttop_exposure_index"] = exposure.astype(float)
    dataframe["away_matchup_ttop_penalty_index"] = penalty.astype(float)
    dataframe["home_starter_fatigue_escape_hatch"] = fatigue_escape_hatch.astype(float)
    return (
        "away_matchup_ttop_exposure_index",
        "away_matchup_ttop_penalty_index",
        "home_starter_fatigue_escape_hatch",
    )


def _add_pitch_archetype_mismatch_features(dataframe: pd.DataFrame) -> tuple[str, ...]:
    starter_k_pct = _series(dataframe, "home_starter_k_pct_30s", 22.3)
    starter_bb_pct = _series(dataframe, "home_starter_bb_pct_30s", 8.0)
    starter_gb_pct = _series(dataframe, "home_starter_gb_pct_30s", 43.0)
    starter_csw_pct = _series(dataframe, "home_starter_csw_pct_30s", 28.0)
    starter_velocity = _series(dataframe, "home_starter_avg_fastball_velocity_30s", 93.0)
    starter_entropy = _series(dataframe, "home_starter_pitch_mix_entropy_30s", 2.3)

    lineup_iso = _series(dataframe, "away_lineup_iso_30g", 0.160)
    lineup_barrel = _series(dataframe, "away_lineup_barrel_pct_30g", 7.2)
    lineup_k_pct = _series(dataframe, "away_lineup_k_pct_30g", 22.8)
    lineup_bb_pct = _series(dataframe, "away_lineup_bb_pct_30g", 8.2)
    lineup_platoon = _series(dataframe, "away_lineup_platoon_advantage_pct", 0.50)
    lineup_contact_gap = _series(dataframe, "away_lineup_woba_minus_xwoba_30g", 0.0)

    power_score = (
        0.35 * ((starter_k_pct - 22.3) / 4.0)
        + 0.30 * ((starter_csw_pct - 28.0) / 3.0)
        + 0.20 * ((starter_velocity - 93.0) / 2.0)
        + 0.15 * ((starter_entropy - 2.3) / 0.35)
    )
    contact_manager_score = (
        0.45 * ((starter_gb_pct - 43.0) / 6.0)
        + 0.30 * ((8.0 - starter_bb_pct) / 2.0)
        + 0.25 * ((2.3 - starter_entropy) / 0.35)
    )
    mismatch_power = (
        0.55 * ((lineup_iso - 0.160) / 0.030)
        + 0.45 * ((lineup_barrel - 7.2) / 2.0)
        - 0.40 * power_score
        + 0.25 * ((43.0 - starter_gb_pct) / 6.0)
    ).clip(-2.0, 2.0)
    mismatch_contact = (
        ((starter_k_pct - lineup_k_pct) / 6.0)
        - 0.35 * contact_manager_score
        + 0.20 * ((lineup_bb_pct - starter_bb_pct) / 2.0)
    ).clip(-2.0, 2.0)
    mismatch_platoon = (
        ((lineup_platoon - 0.50) / 0.20)
        + 0.20 * ((starter_velocity - 93.0) / 2.0)
        + 0.20 * ((starter_entropy - 2.3) / 0.35)
    ).clip(-2.0, 2.0)
    pitch_mix_surprise = (
        ((starter_entropy - 2.3) / 0.35)
        - 0.35 * (lineup_contact_gap / 0.020)
    ).clip(-2.0, 2.0)

    dataframe["home_starter_power_archetype_score"] = power_score.astype(float)
    dataframe["home_starter_contact_manager_score"] = contact_manager_score.astype(float)
    dataframe["away_matchup_power_archetype_mismatch"] = mismatch_power.astype(float)
    dataframe["away_matchup_contact_archetype_mismatch"] = mismatch_contact.astype(float)
    dataframe["away_matchup_platoon_archetype_edge"] = mismatch_platoon.astype(float)
    dataframe["away_matchup_pitch_mix_surprise_gap"] = pitch_mix_surprise.astype(float)
    return (
        "home_starter_power_archetype_score",
        "home_starter_contact_manager_score",
        "away_matchup_power_archetype_mismatch",
        "away_matchup_contact_archetype_mismatch",
        "away_matchup_platoon_archetype_edge",
        "away_matchup_pitch_mix_surprise_gap",
    )


def _add_defense_extension_features(dataframe: pd.DataFrame) -> tuple[str, ...]:
    oaa_30 = _series(dataframe, "home_team_oaa_30g", DEFAULT_HOME_OAA)
    oaa_season = _series(dataframe, "home_team_oaa_season", DEFAULT_HOME_OAA)
    drs_30 = _series(dataframe, "home_team_drs_30g", DEFAULT_HOME_DRS)
    drs_season = _series(dataframe, "home_team_drs_season", DEFAULT_HOME_DRS)
    def_eff_30 = _series(
        dataframe,
        "home_team_defensive_efficiency_30g",
        DEFAULT_HOME_DEFENSIVE_EFFICIENCY,
    )
    def_eff_season = _series(
        dataframe,
        "home_team_defensive_efficiency_season",
        DEFAULT_HOME_DEFENSIVE_EFFICIENCY,
    )
    framing_30 = _series(dataframe, "home_team_adjusted_framing_30g", DEFAULT_HOME_FRAMING)
    framing_season = _series(dataframe, "home_team_adjusted_framing_season", DEFAULT_HOME_FRAMING)
    away_babip = _series(dataframe, "away_lineup_babip_30g", 0.295)
    away_barrel = _series(dataframe, "away_lineup_barrel_pct_30g", 7.2)
    away_iso = _series(dataframe, "away_lineup_iso_30g", 0.160)

    oaa_blend = _blend_defense_series(oaa_30, oaa_season, 0.0)
    drs_blend = _blend_defense_series(drs_30, drs_season, 0.0)
    def_eff_blend = _blend_defense_series(
        def_eff_30,
        def_eff_season,
        DEFAULT_HOME_DEFENSIVE_EFFICIENCY,
    )
    framing_blend = _blend_defense_series(framing_30, framing_season, DEFAULT_HOME_FRAMING)

    range_index = (
        0.32 * (oaa_blend / 4.0)
        + 0.28 * (drs_blend / 6.0)
        + 0.25 * ((def_eff_blend - DEFAULT_HOME_DEFENSIVE_EFFICIENCY) / 0.025)
        + 0.15 * (framing_blend / 6.0)
    ).clip(-1.8, 1.8)
    range_trend = (
        ((oaa_30 - oaa_season) / 4.0)
        + ((drs_30 - drs_season) / 6.0)
        + ((def_eff_30 - def_eff_season) / 0.020)
    ).clip(-2.0, 2.0)
    bip_gap = (
        0.45 * ((away_babip - 0.295) / 0.025)
        + 0.30 * ((away_barrel - 7.2) / 2.0)
        + 0.25 * ((away_iso - 0.160) / 0.030)
        - 0.55 * range_index
    ).clip(-2.0, 2.0)

    dataframe["home_team_range_defense_index"] = range_index.astype(float)
    dataframe["home_team_range_trend_30vseason"] = range_trend.astype(float)
    dataframe["away_bip_quality_vs_home_range_gap"] = bip_gap.astype(float)
    return (
        "home_team_range_defense_index",
        "home_team_range_trend_30vseason",
        "away_bip_quality_vs_home_range_gap",
    )


def _add_bip_baserunning_features(dataframe: pd.DataFrame) -> tuple[str, ...]:
    away_babip = _series(dataframe, "away_lineup_babip_30g", 0.295)
    away_xwoba = _series(dataframe, "away_lineup_xwoba_30g", 0.315)
    away_iso = _series(dataframe, "away_lineup_iso_30g", 0.160)
    away_barrel = _series(dataframe, "away_lineup_barrel_pct_30g", 7.2)
    platoon = _series(dataframe, "away_lineup_platoon_advantage_pct", 0.50)
    range_index = _series(dataframe, "home_team_range_defense_index", 0.0)

    hit_prob = (
        0.235
        + 0.55 * (away_babip - 0.295)
        + 0.40 * (away_xwoba - 0.315)
        - 0.025 * range_index
    ).clip(0.16, 0.42)
    extra_base_prob = (
        0.155
        + 0.60 * (away_iso - 0.160)
        + 0.012 * (away_barrel - 7.2)
        - 0.020 * range_index
    ).clip(0.08, 0.44)
    advancement_prob = (
        0.32
        + 0.35 * (platoon - 0.50)
        + 0.30 * (hit_prob - 0.235)
        + 0.20 * (extra_base_prob - 0.155)
    ).clip(0.18, 0.68)

    dataframe["away_bip_hit_probability_proxy"] = hit_prob.astype(float)
    dataframe["away_bip_extra_base_probability_proxy"] = extra_base_prob.astype(float)
    dataframe["away_bip_advancement_probability_proxy"] = advancement_prob.astype(float)
    return (
        "away_bip_hit_probability_proxy",
        "away_bip_extra_base_probability_proxy",
        "away_bip_advancement_probability_proxy",
    )


def _add_handedness_aware_weather_features(dataframe: pd.DataFrame) -> tuple[str, ...]:
    wind_factor = _series(dataframe, "weather_wind_factor", 0.0)
    air_density = _series(dataframe, "weather_air_density_factor", 1.0)
    weather_composite = _series(dataframe, "weather_composite", 1.0)
    away_lhb_pct = _series(dataframe, "away_lineup_lhb_pct", 0.33)
    away_rhb_pct = _series(dataframe, "away_lineup_rhb_pct", 0.56)
    away_platoon = _series(dataframe, "away_lineup_platoon_advantage_pct", 0.50)
    away_barrel = _series(dataframe, "away_lineup_barrel_pct_30g", 7.2)
    away_iso = _series(dataframe, "away_lineup_iso_30g", 0.160)

    wind_lhb = (wind_factor * away_lhb_pct).clip(-12.0, 12.0)
    wind_rhb = (wind_factor * away_rhb_pct).clip(-12.0, 12.0)
    handedness_gap = (wind_factor * (away_lhb_pct - away_rhb_pct)).clip(-12.0, 12.0)
    power_boost = (
        wind_factor
        * (away_barrel / 7.2)
        * (away_iso / 0.160)
        * (0.80 + 0.40 * away_platoon)
    ).clip(-8.0, 8.0)
    air_density_power = (
        (air_density - 1.0)
        * (away_barrel / 7.2)
        * (weather_composite / 1.0)
    ).clip(-0.40, 0.40)

    dataframe["away_weather_wind_lhb_interaction"] = wind_lhb.astype(float)
    dataframe["away_weather_wind_rhb_interaction"] = wind_rhb.astype(float)
    dataframe["away_weather_handedness_wind_gap"] = handedness_gap.astype(float)
    dataframe["away_weather_power_wind_interaction"] = power_boost.astype(float)
    dataframe["away_weather_air_density_power_interaction"] = air_density_power.astype(float)
    return (
        "away_weather_wind_lhb_interaction",
        "away_weather_wind_rhb_interaction",
        "away_weather_handedness_wind_gap",
        "away_weather_power_wind_interaction",
        "away_weather_air_density_power_interaction",
    )


def _add_umpire_micro_features(dataframe: pd.DataFrame) -> tuple[str, ...]:
    sample_30 = _series(dataframe, "plate_umpire_sample_size_30g", 0.0)
    total_30 = _series(dataframe, "plate_umpire_total_runs_avg_30g", DEFAULT_UMPIRE_TOTAL_RUNS_BASELINE)
    total_90 = _series(dataframe, "plate_umpire_total_runs_avg_90g", DEFAULT_UMPIRE_TOTAL_RUNS_BASELINE)
    f5_30 = _series(dataframe, "plate_umpire_f5_total_runs_avg_30g", DEFAULT_UMPIRE_F5_TOTAL_RUNS_BASELINE)
    f5_90 = _series(dataframe, "plate_umpire_f5_total_runs_avg_90g", DEFAULT_UMPIRE_F5_TOTAL_RUNS_BASELINE)
    home_win_30 = _series(dataframe, "plate_umpire_home_win_pct_30g", DEFAULT_UMPIRE_HOME_WIN_BASELINE)
    home_win_90 = _series(dataframe, "plate_umpire_home_win_pct_90g", DEFAULT_UMPIRE_HOME_WIN_BASELINE)
    abs_share_30 = _series(dataframe, "plate_umpire_abs_active_share_30g", 0.0)
    abs_total_30 = _series(dataframe, "plate_umpire_abs_total_runs_avg_30g", DEFAULT_UMPIRE_TOTAL_RUNS_BASELINE)
    abs_f5_30 = _series(dataframe, "plate_umpire_abs_f5_total_runs_avg_30g", DEFAULT_UMPIRE_F5_TOTAL_RUNS_BASELINE)

    total_blend = _blend_context_series(
        total_30,
        prior=total_90,
        league_average=DEFAULT_UMPIRE_TOTAL_RUNS_BASELINE,
        games_played=sample_30,
    )
    f5_blend = _blend_context_series(
        f5_30,
        prior=f5_90,
        league_average=DEFAULT_UMPIRE_F5_TOTAL_RUNS_BASELINE,
        games_played=sample_30,
    )
    home_win_blend = _blend_context_series(
        home_win_30,
        prior=home_win_90,
        league_average=DEFAULT_UMPIRE_HOME_WIN_BASELINE,
        games_played=sample_30,
    )

    zone_suppression = (
        ((DEFAULT_UMPIRE_TOTAL_RUNS_BASELINE - total_blend) / 1.4)
        + ((DEFAULT_UMPIRE_F5_TOTAL_RUNS_BASELINE - f5_blend) / 0.7)
    ).clip(-2.0, 2.0)
    abs_zone_suppression = (
        ((DEFAULT_UMPIRE_TOTAL_RUNS_BASELINE - abs_total_30) / 1.2)
        + ((DEFAULT_UMPIRE_F5_TOTAL_RUNS_BASELINE - abs_f5_30) / 0.6)
    ).clip(-2.0, 2.0)
    zone_suppression = (
        0.60 * zone_suppression
        + 0.40 * (abs_zone_suppression * (0.35 + 0.65 * abs_share_30))
    ).clip(-2.0, 2.0)
    home_bias = ((home_win_blend - DEFAULT_UMPIRE_HOME_WIN_BASELINE) / 0.06).clip(-1.5, 1.5)
    volatility = (
        abs(total_30 - total_90)
        + abs(f5_30 - f5_90)
        + 0.70 * abs(abs_total_30 - total_30)
        + 1.10 * abs(abs_f5_30 - f5_30)
    ).clip(0.0, 4.0)

    dataframe["plate_umpire_zone_suppression_index"] = zone_suppression.astype(float)
    dataframe["plate_umpire_home_edge_bias_index"] = home_bias.astype(float)
    dataframe["plate_umpire_zone_volatility_index"] = volatility.astype(float)
    return (
        "plate_umpire_zone_suppression_index",
        "plate_umpire_home_edge_bias_index",
        "plate_umpire_zone_volatility_index",
    )


def _add_framing_micro_features(dataframe: pd.DataFrame) -> tuple[str, ...]:
    framing_30 = _series(dataframe, "home_team_adjusted_framing_30g", DEFAULT_HOME_FRAMING)
    framing_60 = _series(dataframe, "home_team_adjusted_framing_60g", DEFAULT_HOME_FRAMING)
    framing_season = _series(dataframe, "home_team_adjusted_framing_season", DEFAULT_HOME_FRAMING)
    retention_30 = _series(dataframe, "home_team_framing_retention_proxy_30g", 1.0)
    retention_season = _series(dataframe, "home_team_framing_retention_proxy_season", 1.0)
    zone_suppression = _series(dataframe, "plate_umpire_zone_suppression_index", 0.0)

    framing_blend = _blend_defense_series(framing_30, framing_season, DEFAULT_HOME_FRAMING)
    retention_blend = _blend_defense_series(retention_30, retention_season, 1.0)
    framing_trend = ((framing_30 - framing_season) / 6.0).clip(-2.0, 2.0)
    stability = (1.0 - (abs(framing_30 - framing_60) / 8.0)).clip(0.0, 1.0)
    zone_support = (
        framing_blend / 6.0
        + 0.35 * zone_suppression
        + 0.20 * ((retention_blend - 0.75) / 0.25)
    ).clip(-2.0, 2.0)

    dataframe["home_team_framing_trend_30vseason"] = framing_trend.astype(float)
    dataframe["home_team_framing_stability_index"] = stability.astype(float)
    dataframe["home_team_framing_zone_support_index"] = zone_support.astype(float)
    return (
        "home_team_framing_trend_30vseason",
        "home_team_framing_stability_index",
        "home_team_framing_zone_support_index",
    )


def _add_abs_regime_features(dataframe: pd.DataFrame) -> tuple[str, ...]:
    feature_columns = (
        "abs_challenge_opportunity_proxy",
        "abs_expected_challenge_pressure_proxy",
        "abs_challenge_conservation_proxy",
        "abs_leverage_framing_retention_proxy",
        "abs_umpire_zone_suppression_proxy",
    )
    if dataframe.empty:
        for column in feature_columns:
            dataframe[column] = pd.Series(dtype=float)
        return feature_columns

    abs_active = _series_any(dataframe, ("abs_active", "is_abs_active"), 1.0).gt(0.5)
    challenge_rows = [
        build_abs_challenge_proxy_context(
            abs_active=bool(active),
            lineup_walk_rate=float(lineup_walk_rate),
            lineup_strikeout_rate=float(lineup_strikeout_rate),
            lineup_quality=float(lineup_quality),
            abs_walk_rate_delta=float(abs_walk_delta),
            abs_strikeout_rate_delta=float(abs_strikeout_delta),
            umpire_zone_suppression=float(zone_suppression),
            umpire_zone_volatility=float(zone_volatility),
            umpire_abs_active_share=float(umpire_abs_share),
            framing_retention_proxy=float(framing_retention),
            framing_stability=float(framing_stability),
            framing_zone_support=float(framing_zone_support),
            run_environment_anchor=float(run_environment_anchor),
            market_anchor_confidence=float(market_anchor_confidence),
        )
        for (
            active,
            lineup_walk_rate,
            lineup_strikeout_rate,
            lineup_quality,
            abs_walk_delta,
            abs_strikeout_delta,
            zone_suppression,
            zone_volatility,
            umpire_abs_share,
            framing_retention,
            framing_stability,
            framing_zone_support,
            run_environment_anchor,
            market_anchor_confidence,
        ) in zip(
            abs_active,
            _series(dataframe, "away_lineup_bb_pct_30g", 8.2),
            _series(dataframe, "away_lineup_k_pct_30g", 22.8),
            _series(dataframe, "away_lineup_woba_30g", 0.315),
            _series(dataframe, "abs_walk_rate_delta", 0.0),
            _series(dataframe, "abs_strikeout_rate_delta", 0.0),
            _series(dataframe, "plate_umpire_zone_suppression_index", 0.0),
            _series(dataframe, "plate_umpire_zone_volatility_index", 0.0),
            _series(dataframe, "plate_umpire_abs_active_share_30g", 0.5),
            _series(dataframe, "home_team_framing_retention_proxy_30g", 1.0),
            _series(dataframe, "home_team_framing_stability_index", 0.75),
            _series(dataframe, "home_team_framing_zone_support_index", 0.0),
            _series(dataframe, "market_run_environment_anchor", 1.0),
            _series(dataframe, "market_anchor_confidence", 0.0),
            strict=False,
        )
    ]
    dataframe["abs_challenge_opportunity_proxy"] = pd.Series(
        [result.challenge_opportunity_proxy for result in challenge_rows],
        index=dataframe.index,
        dtype=float,
    )
    dataframe["abs_expected_challenge_pressure_proxy"] = pd.Series(
        [result.challenge_pressure_proxy for result in challenge_rows],
        index=dataframe.index,
        dtype=float,
    )
    dataframe["abs_challenge_conservation_proxy"] = pd.Series(
        [result.challenge_conservation_proxy for result in challenge_rows],
        index=dataframe.index,
        dtype=float,
    )
    dataframe["abs_leverage_framing_retention_proxy"] = pd.Series(
        [result.leverage_framing_retention_proxy for result in challenge_rows],
        index=dataframe.index,
        dtype=float,
    )
    dataframe["abs_umpire_zone_suppression_proxy"] = pd.Series(
        [result.umpire_zone_suppression_proxy for result in challenge_rows],
        index=dataframe.index,
        dtype=float,
    )
    return feature_columns


def _blend_defense_series(
    current: pd.Series,
    prior: pd.Series,
    league_average: float,
) -> pd.Series:
    return pd.Series(
        [
            blend_value(
                float(current_value),
                games_played=30,
                metric_type="defense",
                prior_value=float(prior_value),
                league_average=float(league_average),
            )
            for current_value, prior_value in zip(current, prior, strict=False)
        ],
        index=current.index,
        dtype=float,
    )


def _blend_context_series(
    current: pd.Series,
    *,
    prior: pd.Series,
    league_average: float,
    games_played: pd.Series,
) -> pd.Series:
    return pd.Series(
        [
            blend_value(
                float(current_value),
                games_played=max(float(sample_size), 0.0),
                metric_type="pitching",
                prior_value=float(prior_value),
                league_average=float(league_average),
            )
            for current_value, prior_value, sample_size in zip(current, prior, games_played, strict=False)
        ],
        index=current.index,
        dtype=float,
    )


def _safe_devig_row(row: pd.Series) -> tuple[float, float, float]:
    return _safe_devig_pair(row.get("market_f5_home_odds"), row.get("market_f5_away_odds"))


def _safe_devig_pair(left_value: Any, right_value: Any) -> tuple[float, float, float]:
    left_odds = _nullable_int(left_value)
    right_odds = _nullable_int(right_value)
    if left_odds is None or right_odds is None:
        return 0.5, 0.5, 0.0
    try:
        left_fair, right_fair = devig_probabilities(left_odds, right_odds)
    except (TypeError, ValueError):
        return 0.5, 0.5, 0.0
    return float(left_fair), float(right_fair), 1.0


def _series(dataframe: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column not in dataframe.columns:
        return pd.Series(np.full(len(dataframe), float(default), dtype=float), index=dataframe.index)
    return pd.to_numeric(dataframe[column], errors="coerce").fillna(float(default)).astype(float)


def _series_any(dataframe: pd.DataFrame, columns: Sequence[str], default: float) -> pd.Series:
    for column in columns:
        if column in dataframe.columns:
            return _series(dataframe, column, default)
    return pd.Series(np.full(len(dataframe), float(default), dtype=float), index=dataframe.index)


def _american_to_implied_or_default(value: Any) -> float:
    resolved = _nullable_int(value)
    if resolved is None:
        return 0.5
    absolute = abs(resolved)
    if absolute == 0:
        return 0.5
    if resolved > 0:
        return float(100.0 / (resolved + 100.0))
    return float(absolute / (absolute + 100.0))


def _nullable_int(value: Any) -> int | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return int(numeric)


def _sigmoid(series: pd.Series) -> pd.Series:
    values = np.asarray(series, dtype=float)
    return pd.Series(1.0 / (1.0 + np.exp(-values)), index=series.index, dtype=float)


def _market_source_name_from_schemas(source_schemas: Sequence[str]) -> str:
    resolved = tuple(sorted({str(schema).strip() for schema in source_schemas if str(schema).strip()}))
    if resolved == ("canonical",):
        return "historical_market_archive_canonical"
    if resolved == ("old_scraper",):
        return "historical_market_archive_old_scraper"
    if resolved:
        return "historical_market_archive_hybrid"
    return "historical_market_archive_unknown"


__all__ = [
    "DEFAULT_MARKET_PRIOR_BOOK_NAME",
    "MarketPriorMetadata",
    "ResearchFeatureAugmentationResult",
    "ResearchFeatureMetadata",
    "augment_run_research_features",
    "metadata_to_dict",
]
