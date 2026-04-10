from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from src.clients.eva_analytics_client import EVA_STATCAST_CORRELATIONS_URL
from src.clients.statcast_client import fetch_statcast_range
from src.db import DEFAULT_DB_PATH, sqlite_connection


DEFAULT_EVA_STATCAST_AUDIT_DIR = Path("data/reports/run_count/eva_statcast")
_FINAL_GAME_STATES = {"final", "game over", "completed early"}


@dataclass(frozen=True, slots=True)
class _MetricSpec:
    metric_key: str
    label: str
    family: str
    existing_repo_feature: bool
    eva_stat: str | None = None


_METRIC_SPECS: tuple[_MetricSpec, ...] = (
    _MetricSpec(
        metric_key="avg_exit_velocity",
        label="Average Exit Velocity",
        family="exit_velocity",
        existing_repo_feature=False,
        eva_stat="EXIT VELOCITY (AVERAGE)",
    ),
    _MetricSpec(
        metric_key="air_100plus_pct",
        label="100+ mph In-Air Rate",
        family="exit_velocity",
        existing_repo_feature=False,
        eva_stat="EXIT VELOCITY (100+ MPH IN AIR%)",
    ),
    _MetricSpec(
        metric_key="hard_hit_pct",
        label="Hard-Hit Rate",
        family="exit_velocity",
        existing_repo_feature=False,
        eva_stat=None,
    ),
    _MetricSpec(
        metric_key="launch_angle_neg4_26_pct",
        label="Launch Angle -4 to 26%",
        family="launch_angle_shape",
        existing_repo_feature=False,
        eva_stat="LAUNCH ANGLE (-4 TO 26%)",
    ),
    _MetricSpec(
        metric_key="launch_angle_23_34_pct",
        label="Launch Angle 23 to 34%",
        family="launch_angle_shape",
        existing_repo_feature=False,
        eva_stat="LAUNCH ANGLE (23 TO 34%)",
    ),
    _MetricSpec(
        metric_key="launch_angle_38_plus_pct",
        label="Launch Angle 38+%",
        family="launch_angle_shape",
        existing_repo_feature=False,
        eva_stat="LAUNCH ANGLE (38+%)",
    ),
    _MetricSpec(
        metric_key="launch_angle_stddev",
        label="Launch Angle StdDev",
        family="launch_angle_shape",
        existing_repo_feature=False,
        eva_stat="LAUNCH ANGLE (STANDARD DEVIATION)",
    ),
    _MetricSpec(
        metric_key="avg_launch_angle",
        label="Average Launch Angle",
        family="launch_angle_shape",
        existing_repo_feature=False,
        eva_stat="LAUNCH ANGLE (AVERAGE)",
    ),
    _MetricSpec(
        metric_key="barrel_pct",
        label="Barrel Rate",
        family="contact_quality_benchmark",
        existing_repo_feature=True,
        eva_stat="BARREL%",
    ),
    _MetricSpec(
        metric_key="xwoba",
        label="xwOBA",
        family="contact_quality_benchmark",
        existing_repo_feature=True,
        eva_stat=None,
    ),
)
_ROLLING_COUNT_COLUMNS: tuple[str, ...] = (
    "pa_count",
    "xwoba_sum",
    "xwoba_count",
    "bbe_count",
    "barrels_count",
    "launch_speed_sum",
    "launch_speed_count",
    "hard_hit_count",
    "air_bbe_count",
    "air_100plus_count",
    "launch_angle_sum",
    "launch_angle_sq_sum",
    "launch_angle_count",
    "launch_angle_neg4_26_count",
    "launch_angle_23_34_count",
    "launch_angle_38_plus_count",
)


def build_eva_statcast_team_game_audit_frame(
    *,
    season: int,
    db_path: str | Path = DEFAULT_DB_PATH,
    rolling_window: int = 30,
    statcast_frame: pd.DataFrame | None = None,
    statcast_fetcher=fetch_statcast_range,
) -> pd.DataFrame:
    """Build a team-game audit frame with anti-leakage-safe rolling Statcast metrics."""

    outcomes = _load_team_game_outcomes(season=season, db_path=db_path)
    if outcomes.empty:
        return pd.DataFrame()

    if statcast_frame is None:
        min_day = outcomes["game_date"].min()
        max_day = outcomes["game_date"].max()
        statcast_frame = statcast_fetcher(
            min_day.isoformat(),
            max_day.isoformat(),
            refresh=False,
        )
    team_game_statcast = _build_team_game_statcast_counts(statcast_frame)

    audit_frame = outcomes.merge(
        team_game_statcast,
        on=["game_pk", "team"],
        how="left",
        suffixes=("", "_statcast"),
    ).sort_values(["team", "game_date", "game_pk", "is_home"]).reset_index(drop=True)

    for column in _ROLLING_COUNT_COLUMNS:
        if column not in audit_frame.columns:
            audit_frame[column] = 0.0
        audit_frame[column] = pd.to_numeric(audit_frame[column], errors="coerce").fillna(0.0)
        rolling_column = f"rolling_{rolling_window}g_{column}"
        audit_frame[rolling_column] = audit_frame.groupby("team", sort=False)[column].transform(
            lambda values: values.shift(1).rolling(rolling_window, min_periods=1).sum()
        )

    metric_prefix = f"rolling_{rolling_window}g"
    audit_frame[f"{metric_prefix}_games_with_history"] = audit_frame.groupby(
        "team",
        sort=False,
    ).cumcount()
    audit_frame[f"{metric_prefix}_xwoba"] = _safe_divide(
        audit_frame[f"{metric_prefix}_xwoba_sum"],
        audit_frame[f"{metric_prefix}_xwoba_count"],
    )
    audit_frame[f"{metric_prefix}_barrel_pct"] = (
        _safe_divide(
            audit_frame[f"{metric_prefix}_barrels_count"],
            audit_frame[f"{metric_prefix}_bbe_count"],
        )
        * 100.0
    )
    audit_frame[f"{metric_prefix}_avg_exit_velocity"] = _safe_divide(
        audit_frame[f"{metric_prefix}_launch_speed_sum"],
        audit_frame[f"{metric_prefix}_launch_speed_count"],
    )
    audit_frame[f"{metric_prefix}_hard_hit_pct"] = (
        _safe_divide(
            audit_frame[f"{metric_prefix}_hard_hit_count"],
            audit_frame[f"{metric_prefix}_bbe_count"],
        )
        * 100.0
    )
    audit_frame[f"{metric_prefix}_air_100plus_pct"] = (
        _safe_divide(
            audit_frame[f"{metric_prefix}_air_100plus_count"],
            audit_frame[f"{metric_prefix}_air_bbe_count"],
        )
        * 100.0
    )
    audit_frame[f"{metric_prefix}_avg_launch_angle"] = _safe_divide(
        audit_frame[f"{metric_prefix}_launch_angle_sum"],
        audit_frame[f"{metric_prefix}_launch_angle_count"],
    )
    mean_square = _safe_divide(
        audit_frame[f"{metric_prefix}_launch_angle_sq_sum"],
        audit_frame[f"{metric_prefix}_launch_angle_count"],
    )
    variance = (mean_square - audit_frame[f"{metric_prefix}_avg_launch_angle"].pow(2)).clip(lower=0.0)
    audit_frame[f"{metric_prefix}_launch_angle_stddev"] = variance.pow(0.5)
    audit_frame[f"{metric_prefix}_launch_angle_neg4_26_pct"] = (
        _safe_divide(
            audit_frame[f"{metric_prefix}_launch_angle_neg4_26_count"],
            audit_frame[f"{metric_prefix}_bbe_count"],
        )
        * 100.0
    )
    audit_frame[f"{metric_prefix}_launch_angle_23_34_pct"] = (
        _safe_divide(
            audit_frame[f"{metric_prefix}_launch_angle_23_34_count"],
            audit_frame[f"{metric_prefix}_bbe_count"],
        )
        * 100.0
    )
    audit_frame[f"{metric_prefix}_launch_angle_38_plus_pct"] = (
        _safe_divide(
            audit_frame[f"{metric_prefix}_launch_angle_38_plus_count"],
            audit_frame[f"{metric_prefix}_bbe_count"],
        )
        * 100.0
    )
    return audit_frame


def build_eva_statcast_metric_summary(
    audit_frame: pd.DataFrame,
    *,
    rolling_window: int = 30,
    eva_correlations: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Summarize repo-local run correlations alongside EVA research guidance."""

    if audit_frame.empty:
        return pd.DataFrame()

    eva_lookup = _build_eva_lookup(eva_correlations)
    rows: list[dict[str, object]] = []
    for spec in _METRIC_SPECS:
        metric_column = f"rolling_{rolling_window}g_{spec.metric_key}"
        if metric_column not in audit_frame.columns:
            continue

        runs_scored_corr, runs_scored_n = _pearson_with_n(
            audit_frame[metric_column],
            audit_frame["runs_scored"],
        )
        f5_runs_scored_corr, f5_runs_scored_n = _pearson_with_n(
            audit_frame[metric_column],
            audit_frame["f5_runs_scored"],
        )
        eva_predictive = eva_lookup.get((spec.eva_stat, "predictive"))
        eva_descriptive = eva_lookup.get((spec.eva_stat, "descriptive"))
        rows.append(
            {
                "metric_key": spec.metric_key,
                "metric_label": spec.label,
                "family": spec.family,
                "existing_repo_feature": spec.existing_repo_feature,
                "rolling_window": int(rolling_window),
                "audit_column": metric_column,
                "sample_size_runs_scored": runs_scored_n,
                "sample_size_f5_runs_scored": f5_runs_scored_n,
                "repo_runs_scored_corr": runs_scored_corr,
                "repo_f5_runs_scored_corr": f5_runs_scored_corr,
                "eva_stat": spec.eva_stat,
                "eva_predictive_woba_corr": _lookup_eva_value(eva_predictive, "woba_corr"),
                "eva_predictive_hr_pct_corr": _lookup_eva_value(eva_predictive, "hr_pct_corr"),
                "eva_predictive_babip_corr": _lookup_eva_value(eva_predictive, "babip_corr"),
                "eva_predictive_iso_corr": _lookup_eva_value(eva_predictive, "iso_corr"),
                "eva_predictive_ba_corr": _lookup_eva_value(eva_predictive, "ba_corr"),
                "eva_descriptive_woba_corr": _lookup_eva_value(eva_descriptive, "woba_corr"),
                "eva_descriptive_hr_pct_corr": _lookup_eva_value(eva_descriptive, "hr_pct_corr"),
                "priority_score": _priority_score(
                    repo_runs_scored_corr=runs_scored_corr,
                    repo_f5_runs_scored_corr=f5_runs_scored_corr,
                    eva_predictive=eva_predictive,
                ),
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    return summary.sort_values(
        ["existing_repo_feature", "priority_score", "metric_label"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def build_eva_statcast_starter_start_audit_frame(
    *,
    season: int,
    db_path: str | Path = DEFAULT_DB_PATH,
    rolling_window: int = 5,
    statcast_frame: pd.DataFrame | None = None,
    statcast_fetcher=fetch_statcast_range,
) -> pd.DataFrame:
    """Build a starter-start audit frame with rolling contact-quality-allowed metrics."""

    outcomes = _load_team_game_outcomes(season=season, db_path=db_path)
    if outcomes.empty:
        return pd.DataFrame()

    start_rows = _load_starter_rows(season=season, db_path=db_path)
    if start_rows.empty:
        return pd.DataFrame()

    if statcast_frame is None:
        min_day = start_rows["game_date"].min()
        max_day = start_rows["game_date"].max()
        statcast_frame = statcast_fetcher(
            min_day.isoformat(),
            max_day.isoformat(),
            refresh=False,
        )

    starter_statcast = _build_starter_start_statcast_counts(start_rows, statcast_frame)
    audit_frame = start_rows.merge(
        outcomes[
            [
                "game_pk",
                "team",
                "runs_allowed",
                "f5_runs_allowed",
                "opponent",
                "is_home",
            ]
        ],
        on=["game_pk", "team"],
        how="left",
    ).merge(
        starter_statcast,
        on=["game_pk", "team", "pitcher_id"],
        how="left",
        suffixes=("", "_statcast"),
    )
    audit_frame = audit_frame.sort_values(
        ["pitcher_id", "game_date", "game_pk", "team"]
    ).reset_index(drop=True)

    for column in _ROLLING_COUNT_COLUMNS:
        if column not in audit_frame.columns:
            audit_frame[column] = 0.0
        audit_frame[column] = pd.to_numeric(audit_frame[column], errors="coerce").fillna(0.0)
        rolling_column = f"rolling_{rolling_window}s_{column}"
        audit_frame[rolling_column] = audit_frame.groupby("pitcher_id", sort=False)[column].transform(
            lambda values: values.shift(1).rolling(rolling_window, min_periods=1).sum()
        )

    metric_prefix = f"rolling_{rolling_window}s"
    audit_frame[f"{metric_prefix}_starts_with_history"] = audit_frame.groupby(
        "pitcher_id",
        sort=False,
    ).cumcount()
    audit_frame[f"{metric_prefix}_xwoba"] = _safe_divide(
        audit_frame[f"{metric_prefix}_xwoba_sum"],
        audit_frame[f"{metric_prefix}_xwoba_count"],
    )
    audit_frame[f"{metric_prefix}_barrel_pct"] = (
        _safe_divide(
            audit_frame[f"{metric_prefix}_barrels_count"],
            audit_frame[f"{metric_prefix}_bbe_count"],
        )
        * 100.0
    )
    audit_frame[f"{metric_prefix}_avg_exit_velocity"] = _safe_divide(
        audit_frame[f"{metric_prefix}_launch_speed_sum"],
        audit_frame[f"{metric_prefix}_launch_speed_count"],
    )
    audit_frame[f"{metric_prefix}_hard_hit_pct"] = (
        _safe_divide(
            audit_frame[f"{metric_prefix}_hard_hit_count"],
            audit_frame[f"{metric_prefix}_bbe_count"],
        )
        * 100.0
    )
    audit_frame[f"{metric_prefix}_air_100plus_pct"] = (
        _safe_divide(
            audit_frame[f"{metric_prefix}_air_100plus_count"],
            audit_frame[f"{metric_prefix}_air_bbe_count"],
        )
        * 100.0
    )
    audit_frame[f"{metric_prefix}_avg_launch_angle"] = _safe_divide(
        audit_frame[f"{metric_prefix}_launch_angle_sum"],
        audit_frame[f"{metric_prefix}_launch_angle_count"],
    )
    mean_square = _safe_divide(
        audit_frame[f"{metric_prefix}_launch_angle_sq_sum"],
        audit_frame[f"{metric_prefix}_launch_angle_count"],
    )
    variance = (mean_square - audit_frame[f"{metric_prefix}_avg_launch_angle"].pow(2)).clip(lower=0.0)
    audit_frame[f"{metric_prefix}_launch_angle_stddev"] = variance.pow(0.5)
    audit_frame[f"{metric_prefix}_launch_angle_neg4_26_pct"] = (
        _safe_divide(
            audit_frame[f"{metric_prefix}_launch_angle_neg4_26_count"],
            audit_frame[f"{metric_prefix}_bbe_count"],
        )
        * 100.0
    )
    audit_frame[f"{metric_prefix}_launch_angle_23_34_pct"] = (
        _safe_divide(
            audit_frame[f"{metric_prefix}_launch_angle_23_34_count"],
            audit_frame[f"{metric_prefix}_bbe_count"],
        )
        * 100.0
    )
    audit_frame[f"{metric_prefix}_launch_angle_38_plus_pct"] = (
        _safe_divide(
            audit_frame[f"{metric_prefix}_launch_angle_38_plus_count"],
            audit_frame[f"{metric_prefix}_bbe_count"],
        )
        * 100.0
    )
    return audit_frame


def build_eva_statcast_starter_metric_summary(
    audit_frame: pd.DataFrame,
    *,
    rolling_window: int = 5,
    eva_correlations: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Summarize starter contact-quality-allowed metrics against runs allowed."""

    if audit_frame.empty:
        return pd.DataFrame()

    eva_lookup = _build_eva_lookup(eva_correlations)
    rows: list[dict[str, object]] = []
    for spec in _METRIC_SPECS:
        metric_column = f"rolling_{rolling_window}s_{spec.metric_key}"
        if metric_column not in audit_frame.columns:
            continue

        runs_allowed_corr, runs_allowed_n = _pearson_with_n(
            audit_frame[metric_column],
            audit_frame["runs_allowed"],
        )
        f5_runs_allowed_corr, f5_runs_allowed_n = _pearson_with_n(
            audit_frame[metric_column],
            audit_frame["f5_runs_allowed"],
        )
        eva_predictive = eva_lookup.get((spec.eva_stat, "predictive"))
        eva_descriptive = eva_lookup.get((spec.eva_stat, "descriptive"))
        rows.append(
            {
                "metric_key": spec.metric_key,
                "metric_label": spec.label,
                "family": spec.family,
                "existing_repo_feature": spec.existing_repo_feature,
                "rolling_window": int(rolling_window),
                "audit_column": metric_column,
                "sample_size_runs_allowed": runs_allowed_n,
                "sample_size_f5_runs_allowed": f5_runs_allowed_n,
                "repo_runs_allowed_corr": runs_allowed_corr,
                "repo_f5_runs_allowed_corr": f5_runs_allowed_corr,
                "eva_stat": spec.eva_stat,
                "eva_predictive_woba_corr": _lookup_eva_value(eva_predictive, "woba_corr"),
                "eva_predictive_hr_pct_corr": _lookup_eva_value(eva_predictive, "hr_pct_corr"),
                "eva_predictive_babip_corr": _lookup_eva_value(eva_predictive, "babip_corr"),
                "eva_predictive_iso_corr": _lookup_eva_value(eva_predictive, "iso_corr"),
                "eva_predictive_ba_corr": _lookup_eva_value(eva_predictive, "ba_corr"),
                "eva_descriptive_woba_corr": _lookup_eva_value(eva_descriptive, "woba_corr"),
                "eva_descriptive_hr_pct_corr": _lookup_eva_value(eva_descriptive, "hr_pct_corr"),
                "priority_score": _priority_score(
                    repo_runs_scored_corr=runs_allowed_corr,
                    repo_f5_runs_scored_corr=f5_runs_allowed_corr,
                    eva_predictive=eva_predictive,
                ),
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    return summary.sort_values(
        ["existing_repo_feature", "priority_score", "metric_label"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def build_eva_statcast_audit_summary(
    audit_frame: pd.DataFrame,
    *,
    metric_summary: pd.DataFrame,
    rolling_window: int = 30,
    eva_correlations: pd.DataFrame | None = None,
    top_n: int = 5,
) -> dict[str, Any]:
    """Build a compact JSON-serializable summary for the EVA Statcast audit."""

    if audit_frame.empty:
        raise ValueError("audit_frame must not be empty")

    history_column = f"rolling_{rolling_window}g_bbe_count"
    team_count = int(audit_frame["team"].nunique())
    season = int(pd.to_numeric(audit_frame["season"], errors="coerce").dropna().iloc[0])
    rows_with_history = int(pd.to_numeric(audit_frame[history_column], errors="coerce").gt(0).sum())
    recommended = _select_metric_rows(
        metric_summary,
        existing_repo_feature=False,
        top_n=top_n,
    )
    benchmarks = _select_metric_rows(
        metric_summary,
        existing_repo_feature=True,
        top_n=top_n,
    )
    return {
        "season": season,
        "rolling_window": int(rolling_window),
        "team_count": team_count,
        "team_game_rows": int(len(audit_frame)),
        "rows_with_statcast_history": rows_with_history,
        "statcast_history_coverage_pct": float(
            rows_with_history / len(audit_frame) if len(audit_frame) else 0.0
        ),
        "eva_snapshot_available": bool(eva_correlations is not None and not eva_correlations.empty),
        "eva_source_page_url": EVA_STATCAST_CORRELATIONS_URL,
        "eva_recommended_usage": "research_snapshot",
        "recommended_feature_targets": recommended,
        "existing_repo_benchmarks": benchmarks,
        "notes": [
            "EVA Analytics is used here as a research snapshot, not as a production feature feed.",
            "The underlying EVA datatable is scrapeable but undocumented and should be treated as brittle.",
            "Repo-native rolling Statcast metrics remain the authoritative source for any later model features.",
        ],
    }


def build_eva_statcast_starter_audit_summary(
    audit_frame: pd.DataFrame,
    *,
    metric_summary: pd.DataFrame,
    rolling_window: int = 5,
    eva_correlations: pd.DataFrame | None = None,
    top_n: int = 5,
) -> dict[str, Any]:
    """Build a compact summary for the starter contact-quality-allowed audit."""

    if audit_frame.empty:
        raise ValueError("audit_frame must not be empty")

    history_column = f"rolling_{rolling_window}s_bbe_count"
    pitcher_count = int(audit_frame["pitcher_id"].nunique())
    season = int(pd.to_numeric(audit_frame["season"], errors="coerce").dropna().iloc[0])
    rows_with_history = int(pd.to_numeric(audit_frame[history_column], errors="coerce").gt(0).sum())
    recommended = _select_metric_rows(
        metric_summary,
        existing_repo_feature=False,
        top_n=top_n,
    )
    benchmarks = _select_metric_rows(
        metric_summary,
        existing_repo_feature=True,
        top_n=top_n,
    )
    return {
        "season": season,
        "rolling_window": int(rolling_window),
        "pitcher_count": pitcher_count,
        "starter_rows": int(len(audit_frame)),
        "rows_with_statcast_history": rows_with_history,
        "statcast_history_coverage_pct": float(
            rows_with_history / len(audit_frame) if len(audit_frame) else 0.0
        ),
        "eva_snapshot_available": bool(eva_correlations is not None and not eva_correlations.empty),
        "eva_source_page_url": EVA_STATCAST_CORRELATIONS_URL,
        "eva_recommended_usage": "research_snapshot",
        "recommended_feature_targets": recommended,
        "existing_repo_benchmarks": benchmarks,
        "notes": [
            "Starter audit uses official game starter ids from the repo and Statcast pitch rows for those starters only.",
            "Runs allowed are team-game outcomes, so opener and bullpen leakage are still possible in edge cases.",
            "Use this as a screening tool before promoting any starter contact-quality-allowed feature into research features.",
        ],
    }


def write_eva_statcast_audit_outputs(
    *,
    audit_frame: pd.DataFrame,
    metric_summary: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: str | Path = DEFAULT_EVA_STATCAST_AUDIT_DIR,
    eva_correlations: pd.DataFrame | None = None,
    starter_audit_frame: pd.DataFrame | None = None,
    starter_metric_summary: pd.DataFrame | None = None,
    starter_summary: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write CSV, JSON, and Markdown outputs for the EVA Statcast audit."""

    if audit_frame.empty:
        raise ValueError("audit_frame must not be empty")

    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    season = int(summary["season"])
    rolling_window = int(summary["rolling_window"])

    audit_csv_path = resolved_output_dir / (
        f"eva_statcast_team_game_audit_{season}_w{rolling_window}.csv"
    )
    summary_csv_path = resolved_output_dir / (
        f"eva_statcast_metric_summary_{season}_w{rolling_window}.csv"
    )
    summary_json_path = resolved_output_dir / (
        f"eva_statcast_audit_{season}_w{rolling_window}.summary.json"
    )
    markdown_path = resolved_output_dir / f"eva_statcast_audit_{season}_w{rolling_window}.md"

    audit_frame.to_csv(audit_csv_path, index=False)
    metric_summary.to_csv(summary_csv_path, index=False)
    summary_json_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(tz=timezone.utc).isoformat(),
                "summary": summary,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    markdown_path.write_text(
        build_eva_statcast_markdown_report(
            metric_summary=metric_summary,
            summary=summary,
            starter_metric_summary=starter_metric_summary,
            starter_summary=starter_summary,
        ),
        encoding="utf-8",
    )

    outputs = {
        "audit_csv": audit_csv_path,
        "metric_summary_csv": summary_csv_path,
        "summary_json": summary_json_path,
        "markdown": markdown_path,
    }
    if eva_correlations is not None and not eva_correlations.empty:
        eva_csv_path = resolved_output_dir / "eva_statcast_correlations_snapshot.csv"
        eva_correlations.to_csv(eva_csv_path, index=False)
        outputs["eva_snapshot_csv"] = eva_csv_path
    if (
        starter_audit_frame is not None
        and not starter_audit_frame.empty
        and starter_metric_summary is not None
        and starter_summary is not None
    ):
        starter_window = int(starter_summary["rolling_window"])
        starter_audit_csv_path = resolved_output_dir / (
            f"eva_statcast_starter_start_audit_{season}_w{starter_window}.csv"
        )
        starter_summary_csv_path = resolved_output_dir / (
            f"eva_statcast_starter_metric_summary_{season}_w{starter_window}.csv"
        )
        starter_summary_json_path = resolved_output_dir / (
            f"eva_statcast_starter_audit_{season}_w{starter_window}.summary.json"
        )
        starter_audit_frame.to_csv(starter_audit_csv_path, index=False)
        starter_metric_summary.to_csv(starter_summary_csv_path, index=False)
        starter_summary_json_path.write_text(
            json.dumps(
                {
                    "generated_at": datetime.now(tz=timezone.utc).isoformat(),
                    "summary": starter_summary,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        outputs["starter_audit_csv"] = starter_audit_csv_path
        outputs["starter_metric_summary_csv"] = starter_summary_csv_path
        outputs["starter_summary_json"] = starter_summary_json_path
    return outputs


def build_eva_statcast_markdown_report(
    *,
    metric_summary: pd.DataFrame,
    summary: dict[str, Any],
    starter_metric_summary: pd.DataFrame | None = None,
    starter_summary: dict[str, Any] | None = None,
) -> str:
    """Render a short markdown report for the EVA Statcast research audit."""

    recommended = _as_table_frame(summary["recommended_feature_targets"])
    benchmarks = _as_table_frame(summary["existing_repo_benchmarks"])
    starter_recommended = _as_table_frame(
        starter_summary["recommended_feature_targets"] if starter_summary else []
    )
    starter_benchmarks = _as_table_frame(
        starter_summary["existing_repo_benchmarks"] if starter_summary else []
    )

    lines = [
        f"# EVA Statcast Research Audit ({summary['season']})",
        "",
        "This report treats EVA Analytics as an external Statcast research reference and"
        " compares that guidance to repo-native rolling team Statcast metrics.",
        "",
        "## Coverage",
        "",
        f"- Rolling window: {summary['rolling_window']} team games",
        f"- Team-game rows in audit: {summary['team_game_rows']}",
        f"- Teams covered: {summary['team_count']}",
        f"- Rows with prior Statcast history: {summary['rows_with_statcast_history']}",
        (
            f"- Rows with prior Statcast history pct: "
            f"{summary['statcast_history_coverage_pct']:.3f}"
        ),
        (
            f"- EVA snapshot available: "
            f"{'yes' if summary['eva_snapshot_available'] else 'no'}"
        ),
        (
            f"- EVA handling mode: {summary['eva_recommended_usage']} "
            f"({summary['eva_source_page_url']})"
        ),
        "",
        "## Recommended Targets",
        "",
        _markdown_table(
            recommended,
            [
                "metric_label",
                "family",
                "repo_runs_scored_corr",
                "repo_f5_runs_scored_corr",
                "eva_predictive_woba_corr",
                "eva_predictive_iso_corr",
            ],
        ),
        "",
        "## Existing Benchmarks",
        "",
        _markdown_table(
            benchmarks,
            [
                "metric_label",
                "repo_runs_scored_corr",
                "repo_f5_runs_scored_corr",
                "eva_predictive_woba_corr",
                "eva_predictive_iso_corr",
            ],
        ),
        "",
        "## Notes",
        "",
        *[f"- {note}" for note in summary["notes"]],
    ]
    if starter_summary is not None:
        lines.extend(
            [
                "",
                "## Starter Allowed Coverage",
                "",
                f"- Rolling starter window: {starter_summary['rolling_window']} starts",
                f"- Starter rows in audit: {starter_summary['starter_rows']}",
                f"- Starters covered: {starter_summary['pitcher_count']}",
                (
                    f"- Rows with prior Statcast history: "
                    f"{starter_summary['rows_with_statcast_history']}"
                ),
                (
                    f"- Rows with prior Statcast history pct: "
                    f"{starter_summary['statcast_history_coverage_pct']:.3f}"
                ),
                "",
                "## Starter Allowed Targets",
                "",
                _markdown_table(
                    starter_recommended,
                    [
                        "metric_label",
                        "family",
                        "repo_runs_allowed_corr",
                        "repo_f5_runs_allowed_corr",
                        "eva_predictive_woba_corr",
                        "eva_predictive_iso_corr",
                    ],
                ),
                "",
                "## Starter Allowed Benchmarks",
                "",
                _markdown_table(
                    starter_benchmarks,
                    [
                        "metric_label",
                        "repo_runs_allowed_corr",
                        "repo_f5_runs_allowed_corr",
                        "eva_predictive_woba_corr",
                        "eva_predictive_iso_corr",
                    ],
                ),
                "",
                "## Starter Notes",
                "",
                *[f"- {note}" for note in starter_summary["notes"]],
            ]
        )
    return "\n".join(lines)


def _load_team_game_outcomes(
    *,
    season: int,
    db_path: str | Path,
) -> pd.DataFrame:
    with sqlite_connection(db_path, builder_optimized=True) as connection:
        games = pd.read_sql_query(
            """
            SELECT
                game_pk,
                date,
                home_team,
                away_team,
                f5_home_score,
                f5_away_score,
                final_home_score,
                final_away_score,
                status
            FROM games
            WHERE substr(date, 1, 4) = ?
            ORDER BY substr(date, 1, 10), game_pk
            """,
            connection,
            params=(str(int(season)),),
        )

    if games.empty:
        return pd.DataFrame()

    games["status"] = games["status"].astype(str).str.strip().str.lower()
    games = games.loc[games["status"].isin(_FINAL_GAME_STATES)].copy()
    if games.empty:
        return pd.DataFrame()

    for column in ("f5_home_score", "f5_away_score", "final_home_score", "final_away_score"):
        games[column] = pd.to_numeric(games[column], errors="coerce")
    games = games.dropna(
        subset=["f5_home_score", "f5_away_score", "final_home_score", "final_away_score"]
    ).copy()
    if games.empty:
        return pd.DataFrame()

    games["game_date"] = pd.to_datetime(games["date"], errors="coerce", utc=True).dt.date
    home_rows = pd.DataFrame(
        {
            "game_pk": games["game_pk"].astype(int),
            "season": int(season),
            "game_date": games["game_date"],
            "team": games["home_team"].astype(str).str.strip().str.upper(),
            "opponent": games["away_team"].astype(str).str.strip().str.upper(),
            "is_home": 1,
            "runs_scored": games["final_home_score"].astype(float),
            "runs_allowed": games["final_away_score"].astype(float),
            "f5_runs_scored": games["f5_home_score"].astype(float),
            "f5_runs_allowed": games["f5_away_score"].astype(float),
        }
    )
    away_rows = pd.DataFrame(
        {
            "game_pk": games["game_pk"].astype(int),
            "season": int(season),
            "game_date": games["game_date"],
            "team": games["away_team"].astype(str).str.strip().str.upper(),
            "opponent": games["home_team"].astype(str).str.strip().str.upper(),
            "is_home": 0,
            "runs_scored": games["final_away_score"].astype(float),
            "runs_allowed": games["final_home_score"].astype(float),
            "f5_runs_scored": games["f5_away_score"].astype(float),
            "f5_runs_allowed": games["f5_home_score"].astype(float),
        }
    )
    outcomes = pd.concat([home_rows, away_rows], ignore_index=True)
    outcomes["total_game_runs"] = outcomes["runs_scored"] + outcomes["runs_allowed"]
    return outcomes


def _load_starter_rows(
    *,
    season: int,
    db_path: str | Path,
) -> pd.DataFrame:
    with sqlite_connection(db_path, builder_optimized=True) as connection:
        games = pd.read_sql_query(
            """
            SELECT game_pk, date, home_team, away_team, home_starter_id, away_starter_id
            FROM games
            WHERE substr(date, 1, 4) = ?
            ORDER BY substr(date, 1, 10), game_pk
            """,
            connection,
            params=(str(int(season)),),
        )

    if games.empty:
        return pd.DataFrame(
            columns=["game_pk", "season", "game_date", "team", "pitcher_id", "opponent", "is_home"]
        )

    rows: list[dict[str, Any]] = []
    for game in games.to_dict(orient="records"):
        game_pk = pd.to_numeric(game.get("game_pk"), errors="coerce")
        game_date = pd.to_datetime(game.get("date"), errors="coerce", utc=True)
        if pd.isna(game_pk) or pd.isna(game_date):
            continue

        for is_home, team_key, opponent_key, starter_key in (
            (1, "home_team", "away_team", "home_starter_id"),
            (0, "away_team", "home_team", "away_starter_id"),
        ):
            pitcher_id = pd.to_numeric(game.get(starter_key), errors="coerce")
            team = str(game.get(team_key) or "").strip().upper()
            opponent = str(game.get(opponent_key) or "").strip().upper()
            if pd.isna(pitcher_id) or not team or not opponent:
                continue
            rows.append(
                {
                    "game_pk": int(game_pk),
                    "season": int(season),
                    "game_date": game_date.date(),
                    "team": team,
                    "pitcher_id": int(pitcher_id),
                    "opponent": opponent,
                    "is_home": int(is_home),
                }
            )

    return pd.DataFrame(rows)


def _build_team_game_statcast_counts(statcast_frame: pd.DataFrame) -> pd.DataFrame:
    if statcast_frame.empty:
        return pd.DataFrame(columns=["game_pk", "team", *_ROLLING_COUNT_COLUMNS])

    terminal = _collapse_plate_appearances(statcast_frame)
    if terminal.empty:
        return pd.DataFrame(columns=["game_pk", "team", *_ROLLING_COUNT_COLUMNS])

    team = _resolve_batting_team(terminal)
    game_pk_column = _first_column(terminal, ("game_pk",))
    game_date_column = _first_column(terminal, ("game_date",))
    xwoba_column = _first_column(terminal, ("estimated_woba_using_speedangle", "xwoba"))
    launch_speed_column = _first_column(terminal, ("launch_speed",))
    launch_angle_column = _first_column(terminal, ("launch_angle",))
    bb_type_column = _first_column(terminal, ("bb_type",))
    if game_pk_column is None:
        return pd.DataFrame(columns=["game_pk", "team", *_ROLLING_COUNT_COLUMNS])

    terminal = terminal.copy()
    terminal["team"] = team
    terminal = terminal.loc[terminal["team"].ne("")].copy()
    if terminal.empty:
        return pd.DataFrame(columns=["game_pk", "team", *_ROLLING_COUNT_COLUMNS])

    launch_speed = (
        pd.to_numeric(terminal[launch_speed_column], errors="coerce")
        if launch_speed_column is not None
        else pd.Series(float("nan"), index=terminal.index)
    )
    launch_angle = (
        pd.to_numeric(terminal[launch_angle_column], errors="coerce")
        if launch_angle_column is not None
        else pd.Series(float("nan"), index=terminal.index)
    )
    xwoba = (
        pd.to_numeric(terminal[xwoba_column], errors="coerce")
        if xwoba_column is not None
        else pd.Series(float("nan"), index=terminal.index)
    )
    bb_type = (
        terminal[bb_type_column].astype(str).str.strip().str.lower()
        if bb_type_column is not None
        else pd.Series("", index=terminal.index, dtype=str)
    )

    batted_ball_mask = launch_speed.notna() & launch_angle.notna()
    in_air_mask = batted_ball_mask & (
        bb_type.isin({"fly_ball", "line_drive", "popup"}) | launch_angle.ge(10.0)
    )
    grouped = terminal.assign(
        game_pk=pd.to_numeric(terminal[game_pk_column], errors="coerce"),
        game_date=(
            pd.to_datetime(terminal[game_date_column], errors="coerce", utc=True).dt.date
            if game_date_column is not None
            else pd.NaT
        ),
        pa_count=1.0,
        xwoba_sum=xwoba.fillna(0.0),
        xwoba_count=xwoba.notna().astype(float),
        bbe_count=batted_ball_mask.astype(float),
        barrels_count=(
            _is_barrel_mask(launch_speed=launch_speed, launch_angle=launch_angle)
            & batted_ball_mask
        ).astype(float),
        launch_speed_sum=launch_speed.where(batted_ball_mask, 0.0).fillna(0.0),
        launch_speed_count=batted_ball_mask.astype(float),
        hard_hit_count=(launch_speed.ge(95.0) & batted_ball_mask).astype(float),
        air_bbe_count=in_air_mask.astype(float),
        air_100plus_count=(launch_speed.ge(100.0) & in_air_mask).astype(float),
        launch_angle_sum=launch_angle.where(batted_ball_mask, 0.0).fillna(0.0),
        launch_angle_sq_sum=launch_angle.pow(2).where(batted_ball_mask, 0.0).fillna(0.0),
        launch_angle_count=batted_ball_mask.astype(float),
        launch_angle_neg4_26_count=(
            launch_angle.ge(-4.0) & launch_angle.le(26.0) & batted_ball_mask
        ).astype(float),
        launch_angle_23_34_count=(
            launch_angle.ge(23.0) & launch_angle.le(34.0) & batted_ball_mask
        ).astype(float),
        launch_angle_38_plus_count=(launch_angle.gt(38.0) & batted_ball_mask).astype(float),
    )
    grouped = grouped.dropna(subset=["game_pk"]).copy()
    if grouped.empty:
        return pd.DataFrame(columns=["game_pk", "team", *_ROLLING_COUNT_COLUMNS])

    summary = (
        grouped.groupby(["game_pk", "team"], as_index=False)
        .agg(
            game_date=("game_date", "first"),
            **{column: (column, "sum") for column in _ROLLING_COUNT_COLUMNS},
        )
        .sort_values(["team", "game_date", "game_pk"])
        .reset_index(drop=True)
    )
    summary["game_pk"] = summary["game_pk"].astype(int)
    return summary


def _build_starter_start_statcast_counts(
    start_rows: pd.DataFrame,
    statcast_frame: pd.DataFrame,
) -> pd.DataFrame:
    if start_rows.empty or statcast_frame.empty:
        return pd.DataFrame(columns=["game_pk", "team", "pitcher_id", *_ROLLING_COUNT_COLUMNS])

    terminal = _collapse_plate_appearances(statcast_frame)
    if terminal.empty:
        return pd.DataFrame(columns=["game_pk", "team", "pitcher_id", *_ROLLING_COUNT_COLUMNS])

    game_pk_column = _first_column(terminal, ("game_pk",))
    pitcher_column = _first_column(terminal, ("pitcher", "pitcher_id"))
    xwoba_column = _first_column(terminal, ("estimated_woba_using_speedangle", "xwoba"))
    launch_speed_column = _first_column(terminal, ("launch_speed",))
    launch_angle_column = _first_column(terminal, ("launch_angle",))
    bb_type_column = _first_column(terminal, ("bb_type",))
    if game_pk_column is None or pitcher_column is None:
        return pd.DataFrame(columns=["game_pk", "team", "pitcher_id", *_ROLLING_COUNT_COLUMNS])

    pitches = terminal.copy()
    pitches["game_pk"] = pd.to_numeric(pitches[game_pk_column], errors="coerce").astype("Int64")
    pitches["pitcher_id"] = pd.to_numeric(pitches[pitcher_column], errors="coerce").astype("Int64")
    pitches = pitches.dropna(subset=["game_pk", "pitcher_id"]).copy()
    if pitches.empty:
        return pd.DataFrame(columns=["game_pk", "team", "pitcher_id", *_ROLLING_COUNT_COLUMNS])

    launch_speed = (
        pd.to_numeric(pitches[launch_speed_column], errors="coerce")
        if launch_speed_column is not None
        else pd.Series(float("nan"), index=pitches.index)
    )
    launch_angle = (
        pd.to_numeric(pitches[launch_angle_column], errors="coerce")
        if launch_angle_column is not None
        else pd.Series(float("nan"), index=pitches.index)
    )
    xwoba = (
        pd.to_numeric(pitches[xwoba_column], errors="coerce")
        if xwoba_column is not None
        else pd.Series(float("nan"), index=pitches.index)
    )
    bb_type = (
        pitches[bb_type_column].astype(str).str.strip().str.lower()
        if bb_type_column is not None
        else pd.Series("", index=pitches.index, dtype=str)
    )
    batted_ball_mask = launch_speed.notna() & launch_angle.notna()
    in_air_mask = batted_ball_mask & (
        bb_type.isin({"fly_ball", "line_drive", "popup"}) | launch_angle.ge(10.0)
    )
    pitches = pitches.assign(
        pa_count=1.0,
        xwoba_sum=xwoba.fillna(0.0),
        xwoba_count=xwoba.notna().astype(float),
        bbe_count=batted_ball_mask.astype(float),
        barrels_count=(
            _is_barrel_mask(launch_speed=launch_speed, launch_angle=launch_angle)
            & batted_ball_mask
        ).astype(float),
        launch_speed_sum=launch_speed.where(batted_ball_mask, 0.0).fillna(0.0),
        launch_speed_count=batted_ball_mask.astype(float),
        hard_hit_count=(launch_speed.ge(95.0) & batted_ball_mask).astype(float),
        air_bbe_count=in_air_mask.astype(float),
        air_100plus_count=(launch_speed.ge(100.0) & in_air_mask).astype(float),
        launch_angle_sum=launch_angle.where(batted_ball_mask, 0.0).fillna(0.0),
        launch_angle_sq_sum=launch_angle.pow(2).where(batted_ball_mask, 0.0).fillna(0.0),
        launch_angle_count=batted_ball_mask.astype(float),
        launch_angle_neg4_26_count=(
            launch_angle.ge(-4.0) & launch_angle.le(26.0) & batted_ball_mask
        ).astype(float),
        launch_angle_23_34_count=(
            launch_angle.ge(23.0) & launch_angle.le(34.0) & batted_ball_mask
        ).astype(float),
        launch_angle_38_plus_count=(launch_angle.gt(38.0) & batted_ball_mask).astype(float),
    )

    starter_keys = start_rows.loc[:, ["game_pk", "team", "pitcher_id"]].copy()
    starter_keys["game_pk"] = pd.to_numeric(starter_keys["game_pk"], errors="coerce").astype("Int64")
    starter_keys["pitcher_id"] = pd.to_numeric(
        starter_keys["pitcher_id"], errors="coerce"
    ).astype("Int64")
    starter_pitches = starter_keys.merge(
        pitches,
        on=["game_pk", "pitcher_id"],
        how="left",
    )
    if starter_pitches.empty:
        return pd.DataFrame(columns=["game_pk", "team", "pitcher_id", *_ROLLING_COUNT_COLUMNS])

    summary = (
        starter_pitches.groupby(["game_pk", "team", "pitcher_id"], as_index=False)
        .agg(**{column: (column, "sum") for column in _ROLLING_COUNT_COLUMNS})
        .sort_values(["pitcher_id", "game_pk", "team"])
        .reset_index(drop=True)
    )
    summary["game_pk"] = summary["game_pk"].astype(int)
    summary["pitcher_id"] = summary["pitcher_id"].astype(int)
    return summary


def _collapse_plate_appearances(pitches: pd.DataFrame) -> pd.DataFrame:
    if pitches.empty:
        return pitches.copy()

    at_bat_column = _first_column(pitches, ("at_bat_number",))
    if at_bat_column is not None:
        sort_columns = [
            column
            for column in ("game_pk", at_bat_column, "pitch_number")
            if column in pitches.columns
        ]
        group_columns = [
            column for column in ("game_pk", at_bat_column) if column in pitches.columns
        ]
        terminal = pitches.sort_values(sort_columns).groupby(group_columns, as_index=False).tail(1)
        return terminal.reset_index(drop=True)

    events_column = _first_column(pitches, ("events",))
    if events_column is not None:
        terminal = pitches.loc[pitches[events_column].notna()].copy()
        if not terminal.empty:
            return terminal.reset_index(drop=True)

    return pitches.tail(1).reset_index(drop=True)


def _resolve_batting_team(dataframe: pd.DataFrame) -> pd.Series:
    batting_team_column = _first_column(dataframe, ("batting_team", "team_batting"))
    if batting_team_column is not None:
        return dataframe[batting_team_column].astype(str).str.strip().str.upper()

    topbot_column = _first_column(dataframe, ("inning_topbot", "inning_top_bot"))
    home_team_column = _first_column(dataframe, ("home_team",))
    away_team_column = _first_column(dataframe, ("away_team",))
    if topbot_column is None or home_team_column is None or away_team_column is None:
        return pd.Series("", index=dataframe.index, dtype=str)

    half_inning = dataframe[topbot_column].astype(str).str.strip().str.lower()
    team = pd.Series("", index=dataframe.index, dtype=str)
    top_mask = half_inning.eq("top")
    bottom_mask = half_inning.eq("bot")
    team.loc[top_mask] = dataframe.loc[top_mask, away_team_column].astype(str)
    team.loc[bottom_mask] = dataframe.loc[bottom_mask, home_team_column].astype(str)
    return team.str.strip().str.upper()


def _first_column(dataframe: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    normalized_columns = {str(column).strip().lower(): str(column) for column in dataframe.columns}
    for candidate in candidates:
        match = normalized_columns.get(candidate.strip().lower())
        if match is not None:
            return match
    return None


def _is_barrel_mask(*, launch_speed: pd.Series, launch_angle: pd.Series) -> pd.Series:
    exit_velocity = pd.to_numeric(launch_speed, errors="coerce")
    angle = pd.to_numeric(launch_angle, errors="coerce")
    min_angle = (26.0 - (exit_velocity - 98.0)).clip(lower=8.0)
    max_angle = (30.0 + ((exit_velocity - 98.0) * 1.2)).clip(upper=50.0)
    return exit_velocity.ge(98.0) & angle.ge(min_angle) & angle.le(max_angle)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator_values = pd.to_numeric(numerator, errors="coerce")
    denominator_values = pd.to_numeric(denominator, errors="coerce")
    result = numerator_values / denominator_values.replace(0, pd.NA)
    return pd.to_numeric(result, errors="coerce")


def _build_eva_lookup(eva_correlations: pd.DataFrame | None) -> dict[tuple[str | None, str], dict[str, Any]]:
    if eva_correlations is None or eva_correlations.empty:
        return {}

    lookup: dict[tuple[str | None, str], dict[str, Any]] = {}
    normalized = eva_correlations.copy()
    normalized["stat"] = normalized["stat"].astype(str).str.strip().str.upper()
    normalized["type"] = normalized["type"].astype(str).str.strip().str.lower()
    for row in normalized.to_dict(orient="records"):
        lookup[(row["stat"], row["type"])] = row
    return lookup


def _lookup_eva_value(row: dict[str, Any] | None, key: str) -> float | None:
    if row is None:
        return None
    value = row.get(key)
    if value is None or pd.isna(value):
        return None
    return float(value)


def _pearson_with_n(left: pd.Series, right: pd.Series) -> tuple[float | None, int]:
    frame = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(frame) < 2:
        return None, int(len(frame))
    if frame["left"].nunique(dropna=True) < 2 or frame["right"].nunique(dropna=True) < 2:
        return None, int(len(frame))
    correlation = frame["left"].corr(frame["right"])
    if pd.isna(correlation):
        return None, int(len(frame))
    return float(correlation), int(len(frame))


def _priority_score(
    *,
    repo_runs_scored_corr: float | None,
    repo_f5_runs_scored_corr: float | None,
    eva_predictive: dict[str, Any] | None,
) -> float:
    eva_woba = _lookup_eva_value(eva_predictive, "woba_corr") or 0.0
    eva_iso = _lookup_eva_value(eva_predictive, "iso_corr") or 0.0
    eva_babip = _lookup_eva_value(eva_predictive, "babip_corr") or 0.0
    return float(
        max(eva_woba, 0.0)
        + max(eva_iso, 0.0)
        + max(eva_babip, 0.0)
        + abs(repo_runs_scored_corr or 0.0)
        + abs(repo_f5_runs_scored_corr or 0.0)
    )


def _select_metric_rows(
    metric_summary: pd.DataFrame,
    *,
    existing_repo_feature: bool,
    top_n: int,
) -> list[dict[str, Any]]:
    if metric_summary.empty:
        return []

    filtered = metric_summary.loc[
        metric_summary["existing_repo_feature"].eq(existing_repo_feature)
    ].copy()
    if filtered.empty:
        return []
    filtered = filtered.sort_values(
        ["priority_score", "metric_label"],
        ascending=[False, True],
    ).head(top_n)
    columns = [
        "metric_key",
        "metric_label",
        "family",
        "eva_predictive_woba_corr",
        "eva_predictive_iso_corr",
    ]
    if "repo_runs_scored_corr" in filtered.columns:
        columns.extend(["repo_runs_scored_corr", "repo_f5_runs_scored_corr"])
    if "repo_runs_allowed_corr" in filtered.columns:
        columns.extend(["repo_runs_allowed_corr", "repo_f5_runs_allowed_corr"])
    return filtered.loc[:, columns].to_dict(orient="records")


def _as_table_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _markdown_table(dataframe: pd.DataFrame, columns: list[str]) -> str:
    if dataframe.empty:
        return "_No rows_"

    subset = dataframe.loc[:, [column for column in columns if column in dataframe.columns]].copy()
    for column in subset.columns:
        if pd.api.types.is_numeric_dtype(subset[column]):
            subset[column] = subset[column].map(_format_numeric)

    header = "| " + " | ".join(subset.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(subset.columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[column]) for column in subset.columns) + " |"
        for row in subset.to_dict(orient="records")
    ]
    return "\n".join([header, divider, *rows])


def _format_numeric(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    numeric = float(value)
    if math.isfinite(numeric) and abs(numeric) >= 100:
        return f"{numeric:.1f}"
    return f"{numeric:.3f}"


__all__ = [
    "DEFAULT_EVA_STATCAST_AUDIT_DIR",
    "build_eva_statcast_audit_summary",
    "build_eva_statcast_markdown_report",
    "build_eva_statcast_metric_summary",
    "build_eva_statcast_starter_audit_summary",
    "build_eva_statcast_starter_metric_summary",
    "build_eva_statcast_starter_start_audit_frame",
    "build_eva_statcast_team_game_audit_frame",
    "write_eva_statcast_audit_outputs",
]
