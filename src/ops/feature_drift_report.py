from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


POSTGAME_OUTCOME_COLUMNS = {
    "f5_home_score",
    "f5_away_score",
    "final_home_score",
    "final_away_score",
    "f5_margin",
    "final_margin",
    "f5_tied_after_5",
    "f5_ml_result",
    "f5_rl_result",
}

METADATA_COLUMNS = {
    "game_pk",
    "season",
    "data_version_hash",
    "build_timestamp",
}

DEFAULT_BASELINE_START = 2021
DEFAULT_BASELINE_END = 2023
DEFAULT_RECENT_START = 2024
DEFAULT_RECENT_END = 2025
VARIANCE_EPSILON = 1e-9


def build_feature_drift_dataframe(
    training_data_path: str | Path,
    *,
    baseline_start: int = DEFAULT_BASELINE_START,
    baseline_end: int = DEFAULT_BASELINE_END,
    recent_start: int = DEFAULT_RECENT_START,
    recent_end: int = DEFAULT_RECENT_END,
    min_non_null_count: int = 100,
    max_rows_per_segment: int | None = None,
) -> pd.DataFrame:
    frame = pd.read_parquet(Path(training_data_path))
    if frame.empty:
        return pd.DataFrame()

    baseline = frame.loc[frame["season"].between(baseline_start, baseline_end)].copy()
    recent = frame.loc[frame["season"].between(recent_start, recent_end)].copy()
    if baseline.empty or recent.empty:
        return pd.DataFrame()

    if max_rows_per_segment is not None and max_rows_per_segment > 0:
        baseline = baseline.head(max_rows_per_segment).copy()
        recent = recent.head(max_rows_per_segment).copy()

    numeric_columns = [
        column
        for column in frame.select_dtypes(include=[np.number, "bool"]).columns
        if column not in METADATA_COLUMNS
    ]

    records: list[dict[str, float | int | str]] = []
    for column in numeric_columns:
        baseline_series = pd.to_numeric(baseline[column], errors="coerce")
        recent_series = pd.to_numeric(recent[column], errors="coerce")
        baseline_non_null = baseline_series.dropna()
        recent_non_null = recent_series.dropna()
        if len(baseline_non_null) < min_non_null_count or len(recent_non_null) < min_non_null_count:
            continue

        baseline_mean = float(baseline_non_null.mean())
        recent_mean = float(recent_non_null.mean())
        baseline_std = float(baseline_non_null.std(ddof=0))
        recent_std = float(recent_non_null.std(ddof=0))
        pooled_std = math.sqrt((baseline_std ** 2 + recent_std ** 2) / 2.0)
        standardized_mean_diff = (
            float((recent_mean - baseline_mean) / pooled_std) if pooled_std > VARIANCE_EPSILON else 0.0
        )
        mean_delta = float(recent_mean - baseline_mean)
        baseline_missing_pct = float(baseline_series.isna().mean())
        recent_missing_pct = float(recent_series.isna().mean())
        missing_pct_delta = float(recent_missing_pct - baseline_missing_pct)
        ks_stat = _ks_statistic(baseline_non_null.to_numpy(), recent_non_null.to_numpy())
        psi = _population_stability_index(baseline_non_null.to_numpy(), recent_non_null.to_numpy())
        drift_score = (
            abs(standardized_mean_diff) * 0.45
            + ks_stat * 0.35
            + psi * 0.10
            + abs(missing_pct_delta) * 0.10
        )
        records.append(
            {
                "column": column,
                "data_role": _classify_data_role(column),
                "family": _classify_family(column),
                "baseline_count": int(len(baseline_non_null)),
                "recent_count": int(len(recent_non_null)),
                "baseline_mean": baseline_mean,
                "recent_mean": recent_mean,
                "mean_delta": mean_delta,
                "baseline_std": baseline_std,
                "recent_std": recent_std,
                "standardized_mean_diff": standardized_mean_diff,
                "baseline_missing_pct": baseline_missing_pct,
                "recent_missing_pct": recent_missing_pct,
                "missing_pct_delta": missing_pct_delta,
                "ks_stat": ks_stat,
                "psi": psi,
                "drift_score": float(drift_score),
            }
        )

    drift_frame = pd.DataFrame.from_records(records)
    if drift_frame.empty:
        return drift_frame
    return drift_frame.sort_values("drift_score", ascending=False, kind="stable").reset_index(drop=True)


def build_family_summary(drift_frame: pd.DataFrame) -> pd.DataFrame:
    if drift_frame.empty:
        return pd.DataFrame()
    summary = (
        drift_frame.groupby(["data_role", "family"], dropna=False)
        .agg(
            feature_count=("column", "count"),
            mean_drift_score=("drift_score", "mean"),
            median_drift_score=("drift_score", "median"),
            max_drift_score=("drift_score", "max"),
            mean_abs_smd=("standardized_mean_diff", lambda value: float(np.mean(np.abs(value)))),
            mean_ks=("ks_stat", "mean"),
            mean_psi=("psi", "mean"),
        )
        .reset_index()
        .sort_values(["mean_drift_score", "max_drift_score"], ascending=[False, False], kind="stable")
        .reset_index(drop=True)
    )
    return summary


def write_feature_drift_report(
    training_data_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    baseline_start: int = DEFAULT_BASELINE_START,
    baseline_end: int = DEFAULT_BASELINE_END,
    recent_start: int = DEFAULT_RECENT_START,
    recent_end: int = DEFAULT_RECENT_END,
    min_non_null_count: int = 100,
    max_rows_per_segment: int | None = None,
) -> dict[str, Path]:
    resolved_training_data_path = Path(training_data_path)
    resolved_output_dir = Path(output_dir) if output_dir is not None else resolved_training_data_path.parent
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    drift_frame = build_feature_drift_dataframe(
        resolved_training_data_path,
        baseline_start=baseline_start,
        baseline_end=baseline_end,
        recent_start=recent_start,
        recent_end=recent_end,
        min_non_null_count=min_non_null_count,
        max_rows_per_segment=max_rows_per_segment,
    )
    family_frame = build_family_summary(drift_frame)

    stem = (
        f"feature_drift_{baseline_start}_{baseline_end}_vs_{recent_start}_{recent_end}"
    )
    csv_path = resolved_output_dir / f"{stem}.csv"
    family_csv_path = resolved_output_dir / f"{stem}_families.csv"
    html_path = resolved_output_dir / f"{stem}.html"
    json_path = resolved_output_dir / f"{stem}.json"

    drift_frame.to_csv(csv_path, index=False)
    family_frame.to_csv(family_csv_path, index=False)

    summary = _build_summary_payload(
        resolved_training_data_path=resolved_training_data_path,
        drift_frame=drift_frame,
        family_frame=family_frame,
        baseline_start=baseline_start,
        baseline_end=baseline_end,
        recent_start=recent_start,
        recent_end=recent_end,
        min_non_null_count=min_non_null_count,
        max_rows_per_segment=max_rows_per_segment,
        csv_path=csv_path,
        family_csv_path=family_csv_path,
        html_path=html_path,
    )
    html_path.write_text(_build_html(summary, drift_frame, family_frame), encoding="utf-8")
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "csv_path": csv_path,
        "family_csv_path": family_csv_path,
        "html_path": html_path,
        "json_path": json_path,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize feature drift between season windows")
    parser.add_argument("--training-data", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--baseline-start", type=int, default=DEFAULT_BASELINE_START)
    parser.add_argument("--baseline-end", type=int, default=DEFAULT_BASELINE_END)
    parser.add_argument("--recent-start", type=int, default=DEFAULT_RECENT_START)
    parser.add_argument("--recent-end", type=int, default=DEFAULT_RECENT_END)
    parser.add_argument("--min-non-null-count", type=int, default=100)
    parser.add_argument("--max-rows-per-segment", type=int)
    args = parser.parse_args(argv)

    result = write_feature_drift_report(
        training_data_path=args.training_data,
        output_dir=args.output_dir,
        baseline_start=args.baseline_start,
        baseline_end=args.baseline_end,
        recent_start=args.recent_start,
        recent_end=args.recent_end,
        min_non_null_count=args.min_non_null_count,
        max_rows_per_segment=args.max_rows_per_segment,
    )
    print(json.dumps({key: str(value) for key, value in result.items()}, indent=2, sort_keys=True))
    return 0


def _build_summary_payload(
    *,
    resolved_training_data_path: Path,
    drift_frame: pd.DataFrame,
    family_frame: pd.DataFrame,
    baseline_start: int,
    baseline_end: int,
    recent_start: int,
    recent_end: int,
    min_non_null_count: int,
    max_rows_per_segment: int | None,
    csv_path: Path,
    family_csv_path: Path,
    html_path: Path,
) -> dict[str, object]:
    top_features = drift_frame.head(25).to_dict(orient="records")
    top_response = (
        drift_frame.loc[drift_frame["data_role"] == "response"]
        .head(10)
        .to_dict(orient="records")
    )
    top_families = family_frame.head(15).to_dict(orient="records")
    return {
        "training_data_path": str(resolved_training_data_path),
        "baseline_window": {"start_season": baseline_start, "end_season": baseline_end},
        "recent_window": {"start_season": recent_start, "end_season": recent_end},
        "min_non_null_count": int(min_non_null_count),
        "max_rows_per_segment": max_rows_per_segment,
        "feature_count": int(len(drift_frame)),
        "family_count": int(len(family_frame)),
        "csv_path": str(csv_path),
        "family_csv_path": str(family_csv_path),
        "html_path": str(html_path),
        "top_features": top_features,
        "top_response_columns": top_response,
        "top_families": top_families,
    }


def _build_html(summary: dict[str, object], drift_frame: pd.DataFrame, family_frame: pd.DataFrame) -> str:
    top_features_html = drift_frame.head(25).to_html(index=False, border=0)
    response_html = (
        drift_frame.loc[drift_frame["data_role"] == "response"].head(10).to_html(index=False, border=0)
        if not drift_frame.empty
        else "<p>No response columns found.</p>"
    )
    family_html = family_frame.head(15).to_html(index=False, border=0)
    baseline_window = summary["baseline_window"]
    recent_window = summary["recent_window"]
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Feature Drift Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #111827; background: #f8fafc; }}
    h1, h2 {{ margin-bottom: 12px; }}
    .cards {{ display: grid; grid-template-columns: repeat(4, minmax(180px, 1fr)); gap: 12px; margin-bottom: 24px; }}
    .card {{ background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 14px; }}
    .label {{ font-size: 12px; text-transform: uppercase; color: #6b7280; margin-bottom: 6px; }}
    .value {{ font-size: 22px; font-weight: 700; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; background: white; margin-bottom: 24px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; }}
    th {{ background: #f3f4f6; }}
    tr:nth-child(even) {{ background: #fafafa; }}
    p.meta {{ color: #4b5563; }}
  </style>
</head>
<body>
  <h1>Feature Drift Report</h1>
  <p class="meta">Baseline seasons: {baseline_window["start_season"]}-{baseline_window["end_season"]} | Recent seasons: {recent_window["start_season"]}-{recent_window["end_season"]}</p>
  <div class="cards">
    <div class="card"><div class="label">Columns Scored</div><div class="value">{summary["feature_count"]}</div></div>
    <div class="card"><div class="label">Families Scored</div><div class="value">{summary["family_count"]}</div></div>
    <div class="card"><div class="label">Top Drift Family</div><div class="value">{family_frame.iloc[0]["family"] if not family_frame.empty else "n/a"}</div></div>
    <div class="card"><div class="label">Top Drift Column</div><div class="value">{drift_frame.iloc[0]["column"] if not drift_frame.empty else "n/a"}</div></div>
  </div>
  <h2>Top Drift Columns</h2>
  {top_features_html}
  <h2>Response / Outcome Drift</h2>
  {response_html}
  <h2>Family Summary</h2>
  {family_html}
</body>
</html>
"""


def _classify_data_role(column: str) -> str:
    if column in POSTGAME_OUTCOME_COLUMNS:
        return "response"
    return "feature"


def _classify_family(column: str) -> str:
    if column in POSTGAME_OUTCOME_COLUMNS:
        return "response"
    if column.startswith("home_lineup_"):
        return "home_lineup"
    if column.startswith("away_lineup_"):
        return "away_lineup"
    if column.startswith("home_starter_"):
        return "home_starter"
    if column.startswith("away_starter_"):
        return "away_starter"
    if column.startswith("home_team_bullpen_"):
        return "home_bullpen"
    if column.startswith("away_team_bullpen_"):
        return "away_bullpen"
    if column.startswith("home_team_"):
        return "home_team"
    if column.startswith("away_team_"):
        return "away_team"
    if column.startswith("weather_"):
        return "weather"
    if column.startswith("park_"):
        return "park"
    if column.startswith("plate_umpire_"):
        return "umpire"
    if column.startswith("abs_") or column == "abs_active":
        return "abs"
    return "other"


def _ks_statistic(baseline: np.ndarray, recent: np.ndarray) -> float:
    if baseline.size == 0 or recent.size == 0:
        return 0.0
    baseline = np.sort(baseline)
    recent = np.sort(recent)
    combined = np.unique(np.concatenate([baseline, recent]))
    baseline_cdf = np.searchsorted(baseline, combined, side="right") / baseline.size
    recent_cdf = np.searchsorted(recent, combined, side="right") / recent.size
    return float(np.max(np.abs(baseline_cdf - recent_cdf)))


def _population_stability_index(
    baseline: np.ndarray,
    recent: np.ndarray,
    *,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    if baseline.size == 0 or recent.size == 0:
        return 0.0
    unique_count = len(np.unique(baseline))
    if unique_count <= 1:
        baseline_share = float(np.mean(baseline == baseline[0]))
        recent_share = float(np.mean(recent == baseline[0]))
        baseline_share = max(baseline_share, epsilon)
        recent_share = max(recent_share, epsilon)
        return float((recent_share - baseline_share) * math.log(recent_share / baseline_share))

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(baseline, quantiles)
    edges = np.unique(edges)
    if len(edges) < 2:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf

    baseline_hist, _ = np.histogram(baseline, bins=edges)
    recent_hist, _ = np.histogram(recent, bins=edges)
    baseline_share = baseline_hist / baseline.size
    recent_share = recent_hist / recent.size
    baseline_share = np.clip(baseline_share, epsilon, None)
    recent_share = np.clip(recent_share, epsilon, None)
    psi = np.sum((recent_share - baseline_share) * np.log(recent_share / baseline_share))
    return float(psi)


if __name__ == "__main__":
    raise SystemExit(main())
