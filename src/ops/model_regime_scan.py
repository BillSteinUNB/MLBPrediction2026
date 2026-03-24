from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

DEFAULT_TREND_COLUMNS = (
    "home_team_bullpen_xfip",
    "away_team_bullpen_xfip",
    "home_starter_pitch_mix_entropy_30s",
    "away_starter_pitch_mix_entropy_30s",
    "home_starter_xfip_30s",
    "away_starter_xfip_30s",
    "home_team_woba_30g",
    "away_team_woba_30g",
    "f5_ml_result",
    "f5_rl_result",
)

DEFAULT_EDGE_BUCKETS = (0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 1.00)


def write_model_regime_scan(
    *,
    training_data_path: str | Path,
    early_predictions_path: str | Path,
    recent_predictions_path: str | Path,
    output_dir: str | Path,
    trend_columns: Sequence[str] = DEFAULT_TREND_COLUMNS,
    edge_buckets: Sequence[float] = DEFAULT_EDGE_BUCKETS,
) -> dict[str, Path]:
    training_data_path = Path(training_data_path)
    early_predictions_path = Path(early_predictions_path)
    recent_predictions_path = Path(recent_predictions_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_frame = pd.read_parquet(training_data_path)
    trend_frame = build_season_trend_dataframe(training_frame, trend_columns=trend_columns)
    feature_relationship_frame = build_feature_relationship_dataframe(
        training_frame,
        columns=[
            "home_team_bullpen_xfip",
            "away_team_bullpen_xfip",
            "home_starter_pitch_mix_entropy_30s",
            "away_starter_pitch_mix_entropy_30s",
            "home_starter_xfip_30s",
            "away_starter_xfip_30s",
            "home_team_woba_30g",
            "away_team_woba_30g",
        ],
        target_column="f5_ml_result",
    )

    early_edge_frame = build_edge_behavior_dataframe(early_predictions_path, label="early", edge_buckets=edge_buckets)
    recent_edge_frame = build_edge_behavior_dataframe(recent_predictions_path, label="recent", edge_buckets=edge_buckets)
    edge_compare_frame = pd.concat([early_edge_frame, recent_edge_frame], ignore_index=True)

    trend_csv = output_dir / "model_regime_trends.csv"
    relationship_csv = output_dir / "model_regime_relationships.csv"
    edge_csv = output_dir / "model_regime_edge_behavior.csv"
    html_path = output_dir / "model_regime_scan.html"
    json_path = output_dir / "model_regime_scan.json"

    trend_frame.to_csv(trend_csv, index=False)
    feature_relationship_frame.to_csv(relationship_csv, index=False)
    edge_compare_frame.to_csv(edge_csv, index=False)

    summary = {
        "training_data_path": str(training_data_path),
        "early_predictions_path": str(early_predictions_path),
        "recent_predictions_path": str(recent_predictions_path),
        "trend_csv": str(trend_csv),
        "relationship_csv": str(relationship_csv),
        "edge_csv": str(edge_csv),
        "top_trend_breaks": trend_frame.sort_values("recent_minus_baseline", key=lambda s: np.abs(s), ascending=False)
        .head(15)
        .to_dict(orient="records"),
        "top_relationship_changes": feature_relationship_frame.head(15).to_dict(orient="records"),
        "edge_bucket_comparison": edge_compare_frame.to_dict(orient="records"),
    }
    html_path.write_text(_build_html(summary, trend_frame, feature_relationship_frame, edge_compare_frame), encoding="utf-8")
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "trend_csv": trend_csv,
        "relationship_csv": relationship_csv,
        "edge_csv": edge_csv,
        "html_path": html_path,
        "json_path": json_path,
    }


def build_season_trend_dataframe(frame: pd.DataFrame, *, trend_columns: Sequence[str]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    seasons = sorted(frame["season"].dropna().astype(int).unique().tolist())
    baseline_mask = frame["season"].between(2021, 2023)
    recent_mask = frame["season"].between(2024, 2025)
    for column in trend_columns:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        season_means = frame.assign(_value=values).groupby("season")["_value"].mean()
        baseline_mean = float(values.loc[baseline_mask].mean())
        recent_mean = float(values.loc[recent_mask].mean())
        record: dict[str, object] = {
            "column": column,
            "baseline_mean_2021_2023": baseline_mean,
            "recent_mean_2024_2025": recent_mean,
            "recent_minus_baseline": float(recent_mean - baseline_mean),
        }
        for season in seasons:
            record[f"season_{season}"] = float(season_means.get(season, np.nan))
        records.append(record)
    return pd.DataFrame.from_records(records)


def build_feature_relationship_dataframe(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str],
    target_column: str,
) -> pd.DataFrame:
    target = pd.to_numeric(frame[target_column], errors="coerce")
    baseline_mask = frame["season"].between(2021, 2023)
    recent_mask = frame["season"].between(2024, 2025)
    records: list[dict[str, object]] = []
    for column in columns:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        baseline_corr = _safe_corr(values.loc[baseline_mask], target.loc[baseline_mask])
        recent_corr = _safe_corr(values.loc[recent_mask], target.loc[recent_mask])
        records.append(
            {
                "column": column,
                "baseline_corr_with_target": baseline_corr,
                "recent_corr_with_target": recent_corr,
                "corr_delta": float(recent_corr - baseline_corr),
            }
        )
    result = pd.DataFrame.from_records(records)
    if result.empty:
        return result
    return result.sort_values("corr_delta", key=lambda s: np.abs(s), ascending=False, kind="stable").reset_index(drop=True)


def build_edge_behavior_dataframe(
    predictions_path: str | Path,
    *,
    label: str,
    edge_buckets: Sequence[float],
) -> pd.DataFrame:
    frame = pd.read_csv(predictions_path)
    bets = frame.loc[frame["is_bet"] == 1].copy()
    if bets.empty:
        return pd.DataFrame()
    bets["bet_edge"] = pd.to_numeric(bets["bet_edge"], errors="coerce")
    bets["bet_profit_units"] = pd.to_numeric(bets["bet_profit_units"], errors="coerce").fillna(0.0)
    bets["bet_stake_units"] = pd.to_numeric(bets["bet_stake_units"], errors="coerce").fillna(0.0)
    bins = sorted({float(value) for value in edge_buckets if float(value) > 0})
    if bins[0] > bets["bet_edge"].min():
        bins = [float(bets["bet_edge"].min())] + bins
    if bins[-1] <= bets["bet_edge"].max():
        bins.append(float(bets["bet_edge"].max()) + 1e-9)
    labels = [f"{bins[index]:.2f}-{bins[index + 1]:.2f}" for index in range(len(bins) - 1)]
    bets["edge_bucket"] = pd.cut(
        bets["bet_edge"],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    )
    grouped = (
        bets.groupby("edge_bucket", observed=True)
        .apply(_summarize_edge_bucket, include_groups=False)
        .reset_index()
    )
    grouped.insert(0, "period", label)
    grouped["edge_bucket"] = grouped["edge_bucket"].astype(str)
    return grouped


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scan model regime changes across data and bet behavior")
    parser.add_argument("--training-data", required=True)
    parser.add_argument("--early-predictions", required=True)
    parser.add_argument("--recent-predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    result = write_model_regime_scan(
        training_data_path=args.training_data,
        early_predictions_path=args.early_predictions,
        recent_predictions_path=args.recent_predictions,
        output_dir=args.output_dir,
    )
    print(json.dumps({key: str(value) for key, value in result.items()}, indent=2, sort_keys=True))
    return 0


def _summarize_edge_bucket(frame: pd.DataFrame) -> pd.Series:
    graded = frame.loc[frame["bet_result"].isin(["WIN", "LOSS"])]
    total_staked = float(frame["bet_stake_units"].sum())
    total_profit = float(frame["bet_profit_units"].sum())
    roi = float(total_profit / total_staked) if total_staked else 0.0
    return pd.Series(
        {
            "bet_count": int(len(frame)),
            "graded_bet_count": int(len(graded)),
            "win_rate": float(graded["bet_result"].eq("WIN").mean()) if len(graded) else 0.0,
            "total_staked_units": total_staked,
            "total_profit_units": total_profit,
            "roi": roi,
            "average_edge": float(frame["bet_edge"].mean()),
            "average_stake_units": float(frame["bet_stake_units"].mean()),
        }
    )


def _safe_corr(x: pd.Series, y: pd.Series) -> float:
    valid = pd.concat([x, y], axis=1).dropna()
    if len(valid) < 50:
        return 0.0
    x_values = valid.iloc[:, 0]
    y_values = valid.iloc[:, 1]
    if float(x_values.std(ddof=0)) < 1e-9 or float(y_values.std(ddof=0)) < 1e-9:
        return 0.0
    return float(x_values.corr(y_values))


def _build_html(
    summary: dict[str, object],
    trend_frame: pd.DataFrame,
    relationship_frame: pd.DataFrame,
    edge_frame: pd.DataFrame,
) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Model Regime Scan</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #111827; background: #f8fafc; }}
    h1, h2 {{ margin-bottom: 12px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; background: white; margin-bottom: 24px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; }}
    th {{ background: #f3f4f6; }}
    tr:nth-child(even) {{ background: #fafafa; }}
  </style>
</head>
<body>
  <h1>Model Regime Scan</h1>
  <h2>Season Trends</h2>
  {trend_frame.to_html(index=False, border=0)}
  <h2>Feature/Target Relationship Changes</h2>
  {relationship_frame.to_html(index=False, border=0)}
  <h2>Edge Bucket Behavior</h2>
  {edge_frame.to_html(index=False, border=0)}
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
