from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd


DEFAULT_BUCKET_BREAKS = (0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 1.00)


def build_edge_bucket_dataframe(
    predictions_path: str | Path,
    *,
    bucket_breaks: Sequence[float] = DEFAULT_BUCKET_BREAKS,
) -> pd.DataFrame:
    frame = pd.read_csv(predictions_path)
    if frame.empty:
        return pd.DataFrame()

    bets = frame.loc[frame["is_bet"] == 1].copy()
    if bets.empty:
        return pd.DataFrame()

    scheduled_start = pd.to_datetime(bets["scheduled_start"], utc=True)
    bets["season"] = scheduled_start.dt.year
    bets["bet_edge"] = pd.to_numeric(bets["bet_edge"], errors="coerce")
    bets["bet_stake_units"] = pd.to_numeric(bets["bet_stake_units"], errors="coerce").fillna(0.0)
    bets["bet_profit_units"] = pd.to_numeric(bets["bet_profit_units"], errors="coerce").fillna(0.0)

    bins = sorted({float(value) for value in bucket_breaks if float(value) > 0})
    if not bins:
        raise ValueError("bucket_breaks must include positive values")
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
    bets["graded"] = bets["bet_result"].isin(["WIN", "LOSS"])
    bets["win"] = bets["bet_result"].eq("WIN").astype(int)

    grouped = (
        bets.groupby(["edge_bucket"], observed=True)
        .apply(_summarize_bucket, include_groups=False)
        .reset_index()
    )
    grouped["season"] = "all"

    by_season = (
        bets.groupby(["season", "edge_bucket"], observed=True)
        .apply(_summarize_bucket, include_groups=False)
        .reset_index()
    )

    combined = pd.concat([grouped, by_season], ignore_index=True)
    combined["edge_bucket"] = combined["edge_bucket"].astype(str)
    combined = combined.sort_values(["season", "edge_bucket"], kind="stable").reset_index(drop=True)
    return combined


def write_edge_bucket_report(
    predictions_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    bucket_breaks: Sequence[float] = DEFAULT_BUCKET_BREAKS,
) -> dict[str, Path]:
    resolved_predictions_path = Path(predictions_path)
    resolved_output_dir = Path(output_dir) if output_dir is not None else resolved_predictions_path.parent
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    frame = build_edge_bucket_dataframe(resolved_predictions_path, bucket_breaks=bucket_breaks)
    parent_slug = resolved_predictions_path.parent.name.replace(" ", "_")
    stem = (
        resolved_predictions_path.stem.replace("walk_forward_predictions", "edge_bucket_report")
        + f"_{parent_slug}"
    )
    csv_path = resolved_output_dir / f"{stem}.csv"
    html_path = resolved_output_dir / f"{stem}.html"
    json_path = resolved_output_dir / f"{stem}.json"

    frame.to_csv(csv_path, index=False)
    html_path.write_text(_build_html(frame, title=f"Edge Bucket Report: {resolved_predictions_path.name}"), encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "predictions_path": str(resolved_predictions_path),
                "bucket_breaks": [float(value) for value in bucket_breaks],
                "row_count": int(len(frame)),
                "csv_path": str(csv_path),
                "html_path": str(html_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {"csv_path": csv_path, "html_path": html_path, "json_path": json_path}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize walk-forward edge buckets")
    parser.add_argument("--predictions-path", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--bucket-breaks", nargs="*", type=float)
    args = parser.parse_args(argv)

    result = write_edge_bucket_report(
        predictions_path=args.predictions_path,
        output_dir=args.output_dir,
        bucket_breaks=args.bucket_breaks or DEFAULT_BUCKET_BREAKS,
    )
    print(
        json.dumps(
            {key: str(value) for key, value in result.items()},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _summarize_bucket(frame: pd.DataFrame) -> pd.Series:
    graded = frame.loc[frame["graded"]]
    total_bets = int(len(frame))
    total_staked = float(frame["bet_stake_units"].sum())
    total_profit = float(frame["bet_profit_units"].sum())
    win_count = int(frame["bet_result"].eq("WIN").sum())
    loss_count = int(frame["bet_result"].eq("LOSS").sum())
    push_count = int(frame["bet_result"].eq("PUSH").sum())
    roi = float(total_profit / total_staked) if total_staked else 0.0
    win_rate = float(win_count / len(graded)) if len(graded) else 0.0
    return pd.Series(
        {
            "bet_count": total_bets,
            "graded_bet_count": int(len(graded)),
            "win_count": win_count,
            "loss_count": loss_count,
            "push_count": push_count,
            "win_rate": win_rate,
            "total_staked_units": total_staked,
            "total_profit_units": total_profit,
            "roi": roi,
            "average_stake_units": float(frame["bet_stake_units"].mean()) if total_bets else 0.0,
            "average_edge": float(frame["bet_edge"].mean()) if total_bets else 0.0,
            "median_edge": float(frame["bet_edge"].median()) if total_bets else 0.0,
        }
    )


def _build_html(frame: pd.DataFrame, *, title: str) -> str:
    table_html = frame.to_html(index=False, border=0)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #111827; background: #f8fafc; }}
    h1 {{ margin-bottom: 16px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 14px; background: white; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; }}
    th {{ background: #f3f4f6; }}
    tr:nth-child(even) {{ background: #fafafa; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  {table_html}
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
