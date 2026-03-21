from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_MODELS_DIR = Path("data") / "models"
DEFAULT_REPORT_DIR = Path("data") / "experiments"


def build_experiment_metrics_dataframe(models_dir: str | Path = DEFAULT_MODELS_DIR) -> pd.DataFrame:
    resolved_models_dir = Path(models_dir)
    rows: list[dict[str, Any]] = []

    for summary_path in sorted(resolved_models_dir.rglob("*_run_*.json")):
        payload = _load_json(summary_path)
        experiment_name = summary_path.parent.name
        summary_name = summary_path.name
        model_version = str(payload.get("model_version", ""))
        run_kind = _classify_summary(summary_name)
        holdout_season = payload.get("holdout_season")
        feature_column_count = len(payload.get("feature_columns", [])) or None
        run_timestamp = _extract_model_timestamp(
            model_version, fallback=summary_path.stat().st_mtime
        )

        for model_name, model_payload in payload.get("models", {}).items():
            target_column = model_payload.get("target_column")
            common = {
                "experiment_name": experiment_name,
                "summary_path": str(summary_path),
                "run_kind": run_kind,
                "model_name": model_name,
                "target_column": target_column,
                "holdout_season": holdout_season,
                "model_version": model_version,
                "data_version_hash": payload.get("data_version_hash"),
                "run_timestamp": run_timestamp,
                "feature_column_count": feature_column_count,
            }

            if run_kind == "training":
                metrics = model_payload.get("holdout_metrics", {})
                rows.append(
                    {
                        **common,
                        "variant": "base",
                        "accuracy": metrics.get("accuracy"),
                        "log_loss": metrics.get("log_loss"),
                        "roc_auc": metrics.get("roc_auc"),
                        "brier": metrics.get("brier"),
                        "ece": metrics.get("ece"),
                        "reliability_gap": metrics.get("reliability_gap"),
                        "comparison_brier_delta": None,
                        "comparison_log_loss_delta": None,
                        "comparison_roc_auc_delta": None,
                        "comparison_accuracy_delta": None,
                    }
                )
            elif run_kind == "stacking":
                metrics = model_payload.get("holdout_metrics", {})
                rows.append(
                    {
                        **common,
                        "variant": "stacked",
                        "accuracy": metrics.get("stacked_accuracy"),
                        "log_loss": metrics.get("stacked_log_loss"),
                        "roc_auc": metrics.get("stacked_roc_auc"),
                        "brier": metrics.get("stacked_brier"),
                        "ece": None,
                        "reliability_gap": None,
                        "comparison_brier_delta": metrics.get("stacked_brier_improvement"),
                        "comparison_log_loss_delta": _subtract(
                            metrics.get("stacked_log_loss"),
                            metrics.get("base_log_loss"),
                        ),
                        "comparison_roc_auc_delta": _subtract(
                            metrics.get("stacked_roc_auc"),
                            metrics.get("base_roc_auc"),
                        ),
                        "comparison_accuracy_delta": _subtract(
                            metrics.get("stacked_accuracy"),
                            metrics.get("base_accuracy"),
                        ),
                    }
                )
            elif run_kind == "calibration":
                metrics = model_payload.get("holdout_metrics", {})
                rows.append(
                    {
                        **common,
                        "variant": str(
                            model_payload.get("calibration_method")
                            or payload.get("calibration_method")
                            or "calibrated"
                        ),
                        "accuracy": metrics.get("calibrated_accuracy"),
                        "log_loss": metrics.get("calibrated_log_loss"),
                        "roc_auc": metrics.get("calibrated_roc_auc"),
                        "brier": metrics.get("calibrated_brier"),
                        "ece": metrics.get("calibrated_ece"),
                        "reliability_gap": metrics.get("max_reliability_gap"),
                        "comparison_brier_delta": metrics.get("brier_improvement"),
                        "comparison_log_loss_delta": _subtract(
                            metrics.get("calibrated_log_loss"),
                            metrics.get("stacked_log_loss"),
                        ),
                        "comparison_roc_auc_delta": _subtract(
                            metrics.get("calibrated_roc_auc"),
                            metrics.get("stacked_roc_auc"),
                        ),
                        "comparison_accuracy_delta": _subtract(
                            metrics.get("calibrated_accuracy"),
                            metrics.get("stacked_accuracy"),
                        ),
                    }
                )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    frame = frame.sort_values(
        by=["holdout_season", "target_column", "variant", "run_timestamp", "experiment_name"],
        kind="stable",
    ).reset_index(drop=True)

    for metric_name in ("accuracy", "log_loss", "roc_auc", "brier", "ece", "reliability_gap"):
        frame[f"delta_vs_prev_{metric_name}"] = frame.groupby(
            ["holdout_season", "target_column", "variant"],
            dropna=False,
        )[metric_name].diff()

    frame["is_best_accuracy"] = _mark_group_best(frame, "accuracy", ascending=False)
    frame["is_best_log_loss"] = _mark_group_best(frame, "log_loss", ascending=True)
    frame["is_best_roc_auc"] = _mark_group_best(frame, "roc_auc", ascending=False)
    frame["is_best_brier"] = _mark_group_best(frame, "brier", ascending=True)
    return frame


def write_experiment_report(
    *,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    output_dir: str | Path = DEFAULT_REPORT_DIR,
) -> dict[str, Path]:
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    metrics = build_experiment_metrics_dataframe(models_dir)
    metrics_csv_path = resolved_output_dir / "experiment_metrics.csv"
    metrics_html_path = resolved_output_dir / "experiment_metrics.html"
    leaderboard_csv_path = resolved_output_dir / "experiment_leaderboard.csv"
    leaderboard_html_path = resolved_output_dir / "experiment_leaderboard.html"
    dashboard_html_path = resolved_output_dir / "experiment_dashboard.html"

    metrics.to_csv(metrics_csv_path, index=False)
    _write_html(metrics, metrics_html_path, title="Experiment Metrics")

    leaderboard = build_leaderboard_dataframe(metrics)
    leaderboard.to_csv(leaderboard_csv_path, index=False)
    _write_html(leaderboard, leaderboard_html_path, title="Experiment Leaderboard")
    _write_dashboard_html(metrics, leaderboard, dashboard_html_path, title="Experiment Dashboard")

    return {
        "metrics_csv": metrics_csv_path,
        "metrics_html": metrics_html_path,
        "leaderboard_csv": leaderboard_csv_path,
        "leaderboard_html": leaderboard_html_path,
        "dashboard_html": dashboard_html_path,
    }


def build_leaderboard_dataframe(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics.copy()

    leaderboard = metrics.copy()
    leaderboard["score_rank"] = leaderboard.groupby(
        ["holdout_season", "target_column", "variant"], dropna=False
    )["roc_auc"].rank(method="dense", ascending=False)
    leaderboard = leaderboard.sort_values(
        by=["holdout_season", "target_column", "variant", "score_rank", "log_loss"],
        kind="stable",
    )
    return leaderboard


def _write_html(frame: pd.DataFrame, path: Path, *, title: str) -> None:
    html = frame.to_html(index=False, border=0)
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    h1 {{ margin-bottom: 16px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; }}
    th {{ background: #f3f4f6; position: sticky; top: 0; }}
    tr:nth-child(even) {{ background: #fafafa; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  {html}
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def _write_dashboard_html(
    metrics: pd.DataFrame,
    leaderboard: pd.DataFrame,
    path: Path,
    *,
    title: str,
) -> None:
    summary_cards = _build_summary_cards(metrics)
    trend_sections = _build_trend_sections(metrics)
    delta_table_html = _build_delta_table(metrics)
    leaderboard_table_html = leaderboard.head(20).to_html(index=False, border=0)

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #111827; background: #f8fafc; }}
    h1, h2, h3 {{ margin: 0 0 12px 0; }}
    p {{ margin: 0 0 12px 0; }}
    .grid {{ display: grid; gap: 16px; }}
    .cards {{ grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); margin-bottom: 24px; }}
    .card, .panel {{ background: white; border: 1px solid #dbe2ea; border-radius: 12px; padding: 16px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }}
    .muted {{ color: #6b7280; font-size: 13px; }}
    .value {{ font-size: 28px; font-weight: 700; margin-top: 8px; }}
    .section {{ margin-top: 24px; }}
    .trend-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 16px; }}
    .chart-grid {{ display: grid; grid-template-columns: repeat(2, minmax(220px, 1fr)); gap: 12px; }}
    .chart-card {{ border: 1px solid #e5e7eb; border-radius: 10px; padding: 12px; }}
    .chart-title {{ font-size: 13px; font-weight: 700; margin-bottom: 8px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 14px; background: white; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; }}
    th {{ background: #f3f4f6; position: sticky; top: 0; }}
    tr:nth-child(even) {{ background: #fafafa; }}
    .good {{ color: #047857; font-weight: 600; }}
    .bad {{ color: #b91c1c; font-weight: 600; }}
    .mono {{ font-family: Consolas, Monaco, monospace; }}
    svg {{ width: 100%; height: 160px; overflow: visible; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p class="muted">Saved-run comparison across training, stacking, and calibration artifacts. Focus on deltas and best-in-group flags, not raw JSON.</p>

  <div class="grid cards">
    {summary_cards}
  </div>

  <div class="section">
    <h2>Metric Trends</h2>
    <div class="trend-grid">
      {trend_sections}
    </div>
  </div>

  <div class="section">
    <h2>Latest Deltas</h2>
    <div class="panel">
      {delta_table_html}
    </div>
  </div>

  <div class="section">
    <h2>Leaderboard Snapshot</h2>
    <div class="panel">
      {leaderboard_table_html}
    </div>
  </div>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def _classify_summary(name: str) -> str:
    if name.startswith("training_run_"):
        return "training"
    if name.startswith("stacking_run_"):
        return "stacking"
    if name.startswith("calibration_run_"):
        return "calibration"
    raise ValueError(f"Unsupported summary file: {name}")


def _extract_model_timestamp(model_version: str, *, fallback: float) -> str:
    if model_version and len(model_version) >= 16 and "T" in model_version:
        return model_version.split("_", 1)[0]
    return str(fallback)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _subtract(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _mark_group_best(frame: pd.DataFrame, metric_name: str, *, ascending: bool) -> pd.Series:
    eligible = frame[metric_name].notna()
    grouped = frame.loc[eligible].groupby(
        ["holdout_season", "target_column", "variant"],
        dropna=False,
    )[metric_name]
    best = grouped.transform("min" if ascending else "max")
    result = pd.Series(False, index=frame.index)
    result.loc[eligible] = frame.loc[eligible, metric_name].eq(best)
    return result


def _build_summary_cards(metrics: pd.DataFrame) -> str:
    if metrics.empty:
        return _panel_card("No Experiments", "0", "No saved experiments found.")

    latest_run = metrics.sort_values("run_timestamp", kind="stable").iloc[-1]
    best_auc = (
        metrics.loc[metrics["roc_auc"].notna()].sort_values("roc_auc", ascending=False).iloc[0]
    )
    best_log_loss = (
        metrics.loc[metrics["log_loss"].notna()].sort_values("log_loss", ascending=True).iloc[0]
    )
    latest_improvement = metrics.loc[metrics["delta_vs_prev_roc_auc"].notna()].sort_values(
        "run_timestamp",
        kind="stable",
    )
    latest_delta = (
        latest_improvement.iloc[-1]["delta_vs_prev_roc_auc"]
        if not latest_improvement.empty
        else None
    )

    cards = [
        _panel_card(
            "Latest Experiment",
            str(latest_run["experiment_name"]),
            f"{latest_run['run_kind']} / {latest_run['variant']} / {latest_run['target_column']}",
        ),
        _panel_card(
            "Best ROC AUC",
            _fmt(best_auc["roc_auc"]),
            f"{best_auc['experiment_name']} | {best_auc['target_column']} | {best_auc['variant']}",
        ),
        _panel_card(
            "Best Log Loss",
            _fmt(best_log_loss["log_loss"]),
            f"{best_log_loss['experiment_name']} | {best_log_loss['target_column']} | {best_log_loss['variant']}",
        ),
        _panel_card(
            "Latest ROC AUC Delta",
            _fmt_signed(latest_delta),
            "Change vs previous experiment in same holdout/target/variant lane",
            value_class=_delta_class(latest_delta, higher_is_better=True),
        ),
    ]
    return "".join(cards)


def _build_trend_sections(metrics: pd.DataFrame) -> str:
    if metrics.empty:
        return '<div class="panel">No metrics available.</div>'

    sections: list[str] = []
    grouped = metrics.groupby(
        ["holdout_season", "target_column", "variant"], dropna=False, sort=True
    )
    for (holdout_season, target_column, variant), group in grouped:
        ordered = group.sort_values("run_timestamp", kind="stable")
        charts = []
        for metric_name, higher_is_better in (
            ("roc_auc", True),
            ("log_loss", False),
            ("accuracy", True),
            ("brier", False),
        ):
            metric_frame = ordered[ordered[metric_name].notna()].copy()
            if metric_frame.empty:
                continue
            charts.append(
                f"""
                <div class="chart-card">
                  <div class="chart-title">{metric_name.upper()}</div>
                  {_render_metric_svg(metric_frame, metric_name, higher_is_better=higher_is_better)}
                </div>
                """
            )

        sections.append(
            f"""
            <div class="panel">
              <h3>{holdout_season} | {target_column} | {variant}</h3>
              <p class="muted">{len(ordered)} saved runs in this comparison lane.</p>
              <div class="chart-grid">
                {"".join(charts)}
              </div>
            </div>
            """
        )
    return "".join(sections)


def _build_delta_table(metrics: pd.DataFrame) -> str:
    delta_columns = [
        "experiment_name",
        "run_kind",
        "variant",
        "target_column",
        "holdout_season",
        "roc_auc",
        "delta_vs_prev_roc_auc",
        "log_loss",
        "delta_vs_prev_log_loss",
        "accuracy",
        "delta_vs_prev_accuracy",
        "brier",
        "delta_vs_prev_brier",
    ]
    latest = (
        metrics.sort_values("run_timestamp", ascending=False, kind="stable")[delta_columns]
        .head(20)
        .copy()
    )
    for column in (
        "roc_auc",
        "delta_vs_prev_roc_auc",
        "log_loss",
        "delta_vs_prev_log_loss",
        "accuracy",
        "delta_vs_prev_accuracy",
        "brier",
        "delta_vs_prev_brier",
    ):
        latest[column] = latest[column].map(_fmt_signed if column.startswith("delta_") else _fmt)
    return latest.to_html(index=False, border=0, escape=False)


def _render_metric_svg(frame: pd.DataFrame, metric_name: str, *, higher_is_better: bool) -> str:
    width = 420.0
    height = 140.0
    padding_left = 42.0
    padding_right = 12.0
    padding_top = 12.0
    padding_bottom = 28.0
    inner_width = width - padding_left - padding_right
    inner_height = height - padding_top - padding_bottom

    values = [float(value) for value in frame[metric_name]]
    labels = [str(value) for value in frame["experiment_name"]]
    x_positions = [
        padding_left + (inner_width * index / max(1, len(values) - 1))
        for index in range(len(values))
    ]
    min_value = min(values)
    max_value = max(values)
    if min_value == max_value:
        min_value -= 0.01
        max_value += 0.01

    def scale_y(value: float) -> float:
        normalized = (value - min_value) / (max_value - min_value)
        return padding_top + (1.0 - normalized) * inner_height

    points = [(x, scale_y(value)) for x, value in zip(x_positions, values, strict=True)]
    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    last_value = values[-1]
    delta_value = None
    if len(values) > 1:
        delta_value = values[-1] - values[-2]
    color = "#0f766e" if higher_is_better else "#1d4ed8"

    point_markup = []
    for (x, y), label, value in zip(points, labels, values, strict=True):
        point_markup.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}">'
            f"<title>{label}: {metric_name}={value:.4f}</title></circle>"
        )

    x_label_markup = []
    for x, label in zip(x_positions, labels, strict=True):
        short_label = label[:18]
        x_label_markup.append(
            f'<text x="{x:.1f}" y="{height - 8:.1f}" font-size="10" text-anchor="middle" fill="#6b7280">{short_label}</text>'
        )

    delta_fill = "#047857" if (delta_value or 0.0) >= 0 else "#b91c1c"
    return f"""
    <svg viewBox="0 0 {width:.0f} {height:.0f}" role="img" aria-label="{metric_name} trend">
      <line x1="{padding_left:.1f}" y1="{padding_top:.1f}" x2="{padding_left:.1f}" y2="{height - padding_bottom:.1f}" stroke="#d1d5db" />
      <line x1="{padding_left:.1f}" y1="{height - padding_bottom:.1f}" x2="{width - padding_right:.1f}" y2="{height - padding_bottom:.1f}" stroke="#d1d5db" />
      <polyline fill="none" stroke="{color}" stroke-width="2.5" points="{polyline}" />
      {"".join(point_markup)}
      {"".join(x_label_markup)}
      <text x="{padding_left:.1f}" y="10" font-size="11" fill="#6b7280">min {_fmt(min_value)}</text>
      <text x="{width - padding_right:.1f}" y="10" font-size="11" text-anchor="end" fill="#6b7280">max {_fmt(max_value)}</text>
      <text x="{width - padding_right:.1f}" y="24" font-size="12" text-anchor="end" fill="{delta_fill}" font-weight="700">latest {_fmt(last_value)} ({_fmt_signed(delta_value)})</text>
    </svg>
    """


def _panel_card(title: str, value: str, detail: str, *, value_class: str = "") -> str:
    class_attr = f" value {value_class}".strip()
    return f"""
    <div class="card">
      <div class="muted">{title}</div>
      <div class="{class_attr}">{value}</div>
      <div class="muted">{detail}</div>
    </div>
    """


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return "-"
    return f"{float(value):.{digits}f}"


def _fmt_signed(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return "-"
    value = float(value)
    return f"{value:+.{digits}f}"


def _delta_class(value: Any, *, higher_is_better: bool) -> str:
    if value is None or value == "":
        return ""
    numeric = float(value)
    positive = numeric >= 0
    is_good = positive if higher_is_better else not positive
    return "good" if is_good else "bad"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate experiment comparison reports from saved run artifacts"
    )
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_REPORT_DIR))
    args = parser.parse_args()

    outputs = write_experiment_report(models_dir=args.models_dir, output_dir=args.output_dir)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
