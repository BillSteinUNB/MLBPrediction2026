from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Force UTF-8 on Windows so rich Unicode characters render properly.
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

REPORT_PATTERNS = {
    "training": "training_run_*.json",
    "stacking": "stacking_run_*.json",
    "calibration": "calibration_run_*.json",
    "walk_forward": "walk_forward_summary_*.json",
}


def _discover_report_files(run_dir: Path) -> dict[str, Path]:
    discovered: dict[str, Path] = {}
    for report_type, pattern in REPORT_PATTERNS.items():
        matches = sorted(run_dir.glob(pattern))
        if matches:
            discovered[report_type] = matches[-1]
    return discovered


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _fmt_float(value: Any, digits: int = 4, empty: str = "-") -> str:
    if value is None:
        return empty
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return empty


def _metric_style(value: Any, high: float, mid: float) -> str:
    if value is None:
        return "dim"
    v = float(value)
    if v > high:
        return "green"
    if v > mid:
        return "yellow"
    return "red"


def _delta_text(value: Any, digits: int = 4) -> Text:
    text = Text(_fmt_float(value, digits))
    if value is None:
        text.stylize("dim")
        return text
    v = float(value)
    text.stylize("green" if v >= 0 else "red")
    return text


def _bar_from_ratio(ratio: float, width: int = 28) -> str:
    ratio = max(0.0, min(1.0, ratio))
    full_blocks = int(ratio * width)
    remainder = ratio * width - full_blocks
    partials = "▏▎▍▌▋▊▉"
    partial = ""
    if remainder > 0 and full_blocks < width:
        idx = min(len(partials) - 1, int(remainder * 8))
        if idx > 0:
            partial = partials[idx - 1]
    consumed = full_blocks + (1 if partial else 0)
    return ("█" * full_blocks) + partial + (" " * max(0, width - consumed))


def _reliability_bar(pred: float, actual: float, width: int = 22) -> str:
    pred_blocks = int(max(0.0, min(1.0, pred)) * width)
    actual_blocks = int(max(0.0, min(1.0, actual)) * width)
    low = min(pred_blocks, actual_blocks)
    high = max(pred_blocks, actual_blocks)
    left = "▓" * low
    gap = "▓" * (high - low) if pred_blocks >= actual_blocks else "░" * (high - low)
    return left + gap + (" " * max(0, width - high))


def _as_run_name(run_dir: Path) -> str:
    return run_dir.name or str(run_dir)


def _render_training(console: Console, run_name: str, data: dict[str, Any]) -> int:
    model_version = data.get("model_version", "-")
    holdout = data.get("holdout_season", "-")
    console.print(
        Panel(
            f"[bold]TRAINING RUN REPORT[/bold]\nRun: [cyan]{run_name}[/cyan]\n"
            f"Version: [magenta]{model_version}[/magenta] | Holdout: [yellow]{holdout}[/yellow]",
            border_style="bright_blue",
            box=box.DOUBLE,
        )
    )

    models = data.get("models", {})
    metrics = Table(title="Model Metrics", box=box.SIMPLE_HEAVY)
    metrics.add_column("Model", style="bold cyan")
    metrics.add_column("Accuracy", justify="right")
    metrics.add_column("Log Loss", justify="right")
    metrics.add_column("ROC AUC", justify="right")
    metrics.add_column("CV Log Loss", justify="right")
    metrics.add_column("Train Rows", justify="right")
    metrics.add_column("Holdout Rows", justify="right")

    for model_key, model_info in models.items():
        hold = model_info.get("holdout_metrics", {})
        acc = hold.get("accuracy")
        auc = hold.get("roc_auc")
        acc_style = _metric_style(acc, high=0.56, mid=0.54)
        auc_style = _metric_style(auc, high=0.58, mid=0.55)
        metrics.add_row(
            model_key,
            Text(_fmt_float(acc), style=acc_style),
            _fmt_float(hold.get("log_loss")),
            Text(_fmt_float(auc), style=auc_style),
            _fmt_float(model_info.get("cv_best_log_loss")),
            f"{model_info.get('train_row_count', '-'):,}"
            if model_info.get("train_row_count") is not None
            else "-",
            f"{model_info.get('holdout_row_count', '-'):,}"
            if model_info.get("holdout_row_count") is not None
            else "-",
        )
    console.print(metrics)

    console.print(Rule("Hyperparameters", style="bright_blue"))
    hp_table = Table(box=box.MINIMAL)
    hp_table.add_column("Model", style="cyan")
    hp_table.add_column("Best Params")
    for model_key, model_info in models.items():
        best_params = model_info.get("best_params", {})
        param_text = ", ".join(f"{k}={v}" for k, v in sorted(best_params.items())) or "-"
        hp_table.add_row(model_key, param_text)
    console.print(hp_table)

    console.print(Rule("Feature Importance", style="bright_blue"))
    for model_key, model_info in models.items():
        fi = model_info.get("feature_importance_rankings", [])[:15]
        fi_table = Table(title=model_key, box=box.SIMPLE)
        fi_table.add_column("#", justify="right")
        fi_table.add_column("Feature", overflow="fold")
        fi_table.add_column("Importance", justify="right")
        fi_table.add_column("Bar")
        max_importance = max((float(row.get("importance", 0.0)) for row in fi), default=1.0)
        for idx, row in enumerate(fi, start=1):
            imp = float(row.get("importance", 0.0))
            bar = _bar_from_ratio(imp / max_importance if max_importance else 0.0)
            fi_table.add_row(
                str(idx), str(row.get("feature", "-")), _fmt_float(imp), Text(bar, style="green")
            )
        console.print(fi_table)
    return len(models)


def _render_stacking(console: Console, data: dict[str, Any]) -> None:
    console.print(Panel("[bold]STACKING RUN REPORT[/bold]", border_style="cyan", box=box.ROUNDED))
    table = Table(title="Stacking Impact", box=box.SIMPLE_HEAVY)
    table.add_column("Model", style="bold cyan")
    table.add_column("Base Brier", justify="right")
    table.add_column("Stacked Brier", justify="right")
    table.add_column("Improvement", justify="right")
    table.add_column("Base AUC", justify="right")
    table.add_column("Stacked AUC", justify="right")
    table.add_column("Meta Features")

    for model_key, model_info in data.get("models", {}).items():
        hold = model_info.get("holdout_metrics", {})
        features = model_info.get("meta_feature_columns", [])
        table.add_row(
            model_key,
            _fmt_float(hold.get("base_brier")),
            _fmt_float(hold.get("stacked_brier")),
            _delta_text(hold.get("stacked_brier_improvement")),
            _fmt_float(hold.get("base_roc_auc")),
            _fmt_float(hold.get("stacked_roc_auc")),
            ", ".join(features) if features else "-",
        )
    console.print(table)


def _render_calibration(console: Console, data: dict[str, Any]) -> int:
    method = data.get("calibration_method", "-")
    console.print(
        Panel(
            f"[bold]CALIBRATION RUN REPORT[/bold]\nMethod: [magenta]{method}[/magenta]",
            border_style="magenta",
            box=box.ROUNDED,
        )
    )

    table = Table(title="Calibration Metrics", box=box.SIMPLE_HEAVY)
    table.add_column("Model", style="bold cyan")
    table.add_column("Stacked Brier", justify="right")
    table.add_column("Calibrated Brier", justify="right")
    table.add_column("Improvement", justify="right")
    table.add_column("Stacked ECE", justify="right")
    table.add_column("Calibrated ECE", justify="right")
    table.add_column("Method", justify="center")

    models = data.get("models", {})
    for model_key, model_info in models.items():
        hold = model_info.get("holdout_metrics", {})
        table.add_row(
            model_key,
            _fmt_float(hold.get("stacked_brier")),
            _fmt_float(hold.get("calibrated_brier")),
            _delta_text(hold.get("brier_improvement")),
            _fmt_float(hold.get("stacked_ece")),
            _fmt_float(hold.get("calibrated_ece")),
            method,
        )
    console.print(table)

    for model_key, model_info in models.items():
        hold = model_info.get("holdout_metrics", {})
        gates = hold.get("quality_gates", {})
        gate_table = Table(title=f"Quality Gates: {model_key}", box=box.MINIMAL)
        gate_table.add_column("Gate", style="cyan")
        gate_table.add_column("Result", justify="center")
        for gate_name in ("brier_lt_0_25", "ece_lt_0_05", "reliability_gap_le_0_05"):
            passed = bool(gates.get(gate_name, False))
            gate_table.add_row(
                gate_name,
                Text("✓ PASS" if passed else "✗ FAIL", style="green" if passed else "red"),
            )
        console.print(gate_table)

        bins = [b for b in hold.get("reliability_diagram", []) if b.get("count", 0) > 0]
        rel_table = Table(title=f"Reliability Diagram (ASCII): {model_key}", box=box.SIMPLE)
        rel_table.add_column("Bin")
        rel_table.add_column("Pred", justify="right")
        rel_table.add_column("Actual", justify="right")
        rel_table.add_column("Gap", justify="right")
        rel_table.add_column("Visual", style="bright_white")
        for bin_row in bins:
            pred = float(bin_row.get("mean_predicted_probability", 0.0))
            actual = float(bin_row.get("empirical_positive_rate", 0.0))
            gap = float(bin_row.get("absolute_gap", abs(pred - actual)))
            label = f"[{bin_row.get('bin_lower', 0):.1f}-{bin_row.get('bin_upper', 0):.1f}]"
            rel_table.add_row(
                label,
                _fmt_float(pred, digits=3),
                _fmt_float(actual, digits=3),
                _fmt_float(gap, digits=3),
                _reliability_bar(pred, actual),
            )
        console.print(rel_table)
    return len(models)


def _render_walk_forward(console: Console, data: dict[str, Any]) -> None:
    roi = data.get("aggregate_roi")
    roi_style = "green" if (roi is not None and float(roi) >= 0) else "red"
    summary_lines = [
        f"Date Range: [cyan]{data.get('start_date', '-')}[/cyan] -> [cyan]{data.get('end_date', '-')}[/cyan]",
        f"Window Count: [bold]{data.get('window_count', '-')}[/bold] | Total Bets: [bold]{data.get('total_bets', '-')}[/bold]",
        f"Total Profit: [bold]{_fmt_float(data.get('total_profit_units'), digits=2)}[/bold]u",
        f"ROI: [{roi_style}]{_fmt_float(roi, digits=4)}[/{roi_style}]",
        f"Edge Threshold: {data.get('edge_threshold', '-')} | Brier Score: {_fmt_float(data.get('aggregate_brier_score'))}",
    ]
    console.print(
        Panel(
            "\n".join(summary_lines),
            title="WALK-FORWARD SUMMARY",
            border_style="green",
            box=box.DOUBLE_EDGE,
        )
    )

    windows = Table(title="Window Builds", box=box.SIMPLE_HEAVY)
    windows.add_column("Window", justify="right")
    windows.add_column("Action")
    windows.add_column("Train Rows", justify="right")
    windows.add_column("Test Rows", justify="right")
    windows.add_column("Data Hash")
    for row in data.get("window_builds", []):
        data_hash = str(row.get("data_version_hash", "-"))
        windows.add_row(
            str(row.get("window_index", "-")),
            str(row.get("build_action", "-")),
            f"{row.get('train_row_count', '-'):,}"
            if row.get("train_row_count") is not None
            else "-",
            f"{row.get('test_row_count', '-'):,}" if row.get("test_row_count") is not None else "-",
            data_hash[:10],
        )
    console.print(windows)

    params = Table(title="Estimator Params", box=box.MINIMAL)
    params.add_column("Param", style="cyan")
    params.add_column("Value")
    for k, v in sorted(data.get("estimator_kwargs", {}).items()):
        params.add_row(str(k), str(v))
    console.print(params)


def _extract_compare_metrics(run_name: str, discovered: dict[str, Path]) -> dict[str, Any]:
    row = {
        "run": run_name,
        "ml_auc": None,
        "rl_auc": None,
        "ml_brier": None,
        "rl_brier": None,
        "stacking_delta_ml": None,
        "calibration_delta_ml": None,
    }
    if "training" in discovered:
        training = _load_json(discovered["training"])
        ml = training.get("models", {}).get("f5_ml_model", {}).get("holdout_metrics", {})
        rl = training.get("models", {}).get("f5_rl_model", {}).get("holdout_metrics", {})
        row["ml_auc"] = ml.get("roc_auc")
        row["rl_auc"] = rl.get("roc_auc")
    if "stacking" in discovered:
        stacking = _load_json(discovered["stacking"])
        ml = stacking.get("models", {}).get("f5_ml_stacking_model", {}).get("holdout_metrics", {})
        rl = stacking.get("models", {}).get("f5_rl_stacking_model", {}).get("holdout_metrics", {})
        row["ml_brier"] = ml.get("base_brier")
        row["rl_brier"] = rl.get("base_brier")
        row["stacking_delta_ml"] = ml.get("stacked_brier_improvement")
    if "calibration" in discovered:
        calibration = _load_json(discovered["calibration"])
        ml = (
            calibration.get("models", {})
            .get("f5_ml_calibrated_model", {})
            .get("holdout_metrics", {})
        )
        row["calibration_delta_ml"] = ml.get("brier_improvement")
    return row


def show_compare_report(run_dirs: list[str | Path], console: Console | None = None) -> None:
    console = console or Console()
    rows: list[dict[str, Any]] = []
    for path in run_dirs:
        run_dir = Path(path).expanduser().resolve()
        if not run_dir.exists() or not run_dir.is_dir():
            continue
        rows.append(
            _extract_compare_metrics(_as_run_name(run_dir), _discover_report_files(run_dir))
        )

    if not rows:
        console.print("[red]No valid run directories for comparison.[/red]")
        return

    console.print(Panel("[bold]RUN COMPARISON[/bold]", border_style="yellow", box=box.DOUBLE))
    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("Run", style="bold cyan")
    table.add_column("ML AUC", justify="right")
    table.add_column("RL AUC", justify="right")
    table.add_column("ML Brier", justify="right")
    table.add_column("RL Brier", justify="right")
    table.add_column("Stacking Δ ML", justify="right")
    table.add_column("Calibration Δ ML", justify="right")

    for row in rows:
        table.add_row(
            str(row["run"]),
            _fmt_float(row["ml_auc"]),
            _fmt_float(row["rl_auc"]),
            _fmt_float(row["ml_brier"]),
            _fmt_float(row["rl_brier"]),
            _delta_text(row["stacking_delta_ml"]),
            _delta_text(row["calibration_delta_ml"]),
        )
    console.print(table)


def show_run_report(run_dir: str | Path, console: Console | None = None) -> dict[str, Any]:
    console = console or Console()
    run_path = Path(run_dir).expanduser().resolve()
    run_name = _as_run_name(run_path)
    if not run_path.exists() or not run_path.is_dir():
        console.print(f"[red]Run directory not found:[/red] {run_path}")
        return {"run": run_name, "trained_models": 0, "calibrated_models": 0}

    discovered = _discover_report_files(run_path)
    console.print(
        Panel(
            "[bold bright_white]MLB PREDICTION -- RUN REPORT[/bold bright_white]",
            border_style="bright_white",
            box=box.HEAVY,
        )
    )
    console.print(f"[dim]Run directory:[/dim] {run_path}")
    console.print(Rule(style="bright_white"))

    trained_models = 0
    calibrated_models = 0

    if "training" in discovered:
        trained_models = _render_training(console, run_name, _load_json(discovered["training"]))
        console.print(Rule(style="bright_white"))
    if "stacking" in discovered:
        _render_stacking(console, _load_json(discovered["stacking"]))
        console.print(Rule(style="bright_white"))
    if "calibration" in discovered:
        calibrated_models = _render_calibration(console, _load_json(discovered["calibration"]))
        console.print(Rule(style="bright_white"))
    if "walk_forward" in discovered:
        _render_walk_forward(console, _load_json(discovered["walk_forward"]))
        console.print(Rule(style="bright_white"))

    if not discovered:
        console.print("[yellow]No report JSON files discovered in run directory.[/yellow]")

    console.print(
        Panel(
            f"[bold green]Report complete.[/bold green] "
            f"{trained_models} models trained, {calibrated_models} calibrated.",
            border_style="green",
            box=box.ROUNDED,
        )
    )

    return {
        "run": run_name,
        "trained_models": trained_models,
        "calibrated_models": calibrated_models,
        "discovered": discovered,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render MLB pipeline run reports")
    parser.add_argument("run_dirs", nargs="+", help="One or more run directories")
    parser.add_argument("--compare", action="store_true", help="Show cross-run comparison table")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    console = Console(force_terminal=True)
    for path in args.run_dirs:
        show_run_report(path, console=console)
    if args.compare and len(args.run_dirs) > 1:
        console.print(Rule("Cross-Run Comparison", style="yellow"))
        show_compare_report(args.run_dirs, console=console)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
