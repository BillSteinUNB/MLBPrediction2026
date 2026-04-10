from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_COUNT_TRACKING_DIR = Path("data/reports/run_count/tracker")
DEFAULT_RUN_COUNT_HISTORY_JSONL = DEFAULT_RUN_COUNT_TRACKING_DIR / "run_history.jsonl"
DEFAULT_RUN_COUNT_HISTORY_CSV = DEFAULT_RUN_COUNT_TRACKING_DIR / "run_history.csv"
DEFAULT_RUN_COUNT_INDEX_JSON = DEFAULT_RUN_COUNT_TRACKING_DIR / "run_history_index.json"
DEFAULT_RUN_COUNT_LATEST_JSON = DEFAULT_RUN_COUNT_TRACKING_DIR / "latest_run.json"
DEFAULT_RUN_COUNT_BENCHMARK_JSON = DEFAULT_RUN_COUNT_TRACKING_DIR / "benchmark.json"
DEFAULT_RUN_COUNT_CONNECTION_JSON = DEFAULT_RUN_COUNT_TRACKING_DIR / "frontend_connection.json"
DEFAULT_RUN_COUNT_CONNECTION_MD = DEFAULT_RUN_COUNT_TRACKING_DIR / "FRONTEND_CONNECTION.md"


def record_run_count_workflow_run(
    *,
    stage3_report_path: str | Path,
    stage3_vs_control_path: str | Path,
    stage4_report_path: str | Path,
    stage4_vs_control_path: str | Path,
    stage4_vs_stage3_path: str | Path,
    stage3_walk_forward_path: str | Path,
    stage4_walk_forward_path: str | Path,
    dual_view_path: str | Path,
    training_data_path: str | Path,
    start_year: int,
    end_year: int,
    holdout_season: int,
    folds: int,
    feature_selection_mode: str,
    forced_delta_count: int,
    xgb_workers: int,
    enable_market_priors: bool,
    historical_odds_db: str | None,
    historical_market_book: str | None,
    mu_delta_mode: str,
    stage3_experiment: str,
    stage4_experiment: str,
    stage3_research_lane_name: str,
    stage4_research_lane_name: str,
    tracking_dir: str | Path = DEFAULT_RUN_COUNT_TRACKING_DIR,
    hypothesis: str | None = None,
    run_label: str | None = None,
    set_as_benchmark: bool = False,
    benchmark_label: str | None = None,
) -> dict[str, Any]:
    resolved_tracking_dir = _resolve_path(tracking_dir)
    resolved_tracking_dir.mkdir(parents=True, exist_ok=True)

    stage3_report = _read_json(stage3_report_path)
    stage3_vs_control = _read_json(stage3_vs_control_path)
    stage4_report = _read_json(stage4_report_path)
    stage4_vs_control = _read_json(stage4_vs_control_path)
    stage4_vs_stage3 = _read_json(stage4_vs_stage3_path)
    stage3_walk_forward = _read_json(stage3_walk_forward_path)
    stage4_walk_forward = _read_json(stage4_walk_forward_path)
    dual_view = _read_json(dual_view_path)

    tracked_at = datetime.now(UTC).isoformat()
    run_id = f"{stage3_report.get('model_version', 'stage3')}__{stage4_report.get('model_version', 'stage4')}"

    benchmark_payload = _load_existing_benchmark(resolved_tracking_dir)
    benchmark_exists = benchmark_payload is not None
    should_set_benchmark = bool(set_as_benchmark or not benchmark_exists)

    record = {
        "run_type": "run_count_workflow",
        "run_id": run_id,
        "tracked_at": tracked_at,
        "run_label": run_label or stage4_experiment,
        "hypothesis": hypothesis or "No explicit hypothesis recorded.",
        "benchmark_status": "candidate",
        "benchmark_label": None,
        "workflow_config": {
            "training_data_path": _relative_to_project(_resolve_path(training_data_path)),
            "start_year": int(start_year),
            "end_year": int(end_year),
            "holdout_season": int(holdout_season),
            "folds": int(folds),
            "feature_selection_mode": str(feature_selection_mode),
            "forced_delta_count": int(forced_delta_count),
            "xgb_workers": int(xgb_workers),
            "enable_market_priors": bool(enable_market_priors),
            "historical_odds_db": historical_odds_db,
            "historical_market_book": historical_market_book,
            "mu_delta_mode": str(mu_delta_mode),
            "stage3_experiment": str(stage3_experiment),
            "stage4_experiment": str(stage4_experiment),
            "stage3_research_lane_name": str(stage3_research_lane_name),
            "stage4_research_lane_name": str(stage4_research_lane_name),
        },
        "artifacts": {
            "stage3_report_path": _relative_to_project(_resolve_path(stage3_report_path)),
            "stage3_vs_control_path": _relative_to_project(_resolve_path(stage3_vs_control_path)),
            "stage4_report_path": _relative_to_project(_resolve_path(stage4_report_path)),
            "stage4_vs_control_path": _relative_to_project(_resolve_path(stage4_vs_control_path)),
            "stage4_vs_stage3_path": _relative_to_project(_resolve_path(stage4_vs_stage3_path)),
            "stage3_walk_forward_path": _relative_to_project(_resolve_path(stage3_walk_forward_path)),
            "stage4_walk_forward_path": _relative_to_project(_resolve_path(stage4_walk_forward_path)),
            "dual_view_path": _relative_to_project(_resolve_path(dual_view_path)),
        },
        "summary": _build_summary_block(
            stage3_report=stage3_report,
            stage3_vs_control=stage3_vs_control,
            stage4_report=stage4_report,
            stage4_vs_control=stage4_vs_control,
            stage4_vs_stage3=stage4_vs_stage3,
            stage3_walk_forward=stage3_walk_forward,
            stage4_walk_forward=stage4_walk_forward,
            dual_view=dual_view,
            hypothesis=hypothesis,
            historical_odds_db=historical_odds_db,
            historical_market_book=historical_market_book,
            enable_market_priors=enable_market_priors,
            mu_delta_mode=mu_delta_mode,
        ),
    }

    benchmark_compare = _build_benchmark_compare(record, benchmark_payload)
    record["benchmark_compare"] = benchmark_compare

    if should_set_benchmark:
        label = benchmark_label or "benchmark_v1"
        record["benchmark_status"] = "benchmark"
        record["benchmark_label"] = label
        benchmark_snapshot = {
            "benchmark_label": label,
            "benchmark_run_id": run_id,
            "tracked_at": tracked_at,
            "summary": record["summary"],
            "artifacts": record["artifacts"],
        }
        (resolved_tracking_dir / DEFAULT_RUN_COUNT_BENCHMARK_JSON.name).write_text(
            json.dumps(benchmark_snapshot, indent=2),
            encoding="utf-8",
        )
    elif benchmark_payload is not None:
        record["benchmark_label"] = benchmark_payload.get("benchmark_label")

    _append_history_record(record=record, resolved_tracking_dir=resolved_tracking_dir)
    _write_latest_views(record=record, resolved_tracking_dir=resolved_tracking_dir)
    return record


def write_frontend_connection_contract(
    *,
    tracking_dir: str | Path = DEFAULT_RUN_COUNT_TRACKING_DIR,
) -> dict[str, Any]:
    resolved_tracking_dir = _resolve_path(tracking_dir)
    resolved_tracking_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "app_key": "run_count_research_tracker",
        "primary_data_sources": {
            "latest_run": _relative_to_project(resolved_tracking_dir / DEFAULT_RUN_COUNT_LATEST_JSON.name),
            "benchmark": _relative_to_project(resolved_tracking_dir / DEFAULT_RUN_COUNT_BENCHMARK_JSON.name),
            "run_history_index": _relative_to_project(resolved_tracking_dir / DEFAULT_RUN_COUNT_INDEX_JSON.name),
            "run_history_jsonl": _relative_to_project(resolved_tracking_dir / DEFAULT_RUN_COUNT_HISTORY_JSONL.name),
            "run_history_csv": _relative_to_project(resolved_tracking_dir / DEFAULT_RUN_COUNT_HISTORY_CSV.name),
            "current_dual_view": _relative_to_project(Path("data/reports/run_count/dual_view/current_dual_view.json")),
        },
        "recommended_default_views": [
            {
                "view_key": "run_ledger",
                "title": "Run Ledger",
                "source": "run_history_index",
                "purpose": "List every workflow run and whether it beat the benchmark, control, or neither.",
            },
            {
                "view_key": "latest_run",
                "title": "Latest Run",
                "source": "latest_run",
                "purpose": "Show the newest completed workflow summary and lane metrics.",
            },
            {
                "view_key": "benchmark_compare",
                "title": "Benchmark Compare",
                "source": "latest_run",
                "purpose": "Compare the selected run against the active benchmark.",
            },
            {
                "view_key": "best_runs",
                "title": "Best Runs",
                "source": "run_history_index",
                "purpose": "Sort runs by ROI, CRPS, NLS, and promotability.",
            },
        ],
        "entities": {
            "workflow_run": {
                "primary_key": "run_id",
                "important_fields": [
                    "run_label",
                    "hypothesis",
                    "benchmark_status",
                    "summary.headline_result",
                    "summary.best_research_lane_key",
                    "summary.production_promotable_lane_key",
                    "summary.stage3.mean_crps",
                    "summary.stage4.mean_crps",
                    "summary.stage3.roi",
                    "summary.stage4.roi",
                ],
            },
            "lane_summary": {
                "child_path": "summary.stage3 / summary.stage4",
                "important_fields": [
                    "mean_crps",
                    "mean_negative_log_score",
                    "rmse",
                    "roi",
                    "bet_count",
                    "market_data_coverage_pct",
                    "catastrophic_regression",
                ],
            },
        },
        "semantics": {
            "profitability": "Use walk-forward betting_evidence.roi when bet_count > 0. If bet_count is 0, mark profitability as unavailable, not false.",
            "accuracy": "Primary distribution metrics are mean_crps and mean_negative_log_score. RMSE is secondary mean-fit support.",
            "benchmarking": "Compare every new run to benchmark.json and to control via the stored delta blocks.",
            "promotion": "production_promotable_lane_key and promoted_second_opinion_lane_key come from the dual-view output and should be rendered prominently.",
        },
    }

    (resolved_tracking_dir / DEFAULT_RUN_COUNT_CONNECTION_JSON.name).write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    (resolved_tracking_dir / DEFAULT_RUN_COUNT_CONNECTION_MD.name).write_text(
        _render_connection_markdown(payload),
        encoding="utf-8",
    )
    return payload


def _build_summary_block(
    *,
    stage3_report: Mapping[str, Any],
    stage3_vs_control: Mapping[str, Any],
    stage4_report: Mapping[str, Any],
    stage4_vs_control: Mapping[str, Any],
    stage4_vs_stage3: Mapping[str, Any],
    stage3_walk_forward: Mapping[str, Any],
    stage4_walk_forward: Mapping[str, Any],
    dual_view: Mapping[str, Any],
    hypothesis: str | None,
    historical_odds_db: str | None,
    historical_market_book: str | None,
    enable_market_priors: bool,
    mu_delta_mode: str,
) -> dict[str, Any]:
    stage3_mean_metrics = dict(stage3_report.get("mean_metrics", {}))
    stage3_distribution_metrics = dict(stage3_report.get("holdout_metrics", {}))
    stage4_mean_metrics = dict(stage4_report.get("mean_metrics", {}))
    stage4_distribution_metrics = dict(stage4_report.get("holdout_metrics", {}))
    promotion_summary = dict(dual_view.get("promotion_summary", {}))
    stage3_betting = dict(stage3_walk_forward.get("betting_evidence", {}))
    stage4_betting = dict(stage4_walk_forward.get("betting_evidence", {}))

    stage3_market_priors = dict(stage3_report.get("research_feature_metadata", {}).get("market_priors", {}))
    stage4_market_priors = dict(stage4_report.get("research_feature_metadata", {}).get("market_priors", {}))

    best_lane_key = promotion_summary.get("best_research_lane_key")
    production_promotable_lane_key = promotion_summary.get("production_promotable_lane_key")
    stage4_beats_stage3_on_crps = bool(stage4_vs_stage3.get("improvement_flags", {}).get("beats_baseline_on_crps"))
    stage4_beats_control_on_crps = bool(stage4_vs_control.get("improvement_flags", {}).get("beats_baseline_on_crps"))
    stage3_beats_control_on_crps = bool(stage3_vs_control.get("improvement_flags", {}).get("beats_control_on_crps"))

    headline_result = _build_headline_result(
        production_promotable_lane_key=production_promotable_lane_key,
        stage3_beats_control_on_crps=stage3_beats_control_on_crps,
        stage4_beats_control_on_crps=stage4_beats_control_on_crps,
        stage4_beats_stage3_on_crps=stage4_beats_stage3_on_crps,
    )
    what_changed = _build_what_changed(
        enable_market_priors=enable_market_priors,
        historical_odds_db=historical_odds_db,
        historical_market_book=historical_market_book,
        mu_delta_mode=mu_delta_mode,
        hypothesis=hypothesis,
    )

    return {
        "headline_result": headline_result,
        "what_changed": what_changed,
        "best_research_lane_key": best_lane_key,
        "best_research_lane_label": promotion_summary.get("best_research_lane_label"),
        "promoted_second_opinion_lane_key": promotion_summary.get("promoted_second_opinion_lane_key"),
        "production_promotable_lane_key": production_promotable_lane_key,
        "is_profitable": _any_profitable(stage3_betting, stage4_betting),
        "is_more_accurate_than_control": bool(stage3_beats_control_on_crps or stage4_beats_control_on_crps),
        "is_more_stable_than_control": not bool(
            stage3_vs_control.get("catastrophic_regression") or stage4_vs_control.get("catastrophic_regression")
        ),
        "main_reason_failed": _build_main_reason_failed(
            production_promotable_lane_key=production_promotable_lane_key,
            stage3_beats_control_on_crps=stage3_beats_control_on_crps,
            stage4_beats_control_on_crps=stage4_beats_control_on_crps,
            stage3_betting=stage3_betting,
            stage4_betting=stage4_betting,
        ),
        "next_action_hint": _build_next_action_hint(
            production_promotable_lane_key=production_promotable_lane_key,
            stage3_beats_control_on_crps=stage3_beats_control_on_crps,
            stage4_beats_control_on_crps=stage4_beats_control_on_crps,
        ),
        "stage3": _build_lane_snapshot(
            report=stage3_report,
            comparison_to_control=stage3_vs_control,
            comparison_to_prior=None,
            betting_evidence=stage3_betting,
            market_priors=stage3_market_priors,
        ),
        "stage4": _build_lane_snapshot(
            report=stage4_report,
            comparison_to_control=stage4_vs_control,
            comparison_to_prior=stage4_vs_stage3,
            betting_evidence=stage4_betting,
            market_priors=stage4_market_priors,
        ),
    }


def _build_lane_snapshot(
    *,
    report: Mapping[str, Any],
    comparison_to_control: Mapping[str, Any],
    comparison_to_prior: Mapping[str, Any] | None,
    betting_evidence: Mapping[str, Any],
    market_priors: Mapping[str, Any],
) -> dict[str, Any]:
    mean_metrics = dict(report.get("mean_metrics", {}))
    holdout_metrics = dict(report.get("holdout_metrics", {}))
    deltas = dict(comparison_to_control.get("deltas", {}))
    prior_deltas = dict((comparison_to_prior or {}).get("deltas", {}))
    return {
        "model_version": report.get("model_version"),
        "research_lane_name": report.get("research_lane_name"),
        "rmse": _safe_float(mean_metrics.get("rmse")),
        "mae": _safe_float(mean_metrics.get("mae")),
        "mean_crps": _safe_float(holdout_metrics.get("mean_crps")),
        "mean_negative_log_score": _safe_float(holdout_metrics.get("mean_negative_log_score")),
        "market_data_coverage_pct": _safe_float(market_priors.get("coverage_pct")),
        "source_origins": list(market_priors.get("source_origins", [])),
        "source_db_paths": list(market_priors.get("source_db_paths", [])),
        "roi": _safe_float(betting_evidence.get("roi")),
        "net_units": _safe_float(betting_evidence.get("net_units")),
        "bet_count": _safe_float(betting_evidence.get("bet_count")),
        "market_anchor_coverage": _safe_float(betting_evidence.get("market_anchor_coverage")),
        "beats_control_on_crps": bool(
            comparison_to_control.get("improvement_flags", {}).get("beats_control_on_crps")
            or comparison_to_control.get("improvement_flags", {}).get("beats_baseline_on_crps")
        ),
        "beats_control_on_negative_log_score": bool(
            comparison_to_control.get("improvement_flags", {}).get("beats_control_on_negative_log_score")
            or comparison_to_control.get("improvement_flags", {}).get("beats_baseline_on_negative_log_score")
        ),
        "catastrophic_regression": bool(comparison_to_control.get("catastrophic_regression")),
        "delta_vs_control": {
            "mean_crps": _safe_float(deltas.get("mean_crps")),
            "mean_negative_log_score": _safe_float(deltas.get("mean_negative_log_score")),
            "rmse": _safe_float(deltas.get("rmse")),
        },
        "delta_vs_prior_lane": {
            "mean_crps": _safe_float(prior_deltas.get("mean_crps")),
            "mean_negative_log_score": _safe_float(prior_deltas.get("mean_negative_log_score")),
            "rmse": _safe_float(prior_deltas.get("rmse")),
        }
        if comparison_to_prior is not None
        else None,
    }


def _build_benchmark_compare(record: Mapping[str, Any], benchmark_payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if benchmark_payload is None:
        return {
            "available": False,
            "benchmark_label": None,
            "notes": ["No prior benchmark was found. This run can seed benchmark_v1."],
        }

    benchmark_summary = dict(benchmark_payload.get("summary", {}))
    benchmark_stage3 = dict(benchmark_summary.get("stage3", {}))
    benchmark_stage4 = dict(benchmark_summary.get("stage4", {}))
    current_summary = dict(record.get("summary", {}))
    current_stage3 = dict(current_summary.get("stage3", {}))
    current_stage4 = dict(current_summary.get("stage4", {}))
    return {
        "available": True,
        "benchmark_label": benchmark_payload.get("benchmark_label"),
        "benchmark_run_id": benchmark_payload.get("benchmark_run_id"),
        "stage3": _metric_delta_block(current=current_stage3, baseline=benchmark_stage3),
        "stage4": _metric_delta_block(current=current_stage4, baseline=benchmark_stage4),
    }


def _metric_delta_block(*, current: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "delta_rmse": _delta(current.get("rmse"), baseline.get("rmse")),
        "delta_mean_crps": _delta(current.get("mean_crps"), baseline.get("mean_crps")),
        "delta_mean_negative_log_score": _delta(
            current.get("mean_negative_log_score"),
            baseline.get("mean_negative_log_score"),
        ),
        "delta_roi": _delta(current.get("roi"), baseline.get("roi")),
        "delta_net_units": _delta(current.get("net_units"), baseline.get("net_units")),
    }


def _build_headline_result(
    *,
    production_promotable_lane_key: str | None,
    stage3_beats_control_on_crps: bool,
    stage4_beats_control_on_crps: bool,
    stage4_beats_stage3_on_crps: bool,
) -> str:
    if production_promotable_lane_key:
        return "A research lane cleared the production promotion guardrails."
    if stage4_beats_control_on_crps:
        return "Stage 4 improved on control but still failed promotion guardrails."
    if stage4_beats_stage3_on_crps:
        return "Stage 4 improved on Stage 3, but no lane beat control cleanly enough to promote."
    if stage3_beats_control_on_crps:
        return "Stage 3 improved on control, but the workflow still did not produce a promotable lane."
    return "No research lane beat control; this run remains an exploratory benchmark candidate."


def _build_what_changed(
    *,
    enable_market_priors: bool,
    historical_odds_db: str | None,
    historical_market_book: str | None,
    mu_delta_mode: str,
    hypothesis: str | None,
) -> str:
    notes: list[str] = []
    if enable_market_priors:
        notes.append("market priors enabled")
    if historical_odds_db:
        notes.append(f"historical archive: {historical_odds_db}")
    if historical_market_book:
        notes.append(f"market book: {historical_market_book}")
    notes.append(f"mu_delta_mode={mu_delta_mode}")
    if hypothesis:
        notes.append(f"hypothesis: {hypothesis}")
    return "; ".join(notes)


def _build_main_reason_failed(
    *,
    production_promotable_lane_key: str | None,
    stage3_beats_control_on_crps: bool,
    stage4_beats_control_on_crps: bool,
    stage3_betting: Mapping[str, Any],
    stage4_betting: Mapping[str, Any],
) -> str | None:
    if production_promotable_lane_key:
        return None
    if not stage3_beats_control_on_crps and not stage4_beats_control_on_crps:
        return "Neither research lane beat control on the key distribution metrics."
    if not _has_betting_signal(stage3_betting) and not _has_betting_signal(stage4_betting):
        return "Profitability remains blocked because the walk-forward betting evidence is unavailable or empty."
    return "The run improved some diagnostics but did not clear the full promotion stack."


def _build_next_action_hint(
    *,
    production_promotable_lane_key: str | None,
    stage3_beats_control_on_crps: bool,
    stage4_beats_control_on_crps: bool,
) -> str:
    if production_promotable_lane_key:
        return "Lock this run as the new benchmark and test stability before any further structural changes."
    if stage4_beats_control_on_crps:
        return "Keep Stage 4 stable and focus on profitability or calibration instead of more structural churn."
    if stage3_beats_control_on_crps:
        return "Treat the Stage 3 lane as the stronger candidate and isolate what changed before tuning Stage 4."
    return "Use this as the new benchmark if desired, then isolate one change at a time against it."


def _any_profitable(*betting_blocks: Mapping[str, Any]) -> bool | None:
    available = [block for block in betting_blocks if _has_betting_signal(block)]
    if not available:
        return None
    return any((_safe_float(block.get("roi")) or 0.0) > 0.0 for block in available)


def _has_betting_signal(block: Mapping[str, Any]) -> bool:
    bet_count = _safe_float(block.get("bet_count"))
    return bet_count is not None and bet_count > 0.0


def _append_history_record(*, record: Mapping[str, Any], resolved_tracking_dir: Path) -> None:
    jsonl_path = resolved_tracking_dir / DEFAULT_RUN_COUNT_HISTORY_JSONL.name
    csv_path = resolved_tracking_dir / DEFAULT_RUN_COUNT_HISTORY_CSV.name

    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")

    flattened_row = _flatten_record(record)
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        combined = pd.concat([existing, pd.DataFrame([flattened_row])], ignore_index=True)
    else:
        combined = pd.DataFrame([flattened_row])
    combined.to_csv(csv_path, index=False)


def _write_latest_views(*, record: Mapping[str, Any], resolved_tracking_dir: Path) -> None:
    latest_path = resolved_tracking_dir / DEFAULT_RUN_COUNT_LATEST_JSON.name
    latest_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    history_records = _load_history_records(resolved_tracking_dir)
    index_rows = []
    for item in history_records:
        summary = dict(item.get("summary", {}))
        stage3 = dict(summary.get("stage3", {}))
        stage4 = dict(summary.get("stage4", {}))
        index_rows.append(
            {
                "run_id": item.get("run_id"),
                "tracked_at": item.get("tracked_at"),
                "run_label": item.get("run_label"),
                "benchmark_status": item.get("benchmark_status"),
                "benchmark_label": item.get("benchmark_label"),
                "headline_result": summary.get("headline_result"),
                "best_research_lane_key": summary.get("best_research_lane_key"),
                "production_promotable_lane_key": summary.get("production_promotable_lane_key"),
                "stage3_mean_crps": stage3.get("mean_crps"),
                "stage3_mean_negative_log_score": stage3.get("mean_negative_log_score"),
                "stage3_rmse": stage3.get("rmse"),
                "stage3_roi": stage3.get("roi"),
                "stage4_mean_crps": stage4.get("mean_crps"),
                "stage4_mean_negative_log_score": stage4.get("mean_negative_log_score"),
                "stage4_rmse": stage4.get("rmse"),
                "stage4_roi": stage4.get("roi"),
                "is_profitable": summary.get("is_profitable"),
                "is_more_accurate_than_control": summary.get("is_more_accurate_than_control"),
            }
        )

    index_payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "row_count": len(index_rows),
        "runs": sorted(index_rows, key=lambda row: str(row.get("tracked_at")), reverse=True),
    }
    (resolved_tracking_dir / DEFAULT_RUN_COUNT_INDEX_JSON.name).write_text(
        json.dumps(index_payload, indent=2),
        encoding="utf-8",
    )


def _load_existing_benchmark(resolved_tracking_dir: Path) -> dict[str, Any] | None:
    benchmark_path = resolved_tracking_dir / DEFAULT_RUN_COUNT_BENCHMARK_JSON.name
    if not benchmark_path.exists():
        return None
    return json.loads(benchmark_path.read_text(encoding="utf-8"))


def _load_history_records(resolved_tracking_dir: Path) -> list[dict[str, Any]]:
    jsonl_path = resolved_tracking_dir / DEFAULT_RUN_COUNT_HISTORY_JSONL.name
    if not jsonl_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            rows.append(json.loads(stripped))
    return rows


def _flatten_record(record: Mapping[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    _flatten_into(flattened, prefix="", value=record)
    return flattened


def _flatten_into(target: dict[str, Any], *, prefix: str, value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested_value in value.items():
            next_prefix = f"{prefix}_{key}" if prefix else str(key)
            _flatten_into(target, prefix=next_prefix, value=nested_value)
        return
    if isinstance(value, list):
        target[prefix] = json.dumps(value, sort_keys=True)
        return
    target[prefix] = value


def _render_connection_markdown(payload: Mapping[str, Any]) -> str:
    primary_sources = payload.get("primary_data_sources", {})
    lines = [
        "# Run Count Tracker Frontend Handoff",
        "",
        "Use the files below as the dashboard contract.",
        "",
        "## Primary Sources",
    ]
    for key, value in primary_sources.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Rendering Notes",
            "- Treat `latest_run.json` as the default landing page payload.",
            "- Treat `benchmark.json` as the active benchmark for all delta cards.",
            "- Treat `run_history_index.json` as the lightweight ledger source.",
            "- Use `run_history.jsonl` only if you need the full historical payloads.",
            "- Profitability is only meaningful when `bet_count > 0`.",
            "- Promotion state comes from the dual-view output and should be shown prominently.",
        ]
    )
    return "\n".join(lines) + "\n"


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(_resolve_path(path).read_text(encoding="utf-8"))


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def _relative_to_project(path: str | Path) -> str:
    resolved = _resolve_path(path)
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _delta(current: Any, baseline: Any) -> float | None:
    current_value = _safe_float(current)
    baseline_value = _safe_float(baseline)
    if current_value is None or baseline_value is None:
        return None
    return current_value - baseline_value
