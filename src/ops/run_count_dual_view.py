from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CURRENT_CONTROL_PATH = Path("data/reports/run_count/registry/current_control.json")
DEFAULT_DISTRIBUTION_REPORT_DIR = Path("data/reports/run_count/distribution_eval")
DEFAULT_MCMC_REPORT_DIR = Path("data/reports/run_count/mcmc")
DEFAULT_WALK_FORWARD_REPORT_DIR = Path("data/reports/run_count/walk_forward")
DEFAULT_DUAL_VIEW_OUTPUT_DIR = Path("data/reports/run_count/dual_view")

MEAN_DISAGREEMENT_THRESHOLD = 0.35
SHUTOUT_DISAGREEMENT_THRESHOLD = 0.05
TAIL_DISAGREEMENT_THRESHOLD = 0.04
ZERO_CALIBRATION_ALERT_THRESHOLD = 0.05
MAX_TAIL_CALIBRATION_ALERT_THRESHOLD = 0.08


@dataclass(frozen=True, slots=True)
class ResolvedStage5Inputs:
    current_control_path: Path
    stage3_report_path: Path
    stage3_vs_control_path: Path
    mcmc_report_path: Path | None
    mcmc_vs_control_path: Path | None
    mcmc_vs_stage3_path: Path | None
    stage3_walk_forward_report_path: Path | None
    mcmc_walk_forward_report_path: Path | None


def resolve_stage5_inputs(
    *,
    current_control_path: str | Path = DEFAULT_CURRENT_CONTROL_PATH,
    stage3_report_json: str | Path | None = None,
    stage3_vs_control_json: str | Path | None = None,
    distribution_report_dir: str | Path = DEFAULT_DISTRIBUTION_REPORT_DIR,
    mcmc_report_json: str | Path | None = None,
    mcmc_vs_control_json: str | Path | None = None,
    mcmc_vs_stage3_json: str | Path | None = None,
    mcmc_report_dir: str | Path = DEFAULT_MCMC_REPORT_DIR,
    stage3_walk_forward_report_json: str | Path | None = None,
    mcmc_walk_forward_report_json: str | Path | None = None,
    walk_forward_report_dir: str | Path = DEFAULT_WALK_FORWARD_REPORT_DIR,
) -> ResolvedStage5Inputs:
    resolved_current_control = _resolve_required_path(current_control_path)
    distribution_dir = _resolve_path(distribution_report_dir)
    resolved_stage3_report = _resolve_optional_path(stage3_report_json) or _find_latest_json(
        distribution_dir,
        pattern="*.distribution_eval.json",
        predicate=lambda payload: "away_runs_distribution_model"
        in str(payload.get("artifact_path", "")),
    )
    if resolved_stage3_report is None:
        raise FileNotFoundError("Could not resolve a Stage 3 distribution evaluation JSON report.")

    resolved_stage3_vs_control = _resolve_optional_path(stage3_vs_control_json) or _find_latest_json(
        distribution_dir,
        pattern="*.vs_control.json",
        predicate=lambda payload: "stage3_distribution_metrics" in payload,
    )
    if resolved_stage3_vs_control is None:
        raise FileNotFoundError("Could not resolve a Stage 3 vs-control comparison JSON report.")

    mcmc_dir = _resolve_path(mcmc_report_dir)
    resolved_mcmc_report = _resolve_optional_path(mcmc_report_json) or _find_latest_json(
        mcmc_dir,
        pattern="*.mcmc_eval.json",
        predicate=lambda payload: "full_game_away_runs_mcmc_model"
        in str(payload.get("model_name", "")),
    )
    resolved_mcmc_vs_control = _resolve_optional_path(mcmc_vs_control_json) or _find_latest_json(
        mcmc_dir,
        pattern="*.vs_control.json",
        predicate=lambda payload: str(payload.get("challenger_label")) == "stage4_mcmc",
    )
    resolved_mcmc_vs_stage3 = _resolve_optional_path(mcmc_vs_stage3_json) or _find_latest_json(
        mcmc_dir,
        pattern="*.vs_stage3.json",
        predicate=lambda payload: str(payload.get("challenger_label")) == "stage4_mcmc",
    )
    walk_forward_dir = _resolve_path(walk_forward_report_dir)
    resolved_stage3_walk_forward = _resolve_optional_path(stage3_walk_forward_report_json) or _find_latest_json(
        walk_forward_dir,
        pattern="*.stage3_walk_forward.json",
        predicate=lambda payload: str(payload.get("lane_key")) == "best_distribution_lane",
    )
    resolved_mcmc_walk_forward = _resolve_optional_path(mcmc_walk_forward_report_json) or _find_latest_json(
        walk_forward_dir,
        pattern="*.mcmc_walk_forward.json",
        predicate=lambda payload: str(payload.get("lane_key")) == "best_mcmc_lane",
    )

    return ResolvedStage5Inputs(
        current_control_path=resolved_current_control,
        stage3_report_path=resolved_stage3_report,
        stage3_vs_control_path=resolved_stage3_vs_control,
        mcmc_report_path=resolved_mcmc_report,
        mcmc_vs_control_path=resolved_mcmc_vs_control,
        mcmc_vs_stage3_path=resolved_mcmc_vs_stage3,
        stage3_walk_forward_report_path=resolved_stage3_walk_forward,
        mcmc_walk_forward_report_path=resolved_mcmc_walk_forward,
    )


def build_dual_view_payload(
    *,
    current_control_payload: Mapping[str, Any],
    stage3_report: Mapping[str, Any],
    stage3_vs_control: Mapping[str, Any],
    mcmc_report: Mapping[str, Any] | None = None,
    mcmc_vs_control: Mapping[str, Any] | None = None,
    mcmc_vs_stage3: Mapping[str, Any] | None = None,
    stage3_walk_forward_report: Mapping[str, Any] | None = None,
    mcmc_walk_forward_report: Mapping[str, Any] | None = None,
    source_paths: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    control_summary = build_control_lane_summary(
        current_control_payload=current_control_payload,
        comparison_to_control=stage3_vs_control,
    )
    stage3_promotion = evaluate_research_lane_promotion(
        lane_key="best_distribution_lane",
        lane_label="Best distribution lane",
        comparison_to_control=stage3_vs_control,
        walk_forward_report=stage3_walk_forward_report,
    )
    stage3_summary = build_research_lane_summary(
        lane_key="best_distribution_lane",
        lane_label="Best distribution lane",
        lane_kind="distribution",
        report_payload=stage3_report,
        artifact_path=str(stage3_report.get("artifact_path") or ""),
        comparison_to_control=stage3_vs_control,
        comparison_to_best_distribution=None,
        promotion_state=stage3_promotion,
        walk_forward_report=stage3_walk_forward_report,
    )

    lane_summaries = [control_summary, stage3_summary]
    if mcmc_report is not None:
        mcmc_promotion = evaluate_research_lane_promotion(
            lane_key="best_mcmc_lane",
            lane_label="Best MCMC lane",
            comparison_to_control=mcmc_vs_control,
            walk_forward_report=mcmc_walk_forward_report,
        )
        lane_summaries.append(
            build_research_lane_summary(
                lane_key="best_mcmc_lane",
                lane_label="Best MCMC lane",
                lane_kind="mcmc",
                report_payload=mcmc_report,
                artifact_path=str(mcmc_report.get("artifact_path") or ""),
                comparison_to_control=mcmc_vs_control,
                comparison_to_best_distribution=mcmc_vs_stage3,
                promotion_state=mcmc_promotion,
                walk_forward_report=mcmc_walk_forward_report,
            )
        )

    pairwise_disagreements = build_pairwise_disagreements(lane_summaries)
    ranked_research_lanes = rank_research_lanes(lane_summaries)
    best_research_lane_key = ranked_research_lanes[0]["lane_key"] if ranked_research_lanes else None
    second_opinion_lane_key = next(
        (
            lane["lane_key"]
            for lane in ranked_research_lanes
            if lane["promotion_state"]["second_opinion_promoted"]
        ),
        None,
    )

    production_promotable_lane_key = next(
        (
            lane["lane_key"]
            for lane in ranked_research_lanes
            if lane["promotion_state"]["production_promotable"]
        ),
        None,
    )
    promotion_summary = {
        "best_research_lane_key": best_research_lane_key,
        "best_research_lane_label": _lane_label_by_key(best_research_lane_key, lane_summaries),
        "promoted_second_opinion_lane_key": second_opinion_lane_key,
        "promoted_second_opinion_lane_label": _lane_label_by_key(second_opinion_lane_key, lane_summaries),
        "production_promotable_lane_key": production_promotable_lane_key,
        "production_promotable_lane_label": _lane_label_by_key(production_promotable_lane_key, lane_summaries),
        "control_lane_intact": True,
        "notes": _promotion_summary_notes(
            second_opinion_lane_key=second_opinion_lane_key,
            production_promotable_lane_key=production_promotable_lane_key,
        ),
    }

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "source_paths": dict(source_paths or {}),
        "lane_summaries": lane_summaries,
        "pairwise_disagreements": pairwise_disagreements,
        "ranked_research_lanes": ranked_research_lanes,
        "promotion_summary": promotion_summary,
    }


def build_control_lane_summary(
    *,
    current_control_payload: Mapping[str, Any],
    comparison_to_control: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_mean_metrics, baseline_distribution_metrics = extract_baseline_sections(comparison_to_control)
    expected_away_runs = _safe_float(baseline_mean_metrics.get("predicted_mean"))
    prediction_summary = baseline_distribution_metrics.get("prediction_summary", {})
    return {
        "lane_key": "control_mean_lane",
        "lane_label": "Control mean lane",
        "lane_kind": "control",
        "lane_status": "control",
        "artifact_path": current_control_payload.get("selected_artifact_path"),
        "expected_away_runs": expected_away_runs,
        "shutout_probability": _safe_float(prediction_summary.get("mean_predicted_p_0")),
        "p_away_runs_ge_3": _safe_float(prediction_summary.get("mean_predicted_p_ge_3")),
        "p_away_runs_ge_5": _safe_float(prediction_summary.get("mean_predicted_p_ge_5")),
        "mean_metrics": dict(baseline_mean_metrics),
        "distribution_metrics": dict(baseline_distribution_metrics),
        "promotion_state": {
            "lane_key": "control_mean_lane",
            "lane_label": "Control mean lane",
            "second_opinion_promoted": False,
            "production_promotable": False,
            "lane_status": "control",
            "summary_reason": "Control lane remains the stable benchmark.",
            "checks": {},
        },
    }


def build_research_lane_summary(
    *,
    lane_key: str,
    lane_label: str,
    lane_kind: str,
    report_payload: Mapping[str, Any],
    artifact_path: str,
    comparison_to_control: Mapping[str, Any] | None,
    comparison_to_best_distribution: Mapping[str, Any] | None,
    promotion_state: Mapping[str, Any],
    walk_forward_report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    holdout_metrics = report_payload.get("holdout_metrics", {})
    prediction_summary = holdout_metrics.get("prediction_summary", {})
    expected_runs = _safe_float(prediction_summary.get("mean_predicted_runs"))
    if expected_runs is None:
        expected_runs = _safe_float(report_payload.get("mean_metrics", {}).get("predicted_mean"))

    return {
        "lane_key": lane_key,
        "lane_label": lane_label,
        "lane_kind": lane_kind,
        "lane_status": promotion_state["lane_status"],
        "artifact_path": artifact_path,
        "expected_away_runs": expected_runs,
        "shutout_probability": _safe_float(prediction_summary.get("mean_predicted_p_0")),
        "p_away_runs_ge_3": _safe_float(prediction_summary.get("mean_predicted_p_ge_3")),
        "p_away_runs_ge_5": _safe_float(prediction_summary.get("mean_predicted_p_ge_5")),
        "mean_metrics": dict(report_payload.get("mean_metrics", {})),
        "distribution_metrics": dict(holdout_metrics),
        "comparison_to_control": dict(comparison_to_control or {}),
        "comparison_to_best_distribution": dict(comparison_to_best_distribution or {}),
        "walk_forward_betting_evidence": dict((walk_forward_report or {}).get("betting_evidence", {})),
        "walk_forward_proxy_market_decision_evidence": dict(
            (walk_forward_report or {}).get("proxy_market_decision_evidence", {})
        ),
        "walk_forward_operational_diagnostics": dict((walk_forward_report or {}).get("operational_diagnostics", {})),
        "promotion_state": dict(promotion_state),
    }


def evaluate_research_lane_promotion(
    *,
    lane_key: str,
    lane_label: str,
    comparison_to_control: Mapping[str, Any] | None,
    walk_forward_report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if comparison_to_control is None:
        return {
            "lane_key": lane_key,
            "lane_label": lane_label,
            "second_opinion_promoted": False,
            "production_promotable": False,
            "lane_status": "exploratory",
            "summary_reason": "Control comparison is missing, so this lane stays exploratory.",
            "checks": {
                "better_distribution_metrics_than_control": {
                    "passed": False,
                    "reason": "No control comparison report was provided.",
                }
            },
        }

    challenger_distribution_metrics = extract_challenger_distribution_metrics(comparison_to_control)
    flags = normalize_improvement_flags(comparison_to_control)
    guardrails = normalize_guardrails(comparison_to_control)
    zero_abs_error = _safe_float(
        challenger_distribution_metrics.get("zero_calibration", {}).get("p_0", {}).get("absolute_error")
    )
    max_tail_abs_error = max_tail_absolute_error(challenger_distribution_metrics)

    distribution_better = flags["beats_crps"] and flags["beats_negative_log_score"]
    mean_acceptable = guardrails["rmse_within_2pct"] and guardrails["mean_bias_within_0_15_runs"]
    calibration_acceptable = (
        guardrails["tail_calibration_stable"]
        and zero_abs_error is not None
        and zero_abs_error <= ZERO_CALIBRATION_ALERT_THRESHOLD
        and max_tail_abs_error <= MAX_TAIL_CALIBRATION_ALERT_THRESHOLD
    )
    betting_check = evaluate_walk_forward_check(walk_forward_report)

    second_opinion_promoted = distribution_better and mean_acceptable and calibration_acceptable
    production_promotable = second_opinion_promoted and betting_check["passed"] is True

    if production_promotable:
        lane_status = "production_promotable"
        summary_reason = "All promotion checks passed, including walk-forward betting."
    elif second_opinion_promoted:
        lane_status = "promoted_second_opinion"
        summary_reason = "Distribution metrics beat control and guardrails hold, but production promotion is blocked."
    else:
        lane_status = "exploratory"
        summary_reason = "This lane does not clear the promotion guardrails and stays exploratory."

    return {
        "lane_key": lane_key,
        "lane_label": lane_label,
        "second_opinion_promoted": second_opinion_promoted,
        "production_promotable": production_promotable,
        "lane_status": lane_status,
        "summary_reason": summary_reason,
        "checks": {
            "better_distribution_metrics_than_control": {
                "passed": distribution_better,
                "details": {
                    "beats_crps": flags["beats_crps"],
                    "beats_negative_log_score": flags["beats_negative_log_score"],
                },
            },
            "acceptable_mean_diagnostics": {
                "passed": mean_acceptable,
                "details": {
                    "rmse_within_2pct": guardrails["rmse_within_2pct"],
                    "mean_bias_within_0_15_runs": guardrails["mean_bias_within_0_15_runs"],
                },
            },
            "no_alarming_calibration_failures": {
                "passed": calibration_acceptable,
                "details": {
                    "tail_calibration_stable": guardrails["tail_calibration_stable"],
                    "zero_abs_error": zero_abs_error,
                    "zero_abs_error_threshold": ZERO_CALIBRATION_ALERT_THRESHOLD,
                    "max_tail_abs_error": max_tail_abs_error,
                    "max_tail_abs_error_threshold": MAX_TAIL_CALIBRATION_ALERT_THRESHOLD,
                },
            },
            "neutral_walk_forward_betting_if_available": betting_check,
        },
    }


def evaluate_walk_forward_check(
    walk_forward_report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if walk_forward_report is None:
        return {
            "available": False,
            "passed": None,
            "reason": "No walk-forward betting report was supplied.",
        }

    betting_payload = walk_forward_report.get("betting_evidence", walk_forward_report)
    if not isinstance(betting_payload, Mapping):
        betting_payload = walk_forward_report

    explicitly_available = betting_payload.get("available")
    if explicitly_available is False:
        return {
            "available": False,
            "passed": None,
            "reason": str(
                betting_payload.get("reason")
                or "Walk-forward report exists, but betting evidence is not available for this lane."
            ),
            "details": {
                "market_anchor_coverage": _first_present_float(betting_payload, ("market_anchor_coverage",)),
            },
        }

    roi = _first_present_float(betting_payload, ("roi", "walk_forward_roi"))
    net_units = _first_present_float(betting_payload, ("net_units", "walk_forward_net_units"))
    max_drawdown = _first_present_float(betting_payload, ("max_drawdown", "walk_forward_max_drawdown"))
    bet_count = _first_present_float(betting_payload, ("bet_count", "walk_forward_bet_count"))
    clv_supported = betting_payload.get("clv_supported")

    if bet_count is not None and bet_count <= 0:
        return {
            "available": True,
            "passed": False,
            "reason": "Walk-forward report matched markets, but no qualifying bets were generated.",
            "details": {
                "roi": roi,
                "net_units": net_units,
                "max_drawdown": max_drawdown,
                "bet_count": bet_count,
            },
        }
    if clv_supported is False:
        return {
            "available": True,
            "passed": False,
            "reason": str(
                betting_payload.get("reason")
                or "Walk-forward report matched the target market, but closing-line validation is unavailable."
            ),
            "details": {
                "roi": roi,
                "net_units": net_units,
                "max_drawdown": max_drawdown,
                "bet_count": bet_count,
                "clv_supported": False,
            },
        }

    neutral_or_better = (roi is not None and roi >= 0.0) or (net_units is not None and net_units >= 0.0)
    not_alarming_drawdown = max_drawdown is None or max_drawdown >= -0.25
    passed = neutral_or_better and not_alarming_drawdown
    return {
        "available": True,
        "passed": passed,
        "details": {
            "roi": roi,
            "net_units": net_units,
            "max_drawdown": max_drawdown,
            "bet_count": bet_count,
        },
        "reason": (
            "Walk-forward betting is neutral or better."
            if passed
            else "Walk-forward betting is negative or shows an alarming drawdown."
        ),
    }


def build_pairwise_disagreements(lane_summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for left_index, left_lane in enumerate(lane_summaries):
        for right_lane in lane_summaries[left_index + 1 :]:
            results.append(
                build_pairwise_disagreement(
                    left_lane=left_lane,
                    right_lane=right_lane,
                )
            )
    return results


def build_pairwise_disagreement(
    *,
    left_lane: Mapping[str, Any],
    right_lane: Mapping[str, Any],
) -> dict[str, Any]:
    mean_gap = abs(_safe_float(left_lane.get("expected_away_runs")) - _safe_float(right_lane.get("expected_away_runs")))
    shutout_gap = abs(
        _safe_float(left_lane.get("shutout_probability")) - _safe_float(right_lane.get("shutout_probability"))
    )
    p_ge_3_gap = abs(
        _safe_float(left_lane.get("p_away_runs_ge_3")) - _safe_float(right_lane.get("p_away_runs_ge_3"))
    )
    p_ge_5_gap = abs(
        _safe_float(left_lane.get("p_away_runs_ge_5")) - _safe_float(right_lane.get("p_away_runs_ge_5"))
    )
    flags = {
        "mean_gap": mean_gap >= MEAN_DISAGREEMENT_THRESHOLD,
        "shutout_gap": shutout_gap >= SHUTOUT_DISAGREEMENT_THRESHOLD,
        "p_ge_3_gap": p_ge_3_gap >= TAIL_DISAGREEMENT_THRESHOLD,
        "p_ge_5_gap": p_ge_5_gap >= TAIL_DISAGREEMENT_THRESHOLD,
    }
    flags["any"] = any(flags.values())
    return {
        "left_lane_key": left_lane.get("lane_key"),
        "right_lane_key": right_lane.get("lane_key"),
        "left_lane_label": left_lane.get("lane_label"),
        "right_lane_label": right_lane.get("lane_label"),
        "mean_gap": mean_gap,
        "shutout_gap": shutout_gap,
        "p_ge_3_gap": p_ge_3_gap,
        "p_ge_5_gap": p_ge_5_gap,
        "flags": flags,
    }


def rank_research_lanes(lane_summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    research_lanes = [lane for lane in lane_summaries if lane.get("lane_kind") != "control"]
    return sorted(
        research_lanes,
        key=lambda lane: (
            not bool(lane["promotion_state"]["second_opinion_promoted"]),
            not bool(lane["promotion_state"]["production_promotable"]),
            _safe_float(lane.get("distribution_metrics", {}).get("mean_negative_log_score")),
            _safe_float(lane.get("distribution_metrics", {}).get("mean_crps")),
        ),
    )


def write_dual_view_outputs(
    *,
    payload: Mapping[str, Any],
    output_dir: str | Path = DEFAULT_DUAL_VIEW_OUTPUT_DIR,
) -> dict[str, str]:
    resolved_output_dir = _resolve_path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    json_path = resolved_output_dir / "current_dual_view.json"
    csv_path = resolved_output_dir / "current_dual_view.csv"
    markdown_path = resolved_output_dir / "current_dual_view.md"

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_lane_summary_csv(csv_path, payload.get("lane_summaries", []))
    markdown_path.write_text(render_dual_view_markdown(payload), encoding="utf-8")

    return {
        "json": _relative_to_project(json_path),
        "csv": _relative_to_project(csv_path),
        "markdown": _relative_to_project(markdown_path),
    }


def render_dual_view_markdown(payload: Mapping[str, Any]) -> str:
    lanes = list(payload.get("lane_summaries", []))
    disagreements = list(payload.get("pairwise_disagreements", []))
    promotion_summary = dict(payload.get("promotion_summary", {}))

    lines = [
        "# Run Count Dual View",
        "",
        f"Generated at: `{payload.get('generated_at')}`",
        "",
        "## Lanes",
        "",
        "| Lane | Status | Expected away runs | Shutout | P(away >= 3) | P(away >= 5) |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for lane in lanes:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(lane.get("lane_label", "")),
                    str(lane.get("lane_status", "")),
                    _format_probability_or_mean(lane.get("expected_away_runs")),
                    _format_probability_or_mean(lane.get("shutout_probability")),
                    _format_probability_or_mean(lane.get("p_away_runs_ge_3")),
                    _format_probability_or_mean(lane.get("p_away_runs_ge_5")),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Promotion",
            "",
            f"- Best current research lane: `{promotion_summary.get('best_research_lane_label')}`",
            f"- Promoted second-opinion lane: `{promotion_summary.get('promoted_second_opinion_lane_label')}`",
            f"- Production-promotable lane: `{promotion_summary.get('production_promotable_lane_label')}`",
        ]
    )
    for note in promotion_summary.get("notes", []):
        lines.append(f"- {note}")

    lines.extend(
        [
            "",
            "## Disagreements",
            "",
            "| Pair | Mean gap | Shutout gap | P>=3 gap | P>=5 gap | Any flag |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for item in disagreements:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{item.get('left_lane_label')} vs {item.get('right_lane_label')}",
                    _format_probability_or_mean(item.get("mean_gap")),
                    _format_probability_or_mean(item.get("shutout_gap")),
                    _format_probability_or_mean(item.get("p_ge_3_gap")),
                    _format_probability_or_mean(item.get("p_ge_5_gap")),
                    "yes" if item.get("flags", {}).get("any") else "no",
                ]
            )
            + " |"
        )

    return "\n".join(lines) + "\n"


def load_dual_view_inputs(
    resolved_inputs: ResolvedStage5Inputs,
) -> dict[str, Any]:
    loaded = {
        "current_control_payload": _read_json(resolved_inputs.current_control_path),
        "stage3_report": _read_json(resolved_inputs.stage3_report_path),
        "stage3_vs_control": _read_json(resolved_inputs.stage3_vs_control_path),
        "source_paths": {
            "current_control_path": _relative_to_project(resolved_inputs.current_control_path),
            "stage3_report_path": _relative_to_project(resolved_inputs.stage3_report_path),
            "stage3_vs_control_path": _relative_to_project(resolved_inputs.stage3_vs_control_path),
        },
    }

    if resolved_inputs.mcmc_report_path is not None and resolved_inputs.mcmc_report_path.exists():
        loaded["mcmc_report"] = _read_json(resolved_inputs.mcmc_report_path)
        loaded["source_paths"]["mcmc_report_path"] = _relative_to_project(resolved_inputs.mcmc_report_path)
    else:
        loaded["mcmc_report"] = None

    if resolved_inputs.mcmc_vs_control_path is not None and resolved_inputs.mcmc_vs_control_path.exists():
        loaded["mcmc_vs_control"] = _read_json(resolved_inputs.mcmc_vs_control_path)
        loaded["source_paths"]["mcmc_vs_control_path"] = _relative_to_project(resolved_inputs.mcmc_vs_control_path)
    else:
        loaded["mcmc_vs_control"] = None

    if resolved_inputs.mcmc_vs_stage3_path is not None and resolved_inputs.mcmc_vs_stage3_path.exists():
        loaded["mcmc_vs_stage3"] = _read_json(resolved_inputs.mcmc_vs_stage3_path)
        loaded["source_paths"]["mcmc_vs_stage3_path"] = _relative_to_project(resolved_inputs.mcmc_vs_stage3_path)
    else:
        loaded["mcmc_vs_stage3"] = None

    if (
        resolved_inputs.stage3_walk_forward_report_path is not None
        and resolved_inputs.stage3_walk_forward_report_path.exists()
    ):
        loaded["stage3_walk_forward_report"] = _read_json(resolved_inputs.stage3_walk_forward_report_path)
        loaded["source_paths"]["stage3_walk_forward_report_path"] = _relative_to_project(
            resolved_inputs.stage3_walk_forward_report_path
        )
    else:
        loaded["stage3_walk_forward_report"] = None

    if (
        resolved_inputs.mcmc_walk_forward_report_path is not None
        and resolved_inputs.mcmc_walk_forward_report_path.exists()
    ):
        loaded["mcmc_walk_forward_report"] = _read_json(resolved_inputs.mcmc_walk_forward_report_path)
        loaded["source_paths"]["mcmc_walk_forward_report_path"] = _relative_to_project(
            resolved_inputs.mcmc_walk_forward_report_path
        )
    else:
        loaded["mcmc_walk_forward_report"] = None

    return loaded


def extract_baseline_sections(comparison_payload: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline_mean_metrics = comparison_payload.get("control_mean_metrics") or comparison_payload.get(
        "baseline_mean_metrics"
    )
    baseline_distribution_metrics = comparison_payload.get("control_distribution_metrics") or comparison_payload.get(
        "baseline_distribution_metrics"
    )
    return dict(baseline_mean_metrics or {}), dict(baseline_distribution_metrics or {})


def extract_challenger_distribution_metrics(comparison_payload: Mapping[str, Any]) -> dict[str, Any]:
    challenger = comparison_payload.get("stage3_distribution_metrics") or comparison_payload.get(
        "challenger_distribution_metrics"
    )
    return dict(challenger or {})


def normalize_improvement_flags(comparison_payload: Mapping[str, Any]) -> dict[str, bool]:
    flags = comparison_payload.get("improvement_flags", {})
    return {
        "beats_crps": bool(
            flags.get("beats_control_on_crps", flags.get("beats_baseline_on_crps", False))
        ),
        "beats_negative_log_score": bool(
            flags.get(
                "beats_control_on_negative_log_score",
                flags.get("beats_baseline_on_negative_log_score", False),
            )
        ),
        "improves_zero_calibration": bool(flags.get("improves_zero_calibration", False)),
        "improves_max_tail_calibration": bool(flags.get("improves_max_tail_calibration", False)),
    }


def normalize_guardrails(comparison_payload: Mapping[str, Any]) -> dict[str, bool]:
    guardrails = comparison_payload.get("guardrails", {})
    return {
        "rmse_within_2pct": bool(guardrails.get("rmse_within_2pct", False)),
        "mean_bias_within_0_15_runs": bool(guardrails.get("mean_bias_within_0_15_runs", False)),
        "tail_calibration_stable": bool(guardrails.get("tail_calibration_stable", False)),
    }


def max_tail_absolute_error(distribution_metrics: Mapping[str, Any]) -> float:
    tail_calibration = distribution_metrics.get("tail_calibration", {})
    values = [
        _safe_float(item.get("absolute_error"))
        for item in tail_calibration.values()
        if isinstance(item, Mapping)
    ]
    filtered = [value for value in values if value is not None]
    return max(filtered, default=0.0)


def _write_lane_summary_csv(path: Path, lane_summaries: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "lane_key",
        "lane_label",
        "lane_kind",
        "lane_status",
        "artifact_path",
        "expected_away_runs",
        "shutout_probability",
        "p_away_runs_ge_3",
        "p_away_runs_ge_5",
        "second_opinion_promoted",
        "production_promotable",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for lane in lane_summaries:
            promotion_state = lane.get("promotion_state", {})
            writer.writerow(
                {
                    "lane_key": lane.get("lane_key"),
                    "lane_label": lane.get("lane_label"),
                    "lane_kind": lane.get("lane_kind"),
                    "lane_status": lane.get("lane_status"),
                    "artifact_path": lane.get("artifact_path"),
                    "expected_away_runs": lane.get("expected_away_runs"),
                    "shutout_probability": lane.get("shutout_probability"),
                    "p_away_runs_ge_3": lane.get("p_away_runs_ge_3"),
                    "p_away_runs_ge_5": lane.get("p_away_runs_ge_5"),
                    "second_opinion_promoted": promotion_state.get("second_opinion_promoted"),
                    "production_promotable": promotion_state.get("production_promotable"),
                }
            )


def _promotion_summary_notes(
    *,
    second_opinion_lane_key: str | None,
    production_promotable_lane_key: str | None,
) -> list[str]:
    notes: list[str] = []
    if second_opinion_lane_key is None:
        notes.append("No research lane clears the second-opinion threshold yet.")
    else:
        notes.append("The best research lane is promoted only as a second opinion until betting evidence is sufficient.")
    if production_promotable_lane_key is None:
        notes.append("No research lane is production-promotable under the current Stage 5 rule set.")
    return notes


def _lane_label_by_key(lane_key: str | None, lane_summaries: Sequence[Mapping[str, Any]]) -> str | None:
    if lane_key is None:
        return None
    for lane in lane_summaries:
        if lane.get("lane_key") == lane_key:
            return str(lane.get("lane_label"))
    return None


def _find_latest_json(
    directory: Path,
    *,
    pattern: str,
    predicate: Any,
) -> Path | None:
    if not directory.exists():
        return None
    candidates = sorted(directory.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    for candidate in candidates:
        try:
            payload = _read_json(candidate)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        if predicate(payload):
            return candidate.resolve()
    return None


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _resolve_required_path(value: str | Path) -> Path:
    resolved = _resolve_path(value)
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved


def _resolve_optional_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    resolved = _resolve_path(value)
    return resolved if resolved.exists() else None


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _relative_to_project(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def _format_probability_or_mean(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.3f}"


def _safe_float(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


def _first_present_float(payload: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        if key in payload and payload[key] is not None:
            return float(payload[key])
    return None

