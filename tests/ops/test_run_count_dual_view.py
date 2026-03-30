from __future__ import annotations

import json

from src.ops.run_count_dual_view import (
    build_dual_view_payload,
    build_pairwise_disagreement,
    evaluate_research_lane_promotion,
    load_dual_view_inputs,
    resolve_stage5_inputs,
    write_dual_view_outputs,
)


def test_stage3_promotion_becomes_second_opinion_when_betting_is_unavailable() -> None:
    comparison = {
        "stage3_distribution_metrics": {
            "zero_calibration": {"p_0": {"absolute_error": 0.01}},
            "tail_calibration": {
                "p_ge_3": {"absolute_error": 0.02},
                "p_ge_5": {"absolute_error": 0.03},
                "p_ge_10": {"absolute_error": 0.04},
            },
        },
        "improvement_flags": {
            "beats_control_on_crps": True,
            "beats_control_on_negative_log_score": True,
        },
        "guardrails": {
            "rmse_within_2pct": True,
            "mean_bias_within_0_15_runs": True,
            "tail_calibration_stable": True,
        },
    }

    result = evaluate_research_lane_promotion(
        lane_key="best_distribution_lane",
        lane_label="Best distribution lane",
        comparison_to_control=comparison,
        walk_forward_report=None,
    )

    assert result["second_opinion_promoted"] is True
    assert result["production_promotable"] is False
    assert result["lane_status"] == "promoted_second_opinion"
    assert result["checks"]["neutral_walk_forward_betting_if_available"]["available"] is False


def test_pairwise_disagreement_flags_large_lane_divergence() -> None:
    disagreement = build_pairwise_disagreement(
        left_lane={
            "lane_key": "control_mean_lane",
            "lane_label": "Control mean lane",
            "expected_away_runs": 4.41,
            "shutout_probability": 0.07,
            "p_away_runs_ge_3": 0.68,
            "p_away_runs_ge_5": 0.41,
        },
        right_lane={
            "lane_key": "best_mcmc_lane",
            "lane_label": "Best MCMC lane",
            "expected_away_runs": 3.94,
            "shutout_probability": 0.17,
            "p_away_runs_ge_3": 0.53,
            "p_away_runs_ge_5": 0.32,
        },
    )

    assert disagreement["flags"]["mean_gap"] is True
    assert disagreement["flags"]["shutout_gap"] is True
    assert disagreement["flags"]["p_ge_3_gap"] is True
    assert disagreement["flags"]["p_ge_5_gap"] is True
    assert disagreement["flags"]["any"] is True


def test_dual_view_payload_marks_stage3_best_and_stage4_exploratory(tmp_path) -> None:
    payload = build_dual_view_payload(
        current_control_payload={
            "selected_artifact_path": "data/models/control.metadata.json",
        },
        stage3_report={
            "artifact_path": "data/models/stage3.metadata.json",
            "mean_metrics": {"predicted_mean": 4.41},
            "holdout_metrics": {
                "mean_crps": 1.78,
                "mean_negative_log_score": 2.47,
                "zero_calibration": {"p_0": {"absolute_error": 0.01}},
                "tail_calibration": {
                    "p_ge_3": {"absolute_error": 0.02},
                    "p_ge_5": {"absolute_error": 0.01},
                    "p_ge_10": {"absolute_error": 0.03},
                },
                "prediction_summary": {
                    "mean_predicted_runs": 4.34,
                    "mean_predicted_p_0": 0.07,
                    "mean_predicted_p_ge_3": 0.68,
                    "mean_predicted_p_ge_5": 0.41,
                },
            },
        },
        stage3_vs_control={
            "control_mean_metrics": {"predicted_mean": 4.41},
            "control_distribution_metrics": {
                "prediction_summary": {
                    "mean_predicted_p_0": 0.07,
                    "mean_predicted_p_ge_3": 0.68,
                    "mean_predicted_p_ge_5": 0.41,
                }
            },
            "stage3_distribution_metrics": {
                "zero_calibration": {"p_0": {"absolute_error": 0.01}},
                "tail_calibration": {
                    "p_ge_3": {"absolute_error": 0.02},
                    "p_ge_5": {"absolute_error": 0.01},
                    "p_ge_10": {"absolute_error": 0.03},
                },
            },
            "improvement_flags": {
                "beats_control_on_crps": True,
                "beats_control_on_negative_log_score": True,
            },
            "guardrails": {
                "rmse_within_2pct": True,
                "mean_bias_within_0_15_runs": True,
                "tail_calibration_stable": True,
            },
        },
        mcmc_report={
            "artifact_path": "data/models/mcmc.metadata.json",
            "mean_metrics": {"predicted_mean": 3.94},
            "holdout_metrics": {
                "mean_crps": 2.18,
                "mean_negative_log_score": 3.10,
                "zero_calibration": {"p_0": {"absolute_error": 0.09}},
                "tail_calibration": {
                    "p_ge_3": {"absolute_error": 0.14},
                    "p_ge_5": {"absolute_error": 0.08},
                    "p_ge_10": {"absolute_error": 0.00},
                },
                "prediction_summary": {
                    "mean_predicted_runs": 3.94,
                    "mean_predicted_p_0": 0.17,
                    "mean_predicted_p_ge_3": 0.53,
                    "mean_predicted_p_ge_5": 0.32,
                },
            },
        },
        mcmc_vs_control={
            "challenger_distribution_metrics": {
                "zero_calibration": {"p_0": {"absolute_error": 0.09}},
                "tail_calibration": {
                    "p_ge_3": {"absolute_error": 0.14},
                    "p_ge_5": {"absolute_error": 0.08},
                    "p_ge_10": {"absolute_error": 0.00},
                },
            },
            "improvement_flags": {
                "beats_baseline_on_crps": False,
                "beats_baseline_on_negative_log_score": False,
            },
            "guardrails": {
                "rmse_within_2pct": False,
                "mean_bias_within_0_15_runs": False,
                "tail_calibration_stable": False,
            },
        },
        mcmc_vs_stage3={},
        stage3_walk_forward_report=None,
        mcmc_walk_forward_report=None,
        source_paths={"stage3_report_path": "data/reports/run_count/distribution_eval/stage3.json"},
    )

    output_paths = write_dual_view_outputs(payload=payload, output_dir=tmp_path)
    written = json.loads((tmp_path / "current_dual_view.json").read_text(encoding="utf-8"))

    assert payload["promotion_summary"]["best_research_lane_key"] == "best_distribution_lane"
    assert payload["promotion_summary"]["promoted_second_opinion_lane_key"] == "best_distribution_lane"
    assert payload["promotion_summary"]["production_promotable_lane_key"] is None
    assert written["lane_summaries"][1]["lane_status"] == "promoted_second_opinion"
    assert written["lane_summaries"][2]["lane_status"] == "exploratory"
    assert output_paths["json"].endswith("current_dual_view.json")


def test_stage3_promotion_reads_lane_specific_walk_forward_betting_blocker() -> None:
    comparison = {
        "stage3_distribution_metrics": {
            "zero_calibration": {"p_0": {"absolute_error": 0.01}},
            "tail_calibration": {
                "p_ge_3": {"absolute_error": 0.02},
                "p_ge_5": {"absolute_error": 0.03},
                "p_ge_10": {"absolute_error": 0.04},
            },
        },
        "improvement_flags": {
            "beats_control_on_crps": True,
            "beats_control_on_negative_log_score": True,
        },
        "guardrails": {
            "rmse_within_2pct": True,
            "mean_bias_within_0_15_runs": True,
            "tail_calibration_stable": True,
        },
    }

    result = evaluate_research_lane_promotion(
        lane_key="best_distribution_lane",
        lane_label="Best distribution lane",
        comparison_to_control=comparison,
        walk_forward_report={
            "lane_key": "best_distribution_lane",
            "betting_evidence": {
                "available": False,
                "market_anchor_coverage": 0.0,
                "reason": "Historical away-run totals or away team totals are not available.",
            },
        },
    )

    assert result["second_opinion_promoted"] is True
    assert result["production_promotable"] is False
    assert result["checks"]["neutral_walk_forward_betting_if_available"]["available"] is False


def test_stage5_input_resolution_loads_lane_specific_walk_forward_reports(tmp_path) -> None:
    current_control = tmp_path / "current_control.json"
    current_control.write_text(json.dumps({"selected_artifact_path": "data/models/control.metadata.json"}), encoding="utf-8")

    stage3_report = tmp_path / "artifact.distribution_eval.json"
    stage3_report.write_text(
        json.dumps({"artifact_path": "data/models/away_runs_distribution_model.metadata.json"}),
        encoding="utf-8",
    )
    stage3_vs_control = tmp_path / "artifact.vs_control.json"
    stage3_vs_control.write_text(json.dumps({"stage3_distribution_metrics": {}}), encoding="utf-8")

    mcmc_dir = tmp_path / "mcmc"
    mcmc_dir.mkdir()
    mcmc_report = mcmc_dir / "artifact.mcmc_eval.json"
    mcmc_report.write_text(json.dumps({"model_name": "full_game_away_runs_mcmc_model"}), encoding="utf-8")
    mcmc_vs_control = mcmc_dir / "artifact.vs_control.json"
    mcmc_vs_control.write_text(json.dumps({"challenger_label": "stage4_mcmc"}), encoding="utf-8")
    mcmc_vs_stage3 = mcmc_dir / "artifact.vs_stage3.json"
    mcmc_vs_stage3.write_text(json.dumps({"challenger_label": "stage4_mcmc"}), encoding="utf-8")

    walk_forward_dir = tmp_path / "walk_forward"
    walk_forward_dir.mkdir()
    (walk_forward_dir / "artifact.stage3_walk_forward.json").write_text(
        json.dumps({"lane_key": "best_distribution_lane", "betting_evidence": {"available": False}}),
        encoding="utf-8",
    )
    (walk_forward_dir / "artifact.mcmc_walk_forward.json").write_text(
        json.dumps({"lane_key": "best_mcmc_lane", "betting_evidence": {"available": False}}),
        encoding="utf-8",
    )

    resolved = resolve_stage5_inputs(
        current_control_path=current_control,
        distribution_report_dir=tmp_path,
        mcmc_report_dir=mcmc_dir,
        walk_forward_report_dir=walk_forward_dir,
    )
    loaded = load_dual_view_inputs(resolved)

    assert loaded["stage3_walk_forward_report"]["lane_key"] == "best_distribution_lane"
    assert loaded["mcmc_walk_forward_report"]["lane_key"] == "best_mcmc_lane"
