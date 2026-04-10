from __future__ import annotations

import json
from pathlib import Path

from src.ops.run_count_tracker import (
    record_run_count_workflow_run,
    write_frontend_connection_contract,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _minimal_report(*, model_version: str, lane_name: str) -> dict:
    return {
        "model_version": model_version,
        "research_lane_name": lane_name,
        "artifact_path": f"data/models/{model_version}.metadata.json",
        "mean_metrics": {"rmse": 3.2, "mae": 2.5},
        "holdout_metrics": {"mean_crps": 1.8, "mean_negative_log_score": 2.4},
        "research_feature_metadata": {
            "market_priors": {
                "coverage_pct": 1.0,
                "source_origins": ["legacy_old_scraper", "oddsportal"],
                "source_db_paths": ["OddsScraper/data/mlb_odds.db", "data/mlb_odds_oddsportal.db"],
            }
        },
    }


def _minimal_comparison(*, beats_crps: bool) -> dict:
    return {
        "deltas": {"mean_crps": -0.01 if beats_crps else 0.01, "mean_negative_log_score": 0.02, "rmse": 0.0},
        "improvement_flags": {
            "beats_control_on_crps": beats_crps,
            "beats_control_on_negative_log_score": False,
            "beats_baseline_on_crps": beats_crps,
            "beats_baseline_on_negative_log_score": False,
        },
        "catastrophic_regression": False,
    }


def _minimal_walk_forward(*, lane_key: str) -> dict:
    return {
        "lane_key": lane_key,
        "betting_evidence": {"bet_count": 0, "roi": None, "net_units": None, "market_anchor_coverage": 1.0},
    }


def _minimal_dual_view(*, best_lane: str) -> dict:
    return {
        "promotion_summary": {
            "best_research_lane_key": best_lane,
            "best_research_lane_label": "Best distribution lane",
            "promoted_second_opinion_lane_key": None,
            "production_promotable_lane_key": None,
        }
    }


def test_run_count_tracker_writes_latest_index_and_benchmark(tmp_path: Path) -> None:
    stage3_report = _write_json(tmp_path / "stage3.json", _minimal_report(model_version="stage3v1", lane_name="stage3"))
    stage4_report = _write_json(tmp_path / "stage4.json", _minimal_report(model_version="stage4v1", lane_name="stage4"))
    stage3_vs_control = _write_json(tmp_path / "stage3_vs_control.json", _minimal_comparison(beats_crps=False))
    stage4_vs_control = _write_json(tmp_path / "stage4_vs_control.json", _minimal_comparison(beats_crps=False))
    stage4_vs_stage3 = _write_json(tmp_path / "stage4_vs_stage3.json", _minimal_comparison(beats_crps=True))
    stage3_walk_forward = _write_json(tmp_path / "stage3_walk_forward.json", _minimal_walk_forward(lane_key="best_distribution_lane"))
    stage4_walk_forward = _write_json(tmp_path / "stage4_walk_forward.json", _minimal_walk_forward(lane_key="best_mcmc_lane"))
    dual_view = _write_json(tmp_path / "dual_view.json", _minimal_dual_view(best_lane="best_distribution_lane"))

    tracker_dir = tmp_path / "tracker"
    record = record_run_count_workflow_run(
        stage3_report_path=stage3_report,
        stage3_vs_control_path=stage3_vs_control,
        stage4_report_path=stage4_report,
        stage4_vs_control_path=stage4_vs_control,
        stage4_vs_stage3_path=stage4_vs_stage3,
        stage3_walk_forward_path=stage3_walk_forward,
        stage4_walk_forward_path=stage4_walk_forward,
        dual_view_path=dual_view,
        training_data_path="data/training/ParquetDefault.parquet",
        start_year=2021,
        end_year=2025,
        holdout_season=2025,
        folds=3,
        feature_selection_mode="flat",
        forced_delta_count=0,
        xgb_workers=4,
        enable_market_priors=True,
        historical_odds_db="OddsScraper/data/mlb_odds.db;data/mlb_odds_oddsportal.db",
        historical_market_book=None,
        mu_delta_mode="off",
        stage3_experiment="stage3-exp",
        stage4_experiment="stage4-exp",
        stage3_research_lane_name="stage3-lane",
        stage4_research_lane_name="stage4-lane",
        tracking_dir=tracker_dir,
        hypothesis="tracker smoke",
        run_label="smoke-run",
        set_as_benchmark=True,
        benchmark_label="benchmark_v1",
    )

    assert record["benchmark_status"] == "benchmark"
    latest_payload = json.loads((tracker_dir / "latest_run.json").read_text(encoding="utf-8"))
    index_payload = json.loads((tracker_dir / "run_history_index.json").read_text(encoding="utf-8"))
    benchmark_payload = json.loads((tracker_dir / "benchmark.json").read_text(encoding="utf-8"))

    assert latest_payload["run_label"] == "smoke-run"
    assert index_payload["row_count"] == 1
    assert benchmark_payload["benchmark_label"] == "benchmark_v1"


def test_frontend_connection_contract_points_to_tracker_files(tmp_path: Path) -> None:
    payload = write_frontend_connection_contract(tracking_dir=tmp_path / "tracker")
    connection_json = json.loads((tmp_path / "tracker" / "frontend_connection.json").read_text(encoding="utf-8"))
    markdown_text = (tmp_path / "tracker" / "FRONTEND_CONNECTION.md").read_text(encoding="utf-8")

    assert payload["app_key"] == "run_count_research_tracker"
    assert connection_json["primary_data_sources"]["latest_run"].endswith("latest_run.json")
    assert "Run Count Tracker Frontend Handoff" in markdown_text
