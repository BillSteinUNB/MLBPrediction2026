from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import sys

AUTORESEARCH_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = AUTORESEARCH_ROOT.parent
for path in (AUTORESEARCH_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import agent  # noqa: E402
import launcher  # noqa: E402
import llm_client  # noqa: E402
import train  # noqa: E402


def test_resolve_bucket_targets_supports_three_value_layout() -> None:
    targets = train.resolve_bucket_targets(max_features=100, bucket_quotas=[24, 36, 20])

    assert targets == {
        "short_form": 24,
        "medium_form": 36,
        "delta": 0,
        "context": 40,
    }


def test_update_train_config_rewrites_agent_block(tmp_path: Path) -> None:
    train_path = tmp_path / "train.py"
    train_path.write_text(
        "\n".join(
            [
                "before = True",
                "# AGENT_CONFIG_START",
                "MAX_FEATURES = 80",
                'SELECTOR_TYPE = "bucketed"',
                "BUCKET_QUOTAS = [24, 28, 12, 16]",
                "EXCLUDE_PATTERNS: list[str] = []",
                "FORCE_INCLUDE_PATTERNS: list[str] = []",
                "FORCED_DELTA_COUNT = 8",
                "TRIALS = 50",
                "FOLDS = 3",
                "# AGENT_CONFIG_END",
                "after = True",
            ]
        ),
        encoding="utf-8",
    )
    proposal = agent.ExperimentProposal(
        max_features=120,
        selector_type="ablation",
        bucket_quotas=[24, 36, 20],
        exclude_patterns=["weather_*"],
        force_include_patterns=["*_7g"],
        forced_delta_count=8,
        trials=50,
        folds=3,
        rationale="test",
    )

    agent.update_train_config(train_path=train_path, proposal=proposal)

    updated = train_path.read_text(encoding="utf-8")
    assert "MAX_FEATURES = 120" in updated
    assert 'SELECTOR_TYPE = "ablation"' in updated
    assert 'BUCKET_QUOTAS = [24, 36, 20]' in updated
    assert 'EXCLUDE_PATTERNS: list[str] = ["weather_*"]' in updated
    assert 'FORCE_INCLUDE_PATTERNS: list[str] = ["*_7g"]' in updated
    assert "FORCED_DELTA_COUNT = 8" in updated


def test_plan_next_experiment_returns_baseline_without_history(monkeypatch) -> None:
    monkeypatch.setattr(llm_client, "load_llm_config", lambda: None)

    decision = agent.plan_next_experiment([], program_text="baseline")

    assert decision.proposal.max_features == 80
    assert decision.proposal.selector_type == "pearson"
    assert "baseline" in decision.hypothesis.lower()
    assert decision.planner_type == "heuristic"


def test_plan_next_experiment_uses_llm_when_configured(monkeypatch) -> None:
    monkeypatch.setattr(
        llm_client,
        "load_llm_config",
        lambda: llm_client.LLMConfig(
            provider="droid",
            model="custom:Test-Model-0",
            command="droid",
            reasoning_effort="medium",
        ),
    )
    monkeypatch.setattr(
        llm_client,
        "generate_text",
        lambda **_kwargs: llm_client.LLMTextResponse(
            provider="droid",
            model="custom:Test-Model-0",
            text=json.dumps(
                {
                    "hypothesis": "Flat Pearson 72-feature selection may improve local refinement.",
                    "reasoning": "The prior winner suggests flat ranking is underexplored.",
                    "config": {
                        "max_features": 72,
                        "selector_type": "pearson",
                        "bucket_quotas": [72, 0, 0, 0],
                        "exclude_patterns": [],
                        "force_include_patterns": [],
                        "trials": 50,
                        "folds": 3,
                    },
                }
            ),
            raw_payload={},
        ),
    )

    decision = agent.plan_next_experiment([], program_text="baseline")

    assert decision.planner_type == "llm"
    assert decision.planner_model == "droid:custom:Test-Model-0"
    assert decision.proposal.max_features == 72
    assert decision.proposal.selector_type == "pearson"


def test_planner_prompt_and_validation_follow_current_max_feature_surface(monkeypatch) -> None:
    prompt_text = agent._planner_user_prompt(
        program_text="baseline",
        history_rows=[],
        session_context=None,
        exploration_mode="fast",
    )

    assert "[72, 80, 88]" in prompt_text

    monkeypatch.setattr(
        llm_client,
        "load_llm_config",
        lambda: llm_client.LLMConfig(
            provider="droid",
            model="custom:Test-Model-0",
            command="droid",
            reasoning_effort="medium",
        ),
    )
    monkeypatch.setattr(
        llm_client,
        "generate_text",
        lambda **_kwargs: llm_client.LLMTextResponse(
            provider="droid",
            model="custom:Test-Model-0",
            text=json.dumps(
                {
                    "hypothesis": "Use the stage-2 local refinement width.",
                    "reasoning": "72 features is part of the current forced-delta local search surface.",
                    "config": {
                        "max_features": 72,
                        "selector_type": "pearson",
                        "bucket_quotas": [72, 0, 0, 0],
                        "exclude_patterns": [],
                        "force_include_patterns": [],
                        "trials": 50,
                        "folds": 3,
                    },
                }
            ),
            raw_payload={},
        ),
    )

    decision = agent.plan_next_experiment([], program_text="baseline")

    assert decision.proposal.max_features == 72


def test_plan_next_experiment_falls_back_when_llm_response_is_invalid(monkeypatch) -> None:
    monkeypatch.setattr(
        llm_client,
        "load_llm_config",
        lambda: llm_client.LLMConfig(
            provider="droid",
            model="custom:Test-Model-0",
            command="droid",
            reasoning_effort="medium",
        ),
    )
    monkeypatch.setattr(
        llm_client,
        "generate_text",
        lambda **_kwargs: llm_client.LLMTextResponse(
            provider="droid",
            model="custom:Test-Model-0",
            text='{"hypothesis":"bad","reasoning":"bad","config":{"max_features":55}}',
            raw_payload={},
        ),
    )

    decision = agent.plan_next_experiment([], program_text="baseline")

    assert decision.planner_type == "heuristic_fallback"
    assert "fallback" in decision.reasoning.lower()


def test_best_fast_experiment_orders_by_holdout_r2_then_cv_rmse(tmp_path: Path) -> None:
    db_path = agent.ensure_experiment_db(tmp_path / "experiments.db")
    first_config = {
        "max_features": 80,
        "selector_type": "bucketed",
        "bucket_quotas": [24, 28, 12, 16],
        "exclude_patterns": [],
        "force_include_patterns": [],
        "trials": 50,
        "folds": 3,
    }
    second_config = {
        "max_features": 100,
        "selector_type": "pearson",
        "bucket_quotas": [24, 28, 12, 16],
        "exclude_patterns": [],
        "force_include_patterns": [],
        "trials": 50,
        "folds": 3,
    }
    started_at = datetime(2026, 3, 27, 22, 0, 0)

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO experiments (
                started_at,
                completed_at,
                mode,
                status,
                hypothesis,
                config_json,
                config_fingerprint,
                experiment_name,
                holdout_r2,
                cv_rmse
            )
            VALUES (?, ?, 'fast', 'succeeded', ?, ?, ?, ?, ?, ?)
            """,
            (
                started_at.isoformat(),
                (started_at + timedelta(minutes=20)).isoformat(),
                "first",
                json.dumps(first_config, sort_keys=True),
                agent.config_fingerprint(first_config),
                "fast-one",
                0.041,
                3.30,
            ),
        )
        connection.execute(
            """
            INSERT INTO experiments (
                started_at,
                completed_at,
                mode,
                status,
                hypothesis,
                config_json,
                config_fingerprint,
                experiment_name,
                holdout_r2,
                cv_rmse
            )
            VALUES (?, ?, 'fast', 'succeeded', ?, ?, ?, ?, ?, ?)
            """,
            (
                (started_at + timedelta(minutes=25)).isoformat(),
                (started_at + timedelta(minutes=45)).isoformat(),
                "second",
                json.dumps(second_config, sort_keys=True),
                agent.config_fingerprint(second_config),
                "fast-two",
                0.041,
                3.12,
            ),
        )
        connection.commit()

    best = agent.best_fast_experiment(db_path)

    assert best is not None
    assert best["experiment_name"] == "fast-two"


def test_ensure_experiment_db_adds_planner_columns(tmp_path: Path) -> None:
    db_path = agent.ensure_experiment_db(tmp_path / "experiments.db")

    with sqlite3.connect(db_path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(experiments)").fetchall()}

    assert "planner_type" in columns
    assert "planner_model" in columns
    assert "planner_prompt_path" in columns
    assert "planner_response_path" in columns


def test_should_start_fast_run_respects_remaining_window() -> None:
    now = datetime(2026, 3, 27, 22, 0, 0)
    stop_at = now + timedelta(minutes=35)

    assert launcher.should_start_fast_run(
        now=now,
        stop_at=stop_at,
        min_fast_window_minutes=30,
    )
    assert not launcher.should_start_fast_run(
        now=now,
        stop_at=now + timedelta(minutes=20),
        min_fast_window_minutes=30,
    )
    assert launcher.should_start_fast_run(
        now=now,
        stop_at=None,
        min_fast_window_minutes=30,
    )


def test_resolve_droid_model_from_settings_payload_prefers_glm_zai() -> None:
    model_id = llm_client._resolve_droid_model_from_settings_payload(
        {
            "customModels": [
                {
                    "id": "custom:Other-Model-0",
                    "displayName": "Other Model",
                    "model": "other",
                    "baseUrl": "https://example.com/v1",
                },
                {
                    "id": "custom:GLM-5.1-(Z.AI)-1",
                    "displayName": "GLM-5.1 (Z.AI)",
                    "model": "glm-5.1",
                    "baseUrl": "https://api.z.ai/v1",
                },
                {
                    "id": "custom:GLM-5-(Z.AI)-2",
                    "displayName": "GLM-5 (Z.AI)",
                    "model": "glm-5",
                    "baseUrl": "https://api.z.ai/v1",
                },
            ]
        }
    )

    assert model_id == "custom:GLM-5.1-(Z.AI)-1"


def test_write_session_summary_persists_report_and_notes(tmp_path: Path) -> None:
    db_path = agent.ensure_experiment_db(tmp_path / "experiments.db")
    session_id = agent.create_session(
        db_path=db_path,
        config=agent.AutoresearchSessionConfig(
            exploration_mode="fast",
            duration_hours=8,
            until_interrupted=False,
            run_full_at_end=True,
        ),
        stop_at=None,
    )

    config = {
        "max_features": 80,
        "selector_type": "pearson",
        "bucket_quotas": [24, 28, 12, 16],
        "exclude_patterns": [],
        "force_include_patterns": [],
        "trials": 50,
        "folds": 3,
    }
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO experiments (
                started_at,
                completed_at,
                mode,
                status,
                hypothesis,
                config_json,
                config_fingerprint,
                experiment_name,
                holdout_r2,
                cv_rmse,
                session_id
            )
            VALUES (?, ?, 'fast', 'succeeded', ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime(2026, 3, 27, 22, 0, 0).isoformat(),
                datetime(2026, 3, 27, 22, 20, 0).isoformat(),
                "best trial",
                json.dumps(config, sort_keys=True),
                agent.config_fingerprint(config),
                "fast-one",
                0.052,
                3.05,
                session_id,
            ),
        )
        connection.commit()

    agent.record_note(
        db_path=db_path,
        session_id=session_id,
        experiment_id=1,
        note_type="new_best",
        importance="high",
        title="New best",
        body="Flat selector improved holdout R².",
        reports_dir=tmp_path / "reports",
    )

    summary, json_path, md_path, review_prompt_path = agent.write_session_summary(
        session_id,
        db_path=db_path,
        reports_dir=tmp_path / "reports",
        status_override="completed",
    )

    assert json_path.exists()
    assert md_path.exists()
    assert review_prompt_path.exists()
    assert summary["best_exploration"]["experiment_name"] == "fast-one"
    assert summary["notes"][0]["title"] == "New best"
    history_log = tmp_path / "reports" / "session_history.jsonl"
    assert history_log.exists()


def test_append_nightly_log_writes_markdown_entry(tmp_path: Path) -> None:
    log_path = agent.append_nightly_log(
        event_type="test_event",
        heading="Test heading",
        body_lines=["session_id: `7`", "status: `ok`"],
        reports_dir=tmp_path / "reports",
    )

    content = log_path.read_text(encoding="utf-8")
    assert "Test heading" in content
    assert "event_type: `test_event`" in content


def test_extract_run_diagnostics_reads_metadata_file(tmp_path: Path) -> None:
    db_path = agent.ensure_experiment_db(tmp_path / "experiments.db")
    output_dir = tmp_path / "model_output"
    output_dir.mkdir()
    metadata_path = output_dir / "model.metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "cv_metric_name": "poisson_deviance",
                "cv_best_score": 2.31,
                "holdout_metrics": {"r2": 0.04, "poisson_deviance": 2.51},
                "feature_columns": ["a", "b"],
                "feature_importance_rankings": [{"feature": "a", "importance": 0.5}],
                "selected_features_by_bucket": {"short_form": ["a"]},
                "omitted_top_features_by_bucket": {"delta": [{"feature": "d", "score": 0.2}]},
                "feature_health_diagnostics": {
                    "expected_delta_columns": {"all_present": False, "missing_columns": ["x_delta"]},
                    "selected_feature_counts": {"family": {"weather": 1}},
                    "selected_feature_fill_health": {
                        "holdout": {
                            "top_default_fill_share": [{"feature": "a", "default_fill_share": 0.9}]
                        }
                    },
                    "selected_feature_drift": {"top_distribution_shift": []},
                },
            }
        ),
        encoding="utf-8",
    )
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO experiments (
                started_at, mode, status, hypothesis, config_json, config_fingerprint, experiment_name, output_dir
            )
            VALUES (?, 'fast', 'succeeded', ?, ?, ?, ?, ?)
            """,
            (
                datetime(2026, 3, 27, 22, 0, 0).isoformat(),
                "hypothesis",
                json.dumps({"max_features": 80}, sort_keys=True),
                "fingerprint",
                "fast-one",
                str(output_dir),
            ),
        )
        connection.commit()

    row = agent.load_experiment(1, db_path=db_path)
    diagnostics = agent._extract_run_diagnostics(row)

    assert diagnostics is not None
    assert diagnostics["cv_metric_name"] == "poisson_deviance"
    assert diagnostics["expected_delta_columns"]["all_present"] is False


def test_build_session_context_includes_artifact_reviews(tmp_path: Path) -> None:
    db_path = agent.ensure_experiment_db(tmp_path / "experiments.db")
    session_id = agent.create_session(
        db_path=db_path,
        config=agent.AutoresearchSessionConfig(
            exploration_mode="fast",
            duration_hours=2,
            until_interrupted=False,
            run_full_at_end=False,
        ),
        stop_at=None,
    )
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "holdout_metrics": {"r2": 0.05},
                "feature_columns": ["a"],
                "feature_importance_rankings": [{"feature": "a", "importance": 1.0}],
                "selected_features_by_bucket": {"short_form": ["a"]},
                "omitted_top_features_by_bucket": {},
                "feature_health_diagnostics": {},
            }
        ),
        encoding="utf-8",
    )
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO experiments (
                started_at, completed_at, mode, status, hypothesis, config_json, config_fingerprint,
                experiment_name, holdout_r2, cv_rmse, session_id, summary_path
            )
            VALUES (?, ?, 'fast', 'succeeded', ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime(2026, 3, 27, 22, 0, 0).isoformat(),
                datetime(2026, 3, 27, 22, 10, 0).isoformat(),
                "good run",
                json.dumps({"max_features": 80, "selector_type": "pearson"}, sort_keys=True),
                "fingerprint",
                "fast-one",
                0.05,
                3.1,
                session_id,
                str(summary_path),
            ),
        )
        connection.commit()

    context = agent.build_session_context(session_id, db_path=db_path)

    assert len(context["artifact_reviews"]) == 1
    assert context["artifact_reviews"][0]["diagnostics"]["feature_column_count"] == 1


def test_session_summary_tracks_artifact_file_names(tmp_path: Path) -> None:
    db_path = agent.ensure_experiment_db(tmp_path / "experiments.db")
    session_id = agent.create_session(
        db_path=db_path,
        config=agent.AutoresearchSessionConfig(
            exploration_mode="fast",
            duration_hours=2,
            until_interrupted=False,
            run_full_at_end=False,
        ),
        stop_at=None,
    )
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    (output_dir / "result.json").write_text("{}", encoding="utf-8")
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO experiments (
                started_at, completed_at, mode, status, hypothesis, config_json, config_fingerprint,
                experiment_name, holdout_r2, cv_rmse, session_id, output_dir
            )
            VALUES (?, ?, 'fast', 'succeeded', ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime(2026, 3, 27, 22, 0, 0).isoformat(),
                datetime(2026, 3, 27, 22, 10, 0).isoformat(),
                "good run",
                json.dumps({"max_features": 80, "selector_type": "pearson"}, sort_keys=True),
                "fingerprint",
                "fast-one",
                0.05,
                3.1,
                session_id,
                str(output_dir),
            ),
        )
        connection.commit()

    summary = agent.build_session_summary(session_id, db_path=db_path)

    assert summary["artifact_references"][0]["files"][0]["file_name"] == "result.json"


def test_maybe_record_experiment_notes_adds_diagnostic_summary(tmp_path: Path) -> None:
    db_path = agent.ensure_experiment_db(tmp_path / "experiments.db")
    session_id = agent.create_session(
        db_path=db_path,
        config=agent.AutoresearchSessionConfig(
            exploration_mode="fast",
            duration_hours=2,
            until_interrupted=False,
            run_full_at_end=False,
        ),
        stop_at=None,
    )
    output_dir = tmp_path / "model_output"
    output_dir.mkdir()
    metadata_path = output_dir / "model.metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "holdout_metrics": {"poisson_deviance": 2.4},
                "feature_columns": ["a", "b"],
                "feature_importance_rankings": [{"feature": "a", "importance": 0.5}],
                "selected_features_by_bucket": {"short_form": ["a"]},
                "omitted_top_features_by_bucket": {"delta": [{"feature": "delta_a"}]},
                "feature_health_diagnostics": {
                    "expected_delta_columns": {"all_present": False, "missing_columns": ["delta_missing"]},
                    "selected_feature_counts": {"family": {"weather": 2}},
                    "selected_feature_fill_health": {
                        "holdout": {
                            "top_default_fill_share": [{"feature": "a", "default_fill_share": 0.9}]
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    proposal = agent.ExperimentProposal(
        max_features=80,
        selector_type="pearson",
        bucket_quotas=[24, 28, 12, 16],
        exclude_patterns=[],
        force_include_patterns=[],
        forced_delta_count=8,
        trials=50,
        folds=3,
        rationale="test",
    )
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO experiments (
                started_at, completed_at, mode, status, hypothesis, config_json, config_fingerprint,
                experiment_name, holdout_r2, cv_rmse, session_id, output_dir
            )
            VALUES (?, ?, 'fast', 'succeeded', ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime(2026, 3, 27, 22, 0, 0).isoformat(),
                datetime(2026, 3, 27, 22, 10, 0).isoformat(),
                "good run",
                json.dumps(agent.proposal_to_snapshot(proposal), sort_keys=True),
                "fingerprint",
                "fast-one",
                0.05,
                3.1,
                session_id,
                str(output_dir),
            ),
        )
        connection.commit()

    note_ids = agent.maybe_record_experiment_notes(
        db_path=db_path,
        experiment_id=1,
        session_id=session_id,
        proposal=proposal,
        exploration_mode="fast",
        prior_session_best=None,
        result_payload={
            "experiment_name": "fast-one",
            "metrics": {"holdout_r2": 0.05, "cv_rmse": 3.1, "holdout_rmse": 3.2},
        },
        status="succeeded",
        error_message=None,
        reports_dir=tmp_path / "reports",
    )

    rows = agent.load_notes(db_path=db_path, session_id=session_id)
    assert any(row["note_type"] == "diagnostic_summary" for row in rows)
    assert len(note_ids) >= 2


def test_pending_suspected_issues_excludes_already_validated_notes(tmp_path: Path) -> None:
    db_path = agent.ensure_experiment_db(tmp_path / "experiments.db")
    session_id = agent.create_session(
        db_path=db_path,
        config=agent.AutoresearchSessionConfig(
            exploration_mode="fast",
            duration_hours=4,
            until_interrupted=False,
            run_full_at_end=False,
        ),
        stop_at=None,
    )
    note_id = agent.record_note(
        db_path=db_path,
        session_id=session_id,
        experiment_id=None,
        note_type="failure",
        importance="high",
        title="Experiment failed",
        body="broken thing",
        metadata={"config": {"max_features": 80, "selector_type": "bucketed", "bucket_quotas": [24, 28, 12, 16], "exclude_patterns": [], "force_include_patterns": [], "trials": 50, "folds": 3}},
        reports_dir=tmp_path / "reports",
    )

    pending_before = agent.pending_suspected_issues(db_path=db_path, session_id=session_id)
    assert [issue.note_id for issue in pending_before] == [note_id]

    agent.record_issue_validation(
        db_path=db_path,
        session_id=session_id,
        issue_note_id=note_id,
        source_experiment_id=None,
        validation_experiment_id=None,
        status="skipped",
        outcome="missing_config",
        details={},
    )

    pending_after = agent.pending_suspected_issues(db_path=db_path, session_id=session_id)
    assert pending_after == []


def test_collect_session_config_uses_interactive_prompts(monkeypatch) -> None:
    answers = iter(["0", "n"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

    config = launcher.collect_session_config(
        duration_hours=None,
        exploration_mode=None,
        run_full_at_end=None,
    )

    assert config.duration_hours == 0
    assert config.until_interrupted is True
    assert config.exploration_mode == "fast"
    assert config.run_full_at_end is False


def test_launcher_runs_planner_self_check_before_full_promotion(monkeypatch) -> None:
    events: list[str] = []

    monkeypatch.setattr(agent, "ensure_experiment_db", lambda _db_path: Path("experiments.db"))
    monkeypatch.setattr(
        launcher,
        "prepare_git_checkpoint",
        lambda: {"branch_name": "AutoResearch-2026-03-27"},
    )
    monkeypatch.setattr(
        launcher,
        "collect_session_config",
        lambda **_kwargs: agent.AutoresearchSessionConfig(
            exploration_mode="fast",
            duration_hours=1,
            until_interrupted=False,
            run_full_at_end=True,
        ),
    )
    monkeypatch.setattr(agent, "create_session", lambda **_kwargs: 7)
    monkeypatch.setattr(agent, "finalize_session", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(agent, "append_nightly_log", lambda **_kwargs: Path("nightly_log.md"))
    monkeypatch.setattr(
        agent,
        "run_planner_self_check",
        lambda: events.append("self-check") or {"provider": "droid", "model": "custom:GLM-5.1-(Z.AI)-1"},
    )
    monkeypatch.setattr(agent, "pending_suspected_issues", lambda **_kwargs: [])
    monkeypatch.setattr(
        launcher,
        "resolve_stop_at",
        lambda **_kwargs: datetime.now() - timedelta(minutes=1),
    )
    monkeypatch.setattr(
        agent,
        "run_best_full",
        lambda **_kwargs: events.append("full-run") or {"status": "succeeded"},
    )
    monkeypatch.setattr(
        agent,
        "write_session_summary",
        lambda *args, **kwargs: (
            {
                "best_exploration": {
                    "experiment_name": "fast-one",
                    "holdout_r2": 0.051,
                    "cv_rmse": 3.1,
                },
                "notes": [{"title": "note"}],
                "recommendations": ["keep going"],
                "session": {
                    "status": kwargs.get("status_override", "completed"),
                    "ended_at": "2026-03-27T00:00:00+00:00",
                },
            },
            Path("summary.json"),
            Path("summary.md"),
            Path("review_prompt.md"),
        ),
    )

    payload = launcher.run_launcher()

    assert payload["status"] == "succeeded"
    assert events == ["self-check", "full-run"]
    assert payload["session_summary"]["summary_json_path"] == "summary.json"


def test_prepare_git_checkpoint_runs_expected_git_flow(monkeypatch) -> None:
    calls: list[tuple[str, ...]] = []

    class Result:
        def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0) -> None:
            self.stdout = stdout
            self.stderr = stderr
            self.returncode = returncode

    def fake_run_git_command(*args: str):
        calls.append(args)
        if args == ("status", "--short"):
            return Result(stdout=" M file.py\n")
        if args == ("rev-parse", "--abbrev-ref", "HEAD"):
            return Result(stdout="main\n")
        return Result(stdout="")

    monkeypatch.setattr(launcher, "_run_git_command", fake_run_git_command)
    monkeypatch.setattr(launcher, "_git_branch_name", lambda now=None: "AutoResearch-2026-03-27")
    monkeypatch.setattr(agent, "append_nightly_log", lambda **_kwargs: Path("nightly_log.md"))

    payload = launcher.prepare_git_checkpoint()

    assert payload["branch_name"] == "AutoResearch-2026-03-27"
    assert ("push", "origin", "main") in calls
    assert ("checkout", "-B", "AutoResearch-2026-03-27") in calls
    assert ("push", "-u", "origin", "AutoResearch-2026-03-27") in calls
