from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.ops.autoresearch import (
    _build_command,
    _constraints_passed,
    _mutate_params,
    _objective_value,
    _parse_constraints,
    _parse_objective,
    _write_leaderboard,
)


def test_build_command_includes_output_dir_and_boolean_flags(tmp_path: Path) -> None:
    command = _build_command(
        module="src.backtest.walk_forward",
        fixed_args={"start": "2021-01-01", "refresh-data": True},
        trial_params={"edge-threshold": 0.05},
        output_dir=tmp_path / "trial-001",
    )

    assert command[:3] == ["python", "-m", "src.backtest.walk_forward"] or command[1:3] == ["-m", "src.backtest.walk_forward"]
    assert "--start" in command
    assert "--refresh-data" in command
    assert "--edge-threshold" in command
    assert "--output-dir" in command

    training_command = _build_command(
        module="src.model.calibration",
        fixed_args={"training-data": "data/training/sample.parquet"},
        trial_params={"edge-threshold": 0.05},
        output_dir=tmp_path / "trial-002",
    )
    assert "--experiment-name" in training_command


def test_mutate_params_changes_subset_of_values() -> None:
    class _StubRandom:
        def sample(self, population, k):
            return population[:k]

        def choice(self, values):
            return values[-1]

    result = _mutate_params(
        base_params={"edge-threshold": 0.05, "staking-mode": "flat"},
        search_space={
            "edge-threshold": {"values": [0.04, 0.05, 0.06]},
            "staking-mode": {"values": ["flat", "edge_scaled"]},
        },
        mutation_count=2,
        rng=_StubRandom(),
    )

    assert result["edge-threshold"] == 0.06
    assert result["staking-mode"] == "edge_scaled"


def test_constraints_and_leaderboard_sorting(tmp_path: Path) -> None:
    objective = _parse_objective({"objective": {"metric": "bankroll_return_pct", "direction": "max"}})
    constraints = _parse_constraints(
        {"constraints": [{"metric": "max_drawdown_pct", "operator": "<=", "value": 0.30}]}
    )

    assert _objective_value({"bankroll_return_pct": 0.25}, objective) == 0.25
    assert _constraints_passed({"max_drawdown_pct": 0.29}, constraints) is True
    assert _constraints_passed({"max_drawdown_pct": 0.31}, constraints) is False

    records = [
        {
            "trial_index": 1,
            "trial_name": "trial-001",
            "experiment_name": "exp-1",
            "status": "success",
            "duration_seconds": 1.0,
            "constraints_passed": True,
            "objective_value": 0.10,
            "git_head": "abc",
            "metrics": {"bankroll_return_pct": 0.10, "max_drawdown_pct": 0.29},
            "params": {"edge-threshold": 0.05},
        },
        {
            "trial_index": 2,
            "trial_name": "trial-002",
            "experiment_name": "exp-2",
            "status": "success",
            "duration_seconds": 1.0,
            "constraints_passed": True,
            "objective_value": 0.20,
            "git_head": "abc",
            "metrics": {"bankroll_return_pct": 0.20, "max_drawdown_pct": 0.28},
            "params": {"edge-threshold": 0.06},
        },
    ]
    leaderboard_path = tmp_path / "leaderboard.csv"
    _write_leaderboard(records, leaderboard_path, objective, constraints)

    leaderboard = pd.read_csv(leaderboard_path)
    assert leaderboard.iloc[0]["experiment_name"] == "exp-2"
