from __future__ import annotations

import argparse
import io
import json
import sqlite3
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd

from src.db import init_db
from src.models.lineup import Lineup, LineupPlayer
from src.models.odds import OddsSnapshot
from src.models.prediction import Prediction
from src.notifications.discord import (
    send_drawdown_alert,
    send_failure_alert,
    send_no_picks,
    send_picks,
)
from src.pipeline import daily


@dataclass(frozen=True)
class Scenario:
    name: str
    pipeline_date: str
    mode: str
    dry_run: bool
    starting_bankroll: float
    schedule: pd.DataFrame
    historical_games: pd.DataFrame
    lineups_fetcher: Callable[[], list[Lineup]]
    odds_fetcher: Callable[[], list[OddsSnapshot]]
    feature_frame_builder: Callable[..., pd.DataFrame]
    prediction_engine: "RecordingPredictionEngine"
    seed_database: Callable[[Path], None] | None = None


class RecordingPredictionEngine:
    model_version = "validation-probe-v1"

    def __init__(
        self,
        predictions: dict[int, Prediction],
        *,
        errors: dict[int, Exception] | None = None,
    ) -> None:
        self._predictions = predictions
        self._errors = errors or {}

    def predict(self, inference_frame: pd.DataFrame) -> Prediction:
        game_pk = int(inference_frame.iloc[0]["game_pk"])
        if game_pk in self._errors:
            raise self._errors[game_pk]
        return self._predictions[game_pk]


class RecordingNotifier:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.network_requests = 0

    def send_picks(self, **payload: Any) -> dict[str, Any]:
        rendered = send_picks(**{**payload, "dry_run": True})
        self.calls.append({"type": "picks", "payload": rendered})
        return rendered

    def send_no_picks(self, **payload: Any) -> dict[str, Any]:
        rendered = send_no_picks(**{**payload, "dry_run": True})
        self.calls.append({"type": "no_picks", "payload": rendered})
        return rendered

    def send_failure_alert(self, **payload: Any) -> dict[str, Any]:
        rendered = send_failure_alert(**{**payload, "dry_run": True})
        self.calls.append({"type": "failure_alert", "payload": rendered})
        return rendered

    def send_drawdown_alert(self, **payload: Any) -> dict[str, Any]:
        rendered = send_drawdown_alert(**{**payload, "dry_run": True})
        self.calls.append({"type": "drawdown_alert", "payload": rendered})
        return rendered


def _players() -> list[LineupPlayer]:
    return [
        LineupPlayer(
            batting_order=slot,
            player_id=10000 + slot,
            player_name=f"Player {slot}",
            position="1B",
        )
        for slot in range(1, 10)
    ]


def _schedule_frame(specs: Sequence[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        rows.append(
            {
                "game_pk": int(spec["game_pk"]),
                "season": 2025,
                "game_date": "2025-09-15",
                "scheduled_start": str(spec.get("scheduled_start") or f"2025-09-15T{spec.get('hour', 18):02d}:05:00+00:00"),
                "home_team": str(spec.get("home_team", "NYY")),
                "away_team": str(spec.get("away_team", "BOS")),
                "home_starter_id": spec.get("home_starter_id", 1000 + int(spec["game_pk"])),
                "away_starter_id": spec.get("away_starter_id", 2000 + int(spec["game_pk"])),
                "venue": str(spec.get("venue", f"Stadium {spec['game_pk']}")),
                "is_dome": bool(spec.get("is_dome", False)),
                "is_abs_active": bool(spec.get("is_abs_active", True)),
                "park_runs_factor": float(spec.get("park_runs_factor", 1.02)),
                "park_hr_factor": float(spec.get("park_hr_factor", 1.01)),
                "game_type": str(spec.get("game_type", "R")),
                "status": str(spec.get("status", "scheduled")),
                "f5_home_score": spec.get("f5_home_score"),
                "f5_away_score": spec.get("f5_away_score"),
                "final_home_score": spec.get("final_home_score", spec.get("f5_home_score")),
                "final_away_score": spec.get("final_away_score", spec.get("f5_away_score")),
            }
        )
    return pd.DataFrame(rows)


def _make_lineups(
    schedule: pd.DataFrame,
    *,
    omit_game_pks: set[int] | None = None,
) -> list[Lineup]:
    omitted = omit_game_pks or set()
    lineups: list[Lineup] = []
    for row in schedule.to_dict(orient="records"):
        game_pk = int(row["game_pk"])
        if game_pk in omitted:
            continue
        for side in ("home", "away"):
            team = str(row[f"{side}_team"])
            starter_id = int(row[f"{side}_starter_id"])
            lineups.append(
                Lineup(
                    game_pk=game_pk,
                    team=team,
                    source="validation-probe",
                    confirmed=True,
                    as_of_timestamp=datetime(2025, 9, 15, tzinfo=UTC),
                    starting_pitcher_id=starter_id,
                    projected_starting_pitcher_id=starter_id,
                    starter_avg_innings_pitched=5.4,
                    is_opener=False,
                    is_bullpen_game=False,
                    players=_players(),
                )
            )
    return lineups


def _prediction(game_pk: int, *, ml_home: float, rl_home: float) -> Prediction:
    return Prediction(
        game_pk=game_pk,
        model_version="validation-probe-v1",
        f5_ml_home_prob=ml_home,
        f5_ml_away_prob=1.0 - ml_home,
        f5_rl_home_prob=rl_home,
        f5_rl_away_prob=1.0 - rl_home,
        predicted_at=datetime(2025, 9, 15, tzinfo=UTC),
    )


def _odds_snapshot(
    game_pk: int,
    *,
    market_type: str,
    home_odds: int,
    away_odds: int,
) -> OddsSnapshot:
    return OddsSnapshot(
        game_pk=game_pk,
        book_name="ValidationBook",
        market_type=market_type,
        home_odds=home_odds,
        away_odds=away_odds,
        fetched_at=datetime(2025, 9, 15, tzinfo=UTC),
    )


def _feature_frame_builder(
    *,
    weather_missing_games: set[int] | None = None,
) -> Callable[..., pd.DataFrame]:
    missing = weather_missing_games or set()

    def _builder(**kwargs: Any) -> pd.DataFrame:
        schedule = kwargs["schedule"]
        rows: list[dict[str, Any]] = []
        for game in schedule.to_dict(orient="records"):
            game_pk = int(game["game_pk"])
            rows.append(
                {
                    "game_pk": game_pk,
                    "scheduled_start": game["scheduled_start"],
                    "home_team": game["home_team"],
                    "away_team": game["away_team"],
                    "venue": game["venue"],
                    "park_runs_factor": float(game.get("park_runs_factor", 1.02)),
                    "park_hr_factor": float(game.get("park_hr_factor", 1.01)),
                    "home_team_f5_pythagorean_wp_30g": 0.56,
                    "home_team_log5_30g": 0.57,
                    "weather_composite": 1.0 if game_pk not in missing else 0.0,
                    "weather_wind_factor": 0.12 if game_pk not in missing else 0.0,
                    "weather_data_missing": 1.0 if game_pk in missing else 0.0,
                }
            )
        return pd.DataFrame(rows)

    return _builder


def _seed_kill_switch_db(db_path: Path) -> None:
    init_db(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "INSERT INTO bankroll_ledger (timestamp, event_type, amount, running_balance, notes) VALUES (?, ?, ?, ?, ?)",
            ("2025-09-14T00:00:00+00:00", "bet_settled", 0.0, 1000.0, "seed peak"),
        )
        connection.execute(
            "INSERT INTO bankroll_ledger (timestamp, event_type, amount, running_balance, notes) VALUES (?, ?, ?, ?, ?)",
            ("2025-09-14T01:00:00+00:00", "bet_settled", -310.0, 690.0, "seed drawdown"),
        )
        connection.commit()


def _scenario(name: str) -> Scenario:
    if name == "dry_run_pick":
        schedule = _schedule_frame([{"game_pk": 1001, "hour": 18, "home_team": "NYY", "away_team": "BOS"}])
        return Scenario(
            name=name,
            pipeline_date="2025-09-15",
            mode="backtest",
            dry_run=True,
            starting_bankroll=1000.0,
            schedule=schedule,
            historical_games=pd.DataFrame(),
            lineups_fetcher=lambda: _make_lineups(schedule),
            odds_fetcher=lambda: [_odds_snapshot(1001, market_type="f5_ml", home_odds=-110, away_odds=100)],
            feature_frame_builder=_feature_frame_builder(),
            prediction_engine=RecordingPredictionEngine({1001: _prediction(1001, ml_home=0.62, rl_home=0.56)}),
        )

    if name == "missing_data":
        schedule = _schedule_frame(
            [
                {"game_pk": 1101, "hour": 18, "home_team": "NYY", "away_team": "BOS"},
                {"game_pk": 1102, "hour": 19, "home_team": "ATL", "away_team": "PHI"},
                {"game_pk": 1103, "hour": 20, "home_team": "LAD", "away_team": "SD"},
            ]
        )
        return Scenario(
            name=name,
            pipeline_date="2025-09-15",
            mode="backtest",
            dry_run=True,
            starting_bankroll=1000.0,
            schedule=schedule,
            historical_games=pd.DataFrame(),
            lineups_fetcher=lambda: _make_lineups(schedule, omit_game_pks={1103}),
            odds_fetcher=lambda: [
                _odds_snapshot(1102, market_type="f5_ml", home_odds=-108, away_odds=102),
                _odds_snapshot(1103, market_type="f5_ml", home_odds=-105, away_odds=-105),
            ],
            feature_frame_builder=_feature_frame_builder(weather_missing_games={1102}),
            prediction_engine=RecordingPredictionEngine(
                {
                    1101: _prediction(1101, ml_home=0.58, rl_home=0.53),
                    1102: _prediction(1102, ml_home=0.57, rl_home=0.52),
                    1103: _prediction(1103, ml_home=0.59, rl_home=0.54),
                }
            ),
        )

    if name == "partial_error":
        schedule = _schedule_frame(
            [
                {"game_pk": 1201, "hour": 18, "home_team": "NYY", "away_team": "BOS"},
                {"game_pk": 1202, "hour": 19, "home_team": "ATL", "away_team": "PHI"},
            ]
        )
        return Scenario(
            name=name,
            pipeline_date="2025-09-15",
            mode="backtest",
            dry_run=True,
            starting_bankroll=1000.0,
            schedule=schedule,
            historical_games=pd.DataFrame(),
            lineups_fetcher=lambda: _make_lineups(schedule),
            odds_fetcher=lambda: [
                _odds_snapshot(1201, market_type="f5_ml", home_odds=-110, away_odds=100),
                _odds_snapshot(1202, market_type="f5_ml", home_odds=-110, away_odds=100),
            ],
            feature_frame_builder=_feature_frame_builder(),
            prediction_engine=RecordingPredictionEngine(
                {1202: _prediction(1202, ml_home=0.63, rl_home=0.57)},
                errors={1201: RuntimeError("boom")},
            ),
        )

    if name == "missing_odds_all":
        schedule = _schedule_frame(
            [
                {"game_pk": 1301, "hour": 18, "home_team": "NYY", "away_team": "BOS"},
                {"game_pk": 1302, "hour": 19, "home_team": "ATL", "away_team": "PHI"},
            ]
        )
        return Scenario(
            name=name,
            pipeline_date="2025-09-15",
            mode="backtest",
            dry_run=True,
            starting_bankroll=1000.0,
            schedule=schedule,
            historical_games=pd.DataFrame(),
            lineups_fetcher=lambda: _make_lineups(schedule),
            odds_fetcher=lambda: [],
            feature_frame_builder=_feature_frame_builder(),
            prediction_engine=RecordingPredictionEngine(
                {
                    1301: _prediction(1301, ml_home=0.58, rl_home=0.53),
                    1302: _prediction(1302, ml_home=0.57, rl_home=0.52),
                }
            ),
        )

    if name == "settle_pick":
        schedule = _schedule_frame(
            [
                {
                    "game_pk": 1401,
                    "hour": 18,
                    "home_team": "NYY",
                    "away_team": "BOS",
                    "status": "final",
                    "f5_home_score": 3,
                    "f5_away_score": 1,
                    "final_home_score": 5,
                    "final_away_score": 2,
                }
            ]
        )
        return Scenario(
            name=name,
            pipeline_date="2025-09-15",
            mode="backtest",
            dry_run=False,
            starting_bankroll=1000.0,
            schedule=schedule,
            historical_games=pd.DataFrame(),
            lineups_fetcher=lambda: _make_lineups(schedule),
            odds_fetcher=lambda: [_odds_snapshot(1401, market_type="f5_ml", home_odds=-110, away_odds=100)],
            feature_frame_builder=_feature_frame_builder(),
            prediction_engine=RecordingPredictionEngine({1401: _prediction(1401, ml_home=0.64, rl_home=0.58)}),
        )

    if name == "kill_switch":
        schedule = _schedule_frame([{"game_pk": 1501, "hour": 18, "home_team": "NYY", "away_team": "BOS"}])
        return Scenario(
            name=name,
            pipeline_date="2025-09-15",
            mode="backtest",
            dry_run=False,
            starting_bankroll=1000.0,
            schedule=schedule,
            historical_games=pd.DataFrame(),
            lineups_fetcher=lambda: _make_lineups(schedule),
            odds_fetcher=lambda: [_odds_snapshot(1501, market_type="f5_ml", home_odds=-110, away_odds=100)],
            feature_frame_builder=_feature_frame_builder(),
            prediction_engine=RecordingPredictionEngine({1501: _prediction(1501, ml_home=0.66, rl_home=0.59)}),
            seed_database=_seed_kill_switch_db,
        )

    raise ValueError(f"Unknown scenario: {name}")


def _rows(connection: sqlite3.Connection, query: str) -> list[dict[str, Any]]:
    connection.row_factory = sqlite3.Row
    cursor = connection.execute(query)
    return [dict(row) for row in cursor.fetchall()]


def _collect_db_summary(db_path: Path) -> dict[str, Any]:
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        return {
            "predictions": _rows(
                connection,
                "SELECT game_pk, model_version, f5_ml_home_prob, f5_rl_home_prob, predicted_at FROM predictions ORDER BY game_pk",
            ),
            "daily_pipeline_results": _rows(
                connection,
                """
                SELECT game_pk, status, selected_market_type, selected_side, odds_at_bet, edge_pct,
                       kelly_stake, no_pick_reason, error_message, notified
                FROM daily_pipeline_results
                ORDER BY game_pk
                """,
            ),
            "odds_snapshots": _rows(
                connection,
                "SELECT game_pk, market_type, home_odds, away_odds, is_frozen FROM odds_snapshots ORDER BY game_pk, id",
            ),
            "bets": _rows(
                connection,
                "SELECT game_pk, market_type, side, kelly_stake, odds_at_bet, result, profit_loss FROM bets ORDER BY id",
            ),
            "bankroll_ledger": _rows(
                connection,
                "SELECT event_type, amount, running_balance, notes FROM bankroll_ledger ORDER BY id",
            ),
        }


def _run_main_with_patches(scenario: Scenario, db_path: Path) -> tuple[int, str, RecordingNotifier]:
    notifier = RecordingNotifier()
    originals = {
        "schedule_fetcher": daily._default_schedule_fetcher,
        "history_fetcher": daily._default_history_fetcher,
        "lineups_fetcher": daily.fetch_confirmed_lineups,
        "odds_fetcher": daily._default_odds_fetcher,
        "feature_builder": daily._default_feature_frame_builder,
        "prediction_engine_factory": daily.ArtifactOrFallbackPredictionEngine,
        "notifier_factory": daily.DiscordNotifier,
    }

    daily._default_schedule_fetcher = lambda target_date, mode: scenario.schedule.copy()
    daily._default_history_fetcher = lambda season, before_date: scenario.historical_games.copy()
    daily.fetch_confirmed_lineups = lambda target_date: scenario.lineups_fetcher()
    daily._default_odds_fetcher = lambda target_date, mode, resolved_db_path: list(
        scenario.odds_fetcher()
    )
    daily._default_feature_frame_builder = scenario.feature_frame_builder
    daily.ArtifactOrFallbackPredictionEngine = lambda: scenario.prediction_engine
    daily.DiscordNotifier = lambda: notifier

    buffer = io.StringIO()
    try:
        with redirect_stdout(buffer):
            exit_code = daily.main(
                [
                    "--date",
                    scenario.pipeline_date,
                    "--mode",
                    scenario.mode,
                    "--db-path",
                    str(db_path),
                    "--starting-bankroll",
                    str(scenario.starting_bankroll),
                    *( ["--dry-run"] if scenario.dry_run else [] ),
                ]
            )
    finally:
        daily._default_schedule_fetcher = originals["schedule_fetcher"]
        daily._default_history_fetcher = originals["history_fetcher"]
        daily.fetch_confirmed_lineups = originals["lineups_fetcher"]
        daily._default_odds_fetcher = originals["odds_fetcher"]
        daily._default_feature_frame_builder = originals["feature_builder"]
        daily.ArtifactOrFallbackPredictionEngine = originals["prediction_engine_factory"]
        daily.DiscordNotifier = originals["notifier_factory"]

    return exit_code, buffer.getvalue(), notifier


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run isolated CLI pipeline validation scenarios")
    parser.add_argument("scenario")
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args(argv)

    scenario = _scenario(args.scenario)
    db_path = Path(args.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)

    if scenario.seed_database is not None:
        scenario.seed_database(db_path)

    exit_code, cli_stdout, notifier = _run_main_with_patches(scenario, db_path)
    parsed_stdout = json.loads(cli_stdout)
    summary = {
        "scenario": scenario.name,
        "invokedCli": {
            "module": "src.pipeline.daily",
            "args": {
                "date": scenario.pipeline_date,
                "mode": scenario.mode,
                "dry_run": scenario.dry_run,
                "db_path": str(db_path),
                "starting_bankroll": scenario.starting_bankroll,
            },
        },
        "exitCode": exit_code,
        "stdoutJson": parsed_stdout,
        "notifier": {
            "callCount": len(notifier.calls),
            "networkRequests": notifier.network_requests,
            "calls": notifier.calls,
        },
        "db": _collect_db_summary(db_path),
    }

    Path(args.summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(cli_stdout.rstrip())
    print(f"EXIT_CODE: {exit_code}")
    print(f"SUMMARY_JSON: {Path(args.summary_json)}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
