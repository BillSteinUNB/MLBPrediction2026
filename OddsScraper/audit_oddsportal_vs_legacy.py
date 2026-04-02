from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.clients.historical_odds_client import load_historical_odds_for_games  # noqa: E402
from src.model.data_builder import validate_run_count_training_data  # noqa: E402
from src.model.xgboost_trainer import _load_training_dataframe  # noqa: E402


def _load_games(training_data_path: str | Path, *, season: int, max_games: int | None) -> pd.DataFrame:
    validated = validate_run_count_training_data(Path(training_data_path))
    frame = _load_training_dataframe(validated).copy()
    season_values = pd.to_numeric(frame["season"], errors="coerce")
    frame = frame.loc[season_values == int(season)].copy()
    frame = frame.sort_values(["game_date", "game_pk"]).reset_index(drop=True)
    if max_games is not None:
        frame = frame.head(int(max_games)).copy()
    return frame


def _favorite_side(frame: pd.DataFrame) -> pd.Series:
    home = pd.to_numeric(frame["home_odds"], errors="coerce")
    away = pd.to_numeric(frame["away_odds"], errors="coerce")
    implied_home = pd.Series(
        [(-v / (-v + 100.0)) if pd.notna(v) and v < 0 else (100.0 / (v + 100.0)) if pd.notna(v) else None for v in home]
    )
    implied_away = pd.Series(
        [(-v / (-v + 100.0)) if pd.notna(v) and v < 0 else (100.0 / (v + 100.0)) if pd.notna(v) else None for v in away]
    )
    return pd.Series(
        ["home" if h > a else "away" if a > h else "tie" for h, a in zip(implied_home, implied_away, strict=False)],
        index=frame.index,
    )


def _runline_is_valid(frame: pd.DataFrame) -> pd.Series:
    home = pd.to_numeric(frame["home_point"], errors="coerce")
    away = pd.to_numeric(frame["away_point"], errors="coerce")
    return (
        home.notna()
        & away.notna()
        & (home * away < 0)
        & ((home.abs() - away.abs()).abs() <= 1e-9)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit OddsPortal historical semantics against legacy historical odds.")
    parser.add_argument("--training-data", default="data/training/ParquetDefault.parquet")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--max-games", type=int, default=250)
    parser.add_argument("--legacy-db", default="OddsScraper/data/mlb_odds.db")
    parser.add_argument("--oddsportal-db", default="data/mlb_odds_oddsportal.db")
    args = parser.parse_args(argv)

    games = _load_games(args.training_data, season=args.season, max_games=args.max_games)
    payload: dict[str, object] = {
        "season": int(args.season),
        "game_sample_size": int(len(games)),
        "markets": {},
    }

    for market_type in ("full_game_ml", "f5_ml", "full_game_rl", "f5_rl", "full_game_total", "f5_total"):
        legacy = load_historical_odds_for_games(
            db_path=args.legacy_db,
            games_frame=games,
            market_type=market_type,
            snapshot_selection="opening",
        )
        oddsportal = load_historical_odds_for_games(
            db_path=args.oddsportal_db,
            games_frame=games,
            market_type=market_type,
            snapshot_selection="opening",
        )
        merged = legacy.merge(
            oddsportal,
            on="game_pk",
            suffixes=("_legacy", "_oddsportal"),
            how="inner",
        )
        market_payload: dict[str, object] = {
            "legacy_rows": int(len(legacy)),
            "oddsportal_rows": int(len(oddsportal)),
            "overlap_rows": int(len(merged)),
        }
        if market_type.endswith("ml") and not merged.empty:
            market_payload["favorite_match_rate"] = float(
                (_favorite_side(merged.rename(columns={"home_odds_legacy": "home_odds", "away_odds_legacy": "away_odds"}))
                 == _favorite_side(merged.rename(columns={"home_odds_oddsportal": "home_odds", "away_odds_oddsportal": "away_odds"}))
                ).mean()
            )
        if market_type.endswith("rl") and not merged.empty:
            legacy_valid = _runline_is_valid(
                merged.rename(columns={"home_point_legacy": "home_point", "away_point_legacy": "away_point"})
            )
            oddsportal_valid = _runline_is_valid(
                merged.rename(columns={"home_point_oddsportal": "home_point", "away_point_oddsportal": "away_point"})
            )
            market_payload["legacy_valid_pair_rate"] = float(legacy_valid.mean())
            market_payload["oddsportal_valid_pair_rate"] = float(oddsportal_valid.mean())
        if market_type.endswith("total") and not merged.empty:
            legacy_total = pd.to_numeric(merged["total_point_legacy"], errors="coerce")
            oddsportal_total = pd.to_numeric(merged["total_point_oddsportal"], errors="coerce")
            market_payload["legacy_point_present_rate"] = float(legacy_total.notna().mean())
            market_payload["oddsportal_point_present_rate"] = float(oddsportal_total.notna().mean())
        payload["markets"][market_type] = market_payload

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
