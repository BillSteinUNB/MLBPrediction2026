from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import requests

from src.config import _load_settings_yaml
from src.db import DEFAULT_DB_PATH, init_db, sqlite_connection


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HISTORICAL_ODDS_ROOT = REPO_ROOT / "data" / "raw" / "historical_odds"
DEFAULT_TRAINING_DATA_PATH = REPO_ROOT / "data" / "training" / "training_data_2018_2025.parquet"
DEFAULT_NORMALIZED_OUTPUT_PATH = DEFAULT_HISTORICAL_ODDS_ROOT / "sbr_f5_ml_2018_2025.parquet"
DEFAULT_CLOSING_OUTPUT_PATH = DEFAULT_HISTORICAL_ODDS_ROOT / "sbr_f5_ml_closing_2018_2025.parquet"
DEFAULT_RAW_OUTPUT_PATH = DEFAULT_HISTORICAL_ODDS_ROOT / "sbr_f5_ml_2018_2025.raw.json"
SBR_F5_ML_URL_TEMPLATE = (
    "https://www.sportsbookreview.com/betting-odds/mlb-baseball/money-line/1st-half/"
    "?date={game_date}"
)
SBR_TIMEOUT = 30.0
NEXT_DATA_PATTERN = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
    re.DOTALL,
)
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
F5_MARKET_TYPE = "f5_ml"


@dataclass(frozen=True, slots=True)
class DateFetchResult:
    game_date: str
    raw_games: list[dict[str, Any]]
    error: str | None = None


def load_training_schedule(training_path: str | Path = DEFAULT_TRAINING_DATA_PATH) -> pd.DataFrame:
    frame = pd.read_parquet(
        Path(training_path),
        columns=[
            "game_pk",
            "season",
            "game_date",
            "scheduled_start",
            "home_team",
            "away_team",
            "venue",
            "status",
            "f5_home_score",
            "f5_away_score",
            "final_home_score",
            "final_away_score",
            "abs_active",
        ],
    ).drop_duplicates(subset=["game_pk"])
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce").dt.date.astype(str)
    frame["scheduled_start"] = pd.to_datetime(frame["scheduled_start"], utc=True, errors="coerce")
    frame = frame.dropna(
        subset=["game_pk", "game_date", "scheduled_start", "home_team", "away_team"]
    )
    return frame.sort_values(["scheduled_start", "game_pk"]).reset_index(drop=True)


def seed_games_from_training_data(
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    training_path: str | Path = DEFAULT_TRAINING_DATA_PATH,
) -> int:
    schedule = load_training_schedule(training_path)
    init_db(db_path)
    settings = _load_settings_yaml()
    stadiums = settings["stadiums"]
    rows = []
    for game in schedule.to_dict(orient="records"):
        home_team = str(game["home_team"])
        stadium = stadiums.get(home_team, {})
        rows.append(
            (
                int(game["game_pk"]),
                pd.Timestamp(game["scheduled_start"]).isoformat(),
                home_team,
                str(game["away_team"]),
                None,
                None,
                str(game.get("venue") or stadium.get("park_name") or home_team),
                int(bool(stadium.get("is_dome", False))),
                int(bool(game.get("abs_active", True))),
                _coerce_nullable_int(game.get("f5_home_score")),
                _coerce_nullable_int(game.get("f5_away_score")),
                _coerce_nullable_int(game.get("final_home_score")),
                _coerce_nullable_int(game.get("final_away_score")),
                _normalize_game_status(str(game.get("status") or "final")),
            )
        )

    with sqlite_connection(init_db(db_path), builder_optimized=True) as connection:
        connection.executemany(
            """
            INSERT OR REPLACE INTO games (
                game_pk,
                date,
                home_team,
                away_team,
                home_starter_id,
                away_starter_id,
                venue,
                is_dome,
                is_abs_active,
                f5_home_score,
                f5_away_score,
                final_home_score,
                final_away_score,
                status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        connection.commit()
    return len(rows)


def fetch_sbr_f5_moneyline_raw(
    *,
    dates: Sequence[str],
    concurrency: int = 8,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    results: list[DateFetchResult] = []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        futures = {
            executor.submit(_fetch_sbr_date, game_date=game_date): game_date for game_date in dates
        }
        for future in as_completed(futures):
            results.append(future.result())

    raw_games: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for result in results:
        if result.error:
            errors.append({"game_date": result.game_date, "error": result.error})
            continue
        raw_games.extend(result.raw_games)
    return raw_games, errors


def normalize_sbr_f5_moneyline(
    raw_games: Sequence[dict[str, Any]],
    *,
    training_path: str | Path = DEFAULT_TRAINING_DATA_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    schedule = load_training_schedule(training_path)
    schedule_lookup = _build_schedule_lookup(schedule)
    code_to_full_name = _build_code_to_full_name()
    team_alias_to_code = _build_team_alias_to_code(code_to_full_name)

    rows: list[dict[str, Any]] = []
    unmatched_games: list[dict[str, Any]] = []
    for raw_game in raw_games:
        game_view = raw_game.get("gameView", {})
        home_name = str(game_view.get("homeTeam", {}).get("fullName") or "").strip()
        away_name = str(game_view.get("awayTeam", {}).get("fullName") or "").strip()
        home_code = team_alias_to_code.get(_normalize_name(home_name))
        away_code = team_alias_to_code.get(_normalize_name(away_name))
        start_time = pd.to_datetime(game_view.get("startDate"), utc=True, errors="coerce")
        if pd.isna(start_time):
            unmatched_games.append(
                {
                    "home_team": home_name,
                    "away_team": away_name,
                    "start_time": str(game_view.get("startDate") or ""),
                    "reason": "invalid_start_time",
                }
            )
            continue
        if home_code is None or away_code is None:
            unmatched_games.append(
                {
                    "home_team": home_name,
                    "away_team": away_name,
                    "start_time": start_time.isoformat(),
                    "reason": "unmapped_team",
                }
            )
            continue

        schedule_row = _match_schedule_row(
            schedule_lookup,
            game_date=str(start_time.date()),
            home_team=home_code,
            away_team=away_code,
            start_time=start_time,
        )
        if schedule_row is None:
            unmatched_games.append(
                {
                    "home_team": home_name,
                    "away_team": away_name,
                    "start_time": start_time.isoformat(),
                    "reason": "schedule_miss",
                }
            )
            continue

        canonical_home_name = code_to_full_name[home_code]
        canonical_away_name = code_to_full_name[away_code]
        game_pk = int(schedule_row["game_pk"])
        commence_time = pd.to_datetime(schedule_row["scheduled_start"], utc=True)
        for odds_view in raw_game.get("oddsViews", []):
            if not isinstance(odds_view, dict):
                continue
            sportsbook = _normalize_book_name(str(odds_view.get("sportsbook") or "unknown"))
            opening_line = odds_view.get("openingLine") or {}
            current_line = odds_view.get("currentLine") or {}
            for is_opening, line, fetched_at in (
                (True, opening_line, commence_time - pd.Timedelta(days=1)),
                (False, current_line, commence_time),
            ):
                home_odds = _coerce_nullable_int(line.get("homeOdds"))
                away_odds = _coerce_nullable_int(line.get("awayOdds"))
                if home_odds is None or away_odds is None:
                    continue
                if not _is_valid_american_odds(home_odds) or not _is_valid_american_odds(away_odds):
                    continue
                rows.append(
                    {
                        "game_pk": game_pk,
                        "commence_time": commence_time.isoformat(),
                        "game_date": str(commence_time.date()),
                        "home_team": canonical_home_name,
                        "away_team": canonical_away_name,
                        "market_type": F5_MARKET_TYPE,
                        "book_name": f"sbr:{sportsbook}",
                        "home_odds": home_odds,
                        "away_odds": away_odds,
                        "fetched_at": fetched_at.isoformat(),
                        "is_opening": is_opening,
                        "source_name": "sbr",
                        "source_event_id": game_view.get("gameId"),
                        "source_start_time": start_time.isoformat(),
                    }
                )

    normalized = pd.DataFrame(rows)
    if not normalized.empty:
        normalized = _dedupe_normalized_rows(normalized)
    unmatched = pd.DataFrame(unmatched_games)
    return normalized, unmatched


def write_outputs(
    *,
    raw_games: Sequence[dict[str, Any]],
    errors: Sequence[dict[str, str]],
    normalized: pd.DataFrame,
    unmatched: pd.DataFrame,
    raw_output_path: str | Path = DEFAULT_RAW_OUTPUT_PATH,
    normalized_output_path: str | Path = DEFAULT_NORMALIZED_OUTPUT_PATH,
    closing_output_path: str | Path = DEFAULT_CLOSING_OUTPUT_PATH,
) -> dict[str, Path]:
    raw_path = Path(raw_output_path)
    normalized_path = Path(normalized_output_path)
    closing_path = Path(closing_output_path)
    metadata_path = normalized_path.with_suffix(normalized_path.suffix + ".metadata.json")
    unmatched_path = normalized_path.with_name(normalized_path.stem + "__unmatched.csv")

    for path in (raw_path, normalized_path, closing_path, metadata_path, unmatched_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    raw_payload = {"games": list(raw_games), "errors": list(errors), "source_name": "sbr"}
    raw_path.write_text(json.dumps(raw_payload, indent=2), encoding="utf-8")
    normalized.to_parquet(normalized_path, index=False)
    closing_frame = normalized.loc[~normalized["is_opening"]].reset_index(drop=True)
    closing_frame.to_parquet(closing_path, index=False)
    unmatched.to_csv(unmatched_path, index=False)

    metadata = {
        "raw_games": len(raw_games),
        "errors": len(errors),
        "normalized_rows": int(len(normalized)),
        "closing_rows": int(len(closing_frame)),
        "opening_rows": int(len(normalized.loc[normalized["is_opening"]]))
        if not normalized.empty
        else 0,
        "unique_games": int(len(normalized["game_pk"].drop_duplicates()))
        if not normalized.empty
        else 0,
        "books": sorted(normalized["book_name"].dropna().unique().tolist())
        if not normalized.empty
        else [],
        "unmatched_games": int(len(unmatched)),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "raw": raw_path,
        "normalized": normalized_path,
        "closing": closing_path,
        "metadata": metadata_path,
        "unmatched": unmatched_path,
    }


def import_closing_lines(
    *,
    closing_output_path: str | Path = DEFAULT_CLOSING_OUTPUT_PATH,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    from src.clients.historical_odds_client import import_historical_odds

    return import_historical_odds(
        source_path=closing_output_path,
        db_path=db_path,
        default_market_type=F5_MARKET_TYPE,
        default_book_name="sbr",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Acquire historical MLB F5 moneyline odds from SBR"
    )
    parser.add_argument("--training-path", default=str(DEFAULT_TRAINING_DATA_PATH))
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--raw-output", default=str(DEFAULT_RAW_OUTPUT_PATH))
    parser.add_argument("--normalized-output", default=str(DEFAULT_NORMALIZED_OUTPUT_PATH))
    parser.add_argument("--closing-output", default=str(DEFAULT_CLOSING_OUTPUT_PATH))
    parser.add_argument("--seed-games", action="store_true")
    parser.add_argument("--import-closing", action="store_true")
    args = parser.parse_args(argv)

    schedule = load_training_schedule(args.training_path)
    if args.start_date:
        schedule = schedule.loc[schedule["game_date"] >= args.start_date].copy()
    if args.end_date:
        schedule = schedule.loc[schedule["game_date"] <= args.end_date].copy()
    dates = sorted(schedule["game_date"].astype(str).unique().tolist())

    raw_games, errors = fetch_sbr_f5_moneyline_raw(dates=dates, concurrency=args.concurrency)
    normalized, unmatched = normalize_sbr_f5_moneyline(raw_games, training_path=args.training_path)
    outputs = write_outputs(
        raw_games=raw_games,
        errors=errors,
        normalized=normalized,
        unmatched=unmatched,
        raw_output_path=args.raw_output,
        normalized_output_path=args.normalized_output,
        closing_output_path=args.closing_output,
    )

    seeded_games = 0
    if args.seed_games:
        seeded_games = seed_games_from_training_data(
            db_path=args.db_path, training_path=args.training_path
        )

    imported_rows = 0
    if args.import_closing:
        imported_rows = import_closing_lines(
            closing_output_path=outputs["closing"], db_path=args.db_path
        )

    print(
        json.dumps(
            {
                "dates_requested": len(dates),
                "raw_games": len(raw_games),
                "errors": len(errors),
                "normalized_rows": int(len(normalized)),
                "unique_games": (
                    int(len(normalized["game_pk"].drop_duplicates())) if not normalized.empty else 0
                ),
                "seeded_games": seeded_games,
                "imported_rows": imported_rows,
                "outputs": {key: str(value) for key, value in outputs.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _fetch_sbr_date(*, game_date: str) -> DateFetchResult:
    url = SBR_F5_ML_URL_TEMPLATE.format(game_date=game_date)
    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=SBR_TIMEOUT)
        if response.status_code == 404:
            return DateFetchResult(game_date=game_date, raw_games=[])
        response.raise_for_status()
        match = NEXT_DATA_PATTERN.search(response.text)
        if match is None:
            return DateFetchResult(game_date=game_date, raw_games=[], error="missing_next_data")
        payload = json.loads(match.group(1))
        page_props = payload.get("props", {}).get("pageProps", {})
        odds_tables = page_props.get("oddsTables", [])
        if not odds_tables:
            return DateFetchResult(game_date=game_date, raw_games=[])
        raw_games = odds_tables[0].get("oddsTableModel", {}).get("gameRows", [])
        return DateFetchResult(game_date=game_date, raw_games=raw_games)
    except Exception as exc:
        return DateFetchResult(game_date=game_date, raw_games=[], error=str(exc))


def _build_code_to_full_name() -> dict[str, str]:
    settings = _load_settings_yaml()
    return {team_code: str(team["full_name"]) for team_code, team in settings["teams"].items()}


def _build_team_alias_to_code(code_to_full_name: dict[str, str]) -> dict[str, str]:
    settings = _load_settings_yaml()
    alias_to_code: dict[str, str] = {}
    for team_code, team_payload in settings["teams"].items():
        full_name = str(team_payload["full_name"])
        city = str(team_payload["city"])
        nickname = str(team_payload["nickname"])
        for alias in {full_name, city, nickname, f"{city} {nickname}"}:
            alias_to_code[_normalize_name(alias)] = team_code

    alias_to_code[_normalize_name("Oakland Athletics")] = "OAK"
    alias_to_code[_normalize_name("Oakland A's")] = "OAK"
    alias_to_code[_normalize_name("Athletics")] = "OAK"
    alias_to_code[_normalize_name("Cleveland Indians")] = "CLE"
    alias_to_code[_normalize_name("Kansas City Royals")] = "KC"
    alias_to_code[_normalize_name("San Diego Padres")] = "SD"
    alias_to_code[_normalize_name("San Francisco Giants")] = "SF"
    alias_to_code[_normalize_name("St Louis Cardinals")] = "STL"
    alias_to_code[_normalize_name("Tampa Bay Rays")] = "TB"
    alias_to_code[_normalize_name("LA Dodgers")] = "LAD"
    alias_to_code[_normalize_name("LA Angels")] = "LAA"
    alias_to_code[_normalize_name("Chi Cubs")] = "CHC"
    alias_to_code[_normalize_name("Chi White Sox")] = "CWS"
    return alias_to_code


def _build_schedule_lookup(
    schedule: pd.DataFrame,
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    lookup: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for record in schedule.to_dict(orient="records"):
        key = (str(record["game_date"]), str(record["home_team"]), str(record["away_team"]))
        lookup.setdefault(key, []).append(record)
    for key, records in lookup.items():
        lookup[key] = sorted(records, key=lambda record: pd.Timestamp(record["scheduled_start"]))
    return lookup


def _match_schedule_row(
    schedule_lookup: dict[tuple[str, str, str], list[dict[str, Any]]],
    *,
    game_date: str,
    home_team: str,
    away_team: str,
    start_time: pd.Timestamp,
) -> dict[str, Any] | None:
    candidates = schedule_lookup.get((game_date, home_team, away_team))
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    return min(
        candidates,
        key=lambda candidate: abs(
            _timestamp_seconds(candidate["scheduled_start"]) - start_time.timestamp()
        ),
    )


def _normalize_book_name(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    return normalized.strip("_") or "unknown"


def _normalize_name(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()
    tokens = normalized.split()
    deduped_tokens: list[str] = []
    for token in tokens:
        if deduped_tokens and deduped_tokens[-1] == token:
            continue
        deduped_tokens.append(token)
    return " ".join(deduped_tokens)


def _coerce_nullable_int(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_valid_american_odds(value: int) -> bool:
    return value <= -100 or value >= 100


def _normalize_game_status(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"final", "finished", "closed"}:
        return "final"
    if normalized in {"scheduled", "pregame", "pre-game"}:
        return "scheduled"
    if normalized in {"postponed", "ppd"}:
        return "postponed"
    if normalized in {"suspended"}:
        return "suspended"
    if normalized in {"cancelled", "canceled"}:
        return "cancelled"
    return "final"


def _dedupe_normalized_rows(frame: pd.DataFrame) -> pd.DataFrame:
    deduped = frame.copy()
    deduped["source_start_time"] = pd.to_datetime(
        deduped["source_start_time"], utc=True, errors="coerce"
    )
    deduped["commence_time"] = pd.to_datetime(deduped["commence_time"], utc=True, errors="coerce")
    deduped["match_delta_seconds"] = (
        (deduped["source_start_time"] - deduped["commence_time"]).abs().dt.total_seconds()
    )
    deduped["match_delta_seconds"] = deduped["match_delta_seconds"].fillna(float("inf"))
    deduped = deduped.sort_values(
        ["game_pk", "book_name", "is_opening", "match_delta_seconds", "source_start_time"]
    )
    deduped = deduped.drop_duplicates(subset=["game_pk", "book_name", "is_opening"], keep="first")
    deduped = deduped.drop(columns=["match_delta_seconds"])
    deduped["source_start_time"] = deduped["source_start_time"].apply(
        lambda value: value.isoformat() if pd.notna(value) else None
    )
    deduped["commence_time"] = deduped["commence_time"].apply(
        lambda value: value.isoformat() if pd.notna(value) else None
    )
    return deduped.reset_index(drop=True)


def _timestamp_seconds(value: Any) -> float:
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        return 0.0
    return float(parsed.timestamp())


__all__ = [
    "DEFAULT_CLOSING_OUTPUT_PATH",
    "DEFAULT_HISTORICAL_ODDS_ROOT",
    "DEFAULT_NORMALIZED_OUTPUT_PATH",
    "DEFAULT_RAW_OUTPUT_PATH",
    "DEFAULT_TRAINING_DATA_PATH",
    "fetch_sbr_f5_moneyline_raw",
    "import_closing_lines",
    "load_training_schedule",
    "normalize_sbr_f5_moneyline",
    "seed_games_from_training_data",
    "write_outputs",
]


if __name__ == "__main__":
    raise SystemExit(main())
