from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

from src.models.odds import OddsSnapshot
from src.ops.error_handler import CircuitBreaker
from src.pipeline.daily import _default_lineups_fetcher, _default_odds_fetcher


def _sample_snapshot(game_pk: int = 1001) -> OddsSnapshot:
    return OddsSnapshot(
        game_pk=game_pk,
        book_name="TestBook",
        market_type="f5_ml",
        home_odds=-110,
        away_odds=100,
        fetched_at=datetime(2025, 9, 15, tzinfo=UTC),
    )


def test_default_odds_fetcher_retries_three_times_before_success(
    monkeypatch,
    tmp_path: Path,
) -> None:
    attempts = {"count": 0}
    sleeps: list[float] = []
    expected = [_sample_snapshot()]

    def _flaky_fetch(**_: object) -> list[OddsSnapshot]:
        attempts["count"] += 1
        if attempts["count"] < 4:
            raise RuntimeError("temporary odds outage")
        return expected

    monkeypatch.setattr("src.pipeline.daily.fetch_mlb_odds", _flaky_fetch)
    monkeypatch.setattr("src.pipeline.daily._ODDS_CIRCUIT", CircuitBreaker(name="odds"))
    monkeypatch.setattr("src.ops.error_handler.time.sleep", sleeps.append)

    result = _default_odds_fetcher(date(2025, 9, 15), "prod", tmp_path / "mlb.db")

    assert result == expected
    assert attempts["count"] == 4
    assert sleeps == [2.0, 4.0, 8.0]


def test_default_lineups_fetcher_gracefully_degrades_on_warning(monkeypatch) -> None:
    attempts = {"count": 0}
    sleeps: list[float] = []

    def _always_fail(target_date: str) -> list[object]:
        attempts["count"] += 1
        raise RuntimeError(f"lineups unavailable for {target_date}")

    monkeypatch.setattr("src.pipeline.daily.fetch_confirmed_lineups", _always_fail)
    monkeypatch.setattr("src.pipeline.daily._LINEUPS_CIRCUIT", CircuitBreaker(name="lineups"))
    monkeypatch.setattr("src.ops.error_handler.time.sleep", sleeps.append)

    result = _default_lineups_fetcher("2025-09-15")

    assert result == []
    assert attempts["count"] == 4
    assert sleeps == [2.0, 4.0, 8.0]
