from __future__ import annotations

import logging

import pytest

from src.ops.error_handler import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    call_with_graceful_degradation,
    notify_fatal_error,
    retry,
)


class _RecordingNotifier:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def send_failure_alert(self, **payload: object) -> dict[str, object]:
        self.calls.append(dict(payload))
        return {"type": "failure_alert", **payload}


def test_retry_retries_three_times_with_exponential_backoff_then_succeeds(monkeypatch) -> None:
    sleeps: list[float] = []
    attempts = {"count": 0}

    monkeypatch.setattr("src.ops.error_handler.time.sleep", sleeps.append)

    @retry()
    def _flaky_operation() -> str:
        attempts["count"] += 1
        if attempts["count"] < 4:
            raise RuntimeError("temporary failure")
        return "ok"

    assert _flaky_operation() == "ok"
    assert attempts["count"] == 4
    assert sleeps == [2.0, 4.0, 8.0]


def test_circuit_breaker_opens_after_five_consecutive_failures() -> None:
    breaker = CircuitBreaker(name="odds", failure_threshold=5)
    attempts = {"count": 0}

    def _always_fail() -> None:
        attempts["count"] += 1
        raise RuntimeError("boom")

    for _ in range(4):
        with pytest.raises(RuntimeError, match="boom"):
            breaker.call(_always_fail)

    with pytest.raises(CircuitBreakerOpenError, match="odds"):
        breaker.call(_always_fail)

    assert attempts["count"] == 5


def test_retry_composed_around_circuit_breaker_stops_after_five_raw_failures(monkeypatch) -> None:
    sleeps: list[float] = []
    breaker = CircuitBreaker(name="odds", failure_threshold=5)
    attempts = {"count": 0}

    monkeypatch.setattr("src.ops.error_handler.time.sleep", sleeps.append)

    def _always_fail() -> None:
        attempts["count"] += 1
        raise RuntimeError("boom")

    @retry()
    def _wrapped() -> None:
        breaker.call(_always_fail)

    with pytest.raises(RuntimeError, match="boom"):
        _wrapped()

    with pytest.raises(CircuitBreakerOpenError, match="odds"):
        _wrapped()

    assert attempts["count"] == 5
    assert sleeps == [2.0, 4.0, 8.0]


def test_call_with_graceful_degradation_returns_fallback_on_warning(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        result = call_with_graceful_degradation(
            lambda: (_ for _ in ()).throw(RuntimeError("missing odds")),
            operation_name="odds fetch",
            fallback=[],
        )

    assert result == []
    assert "odds fetch" in caplog.text


def test_notify_fatal_error_sends_discord_alert() -> None:
    notifier = _RecordingNotifier()

    payload = notify_fatal_error(
        pipeline_date="2025-09-15",
        error=RuntimeError("model artifact missing"),
        notifier=notifier,
        dry_run=True,
    )

    assert payload == {
        "type": "failure_alert",
        "pipeline_date": "2025-09-15",
        "error_message": "model artifact missing",
        "dry_run": True,
    }
    assert notifier.calls == [
        {
            "pipeline_date": "2025-09-15",
            "error_message": "model artifact missing",
            "dry_run": True,
        }
    ]
