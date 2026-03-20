from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, ParamSpec, Protocol, TypeVar

from src.notifications.discord import send_failure_alert


logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


class FailureAlertNotifier(Protocol):
    def send_failure_alert(self, **payload: Any) -> dict[str, Any]: ...


class CircuitBreakerOpenError(RuntimeError):
    """Raised when a circuit breaker has opened and refuses new calls."""


class _DefaultFailureNotifier:
    def send_failure_alert(self, **payload: Any) -> dict[str, Any]:
        return send_failure_alert(**payload)


@dataclass(slots=True)
class CircuitBreaker:
    """Track consecutive failures and open after a configurable threshold."""

    name: str
    failure_threshold: int = 5
    consecutive_failures: int = 0
    is_open: bool = False

    def before_call(self) -> None:
        if self.is_open:
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self.name}' is open after "
                f"{self.consecutive_failures} consecutive failures"
            )

    def record_success(self) -> None:
        self.consecutive_failures = 0
        self.is_open = False

    def record_failure(self) -> None:
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.failure_threshold:
            self.is_open = True
            logger.error(
                "Circuit breaker '%s' opened after %s consecutive failures",
                self.name,
                self.consecutive_failures,
            )

    def call(self, operation: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        self.before_call()
        try:
            result = operation(*args, **kwargs)
        except Exception as exc:
            self.record_failure()
            if self.is_open:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is open after "
                    f"{self.consecutive_failures} consecutive failures"
                ) from exc
            raise

        self.record_success()
        return result

    def reset(self) -> None:
        self.consecutive_failures = 0
        self.is_open = False


def retry(
    *,
    max_retries: int = 3,
    initial_delay: float = 2.0,
    backoff_factor: float = 2.0,
    retry_exceptions: tuple[type[Exception], ...] = (Exception,),
    logger_: logging.Logger | None = None,
    sleep_func: Callable[[float], None] | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Retry a callable with exponential backoff delays of 2/4/8 by default."""

    def decorator(operation: Callable[P, T]) -> Callable[P, T]:
        @wraps(operation)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            active_logger = logger_ or logging.getLogger(operation.__module__)
            resolved_sleep = sleep_func or time.sleep
            delay = initial_delay

            for attempt in range(max_retries + 1):
                try:
                    return operation(*args, **kwargs)
                except retry_exceptions as exc:
                    if isinstance(exc, CircuitBreakerOpenError):
                        raise
                    if attempt >= max_retries:
                        active_logger.error(
                            "%s failed after %s total attempts",
                            operation.__name__,
                            max_retries + 1,
                            exc_info=True,
                        )
                        raise

                    active_logger.warning(
                        "%s failed on attempt %s/%s; retrying in %.0f seconds",
                        operation.__name__,
                        attempt + 1,
                        max_retries + 1,
                        delay,
                        exc_info=True,
                    )
                    resolved_sleep(delay)
                    delay *= backoff_factor

            raise RuntimeError("retry wrapper exhausted unexpectedly")

        return wrapper

    return decorator


def call_with_graceful_degradation(
    operation: Callable[[], T],
    *,
    operation_name: str,
    fallback: T | Callable[[Exception], T],
    logger_: logging.Logger | None = None,
) -> T:
    """Run an operation and fall back with a warning when it fails."""

    active_logger = logger_ or logger
    try:
        return operation()
    except Exception as exc:
        active_logger.warning(
            "%s failed; continuing with graceful degradation",
            operation_name,
            exc_info=True,
        )
        return fallback(exc) if callable(fallback) else fallback


def notify_fatal_error(
    *,
    pipeline_date: str,
    error: Exception | str,
    notifier: FailureAlertNotifier | None = None,
    dry_run: bool = False,
    logger_: logging.Logger | None = None,
) -> dict[str, Any]:
    """Log and send a Discord failure alert for a fatal pipeline error."""

    error_message = str(error)
    active_logger = logger_ or logger
    if isinstance(error, BaseException):
        active_logger.error(
            "Fatal pipeline error for %s: %s",
            pipeline_date,
            error_message,
            exc_info=(type(error), error, error.__traceback__),
        )
    else:
        active_logger.error("Fatal pipeline error for %s: %s", pipeline_date, error_message)

    resolved_notifier = notifier or _DefaultFailureNotifier()
    return resolved_notifier.send_failure_alert(
        pipeline_date=pipeline_date,
        error_message=error_message,
        dry_run=dry_run,
    )


__all__ = [
    "CircuitBreaker",
    "CircuitBreakerOpenError",
    "FailureAlertNotifier",
    "call_with_graceful_degradation",
    "notify_fatal_error",
    "retry",
]
