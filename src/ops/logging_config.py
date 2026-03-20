from __future__ import annotations

import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


_HANDLER_SENTINEL = "_mlbprediction2026_logging_handler"


def _mark_handler(handler: logging.Handler) -> logging.Handler:
    setattr(handler, _HANDLER_SENTINEL, True)
    return handler


def _remove_existing_project_handlers(root_logger: logging.Logger) -> None:
    for handler in list(root_logger.handlers):
        if getattr(handler, _HANDLER_SENTINEL, False):
            root_logger.removeHandler(handler)
            handler.close()


def configure_logging(
    *,
    log_dir: str | Path = Path("data") / "logs",
    log_name: str = "pipeline",
    level: int = logging.INFO,
) -> Path:
    """Configure console logging plus daily rotating file logs with 30-day retention."""

    resolved_log_dir = Path(log_dir)
    resolved_log_dir.mkdir(parents=True, exist_ok=True)
    log_path = resolved_log_dir / f"{log_name}.log"

    root_logger = logging.getLogger()
    _remove_existing_project_handlers(root_logger)
    root_logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    console_handler = _mark_handler(logging.StreamHandler())
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    file_handler = _mark_handler(
        TimedRotatingFileHandler(
            log_path,
            when="midnight",
            interval=1,
            backupCount=30,
            encoding="utf-8",
            utc=True,
        )
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    file_handler.suffix = "%Y-%m-%d"

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    return log_path


__all__ = ["configure_logging"]
