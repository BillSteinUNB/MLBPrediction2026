from __future__ import annotations

import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from src.ops.logging_config import configure_logging


def test_configure_logging_uses_daily_rotation_and_thirty_day_retention(tmp_path: Path) -> None:
    log_path = configure_logging(log_dir=tmp_path, log_name="pipeline-test")
    logger = logging.getLogger("src.pipeline.daily")
    logger.info("logging configured")

    handlers = [
        handler
        for handler in logging.getLogger().handlers
        if isinstance(handler, TimedRotatingFileHandler)
    ]

    assert log_path == tmp_path / "pipeline-test.log"
    assert len(handlers) == 1
    assert Path(handlers[0].baseFilename) == log_path
    assert handlers[0].when == "MIDNIGHT"
    assert handlers[0].backupCount == 30

    handlers[0].flush()
    assert log_path.exists()
    assert "logging configured" in log_path.read_text(encoding="utf-8")
