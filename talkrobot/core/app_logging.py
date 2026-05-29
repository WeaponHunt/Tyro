"""Application logging helpers."""
from __future__ import annotations

import os
import sys
import threading
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import TextIO

try:
    from loguru import logger
except Exception:
    logger = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = PROJECT_ROOT / "talkrobot" / "logs"


class DailyFileSink:
    """A loguru sink that writes to one file per local calendar day."""

    def __init__(
        self,
        log_dir: str | os.PathLike[str] | None = None,
        prefix: str = "robot",
        suffix: str = ".log",
        retention_days: int = 30,
    ) -> None:
        self.log_dir = Path(log_dir or DEFAULT_LOG_DIR)
        self.prefix = prefix
        self.suffix = suffix
        self.retention_days = max(1, int(retention_days))
        self._lock = threading.Lock()
        self._current_day = ""
        self._file: TextIO | None = None

    def write(self, message) -> None:
        day = datetime.now().strftime("%Y-%m-%d")
        with self._lock:
            if day != self._current_day:
                self._open_for_day(day)
            if self._file is not None:
                self._file.write(str(message))
                self._file.flush()

    def stop(self) -> None:
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None

    def _open_for_day(self, day: str) -> None:
        if self._file is not None:
            self._file.close()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._current_day = day
        self._file = (self.log_dir / f"{self.prefix}_{day}{self.suffix}").open(
            "a",
            encoding="utf-8",
        )
        self._cleanup_old_files()

    def _cleanup_old_files(self) -> None:
        cutoff = datetime.now().date() - timedelta(days=self.retention_days)
        pattern = f"{self.prefix}_*{self.suffix}"
        for path in self.log_dir.glob(pattern):
            day_text = path.name[len(self.prefix) + 1 : -len(self.suffix)]
            try:
                file_day = datetime.strptime(day_text, "%Y-%m-%d").date()
            except ValueError:
                continue
            if file_day < cutoff:
                try:
                    path.unlink()
                except OSError:
                    pass


def configure_logging(debug: bool = False, log_dir: str | os.PathLike[str] | None = None) -> None:
    """Configure console logging plus daily file logging."""
    level = "DEBUG" if debug else "INFO"
    if logger is None:
        logging.basicConfig(level=getattr(logging, level))
        return

    retention_days = _env_int("TALKROBOT_LOG_RETENTION_DAYS", 30)
    selected_log_dir = log_dir or os.getenv("TALKROBOT_LOG_DIR") or DEFAULT_LOG_DIR

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:8}</level> | <level>{message}</level>",
        level=level,
    )
    logger.add(
        DailyFileSink(selected_log_dir, retention_days=retention_days),
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:8} | {name}:{function}:{line} | {message}",
        level=level,
        enqueue=True,
        backtrace=debug,
        diagnose=debug,
    )


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default
