"""Structured per-turn interaction logging."""
from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

try:
    from loguru import logger
except Exception:
    logger = logging.getLogger(__name__)

from talkrobot.core.app_logging import DEFAULT_LOG_DIR


_LOCK = threading.Lock()


def log_interaction(record: Dict[str, Any]) -> None:
    """Append one interaction record to the current day's JSONL file."""
    if _env_bool("TALKROBOT_INTERACTION_LOG_ENABLED", True) is False:
        return

    now = datetime.now().astimezone()
    payload = {
        "interaction_id": str(uuid.uuid4()),
        "timestamp": now.isoformat(timespec="milliseconds"),
        **record,
    }

    try:
        log_dir = Path(os.getenv("TALKROBOT_INTERACTION_LOG_DIR") or DEFAULT_LOG_DIR / "interactions")
        log_dir.mkdir(parents=True, exist_ok=True)
        _cleanup_old_files(log_dir)
        path = log_dir / f"interactions_{now.strftime('%Y-%m-%d')}.jsonl"
        line = json.dumps(payload, ensure_ascii=False, default=str)
        with _LOCK:
            with path.open("a", encoding="utf-8") as file:
                file.write(line + "\n")
    except Exception as exc:
        logger.warning(f"写入交互日志失败: {exc}")


def _cleanup_old_files(log_dir: Path) -> None:
    retention_days = _env_int("TALKROBOT_LOG_RETENTION_DAYS", 30)
    cutoff = datetime.now().date() - timedelta(days=max(1, retention_days))
    for path in log_dir.glob("interactions_*.jsonl"):
        day_text = path.name[len("interactions_") : -len(".jsonl")]
        try:
            file_day = datetime.strptime(day_text, "%Y-%m-%d").date()
        except ValueError:
            continue
        if file_day < cutoff:
            try:
                path.unlink()
            except OSError:
                pass


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default
