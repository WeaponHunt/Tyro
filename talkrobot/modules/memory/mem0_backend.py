"""Mem0-backed memory implementation."""
from __future__ import annotations

import os
import threading
from typing import List

try:
    from mem0 import Memory
except Exception:
    Memory = None

from talkrobot.modules.memory.base import MemoryBackend, MemoryRecord


class Mem0MemoryBackend(MemoryBackend):
    """Adapter around the open-source mem0 Memory API."""

    def __init__(self, config: dict, user_id: str):
        db_path = config.get("vector_store", {}).get("config", {}).get("path")
        if db_path:
            os.makedirs(db_path, exist_ok=True)

        if Memory is None:
            raise ImportError("mem0 is required to initialize Mem0MemoryBackend")

        self.memory = Memory.from_config(config)
        self.user_id = user_id
        self.db_path = db_path
        self._lock = threading.Lock()

    def add(self, text: str) -> None:
        with self._lock:
            self.memory.add(text, user_id=self.user_id)

    def search(self, query: str, limit: int = 3) -> List[MemoryRecord]:
        with self._lock:
            raw_results = self.memory.search(query, user_id=self.user_id, limit=limit)
        return normalize_memory_records(raw_results)

    def get_all(self) -> List[MemoryRecord]:
        with self._lock:
            raw_results = self.memory.get_all(user_id=self.user_id)
        return normalize_memory_records(raw_results)


def normalize_memory_records(raw_results) -> List[MemoryRecord]:
    if isinstance(raw_results, dict) and "results" in raw_results:
        raw_results = raw_results["results"]
    if not raw_results:
        return []

    records: List[MemoryRecord] = []
    for item in raw_results:
        text = ""
        score = None
        metadata = {}

        if isinstance(item, str):
            text = item
        elif isinstance(item, dict):
            text = item.get("memory") or item.get("text") or item.get("content") or item.get("value") or ""
            score = item.get("score")
            metadata = {key: value for key, value in item.items() if key not in {"memory", "text", "content", "value"}}
        else:
            text = (
                getattr(item, "memory", None)
                or getattr(item, "text", None)
                or getattr(item, "content", None)
                or ""
            )
            score = getattr(item, "score", None)

        text = str(text).strip()
        if text:
            records.append(MemoryRecord(text=text, score=score, metadata=metadata))
    return records
