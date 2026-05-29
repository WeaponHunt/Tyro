"""Small local JSON memory backend for tests and offline development."""
from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime
from typing import Dict, List

from talkrobot.modules.memory.base import MemoryBackend, MemoryRecord


class SimpleJsonMemoryBackend(MemoryBackend):
    """A dependency-free persistent memory backend.

    It stores records in JSON and retrieves them by lightweight token overlap.
    This is intentionally simple, but useful as a local open-source reference
    implementation and as a stable test backend.
    """

    def __init__(self, db_path: str, user_id: str):
        self.db_path = os.path.abspath(db_path)
        self.user_id = user_id
        self.file_path = os.path.join(self.db_path, f"{self._safe_user_id(user_id)}.json")
        self._lock = threading.Lock()
        os.makedirs(self.db_path, exist_ok=True)
        self._records = self._load()

    def add(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return

        with self._lock:
            self._records.append(
                {
                    "memory": text,
                    "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                }
            )
            self._save()

    def search(self, query: str, limit: int = 3) -> List[MemoryRecord]:
        query_tokens = self._tokens(query)
        with self._lock:
            scored = []
            for index, item in enumerate(self._records):
                text = str(item.get("memory") or "")
                memory_tokens = self._tokens(text)
                overlap = len(query_tokens.intersection(memory_tokens))
                score = overlap / max(1, len(query_tokens))
                if overlap > 0:
                    scored.append((score, index, item))

            scored.sort(key=lambda row: (-row[0], -row[1]))
            selected = scored[: max(1, int(limit))]

        return [
            MemoryRecord(
                text=str(item.get("memory") or ""),
                score=score,
                metadata={key: value for key, value in item.items() if key != "memory"},
            )
            for score, _, item in selected
        ]

    def get_all(self) -> List[MemoryRecord]:
        with self._lock:
            return [
                MemoryRecord(
                    text=str(item.get("memory") or ""),
                    metadata={key: value for key, value in item.items() if key != "memory"},
                )
                for item in self._records
                if item.get("memory")
            ]

    def _load(self) -> List[Dict[str, str]]:
        if not os.path.isfile(self.file_path):
            return []
        try:
            with open(self.file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            if isinstance(data, list):
                return [item for item in data if isinstance(item, dict)]
        except Exception:
            return []
        return []

    def _save(self) -> None:
        with open(self.file_path, "w", encoding="utf-8") as file:
            json.dump(self._records, file, ensure_ascii=False, indent=2)

    @staticmethod
    def _tokens(text: str) -> set[str]:
        text = (text or "").casefold()
        words = set(re.findall(r"[a-z0-9_]+", text))
        cjk = set(re.findall(r"[\u4e00-\u9fff]", text))
        return words.union(cjk)

    @staticmethod
    def _safe_user_id(user_id: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", user_id or "default")
