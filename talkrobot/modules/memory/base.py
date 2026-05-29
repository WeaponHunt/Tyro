"""Common memory backend interfaces."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MemoryRecord:
    text: str
    score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"memory": self.text}
        if self.score is not None:
            data["score"] = self.score
        if self.metadata:
            data.update(self.metadata)
        return data


class MemoryBackend(ABC):
    """Backend contract for long-term memory implementations."""

    @abstractmethod
    def add(self, text: str) -> None:
        """Persist one memory item."""

    @abstractmethod
    def search(self, query: str, limit: int = 3) -> List[MemoryRecord]:
        """Return memory records relevant to query."""

    @abstractmethod
    def get_all(self) -> List[MemoryRecord]:
        """Return all memory records for the current user."""

    def shutdown(self) -> None:
        """Release resources held by the backend."""
