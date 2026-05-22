"""Agent event primitives."""
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class AgentEvent:
    """A small event emitted by AgentRuntime during one user turn."""

    type: str
    text: str = ""
    speakable: bool = False
    data: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def status(cls, text: str, speakable: bool = True, **data) -> "AgentEvent":
        return cls(type="status", text=text, speakable=speakable, data=data)

    @classmethod
    def error(cls, text: str, stage: str, **data) -> "AgentEvent":
        payload = {"stage": stage, **data}
        return cls(type="error", text=text, speakable=True, data=payload)
