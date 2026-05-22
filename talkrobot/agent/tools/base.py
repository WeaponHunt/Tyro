"""Tool interfaces for the lightweight agent runtime."""
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ToolResult:
    ok: bool
    content: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    error: str = ""


class BaseTool:
    name = ""
    description = ""
    speakable_start = ""

    def run(self, **kwargs) -> ToolResult:
        raise NotImplementedError
