"""Tool interfaces for the lightweight agent runtime."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ToolResult:
    ok: bool
    content: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    error: str = ""


class BaseTool:
    name = ""
    description = ""
    input_schema: Dict[str, Any] = {"type": "object", "properties": {}}
    speakable_start = ""
    context_label = ""
    keywords = ()

    def run(self, **kwargs) -> ToolResult:
        raise NotImplementedError

    def plan(self, user_text: str):
        """Optionally build a ToolStep from user text.

        Most built-in tools are planned by AgentPlanner directly. Extension
        tools can override this hook to make themselves auto-runnable without
        editing the central planner.
        """
        return None

    def plan_from_skill(self, user_text: str, skill=None):
        """Optionally build a ToolStep when a matched skill names this tool."""
        return self.plan(user_text)

    def to_mcp_tool(self) -> Dict[str, Any]:
        """Return an MCP-style tool definition for prompts and clients."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": copy.deepcopy(getattr(self, "input_schema", {}) or {"type": "object", "properties": {}}),
        }


@dataclass
class RuntimeToolContext:
    project_root: str
    memory_module: Optional[Any] = None
    long_term_memory: bool = False
    language: str = "zh"
    tool_authorization: Optional[Any] = None


class ToolProvider:
    """Builds tools for one runtime turn."""

    def build_tools(self, context: RuntimeToolContext) -> Dict[str, BaseTool]:
        raise NotImplementedError
