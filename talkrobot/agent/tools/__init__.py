"""Agent tool interfaces, registry, and built-in tools."""

from talkrobot.agent.tools.base import BaseTool, RuntimeToolContext, ToolProvider, ToolResult
from talkrobot.agent.tools.builtin import (
    CalculatorTool,
    CurrentTimeTool,
    FileEditTool,
    MemoryWriteTool,
    MemorySearchTool,
    ProjectFileListTool,
    ProjectFileReadTool,
    ProjectFileSearchTool,
    RepoBootstrapTool,
    ShellCommandTool,
    WebFetchTool,
)
from talkrobot.agent.tools.registry import BuiltinToolProvider, ToolRegistry

__all__ = [
    "BaseTool",
    "RuntimeToolContext",
    "ToolProvider",
    "ToolResult",
    "BuiltinToolProvider",
    "ToolRegistry",
    "MCPToolProvider",
    "mcp_stdio_server",
    "mcp_tool",
    "CalculatorTool",
    "CurrentTimeTool",
    "FileEditTool",
    "MemoryWriteTool",
    "MemorySearchTool",
    "ProjectFileListTool",
    "ProjectFileReadTool",
    "ProjectFileSearchTool",
    "RepoBootstrapTool",
    "ShellCommandTool",
    "WebFetchTool",
]


def __getattr__(name):
    if name in {"MCPToolProvider", "mcp_stdio_server", "mcp_tool"}:
        from talkrobot.agent.tools import mcp

        return getattr(mcp, name)
    raise AttributeError(f"module 'talkrobot.agent.tools' has no attribute '{name}'")
