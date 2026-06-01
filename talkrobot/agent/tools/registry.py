"""Tool registry and providers for AgentRuntime."""
from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional

try:
    from loguru import logger
except Exception:
    import logging

    logger = logging.getLogger(__name__)

from talkrobot.agent.tools.base import BaseTool, RuntimeToolContext, ToolProvider
from talkrobot.agent.tools.builtin import (
    CalculatorTool,
    CurrentTimeTool,
    FileEditTool,
    MemorySearchTool,
    MemoryWriteTool,
    ProjectFileListTool,
    ProjectFileReadTool,
    ProjectFileSearchTool,
    RepoBootstrapTool,
    ShellCommandTool,
    WebFetchTool,
)


class BuiltinToolProvider(ToolProvider):
    """Builds the built-in low-risk tools."""

    def build_tools(self, context: RuntimeToolContext) -> Dict[str, BaseTool]:
        return {
            "memory_search": MemorySearchTool(
                context.memory_module,
                enabled=context.long_term_memory,
            ),
            "memory_write": MemoryWriteTool(
                context.memory_module,
                enabled=context.long_term_memory,
            ),
            "current_time": CurrentTimeTool(),
            "calculator": CalculatorTool(),
            "project_file_search": ProjectFileSearchTool(context.project_root),
            "project_file_list": ProjectFileListTool(context.project_root),
            "project_file_read": ProjectFileReadTool(context.project_root),
            "repo_bootstrap": RepoBootstrapTool(context.project_root),
            "shell_command": ShellCommandTool(context.project_root),
            "file_edit": FileEditTool(context.project_root),
            "web_fetch": WebFetchTool(),
        }


class ToolRegistry:
    """Combines tool providers into the tool map used by the agent."""

    def __init__(self, providers: Optional[Iterable[ToolProvider]] = None):
        self.providers: List[ToolProvider] = list(providers or [])

    @classmethod
    def with_mcp_servers(
        cls,
        servers: Iterable[object],
        include_builtin: bool = True,
    ) -> "ToolRegistry":
        providers: List[ToolProvider] = []
        if include_builtin:
            providers.append(BuiltinToolProvider())

        from talkrobot.agent.tools.mcp import MCPToolProvider

        providers.append(MCPToolProvider.from_servers(servers))
        return cls(providers)

    @classmethod
    def default(cls, project_root: str) -> "ToolRegistry":
        providers: List[ToolProvider] = [BuiltinToolProvider()]
        mcp_configs = _mcp_config_paths(project_root)
        if mcp_configs:
            try:
                from talkrobot.agent.tools.mcp import MCPToolProvider

                providers.append(MCPToolProvider.from_files(mcp_configs))
            except Exception as exc:
                logger.warning(f"加载 MCP 工具配置失败: {exc}")
        return cls(providers)

    def build_tools(self, context: RuntimeToolContext) -> Dict[str, BaseTool]:
        tools: Dict[str, BaseTool] = {}
        for provider in self.providers:
            try:
                provided = provider.build_tools(context)
            except Exception as exc:
                logger.warning(f"工具 provider 加载失败: {provider.__class__.__name__}: {exc}")
                continue

            for name, tool in provided.items():
                tool_name = (getattr(tool, "name", "") or name or "").strip()
                if not tool_name:
                    logger.warning(f"跳过未命名工具: {tool!r}")
                    continue
                if tool_name in tools:
                    logger.warning(f"跳过重复工具: {tool_name}")
                    continue
                tools[tool_name] = tool
        return tools


def _mcp_config_paths(project_root: str) -> List[str]:
    paths = []
    default_path = os.path.join(project_root, "talkrobot", "agent", "mcp_servers.json")
    primary = os.getenv("TALKROBOT_MCP_CONFIG", default_path)
    if primary:
        paths.append(primary)

    extra = os.getenv("TALKROBOT_MCP_CONFIGS", "")
    for path in extra.split(os.pathsep):
        path = path.strip()
        if path:
            paths.append(path)

    resolved = []
    seen = set()
    for path in paths:
        expanded = os.path.abspath(os.path.expanduser(os.path.expandvars(path)))
        if expanded in seen or not os.path.isfile(expanded):
            continue
        seen.add(expanded)
        resolved.append(expanded)
    return resolved
