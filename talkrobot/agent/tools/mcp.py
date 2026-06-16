"""Optional Model Context Protocol tool adapter."""
from __future__ import annotations

import asyncio
import json
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

try:
    from loguru import logger
except Exception:
    import logging

    logger = logging.getLogger(__name__)

from talkrobot.agent.planner import ToolStep
from talkrobot.agent.tools.base import BaseTool, RuntimeToolContext, ToolProvider, ToolResult


def _run_async(factory):
    """Run a coroutine factory from sync code, even if a loop is active."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(factory())

    result: Dict[str, Any] = {}

    def worker() -> None:
        try:
            result["value"] = asyncio.run(factory())
        except Exception as exc:
            result["error"] = exc

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]
    return result.get("value")


def _build_stdio_params(server, StdioServerParameters):
    env = os.environ.copy()
    env.update(server.env or {})
    params_kwargs = {
        "command": server.command,
        "args": server.args,
        "env": env,
    }
    if server.cwd:
        params_kwargs["cwd"] = server.cwd
    try:
        return StdioServerParameters(**params_kwargs)
    except TypeError:
        params_kwargs.pop("cwd", None)
        return StdioServerParameters(**params_kwargs)


@dataclass
class MCPServerConfig:
    name: str
    command: str
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    cwd: str = ""
    enabled: bool = True
    discover: bool = False
    tools: List[Dict[str, Any]] = field(default_factory=list)


def mcp_tool(
    name: str,
    *,
    alias: str = "",
    description: str = "",
    input_schema: Optional[Dict[str, Any]] = None,
    triggers: Optional[Iterable[str]] = None,
    argument_template: Optional[Dict[str, Any]] = None,
    skill_argument_template: Optional[Dict[str, Any]] = None,
    context_label: str = "",
    speakable_start: str = "",
) -> Dict[str, Any]:
    """Create one MCP tool config entry.

    This is a convenience API for wiring open-source MCP servers from Python
    without hand-writing JSON.
    """
    data: Dict[str, Any] = {"name": name}
    if alias:
        data["alias"] = alias
    if description:
        data["description"] = description
    if input_schema is not None:
        data["input_schema"] = input_schema
    if triggers:
        data["triggers"] = list(triggers)
    if argument_template is not None:
        data["argument_template"] = argument_template
    if skill_argument_template is not None:
        data["skill_argument_template"] = skill_argument_template
    if context_label:
        data["context_label"] = context_label
    if speakable_start:
        data["speakable_start"] = speakable_start
    return data


def mcp_stdio_server(
    name: str,
    command: str,
    args: Optional[Iterable[str]] = None,
    *,
    env: Optional[Dict[str, str]] = None,
    cwd: str = "",
    enabled: bool = True,
    discover: bool = False,
    tools: Optional[Iterable[Dict[str, Any]]] = None,
) -> MCPServerConfig:
    """Create an stdio MCP server config for ToolRegistry/MCPToolProvider."""
    return MCPServerConfig(
        name=name,
        command=command,
        args=[str(arg) for arg in (args or [])],
        env={str(key): str(value) for key, value in (env or {}).items()},
        cwd=cwd,
        enabled=enabled,
        discover=discover,
        tools=list(tools or []),
    )


def _expand(value: str) -> str:
    return os.path.expandvars(os.path.expanduser(str(value or "")))


class MCPTool(BaseTool):
    """A sync BaseTool wrapper around one MCP server tool."""

    speakable_start = "我调用一个外部工具。"
    context_label = "外部工具结果"

    def __init__(
        self,
        server: MCPServerConfig,
        remote_name: str,
        alias: str = "",
        description: str = "",
        input_schema: Optional[Dict[str, Any]] = None,
        triggers: Optional[Iterable[str]] = None,
        argument_template: Optional[Dict[str, Any]] = None,
        skill_argument_template: Optional[Dict[str, Any]] = None,
        speakable_start: str = "",
        context_label: str = "",
    ):
        self.server = server
        self.remote_name = remote_name
        self.name = alias or f"mcp_{server.name}_{remote_name}".replace("-", "_")
        self.description = description or f"MCP tool {server.name}.{remote_name}"
        self.input_schema = input_schema or {}
        self.keywords = tuple(triggers or ())
        self.argument_template = argument_template or {}
        self.skill_argument_template = skill_argument_template
        if speakable_start:
            self.speakable_start = speakable_start
        if context_label:
            self.context_label = context_label

    def plan(self, user_text: str):
        if not self._matches(user_text):
            return None

        args = self._render_arguments(user_text)
        if args is None:
            return None
        return ToolStep(self.name, args, f"matched MCP tool {self.server.name}.{self.remote_name}")

    def plan_from_skill(self, user_text: str, skill=None):
        if self.skill_argument_template is None:
            return None
        args = {
            str(key): self._render_value(value, user_text)
            for key, value in self.skill_argument_template.items()
        }
        skill_name = getattr(skill, "name", "")
        reason = f"matched skill {skill_name}" if skill_name else "matched skill"
        return ToolStep(self.name, args, reason)

    def run(self, **kwargs) -> ToolResult:
        try:
            result = _run_async(lambda: self._call_tool(kwargs))
            content = self._content_to_text(result)
            is_error = bool(getattr(result, "isError", False))
            return ToolResult(
                ok=not is_error,
                content=content,
                data={"server": self.server.name, "remote_tool": self.remote_name},
                error=content if is_error else "",
            )
        except Exception as exc:
            return ToolResult(ok=False, error=str(exc))

    def to_mcp_tool(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "remoteName": self.remote_name,
            "server": self.server.name,
            "description": self.description,
            "inputSchema": self.input_schema or {"type": "object", "properties": {}},
        }

    def _matches(self, user_text: str) -> bool:
        text = (user_text or "").casefold()
        return any(keyword.casefold() in text for keyword in self.keywords if keyword)

    def _render_arguments(self, user_text: str) -> Optional[Dict[str, Any]]:
        if self.argument_template:
            return {
                str(key): self._render_value(value, user_text)
                for key, value in self.argument_template.items()
            }

        properties = self.input_schema.get("properties") or {}
        required = self.input_schema.get("required") or []
        if not required:
            return {}
        if len(required) == 1 and required[0] in properties:
            return {str(required[0]): user_text}
        return None

    def _render_value(self, value, user_text: str):
        if isinstance(value, str):
            return value.replace("{user_text}", user_text or "")
        if isinstance(value, list):
            return [self._render_value(item, user_text) for item in value]
        if isinstance(value, dict):
            return {key: self._render_value(item, user_text) for key, item in value.items()}
        return value

    async def _call_tool(self, arguments: Dict[str, Any]):
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except Exception as exc:
            raise RuntimeError("MCP Python SDK is not installed. Install package `mcp` to enable MCP tools.") from exc

        server_params = _build_stdio_params(self.server, StdioServerParameters)

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session.call_tool(self.remote_name, arguments)

    @staticmethod
    def _content_to_text(result) -> str:
        content = getattr(result, "content", None)
        if content is None:
            return str(result)

        parts: List[str] = []
        for item in content:
            text = getattr(item, "text", None)
            if text is not None:
                parts.append(str(text))
                continue
            data = getattr(item, "data", None)
            if data is not None:
                parts.append(str(data))
                continue
            parts.append(str(item))
        return "\n".join(part for part in parts if part).strip()


class MCPToolProvider(ToolProvider):
    """Loads MCP tools from a JSON config file."""

    def __init__(self, servers: Iterable[MCPServerConfig]):
        self.servers = list(servers)

    @classmethod
    def from_file(cls, path: str) -> "MCPToolProvider":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return cls.from_config(raw)

    @classmethod
    def from_files(cls, paths: Iterable[str]) -> "MCPToolProvider":
        servers: List[MCPServerConfig] = []
        for path in paths:
            if not path:
                continue
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            servers.extend(cls._parse_servers(raw))
        return cls(servers)

    @classmethod
    def from_config(cls, raw: Dict[str, Any]) -> "MCPToolProvider":
        return cls(cls._parse_servers(raw))

    @classmethod
    def from_servers(cls, servers: Iterable[MCPServerConfig]) -> "MCPToolProvider":
        return cls(servers)

    @classmethod
    def from_env(cls, env_name: str = "TALKROBOT_MCP_CONFIGS") -> "MCPToolProvider":
        paths = [
            path.strip()
            for path in os.getenv(env_name, "").split(os.pathsep)
            if path.strip()
        ]
        return cls.from_files(paths)

    @staticmethod
    def _parse_servers(raw: Dict[str, Any]) -> List[MCPServerConfig]:
        servers = raw.get("servers", raw)
        configs: List[MCPServerConfig] = []
        if isinstance(servers, dict):
            items = servers.items()
        else:
            items = ((item.get("name", ""), item) for item in servers if isinstance(item, dict))

        for name, data in items:
            if not isinstance(data, dict):
                continue
            config = MCPServerConfig(
                name=str(data.get("name") or name).strip(),
                command=_expand(str(data.get("command") or "").strip()),
                args=[_expand(str(arg)) for arg in data.get("args", [])],
                env={str(k): _expand(str(v)) for k, v in (data.get("env") or {}).items()},
                cwd=_expand(str(data.get("cwd") or "").strip()),
                enabled=bool(data.get("enabled", True)),
                discover=bool(data.get("discover", False)),
                tools=list(data.get("tools") or []),
            )
            if config.name and config.command and config.enabled:
                configs.append(config)
        return configs

    def build_tools(self, context: RuntimeToolContext) -> Dict[str, BaseTool]:
        tools: Dict[str, BaseTool] = {}
        for server in self.servers:
            for tool in self._configured_tools(server):
                tools[tool.name] = tool
            if server.discover:
                for tool in self._discover_tools(server):
                    tools.setdefault(tool.name, tool)
        return tools

    def _configured_tools(self, server: MCPServerConfig) -> List[MCPTool]:
        items = []
        for data in server.tools:
            if not isinstance(data, dict):
                continue
            remote_name = str(data.get("remote_name") or data.get("name") or "").strip()
            if not remote_name:
                continue
            items.append(
                MCPTool(
                    server=server,
                    remote_name=remote_name,
                    alias=str(data.get("alias") or "").strip(),
                    description=str(data.get("description") or "").strip(),
                    input_schema=data.get("input_schema") or data.get("inputSchema") or {},
                    triggers=data.get("triggers") or [],
                    argument_template=data.get("argument_template") or {},
                    skill_argument_template=data.get("skill_argument_template") if "skill_argument_template" in data else None,
                    speakable_start=str(data.get("speakable_start") or "").strip(),
                    context_label=str(data.get("context_label") or "").strip(),
                )
            )
        return items

    def _discover_tools(self, server: MCPServerConfig) -> List[MCPTool]:
        try:
            discovered = _run_async(lambda: self._list_tools(server))
        except Exception as exc:
            logger.warning(f"MCP 工具发现失败: {server.name}: {exc}")
            return []

        tools = []
        for item in discovered:
            remote_name = str(getattr(item, "name", "") or "").strip()
            if not remote_name:
                continue
            tools.append(
                MCPTool(
                    server=server,
                    remote_name=remote_name,
                    description=str(getattr(item, "description", "") or ""),
                    input_schema=getattr(item, "inputSchema", None) or {},
                )
            )
        return tools

    async def _list_tools(self, server: MCPServerConfig):
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except Exception as exc:
            raise RuntimeError("MCP Python SDK is not installed. Install package `mcp` to enable MCP tools.") from exc

        server_params = _build_stdio_params(server, StdioServerParameters)

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                return list(getattr(result, "tools", []) or [])
