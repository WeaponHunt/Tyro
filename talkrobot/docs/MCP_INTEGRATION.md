# MCP 接入说明

TalkRobot Agent 支持两种方式接入开源 MCP server：

1. 写 JSON 配置，由 `ToolRegistry.default()` 自动加载。
2. 在 Python 代码里用 `mcp_stdio_server()` / `mcp_tool()` 构造 server，再传给 `AgentRuntime.with_mcp_servers()`。

## JSON 配置

默认读取：

```text
talkrobot/agent/mcp_servers.json
```

也可以用环境变量覆盖或追加：

```bash
TALKROBOT_MCP_CONFIG=/path/to/primary.json
TALKROBOT_MCP_CONFIGS=/path/to/extra1.json:/path/to/extra2.json
```

配置示例：

```json
{
  "servers": {
    "external_fetch": {
      "enabled": true,
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-fetch"],
      "discover": false,
      "tools": [
        {
          "name": "fetch",
          "alias": "mcp_external_fetch",
          "description": "Fetch readable text from a URL.",
          "input_schema": {
            "type": "object",
            "properties": {
              "url": {"type": "string"}
            },
            "required": ["url"]
          },
          "triggers": ["external fetch", "读取网页"],
          "context_label": "外部网页读取结果"
        }
      ]
    }
  }
}
```

字段说明：

- `command` / `args`: MCP server 的 stdio 启动命令。
- `env`: 传给 server 的环境变量。
- `cwd`: server 启动目录。
- `discover`: 是否启动 server 发现 tool。实验阶段建议先 `false`，手动声明 tool 更稳定。
- `tools[].name`: MCP server 内真实 tool 名。
- `tools[].alias`: 暴露给 Agent 的 tool 名。
- `tools[].triggers`: 规则 planner 或 tool 自规划时的触发词。
- `tools[].argument_template`: 命中 triggers 时如何生成参数，支持 `{user_text}`。
- `tools[].skill_argument_template`: skill 命中时如何生成参数；不设置则 skill 不会直接触发该 tool。

## Python 接口

可以直接在代码里接入外部 MCP server：

```python
from talkrobot.agent import AgentRuntime
from talkrobot.agent.tools import mcp_stdio_server, mcp_tool

server = mcp_stdio_server(
    "external_fetch",
    "npx",
    ["-y", "@modelcontextprotocol/server-fetch"],
    tools=[
        mcp_tool(
            "fetch",
            alias="mcp_external_fetch",
            description="Fetch readable text from a URL.",
            input_schema={
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
            triggers=["external fetch", "读取网页"],
        )
    ],
)

runtime = AgentRuntime.with_mcp_servers([server], project_root="/home/acir/Tyro")
```

这个接口适合快速试开源 MCP server，或在测试里动态拼装工具。

## 当前内置实验 MCP Server

本仓库提供了一个本地实验 server：

```bash
python -m talkrobot.agent.mcp_servers.dev_tools
```

配置文件：

```text
talkrobot/agent/mcp_servers.json
```

暴露给 Agent 的工具：

- `mcp_git_status`
- `mcp_git_diff`
- `mcp_git_log`
- `mcp_sqlite_schema`
- `mcp_sqlite_query`
- `mcp_fetch_url`

对应实验 skill：

- `code_review_helper`
- `local_data_analyst`
- `research_helper`

## 注意

- 运行 MCP tool 需要安装 `mcp` Python SDK。
- 某些开源 MCP server 需要 Node.js / `npx` / Docker / API key。
- 如果开启 `discover`，Agent 会在构建工具时启动 MCP server 做 tool discovery；启动慢或不稳定的 server 建议手动声明 tools。
