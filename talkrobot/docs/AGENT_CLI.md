# Agent CLI

Tyro Agent 可以通过独立命令行入口使用，不依赖 ASR、TTS 或 Web UI。

## 启动方式

开发环境中直接运行：

```bash
conda run -n robot_sys python -m talkrobot.agent_cli ask "帮我 review 当前改动"
```

安装为 editable package 后可以使用 console script：

```bash
conda run -n robot_sys python -m pip install -e .
tyro-agent ask "帮我 review 当前改动"
```

## 命令

### 单轮提问

```bash
python -m talkrobot.agent_cli ask "读取这个网页并总结：https://example.com"
```

### 多轮终端交互

```bash
python -m talkrobot.agent_cli chat
```

输入 `q`、`quit` 或 `exit` 退出。

保留最近 5 轮短期上下文：

```bash
python -m talkrobot.agent_cli chat --history-rounds 5
```

### 查看工具

```bash
python -m talkrobot.agent_cli tools
python -m talkrobot.agent_cli tools --json
```

### 查看 Skill

```bash
python -m talkrobot.agent_cli skills
python -m talkrobot.agent_cli skills --json
```

### 查看 MCP Tools

```bash
python -m talkrobot.agent_cli mcp
python -m talkrobot.agent_cli mcp --json
```

## 常用参数

```bash
--planner llm|rule
--show-events
--json
--project-root /path/to/project
--mcp-config /path/to/mcp.json
--mcp-configs "/path/one.json:/path/two.json"
--language zh|en
--history-rounds 5
--memory-provider none|simple|mem0
--no-memory
--max-react-iterations 4
```

默认长期记忆关闭，即 `--memory-provider none`。如果要用本地 JSON memory backend：

```bash
python -m talkrobot.agent_cli ask "我叫小明，记住" --memory-provider simple
```

## 调试 ReAct 过程

使用 `--show-events` 可以看到规划、Skill 加载、工具调用和错误：

```bash
python -m talkrobot.agent_cli ask "帮我 review 当前改动" --show-events
```

示例事件：

```text
[plan] planner=llm mode=tool_assisted steps=mcp_git_status,mcp_git_diff skills=code_review_helper
[tool:start] mcp_git_status reason=...
[tool:result] mcp_git_status ok=True elapsed_ms=120
[llm] context_chars=2300
```

## 代理环境

如果本机 VPN 设置了 `ALL_PROXY=socks://...`、`socks4://...` 或 `socks5://...`，但当前 Python 环境的 `httpx/OpenAI` 不支持 socks 代理，CLI 会在启动时自动忽略这些代理环境变量，避免 OpenAI client 初始化时报 `Unknown scheme for proxy URL`。

## JSON 输出

适合脚本集成：

```bash
python -m talkrobot.agent_cli ask "现在几点" --planner rule --json
```

输出会包含：

- `reply`
- `used_tools`
- `used_skills`
- `tool_results`
- `observations`
- `react_iterations`
- `events`

## 接入外部 MCP 配置

默认会读取项目内 `talkrobot/agent/mcp_servers.json`。实验其他开源 MCP server 时，可以通过参数临时替换或追加配置：

```bash
python -m talkrobot.agent_cli mcp --mcp-config ./my_mcp_servers.json
python -m talkrobot.agent_cli ask "用外部 fetch 读取 https://example.com" --mcp-config ./my_mcp_servers.json --show-events
```

`--mcp-configs` 支持多个配置文件，分隔符使用当前系统的路径分隔符，Linux/macOS 为 `:`，Windows 为 `;`。

## 推荐测试问题

```text
帮我 review 当前改动
当前仓库状态怎么样
读取这个网页并总结：https://example.com
查看 example/demo.sqlite 的表结构
项目里 AgentRuntime 在哪里定义
```
