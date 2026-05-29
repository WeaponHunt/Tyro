# TalkRobot 代码结构

本文档描述当前主项目代码结构。正式运行链路集中在 `talkrobot/` 下，顶层 `example/`、`record/`、`expression/`、`expression_video/` 为独立示例或辅助脚本。

## 入口

- `talkrobot/main.py`: 命令行入口，负责初始化 ASR、TTS、LLM、Memory、Expression、Face、ConversationManager，并提供 `chat` 与 `add-memory` 子命令。
- `talkrobot/web_app.py`: FastAPI Web 入口，复用 LLM、Memory、Persona、AgentRuntime，提供文本聊天、记忆管理和静态页面服务。
- `talkrobot/config.py`: 全局配置与环境变量读取，集中管理模型、音频、TTS、LLM、记忆、人脸识别、人格和日志配置。

## Core 层

`talkrobot/core/` 放置运行编排和跨模块基础能力。

- `conversation_manager.py`: 语音/文本对话主编排，处理唤醒、用户切换、ASR、AgentRuntime、TTS、表情、记忆写入和交互日志。
- `audio_recorder.py`: 录音与 VAD 检测，支持按键录音和持续监听。
- `tts_playback_controller.py`: TTS 播放控制与打断。
- `dialogue_history.py`: 按用户隔离的滑动窗口短期对话历史。
- `memory_router.py`: 多用户长期记忆实例管理。
- `persona_manager.py`: 用户人格 prompt 读取与写回。
- `face_identity_resolver.py`: 人脸识别结果到当前用户的解析。
- `expression_server_manager.py`: 表情服务进程管理。
- `app_logging.py`: 控制台日志与按日期滚动文件日志。
- `interaction_log.py`: 每轮交互 JSONL 结构化日志。

## Modules 层

`talkrobot/modules/` 放置可独立替换的功能模块。

- `asr/asr_module.py`: 语音识别封装。
- `tts/tts_module.py`: TTS 合成与播放封装。
- `llm/llm_module.py`: OpenAI-compatible Chat Completions 调用。
- `llm/persona_update_agent.py`: 后台人格更新 Agent。
- `memory/memory_module.py`: 长期记忆 facade，保持运行代码调用接口稳定。
- `memory/base.py`: 可替换 memory backend 接口与 `MemoryRecord`。
- `memory/mem0_backend.py`: 原有 Mem0 backend 适配。
- `memory/simple_backend.py`: 本地 JSON backend，适合离线测试和替换链路验证。
- `memory/factory.py`: 根据 provider 创建 memory backend。
- `memory/filters.py`: 稳定记忆过滤规则。
- `expression/expression_module.py`: 表情标签解析和表情服务调用。
- `face_recognize/face_recognition.py`: 人脸识别模块。

## Agent 层

`talkrobot/agent/` 是正式 agent 架构，采用 ReAct 链路：“规划工具 -> 执行工具 -> 记录 observation -> 必要时继续规划 -> 把结果注入上下文 -> LLM 生成”。

- `runtime.py`: 单轮 agent runtime，负责构建工具、匹配 skill、执行 ReAct 循环、汇总上下文并调用 LLM。
- `planner.py`: planner facade，支持 `llm` 和 `rule` 两种规划模式；默认由大模型输出 JSON 工具计划，失败时回退规则规划，也会调用扩展工具的 `plan()` / `plan_from_skill()`。
- `events.py`: AgentRuntime 对外发出的事件类型。
- `skills.py`: 递归加载 `SKILL.md`，根据触发词和工具名匹配 skill。
- `skills/<name>/SKILL.md`: skill 描述文件，可声明 `name`、`description`、`triggers`、`tools`。
- `mcp_servers.example.json`: MCP 工具配置模板。
- `mcp_servers.json`: 本地实验 MCP 工具配置，默认接入 git、sqlite、fetch 工具。
- `mcp_servers/dev_tools.py`: 本地实验 MCP server，使用 stdio 暴露 `git_status`、`git_diff`、`git_log`、`sqlite_schema`、`sqlite_query`、`fetch_url`。

### Tools

`talkrobot/agent/tools/` 是工具扩展层。

- `base.py`: `BaseTool`、`ToolResult`、`ToolProvider`、`RuntimeToolContext` 基础接口。
- `builtin.py`: 内置低风险工具，包括记忆检索/写入、时间、计算、项目文件检索/读取/列表、网页读取。
- `registry.py`: `ToolRegistry` 汇总多个 `ToolProvider`，默认加载内置工具和可选 MCP 配置。
- `mcp.py`: MCP stdio tool adapter，把 MCP server tool 包装成 `BaseTool`。

新增普通工具时，实现 `BaseTool.run()`，需要自动规划时实现 `plan()`。新增一组工具时，实现 `ToolProvider.build_tools()` 并传给 `ToolRegistry`。

新增 MCP 工具时，复制 `mcp_servers.example.json` 为 `mcp_servers.json`，配置 server 启动命令、tool 名称、触发词和 `argument_template`。也可以用 `TALKROBOT_MCP_CONFIG` 指向外部配置。

接入开源 MCP server 的详细方式见 `talkrobot/docs/MCP_INTEGRATION.md`。代码里也可以使用 `mcp_stdio_server()` / `mcp_tool()` 构造 server，并通过 `AgentRuntime.with_mcp_servers()` 直接创建 runtime。

新增 skill 时，在 `talkrobot/agent/skills/<skill_name>/SKILL.md` 写 front matter。外部 skill 目录可通过 `TALKROBOT_SKILL_DIRS` 追加。

当前内置实验 skill：

- `code_review_helper`: 结合 MCP git 工具审查当前 diff。
- `local_data_analyst`: 使用 MCP sqlite 工具查看 schema 并执行只读查询。
- `research_helper`: 使用 MCP fetch 或内置 web fetch 读取 URL 内容。

planner 可通过环境变量切换：

```bash
TALKROBOT_AGENT_PLANNER=llm
TALKROBOT_AGENT_PLANNER=rule
TALKROBOT_AGENT_REACT_MAX_ITERATIONS=4
```

## 测试

`talkrobot/tests/` 放置模块级测试和 agent 架构测试。部分测试需要音频设备、模型文件、API key 或本地服务；仅做语法和架构验证时，优先运行：

```bash
conda run -n robot_sys python -m compileall talkrobot/agent talkrobot/core talkrobot/modules talkrobot/web_app.py talkrobot/main.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n robot_sys python -m pytest talkrobot/tests/test_agent_runtime.py
```

## 日志与运行数据

- `talkrobot/logs/`: 运行日志目录，不应提交。
- `talkrobot/logs/interactions/`: 每轮交互 JSONL 日志目录，不应提交。
- `talkrobot/mem_db/`: 默认长期记忆数据库目录，不应提交。

## 当前清理原则

- 主运行代码集中保留在 `talkrobot/`。
- 示例和实验代码不参与主链路时，不放在主包内。
- 工具扩展走 `ToolRegistry` / `ToolProvider`，避免在 `AgentRuntime` 内继续堆分支。
- Skill 只负责注入行为说明和声明建议工具，具体执行逻辑留在 tool。
