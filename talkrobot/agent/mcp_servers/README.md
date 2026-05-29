# MCP Servers

本目录存放 TalkRobot Agent 的本地实验 MCP server。目前已有一个本地 stdio server：

```bash
python -m talkrobot.agent.mcp_servers.dev_tools
```

对应配置文件：

```text
talkrobot/agent/mcp_servers.json
```

Agent 启动时会通过 `ToolRegistry.default()` 自动读取该配置，并把 MCP tool 包装成可规划、可执行的 Agent tool。

## 当前 MCP Tools

### Git Tools

#### `mcp_git_status`

底层 MCP tool：`git_status`

功能：查看当前仓库的 `git status --short`。

适合问题：

- 当前仓库状态怎么样
- 帮我看看当前改动
- 有哪些文件被修改了

#### `mcp_git_diff`

底层 MCP tool：`git_diff`

功能：查看当前仓库 diff。支持可选参数：

- `path`: 限制到某个项目内相对路径
- `staged`: 是否查看 staged diff

适合问题：

- 帮我 review 当前改动
- 查看当前 git diff，有没有风险
- 只看 `talkrobot/agent/runtime.py` 的改动

#### `mcp_git_log`

底层 MCP tool：`git_log`

功能：查看最近提交记录。支持可选参数：

- `limit`: 返回提交数量，默认 5，最大 20

适合问题：

- 最近提交了什么
- 看一下最近 5 个 commit

## SQLite Tools

SQLite 工具只允许读取项目目录内的数据库文件，并限制为只读查询。

#### `mcp_sqlite_schema`

底层 MCP tool：`sqlite_schema`

功能：读取 SQLite 数据库的表和字段结构。

必需参数：

- `database_path`: 项目内 SQLite 文件相对路径

适合问题：

- 查看 `example/demo.sqlite` 的表结构
- 这个数据库有哪些表

#### `mcp_sqlite_query`

底层 MCP tool：`sqlite_query`

功能：执行只读 SQLite 查询。

必需参数：

- `database_path`: 项目内 SQLite 文件相对路径
- `query`: 只读 SQL

可选参数：

- `limit`: 最大返回行数，默认 50，最大 200

允许的 SQL 类型：

- `SELECT`
- `PRAGMA`
- `EXPLAIN`
- 只读 `WITH`

禁止写入、删除、建表、修改数据等操作。

适合问题：

- 查询 `example/demo.sqlite` 里 users 表前 10 行
- 统计某张表的数据量

## Fetch Tool

#### `mcp_fetch_url`

底层 MCP tool：`fetch_url`

功能：读取公开 `http/https` URL 的文本内容。HTML 页面会被提取为可读文本。

必需参数：

- `url`: 公开网页 URL

可选参数：

- `max_chars`: 最大返回字符数，默认 8000

适合问题：

- 读取这个网页并总结：https://example.com
- 帮我看看这个链接主要讲了什么

## 关联 Skills

当前 MCP tools 主要配合以下 Skill 使用：

- `code_review_helper`: 使用 git tools 做代码审查、diff 分析。
- `local_data_analyst`: 使用 SQLite tools 查看 schema 和执行只读查询。
- `research_helper`: 使用 fetch tool 读取网页资料。

Skill 文件位置：

```text
talkrobot/agent/skills/code_review_helper/SKILL.md
talkrobot/agent/skills/local_data_analyst/SKILL.md
talkrobot/agent/skills/research_helper/SKILL.md
```

## 安全边界

当前本地 MCP server 做了几项基础限制：

- Git 命令固定在项目根目录执行。
- SQLite 数据库路径必须位于项目目录内。
- SQLite 查询只允许只读 SQL。
- Fetch 只允许 `http/https` URL。
- 返回文本会截断，避免过大内容塞进上下文。

## 新增 MCP Tool 的位置

如果要新增本地实验工具，通常改两个地方：

```text
talkrobot/agent/mcp_servers/dev_tools.py
talkrobot/agent/mcp_servers.json
```

如果要接入外部开源 MCP server，通常只需要新增或修改：

```text
talkrobot/agent/mcp_servers.json
```

也可以通过环境变量追加配置：

```bash
TALKROBOT_MCP_CONFIG=/path/to/main.json
TALKROBOT_MCP_CONFIGS=/path/to/extra1.json:/path/to/extra2.json
```
