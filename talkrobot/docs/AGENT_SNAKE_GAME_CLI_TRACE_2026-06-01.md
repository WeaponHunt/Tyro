# Agent CLI 贪吃蛇小游戏实测 Trace

日期：2026-06-01

## 测试目标

验证 Tyro Agent CLI 在一个空目录里创建可直接打开的网页版贪吃蛇小游戏的效果，并把底层 JSON trace 整理成人类可读的过程记录。

用户任务：

```text
请在这个空文件夹里创建一个可玩的网页版贪吃蛇小游戏。要求只使用 HTML/CSS/JavaScript，直接打开 index.html 就能玩；需要有开始/暂停/重新开始、方向键和 WASD 控制、分数与最高分、游戏结束提示，并且界面在桌面和手机上都能正常显示。
```

运行命令：

```bash
TALKROBOT_AGENT_LLM_TRACE_DIR=/tmp/tyro-agent-snake-trace-IURpa4 \
timeout 300s conda run -n robot_sys python -m talkrobot.agent_cli ask \
  --project-root /tmp/tyro-agent-snake-k1m7lj \
  --allow-file-write \
  --allowed-write-path . \
  --allow-shell-command \
  --allowed-command-prefix "node --check" \
  --max-react-iterations 12 \
  --step-trace live \
  --show-events \
  --json \
  "<用户任务>"
```

产物目录：

```text
项目目录: /tmp/tyro-agent-snake-k1m7lj
LLM trace: /tmp/tyro-agent-snake-trace-IURpa4/agent_llm_2026-06-01.jsonl
生成文件: /tmp/tyro-agent-snake-k1m7lj/index.html
```

## 总览

```text
结果: 成功生成 index.html
耗时: 107912 ms
ReAct 迭代: 1 次工具迭代，随后完成
模型调用: 2 次 react_planner
使用工具: file_edit
写入文件: index.html
文件大小: 8105 bytes
JS 语法检查: 通过
浏览器实测: 未执行，本机未安装 Playwright/Chromium
```

## 时间线

| 顺序 | 阶段 | 发生了什么 | 结果 |
| --- | --- | --- | --- |
| 1 | planner call 1 | 模型生成完整 `index.html`，并要求调用 `file_edit` | 95322 ms |
| 2 | plan event | runtime 判断下一步为 `file_edit` | tool_assisted |
| 3 | tool_start | 执行 `file_edit` 写入 `index.html` | started |
| 4 | tool_result | 文件从 0 chars 写到 7855 chars | ok，8105 bytes |
| 5 | planner call 2 | 模型读取上一步观察后判断任务完成 | 12584 ms |
| 6 | plan event | 下一步为空 | final |
| 7 | final_response | 返回中文完成说明 | success |

## ReAct Trace

### Step 1: 决策

runtime 收到第一轮 planner 输出。值得注意的是，模型输出里同时包含：

```text
done=true
tool=file_edit
```

当前 runtime 采用“有工具调用时工具优先”的策略，所以没有直接相信 `done=true`，而是先执行写文件动作。这一点是好的，否则可能出现模型口头宣称完成但文件没有落盘的问题。

公开事件：

```text
[决策] 下一步=file_edit
[plan] planner=llm mode=tool_assisted steps=file_edit skills=
```

模型给出的决策理由：

```text
The user requested a fully functional, responsive Snake game using HTML/CSS/JS in a single index.html file. I have generated the complete code including game logic, keyboard/touch controls, UI, and responsive scaling, and will create the file now.
```

工具调用：

```text
tool: file_edit
path: index.html
create: true
content: 完整 HTML/CSS/JavaScript，最终落盘为 /tmp/tyro-agent-snake-k1m7lj/index.html
```

### Step 2: 执行

公开事件：

```text
[执行] file_edit
[tool:start] file_edit
[观察] file_edit ok elapsed_ms=0
[tool:result] file_edit ok=True elapsed_ms=0
```

工具观察：

```text
ok: true
content: index.html: changed 0 -> 7855 chars
data.path: index.html
data.created: true
data.bytes: 8105
retryable: false
```

### Step 3: 完成判断

第二轮 planner 收到 `file_edit` 成功观察后，判断无需继续调用工具。

公开事件：

```text
[决策] 下一步=none
[plan] planner=llm mode=final steps= skills=
```

模型给出的完成理由：

```text
The previous step successfully created the index.html file containing a fully functional Snake game with all requested features (HTML/CSS/JS only, start/pause/restart, keyboard & touch controls, score/high score, game over overlay, and responsive design). No further actions are needed.
```

最终回复：

```text
贪吃蛇小游戏已成功创建在 `index.html` 文件中。您可以直接在浏览器中打开它进行游玩。

游戏包含以下功能：
1. **控制方式**：支持键盘方向键和 WASD 控制，手机端支持滑动屏幕控制方向。
2. **游戏按钮**：提供“开始”、“暂停”和“重新开始”按钮，同时界面上也有覆盖层按钮方便操作。
3. **分数系统**：实时显示当前分数，并使用 `localStorage` 记录历史最高分。
4. **游戏结束提示**：撞墙或撞到自己时会弹出游戏结束提示及最终分数。
5. **响应式界面**：Canvas 会根据屏幕大小自动缩放，确保在桌面和移动设备上都能获得良好的视觉体验。
```

## LLM 调用明细

trace 文件：

```text
/tmp/tyro-agent-snake-trace-IURpa4/agent_llm_2026-06-01.jsonl
```

调用摘要：

| call | timestamp | stage | elapsed_ms | model | response_chars | error |
| --- | --- | --- | ---: | --- | ---: | --- |
| 1 | 2026-06-01T15:43:10.840+08:00 | react_planner | 95322 | qwen3.7-max | 8702 | none |
| 2 | 2026-06-01T15:43:23.427+08:00 | react_planner | 12584 | qwen3.7-max | 668 | none |

trace id：

```text
record.trace_id: 734b040b-0635-4f20-9fc8-610608577195
metadata.trace_id: 5bcf1a2a-c1ab-410c-85b4-d06402a5ac9b
```

说明：`record.trace_id` 来自 `AgentLLMTraceWrapper`，`metadata.trace_id` 来自 CLI 本轮请求元数据。两者都能定位这次调用，但当前实现会生成两个不同 ID。

## 生成文件检查

文件：

```text
/tmp/tyro-agent-snake-k1m7lj/index.html
```

大小：

```text
8105 bytes
```

关键功能定位：

```text
Canvas: <canvas id="gameCanvas" width="400" height="400">
开始按钮: id="startBtn"
暂停按钮: id="pauseBtn"
重新开始按钮: id="restartBtn"
方向键/WASD: document.addEventListener('keydown', ...)
移动端滑动: touchstart / touchend
当前分数: id="score"
最高分: id="highScore" + localStorage.setItem('snakeHighScore', ...)
游戏结束提示: gameOver() 更新 overlayTitle / overlayMsg
响应式尺寸: resizeCanvas() 使用 window.innerWidth / innerHeight
```

语法检查：

```bash
awk '/<script>/{flag=1;next}/<\/script>/{flag=0}flag' \
  /tmp/tyro-agent-snake-k1m7lj/index.html \
  > /tmp/tyro-agent-snake-k1m7lj/script.js

node --check /tmp/tyro-agent-snake-k1m7lj/script.js
```

结果：

```text
通过，无输出
```

浏览器自动化检查：

```text
未执行。当前环境未找到 playwright-python、playwright-node 或 Chromium/Chrome 可执行文件。
```

## 效果评价

整体评价：这次 CLI 表现是成功的。它在一次工具迭代里完成了单文件网页小游戏创建，需求覆盖较完整，生成代码可直接打开，JS 语法通过，结构也比较清楚。

亮点：

```text
1. 文件确实落盘，没有只给口头答复。
2. 覆盖了开始、暂停、重开、分数、最高分、游戏结束提示。
3. 键盘方向键和 WASD 都有处理。
4. 额外提供了移动端滑动控制。
5. 第二轮 planner 能基于工具观察结束任务，没有继续无意义 replan。
```

不足或风险：

```text
1. 未做真实浏览器点击/键盘/触摸自动化验证，所以运行时交互只能基于代码静态判断。
2. UI 配色和布局可玩但偏基础，按钮区在极窄屏上没有 flex-wrap，仍有小概率拥挤。
3. touchend 里 `if (!isRunning) return;`，所以手机端滑动不能从初始遮罩直接开局，只能点按钮开始后再滑动控制。
4. 自撞检测在 `snake.pop()` 之前检查完整身体，某些蛇类实现中“移动到当前尾巴位置”本应合法，这里会被判定为撞到自己。
5. trace 的 CLI live 输出仍偏短，真正完整的信息主要在 JSONL；这份文档是后处理后的可读版本。
```

## 原始 CLI 可见输出摘要

```text
[决策] reason=<任务需要创建完整 Snake 单文件网页> 下一步=file_edit
[plan] planner=llm mode=tool_assisted steps=file_edit skills=
[执行] file_edit reason=<创建完整 HTML/CSS/JS 文件>
[tool:start] file_edit reason=<创建完整 HTML/CSS/JS 文件>
[观察] file_edit ok elapsed_ms=0
[tool:result] file_edit ok=True elapsed_ms=0
[决策] reason=<文件已成功创建，无需继续> 下一步=none
[plan] planner=llm mode=final steps= skills=
[final_response] 贪吃蛇小游戏已成功创建在 `index.html` 文件中...
```

