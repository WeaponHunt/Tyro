# Agent 复杂任务实测记录：网页版记忆翻牌小游戏

## 测试概览

- 测试时间：2026-06-01 13:08 Asia/Shanghai
- 测试目录：`/tmp/tyro-agent-memory-game-20260601-130812`
- LLM trace 目录：`/tmp/tyro-agent-memory-game-20260601-130812-llm-trace`
- 生成文件：`/tmp/tyro-agent-memory-game-20260601-130812/index.html`
- 最终命令状态：`124`，即被 `timeout 300s` 终止

用户请求：

```text
请在这个空文件夹里创建一个可玩的网页版记忆翻牌小游戏。要求只使用 HTML/CSS/JavaScript，直接打开 index.html 就能玩；需要有 4x4 卡牌、步数统计、计时器、重新开始按钮、胜利提示，并且界面在桌面和手机上都能正常显示。
```

执行命令：

```bash
TALKROBOT_AGENT_LLM_TRACE_DIR=/tmp/tyro-agent-memory-game-20260601-130812-llm-trace \
timeout 300s conda run -n robot_sys tyro-agent ask \
  --project-root /tmp/tyro-agent-memory-game-20260601-130812 \
  --allow-file-write \
  --allowed-write-path . \
  --max-react-iterations 8 \
  --step-trace live \
  "请在这个空文件夹里创建一个可玩的网页版记忆翻牌小游戏。要求只使用 HTML/CSS/JavaScript，直接打开 index.html 就能玩；需要有 4x4 卡牌、步数统计、计时器、重新开始按钮、胜利提示，并且界面在桌面和手机上都能正常显示。"
```

## 产物检查

`index.html` 已生成，大小约 9.8 KB。静态检查结果：

```text
node --check script.js 通过
index.html: 9853 bytes
script.js: 4051 bytes
```

从文件内容看，已包含：

- 4x4 CSS grid：`grid-template-columns: repeat(4, 1fr)`
- 卡牌点击事件：`card.addEventListener('click', () => flipCard(card))`
- 步数统计：`moves`
- 计时器：`setInterval`
- 重新开始按钮：`initGame()`
- 胜利提示：`showVictory()`
- 响应式样式：`@media (max-width: 600px)`

结论：**产物本身基本成功生成，且具备请求的核心功能。**

## LLM 调用时间线

LLM trace 文件：

```text
/tmp/tyro-agent-memory-game-20260601-130812-llm-trace/agent_llm_2026-06-01.jsonl
```

本轮共有 12 次模型调用：

| call | stage | elapsed_ms | 实际行为 |
| --- | --- | ---: | --- |
| 1 | `first_turn_decision` | 14847 | 判定为 `plan_then_react`，生成 coarse plan：先创建 `index.html`，再验证。 |
| 2 | `react_planner` | 42792 | 在 phase_1 生成 `file_edit`，创建完整 `index.html`。 |
| 3 | `react_planner` | 4700 | 判断 `index.html` 已创建，返回 `mode=chat`。 |
| 4 | `final_response` | 44851 | 生成第一次完成总结：“已经创建好了”。 |
| 5 | `replan` | 20069 | 又触发 replan；模型返回的 schema 使用 `name/description`，不是当前代码期望的 `id/objective`。 |
| 6 | `react_planner` | 4402 | 进入验证阶段，计划 `project_file_list` + `project_file_read`。 |
| 7 | `react_planner` | 7371 | 判断验证完成，返回 `mode=chat`。 |
| 8 | `final_response` | 24371 | 生成第二次完成总结。 |
| 9 | `replan` | 29184 | 又触发 replan，模型再次输出“已完成”的计划形态。 |
| 10 | `react_planner` | 11161 | 在 `replan_1` 阶段判断无需工具，`mode=chat`。 |
| 11 | `final_response` | 80283 | 生成第三次完成总结。 |
| 12 | `replan` | 13823 | 再次 replan，准备进入 `verify_and_complete`，随后进程被 timeout 终止。 |

## 关键发现

### 1. 文件写入成功

这次不是“模型说写了但实际没写”。证据：

- `index.html` 存在。
- LLM trace 第 2 次调用明确规划了 `file_edit`。
- 文件内容包含完整 HTML/CSS/JS。
- JS 语法检查通过。

### 2. CLI 输出没有落盘

`agent.stdout.txt` 和 `agent.stderr.txt` 都是 0 字节。原因不是没有事件，而是：

- 命令通过 `conda run` 执行；
- 进程最终由 `timeout 300s` 杀掉；
- `conda run` 对输出有缓冲，导致 step trace 没来得及刷出。

所以这次主要依靠 `agent_llm` JSONL trace 和生成文件判断真实执行过程。

### 3. 流程没有正常停止

最大问题是：**agent 已经完成创建和验证，却没有结束任务，而是继续 replan。**

表现：

- 第 4、8、11 次调用都已经生成“任务完成”的最终回复。
- 但 phase loop 仍继续触发 replan。
- 最后超过 300 秒被杀。

这说明当前阶段验证/终止逻辑仍不够稳。它能做事，但在“何时停止”上还有问题。

### 4. replan schema 容错不足

第 5 次和第 9 次 replan，模型返回了类似：

```json
{
  "phases": [
    {
      "name": "phase_1",
      "description": "...",
      "done_when": "..."
    }
  ]
}
```

而代码里的 coarse plan parser 期望字段是：

```json
{
  "id": "phase_1",
  "objective": "...",
  "done_when": "..."
}
```

这会导致 replan 结果不稳定，可能回退到本地 fallback plan，进一步增加循环概率。

## 初步结论

这次测试说明 agent 已经具备“实际创建复杂前端文件”的能力，但流程控制仍有明显缺陷：

1. 写文件能力正常。
2. 复杂任务产物基本可用。
3. LLM trace 能定位每一次模型调用。
4. 主要问题从“不会写”转移到了“完成后不会稳定停”。

## 建议后续修复点

1. **阶段完成后应立即终止最终阶段**
   如果最后一个 phase 已经产生 final response，且 `changed_files` 或验证证据满足任务目标，应直接 return，不应继续 replan。

2. **增强 `PhaseVerifier`**
   对文件创建任务，`file_edit ok` + `project_file_read ok` + 关键功能文本存在，应判定为 `phase_done`。

3. **增强 replan schema 兼容**
   支持模型返回的 `name -> id`、`description -> objective`，减少无意义 fallback。

4. **增加 no-progress guard**
   如果连续多次 final/replan 没有新增工具结果、没有新增 changed_files、没有新增 verification evidence，应停止并输出当前状态。

5. **避免 `conda run` 输出缓冲影响调试**
   调试时可以优先用环境内 Python 直接运行入口，或让 CLI step trace 同时写入文件。
