# Agent 最小 ReAct 记忆翻牌游戏实测记录

日期：2026-06-01

## 测试目标

验证砍掉 coarse plan、phase、replan、final_response 二次总结后的最小 ReAct 流程，在一个空目录中创建可直接打开的网页版记忆翻牌小游戏时的实际表现。

用户任务：

```text
请在这个空文件夹里创建一个可玩的网页版记忆翻牌小游戏。要求只使用 HTML/CSS/JavaScript，直接打开 index.html 就能玩；需要有 4x4 卡牌、步数统计、计时器、重新开始按钮、胜利提示，并且界面在桌面和手机上都能正常显示。
```

运行参数：

```bash
TALKROBOT_AGENT_LLM_TRACE_DIR=<tmp>/llm-trace \
timeout 300s conda run -n robot_sys tyro-agent ask \
  --project-root <tmp> \
  --allow-file-write \
  --allowed-write-path . \
  --allow-shell-command \
  --allowed-command-prefix "node --check" \
  --max-react-iterations 12 \
  --step-trace live \
  --json \
  "<用户任务>"
```

## 第一次运行：发现工具参数兼容问题

测试目录：

```text
/tmp/tyro-agent-memory-game-react-YILCLy
```

事件序列：

```text
plan -> tool_start(file_edit) -> tool_result(ok) -> plan -> final_response
```

LLM 调用：

```text
call=1 stage=react_planner elapsed_ms=46262
call=2 stage=react_planner elapsed_ms=10398
```

现象：

```text
file_edit ok
index.html: changed 0 -> 0 chars
```

实际检查：

```bash
wc -c /tmp/tyro-agent-memory-game-react-YILCLy/index.html
# 0
```

原因：

模型第一次返回了创建文件的工具参数，但使用的是 `content` 字段，而当时 `file_edit` 工具只接受 `new_text`。runtime 会按工具函数签名过滤参数，所以 `content` 被丢弃，最终创建了一个 0 字节文件。

这个问题不是记忆游戏专属，而是通用工具协议兼容问题：模型很容易把“创建文件内容”表达成 `content`。

## 泛化修复

修复点：

```text
talkrobot/agent/tools/builtin.py
```

修改内容：

```text
FileEditTool.run 新增 content 参数。
当 new_text 为空且 content 非空时，将 content 作为 new_text 的别名使用。
FileEditTool.description 也补充说明 content 是 new_text alias。
```

同时修了一个日志清晰度问题：

```text
talkrobot/agent/planner.py
```

修改内容：

```text
正常的 {done:false, tool, args} 不再被误标为 legacy_tool_envelope。
如果模型同时给出 done=true 和 tool，则 tool 优先，先执行工具，再进入下一轮判断。
```

新增/更新测试后结果：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n robot_sys python -m pytest \
  talkrobot/tests/test_agent_runtime.py talkrobot/tests/test_agent_cli.py -q

# 24 passed in 0.30s
```

## 最终复测结果

最终测试目录：

```text
/tmp/tyro-agent-memory-game-react-final-4SIxNy
```

最终生成文件：

```text
/tmp/tyro-agent-memory-game-react-final-4SIxNy/index.html
```

文件大小：

```bash
wc -c /tmp/tyro-agent-memory-game-react-final-4SIxNy/index.html
# 5769
```

事件序列：

```text
[决策] 下一步=file_edit
[执行] file_edit
[观察] file_edit ok
[决策] 下一步=none
final_response
```

完整 runtime 事件形态：

```text
plan -> tool_start(file_edit) -> tool_result(ok) -> plan -> final_response
```

LLM trace：

```text
call=1 stage=react_planner elapsed_ms=65078
response: done=false, tool=file_edit, args.path=index.html, args.create=true, args.new_text=<完整 HTML>

call=2 stage=react_planner elapsed_ms=6567
response: done=true, answer=<中文完成说明>
```

工具执行结果：

```text
file_edit ok
index.html: changed 0 -> 5670 chars
bytes=5769
changed_files=["index.html"]
```

最终回复：

```text
已成功创建记忆翻牌小游戏！你可以直接在浏览器中打开 `index.html` 文件开始游玩。

游戏包含以下功能：
- 4x4 卡牌网格
- 步数统计
- 计时器
- 重新开始
- 胜利提示
- 响应式设计
```

## 静态验证

关键功能定位：

```text
4x4 grid: grid-template-columns: repeat(4, 1fr)
步数: id="moves"
计时器: id="timer"
重新开始: id="restart"
胜利提示: modal.show
胜利统计: final-moves / final-time
响应式/自适应: max-width, aspect-ratio, clamp
游戏逻辑: flipCard, matches, lockBoard
```

脚本语法检查：

```bash
awk '/<script>/{flag=1;next}/<\/script>/{flag=0}flag' index.html > script.js
node --check script.js

# 通过，无输出
```

这次没有做真实浏览器点击测试，所以只能确认文件落盘、结构存在、JS 语法通过；没有自动验证翻牌交互的运行时行为。

## 和旧流程对比

这次最明显的改善：

```text
没有 first_turn_decision。
没有 coarse plan。
没有 phase_start / phase_verdict。
没有 replan。
没有每个 phase 之后的 final_response。
没有 300s timeout。
最终 2 次 LLM 调用就结束。
```

旧流程里出现过的问题：

```text
模型已经在 final_response 声称完成，但 phase verifier 判定 continue，于是进入 replan。
最终任务完成状态和模型总结状态脱节。
```

新流程里这个问题消失了。模型只做一件事：根据上一轮 observation 决定下一步是 tool 还是 final。

## 当前仍然存在的问题

1. 模型没有主动运行验证工具。

虽然本次命令授权了：

```text
--allow-shell-command --allowed-command-prefix "node --check"
```

但模型在文件写入成功后直接 final，没有选择 `shell_command`。这符合“最小 ReAct”的简单性，但也说明验证能力需要更明确的提示或外部策略。

2. 最小 ReAct 只看上一轮 observation，信息压力小，但长期任务可能遗忘较多。

本任务只需要一次创建文件，所以表现很好。如果任务需要多文件、多步骤定位，可能需要在 observation 之外加入一个很小的 running summary，而不是恢复复杂 phase 机制。

3. `file_edit` 仍然依赖模型一次性给出完整文件内容。

对于大文件或精细修改，这种 `old_text/new_text` 方案仍然脆弱。后续可以考虑增加更直接的 `write_file` 或 patch 工具，但保持同样简单的 ReAct 控制流。

## 结论

最小 ReAct 版本在这个任务上明显比旧流程更稳定：它没有陷入 replan/final 循环，能在一次工具调用后收敛。

这次测试也暴露并修复了一个通用工具兼容问题：`file_edit` 需要接受 `content` 作为 `new_text` 的别名。修复后，agent 能实际创建非空 `index.html`，并给出一致的最终反馈。
