import json

from talkrobot.agent import AgentRuntime
from talkrobot.agent.planner import AgentPlanner, ToolStep
from talkrobot.agent.policy import AgentToolAuthorization, ToolPolicyGate
from talkrobot.agent.state import TaskState
from talkrobot.agent.tools import BaseTool, ToolProvider, ToolRegistry, ToolResult
from talkrobot.agent.tools.builtin import CalculatorTool, FileEditTool, ProjectFileReadTool


class SequenceLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []
        self.system_prompts = []

    def generate_response(self, user_input: str, context: str = "", system_prompt_override: str = "") -> str:
        self.prompts.append(user_input)
        self.system_prompts.append(system_prompt_override)
        if self.responses:
            item = self.responses.pop(0)
            return item() if callable(item) else item
        return json.dumps({"done": True, "answer": "done", "tool": "", "args": {}, "reason": "fallback"})


class StreamingSequenceLLM(SequenceLLM):
    def __init__(self, responses, stream_chunks):
        super().__init__(responses)
        self.stream_chunks = list(stream_chunks)
        self.stream_prompts = []
        self.stream_contexts = []

    def generate_response_stream(self, user_input: str, context: str = "", system_prompt_override: str = ""):
        self.stream_prompts.append(user_input)
        self.stream_contexts.append(context)
        yield from self.stream_chunks


class EchoTool(BaseTool):
    name = "echo_tool"
    description = "Echoes text."
    speakable_start = "echo"

    def run(self, text: str) -> ToolResult:
        return ToolResult(ok=True, content=f"echoed: {text}")


class StrictArgTool(BaseTool):
    name = "strict_arg_tool"
    description = "Accepts one arg."
    speakable_start = "strict"

    def run(self, query: str) -> ToolResult:
        return ToolResult(ok=True, content=f"strict: {query}")


class TestProvider(ToolProvider):
    def build_tools(self, context):
        return {
            "calculator": CalculatorTool(),
            "echo_tool": EchoTool(),
            "strict_arg_tool": StrictArgTool(),
            "project_file_read": ProjectFileReadTool(context.project_root),
        }


def _runtime(tmp_path, **kwargs):
    return AgentRuntime(project_root=str(tmp_path), tool_registry=ToolRegistry([TestProvider()]), **kwargs)


def _final(events):
    return [event for event in events if event.type == "final_response"][-1]


def test_minimal_react_executes_one_tool_then_final(tmp_path):
    llm = SequenceLLM(
        [
            json.dumps(
                {
                    "done": False,
                    "tool": "calculator",
                    "args": {"expression": "2+3"},
                    "reason": "calculate",
                }
            ),
            json.dumps({"done": True, "answer": "结果是 5。", "tool": "", "args": {}, "reason": "enough"}),
        ]
    )

    events = list(_runtime(tmp_path).run_stream("2+3 等于多少", llm))

    assert [event.type for event in events] == ["plan", "tool_start", "tool_result", "plan", "final_response"]
    assert events[2].data["observation"]["content"] == "5"
    assert _final(events).text == "结果是 5。"
    assert "Previous step:\n(none)" in llm.prompts[0]
    assert '"content": "5"' in llm.prompts[1]


def test_progress_events_mark_planner_work(tmp_path):
    llm = SequenceLLM([json.dumps({"done": True, "answer": "完成", "tool": "", "args": {}, "reason": "done"})])

    events = list(_runtime(tmp_path).run_stream("说你好", llm, progress_events=True))

    assert events[0].type == "llm_start"
    assert events[0].data["stage"] == "react_planner"
    assert events[1].type == "plan"


def test_streaming_final_response_emits_deltas(tmp_path):
    llm = StreamingSequenceLLM(
        [json.dumps({"done": True, "answer": "非流式答案", "tool": "", "args": {}, "reason": "done"})],
        ["流式", "答案"],
    )

    events = list(_runtime(tmp_path).run_stream("说你好", llm, streaming=True, progress_events=True))

    assert [event.text for event in events if event.type == "final_response_delta"] == ["流式", "答案"]
    assert _final(events).text == "流式答案"
    assert llm.stream_prompts
    assert "planner_answer" in llm.stream_contexts[0]


def test_tool_request_takes_precedence_over_done_flag(tmp_path):
    llm = SequenceLLM(
        [
            json.dumps(
                {
                    "done": True,
                    "answer": "premature final",
                    "tool": "calculator",
                    "args": {"expression": "2+3"},
                    "reason": "tool first",
                }
            ),
            json.dumps({"done": True, "answer": "结果是 5。", "tool": "", "args": {}, "reason": "done"}),
        ]
    )

    events = list(_runtime(tmp_path).run_stream("2+3 等于多少", llm))

    assert events[0].data["reason"] == "tool first"
    assert events[0].data["steps"] == ["calculator"]
    assert events[2].data["observation"]["content"] == "5"
    assert _final(events).text == "结果是 5。"


def test_each_react_round_sees_previous_and_recent_observations(tmp_path):
    llm = SequenceLLM(
        [
            json.dumps({"done": False, "tool": "echo_tool", "args": {"text": "first"}, "reason": "first"}),
            json.dumps({"done": False, "tool": "echo_tool", "args": {"text": "second"}, "reason": "second"}),
            json.dumps({"done": True, "answer": "完成", "tool": "", "args": {}, "reason": "done"}),
        ]
    )

    events = list(_runtime(tmp_path).run_stream("连续执行", llm))

    assert _final(events).text == "完成"
    assert "echoed: first" in llm.prompts[1]
    assert "Previous step:" in llm.prompts[2]
    assert "echoed: second" in llm.prompts[2]
    assert "Recent steps:" in llm.prompts[2]
    assert "echoed: first" in llm.prompts[2]


def test_file_edit_writes_when_authorized(tmp_path):
    llm = SequenceLLM(
        [
            json.dumps(
                {
                    "done": False,
                    "tool": "file_edit",
                    "args": {"path": "index.html", "new_text": "<h1>Hello</h1>\n", "create": True},
                    "reason": "create file",
                }
            ),
            json.dumps({"done": True, "answer": "已创建 index.html。", "tool": "", "args": {}, "reason": "written"}),
        ]
    )
    runtime = AgentRuntime(
        project_root=str(tmp_path),
        tool_authorization=AgentToolAuthorization(allow_file_writes=True),
    )

    events = list(runtime.run_stream("创建一个网页文件", llm))

    assert (tmp_path / "index.html").read_text(encoding="utf-8") == "<h1>Hello</h1>\n"
    assert _final(events).text == "已创建 index.html。"
    assert _final(events).data["task_state"]["changed_files"] == ["index.html"]


def test_file_edit_accepts_content_alias_for_created_files(tmp_path):
    llm = SequenceLLM(
        [
            json.dumps(
                {
                    "done": False,
                    "tool": "file_edit",
                    "args": {"path": "index.html", "content": "<main>Game</main>\n", "create": True},
                    "reason": "create file",
                }
            ),
            json.dumps({"done": True, "answer": "已创建。", "tool": "", "args": {}, "reason": "written"}),
        ]
    )
    runtime = AgentRuntime(
        project_root=str(tmp_path),
        tool_authorization=AgentToolAuthorization(allow_file_writes=True),
    )

    events = list(runtime.run_stream("创建 index.html", llm))

    assert (tmp_path / "index.html").read_text(encoding="utf-8") == "<main>Game</main>\n"
    assert "changed 0 -> 18 chars" in events[2].data["observation"]["content"]


def test_file_edit_can_overwrite_existing_file(tmp_path):
    (tmp_path / "index.html").write_text("<p>Old</p>\n", encoding="utf-8")
    llm = SequenceLLM(
        [
            json.dumps(
                {
                    "done": False,
                    "tool": "file_edit",
                    "args": {"path": "index.html", "content": "<main>New</main>\n", "overwrite": True},
                    "reason": "replace existing file",
                }
            ),
            json.dumps({"done": True, "answer": "已覆盖。", "tool": "", "args": {}, "reason": "written"}),
        ]
    )
    runtime = AgentRuntime(
        project_root=str(tmp_path),
        tool_authorization=AgentToolAuthorization(allow_file_writes=True),
    )

    events = list(runtime.run_stream("替换 index.html", llm))

    assert (tmp_path / "index.html").read_text(encoding="utf-8") == "<main>New</main>\n"
    assert "changed 11 -> 17 chars" in events[2].data["observation"]["content"]
    assert _final(events).text == "已覆盖。"


def test_policy_rejection_is_returned_to_next_round(tmp_path):
    llm = SequenceLLM(
        [
            json.dumps(
                {
                    "done": False,
                    "tool": "file_edit",
                    "args": {"path": "index.html", "new_text": "x", "create": True},
                    "reason": "try write",
                }
            ),
            json.dumps({"done": True, "answer": "没有写入，因为缺少授权。", "tool": "", "args": {}, "reason": "blocked"}),
        ]
    )

    events = list(AgentRuntime(project_root=str(tmp_path)).run_stream("写入 index.html", llm))

    assert any(event.type == "approval_requested" for event in events)
    assert any(event.type == "approval_result" and event.data["approved"] is False for event in events)
    assert "policy rejected" in llm.prompts[1]
    assert _final(events).text == "我还没有把修改写入文件：本轮没有任何成功的文件修改记录。"


def test_final_response_cannot_claim_write_without_file_edit(tmp_path):
    llm = SequenceLLM([json.dumps({"done": True, "answer": "已经修改完成。", "tool": "", "args": {}, "reason": "claim"})])

    events = list(AgentRuntime(project_root=str(tmp_path)).run_stream("帮我修改 index.html", llm))

    assert _final(events).text == "我还没有把修改写入文件：本轮没有任何成功的文件修改记录。"
    assert _final(events).data["raw_final_replaced_reason"] == "write_not_applied"


def test_unknown_tool_args_fail_instead_of_being_silently_dropped(tmp_path):
    llm = SequenceLLM(
        [
            json.dumps(
                {
                    "done": False,
                    "tool": "strict_arg_tool",
                    "args": {"query": "weather", "max_chars": 2048},
                    "reason": "call strict",
                }
            ),
            json.dumps({"done": True, "answer": "参数错误已反馈。", "tool": "", "args": {}, "reason": "done"}),
        ]
    )

    events = list(_runtime(tmp_path).run_stream("strict", llm))
    tool_result = next(event for event in events if event.type == "tool_result")

    assert tool_result.data["observation"]["ok"] is False
    assert "unknown tool argument(s): max_chars" in tool_result.data["observation"]["error"]
    assert "unknown tool argument(s): max_chars" in llm.prompts[1]
    assert _final(events).text == "参数错误已反馈。"


def test_react_prompt_exposes_tools_as_mcp_json_schema():
    prompt = AgentPlanner._build_react_prompt(
        "创建 index.html",
        [FileEditTool("/tmp")],
        previous_observation=None,
        authorization_context="file_edit authorized: True",
    )

    assert "Available tools (MCP JSON" in prompt
    assert '"name": "file_edit"' in prompt
    assert '"inputSchema"' in prompt
    assert '"new_text"' in prompt
    assert '"content"' in prompt
    assert '"overwrite"' in prompt
    assert '"additionalProperties": false' in prompt


def test_react_prompt_includes_recent_compact_observation_history():
    prompt = AgentPlanner._build_react_prompt(
        "创建俄罗斯方块",
        [FileEditTool("/tmp")],
        previous_observation={"tool": "project_file_list", "args": {"path": "."}, "ok": True, "content": "index.html", "error": ""},
        authorization_context="file_edit authorized: True",
        observation_history=[
            {
                "iteration": 1,
                "tool": "file_edit",
                "args": {"path": "index.html", "content": "x" * 500, "create": True},
                "ok": False,
                "content": "",
                "error": "old_text is required for existing files unless overwrite=true",
                "retryable": False,
            },
            {
                "iteration": 2,
                "tool": "project_file_list",
                "args": {"path": "."},
                "ok": True,
                "content": "index.html\nscript.js",
                "error": "",
                "retryable": False,
            },
        ],
    )

    assert "Recent steps:" in prompt
    assert "old_text is required for existing files unless overwrite=true" in prompt
    assert '"content_chars": 500' in prompt
    assert "x" * 300 not in prompt


def test_rule_planner_still_handles_simple_math(tmp_path):
    events = list(_runtime(tmp_path, planner_provider="rule").run_stream("2+3", llm=object()))

    assert events[2].data["observation"]["content"] == "5"
    assert _final(events).text.startswith("calculator 执行完成")


def test_policy_gate_keeps_file_edit_inside_authorized_paths(tmp_path):
    gate = ToolPolicyGate(
        AgentToolAuthorization(allow_file_writes=True, allowed_write_paths=["src"]),
        project_root=str(tmp_path),
    )

    assert gate.assess(ToolStep("file_edit", {"path": "src/app.py"}), user_text="修改").allowed is True
    blocked = gate.assess(ToolStep("file_edit", {"path": "other/app.py"}), user_text="修改")
    assert blocked.allowed is False
    assert blocked.requires_confirmation is True


def test_task_state_records_command_verification():
    state = TaskState(goal="run tests")

    state.note_tool_result(
        "shell_command",
        {"command": "python -m pytest"},
        True,
        "$ python -m pytest\nexit_code=0\nstdout:\n================ 1 passed ================",
        "",
        {"command": ["python", "-m", "pytest"], "exit_code": 0, "stdout": "================ 1 passed ================"},
    )

    assert state.command_results[0]["check_type"] == "pytest"
    assert state.verification_results
