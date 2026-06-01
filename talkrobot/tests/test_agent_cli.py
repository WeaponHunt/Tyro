import json
import os
from io import StringIO

from talkrobot import agent_cli
from talkrobot.agent.events import AgentEvent
from talkrobot.agent.llm_trace import AgentLLMTraceWrapper, classify_agent_llm_stage
from talkrobot.agent.policy import ToolApprovalRequest
from talkrobot.core.proxy_env import drop_unsupported_proxy_env


class FakeLLM:
    def __init__(self):
        self.context = ""

    def generate_response(self, user_input: str, context: str = "", system_prompt_override: str = "") -> str:
        self.context = context
        return "fake reply"


def _json_output(capsys):
    captured = capsys.readouterr()
    return json.loads(captured.out)


def test_cli_ask_runs_agent_turn_with_rule_planner(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(agent_cli, "build_llm", lambda language: FakeLLM())

    code = agent_cli.main(
        [
            "ask",
            "--project-root",
            str(tmp_path),
            "--planner",
            "rule",
            "--json",
            "2+3 等于多少",
        ]
    )

    payload = _json_output(capsys)
    assert code == 0
    assert payload["reply"].startswith("calculator 执行完成")
    assert payload["used_tools"] == ["calculator"]
    assert payload["observations"][0]["content"] == "5"


def test_cli_lists_builtin_tools(tmp_path, capsys):
    code = agent_cli.main(["tools", "--project-root", str(tmp_path), "--json"])

    payload = _json_output(capsys)
    names = {item["name"] for item in payload}
    assert code == 0
    assert "calculator" in names
    assert "current_time" in names
    assert "project_file_search" in names
    assert "repo_bootstrap" in names
    assert "shell_command" in names
    assert "file_edit" in names


def test_cli_builds_tool_authorization_from_flags(tmp_path):
    parser = agent_cli.build_parser()
    args = parser.parse_args(
        [
            "ask",
            "--project-root",
            str(tmp_path),
            "--allow-shell-command",
            "--allowed-command-prefix",
            "python -m pytest",
            "--allow-file-write",
            "--allowed-write-path",
            "src",
            "fix issue",
        ]
    )

    authorization = agent_cli.build_tool_authorization(args)

    assert authorization.allow_shell_commands is True
    assert authorization.allowed_command_prefixes == [["python", "-m", "pytest"]]
    assert authorization.allow_file_writes is True
    assert authorization.allowed_write_paths == ["src"]


def test_cli_approval_callback_respects_json_and_user_input(monkeypatch):
    parser = agent_cli.build_parser()
    json_args = parser.parse_args(["ask", "--json", "fix issue"])
    assert agent_cli.build_approval_callback(json_args) is None

    args = parser.parse_args(["ask", "fix issue"])
    callback = agent_cli.build_approval_callback(args)
    request = ToolApprovalRequest(
        tool="shell_command",
        args={"command": "python -m pytest"},
        reason="run tests",
        policy_reason="shell_command requires user authorization",
        risk_level="medium",
    )

    monkeypatch.setattr("builtins.input", lambda prompt="": "y")
    assert callback(request) is True

    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    assert callback(request) is False


def test_cli_step_trace_argument_and_formatting():
    parser = agent_cli.build_parser()
    args = parser.parse_args(["ask", "--step-trace", "clear", "fix issue"])

    assert args.step_trace == "clear"
    assert "next=file_edit" in agent_cli.format_step_trace_event(
        AgentEvent(type="plan", data={"reason": "apply fix", "steps": ["file_edit"]})
    )


def test_cli_step_trace_redacts_internal_prompt_unless_debug():
    event = AgentEvent(
        type="plan",
        data={
            "reason": "needs work Available tools:\n- shell_command\n\nTool authorization:\nsecret",
            "steps": ["shell_command"],
        },
    )

    public = agent_cli.format_step_trace_event(event, language="en", debug=False)
    debug = agent_cli.format_step_trace_event(event, language="en", debug=True)

    assert "Available tools" not in public
    assert "Tool authorization" not in public
    assert "Available tools" in debug


def test_cli_detects_turn_language():
    assert agent_cli.resolve_turn_language("请创建一个网页应用", "auto") == "zh"
    assert agent_cli.resolve_turn_language("Create a web app", "auto") == "en"
    assert agent_cli.resolve_turn_language("Create a web app", "zh") == "zh"
    assert agent_cli.chat_prompt_label("auto") == "你: "
    assert agent_cli.chat_prompt_label("en") == "You: "


def test_step_trace_printer_clear_mode_clears_completed_step():
    stream = StringIO()
    printer = agent_cli.StepTracePrinter(mode="clear", stream=stream)

    printer.handle(AgentEvent(type="tool_start", data={"tool": "file_edit", "reason": "apply fix"}))
    printer.handle(AgentEvent(type="tool_result", data={"tool": "file_edit", "ok": True, "elapsed_ms": 3}))

    output = stream.getvalue()
    assert "file_edit reason=apply fix" in output
    assert "file_edit ok" in output
    assert "\033[1A\033[2K" in output


def test_cli_lists_packaged_skills(capsys):
    code = agent_cli.main(["skills", "--json"])

    payload = _json_output(capsys)
    names = {item["name"] for item in payload}
    assert code == 0
    assert "project_helper" in names
    assert "research_helper" in names


def test_cli_mcp_config_argument_registers_external_tools(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "mcp.json"
    config_path.write_text(
        json.dumps(
            {
                "servers": {
                    "demo": {
                        "command": "python",
                        "args": ["-m", "demo_server"],
                        "tools": [
                            {
                                "name": "ping",
                                "alias": "mcp_cli_ping",
                                "description": "Ping demo MCP server.",
                                "triggers": ["ping"],
                            }
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("TALKROBOT_MCP_CONFIG", os.path.join(str(tmp_path), "missing.json"))
    code = agent_cli.main(
        [
            "mcp",
            "--project-root",
            str(tmp_path),
            "--mcp-config",
            str(config_path),
            "--json",
        ]
    )

    payload = _json_output(capsys)
    names = {item["name"] for item in payload}
    assert code == 0
    assert "mcp_cli_ping" in names
    assert os.environ["TALKROBOT_MCP_CONFIG"].endswith("missing.json")


def test_drop_unsupported_socks_proxy_env(monkeypatch):
    monkeypatch.setenv("ALL_PROXY", "socks://127.0.0.1:7897")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7890")

    removed = drop_unsupported_proxy_env(prefix="test")

    assert removed == ["ALL_PROXY"]
    assert "ALL_PROXY" not in os.environ
    assert os.environ["HTTPS_PROXY"] == "http://127.0.0.1:7890"


def test_agent_llm_trace_wrapper_writes_jsonl(tmp_path, monkeypatch):
    monkeypatch.setenv("TALKROBOT_AGENT_LLM_TRACE_ENABLED", "1")
    llm = FakeLLM()
    wrapped = AgentLLMTraceWrapper(
        llm,
        trace_id="trace-test",
        metadata={"project_root": str(tmp_path)},
        log_dir=tmp_path / "agent_llm",
    )

    response = wrapped.generate_response(
        "user prompt",
        context="tool context",
        system_prompt_override="You are Tyro's minimal ReAct controller.",
    )

    path = tmp_path / "agent_llm"
    files = list(path.glob("agent_llm_*.jsonl"))
    assert response == "fake reply"
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8").strip())
    assert payload["trace_id"] == "trace-test"
    assert payload["stage"] == "react_planner"
    assert payload["user_input"] == "user prompt"
    assert payload["context"] == "tool context"
    assert payload["response"] == "fake reply"
    assert payload["metadata"]["project_root"] == str(tmp_path)


def test_agent_llm_trace_stage_classifier():
    assert classify_agent_llm_stage("You are Tyro's minimal ReAct controller.", "") == "react_planner"
    assert classify_agent_llm_stage("", "context") == "final_response"
