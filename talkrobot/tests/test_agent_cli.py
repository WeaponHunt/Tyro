import json
import os

from talkrobot import agent_cli
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
    assert payload["reply"] == "fake reply"
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
