import json
import os

from talkrobot.agent import AgentRuntime
from talkrobot.agent.planner import AgentPlanner, ToolStep
from talkrobot.agent.tools.builtin import CalculatorTool
from talkrobot.agent.tools import BaseTool, ToolProvider, ToolRegistry, ToolResult, mcp_stdio_server, mcp_tool
from talkrobot.agent.tools.mcp import MCPToolProvider


class EchoTool(BaseTool):
    name = "echo_tool"
    description = "Echoes user text for tests."
    speakable_start = "echo"
    context_label = "Echo Tool Result"
    keywords = ("echo",)

    def plan(self, user_text: str):
        if "echo" in user_text.lower():
            return ToolStep(self.name, {"text": user_text}, "test echo")
        return None

    def run(self, text: str) -> ToolResult:
        return ToolResult(ok=True, content=f"echoed: {text}")


class EchoProvider(ToolProvider):
    def build_tools(self, context):
        return {"echo_tool": EchoTool()}


class SkillOnlyTool(BaseTool):
    name = "skill_only_tool"
    speakable_start = "skill"
    context_label = "Skill Tool Result"

    def plan_from_skill(self, user_text: str, skill=None):
        return ToolStep(self.name, {"text": user_text}, "skill activated")

    def run(self, text: str) -> ToolResult:
        return ToolResult(ok=True, content=f"skill: {text}")


class SkillOnlyProvider(ToolProvider):
    def build_tools(self, context):
        return {"skill_only_tool": SkillOnlyTool()}


class UnstableTool(BaseTool):
    name = "unstable_tool"
    description = "Fails with a retryable error for ReAct tests."
    speakable_start = "unstable"
    context_label = "Unstable Tool Result"

    def run(self) -> ToolResult:
        return ToolResult(ok=False, error="temporary bad arguments")


class ReActProvider(ToolProvider):
    def build_tools(self, context):
        return {
            "unstable_tool": UnstableTool(),
            "echo_tool": EchoTool(),
        }


class FakeLLM:
    def __init__(self):
        self.context = ""

    def generate_response(self, user_input: str, context: str = "", system_prompt_override: str = "") -> str:
        self.context = context
        return "ok"


class PlannerLLM:
    def generate_response(self, user_input: str, context: str = "", system_prompt_override: str = "") -> str:
        if "tool planner" in system_prompt_override.lower():
            return (
                '{"mode":"tool_assisted","reason":"needs arithmetic",'
                '"steps":[{"tool":"calculator","args":{"expression":"2+3"},"reason":"calculate"}]}'
            )
        return "ok"


class ReActLLM:
    def __init__(self):
        self.context = ""

    def generate_response(self, user_input: str, context: str = "", system_prompt_override: str = "") -> str:
        if "tool planner" not in system_prompt_override.lower():
            self.context = context
            return "done"
        if "Observations:" not in user_input:
            return (
                '{"mode":"tool_assisted","reason":"try unstable first",'
                '"steps":[{"tool":"unstable_tool","args":{},"reason":"first attempt"}]}'
            )
        if "temporary bad arguments" in user_input and "echoed: fallback" not in user_input:
            return (
                '{"mode":"tool_assisted","reason":"recover with echo",'
                '"steps":[{"tool":"echo_tool","args":{"text":"fallback"},"reason":"recover"}]}'
            )
        return '{"mode":"chat","reason":"enough observations","steps":[]}'


def test_llm_planner_parses_json_tool_plan():
    planner = AgentPlanner(provider="llm")
    plan = planner.plan("2+3 等于多少", tools=[CalculatorTool()], llm=PlannerLLM())

    assert plan.mode == "tool_assisted"
    assert plan.reason == "needs arithmetic"
    assert plan.steps == [ToolStep("calculator", {"expression": "2+3"}, "calculate")]


def test_runtime_reacts_after_retryable_tool_failure(tmp_path):
    llm = ReActLLM()
    runtime = AgentRuntime(
        project_root=str(tmp_path),
        tool_registry=ToolRegistry([ReActProvider()]),
        planner_provider="llm",
        max_react_iterations=3,
    )

    events = list(runtime.run_stream("recover please", llm))
    final = events[-1]

    assert final.type == "final_response"
    assert final.data["used_tools"] == ["unstable_tool", "echo_tool"]
    assert final.data["react_iterations"] == 2
    assert final.data["observations"][0]["retryable"] is True
    assert "failed" in llm.context
    assert "temporary bad arguments" in llm.context
    assert "echoed: fallback" in llm.context


def test_runtime_uses_registered_tool_and_skill(tmp_path):
    skill_dir = tmp_path / "skills" / "echo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: echo_skill
description: test skill
triggers:
  - echo
tools:
  - echo_tool
---
# Echo Skill

Use echo tool results directly.
""",
        encoding="utf-8",
    )

    llm = FakeLLM()
    runtime = AgentRuntime(
        project_root=str(tmp_path),
        tool_registry=ToolRegistry([EchoProvider()]),
        skill_dirs=[str(tmp_path / "skills")],
    )

    events = list(runtime.run_stream("please echo this", llm))
    final = events[-1]

    assert final.type == "final_response"
    assert final.data["used_tools"] == ["echo_tool"]
    assert final.data["used_skills"] == ["echo_skill"]
    assert "Echo Tool Result" in llm.context
    assert "echoed: please echo this" in llm.context
    assert "Echo Skill" in llm.context


def test_skill_can_activate_named_tool(tmp_path):
    skill_dir = tmp_path / "skills" / "skill_tool"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: skill_tool
triggers:
  - skill-run
tools:
  - skill_only_tool
---
# Skill Tool
""",
        encoding="utf-8",
    )

    llm = FakeLLM()
    runtime = AgentRuntime(
        project_root=str(tmp_path),
        tool_registry=ToolRegistry([SkillOnlyProvider()]),
        skill_dirs=[str(tmp_path / "skills")],
    )

    events = list(runtime.run_stream("please skill-run now", llm))
    final = events[-1]

    assert final.data["used_tools"] == ["skill_only_tool"]
    assert final.data["used_skills"] == ["skill_tool"]
    assert "Skill Tool Result" in llm.context
    assert "skill: please skill-run now" in llm.context


def test_mcp_provider_loads_configured_tool_and_plans(tmp_path):
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
                                "name": "search",
                                "alias": "mcp_demo_search",
                                "triggers": ["external search"],
                                "argument_template": {"query": "{user_text}"},
                            }
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    provider = MCPToolProvider.from_file(str(config_path))
    tools = provider.build_tools(context=None)

    assert "mcp_demo_search" in tools
    step = tools["mcp_demo_search"].plan("please external search Tyro")
    assert step.tool == "mcp_demo_search"
    assert step.args == {"query": "please external search Tyro"}


def test_dev_mcp_config_registers_experiment_tools():
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "agent", "mcp_servers.json")
    provider = MCPToolProvider.from_file(config_path)
    tools = provider.build_tools(context=None)

    assert "mcp_git_status" in tools
    assert "mcp_git_diff" in tools
    assert "mcp_sqlite_schema" in tools
    assert "mcp_sqlite_query" in tools
    assert "mcp_fetch_url" in tools
    assert tools["mcp_git_status"].plan_from_skill("review", skill=None).args == {}
    assert tools["mcp_fetch_url"].plan_from_skill("https://example.com", skill=None) is None


def test_python_mcp_server_interface_registers_external_server():
    server = mcp_stdio_server(
        "external_fetch",
        "npx",
        ["-y", "@modelcontextprotocol/server-fetch"],
        discover=False,
        tools=[
            mcp_tool(
                "fetch",
                alias="mcp_external_fetch",
                description="Fetch a URL with an external open-source MCP server.",
                input_schema={
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"],
                },
                triggers=["external fetch"],
            )
        ],
    )

    registry = ToolRegistry.with_mcp_servers([server], include_builtin=False)
    tools = registry.build_tools(context=None)
    step = tools["mcp_external_fetch"].plan("please external fetch https://example.com")

    assert "mcp_external_fetch" in tools
    assert step.tool == "mcp_external_fetch"
    assert step.args == {"url": "please external fetch https://example.com"}


def test_mcp_provider_loads_multiple_config_files(tmp_path, monkeypatch):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(
        json.dumps(
            {
                "servers": {
                    "one": {
                        "command": "python",
                        "args": ["-m", "one"],
                        "tools": [{"name": "status", "alias": "mcp_one_status"}],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps(
            {
                "servers": {
                    "two": {
                        "command": "python",
                        "args": ["-m", "two"],
                        "tools": [{"name": "status", "alias": "mcp_two_status"}],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("TALKROBOT_MCP_CONFIG", str(first))
    monkeypatch.setenv("TALKROBOT_MCP_CONFIGS", str(second))
    registry = ToolRegistry.default(str(tmp_path))
    tools = registry.build_tools(context=None)

    assert "mcp_one_status" in tools
    assert "mcp_two_status" in tools
