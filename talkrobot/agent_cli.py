"""Command-line interface for the TalkRobot Agent."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Iterable, Optional

from talkrobot.agent import AgentRuntime
from talkrobot.agent.events import AgentEvent
from talkrobot.agent.tools import RuntimeToolContext, ToolRegistry
from talkrobot.config import Config
from talkrobot.core.app_logging import configure_logging
from talkrobot.core.dialogue_history import SlidingWindowDialogueHistory
from talkrobot.core.proxy_env import drop_unsupported_proxy_env
from talkrobot.modules.llm.llm_module import LLMModule


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--user", default=Config.DEFAULT_USER, help="用户名称")
    common.add_argument("--language", choices=["zh", "en"], default=Config.LANGUAGE, help="输出语言")
    common.add_argument("--planner", choices=["llm", "rule"], default=Config.AGENT_PLANNER_PROVIDER, help="planner 类型")
    common.add_argument("--project-root", default=os.getcwd(), help="项目根目录")
    common.add_argument("--mcp-config", default="", help="覆盖 TALKROBOT_MCP_CONFIG 指向的 MCP 配置文件")
    common.add_argument("--mcp-configs", default="", help="追加 MCP 配置文件，多个路径用系统路径分隔符分隔")
    common.add_argument("--max-react-iterations", type=int, default=Config.AGENT_REACT_MAX_ITERATIONS, help="ReAct 最大迭代次数")
    common.add_argument("--history-rounds", type=int, default=0, help="chat 模式保留最近 n 轮短期上下文")
    common.add_argument("--memory-provider", choices=["none", "simple", "mem0"], default="none", help="长期记忆 provider，默认关闭")
    common.add_argument("--no-memory", action="store_true", default=False, help="关闭长期记忆")
    common.add_argument("--show-events", action="store_true", default=False, help="显示 plan/tool/error 等事件")
    common.add_argument("--json", action="store_true", default=False, help="输出 JSON")

    parser = argparse.ArgumentParser(description="TalkRobot Agent CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask = subparsers.add_parser("ask", parents=[common], help="单轮询问 agent")
    ask.add_argument("question", nargs="+", help="用户问题")

    subparsers.add_parser("chat", parents=[common], help="终端多轮交互")
    subparsers.add_parser("tools", parents=[common], help="列出当前注册 tools")
    subparsers.add_parser("skills", parents=[common], help="列出当前加载 skills")
    subparsers.add_parser("mcp", parents=[common], help="列出当前 MCP servers/tools")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    configure_logging(debug=Config.DEBUG)
    drop_unsupported_proxy_env(prefix="Agent CLI")

    if args.command == "ask":
        return command_ask(args)
    if args.command == "chat":
        return command_chat(args)
    if args.command == "tools":
        return command_tools(args)
    if args.command == "skills":
        return command_skills(args)
    if args.command == "mcp":
        return command_mcp(args)
    parser.error(f"unsupported command: {args.command}")
    return 2


def command_ask(args) -> int:
    question = " ".join(args.question).strip()
    result = run_agent_turn(question, args)
    emit_answer(result, json_mode=args.json)
    return 0


def command_chat(args) -> int:
    llm = build_llm(args.language)
    memory = build_memory(args)
    runtime = build_runtime(args)
    history = SlidingWindowDialogueHistory(args.history_rounds)
    print("Tyro Agent CLI. 输入 q / quit / exit 退出。", file=sys.stderr if args.json else sys.stdout)
    while True:
        try:
            prompt = "" if args.json else ("你: " if args.language == "zh" else "You: ")
            user_text = input(prompt).strip()
        except EOFError:
            break
        if not user_text:
            continue
        if user_text.lower() in {"q", "quit", "exit"}:
            break
        result = run_agent_turn(
            user_text,
            args,
            runtime=runtime,
            llm=llm,
            memory=memory,
            sliding_window_context=history.build_context(args.user),
        )
        emit_answer(result, json_mode=args.json)
        history.append(args.user, user_text, result.get("reply", ""))
    shutdown_memory(memory)
    return 0


def command_tools(args) -> int:
    tools = build_tool_registry(args).build_tools(
        RuntimeToolContext(project_root=os.path.abspath(args.project_root), language=args.language)
    )
    payload = [
        {
            "name": name,
            "description": getattr(tool, "description", ""),
            "context_label": getattr(tool, "context_label", ""),
            "type": tool.__class__.__name__,
        }
        for name, tool in sorted(tools.items())
    ]
    emit_list(payload, args.json, title="Tools")
    return 0


def command_skills(args) -> int:
    runtime = build_runtime(args)
    payload = [
        {
            "name": skill.name,
            "description": skill.description,
            "triggers": skill.triggers,
            "tools": skill.tools,
            "path": skill.path,
        }
        for skill in runtime.skills.skills
    ]
    emit_list(payload, args.json, title="Skills")
    return 0


def command_mcp(args) -> int:
    tools = build_tool_registry(args).build_tools(
        RuntimeToolContext(project_root=os.path.abspath(args.project_root), language=args.language)
    )
    payload = []
    for name, tool in sorted(tools.items()):
        server = getattr(tool, "server", None)
        if server is None:
            continue
        payload.append(
            {
                "name": name,
                "remote_name": getattr(tool, "remote_name", ""),
                "server": getattr(server, "name", ""),
                "command": getattr(server, "command", ""),
                "args": getattr(server, "args", []),
                "description": getattr(tool, "description", ""),
                "triggers": list(getattr(tool, "keywords", ()) or ()),
            }
        )
    emit_list(payload, args.json, title="MCP Tools")
    return 0


def run_agent_turn(
    question: str,
    args,
    runtime: Optional[AgentRuntime] = None,
    llm=None,
    memory=None,
    sliding_window_context: str = "",
) -> Dict[str, Any]:
    runtime = runtime or build_runtime(args)
    llm = llm or build_llm(args.language)
    owns_memory = memory is None
    memory = build_memory(args) if memory is None else memory

    final_text = ""
    final_data: Dict[str, Any] = {}
    events = []
    try:
        for event in runtime.run_stream(
            user_text=question,
            llm=llm,
            memory_module=memory,
            long_term_memory=memory is not None,
            sliding_window_context=sliding_window_context,
            streaming=False,
        ):
            if args.show_events:
                print(format_event(event), file=sys.stderr if args.json else sys.stdout)
            events.append(event_to_dict(event))
            if event.type == "final_response":
                final_text = event.text
                final_data = event.data or {}
    finally:
        if owns_memory:
            shutdown_memory(memory)

    return {
        "reply": final_text,
        "events": events,
        **final_data,
    }


def build_runtime(args) -> AgentRuntime:
    return AgentRuntime(
        project_root=os.path.abspath(args.project_root),
        language=args.language,
        tool_registry=build_tool_registry(args),
        planner_provider=args.planner,
        max_react_iterations=args.max_react_iterations,
    )


def build_tool_registry(args) -> ToolRegistry:
    env_overrides = {
        "TALKROBOT_MCP_CONFIG": (getattr(args, "mcp_config", "") or "").strip(),
        "TALKROBOT_MCP_CONFIGS": (getattr(args, "mcp_configs", "") or "").strip(),
    }
    if not any(env_overrides.values()):
        return ToolRegistry.default(os.path.abspath(args.project_root))

    old_values = {key: os.environ.get(key) for key in env_overrides}
    try:
        for key, value in env_overrides.items():
            if value:
                os.environ[key] = value
        return ToolRegistry.default(os.path.abspath(args.project_root))
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def build_llm(language: str):
    system_prompt = Config.SYSTEM_PROMPT_EN if language == "en" else Config.SYSTEM_PROMPT
    global_prompt = Config.GLOBAL_SYSTEM_PROMPT_EN if language == "en" else Config.GLOBAL_SYSTEM_PROMPT
    prompt = "\n\n".join(part for part in (system_prompt, global_prompt) if part)
    return LLMModule(
        api_key=Config.LLM_API_KEY,
        base_url=Config.LLM_BASE_URL,
        model=Config.LLM_MODEL,
        system_prompt=prompt,
        language=language,
    )


def build_memory(args):
    provider = "none" if args.no_memory else args.memory_provider
    if provider == "none":
        return None

    from talkrobot.modules.memory.factory import create_memory_for_user

    return create_memory_for_user(args.user, provider=provider)


def shutdown_memory(memory) -> None:
    if memory is None:
        return
    try:
        memory.shutdown()
    except Exception:
        pass


def emit_answer(result: Dict[str, Any], json_mode: bool = False) -> None:
    if json_mode:
        print(json.dumps(result, ensure_ascii=False, default=str))
        return
    print(result.get("reply", ""))


def emit_list(items, json_mode: bool, title: str) -> None:
    if json_mode:
        print(json.dumps(items, ensure_ascii=False, indent=2, default=str))
        return
    print(title)
    for item in items:
        name = item.get("name", "")
        description = item.get("description", "")
        detail = f" - {description}" if description else ""
        print(f"- {name}{detail}")


def format_event(event: AgentEvent) -> str:
    data = event.data or {}
    if event.type == "plan":
        steps = ",".join(data.get("steps", []) or [])
        skills = ",".join(data.get("skills", []) or [])
        return f"[plan] planner={data.get('planner')} mode={data.get('mode')} steps={steps} skills={skills}"
    if event.type == "tool_start":
        return f"[tool:start] {data.get('tool')} reason={data.get('reason', '')}"
    if event.type == "tool_result":
        return f"[tool:result] {data.get('tool')} ok={data.get('ok')} elapsed_ms={data.get('elapsed_ms')}"
    if event.type == "skill_loaded":
        return f"[skill] {','.join(data.get('skills', []) or [])}"
    if event.type == "error":
        return f"[error] stage={data.get('stage')} error={data.get('error')}"
    if event.type == "llm_start":
        return f"[llm] context_chars={data.get('context_chars')}"
    return f"[{event.type}] {event.text}"


def event_to_dict(event: AgentEvent) -> Dict[str, Any]:
    return {
        "type": event.type,
        "text": event.text,
        "speakable": event.speakable,
        "data": event.data,
    }


if __name__ == "__main__":
    raise SystemExit(main())
