"""Command-line interface for the TalkRobot Agent."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import uuid
from typing import Any, Dict, Iterable, Optional

from talkrobot.agent import AgentRuntime
from talkrobot.agent.events import AgentEvent
from talkrobot.agent.llm_trace import wrap_agent_llm_trace
from talkrobot.agent.policy import AgentToolAuthorization, ToolApprovalRequest
from talkrobot.agent.tools import RuntimeToolContext, ToolRegistry
from talkrobot.config import Config
from talkrobot.core.app_logging import configure_logging
from talkrobot.core.dialogue_history import SlidingWindowDialogueHistory
from talkrobot.core.proxy_env import drop_unsupported_proxy_env
from talkrobot.modules.llm.llm_module import LLMModule


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--user", default=Config.DEFAULT_USER, help="用户名称")
    common.add_argument("--language", choices=["auto", "zh", "en"], default="auto", help="输出语言，auto 会跟随本轮用户输入")
    common.add_argument("--planner", choices=["llm", "rule"], default=Config.AGENT_PLANNER_PROVIDER, help="planner 类型")
    common.add_argument("--project-root", default=os.getcwd(), help="项目根目录")
    common.add_argument("--mcp-config", default="", help="覆盖 TALKROBOT_MCP_CONFIG 指向的 MCP 配置文件")
    common.add_argument("--mcp-configs", default="", help="追加 MCP 配置文件，多个路径用系统路径分隔符分隔")
    common.add_argument("--max-react-iterations", type=int, default=Config.AGENT_REACT_MAX_ITERATIONS, help="ReAct 最大迭代次数")
    common.add_argument("--history-rounds", type=int, default=0, help="chat 模式保留最近 n 轮短期上下文")
    common.add_argument("--memory-provider", choices=["none", "simple", "mem0"], default="none", help="长期记忆 provider，默认关闭")
    common.add_argument("--no-memory", action="store_true", default=False, help="关闭长期记忆")
    common.add_argument("--allow-shell-command", action="store_true", default=False, help="授权 agent 运行安全命令")
    common.add_argument("--allowed-command-prefix", action="append", default=[], help="授权命令前缀，可重复，如: 'conda run -n robot_sys python -m pytest'")
    common.add_argument("--allow-file-write", action="store_true", default=False, help="授权 agent 修改项目内文件")
    common.add_argument("--allowed-write-path", action="append", default=[], help="授权写入路径，可重复，默认项目根目录")
    common.add_argument("--show-events", action="store_true", default=False, help="显示 plan/tool/error 等事件")
    common.add_argument("--debug", action="store_true", default=False, help="显示调试级 agent 事件细节")
    common.add_argument(
        "--step-trace",
        choices=["off", "live", "clear"],
        default="off",
        help="流式显示可见执行过程；clear 会在每步结束后清空终端中的过程块",
    )
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
    if getattr(args, "debug", False):
        Config.DEBUG = True
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
    llm = None
    memory = build_memory(args)
    runtime = build_runtime(args)
    history = SlidingWindowDialogueHistory(args.history_rounds)
    print("Tyro Agent CLI. 输入 q / quit / exit 退出。", file=sys.stderr if args.json else sys.stdout)
    while True:
        try:
            prompt = "" if args.json else chat_prompt_label(getattr(args, "language", "auto"))
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
    effective_language = resolve_turn_language(question, getattr(args, "language", "auto"))
    runtime = runtime or build_runtime(args, language_override=effective_language)
    runtime.language = effective_language
    raw_llm = llm or build_llm(effective_language)
    trace_id = str(uuid.uuid4())
    llm = wrap_agent_llm_trace(
        raw_llm,
        metadata={
            "trace_id": trace_id,
            "command": getattr(args, "command", ""),
            "user": getattr(args, "user", ""),
            "project_root": os.path.abspath(getattr(args, "project_root", os.getcwd())),
            "question": question,
            "language": effective_language,
            "planner": getattr(args, "planner", ""),
        },
    )
    owns_memory = memory is None
    memory = build_memory(args) if memory is None else memory

    final_text = ""
    final_data: Dict[str, Any] = {}
    events = []
    step_trace = StepTracePrinter(
        getattr(args, "step_trace", "off"),
        language=effective_language,
        debug=debug_enabled(args),
    )
    try:
        for event in runtime.run_stream(
            user_text=question,
            llm=llm,
            memory_module=memory,
            long_term_memory=memory is not None,
            sliding_window_context=sliding_window_context,
            streaming=False,
            approval_callback=build_approval_callback(args),
        ):
            step_trace.handle(event)
            if args.show_events:
                print(format_event(event, debug=debug_enabled(args), language=effective_language), file=sys.stderr if args.json else sys.stdout)
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


def build_runtime(args, language_override: str = "") -> AgentRuntime:
    return AgentRuntime(
        project_root=os.path.abspath(args.project_root),
        language=language_override or resolve_turn_language("", getattr(args, "language", "auto")),
        tool_registry=build_tool_registry(args),
        planner_provider=args.planner,
        max_react_iterations=args.max_react_iterations,
        tool_authorization=build_tool_authorization(args),
    )


def build_tool_authorization(args) -> AgentToolAuthorization:
    prefixes = []
    for item in getattr(args, "allowed_command_prefix", []) or []:
        try:
            parts = shlex.split(item)
        except ValueError:
            parts = []
        if parts:
            prefixes.append(parts)
    write_paths = [path for path in (getattr(args, "allowed_write_path", []) or []) if path]
    return AgentToolAuthorization(
        allow_shell_commands=bool(getattr(args, "allow_shell_command", False)),
        allowed_command_prefixes=prefixes,
        allow_file_writes=bool(getattr(args, "allow_file_write", False)),
        allowed_write_paths=write_paths or ["."],
    )


def debug_enabled(args) -> bool:
    return bool(getattr(args, "debug", False) or Config.DEBUG)


def resolve_turn_language(user_text: str, requested_language: str = "auto") -> str:
    requested = (requested_language or "auto").strip().lower()
    if requested in {"zh", "en"}:
        return requested
    return detect_user_language(user_text)


def chat_prompt_label(requested_language: str = "auto") -> str:
    requested = (requested_language or "auto").strip().lower()
    if requested == "en":
        return "You: "
    return "你: "


def detect_user_language(text: str) -> str:
    cjk = sum(1 for char in text or "" if "\u4e00" <= char <= "\u9fff")
    letters = sum(1 for char in text or "" if ("a" <= char.lower() <= "z"))
    if cjk:
        return "zh"
    if letters:
        return "en"
    return "zh"


def build_approval_callback(args):
    if getattr(args, "json", False):
        return None

    def approve(request: ToolApprovalRequest) -> bool:
        print(format_approval_request(request), file=sys.stderr)
        try:
            answer = input("Approve this one operation? [y/N]: ").strip().lower()
        except EOFError:
            return False
        return answer in {"y", "yes", "是", "批准", "同意"}

    return approve


def format_approval_request(request: ToolApprovalRequest) -> str:
    args_preview = json.dumps(request.args, ensure_ascii=False, default=str)
    if len(args_preview) > 800:
        args_preview = args_preview[:800].rstrip() + "..."
    return (
        "\n[approval requested]\n"
        f"tool: {request.tool}\n"
        f"risk: {request.risk_level}\n"
        f"reason: {request.reason}\n"
        f"policy: {request.policy_reason}\n"
        f"args: {args_preview}"
    )


class StepTracePrinter:
    """Displays public execution steps without exposing private model reasoning."""

    TERMINAL_EVENTS = {"tool_result", "approval_result", "final_response"}

    def __init__(self, mode: str = "off", stream=None, language: str = "zh", debug: bool = False):
        self.mode = mode if mode in {"off", "live", "clear"} else "off"
        self.stream = stream or sys.stderr
        self.language = language if language in {"zh", "en"} else "zh"
        self.debug = debug
        self._line_count = 0

    def handle(self, event: AgentEvent) -> None:
        if self.mode == "off":
            return
        text = format_step_trace_event(event, language=self.language, debug=self.debug)
        if not text:
            return
        if self.mode == "clear" and self._line_count and event.type not in self.TERMINAL_EVENTS:
            self.clear()
        print(text, file=self.stream)
        self._line_count += text.count("\n") + 1
        if self.mode == "clear" and event.type in self.TERMINAL_EVENTS:
            self.clear()

    def clear(self) -> None:
        if self._line_count <= 0:
            return
        for _ in range(self._line_count):
            print("\033[1A\033[2K", end="", file=self.stream)
        self.stream.flush()
        self._line_count = 0


def format_step_trace_event(event: AgentEvent, language: str = "en", debug: bool = False) -> str:
    data = event.data or {}
    zh = language == "zh"
    if event.type == "plan":
        steps = ", ".join(data.get("steps", []) or []) or "none"
        label = "决策" if zh else "decision"
        next_label = "下一步" if zh else "next"
        return f"[{label}] reason={public_detail(data.get('reason', ''), debug)} {next_label}={steps}"
    if event.type == "tool_start":
        label = "执行" if zh else "act"
        return f"[{label}] {data.get('tool')} reason={public_detail(data.get('reason', ''), debug)}"
    if event.type == "tool_result":
        status = "ok" if data.get("ok") else "failed"
        label = "观察" if zh else "observe"
        return f"[{label}] {data.get('tool')} {status} elapsed_ms={data.get('elapsed_ms')}"
    if event.type == "approval_requested":
        label = "审批" if zh else "approval"
        needs = "需要确认" if zh else "needs approval"
        return f"[{label}] {data.get('tool')} {needs}: {public_detail(data.get('policy_reason', ''), debug)}"
    if event.type == "approval_result":
        label = "审批" if zh else "approval"
        return f"[{label}] {data.get('tool')} approved={data.get('approved')}"
    if event.type == "skill_loaded":
        return f"[skill] {','.join(data.get('skills', []) or [])}"
    if event.type == "error":
        return f"[error] {data.get('stage')} {public_detail(data.get('error'), debug)}"
    if event.type == "llm_start":
        label = "整理回复" if zh else "compose"
        return f"[{label}] context_chars={data.get('context_chars')}"
    return ""


def public_detail(value, debug: bool = False, limit: int = 120) -> str:
    text = str(value or "").replace("\n", " ").strip()
    if debug:
        return text
    for marker in ("Available tools:", "Tool authorization:", "User input:"):
        if marker in text:
            text = text.split(marker, 1)[0].strip()
    if len(text) > limit:
        text = text[:limit].rstrip() + "..."
    return text


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


def format_event(event: AgentEvent, debug: bool = False, language: str = "zh") -> str:
    data = event.data or {}
    def clean(value):
        return public_detail(value, debug)
    if event.type == "plan":
        steps = ",".join(data.get("steps", []) or [])
        skills = ",".join(data.get("skills", []) or [])
        reason = f" reason={clean(data.get('reason', ''))}" if debug else ""
        return f"[plan] planner={data.get('planner')} mode={data.get('mode')} steps={steps} skills={skills}{reason}"
    if event.type == "tool_start":
        return f"[tool:start] {data.get('tool')} reason={clean(data.get('reason', ''))}"
    if event.type == "tool_result":
        return f"[tool:result] {data.get('tool')} ok={data.get('ok')} elapsed_ms={data.get('elapsed_ms')}"
    if event.type == "approval_requested":
        return f"[approval:requested] {data.get('tool')} reason={clean(data.get('policy_reason'))}"
    if event.type == "approval_result":
        return f"[approval:result] {data.get('tool')} approved={data.get('approved')}"
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
