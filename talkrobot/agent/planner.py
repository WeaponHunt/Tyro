"""Minimal ReAct planner for choosing one tool call or a final answer."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

try:
    from loguru import logger
except Exception:
    import logging

    logger = logging.getLogger(__name__)

from talkrobot.agent.tools.builtin import extract_file_path, extract_url


@dataclass
class ToolStep:
    tool: str
    args: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""


@dataclass
class AgentPlan:
    mode: str
    steps: List[ToolStep] = field(default_factory=list)
    reason: str = ""


@dataclass
class ReactDecision:
    done: bool = False
    answer: str = ""
    step: Optional[ToolStep] = None
    reason: str = ""


class AgentPlanner:
    """One-step ReAct planner.

    Each call receives the original user task and only the previous tool
    observation. It returns either one tool call or a final answer.
    """

    def __init__(self, provider: str = "llm"):
        provider = (provider or "llm").strip().lower()
        self.provider = provider if provider in {"llm", "rule"} else "llm"

    def decide(
        self,
        user_text: str,
        tools: Iterable[object] = (),
        llm=None,
        previous_observation: Optional[Dict[str, Any]] = None,
        authorization_context: str = "",
    ) -> ReactDecision:
        tools = list(tools or ())
        if self.provider == "rule":
            return self._decide_rule(user_text, tools, previous_observation)

        decision = self._decide_llm(
            user_text=user_text,
            tools=tools,
            llm=llm,
            previous_observation=previous_observation,
            authorization_context=authorization_context,
        )
        if decision is not None:
            return decision
        return self._decide_rule(user_text, tools, previous_observation)

    def plan(
        self,
        user_text: str,
        use_memory: bool = True,
        tools: Iterable[object] = (),
        llm=None,
        observations: str = "",
        completed_steps: Iterable[str] = (),
        authorization_context: str = "",
    ) -> AgentPlan:
        del use_memory, observations, completed_steps
        decision = self.decide(
            user_text,
            tools=tools,
            llm=llm,
            previous_observation=None,
            authorization_context=authorization_context,
        )
        steps = [decision.step] if decision.step is not None else []
        return AgentPlan(mode="tool_assisted" if steps else "chat", steps=steps, reason=decision.reason)

    def _decide_llm(
        self,
        *,
        user_text: str,
        tools: Iterable[object],
        llm,
        previous_observation: Optional[Dict[str, Any]],
        authorization_context: str = "",
    ) -> ReactDecision | None:
        if llm is None or not hasattr(llm, "generate_response"):
            return None

        tool_map = {getattr(tool, "name", ""): tool for tool in tools if getattr(tool, "name", "")}
        try:
            raw = llm.generate_response(
                self._build_react_prompt(user_text, tool_map.values(), previous_observation, authorization_context),
                context="",
                system_prompt_override=self._react_system_prompt(),
            )
            payload = _extract_json_object(raw)
            return self._payload_to_decision(payload, tool_map)
        except Exception as exc:
            logger.warning(f"LLM ReAct decision failed, falling back to rule planner: {exc}")
            return None

    def _decide_rule(
        self,
        user_text: str,
        tools: Iterable[object],
        previous_observation: Optional[Dict[str, Any]],
    ) -> ReactDecision:
        if previous_observation is not None:
            tool = previous_observation.get("tool") or "tool"
            ok = bool(previous_observation.get("ok"))
            content = previous_observation.get("content") or previous_observation.get("error") or ""
            status = "完成" if ok else "失败"
            answer = f"{tool} 执行{status}。"
            if content:
                answer += f"\n{content}"
            return ReactDecision(done=True, answer=answer, reason="rule_observation_final")

        step = self._rule_step(user_text, tools)
        if step is not None:
            return ReactDecision(done=False, step=step, reason="rule_tool")
        return ReactDecision(done=True, answer="", reason="rule_no_tool")

    def _payload_to_decision(self, payload: Dict[str, Any], tool_map: Dict[str, object]) -> ReactDecision:
        answer = str(payload.get("answer") or "")
        reason = str(payload.get("reason") or "llm")
        tool_name = str(payload.get("tool") or payload.get("name") or "").strip()
        if tool_name:
            if tool_name not in tool_map:
                raise ValueError(f"unknown tool: {tool_name}")
            args = payload.get("args")
            if args is None:
                args = payload.get("arguments")
            args = args or {}
            if not isinstance(args, dict):
                args = {}
            return ReactDecision(done=False, step=ToolStep(tool_name, args, reason), reason=reason)

        if bool(payload.get("done", False)):
            return ReactDecision(done=True, answer=answer, reason=reason)
        return ReactDecision(done=True, answer=answer, reason=reason or "llm_no_tool")

    @staticmethod
    def _payload_to_single_tool_step(
        payload: Dict[str, Any],
        tool_map: Dict[str, object],
        *,
        reason: str,
    ) -> ToolStep | None:
        tool_name = str(payload.get("tool") or payload.get("name") or "").strip()
        if not tool_name or tool_name not in tool_map:
            return None
        args = payload.get("args")
        if args is None:
            args = payload.get("arguments")
        if args is None:
            args = {}
        if not isinstance(args, dict):
            args = {}
        return ToolStep(tool_name, args, str(payload.get("reason") or reason))

    def _rule_step(self, user_text: str, tools: Iterable[object]) -> ToolStep | None:
        del tools
        text = (user_text or "").strip()
        normalized = text.lower()

        if any(word in text for word in ("几点", "时间", "日期", "今天几号", "星期几")) or any(
            word in normalized for word in ("what time", "date today", "current time")
        ):
            return ToolStep("current_time", {}, "answer current time/date question")

        expression = _extract_math_expression(text)
        if expression:
            return ToolStep("calculator", {"expression": expression}, "answer arithmetic question")

        url = extract_url(text)
        if url:
            return ToolStep("web_fetch", {"url": url}, "read requested web page")

        file_path = extract_file_path(text)
        if file_path and any(word in text for word in ("读", "打开", "查看", "看看", "read", "open", "show")):
            return ToolStep("project_file_read", {"path": file_path}, "read requested project file")

        if any(word in text for word in ("列出", "目录结构", "有哪些文件", "文件列表", "模块列表")) or any(
            word in normalized for word in ("list files", "directory tree", "what files", "show files")
        ):
            return ToolStep("project_file_list", {"path": "."}, "list project files")

        if any(word in text for word in ("项目", "代码", "文件", "搜索", "查找", "在哪", "哪里定义")) or any(
            word in normalized for word in ("project", "code", "file", "search", "find", "where is")
        ):
            return ToolStep("project_file_search", {"query": _extract_search_query(text) or text}, "search project files")

        return None

    @staticmethod
    def _react_system_prompt() -> str:
        return (
            "You are Tyro's minimal ReAct controller. Return exactly one JSON object and no markdown.\n"
            "Schema: {\"done\":true|false,\"answer\":\"...\",\"tool\":\"tool_name_or_empty\",\"args\":{},\"reason\":\"...\"}.\n"
            "If the task is complete, set done=true and put the user-facing reply in answer.\n"
            "If more work is needed, set done=false and choose exactly one tool from the list.\n"
            "Use the same language as the user for answer. Do not invent tools."
        )

    @staticmethod
    def _build_react_prompt(
        user_text: str,
        tools: Iterable[object],
        previous_observation: Optional[Dict[str, Any]],
        authorization_context: str = "",
    ) -> str:
        tool_lines = []
        for tool in tools:
            name = getattr(tool, "name", "")
            if not name:
                continue
            description = getattr(tool, "description", "") or ""
            tool_lines.append(f"- {name}: {description}")

        prompt = [
            f"User task:\n{user_text}",
            "Previous step:\n" + _format_previous_observation(previous_observation),
            "Available tools:\n" + "\n".join(tool_lines),
        ]
        if authorization_context:
            prompt.append("Tool authorization:\n" + authorization_context)
        return "\n\n".join(prompt)


def _format_previous_observation(previous_observation: Optional[Dict[str, Any]]) -> str:
    if not previous_observation:
        return "(none)"
    content = str(previous_observation.get("content") or "")
    if len(content) > 4000:
        content = content[:4000].rstrip() + "\n..."
    error = str(previous_observation.get("error") or "")
    return json.dumps(
        {
            "tool": previous_observation.get("tool"),
            "args": previous_observation.get("args"),
            "ok": previous_observation.get("ok"),
            "content": content,
            "error": error,
        },
        ensure_ascii=False,
        default=str,
    )


def _extract_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(text[start : end + 1])

    if not isinstance(payload, dict):
        raise ValueError("planner response must be a JSON object")
    return payload


def _extract_search_query(text: str) -> str:
    symbol_match = re.search(r"\b[A-Z][A-Za-z0-9_]{2,}\b", text or "")
    if symbol_match:
        return symbol_match.group(0)

    patterns = [
        r"(?:搜索|查找|找一下|看看)\s*(?:项目里|代码里|文件里)?\s*([A-Za-z0-9_\-\u4e00-\u9fff./]+)\s*(?:在哪|哪里|如何|怎么|的定义|定义)?",
        r"([A-Za-z_][A-Za-z0-9_]{2,})\s*(?:在哪|哪里|如何|怎么|的定义|定义)",
        r"(?:搜索|查找|找一下|看看)\s*([A-Za-z0-9_\-\u4e00-\u9fff./]+)",
        r"(?:where is|find|search)\s+([A-Za-z0-9_\-./]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return ""


def _extract_math_expression(text: str) -> str:
    if not any(op in text for op in ("+", "-", "*", "/", "×", "÷")):
        return ""
    candidate = text.replace("×", "*").replace("÷", "/")
    matches = re.findall(r"[0-9][0-9+\-*/().\s]+[0-9)]", candidate)
    if not matches:
        return ""
    expression = max(matches, key=len).strip()
    if len(expression) > 120:
        return ""
    return expression
