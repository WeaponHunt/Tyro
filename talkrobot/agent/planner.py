"""Agent planners for choosing low-risk tool calls."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

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


class AgentPlanner:
    """Planner facade with LLM and deterministic rule modes."""

    def __init__(self, provider: str = "llm"):
        provider = (provider or "llm").strip().lower()
        self.provider = provider if provider in {"llm", "rule"} else "llm"

    def plan(
        self,
        user_text: str,
        use_memory: bool = True,
        tools: Iterable[object] = (),
        llm=None,
        observations: str = "",
        completed_steps: Iterable[str] = (),
    ) -> AgentPlan:
        tools = list(tools or ())
        if self.provider == "rule":
            return self._plan_rule(user_text, use_memory=use_memory, tools=tools)

        plan = self._plan_llm(
            user_text,
            use_memory=use_memory,
            tools=tools,
            llm=llm,
            observations=observations,
            completed_steps=completed_steps,
        )
        if plan is not None:
            return plan
        return self._plan_rule(user_text, use_memory=use_memory, tools=tools)

    def _plan_rule(self, user_text: str, use_memory: bool = True, tools: Iterable[object] = ()) -> AgentPlan:
        text = (user_text or "").strip()
        normalized = text.lower()
        steps: List[ToolStep] = []

        if use_memory:
            steps.append(ToolStep("memory_search", {"query": text}, "retrieve user context"))

        if self._asks_memory_write(text):
            steps.append(ToolStep("memory_write", {"content": text}, "save explicit user memory"))

        if self._asks_time(text, normalized):
            steps.append(ToolStep("current_time", {}, "answer current time/date questions"))

        expression = self._extract_math_expression(text)
        if expression:
            steps.append(ToolStep("calculator", {"expression": expression}, "answer arithmetic questions"))

        url = extract_url(text)
        if url:
            steps.append(ToolStep("web_fetch", {"url": url}, "read requested web page"))

        file_path = extract_file_path(text)
        if file_path and self._mentions_read(text):
            steps.append(ToolStep("project_file_read", {"path": file_path}, "read requested project file"))
        elif self._mentions_project_list(text, normalized):
            path = file_path or self._extract_dir_path(text) or "."
            steps.append(ToolStep("project_file_list", {"path": path}, "list project files"))
        elif self._mentions_project_search(text, normalized):
            query = self._extract_search_query(text) or text
            steps.append(ToolStep("project_file_search", {"query": query}, "search project files"))

        planned_tools = {step.tool for step in steps}
        for tool in tools or ():
            tool_name = getattr(tool, "name", "")
            if not tool_name or tool_name in planned_tools:
                continue
            plan_hook = getattr(tool, "plan", None)
            if plan_hook is None:
                continue
            step = plan_hook(text)
            if step is None or not getattr(step, "tool", ""):
                continue
            if step.tool in planned_tools:
                continue
            steps.append(step)
            planned_tools.add(step.tool)

        mode = "tool_assisted" if len(steps) > (1 if use_memory else 0) else "chat"
        return AgentPlan(mode=mode, steps=steps, reason="rule_based")

    def _plan_llm(
        self,
        user_text: str,
        use_memory: bool,
        tools: Iterable[object],
        llm,
        observations: str = "",
        completed_steps: Iterable[str] = (),
    ) -> AgentPlan | None:
        if llm is None or not hasattr(llm, "generate_response"):
            return None

        tool_map = {getattr(tool, "name", ""): tool for tool in tools if getattr(tool, "name", "")}
        if not tool_map:
            return AgentPlan(mode="chat", steps=[], reason="llm_no_tools")

        try:
            raw = llm.generate_response(
                self._build_planner_user_prompt(
                    user_text,
                    use_memory,
                    tool_map.values(),
                    observations=observations,
                    completed_steps=completed_steps,
                ),
                context="",
                system_prompt_override=self._planner_system_prompt(),
            )
            payload = self._extract_json_object(raw)
            plan = self._payload_to_plan(payload, tool_map)
            plan.reason = str(payload.get("reason") or "llm")
            return plan
        except Exception as exc:
            logger.warning(f"LLM planner failed, falling back to rule planner: {exc}")
            return None

    def _payload_to_plan(self, payload: Dict[str, Any], tool_map: Dict[str, object]) -> AgentPlan:
        raw_steps = payload.get("steps") or []
        if not isinstance(raw_steps, list):
            raw_steps = []

        steps: List[ToolStep] = []
        planned = set()
        for item in raw_steps:
            if not isinstance(item, dict):
                continue
            tool_name = str(item.get("tool") or "").strip()
            if not tool_name or tool_name not in tool_map or tool_name in planned:
                continue
            args = item.get("args") or {}
            if not isinstance(args, dict):
                args = {}
            steps.append(
                ToolStep(
                    tool=tool_name,
                    args=args,
                    reason=str(item.get("reason") or "llm planned tool"),
                )
            )
            planned.add(tool_name)

        mode = str(payload.get("mode") or "").strip().lower()
        if mode not in {"chat", "tool_assisted"}:
            mode = "tool_assisted" if steps else "chat"
        if steps:
            mode = "tool_assisted"
        return AgentPlan(mode=mode, steps=steps, reason="llm")

    @staticmethod
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
            payload = json.loads(text[start:end + 1])

        if not isinstance(payload, dict):
            raise ValueError("planner response must be a JSON object")
        return payload

    @staticmethod
    def _planner_system_prompt() -> str:
        return (
            "You are Tyro's tool planner. Return only one JSON object, no markdown.\n"
            "Choose zero or more safe tools needed before the assistant answers.\n"
            "You may be called repeatedly in a ReAct loop. Use observations to decide the next action.\n"
            "Schema: {\"mode\":\"chat|tool_assisted\",\"reason\":\"...\",\"steps\":[{\"tool\":\"name\",\"args\":{},\"reason\":\"...\"}]}.\n"
            "Use only tools from the provided tool list. Do not invent tools.\n"
            "Do not repeat completed steps unless the previous observation shows retryable bad arguments.\n"
            "For memory_search use args {\"query\": user text}. For memory_write use {\"content\": stable memory text}.\n"
            "For current_time use {}. For calculator use {\"expression\":\"...\"}.\n"
            "For project_file_search use {\"query\":\"...\"}; project_file_list use {\"path\":\".\"}; project_file_read use {\"path\":\"relative/path\"}.\n"
            "For web_fetch use {\"url\":\"https://...\"}.\n"
            "If observations are enough to answer, return mode chat and an empty steps array."
        )

    @staticmethod
    def _build_planner_user_prompt(
        user_text: str,
        use_memory: bool,
        tools: Iterable[object],
        observations: str = "",
        completed_steps: Iterable[str] = (),
    ) -> str:
        tool_lines = []
        for tool in tools:
            name = getattr(tool, "name", "")
            if not name:
                continue
            description = getattr(tool, "description", "") or ""
            tool_lines.append(f"- {name}: {description}")

        memory_hint = (
            "Long-term memory is available; include memory_search when user context or preferences may help."
            if use_memory
            else "Long-term memory is unavailable; do not choose memory_search or memory_write."
        )
        completed = "\n".join(f"- {step}" for step in completed_steps if step)
        prompt = (
            f"User input:\n{user_text}\n\n"
            f"{memory_hint}\n\n"
            "Available tools:\n"
            + "\n".join(tool_lines)
        )
        if completed:
            prompt += "\n\nCompleted steps:\n" + completed
        if observations:
            prompt += "\n\nObservations:\n" + observations
        return prompt

    def plan_skill_tools(
        self,
        user_text: str,
        tools: Iterable[object],
        skills: Iterable[object],
        planned_tools: Iterable[str] = (),
    ) -> List[ToolStep]:
        tool_map = {getattr(tool, "name", ""): tool for tool in tools if getattr(tool, "name", "")}
        planned = {name for name in planned_tools if name}
        steps: List[ToolStep] = []

        for skill in skills or ():
            for tool_name in getattr(skill, "tools", []) or []:
                if tool_name in planned:
                    continue
                tool = tool_map.get(tool_name)
                if tool is None:
                    continue
                plan_hook = getattr(tool, "plan_from_skill", None)
                if plan_hook is None:
                    continue
                step = plan_hook(user_text, skill)
                if step is None or not getattr(step, "tool", ""):
                    continue
                if step.tool in planned:
                    continue
                steps.append(step)
                planned.add(step.tool)
        return steps

    @staticmethod
    def _asks_time(text: str, normalized: str) -> bool:
        zh = any(word in text for word in ("几点", "时间", "日期", "今天几号", "星期几"))
        en = any(word in normalized for word in ("what time", "date today", "current time"))
        return zh or en

    @staticmethod
    def _mentions_read(text: str) -> bool:
        return any(word in text for word in ("读", "打开", "查看", "看看", "read", "open", "show"))

    @staticmethod
    def _mentions_project_search(text: str, normalized: str) -> bool:
        zh = any(word in text for word in ("项目", "代码", "文件", "搜索", "查找", "在哪", "哪里定义"))
        en = any(word in normalized for word in ("project", "code", "file", "search", "find", "where is"))
        return zh or en

    @staticmethod
    def _mentions_project_list(text: str, normalized: str) -> bool:
        zh = any(word in text for word in ("列出", "目录结构", "有哪些文件", "文件列表", "模块列表"))
        en = any(word in normalized for word in ("list files", "directory tree", "what files", "show files"))
        return zh or en

    @staticmethod
    def _asks_memory_write(text: str) -> bool:
        return any(word in text for word in ("记住", "记一下", "帮我记住", "以后记得"))

    @staticmethod
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

    @staticmethod
    def _extract_dir_path(text: str) -> str:
        match = re.search(r"([\w./-]+/)", text or "")
        return match.group(1).rstrip("/") if match else ""

    @staticmethod
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
