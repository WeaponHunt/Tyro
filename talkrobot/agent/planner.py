"""A small deterministic planner for first-version agent capabilities."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

from talkrobot.agent.tools.builtin import extract_file_path


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
    """Rule-based planner: cheap, predictable, and easy to debug."""

    def plan(self, user_text: str, use_memory: bool = True) -> AgentPlan:
        text = (user_text or "").strip()
        normalized = text.lower()
        steps: List[ToolStep] = []

        if use_memory:
            steps.append(ToolStep("memory_search", {"query": text}, "retrieve user context"))

        if self._asks_time(text, normalized):
            steps.append(ToolStep("current_time", {}, "answer current time/date questions"))

        expression = self._extract_math_expression(text)
        if expression:
            steps.append(ToolStep("calculator", {"expression": expression}, "answer arithmetic questions"))

        file_path = extract_file_path(text)
        if file_path and self._mentions_read(text):
            steps.append(ToolStep("project_file_read", {"path": file_path}, "read requested project file"))
        elif self._mentions_project_search(text, normalized):
            query = self._extract_search_query(text) or text
            steps.append(ToolStep("project_file_search", {"query": query}, "search project files"))

        mode = "tool_assisted" if len(steps) > (1 if use_memory else 0) else "chat"
        return AgentPlan(mode=mode, steps=steps, reason="rule_based")

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
    def _extract_search_query(text: str) -> str:
        patterns = [
            r"(?:搜索|查找|找一下|看看)\s*([A-Za-z0-9_\-\u4e00-\u9fff./]+)",
            r"(?:where is|find|search)\s+([A-Za-z0-9_\-./]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(1)
        return ""

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
