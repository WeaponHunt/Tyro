"""Low-risk built-in tools used by the first agent runtime."""
from __future__ import annotations

import ast
import operator
import os
import re
from datetime import datetime
from typing import Iterable, List, Optional
from zoneinfo import ZoneInfo

from talkrobot.agent.tools.base import BaseTool, ToolResult


class MemorySearchTool(BaseTool):
    name = "memory_search"
    description = "Searches the user's long-term memory."
    speakable_start = "我查一下相关记忆。"

    def __init__(self, memory_module=None, enabled: bool = True):
        self.memory = memory_module
        self.enabled = bool(enabled and memory_module is not None)

    def run(self, query: str, limit: int = 3) -> ToolResult:
        if not self.enabled:
            return ToolResult(ok=True, content="", data={"skipped": True})
        try:
            content = self.memory.search_memory(query, limit=limit)
            return ToolResult(ok=True, content=content or "")
        except Exception as exc:
            return ToolResult(ok=False, error=str(exc))


class CurrentTimeTool(BaseTool):
    name = "current_time"
    description = "Returns current date and time."
    speakable_start = "我看一下当前时间。"

    def __init__(self, timezone: str = "Asia/Shanghai"):
        self.timezone = timezone

    def run(self) -> ToolResult:
        try:
            now = datetime.now(ZoneInfo(self.timezone))
        except Exception:
            now = datetime.now()
        return ToolResult(
            ok=True,
            content=now.strftime("%Y-%m-%d %H:%M:%S %Z").strip(),
            data={"iso": now.isoformat()},
        )


class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Evaluates a simple arithmetic expression."
    speakable_start = "我算一下。"

    _ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def run(self, expression: str) -> ToolResult:
        try:
            node = ast.parse(expression, mode="eval")
            value = self._eval(node.body)
            return ToolResult(ok=True, content=str(value), data={"expression": expression})
        except Exception as exc:
            return ToolResult(ok=False, error=str(exc))

    def _eval(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in self._ops:
            return self._ops[type(node.op)](self._eval(node.left), self._eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in self._ops:
            return self._ops[type(node.op)](self._eval(node.operand))
        raise ValueError("unsupported expression")


class ProjectFileSearchTool(BaseTool):
    name = "project_file_search"
    description = "Searches project files by keyword."
    speakable_start = "我在项目里找一下。"

    def __init__(self, project_root: str, max_matches: int = 20):
        self.project_root = os.path.abspath(project_root)
        self.max_matches = max(1, int(max_matches))

    def run(self, query: str) -> ToolResult:
        query = (query or "").strip()
        if not query:
            return ToolResult(ok=False, error="empty query")

        matches: List[str] = []
        lowered = query.lower()
        for path in self._iter_files():
            rel = os.path.relpath(path, self.project_root)
            if lowered in rel.lower():
                matches.append(f"{rel}: path match")
            else:
                snippet = self._first_text_match(path, lowered)
                if snippet:
                    matches.append(f"{rel}: {snippet}")
            if len(matches) >= self.max_matches:
                break

        return ToolResult(ok=True, content="\n".join(matches), data={"matches": len(matches)})

    def _iter_files(self) -> Iterable[str]:
        ignored_dirs = {".git", "__pycache__", ".pytest_cache", "mem_db", "logs", "node_modules"}
        for root, dirs, files in os.walk(self.project_root):
            dirs[:] = [d for d in dirs if d not in ignored_dirs and not d.startswith(".venv")]
            for filename in files:
                if filename.endswith((".pyc", ".png", ".jpg", ".jpeg", ".wav", ".mp4", ".bin")):
                    continue
                yield os.path.join(root, filename)

    @staticmethod
    def _first_text_match(path: str, lowered_query: str) -> str:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line_no, line in enumerate(f, start=1):
                    if lowered_query in line.lower():
                        return f"L{line_no} {line.strip()[:120]}"
        except Exception:
            return ""
        return ""


class ProjectFileReadTool(BaseTool):
    name = "project_file_read"
    description = "Reads a small project text file."
    speakable_start = "我打开文件看一下。"

    def __init__(self, project_root: str, max_chars: int = 6000):
        self.project_root = os.path.abspath(project_root)
        self.max_chars = max(1000, int(max_chars))

    def run(self, path: str) -> ToolResult:
        resolved = self._resolve(path)
        if resolved is None:
            return ToolResult(ok=False, error="path is outside project or not a file")
        try:
            with open(resolved, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read(self.max_chars)
            rel = os.path.relpath(resolved, self.project_root)
            return ToolResult(ok=True, content=f"{rel}\n{content}", data={"path": rel})
        except Exception as exc:
            return ToolResult(ok=False, error=str(exc))

    def _resolve(self, path: str) -> Optional[str]:
        raw = (path or "").strip().strip("`'\"")
        if not raw:
            return None
        candidate = raw if os.path.isabs(raw) else os.path.join(self.project_root, raw)
        resolved = os.path.abspath(candidate)
        if not resolved.startswith(self.project_root + os.sep) and resolved != self.project_root:
            return None
        if not os.path.isfile(resolved):
            return None
        return resolved


def extract_file_path(text: str) -> str:
    match = re.search(r"([\w./-]+\.(?:py|md|txt|json|toml|yaml|yml|css|js|html))", text or "")
    return match.group(1) if match else ""
