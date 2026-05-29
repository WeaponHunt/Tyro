"""Low-risk built-in tools used by the first agent runtime."""
from __future__ import annotations

import ast
import operator
import os
import re
from html.parser import HTMLParser
from datetime import datetime
from urllib.parse import urlparse
from typing import Iterable, List, Optional
from zoneinfo import ZoneInfo

from talkrobot.agent.tools.base import BaseTool, ToolResult


IGNORED_DIRS = {".git", "__pycache__", ".pytest_cache", "mem_db", "logs", "node_modules"}
IGNORED_BINARY_SUFFIXES = (".pyc", ".png", ".jpg", ".jpeg", ".wav", ".mp4", ".bin")


def _resolve_project_path(
    project_root: str,
    path: str,
    *,
    require_file: bool = False,
    require_dir: bool = False,
) -> Optional[str]:
    raw = (path or ".").strip().strip("`'\"")
    if require_file and not raw:
        return None

    candidate = raw if os.path.isabs(raw) else os.path.join(project_root, raw)
    resolved = os.path.abspath(candidate)
    if not resolved.startswith(project_root + os.sep) and resolved != project_root:
        return None
    if require_file and not os.path.isfile(resolved):
        return None
    if require_dir and not os.path.isdir(resolved):
        return None
    return resolved


class MemorySearchTool(BaseTool):
    name = "memory_search"
    description = "Searches the user's long-term memory."
    speakable_start = "我查一下相关记忆。"
    context_label = "检索到的相关记忆"

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


class MemoryWriteTool(BaseTool):
    name = "memory_write"
    description = "Writes explicit stable user memory."
    speakable_start = "我帮你记下来。"
    context_label = "记忆写入工具结果"

    def __init__(self, memory_module=None, enabled: bool = True):
        self.memory = memory_module
        self.enabled = bool(enabled and memory_module is not None)

    def run(self, content: str) -> ToolResult:
        content = (content or "").strip()
        if not self.enabled:
            return ToolResult(ok=False, error="memory unavailable")
        if not content:
            return ToolResult(ok=False, error="empty memory")
        try:
            if hasattr(self.memory, "add_user_memory_if_stable"):
                saved = self.memory.add_user_memory_if_stable(content, async_mode=True)
            else:
                self.memory.add_memory(content, async_mode=True)
                saved = True
            if not saved:
                self.memory.add_memory(f"用户明确要求记住: {content}", async_mode=True)
            return ToolResult(ok=True, content=f"已提交记忆: {content}", data={"saved": True})
        except Exception as exc:
            return ToolResult(ok=False, error=str(exc))


class CurrentTimeTool(BaseTool):
    name = "current_time"
    description = "Returns current date and time."
    speakable_start = "我看一下当前时间。"
    context_label = "当前时间工具结果"

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
    context_label = "计算工具结果"

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
    context_label = "项目文件搜索结果"

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
        for root, dirs, files in os.walk(self.project_root):
            dirs[:] = [d for d in dirs if d not in IGNORED_DIRS and not d.startswith(".venv")]
            for filename in files:
                if filename.endswith(IGNORED_BINARY_SUFFIXES):
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


class ProjectFileListTool(BaseTool):
    name = "project_file_list"
    description = "Lists project files under a directory."
    speakable_start = "我列一下项目文件。"
    context_label = "项目文件列表"

    def __init__(self, project_root: str, max_entries: int = 80):
        self.project_root = os.path.abspath(project_root)
        self.max_entries = max(10, int(max_entries))

    def run(self, path: str = ".") -> ToolResult:
        resolved = _resolve_project_path(self.project_root, path, require_dir=True)
        if resolved is None:
            return ToolResult(ok=False, error="path is outside project or not a directory")

        entries = []
        for root, dirs, files in os.walk(resolved):
            dirs[:] = [d for d in dirs if d not in IGNORED_DIRS and not d.startswith(".venv")]
            depth = os.path.relpath(root, resolved).count(os.sep)
            if depth > 1:
                dirs[:] = []
                continue
            for dirname in dirs:
                entries.append(os.path.relpath(os.path.join(root, dirname), self.project_root) + "/")
            for filename in files:
                if filename.endswith(".pyc"):
                    continue
                entries.append(os.path.relpath(os.path.join(root, filename), self.project_root))
            if len(entries) >= self.max_entries:
                break

        entries = entries[: self.max_entries]
        return ToolResult(ok=True, content="\n".join(entries), data={"entries": len(entries)})


class ProjectFileReadTool(BaseTool):
    name = "project_file_read"
    description = "Reads a small project text file."
    speakable_start = "我打开文件看一下。"
    context_label = "项目文件内容"

    def __init__(self, project_root: str, max_chars: int = 6000):
        self.project_root = os.path.abspath(project_root)
        self.max_chars = max(1000, int(max_chars))

    def run(self, path: str) -> ToolResult:
        resolved = _resolve_project_path(self.project_root, path, require_file=True)
        if resolved is None:
            return ToolResult(ok=False, error="path is outside project or not a file")
        try:
            with open(resolved, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read(self.max_chars)
            rel = os.path.relpath(resolved, self.project_root)
            return ToolResult(ok=True, content=f"{rel}\n{content}", data={"path": rel})
        except Exception as exc:
            return ToolResult(ok=False, error=str(exc))


def extract_file_path(text: str) -> str:
    match = re.search(r"([\w./-]+\.(?:py|md|txt|json|toml|yaml|yml|css|js|html))", text or "")
    return match.group(1) if match else ""


class _TextHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: List[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag.lower() in {"script", "style", "noscript"}:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag.lower() in {"script", "style", "noscript"} and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth == 0:
            text = re.sub(r"\s+", " ", data).strip()
            if text:
                self.parts.append(text)


class WebFetchTool(BaseTool):
    name = "web_fetch"
    description = "Fetches text from a public http/https URL."
    speakable_start = "我打开网页看一下。"
    context_label = "网页读取结果"

    def __init__(self, max_chars: int = 6000):
        self.max_chars = max(1000, int(max_chars))

    def run(self, url: str) -> ToolResult:
        url = (url or "").strip()
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return ToolResult(ok=False, error="only http/https urls are supported")

        try:
            import requests

            response = requests.get(url, timeout=8)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            text = response.text
            if "html" in content_type.lower():
                parser = _TextHTMLParser()
                parser.feed(text)
                text = "\n".join(parser.parts)
            text = text.strip()
            if len(text) > self.max_chars:
                text = text[: self.max_chars].rstrip() + "\n..."
            return ToolResult(ok=True, content=text, data={"url": url, "content_type": content_type})
        except Exception as exc:
            return ToolResult(ok=False, error=str(exc))


def extract_url(text: str) -> str:
    match = re.search(r"https?://[^\s，。！？)）]+", text or "")
    return match.group(0) if match else ""
