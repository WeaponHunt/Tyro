"""Local MCP dev tools for experiments.

Run through stdio with:
    python -m talkrobot.agent.mcp_servers.dev_tools
"""
from __future__ import annotations

import sqlite3
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import requests
from mcp.server.fastmcp import FastMCP


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MAX_TEXT_CHARS = 12000

mcp = FastMCP("talkrobot-dev-tools")


class _TextHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs) -> None:
        if tag.lower() in {"script", "style", "noscript"}:
            self._skip_depth += 1

    def handle_endtag(self, tag) -> None:
        if tag.lower() in {"script", "style", "noscript"} and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data) -> None:
        if self._skip_depth == 0:
            text = " ".join(str(data).split())
            if text:
                self.parts.append(text)


def _truncate(text: str, max_chars: int = MAX_TEXT_CHARS) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n..."


def _git(args: Iterable[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    if completed.returncode != 0:
        raise RuntimeError(output.strip() or f"git exited with {completed.returncode}")
    return _truncate(output.strip())


def _resolve_project_path(path: str) -> Path:
    raw = (path or ".").strip().strip("`'\"")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    resolved = candidate.resolve()
    if resolved != PROJECT_ROOT and PROJECT_ROOT not in resolved.parents:
        raise ValueError("path is outside project")
    return resolved


def _validate_readonly_sql(query: str) -> str:
    sql = (query or "").strip()
    if not sql:
        raise ValueError("empty SQL query")
    first = sql.split(None, 1)[0].casefold()
    if first not in {"select", "pragma", "explain", "with"}:
        raise ValueError("only read-only SQL is allowed")
    blocked = {"insert", "update", "delete", "drop", "alter", "create", "replace", "attach", "detach", "vacuum"}
    lowered = sql.casefold()
    if any(word in lowered for word in blocked):
        raise ValueError("SQL contains a blocked write keyword")
    return sql


def _quote_sqlite_identifier(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


@mcp.tool()
def git_status() -> str:
    """Return git status --short for the TalkRobot repository."""
    return _git(["status", "--short"])


@mcp.tool()
def git_diff(path: str = "", staged: bool = False) -> str:
    """Return git diff for the repository or one relative path."""
    args = ["diff", "--staged" if staged else "--"]
    if staged:
        args = ["diff", "--staged", "--"]
    if path:
        resolved = _resolve_project_path(path)
        args.append(str(resolved.relative_to(PROJECT_ROOT)))
    return _git(args)


@mcp.tool()
def git_log(limit: int = 5) -> str:
    """Return recent git commits."""
    limit = max(1, min(int(limit or 5), 20))
    return _git(["log", f"--max-count={limit}", "--oneline", "--decorate"])


@mcp.tool()
def sqlite_schema(database_path: str) -> str:
    """Return table and column schema for a sqlite database under the project."""
    db_path = _resolve_project_path(database_path)
    if not db_path.is_file():
        raise ValueError("database_path is not a file")
    rows = []
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        table_rows = conn.execute(
            "SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view') ORDER BY name"
        ).fetchall()
        for name, kind in table_rows:
            rows.append(f"{kind}: {name}")
            for column in conn.execute(f"PRAGMA table_info({_quote_sqlite_identifier(name)})").fetchall():
                rows.append(f"  - {column[1]} {column[2]}")
    return "\n".join(rows)


@mcp.tool()
def sqlite_query(database_path: str, query: str, limit: int = 50) -> str:
    """Run a read-only sqlite query against a database under the project."""
    db_path = _resolve_project_path(database_path)
    if not db_path.is_file():
        raise ValueError("database_path is not a file")
    sql = _validate_readonly_sql(query)
    limit = max(1, min(int(limit or 50), 200))
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql).fetchmany(limit)
    if not rows:
        return ""
    headers = rows[0].keys()
    lines = [" | ".join(headers)]
    for row in rows:
        lines.append(" | ".join(str(row[key]) for key in headers))
    return _truncate("\n".join(lines))


@mcp.tool()
def fetch_url(url: str, max_chars: int = 8000) -> str:
    """Fetch readable text from a public http/https URL."""
    parsed = urlparse((url or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("only http/https URLs are supported")
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    text = response.text
    content_type = response.headers.get("content-type", "")
    if "html" in content_type.casefold():
        parser = _TextHTMLParser()
        parser.feed(text)
        text = "\n".join(parser.parts)
    return _truncate(text.strip(), max(1000, min(int(max_chars or 8000), MAX_TEXT_CHARS)))


if __name__ == "__main__":
    mcp.run()
