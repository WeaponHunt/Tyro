"""Small task state for the minimal ReAct runtime."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TaskState:
    goal: str
    mode: str = "react"
    facts: List[str] = field(default_factory=list)
    attempted_actions: List[str] = field(default_factory=list)
    failed_attempts: List[str] = field(default_factory=list)
    changed_files: List[str] = field(default_factory=list)
    verification_results: List[str] = field(default_factory=list)
    repo_summary: str = ""
    test_commands: List[str] = field(default_factory=list)
    command_results: List[Dict[str, Any]] = field(default_factory=list)

    def note_tool_result(
        self,
        tool: str,
        args: Dict[str, Any],
        ok: bool,
        content: str = "",
        error: str = "",
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        data = data or {}
        signature = f"{tool} {args}".strip()
        if signature:
            self.attempted_actions.append(signature)

        if tool == "repo_bootstrap" and ok:
            self.repo_summary = _compact(content, limit=800)
            self.test_commands = [str(item) for item in data.get("test_commands", []) if item]

        if tool == "shell_command":
            parsed = parse_command_result(data, content, error)
            self.command_results.append(parsed)
            if parsed["is_verification"]:
                self.verification_results.append(parsed["summary"])
            if not ok:
                self.failed_attempts.append(parsed["summary"])

        if ok:
            snippet = _compact(content)
            if snippet:
                self.facts.append(f"{tool}: {snippet}")
            path = _path_from_args_or_content(args, content)
            if path and _looks_write_tool(tool):
                self.changed_files.append(path)
            if _looks_verification(tool, content):
                self.verification_results.append(f"{tool}: {snippet or 'ok'}")
        else:
            self.failed_attempts.append(f"{tool}: {error or 'failed'}")


def parse_command_result(data: Dict[str, Any], content: str = "", error: str = "") -> Dict[str, Any]:
    command = data.get("command") or []
    if isinstance(command, str):
        command_text = command
    else:
        command_text = " ".join(str(part) for part in command)
    exit_code = data.get("exit_code")
    output = "\n".join(str(part or "") for part in (data.get("stdout"), data.get("stderr"), content))
    check_type = _classify_command(command_text)
    summary = _summarize_command(command_text, check_type, exit_code, output, error)
    return {
        "command": command_text,
        "exit_code": exit_code,
        "check_type": check_type,
        "is_verification": check_type in {"pytest", "compile", "lint", "typecheck", "test"},
        "summary": summary,
        "environment_blocked": _looks_environment_blocked(output + "\n" + str(error or "")),
    }


def _classify_command(command: str) -> str:
    lowered = (command or "").casefold()
    if "pytest" in lowered:
        return "pytest"
    if "compileall" in lowered:
        return "compile"
    if "ruff" in lowered:
        return "lint"
    if "mypy" in lowered or "pyright" in lowered or "tsc" in lowered:
        return "typecheck"
    if "test" in lowered:
        return "test"
    return "command"


def _summarize_command(command: str, check_type: str, exit_code, output: str, error: str) -> str:
    status = "passed" if exit_code == 0 else "failed"
    if exit_code is None and error:
        status = "blocked"
    detail = ""
    if check_type == "pytest":
        match = re.search(r"=+\s*(\d+\s+failed.*?)\s*=+", output, flags=re.IGNORECASE | re.DOTALL)
        if match:
            detail = _compact(match.group(1), limit=160)
        elif "passed" in output.casefold():
            match = re.search(r"=+\s*(\d+\s+passed.*?)\s*=+", output, flags=re.IGNORECASE | re.DOTALL)
            detail = _compact(match.group(1), limit=160) if match else ""
    if not detail:
        first_error = re.search(
            r"(error:.*|failed:.*|traceback.*|no module named .*|module not found.*)",
            output,
            flags=re.IGNORECASE,
        )
        detail = _compact(first_error.group(1), limit=160) if first_error else _compact(error, limit=160)
    suffix = f": {detail}" if detail else ""
    return f"{check_type} {status}: {command}{suffix}"


def _looks_environment_blocked(text: str) -> bool:
    lowered = (text or "").casefold()
    markers = (
        "no module named",
        "module not found",
        "permission denied",
        "device not found",
        "connection refused",
        "api key",
        "not installed",
        "command not found",
    )
    return any(marker in lowered for marker in markers)


def _path_from_args_or_content(args: Dict[str, Any], content: str) -> str:
    path = str((args or {}).get("path") or "").strip()
    if path:
        return path
    match = re.search(r"^([^:\n]+): changed\b", content or "")
    return match.group(1).strip() if match else ""


def _looks_write_tool(tool: str) -> bool:
    lowered = (tool or "").casefold()
    return any(marker in lowered for marker in ("write", "edit", "patch", "create", "file_edit"))


def _looks_verification(tool: str, content: str) -> bool:
    text = f"{tool}\n{content}".casefold()
    return any(marker in text for marker in ("pytest", "compileall", "ruff", "mypy", "passed", "tests pass"))


def _compact(text: str, limit: int = 260) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) > limit:
        return text[:limit].rstrip() + "..."
    return text
