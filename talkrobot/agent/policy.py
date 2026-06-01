"""Tool policy gate for agent runtime tool calls."""
from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from talkrobot.agent.planner import ToolStep


READ_ONLY_TOOLS = {
    "memory_search",
    "current_time",
    "calculator",
    "project_file_search",
    "project_file_list",
    "project_file_read",
    "repo_bootstrap",
    "web_fetch",
}

HIGH_RISK_MARKERS = (
    "write",
    "edit",
    "delete",
    "remove",
    "send",
    "email",
    "push",
    "deploy",
    "migrate",
    "migration",
    "shell",
    "exec",
    "run_command",
    "database",
    "sqlite_query",
)


SAFE_COMMAND_PREFIXES = (
    ("python", "-m", "pytest"),
    ("python", "-m", "compileall"),
    ("conda", "run"),
    ("pytest",),
    ("ruff",),
    ("mypy",),
    ("npm", "test"),
    ("npm", "run"),
    ("pnpm", "test"),
    ("pnpm", "run"),
    ("yarn", "test"),
    ("yarn", "run"),
    ("go", "test"),
    ("cargo", "test"),
)

DANGEROUS_COMMAND_TOKENS = {
    "rm",
    "sudo",
    "su",
    "chmod",
    "chown",
    "mkfs",
    "mount",
    "umount",
    "ssh",
    "scp",
    "rsync",
}

DANGEROUS_COMMAND_SEQUENCES = (
    ("git", "reset"),
    ("git", "clean"),
    ("git", "push"),
    ("git", "checkout"),
    ("git", "restore"),
    ("docker", "run"),
    ("docker", "compose"),
    ("kubectl",),
)


@dataclass
class AgentToolAuthorization:
    """User-granted tool permissions for one runtime."""

    allow_shell_commands: bool = False
    allowed_command_prefixes: List[List[str]] = field(default_factory=list)
    allow_file_writes: bool = False
    allowed_write_paths: List[str] = field(default_factory=lambda: ["."])

    def command_prefixes(self) -> List[List[str]]:
        if self.allowed_command_prefixes:
            return [list(prefix) for prefix in self.allowed_command_prefixes if prefix]
        return [list(prefix) for prefix in SAFE_COMMAND_PREFIXES]


@dataclass
class PolicyDecision:
    allowed: bool
    reason: str = ""
    requires_confirmation: bool = False


@dataclass
class ToolApprovalRequest:
    tool: str
    args: Dict[str, Any]
    reason: str
    policy_reason: str
    risk_level: str = "low"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool,
            "args": self.args,
            "reason": self.reason,
            "policy_reason": self.policy_reason,
            "risk_level": self.risk_level,
        }


class ToolPolicyGate:
    """Central policy gate used before every tool invocation."""

    def __init__(self, authorization: Optional[AgentToolAuthorization] = None, project_root: str = ""):
        self.authorization = authorization or AgentToolAuthorization()
        self.project_root = os.path.abspath(project_root or os.getcwd())

    def assess(
        self,
        step: ToolStep,
        *,
        user_text: str,
        allowed_tools: Optional[Iterable[str]] = None,
        risk_level: str = "low",
    ) -> PolicyDecision:
        del allowed_tools
        tool_name = (step.tool or "").strip()
        if not tool_name:
            return PolicyDecision(False, "empty tool name")

        if tool_name in READ_ONLY_TOOLS:
            return PolicyDecision(True, "read-only tool")

        if tool_name == "memory_write" and self._explicit_memory_authorization(user_text):
            return PolicyDecision(True, "explicit user memory request")

        if tool_name == "shell_command":
            return self._assess_shell(step)

        if tool_name == "file_edit":
            return self._assess_file_write(step)

        if self._is_high_risk_tool(tool_name) or risk_level == "high":
            return PolicyDecision(
                False,
                f"tool {tool_name} requires confirmation before execution",
                requires_confirmation=True,
            )

        return PolicyDecision(True, "default low-risk tool")

    @staticmethod
    def _is_high_risk_tool(tool_name: str) -> bool:
        lowered = tool_name.casefold()
        return any(marker in lowered for marker in HIGH_RISK_MARKERS)

    def _assess_shell(self, step: ToolStep) -> PolicyDecision:
        if not self.authorization.allow_shell_commands:
            return PolicyDecision(False, "shell_command requires user authorization", True)

        argv = _command_argv(step.args.get("command"))
        if not argv:
            return PolicyDecision(False, "empty shell command")
        if _has_shell_metacharacters(step.args.get("command")):
            return PolicyDecision(False, "shell metacharacters are not allowed")
        if _has_dangerous_command(argv):
            return PolicyDecision(False, "dangerous command is not allowed", True)
        if not _matches_prefix(argv, self.authorization.command_prefixes()):
            return PolicyDecision(False, f"command prefix is not authorized: {' '.join(argv[:3])}", True)

        cwd = str(step.args.get("cwd") or ".")
        if not _path_inside_any(self.project_root, cwd, ["."]):
            return PolicyDecision(False, "command cwd must stay inside project root")
        return PolicyDecision(True, "authorized safe shell command")

    def _assess_file_write(self, step: ToolStep) -> PolicyDecision:
        if not self.authorization.allow_file_writes:
            return PolicyDecision(False, "file_edit requires user authorization", True)
        path = str(step.args.get("path") or "")
        if not path:
            return PolicyDecision(False, "file_edit requires path")
        if not _path_inside_any(self.project_root, path, self.authorization.allowed_write_paths):
            return PolicyDecision(False, f"write path is not authorized: {path}", True)
        return PolicyDecision(True, "authorized file edit")

    @staticmethod
    def _explicit_memory_authorization(user_text: str) -> bool:
        text = (user_text or "").casefold()
        return any(marker in text for marker in ("记住", "记一下", "帮我记住", "remember", "save this memory"))


def _command_argv(command) -> List[str]:
    if isinstance(command, list):
        return [str(part) for part in command if str(part)]
    try:
        return shlex.split(str(command or ""))
    except ValueError:
        return []


def _has_shell_metacharacters(command) -> bool:
    if isinstance(command, list):
        return False
    text = str(command or "")
    return any(marker in text for marker in (";", "&&", "||", "|", "`", "$(", ">", "<"))


def _has_dangerous_command(argv: Sequence[str]) -> bool:
    lowered = [item.casefold() for item in argv]
    if any(token in DANGEROUS_COMMAND_TOKENS for token in lowered):
        return True
    for sequence in DANGEROUS_COMMAND_SEQUENCES:
        if len(lowered) >= len(sequence) and tuple(lowered[: len(sequence)]) == sequence:
            return True
    return False


def _matches_prefix(argv: Sequence[str], prefixes: Iterable[Sequence[str]]) -> bool:
    lowered = [item.casefold() for item in argv]
    for prefix in prefixes:
        lowered_prefix = [str(item).casefold() for item in prefix if str(item)]
        if lowered_prefix and lowered[: len(lowered_prefix)] == lowered_prefix:
            return True
    return False


def _path_inside_any(project_root: str, path: str, allowed_roots: Iterable[str]) -> bool:
    root = os.path.abspath(project_root)
    candidate = path if os.path.isabs(path) else os.path.join(root, path)
    resolved = os.path.abspath(candidate)
    if not (resolved == root or resolved.startswith(root + os.sep)):
        return False

    for allowed in allowed_roots or (".",):
        allowed_candidate = allowed if os.path.isabs(str(allowed)) else os.path.join(root, str(allowed))
        allowed_resolved = os.path.abspath(allowed_candidate)
        if resolved == allowed_resolved or resolved.startswith(allowed_resolved + os.sep):
            return True
    return False
