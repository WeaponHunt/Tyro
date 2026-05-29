"""Markdown skill loading for AgentRuntime."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterable, List


@dataclass
class AgentSkill:
    name: str
    description: str = ""
    triggers: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    body: str = ""
    path: str = ""

    def to_context(self, max_chars: int = 3000) -> str:
        body = self.body.strip()
        if len(body) > max_chars:
            body = body[:max_chars].rstrip() + "\n..."
        header = f"技能: {self.name}"
        if self.description:
            header += f"\n说明: {self.description}"
        if self.tools:
            header += f"\n建议工具: {', '.join(self.tools)}"
        return f"{header}\n{body}".strip()


class SkillRegistry:
    """Loads SKILL.md files and matches them by triggers/planned tools."""

    def __init__(self, skills_dir: str | Iterable[str]):
        if isinstance(skills_dir, str):
            dirs = [skills_dir]
        else:
            dirs = list(skills_dir)
        self.skill_dirs = [os.path.abspath(path) for path in dirs if path]
        self._skills: List[AgentSkill] = []
        self.reload()

    @property
    def skills(self) -> List[AgentSkill]:
        return list(self._skills)

    def reload(self) -> None:
        self._skills = []
        seen_paths = set()
        for skills_dir in self.skill_dirs:
            if not os.path.isdir(skills_dir):
                continue

            for root, _, files in os.walk(skills_dir):
                for filename in files:
                    if filename.lower() != "skill.md":
                        continue
                    path = os.path.abspath(os.path.join(root, filename))
                    if path in seen_paths:
                        continue
                    seen_paths.add(path)
                    skill = self._load_skill(path)
                    if skill is not None:
                        self._skills.append(skill)

    def match(self, user_text: str, planned_tools: Iterable[str], limit: int = 3) -> List[AgentSkill]:
        text = (user_text or "").casefold()
        tool_set = {tool for tool in planned_tools if tool}
        scored = []
        for skill in self._skills:
            score = 0
            for trigger in skill.triggers:
                if trigger and trigger.casefold() in text:
                    score += 2
            if tool_set and set(skill.tools).intersection(tool_set):
                score += 1
            if score > 0:
                scored.append((score, skill.name, skill))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [skill for _, _, skill in scored[:limit]]

    def _load_skill(self, path: str) -> AgentSkill | None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()
        except Exception:
            return None

        meta = {}
        body = raw
        if raw.startswith("---"):
            end = raw.find("\n---", 3)
            if end >= 0:
                meta_text = raw[3:end].strip()
                body = raw[end + 4 :].strip()
                meta = self._parse_front_matter(meta_text)

        name = str(meta.get("name") or os.path.basename(os.path.dirname(path)) or "skill").strip()
        return AgentSkill(
            name=name,
            description=str(meta.get("description") or "").strip(),
            triggers=self._as_list(meta.get("triggers")),
            tools=self._as_list(meta.get("tools")),
            body=body,
            path=path,
        )

    @staticmethod
    def _parse_front_matter(text: str) -> dict:
        meta = {}
        current_key = ""
        for raw_line in text.splitlines():
            line = raw_line.rstrip()
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("- ") and current_key:
                meta.setdefault(current_key, []).append(stripped[2:].strip())
                continue
            if ":" not in stripped:
                continue
            key, value = stripped.split(":", 1)
            current_key = key.strip()
            value = value.strip()
            if value:
                meta[current_key] = value
            else:
                meta[current_key] = []
        return meta

    @staticmethod
    def _as_list(value) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return [part.strip() for part in str(value).split(",") if part.strip()]
