"""Event-driven first-version agent runtime."""
from __future__ import annotations

import os
import json
import time
from typing import Iterable, Iterator, Optional

from talkrobot.agent.events import AgentEvent
from talkrobot.agent.planner import AgentPlanner
from talkrobot.agent.skills import SkillRegistry
from talkrobot.agent.tools import RuntimeToolContext, ToolRegistry
from talkrobot.config import Config


class AgentRuntime:
    """Plans low-risk tool calls, executes them, then asks the LLM to respond."""

    def __init__(
        self,
        project_root: Optional[str] = None,
        language: str = "zh",
        tool_registry: Optional[ToolRegistry] = None,
        skill_dirs: Optional[Iterable[str]] = None,
        planner_provider: Optional[str] = None,
        max_react_iterations: Optional[int] = None,
    ):
        self.project_root = os.path.abspath(project_root or os.getcwd())
        self.language = (language or "zh").strip().lower()
        self.planner = AgentPlanner(planner_provider or Config.AGENT_PLANNER_PROVIDER)
        self.tool_registry = tool_registry or ToolRegistry.default(self.project_root)
        self.skills = SkillRegistry(self._resolve_skill_dirs(skill_dirs))
        self.max_react_iterations = max(1, int(max_react_iterations or Config.AGENT_REACT_MAX_ITERATIONS))

    @classmethod
    def with_mcp_servers(
        cls,
        mcp_servers: Iterable[object],
        project_root: Optional[str] = None,
        **kwargs,
    ) -> "AgentRuntime":
        """Build an AgentRuntime with built-ins plus explicit MCP servers."""
        root = os.path.abspath(project_root or os.getcwd())
        return cls(
            project_root=root,
            tool_registry=ToolRegistry.with_mcp_servers(mcp_servers),
            **kwargs,
        )

    @property
    def _is_english(self) -> bool:
        return self.language == "en"

    def _msg(self, zh: str, en: str) -> str:
        return en if self._is_english else zh

    def run_stream(
        self,
        user_text: str,
        llm,
        memory_module=None,
        long_term_memory: bool = False,
        sliding_window_context: str = "",
        switch_notice: str = "",
        system_prompt_override: str = "",
        streaming: bool = False,
    ) -> Iterator[AgentEvent]:
        turn_start = time.perf_counter()
        tools = self.tool_registry.build_tools(
            RuntimeToolContext(
                project_root=self.project_root,
                memory_module=memory_module,
                long_term_memory=long_term_memory,
                language=self.language,
            )
        )
        tool_context_parts = []
        matched_skills = []
        matched_skill_names = set()
        tool_results = []
        observations = []
        completed_step_signatures = set()
        react_iterations = self.max_react_iterations if self.planner.provider == "llm" else 1

        for iteration in range(1, react_iterations + 1):
            plan = self.planner.plan(
                user_text,
                use_memory=long_term_memory and memory_module is not None,
                tools=tools.values(),
                llm=llm,
                observations=self._format_observations(observations),
                completed_steps=sorted(completed_step_signatures),
            )
            planned_tools = [step.tool for step in plan.steps]
            new_skills = [
                skill
                for skill in self.skills.match(user_text, planned_tools)
                if skill.name not in matched_skill_names
            ]
            if new_skills:
                matched_skills.extend(new_skills)
                matched_skill_names.update(skill.name for skill in new_skills)

            skill_steps = self.planner.plan_skill_tools(
                user_text,
                tools.values(),
                new_skills,
                planned_tools + list(completed_step_signatures),
            )
            if skill_steps:
                plan.steps.extend(skill_steps)
                planned_tools = [step.tool for step in plan.steps]
                plan.mode = "tool_assisted"

            executable_steps = [
                step
                for step in plan.steps
                if step.tool in tools and self._step_signature(step) not in completed_step_signatures
            ]
            yield AgentEvent(
                type="plan",
                text=self._msg("我先规划一下。", "I'll plan this out first."),
                speakable=False,
                data={
                    "mode": plan.mode,
                    "planner": self.planner.provider,
                    "reason": plan.reason,
                    "iteration": iteration,
                    "steps": [step.tool for step in executable_steps],
                    "skills": [skill.name for skill in matched_skills],
                    "observations": len(observations),
                },
            )

            if new_skills:
                skill_context = "\n\n".join(skill.to_context() for skill in new_skills)
                tool_context_parts.append(f"已加载技能说明:\n{skill_context}")
                yield AgentEvent(
                    type="skill_loaded",
                    text=self._msg("我加载了相关技能。", "I loaded a relevant skill."),
                    speakable=False,
                    data={"skills": [skill.name for skill in new_skills]},
                )

            if not executable_steps:
                break

            any_retryable_failure = False
            for step in executable_steps:
                signature = self._step_signature(step)
                completed_step_signatures.add(signature)
                tool = tools.get(step.tool)
                if tool is None:
                    continue

                yield AgentEvent(
                    type="tool_start",
                    text=tool.speakable_start,
                    speakable=step.tool != "memory_search",
                    data={"tool": step.tool, "reason": step.reason, "iteration": iteration},
                )
                started = time.perf_counter()
                result = tool.run(**step.args)
                elapsed_ms = round((time.perf_counter() - started) * 1000)
                retryable = (not result.ok) and self._is_retryable_tool_error(result.error)
                any_retryable_failure = any_retryable_failure or retryable
                observation = {
                    "iteration": iteration,
                    "tool": step.tool,
                    "args": step.args,
                    "ok": result.ok,
                    "content": result.content,
                    "error": result.error,
                    "retryable": retryable,
                }
                observations.append(observation)
                tool_results.append(
                    {
                        "tool": step.tool,
                        "ok": result.ok,
                        "elapsed_ms": elapsed_ms,
                        "error": result.error,
                        "retryable": retryable,
                    }
                )

                if result.ok and result.content:
                    label = self._tool_label(step.tool, tool)
                    tool_context_parts.append(f"{label}:\n{result.content}")
                elif not result.ok:
                    yield AgentEvent.error(
                        self._msg("有个工具执行失败了，我会先用已有信息继续。", "A tool failed, I'll continue with what I have."),
                        stage=step.tool,
                        error=result.error,
                        elapsed_ms=elapsed_ms,
                        retryable=retryable,
                    )

                yield AgentEvent(
                    type="tool_result",
                    text=self._tool_result_text(step.tool, result.ok, result.content),
                    speakable=False,
                    data={
                        "tool": step.tool,
                        "ok": result.ok,
                        "elapsed_ms": elapsed_ms,
                        "iteration": iteration,
                        "retryable": retryable,
                    },
                )

            if self.planner.provider != "llm":
                break
            if iteration >= react_iterations:
                break
            if plan.mode == "chat":
                break
            if not any_retryable_failure and self._all_recent_steps_empty_or_terminal(executable_steps, observations):
                break

        if observations:
            tool_context_parts.append(
                "工具执行观察:\n" + self._format_observations(observations, include_content=True)
            )

        context = self._merge_context(switch_notice, tool_context_parts, sliding_window_context)
        yield AgentEvent(
            type="llm_start",
            text=self._msg("我开始整理回复。", "I'll compose the reply now."),
            speakable=False,
            data={"context_chars": len(context)},
        )

        llm_start = time.perf_counter()
        if streaming and hasattr(llm, "generate_response_stream"):
            parts = []
            for chunk in llm.generate_response_stream(
                user_text,
                context,
                system_prompt_override=system_prompt_override,
            ):
                parts.append(chunk)
                yield AgentEvent(type="llm_chunk", text=chunk, speakable=False)
            raw_response = "".join(parts)
        else:
            raw_response = llm.generate_response(
                user_text,
                context,
                system_prompt_override=system_prompt_override,
            )

        yield AgentEvent(
            type="final_response",
            text=raw_response,
            speakable=True,
            data={
                "context": context,
                "tool_results": tool_results,
                "observations": observations,
                "react_iterations": len({item["iteration"] for item in observations}),
                "llm_elapsed_ms": round((time.perf_counter() - llm_start) * 1000),
                "total_elapsed_ms": round((time.perf_counter() - turn_start) * 1000),
                "used_memory": any(r["tool"] == "memory_search" and r["ok"] for r in tool_results),
                "used_tools": [r["tool"] for r in tool_results if r["tool"] != "memory_search"],
                "used_skills": [skill.name for skill in matched_skills],
            },
        )

    def _merge_context(
        self,
        switch_notice: str,
        tool_context_parts: Iterable[str],
        sliding_window_context: str,
    ) -> str:
        sections = []
        if switch_notice:
            sections.append(switch_notice)
        sections.extend(part for part in tool_context_parts if part)
        if sliding_window_context:
            sections.append(sliding_window_context)
        return "\n\n".join(sections)

    def _tool_label(self, tool_name: str, tool=None) -> str:
        context_label = getattr(tool, "context_label", "")
        if context_label:
            return context_label
        return f"工具结果 {tool_name}"

    def _tool_result_text(self, tool_name: str, ok: bool, content: str) -> str:
        if not ok:
            return self._msg("工具执行失败。", "Tool execution failed.")
        if tool_name == "memory_search":
            return self._msg("记忆检索完成。", "Memory search finished.")
        if content:
            return self._msg("工具执行完成。", "Tool execution finished.")
        return self._msg("工具没有返回额外内容。", "Tool returned no extra content.")

    @staticmethod
    def _step_signature(step) -> str:
        return json.dumps(
            {"tool": step.tool, "args": step.args},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )

    @staticmethod
    def _is_retryable_tool_error(error: str) -> bool:
        text = (error or "").casefold()
        if not text:
            return False
        non_retryable = (
            "unavailable",
            "unsupported",
            "outside project",
            "not a file",
            "not a directory",
            "only http/https",
            "empty memory",
        )
        return not any(marker in text for marker in non_retryable)

    @staticmethod
    def _format_observations(observations, include_content: bool = False) -> str:
        lines = []
        for index, item in enumerate(observations, start=1):
            status = "ok" if item.get("ok") else "failed"
            line = f"{index}. [{status}] {item.get('tool')} args={item.get('args')}"
            if item.get("error"):
                line += f" error={item.get('error')}"
            if include_content and item.get("content"):
                content = str(item.get("content"))
                if len(content) > 1200:
                    content = content[:1200].rstrip() + "\n..."
                line += f"\n{content}"
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _all_recent_steps_empty_or_terminal(steps, observations) -> bool:
        recent = observations[-len(steps):] if steps else []
        if not recent:
            return True
        return all((item.get("ok") and not item.get("content")) or (not item.get("ok") and not item.get("retryable")) for item in recent)

    def _resolve_skill_dirs(self, skill_dirs: Optional[Iterable[str]]) -> Iterable[str]:
        dirs = list(skill_dirs or [])
        dirs.append(os.path.join(os.path.dirname(__file__), "skills"))
        extra_dirs = os.getenv("TALKROBOT_SKILL_DIRS", "")
        for path in extra_dirs.split(os.pathsep):
            path = path.strip()
            if path:
                dirs.append(path)
        return dirs
