"""Event-driven first-version agent runtime."""
from __future__ import annotations

import os
import time
from typing import Dict, Iterable, Iterator, Optional

try:
    from loguru import logger
except Exception:
    import logging

    logger = logging.getLogger(__name__)

from talkrobot.agent.events import AgentEvent
from talkrobot.agent.planner import AgentPlan, AgentPlanner
from talkrobot.agent.tools import (
    CalculatorTool,
    CurrentTimeTool,
    MemorySearchTool,
    ProjectFileReadTool,
    ProjectFileSearchTool,
)


class AgentRuntime:
    """Plans low-risk tool calls, executes them, then asks the LLM to respond."""

    def __init__(self, project_root: Optional[str] = None, language: str = "zh"):
        self.project_root = os.path.abspath(project_root or os.getcwd())
        self.language = (language or "zh").strip().lower()
        self.planner = AgentPlanner()

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
        tools = self._build_tools(memory_module, long_term_memory)
        plan = self.planner.plan(user_text, use_memory=long_term_memory and memory_module is not None)
        yield AgentEvent(
            type="plan",
            text=self._msg("我先规划一下。", "I'll plan this out first."),
            speakable=False,
            data={"mode": plan.mode, "steps": [step.tool for step in plan.steps]},
        )

        tool_context_parts = []
        tool_results = []
        for step in plan.steps:
            tool = tools.get(step.tool)
            if tool is None:
                continue

            yield AgentEvent(
                type="tool_start",
                text=tool.speakable_start,
                speakable=step.tool != "memory_search",
                data={"tool": step.tool, "reason": step.reason},
            )
            started = time.perf_counter()
            result = tool.run(**step.args)
            elapsed_ms = round((time.perf_counter() - started) * 1000)
            tool_results.append(
                {
                    "tool": step.tool,
                    "ok": result.ok,
                    "elapsed_ms": elapsed_ms,
                    "error": result.error,
                }
            )

            if result.ok and result.content:
                label = self._tool_label(step.tool)
                tool_context_parts.append(f"{label}:\n{result.content}")
            elif not result.ok:
                yield AgentEvent.error(
                    self._msg("有个工具执行失败了，我会先用已有信息继续。", "A tool failed, I'll continue with what I have."),
                    stage=step.tool,
                    error=result.error,
                    elapsed_ms=elapsed_ms,
                )

            yield AgentEvent(
                type="tool_result",
                text=self._tool_result_text(step.tool, result.ok, result.content),
                speakable=False,
                data={"tool": step.tool, "ok": result.ok, "elapsed_ms": elapsed_ms},
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
                "llm_elapsed_ms": round((time.perf_counter() - llm_start) * 1000),
                "total_elapsed_ms": round((time.perf_counter() - turn_start) * 1000),
                "used_memory": any(part.startswith("检索到的相关记忆") for part in tool_context_parts),
                "used_tools": [r["tool"] for r in tool_results if r["tool"] != "memory_search"],
            },
        )

    def _build_tools(self, memory_module, long_term_memory: bool) -> Dict[str, object]:
        return {
            "memory_search": MemorySearchTool(memory_module, enabled=long_term_memory),
            "current_time": CurrentTimeTool(),
            "calculator": CalculatorTool(),
            "project_file_search": ProjectFileSearchTool(self.project_root),
            "project_file_read": ProjectFileReadTool(self.project_root),
        }

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

    def _tool_label(self, tool_name: str) -> str:
        labels = {
            "memory_search": "检索到的相关记忆",
            "current_time": "当前时间工具结果",
            "calculator": "计算工具结果",
            "project_file_search": "项目文件搜索结果",
            "project_file_read": "项目文件内容",
        }
        return labels.get(tool_name, f"工具结果 {tool_name}")

    def _tool_result_text(self, tool_name: str, ok: bool, content: str) -> str:
        if not ok:
            return self._msg("工具执行失败。", "Tool execution failed.")
        if tool_name == "memory_search":
            return self._msg("记忆检索完成。", "Memory search finished.")
        if content:
            return self._msg("工具执行完成。", "Tool execution finished.")
        return self._msg("工具没有返回额外内容。", "Tool returned no extra content.")
