"""Small event-driven ReAct runtime."""
from __future__ import annotations

import inspect
import json
import os
import re
import time
from typing import Callable, Dict, Iterable, Iterator, Optional

from talkrobot.agent.events import AgentEvent
from talkrobot.agent.planner import AgentPlanner, ReactDecision, ToolStep
from talkrobot.agent.policy import AgentToolAuthorization, ToolApprovalRequest, ToolPolicyGate
from talkrobot.agent.skills import SkillRegistry
from talkrobot.agent.state import TaskState
from talkrobot.agent.tools import RuntimeToolContext, ToolRegistry, ToolResult
from talkrobot.config import Config


class AgentRuntime:
    """Runs one simple ReAct loop for a user turn."""

    def __init__(
        self,
        project_root: Optional[str] = None,
        language: str = "zh",
        tool_registry: Optional[ToolRegistry] = None,
        skill_dirs: Optional[Iterable[str]] = None,
        planner_provider: Optional[str] = None,
        max_react_iterations: Optional[int] = None,
        tool_authorization: Optional[AgentToolAuthorization] = None,
    ):
        self.project_root = os.path.abspath(project_root or os.getcwd())
        self.language = (language or "zh").strip().lower()
        self.planner = AgentPlanner(planner_provider or Config.AGENT_PLANNER_PROVIDER)
        self.tool_registry = tool_registry or ToolRegistry.default(self.project_root)
        self.skills = SkillRegistry(self._resolve_skill_dirs(skill_dirs))
        self.max_react_iterations = max(1, int(max_react_iterations or Config.AGENT_REACT_MAX_ITERATIONS))
        self.tool_authorization = tool_authorization or AgentToolAuthorization()
        self.policy_gate = ToolPolicyGate(self.tool_authorization, project_root=self.project_root)

    @classmethod
    def with_mcp_servers(
        cls,
        mcp_servers: Iterable[object],
        project_root: Optional[str] = None,
        **kwargs,
    ) -> "AgentRuntime":
        root = os.path.abspath(project_root or os.getcwd())
        return cls(project_root=root, tool_registry=ToolRegistry.with_mcp_servers(mcp_servers), **kwargs)

    @property
    def _is_english(self) -> bool:
        return self.language == "en"

    def _msg(self, zh: str, en: str) -> str:
        return en if self._is_english else zh

    def _tool_context(self, memory_module=None, long_term_memory: bool = False) -> RuntimeToolContext:
        return RuntimeToolContext(
            project_root=self.project_root,
            memory_module=memory_module,
            long_term_memory=long_term_memory,
            language=self.language,
            tool_authorization=self.tool_authorization,
        )

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
        progress_events: bool = False,
        approval_callback: Optional[Callable[[ToolApprovalRequest], bool]] = None,
    ) -> Iterator[AgentEvent]:
        del sliding_window_context, switch_notice, system_prompt_override
        turn_start = time.perf_counter()
        tools = self.tool_registry.build_tools(self._tool_context(memory_module, long_term_memory))
        task_state = TaskState(goal=user_text, mode="react")
        observations = []
        tool_results = []
        previous_observation = None

        for iteration in range(1, self.max_react_iterations + 1):
            if progress_events:
                yield self._llm_start_event(
                    stage="react_planner",
                    iteration=iteration,
                    context_chars=len(str(previous_observation or "")),
                )
            decision = self.planner.decide(
                user_text,
                tools=tools.values(),
                llm=llm,
                previous_observation=previous_observation,
                observation_history=observations,
                authorization_context=self._authorization_context_text(),
            )

            step = decision.step
            yield self._plan_event(iteration, decision)

            if decision.done or step is None:
                text = decision.answer or self._msg("我没有更多需要执行的操作。", "I have no more actions to run.")
                yield from self._final_events(
                    text,
                    task_state,
                    observations,
                    tool_results,
                    turn_start,
                    decision.reason,
                    llm=llm,
                    streaming=streaming,
                    progress_events=progress_events,
                )
                return

            if step.tool not in tools:
                previous_observation = self._record_observation(
                    observations,
                    tool_results,
                    task_state,
                    step,
                    ToolResult(ok=False, error=f"unknown tool: {step.tool}"),
                    iteration,
                    elapsed_ms=0,
                    retryable=False,
                )
                yield AgentEvent.error(
                    self._msg("模型选择了不存在的工具。", "The model selected an unknown tool."),
                    stage=step.tool,
                    error=previous_observation["error"],
                    elapsed_ms=0,
                    retryable=False,
                )
                continue

            previous_observation = yield from self._execute_one_step(
                step,
                tools[step.tool],
                task_state,
                observations,
                tool_results,
                user_text=user_text,
                iteration=iteration,
                approval_callback=approval_callback,
            )

        yield from self._final_events(
            self._msg(
                "我已达到本轮最大 ReAct 迭代次数，先停止以避免无限循环。当前任务还没有可靠完成。",
                "I reached the maximum ReAct iterations and stopped to avoid an infinite loop. The task is not reliably complete yet.",
            ),
            task_state,
            observations,
            tool_results,
            turn_start,
            reason="max_iterations",
            llm=llm,
            streaming=streaming,
            progress_events=progress_events,
        )

    def _llm_start_event(self, *, stage: str, iteration: int, context_chars: int = 0) -> AgentEvent:
        return AgentEvent(
            type="llm_start",
            text=self._msg("我在整理下一步。", "I am working out the next step."),
            speakable=False,
            data={"stage": stage, "iteration": iteration, "context_chars": context_chars},
        )

    def _execute_one_step(
        self,
        step: ToolStep,
        tool,
        task_state: TaskState,
        observations,
        tool_results,
        *,
        user_text: str,
        iteration: int,
        approval_callback: Optional[Callable[[ToolApprovalRequest], bool]],
    ) -> Iterator[AgentEvent]:
        policy = self.policy_gate.assess(step, user_text=user_text, risk_level="low")
        if not policy.allowed:
            approved_by_user = False
            if policy.requires_confirmation:
                request = ToolApprovalRequest(
                    tool=step.tool,
                    args=step.args if isinstance(step.args, dict) else {},
                    reason=step.reason,
                    policy_reason=policy.reason,
                    risk_level="low",
                )
                yield AgentEvent(
                    type="approval_requested",
                    text=self._msg("这个操作需要你的确认。", "This operation needs your approval."),
                    speakable=True,
                    data=request.to_dict(),
                )
                approved_by_user = self._resolve_approval(approval_callback, request)
                yield AgentEvent(
                    type="approval_result",
                    text=self._msg("已批准。", "Approved.") if approved_by_user else self._msg("未批准。", "Not approved."),
                    speakable=False,
                    data={**request.to_dict(), "approved": approved_by_user},
                )

            if not approved_by_user:
                result = ToolResult(ok=False, error=f"policy rejected: {policy.reason}")
                observation = self._record_observation(
                    observations,
                    tool_results,
                    task_state,
                    step,
                    result,
                    iteration,
                    elapsed_ms=0,
                    retryable=False,
                )
                yield AgentEvent.error(
                    self._msg("这个工具调用被安全策略拦截了。", "The tool call was blocked by policy."),
                    stage=step.tool,
                    error=result.error,
                    elapsed_ms=0,
                    retryable=False,
                )
                yield self._tool_result_event(step.tool, result, iteration, 0, False, observation, policy_rejected=True)
                return observation

        yield AgentEvent(
            type="tool_start",
            text=getattr(tool, "speakable_start", ""),
            speakable=step.tool != "memory_search",
            data={"tool": step.tool, "reason": step.reason, "iteration": iteration},
        )
        started = time.perf_counter()
        result = self._run_tool(tool, step.args)
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        retryable = (not result.ok) and self._is_retryable_tool_error(result.error)
        observation = self._record_observation(
            observations,
            tool_results,
            task_state,
            step,
            result,
            iteration,
            elapsed_ms=elapsed_ms,
            retryable=retryable,
        )

        if not result.ok:
            yield AgentEvent.error(
                self._msg("工具执行失败。", "Tool execution failed."),
                stage=step.tool,
                error=result.error,
                elapsed_ms=elapsed_ms,
                retryable=retryable,
            )
        yield self._tool_result_event(step.tool, result, iteration, elapsed_ms, retryable, observation)
        return observation

    def _record_observation(
        self,
        observations,
        tool_results,
        task_state: TaskState,
        step: ToolStep,
        result: ToolResult,
        iteration: int,
        *,
        elapsed_ms: int,
        retryable: bool,
    ) -> dict:
        observation = {
            "iteration": iteration,
            "tool": step.tool,
            "args": step.args,
            "ok": result.ok,
            "content": result.content,
            "error": result.error,
            "data": result.data,
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
        task_state.note_tool_result(step.tool, step.args, result.ok, result.content, result.error, result.data)
        return observation

    def _plan_event(self, iteration: int, decision: ReactDecision) -> AgentEvent:
        step = decision.step
        return AgentEvent(
            type="plan",
            text=self._msg("我判断下一步。", "I'll choose the next step."),
            speakable=False,
            data={
                "mode": "final" if decision.done or step is None else "tool_assisted",
                "planner": self.planner.provider,
                "reason": decision.reason,
                "iteration": iteration,
                "steps": [step.tool] if step is not None else [],
            },
        )

    def _tool_result_event(
        self,
        tool_name: str,
        result: ToolResult,
        iteration: int,
        elapsed_ms: int,
        retryable: bool,
        observation: dict,
        *,
        policy_rejected: bool = False,
    ) -> AgentEvent:
        data = {
            "tool": tool_name,
            "ok": result.ok,
            "elapsed_ms": elapsed_ms,
            "iteration": iteration,
            "retryable": retryable,
            "observation": observation,
        }
        if policy_rejected:
            data["policy_rejected"] = True
        return AgentEvent(
            type="tool_result",
            text=self._tool_result_text(tool_name, result.ok, result.content),
            speakable=False,
            data=data,
        )

    def _final_event(
        self,
        text: str,
        task_state: TaskState,
        observations,
        tool_results,
        turn_start: float,
        reason: str = "",
    ) -> AgentEvent:
        raw_text = str(text or "")
        replacement_reason = self._final_replacement_reason(raw_text, task_state)
        final_text = self._final_text_or_incomplete(raw_text, task_state)
        return AgentEvent(
            type="final_response",
            text=final_text,
            speakable=True,
            data={
                "observations": observations,
                "tool_results": tool_results,
                "react_iterations": len({item["iteration"] for item in observations}),
                "llm_elapsed_ms": 0,
                "total_elapsed_ms": round((time.perf_counter() - turn_start) * 1000),
                "used_memory": any(r["tool"] == "memory_search" and r["ok"] for r in tool_results),
                "used_tools": [r["tool"] for r in tool_results if r["tool"] != "memory_search"],
                "mode": "react",
                "risk_level": "low",
                "reason": reason,
                "raw_final_was_tool_directive": replacement_reason == "tool_directive",
                "raw_final_replaced_reason": replacement_reason,
                "task_state": self._task_state_payload(task_state),
            },
        )

    def _final_events(
        self,
        text: str,
        task_state: TaskState,
        observations,
        tool_results,
        turn_start: float,
        reason: str = "",
        *,
        llm=None,
        streaming: bool = False,
        progress_events: bool = False,
    ) -> Iterator[AgentEvent]:
        final_text = self._final_text_or_incomplete(str(text or ""), task_state)
        can_stream = (
            bool(streaming)
            and hasattr(llm, "generate_response_stream")
            and not self._final_replacement_reason(str(text or ""), task_state)
        )
        if can_stream:
            if progress_events:
                yield self._llm_start_event(
                    stage="final_response",
                    iteration=len({item["iteration"] for item in observations}) + 1,
                    context_chars=len(json.dumps(observations, ensure_ascii=False, default=str)),
                )
            chunks = []
            try:
                for chunk in llm.generate_response_stream(
                    self._final_stream_user_prompt(task_state.goal),
                    context=self._final_stream_context(text, observations, tool_results, task_state),
                    system_prompt_override=self._final_stream_system_prompt(),
                ):
                    if not chunk:
                        continue
                    chunks.append(str(chunk))
                    yield AgentEvent(
                        type="final_response_delta",
                        text=str(chunk),
                        speakable=True,
                        data={"stage": "final_response"},
                    )
            except Exception:
                chunks = []
            streamed_text = "".join(chunks).strip()
            if streamed_text:
                final_text = streamed_text
        yield self._final_event(final_text, task_state, observations, tool_results, turn_start, reason)

    def _final_stream_context(self, planner_answer: str, observations, tool_results, task_state: TaskState) -> str:
        payload = {
            "planner_answer": planner_answer,
            "observations": observations,
            "tool_results": tool_results,
            "changed_files": task_state.changed_files,
            "failed_attempts": task_state.failed_attempts,
        }
        return json.dumps(payload, ensure_ascii=False, default=str)

    def _final_stream_user_prompt(self, goal: str) -> str:
        return self._msg(
            f"请基于已完成的操作，给用户一个简洁的最终回复。\n用户任务：{goal}",
            f"Based on the completed actions, give the user a concise final reply.\nUser task: {goal}",
        )

    @staticmethod
    def _final_stream_system_prompt() -> str:
        return (
            "You are Tyro composing the final user-facing answer for an agent run. "
            "Use only the provided context and do not reveal hidden chain-of-thought. "
            "Mention important completed actions, changed files, failures, or verification results. "
            "Keep it concise and use the user's language."
        )

    def _final_text_or_incomplete(self, text: str, task_state: Optional[TaskState] = None) -> str:
        reason = self._final_replacement_reason(text, task_state)
        if reason == "tool_directive":
            return self._msg(
                "我还没有完成这次任务：模型在最终回复里又请求调用工具，已停止以避免误报。",
                "I have not completed this task: the model requested another tool call in the final reply, so I stopped instead of reporting success.",
            )
        if reason == "write_not_applied":
            return self._msg(
                "我还没有把修改写入文件：本轮没有任何成功的文件修改记录。",
                "I have not written changes to disk: this turn has no successful file-edit record.",
            )
        return text

    def _final_replacement_reason(self, text: str, task_state: Optional[TaskState] = None) -> str:
        if self._looks_like_tool_directive(text):
            return "tool_directive"
        if self._write_requested_without_changes(task_state):
            return "write_not_applied"
        return ""

    @staticmethod
    def _write_requested_without_changes(task_state: Optional[TaskState]) -> bool:
        if task_state is None or task_state.changed_files:
            return False
        goal = str(task_state.goal or "").casefold()
        markers = (
            "帮我修复",
            "修复这个",
            "修复bug",
            "修复 bug",
            "修改",
            "改一下",
            "替换",
            "写入",
            "写到文件",
            "fix this",
            "fix the",
            "fix bug",
            "modify",
            "update",
            "patch",
            "write",
            "replace",
        )
        return any(marker in goal for marker in markers)

    @staticmethod
    def _looks_like_tool_directive(text: str) -> bool:
        stripped = str(text or "").strip()
        if not stripped:
            return False
        if re.search(r"<\s*/?\s*tool(?:_use)?\s*>", stripped, flags=re.IGNORECASE):
            return True
        return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\s*\n\s*\{[\s\S]*\}\s*", stripped))

    @staticmethod
    def _task_state_payload(task_state: TaskState) -> dict:
        return {
            "goal": task_state.goal,
            "mode": task_state.mode,
            "facts": task_state.facts,
            "attempted_actions": task_state.attempted_actions,
            "failed_attempts": task_state.failed_attempts,
            "changed_files": task_state.changed_files,
            "verification_results": task_state.verification_results,
            "repo_summary": task_state.repo_summary,
            "test_commands": task_state.test_commands,
            "command_results": task_state.command_results,
        }

    def _authorization_payload(self) -> dict:
        return {
            "allow_shell_commands": self.tool_authorization.allow_shell_commands,
            "allowed_command_prefixes": self.tool_authorization.command_prefixes(),
            "allow_file_writes": self.tool_authorization.allow_file_writes,
            "allowed_write_paths": self.tool_authorization.allowed_write_paths,
        }

    def _authorization_context_text(self) -> str:
        payload = self._authorization_payload()
        return (
            f"shell_command authorized: {payload['allow_shell_commands']}; "
            f"allowed command prefixes: {payload['allowed_command_prefixes']}; "
            f"file_edit authorized: {payload['allow_file_writes']}; "
            f"allowed write paths: {payload['allowed_write_paths']}"
        )

    def _tool_result_text(self, tool_name: str, ok: bool, content: str) -> str:
        if not ok:
            return self._msg("工具执行失败。", "Tool execution failed.")
        if tool_name == "memory_search":
            return self._msg("记忆检索完成。", "Memory search finished.")
        if content:
            return self._msg("工具执行完成。", "Tool execution finished.")
        return self._msg("工具没有返回额外内容。", "Tool returned no extra content.")

    def _run_tool(self, tool, args) -> ToolResult:
        try:
            safe_args = self._filter_tool_args(tool.run, args or {})
            return tool.run(**safe_args)
        except Exception as exc:
            return ToolResult(ok=False, error=f"{exc.__class__.__name__}: {exc}")

    @staticmethod
    def _filter_tool_args(run_callable, args) -> dict:
        if not isinstance(args, dict):
            return {}
        try:
            signature = inspect.signature(run_callable)
        except (TypeError, ValueError):
            return dict(args)

        parameters = signature.parameters.values()
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters):
            return dict(args)

        allowed = {
            name
            for name, param in signature.parameters.items()
            if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        unknown = sorted(str(key) for key in args if key not in allowed)
        if unknown:
            raise ValueError(f"unknown tool argument(s): {', '.join(unknown)}")
        return {key: value for key, value in args.items() if key in allowed}

    @staticmethod
    def _resolve_approval(
        approval_callback: Optional[Callable[[ToolApprovalRequest], bool]],
        request: ToolApprovalRequest,
    ) -> bool:
        if approval_callback is None:
            return False
        try:
            return bool(approval_callback(request))
        except Exception:
            return False

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

    def _resolve_skill_dirs(self, skill_dirs: Optional[Iterable[str]]) -> Iterable[str]:
        dirs = list(skill_dirs or [])
        dirs.append(os.path.join(os.path.dirname(__file__), "skills"))
        extra_dirs = os.getenv("TALKROBOT_SKILL_DIRS", "")
        for path in extra_dirs.split(os.pathsep):
            path = path.strip()
            if path:
                dirs.append(path)
        return dirs
