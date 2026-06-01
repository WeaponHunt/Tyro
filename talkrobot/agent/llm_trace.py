"""Dedicated JSONL tracing for Agent LLM calls."""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

try:
    from loguru import logger
except Exception:
    import logging

    logger = logging.getLogger(__name__)

from talkrobot.core.app_logging import DEFAULT_LOG_DIR


_LOCK = threading.Lock()


class AgentLLMTraceWrapper:
    """Wrap an LLM object and persist every model call made by the Agent."""

    def __init__(
        self,
        llm,
        *,
        trace_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        log_dir: str | os.PathLike[str] | None = None,
    ):
        self._llm = llm
        self.trace_id = trace_id or str(uuid.uuid4())
        self.metadata = dict(metadata or {})
        self.log_dir = Path(log_dir or os.getenv("TALKROBOT_AGENT_LLM_TRACE_DIR") or DEFAULT_LOG_DIR / "agent_llm")
        self._call_index = 0

    def __getattr__(self, name: str):
        return getattr(self._llm, name)

    def generate_response(self, user_input: str, context: str = "", system_prompt_override: str = "") -> str:
        self._call_index += 1
        call_index = self._call_index
        started = time.perf_counter()
        response = ""
        error = ""
        try:
            response = self._llm.generate_response(
                user_input,
                context=context,
                system_prompt_override=system_prompt_override,
            )
            return response
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            self._write_call(
                call_index=call_index,
                user_input=user_input,
                context=context,
                system_prompt_override=system_prompt_override,
                response=response,
                error=error,
                elapsed_ms=round((time.perf_counter() - started) * 1000),
                streaming=False,
            )

    def generate_response_stream(self, user_input: str, context: str = "", system_prompt_override: str = "") -> Iterator[str]:
        self._call_index += 1
        call_index = self._call_index
        started = time.perf_counter()
        parts = []
        error = ""
        try:
            for chunk in self._llm.generate_response_stream(
                user_input,
                context=context,
                system_prompt_override=system_prompt_override,
            ):
                parts.append(chunk)
                yield chunk
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            self._write_call(
                call_index=call_index,
                user_input=user_input,
                context=context,
                system_prompt_override=system_prompt_override,
                response="".join(parts),
                error=error,
                elapsed_ms=round((time.perf_counter() - started) * 1000),
                streaming=True,
            )

    def _write_call(
        self,
        *,
        call_index: int,
        user_input: str,
        context: str,
        system_prompt_override: str,
        response: str,
        error: str,
        elapsed_ms: int,
        streaming: bool,
    ) -> None:
        if not agent_llm_trace_enabled():
            return
        now = datetime.now().astimezone()
        system_prompt = (system_prompt_override or "").strip() or str(getattr(self._llm, "system_prompt", "") or "")
        record = {
            "trace_id": self.trace_id,
            "call_id": str(uuid.uuid4()),
            "call_index": call_index,
            "timestamp": now.isoformat(timespec="milliseconds"),
            "stage": classify_agent_llm_stage(system_prompt_override, context),
            "streaming": streaming,
            "elapsed_ms": elapsed_ms,
            "model": getattr(self._llm, "model", ""),
            "language": getattr(self._llm, "language", ""),
            "metadata": self.metadata,
            "system_prompt": system_prompt,
            "user_input": user_input,
            "context": context,
            "response": response,
            "error": error,
        }
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            _cleanup_old_files(self.log_dir)
            path = self.log_dir / f"agent_llm_{now.strftime('%Y-%m-%d')}.jsonl"
            line = json.dumps(record, ensure_ascii=False, default=str)
            with _LOCK:
                with path.open("a", encoding="utf-8") as file:
                    file.write(line + "\n")
        except Exception as exc:
            logger.warning(f"写入 Agent LLM trace 失败: {exc}")


def wrap_agent_llm_trace(llm, *, metadata: Optional[Dict[str, Any]] = None, log_dir: str | os.PathLike[str] | None = None):
    if isinstance(llm, AgentLLMTraceWrapper):
        return llm
    return AgentLLMTraceWrapper(llm, metadata=metadata, log_dir=log_dir)


def classify_agent_llm_stage(system_prompt_override: str = "", context: str = "") -> str:
    prompt = (system_prompt_override or "").casefold()
    if "minimal react controller" in prompt:
        return "react_planner"
    if context:
        return "final_response"
    return "llm"


def agent_llm_trace_enabled() -> bool:
    value = os.getenv("TALKROBOT_AGENT_LLM_TRACE_ENABLED")
    if value is None:
        return True
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _cleanup_old_files(log_dir: Path) -> None:
    cutoff = datetime.now().date() - timedelta(days=_env_int("TALKROBOT_LOG_RETENTION_DAYS", 30))
    for path in log_dir.glob("agent_llm_*.jsonl"):
        day_text = path.name[len("agent_llm_") : -len(".jsonl")]
        try:
            file_day = datetime.strptime(day_text, "%Y-%m-%d").date()
        except ValueError:
            continue
        if file_day < cutoff:
            try:
                path.unlink()
            except OSError:
                pass


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default
