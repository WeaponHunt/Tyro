"""
FastAPI web UI for TalkRobot.

Run with:
    python -m talkrobot.web_app
"""
from __future__ import annotations

import os
import re
import time
import threading
import importlib.util
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple

import uvicorn
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
try:
    from loguru import logger
except Exception:
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
from openai import OpenAI
from pydantic import BaseModel, Field

from talkrobot.agent import AgentRuntime
from talkrobot.config import Config
from talkrobot.core.dialogue_history import SlidingWindowDialogueHistory
from talkrobot.core.persona_manager import PersonaManager


WEB_DIR = os.path.join(os.path.dirname(__file__), "web")
STATIC_DIR = os.path.join(WEB_DIR, "static")
EXPRESSION_PATTERN = re.compile(r"\[expression:(\w[\w-]*)\]")
AVAILABLE_EXPRESSIONS = [
    "happy",
    "angry",
    "sad",
    "scared",
    "surprised",
    "more-happy",
    "dizzy",
    "evil_smile",
    "nauty_smile",
    "pitying",
]
PROXY_ENV_NAMES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)
UNSUPPORTED_PROXY_SCHEMES = ("socks://", "socks4://", "socks5://")


def _drop_unsupported_proxy_env() -> None:
    """Avoid httpx/OpenAI failures when socks proxy support is not installed."""
    removed = []
    for name in PROXY_ENV_NAMES:
        value = (os.environ.get(name) or "").strip()
        if value.lower().startswith(UNSUPPORTED_PROXY_SCHEMES):
            os.environ.pop(name, None)
            removed.append(name)

    if removed:
        logger.warning(
            "Web UI 已忽略不受当前 httpx 环境支持的 socks 代理变量: "
            + ", ".join(removed)
        )


_drop_unsupported_proxy_env()


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    user: str = Field(default=Config.DEFAULT_USER, min_length=1, max_length=80)
    language: str = Field(default=Config.LANGUAGE)
    history_rounds: int = Field(default=Config.SLIDING_WINDOW_ROUNDS, ge=0, le=50)
    use_memory: bool = True


class MemoryRequest(BaseModel):
    user: str = Field(default=Config.DEFAULT_USER, min_length=1, max_length=80)
    content: str = Field(..., min_length=1, max_length=8000)


class ClearHistoryRequest(BaseModel):
    user: str = Field(default=Config.DEFAULT_USER, min_length=1, max_length=80)
    language: str = Field(default=Config.LANGUAGE)
    history_rounds: int = Field(default=Config.SLIDING_WINDOW_ROUNDS, ge=0, le=50)
    use_memory: bool = True


def _normalize_language(language: str) -> str:
    language = (language or Config.LANGUAGE or "zh").strip().lower()
    return language if language in {"zh", "en"} else "zh"


def _selected_prompts(language: str) -> Tuple[str, str]:
    if _normalize_language(language) == "en":
        return Config.SYSTEM_PROMPT_EN, Config.GLOBAL_SYSTEM_PROMPT_EN
    return Config.SYSTEM_PROMPT, Config.GLOBAL_SYSTEM_PROMPT


def _merge_context(memory_context: str, history_context: str) -> str:
    sections = []
    if memory_context:
        sections.append(f"检索到的相关记忆:\n{memory_context}")
    if history_context:
        sections.append(history_context)
    return "\n\n".join(sections)


def _normalize_memories(raw: Any) -> List[Dict[str, str]]:
    if isinstance(raw, dict) and "results" in raw:
        raw = raw["results"]
    if not isinstance(raw, list):
        return []

    memories: List[Dict[str, str]] = []
    for item in raw:
        if isinstance(item, str):
            text = item
        elif isinstance(item, dict):
            text = (
                item.get("memory")
                or item.get("text")
                or item.get("content")
                or item.get("value")
                or ""
            )
        else:
            text = (
                getattr(item, "memory", None)
                or getattr(item, "text", None)
                or getattr(item, "content", None)
                or ""
            )
        text = str(text).strip()
        if text:
            memories.append({"text": text})
    return memories


def _expression_prompt() -> str:
    expr_list = ", ".join(AVAILABLE_EXPRESSIONS)
    return (
        "\n\n【表情指令】在每次回复的开头，根据回复内容的情感，"
        "添加一个表情标签，格式为 [expression:表情名称]。"
        f"可选的表情有: {expr_list}。"
        "只需要一个表情标签，放在回复最前面。"
    )


def _parse_expression(response: str) -> Tuple[str, Optional[str]]:
    match = EXPRESSION_PATTERN.search(response)
    if not match:
        return response, None
    return EXPRESSION_PATTERN.sub("", response).strip(), match.group(1)


class WebChatSession:
    """A text-chat session that reuses TalkRobot's LLM, memory and persona modules."""

    def __init__(self, user: str, language: str, history_rounds: int, use_memory: bool):
        self.user = (user or Config.DEFAULT_USER).strip() or Config.DEFAULT_USER
        self.language = _normalize_language(language)
        self.history_rounds = max(0, int(history_rounds))
        self.use_memory = bool(use_memory)
        self._lock = threading.Lock()
        self._history = SlidingWindowDialogueHistory(self.history_rounds)

        system_prompt, global_prompt = _selected_prompts(self.language)
        self._global_prompt = (global_prompt or "").strip()
        self._expression_prompt = _expression_prompt() if Config.EXPRESSION_ENABLED else ""
        self._persona_manager = PersonaManager(
            profile_path=Config.PERSONA_PROFILE_PATH,
            fallback_prompt=system_prompt,
        )
        self._http_client = httpx.Client(trust_env=False)
        self._llm_client = OpenAI(
            api_key=Config.LLM_API_KEY,
            base_url=Config.LLM_BASE_URL,
            http_client=self._http_client,
        )
        self._llm_model = Config.LLM_MODEL
        self._system_prompt = system_prompt + self._expression_prompt
        self._memory: Optional[Any] = None
        self._memory_error = ""
        self._agent_runtime = AgentRuntime(
            project_root=os.path.dirname(os.path.dirname(__file__)),
            language=self.language,
        )
        if self.use_memory:
            self._init_memory()

    @property
    def memory_enabled(self) -> bool:
        return self._memory is not None

    @property
    def memory_error(self) -> str:
        return self._memory_error

    def _init_memory(self) -> None:
        try:
            from talkrobot.modules.memory.memory_module import MemoryModule

            self._memory = MemoryModule(
                config=Config.get_memory_config(self.user),
                user_id=Config.get_user_id(self.user),
            )
        except Exception as exc:
            self._memory = None
            self._memory_error = str(exc)
            logger.warning(f"Web UI 记忆模块初始化失败，已降级为短期上下文: {exc}")

    def _persona_prompt(self) -> str:
        persona_prompt = self._persona_manager.get_prompt_for_user(self.user)
        sections = []
        if persona_prompt and persona_prompt.strip():
            sections.append(persona_prompt.strip())
        if self._global_prompt:
            sections.append(self._global_prompt)
        if self._expression_prompt:
            sections.append(self._expression_prompt.strip())
        return "\n\n".join(sections)

    def _generate_response(self, message: str, context: str, system_prompt_override: str = "") -> str:
        system_prompt = (system_prompt_override or "").strip() or self._persona_prompt() or self._system_prompt
        messages = [{"role": "system", "content": system_prompt}]
        if context:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        f"Relevant background information about the user:\n{context}"
                        if self.language == "en"
                        else f"关于用户的相关背景信息:\n{context}"
                    ),
                }
            )
        messages.append({"role": "user", "content": message})

        try:
            completion = self._llm_client.chat.completions.create(
                model=self._llm_model,
                messages=messages,
            )
            return completion.choices[0].message.content or ""
        except Exception as exc:
            logger.error(f"Web UI LLM 生成失败: {exc}")
            if self.language == "en":
                return "Sorry, I can't answer your question right now."
            return "抱歉，我现在无法回答您的问题。"

    def generate_response(self, user_input: str, context: str = "", system_prompt_override: str = "") -> str:
        return self._generate_response(user_input, context, system_prompt_override)

    def chat(self, message: str) -> Dict[str, Any]:
        message = message.strip()
        if not message:
            raise ValueError("message is empty")

        with self._lock:
            started = time.perf_counter()
            history_context = self._history.build_context(self.user)
            raw_response = ""
            agent_context = ""
            used_memory = False
            used_tools = []

            for event in self._agent_runtime.run_stream(
                user_text=message,
                llm=self,
                memory_module=self._memory,
                long_term_memory=self._memory is not None,
                sliding_window_context=history_context,
                system_prompt_override=self._persona_prompt(),
                streaming=False,
            ):
                if event.type == "final_response":
                    raw_response = event.text
                    agent_context = event.data.get("context", "")
                    used_memory = bool(event.data.get("used_memory", False))
                    used_tools = event.data.get("used_tools", [])
                elif event.type == "error":
                    logger.warning(f"Web Agent事件错误: {event.data}")

            response, expression = _parse_expression(raw_response)
            response = response.strip() or raw_response.strip()

            self._history.append(self.user, message, response)
            if self._memory is not None:
                if hasattr(self._memory, "add_user_memory_if_stable"):
                    self._memory.add_user_memory_if_stable(message, async_mode=True)
                else:
                    self._memory.add_memory(f"用户说: {message}", async_mode=True)

            elapsed_ms = round((time.perf_counter() - started) * 1000)
            return {
                "reply": response,
                "expression": expression or "neutral",
                "user": self.user,
                "language": self.language,
                "memory_enabled": self.memory_enabled,
                "memory_error": self.memory_error,
                "used_memory": used_memory,
                "used_history": bool(history_context),
                "used_tools": used_tools,
                "context_chars": len(agent_context),
                "elapsed_ms": elapsed_ms,
            }

    def add_memory(self, content: str) -> None:
        if self._memory is None:
            self._init_memory()
        if self._memory is None:
            raise RuntimeError(self._memory_error or "memory module unavailable")
        self._memory.add_memory(content.strip(), async_mode=False)

    def list_memories(self) -> List[Dict[str, str]]:
        if self._memory is None:
            self._init_memory()
        if self._memory is None:
            return []
        return _normalize_memories(self._memory.get_all_memories())

    def clear_history(self) -> None:
        self._history.clear(self.user)

    def shutdown(self) -> None:
        if self._memory is not None:
            self._memory.shutdown()
        self._http_client.close()


class SessionStore:
    def __init__(self) -> None:
        self._sessions: Dict[Tuple[str, str, int, bool], WebChatSession] = {}
        self._lock = threading.Lock()

    def get(self, user: str, language: str, history_rounds: int, use_memory: bool) -> WebChatSession:
        normalized_user = (user or Config.DEFAULT_USER).strip() or Config.DEFAULT_USER
        normalized_language = _normalize_language(language)
        key = (normalized_user, normalized_language, max(0, int(history_rounds)), bool(use_memory))
        with self._lock:
            session = self._sessions.get(key)
            if session is None:
                session = WebChatSession(*key)
                self._sessions[key] = session
            return session

    def shutdown(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            try:
                session.shutdown()
            except Exception as exc:
                logger.warning(f"Web UI 会话关闭异常: {exc}")


store = SessionStore()


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    store.shutdown()


app = FastAPI(title="TalkRobot Web UI", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/config")
def get_config():
    return {
        "default_user": Config.DEFAULT_USER,
        "language": _normalize_language(Config.LANGUAGE),
        "history_rounds": Config.SLIDING_WINDOW_ROUNDS,
        "model": Config.LLM_MODEL,
        "memory_available": importlib.util.find_spec("mem0") is not None,
    }


@app.get("/api/health")
def health():
    return {"ok": True, "service": "talkrobot-web"}


@app.post("/api/chat")
def chat(payload: ChatRequest):
    try:
        session = store.get(
            payload.user,
            payload.language,
            payload.history_rounds,
            payload.use_memory,
        )
        return session.chat(payload.message)
    except Exception as exc:
        logger.exception(f"Web UI 对话失败: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/memory")
def add_memory(payload: MemoryRequest):
    try:
        session = store.get(payload.user, Config.LANGUAGE, Config.SLIDING_WINDOW_ROUNDS, True)
        session.add_memory(payload.content)
        return {"ok": True}
    except Exception as exc:
        logger.exception(f"Web UI 添加记忆失败: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/memories")
def list_memories(user: str = Query(default=Config.DEFAULT_USER, min_length=1, max_length=80)):
    try:
        session = store.get(user, Config.LANGUAGE, Config.SLIDING_WINDOW_ROUNDS, True)
        return {"memories": session.list_memories()}
    except Exception as exc:
        logger.exception(f"Web UI 读取记忆失败: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/history/clear")
def clear_history(payload: ClearHistoryRequest):
    session = store.get(
        payload.user,
        payload.language,
        payload.history_rounds,
        payload.use_memory,
    )
    session.clear_history()
    return {"ok": True}


def main() -> None:
    host = os.getenv("TALKROBOT_WEB_HOST", "127.0.0.1")
    port = int(os.getenv("TALKROBOT_WEB_PORT", "7860"))
    uvicorn.run("talkrobot.web_app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
