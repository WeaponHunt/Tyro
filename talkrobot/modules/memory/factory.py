"""Factory helpers for memory backends and facade modules."""
from __future__ import annotations

from talkrobot.config import Config
from talkrobot.modules.memory.base import MemoryBackend
from talkrobot.modules.memory.mem0_backend import Mem0MemoryBackend
from talkrobot.modules.memory.simple_backend import SimpleJsonMemoryBackend


def create_memory_backend(provider: str, config: dict, user_id: str) -> MemoryBackend:
    provider = (provider or "mem0").strip().lower()
    if provider == "mem0":
        return Mem0MemoryBackend(config=config, user_id=user_id)
    if provider in {"simple", "json", "local"}:
        db_path = config.get("path") or config.get("db_path") or Config.get_simple_memory_path()
        return SimpleJsonMemoryBackend(db_path=db_path, user_id=user_id)
    raise ValueError(f"Unsupported memory provider: {provider}")


def create_memory_for_user(user: str, provider: str | None = None):
    from talkrobot.modules.memory.memory_module import MemoryModule

    selected_provider = provider or Config.MEMORY_PROVIDER
    if selected_provider == "mem0":
        config = Config.get_memory_config(user)
    else:
        config = {"path": Config.get_simple_memory_path()}
    return MemoryModule(
        config=config,
        user_id=Config.get_user_id(user),
        provider=selected_provider,
    )
