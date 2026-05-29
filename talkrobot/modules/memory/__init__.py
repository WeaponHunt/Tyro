"""Pluggable memory backends."""

from talkrobot.modules.memory.base import MemoryBackend, MemoryRecord
from talkrobot.modules.memory.memory_module import MemoryModule
from talkrobot.modules.memory.simple_backend import SimpleJsonMemoryBackend

__all__ = [
    "MemoryBackend",
    "MemoryRecord",
    "MemoryModule",
    "SimpleJsonMemoryBackend",
]
