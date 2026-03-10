"""
TalkRobot - 智能对话机器人
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from talkrobot.config import Config

__all__ = [
    "Config",
    "ASRModule",
    "TTSModule",
    "LLMModule",
    "MemoryModule",
]


def __getattr__(name):
    if name == "ASRModule":
        from talkrobot.modules.asr_module import ASRModule
        return ASRModule
    if name == "TTSModule":
        from talkrobot.modules.tts_module import TTSModule
        return TTSModule
    if name == "LLMModule":
        from talkrobot.modules.llm_module import LLMModule
        return LLMModule
    if name == "MemoryModule":
        from talkrobot.modules.memory_module import MemoryModule
        return MemoryModule
    raise AttributeError(f"module 'talkrobot' has no attribute '{name}'")