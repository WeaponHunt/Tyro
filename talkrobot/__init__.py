"""
TalkRobot - 智能对话机器人
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from talkrobot.config import Config
from talkrobot.modules.asr_module import ASRModule
from talkrobot.modules.tts_module import TTSModule
from talkrobot.modules.llm_module import LLMModule
from talkrobot.modules.memory_module import MemoryModule

__all__ = [
    "Config",
    "ASRModule",
    "TTSModule",
    "LLMModule",
    "MemoryModule",
]