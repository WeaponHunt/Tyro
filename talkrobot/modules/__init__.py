"""
TalkRobot 功能模块
"""

__all__ = [
    "ASRModule",
    "TTSModule",
    "LLMModule",
    "MemoryModule",
    "ExpressionModule",
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
    if name == "ExpressionModule":
        from talkrobot.modules.expression_module import ExpressionModule
        return ExpressionModule
    raise AttributeError(f"module 'talkrobot.modules' has no attribute '{name}'")