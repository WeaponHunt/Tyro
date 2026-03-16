"""
TalkRobot 功能模块
"""

__all__ = [
    "ASRModule",
    "TTSModule",
    "LLMModule",
    "MemoryModule",
    "ExpressionModule",
    "FaceRecognitionModule",
]


def __getattr__(name):
    if name == "ASRModule":
        from talkrobot.modules.asr.asr_module import ASRModule
        return ASRModule
    if name == "TTSModule":
        from talkrobot.modules.tts.tts_module import TTSModule
        return TTSModule
    if name == "LLMModule":
        from talkrobot.modules.llm.llm_module import LLMModule
        return LLMModule
    if name == "MemoryModule":
        from talkrobot.modules.memory.memory_module import MemoryModule
        return MemoryModule
    if name == "ExpressionModule":
        from talkrobot.modules.expression.expression_module import ExpressionModule
        return ExpressionModule
    if name == "FaceRecognitionModule":
        from talkrobot.modules.face_recognize.face_recognition import FaceRecognitionModule
        return FaceRecognitionModule
    raise AttributeError(f"module 'talkrobot.modules' has no attribute '{name}'")