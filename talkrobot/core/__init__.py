"""
TalkRobot 核心模块
"""

__all__ = [
    "AudioRecorder",
    "ConversationManager",
    "FaceIdentityResolver",
    "SlidingWindowDialogueHistory",
    "TTSPlaybackController",
    "UserMemoryRouter",
]


def __getattr__(name):
    if name == "AudioRecorder":
        from talkrobot.core.audio_recorder import AudioRecorder
        return AudioRecorder
    if name == "ConversationManager":
        from talkrobot.core.conversation_manager import ConversationManager
        return ConversationManager
    if name == "FaceIdentityResolver":
        from talkrobot.core.face_identity_resolver import FaceIdentityResolver
        return FaceIdentityResolver
    if name == "SlidingWindowDialogueHistory":
        from talkrobot.core.dialogue_history import SlidingWindowDialogueHistory
        return SlidingWindowDialogueHistory
    if name == "TTSPlaybackController":
        from talkrobot.core.tts_playback_controller import TTSPlaybackController
        return TTSPlaybackController
    if name == "UserMemoryRouter":
        from talkrobot.core.memory_router import UserMemoryRouter
        return UserMemoryRouter
    raise AttributeError(f"module 'talkrobot.core' has no attribute '{name}'")
