"""TalkRobot core modules."""

__all__ = ["AudioRecorder", "ConversationManager", "VideoRecorder"]


def __getattr__(name):
    if name == "AudioRecorder":
        from talkrobot.core.audio_recorder import AudioRecorder

        return AudioRecorder
    if name == "ConversationManager":
        from talkrobot.core.conversation_manager import ConversationManager

        return ConversationManager
    if name == "VideoRecorder":
        from talkrobot.core.video_recorder import VideoRecorder

        return VideoRecorder
    raise AttributeError(name)
