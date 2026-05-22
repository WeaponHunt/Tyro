"""
TTS playback coordination.
"""
from typing import Iterable, Union

from loguru import logger
from pynput import keyboard


class TTSPlaybackController:
    """Coordinates keyboard interruption and recorder mute state during TTS."""

    def __init__(self, tts_module, audio_recorder=None):
        self.tts = tts_module
        self.audio_recorder = audio_recorder

    def play(self, text: Union[str, Iterable[str]]) -> bool:
        """Play TTS and return whether playback was interrupted."""
        if self.tts is None:
            return False

        def _on_press_interrupt(key):
            try:
                if hasattr(key, "char") and key.char == "s":
                    logger.info("检测到S键，触发TTS打断")
                    if hasattr(self.tts, "interrupt"):
                        self.tts.interrupt()
                    else:
                        self.tts._interrupted.set()
            except Exception as e:
                logger.debug(f"回调异常: {e}")

        interrupt_listener = keyboard.Listener(on_press=_on_press_interrupt, daemon=True)
        interrupt_listener.start()
        logger.debug("TTS打断监听器已启动（daemon模式）")

        if self.audio_recorder:
            logger.debug("设置 is_tts_playing = True")
            self.audio_recorder.is_tts_playing = True
        try:
            self.tts.synthesize(text, play_audio=True)
        finally:
            if self.audio_recorder:
                logger.debug("设置 is_tts_playing = False")
                self.audio_recorder.is_tts_playing = False
            logger.debug("TTS播放流程已结束")

        if hasattr(self.tts, "is_interrupted"):
            return bool(self.tts.is_interrupted)
        return bool(getattr(self.tts, "_interrupted", None) and self.tts._interrupted.is_set())
