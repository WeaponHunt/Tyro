"""
VAD + ASR + EOT turn detector for duplex voice mode.
"""
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional

import numpy as np
from loguru import logger


class TurnState(str, Enum):
    IDLE = "idle"
    SPEECH = "speech"
    EXTRA_WAIT = "extra_wait"


@dataclass
class DuplexTurnConfig:
    sample_rate: int = 16000
    vad_mode: int = 2
    vad_activation_threshold: float = 0.5
    short_silence_seconds: float = 0.35
    extra_wait_seconds: float = 2.0
    pre_speech_padding_seconds: float = 0.4
    min_segment_seconds: float = 0.1
    verbose: bool = True


@dataclass
class InterruptionConfig:
    vad_activation_threshold: float = 0.75
    min_speech_seconds: float = 0.25
    max_interruption_seconds: float = 30.0


@dataclass
class DetectedTurn:
    user_text: str
    eot_probability: float
    finish_reason: str


class WebRTCSpeechVAD:
    """Frame-level wrapper around WebRTC VAD.

    WebRTC VAD returns a speech flag rather than a probability. We expose it as
    1.0/0.0 so the rest of the turn state machine can stay threshold based.
    """

    def __init__(self, sample_rate: int = 16000, mode: int = 2) -> None:
        if sample_rate != 16000:
            raise ValueError("WebRTC duplex VAD currently expects 16 kHz audio.")
        import webrtcvad

        self.vad = webrtcvad.Vad(max(0, min(3, int(mode))))
        self.sample_rate = int(sample_rate)
        self.window_size_samples = int(self.sample_rate * 30 / 1000)

    def reset(self) -> None:
        return

    def __call__(self, audio_window: np.ndarray) -> float:
        audio = np.asarray(audio_window, dtype=np.float32).flatten()
        if len(audio) != self.window_size_samples:
            if len(audio) > self.window_size_samples:
                audio = audio[: self.window_size_samples]
            else:
                audio = np.pad(audio, (0, self.window_size_samples - len(audio)))
        pcm = (np.clip(audio, -1.0, 1.0) * np.iinfo(np.int16).max).astype(np.int16)
        return 1.0 if self.vad.is_speech(pcm.tobytes(), self.sample_rate) else 0.0


class InterruptionGate:
    """Confirm that AEC-processed mic audio contains user speech during TTS."""

    def __init__(
        self,
        interruption_config: InterruptionConfig,
        turn_config: DuplexTurnConfig,
    ) -> None:
        self.threshold = float(interruption_config.vad_activation_threshold)
        self.min_speech_seconds = float(interruption_config.min_speech_seconds)
        self.short_silence_seconds = float(turn_config.short_silence_seconds)
        self.vad = WebRTCSpeechVAD(
            sample_rate=turn_config.sample_rate,
            mode=turn_config.vad_mode,
        )
        self.window_samples = self.vad.window_size_samples
        self.window_seconds = self.window_samples / float(turn_config.sample_rate)
        max_prefix_windows = max(
            0,
            int(round(turn_config.pre_speech_padding_seconds / self.window_seconds)),
        )
        self.pre_speech_buffer: deque[np.ndarray] = deque(maxlen=max_prefix_windows)
        self.tail = np.empty(0, dtype=np.float32)
        self.candidate_chunks: list[np.ndarray] = []
        self.consecutive_speech_seconds = 0.0
        self.silence_seconds = 0.0

    def process_chunk(self, chunk: np.ndarray) -> Optional[list[np.ndarray]]:
        for window in self._iter_windows(chunk):
            confirmed = self._process_window(window)
            if confirmed is not None:
                return confirmed
        return None

    def reset(self) -> None:
        self.vad.reset()
        self.pre_speech_buffer.clear()
        self.tail = np.empty(0, dtype=np.float32)
        self.candidate_chunks = []
        self.consecutive_speech_seconds = 0.0
        self.silence_seconds = 0.0

    def _iter_windows(self, chunk: np.ndarray) -> Iterable[np.ndarray]:
        audio = np.asarray(chunk, dtype=np.float32).flatten()
        if len(audio) == 0:
            return
        if len(self.tail) > 0:
            audio = np.concatenate([self.tail, audio])
            self.tail = np.empty(0, dtype=np.float32)

        usable = (len(audio) // self.window_samples) * self.window_samples
        for start in range(0, usable, self.window_samples):
            yield audio[start : start + self.window_samples]

        if usable < len(audio):
            self.tail = audio[usable:]

    def _process_window(self, window: np.ndarray) -> Optional[list[np.ndarray]]:
        probability = self.vad(window)
        has_speech = probability >= self.threshold

        if has_speech:
            if not self.candidate_chunks:
                self.candidate_chunks = [item.copy() for item in self.pre_speech_buffer]
                self.pre_speech_buffer.clear()
            self.candidate_chunks.append(window.copy())
            self.consecutive_speech_seconds += self.window_seconds
            self.silence_seconds = 0.0
            if self.consecutive_speech_seconds >= self.min_speech_seconds:
                confirmed = [item.copy() for item in self.candidate_chunks]
                self.reset()
                return confirmed
            return None

        if self.candidate_chunks:
            self.candidate_chunks.append(window.copy())
            self.consecutive_speech_seconds = 0.0
            self.silence_seconds += self.window_seconds
            if self.silence_seconds >= self.short_silence_seconds:
                self.reset()
            return None

        self.pre_speech_buffer.append(window.copy())
        return None


class DuplexTurnDetector:
    """Accumulate speech until EOT says a user turn is complete."""

    def __init__(self, config: DuplexTurnConfig, asr_module, eot_module) -> None:
        self.config = config
        self.asr = asr_module
        self.eot = eot_module
        self.vad = WebRTCSpeechVAD(sample_rate=config.sample_rate, mode=config.vad_mode)
        self.window_samples = self.vad.window_size_samples
        self.window_seconds = self.window_samples / float(config.sample_rate)
        self.pre_speech_max_windows = max(
            0,
            int(round(config.pre_speech_padding_seconds / self.window_seconds)),
        )
        self.total_samples_processed = 0
        self.asr_segment_index = 0
        self.reset_turn()

    def reset_turn(self) -> None:
        self.vad.reset()
        self.state = TurnState.IDLE
        self.audio_buffer: list[np.ndarray] = []
        self.turn_audio_buffer: list[np.ndarray] = []
        self.turn_audio_start_sample: Optional[int] = None
        self.pre_speech_buffer: deque[tuple[int, np.ndarray]] = deque(
            maxlen=self.pre_speech_max_windows
        )
        self.audio_buffer_start_sample: Optional[int] = None
        self.detected_speech_start_sample: Optional[int] = None
        self.tail = np.empty(0, dtype=np.float32)
        self.accumulated_text = ""
        self.last_eot_probability = 0.0
        self.silence_seconds = 0.0
        self.extra_wait_seconds = 0.0

    def process_audio_chunks(
        self,
        chunks: Iterable[np.ndarray],
        finalize_at_end: bool = True,
    ) -> Optional[DetectedTurn]:
        for chunk in chunks:
            for window in self._iter_windows(chunk):
                window_start_sample = self.total_samples_processed
                result = self._handle_window(window, window_start_sample)
                self.total_samples_processed += len(window)
                if result is not None:
                    return result

        if finalize_at_end:
            return self.finish_stream()
        return None

    def finish_stream(self) -> Optional[DetectedTurn]:
        if self.state == TurnState.SPEECH and self.audio_buffer:
            self._run_asr_on_current_segment()
        if self.accumulated_text:
            return self._finalize("stream_end")
        self.reset_turn()
        return None

    def _iter_windows(self, chunk: np.ndarray) -> Iterable[np.ndarray]:
        audio = np.asarray(chunk, dtype=np.float32).flatten()
        if len(audio) == 0:
            return

        if len(self.tail) > 0:
            audio = np.concatenate([self.tail, audio])
            self.tail = np.empty(0, dtype=np.float32)

        usable = (len(audio) // self.window_samples) * self.window_samples
        for start in range(0, usable, self.window_samples):
            yield audio[start : start + self.window_samples]

        if usable < len(audio):
            self.tail = audio[usable:]

    def _handle_window(
        self,
        window: np.ndarray,
        window_start_sample: int,
    ) -> Optional[DetectedTurn]:
        probability = self.vad(window)
        has_speech = probability >= self.config.vad_activation_threshold

        if self.state == TurnState.IDLE:
            if has_speech:
                self._start_audio_buffer(window, window_start_sample)
                self._log(f"VAD: speech start p={probability:.3f}")
                self.state = TurnState.SPEECH
                self.silence_seconds = 0.0
            else:
                self._remember_pre_speech_window(window, window_start_sample)
            return None

        if self.state == TurnState.SPEECH:
            self.audio_buffer.append(window.copy())
            if has_speech:
                self.silence_seconds = 0.0
            else:
                self.silence_seconds += self.window_seconds

            if self.silence_seconds >= self.config.short_silence_seconds:
                segment_text = self._run_asr_on_current_segment()
                if segment_text:
                    prediction = self.eot.predict(self.accumulated_text)
                    self.last_eot_probability = prediction.probability
                    self._log(
                        f"EOT: p={prediction.probability:.3f}, "
                        f"end={prediction.is_end}, text={self.accumulated_text}"
                    )
                    if prediction.is_end:
                        return self._finalize("eot_high")

                self.state = TurnState.EXTRA_WAIT
                self.extra_wait_seconds = 0.0
                self.silence_seconds = 0.0
            return None

        if self.state == TurnState.EXTRA_WAIT:
            if has_speech:
                self._start_audio_buffer(window, window_start_sample)
                self._log(f"VAD: speech resumed p={probability:.3f}")
                self.state = TurnState.SPEECH
                self.silence_seconds = 0.0
                self.extra_wait_seconds = 0.0
                return None

            self._append_turn_audio(window, window_start_sample)
            self._remember_pre_speech_window(window, window_start_sample)
            self.extra_wait_seconds += self.window_seconds
            if self.extra_wait_seconds >= self.config.extra_wait_seconds:
                if self.accumulated_text:
                    return self._finalize("extra_wait_timeout")
                self.reset_turn()
            return None

        return None

    def _start_audio_buffer(self, window: np.ndarray, window_start_sample: int) -> None:
        if self.turn_audio_buffer:
            self.audio_buffer_start_sample = window_start_sample
            self.detected_speech_start_sample = window_start_sample
            self.audio_buffer = [window.copy()]
            self.pre_speech_buffer.clear()
            return

        prefix_windows = list(self.pre_speech_buffer)
        self.pre_speech_buffer.clear()

        if prefix_windows:
            self.audio_buffer_start_sample = prefix_windows[0][0]
            self.detected_speech_start_sample = window_start_sample
            self.audio_buffer = [buffered.copy() for _start, buffered in prefix_windows]
            self.audio_buffer.append(window.copy())
            return

        self.audio_buffer_start_sample = window_start_sample
        self.detected_speech_start_sample = window_start_sample
        self.audio_buffer = [window.copy()]

    def _remember_pre_speech_window(self, window: np.ndarray, window_start_sample: int) -> None:
        if self.pre_speech_max_windows <= 0:
            return
        self.pre_speech_buffer.append((window_start_sample, window.copy()))

    def _append_turn_audio(self, audio_data: np.ndarray, start_sample: int) -> None:
        if self.turn_audio_start_sample is None:
            self.turn_audio_start_sample = start_sample
        self.turn_audio_buffer.append(audio_data.copy())

    def _run_asr_on_current_segment(self) -> str:
        if not self.audio_buffer:
            return ""

        new_audio_data = np.concatenate(self.audio_buffer).astype(np.float32)
        start_sample = self.audio_buffer_start_sample
        self.audio_buffer = []
        self.audio_buffer_start_sample = None
        self.detected_speech_start_sample = None
        if len(new_audio_data) < int(self.config.min_segment_seconds * self.config.sample_rate):
            return ""

        if start_sample is None:
            start_sample = max(0, self.total_samples_processed - len(new_audio_data))

        self._append_turn_audio(new_audio_data, start_sample)
        audio_data = np.concatenate(self.turn_audio_buffer).astype(np.float32)
        self._log(
            "ASR audio: "
            f"pass={self.asr_segment_index}, "
            f"duration={len(audio_data) / self.config.sample_rate:.3f}s, "
            f"samples={len(audio_data)}"
        )
        self.asr_segment_index += 1

        text = self.asr.transcribe(audio_data).strip()
        self._log(f"ASR: {text or '<empty>'}")
        if text:
            self.accumulated_text = text
        return text

    def _finalize(self, reason: str) -> DetectedTurn:
        user_text = self.accumulated_text.strip()
        result = DetectedTurn(
            user_text=user_text,
            eot_probability=self.last_eot_probability,
            finish_reason=reason,
        )
        self.reset_turn()
        return result

    def _log(self, message: str) -> None:
        if self.config.verbose:
            logger.info(message)
