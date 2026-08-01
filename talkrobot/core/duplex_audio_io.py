"""
Full-duplex audio IO with WebRTC APM echo cancellation.
"""
import queue
import threading
from typing import Iterable, Optional

import numpy as np
from loguru import logger


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).flatten()
    if source_rate == target_rate or len(audio) == 0:
        return audio

    old_index = np.linspace(0, 1, num=len(audio), endpoint=False)
    new_length = int(round(len(audio) * target_rate / source_rate))
    new_index = np.linspace(0, 1, num=new_length, endpoint=False)
    return np.interp(new_index, old_index, audio).astype(np.float32)


def float_to_int16(audio: np.ndarray) -> np.ndarray:
    return (np.clip(audio, -1.0, 1.0) * np.iinfo(np.int16).max).astype(np.int16)


def int16_to_float(audio: np.ndarray) -> np.ndarray:
    return np.asarray(audio, dtype=np.float32) / np.iinfo(np.int16).max


class DuplexAudioIO:
    """One sounddevice stream for playback, mic capture, and AEC."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 10,
        stream_delay_ms: int = 120,
        latency: str = "high",
        high_pass_filter: bool = True,
        noise_suppression: bool = False,
        auto_gain_control: bool = False,
        mic_queue_size: int = 400,
    ) -> None:
        import sounddevice as sd
        from livekit import rtc
        from livekit.rtc.apm import AudioProcessingModule

        self.sd = sd
        self.rtc = rtc
        self.sample_rate = int(sample_rate)
        self.frame_samples = int(round(self.sample_rate * int(frame_ms) / 1000))
        if self.frame_samples <= 0:
            raise ValueError("AEC frame size must be positive.")

        self.stream_delay_ms = int(stream_delay_ms)
        self.apm = AudioProcessingModule(
            echo_cancellation=True,
            noise_suppression=bool(noise_suppression),
            high_pass_filter=bool(high_pass_filter),
            auto_gain_control=bool(auto_gain_control),
        )
        self.apm.set_stream_delay_ms(self.stream_delay_ms)

        self._mic_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=int(mic_queue_size))
        self._lock = threading.Condition()
        self._playback_buffer = np.empty(0, dtype=np.int16)
        self._discard_input = False
        self._closed = False
        self.status_messages: list[str] = []
        self.stream = self.sd.Stream(
            samplerate=self.sample_rate,
            blocksize=self.frame_samples,
            dtype="int16",
            channels=1,
            latency=latency,
            callback=self._callback,
        )

    def start(self) -> None:
        self.stream.start()
        logger.info(
            f"Duplex AEC 音频流已启动: sample_rate={self.sample_rate}, "
            f"frame_samples={self.frame_samples}, delay={self.stream_delay_ms}ms"
        )

    def close(self) -> None:
        self._closed = True
        with self._lock:
            self._lock.notify_all()
        try:
            self.stream.stop()
        finally:
            self.stream.close()

    def mic_chunks(self) -> Iterable[np.ndarray]:
        while not self._closed:
            yield self._mic_queue.get()

    def get_mic_chunk(self, timeout: float = 0.05) -> Optional[np.ndarray]:
        try:
            return self._mic_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def play_audio(
        self,
        audio_chunks: list[np.ndarray],
        sample_rate: int,
        discard_input: bool = True,
    ) -> None:
        self.start_playback(audio_chunks, sample_rate, discard_input=discard_input)
        self.wait_for_playback(drain_input=True)

    def start_playback(
        self,
        audio_chunks: list[np.ndarray],
        sample_rate: int,
        discard_input: bool = True,
    ) -> None:
        if not audio_chunks:
            return

        audio = np.concatenate([np.asarray(chunk, dtype=np.float32).flatten() for chunk in audio_chunks])
        audio = resample_audio(audio, sample_rate, self.sample_rate)
        playback = float_to_int16(audio)

        with self._lock:
            self._discard_input = bool(discard_input)
            self._drain_mic_queue()
            self._playback_buffer = np.concatenate([self._playback_buffer, playback])
            self._lock.notify_all()

    def wait_for_playback(self, drain_input: bool = True) -> None:
        while True:
            with self._lock:
                if len(self._playback_buffer) == 0:
                    break
            self.sd.sleep(20)

        self.sd.sleep(max(50, self.stream_delay_ms))
        with self._lock:
            self._discard_input = False
            if drain_input:
                self._drain_mic_queue()

    def is_playing(self) -> bool:
        with self._lock:
            return len(self._playback_buffer) > 0

    def stop_playback(self, drain_input: bool = False) -> None:
        with self._lock:
            self._playback_buffer = np.empty(0, dtype=np.int16)
            self._discard_input = False
            if drain_input:
                self._drain_mic_queue()

    def _callback(self, indata, outdata, frames, _time_info, status) -> None:
        if status:
            self.status_messages.append(str(status))

        if frames != self.frame_samples:
            raise self.sd.CallbackAbort(f"Expected {self.frame_samples} frames, got {frames}")

        with self._lock:
            take = min(frames, len(self._playback_buffer))
            far_block = np.zeros(frames, dtype=np.int16)
            if take:
                far_block[:take] = self._playback_buffer[:take]
                self._playback_buffer = self._playback_buffer[take:]
            discard_input = self._discard_input

        outdata[:, 0] = far_block
        raw_block = np.asarray(indata[:, 0], dtype=np.int16).copy()

        far_frame = self._make_audio_frame(far_block)
        near_frame = self._make_audio_frame(raw_block)
        self.apm.process_reverse_stream(far_frame)
        self.apm.process_stream(near_frame)

        if discard_input:
            return

        processed = np.asarray(near_frame.data, dtype=np.int16).copy()
        try:
            self._mic_queue.put_nowait(int16_to_float(processed))
        except queue.Full:
            try:
                self._mic_queue.get_nowait()
            except queue.Empty:
                pass
            self._mic_queue.put_nowait(int16_to_float(processed))

    def _make_audio_frame(self, samples: np.ndarray):
        audio = np.asarray(samples, dtype=np.int16)
        return self.rtc.AudioFrame(
            data=bytearray(audio.tobytes()),
            sample_rate=self.sample_rate,
            num_channels=1,
            samples_per_channel=len(audio),
        )

    def _drain_mic_queue(self) -> None:
        while True:
            try:
                self._mic_queue.get_nowait()
            except queue.Empty:
                return
