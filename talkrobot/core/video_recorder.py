"""
视频录制模块
按热键切换摄像头录像，并在录制状态变化时通知外部回调。
"""
from __future__ import annotations

import os
import queue
import shutil
import subprocess
import threading
import time
import wave
from datetime import datetime
from typing import Callable, Optional

import cv2
import numpy as np
from loguru import logger
from pynput import keyboard


class VideoRecorder:
    """按键触发的视频录制器。"""

    def __init__(
        self,
        camera_index: int,
        output_dir: str,
        toggle_key: str = "r",
        fps: float = 25.0,
        audio_enabled: bool = True,
        audio_sample_rate: int = 16000,
        audio_channels: int = 1,
        frame_getter: Optional[Callable[[], object]] = None,
        on_recording_changed: Optional[Callable[[bool], None]] = None,
    ):
        self.camera_index = camera_index
        self.output_dir = os.path.abspath(output_dir)
        self.toggle_key = self._normalize_key_char(toggle_key, "r")
        self.toggle_key_label = self._format_key_label(self.toggle_key)
        self.fps = max(1.0, float(fps or 25.0))
        self.audio_enabled = bool(audio_enabled)
        self.audio_sample_rate = int(audio_sample_rate or 16000)
        self.audio_channels = max(1, int(audio_channels or 1))
        self.frame_getter = frame_getter
        self.on_recording_changed = on_recording_changed

        self._keyboard_listener = None
        self._record_thread = None
        self._audio_thread = None
        self._stop_event = threading.Event()
        self._recording_event = threading.Event()
        self._audio_stop_event = threading.Event()
        self._toggle_lock = threading.Lock()
        self._cap = None
        self._writer = None
        self._audio_stream = None
        self._audio_queue = None
        self._current_path = None
        self._temp_video_path = None
        self._temp_audio_path = None
        self._audio_chunk_count = 0
        self._last_toggle_time = 0.0
        self._toggle_debounce_seconds = 0.35

    @staticmethod
    def _normalize_key_char(key: str, fallback: str) -> str:
        value = str(key or "").strip().lower()
        if not value:
            return fallback
        aliases = {
            "return": "enter",
            "newline": "enter",
            "esc": "escape",
            "spacebar": "space",
        }
        value = aliases.get(value, value)
        if value in {"enter", "space", "tab", "escape"}:
            return value
        return value[0]

    @staticmethod
    def _format_key_label(key_value: str) -> str:
        labels = {
            "enter": "ENTER",
            "space": "SPACE",
            "tab": "TAB",
            "escape": "ESC",
        }
        return labels.get(key_value, key_value.upper())

    @staticmethod
    def _is_key_pressed(key, expected_key: str) -> bool:
        key_name = {
            "enter": "enter",
            "space": "space",
            "tab": "tab",
            "escape": "esc",
        }.get(expected_key)

        if key_name:
            try:
                if key == getattr(keyboard.Key, key_name):
                    return True
            except Exception:
                pass

        if expected_key == "space":
            try:
                if getattr(key, "char", None) == " ":
                    return True
            except Exception:
                pass

        try:
            return getattr(key, "char", None) == expected_key
        except Exception:
            return False

    @property
    def is_recording(self) -> bool:
        return self._recording_event.is_set()

    def start(self) -> None:
        """启动按键监听。"""
        if self._keyboard_listener is not None:
            return

        os.makedirs(self.output_dir, exist_ok=True)
        self._stop_event.clear()
        self._keyboard_listener = keyboard.Listener(on_press=self._on_press, daemon=True)
        self._keyboard_listener.start()
        logger.info(
            f"视频录制热键监听已启动 (按 {self.toggle_key_label} 开始/结束录像，保存目录: {self.output_dir})"
        )
        print(f"   - 按 '{self.toggle_key_label}' 键开始/结束视频录制")

    def stop(self) -> None:
        """停止按键监听和正在进行的录像。"""
        self._stop_event.set()
        if self.is_recording:
            self._stop_recording()

        if self._keyboard_listener is not None:
            try:
                self._keyboard_listener.stop()
            except Exception as e:
                logger.warning(f"停止视频录制键盘监听器异常: {e}")
            finally:
                self._keyboard_listener = None

        if self._record_thread and self._record_thread.is_alive():
            self._record_thread.join(timeout=3.0)
        self._record_thread = None
        self._stop_audio_recording()
        self._release_capture()
        logger.info("视频录制器已停止")

    def _on_press(self, key) -> None:
        try:
            if self._is_key_pressed(key, self.toggle_key):
                self.toggle_recording()
        except Exception as e:
            logger.debug(f"视频录制按键监听异常: {e}")

    def toggle_recording(self) -> None:
        with self._toggle_lock:
            now = time.monotonic()
            if now - self._last_toggle_time < self._toggle_debounce_seconds:
                return
            self._last_toggle_time = now
            if self.is_recording:
                self._stop_recording()
            else:
                self._start_recording()

    def _start_recording(self) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._current_path = os.path.join(self.output_dir, f"record_{timestamp}.mp4")
        self._temp_video_path = os.path.join(self.output_dir, f".record_{timestamp}.video.mp4")
        self._temp_audio_path = os.path.join(self.output_dir, f".record_{timestamp}.audio.wav")
        self._audio_chunk_count = 0
        self._recording_event.set()
        self._notify_recording_changed(True)

        self._start_audio_recording()
        self._record_thread = threading.Thread(target=self._record_loop, daemon=True)
        self._record_thread.start()
        print(f"\n🔴 视频录制开始: {self._current_path}")
        logger.info(f"视频录制开始: {self._current_path}")

    def _stop_recording(self) -> None:
        self._recording_event.clear()
        if self._record_thread and self._record_thread.is_alive() and self._record_thread != threading.current_thread():
            self._record_thread.join(timeout=3.0)
        self._record_thread = None
        self._release_writer()
        self._stop_audio_recording()
        self._finalize_recording()
        self._notify_recording_changed(False)
        if self._current_path:
            print(f"\n✅ 视频录制结束: {self._current_path}")
            logger.info(f"视频录制结束: {self._current_path}")
        self._current_path = None
        self._temp_video_path = None
        self._temp_audio_path = None

    def _notify_recording_changed(self, recording: bool) -> None:
        if self.on_recording_changed is None:
            return
        try:
            self.on_recording_changed(bool(recording))
        except Exception as e:
            logger.warning(f"通知录制状态变化失败: {e}")

    def _get_frame(self):
        if self.frame_getter is not None:
            try:
                frame = self.frame_getter()
                if frame is not None:
                    return frame
            except Exception as e:
                logger.debug(f"读取共享摄像头帧失败: {e}")

        if self._cap is None:
            self._cap = cv2.VideoCapture(self.camera_index)
            if not self._cap.isOpened():
                raise RuntimeError(f"无法打开录像摄像头: index={self.camera_index}")

        ok, frame = self._cap.read()
        if not ok:
            return None
        return frame

    def _record_loop(self) -> None:
        frame_interval = 1.0 / self.fps
        next_frame_time = time.perf_counter()

        try:
            while self.is_recording and not self._stop_event.is_set():
                frame = self._get_frame()
                if frame is None:
                    time.sleep(0.03)
                    continue

                height, width = frame.shape[:2]
                if self._writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    self._writer = cv2.VideoWriter(self._temp_video_path, fourcc, self.fps, (width, height))
                    if not self._writer.isOpened():
                        raise RuntimeError(f"无法创建视频文件: {self._temp_video_path}")

                self._writer.write(frame)

                next_frame_time += frame_interval
                sleep_time = next_frame_time - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    next_frame_time = time.perf_counter()
        except Exception as e:
            logger.error(f"视频录制失败: {e}")
            print(f"\n❌ 视频录制失败: {e}")
            self._recording_event.clear()
            self._notify_recording_changed(False)
        finally:
            self._release_writer()

    def _start_audio_recording(self) -> None:
        if not self.audio_enabled:
            return

        self._audio_stop_event.clear()
        self._audio_queue = queue.Queue()

        def audio_callback(indata, frames, time_info, status):
            if status:
                logger.debug(f"视频录制音频状态: {status}")
            if self.is_recording:
                try:
                    self._audio_queue.put_nowait(indata.copy())
                    self._audio_chunk_count += 1
                except Exception as e:
                    logger.debug(f"缓存录像音频失败: {e}")

        try:
            import sounddevice as sd

            self._audio_thread = threading.Thread(target=self._audio_writer_loop, daemon=True)
            self._audio_thread.start()
            self._audio_stream = sd.InputStream(
                samplerate=self.audio_sample_rate,
                channels=self.audio_channels,
                dtype="float32",
                callback=audio_callback,
            )
            self._audio_stream.start()
            logger.info("视频录制音频采集已启动")
        except Exception as e:
            logger.warning(f"视频录制音频采集启动失败，将录制无声视频: {e}")
            self._stop_audio_recording()

    def _stop_audio_recording(self) -> None:
        self._audio_stop_event.set()

        if self._audio_stream is not None:
            try:
                if self._audio_stream.active:
                    self._audio_stream.stop()
            except Exception as e:
                logger.warning(f"停止录像音频流异常: {e}")
            finally:
                try:
                    self._audio_stream.close()
                except Exception as e:
                    logger.warning(f"关闭录像音频流异常: {e}")
                self._audio_stream = None

        if self._audio_thread and self._audio_thread.is_alive():
            self._audio_thread.join(timeout=3.0)
        self._audio_thread = None
        self._audio_queue = None

    def _audio_writer_loop(self) -> None:
        if not self._temp_audio_path:
            return

        try:
            with wave.open(self._temp_audio_path, "wb") as audio_file:
                audio_file.setnchannels(self.audio_channels)
                audio_file.setsampwidth(2)
                audio_file.setframerate(self.audio_sample_rate)
                while not self._audio_stop_event.is_set() or (
                    self._audio_queue is not None and not self._audio_queue.empty()
                ):
                    try:
                        chunk = self._audio_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    audio = np.asarray(chunk, dtype=np.float32)
                    audio = np.clip(audio, -1.0, 1.0)
                    audio_i16 = (audio * 32767.0).astype(np.int16)
                    audio_file.writeframes(audio_i16.tobytes())
        except Exception as e:
            logger.warning(f"写入录像音频失败，将录制无声视频: {e}")

    def _finalize_recording(self) -> None:
        if not self._current_path or not self._temp_video_path:
            return
        if not os.path.exists(self._temp_video_path):
            return

        has_audio = (
            self.audio_enabled
            and self._audio_chunk_count > 0
            and self._temp_audio_path
            and os.path.exists(self._temp_audio_path)
            and os.path.getsize(self._temp_audio_path) > 44
        )

        if not has_audio:
            logger.warning("未采集到录像音频，保留无声视频")
            shutil.move(self._temp_video_path, self._current_path)
            self._cleanup_temp_audio()
            return

        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            self._temp_video_path,
            "-i",
            self._temp_audio_path,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            self._current_path,
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            os.remove(self._temp_video_path)
            self._cleanup_temp_audio()
            logger.info("视频音频合成完成")
        except Exception as e:
            logger.warning(f"视频音频合成失败，保留无声视频: {e}")
            shutil.move(self._temp_video_path, self._current_path)
            self._cleanup_temp_audio()

    def _cleanup_temp_audio(self) -> None:
        if self._temp_audio_path and os.path.exists(self._temp_audio_path):
            try:
                os.remove(self._temp_audio_path)
            except Exception as e:
                logger.debug(f"清理临时音频文件失败: {e}")

    def _release_writer(self) -> None:
        if self._writer is not None:
            try:
                self._writer.release()
            except Exception as e:
                logger.warning(f"释放视频写入器失败: {e}")
            finally:
                self._writer = None

    def _release_capture(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception as e:
                logger.warning(f"释放录像摄像头失败: {e}")
            finally:
                self._cap = None
