"""
音频录制核心模块
负责录制音频并触发ASR
支持两种模式:
  - push:       按住 Q 键录音，松开结束
  - continuous:  持续监听，基于 Silero VAD 自动检测语音段
"""
import sys
import time as _time
import threading
import shutil
import numpy as np
import sounddevice as sd
from pynput import keyboard
from loguru import logger
from typing import Callable


class VADIterator:
    """A small helper that wraps Silero's get_speech_timestamps into a
    per-chunk callable iterator which emits simple start/end events.

    Usage:
        vad = VADIterator(model, sampling_rate=16000, silence_duration=1.5)
        event_list = vad(chunk_numpy_array, return_seconds=False)
        # event_list may be [] or like [{'start': 123}] or [{'end': True}]
    """

    def __init__(self, model, sampling_rate: int = 16000, silence_duration: float = 1.5):
        self.model = model
        self.sr = sampling_rate
        self.silence_samples_threshold = int(max(0.0, silence_duration) * sampling_rate)
        self.in_speech = False
        self._silent_samples = 0

    def reset(self):
        """Reset internal state (useful when flushing or skipping audio)."""
        self.in_speech = False
        self._silent_samples = 0
        try:
            # model may implement reset_states()
            self.model.reset_states()
        except Exception:
            pass

    def __call__(self, chunk_audio: np.ndarray, return_seconds: bool = False):
        """Process a single chunk (1-D float32 array).

        Returns a list of simple events. Each event is a dict:
            {'start': <sample_index_in_chunk>} or {'end': True}
        The caller can pass return_seconds=True to receive times in seconds.
        """
        # Defensive: ensure 1-D float32 mono array
        if chunk_audio is None or chunk_audio.size == 0:
            return []

        try:
            import torch
            from silero_vad import get_speech_timestamps
        except Exception:
            # can't run VAD here — return no events (caller should handle import errors earlier)
            return []

        # Ensure shape: 1-D array of samples
        if chunk_audio.ndim > 1:
            audio_flat = chunk_audio.flatten().astype(np.float32)
        else:
            audio_flat = chunk_audio.astype(np.float32)

        try:
            tensor = torch.from_numpy(audio_flat)
            timestamps = get_speech_timestamps(tensor, self.model, sampling_rate=self.sr, return_seconds=False)
        except Exception:
            return []

        events = []
        if timestamps:
            # Found speech inside this chunk
            # Reset silent counter
            self._silent_samples = 0
            if not self.in_speech:
                # Emit start at the first detected start sample (relative to this chunk)
                first = timestamps[0]
                if isinstance(first, dict):
                    start_sample = int(first.get('start', 0))
                elif isinstance(first, (list, tuple)) and len(first) >= 1:
                    start_sample = int(first[0])
                else:
                    start_sample = 0

                self.in_speech = True
                if return_seconds:
                    events.append({'start': start_sample / float(self.sr)})
                else:
                    events.append({'start': start_sample})
            # otherwise we're already in speech; update state only
        else:
            # No speech in this chunk
            if self.in_speech:
                # Accumulate silence samples and possibly emit end when threshold reached
                self._silent_samples += audio_flat.shape[0]
                if self.silence_samples_threshold > 0 and self._silent_samples >= self.silence_samples_threshold:
                    # End of utterance
                    self.in_speech = False
                    self._silent_samples = 0
                    events.append({'end': True if not return_seconds else True})

        return events


class AudioRecorder:
    """音频录制器"""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        listen_mode: str = "push",
        vad_check_interval: float = 0.25,
        pre_speech_duration: float = 0.25,
        silence_duration: float = 1.5,
        min_speech_duration: float = 0.3,
    ):
        """
        初始化录音器

        Args:
            sample_rate: 采样率
            channels: 通道数
            listen_mode: 监听模式 ("push" | "continuous")
            vad_check_interval: VAD 检测间隔（秒），每隔此时间检测一次语音 (continuous 模式)
            pre_speech_duration: VAD 检测到说话时，向前补偿的音频时长（秒）
            silence_duration: 静默多少秒后视为说话结束 (continuous 模式)
            min_speech_duration: 最短语音时长，过短丢弃 (continuous 模式)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.listen_mode = listen_mode

        # 通用状态
        self.is_recording = False
        self.audio_frames = []
        self.on_audio_complete: Callable = None

        # 外部可设置的标志: TTS 正在播放时置 True，用于 continuous 模式屏蔽自身语音
        self.is_tts_playing = False

        # continuous 模式参数
        self.vad_check_interval = vad_check_interval
        self.pre_speech_duration = max(0.0, pre_speech_duration)
        self.silence_duration = silence_duration
        self.min_speech_duration = min_speech_duration
        self._last_voice_time: float = 0.0
        self._speech_start_time: float = 0.0
        self._processing = False  # 正在处理上一段语音时不再采集新段
        self._pre_speech_samples = int(self.sample_rate * self.pre_speech_duration)
        self._pre_speech_audio = np.empty(0, dtype=np.float32)

        # continuous 模式: Silero VAD + 缓冲区
        self._vad_model = None
        self._chunk_buffer = []       # 音频回调写入的临时缓冲
        self._chunk_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._vad_thread: threading.Thread = None
        self._keyboard_listener = None
        self._stream = None

        if listen_mode == "continuous":
            self._init_vad()
        
        # VAD 状态栏相关
        self._current_rms: float = 0.0
        self._current_has_speech: bool = False

    # ------------------------------------------------------------------
    # VAD 实时状态栏 (原地刷新，不叠加)
    # ------------------------------------------------------------------
    def _print_vad_status(self, rms: float = 0.0, has_speech: bool = False):
        """在终端原地刷新 VAD 状态行"""
        if not sys.stdout.isatty():
            return

        if self._processing:
            status = "⏳ 处理中"
        elif self.is_tts_playing:
            status = "🔊 播放中"
        elif self.is_recording:
            duration = _time.time() - self._speech_start_time
            status = f"🔴 录音中 [{duration:.1f}s]"
        else:
            status = "🎙️  监听中"

        # 音量条
        bar_len = 20
        level = min(rms / 0.08, 1.0)  # 归一化，假设 0.08 为较大音量
        filled = int(level * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)

        speech_icon = "🗣️ " if has_speech else "   "

        line = f"{status} {speech_icon}|{bar}| RMS:{rms:.4f}"

        # 控制行宽，避免换行导致刷屏
        term_width = shutil.get_terminal_size((80, 20)).columns
        max_width = max(1, term_width - 1)
        if len(line) > max_width:
            line = line[:max_width]

        # 清行 + 回到行首后重写
        sys.stdout.write("\r\x1b[2K" + line)
        sys.stdout.flush()

    def _clear_vad_status(self):
        """清除状态行（在打印重要消息前调用）"""
        if not sys.stdout.isatty():
            return
        sys.stdout.write("\r\x1b[2K")
        sys.stdout.flush()

    def _append_pre_speech_audio(self, audio: np.ndarray):
        """维护说话前滚动缓冲（仅保留最近 pre_speech_duration 秒）。"""
        if self._pre_speech_samples <= 0 or audio.size == 0:
            return
        if self._pre_speech_audio.size == 0:
            merged = audio
        else:
            merged = np.concatenate([self._pre_speech_audio, audio])
        self._pre_speech_audio = merged[-self._pre_speech_samples:]

    def _consume_pre_speech_audio(self, current_audio: np.ndarray, speech_start_sample: int) -> np.ndarray:
        """根据当前检测到的起点，提取最多 pre_speech_duration 秒的前置音频。"""
        if self._pre_speech_samples <= 0:
            return np.empty(0, dtype=np.float32)

        clamped_start = max(0, min(int(speech_start_sample), len(current_audio)))
        current_prefix = current_audio[:clamped_start]

        if self._pre_speech_audio.size and current_prefix.size:
            merged = np.concatenate([self._pre_speech_audio, current_prefix])
        elif self._pre_speech_audio.size:
            merged = self._pre_speech_audio
        else:
            merged = current_prefix

        if merged.size == 0:
            return np.empty(0, dtype=np.float32)
        return merged[-self._pre_speech_samples:]

    # ------------------------------------------------------------------
    # Silero VAD 初始化
    # ------------------------------------------------------------------
    def _init_vad(self):
        """加载 Silero VAD 模型"""
        try:
            from silero_vad import load_silero_vad
            logger.info("正在加载 Silero VAD 模型...")
            self._vad_model = load_silero_vad()
            # wrap into VADIterator for per-chunk processing
            self._vad_iterator = VADIterator(self._vad_model, sampling_rate=self.sample_rate, silence_duration=self.silence_duration)
            logger.info("Silero VAD 模型加载完成")
        except ImportError:
            logger.error("未安装 silero-vad，请运行: pip install silero-vad")
            raise
        except Exception as e:
            logger.error(f"加载 Silero VAD 失败: {e}")
            raise

    # ------------------------------------------------------------------
    # 音频流回调 (两种模式共用)
    # ------------------------------------------------------------------
    def audio_callback(self, indata, frames, time, status):
        """音频流回调函数"""
        if status:
            logger.warning(f"录音状态: {status}")

        if self.listen_mode == "push":
            # push 模式: 只有手动按键时才采集
            if self.is_recording:
                self.audio_frames.append(indata.copy())
        else:
            # continuous 模式: 所有音频写入缓冲区，由 VAD 监控线程处理
            if not self.is_tts_playing and not self._processing:
                with self._chunk_lock:
                    self._chunk_buffer.append(indata.copy())

    # ------------------------------------------------------------------
    # continuous 模式: VAD 监控线程
    # ------------------------------------------------------------------
    def _vad_monitor_loop(self):
        """
        在独立线程中运行，每隔 vad_check_interval 秒从缓冲区取出音频，
        用 Silero VAD 检测是否有语音，驱动录音状态机。
        """
        logger.debug("VAD 监控线程已启动")

        while not self._stop_event.is_set():
            _time.sleep(self.vad_check_interval)

            # TTS 播放中或正在处理上一段 → 跳过并清空缓冲，不显示状态栏
            if self.is_tts_playing or self._processing:
                logger.debug(f"VAD跳过: is_tts_playing={self.is_tts_playing}, _processing={self._processing}")
                if self.is_recording:
                    self.is_recording = False
                    self.audio_frames = []
                with self._chunk_lock:
                    self._chunk_buffer = []
                self._pre_speech_audio = np.empty(0, dtype=np.float32)
                # 重置 iterator/model 状态
                try:
                    self._vad_iterator.reset()
                except Exception:
                    try:
                        self._vad_model.reset_states()
                    except Exception:
                        pass
                continue

            # 取出缓冲区中的所有音频块
            with self._chunk_lock:
                if not self._chunk_buffer:
                    self._print_vad_status(rms=0.0, has_speech=False)
                    continue
                current_chunks = self._chunk_buffer.copy()
                self._chunk_buffer = []

            logger.debug(f"VAD检测: 处理 {len(current_chunks)} 个音频块")

            # 合并为一维 float32 数组
            chunk_audio = np.concatenate(current_chunks, axis=0).flatten().astype(np.float32)

            # 计算当前音频块的 RMS
            rms = float(np.sqrt(np.mean(chunk_audio ** 2)))

            # Use VADIterator to get simple events
            try:
                events = self._vad_iterator(chunk_audio, return_seconds=False)
            except Exception as e:
                logger.warning(f"VAD 检测异常: {e}")
                events = []

            has_speech = any('start' in ev for ev in events) or self._vad_iterator.in_speech
            now = _time.time()

            logger.debug(f"VAD结果: events={events}, rms={rms:.4f}, is_recording={self.is_recording}")

            for ev in events:
                if 'start' in ev:
                    self._last_voice_time = now
                    if not self.is_recording:
                        # Begin recording, include pre-speech buffer
                        self.is_recording = True
                        self._speech_start_time = now
                        self.audio_frames = []

                        speech_start_sample = int(ev.get('start', 0))
                        pre_speech_audio = self._consume_pre_speech_audio(chunk_audio, speech_start_sample)
                        if pre_speech_audio.size > 0:
                            self.audio_frames.append(pre_speech_audio.reshape(-1, self.channels))

                        self._clear_vad_status()
                        print("🔴 检测到语音，正在录音...")
                        logger.debug("开始录音")
                    # always append current chunk when speech detected
                    self.audio_frames.extend(current_chunks)
                    self._pre_speech_audio = np.empty(0, dtype=np.float32)

                if 'end' in ev:
                    # end event from iterator: finalize
                    self.is_recording = False
                    speech_len = now - self._speech_start_time
                    if speech_len < self.min_speech_duration:
                        self._clear_vad_status()
                        print("⚠️ 语音过短，已忽略")
                        logger.debug(f"语音过短: {speech_len:.2f}s")
                        self.audio_frames = []
                        self._pre_speech_audio = np.empty(0, dtype=np.float32)
                        try:
                            self._vad_iterator.reset()
                        except Exception:
                            try:
                                self._vad_model.reset_states()
                            except Exception:
                                pass
                        continue

                    self._clear_vad_status()
                    print("✅ 录音结束")
                    logger.debug(f"录音结束，时长: {speech_len:.2f}s")
                    if self.audio_frames and self.on_audio_complete:
                        audio_data = np.concatenate(self.audio_frames, axis=0)
                        self._processing = True
                        try:
                            self._vad_iterator.reset()
                        except Exception:
                            try:
                                self._vad_model.reset_states()
                            except Exception:
                                pass
                        logger.debug("触发音频处理回调")
                        self.on_audio_complete(audio_data)

            # if no events and we're recording, keep appending and check timeout using last voice time
            if not events and self.is_recording:
                self.audio_frames.extend(current_chunks)
                silence_time = now - self._last_voice_time
                logger.debug(f"录音中，累计静默: {silence_time:.2f}s")
                if silence_time >= self.silence_duration:
                    # treat as end
                    self.is_recording = False
                    speech_len = now - self._speech_start_time
                    if speech_len < self.min_speech_duration:
                        self._clear_vad_status()
                        print("⚠️ 语音过短，已忽略")
                        logger.debug(f"语音过短: {speech_len:.2f}s")
                        self.audio_frames = []
                        self._pre_speech_audio = np.empty(0, dtype=np.float32)
                        try:
                            self._vad_iterator.reset()
                        except Exception:
                            try:
                                self._vad_model.reset_states()
                            except Exception:
                                pass
                    else:
                        self._clear_vad_status()
                        print("✅ 录音结束")
                        logger.debug(f"录音结束，时长: {speech_len:.2f}s")
                        if self.audio_frames and self.on_audio_complete:
                            audio_data = np.concatenate(self.audio_frames, axis=0)
                            self._processing = True
                            try:
                                self._vad_iterator.reset()
                            except Exception:
                                try:
                                    self._vad_model.reset_states()
                                except Exception:
                                    pass
                            logger.debug("触发音频处理回调")
                            self.on_audio_complete(audio_data)
            elif not events and not self.is_recording:
                # update pre-speech rolling buffer
                self._append_pre_speech_audio(chunk_audio)

            # 刷新状态栏
            self._print_vad_status(rms=rms, has_speech=has_speech)

        logger.debug("VAD 监控线程已退出")

    def notify_process_done(self):
        """对话处理完成后调用，允许继续采集下一段语音"""
        logger.debug("对话处理完成，重置 _processing 标志")
        self._processing = False

    def stop(self):
        """停止录音器并释放底层资源。"""
        logger.info("正在停止音频录制...")
        self._stop_event.set()
        self.is_recording = False

        with self._chunk_lock:
            self._chunk_buffer = []
        self.audio_frames = []

        if self._keyboard_listener:
            try:
                self._keyboard_listener.stop()
            except Exception as e:
                logger.warning(f"停止键盘监听器异常: {e}")
            finally:
                self._keyboard_listener = None

        if self._vad_thread and self._vad_thread.is_alive() and self._vad_thread != threading.current_thread():
            self._vad_thread.join(timeout=2.0)
        self._vad_thread = None

        if self._stream is not None:
            try:
                if self._stream.active:
                    self._stream.stop()
            except Exception as e:
                logger.warning(f"停止输入流异常: {e}")
            finally:
                try:
                    self._stream.close()
                except Exception as e:
                    logger.warning(f"关闭输入流异常: {e}")
                self._stream = None

        self._clear_vad_status()
        logger.info("音频录制已停止")

    # ------------------------------------------------------------------
    # push 模式: 键盘事件
    # ------------------------------------------------------------------
    def on_press(self, key):
        """键盘按下事件"""
        try:
            k = key.char if hasattr(key, 'char') else None
            if k == 'q' and not self.is_recording:
                self.is_recording = True
                self.audio_frames = []
                print("\n🔴 正在录音... (松开 'Q' 结束)", end='', flush=True)
        except Exception:
            pass

    def on_release(self, key):
        """键盘松开事件"""
        try:
            k = key.char if hasattr(key, 'char') else None
            if k == 'q' and self.is_recording:
                self.is_recording = False
                print("\n✅ 录音结束")

                if self.audio_frames and self.on_audio_complete:
                    # 合并音频数据
                    audio_data = np.concatenate(self.audio_frames, axis=0)
                    # 触发回调
                    self.on_audio_complete(audio_data)

        except Exception as e:
            logger.error(f"按键释放处理错误: {e}")

    # ------------------------------------------------------------------
    # 启动
    # ------------------------------------------------------------------
    def start(self, on_audio_complete: Callable):
        """
        启动录音器

        Args:
            on_audio_complete: 录音完成时的回调函数
        """
        self.on_audio_complete = on_audio_complete

        logger.info(f"启动音频录制 (模式: {self.listen_mode})...")
        print("\n" + "=" * 50)
        print("🎤 对话机器人已启动")
        print("=" * 50)

        if self.listen_mode == "push":
            print("📌 操作说明 [按键模式]:")
            print("   - 按住 'Q' 键说话")
            print("   - 松开 'Q' 键结束")
            print("   - 按 'S' 键打断语音播放")
            print("   - 按 Ctrl+C 退出程序")
        else:
            print("📌 操作说明 [持续监听模式 - Silero VAD]:")
            print("   - 直接说话即可，系统自动检测语音")
            print("   - 说“你好”进入响应模式，机器人才会回复")
            print("   - 说“再见”退出响应模式，后续语音将忽略")
            print("   - 停顿超过 {:.1f} 秒视为说话结束".format(self.silence_duration))
            print("   - 机器人说话时会自动屏蔽麦克风")
            print("   - 按 'S' 键打断语音播放")
            print("   - 按 Ctrl+C 退出程序")

        print("=" * 50 + "\n")

        self._stop_event.clear()
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self.audio_callback,
        )
        self._stream.start()

        try:
            if self.listen_mode == "push":
                self._keyboard_listener = keyboard.Listener(
                    on_press=self.on_press,
                    on_release=self.on_release,
                )
                self._keyboard_listener.start()
                while not self._stop_event.wait(0.1):
                    pass
            else:
                # continuous 模式: 启动 VAD 监控线程，主线程等待
                self._vad_thread = threading.Thread(target=self._vad_monitor_loop)
                self._vad_thread.start()
                while not self._stop_event.wait(0.1):
                    pass
        except KeyboardInterrupt:
            self.stop()
            raise
        finally:
            self.stop()