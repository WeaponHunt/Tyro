"""
音频录制核心模块
负责录制音频并触发ASR
支持两种模式:
  - push:       按住 Q 键录音，松开结束
  - continuous:  持续监听，基于 Silero VAD 自动检测语音段
"""
import time as _time
import threading
import numpy as np
import sounddevice as sd
from pynput import keyboard
from loguru import logger
from typing import Callable


class AudioRecorder:
    """音频录制器"""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        listen_mode: str = "push",
        vad_check_interval: float = 0.25,
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
        self.silence_duration = silence_duration
        self.min_speech_duration = min_speech_duration
        self._last_voice_time: float = 0.0
        self._speech_start_time: float = 0.0
        self._processing = False  # 正在处理上一段语音时不再采集新段

        # continuous 模式: Silero VAD + 缓冲区
        self._vad_model = None
        self._chunk_buffer = []       # 音频回调写入的临时缓冲
        self._chunk_lock = threading.Lock()
        self._stop_event = threading.Event()

        if listen_mode == "continuous":
            self._init_vad()

    # ------------------------------------------------------------------
    # Silero VAD 初始化
    # ------------------------------------------------------------------
    def _init_vad(self):
        """加载 Silero VAD 模型"""
        try:
            from silero_vad import load_silero_vad
            logger.info("正在加载 Silero VAD 模型...")
            self._vad_model = load_silero_vad()
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
        import torch
        from silero_vad import get_speech_timestamps

        while not self._stop_event.is_set():
            _time.sleep(self.vad_check_interval)

            # TTS 播放中或正在处理上一段 → 跳过并清空缓冲
            if self.is_tts_playing or self._processing:
                if self.is_recording:
                    self.is_recording = False
                    self.audio_frames = []
                with self._chunk_lock:
                    self._chunk_buffer = []
                # 重置 VAD 内部状态
                self._vad_model.reset_states()
                continue

            # 取出缓冲区中的所有音频块
            with self._chunk_lock:
                if not self._chunk_buffer:
                    continue
                current_chunks = self._chunk_buffer.copy()
                self._chunk_buffer = []

            # 合并为一维 float32 数组并转为 tensor
            chunk_audio = np.concatenate(current_chunks, axis=0).flatten().astype(np.float32)
            audio_tensor = torch.from_numpy(chunk_audio)

            # 用 Silero VAD 检测这段音频中是否存在语音
            try:
                timestamps = get_speech_timestamps(
                    audio_tensor,
                    self._vad_model,
                    sampling_rate=self.sample_rate,
                    return_seconds=False,
                )
            except Exception as e:
                logger.warning(f"VAD 检测异常: {e}")
                continue

            has_speech = len(timestamps) > 0
            now = _time.time()

            if has_speech:
                # 检测到语音
                self._last_voice_time = now
                if not self.is_recording:
                    # ✅ 开始录音，触发检测的这段音频也包含在内
                    self.is_recording = True
                    self._speech_start_time = now
                    self.audio_frames = []
                    print("\n🔴 检测到语音，正在录音...", end='', flush=True)
                self.audio_frames.extend(current_chunks)

            elif self.is_recording:
                # 未检测到语音但正在录音 → 积累静默段，等待超时
                self.audio_frames.extend(current_chunks)
                if now - self._last_voice_time >= self.silence_duration:
                    # 静默超时 → 结束本段
                    self.is_recording = False
                    speech_len = now - self._speech_start_time
                    if speech_len < self.min_speech_duration:
                        print("\n⚠️ 语音过短，已忽略")
                        self.audio_frames = []
                        self._vad_model.reset_states()
                        continue
                    print("\n✅ 录音结束")
                    if self.audio_frames and self.on_audio_complete:
                        audio_data = np.concatenate(self.audio_frames, axis=0)
                        self._processing = True
                        self._vad_model.reset_states()
                        self.on_audio_complete(audio_data)
            # else: 不在录音且无语音 → 丢弃，不做任何事

    def notify_process_done(self):
        """对话处理完成后调用，允许继续采集下一段语音"""
        self._processing = False

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
            print("   - 按 Ctrl+C 退出程序")
        else:
            print("📌 操作说明 [持续监听模式 - Silero VAD]:")
            print("   - 直接说话即可，系统自动检测语音")
            print("   - 停顿超过 {:.1f} 秒视为说话结束".format(self.silence_duration))
            print("   - 机器人说话时会自动屏蔽麦克风")
            print("   - 按 Ctrl+C 退出程序")

        print("=" * 50 + "\n")

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self.audio_callback,
        ):
            if self.listen_mode == "push":
                with keyboard.Listener(
                    on_press=self.on_press,
                    on_release=self.on_release,
                ) as listener:
                    listener.join()
            else:
                # continuous 模式: 启动 VAD 监控线程，主线程等待
                self._stop_event.clear()
                vad_thread = threading.Thread(
                    target=self._vad_monitor_loop, daemon=True
                )
                vad_thread.start()
                try:
                    self._stop_event.wait()
                except KeyboardInterrupt:
                    self._stop_event.set()
                    raise