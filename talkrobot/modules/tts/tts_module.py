"""
语音合成(TTS)模块
负责将文本转换为语音
"""
import threading
import time
from collections.abc import Iterable
from typing import List, Optional, Union

import sounddevice as sd
import numpy as np
from loguru import logger

class TTSModule:
    """语音合成模块"""
    
    def __init__(
        self,
        lang_code: str = 'z',
        voice: str = 'zf_xiaoyi',
        speed: float = 1.0,
        provider: str = 'kokoro',
        language: Optional[str] = None,
        sample_rate: int = 24000,
    ):
        """
        初始化TTS模块
        
        Args:
            lang_code: Kokoro语言代码（兼容保留，默认'z'）
            voice: 音色名称
            speed: 语速
            provider: TTS后端，支持 'kokoro' / 'easy_tts_server'
            language: 语言，支持 'zh' / 'en'，不传则沿用 lang_code
            sample_rate: 采样率
        """
        self.provider = str(provider).strip().lower()
        self.language = self._normalize_language(language) if language is not None else None
        self.sample_rate = sample_rate
        self.voice = voice
        self.speed = speed
        self.pipeline = None
        self.easy_tts_engine = None
        self._state_lock = threading.Lock()
        self._paused = threading.Event()
        self._resume_segments: List[str] = []
        self._resume_segment_index = 0
        self._resume_play_audio = True

        logger.info(
            f"正在初始化TTS模块: provider={self.provider}, language={self.language}, "
            f"lang_code={lang_code}, voice={voice}"
        )

        if self.provider == "kokoro":
            from kokoro import KPipeline

            resolved_lang_code = self._resolve_kokoro_lang_code(self.language, lang_code)
            self.pipeline = KPipeline(lang_code=resolved_lang_code)
        elif self.provider == "easy_tts_server":
            try:
                from easy_tts_server import create_tts_engine
            except ImportError as e:
                raise ImportError(
                    "使用 easy_tts_server 作为TTS后端时，请先安装依赖: pip install easy_tts_server"
                ) from e

            self.easy_tts_engine = create_tts_engine()
            self.sample_rate = self._get_easy_sample_rate(default=sample_rate)
        else:
            raise ValueError(f"不支持的TTS后端: {provider}，可选: kokoro / easy_tts_server")

        self._interrupted = threading.Event()
        logger.info("TTS模块初始化完成")

    @staticmethod
    def _normalize_language(language: str) -> str:
        language = str(language).strip().lower()
        mapping = {
            "zh": "zh",
            "cn": "zh",
            "chinese": "zh",
            "z": "zh",
            "en": "en",
            "english": "en",
            "e": "en",
        }
        return mapping.get(language, "zh")

    @staticmethod
    def _resolve_kokoro_lang_code(language: Optional[str], lang_code: str) -> str:
        if language is not None:
            return "z" if language == "zh" else "a"

        if lang_code and str(lang_code).strip():
            normalized = str(lang_code).strip().lower()
            if normalized in {"zh", "z"}:
                return "z"
            if normalized in {"en", "a"}:
                return "a"
            return normalized

        return "z"

    def _get_easy_sample_rate(self, default: int) -> int:
        for attr_name in ("sample_rate", "sampling_rate", "sr"):
            value = getattr(self.easy_tts_engine, attr_name, None)
            if isinstance(value, int) and value > 0:
                logger.info(f"easy_tts_server 采样率: {value}")
                return value
        return default
    
    def stop(self):
        """打断当前TTS播放"""
        self._interrupted.set()
        self._paused.clear()
        with self._state_lock:
            self._resume_segments = []
            self._resume_segment_index = 0
        try:
            sd.stop()
        except Exception as e:
            logger.debug(f"TTS stop 调用 sd.stop 异常(可忽略): {e}")

    def pause(self):
        """暂停当前TTS播放，记录到当前段。"""
        self._paused.set()
        try:
            sd.stop()
        except Exception as e:
            logger.debug(f"TTS pause 调用 sd.stop 异常(可忽略): {e}")

    def resume(self, play_audio: Optional[bool] = None) -> list:
        """恢复暂停的TTS播放，从暂停段重新开始。"""
        if not self._paused.is_set():
            logger.info("当前未处于暂停状态，无需恢复")
            return []

        with self._state_lock:
            segments = list(self._resume_segments)
            start_idx = self._resume_segment_index
            resolved_play_audio = self._resume_play_audio if play_audio is None else bool(play_audio)

        if not segments or start_idx >= len(segments):
            logger.info("没有可恢复的TTS段")
            self._paused.clear()
            return []

        logger.info(f"恢复TTS播放，从第{start_idx}段开始")
        self._paused.clear()
        self._interrupted.clear()
        audio_chunks, next_segment_idx = self._synthesize_segments(
            segments=segments,
            start_segment_idx=start_idx,
            play_audio=resolved_play_audio,
        )

        with self._state_lock:
            self._resume_segment_index = next_segment_idx
            self._resume_play_audio = resolved_play_audio
            if next_segment_idx >= len(segments) or self._interrupted.is_set():
                self._resume_segments = []
                self._resume_segment_index = 0

        return audio_chunks

    def _play_audio_chunk(self, audio: np.ndarray, index: int) -> bool:
        """播放单段音频，返回是否被中断。"""
        logger.debug(f"播放第{index}段音频，长度{len(audio)}")
        sd.play(audio, self.sample_rate, blocking=False)

        # 某些设备/驱动下 get_stream().latency 并不表示已播放采样数，
        # 使用预计时长 + 安全余量避免等待循环卡死。
        expected_seconds = max(0.0, float(len(audio)) / float(self.sample_rate))
        deadline = time.monotonic() + expected_seconds + 1.0

        while True:
            if self._paused.is_set():
                logger.info("TTS播放已暂停，停止当前播放")
                sd.stop()
                return True

            if self._interrupted.is_set():
                logger.info("TTS播放被打断，停止播放（播放中）")
                sd.stop()
                return True

            stream = sd.get_stream()
            if stream is None or not stream.active:
                logger.debug(f"第{index}段播放完成")
                break

            if time.monotonic() >= deadline:
                logger.warning(
                    f"第{index}段播放等待超时，已强制停止（预计{expected_seconds:.2f}s）"
                )
                sd.stop()
                break

            sd.sleep(10)

        return self._interrupted.is_set() or self._paused.is_set()

    @staticmethod
    def _split_text_by_punctuation(text: str) -> List[str]:
        """按中英文逗号和句号切分文本，保留标点。"""
        if not text:
            return []

        delimiters = {"，", ",", "。", "."}
        segments: List[str] = []
        buffer = ""

        for ch in str(text):
            buffer += ch
            if ch in delimiters:
                segment = buffer.strip()
                if segment:
                    segments.append(segment)
                buffer = ""

        tail = buffer.strip()
        if tail:
            segments.append(tail)

        return segments

    def _synthesize_segments(
        self,
        segments: List[str],
        start_segment_idx: int,
        play_audio: bool,
    ) -> tuple[List[np.ndarray], int]:
        """从指定段号开始合成，返回(音频块, 下一个待播段号)。"""
        audio_chunks: List[np.ndarray] = []
        next_segment_idx = start_segment_idx
        audio_index = 0

        for segment_idx in range(start_segment_idx, len(segments)):
            if self._paused.is_set() or self._interrupted.is_set():
                next_segment_idx = segment_idx
                break

            segment = segments[segment_idx].strip()
            if not segment:
                next_segment_idx = segment_idx + 1
                continue

            logger.debug(f"开始合成第{segment_idx}段: {segment}")
            segment_audio_chunks, interrupted = self._synthesize_single_text(
                segment,
                play_audio=play_audio,
                start_index=audio_index,
            )
            audio_chunks.extend(segment_audio_chunks)
            audio_index += len(segment_audio_chunks)

            if self._paused.is_set():
                next_segment_idx = segment_idx
                logger.info(f"TTS暂停在第{segment_idx}段，恢复时将从该段重新开始")
                break

            if interrupted or self._interrupted.is_set():
                next_segment_idx = segment_idx + 1
                break

            next_segment_idx = segment_idx + 1

        return audio_chunks, next_segment_idx

    def _synthesize_single_text(self, text: str, play_audio: bool, start_index: int = 0) -> tuple[List[np.ndarray], bool]:
        """合成单条文本，返回(音频列表, 是否被中断)。"""
        if self.provider == "kokoro":
            audio_chunks: List[np.ndarray] = []
            generator = self.pipeline(text, voice=self.voice, speed=self.speed)
            interrupted = False

            try:
                for offset, (_gs, _ps, audio) in enumerate(generator):
                    if self._paused.is_set():
                        logger.info("TTS已暂停，停止合成（生成前）")
                        interrupted = True
                        break

                    if self._interrupted.is_set():
                        logger.info("TTS播放被打断，停止合成（生成前）")
                        interrupted = True
                        break

                    audio_chunks.append(audio)

                    if play_audio:
                        interrupted = self._play_audio_chunk(audio, start_index + offset)
                        if interrupted:
                            if self._paused.is_set():
                                logger.info("TTS已暂停，停止合成（播放后）")
                            else:
                                logger.info("TTS播放被打断，停止合成（播放后）")
                            break

                    logger.debug(f"第 {start_index + offset} 段完成")
            except GeneratorExit:
                logger.debug("生成器已关闭")

            return audio_chunks, interrupted

        if self.provider == "easy_tts_server":
            if self._paused.is_set() or self._interrupted.is_set():
                return [], True

            language = self.language or "zh"
            audio = self.easy_tts_engine.tts(text, language=language, voice=self.voice)
            audio_array = np.asarray(audio)
            audio_chunks = [audio_array]
            interrupted = False

            if play_audio:
                interrupted = self._play_audio_chunk(audio_array, start_index)
            return audio_chunks, interrupted

        raise ValueError(f"不支持的TTS后端: {self.provider}")

    def _synthesize_from_iterable(self, text_stream: Iterable[str], play_audio: bool) -> List[np.ndarray]:
        """从字符串迭代器合并文本后按逗号/句号切段合成。"""
        merged_text_parts: List[str] = []
        for part in text_stream:
            if self._paused.is_set() or self._interrupted.is_set():
                break
            if part is None:
                continue
            merged_text_parts.append(str(part))

        merged_text = "".join(merged_text_parts)
        segments = self._split_text_by_punctuation(merged_text)

        with self._state_lock:
            self._resume_segments = segments
            self._resume_segment_index = 0
            self._resume_play_audio = play_audio

        audio_chunks, next_segment_idx = self._synthesize_segments(
            segments=segments,
            start_segment_idx=0,
            play_audio=play_audio,
        )

        with self._state_lock:
            self._resume_segment_index = next_segment_idx
            if next_segment_idx >= len(segments) or self._interrupted.is_set():
                self._resume_segments = []
                self._resume_segment_index = 0

        return audio_chunks
    
    def synthesize(self, text: Union[str, Iterable[str]], play_audio: bool = True) -> list:
        """
        将文本合成为语音
        
        Args:
            text: 待合成的文本，支持字符串或字符串生成器/迭代器
            play_audio: 是否直接播放音频
            
        Returns:
            list: 音频数据列表
        """
        try:
            logger.info(f"正在合成语音，输入类型: {type(text).__name__}")
            self._interrupted.clear()
            self._paused.clear()

            if isinstance(text, str):
                logger.info(f"正在合成语音: {text}")
                segments = self._split_text_by_punctuation(text)
                with self._state_lock:
                    self._resume_segments = segments
                    self._resume_segment_index = 0
                    self._resume_play_audio = play_audio

                audio_chunks, next_segment_idx = self._synthesize_segments(
                    segments=segments,
                    start_segment_idx=0,
                    play_audio=play_audio,
                )

                with self._state_lock:
                    self._resume_segment_index = next_segment_idx
                    if next_segment_idx >= len(segments) or self._interrupted.is_set():
                        self._resume_segments = []
                        self._resume_segment_index = 0
            else:
                audio_chunks = self._synthesize_from_iterable(text, play_audio=play_audio)
            
            if self._paused.is_set():
                logger.info("语音合成已暂停")
            elif self._interrupted.is_set():
                logger.info("语音合成被用户打断")
            else:
                logger.info("语音合成完成")
            return audio_chunks
            
        except Exception as e:
            logger.error(f"TTS合成出错: {e}")
            return []
    
    def save_audio(self, audio_chunks: list, filepath: str) -> bool:
        """
        保存音频到文件
        
        Args:
            audio_chunks: 音频数据列表
            filepath: 保存路径
            
        Returns:
            bool: 是否成功
        """
        try:
            import soundfile as sf
            # 合并所有音频片段
            full_audio = np.concatenate(audio_chunks)
            sf.write(filepath, full_audio, self.sample_rate)
            logger.info(f"音频已保存至: {filepath}")
            return True
        except Exception as e:
            logger.error(f"保存音频失败: {e}")
            return False