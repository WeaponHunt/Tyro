"""
语音合成(TTS)模块
负责将文本转换为语音
"""
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Iterable
from typing import List, Optional, Union

# sounddevice is imported lazily inside methods that actually play audio to avoid
# hard dependency at module import time in environments without audio devices.
import numpy as np
import unicodedata
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

    @staticmethod
    def _filter_text(text: str) -> str:
        """
        过滤输入文本，删除非文本且非标点的符号（例如表情符号、其他 Unicode 符号类别）。

        保留的 Unicode 类别首字母：
          - L: Letter (字母)
          - N: Number (数字)
          - P: Punctuation (标点)
          - Z: Separator (空白)
          - M: Mark (组合符号)

        删除的类别示例：S (Symbol，包含 emoji)、C (Other 控制字符等)
        """
        if not text:
            return text

        filtered_chars = []
        removed_chars = []
        for ch in text:
            try:
                cat = unicodedata.category(ch)
            except Exception:
                cat = "C"

            if cat and cat[0] in ("L", "N", "P", "Z", "M"):
                filtered_chars.append(ch)
            else:
                removed_chars.append(ch)

        if removed_chars:
            # 只打印部分以避免日志过长
            sample = removed_chars[:10]
            logger.debug(f"过滤掉非文本/标点字符: {sample}{'...' if len(removed_chars)>10 else ''}")

        return "".join(filtered_chars)
    
    def stop(self):
        """打断当前TTS播放"""
        self._interrupted.set()
        try:
            import sounddevice as sd
            sd.stop()
        except Exception as e:
            logger.debug(f"TTS stop 调用 sd.stop 异常(可忽略): {e}")

    def _play_audio_chunk(self, audio: np.ndarray, index: int) -> bool:
        """播放单段音频，返回是否被中断。"""
        logger.debug(f"播放第{index}段音频，长度{len(audio)}")
        try:
            import sounddevice as sd
        except Exception as e:
            logger.debug(f"无法导入 sounddevice，跳过播放: {e}")
            return False

        sd.play(audio, self.sample_rate, blocking=False)

        total_samples = len(audio)
        played_samples = 0
        while played_samples < total_samples:
            if self._interrupted.is_set():
                logger.info("TTS播放被打断，停止播放（播放中）")
                sd.stop()
                return True

            stream = sd.get_stream()
            if stream is None or not stream.active:
                logger.debug(f"第{index}段播放完成")
                break

            try:
                played_samples = int(stream.latency[1] * self.sample_rate)
            except Exception:
                pass

            sd.sleep(10)

        return self._interrupted.is_set()

    def _synthesize_single_text(self, text: str, play_audio: bool, start_index: int = 0) -> tuple[List[np.ndarray], bool]:
        """合成单条文本，返回(音频列表, 是否被中断)。"""
        if self.provider == "kokoro":
            audio_chunks: List[np.ndarray] = []
            generator = self.pipeline(text, voice=self.voice, speed=self.speed)
            interrupted = False

            try:
                for offset, (_gs, _ps, audio) in enumerate(generator):
                    if self._interrupted.is_set():
                        logger.info("TTS播放被打断，停止合成（生成前）")
                        interrupted = True
                        break

                    audio_chunks.append(audio)

                    if play_audio:
                        interrupted = self._play_audio_chunk(audio, start_index + offset)
                        if interrupted:
                            logger.info("TTS播放被打断，停止合成（播放后）")
                            break

                    logger.debug(f"第 {start_index + offset} 段完成")
            except GeneratorExit:
                logger.debug("生成器已关闭")

            return audio_chunks, interrupted

        if self.provider == "easy_tts_server":
            if self._interrupted.is_set():
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
        """从字符串迭代器持续合成；按逗号聚句并并行合成。"""
        audio_chunks: List[np.ndarray] = []
        chunk_index = 0
        future_queue: queue.Queue = queue.Queue()
        sentinel = object()

        def _sentence_iter(stream: Iterable[str]) -> Iterable[str]:
            """从增量文本中按逗号聚合完整句子并依次产出。"""
            buffer = ""
            delimiters = ("。", "：", "？", "?", "！", "!", "；", ";", "，", ",", ".", ":", "\n", "～")

            for part in stream:
                if self._interrupted.is_set():
                    break
                if part is None:
                    continue
                # 先过滤掉非文本/非标点的字符（例如表情符号）
                filtered_part = self._filter_text(str(part))
                if not filtered_part:
                    logger.debug(f"迭代器部分被过滤（非文本/标点），跳过: {part}")
                    continue
                buffer += filtered_part

                while True:
                    positions = [buffer.find(d) for d in delimiters if d in buffer]
                    if not positions:
                        break

                    split_pos = min(pos for pos in positions if pos >= 0) + 1
                    sentence = buffer[:split_pos].strip()
                    buffer = buffer[split_pos:]
                    if sentence:
                        yield sentence

            tail = buffer.strip()
            if tail and not self._interrupted.is_set():
                yield tail

        def _synthesize_sentence(sentence: str) -> List[np.ndarray]:
            chunks, _ = self._synthesize_single_text(sentence, play_audio=False)
            return chunks

        def producer() -> None:
            try:
                with ThreadPoolExecutor(max_workers=3) as executor:
                    sentence_idx = 0
                    for sentence in _sentence_iter(text_stream):
                        if self._interrupted.is_set():
                            break
                        logger.debug(f"生成器模式收到完整句子[{sentence_idx}]: {sentence}")
                        future = executor.submit(_synthesize_sentence, sentence)
                        future_queue.put((sentence_idx, future))
                        sentence_idx += 1
            except Exception as e:
                logger.error(f"生成器模式生产线程出错: {e}")
            finally:
                future_queue.put(sentinel)

        producer_thread = threading.Thread(target=producer, daemon=True)
        producer_thread.start()

        pending_futures = {}
        next_sentence_idx = 0
        producer_done = False

        while not self._interrupted.is_set():
            if not producer_done:
                item = future_queue.get()
                if item is sentinel:
                    producer_done = True
                else:
                    sentence_idx, future = item
                    pending_futures[sentence_idx] = future

            while next_sentence_idx in pending_futures:
                future = pending_futures[next_sentence_idx]
                if not future.done():
                    break

                try:
                    sentence_chunks = future.result()
                except Exception as e:
                    logger.error(f"句子[{next_sentence_idx}] 合成失败: {e}")
                    sentence_chunks = []

                for audio in sentence_chunks:
                    if self._interrupted.is_set():
                        break
                    audio_chunks.append(audio)
                    if play_audio:
                        interrupted = self._play_audio_chunk(audio, chunk_index)
                        chunk_index += 1
                        if interrupted:
                            break

                del pending_futures[next_sentence_idx]
                next_sentence_idx += 1

            if producer_done and not pending_futures:
                break

        producer_thread.join(timeout=0.5)
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

            if isinstance(text, str):
                # 过滤掉非文本/标点的字符（例如仅包含表情符号的字符串）
                filtered = self._filter_text(text)
                if not filtered:
                    logger.info("输入文本被过滤为空（可能只包含 emoji/符号），跳过合成")
                    return []

                logger.info(f"正在合成语音: {filtered}")
                audio_chunks, _ = self._synthesize_single_text(filtered, play_audio=play_audio)
            else:
                audio_chunks = self._synthesize_from_iterable(text, play_audio=play_audio)
            
            if self._interrupted.is_set():
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