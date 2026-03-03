"""
语音合成(TTS)模块
负责将文本转换为语音
"""
import threading

from kokoro import KPipeline
import sounddevice as sd
import numpy as np
from loguru import logger

class TTSModule:
    """语音合成模块"""
    
    def __init__(self, lang_code: str = 'z', voice: str = 'zf_xiaoyi', speed: float = 1.0):
        """
        初始化TTS模块
        
        Args:
            lang_code: 语言代码 ('z' 代表中文)
            voice: 音色名称
            speed: 语速
        """
        logger.info(f"正在初始化TTS模块: lang={lang_code}, voice={voice}")
        self.pipeline = KPipeline(lang_code=lang_code)
        self.voice = voice
        self.speed = speed
        self._interrupted = threading.Event()
        logger.info("TTS模块初始化完成")
    
    def stop(self):
        """打断当前TTS播放"""
        self._interrupted.set()
        try:
            sd.stop()
        except Exception as e:
            logger.debug(f"TTS stop 调用 sd.stop 异常(可忽略): {e}")
    
    def synthesize(self, text: str, play_audio: bool = True) -> list:
        """
        将文本合成为语音
        
        Args:
            text: 待合成的文本
            play_audio: 是否直接播放音频
            
        Returns:
            list: 音频数据列表
        """
        try:
            logger.info(f"正在合成语音: {text}")
            self._interrupted.clear()
            
            generator = self.pipeline(text, voice=self.voice, speed=self.speed)
            
            audio_chunks = []
            try:
                for i, (gs, ps, audio) in enumerate(generator):
                    # 每个循环开头检查是否被打断
                    if self._interrupted.is_set():
                        logger.info("TTS播放被打断，停止合成（生成前）")
                        break
                    
                    audio_chunks.append(audio)
                    
                    # 实时播放
                    if play_audio:
                        logger.debug(f"播放第{i}段音频，长度{len(audio)}")
                        sd.play(audio, 24000, blocking=False)
                        
                        # 使用轮询等待播放完成，每 10ms 检查一次是否被中断
                        total_samples = len(audio)
                        played_samples = 0
                        while played_samples < total_samples:
                            if self._interrupted.is_set():
                                logger.info("TTS播放被打断，停止播放（播放中）")
                                sd.stop()
                                break
                            
                            stream = sd.get_stream()
                            if stream is None or not stream.active:
                                logger.debug(f"第{i}段播放完成")
                                break
                            
                            # 估算已播放的样本数
                            try:
                                played_samples = int(stream.latency[1] * 24000)
                            except:
                                pass
                            
                            sd.sleep(10)
                        
                        # 播放完成后再检查一次
                        if self._interrupted.is_set():
                            logger.info("TTS播放被打断，停止合成（播放后）")
                            break
                    
                    logger.debug(f"第 {i} 段完成")
            except GeneratorExit:
                logger.debug("生成器已关闭")
            
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
            sf.write(filepath, full_audio, 24000)
            logger.info(f"音频已保存至: {filepath}")
            return True
        except Exception as e:
            logger.error(f"保存音频失败: {e}")
            return False