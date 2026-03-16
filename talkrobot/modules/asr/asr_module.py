"""
语音识别(ASR)模块
负责将音频转换为文本
"""
import numpy as np
from funasr import AutoModel
from loguru import logger

class ASRModule:
    """语音识别模块"""
    
    def __init__(self, model_name: str, device: str = "cuda"):
        """
        初始化ASR模块
        
        Args:
            model_name: 模型名称
            device: 运行设备 (cuda/cpu)
        """
        logger.info(f"正在加载ASR模型: {model_name}")
        self.model = AutoModel(
            model=model_name,
            device=device,
            disable_update=True
        )
        logger.info("ASR模型加载完成")
    
    def transcribe(self, audio_data: np.ndarray) -> str:
        """
        将音频转换为文本
        
        Args:
            audio_data: 音频数据 (numpy数组)
            
        Returns:
            str: 识别的文本
        """
        try:
            # 确保音频数据格式正确
            audio_input = audio_data.flatten().astype(np.float32)
            
            # 检查音频长度
            if len(audio_input) < 1600:  # 少于0.1秒
                logger.warning("音频过短,跳过识别")
                return ""
            
            # 执行识别
            result = self.model.generate(
                input=audio_input,
                cache={},
                language="auto",
                use_itn=True,
            )
            
            if result and len(result) > 0:
                text = result[0]['text']
                logger.info(f"识别结果: {text}")
                return text
            
            return ""
            
        except Exception as e:
            logger.error(f"ASR识别出错: {e}")
            return ""