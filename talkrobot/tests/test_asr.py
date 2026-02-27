"""
ASR模块测试
"""
import numpy as np
from talkrobot.config import Config
from talkrobot.modules.asr_module import ASRModule

def test_asr():
    """测试ASR模块"""
    print("="*50)
    print("测试 ASR 模块")
    print("="*50)
    
    asr = ASRModule(
        model_name=Config.ASR_MODEL,
        device=Config.ASR_DEVICE
    )
    
    # 生成测试音频 (静音)
    duration = 2  # 秒
    test_audio = np.random.randn(Config.SAMPLE_RATE * duration).astype(np.float32) * 0.01
    
    print("\n正在测试识别...")
    result = asr.transcribe(test_audio)
    print(f"识别结果: {result}")
    print("\n✅ ASR模块测试完成")

if __name__ == "__main__":
    test_asr()