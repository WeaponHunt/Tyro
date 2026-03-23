"""
TTS模块测试
"""
import time

from talkrobot.config import Config
from talkrobot.modules.tts.tts_module import TTSModule

def test_tts():
    """测试TTS模块"""
    test_start = time.perf_counter()

    print("="*50)
    print("测试 TTS 模块")
    print("="*50)

    init_start = time.perf_counter()
    tts = TTSModule(
        lang_code=Config.TTS_LANG_CODE,
        voice=Config.TTS_VOICE,
        speed=Config.TTS_SPEED,
        provider=Config.TTS_PROVIDER,
        language=Config.LANGUAGE,
        sample_rate=Config.TTS_SAMPLE_RATE,
    )
    init_elapsed = time.perf_counter() - init_start
    print(f"初始化耗时: {init_elapsed:.3f}s")
    
    test_text = "你好,我是小算算机器人,很高兴为您服务。"
    print(f"\n测试文本: {test_text}")
    print("正在合成并播放...")

    synth_start = time.perf_counter()
    audio_chunks = tts.synthesize(test_text, play_audio=True)
    synth_elapsed = time.perf_counter() - synth_start

    print(f"生成了 {len(audio_chunks)} 个音频片段")
    print(f"TTS合成/播放耗时: {synth_elapsed:.3f}s")
    print(f"总耗时: {time.perf_counter() - test_start:.3f}s")
    print("\n✅ TTS模块测试完成")

if __name__ == "__main__":
    test_tts()