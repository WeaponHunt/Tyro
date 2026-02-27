"""
TTS模块测试
"""
from talkrobot.config import Config
from talkrobot.modules.tts_module import TTSModule

def test_tts():
    """测试TTS模块"""
    print("="*50)
    print("测试 TTS 模块")
    print("="*50)
    
    tts = TTSModule(
        lang_code=Config.TTS_LANG_CODE,
        voice=Config.TTS_VOICE,
        speed=Config.TTS_SPEED
    )
    
    test_text = "你好,我是小算算机器人,很高兴为您服务。"
    print(f"\n测试文本: {test_text}")
    print("正在合成并播放...")
    
    audio_chunks = tts.synthesize(test_text, play_audio=True)
    print(f"生成了 {len(audio_chunks)} 个音频片段")
    print("\n✅ TTS模块测试完成")

if __name__ == "__main__":
    test_tts()