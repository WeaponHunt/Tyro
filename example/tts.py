from kokoro import KModel, KPipeline
import soundfile as sf

# 1. 初始化 (自动下载模型)
pipeline = KPipeline(lang_code='z') # 'z' 代表中文

# 2. 直接选预设音色，输入文本
# 这里不需要 ref_audio，不需要 ref_text
generator = pipeline("你好，我是小算算机器人", voice='zf_xiaoyi', speed=1.0)

# 3. 循环获取结果 (支持流式输出)
for i, (gs, ps, audio) in enumerate(generator):
    sf.write(f'out_{i}.wav', audio, 24000)
    print(f"第 {i} 段合成完成")