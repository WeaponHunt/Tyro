import numpy as np
import sounddevice as sd
from funasr import AutoModel
from pynput import keyboard
import threading
import queue

# 1. 加载模型
print("正在加载模型...")
model = AutoModel(
    model="iic/SenseVoiceSmall",
    device="cuda", # 如果没有显卡请改为 "cpu"
    disable_update=True
)

# 全局变量
FS = 16000  
is_recording = False
recording_data = []

def run_asr(audio_data):
    """单独的 ASR 执行函数"""
    try:
        # 【关键修复】：将 2D 数组转为 1D，并确保是 float32 类型
        audio_input = audio_data.flatten().astype(np.float32)
        
        # 如果录音时间太短，跳过识别
        if len(audio_input) < 1600: # 少于 0.1 秒
            return

        res = model.generate(
            input=audio_input,
            cache={},
            language="auto",
            use_itn=True,
        )
        if res:
            print(f"\n👉 识别结果: {res[0]['text']}")
            print("-" * 30 + "\n按住 'Q' 键继续录音...")
    except Exception as e:
        print(f"\n❌ 识别出错: {e}")

def on_press(key):
    global is_recording, recording_data
    try:
        # 兼容不同系统的键位判断
        k = key.char if hasattr(key, 'char') else None
        if k == 'q' and not is_recording:
            print("\n🔴 正在录制... (松开 'Q' 结束)", end='', flush=True)
            is_recording = True
            recording_data = []
    except Exception:
        pass

def on_release(key):
    global is_recording, recording_data
    try:
        k = key.char if hasattr(key, 'char') else None
        if k == 'q' and is_recording:
            is_recording = False
            print("\n✅ 录制结束，识别中...")
            
            if recording_data:
                # 合并数据
                full_audio = np.concatenate(recording_data, axis=0)
                # 使用线程运行 ASR，避免阻塞监听器
                threading.Thread(target=run_asr, args=(full_audio,)).start()
    except Exception:
        pass

def audio_callback(indata, frames, time, status):
    """sounddevice 的回调函数，用于持续捕获音频"""
    if is_recording:
        recording_data.append(indata.copy())

# 启动持续的音频流
with sd.InputStream(samplerate=FS, channels=1, callback=audio_callback):
    print("程序已就绪：按住 'Q' 键说话。 (Ctrl+C 退出)")
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()