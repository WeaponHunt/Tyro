import sounddevice as sd
from scipy.io import wavfile
import numpy as np
from pynput import keyboard
import threading

# 参数设置
sample_rate = 44100  # 采样率：44.1kHz
channels = 2  # 通道数：立体声

# 全局变量
is_recording = False
audio_frames = []
stream = None

def record_audio():
    """持续录制音频"""
    global is_recording, audio_frames, stream
    
    def audio_callback(indata, frames, time, status):
        if status:
            print(f"录音错误: {status}")
        if is_recording:
            audio_frames.append(indata.copy())
    
    # 创建音频流
    stream = sd.InputStream(
        samplerate=sample_rate,
        channels=channels,
        callback=audio_callback,
        blocksize=2048
    )
    
    with stream:
        print("准备完毕，按住 'R' 键开始录音...")
        while True:
            sd.sleep(100)

def on_press(key):
    """处理键盘按下事件"""
    global is_recording, audio_frames
    
    try:
        # 兼容不同系统的键位判断
        k = key.char if hasattr(key, 'char') else None
        if k == 'r' or k == 'R':
            if not is_recording:
                is_recording = True
                audio_frames = []
                print("🔴 开始录制... (松开 'R' 结束)", end='', flush=True)
    except Exception:
        pass

def on_release(key):
    """处理键盘松开事件"""
    global is_recording, audio_frames
    
    try:
        k = key.char if hasattr(key, 'char') else None
        if k == 'r' or k == 'R':
            if is_recording:
                is_recording = False
                print("\n✅ 录制完成，正在保存...")
                
                if audio_frames:
                    # 合并所有音频帧
                    audio_data = np.concatenate(audio_frames, axis=0)
                    # 保存为WAV文件
                    wavfile.write("audio.wav", sample_rate, (audio_data * 32767).astype(np.int16))
                    print("🎵 音频已保存为 audio.wav\n")
    except Exception:
        pass

# 在单独的线程中进行音频录制
audio_thread = threading.Thread(target=record_audio, daemon=True)
audio_thread.start()

# 启动键盘监听
print("程序已就绪：按住 'R' 键录音。 (Ctrl+C 退出)\n")
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()