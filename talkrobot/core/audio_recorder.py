"""
音频录制核心模块
负责录制音频并触发ASR
"""
import numpy as np
import sounddevice as sd
from pynput import keyboard
from loguru import logger
from typing import Callable

class AudioRecorder:
    """音频录制器"""
    
    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        """
        初始化录音器
        
        Args:
            sample_rate: 采样率
            channels: 通道数
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.is_recording = False
        self.audio_frames = []
        self.on_audio_complete: Callable = None
        
    def audio_callback(self, indata, frames, time, status):
        """音频流回调函数"""
        if status:
            logger.warning(f"录音状态: {status}")
        if self.is_recording:
            self.audio_frames.append(indata.copy())
    
    def on_press(self, key):
        """键盘按下事件"""
        try:
            k = key.char if hasattr(key, 'char') else None
            if k == 'q' and not self.is_recording:
                self.is_recording = True
                self.audio_frames = []
                print("\n🔴 正在录音... (松开 'Q' 结束)", end='', flush=True)
        except Exception:
            pass
    
    def on_release(self, key):
        """键盘松开事件"""
        try:
            k = key.char if hasattr(key, 'char') else None
            if k == 'q' and self.is_recording:
                self.is_recording = False
                print("\n✅ 录音结束")
                
                if self.audio_frames and self.on_audio_complete:
                    # 合并音频数据
                    audio_data = np.concatenate(self.audio_frames, axis=0)
                    # 触发回调
                    self.on_audio_complete(audio_data)
                    
        except Exception as e:
            logger.error(f"按键释放处理错误: {e}")
    
    def start(self, on_audio_complete: Callable):
        """
        启动录音器
        
        Args:
            on_audio_complete: 录音完成时的回调函数
        """
        self.on_audio_complete = on_audio_complete
        
        logger.info("启动音频录制...")
        print("\n" + "="*50)
        print("🎤 对话机器人已启动")
        print("="*50)
        print("📌 操作说明:")
        print("   - 按住 'Q' 键说话")
        print("   - 松开 'Q' 键结束")
        print("   - 按 Ctrl+C 退出程序")
        print("="*50 + "\n")
        
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self.audio_callback
        ):
            with keyboard.Listener(
                on_press=self.on_press,
                on_release=self.on_release
            ) as listener:
                listener.join()