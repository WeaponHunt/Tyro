"""
对话管理器
协调各个模块完成完整的对话流程
"""
import numpy as np
from loguru import logger
from typing import Optional
import threading

class ConversationManager:
    """对话管理器"""
    
    def __init__(self, asr_module, tts_module, llm_module, memory_module):
        """
        初始化对话管理器
        
        Args:
            asr_module: ASR模块实例
            tts_module: TTS模块实例
            llm_module: LLM模块实例
            memory_module: Memory模块实例
        """
        self.asr = asr_module
        self.tts = tts_module
        self.llm = llm_module
        self.memory = memory_module
        
        logger.info("对话管理器初始化完成")
    
    def process_audio(self, audio_data: np.ndarray) -> None:
        """
        处理录制的音频,完成完整的对话流程
        
        Args:
            audio_data: 音频数据
        """
        try:
            # 1. ASR: 语音转文字
            print("\n🎯 正在识别...")
            user_text = self.asr.transcribe(audio_data)
            
            if not user_text or user_text.strip() == "":
                print("⚠️ 未识别到有效内容,请重试")
                return
            
            print(f"👤 您说: {user_text}")
            
            # 2. 异步存储用户输入到记忆 (不阻塞后续流程)
            self.memory.add_memory(f"用户说: {user_text}", async_mode=True)
            
            # 3. 从记忆中检索相关上下文
            print("🔍 正在检索相关记忆...")
            context = self.memory.search_memory(user_text)
            
            # 4. LLM: 生成回复
            print("🤖 正在思考回复...")
            response = self.llm.generate_response(user_text, context)
            print(f"🤖 机器人: {response}")
            
            # 5. 异步存储机器人回复到记忆 (不阻塞后续流程)
            self.memory.add_memory(f"机器人回复: {response}", async_mode=True)
            
            # 6. TTS: 文字转语音并播放
            print("🔊 正在播放语音...")
            self.tts.synthesize(response, play_audio=True)
            
            print("\n" + "-"*50)
            print("✅ 对话完成,请继续按 'Q' 说话")
            print("-"*50 + "\n")
            
        except Exception as e:
            logger.error(f"对话处理出错: {e}")
            print(f"❌ 处理出错: {e}")
    
    def process_audio_async(self, audio_data: np.ndarray) -> None:
        """
        异步处理音频(避免阻塞录音)
        
        Args:
            audio_data: 音频数据
        """
        thread = threading.Thread(target=self.process_audio, args=(audio_data,))
        thread.daemon = True
        thread.start()