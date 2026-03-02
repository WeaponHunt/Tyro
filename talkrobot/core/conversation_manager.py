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
    
    def __init__(self, asr_module, tts_module, llm_module, memory_module,
                 tts_enabled: bool = True, expression_module=None,
                 audio_recorder=None,
                 audio_min_duration: float = 0.5,
                 audio_min_rms: float = 0.005,
                 sample_rate: int = 16000):
        """
        初始化对话管理器
        
        Args:
            asr_module: ASR模块实例
            tts_module: TTS模块实例
            llm_module: LLM模块实例
            memory_module: Memory模块实例
            tts_enabled: 是否启用TTS语音播放（默认True）
            expression_module: 表情模块实例（可选）
            audio_recorder: 音频录制器实例（可选，用于 continuous 模式通知状态）
            audio_min_duration: 最短音频时长（秒），低于此值不送 ASR
            audio_min_rms: 最低音量 (RMS)，低于此值视为静音
            sample_rate: 采样率（用于计算时长）
        """
        self.asr = asr_module
        self.tts = tts_module
        self.llm = llm_module
        self.memory = memory_module
        self.tts_enabled = tts_enabled
        self.expression = expression_module
        self.audio_recorder = audio_recorder
        self.audio_min_duration = audio_min_duration
        self.audio_min_rms = audio_min_rms
        self.sample_rate = sample_rate
        
        expr_status = '开启' if (expression_module and expression_module.is_available) else '关闭'
        logger.info(f"对话管理器初始化完成 (TTS: {'开启' if tts_enabled else '关闭'}, 表情: {expr_status})")
    
    def process_audio(self, audio_data: np.ndarray) -> None:
        """
        处理录制的音频,完成完整的对话流程
        
        Args:
            audio_data: 音频数据
        """
        try:
            # 0. 音频前置过滤：时长和音量检查
            duration = len(audio_data) / self.sample_rate
            rms = float(np.sqrt(np.mean(audio_data ** 2)))
            
            if duration < self.audio_min_duration:
                logger.info(f"音频过短 ({duration:.2f}s < {self.audio_min_duration}s)，已忽略")
                print(f"\n⚠️ 音频过短 ({duration:.2f}秒)，已忽略")
                return
            
            if rms < self.audio_min_rms:
                logger.info(f"音量过低 (RMS={rms:.4f} < {self.audio_min_rms})，已忽略")
                print(f"\n⚠️ 音量过低，已忽略")
                return
            
            logger.debug(f"音频检查通过: 时长={duration:.2f}s, RMS={rms:.4f}")
            
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
            raw_response = self.llm.generate_response(user_text, context)
            
            # 4.1 解析表情标签
            expression_name = None
            if self.expression and self.expression.is_available:
                from talkrobot.modules.expression_module import ExpressionModule
                response, expression_name = ExpressionModule.parse_expression_from_response(raw_response)
            else:
                response = raw_response
            
            print(f"🤖 机器人: {response}")
            if expression_name:
                logger.info(f"检测到表情: {expression_name}")
            
            # 4.2 切换表情
            if expression_name and self.expression:
                self.expression.set_expression(expression_name)
            
            # 5. 异步存储机器人回复到记忆 (不阻塞后续流程)
            self.memory.add_memory(f"机器人回复: {response}", async_mode=True)
            
            # 6. TTS: 文字转语音并播放
            if self.tts_enabled:
                print("🔊 正在播放语音...")
                # 通知录音器 TTS 正在播放（continuous 模式下屏蔽麦克风）
                if self.audio_recorder:
                    self.audio_recorder.is_tts_playing = True
                try:
                    self.tts.synthesize(response, play_audio=True)
                finally:
                    if self.audio_recorder:
                        self.audio_recorder.is_tts_playing = False
            else:
                logger.info("TTS已关闭，跳过语音播放")
            
            # 7. 回复结束后重置为默认表情
            if expression_name and self.expression:
                self.expression.reset_expression()
            
            print("\n" + "-"*50)
            if self.audio_recorder and self.audio_recorder.listen_mode == "continuous":
                print("✅ 对话完成，请继续说话")
            else:
                print("✅ 对话完成，请继续按 'Q' 说话")
            print("-"*50 + "\n")
            
        except Exception as e:
            logger.error(f"对话处理出错: {e}")
            print(f"❌ 处理出错: {e}")
        finally:
            # 通知录音器处理完成，可以采集下一段语音
            if self.audio_recorder:
                self.audio_recorder.notify_process_done()
    
    def process_audio_async(self, audio_data: np.ndarray) -> None:
        """
        异步处理音频(避免阻塞录音)
        
        Args:
            audio_data: 音频数据
        """
        thread = threading.Thread(target=self.process_audio, args=(audio_data,))
        thread.daemon = True
        thread.start()