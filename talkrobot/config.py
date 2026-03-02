"""
配置管理模块
集中管理所有组件的配置参数
"""
import os

class Config:
    """全局配置类"""
    
    # 音频配置
    SAMPLE_RATE = 16000
    CHANNELS = 1
    
    # 监听模式: "push"=按住Q键说话, "continuous"=持续监听
    DEFAULT_LISTEN_MODE = "push"
    
    # 持续监听模式 VAD 配置 (Silero VAD)
    VAD_CHECK_INTERVAL = 0.25        # VAD 检测间隔（秒），每隔此时间检测一次语音
    VAD_SILENCE_DURATION = 1      # 静默多少秒后判定说话结束
    VAD_MIN_SPEECH_DURATION = 0.3    # 最短语音时长（秒），过短的丢弃
    
    # 音频过滤配置（ASR 前置检查）
    AUDIO_MIN_DURATION = 0.5          # 最短音频时长（秒），低于此值不送 ASR
    AUDIO_MIN_RMS = 0                 # 最低音量 (RMS)，低于此值视为静音
    
    # ASR 配置
    ASR_MODEL = "iic/SenseVoiceSmall"
    ASR_DEVICE = "cuda"  # 或 "cpu"
    
    # TTS 配置
    TTS_LANG_CODE = 'z'  # 中文
    TTS_VOICE = 'zf_xiaoyi'
    TTS_SPEED = 1.0
    TTS_SAMPLE_RATE = 24000
    
    # LLM 配置
    LLM_API_KEY = "put your api key here"
    LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    LLM_MODEL = "qwen-plus"
    
    # 表情服务器配置
    EXPRESSION_SERVER_URL = "http://localhost:8001"
    EXPRESSION_DEFAULT = "neutral"
    EXPRESSION_ENABLED = True  # 是否启用表情功能

    # Memory 基础数据库路径
    MEMORY_DB_BASE_PATH = os.path.join(os.path.dirname(__file__), "mem_db")
    
    # 默认用户
    DEFAULT_USER = "default"
    
    # 系统提示词
    SYSTEM_PROMPT = "你名叫Tyro,一个友好、乐于助人且高效的AI助手。请用简洁、自然的方式回答用户的问题,请尽量不要生成英文。"
    
    @classmethod
    def get_user_id(cls, user: str) -> str:
        """根据用户名生成 user_id"""
        return f"user_{user}"
    
    @classmethod
    def get_memory_db_path(cls, user: str) -> str:
        """根据用户名生成独立的记忆数据库路径"""
        return os.path.join(cls.MEMORY_DB_BASE_PATH, user)
    
    @classmethod
    def get_memory_config(cls, user: str) -> dict:
        """根据用户名生成独立的 Memory 配置"""
        return {
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": f"talkrobot_memories_{user}",
                    "path": cls.get_memory_db_path(user)
                }
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "qwen-plus",
                    "api_key": cls.LLM_API_KEY,
                    "openai_base_url": cls.LLM_BASE_URL,
                    "max_tokens": 1500,
                    "temperature": 0.1
                }
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": "text-embedding-v2",
                    "api_key": cls.LLM_API_KEY,
                    "openai_base_url": cls.LLM_BASE_URL
                }
            }
        }