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
    
    # ASR 配置
    ASR_MODEL = "iic/SenseVoiceSmall"
    ASR_DEVICE = "cuda"  # 或 "cpu"
    
    # TTS 配置
    TTS_LANG_CODE = 'z'  # 中文
    TTS_VOICE = 'zf_xiaoyi'
    TTS_SPEED = 1.0
    TTS_SAMPLE_RATE = 24000
    
    # LLM 配置
    LLM_API_KEY = "sk-9d42be35fbba4ef8ab8d217c2a613869"
    LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    LLM_MODEL = "qwen-plus"
    
    # Memory 基础数据库路径
    MEMORY_DB_BASE_PATH = os.path.join(os.path.dirname(__file__), "mem_db")
    
    # 默认用户
    DEFAULT_USER = "default"
    
    # 系统提示词
    SYSTEM_PROMPT = "你名叫Tyro,一个友好、乐于助人且高效的AI助手。请用简洁、自然的方式回答用户的问题。"
    
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