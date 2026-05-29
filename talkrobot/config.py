"""
配置管理模块
集中管理所有组件的配置参数
"""
import os
import re

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


class Config:
    """全局配置类"""
    
    # 调试模式（运行时由命令行参数设置）
    DEBUG = _env_bool("TALKROBOT_DEBUG", False)
    
    # 音频配置
    SAMPLE_RATE = _env_int("TALKROBOT_SAMPLE_RATE", 16000)
    CHANNELS = _env_int("TALKROBOT_CHANNELS", 1)
    
    # 监听模式: "push"=按住Q键说话, "continuous"=持续监听
    DEFAULT_LISTEN_MODE = os.getenv("TALKROBOT_LISTEN_MODE", "push")
    
    # 持续监听模式 VAD 配置 (Silero VAD)
    VAD_CHECK_INTERVAL = _env_float("TALKROBOT_VAD_CHECK_INTERVAL", 0.5)
    VAD_PRE_SPEECH_DURATION = _env_float("TALKROBOT_VAD_PRE_SPEECH_DURATION", 0.25)
    VAD_SILENCE_DURATION = _env_float("TALKROBOT_VAD_SILENCE_DURATION", 1)
    VAD_MIN_SPEECH_DURATION = _env_float("TALKROBOT_VAD_MIN_SPEECH_DURATION", 0.3)
    
    # 音频过滤配置（ASR 前置检查）
    AUDIO_MIN_DURATION = _env_float("TALKROBOT_AUDIO_MIN_DURATION", 0.3)
    AUDIO_MIN_RMS = _env_float("TALKROBOT_AUDIO_MIN_RMS", 0)
    
    # ASR 配置
    ASR_MODEL = os.getenv("TALKROBOT_ASR_MODEL", "iic/SenseVoiceSmall")
    ASR_DEVICE = os.getenv("TALKROBOT_ASR_DEVICE", "cuda")  # 或 "cpu"
    
    # TTS 配置
    TTS_PROVIDER = os.getenv("TALKROBOT_TTS_PROVIDER", "easy_tts_server")
    LANGUAGE = os.getenv("TALKROBOT_LANGUAGE", "zh")
    TTS_LANG_CODE = os.getenv("TALKROBOT_TTS_LANG_CODE", "z")
    TTS_VOICE = os.getenv("TALKROBOT_TTS_VOICE", "zf_xiaoyi")
    TTS_SPEED = _env_float("TALKROBOT_TTS_SPEED", 1.0)
    TTS_SAMPLE_RATE = _env_int("TALKROBOT_TTS_SAMPLE_RATE", 24000)
    
    # LLM 配置
    LLM_API_KEY = os.getenv("TALKROBOT_LLM_API_KEY") or os.getenv("DASHSCOPE_API_KEY", "")
    LLM_BASE_URL = os.getenv(
        "TALKROBOT_LLM_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    LLM_MODEL = os.getenv("TALKROBOT_LLM_MODEL", "qwen-flash")

    # Agent planner: "llm" uses the chat model to choose tools; "rule" uses deterministic heuristics.
    AGENT_PLANNER_PROVIDER = os.getenv("TALKROBOT_AGENT_PLANNER", "llm").strip().lower()
    AGENT_REACT_MAX_ITERATIONS = _env_int("TALKROBOT_AGENT_REACT_MAX_ITERATIONS", 4)
    
    # 表情服务器配置
    EXPRESSION_SERVER_URL = os.getenv("TALKROBOT_EXPRESSION_SERVER_URL", "http://localhost:8001")
    EXPRESSION_DEFAULT = os.getenv("TALKROBOT_EXPRESSION_DEFAULT", "neutral")
    EXPRESSION_ENABLED = _env_bool("TALKROBOT_EXPRESSION_ENABLED", True)

    # 人脸识别配置
    FACE_ENABLED = _env_bool("TALKROBOT_FACE_ENABLED", False)
    FACE_CAMERA_INDEX = _env_int("TALKROBOT_FACE_CAMERA_INDEX", 0)
    FACE_POLL_INTERVAL = _env_float("TALKROBOT_FACE_POLL_INTERVAL", 0.03)
    FACE_USE_GPU = _env_bool("TALKROBOT_FACE_USE_GPU", True)
    FACE_MODEL_NAME = os.getenv("TALKROBOT_FACE_MODEL_NAME", "buffalo_s")
    FACE_UNKNOWN_USER = os.getenv("TALKROBOT_FACE_UNKNOWN_USER", "guest")
    FACE_KNOWN_FACES_DIR = os.getenv(
        "TALKROBOT_FACE_KNOWN_FACES_DIR",
        os.path.join(
            os.path.dirname(__file__),
            "modules",
            "face_recognize",
            "known_faces",
        ),
    )

    # Memory 基础数据库路径
    MEMORY_PROVIDER = os.getenv("TALKROBOT_MEMORY_PROVIDER", "mem0").strip().lower()
    MEMORY_DB_BASE_PATH = os.getenv(
        "TALKROBOT_MEMORY_DB_BASE_PATH",
        os.path.join(os.path.dirname(__file__), "mem_db"),
    )
    SIMPLE_MEMORY_DB_PATH = os.getenv(
        "TALKROBOT_SIMPLE_MEMORY_DB_PATH",
        os.path.join(MEMORY_DB_BASE_PATH, "simple"),
    )
    
    # 默认用户
    DEFAULT_USER = os.getenv("TALKROBOT_DEFAULT_USER", "default")

    # 人格配置文件路径
    PERSONA_PROFILE_PATH = os.getenv(
        "TALKROBOT_PERSONA_PROFILE_PATH",
        os.path.join(os.path.dirname(__file__), "persona_profiles.json"),
    )

    # 全局提示词：对所有用户生效，拼接在用户人格 prompt 后
    GLOBAL_SYSTEM_PROMPT = """请遵循以下原则：
        语义优先：如果一句话字面上不通顺，请结合上下文推测用户最可能想表达的意思（例如“我想看电影”被误识为“我想看点影”）。
        音近替换：对于模糊的词汇，优先考虑发音相似的正确词汇。
        生成的文本要便于tts朗读，比如”10-15“应该生成为“10到15”避免tts把”-“读成”减“。"""

    GLOBAL_SYSTEM_PROMPT_EN = """Please follow these principles:
        Semantic-first: if an utterance is awkward or unclear, infer the most likely user intent from context (for example, treat minor ASR mistakes as likely homophone errors).
        Homophone correction: for ambiguous words, prefer phonetically similar and contextually correct terms.
        TTS-friendly output: generate text that is easy to read aloud by TTS. For example, write ranges as "10 to 15" instead of "10-15" to avoid reading '-' as "minus".
        Language policy: answer the user in English no matter what."""

    # 是否启用后台人格自动更新（LangGraph Agent）
    ENABLE_PERSONA_AUTO_UPDATE = _env_bool("TALKROBOT_ENABLE_PERSONA_AUTO_UPDATE", True)
    PERSONA_SENTIMENT_THRESHOLD = _env_float("TALKROBOT_PERSONA_SENTIMENT_THRESHOLD", 0.82)

    # 滑动窗口记忆：大模型可见的最近对话轮数（0 表示关闭）
    SLIDING_WINDOW_ROUNDS = _env_int("TALKROBOT_SLIDING_WINDOW_ROUNDS", 5)
    
    # 系统提示词
    SYSTEM_PROMPT = """你名叫Tyro,一个友好、乐于助人且高效的AI助手。请用简洁、自然的方式回答用户的问题,请尽量不要生成英文。
    请注意：你的输入来自 ASR（语音识别）系统，可能存在同音字错误、漏词、多词或断句不准的情况。
    在处理用户输入时，请遵循以下原则：
        语义优先：如果一句话字面上不通顺，请结合上下文推测用户最可能想表达的意思（例如“我想看电影”被误识为“我想看点影”）。
        音近替换：对于模糊的词汇，优先考虑发音相似的正确词汇。
        保持自然：直接回答用户的潜在意图，除非完全无法理解，否则不要反复询问用户是否说错了。"""

    SYSTEM_PROMPT_EN = """Your name is Tyro, a friendly, helpful, and efficient AI assistant. Answer user questions in concise, natural English.
    Note: your input comes from an ASR (speech recognition) system, so it may contain homophone errors, missing words, extra words, or incorrect sentence boundaries.
    When handling user input, follow these principles:
        Semantic-first: if a sentence is not fluent literally, infer the user's most likely intent from context.
        Homophone correction: for ambiguous terms, prioritize phonetically similar and contextually correct words.
        Keep it natural: directly answer the user's likely intent; unless it is completely unintelligible, avoid repeatedly asking whether the user spoke incorrectly."""
    

    @classmethod
    def get_user_id(cls, user: str) -> str:
        """根据用户名生成 user_id"""
        return f"user_{user}"
    
    @classmethod
    def get_memory_db_path(cls, user: str) -> str:
        """根据用户名生成独立的记忆数据库路径"""
        return os.path.join(cls.MEMORY_DB_BASE_PATH, user)

    @classmethod
    def get_simple_memory_path(cls) -> str:
        """本地 JSON memory backend 的存储目录。"""
        return cls.SIMPLE_MEMORY_DB_PATH

    @classmethod
    def has_persistent_memory(cls, user: str) -> bool:
        """判断用户是否已有持久化记忆数据。"""
        if cls.MEMORY_PROVIDER in {"simple", "json", "local"}:
            safe_user_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", cls.get_user_id(user))
            return os.path.isfile(os.path.join(cls.get_simple_memory_path(), f"{safe_user_id}.json"))

        db_path = cls.get_memory_db_path(user)
        if not os.path.isdir(db_path):
            return False

        for _, _, files in os.walk(db_path):
            if files:
                return True
        return False
    
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
