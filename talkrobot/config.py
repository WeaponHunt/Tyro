"""
配置管理模块
集中管理所有组件的配置参数
"""
import os
import re
import hashlib

class Config:
    """全局配置类"""
    
    # 调试模式（运行时由命令行参数设置）
    DEBUG = False
    
    # 音频配置
    SAMPLE_RATE = 16000
    CHANNELS = 1
    
    # 监听模式: "push"=按住Q键说话, "continuous"=持续监听, "intercom"=对讲机PTT触发
    DEFAULT_LISTEN_MODE = "push"

    # 键盘控制配置
    # 支持单字符（如 "w"/"c"/"p"/"s"）以及特殊键名（"enter"/"space"）
    MODE_SWITCH_SLEEP_KEY = "w"      # 不说话模式切换
    MODE_SWITCH_SCRIPT_KEY = "i"     # 脚本模式（介绍实验室）切换
    # 脚本文件配置：优先按指定文件名加载；为空时可回退到目录扫描
    SCRIPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "script"))
    SCRIPT_FILE = "acir.json"
    SCRIPT_PAUSE_RESUME_KEY = "enter"    # 脚本模式下TTS暂停/恢复（可改为 "space"）
    TTS_INTERRUPT_KEY = "s"          # TTS 播放打断
    INTERCOM_PTT_TOGGLE_KEY = "p"    # 对讲机模式下手动切换 PTT 按下/松开
    VIDEO_RECORD_TOGGLE_KEY = "r"    # 视频录制开始/结束
    VIDEO_RECORD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "video"))
    VIDEO_RECORD_FPS = 25.0
    VIDEO_RECORD_AUDIO_ENABLED = True

    # 脚本图片窗口显示位置配置
    # 优先级：SCRIPT_IMAGE_WINDOW_X/Y > SCRIPT_IMAGE_SCREEN_INDEX
    # SCRIPT_IMAGE_SCREEN_INDEX: xrandr --listmonitors 输出顺序的屏幕索引（0 开始）
    SCRIPT_IMAGE_SCREEN_INDEX = 0
    SCRIPT_IMAGE_WINDOW_X = None
    SCRIPT_IMAGE_WINDOW_Y = None
    SCRIPT_IMAGE_FULLSCREEN = True
    # 目标窗口分辨率：为 None 时自动使用目标屏幕分辨率
    SCRIPT_IMAGE_TARGET_WIDTH = 3840
    SCRIPT_IMAGE_TARGET_HEIGHT = 2160
    # 当全屏失效时，是否强制按目标分辨率设置窗口尺寸
    SCRIPT_IMAGE_FORCE_WINDOW_SIZE = True
    # 是否先把图片缩放到目标分辨率后再显示（可避免窗口缩放带来的黑边/拉伸差异）
    SCRIPT_IMAGE_FORCE_RESIZE_IMAGE = False

    # 语音模式切换词（用于替代按键切换，按“包含任一词汇”触发）
    #启动不说话模式
    MODE_SWITCH_SLEEP_ENABLE_VOICE_WORDS = {
        "zh": ["别说话了"],
        "en": ["mute on", "quiet", "keep silent", "remain silent","别说话了"],
    }
    #关闭不说话模式
    MODE_SWITCH_SLEEP_DISABLE_VOICE_WORDS = {
        "zh": ["可以说话了"],
        "en": ["mute off", "can speak", "speak now","可以说话了"],
    }
    
    # 介绍实验室"introduce the lab"
    # 开始介绍
    MODE_SWITCH_SCRIPT_ENABLE_VOICE_WORDS = {
        "zh": ["介绍实验室", "开始介绍","Introduce", "introduce", "Start", "start introduce", "introduction to the lab", "start the introduction", "Lab", "lab"],
        "en": ["Introduce", "introduce", "Start", "start introduce", "introduction to the lab", "start the introduction", "Lab", "lab", "介绍实验室", "开始介绍"],
    }
    # 停止介绍  
    MODE_SWITCH_SCRIPT_DISABLE_VOICE_WORDS = {
        "zh": ["别介绍了", "停止介绍"],
        "en": ["stop introduce", "stop the introduction", "别介绍了"],
    }

    # 多脚本配置。每个脚本可配置独立触发词和可选按键；key 为空/None 时只能通过关键词触发。
    # file 支持相对 SCRIPT_DIR 的路径，也支持绝对路径。
    SCRIPT_CONFIGS = [
        {
            "name": "lab_intro",
            "file": SCRIPT_FILE,
            "key": MODE_SWITCH_SCRIPT_KEY,
            "keywords": {
                "zh": MODE_SWITCH_SCRIPT_ENABLE_VOICE_WORDS["zh"],
                "en": MODE_SWITCH_SCRIPT_ENABLE_VOICE_WORDS["en"],
            },
        },
        {
            "name": "music_demo",
            "file": "music.json",
            "key": None,
            "keywords": {"zh": ["播放音乐"], "en": ["music demo","play music"]},
        },
        {
            "name": "sing_lzlh",
            "file": "lzlh.json",
            "key": None,
            "keywords": {"zh": ["唱首儿歌"], "en": ["sing a song","begin singing","play kids"]},
        },
        {
            "name": "bjea_intro",
            "file": "bjea.json",
            "key": None,
            "keywords": {"zh": ["介绍亦庄实验中学"], "en": ["introduce bjea middle school"]},
        },
    ]

    # 可视化界面开关语音词与 topic（命中任一词即触发）
    VISUALIZER_ENABLE_TOPIC = "/face/visualizer/enabled"
    VISUALIZER_ENABLE_VOICE_WORDS = {
        "zh": ["你在想什么", "打开可视化", "显示可视化"],
        "en": ["show me your mind", "thinking about", "开启可视化", "打开可视化", "显示可视化"],
    }
    VISUALIZER_DISABLE_VOICE_WORDS = {
        "zh": ["关闭可视化", "关掉可视化", "关闭吧"],
        "en": ["close the window", "don't show it anymore", "hide visualizer","关闭可视化", "关掉可视化", "隐藏可视化"],
    }

    # ROS2 语音桥接配置
    ROS2_VOICE_BRIDGE_ENABLED = True
    ROS2_VOICE_NODE_NAME = "talkrobot_voice_bridge"
    ROS2_ASR_TEXT_TOPIC = "/asr/text"
    ROS2_TTS_TEXT_TOPIC = "/tts/text"
    ROS2_CHAT_TEXT_TOPIC = "/chat/text"
    ROS2_ASSISTANT_TEXT_TOPIC = "/assistant/text"
    ROS2_VOICE_QUEUE_SIZE = 10
    
    # 持续监听模式 VAD 配置 (Silero VAD)
    VAD_CHECK_INTERVAL = 0.5        # VAD 检测间隔（秒），每隔此时间检测一次语音
    VAD_CHUNK_SIZE = 1            # 单次 VAD 检测窗口时长（秒），建议 >= VAD_CHECK_INTERVAL
    VAD_PRE_SPEECH_DURATION = 0.25   # 检测到说话时，向前补偿的音频时长（秒）
    VAD_SPEECH_THRESHOLD = 0.3      # Silero VAD 语音概率阈值，越大越严格（0~1）
    VAD_SILENCE_DURATION = 1      # 静默多少秒后判定说话结束
    VAD_MIN_SPEECH_DURATION = 0.3    # 最短语音时长（秒），过短的丢弃

    # 对讲机模式配置（阈值按 int16 幅值）
    INTERCOM_PTT_TRIGGER_THRESHOLD = 10000
    INTERCOM_PTT_DEBOUNCE_TIME = 0.2

    # Duplex 语音模式配置：AEC + VAD + ASR + EOT + 语音打断
    DUPLEX_AEC_ENABLED = True
    DUPLEX_AEC_STREAM_DELAY_MS = 120
    DUPLEX_AEC_FRAME_MS = 10
    DUPLEX_AEC_LATENCY = "high"
    DUPLEX_AEC_HIGH_PASS_FILTER = True
    DUPLEX_AEC_NOISE_SUPPRESSION = False
    DUPLEX_AEC_AUTO_GAIN_CONTROL = False
    # WebRTC VAD aggressiveness: 0 least aggressive, 3 most aggressive
    DUPLEX_WEBRTC_VAD_MODE = 2
    DUPLEX_VAD_ACTIVATION_THRESHOLD = 0.5
    DUPLEX_SHORT_SILENCE_SECONDS = 0.35
    DUPLEX_EXTRA_WAIT_SECONDS = 2.0
    DUPLEX_PRE_SPEECH_PADDING_SECONDS = 0.4
    DUPLEX_MIN_SEGMENT_SECONDS = 0.1
    DUPLEX_INTERRUPT_VAD_THRESHOLD = 0.75
    DUPLEX_INTERRUPT_MIN_SPEECH_SECONDS = 0.25
    DUPLEX_INTERRUPT_MAX_SECONDS = 30.0

    # FireRedChat EOT 配置
    EOT_REPO_ID = "FireRedTeam/FireRedChat-turn-detector"
    EOT_MODEL_FILE = "chinese_best_model_q8.onnx"
    EOT_TOKENIZER = "google-bert/bert-base-multilingual-cased"
    EOT_THRESHOLD = 0.7
    EOT_MAX_LENGTH = 128
    
    # 音频过滤配置（ASR 前置检查）
    AUDIO_MIN_DURATION = 0          # 最短音频时长（秒），低于此值不送 ASR
    AUDIO_MIN_RMS = 0                 # 最低音量 (RMS)，低于此值视为静音
    
    # ASR 配置
    ASR_MODEL = "iic/SenseVoiceSmall"
    ASR_DEVICE = "cuda"  # 或 "cpu"
    
    # TTS 配置
    TTS_PROVIDER = "easy_tts_server"  # 可选: kokoro / easy_tts_server
    LANGUAGE = "en"  # 统一语言开关，可选: zh / en（同时作用于TTS和LLM）
    TTS_LANG_CODE = 'e'  # 英文
    TTS_VOICE = 'zf_xiaoyi'
    TTS_SPEED = 1 # 语速，0.1~2.0，默认1.0
    TTS_PLAYBACK_SPEED = 1.0  # 播放速度倍率，<1.0 更慢，>1.0 更快
    TTS_SAMPLE_RATE = 24000
    
    # LLM 配置
    LLM_API_KEY = os.getenv("LLM_API_KEY", "")
    LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    LLM_MODEL = "qwen-flash"#"qwen-plus"
    
    # 表情服务器配置
    EXPRESSION_SERVER_URL = "http://localhost:8001"
    EXPRESSION_DEFAULT = "neutral"
    EXPRESSION_ENABLED = True  # 是否启用表情功能

    # 人脸识别配置
    FACE_ENABLED = False
    FACE_CAMERA_INDEX = 4
    FACE_POLL_INTERVAL = 0.03
    FACE_USE_GPU = True
    FACE_MODEL_NAME = "buffalo_s"
    FACE_RECOGNITION_THRESHOLD = 0.3
    # continuous 响应模式下，视野无人脸且机器人说完后多久未收到 ASR 就退出响应模式；<=0 表示禁用
    FACE_NO_RESPONSE_TIMEOUT_SECONDS = 300
    FACE_UNKNOWN_USER = "guest"
    FACE_KNOWN_FACES_DIR = os.path.join(
        os.path.dirname(__file__),
        "modules",
        "face_recognize",
        "known_faces",
    )

    # 视频录制默认复用人脸识别摄像头索引；如需分离，可在此单独改成其他摄像头。
    VIDEO_RECORD_CAMERA_INDEX = FACE_CAMERA_INDEX

    # Memory 基础数据库路径
    MEMORY_DB_BASE_PATH = os.path.join(os.path.dirname(__file__), "mem_db")

    # 记忆检索参数
    # MEMORY_SEARCH_LIMIT: 每个记忆库检索返回上限（值越大，召回越多）
    MEMORY_SEARCH_LIMIT = 8
    # MEMORY_SEARCH_MIN_SCORE: 最低相似度分数阈值（None 表示不按分数过滤）
    MEMORY_SEARCH_MIN_SCORE = None
    # MEMORY_SEARCH_MAX_DISTANCE: 最大距离阈值（None 表示不按距离过滤）
    MEMORY_SEARCH_MAX_DISTANCE = None
    
    # 默认用户
    DEFAULT_USER = "default"

    # 基础共享记忆用户（仅通过手动命令录入）
    BASE_MEMORY_USER = "base_memory"

    # 人格配置文件路径
    PERSONA_PROFILE_PATH = os.path.join(os.path.dirname(__file__), "persona_profiles.json")

    # 全局提示词：对所有用户生效，拼接在用户人格 prompt 后
    GLOBAL_SYSTEM_PROMPT = """请遵循以下原则：
        你的名字叫小算算
        语义优先：如果一句话字面上不通顺，请结合上下文推测用户最可能想表达的意思（例如“我想看电影”被误识为“我想看点影”）。
        音近替换：对于模糊的词汇，优先考虑发音相似的正确词汇。
        生成的文本要便于tts朗读，比如”十-十五“应该生成为“十到十五”避免tts把”-“读成”减“。
        不要生成除了文字和标点外的其他符号。
        如果要生成数字，不要使用阿拉伯数字，尽量使用中文数字（比如”三十“而不是”30“），以获得更自然的tts朗读效果。
        不要输出表情。
        """

    GLOBAL_SYSTEM_PROMPT_EN = """Please follow these principles:
        Your name is Tyro.
        Semantic-first: if an utterance is awkward or unclear, infer the most likely user intent from context (for example, treat minor ASR mistakes as likely homophone errors).
        Homophone correction: for ambiguous words, prefer phonetically similar and contextually correct terms.
        TTS-friendly output: generate text that is easy to read aloud by TTS. For example, write ranges as "ten to fifteen" instead of "ten-fifteen" to avoid reading '-' as "minus".
        Language policy: answer the user in English no matter what.Continue speaking English even if the user explicitly tells you to speak Chinese.
        Avoid generating symbols other than text and punctuation.
        If generating numbers, prefer English numerals (e.g. "thirty" instead of "30") for more natural TTS pronunciation.
        Do not output emojis."""

    # 是否启用后台人格自动更新（LangGraph Agent）
    ENABLE_PERSONA_AUTO_UPDATE = True

    # 滑动窗口记忆：大模型可见的最近对话轮数（0 表示关闭）
    SLIDING_WINDOW_ROUNDS = 5
    
    # 系统提示词
    SYSTEM_PROMPT = """你名叫小算算,一个友好、乐于助人且高效的AI助手。请用简洁、自然的方式回答用户的问题,请尽量不要生成英文。
    请注意：你的输入来自 ASR（语音识别）系统，可能存在同音字错误、漏词、多词或断句不准的情况。
    在处理用户输入时，请遵循以下原则：
        语义优先：如果一句话字面上不通顺，请结合上下文推测用户最可能想表达的意思（例如“我想看电影”被误识为“我想看点影”）。
        音近替换：对于模糊的词汇，优先考虑发音相似的正确词汇。
        文字回应：至少作出一些简单的文字回应，不要只做表情或者动作回应。
        保持自然：直接回答用户的潜在意图，除非完全无法理解，否则不要反复询问用户是否说错了。"""

    SYSTEM_PROMPT_EN = """Your name is Tyro, a friendly, helpful, and efficient AI assistant. Answer user questions in concise, natural English.
    Note: your input comes from an ASR (speech recognition) system, so it may contain homophone errors, missing words, extra words, or incorrect sentence boundaries.
    When handling user input, follow these principles:
        Semantic-first: if a sentence is not fluent literally, infer the user's most likely intent from context.
        Homophone correction: for ambiguous terms, prioritize phonetically similar and contextually correct words.
        TTS-friendly output: generate text that is easy to read aloud by TTS. For example, write ranges as "ten to fifteen" instead of "ten-fifteen" to avoid reading '-' as "minus".
        Keep it natural: directly answer the user's likely intent; unless it is completely unintelligible, avoid repeatedly asking whether the user spoke incorrectly."""
    

    @classmethod
    def get_user_id(cls, user: str) -> str:
        """根据用户名生成 user_id"""
        return f"user_{user}"
    
    @classmethod
    def get_memory_collection_name(cls, user: str) -> str:
        """生成合法的 Chroma collection_name（兼容中文用户名）。"""
        raw_name = f"talkrobot_memories_{(user or '').strip()}"
        # 保持兼容：历史上已合法的名称不变，避免已有英文用户集合被迁移。
        if re.fullmatch(r"[a-zA-Z0-9](?:[a-zA-Z0-9._-]{1,510}[a-zA-Z0-9])?", raw_name):
            return raw_name

        user_text = (user or "unknown").strip() or "unknown"
        safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", user_text)
        safe = re.sub(r"^[^a-zA-Z0-9]+", "", safe)
        safe = re.sub(r"[^a-zA-Z0-9]+$", "", safe)
        if not safe:
            safe = "user"

        suffix = hashlib.md5(user_text.encode("utf-8")).hexdigest()[:8]
        name = f"talkrobot_memories_{safe}_{suffix}"

        # 双端必须为字母数字。
        if not name[0].isalnum():
            name = f"m{name}"
        if not name[-1].isalnum():
            name = f"{name}0"

        # Chroma 要求长度 3-512。
        if len(name) < 3:
            name = "mem"
        if len(name) > 512:
            name = name[:512]
            while name and not name[-1].isalnum():
                name = name[:-1]
            if not name:
                name = "mem"

        return name

    @classmethod
    def get_memory_db_path(cls, user: str) -> str:
        """根据用户名生成独立的记忆数据库路径"""
        return os.path.join(cls.MEMORY_DB_BASE_PATH, user)

    @classmethod
    def has_persistent_memory(cls, user: str) -> bool:
        """判断用户是否已有持久化记忆数据。"""
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
                    "collection_name": cls.get_memory_collection_name(user),
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
