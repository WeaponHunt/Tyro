"""
对话机器人主程序
整合所有模块,启动对话机器人
"""
import sys
import os
import time
import subprocess
import atexit
import argparse
import re
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Dict, Tuple
import numpy as np
from loguru import logger

from talkrobot.config import Config

# 全局变量：表情服务器子进程
_expression_server_process = None

os.environ['HF_HUB_OFFLINE'] = '1'


class FaceIdentityResolver:
    """根据当前摄像头识别人脸，解析当前交互对象。"""

    _NON_TARGET_LABELS = {"", "无人脸", "识别中"}

    def __init__(
        self,
        enabled: bool,
        default_user: str,
        camera_index: int,
        poll_interval: float = Config.FACE_POLL_INTERVAL,
    ):
        self.enabled = False
        self.default_user = default_user
        self.camera_index = camera_index
        self.unknown_user = Config.FACE_UNKNOWN_USER
        self.poll_interval = max(0.0, float(poll_interval))
        self._cap = None
        self._module = None
        self._on_user_change = None
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._track_thread = None
        self._current_user = self.unknown_user
        self._current_is_familiar = False
        self._current_has_face = False
        self._latest_frame = None

        if not enabled:
            return

        try:
            import cv2
            from talkrobot.modules.face_recognize.face_recognition import FaceRecognitionModule

            self._module = FaceRecognitionModule(
                known_faces_dir=Config.FACE_KNOWN_FACES_DIR,
                model_name=Config.FACE_MODEL_NAME,
                use_gpu=Config.FACE_USE_GPU,
                recognition_threshold=Config.FACE_RECOGNITION_THRESHOLD,
            )

            self._cap = cv2.VideoCapture(camera_index)
            if not self._cap.isOpened():
                raise RuntimeError(f"无法打开摄像头: index={camera_index}")

            self.enabled = True
            logger.info(f"人脸识别已启用，摄像头 index={camera_index}")
            self._track_thread = threading.Thread(target=self._tracking_loop, daemon=True)
            self._track_thread.start()
        except Exception as e:
            logger.warning(f"人脸识别初始化失败，已回退普通模式: {e}")
            self.shutdown()

    @staticmethod
    def _sanitize_user_name(name: str) -> str:
        clean = re.sub(r"\s+", "_", name.strip())
        clean = re.sub(r"[^0-9a-zA-Z_\-\u4e00-\u9fff]", "_", clean)
        return clean or Config.DEFAULT_USER

    def _detect_user_once(self) -> Tuple[str, bool, bool]:
        """执行单次人脸检测并返回 (用户, 是否熟人, 是否有人脸)。"""
        if not self.enabled or self._cap is None or self._module is None:
            return self.default_user, False, False

        ok, frame = self._cap.read()
        if not ok:
            logger.debug("读取摄像头帧失败，回退陌生人用户")
            return self.unknown_user, False, False
        with self._state_lock:
            self._latest_frame = frame.copy()

        try:
            result = self._module.process_frame(frame)
            label = str(result.get("label", "")).strip()
            has_face = label != "无人脸" and bool(result.get("tracked", False))
        except Exception as e:
            logger.warning(f"人脸识别处理失败，回退陌生人用户: {e}")
            return self.unknown_user, False, False

        if label in self._NON_TARGET_LABELS:
            return self.unknown_user, False, has_face
        if label == "陌生人":
            return self.unknown_user, False, True
        return self._sanitize_user_name(label), True, True

    def _tracking_loop(self) -> None:
        """后台持续人脸追踪，检测到对象变化时触发回调。"""
        while not self._stop_event.is_set():
            user, is_familiar, has_face = self._detect_user_once()
            changed = False
            with self._state_lock:
                if user != self._current_user or has_face != self._current_has_face:
                    self._current_user = user
                    self._current_is_familiar = is_familiar
                    self._current_has_face = has_face
                    changed = True

            if changed:
                logger.info(f"人脸交互对象变化: {user}, has_face={has_face}")
                callback = self._on_user_change
                if callback is not None:
                    try:
                        callback(user, is_familiar, has_face)
                    except Exception as e:
                        logger.warning(f"人脸用户切换回调异常: {e}")

            self._stop_event.wait(self.poll_interval)

    def set_on_user_change(self, callback) -> None:
        """设置交互对象变化回调。"""
        self._on_user_change = callback

    def resolve_user(self) -> str:
        """返回当前交互对象用户名。"""
        if not self.enabled:
            return self.default_user
        with self._state_lock:
            return self._current_user

    def is_current_user_familiar(self) -> bool:
        """返回当前交互对象是否熟人。"""
        if not self.enabled:
            return False
        with self._state_lock:
            return self._current_is_familiar

    def has_current_face(self) -> bool:
        """返回当前视野中是否有人脸。"""
        if not self.enabled:
            return False
        with self._state_lock:
            return self._current_has_face

    def get_latest_frame(self):
        """返回人脸识别摄像头最新帧副本，用于复用同一路摄像头录像。"""
        with self._state_lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def shutdown(self) -> None:
        self._stop_event.set()
        if self._track_thread is not None and self._track_thread.is_alive():
            self._track_thread.join(timeout=1.0)
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception as e:
                logger.warning(f"释放摄像头失败: {e}")
        self._cap = None
        self._module = None
        self._track_thread = None
        self.enabled = False


class UserMemoryRouter:
    """按用户动态提供记忆模块；无持久记忆时仅启用滑动窗口短期记忆。"""

    def __init__(self):
        self._cache: Dict[str, Tuple[Optional[object], bool]] = {}

    @staticmethod
    def _is_known_face_user(user: str) -> bool:
        """判断用户是否来自 known_faces 人脸图库。"""
        user_key = (user or "").strip()
        if not user_key or user_key == Config.FACE_UNKNOWN_USER:
            return False

        known_dir = Path(Config.FACE_KNOWN_FACES_DIR)
        if not known_dir.is_dir():
            return False

        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        try:
            for file_path in known_dir.iterdir():
                if file_path.suffix.lower() not in image_exts:
                    continue
                raw_name = file_path.stem.strip()
                if not raw_name:
                    continue
                if raw_name == user_key or FaceIdentityResolver._sanitize_user_name(raw_name) == user_key:
                    return True
        except Exception as e:
            logger.warning(f"读取人脸图库失败，无法判断是否为已登记用户: {e}")
        return False

    def get_memory_for_user(self, user: str) -> Tuple[Optional[object], bool]:
        user = (user or Config.DEFAULT_USER).strip()
        if user in self._cache:
            return self._cache[user]

        has_persistent = Config.has_persistent_memory(user)
        # 人脸图库中的已登记用户，即使当前无历史记忆文件，也启用长期记忆并创建新库。
        if not has_persistent and self._is_known_face_user(user):
            has_persistent = True
            logger.info(f"用户[{user}]命中人脸图库，启用长期记忆并自动创建用户记忆库")

        if not has_persistent:
            logger.info(f"用户[{user}]不存在长期记忆，使用短期记忆模式")
            self._cache[user] = (None, False)
            return self._cache[user]

        from talkrobot.modules.memory.memory_module import MemoryModule

        module = MemoryModule(
            config=Config.get_memory_config(user),
            user_id=Config.get_user_id(user)
        )
        self._cache[user] = (module, True)
        logger.info(f"用户[{user}]已接入长期记忆")
        return self._cache[user]

    def shutdown(self) -> None:
        for user, (module, _) in self._cache.items():
            if module is None:
                continue
            try:
                module.shutdown()
            except Exception as e:
                logger.warning(f"关闭用户[{user}]记忆模块异常: {e}")


def _start_expression_server():
    """自动启动表情服务器子进程"""
    global _expression_server_process
    script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "expression", "expression_server.py")
    if not os.path.exists(script_path):
        logger.warning(f"表情服务器脚本不存在: {script_path}")
        return False
    try:
        logger.info(f"正在自动启动表情服务器: {script_path}")
        # 继承环境变量（包括 DISPLAY）以确保 OpenCV 窗口能正常弹出
        env = os.environ.copy()
        _expression_server_process = subprocess.Popen(
            [sys.executable, script_path],
            env=env,
            cwd=os.path.dirname(os.path.dirname(__file__)),
        )
        atexit.register(_stop_expression_server)
        # 等待服务器就绪
        time.sleep(2)
        logger.info(f"表情服务器已启动 (PID: {_expression_server_process.pid})")
        return True
    except Exception as e:
        logger.error(f"启动表情服务器失败: {e}")
        return False


def _stop_expression_server():
    """终止表情服务器子进程"""
    global _expression_server_process
    if _expression_server_process and _expression_server_process.poll() is None:
        logger.info("正在关闭表情服务器...")
        _expression_server_process.terminate()
        try:
            _expression_server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _expression_server_process.kill()
        logger.info("表情服务器已关闭")
    _expression_server_process = None


class DuplexVoiceRuntime:
    """运行 duplex 语音循环，并在 TTS 播放期间捕获语音打断。"""

    def __init__(self, audio_io, turn_detector, tts_module, turn_config, interruption_config):
        self.audio_io = audio_io
        self.turn_detector = turn_detector
        self.tts = tts_module
        self.turn_config = turn_config
        self.interruption_config = interruption_config
        self._pending_turn = None
        self._pending_lock = threading.Lock()

    def start(self) -> None:
        self.audio_io.start()

    def close(self) -> None:
        self.audio_io.close()

    def pop_pending_turn(self):
        with self._pending_lock:
            pending = self._pending_turn
            self._pending_turn = None
        return pending

    def _set_pending_turn(self, turn) -> None:
        with self._pending_lock:
            self._pending_turn = turn

    def play_text(self, text: str) -> None:
        """Synthesize text, play through AEC far-end stream, and listen for interruption."""
        if not text:
            return

        from talkrobot.core.duplex_turn_detector import InterruptionGate, TurnState

        audio_chunks = self.tts.synthesize(text, play_audio=False)
        if not audio_chunks:
            return

        self.audio_io.start_playback(audio_chunks, self.tts.sample_rate, discard_input=False)
        interrupted = False
        started_at = time.perf_counter()
        interruption_gate = InterruptionGate(self.interruption_config, self.turn_config)

        while self.audio_io.is_playing():
            interrupted_flag = getattr(self.tts, "_interrupted", None)
            if interrupted_flag is not None and interrupted_flag.is_set():
                self.audio_io.stop_playback(drain_input=True)
                return

            chunk = self.audio_io.get_mic_chunk(timeout=0.05)
            if chunk is None:
                continue

            confirmed_chunks = interruption_gate.process_chunk(chunk)
            if confirmed_chunks is None:
                continue

            interrupted = True
            self.audio_io.stop_playback(drain_input=False)
            print("🎙️ 检测到用户语音打断，已停止当前播报")
            turn = self.turn_detector.process_audio_chunks(confirmed_chunks, finalize_at_end=False)
            if turn is not None:
                self._set_pending_turn(turn)
                return
            break

        if not interrupted:
            self.audio_io.wait_for_playback(drain_input=True)
            return

        while time.perf_counter() - started_at < self.interruption_config.max_interruption_seconds:
            chunk = self.audio_io.get_mic_chunk(timeout=0.1)
            if chunk is None:
                continue

            turn = self.turn_detector.process_audio_chunks([chunk], finalize_at_end=False)
            if turn is not None:
                self._set_pending_turn(turn)
                return

            if self.turn_detector.state == TurnState.IDLE:
                logger.info("duplex 语音打断被判定为误触发，已忽略")
                return

        logger.info("duplex 语音打断等待完整用户 turn 超时，已重置状态")
        self.turn_detector.reset_turn()

    def run_forever(self, conversation_manager) -> None:
        print("\n🎧 duplex 模式已启动：AEC + VAD/EOT 结束判断 + 语音打断")
        while True:
            turn = self.pop_pending_turn()
            while turn is None:
                chunk = self.audio_io.get_mic_chunk(timeout=0.1)
                if chunk is None:
                    continue
                turn = self.turn_detector.process_audio_chunks([chunk], finalize_at_end=False)

            if turn.user_text:
                logger.info(
                    f"duplex turn complete: reason={turn.finish_reason}, "
                    f"eot={turn.eot_probability:.3f}, text={turn.user_text}"
                )
                conversation_manager.process_recognized_text(turn.user_text)


def _setup_logger():
    """配置日志"""
    level = "DEBUG" if Config.DEBUG else "INFO"
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:8}</level> | <level>{message}</level>",
        level=level
    )
    logger.add("talkrobot/logs/robot_{time}.log", rotation="1 day", retention="7 days", level=level)


def run_chat(args):
    """启动对话机器人"""
    from talkrobot.modules.tts.tts_module import TTSModule
    from talkrobot.modules.llm.llm_module import LLMModule
    from talkrobot.modules.llm.persona_update_agent import PersonaUpdateAgent
    from talkrobot.modules.memory.memory_module import MemoryModule
    from talkrobot.core.persona_manager import PersonaManager
    from talkrobot.core.conversation_manager import ConversationManager

    user = args.user
    memory_module = None
    base_memory_module = None
    memory_router = None
    persona_manager = None
    persona_update_agent = None
    face_resolver = None
    tts_module = None
    audio_recorder = None
    duplex_runtime = None
    ros2_bridge = None
    video_recorder = None
    conversation_manager = None

    try:
        _setup_logger()

        language = (getattr(args, "language", None) or Config.LANGUAGE or "zh").strip().lower()
        if language not in {"zh", "en"}:
            logger.warning(f"无效的语言配置: {language}，回退为 zh")
            language = "zh"

        if language == "en":
            selected_system_prompt = Config.SYSTEM_PROMPT_EN
            selected_global_system_prompt = Config.GLOBAL_SYSTEM_PROMPT_EN
        else:
            selected_system_prompt = Config.SYSTEM_PROMPT
            selected_global_system_prompt = Config.GLOBAL_SYSTEM_PROMPT

        logger.info("="*50)
        logger.info(f"正在初始化对话机器人系统... (用户: {user})")
        logger.info("="*50)

        # 1. 初始化各个模块
        no_asr_mode = getattr(args, "no_asr", False)
        asr_module = None
        if not no_asr_mode:
            from talkrobot.modules.asr.asr_module import ASRModule
            asr_module = ASRModule(
                model_name=Config.ASR_MODEL,
                device=Config.ASR_DEVICE
            )

        tts_provider = getattr(args, "tts_provider", None) or Config.TTS_PROVIDER

        tts_module = TTSModule(
            lang_code=Config.TTS_LANG_CODE,
            voice=Config.TTS_VOICE,
            speed=Config.TTS_SPEED,
            playback_speed=Config.TTS_PLAYBACK_SPEED,
            provider=tts_provider,
            language=language,
            sample_rate=Config.TTS_SAMPLE_RATE,
        )

        # 初始化表情模块（可选）
        expression_module = None
        expression_prompt = ""
        if Config.EXPRESSION_ENABLED:
            from talkrobot.modules.expression.expression_module import ExpressionModule
            # 自动启动表情服务器
            _start_expression_server()
            expression_module = ExpressionModule(
                server_url=Config.EXPRESSION_SERVER_URL,
                default_expression=Config.EXPRESSION_DEFAULT
            )
            if expression_module.is_available:
                expression_prompt = ExpressionModule.get_expression_prompt()

        llm_module = LLMModule(
            api_key=Config.LLM_API_KEY,
            base_url=Config.LLM_BASE_URL,
            model=Config.LLM_MODEL,
            system_prompt=selected_system_prompt,
            expression_prompt=expression_prompt,
            language=language,
            visualizer_enable_topic=Config.VISUALIZER_ENABLE_TOPIC,
        )

        disable_persona_auto_update = bool(getattr(args, "disable_persona_auto_update", False))
        enable_persona_auto_update = bool(Config.ENABLE_PERSONA_AUTO_UPDATE and not disable_persona_auto_update)
        logger.info(
            f"后台人格自动更新: {'开启' if enable_persona_auto_update else '关闭'}"
            f" (config={Config.ENABLE_PERSONA_AUTO_UPDATE}, cli_disable={disable_persona_auto_update})"
        )

        persona_manager = PersonaManager(
            profile_path=Config.PERSONA_PROFILE_PATH,
            fallback_prompt=selected_system_prompt,
        )

        if enable_persona_auto_update:
            persona_update_agent = PersonaUpdateAgent(
                llm_client=llm_module.client,
                model=Config.LLM_MODEL,
            )

        def _persona_provider(current_user: str) -> str:
            persona_prompt = persona_manager.get_prompt_for_user(current_user, language=language)
            global_prompt = (selected_global_system_prompt or "").strip()
            sections = []
            if persona_prompt and str(persona_prompt).strip():
                sections.append(str(persona_prompt).strip())
            if global_prompt:
                sections.append(global_prompt)
            if expression_prompt and str(expression_prompt).strip():
                sections.append(str(expression_prompt).strip())
            return "\n\n".join(sections)

        persona_update_handler = None
        if enable_persona_auto_update:
            def _persona_update_handler(current_user: str, user_text: str, context: str) -> None:
                handler_start = time.perf_counter()
                if persona_update_agent is None:
                    return

                current_prompt = persona_manager.get_prompt_for_user(current_user, language=language)
                agent_start = time.perf_counter()
                result = persona_update_agent.run(
                    user=current_user,
                    user_input=user_text,
                    context=context,
                    current_prompt=current_prompt,
                )
                agent_elapsed = time.perf_counter() - agent_start
                logger.debug(
                    "人格更新handler耗时: "
                    f"user={current_user}, agent_call={agent_elapsed:.3f}s, "
                    f"agent_total={float(result.get('total_elapsed_s', 0.0) or 0.0):.3f}s, "
                    f"graph={float(result.get('graph_elapsed_s', 0.0) or 0.0):.3f}s, "
                    f"decide={float(result.get('decide_elapsed_s', 0.0) or 0.0):.3f}s, "
                    f"propose={float(result.get('propose_elapsed_s', 0.0) or 0.0):.3f}s"
                )

                if not result.get("should_update", False):
                    reason = result.get("reason", "skip")
                    logger.debug(f"人格更新跳过: user={current_user}, reason={reason}")
                    logger.debug(
                        f"人格更新handler总耗时: user={current_user}, total={time.perf_counter() - handler_start:.3f}s"
                    )
                    return

                updated_prompt = str(result.get("updated_prompt", "") or "").strip()
                if not updated_prompt:
                    logger.debug(f"人格更新跳过: user={current_user}, reason=empty_prompt")
                    logger.debug(
                        f"人格更新handler总耗时: user={current_user}, total={time.perf_counter() - handler_start:.3f}s"
                    )
                    return

                persist_start = time.perf_counter()
                if persona_manager.update_user_prompt(current_user, updated_prompt, language=language):
                    persist_elapsed = time.perf_counter() - persist_start
                    logger.info(
                        f"人格提示词已后台更新: user={current_user}, confidence={result.get('confidence', 0):.2f}"
                    )
                    logger.debug(
                        f"人格更新写回耗时: user={current_user}, persist={persist_elapsed:.3f}s"
                    )

                logger.debug(
                    f"人格更新handler总耗时: user={current_user}, total={time.perf_counter() - handler_start:.3f}s"
                )

            persona_update_handler = _persona_update_handler

        enable_face = bool(getattr(args, "enable_face", False))
        memory_provider = None
        user_resolver = None

        if enable_face:
            face_resolver = FaceIdentityResolver(
                enabled=True,
                default_user=user,
                camera_index=getattr(args, "face_camera_index", Config.FACE_CAMERA_INDEX),
                poll_interval=getattr(args, "face_poll_interval", Config.FACE_POLL_INTERVAL),
            )
            if face_resolver.enabled:
                memory_router = UserMemoryRouter()
                user_resolver = face_resolver.resolve_user
                memory_provider = memory_router.get_memory_for_user
                memory_module, _ = memory_provider(user)
                logger.info("已启用人脸驱动交互对象切换")
            else:
                logger.warning("人脸识别不可用，继续使用固定用户模式")

        if memory_provider is None:
            memory_module = MemoryModule(
                config=Config.get_memory_config(user),
                user_id=Config.get_user_id(user)
            )

        # 基础共享记忆：所有用户仅参与检索，不参与日常自动写入。
        base_memory_module = MemoryModule(
            config=Config.get_memory_config(Config.BASE_MEMORY_USER),
            user_id=Config.get_user_id(Config.BASE_MEMORY_USER)
        )

        # 2. 创建对话管理器
        tts_enabled = not args.no_tts
        listen_mode = getattr(args, 'listen_mode', None) or Config.DEFAULT_LISTEN_MODE
        history_rounds = getattr(args, 'history_rounds', None)
        if history_rounds is None:
            history_rounds = Config.SLIDING_WINDOW_ROUNDS
        if history_rounds < 0:
            raise ValueError("history-rounds 不能小于 0")

        if no_asr_mode and listen_mode != Config.DEFAULT_LISTEN_MODE:
            logger.warning("no-asr 模式下 listen-mode 参数无效，将使用终端文本输入")
        if no_asr_mode and listen_mode == "duplex":
            raise ValueError("duplex 模式依赖 ASR，不能与 --no-asr 同时使用")
        if listen_mode == "duplex" and getattr(args, "streaming", False):
            logger.warning("duplex 模式需要完整 TTS 音频作为 AEC far-end，已自动关闭 streaming")
            args.streaming = False

        # 3. 创建音频录制器（no-asr 模式下跳过）
        if not no_asr_mode and listen_mode != "duplex":
            from talkrobot.core.audio_recorder import AudioRecorder
            audio_recorder = AudioRecorder(
                sample_rate=Config.SAMPLE_RATE,
                channels=Config.CHANNELS,
                listen_mode=listen_mode,
                tts_interrupt_key=Config.TTS_INTERRUPT_KEY,
                intercom_toggle_key=Config.INTERCOM_PTT_TOGGLE_KEY,
                vad_check_interval=Config.VAD_CHECK_INTERVAL,
                vad_chunk_size=Config.VAD_CHUNK_SIZE,
                pre_speech_duration=Config.VAD_PRE_SPEECH_DURATION,
                vad_speech_threshold=Config.VAD_SPEECH_THRESHOLD,
                silence_duration=Config.VAD_SILENCE_DURATION,
                min_speech_duration=Config.VAD_MIN_SPEECH_DURATION,
                ptt_trigger_threshold=Config.INTERCOM_PTT_TRIGGER_THRESHOLD,
                ptt_debounce_time=Config.INTERCOM_PTT_DEBOUNCE_TIME,
            )
        else:
            audio_recorder = None

        tts_playback_handler = None
        if listen_mode == "duplex":
            from talkrobot.core.duplex_audio_io import DuplexAudioIO
            from talkrobot.core.duplex_turn_detector import (
                DuplexTurnConfig,
                DuplexTurnDetector,
                InterruptionConfig,
            )
            from talkrobot.modules.eot.eot_module import EOTModule

            turn_config = DuplexTurnConfig(
                sample_rate=Config.SAMPLE_RATE,
                vad_mode=Config.DUPLEX_WEBRTC_VAD_MODE,
                vad_activation_threshold=Config.DUPLEX_VAD_ACTIVATION_THRESHOLD,
                short_silence_seconds=Config.DUPLEX_SHORT_SILENCE_SECONDS,
                extra_wait_seconds=Config.DUPLEX_EXTRA_WAIT_SECONDS,
                pre_speech_padding_seconds=Config.DUPLEX_PRE_SPEECH_PADDING_SECONDS,
                min_segment_seconds=Config.DUPLEX_MIN_SEGMENT_SECONDS,
                verbose=Config.DEBUG,
            )
            interruption_config = InterruptionConfig(
                vad_activation_threshold=Config.DUPLEX_INTERRUPT_VAD_THRESHOLD,
                min_speech_seconds=Config.DUPLEX_INTERRUPT_MIN_SPEECH_SECONDS,
                max_interruption_seconds=Config.DUPLEX_INTERRUPT_MAX_SECONDS,
            )
            duplex_audio_io = DuplexAudioIO(
                sample_rate=Config.SAMPLE_RATE,
                frame_ms=Config.DUPLEX_AEC_FRAME_MS,
                stream_delay_ms=Config.DUPLEX_AEC_STREAM_DELAY_MS,
                latency=Config.DUPLEX_AEC_LATENCY,
                high_pass_filter=Config.DUPLEX_AEC_HIGH_PASS_FILTER,
                noise_suppression=Config.DUPLEX_AEC_NOISE_SUPPRESSION,
                auto_gain_control=Config.DUPLEX_AEC_AUTO_GAIN_CONTROL,
            )
            eot_module = EOTModule(
                repo_id=Config.EOT_REPO_ID,
                model_file=Config.EOT_MODEL_FILE,
                tokenizer_name=Config.EOT_TOKENIZER,
                threshold=Config.EOT_THRESHOLD,
                max_length=Config.EOT_MAX_LENGTH,
                allow_download=False,
            )
            duplex_turn_detector = DuplexTurnDetector(turn_config, asr_module, eot_module)
            duplex_runtime = DuplexVoiceRuntime(
                audio_io=duplex_audio_io,
                turn_detector=duplex_turn_detector,
                tts_module=tts_module,
                turn_config=turn_config,
                interruption_config=interruption_config,
            )
            tts_playback_handler = duplex_runtime.play_text

        ros2_voice_enabled = bool(getattr(args, "ros2_voice", Config.ROS2_VOICE_BRIDGE_ENABLED))
        if ros2_voice_enabled:
            from talkrobot.modules.ros2.ros2_bridge import Ros2VoiceBridge
            ros2_bridge = Ros2VoiceBridge(
                tts_module=tts_module,
                audio_recorder=audio_recorder,
                node_name=getattr(args, "ros2_voice_node_name", Config.ROS2_VOICE_NODE_NAME),
                asr_text_topic=getattr(args, "ros2_asr_topic", Config.ROS2_ASR_TEXT_TOPIC),
                tts_text_topic=getattr(args, "ros2_tts_topic", Config.ROS2_TTS_TEXT_TOPIC),
                chat_text_topic=getattr(args, "ros2_chat_topic", Config.ROS2_CHAT_TEXT_TOPIC),
                assistant_text_topic=getattr(args, "ros2_assistant_topic", Config.ROS2_ASSISTANT_TEXT_TOPIC),
                queue_size=getattr(args, "ros2_voice_queue_size", Config.ROS2_VOICE_QUEUE_SIZE),
            )
            ros2_bridge.start()
        else:
            logger.info("ROS2 语音 bridge 已关闭")

        conversation_manager = ConversationManager(
            asr_module=asr_module,
            tts_module=tts_module,
            llm_module=llm_module,
            memory_module=memory_module,
            base_memory_module=base_memory_module,
            tts_enabled=tts_enabled,
            streaming=getattr(args, 'streaming', False),
            expression_module=expression_module,
            audio_recorder=audio_recorder,
            audio_min_duration=Config.AUDIO_MIN_DURATION,
            audio_min_rms=Config.AUDIO_MIN_RMS,
            sample_rate=Config.SAMPLE_RATE,
            debug_timing=Config.DEBUG,
            history_rounds=history_rounds,
            default_user=user,
            user_resolver=user_resolver,
            memory_provider=memory_provider,
            persona_provider=_persona_provider,
            persona_update_handler=persona_update_handler,
            language=language,
            say_hallo=getattr(args, 'say_hallo', False),
            greeting_cooldown_seconds=getattr(args, 'hallo_cooldown_seconds', 600.0),
            sleep_toggle_key=Config.MODE_SWITCH_SLEEP_KEY,
            script_toggle_key=Config.MODE_SWITCH_SCRIPT_KEY,
            script_dir=Config.SCRIPT_DIR,
            script_file=Config.SCRIPT_FILE,
            script_configs=Config.SCRIPT_CONFIGS,
            script_pause_resume_key=Config.SCRIPT_PAUSE_RESUME_KEY,
            tts_interrupt_key=Config.TTS_INTERRUPT_KEY,
            sleep_enable_voice_words=Config.MODE_SWITCH_SLEEP_ENABLE_VOICE_WORDS.get(
                language,
                Config.MODE_SWITCH_SLEEP_ENABLE_VOICE_WORDS.get("zh", []),
            ),
            sleep_disable_voice_words=Config.MODE_SWITCH_SLEEP_DISABLE_VOICE_WORDS.get(
                language,
                Config.MODE_SWITCH_SLEEP_DISABLE_VOICE_WORDS.get("zh", []),
            ),
            script_enable_voice_words=Config.MODE_SWITCH_SCRIPT_ENABLE_VOICE_WORDS.get(
                language,
                Config.MODE_SWITCH_SCRIPT_ENABLE_VOICE_WORDS.get("zh", []),
            ),
            script_disable_voice_words=Config.MODE_SWITCH_SCRIPT_DISABLE_VOICE_WORDS.get(
                language,
                Config.MODE_SWITCH_SCRIPT_DISABLE_VOICE_WORDS.get("zh", []),
            ),
            visualizer_enable_topic=Config.VISUALIZER_ENABLE_TOPIC,
            visualizer_enable_voice_words=Config.VISUALIZER_ENABLE_VOICE_WORDS.get(
                language,
                Config.VISUALIZER_ENABLE_VOICE_WORDS.get("zh", []),
            ),
            visualizer_disable_voice_words=Config.VISUALIZER_DISABLE_VOICE_WORDS.get(
                language,
                Config.VISUALIZER_DISABLE_VOICE_WORDS.get("zh", []),
            ),
            script_image_screen_index=Config.SCRIPT_IMAGE_SCREEN_INDEX,
            script_image_window_x=Config.SCRIPT_IMAGE_WINDOW_X,
            script_image_window_y=Config.SCRIPT_IMAGE_WINDOW_Y,
            script_image_fullscreen=Config.SCRIPT_IMAGE_FULLSCREEN,
            script_image_target_width=Config.SCRIPT_IMAGE_TARGET_WIDTH,
            script_image_target_height=Config.SCRIPT_IMAGE_TARGET_HEIGHT,
            script_image_force_window_size=Config.SCRIPT_IMAGE_FORCE_WINDOW_SIZE,
            script_image_force_resize_image=Config.SCRIPT_IMAGE_FORCE_RESIZE_IMAGE,
            no_face_response_timeout_seconds=Config.FACE_NO_RESPONSE_TIMEOUT_SECONDS,
            tts_playback_handler=tts_playback_handler,
            ros2_bridge=ros2_bridge,
        )
        if ros2_bridge is not None:
            ros2_bridge.set_chat_text_callback(conversation_manager.process_text)

        if face_resolver is not None and face_resolver.enabled:
            face_resolver.set_on_user_change(conversation_manager.on_face_user_change)
            conversation_manager.on_face_user_change(
                face_resolver.resolve_user(),
                face_resolver.is_current_user_familiar(),
                face_resolver.has_current_face(),
            )

        def _on_video_recording_changed(recording: bool) -> None:
            if expression_module is not None and expression_module.is_available:
                expression_module.set_recording(recording)

        video_frame_getter = None
        video_camera_index = Config.VIDEO_RECORD_CAMERA_INDEX
        if face_resolver is not None and face_resolver.enabled:
            if int(video_camera_index) == int(getattr(args, "face_camera_index", Config.FACE_CAMERA_INDEX)):
                video_frame_getter = face_resolver.get_latest_frame
                logger.info("视频录制将复用人脸识别摄像头帧")

        from talkrobot.core.video_recorder import VideoRecorder
        video_recorder = VideoRecorder(
            camera_index=video_camera_index,
            output_dir=Config.VIDEO_RECORD_DIR,
            toggle_key=Config.VIDEO_RECORD_TOGGLE_KEY,
            fps=Config.VIDEO_RECORD_FPS,
            audio_enabled=Config.VIDEO_RECORD_AUDIO_ENABLED,
            audio_sample_rate=Config.SAMPLE_RATE,
            audio_channels=Config.CHANNELS,
            frame_getter=video_frame_getter,
            on_recording_changed=_on_video_recording_changed,
        )
        video_recorder.start()

        logger.info("所有模块初始化完成!")
        logger.info("="*50)

        # 4. 开始对话：语音模式 or 终端文本模式
        if no_asr_mode:
            if language == "en":
                print("\n⌨️ no-asr mode enabled: type text directly in terminal")
                print("   Enter q / quit / exit to quit\n")
            else:
                print("\n⌨️ 已启用 no-asr 模式：请直接在终端输入文本对话")
                print("   输入 q / quit / exit 退出\n")
            while True:
                try:
                    user_text = input("You: " if language == "en" else "你: ").strip()
                except EOFError:
                    break

                if not user_text:
                    continue
                if user_text.lower() in ("q", "quit", "exit"):
                    break

                conversation_manager.process_text(user_text)
        else:
            if duplex_runtime is not None:
                duplex_runtime.start()
                duplex_runtime.run_forever(conversation_manager)
            else:
                audio_recorder.start(
                    on_audio_complete=conversation_manager.process_audio_async
                )

    except KeyboardInterrupt:
        if 'language' in locals() and language == "en":
            print("\n\n👋 Program exited. Goodbye!")
        else:
            print("\n\n👋 程序已退出,再见!")
        logger.info("用户主动退出程序")
        
        # Debug模式：打印所有线程堆栈
        if Config.DEBUG:
            import threading
            import traceback
            print("\n" + "="*60)
            print("🐛 DEBUG模式 - 线程堆栈信息:")
            print("="*60)
            for thread in threading.enumerate():
                print(f"\n线程: {thread.name} (ID: {thread.ident}, Alive: {thread.is_alive()})")
                if thread.ident:
                    frame = sys._current_frames().get(thread.ident)
                    if frame:
                        print("堆栈:")
                        traceback.print_stack(frame)
            print("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"程序运行出错: {e}", exc_info=True)
        print(f"\n❌ 程序出错: {e}")
    finally:
        # 统一收尾：先停播放，再停录音，再等待处理线程，最后关闭外部资源
        if ros2_bridge is not None:
            try:
                ros2_bridge.stop()
            except Exception as e:
                logger.warning(f"停止 ROS2 语音 bridge 异常: {e}")

        if tts_module is not None:
            try:
                tts_module.stop()
            except Exception as e:
                logger.warning(f"停止TTS异常: {e}")

        if audio_recorder is not None:
            try:
                audio_recorder.stop()
            except Exception as e:
                logger.warning(f"停止录音器异常: {e}")

        if duplex_runtime is not None:
            try:
                duplex_runtime.close()
            except Exception as e:
                logger.warning(f"停止 duplex 音频流异常: {e}")

        if video_recorder is not None:
            try:
                video_recorder.stop()
            except Exception as e:
                logger.warning(f"停止视频录制器异常: {e}")

        if conversation_manager is not None:
            try:
                conversation_manager.shutdown(timeout=5.0)
            except Exception as e:
                logger.warning(f"关闭对话管理器异常: {e}")

        if memory_router is not None:
            try:
                memory_router.shutdown()
            except Exception as e:
                logger.warning(f"关闭多用户记忆路由器异常: {e}")
        elif memory_module is not None:
            try:
                memory_module.shutdown()
            except Exception as e:
                logger.warning(f"关闭记忆模块异常: {e}")

        if base_memory_module is not None:
            try:
                base_memory_module.shutdown()
            except Exception as e:
                logger.warning(f"关闭基础记忆模块异常: {e}")

        if face_resolver is not None:
            try:
                face_resolver.shutdown()
            except Exception as e:
                logger.warning(f"关闭人脸识别异常: {e}")

        _stop_expression_server()


def run_add_memory(args):
    """手动添加记忆"""
    _setup_logger()
    from talkrobot.modules.memory.memory_module import MemoryModule

    user = (args.user or Config.DEFAULT_USER).strip()
    content = args.content
    all_users_mode = bool(getattr(args, "all_users", False))

    if all_users_mode:
        users = _collect_all_memory_users()
        logger.info(f"正在为全部用户添加记忆，共 {len(users)} 人: {users}")
    else:
        users = [user]
        logger.info(f"正在为用户 [{user}] 添加记忆...")

    memory_modules = {}
    for target_user in users:
        memory_modules[target_user] = MemoryModule(
            config=Config.get_memory_config(target_user),
            user_id=Config.get_user_id(target_user)
        )

    def _add_to_targets(text: str) -> None:
        for target_user in users:
            memory_modules[target_user].add_memory(text, async_mode=False)
            if all_users_mode:
                print(f"  ✅ 已为用户 [{target_user}] 添加记忆")

    try:
        if content:
            # 直接通过命令行参数添加
            _add_to_targets(content)
            if all_users_mode:
                print(f"✅ 已为全部用户添加记忆: {content}")
            else:
                print(f"✅ 已为用户 [{user}] 添加记忆: {content}")
        else:
            # 交互式添加模式
            if all_users_mode:
                print(f"📝 进入交互式记忆添加模式 (全部用户，共 {len(users)} 人)")
            else:
                print(f"📝 进入交互式记忆添加模式 (用户: {user})")
            print("   输入记忆内容后回车添加，输入 q 或 quit 退出\n")
            while True:
                try:
                    text = input("请输入记忆内容: ").strip()
                except EOFError:
                    break
                if not text:
                    continue
                if text.lower() in ("q", "quit", "exit"):
                    break
                _add_to_targets(text)
                print(f"  ✅ 已添加: {text}\n")
    finally:
        for memory_module in memory_modules.values():
            memory_module.shutdown()
        print("👋 记忆模块已关闭")


def run_add_base_memory(args):
    """手动添加基础共享记忆（仅手动录入）。"""
    _setup_logger()
    from talkrobot.modules.memory.memory_module import MemoryModule

    def _split_base_memory_sentences(text: str) -> list:
        """按句子切分基础共享记忆，避免整段长文只存成一条。"""
        pieces = re.split(r"[。！？!?；;\n]+", str(text or ""))
        sentences = []
        for piece in pieces:
            sentence = piece.strip(" \t\r,，。！？!?；;")
            if sentence:
                sentences.append(sentence)
        return sentences

    def _add_base_sentences(raw_text: str) -> int:
        sentences = _split_base_memory_sentences(raw_text)
        for sentence in sentences:
            base_memory.add_memory(sentence, async_mode=False, infer=False)
        return len(sentences)

    content = args.content
    base_memory = MemoryModule(
        config=Config.get_memory_config(Config.BASE_MEMORY_USER),
        user_id=Config.get_user_id(Config.BASE_MEMORY_USER)
    )

    logger.info(f"正在向基础记忆[{Config.BASE_MEMORY_USER}]添加内容...")
    try:
        if content:
            count = _add_base_sentences(content)
            print(f"✅ 已按句切分并添加基础记忆 {count} 条")
        else:
            print(f"📝 进入基础记忆交互式添加模式 ({Config.BASE_MEMORY_USER})")
            print("   输入记忆内容后回车添加，输入 q 或 quit 退出\n")
            while True:
                try:
                    text = input("请输入基础记忆内容: ").strip()
                except EOFError:
                    break
                if not text:
                    continue
                if text.lower() in ("q", "quit", "exit"):
                    break
                count = _add_base_sentences(text)
                print(f"  ✅ 已按句切分并添加基础记忆 {count} 条\n")
    finally:
        base_memory.shutdown()
        print("👋 基础记忆模块已关闭")


def run_query_memory(args):
    """手动检索记忆（只读，不写入），用于快速验证记忆是否已存储。"""
    _setup_logger()
    from talkrobot.modules.memory.memory_module import MemoryModule

    user = (args.user or Config.DEFAULT_USER).strip()
    query = args.query
    limit = max(1, int(args.limit))
    include_base = bool(getattr(args, "include_base", False))

    user_memory = MemoryModule(
        config=Config.get_memory_config(user),
        user_id=Config.get_user_id(user),
    )
    base_memory = None
    if include_base:
        base_memory = MemoryModule(
            config=Config.get_memory_config(Config.BASE_MEMORY_USER),
            user_id=Config.get_user_id(Config.BASE_MEMORY_USER),
        )

    def _run_query_once(text: str) -> None:
        q = (text or "").strip()
        if not q:
            print("⚠️ 查询内容为空，已跳过")
            return

        user_context = ""
        base_context = ""

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                "user": executor.submit(
                    user_memory.search_memory,
                    q,
                    limit,
                    Config.MEMORY_SEARCH_MIN_SCORE,
                    Config.MEMORY_SEARCH_MAX_DISTANCE,
                )
            }
            if base_memory is not None:
                futures["base"] = executor.submit(
                    base_memory.search_memory,
                    q,
                    limit,
                    Config.MEMORY_SEARCH_MIN_SCORE,
                    Config.MEMORY_SEARCH_MAX_DISTANCE,
                )

            for name, future in futures.items():
                try:
                    result = future.result()
                except Exception as e:
                    logger.warning(f"{name}记忆检索失败: {e}")
                    result = ""
                if name == "user":
                    user_context = result or ""
                else:
                    base_context = result or ""

        print("\n" + "=" * 56)
        print(f"🔎 查询: {q}")
        print(f"👤 用户记忆 [{user}] 命中:")
        print(user_context if user_context else "(无命中)")

        if base_memory is not None:
            print(f"\n📚 基础记忆 [{Config.BASE_MEMORY_USER}] 命中:")
            print(base_context if base_context else "(无命中)")

        merged_parts = []
        if user_context:
            merged_parts.append(f"用户长期记忆:\n{user_context}")
        if base_context:
            merged_parts.append(f"基础共享记忆:\n{base_context}")
        merged = "\n\n".join(merged_parts)
        print("\n🧠 合并结果:")
        print(merged if merged else "(无命中)")
        print("=" * 56 + "\n")

    try:
        if query:
            _run_query_once(query)
        else:
            print(f"🧪 进入记忆检索测试模式 (user={user}, include_base={include_base}, limit={limit})")
            print("   输入查询内容后回车检索，输入 q 或 quit 退出\n")
            while True:
                try:
                    text = input("请输入检索查询: ").strip()
                except EOFError:
                    break
                if not text:
                    continue
                if text.lower() in ("q", "quit", "exit"):
                    break
                _run_query_once(text)
    finally:
        user_memory.shutdown()
        if base_memory is not None:
            base_memory.shutdown()
        print("👋 记忆检索测试已结束")


def _collect_all_memory_users() -> list:
    """收集可写入记忆的全部用户（包含陌生人用户）。"""
    users = set()

    # 已有人格配置中的用户
    profile_path = Config.PERSONA_PROFILE_PATH
    if profile_path and os.path.exists(profile_path):
        try:
            with open(profile_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                profiles = raw.get("users") if isinstance(raw.get("users"), dict) else raw
                for key in profiles.keys():
                    if isinstance(key, str) and key.strip():
                        users.add(key.strip())
        except Exception as e:
            logger.warning(f"读取人格配置失败，跳过该来源: {e}")

    # 已有持久化记忆目录中的用户
    db_base = Config.MEMORY_DB_BASE_PATH
    if db_base and os.path.isdir(db_base):
        try:
            for name in os.listdir(db_base):
                path = os.path.join(db_base, name)
                if os.path.isdir(path) and name.strip() and name.strip() != Config.BASE_MEMORY_USER:
                    users.add(name.strip())
        except Exception as e:
            logger.warning(f"读取记忆目录失败，跳过该来源: {e}")

    users.add(Config.DEFAULT_USER)
    users.add(Config.FACE_UNKNOWN_USER)
    return sorted(users)


def run_vad_debug(args):
    """仅启动 VAD 监听模块，用于阈值与分数调试。"""
    _setup_logger()

    from talkrobot.core.audio_recorder import AudioRecorder

    def on_audio_complete(audio_data):
        # score-only 调试模式下不会触发到这里，保留仅为接口兼容。
        return

    recorder = AudioRecorder(
        sample_rate=args.sample_rate,
        channels=args.channels,
        listen_mode="continuous",
        vad_check_interval=args.vad_check_interval,
        vad_chunk_size=args.vad_chunk_size,
        pre_speech_duration=args.vad_pre_speech_duration,
        vad_speech_threshold=args.vad_speech_threshold,
        silence_duration=args.vad_silence_duration,
        min_speech_duration=args.vad_min_speech_duration,
        vad_debug_scores=True,
        vad_scores_only_mode=True,
    )

    print("\n🔧 VAD Debug 模式")
    print("   - 仅启动麦克风 + VAD，不加载 ASR/LLM/TTS")
    print("   - 仅输出每个 chunk 的 VAD score/rms/events，不做语音段处理")
    print(f"   - check_interval={args.vad_check_interval}s, chunk_size={args.vad_chunk_size}s")
    print(f"   - threshold={args.vad_speech_threshold}, silence={args.vad_silence_duration}s, min_speech={args.vad_min_speech_duration}s")

    recorder.start(on_audio_complete=on_audio_complete)


def run_setup_duplex(args):
    """下载并检查 duplex 模式所需资源。"""
    _setup_logger()
    os.environ.pop("HF_HUB_OFFLINE", None)

    from talkrobot.modules.eot.eot_module import EOTModule, download_eot_resources
    from talkrobot.core.duplex_turn_detector import WebRTCSpeechVAD

    print("🔧 正在准备 duplex 语音模式环境...")
    download_eot_resources(
        repo_id=args.eot_repo_id,
        model_file=args.eot_model_file,
        tokenizer_name=args.eot_tokenizer,
    )

    detector = EOTModule(
        repo_id=args.eot_repo_id,
        model_file=args.eot_model_file,
        tokenizer_name=args.eot_tokenizer,
        threshold=args.eot_threshold,
        max_length=args.eot_max_length,
        allow_download=True,
    )
    prediction = detector.predict(args.eot_test_text)
    print(
        f"✅ EOT smoke test: p={prediction.probability:.6f}, "
        f"is_end={prediction.is_end}, text={args.eot_test_text}"
    )

    vad = WebRTCSpeechVAD(sample_rate=Config.SAMPLE_RATE, mode=Config.DUPLEX_WEBRTC_VAD_MODE)
    silence_probability = vad(np.zeros(vad.window_size_samples, dtype=np.float32))
    print(f"✅ WebRTC VAD smoke test: silence_probability={silence_probability:.6f}")

    from livekit.rtc.apm import AudioProcessingModule

    _ = AudioProcessingModule(echo_cancellation=True)
    print("✅ LiveKit WebRTC APM smoke test: echo_cancellation module created")
    print("✅ duplex 环境准备完成")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="TalkRobot - 智能对话机器人")
    subparsers = parser.add_subparsers(dest="command")

    # 子命令: chat (默认)
    chat_parser = subparsers.add_parser("chat", help="启动对话机器人")
    chat_parser.add_argument(
        "--user", type=str, default=Config.DEFAULT_USER,
        help="用户名称，不同用户的记忆互相隔离 (默认: default)"
    )
    chat_parser.add_argument(
        "--no-tts", action="store_true", default=False,
        help="禁用TTS语音播放，仅显示文字回复"
    )
    chat_parser.add_argument(
        "--tts-provider", type=str, choices=["kokoro", "easy_tts_server"],
        default=Config.TTS_PROVIDER,
        help=f"TTS后端选择 (默认: {Config.TTS_PROVIDER})"
    )
    chat_parser.add_argument(
        "--language", type=str, choices=["zh", "en"],
        default=Config.LANGUAGE,
        help=f"统一语言开关：TTS与LLM同时生效 (默认: {Config.LANGUAGE})"
    )
    chat_parser.add_argument(
        "--listen-mode", type=str, choices=["push", "continuous", "intercom", "duplex"],
        default=Config.DEFAULT_LISTEN_MODE,
        help="监听模式: push=按住Q键说话, continuous=持续监听, intercom=对讲机PTT触发, duplex=AEC+EOT+语音打断 (默认: {})".format(Config.DEFAULT_LISTEN_MODE)
    )
    chat_parser.add_argument(
        "--no-asr", action="store_true", default=False,
        help="禁用ASR语音输入，改为终端键盘输入文本对话"
    )
    chat_parser.add_argument(
        "--debug", action="store_true", default=False,
        help="启用调试模式，输出详细日志，Ctrl+C时显示线程堆栈"
    )
    chat_parser.add_argument(
        "--history-rounds", type=int, default=Config.SLIDING_WINDOW_ROUNDS,
        help=f"滑动窗口对话轮数（仅包含最近历史，不含当前轮，0=关闭，默认: {Config.SLIDING_WINDOW_ROUNDS}）"
    )
    chat_parser.add_argument(
        "--streaming", action="store_true", default=False,
        help="启用流式回复生成（边生成边TTS播放）"
    )
    chat_parser.add_argument(
        "--enable-face", action="store_true", default=Config.FACE_ENABLED,
        help="启用人脸识别并根据识别人脸热切换交互对象"
    )
    chat_parser.add_argument(
        "--face-camera-index", type=int, default=Config.FACE_CAMERA_INDEX,
        help=f"人脸识别摄像头索引 (默认: {Config.FACE_CAMERA_INDEX})"
    )
    chat_parser.add_argument(
        "--face-poll-interval", type=float, default=Config.FACE_POLL_INTERVAL,
        help=f"人脸识别轮询间隔（秒，越小越高频，默认: {Config.FACE_POLL_INTERVAL}）"
    )
    chat_parser.add_argument(
        "--say-hallo", action="store_true", default=False,
        help="在 continuous 的非响应阶段，检测到熟人时主动问好"
    )
    chat_parser.add_argument(
        "--hallo-cooldown-seconds", type=float, default=600.0,
        help="主动问好冷却时间（秒），默认600秒=10分钟"
    )
    chat_parser.add_argument(
        "--disable-persona-auto-update", action="store_true", default=False,
        help="关闭后台人格自动更新（LangGraph Agent）"
    )
    chat_parser.add_argument(
        "--no-ros2-voice", action="store_false", dest="ros2_voice",
        default=Config.ROS2_VOICE_BRIDGE_ENABLED,
        help="关闭 ROS2 语音 bridge"
    )
    chat_parser.add_argument(
        "--ros2-asr-topic", type=str, default=Config.ROS2_ASR_TEXT_TOPIC,
        help=f"ASR 文本发布 topic (默认: {Config.ROS2_ASR_TEXT_TOPIC})"
    )
    chat_parser.add_argument(
        "--ros2-tts-topic", type=str, default=Config.ROS2_TTS_TEXT_TOPIC,
        help=f"TTS 文本订阅 topic (默认: {Config.ROS2_TTS_TEXT_TOPIC})"
    )
    chat_parser.add_argument(
        "--ros2-chat-topic", type=str, default=Config.ROS2_CHAT_TEXT_TOPIC,
        help=f"远程文本聊天订阅 topic (默认: {Config.ROS2_CHAT_TEXT_TOPIC})"
    )
    chat_parser.add_argument(
        "--ros2-assistant-topic", type=str, default=Config.ROS2_ASSISTANT_TEXT_TOPIC,
        help=f"机器人回复文本发布 topic (默认: {Config.ROS2_ASSISTANT_TEXT_TOPIC})"
    )
    chat_parser.add_argument(
        "--ros2-voice-node-name", type=str, default=Config.ROS2_VOICE_NODE_NAME,
        help=f"ROS2 语音 bridge 节点名 (默认: {Config.ROS2_VOICE_NODE_NAME})"
    )
    chat_parser.add_argument(
        "--ros2-voice-queue-size", type=int, default=Config.ROS2_VOICE_QUEUE_SIZE,
        help=f"ROS2 语音 bridge topic 队列大小 (默认: {Config.ROS2_VOICE_QUEUE_SIZE})"
    )

    # 子命令: add-memory
    mem_parser = subparsers.add_parser("add-memory", help="手动为指定用户或全部用户添加记忆")
    mem_parser.add_argument(
        "--user", type=str, default=Config.DEFAULT_USER,
        help=f"用户名称 (默认: {Config.DEFAULT_USER})"
    )
    mem_parser.add_argument(
        "--all-users", action="store_true", default=False,
        help=f"为全部用户添加记忆（包含未知用户: {Config.FACE_UNKNOWN_USER}）"
    )
    mem_parser.add_argument(
        "--content", type=str, default=None,
        help="要添加的记忆内容 (不提供则进入交互式添加模式)"
    )

    # 子命令: add-base-memory
    base_mem_parser = subparsers.add_parser("add-base-memory", help="手动添加基础共享记忆（所有用户可检索）")
    base_mem_parser.add_argument(
        "--content", type=str, default=None,
        help="要添加到基础记忆的内容 (不提供则进入交互式添加模式)"
    )

    # 子命令: query-memory
    query_mem_parser = subparsers.add_parser("query-memory", help="手动测试记忆检索（只读，不写入）")
    query_mem_parser.add_argument(
        "--user", type=str, default=Config.DEFAULT_USER,
        help=f"要检索的用户记忆 (默认: {Config.DEFAULT_USER})"
    )
    query_mem_parser.add_argument(
        "--query", type=str, default=None,
        help="检索查询文本 (不提供则进入交互式检索模式)"
    )
    query_mem_parser.add_argument(
        "--limit", type=int, default=Config.MEMORY_SEARCH_LIMIT,
        help=f"每个记忆库返回条数上限 (默认: {Config.MEMORY_SEARCH_LIMIT})"
    )
    query_mem_parser.add_argument(
        "--include-base", action="store_true", default=False,
        help=f"并行检索基础共享记忆 [{Config.BASE_MEMORY_USER}]"
    )

    # 子命令: vad-debug
    vad_parser = subparsers.add_parser("vad-debug", help="仅启动 VAD 监听调试（实时输出每个chunk分数）")
    vad_parser.add_argument("--sample-rate", type=int, default=Config.SAMPLE_RATE, help=f"采样率 (默认: {Config.SAMPLE_RATE})")
    vad_parser.add_argument("--channels", type=int, default=Config.CHANNELS, help=f"通道数 (默认: {Config.CHANNELS})")
    vad_parser.add_argument("--vad-check-interval", type=float, default=Config.VAD_CHECK_INTERVAL, help=f"VAD 检测间隔秒 (默认: {Config.VAD_CHECK_INTERVAL})")
    vad_parser.add_argument("--vad-chunk-size", type=float, default=Config.VAD_CHUNK_SIZE, help=f"单次 VAD 检测窗口秒 (默认: {Config.VAD_CHUNK_SIZE})")
    vad_parser.add_argument("--vad-pre-speech-duration", type=float, default=Config.VAD_PRE_SPEECH_DURATION, help=f"前置补偿秒 (默认: {Config.VAD_PRE_SPEECH_DURATION})")
    vad_parser.add_argument("--vad-speech-threshold", type=float, default=Config.VAD_SPEECH_THRESHOLD, help=f"VAD 语音阈值 (默认: {Config.VAD_SPEECH_THRESHOLD})")
    vad_parser.add_argument("--vad-silence-duration", type=float, default=Config.VAD_SILENCE_DURATION, help=f"静默结束阈值秒 (默认: {Config.VAD_SILENCE_DURATION})")
    vad_parser.add_argument("--vad-min-speech-duration", type=float, default=Config.VAD_MIN_SPEECH_DURATION, help=f"最短语音秒 (默认: {Config.VAD_MIN_SPEECH_DURATION})")

    setup_duplex_parser = subparsers.add_parser("setup-duplex", help="下载并检查 duplex 模式所需 EOT/VAD/AEC 资源")
    setup_duplex_parser.add_argument("--eot-repo-id", default=Config.EOT_REPO_ID)
    setup_duplex_parser.add_argument("--eot-model-file", default=Config.EOT_MODEL_FILE)
    setup_duplex_parser.add_argument("--eot-tokenizer", default=Config.EOT_TOKENIZER)
    setup_duplex_parser.add_argument("--eot-threshold", type=float, default=Config.EOT_THRESHOLD)
    setup_duplex_parser.add_argument("--eot-max-length", type=int, default=Config.EOT_MAX_LENGTH)
    setup_duplex_parser.add_argument("--eot-test-text", default="好的，那我们明天再聊")

    # 兼容旧版: python -m talkrobot.main --user xxx
    parser.add_argument(
        "--user", type=str, default=None,
        help="(兼容旧版) 用户名称，等同于 chat --user"
    )
    parser.add_argument(
        "--no-tts", action="store_true", default=False,
        help="(兼容旧版) 禁用TTS语音播放"
    )
    parser.add_argument(
        "--tts-provider", type=str, choices=["kokoro", "easy_tts_server"],
        default=None,
        help="(兼容旧版) TTS后端选择"
    )
    parser.add_argument(
        "--language", type=str, choices=["zh", "en"],
        default=None,
        help="(兼容旧版) 统一语言开关：TTS与LLM同时生效"
    )
    parser.add_argument(
        "--listen-mode", type=str, choices=["push", "continuous", "intercom", "duplex"],
        default=None,
        help="(兼容旧版) 监听模式: push=按住Q键, continuous=持续监听, intercom=对讲机PTT触发, duplex=AEC+EOT+语音打断"
    )
    parser.add_argument(
        "--no-asr", action="store_true", default=False,
        help="(兼容旧版) 禁用ASR语音输入，改为终端键盘输入"
    )
    parser.add_argument(
        "--debug", action="store_true", default=False,
        help="(兼容旧版) 启用调试模式"
    )
    parser.add_argument(
        "--history-rounds", type=int, default=None,
        help="(兼容旧版) 滑动窗口对话轮数（0=关闭）"
    )
    parser.add_argument(
        "--streaming", action="store_true", default=False,
        help="(兼容旧版) 启用流式回复生成"
    )
    parser.add_argument(
        "--enable-face", action="store_true", default=False,
        help="(兼容旧版) 启用人脸识别并根据识别人脸热切换交互对象"
    )
    parser.add_argument(
        "--face-camera-index", type=int, default=None,
        help="(兼容旧版) 人脸识别摄像头索引"
    )
    parser.add_argument(
        "--face-poll-interval", type=float, default=None,
        help="(兼容旧版) 人脸识别轮询间隔（秒）"
    )
    parser.add_argument(
        "--say-hallo", action="store_true", default=False,
        help="(兼容旧版) 在 continuous 的非响应阶段，检测到熟人时主动问好"
    )
    parser.add_argument(
        "--disable-persona-auto-update", action="store_true", default=False,
        help="(兼容旧版) 关闭后台人格自动更新（LangGraph Agent）"
    )
    parser.add_argument(
        "--no-ros2-voice", action="store_false", dest="ros2_voice",
        default=Config.ROS2_VOICE_BRIDGE_ENABLED,
        help="(兼容旧版) 关闭 ROS2 语音 bridge"
    )
    parser.add_argument(
        "--ros2-asr-topic", type=str, default=Config.ROS2_ASR_TEXT_TOPIC,
        help="(兼容旧版) ASR 文本发布 topic"
    )
    parser.add_argument(
        "--ros2-tts-topic", type=str, default=Config.ROS2_TTS_TEXT_TOPIC,
        help="(兼容旧版) TTS 文本订阅 topic"
    )
    parser.add_argument(
        "--ros2-chat-topic", type=str, default=Config.ROS2_CHAT_TEXT_TOPIC,
        help="(兼容旧版) 远程文本聊天订阅 topic"
    )
    parser.add_argument(
        "--ros2-assistant-topic", type=str, default=Config.ROS2_ASSISTANT_TEXT_TOPIC,
        help="(兼容旧版) 机器人回复文本发布 topic"
    )
    parser.add_argument(
        "--ros2-voice-node-name", type=str, default=Config.ROS2_VOICE_NODE_NAME,
        help="(兼容旧版) ROS2 语音 bridge 节点名"
    )
    parser.add_argument(
        "--ros2-voice-queue-size", type=int, default=Config.ROS2_VOICE_QUEUE_SIZE,
        help="(兼容旧版) ROS2 语音 bridge topic 队列大小"
    )

    args = parser.parse_args()
    
    # 设置全局DEBUG标志
    Config.DEBUG = getattr(args, 'debug', False)

    if args.command == "add-memory":
        run_add_memory(args)
    elif args.command == "add-base-memory":
        run_add_base_memory(args)
    elif args.command == "query-memory":
        run_query_memory(args)
    elif args.command == "vad-debug":
        run_vad_debug(args)
    elif args.command == "setup-duplex":
        run_setup_duplex(args)
    elif args.command == "chat":
        run_chat(args)
    else:
        # 兼容旧版调用方式: python -m talkrobot.main --user ljc
        if args.user is None:
            args.user = Config.DEFAULT_USER
        if not hasattr(args, 'listen_mode') or args.listen_mode is None:
            args.listen_mode = Config.DEFAULT_LISTEN_MODE
        if not hasattr(args, 'face_camera_index') or args.face_camera_index is None:
            args.face_camera_index = Config.FACE_CAMERA_INDEX
        if not hasattr(args, 'face_poll_interval') or args.face_poll_interval is None:
            args.face_poll_interval = Config.FACE_POLL_INTERVAL
        run_chat(args)


if __name__ == "__main__":
    main()
