"""
对话管理器
协调各个模块完成完整的对话流程
"""
import numpy as np
import time
import re
import os
import json
import copy
from loguru import logger
from typing import Optional, Callable, Tuple
import threading
from pynput import keyboard

class ConversationManager:
    """对话管理器"""
    
    def __init__(self, asr_module, tts_module, llm_module, memory_module,
                 tts_enabled: bool = True, expression_module=None,
                 streaming: bool = False,
                 audio_recorder=None,
                 audio_min_duration: float = 0.5,
                 audio_min_rms: float = 0.005,
                 sample_rate: int = 16000,
                 debug_timing: bool = False,
                 history_rounds: int = 3,
                 default_user: str = "default",
                 user_resolver: Optional[Callable[[], str]] = None,
                 memory_provider: Optional[Callable[[str], Tuple[Optional[object], bool]]] = None,
                 persona_provider: Optional[Callable[[str], str]] = None,
                 persona_update_handler: Optional[Callable[[str, str, str], None]] = None,
                 language: str = "zh",
                 say_hallo: bool = False,
                 greeting_cooldown_seconds: float = 600.0):
        """
        初始化对话管理器
        
        Args:
            asr_module: ASR模块实例
            tts_module: TTS模块实例
            llm_module: LLM模块实例
            memory_module: Memory模块实例
            tts_enabled: 是否启用TTS语音播放（默认True）
            expression_module: 表情模块实例（可选）
            streaming: 是否启用流式回复生成
            audio_recorder: 音频录制器实例（可选，用于 continuous 模式通知状态）
            audio_min_duration: 最短音频时长（秒），低于此值不送 ASR
            audio_min_rms: 最低音量 (RMS)，低于此值视为静音
            sample_rate: 采样率（用于计算时长）
            debug_timing: 是否输出各环节耗时（仅调试）
            history_rounds: 滑动窗口历史轮数（0 表示关闭）
            default_user: 默认交互用户
            user_resolver: 当前交互用户解析器，返回用户名
            memory_provider: 根据用户名返回 (memory_module, 是否启用长期记忆)
            persona_provider: 根据用户名返回人格 system prompt
            persona_update_handler: 后台人格更新处理器，参数为 (user, user_text, context)
            say_hallo: 是否在非响应阶段见到熟人后主动问好
        """
        self.asr = asr_module
        self.tts = tts_module
        self.llm = llm_module
        self.memory = memory_module
        self.tts_enabled = tts_enabled
        self.streaming = streaming
        self.expression = expression_module
        self.audio_recorder = audio_recorder
        self.audio_min_duration = audio_min_duration
        self.audio_min_rms = audio_min_rms
        self.sample_rate = sample_rate
        self.debug_timing = debug_timing
        self.history_rounds = max(0, int(history_rounds))
        self.default_user = default_user
        self._user_resolver = user_resolver or (lambda: self.default_user)
        self._memory_provider = memory_provider
        self._persona_provider = persona_provider or (lambda _: "")
        self._persona_update_handler = persona_update_handler
        self._worker_threads = []
        self._persona_update_threads = []
        self._worker_lock = threading.Lock()
        self._persona_update_lock = threading.Lock()
        self._user_state_lock = threading.Lock()
        self._proactive_lock = threading.Lock()
        self._history_lock = threading.Lock()
        self._sleep_lock = threading.Lock()
        self._script_lock = threading.Lock()
        self._script_image_lock = threading.Lock()
        self._recent_dialogue_rounds_by_user = {}  # dict[user, list[(user_text, assistant_text)]]
        self._active_user = self.default_user
        self._active_user_has_long_term_memory = True
        self._active_user_initialized = False
        self._pending_user_switch_notice = ""
        self._is_continuous_mode = bool(audio_recorder and audio_recorder.listen_mode == "continuous")
        self._response_enabled = not self._is_continuous_mode
        self._sleep_mode = False
        self._sleep_listener = None
        self._script_mode = False
        self._script_thread = None
        self._script_stop_event = threading.Event()
        self._script_context_snapshot = None
        self._script_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "script")
        )
        self._script_image_window = "ScriptImage"
        self._script_image_open = False
        self._script_image_thread = None
        self._script_image_stop_event = threading.Event()
        self._script_image_update_event = threading.Event()
        self._script_image_current_path = None
        self.language = (language or "zh").strip().lower()
        if self.language not in {"zh", "en"}:
            self.language = "zh"
        self._wake_words = ["hello", "hi", "hey"] if self._is_english else ["你好"]
        self._sleep_words = ["goodbye", "bye", "seeyou"] if self._is_english else ["再见"]
        self._say_hallo = bool(say_hallo)
        self._greeting_response_window_seconds = 10.0
        self._pending_greeting_deadline: Optional[float] = None
        self._greeting_cooldown_seconds = float(greeting_cooldown_seconds)
        self._last_greet_ts_by_user = {}
        self._script_triggers = ["介绍一下实验室"] if not self._is_english else ["introduce the lab"]

        if self._memory_provider is None:
            self._memory_provider = lambda _: (memory_module, True)
        self._switch_user_if_needed(self.default_user, silent=True)
        self._start_sleep_listener()
        
        expr_status = '开启' if (expression_module and expression_module.is_available) else '关闭'
        logger.info(
            f"对话管理器初始化完成 (TTS: {'开启' if tts_enabled else '关闭'}, "
            f"流式: {'开启' if streaming else '关闭'}, 表情: {expr_status}, 滑动窗口轮数: {self.history_rounds})"
        )
        if self._is_continuous_mode:
            if self._is_english:
                logger.info("continuous mode wake/sleep commands enabled: say 'hello' to wake, 'goodbye' to sleep")
            else:
                logger.info("continuous 模式已启用响应开关：说“你好”进入响应，说“再见”退出响应")

    @property
    def _is_english(self) -> bool:
        return self.language == "en"

    def _msg(self, zh: str, en: str) -> str:
        return en if self._is_english else zh

    def _resolve_active_user(self) -> str:
        """解析当前交互用户。"""
        try:
            user = self._user_resolver()
        except Exception as e:
            logger.warning(f"用户解析失败，回退默认用户: {e}")
            user = self.default_user

        if not user or not str(user).strip():
            return self.default_user
        return str(user).strip()

    def _start_sleep_listener(self) -> None:
        """启动睡眠/脚本模式切换的键盘监听器（W/C）。"""
        if self._sleep_listener is not None:
            return

        def _on_press_sleep(key):
            try:
                k = key.char if hasattr(key, 'char') else None
                if not k:
                    return
                k = k.lower()
                if k == 'w':
                    self._toggle_sleep_mode()
                elif k == 'c':
                    self._toggle_script_mode()
            except Exception as e:
                logger.debug(f"睡眠模式按键监听异常: {e}")

        self._sleep_listener = keyboard.Listener(on_press=_on_press_sleep, daemon=True)
        self._sleep_listener.start()
        logger.info("睡眠模式切换监听已启动 (按 W 切换)")

    def _toggle_sleep_mode(self) -> None:
        """切换睡眠模式。"""
        with self._sleep_lock:
            self._sleep_mode = not self._sleep_mode
            is_sleeping = self._sleep_mode

        if is_sleeping:
            if self._script_mode:
                self._exit_script_mode()
            if self.expression and self.expression.is_available:
                self.expression.set_expression("sleep")
            logger.info("进入睡眠模式，暂停外部输入响应")
            print(self._msg("😴 已进入睡眠模式（按 W 退出）", "😴 Sleep mode enabled (press W to wake)"))
            self._play_tts_notice(self._msg("我先去睡会。", "I'm going to sleep for a bit."))
        else:
            self._reset_context_on_wake()
            logger.info("退出睡眠模式，已重置上下文")
            print(self._msg("✅ 已退出睡眠模式（上下文已重置）", "✅ Sleep mode disabled (context reset)"))
            self._play_tts_notice(self._msg("我醒了。", "I'm awake."))
            if self.expression and self.expression.is_available:
                self.expression.reset_expression()

    def _toggle_script_mode(self) -> None:
        """切换脚本模式。"""
        if self._sleep_mode:
            print(self._msg("😴 当前为睡眠模式，无法进入脚本模式", "😴 Sleep mode active, script mode disabled"))
            return

        if self._script_mode and self._script_thread and not self._script_thread.is_alive():
            logger.warning("检测到脚本线程异常退出，自动恢复脚本状态")
            self._exit_script_mode()

        if not self._script_mode:
            self._enter_script_mode()
        else:
            self._exit_script_mode()

    def _enter_script_mode(self) -> None:
        with self._script_lock:
            if self._script_mode:
                return

            self._script_context_snapshot = {
                "recent_dialogue_rounds_by_user": copy.deepcopy(self._recent_dialogue_rounds_by_user),
                "pending_user_switch_notice": self._pending_user_switch_notice,
                "pending_greeting_deadline": self._pending_greeting_deadline,
                "last_greet_ts_by_user": copy.deepcopy(self._last_greet_ts_by_user),
                "response_enabled": self._response_enabled,
                "active_user": self._active_user,
                "active_user_has_long_term_memory": self._active_user_has_long_term_memory,
                "active_user_initialized": self._active_user_initialized,
            }

            self._script_mode = True
            self._script_stop_event.clear()
            self._script_thread = threading.Thread(target=self._run_script_mode, daemon=True)
            self._script_thread.start()

        logger.info("进入脚本模式")
        print(self._msg("🎬 已进入脚本模式（按 C 退出）", "🎬 Script mode enabled (press C to exit)"))

    def _exit_script_mode(self) -> None:
        with self._script_lock:
            if not self._script_mode:
                return
            self._script_mode = False
            self._script_stop_event.set()
            snapshot = self._script_context_snapshot
            self._script_context_snapshot = None

        if snapshot:
            with self._history_lock:
                self._recent_dialogue_rounds_by_user = snapshot.get("recent_dialogue_rounds_by_user", {})
            self._pending_user_switch_notice = snapshot.get("pending_user_switch_notice", "")
            self._pending_greeting_deadline = snapshot.get("pending_greeting_deadline")
            self._last_greet_ts_by_user = snapshot.get("last_greet_ts_by_user", {})
            self._response_enabled = snapshot.get("response_enabled", self._response_enabled)
            self._active_user = snapshot.get("active_user", self._active_user)
            self._active_user_has_long_term_memory = snapshot.get(
                "active_user_has_long_term_memory", self._active_user_has_long_term_memory
            )
            self._active_user_initialized = snapshot.get(
                "active_user_initialized", self._active_user_initialized
            )

        self._close_script_image_window()
        logger.info("退出脚本模式，恢复交互上下文")
        print(self._msg("✅ 已退出脚本模式", "✅ Script mode disabled"))

    def _resolve_script_path(self) -> Optional[str]:
        if not os.path.isdir(self._script_dir):
            return None

        script_candidates = []
        for name in os.listdir(self._script_dir):
            if name.lower().endswith(".json"):
                script_candidates.append(os.path.join(self._script_dir, name))

        script_candidates.sort()
        return script_candidates[0] if script_candidates else None

    def _load_script_steps(self) -> list:
        script_path = self._resolve_script_path()
        if not script_path:
            logger.warning("脚本模式未找到脚本文件 (.json)")
            return []

        try:
            with open(script_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            logger.warning(f"脚本文件读取失败: {e}")
            return []

        steps = payload.get("steps", []) if isinstance(payload, dict) else payload
        if not isinstance(steps, list):
            logger.warning("脚本格式无效：steps 必须是列表")
            return []
        return steps

    def _run_script_mode(self) -> None:
        try:
            steps = self._load_script_steps()
            if not steps:
                self._exit_script_mode()
                return

            for step in steps:
                if self._script_stop_event.is_set() or not self._script_mode:
                    break

                if not isinstance(step, dict):
                    continue

                text = str(step.get("text", "") or "").strip()
                expression = str(step.get("expression", "") or "").strip()
                image = str(step.get("image", "") or "").strip()
                delay = step.get("delay", 0.0)

                if expression and self.expression and self.expression.is_available:
                    self.expression.set_expression(expression)

                if image:
                    self._show_script_image_async(image)

                if text:
                    print(f"🤖 {self._msg('脚本', 'Script')}: {text}")
                    self._play_tts_notice(text)

                try:
                    delay_seconds = float(delay)
                except Exception:
                    delay_seconds = 0.0

                if delay_seconds > 0:
                    end_ts = time.time() + delay_seconds
                    while time.time() < end_ts:
                        if self._script_stop_event.is_set() or not self._script_mode:
                            break
                        time.sleep(0.05)
        except Exception as e:
            logger.warning(f"脚本模式运行异常: {e}")
        finally:
            if self.expression and self.expression.is_available:
                self.expression.reset_expression()
            self._close_script_image_window()

            if self._script_mode:
                self._exit_script_mode()

    def _resolve_script_image_path(self, image_path: str) -> Optional[str]:
        if not image_path:
            return None

        resolved_path = image_path
        if not os.path.isabs(resolved_path):
            resolved_path = os.path.join(self._script_dir, image_path)

        if not os.path.isfile(resolved_path):
            logger.warning(f"脚本图片不存在: {resolved_path}")
            return None
        return resolved_path

    def _script_image_loop(self) -> None:
        try:
            import cv2
        except Exception as e:
            logger.warning(f"脚本图片显示失败，缺少 OpenCV: {e}")
            return

        cv2.startWindowThread()

        while not self._script_image_stop_event.is_set():
            self._script_image_update_event.wait(0.1)
            self._script_image_update_event.clear()

            with self._script_image_lock:
                resolved_path = self._script_image_current_path

            if not resolved_path:
                if self._script_image_open:
                    try:
                        cv2.destroyWindow(self._script_image_window)
                    except Exception as e:
                        logger.debug(f"关闭脚本图片窗口失败: {e}")
                    finally:
                        self._script_image_open = False
                time.sleep(0.05)
                continue

            try:
                image = cv2.imread(resolved_path)
                if image is None:
                    logger.warning(f"脚本图片读取失败: {resolved_path}")
                    continue
                cv2.namedWindow(self._script_image_window, cv2.WINDOW_NORMAL)
                cv2.setWindowProperty(
                    self._script_image_window,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN,
                )
                cv2.imshow(self._script_image_window, image)
                self._script_image_open = True
                cv2.waitKey(1)
            except Exception as e:
                logger.warning(f"脚本图片显示异常: {e}")

        if self._script_image_open:
            try:
                cv2.destroyWindow(self._script_image_window)
            except Exception as e:
                logger.debug(f"关闭脚本图片窗口失败: {e}")
            finally:
                self._script_image_open = False

    def _show_script_image_async(self, image_path: str) -> None:
        resolved_path = self._resolve_script_image_path(image_path)
        if not resolved_path:
            return

        with self._script_image_lock:
            self._script_image_current_path = resolved_path
        self._script_image_update_event.set()

        if self._script_image_thread is None or not self._script_image_thread.is_alive():
            self._script_image_stop_event.clear()
            self._script_image_thread = threading.Thread(target=self._script_image_loop, daemon=True)
            self._script_image_thread.start()

    def _close_script_image_window(self) -> None:
        with self._script_image_lock:
            self._script_image_current_path = None
        self._script_image_update_event.set()

    def _play_tts_notice(self, text: str) -> None:
        if not text or not self.tts_enabled or self.tts is None:
            return
        if self.audio_recorder:
            self.audio_recorder.is_tts_playing = True
        try:
            self.tts.synthesize(text, play_audio=True)
        finally:
            if self.audio_recorder:
                self.audio_recorder.is_tts_playing = False

    def _reset_context_on_wake(self) -> None:
        """从睡眠模式唤醒时重置对话上下文。"""
        with self._history_lock:
            self._recent_dialogue_rounds_by_user.clear()
        self._pending_user_switch_notice = ""
        self._pending_greeting_deadline = None
        self._last_greet_ts_by_user = {}
        self._response_enabled = not self._is_continuous_mode
        logger.info("已重置滑动窗口、问好窗口与响应模式")

    def _switch_user_if_needed(self, user: str, silent: bool = False) -> None:
        """当用户变化时切换记忆模块与会话上下文。"""
        with self._user_state_lock:
            if not user:
                user = self.default_user

            if user == self._active_user and self._active_user_initialized:
                return

            previous_user = self._active_user if self._active_user_initialized else None

            if previous_user is not None and previous_user != user:
                with self._history_lock:
                    self._recent_dialogue_rounds_by_user.clear()
                self._pending_user_switch_notice = (
                    f"会话对象已切换：上一位交互对象是[{previous_user}]，当前对象是[{user}]。"
                    "请不要把上一位对象的对话内容当作当前对象的个人信息。"
                )

            memory_module, has_long_term_memory = self._memory_provider(user)
            self.memory = memory_module
            self._active_user = user
            self._active_user_has_long_term_memory = bool(has_long_term_memory and memory_module is not None)
            self._active_user_initialized = True

        if not silent:
            print(f"🧑 当前交互对象: {self._active_user}")
            if not self._active_user_has_long_term_memory:
                print("📝 当前对象无长期记忆，已切换为仅滑动窗口短期记忆模式")

        logger.info(
            f"已切换交互对象: user={self._active_user}, "
            f"long_term_memory={'开启' if self._active_user_has_long_term_memory else '关闭'}"
        )

    def _resolve_persona_prompt(self, user: str) -> str:
        """获取当前用户人格提示词。"""
        try:
            prompt = self._persona_provider(user)
        except Exception as e:
            logger.warning(f"人格提示词解析失败，回退模块默认 system prompt: {e}")
            return ""

        if not prompt or not str(prompt).strip():
            return ""
        return str(prompt).strip()

    def _start_persona_update_async(self, user: str, user_text: str, context: str) -> float:
        """异步触发人格更新 agent，不阻塞主回复链路。"""
        if self._persona_update_handler is None:
            return 0.0

        schedule_ts = time.perf_counter()

        def _runner():
            run_start = time.perf_counter()
            queue_delay = run_start - schedule_ts
            if self.debug_timing:
                msg = f"⏱️ 人格更新线程排队耗时: {queue_delay:.3f}s"
                print(msg)
                logger.debug(msg)

            try:
                self._persona_update_handler(user, user_text, context)
            except Exception as e:
                logger.warning(f"后台人格更新线程异常: {e}")
            finally:
                if self.debug_timing:
                    run_elapsed = time.perf_counter() - run_start
                    msg = f"⏱️ 人格更新后台总耗时: {run_elapsed:.3f}s"
                    print(msg)
                    logger.debug(msg)

        with self._persona_update_lock:
            self._persona_update_threads = [t for t in self._persona_update_threads if t.is_alive()]
            thread = threading.Thread(target=_runner, daemon=True)
            self._persona_update_threads.append(thread)
        thread.start()
        return schedule_ts

    def switch_active_user(self, user: str) -> None:
        """供外部线程调用的用户切换入口（如人脸追踪线程）。"""
        self._switch_user_if_needed(user)

    def on_face_user_change(self, user: str, is_familiar: bool = False) -> None:
        """人脸追踪回调入口：切换对象，并在条件满足时主动问好。"""
        if self._sleep_mode or self._script_mode:
            logger.debug("睡眠模式中，忽略人脸用户变更")
            return
        self._switch_user_if_needed(user)

        if not self._say_hallo:
            return
        if not is_familiar:
            return
        if not self._is_continuous_mode:
            return
        if self._response_enabled:
            return

        self._say_hello_to_familiar_user()

    def _say_hello_to_familiar_user(self) -> None:
        """在非响应阶段，对熟人执行一次主动问好。"""
        with self._proactive_lock:
            if not self._is_continuous_mode or self._response_enabled:
                return

            now = time.time()
            last_greet_ts = self._last_greet_ts_by_user.get(self._active_user)
            if last_greet_ts is not None:
                elapsed = now - last_greet_ts
                if elapsed < self._greeting_cooldown_seconds:
                    remaining = self._greeting_cooldown_seconds - elapsed
                    logger.info(
                        f"say_hallo 命中冷却: user={self._active_user}, 剩余{remaining:.1f}s，跳过主动问好"
                    )
                    return

            if not self._active_user_has_long_term_memory or self.memory is None:
                logger.info("say_hallo 已启用，但当前熟人无长期记忆，跳过主动问好")
                return

            try:
                memory_context = self.memory.search_memory(
                    "What is this user's name and how should I address them?"
                    if self._is_english
                    else "这个用户叫什么名字？我应该如何称呼TA？"
                )
            except Exception as e:
                logger.warning(f"say_hallo 记忆检索失败: {e}")
                memory_context = ""

            if not memory_context:
                logger.info("say_hallo 未检索到可用姓名记忆，跳过主动问好")
                return

            greeting_instruction = (
                "You just met a familiar person. Use only the provided memory to decide how to address them. "
                "Say one short and natural greeting in English. "
                "If the name cannot be confirmed from memory, say exactly: 'Hello, welcome back.' "
                "Do not make up extra details and do not mention memory retrieval."
                if self._is_english else
                "你刚见到一位熟人。请仅基于提供的记忆判断对方称呼，"
                "用一句简短自然的中文主动问好。"
                "如果记忆里无法确认名字，直接说“你好，欢迎回来”。"
                "不要编造额外信息，不要提及你在检索记忆。"
            )
            persona_prompt = self._resolve_persona_prompt(self._active_user)

            try:
                raw_response = self.llm.generate_response(
                    greeting_instruction,
                    memory_context,
                    system_prompt_override=persona_prompt,
                ).strip()
            except Exception as e:
                logger.warning(f"say_hallo LLM 生成失败: {e}")
                return

            if not raw_response:
                return

            expression_name = None
            if self.expression and self.expression.is_available:
                from talkrobot.modules.expression.expression_module import ExpressionModule
                response, expression_name = ExpressionModule.parse_expression_from_response(raw_response)
            else:
                response = raw_response

            response = response.strip()
            if not response:
                return

            logger.info(f"say_hallo 主动问好: user={self._active_user}, response={response}")

            if expression_name and self.expression and self.expression.is_available:
                self.expression.set_expression(expression_name)

            print(f"🤖 {self._msg('主动问好', 'Greeting')}: {response}")

            self._last_greet_ts_by_user[self._active_user] = time.time()

            # 将主动问好写入短期滑动窗口，便于后续上下文连续
            self._append_dialogue_round(
                self._active_user,
                self._msg("（机器人主动问好）", "(Assistant proactive greeting)"),
                response,
            )

            # 开启“问好后应答窗口”：5秒内用户应答可直接进入响应模式
            self._pending_greeting_deadline = time.time() + self._greeting_response_window_seconds

            if self.tts_enabled and self.tts is not None:
                if self.audio_recorder:
                    self.audio_recorder.is_tts_playing = True
                try:
                    self.tts.synthesize(response, play_audio=True)
                finally:
                    if self.audio_recorder:
                        self.audio_recorder.is_tts_playing = False

            if expression_name and self.expression and self.expression.is_available:
                self.expression.reset_expression()

    def _consume_user_switch_notice(self) -> str:
        """取出并清空一次性用户切换提示。"""
        notice = self._pending_user_switch_notice
        self._pending_user_switch_notice = ""
        return notice

    def _build_sliding_window_context(self, user: str) -> str:
        """构建最近 n 轮对话窗口文本。"""
        if self.history_rounds <= 0:
            return ""

        with self._history_lock:
            rounds = list(self._recent_dialogue_rounds_by_user.get(user, []))

        if not rounds:
            return ""

        lines = [f"最近对话窗口（最近{len(rounds)}轮）:"]
        for idx, (user_text, assistant_text) in enumerate(rounds, start=1):
            lines.append(f"第{idx}轮 用户: {user_text}")
            lines.append(f"第{idx}轮 机器人: {assistant_text}")
        return "\n".join(lines)

    def _append_dialogue_round(self, user: str, user_text: str, assistant_text: str) -> None:
        """将一轮对话写入滑动窗口。"""
        if self.history_rounds <= 0:
            return

        with self._history_lock:
            rounds = self._recent_dialogue_rounds_by_user.setdefault(user, [])
            rounds.append((user_text, assistant_text))
            if len(rounds) > self.history_rounds:
                self._recent_dialogue_rounds_by_user[user] = rounds[-self.history_rounds:]

    def _clear_sliding_window(self, user: Optional[str] = None) -> None:
        """清空指定用户（默认当前用户）的滑动窗口短期记忆。"""
        if self.history_rounds <= 0:
            return

        target_user = user or self._active_user
        with self._history_lock:
            if target_user in self._recent_dialogue_rounds_by_user:
                self._recent_dialogue_rounds_by_user[target_user] = []
                logger.info(f"已清空滑动窗口短期记忆: user={target_user}")

    @staticmethod
    def _merge_context(memory_context, sliding_window_context: str) -> str:
        """合并检索记忆与滑动窗口上下文。"""
        sections = []
        if memory_context:
            sections.append(f"检索到的相关记忆:\n{memory_context}")
        if sliding_window_context:
            sections.append(sliding_window_context)
        return "\n\n".join(sections)

    def _normalize_text(self, text: str) -> str:
        """标准化识别文本，用于关键词匹配。"""
        if not text:
            return ""
        cleaned = re.sub(r"<\|[^>]*\|>", "", text)
        cleaned = cleaned.casefold()
        cleaned = re.sub(r"\s+", "", cleaned)
        cleaned = re.sub(r"[，。！？、；：,.!?;:\"'‘’“”\[\]()（）]", "", cleaned)
        return cleaned

    def _maybe_trigger_script_mode(self, user_text: str) -> bool:
        """监听特定ASR内容触发脚本模式。返回是否已触发。"""
        if self._sleep_mode or self._script_mode:
            return False

        normalized_text = self._normalize_text(user_text)
        for trigger in self._script_triggers:
            if trigger and trigger in normalized_text:
                logger.info(f"命中脚本触发词: {trigger}")
                self._enter_script_mode()
                return True
        return False

    def _print_stage_timing(self, stage: str, elapsed: float) -> None:
        """在 debug 模式下输出阶段耗时。"""
        if not self.debug_timing:
            return
        msg = f"⏱️ {stage} 用时: {elapsed:.3f}s"
        print(msg)
        logger.debug(msg)

    def _handle_continuous_mode_command(self, user_text: str) -> bool:
        """处理 continuous 模式下的唤醒/休眠指令。返回是否继续后续对话流程。"""
        if not self._is_continuous_mode:
            return True

        normalized_text = self._normalize_text(user_text)

        # 主动问好后的短窗口：5秒内用户若应答，直接进入响应模式
        if not self._response_enabled and self._pending_greeting_deadline is not None:
            if time.time() <= self._pending_greeting_deadline:
                self._response_enabled = True
                self._pending_greeting_deadline = None
                logger.info(self._msg("检测到问好后5秒内应答，自动进入响应模式", "Detected a response within the greeting window, entering response mode"))
                print(self._msg("✅ 已进入响应模式（已识别为对问好的应答）", "✅ Response mode enabled (detected as greeting reply)"))
            else:
                self._pending_greeting_deadline = None

        if not self._response_enabled:
            if any(word in normalized_text for word in self._wake_words):
                self._response_enabled = True
                self._pending_greeting_deadline = None
                logger.info(self._msg("检测到唤醒词“你好”，进入响应模式", "Wake word detected, entering response mode"))
                print(self._msg("✅ 已进入响应模式", "✅ Response mode enabled"))
            else:
                logger.debug("当前为非响应模式，忽略本次语音")
                print(self._msg("🤫 当前非响应模式（说“你好”可唤醒）", "🤫 Non-response mode (say 'hello' to wake)"))
                return False

        if any(word in normalized_text for word in self._sleep_words):
            # 新增：先主动告别
            farewell = self._msg("再见，下次再聊！", "Goodbye, talk to you next time!")
            print(f"🤖 {self._msg('机器人', 'Assistant')}: {farewell}")
            logger.info(self._msg("检测到“再见”，机器人主动告别", "Sleep word detected, assistant says goodbye"))
            if self.tts_enabled and self.tts is not None:
                if self.audio_recorder:
                    self.audio_recorder.is_tts_playing = True
                try:
                    self.tts.synthesize(farewell, play_audio=True)
                finally:
                    if self.audio_recorder:
                        self.audio_recorder.is_tts_playing = False

            self._clear_sliding_window(self._active_user)
            self._response_enabled = False
            self._pending_greeting_deadline = None
            logger.info(self._msg("检测到“再见”，退出响应模式", "Sleep word detected, exit response mode"))
            print(self._msg("🛑 已退出响应模式（后续语音将忽略，说“你好”可重新唤醒）", "🛑 Response mode disabled (later speech will be ignored, say 'hello' to wake again)"))
            return False

        return True

    def _process_user_text(self, user_text: str) -> None:
        """处理已获得的用户文本（来自 ASR 或终端输入）。"""
        logger.debug(f"处理用户文本: {user_text}")

        current_user = self._resolve_active_user()
        if current_user != self._active_user or not self._active_user_initialized:
            self._switch_user_if_needed(current_user)

        # 2. 异步存储用户输入到记忆 (不阻塞后续流程)
        if self._active_user_has_long_term_memory and self.memory is not None:
            logger.debug("开始存储用户输入到记忆")
            self.memory.add_memory(f"用户说: {user_text}", async_mode=True)
        else:
            logger.debug("当前对象无长期记忆，跳过用户输入写入")

        # 3. 从记忆中检索相关上下文
        logger.debug("开始检索记忆")
        memory_start = time.perf_counter()
        memory_context = ""
        if self._active_user_has_long_term_memory and self.memory is not None:
            print(self._msg("🔍 正在检索相关记忆...", "🔍 Retrieving relevant memory..."))
            memory_context = self.memory.search_memory(user_text)
        else:
            print(self._msg("🧠 当前仅使用短期记忆（滑动窗口）...", "🧠 Using short-term memory only (sliding window)..."))

        sliding_window_context = self._build_sliding_window_context(self._active_user)
        context = self._merge_context(memory_context, sliding_window_context)
        switch_notice = self._consume_user_switch_notice()
        if switch_notice:
            context = f"{switch_notice}\n\n{context}" if context else switch_notice
        memory_elapsed = time.perf_counter() - memory_start
        self._print_stage_timing("查询相关记忆", memory_elapsed)
        logger.debug(f"记忆检索完成，记忆上下文长度: {len(str(memory_context)) if memory_context else 0}")
        logger.debug(f"滑动窗口上下文长度: {len(sliding_window_context) if sliding_window_context else 0}")

        # 3.1 并行后台人格更新（不影响当前轮回复）
        persona_async_schedule_ts = self._start_persona_update_async(self._active_user, user_text, context)

        # 4. LLM: 生成回复
        print(self._msg("🤖 正在思考回复...", "🤖 Thinking..."))
        logger.debug("开始LLM生成回复")
        llm_start = time.perf_counter()
        if self.debug_timing and persona_async_schedule_ts > 0:
            overlap_start_delta = llm_start - persona_async_schedule_ts
            msg = f"⏱️ 人格更新与主回复并行启动差: {overlap_start_delta:.3f}s"
            print(msg)
            logger.debug(msg)
        logger.debug(f"💡 输入给LLM的上下文: {context}")
        persona_prompt = self._resolve_persona_prompt(self._active_user)
        tts_played_in_streaming = False
        expression_name = None
        expression_set_in_streaming = False
        if self.streaming:
            logger.debug("使用流式模式生成回复")
            raw_response_parts = []
            buffered_prefix = ""
            expression_prefix_finalized = False

            from talkrobot.modules.expression.expression_module import ExpressionModule

            def _handle_stream_chunk(chunk: str) -> str:
                """处理流式增量，剥离起始表情标签并返回可展示/可播报文本。"""
                nonlocal buffered_prefix, expression_prefix_finalized, expression_name, expression_set_in_streaming

                if expression_prefix_finalized:
                    return chunk

                buffered_prefix += chunk
                stripped = buffered_prefix.lstrip()

                # 1) 明确不是表情前缀，直接透传
                if not stripped.startswith("["):
                    expression_prefix_finalized = True
                    output = buffered_prefix
                    buffered_prefix = ""
                    return output

                # 2) 可能是 [expression:...] 前缀，等待足够字符
                expected_prefix = "[expression:"
                if expected_prefix.startswith(stripped) and "]" not in stripped:
                    return ""

                # 3) 不是合法 expression 前缀，直接透传
                if not stripped.startswith(expected_prefix):
                    expression_prefix_finalized = True
                    output = buffered_prefix
                    buffered_prefix = ""
                    return output

                # 4) 是 expression 前缀，等到 ] 再解析
                close_idx = stripped.find("]")
                if close_idx < 0:
                    return ""

                tag = stripped[:close_idx + 1]
                rest = stripped[close_idx + 1:]
                _, parsed_expression = ExpressionModule.parse_expression_from_response(tag)
                if parsed_expression:
                    expression_name = parsed_expression
                    logger.info(f"流式前缀检测到表情: {expression_name}")
                    if self.expression and self.expression.is_available:
                        self.expression.set_expression(expression_name)
                        expression_set_in_streaming = True

                expression_prefix_finalized = True
                buffered_prefix = ""
                return rest

            def _stream_with_capture():
                for chunk in self.llm.generate_response_stream(
                    user_text,
                    context,
                    system_prompt_override=persona_prompt,
                ):
                    raw_response_parts.append(chunk)
                    clean_chunk = _handle_stream_chunk(chunk)
                    if clean_chunk:
                        print(clean_chunk, end="", flush=True)
                        yield clean_chunk

                # 若直到结束都未最终确定前缀，则把缓存内容透传（兜底）
                if buffered_prefix and not expression_prefix_finalized:
                    print(buffered_prefix, end="", flush=True)
                    yield buffered_prefix

            print(self._msg("🤖 机器人: ", "🤖 Assistant: "), end="", flush=True)

            if self.tts_enabled:
                print(self._msg("\n🔊 正在播放语音... (按 'S' 键可打断)", "\n🔊 Playing audio... (press 'S' to interrupt)"))
                logger.debug("开始流式TTS播放")

                def _on_press_interrupt_streaming(key):
                    try:
                        if hasattr(key, 'char') and key.char == 's':
                            logger.info("检测到S键，触发TTS打断")
                            self.tts._interrupted.set()
                    except Exception as e:
                        logger.debug(f"回调异常: {e}")

                interrupt_listener = keyboard.Listener(on_press=_on_press_interrupt_streaming, daemon=True)
                interrupt_listener.start()
                logger.debug("流式TTS打断监听器已启动（daemon模式）")

                if self.audio_recorder:
                    logger.debug("设置 is_tts_playing = True")
                    self.audio_recorder.is_tts_playing = True

                tts_start = time.perf_counter()
                try:
                    self.tts.synthesize(_stream_with_capture(), play_audio=True)
                    tts_elapsed = time.perf_counter() - tts_start
                    self._print_stage_timing("TTS合成", tts_elapsed)
                    tts_played_in_streaming = True
                finally:
                    if self.audio_recorder:
                        logger.debug("设置 is_tts_playing = False")
                        self.audio_recorder.is_tts_playing = False
                    logger.debug("流式TTS播放流程已结束")
            else:
                for _ in _stream_with_capture():
                    pass

            print()
            raw_response = "".join(raw_response_parts)
        else:
            raw_response = self.llm.generate_response(
                user_text,
                context,
                system_prompt_override=persona_prompt,
            )

        llm_elapsed = time.perf_counter() - llm_start
        self._print_stage_timing("大模型生成回复", llm_elapsed)
        logger.debug(f"LLM回复生成完成: {raw_response[:50]}...")

        # 4.1 解析表情标签
        if self.expression and self.expression.is_available:
            from talkrobot.modules.expression.expression_module import ExpressionModule
            response, parsed_expression_name = ExpressionModule.parse_expression_from_response(raw_response)
            if not expression_name:
                expression_name = parsed_expression_name
        else:
            response = raw_response

        if not self.streaming:
            print(f"🤖 {self._msg('机器人', 'Assistant')}: {response}")
        if expression_name:
            logger.info(f"检测到表情: {expression_name}")

        # 4.2 切换表情
        if expression_name and self.expression and not expression_set_in_streaming:
            logger.debug(f"切换表情: {expression_name}")
            self.expression.set_expression(expression_name)

        # 5. 异步存储机器人回复到记忆 (不阻塞后续流程)
        if self._active_user_has_long_term_memory and self.memory is not None:
            logger.debug("开始存储机器人回复到记忆")
            self.memory.add_memory(f"机器人回复: {response}", async_mode=True)
        else:
            logger.debug("当前对象无长期记忆，跳过机器人回复写入")

        # 5.1 更新滑动窗口
        self._append_dialogue_round(self._active_user, user_text, response)

        # 6. TTS: 文字转语音并播放
        if self.tts_enabled and not tts_played_in_streaming:
            print(self._msg("🔊 正在播放语音... (按 'S' 键可打断)", "🔊 Playing audio... (press 'S' to interrupt)"))
            logger.debug("开始TTS播放")

            # 启动临时键盘监听器，按 S 键打断 TTS
            # 重要：不在回调里返回 False，避免监听器自阻塞；用 daemon=True 让主线程不等待
            def _on_press_interrupt(key):
                try:
                    if hasattr(key, 'char') and key.char == 's':
                        logger.info("检测到S键，触发TTS打断")
                        self.tts._interrupted.set()
                        # 不调用 sd.stop()，让 TTS 模块自己在轮询里检查到中断后调用
                except Exception as e:
                    logger.debug(f"回调异常: {e}")

            interrupt_listener = keyboard.Listener(on_press=_on_press_interrupt, daemon=True)
            interrupt_listener.start()
            logger.debug("TTS打断监听器已启动（daemon模式）")

            # 通知录音器 TTS 正在播放（continuous 模式下屏蔽麦克风）
            if self.audio_recorder:
                logger.debug("设置 is_tts_playing = True")
                self.audio_recorder.is_tts_playing = True
            tts_start = time.perf_counter()
            try:
                logger.debug("开始调用 tts.synthesize()")
                self.tts.synthesize(response, play_audio=True)
                tts_elapsed = time.perf_counter() - tts_start
                self._print_stage_timing("TTS合成", tts_elapsed)
                logger.debug("tts.synthesize() 返回")
            finally:
                # 先恢复录音器状态，daemon 监听器自动退出，无需显式 stop()
                if self.audio_recorder:
                    logger.debug("设置 is_tts_playing = False")
                    self.audio_recorder.is_tts_playing = False
                logger.debug("TTS播放流程已结束")

            if self.tts._interrupted.is_set():
                print(self._msg("⏹️ 语音播放已打断", "⏹️ Audio playback interrupted"))
                logger.debug("TTS播放被打断")
            else:
                logger.debug("TTS播放正常完成")
        else:
            logger.info("TTS已关闭，跳过语音播放")

        # 7. 回复结束后重置为默认表情
        if expression_name and self.expression:
            logger.debug("重置表情为默认")
            self.expression.reset_expression()

        print("\n" + "-"*50)
        if self.audio_recorder and self.audio_recorder.listen_mode == "continuous":
            print(self._msg("✅ 对话完成，请继续说话", "✅ Done. Please continue speaking"))
        elif self.audio_recorder:
            print(self._msg("✅ 对话完成，请继续按 'Q' 说话", "✅ Done. Hold 'Q' to speak again"))
        else:
            print(self._msg("✅ 对话完成，请继续输入", "✅ Done. Please continue typing"))
        print("-"*50 + "\n")
        logger.debug("对话流程完成")
    
    def process_audio(self, audio_data: np.ndarray) -> None:
        """
        处理录制的音频,完成完整的对话流程
        
        Args:
            audio_data: 音频数据
        """
        try:
            if self._sleep_mode or self._script_mode:
                logger.debug("睡眠模式中，忽略音频输入")
                return
            logger.debug("开始处理音频数据")
            # 0. 音频前置过滤：时长和音量检查
            duration = len(audio_data) / self.sample_rate
            rms = float(np.sqrt(np.mean(audio_data ** 2)))
            
            logger.debug(f"音频参数: 时长={duration:.2f}s, RMS={rms:.4f}")
            
            if duration < self.audio_min_duration:
                logger.info(f"音频过短 ({duration:.2f}s < {self.audio_min_duration}s)，已忽略")
                print(self._msg(f"\n⚠️ 音频过短 ({duration:.2f}秒)，已忽略", f"\n⚠️ Audio too short ({duration:.2f}s), ignored"))
                return
            
            if rms < self.audio_min_rms:
                logger.info(f"音量过低 (RMS={rms:.4f} < {self.audio_min_rms})，已忽略")
                print(self._msg("\n⚠️ 音量过低，已忽略", "\n⚠️ Volume too low, ignored"))
                return
            
            logger.debug(f"音频检查通过: 时长={duration:.2f}s, RMS={rms:.4f}")
            
            # 1. ASR: 语音转文字
            print(self._msg("\n🎯 正在识别...", "\n🎯 Recognizing..."))
            logger.debug("开始ASR识别")
            asr_start = time.perf_counter()
            user_text = self.asr.transcribe(audio_data)
            asr_elapsed = time.perf_counter() - asr_start
            self._print_stage_timing("ASR生成文本", asr_elapsed)
            logger.debug(f"ASR识别完成: {user_text}")
            
            if not user_text or user_text.strip() == "":
                print(self._msg("⚠️ 未识别到有效内容,请重试", "⚠️ No valid speech recognized, please try again"))
                logger.debug("ASR识别为空")
                return
            
            print(f"👤 {self._msg('您说', 'You said')}: {user_text}")
            if self._maybe_trigger_script_mode(user_text):
                return
            if not self._handle_continuous_mode_command(user_text):
                return

            self._process_user_text(user_text)
            
        except Exception as e:
            logger.error(f"对话处理出错: {e}", exc_info=True)
            print(f"❌ 处理出错: {e}")
        finally:
            # 通知录音器处理完成，可以采集下一段语音
            logger.debug("准备通知录音器处理完成")
            if self.audio_recorder:
                self.audio_recorder.notify_process_done()
            logger.debug("已通知录音器处理完成")

    def process_text(self, user_text: str) -> None:
        """处理终端输入文本,完成完整对话流程。"""
        try:
            if self._sleep_mode or self._script_mode:
                logger.debug("睡眠模式中，忽略文本输入")
                return
            if not user_text or user_text.strip() == "":
                print(self._msg("⚠️ 输入为空，请重试", "⚠️ Empty input, please try again"))
                return

            user_text = user_text.strip()
            print(f"👤 {self._msg('您输入', 'You typed')}: {user_text}")

            if self._maybe_trigger_script_mode(user_text):
                return

            if not self._handle_continuous_mode_command(user_text):
                return

            self._process_user_text(user_text)
        except Exception as e:
            logger.error(f"文本对话处理出错: {e}", exc_info=True)
            print(f"❌ 处理出错: {e}")
    
    def process_audio_async(self, audio_data: np.ndarray) -> None:
        """
        异步处理音频(避免阻塞录音)
        
        Args:
            audio_data: 音频数据
        """
        with self._worker_lock:
            self._worker_threads = [t for t in self._worker_threads if t.is_alive()]
            thread = threading.Thread(target=self.process_audio, args=(audio_data,))
            thread.daemon = False
            self._worker_threads.append(thread)
        thread.start()

    def shutdown(self, timeout: float = 5.0):
        """等待后台音频处理线程退出。"""
        logger.info("正在关闭对话管理器...")
        with self._worker_lock:
            threads = list(self._worker_threads)
            self._worker_threads = []

        with self._persona_update_lock:
            persona_threads = list(self._persona_update_threads)
            self._persona_update_threads = []

        for thread in threads:
            if thread.is_alive() and thread != threading.current_thread():
                thread.join(timeout=timeout)
        for thread in persona_threads:
            if thread.is_alive() and thread != threading.current_thread():
                thread.join(timeout=min(timeout, 1.0))
        if self._script_thread and self._script_thread.is_alive():
            self._script_stop_event.set()
            self._script_thread.join(timeout=min(timeout, 1.0))
        if self._script_image_thread and self._script_image_thread.is_alive():
            self._script_image_stop_event.set()
            self._script_image_update_event.set()
            self._script_image_thread.join(timeout=min(timeout, 1.0))
            self._script_image_thread = None
        self._close_script_image_window()
        if self._sleep_listener is not None:
            try:
                self._sleep_listener.stop()
            except Exception as e:
                logger.warning(f"停止睡眠模式监听异常: {e}")
            self._sleep_listener = None
        logger.info("对话管理器已关闭")