"""
对话管理器
协调各个模块完成完整的对话流程
"""
import numpy as np
import time
import re
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
        self._worker_threads = []
        self._worker_lock = threading.Lock()
        self._user_state_lock = threading.Lock()
        self._proactive_lock = threading.Lock()
        self._history_lock = threading.Lock()
        self._recent_dialogue_rounds_by_user = {}  # dict[user, list[(user_text, assistant_text)]]
        self._active_user = self.default_user
        self._active_user_has_long_term_memory = True
        self._active_user_initialized = False
        self._pending_user_switch_notice = ""
        self._is_continuous_mode = bool(audio_recorder and audio_recorder.listen_mode == "continuous")
        self._response_enabled = not self._is_continuous_mode
        self._wake_word = "你好"
        self._sleep_word = "再见"
        self._say_hallo = bool(say_hallo)
        self._greeting_response_window_seconds = 10.0
        self._pending_greeting_deadline: Optional[float] = None
        self._greeting_cooldown_seconds = float(greeting_cooldown_seconds)
        self._last_greet_ts_by_user = {}

        if self._memory_provider is None:
            self._memory_provider = lambda _: (memory_module, True)
        self._switch_user_if_needed(self.default_user, silent=True)
        
        expr_status = '开启' if (expression_module and expression_module.is_available) else '关闭'
        logger.info(
            f"对话管理器初始化完成 (TTS: {'开启' if tts_enabled else '关闭'}, "
            f"流式: {'开启' if streaming else '关闭'}, 表情: {expr_status}, 滑动窗口轮数: {self.history_rounds})"
        )
        if self._is_continuous_mode:
            logger.info("continuous 模式已启用响应开关：说“你好”进入响应，说“再见”退出响应")

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

    def switch_active_user(self, user: str) -> None:
        """供外部线程调用的用户切换入口（如人脸追踪线程）。"""
        self._switch_user_if_needed(user)

    def on_face_user_change(self, user: str, is_familiar: bool = False) -> None:
        """人脸追踪回调入口：切换对象，并在条件满足时主动问好。"""
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
                memory_context = self.memory.search_memory("这个用户叫什么名字？我应该如何称呼TA？")
            except Exception as e:
                logger.warning(f"say_hallo 记忆检索失败: {e}")
                memory_context = ""

            if not memory_context:
                logger.info("say_hallo 未检索到可用姓名记忆，跳过主动问好")
                return

            greeting_instruction = (
                "你刚见到一位熟人。请仅基于提供的记忆判断对方称呼，"
                "用一句简短自然的中文主动问好。"
                "如果记忆里无法确认名字，直接说“你好，欢迎回来”。"
                "不要编造额外信息，不要提及你在检索记忆。"
            )

            try:
                raw_response = self.llm.generate_response(greeting_instruction, memory_context).strip()
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

            print(f"🤖 主动问好: {response}")

            self._last_greet_ts_by_user[self._active_user] = time.time()

            # 将主动问好写入短期滑动窗口，便于后续上下文连续
            self._append_dialogue_round(self._active_user, "（机器人主动问好）", response)

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
        cleaned = re.sub(r"\s+", "", cleaned)
        cleaned = re.sub(r"[，。！？、；：,.!?;:\"'‘’“”\[\]()（）]", "", cleaned)
        return cleaned

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
                logger.info("检测到问好后5秒内应答，自动进入响应模式")
                print("✅ 已进入响应模式（已识别为对问好的应答）")
            else:
                self._pending_greeting_deadline = None

        if not self._response_enabled:
            if self._wake_word in normalized_text:
                self._response_enabled = True
                self._pending_greeting_deadline = None
                logger.info("检测到唤醒词“你好”，进入响应模式")
                print("✅ 已进入响应模式")
            else:
                logger.debug("当前为非响应模式，忽略本次语音")
                print("🤫 当前非响应模式（说“你好”可唤醒）")
                return False

        if self._sleep_word in normalized_text:
            # 新增：先主动告别
            farewell = "再见，下次再聊！"
            print(f"🤖 机器人: {farewell}")
            logger.info("检测到“再见”，机器人主动告别")
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
            logger.info("检测到“再见”，退出响应模式")
            print("🛑 已退出响应模式（后续语音将忽略，说“你好”可重新唤醒）")
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
            print("🔍 正在检索相关记忆...")
            memory_context = self.memory.search_memory(user_text)
        else:
            print("🧠 当前仅使用短期记忆（滑动窗口）...")

        sliding_window_context = self._build_sliding_window_context(self._active_user)
        context = self._merge_context(memory_context, sliding_window_context)
        switch_notice = self._consume_user_switch_notice()
        if switch_notice:
            context = f"{switch_notice}\n\n{context}" if context else switch_notice
        memory_elapsed = time.perf_counter() - memory_start
        self._print_stage_timing("查询相关记忆", memory_elapsed)
        logger.debug(f"记忆检索完成，记忆上下文长度: {len(str(memory_context)) if memory_context else 0}")
        logger.debug(f"滑动窗口上下文长度: {len(sliding_window_context) if sliding_window_context else 0}")

        # 4. LLM: 生成回复
        print("🤖 正在思考回复...")
        logger.debug("开始LLM生成回复")
        llm_start = time.perf_counter()
        logger.debug(f"💡 输入给LLM的上下文: {context}")
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
                for chunk in self.llm.generate_response_stream(user_text, context):
                    raw_response_parts.append(chunk)
                    clean_chunk = _handle_stream_chunk(chunk)
                    if clean_chunk:
                        print(clean_chunk, end="", flush=True)
                        yield clean_chunk

                # 若直到结束都未最终确定前缀，则把缓存内容透传（兜底）
                if buffered_prefix and not expression_prefix_finalized:
                    print(buffered_prefix, end="", flush=True)
                    yield buffered_prefix

            print("🤖 机器人: ", end="", flush=True)

            if self.tts_enabled:
                print("\n🔊 正在播放语音... (按 'S' 键可打断)")
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
            raw_response = self.llm.generate_response(user_text, context)

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
            print(f"🤖 机器人: {response}")
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
            print("🔊 正在播放语音... (按 'S' 键可打断)")
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
                print("⏹️ 语音播放已打断")
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
            print("✅ 对话完成，请继续说话")
        elif self.audio_recorder:
            print("✅ 对话完成，请继续按 'Q' 说话")
        else:
            print("✅ 对话完成，请继续输入")
        print("-"*50 + "\n")
        logger.debug("对话流程完成")
    
    def process_audio(self, audio_data: np.ndarray) -> None:
        """
        处理录制的音频,完成完整的对话流程
        
        Args:
            audio_data: 音频数据
        """
        try:
            logger.debug("开始处理音频数据")
            # 0. 音频前置过滤：时长和音量检查
            duration = len(audio_data) / self.sample_rate
            rms = float(np.sqrt(np.mean(audio_data ** 2)))
            
            logger.debug(f"音频参数: 时长={duration:.2f}s, RMS={rms:.4f}")
            
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
            logger.debug("开始ASR识别")
            asr_start = time.perf_counter()
            user_text = self.asr.transcribe(audio_data)
            asr_elapsed = time.perf_counter() - asr_start
            self._print_stage_timing("ASR生成文本", asr_elapsed)
            logger.debug(f"ASR识别完成: {user_text}")
            
            if not user_text or user_text.strip() == "":
                print("⚠️ 未识别到有效内容,请重试")
                logger.debug("ASR识别为空")
                return
            
            print(f"👤 您说: {user_text}")
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
            if not user_text or user_text.strip() == "":
                print("⚠️ 输入为空，请重试")
                return

            user_text = user_text.strip()
            print(f"👤 您输入: {user_text}")

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

        for thread in threads:
            if thread.is_alive() and thread != threading.current_thread():
                thread.join(timeout=timeout)
        logger.info("对话管理器已关闭")