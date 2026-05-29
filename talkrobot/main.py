"""
对话机器人主程序
整合所有模块,启动对话机器人
"""
import sys
import os
import time
import argparse
from loguru import logger

from talkrobot.config import Config
from talkrobot.core.app_logging import configure_logging
from talkrobot.core.persona_manager import PersonaManager
from talkrobot.core.expression_server_manager import ExpressionServerManager
from talkrobot.core.face_identity_resolver import FaceIdentityResolver
from talkrobot.core.memory_router import UserMemoryRouter

os.environ.setdefault('HF_HUB_OFFLINE', '1')


def run_chat(args):
    """启动对话机器人"""
    from talkrobot.modules.tts.tts_module import TTSModule
    from talkrobot.modules.llm.llm_module import LLMModule
    from talkrobot.modules.llm.persona_update_agent import PersonaUpdateAgent
    from talkrobot.core.conversation_manager import ConversationManager

    user = args.user
    memory_module = None
    memory_router = None
    persona_manager = None
    persona_update_agent = None
    face_resolver = None
    tts_module = None
    audio_recorder = None
    conversation_manager = None
    expression_server_manager = None

    try:
        configure_logging(debug=Config.DEBUG)

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
            provider=tts_provider,
            language=language,
            sample_rate=Config.TTS_SAMPLE_RATE,
        )

        # 初始化表情模块（可选）
        expression_module = None
        expression_prompt = ""
        if Config.EXPRESSION_ENABLED:
            from talkrobot.modules.expression.expression_module import ExpressionModule
            expression_server_manager = ExpressionServerManager(
                project_root=os.path.dirname(os.path.dirname(__file__))
            )
            expression_server_manager.start()
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
                sentiment_threshold=Config.PERSONA_SENTIMENT_THRESHOLD,
            )

        def _persona_provider(current_user: str) -> str:
            persona_prompt = persona_manager.get_prompt_for_user(current_user)
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

                current_prompt = persona_manager.get_prompt_for_user(current_user)
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
                    f"sentiment={float(result.get('sentiment_elapsed_s', 0.0) or 0.0):.3f}s, "
                    f"reflection={float(result.get('reflection_elapsed_s', 0.0) or 0.0):.3f}s, "
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
                if persona_manager.update_user_prompt(current_user, updated_prompt):
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
            from talkrobot.modules.memory.memory_module import MemoryModule

            memory_module = MemoryModule(
                config=Config.get_memory_config(user),
                user_id=Config.get_user_id(user)
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

        # 3. 创建音频录制器（no-asr 模式下跳过）
        if not no_asr_mode:
            from talkrobot.core.audio_recorder import AudioRecorder
            audio_recorder = AudioRecorder(
                sample_rate=Config.SAMPLE_RATE,
                channels=Config.CHANNELS,
                listen_mode=listen_mode,
                vad_check_interval=Config.VAD_CHECK_INTERVAL,
                pre_speech_duration=Config.VAD_PRE_SPEECH_DURATION,
                silence_duration=Config.VAD_SILENCE_DURATION,
                min_speech_duration=Config.VAD_MIN_SPEECH_DURATION,
            )
        else:
            audio_recorder = None

        conversation_manager = ConversationManager(
            asr_module=asr_module,
            tts_module=tts_module,
            llm_module=llm_module,
            memory_module=memory_module,
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
        )

        if face_resolver is not None and face_resolver.enabled:
            face_resolver.set_on_user_change(conversation_manager.on_face_user_change)
            conversation_manager.on_face_user_change(
                face_resolver.resolve_user(),
                face_resolver.is_current_user_familiar(),
            )

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

        if face_resolver is not None:
            try:
                face_resolver.shutdown()
            except Exception as e:
                logger.warning(f"关闭人脸识别异常: {e}")

        if expression_server_manager is not None:
            expression_server_manager.stop()


def run_add_memory(args):
    """手动添加记忆"""
    from talkrobot.modules.memory.memory_module import MemoryModule

    configure_logging(debug=Config.DEBUG)

    user = args.user
    content = args.content

    logger.info(f"正在为用户 [{user}] 添加记忆...")

    memory_module = MemoryModule(
        config=Config.get_memory_config(user),
        user_id=Config.get_user_id(user)
    )

    try:
        if content:
            # 直接通过命令行参数添加
            memory_module.add_memory(content, async_mode=False)
            print(f"✅ 已为用户 [{user}] 添加记忆: {content}")
        else:
            # 交互式添加模式
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
                memory_module.add_memory(text, async_mode=False)
                print(f"  ✅ 已添加: {text}\n")
    finally:
        memory_module.shutdown()
        print("👋 记忆模块已关闭")


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
        "--listen-mode", type=str, choices=["push", "continuous"],
        default=Config.DEFAULT_LISTEN_MODE,
        help="监听模式: push=按住Q键说话, continuous=持续监听 (默认: {})".format(Config.DEFAULT_LISTEN_MODE)
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

    # 子命令: add-memory
    mem_parser = subparsers.add_parser("add-memory", help="手动为指定用户添加记忆")
    mem_parser.add_argument(
        "--user", type=str, required=True,
        help="用户名称"
    )
    mem_parser.add_argument(
        "--content", type=str, default=None,
        help="要添加的记忆内容 (不提供则进入交互式添加模式)"
    )

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
        "--listen-mode", type=str, choices=["push", "continuous"],
        default=None,
        help="(兼容旧版) 监听模式: push=按住Q键, continuous=持续监听"
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

    args = parser.parse_args()
    
    # 设置全局DEBUG标志
    Config.DEBUG = getattr(args, 'debug', False)

    if args.command == "add-memory":
        run_add_memory(args)
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
