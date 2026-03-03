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
from loguru import logger

from talkrobot.config import Config
from talkrobot.modules.memory_module import MemoryModule

# 全局变量：表情服务器子进程
_expression_server_process = None


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
    from talkrobot.modules.asr_module import ASRModule
    from talkrobot.modules.tts_module import TTSModule
    from talkrobot.modules.llm_module import LLMModule
    from talkrobot.core.audio_recorder import AudioRecorder
    from talkrobot.core.conversation_manager import ConversationManager

    user = args.user
    memory_module = None
    tts_module = None
    audio_recorder = None
    conversation_manager = None

    try:
        _setup_logger()

        logger.info("="*50)
        logger.info(f"正在初始化对话机器人系统... (用户: {user})")
        logger.info("="*50)

        # 1. 初始化各个模块
        asr_module = ASRModule(
            model_name=Config.ASR_MODEL,
            device=Config.ASR_DEVICE
        )

        tts_module = TTSModule(
            lang_code=Config.TTS_LANG_CODE,
            voice=Config.TTS_VOICE,
            speed=Config.TTS_SPEED
        )

        # 初始化表情模块（可选）
        expression_module = None
        expression_prompt = ""
        if Config.EXPRESSION_ENABLED:
            from talkrobot.modules.expression_module import ExpressionModule
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
            system_prompt=Config.SYSTEM_PROMPT,
            expression_prompt=expression_prompt
        )

        memory_module = MemoryModule(
            config=Config.get_memory_config(user),
            user_id=Config.get_user_id(user)
        )

        # 2. 创建对话管理器
        tts_enabled = not args.no_tts
        listen_mode = getattr(args, 'listen_mode', None) or Config.DEFAULT_LISTEN_MODE

        # 3. 创建音频录制器
        audio_recorder = AudioRecorder(
            sample_rate=Config.SAMPLE_RATE,
            channels=Config.CHANNELS,
            listen_mode=listen_mode,
            vad_check_interval=Config.VAD_CHECK_INTERVAL,
            pre_speech_duration=Config.VAD_PRE_SPEECH_DURATION,
            silence_duration=Config.VAD_SILENCE_DURATION,
            min_speech_duration=Config.VAD_MIN_SPEECH_DURATION,
        )

        conversation_manager = ConversationManager(
            asr_module=asr_module,
            tts_module=tts_module,
            llm_module=llm_module,
            memory_module=memory_module,
            tts_enabled=tts_enabled,
            expression_module=expression_module,
            audio_recorder=audio_recorder,
            audio_min_duration=Config.AUDIO_MIN_DURATION,
            audio_min_rms=Config.AUDIO_MIN_RMS,
            sample_rate=Config.SAMPLE_RATE,
        )

        logger.info("所有模块初始化完成!")
        logger.info("="*50)

        # 4. 启动录音并开始对话
        audio_recorder.start(
            on_audio_complete=conversation_manager.process_audio_async
        )

    except KeyboardInterrupt:
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

        if memory_module is not None:
            try:
                memory_module.shutdown()
            except Exception as e:
                logger.warning(f"关闭记忆模块异常: {e}")

        _stop_expression_server()


def run_add_memory(args):
    """手动添加记忆"""
    _setup_logger()

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
        "--listen-mode", type=str, choices=["push", "continuous"],
        default=Config.DEFAULT_LISTEN_MODE,
        help="监听模式: push=按住Q键说话, continuous=持续监听 (默认: {})".format(Config.DEFAULT_LISTEN_MODE)
    )
    chat_parser.add_argument(
        "--debug", action="store_true", default=False,
        help="启用调试模式，输出详细日志，Ctrl+C时显示线程堆栈"
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
        "--listen-mode", type=str, choices=["push", "continuous"],
        default=None,
        help="(兼容旧版) 监听模式: push=按住Q键, continuous=持续监听"
    )
    parser.add_argument(
        "--debug", action="store_true", default=False,
        help="(兼容旧版) 启用调试模式"
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
        run_chat(args)


if __name__ == "__main__":
    main()