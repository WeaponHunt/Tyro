"""
对话机器人主程序
整合所有模块,启动对话机器人
"""
import sys
import argparse
from loguru import logger

from talkrobot.config import Config
from talkrobot.modules.memory_module import MemoryModule


def _setup_logger():
    """配置日志"""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:8}</level> | <level>{message}</level>",
        level="INFO"
    )
    logger.add("talkrobot/logs/robot_{time}.log", rotation="1 day", retention="7 days")


def run_chat(args):
    """启动对话机器人"""
    from talkrobot.modules.asr_module import ASRModule
    from talkrobot.modules.tts_module import TTSModule
    from talkrobot.modules.llm_module import LLMModule
    from talkrobot.core.audio_recorder import AudioRecorder
    from talkrobot.core.conversation_manager import ConversationManager

    user = args.user

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

        llm_module = LLMModule(
            api_key=Config.LLM_API_KEY,
            base_url=Config.LLM_BASE_URL,
            model=Config.LLM_MODEL,
            system_prompt=Config.SYSTEM_PROMPT
        )

        memory_module = MemoryModule(
            config=Config.get_memory_config(user),
            user_id=Config.get_user_id(user)
        )

        # 2. 创建对话管理器
        conversation_manager = ConversationManager(
            asr_module=asr_module,
            tts_module=tts_module,
            llm_module=llm_module,
            memory_module=memory_module
        )

        # 3. 创建音频录制器
        audio_recorder = AudioRecorder(
            sample_rate=Config.SAMPLE_RATE,
            channels=Config.CHANNELS
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
        # 等待所有内存任务完成
        if 'memory_module' in locals():
            memory_module.shutdown()
    except Exception as e:
        logger.error(f"程序运行出错: {e}", exc_info=True)
        print(f"\n❌ 程序出错: {e}")
        # 异常发生时也要关闭线程池
        if 'memory_module' in locals():
            memory_module.shutdown()


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

    args = parser.parse_args()

    if args.command == "add-memory":
        run_add_memory(args)
    elif args.command == "chat":
        run_chat(args)
    else:
        # 兼容旧版调用方式: python -m talkrobot.main --user ljc
        if args.user is None:
            args.user = Config.DEFAULT_USER
        run_chat(args)


if __name__ == "__main__":
    main()