"""
ROS2 voice bridge.

Publishes recognized ASR text and subscribes to external TTS text commands.
"""
import queue
import threading
from typing import Callable, Optional

from loguru import logger


try:
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from std_msgs.msg import String
except Exception:
    rclpy = None
    SingleThreadedExecutor = None
    Node = object
    String = None


class Ros2VoiceBridge:
    """Bridge between TalkRobot voice modules and ROS2 String topics."""

    def __init__(
        self,
        tts_module,
        audio_recorder=None,
        node_name: str = "talkrobot_voice_bridge",
        asr_text_topic: str = "/asr/text",
        tts_text_topic: str = "/tts/text",
        chat_text_topic: str = "/chat/text",
        assistant_text_topic: str = "/assistant/text",
        chat_text_callback: Optional[Callable[[str], None]] = None,
        queue_size: int = 10,
    ):
        self.tts = tts_module
        self.audio_recorder = audio_recorder
        self.node_name = str(node_name or "talkrobot_voice_bridge").strip() or "talkrobot_voice_bridge"
        self.asr_text_topic = str(asr_text_topic or "/asr/text").strip() or "/asr/text"
        self.tts_text_topic = str(tts_text_topic or "/tts/text").strip() or "/tts/text"
        self.chat_text_topic = str(chat_text_topic or "/chat/text").strip() or "/chat/text"
        self.assistant_text_topic = str(assistant_text_topic or "/assistant/text").strip() or "/assistant/text"
        self._chat_text_callback = chat_text_callback
        try:
            resolved_queue_size = int(queue_size)
        except (TypeError, ValueError):
            resolved_queue_size = 10
        self.queue_size = resolved_queue_size if resolved_queue_size > 0 else 10

        self.enabled = False
        self._node = None
        self._executor = None
        self._asr_publisher = None
        self._assistant_publisher = None
        self._spin_thread = None
        self._tts_thread = None
        self._chat_thread = None
        self._stop_event = threading.Event()
        self._tts_queue: queue.Queue[Optional[str]] = queue.Queue()
        self._chat_queue: queue.Queue[Optional[str]] = queue.Queue()

    def set_chat_text_callback(self, callback: Optional[Callable[[str], None]]) -> None:
        """Set callback used for external text-chat input."""
        self._chat_text_callback = callback

    def start(self) -> bool:
        """Start ROS2 node, executor thread, and TTS worker."""
        if self.enabled:
            return True

        if rclpy is None or String is None or SingleThreadedExecutor is None:
            logger.warning("ROS2 环境不可用，语音 bridge 未启动")
            return False

        try:
            try:
                rclpy.init()
            except Exception:
                logger.debug("rclpy 可能已经初始化，继续创建 voice bridge")

            class _VoiceBridgeNode(Node):
                def __init__(node_self):
                    super().__init__(self.node_name)

            self._node = _VoiceBridgeNode()
            self._asr_publisher = self._node.create_publisher(
                String,
                self.asr_text_topic,
                self.queue_size,
            )
            self._assistant_publisher = self._node.create_publisher(
                String,
                self.assistant_text_topic,
                self.queue_size,
            )
            self._node.create_subscription(
                String,
                self.tts_text_topic,
                self._on_tts_text,
                self.queue_size,
            )
            self._node.create_subscription(
                String,
                self.chat_text_topic,
                self._on_chat_text,
                self.queue_size,
            )

            self._executor = SingleThreadedExecutor()
            self._executor.add_node(self._node)
            self._stop_event.clear()

            self._spin_thread = threading.Thread(
                target=self._spin,
                name="ros2_voice_bridge_spin",
                daemon=True,
            )
            self._tts_thread = threading.Thread(
                target=self._tts_worker,
                name="ros2_voice_bridge_tts",
                daemon=True,
            )
            self._chat_thread = threading.Thread(
                target=self._chat_worker,
                name="ros2_voice_bridge_chat",
                daemon=True,
            )
            self._spin_thread.start()
            self._tts_thread.start()
            self._chat_thread.start()
            self.enabled = True
            logger.info(
                f"ROS2 语音 bridge 已启动: ASR publish={self.asr_text_topic}, "
                f"TTS subscribe={self.tts_text_topic}"
            )
            return True
        except Exception as e:
            logger.warning(f"启动 ROS2 语音 bridge 失败: {e}")
            self.stop()
            return False

    def _spin(self) -> None:
        try:
            self._executor.spin()
        except Exception as e:
            if not self._stop_event.is_set():
                logger.warning(f"ROS2 voice bridge spin 异常: {e}")

    def _on_tts_text(self, msg) -> None:
        text = str(getattr(msg, "data", "") or "").strip()
        if not text:
            return
        self._tts_queue.put(text)
        logger.info(f"收到 ROS2 TTS 文本: {text}")

    def _on_chat_text(self, msg) -> None:
        text = str(getattr(msg, "data", "") or "").strip()
        if not text:
            return
        self._chat_queue.put(text)
        logger.info(f"Received ROS2 chat text: {text}")

    def _tts_worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                text = self._tts_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if text is None:
                break
            if self._stop_event.is_set():
                break

            if self.tts is None:
                logger.warning("收到 ROS2 TTS 文本，但 TTS 模块不可用")
                continue

            try:
                if self.audio_recorder is not None:
                    self.audio_recorder.is_tts_playing = True
                self.tts.synthesize(text, play_audio=True)
            except Exception as e:
                logger.warning(f"ROS2 TTS 播放失败: {e}")
            finally:
                if self.audio_recorder is not None:
                    self.audio_recorder.is_tts_playing = False

    def _chat_worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                text = self._chat_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if text is None:
                break
            if self._stop_event.is_set():
                break

            callback = self._chat_text_callback
            if callback is None:
                logger.warning("Received ROS2 chat text, but chat callback is not ready")
                continue

            try:
                callback(text)
            except Exception as e:
                logger.warning(f"ROS2 chat text handling failed: {e}")

    def publish_asr_text(self, text: str) -> bool:
        """Publish recognized ASR text to ROS2."""
        text = str(text or "").strip()
        if not text:
            return False
        if not self.enabled or self._asr_publisher is None or String is None:
            return False

        try:
            msg = String()
            msg.data = text
            self._asr_publisher.publish(msg)
            logger.info(f"已发布 ASR 文本到 {self.asr_text_topic}: {text}")
            return True
        except Exception as e:
            logger.warning(f"发布 ASR 文本到 ROS2 失败: {e}")
            return False

    def publish_assistant_text(self, text: str) -> bool:
        """Publish assistant reply text to ROS2."""
        text = str(text or "").strip()
        if not text:
            return False
        if not self.enabled or self._assistant_publisher is None or String is None:
            return False

        try:
            msg = String()
            msg.data = text
            self._assistant_publisher.publish(msg)
            logger.info(f"Published assistant text to {self.assistant_text_topic}: {text}")
            return True
        except Exception as e:
            logger.warning(f"Publishing assistant text to ROS2 failed: {e}")
            return False

    def stop(self) -> None:
        """Stop worker threads and destroy ROS2 node resources."""
        self._stop_event.set()
        try:
            self._tts_queue.put_nowait(None)
        except Exception:
            pass
        try:
            self._chat_queue.put_nowait(None)
        except Exception:
            pass

        if self._executor is not None:
            try:
                self._executor.shutdown()
            except Exception as e:
                logger.debug(f"关闭 ROS2 voice bridge executor 异常: {e}")

        for thread in (self._spin_thread, self._tts_thread, self._chat_thread):
            if thread is not None and thread.is_alive() and thread != threading.current_thread():
                thread.join(timeout=1.0)

        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception as e:
                logger.debug(f"销毁 ROS2 voice bridge node 异常: {e}")

        self._node = None
        self._executor = None
        self._asr_publisher = None
        self._assistant_publisher = None
        self._spin_thread = None
        self._tts_thread = None
        self._chat_thread = None
        self.enabled = False
