"""
Face-driven user identity resolver.

This module owns camera polling and face recognition state so the CLI entry
point can stay focused on wiring the application together.
"""
import re
import threading
from typing import Callable, Optional, Tuple

from loguru import logger

from talkrobot.config import Config


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
        self._on_user_change: Optional[Callable[[str, bool], None]] = None
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._track_thread = None
        self._current_user = self.unknown_user
        self._current_is_familiar = False

        if not enabled:
            return

        try:
            import cv2
            from talkrobot.modules.face_recognize.face_recognition import FaceRecognitionModule

            self._module = FaceRecognitionModule(
                known_faces_dir=Config.FACE_KNOWN_FACES_DIR,
                model_name=Config.FACE_MODEL_NAME,
                use_gpu=Config.FACE_USE_GPU,
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

    def _detect_user_once(self) -> Tuple[str, bool]:
        """执行单次人脸检测并返回 (用户, 是否熟人)。"""
        if not self.enabled or self._cap is None or self._module is None:
            return self.default_user, False

        ok, frame = self._cap.read()
        if not ok:
            logger.debug("读取摄像头帧失败，回退陌生人用户")
            return self.unknown_user, False

        try:
            result = self._module.process_frame(frame)
            label = str(result.get("label", "")).strip()
        except Exception as e:
            logger.warning(f"人脸识别处理失败，回退陌生人用户: {e}")
            return self.unknown_user, False

        if label in self._NON_TARGET_LABELS:
            return self.unknown_user, False
        if label == "陌生人":
            return self.unknown_user, False
        return self._sanitize_user_name(label), True

    def _tracking_loop(self) -> None:
        """后台持续人脸追踪，检测到对象变化时触发回调。"""
        while not self._stop_event.is_set():
            user, is_familiar = self._detect_user_once()
            changed = False
            with self._state_lock:
                if user != self._current_user:
                    self._current_user = user
                    self._current_is_familiar = is_familiar
                    changed = True

            if changed:
                logger.info(f"人脸交互对象变化: {user}")
                callback = self._on_user_change
                if callback is not None:
                    try:
                        callback(user, is_familiar)
                    except Exception as e:
                        logger.warning(f"人脸用户切换回调异常: {e}")

            self._stop_event.wait(self.poll_interval)

    def set_on_user_change(self, callback: Callable[[str, bool], None]) -> None:
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

    def shutdown(self) -> None:
        self._stop_event.set()
        if self._track_thread is not None and self._track_thread.is_alive():
            self._track_thread.join(timeout=1.0)
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception as e:
                logger.warning(f"释放摄像头失败: {e}")
        if self._module is not None:
            try:
                self._module.shutdown()
            except Exception as e:
                logger.warning(f"关闭人脸识别模块失败: {e}")
        self._cap = None
        self._module = None
        self._track_thread = None
        self.enabled = False
