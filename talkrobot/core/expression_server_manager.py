"""
Lifecycle management for the optional expression video server.
"""
import atexit
import os
import subprocess
import sys
import time

from loguru import logger


class ExpressionServerManager:
    """Start and stop the expression server subprocess."""

    def __init__(self, project_root: str):
        self.project_root = project_root
        self._process = None

    @property
    def process(self):
        return self._process

    def start(self) -> bool:
        """自动启动表情服务器子进程。"""
        script_path = os.path.join(self.project_root, "expression", "expression_server.py")
        if not os.path.exists(script_path):
            logger.warning(f"表情服务器脚本不存在: {script_path}")
            return False
        if self._process is not None and self._process.poll() is None:
            return True

        try:
            logger.info(f"正在自动启动表情服务器: {script_path}")
            env = os.environ.copy()
            self._process = subprocess.Popen(
                [sys.executable, script_path],
                env=env,
                cwd=self.project_root,
            )
            atexit.register(self.stop)
            time.sleep(2)
            logger.info(f"表情服务器已启动 (PID: {self._process.pid})")
            return True
        except Exception as e:
            logger.error(f"启动表情服务器失败: {e}")
            return False

    def stop(self) -> None:
        """终止表情服务器子进程。"""
        if self._process and self._process.poll() is None:
            logger.info("正在关闭表情服务器...")
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            logger.info("表情服务器已关闭")
        self._process = None
