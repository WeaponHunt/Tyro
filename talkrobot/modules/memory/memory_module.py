"""Memory facade used by TalkRobot runtime code."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

try:
    from loguru import logger
except Exception:
    import logging

    logger = logging.getLogger(__name__)

from talkrobot.config import Config
from talkrobot.modules.memory.base import MemoryBackend, MemoryRecord
from talkrobot.modules.memory.factory import create_memory_backend
from talkrobot.modules.memory.filters import is_stable_user_memory, strip_memory_prefix


class MemoryModule:
    """Stable facade over pluggable memory backends.

    The public methods keep the previous API intact:
    `add_memory`, `search_memory`, `get_all_memories`,
    `add_user_memory_if_stable`, and `shutdown`.
    """

    def __init__(
        self,
        config: dict,
        user_id: str,
        max_workers: int = 2,
        provider: Optional[str] = None,
        backend: Optional[MemoryBackend] = None,
    ):
        self.provider = (provider or Config.MEMORY_PROVIDER or "mem0").strip().lower()
        self.user_id = user_id
        self.backend = backend or create_memory_backend(self.provider, config, user_id)
        self.db_path = getattr(self.backend, "db_path", None)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        logger.info(f"Memory模块初始化完成: provider={self.provider}, path={self.db_path}")

    def add_memory(self, text: str, async_mode: bool = True) -> None:
        text = (text or "").strip()
        if not text:
            return
        if async_mode:
            self.executor.submit(self._add_memory_sync, text)
            logger.info(f"已提交记忆任务: {text[:50]}...")
            return
        self._add_memory_sync(text)

    @staticmethod
    def is_stable_user_memory(text: str) -> bool:
        return is_stable_user_memory(text)

    def add_user_memory_if_stable(self, user_text: str, async_mode: bool = True) -> bool:
        clean_text = strip_memory_prefix(user_text)
        if not self.is_stable_user_memory(clean_text):
            logger.info(f"跳过非稳定用户记忆: {clean_text[:50]}...")
            return False

        self.add_memory(f"用户稳定信息: {clean_text}", async_mode=async_mode)
        return True

    def search_memory(self, query: str, limit: int = 3) -> str:
        try:
            records = self.backend.search(query, limit=limit)
        except Exception as exc:
            logger.error(f"搜索记忆失败: {exc}")
            return ""

        context = self._format_context(records)
        logger.info(f"检索到 {len(records)} 条相关记忆")
        return context

    def get_all_memories(self) -> list:
        try:
            return [record.as_dict() for record in self.backend.get_all()]
        except Exception as exc:
            logger.error(f"获取记忆失败: {exc}")
            return []

    def shutdown(self) -> None:
        logger.info("正在等待所有记忆任务完成...")
        logger.info(f"记忆数据已保存至: {self.db_path}")
        self.executor.shutdown(wait=True)
        try:
            self.backend.shutdown()
        finally:
            logger.info("Memory模块已安全关闭")

    def _add_memory_sync(self, text: str) -> None:
        try:
            self.backend.add(text)
            logger.info(f"已添加记忆: {text[:50]}...")
        except Exception as exc:
            logger.error(f"添加记忆失败: {exc}")

    @staticmethod
    def _format_context(records: List[MemoryRecord]) -> str:
        return "\n".join(
            f"{index}. {record.text}"
            for index, record in enumerate(records, start=1)
            if record.text
        )
