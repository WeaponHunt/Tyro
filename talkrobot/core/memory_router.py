"""
Memory routing for face-driven multi-user conversations.
"""
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from loguru import logger

from talkrobot.config import Config

if TYPE_CHECKING:
    from talkrobot.modules.memory.memory_module import MemoryModule


class UserMemoryRouter:
    """按用户动态提供记忆模块；无持久记忆时仅启用滑动窗口短期记忆。"""

    def __init__(self):
        self._cache: Dict[str, Tuple[Optional["MemoryModule"], bool]] = {}

    def get_memory_for_user(self, user: str) -> Tuple[Optional["MemoryModule"], bool]:
        user = (user or Config.DEFAULT_USER).strip()
        if user in self._cache:
            return self._cache[user]

        has_persistent = Config.has_persistent_memory(user)
        if not has_persistent:
            logger.info(f"用户[{user}]不存在长期记忆，使用短期记忆模式")
            self._cache[user] = (None, False)
            return self._cache[user]

        from talkrobot.modules.memory.memory_module import MemoryModule

        module = MemoryModule(
            config=Config.get_memory_config(user),
            user_id=Config.get_user_id(user),
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
