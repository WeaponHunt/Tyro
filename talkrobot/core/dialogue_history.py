"""
Short-term dialogue history helpers.
"""
import threading
from typing import Dict, List, Optional, Tuple


class SlidingWindowDialogueHistory:
    """按用户维护最近 n 轮短期对话。"""

    def __init__(self, max_rounds: int):
        self.max_rounds = max(0, int(max_rounds))
        self._lock = threading.Lock()
        self._rounds_by_user: Dict[str, List[Tuple[str, str]]] = {}

    def build_context(self, user: str) -> str:
        """构建最近 n 轮对话窗口文本。"""
        if self.max_rounds <= 0:
            return ""

        with self._lock:
            rounds = list(self._rounds_by_user.get(user, []))

        if not rounds:
            return ""

        lines = [f"最近对话窗口（最近{len(rounds)}轮）:"]
        for idx, (user_text, assistant_text) in enumerate(rounds, start=1):
            lines.append(f"第{idx}轮 用户: {user_text}")
            lines.append(f"第{idx}轮 机器人: {assistant_text}")
        return "\n".join(lines)

    def append(self, user: str, user_text: str, assistant_text: str) -> None:
        """将一轮对话写入滑动窗口。"""
        if self.max_rounds <= 0:
            return

        with self._lock:
            rounds = self._rounds_by_user.setdefault(user, [])
            rounds.append((user_text, assistant_text))
            if len(rounds) > self.max_rounds:
                self._rounds_by_user[user] = rounds[-self.max_rounds :]

    def clear(self, user: Optional[str] = None) -> None:
        """清空指定用户或全部用户的短期对话。"""
        if self.max_rounds <= 0:
            return

        with self._lock:
            if user is None:
                self._rounds_by_user.clear()
            elif user in self._rounds_by_user:
                self._rounds_by_user[user] = []
