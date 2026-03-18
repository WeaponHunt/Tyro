"""
人格提示词管理器
负责按用户加载人格 prompt，并提供默认回退
"""
import json
import os
import threading
from typing import Dict, Any

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


class PersonaManager:
    """按用户返回人格 prompt 的管理器。"""

    def __init__(self, profile_path: str, fallback_prompt: str):
        self.profile_path = profile_path
        self.fallback_prompt = (fallback_prompt or "").strip()
        self._profiles: Dict[str, Any] = {}
        self._default_prompt = self.fallback_prompt
        self._file_lock = threading.Lock()
        self.reload()

    def _load_raw_profiles(self) -> Dict[str, Any]:
        """读取原始人格配置（根对象）。"""
        if not self.profile_path or not os.path.exists(self.profile_path):
            return {}
        try:
            with open(self.profile_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                return raw
        except Exception as e:
            logger.warning(f"读取人格配置失败，将使用空配置写入: {e}")
        return {}

    def _dump_profiles(self, raw: Dict[str, Any]) -> None:
        """原子写回人格配置。"""
        if not self.profile_path:
            raise ValueError("persona profile path 为空")

        dir_path = os.path.dirname(self.profile_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        temp_path = f"{self.profile_path}.tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(temp_path, self.profile_path)

    @staticmethod
    def _extract_system_prompt(entry: Any) -> str:
        if isinstance(entry, str):
            return entry.strip()
        if isinstance(entry, dict):
            prompt = entry.get("system_prompt", "")
            if isinstance(prompt, str):
                return prompt.strip()
        return ""

    def reload(self) -> None:
        self._profiles = {}
        self._default_prompt = self.fallback_prompt

        if not self.profile_path:
            logger.warning("人格配置路径为空，使用 Config.SYSTEM_PROMPT 回退")
            return

        if not os.path.exists(self.profile_path):
            logger.warning(f"人格配置文件不存在，使用默认人格回退: {self.profile_path}")
            return

        try:
            with open(self.profile_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as e:
            logger.warning(f"加载人格配置失败，使用默认人格回退: {e}")
            return

        if not isinstance(raw, dict):
            logger.warning("人格配置格式错误（根节点应为对象），使用默认人格回退")
            return

        users_block = raw.get("users")
        profiles = users_block if isinstance(users_block, dict) else raw

        default_prompt = self._extract_system_prompt(profiles.get("default"))
        if default_prompt:
            self._default_prompt = default_prompt

        normalized = {}
        for user, entry in profiles.items():
            if not isinstance(user, str):
                continue
            key = user.strip()
            if not key:
                continue
            prompt = self._extract_system_prompt(entry)
            if prompt:
                normalized[key] = prompt

        self._profiles = normalized
        logger.info(
            f"人格配置已加载: users={len(self._profiles)}, profile_path={self.profile_path}"
        )

    def get_prompt_for_user(self, user: str) -> str:
        user_key = (user or "").strip()
        if user_key and user_key in self._profiles:
            return self._profiles[user_key]

        default_prompt = self._profiles.get("default", "")
        if default_prompt:
            return default_prompt

        return self._default_prompt

    def update_user_prompt(self, user: str, prompt: str) -> bool:
        """更新指定用户人格 prompt 并持久化。"""
        user_key = (user or "").strip()
        new_prompt = (prompt or "").strip()
        if not user_key or not new_prompt:
            return False

        with self._file_lock:
            raw = self._load_raw_profiles()
            users_block = raw.get("users")

            if isinstance(users_block, dict):
                target = users_block
            else:
                if not isinstance(raw, dict):
                    raw = {}
                target = raw

            current_entry = target.get(user_key, {})
            if isinstance(current_entry, dict):
                current_entry["system_prompt"] = new_prompt
                target[user_key] = current_entry
            else:
                target[user_key] = {"system_prompt": new_prompt}

            self._dump_profiles(raw)
            self.reload()

        logger.info(f"已更新用户人格提示词: user={user_key}")
        return True
