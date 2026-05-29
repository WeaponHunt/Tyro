"""Helpers for keeping httpx/OpenAI proxy environment compatible."""
from __future__ import annotations

import os
from typing import Iterable, List

try:
    from loguru import logger
except Exception:
    import logging

    logger = logging.getLogger(__name__)


PROXY_ENV_NAMES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)
UNSUPPORTED_PROXY_SCHEMES = ("socks://", "socks4://", "socks5://")


def drop_unsupported_proxy_env(
    *,
    prefix: str = "TalkRobot",
    env_names: Iterable[str] = PROXY_ENV_NAMES,
) -> List[str]:
    """Drop proxy env vars that this httpx installation cannot parse/use."""
    removed = []
    for name in env_names:
        value = (os.environ.get(name) or "").strip()
        if value.lower().startswith(UNSUPPORTED_PROXY_SCHEMES):
            os.environ.pop(name, None)
            removed.append(name)

    if removed:
        logger.warning(
            f"{prefix} 已忽略不受当前 httpx/OpenAI 环境支持的 socks 代理变量: "
            + ", ".join(removed)
        )
    return removed
