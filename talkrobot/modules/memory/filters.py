"""Stable-memory heuristics shared by memory backends."""
from __future__ import annotations

import re


STABLE_MEMORY_PATTERNS = [
    r"(?:我|本人)(?:叫|名叫|名字叫|的名字是)",
    r"(?:我的)?(?:名字|姓名|昵称|外号)(?:是|叫)",
    r"(?:我|本人)(?:今年)?\d{1,3}岁",
    r"(?:我的)?(?:生日|出生日期|年龄|星座|生肖)(?:是|在|为)",
    r"(?:我|本人)(?:现在)?(?:来自|住在|居住在|家在|老家在)",
    r"(?:我的)?(?:家乡|住址|地址|城市|学校|专业|公司|职业|工作|岗位)(?:是|在|为)",
    r"(?:我|本人)(?:在|就读于|毕业于|任职于|工作于)",
    r"(?:我|本人)(?:喜欢|爱好|偏好|讨厌|不喜欢|不爱|害怕|擅长|习惯)",
    r"(?:我的)?(?:爱好|偏好|口味|习惯|忌口|过敏源|过敏|禁忌)(?:是|有|包括|为)",
    r"(?:请)?(?:记住|记一下|帮我记住|以后记得)",
    r"(?:以后|今后)(?:叫我|称呼我|不要叫我|请叫我)",
]

TRANSIENT_MEMORY_PATTERNS = [
    r"^(?:帮我|请你|能不能|可以|给我|告诉我|查一下|解释|写|生成|总结|翻译|打开|播放)",
    r"(?:今天|昨天|明天|刚才|刚刚|现在|正在|这次|这会儿|等下|一会儿)",
    r"[?？]$",
]


def strip_memory_prefix(text: str) -> str:
    return re.sub(r"^\s*(?:用户说|用户|我)\s*[:：]\s*", "", text or "").strip()


def is_stable_user_memory(text: str) -> bool:
    """Return True when user text looks useful as long-term memory."""
    clean_text = strip_memory_prefix(text)
    if not clean_text:
        return False

    if re.search(r"(?:请)?(?:记住|记一下|帮我记住|以后记得)", clean_text):
        return True

    has_stable_signal = any(re.search(pattern, clean_text) for pattern in STABLE_MEMORY_PATTERNS)
    if not has_stable_signal:
        return False

    if any(re.search(pattern, clean_text) for pattern in TRANSIENT_MEMORY_PATTERNS):
        return bool(re.search(r"(?:住在|居住在|家在|老家在|工作|就读|任职)", clean_text))

    return True
