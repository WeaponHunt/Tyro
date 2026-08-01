"""
表情模块
负责与表情服务器通信，控制机器人表情显示
"""
import re
import requests
from loguru import logger
from typing import Optional, Tuple


# LLM 可选的表情列表（用于 system prompt）
AVAILABLE_EXPRESSIONS = [
    "happy",       # 开心
    "angry",       # 生气
    "sad",         # 悲伤
    "scared",      # 害怕
    "surprised",   # 惊讶
    "more-happy",  # 非常开心
    "dizzy",       # 头晕/困惑
    "evil_smile",  # 坏笑
    "nauty_smile", # 调皮笑
    "pitying",     # 同情
]

# 表情标签正则：匹配 [expression:xxx]
EXPRESSION_PATTERN = re.compile(r'\[expression:(\w[\w-]*)\]')


class ExpressionModule:
    """表情控制模块"""

    def __init__(self, server_url: str, default_expression: str = "neutral"):
        """
        初始化表情模块

        Args:
            server_url: 表情服务器地址，例如 http://localhost:8001
            default_expression: 默认表情名称
        """
        self.server_url = server_url.rstrip("/")
        self.default_expression = default_expression
        self._available = self._check_server()
        if self._available:
            logger.info(f"表情模块初始化完成: server={self.server_url}")
        else:
            logger.warning(f"表情服务器不可用({self.server_url})，表情功能将被禁用")

    def _check_server(self) -> bool:
        """检查表情服务器是否可用"""
        try:
            resp = requests.get(f"{self.server_url}/expressions", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    @property
    def is_available(self) -> bool:
        return self._available

    def set_expression(self, expression: str) -> bool:
        """
        设置表情

        Args:
            expression: 表情名称

        Returns:
            bool: 是否成功
        """
        if not self._available:
            logger.debug("表情服务器不可用，跳过表情设置")
            return False
        try:
            resp = requests.post(
                f"{self.server_url}/expression/{expression}", timeout=5
            )
            if resp.status_code == 200:
                logger.info(f"表情已切换: {expression}")
                return True
            else:
                logger.warning(f"表情切换失败: {expression}, status={resp.status_code}")
                return False
        except Exception as e:
            logger.error(f"表情切换请求出错: {e}")
            return False

    def reset_expression(self) -> bool:
        """
        重置为默认表情

        Returns:
            bool: 是否成功
        """
        if not self._available:
            return False
        try:
            resp = requests.post(f"{self.server_url}/reset", timeout=5)
            if resp.status_code == 200:
                logger.info("表情已重置为默认")
                return True
            else:
                logger.warning(f"表情重置失败, status={resp.status_code}")
                return False
        except Exception as e:
            logger.error(f"表情重置请求出错: {e}")
            return False

    def set_recording(self, recording: bool) -> bool:
        """
        设置表情窗口的录像提示叠加层。

        Args:
            recording: True 显示 REC 提示，False 隐藏

        Returns:
            bool: 是否成功
        """
        if not self._available:
            return False
        state = "true" if recording else "false"
        try:
            resp = requests.post(f"{self.server_url}/recording/{state}", timeout=3)
            if resp.status_code == 200:
                logger.info(f"表情录像提示已{'开启' if recording else '关闭'}")
                return True
            logger.warning(f"设置表情录像提示失败: recording={recording}, status={resp.status_code}")
            return False
        except Exception as e:
            logger.error(f"设置表情录像提示请求出错: {e}")
            return False

    def set_mouth_speaking(self, speaking: bool) -> bool:
        """
        设置表情窗口的嘴部说话动画叠加层。

        Args:
            speaking: True 开启嘴部动画，False 关闭

        Returns:
            bool: 是否成功
        """
        if not self._available:
            return False
        state = "true" if speaking else "false"
        try:
            resp = requests.post(f"{self.server_url}/mouth/{state}", timeout=3)
            if resp.status_code == 200:
                logger.debug(f"嘴部说话动画已{'开启' if speaking else '关闭'}")
                return True
            logger.warning(f"设置嘴部说话动画失败: speaking={speaking}, status={resp.status_code}")
            return False
        except Exception as e:
            logger.error(f"设置嘴部说话动画请求出错: {e}")
            return False

    def set_status(self, status: str, visible: bool = True) -> bool:
        """
        设置表情窗口的机器人状态文字叠加层。

        Args:
            status: 要显示的状态文字
            visible: 是否显示状态文字

        Returns:
            bool: 是否成功
        """
        if not self._available:
            return False
        try:
            resp = requests.post(
                f"{self.server_url}/status",
                json={"status": str(status or ""), "visible": bool(visible)},
                timeout=3,
            )
            if resp.status_code == 200:
                logger.debug(f"表情状态文字已更新: {status}")
                return True
            logger.warning(f"设置表情状态文字失败: status={status}, http_status={resp.status_code}")
            return False
        except Exception as e:
            logger.error(f"设置表情状态文字请求出错: {e}")
            return False

    def set_heard_text(self, text: str, visible: bool = True) -> bool:
        """
        设置表情窗口底部的 ASR 识别文本叠加层。

        Args:
            text: 要显示的识别文本
            visible: 是否显示识别文本

        Returns:
            bool: 是否成功
        """
        if not self._available:
            return False
        try:
            resp = requests.post(
                f"{self.server_url}/heard",
                json={"text": str(text or ""), "visible": bool(visible)},
                timeout=3,
            )
            if resp.status_code == 200:
                logger.debug(f"表情识别文本已更新: {text}")
                return True
            logger.warning(f"设置表情识别文本失败: text={text}, http_status={resp.status_code}")
            return False
        except Exception as e:
            logger.error(f"设置表情识别文本请求出错: {e}")
            return False

    @staticmethod
    def parse_expression_from_response(response: str) -> Tuple[str, Optional[str]]:
        """
        从 LLM 回复中解析表情标签并返回纯文本

        Args:
            response: LLM 原始回复，可能包含 [expression:xxx] 标签

        Returns:
            Tuple[str, Optional[str]]: (纯文本回复, 表情名称或None)
        """
        match = EXPRESSION_PATTERN.search(response)
        if match:
            expression = match.group(1)
            # 移除表情标签，得到纯文本
            clean_text = EXPRESSION_PATTERN.sub("", response).strip()
            return clean_text, expression
        return response, None

    @staticmethod
    def get_expression_prompt() -> str:
        """
        生成用于 system prompt 的表情指令

        Returns:
            str: 关于表情选择的提示词片段
        """
        expr_list = ", ".join(AVAILABLE_EXPRESSIONS)
        return (
            f"\n\n【表情指令】在每次回复的开头，根据回复内容的情感，"
            f"添加一个表情标签，格式为 [expression:表情名称]。"
            f"可选的表情有: {expr_list}。"
            f"例如：如果回复是开心的内容，就写 [expression:happy] 然后接回复内容。"
            f"只需要一个表情标签，放在回复最前面。"
        )
