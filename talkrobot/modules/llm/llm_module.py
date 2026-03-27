"""
大语言模型(LLM)模块
负责生成对话回复
"""
from typing import Iterator, List, Dict, Tuple, Optional
import re

from openai import OpenAI
from loguru import logger

# 可用动作描述字典（用于让模型选择动作并校验）
gesture_descriptions = {
    "wave_left_hand()": "左手招手动作：从 0°(下垂) → 90° → 70°-110°摆动 → 复位到 0°",
    "wave_right_hand()": "右手招手动作：从 0°(下垂) → 90° → 70°-110°摆动 → 复位到 0°",
    "head_nod()": "点头动作：平视(0°) → 低头(-15°) → 平视, 表示赞同、认可、感谢",
    "head_shake()": "摇头动作：中心(0°) → 右转(+30°) → 左转(-30°) → 中心，表示反对、否认、拒绝",
    "greet()": "欢迎动作：点头 + 右手招手",
    "surprise()": "惊讶动作：头部微抬 + 双臂抬起40°",
    "reset()": "复位动作/立正：所有关节复位到初始位置",
    # ---- 参数化动作 ----
    "arm_left_up()": "左臂向上抬起（带参数angle，例如 arm_left_up(60)）",
    "arm_left_down()": "左臂向下放（带参数angle，例如 arm_left_down(30)）",
    "arm_right_up()": "右臂向上抬起（带参数angle，例如 arm_right_up(60)）",
    "arm_right_down()": "右臂向下放（带参数angle，例如 arm_right_down(30)）",
}

# 大模型中嵌入给机器人的动作指令说明（中/英）
GESTURE_INSTRUCTION_ZH = (
    "在回复中包含一个严格格式化的动作标签，格式为 [gesture:动作函数调用]，例如："
    "[gesture:head_nod()] 或 [gesture:arm_left_up(45)]。\n"
    "请只插入一个这样的标签（如果需要动作），并且不要在标签附近写多余的解释性文字。\n"
    "可用动作包括：wave_left_hand(), wave_right_hand(), head_nod(), head_shake(), greet(), surprise(), reset(), "
    "以及参数化动作 arm_left_up(angle), arm_left_down(angle), arm_right_up(angle), arm_right_down(angle)。"
)

GESTURE_INSTRUCTION_EN = (
    "for example: [gesture:head_nod()] or [gesture:arm_left_up(45)].\n"
    "Insert exactly one such tag when a gesture is appropriate and do not add explanatory text around the tag.\n"
    "Available gestures: wave_left_hand(), wave_right_hand(), head_nod(), head_shake(), greet(), surprise(), reset(), "
    "and parameterized gestures arm_left_up(angle), arm_left_down(angle), arm_right_up(angle), arm_right_down(angle)."
)

# 尝试可选导入 ROS2，如果不可用则保持为 None
_rclpy = None
_ros_String = None
_ros_Bool = None
_ros_Point = None
try:
    import rclpy as _tmp_rclpy  # type: ignore
    from std_msgs.msg import String as _tmp_String  # type: ignore
    from std_msgs.msg import Bool as _tmp_Bool  # type: ignore
    from geometry_msgs.msg import Point as _tmp_Point  # type: ignore
    _rclpy = _tmp_rclpy
    _ros_String = _tmp_String
    _ros_Bool = _tmp_Bool
    _ros_Point = _tmp_Point
except Exception:
    # ROS2 环境不可用，后续发布将被跳过
    _rclpy = None
    _ros_String = None
    _ros_Bool = None
    _ros_Point = None

class LLMModule:
    """大语言模型模块"""
    
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        system_prompt: str,
        expression_prompt: str = "",
        language: str = "zh",
        visualizer_enable_topic: str = "/face/visualizer/enabled",
    ):
        """
        初始化LLM模块
        
        Args:
            api_key: API密钥
            base_url: API地址
            model: 模型名称
            system_prompt: 系统提示词
            expression_prompt: 表情指令提示词（追加到 system_prompt 后）
            language: 输出语言（zh/en）
            visualizer_enable_topic: 可视化开关 topic（std_msgs/Bool）
        """
        logger.info(f"正在初始化LLM模块: model={model}")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        # 先记录 language，再基于语言选择合适的 gesture instruction
        self.language = (language or "zh").strip().lower()
        if self.language not in {"zh", "en"}:
            self.language = "zh"
        # 将 global system prompt + expression prompt + gesture instruction 合并
        if self.language == "en":
            self.system_prompt = system_prompt + expression_prompt + "\n\n" + GESTURE_INSTRUCTION_EN
        else:
            self.system_prompt = system_prompt + expression_prompt + "\n\n" + GESTURE_INSTRUCTION_ZH
        logger.info("LLM模块初始化完成")

        # 可选的 ROS2 发布器（如果环境中有 rclpy）
        self.ros2_enabled: bool = False
        self.gesture_publisher = None
        self.face_tracking_enable_publisher = None
        self.head_manual_control_deg_publisher = None
        self.visualizer_enable_publisher = None
        self.visualizer_enable_topic = str(visualizer_enable_topic or "/face/visualizer/enabled").strip() or "/face/visualizer/enabled"
        self._ros_node = None

        if _rclpy is not None and _ros_String is not None and _ros_Bool is not None and _ros_Point is not None:
            try:
                # 尝试初始化 rclpy（如果尚未初始化）并创建一个临时节点用于发布
                try:
                    _rclpy.init()
                except Exception:
                    # 如果 rclpy 已经初始化，则 init 可能报错，忽略
                    pass

                # 创建一个临时节点用于发布（在某些环境 create_node 可用）
                try:
                    # rclpy.create_node 在新版 rclpy 中可用
                    self._ros_node = _rclpy.create_node('llm_module_gesture_publisher')
                except Exception:
                    # 如果 create_node 不可用，尝试直接使用 Node 类
                    try:
                        from rclpy.node import Node as _Node  # type: ignore
                        class _TmpNode(_Node):
                            def __init__(self):
                                super().__init__('llm_module_gesture_publisher')
                        self._ros_node = _TmpNode()
                    except Exception:
                        self._ros_node = None

                if self._ros_node is not None:
                    self.gesture_publisher = self._ros_node.create_publisher(_ros_String, '/gesture', 10)
                    self.face_tracking_enable_publisher = self._ros_node.create_publisher(_ros_Bool, '/face_tracking/enable', 10)
                    self.head_manual_control_deg_publisher = self._ros_node.create_publisher(_ros_Point, '/head/manual_control_deg', 10)
                    self.visualizer_enable_publisher = self._ros_node.create_publisher(_ros_Bool, self.visualizer_enable_topic, 10)
                    self.ros2_enabled = True
                    logger.info("ROS2 可用：已创建 /gesture 发布器")
                    logger.info("人脸追踪开关 topic: /face_tracking/enable (Bool, True=开启, False=关闭)")
                    logger.info("手动头控 topic: /head/manual_control_deg (geometry_msgs/Point, 单位: 度)")
                    logger.info(f"可视化开关 topic: {self.visualizer_enable_topic} (Bool, True=开启, False=关闭)")
                else:
                    logger.info("ROS2 环境检测到但无法创建节点，跳过 /gesture 发布器创建")
            except Exception as e:
                logger.info(f"初始化 ROS2 发布器失败，已跳过：{e}")
                self.ros2_enabled = False

    @property
    def _is_english(self) -> bool:
        return self.language == "en"

    def _build_messages(self, user_input: str, context: str = "", system_prompt_override: str = "") -> List[Dict[str, str]]:
        """构建发送给大模型的消息列表。"""
        override = (system_prompt_override or "").strip()
        if override:
            # 即使有 override，也要追加 gesture instruction（中/英）
            if self._is_english:
                system_prompt = override + "\n\n" + GESTURE_INSTRUCTION_EN
            else:
                system_prompt = override + "\n\n" + GESTURE_INSTRUCTION_ZH
        else:
            system_prompt = self.system_prompt

        messages = [
            {"role": "system", "content": system_prompt}
        ]

        if context:
            messages.append({
                "role": "system",
                "content": (
                    f"Relevant background information about the user:\n{context}"
                    if self._is_english
                    else f"关于用户的相关背景信息:\n{context}"
                )
            })

        messages.append({"role": "user", "content": user_input})
        return messages
    
    def generate_response(self, user_input: str, context: str = "", system_prompt_override: str = "") -> str:
        """
        生成对话回复
        
        Args:
            user_input: 用户输入
            context: 上下文信息(来自记忆)
            
        Returns:
            str: AI回复
        """
        # Note: return_gesture is internal-only signal (keeps backward compatibility)
        return_gesture = False
        # allow caller to request (response, gesture) by passing a keyword arg
        # preserve signature for callers that don't use kwargs by only checking **kwargs
        # (we can't change function signature without breaking callers)
        # However Python doesn't expose kwargs here; instead support attribute injected by caller
        # We'll support an attribute on this instance used by internal helper generate_response_with_gesture
        if hasattr(self, "_llm_return_gesture") and self._llm_return_gesture:
            return_gesture = True
            # reset flag
            self._llm_return_gesture = False

        try:

            messages = self._build_messages(user_input, context, system_prompt_override)

            logger.info(f"构建的消息: {messages}")
            logger.info(f"正在生成回复,用户输入: {user_input}")

            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )

            raw = completion.choices[0].message.content
            logger.info(f"生成原始回复: {raw}")

            # 尝试从回复中解析 gesture 标签，格式: [gesture:arm_left_up(45)]
            gesture = ""
            m = re.search(r"\[\s*gesture\s*:\s*([^\]]+)\]", raw, flags=re.IGNORECASE)
            if m:
                gesture_candidate = m.group(1).strip()
                # 验证动作是否合法（仅比较函数名）
                func_name = gesture_candidate.split('(')[0].strip() + '()' if '(' in gesture_candidate else gesture_candidate.strip()
                if func_name in gesture_descriptions:
                    gesture = gesture_candidate
                    logger.info(f"解析到动作: {gesture}")
                else:
                    logger.warning(f"解析到非法动作 '{gesture_candidate}'，使用默认动作 'reset()'")
                    gesture = "reset()"

                # 从 raw 中移除该标签以得到纯回复文本
                raw = re.sub(r"\[\s*gesture\s*:[^\]]+\]", "", raw, flags=re.IGNORECASE).strip()

            # 发布 ROS2 动作（如果可用且有解析到动作）
            if gesture:
                try:
                    self.publish_gesture(gesture)
                except Exception as e:
                    logger.warning(f"发布动作失败: {e}")

            logger.info(f"生成回复: {raw}")

            if return_gesture:
                return raw, (gesture or "reset()")
            return raw

        except Exception as e:
            logger.error(f"LLM生成回复出错: {e}")
            if self._is_english:
                if return_gesture:
                    return "Sorry, I can't answer your question right now.", "reset()"
                return "Sorry, I can't answer your question right now."
            if return_gesture:
                return "抱歉,我现在无法回答您的问题。", "reset()"
            return "抱歉,我现在无法回答您的问题。"

    # select_gesture removed: gesture selection is parsed from model reply inside generate_response

    def publish_gesture(self, gesture_str: str) -> bool:
        """
        将动作字符串作为ROS2 String消息发布到 /gesture topic（如果ROS2可用）。
        返回 True 表示已发布，False 表示未发布（例如 ROS2 不可用）。
        """
        # if not self.ros2_enabled or self.gesture_publisher is None:
        #     logger.debug("ROS2 不可用，跳过发布动作")
        #     return False

        try:
            msg = _ros_String()
            msg.data = gesture_str
            self.gesture_publisher.publish(msg)
            logger.info(f"已发布动作到 /gesture: {gesture_str}")
            return True
        except Exception as e:
            logger.error(f"发布动作到 /gesture 失败: {e}")
            return False

    def publish_face_tracking_enable(self, enabled: bool) -> bool:
        """发布人脸追踪开关到 /face_tracking/enable。"""
        if self.face_tracking_enable_publisher is None or _ros_Bool is None:
            logger.debug("/face_tracking/enable 发布器不可用，跳过发布")
            return False

        try:
            msg = _ros_Bool()
            msg.data = bool(enabled)
            self.face_tracking_enable_publisher.publish(msg)
            logger.info(f"已发布人脸追踪开关到 /face_tracking/enable: {msg.data}")
            return True
        except Exception as e:
            logger.error(f"发布人脸追踪开关失败: {e}")
            return False

    def publish_head_manual_control_deg(self, x_deg: float, y_deg: float = 0.0) -> bool:
        """发布手动头控角度到 /head/manual_control_deg（单位：度，仅使用 x/y）。"""
        if self.head_manual_control_deg_publisher is None or _ros_Point is None:
            logger.debug("/head/manual_control_deg 发布器不可用，跳过发布")
            return False

        try:
            msg = _ros_Point()
            msg.x = float(x_deg)
            msg.y = float(y_deg)
            # 该 topic 仅约定使用 x/y，z 固定为 0
            msg.z = 0.0
            self.head_manual_control_deg_publisher.publish(msg)
            logger.info(
                f"已发布手动头控到 /head/manual_control_deg: x={msg.x:.2f}, y={msg.y:.2f}"
            )
            return True
        except Exception as e:
            logger.error(f"发布手动头控失败: {e}")
            return False

    def publish_visualizer_enable(self, enabled: bool) -> bool:
        """发布可视化界面开关到配置的 Bool topic。"""
        if self.visualizer_enable_publisher is None or _ros_Bool is None:
            logger.debug(f"{self.visualizer_enable_topic} 发布器不可用，跳过发布")
            return False

        try:
            msg = _ros_Bool()
            msg.data = bool(enabled)
            self.visualizer_enable_publisher.publish(msg)
            logger.info(
                f"已发布可视化开关到 {self.visualizer_enable_topic}: {int(msg.data)}"
            )
            return True
        except Exception as e:
            logger.error(f"发布可视化开关失败: {e}")
            return False

    def generate_response_with_gesture(self, user_input: str, context: str = "", system_prompt_override: str = "") -> Tuple[str, str]:
        """
        生成对话回复并同时选择一个配套的动作函数。

        Returns:
            (response_str, gesture_str)
        """
        try:
            # use internal flag to ask generate_response to also return gesture
            self._llm_return_gesture = True
            res = self.generate_response(user_input, context, system_prompt_override)
            if isinstance(res, tuple) and len(res) == 2:
                response, gesture = res
            else:
                response, gesture = res, "reset()"
            return response, (gesture or "reset()")
        except Exception as e:
            logger.error(f"生成回复并选择动作出错: {e}")
            if self._is_english:
                return "Sorry, I can't answer your question right now.", "reset()"
            return "抱歉,我现在无法回答您的问题。", "reset()"

    def generate_response_stream(self, user_input: str, context: str = "", system_prompt_override: str = "") -> Iterator[str]:
        """
        流式生成对话回复（按增量文本逐段返回）

        Args:
            user_input: 用户输入
            context: 上下文信息(来自记忆)

        Yields:
            str: 流式增量文本片段
        """
        try:
            messages = self._build_messages(user_input, context, system_prompt_override)
            logger.info(f"正在流式生成回复,用户输入: {user_input}")

            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                stream_options={"include_usage": True}
            )

            buffer = ""
            # 支持跨 chunk 的 [gesture:...] 标签解析
            for chunk in completion:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                if not (delta and delta.content):
                    continue

                buffer += delta.content

                # 处理所有完整的 gesture 标签
                while True:
                    m = re.search(r"\[\s*gesture\s*:[^\]]+\]", buffer, flags=re.IGNORECASE)
                    if not m:
                        break
                    gesture_tag = m.group(0)
                    # 提取并验证
                    inner = re.search(r"\[\s*gesture\s*:\s*([^\]]+)\]", gesture_tag, flags=re.IGNORECASE)
                    gesture = ""
                    if inner:
                        candidate = inner.group(1).strip()
                        func_name = candidate.split('(')[0].strip() + '()' if '(' in candidate else candidate.strip()
                        if func_name in gesture_descriptions:
                            gesture = candidate
                        else:
                            gesture = "reset()"
                    if gesture:
                        try:
                            self.publish_gesture(gesture)
                            logger.info(f"流式解析并发布动作: {gesture}")
                        except Exception as e:
                            logger.warning(f"流式发布动作失败: {e}")

                    # 删除该标签并继续处理 buffer
                    buffer = buffer[:m.start()] + buffer[m.end():]

                # 若 buffer 中尚有内容且不以不完整的 gesture 前缀开始，则输出
                # 如果 buffer 以完整的待完成 gesture 前缀开始 (如 "[gestu" ), 则保留等待后续 chunk
                prefix_idx = re.search(r"\[\s*gesture\s*:", buffer, flags=re.IGNORECASE)
                if prefix_idx and buffer.find(']') == -1:
                    # 不输出，等待更多数据
                    continue

                if buffer:
                    yield buffer
                    buffer = ""

            # 结束时若有残留未处理的 buffer，输出它（兜底）
            if buffer:
                yield buffer

            logger.info("流式回复生成完成")

        except Exception as e:
            logger.error(f"LLM流式生成回复出错: {e}")
            if self._is_english:
                yield "Sorry, I can't answer your question right now."
            else:
                yield "抱歉,我现在无法回答您的问题。"