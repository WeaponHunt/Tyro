"""
大语言模型(LLM)模块
负责生成对话回复
"""
from typing import Iterator, List, Dict

from openai import OpenAI
from loguru import logger

class LLMModule:
    """大语言模型模块"""
    
    def __init__(self, api_key: str, base_url: str, model: str, system_prompt: str, expression_prompt: str = ""):
        """
        初始化LLM模块
        
        Args:
            api_key: API密钥
            base_url: API地址
            model: 模型名称
            system_prompt: 系统提示词
            expression_prompt: 表情指令提示词（追加到 system_prompt 后）
        """
        logger.info(f"正在初始化LLM模块: model={model}")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.system_prompt = system_prompt + expression_prompt
        logger.info("LLM模块初始化完成")

    def _build_messages(self, user_input: str, context: str = "", system_prompt_override: str = "") -> List[Dict[str, str]]:
        """构建发送给大模型的消息列表。"""
        system_prompt = (system_prompt_override or "").strip() or self.system_prompt
        messages = [
            {"role": "system", "content": system_prompt}
        ]

        if context:
            messages.append({
                "role": "system",
                "content": f"关于用户的相关背景信息:\n{context}"
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
        try:
            messages = self._build_messages(user_input, context, system_prompt_override)
            
            logger.info(f"正在生成回复,用户输入: {user_input}")
            
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            
            response = completion.choices[0].message.content
            logger.info(f"生成回复: {response}")
            return response
            
        except Exception as e:
            logger.error(f"LLM生成回复出错: {e}")
            return "抱歉,我现在无法回答您的问题。"

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

            for chunk in completion:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content

            logger.info("流式回复生成完成")

        except Exception as e:
            logger.error(f"LLM流式生成回复出错: {e}")
            yield "抱歉,我现在无法回答您的问题。"