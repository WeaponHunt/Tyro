"""
记忆管理模块
负责存储和检索对话历史
"""
import os
import re
from concurrent.futures import ThreadPoolExecutor
import threading

try:
    from loguru import logger
except Exception:
    import logging

    logger = logging.getLogger(__name__)

try:
    from mem0 import Memory
except Exception:
    Memory = None


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


def _strip_memory_prefix(text: str) -> str:
    return re.sub(r"^\s*(?:用户说|用户|我)\s*[:：]\s*", "", text or "").strip()


class MemoryModule:
    """记忆管理模块"""
    
    def __init__(self, config: dict, user_id: str, max_workers: int = 2):
        """
        初始化记忆模块
        
        Args:
            config: mem0配置
            user_id: 用户ID
            max_workers: 线程池最大工作线程数
        """
        logger.info("正在初始化Memory模块")
        
        # 确保数据库目录存在
        db_path = config.get("vector_store", {}).get("config", {}).get("path")
        if db_path and not os.path.exists(db_path):
            os.makedirs(db_path, exist_ok=True)
            logger.info(f"创建Memory数据库目录: {db_path}")
        
        # 初始化Memory
        if Memory is None:
            raise ImportError("mem0 is required to initialize MemoryModule")
        self.memory = Memory.from_config(config)
        self.user_id = user_id
        self.db_path = db_path
        
        # 创建线程池用于异步添加记忆
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.lock = threading.Lock()
        
        logger.info(f"Memory模块初始化完成，数据库路径: {self.db_path}")
    
    def add_memory(self, text: str, async_mode: bool = True) -> None:
        """
        添加记忆 (会自动持久化到本地数据库)
        
        Args:
            text: 要记忆的内容
            async_mode: 是否异步添加（默认True）
        """
        if async_mode:
            # 异步添加，立即返回
            self.executor.submit(self._add_memory_sync, text)
            logger.info(f"已提交记忆任务: {text[:50]}...")
        else:
            # 同步添加
            self._add_memory_sync(text)

    @staticmethod
    def is_stable_user_memory(text: str) -> bool:
        """判断用户输入是否像长期稳定信息，避免把普通闲聊写入长期记忆。"""
        clean_text = _strip_memory_prefix(text)
        if not clean_text:
            return False

        # 显式记忆请求优先通过，即使句子里包含“今天”等时间词。
        if re.search(r"(?:请)?(?:记住|记一下|帮我记住|以后记得)", clean_text):
            return True

        has_stable_signal = any(
            re.search(pattern, clean_text)
            for pattern in STABLE_MEMORY_PATTERNS
        )
        if not has_stable_signal:
            return False

        if any(re.search(pattern, clean_text) for pattern in TRANSIENT_MEMORY_PATTERNS):
            # “我现在住在上海”这类仍是稳定信息。
            if re.search(r"(?:住在|居住在|家在|老家在|工作|就读|任职)", clean_text):
                return True
            return False

        return True

    def add_user_memory_if_stable(self, user_text: str, async_mode: bool = True) -> bool:
        """只在用户输入包含稳定信息时写入长期记忆。"""
        clean_text = _strip_memory_prefix(user_text)
        if not self.is_stable_user_memory(clean_text):
            logger.info(f"跳过非稳定用户记忆: {clean_text[:50]}...")
            return False

        self.add_memory(f"用户稳定信息: {clean_text}", async_mode=async_mode)
        return True
    
    def _add_memory_sync(self, text: str) -> None:
        """
        同步添加记忆的内部方法
        
        Args:
            text: 要记忆的内容
        """
        try:
            with self.lock:
                self.memory.add(text, user_id=self.user_id)
            logger.info(f"已添加记忆: {text[:50]}...")
        except Exception as e:
            logger.error(f"添加记忆失败: {e}")
    
    def search_memory(self, query: str, limit: int = 3) -> str:
        """
        搜索相关记忆
        
        Args:
            query: 搜索查询
            limit: 返回结果数量
            
        Returns:
            str: 相关记忆的文本
        """
        try:
            with self.lock:
                results = self.memory.search(query, user_id=self.user_id, limit=limit)
            
            if not results:
                return ""

            if isinstance(results, dict) and "results" in results:
                results = results["results"]
            
            # 合并搜索结果
            context_parts = []
            for i, result in enumerate(results, 1):
                mem = None
                if isinstance(result, str):
                    mem = result
                elif isinstance(result, dict):
                    mem = result.get("memory") or result.get("text") or result.get("content") or result.get("value")
                else:
                    mem = getattr(result, "memory", None) or getattr(result, "text", None) or getattr(result, "content", None)
                
                if mem:
                    context_parts.append(f"{i}. {mem}")
            
            context = "\n".join(context_parts)
            logger.info(f"检索到 {len(context_parts)} 条相关记忆")
            return context
            
        except Exception as e:
            logger.error(f"搜索记忆失败: {e}")
            return ""
    
    def get_all_memories(self) -> list:
        """
        获取所有记忆
        
        Returns:
            list: 所有记忆列表
        """
        try:
            with self.lock:
                return self.memory.get_all(user_id=self.user_id)
        except Exception as e:
            logger.error(f"获取记忆失败: {e}")
            return []
    
    def shutdown(self) -> None:
        """
        关闭线程池，等待所有任务完成
        记忆会自动保存到本地数据库
        """
        logger.info(f"正在等待所有记忆任务完成...")
        logger.info(f"记忆数据已保存至: {self.db_path}")
        self.executor.shutdown(wait=True)
        logger.info("Memory模块已安全关闭")
