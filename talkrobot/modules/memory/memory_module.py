"""
记忆管理模块
负责存储和检索对话历史
"""
import os
from mem0 import Memory
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
import threading

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