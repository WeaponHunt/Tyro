"""
记忆管理模块
负责存储和检索对话历史
"""
import os
import re
from mem0 import Memory
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
import threading
from typing import Optional

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
    
    def add_memory(self, text: str, async_mode: bool = True, infer: bool = True) -> None:
        """
        添加记忆 (会自动持久化到本地数据库)
        
        Args:
            text: 要记忆的内容
            async_mode: 是否异步添加（默认True）
            infer: 是否启用LLM事实抽取（默认True）
        """
        if async_mode:
            # 异步添加，立即返回
            self.executor.submit(self._add_memory_sync, text, infer)
            logger.info(f"已提交记忆任务: {text[:50]}...")
        else:
            # 同步添加
            self._add_memory_sync(text, infer)
    
    def _add_memory_sync(self, text: str, infer: bool = True) -> None:
        """
        同步添加记忆的内部方法
        
        Args:
            text: 要记忆的内容
        """
        try:
            with self.lock:
                result = self.memory.add(text, user_id=self.user_id, infer=infer)

            if not infer:
                if self._has_add_results(result):
                    logger.info(f"已添加记忆(原文直存): {text[:50]}...")
                else:
                    logger.warning(f"原文直存返回空结果: {text[:80]}...")
                return

            if self._has_add_results(result):
                logger.info(f"已添加记忆: {text[:50]}...")
                return

            # 对较长/混合语言文本，mem0 可能提取不到事实；按句拆分重试一次。
            saved_count = self._retry_add_by_sentence(text)
            if saved_count > 0:
                logger.info(f"原文提取为空，拆句重试后已添加 {saved_count} 条记忆")
            else:
                logger.warning(
                    f"未提取到可存储记忆，已跳过。建议改为短句事实表达: {text[:80]}..."
                )
        except Exception as e:
            logger.error(f"添加记忆失败: {e}")

    @staticmethod
    def _has_add_results(result) -> bool:
        if not result:
            return False
        if isinstance(result, dict):
            results = result.get("results")
            return bool(results)
        if isinstance(result, list):
            return bool(result)
        return True

    @staticmethod
    def _split_text_for_retry(text: str) -> list:
        pieces = re.split(r"[。！？；;\n]+", str(text or ""))
        cleaned = []
        for piece in pieces:
            p = piece.strip(" ，,。.！!？?；;")
            if len(p) >= 4:
                cleaned.append(p)
        return cleaned

    def _retry_add_by_sentence(self, text: str) -> int:
        saved = 0
        for piece in self._split_text_for_retry(text):
            with self.lock:
                result = self.memory.add(piece, user_id=self.user_id)
            if self._has_add_results(result):
                saved += 1
        return saved
    
    @staticmethod
    def _to_float(value) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _extract_score_distance(cls, result):
        score = None
        distance = None
        if isinstance(result, dict):
            score = (
                cls._to_float(result.get("score"))
                or cls._to_float(result.get("similarity"))
                or cls._to_float(result.get("relevance"))
            )
            distance = (
                cls._to_float(result.get("distance"))
                or cls._to_float(result.get("dist"))
            )
        else:
            score = (
                cls._to_float(getattr(result, "score", None))
                or cls._to_float(getattr(result, "similarity", None))
                or cls._to_float(getattr(result, "relevance", None))
            )
            distance = (
                cls._to_float(getattr(result, "distance", None))
                or cls._to_float(getattr(result, "dist", None))
            )
        return score, distance

    def search_memory(
        self,
        query: str,
        limit: int = 3,
        min_score: Optional[float] = None,
        max_distance: Optional[float] = None,
    ) -> str:
        """
        搜索相关记忆
        
        Args:
            query: 搜索查询
            limit: 返回结果数量
            min_score: 最低相似度分数阈值（None 表示不按分数过滤）
            max_distance: 最大距离阈值（None 表示不按距离过滤）
            
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
                score, distance = self._extract_score_distance(result)
                if min_score is not None and score is not None and score < min_score:
                    continue
                if max_distance is not None and distance is not None and distance > max_distance:
                    continue

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