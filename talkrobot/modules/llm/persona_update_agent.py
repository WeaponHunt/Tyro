"""
人格更新 Agent
并行判断是否更新人格 prompt，并生成候选 prompt
"""
import json
import re
import time
from typing import Any, Dict, TypedDict

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


class PersonaUpdateState(TypedDict, total=False):
    user_input: str
    context: str
    current_prompt: str
    need_update: bool
    confidence: float
    decision_reason: str
    candidate_prompt: str
    candidate_reason: str
    decide_elapsed_s: float
    propose_elapsed_s: float


class PersonaUpdateAgent:
    """基于 LangGraph 的人格更新 Agent。"""

    def __init__(self, llm_client, model: str, cooldown_seconds: float = 20.0, min_confidence: float = 0.65):
        self.client = llm_client
        self.model = model
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self.min_confidence = float(min_confidence)
        self._last_update_ts_by_user: Dict[str, float] = {}
        self._graph = None
        self._langgraph_available = False
        self._build_graph_if_available()

    def _build_graph_if_available(self) -> None:
        try:
            from langgraph.graph import StateGraph, START, END
        except Exception as e:
            logger.warning(f"LangGraph 不可用，跳过人格更新 Agent: {e}")
            self._langgraph_available = False
            return

        graph = StateGraph(PersonaUpdateState)
        graph.add_node("decide_update", self._node_decide_update)
        graph.add_node("propose_prompt", self._node_propose_prompt)
        graph.add_node("merge", self._node_merge)

        graph.add_edge(START, "decide_update")
        graph.add_edge(START, "propose_prompt")
        graph.add_edge("decide_update", "merge")
        graph.add_edge("propose_prompt", "merge")
        graph.add_edge("merge", END)

        self._graph = graph.compile()
        self._langgraph_available = True
        logger.info("PersonaUpdateAgent 已启用 LangGraph 并行节点")

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        if not text:
            return {}

        raw = text.strip()
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
        except Exception:
            pass

        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            return {}

        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                return data
        except Exception:
            return {}
        return {}

    def _invoke_json_task(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        content = completion.choices[0].message.content or ""
        return self._extract_json(content)

    def _node_decide_update(self, state: PersonaUpdateState) -> PersonaUpdateState:
        start = time.perf_counter()
        system_prompt = (
            "你是一个严格的 JSON 决策器。"
            "请仅输出 JSON，不要输出其它文本。"
        )
        user_prompt = (
            "你作为一个对话机器人"
            "请判断用户输入是否明确提出了‘改变你说话风格/人格/语气/行为规则’的请求。\n"
            "若只是普通问答、闲聊、任务请求，need_update 应为 false。\n"
            "输出格式："
            "{\"need_update\": boolean, \"confidence\": number, \"reason\": string}\n"
            f"用户输入: {state.get('user_input', '')}\n"
            f"当前人格 prompt: {state.get('current_prompt', '')}"
        )

        try:
            data = self._invoke_json_task(system_prompt, user_prompt)
            print(f"decide_update data: {data}")
        except Exception as e:
            logger.warning(f"decide_update 调用失败: {e}")
            return {
                "need_update": False,
                "confidence": 0.0,
                "decision_reason": "llm_error",
                "decide_elapsed_s": time.perf_counter() - start,
            }

        return {
            "need_update": bool(data.get("need_update", False)),
            "confidence": float(data.get("confidence", 0.0) or 0.0),
            "decision_reason": str(data.get("reason", "") or "").strip(),
            "decide_elapsed_s": time.perf_counter() - start,
        }

    def _node_propose_prompt(self, state: PersonaUpdateState) -> PersonaUpdateState:
        start = time.perf_counter()
        system_prompt = (
            "你是一个人格 prompt 生成器。"
            "请仅输出 JSON，不要输出其它文本。"
        )
        context = state.get("context", "")
        if len(context) > 1200:
            context = context[:1200]

        user_prompt = (
            "请基于用户输入，生成一个可能的新人格 system_prompt。\n"
            "要求：中文，简洁，不超过220字，不包含模型供应商、API、密钥、系统路径等信息。\n"
            "如果用户输入的内容和当前人格的部分特质矛盾，删掉矛盾的特质并按用户的新需求生成，形成新的 prompt；"
            "输出格式："
            "{\"candidate_prompt\": string, \"reason\": string}\n"
            f"当前人格 prompt: {state.get('current_prompt', '')}\n"
            f"用户输入: {state.get('user_input', '')}\n"
            #f"辅助上下文: {context}"
        )

        try:
            data = self._invoke_json_task(system_prompt, user_prompt)
            print(f"propose_prompt data: {data}")
        except Exception as e:
            logger.warning(f"propose_prompt 调用失败: {e}")
            return {
                "candidate_prompt": "",
                "candidate_reason": "llm_error",
                "propose_elapsed_s": time.perf_counter() - start,
            }

        return {
            "candidate_prompt": str(data.get("candidate_prompt", "") or "").strip(),
            "candidate_reason": str(data.get("reason", "") or "").strip(),
            "propose_elapsed_s": time.perf_counter() - start,
        }

    def _node_merge(self, state: PersonaUpdateState) -> PersonaUpdateState:
        return state

    @staticmethod
    def _sanitize_prompt(prompt: str) -> str:
        text = (prompt or "").strip()
        if not text:
            return ""
        text = re.sub(r"\s+", " ", text)
        return text[:600]

    def run(self, user: str, user_input: str, context: str, current_prompt: str) -> Dict[str, Any]:
        """执行单轮人格更新推理（由外部线程调用）。"""
        total_start = time.perf_counter()

        def _with_timing(payload: Dict[str, Any], graph_elapsed_s: float = 0.0) -> Dict[str, Any]:
            payload["graph_elapsed_s"] = graph_elapsed_s
            payload["total_elapsed_s"] = time.perf_counter() - total_start
            return payload

        if not self._langgraph_available or self._graph is None:
            return _with_timing({"should_update": False, "reason": "langgraph_unavailable"})

        user_key = (user or "").strip()
        now = time.time()
        last_ts = self._last_update_ts_by_user.get(user_key)
        if last_ts is not None and (now - last_ts) < self.cooldown_seconds:
            return _with_timing({"should_update": False, "reason": "cooldown"})

        state: PersonaUpdateState = {
            "user_input": (user_input or "").strip(),
            "context": context or "",
            "current_prompt": (current_prompt or "").strip(),
        }
        if not state["user_input"]:
            return _with_timing({"should_update": False, "reason": "empty_input"})

        graph_start = time.perf_counter()
        try:
            result = self._graph.invoke(state)
        except Exception as e:
            logger.warning(f"人格更新 Agent 执行失败: {e}")
            return _with_timing({"should_update": False, "reason": "graph_error"})
        graph_elapsed = time.perf_counter() - graph_start

        need_update = bool(result.get("need_update", False))
        confidence = float(result.get("confidence", 0.0) or 0.0)
        candidate_prompt = self._sanitize_prompt(str(result.get("candidate_prompt", "") or ""))
        decision_reason = str(result.get("decision_reason", "") or "").strip()
        decide_elapsed_s = float(result.get("decide_elapsed_s", 0.0) or 0.0)
        propose_elapsed_s = float(result.get("propose_elapsed_s", 0.0) or 0.0)

        logger.debug(
            "人格更新Agent耗时: "
            f"user={user_key}, graph={graph_elapsed:.3f}s, "
            f"decide={decide_elapsed_s:.3f}s, propose={propose_elapsed_s:.3f}s"
        )

        if not need_update:
            return _with_timing(
                {
                    "should_update": False,
                    "reason": f"need_update_false:{decision_reason}",
                    "decide_elapsed_s": decide_elapsed_s,
                    "propose_elapsed_s": propose_elapsed_s,
                },
                graph_elapsed,
            )
        if confidence < self.min_confidence:
            return _with_timing(
                {
                    "should_update": False,
                    "reason": f"low_confidence:{confidence:.2f}",
                    "decide_elapsed_s": decide_elapsed_s,
                    "propose_elapsed_s": propose_elapsed_s,
                },
                graph_elapsed,
            )
        if not candidate_prompt:
            return _with_timing(
                {
                    "should_update": False,
                    "reason": "empty_candidate_prompt",
                    "decide_elapsed_s": decide_elapsed_s,
                    "propose_elapsed_s": propose_elapsed_s,
                },
                graph_elapsed,
            )
        if candidate_prompt == (current_prompt or "").strip():
            return _with_timing(
                {
                    "should_update": False,
                    "reason": "unchanged",
                    "decide_elapsed_s": decide_elapsed_s,
                    "propose_elapsed_s": propose_elapsed_s,
                },
                graph_elapsed,
            )

        self._last_update_ts_by_user[user_key] = now
        return _with_timing(
            {
                "should_update": True,
                "updated_prompt": candidate_prompt,
                "confidence": confidence,
                "decision_reason": decision_reason,
                "candidate_reason": str(result.get("candidate_reason", "") or "").strip(),
                "decide_elapsed_s": decide_elapsed_s,
                "propose_elapsed_s": propose_elapsed_s,
            },
            graph_elapsed,
        )
