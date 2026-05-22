"""
人格更新 Agent
先用本地情绪模型做门控，再用 LLM 反思是否更新人格 prompt。
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
    reflection_elapsed_s: float
    sentiment_label: int
    sentiment_negative: float
    sentiment_positive: float


class SentimentGate:
    """Lazy local sentiment classifier used before expensive persona reflection."""

    def __init__(
        self,
        model_name: str = "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment",
        threshold: float = 0.82,
    ):
        self.model_name = model_name
        self.threshold = float(threshold)
        self._tokenizer = None
        self._model = None
        self._load_error = ""

    @property
    def load_error(self) -> str:
        return self._load_error

    def _ensure_loaded(self) -> bool:
        if self._tokenizer is not None and self._model is not None:
            return True
        if self._load_error:
            return False

        try:
            import torch
            import torch.nn.functional as F  # noqa: F401
            from transformers import BertForSequenceClassification, BertTokenizer

            self._torch = torch
            self._softmax = F.softmax
            self._tokenizer = BertTokenizer.from_pretrained(self.model_name)
            self._model = BertForSequenceClassification.from_pretrained(self.model_name)
            self._model.eval()
            logger.info(f"情绪门控模型已加载: {self.model_name}")
            return True
        except Exception as e:
            self._load_error = str(e)
            logger.warning(f"情绪门控模型加载失败，跳过人格自动更新: {e}")
            return False

    def score(self, text: str) -> Dict[str, Any]:
        if not self._ensure_loaded():
            return {
                "should_reflect": False,
                "reason": "sentiment_model_unavailable",
                "load_error": self._load_error,
            }

        clean_text = (text or "").strip()
        if not clean_text:
            return {"should_reflect": False, "reason": "empty_input"}

        try:
            encoded = self._tokenizer.encode(
                clean_text,
                truncation=True,
                max_length=256,
            )
            with self._torch.no_grad():
                output = self._model(self._torch.tensor([encoded]))
                probs = self._softmax(output.logits, dim=1)[0]
            negative = float(probs[0].item())
            positive = float(probs[1].item())
            label = 1 if positive >= negative else 0
            should_reflect = max(negative, positive) >= self.threshold
            return {
                "should_reflect": should_reflect,
                "label": label,
                "negative": negative,
                "positive": positive,
                "threshold": self.threshold,
                "reason": "above_threshold" if should_reflect else "below_threshold",
            }
        except Exception as e:
            logger.warning(f"情绪门控推理失败，跳过人格自动更新: {e}")
            return {"should_reflect": False, "reason": "sentiment_infer_error"}


class PersonaUpdateAgent:
    """人格更新 Agent。"""

    def __init__(
        self,
        llm_client,
        model: str,
        cooldown_seconds: float = 20.0,
        min_confidence: float = 0.65,
        sentiment_threshold: float = 0.82,
    ):
        self.client = llm_client
        self.model = model
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self.min_confidence = float(min_confidence)
        self.sentiment_gate = SentimentGate(threshold=sentiment_threshold)
        self._last_update_ts_by_user: Dict[str, float] = {}

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

    def _reflect_update(self, state: PersonaUpdateState) -> PersonaUpdateState:
        start = time.perf_counter()
        system_prompt = (
            "你是一个严格的人格更新反思器。"
            "请仅输出 JSON，不要输出其它文本。"
        )
        user_prompt = (
            "请判断用户输入是否明确提出了‘改变你说话风格/人格/语气/行为规则’的请求，"
            "并在确实需要更新时生成新人格 system_prompt。\n"
            "若只是普通问答、闲聊、任务请求，need_update 应为 false。\n"
            "若需要更新，candidate_prompt 要中文、简洁、不超过220字，"
            "不包含模型供应商、API、密钥、系统路径等信息。"
            "如果用户输入的内容和当前人格的部分特质矛盾，删掉矛盾的特质并按用户的新需求生成。\n"
            "输出格式："
            "{\"need_update\": boolean, \"confidence\": number, "
            "\"candidate_prompt\": string, \"reason\": string}\n"
            f"用户输入: {state.get('user_input', '')}\n"
            f"当前人格 prompt: {state.get('current_prompt', '')}"
        )

        try:
            data = self._invoke_json_task(system_prompt, user_prompt)
        except Exception as e:
            logger.warning(f"人格更新反思调用失败: {e}")
            return {
                "need_update": False,
                "confidence": 0.0,
                "decision_reason": "llm_error",
                "candidate_prompt": "",
                "reflection_elapsed_s": time.perf_counter() - start,
            }

        return {
            "need_update": bool(data.get("need_update", False)),
            "confidence": float(data.get("confidence", 0.0) or 0.0),
            "candidate_prompt": str(data.get("candidate_prompt", "") or "").strip(),
            "decision_reason": str(data.get("reason", "") or "").strip(),
            "reflection_elapsed_s": time.perf_counter() - start,
        }

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

        sentiment_start = time.perf_counter()
        sentiment = self.sentiment_gate.score(state["user_input"])
        sentiment_elapsed = time.perf_counter() - sentiment_start
        if not sentiment.get("should_reflect", False):
            return _with_timing(
                {
                    "should_update": False,
                    "reason": sentiment.get("reason", "sentiment_gate_skip"),
                    "sentiment_elapsed_s": sentiment_elapsed,
                    "sentiment_positive": float(sentiment.get("positive", 0.0) or 0.0),
                    "sentiment_negative": float(sentiment.get("negative", 0.0) or 0.0),
                    "sentiment_label": int(sentiment.get("label", -1)),
                }
            )

        reflection_start = time.perf_counter()
        reflection = self._reflect_update(state)
        reflection_elapsed = time.perf_counter() - reflection_start
        need_update = bool(reflection.get("need_update", False))
        confidence = float(reflection.get("confidence", 0.0) or 0.0)
        decision_reason = str(reflection.get("decision_reason", "") or "").strip()
        candidate_prompt = self._sanitize_prompt(str(reflection.get("candidate_prompt", "") or ""))
        decide_elapsed_s = reflection_elapsed
        propose_elapsed_s = 0.0

        logger.debug(
            "人格更新Agent耗时: "
            f"user={user_key}, sentiment={sentiment_elapsed:.3f}s, "
            f"reflection={reflection_elapsed:.3f}s, decide={decide_elapsed_s:.3f}s, "
            f"propose={propose_elapsed_s:.3f}s"
        )

        if not need_update:
            return _with_timing(
                {
                    "should_update": False,
                    "reason": f"need_update_false:{decision_reason}",
                    "decide_elapsed_s": decide_elapsed_s,
                    "propose_elapsed_s": propose_elapsed_s,
                    "reflection_elapsed_s": reflection_elapsed,
                    "sentiment_elapsed_s": sentiment_elapsed,
                    "sentiment_positive": float(sentiment.get("positive", 0.0) or 0.0),
                    "sentiment_negative": float(sentiment.get("negative", 0.0) or 0.0),
                    "sentiment_label": int(sentiment.get("label", -1)),
                },
            )
        if confidence < self.min_confidence:
            return _with_timing(
                {
                    "should_update": False,
                    "reason": f"low_confidence:{confidence:.2f}",
                    "decide_elapsed_s": decide_elapsed_s,
                    "propose_elapsed_s": propose_elapsed_s,
                    "reflection_elapsed_s": reflection_elapsed,
                    "sentiment_elapsed_s": sentiment_elapsed,
                    "sentiment_positive": float(sentiment.get("positive", 0.0) or 0.0),
                    "sentiment_negative": float(sentiment.get("negative", 0.0) or 0.0),
                    "sentiment_label": int(sentiment.get("label", -1)),
                },
            )
        if not candidate_prompt:
            return _with_timing(
                {
                    "should_update": False,
                    "reason": "empty_candidate_prompt",
                    "decide_elapsed_s": decide_elapsed_s,
                    "propose_elapsed_s": propose_elapsed_s,
                    "reflection_elapsed_s": reflection_elapsed,
                    "sentiment_elapsed_s": sentiment_elapsed,
                    "sentiment_positive": float(sentiment.get("positive", 0.0) or 0.0),
                    "sentiment_negative": float(sentiment.get("negative", 0.0) or 0.0),
                    "sentiment_label": int(sentiment.get("label", -1)),
                },
            )
        if candidate_prompt == (current_prompt or "").strip():
            return _with_timing(
                {
                    "should_update": False,
                    "reason": "unchanged",
                    "decide_elapsed_s": decide_elapsed_s,
                    "propose_elapsed_s": propose_elapsed_s,
                    "reflection_elapsed_s": reflection_elapsed,
                    "sentiment_elapsed_s": sentiment_elapsed,
                    "sentiment_positive": float(sentiment.get("positive", 0.0) or 0.0),
                    "sentiment_negative": float(sentiment.get("negative", 0.0) or 0.0),
                    "sentiment_label": int(sentiment.get("label", -1)),
                },
            )

        self._last_update_ts_by_user[user_key] = now
        return _with_timing(
            {
                "should_update": True,
                "updated_prompt": candidate_prompt,
                "confidence": confidence,
                "decision_reason": decision_reason,
                "candidate_reason": decision_reason,
                "decide_elapsed_s": decide_elapsed_s,
                "propose_elapsed_s": propose_elapsed_s,
                "reflection_elapsed_s": reflection_elapsed,
                "sentiment_elapsed_s": sentiment_elapsed,
                "sentiment_positive": float(sentiment.get("positive", 0.0) or 0.0),
                "sentiment_negative": float(sentiment.get("negative", 0.0) or 0.0),
                "sentiment_label": int(sentiment.get("label", -1)),
            },
        )
