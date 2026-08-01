"""
End-of-turn detector used by duplex voice mode.
"""
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger


DEFAULT_REPO_ID = os.getenv("TALKROBOT_EOT_REPO", "FireRedTeam/FireRedChat-turn-detector")
DEFAULT_MODEL_FILE = os.getenv("TALKROBOT_EOT_MODEL_FILE", "chinese_best_model_q8.onnx")
DEFAULT_TOKENIZER = os.getenv("TALKROBOT_EOT_TOKENIZER", "google-bert/bert-base-multilingual-cased")
DEFAULT_THRESHOLD = float(os.getenv("TALKROBOT_EOT_THRESHOLD", "0.7"))
DEFAULT_MAX_LENGTH = int(os.getenv("TALKROBOT_EOT_MAX_LENGTH", "128"))
LOCAL_RESOURCE_DIR = Path(__file__).resolve().parents[2] / "resources" / "eot"
TOKENIZER_FILES = (
    "config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
)


@dataclass(frozen=True)
class EOTPrediction:
    text: str
    probability: float
    is_end: bool


def fix_unsupported_socks_proxy() -> None:
    has_http_proxy = any(
        os.environ.get(key)
        for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy")
    )
    if not has_http_proxy:
        return

    for key in ("ALL_PROXY", "all_proxy"):
        value = os.environ.get(key, "")
        if value.lower().startswith("socks://"):
            os.environ.pop(key, None)


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


def _hub_resolve_url(repo_id: str, filename: str) -> str:
    endpoint = os.getenv("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    return f"{endpoint}/{repo_id}/resolve/main/{filename}"


def _download_url(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    logger.info(f"下载: {url} -> {output_path}")
    urllib.request.urlretrieve(url, tmp_path)
    tmp_path.replace(output_path)


def _download_direct_fallback(repo_id: str, model_file: str, tokenizer_name: str) -> tuple[Path, Path]:
    model_path = LOCAL_RESOURCE_DIR / model_file
    tokenizer_dir = LOCAL_RESOURCE_DIR / "tokenizer"

    if not model_path.exists():
        _download_url(_hub_resolve_url(repo_id, model_file), model_path)

    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    for filename in TOKENIZER_FILES:
        target = tokenizer_dir / filename
        if target.exists():
            continue
        try:
            _download_url(_hub_resolve_url(tokenizer_name, filename), target)
        except Exception as exc:
            if filename in {"tokenizer.json", "special_tokens_map.json"}:
                logger.warning(f"可选 tokenizer 文件下载失败，已跳过: {filename}: {exc}")
                continue
            raise

    return model_path, tokenizer_dir


def download_eot_resources(
    repo_id: str = DEFAULT_REPO_ID,
    model_file: str = DEFAULT_MODEL_FILE,
    tokenizer_name: str = DEFAULT_TOKENIZER,
) -> None:
    """Download EOT ONNX model and tokenizer into the Hugging Face cache."""
    fix_unsupported_socks_proxy()
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    from transformers import BertTokenizer

    local_model_path = LOCAL_RESOURCE_DIR / model_file
    local_tokenizer_dir = LOCAL_RESOURCE_DIR / "tokenizer"
    if local_model_path.exists() and (local_tokenizer_dir / "vocab.txt").exists():
        BertTokenizer.from_pretrained(str(local_tokenizer_dir), truncation_side="left")
        logger.info(f"EOT 本地资源已存在: {local_model_path}")
        return

    try:
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(repo_id=repo_id, filename=model_file)
        BertTokenizer.from_pretrained(tokenizer_name, truncation_side="left")
        logger.info(f"EOT 模型已缓存: {model_path}")
        return
    except Exception as exc:
        logger.warning(f"huggingface_hub 下载失败，改用直接 URL 下载: {exc}")

    model_path, tokenizer_dir = _download_direct_fallback(repo_id, model_file, tokenizer_name)
    BertTokenizer.from_pretrained(str(tokenizer_dir), truncation_side="left")
    logger.info(f"EOT 模型已缓存: {model_path}")


class EOTModule:
    """FireRedChat end-of-turn ONNX wrapper."""

    def __init__(
        self,
        repo_id: str = DEFAULT_REPO_ID,
        model_file: str = DEFAULT_MODEL_FILE,
        tokenizer_name: str = DEFAULT_TOKENIZER,
        threshold: float = DEFAULT_THRESHOLD,
        max_length: int = DEFAULT_MAX_LENGTH,
        allow_download: bool = False,
    ) -> None:
        fix_unsupported_socks_proxy()
        if allow_download:
            os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

        import onnxruntime as ort
        from transformers import BertTokenizer

        self.threshold = float(threshold)
        self.max_length = int(max_length)
        logger.info(
            f"正在加载 EOT 模型: repo={repo_id}, file={model_file}, threshold={self.threshold}"
        )
        local_model_path = LOCAL_RESOURCE_DIR / model_file
        local_tokenizer_dir = LOCAL_RESOURCE_DIR / "tokenizer"
        if local_model_path.exists() and (local_tokenizer_dir / "vocab.txt").exists():
            model_path = str(local_model_path)
            tokenizer_source = str(local_tokenizer_dir)
        else:
            from huggingface_hub import hf_hub_download

            try:
                model_path = hf_hub_download(repo_id=repo_id, filename=model_file)
                tokenizer_source = tokenizer_name
            except Exception as exc:
                if not allow_download:
                    raise RuntimeError(
                        "EOT 模型未缓存且下载失败。请先运行: "
                        ".venv/bin/python -m talkrobot.main setup-duplex"
                    ) from exc
                model_path_obj, tokenizer_dir = _download_direct_fallback(
                    repo_id,
                    model_file,
                    tokenizer_name,
                )
                model_path = str(model_path_obj)
                tokenizer_source = str(tokenizer_dir)

        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_source, truncation_side="left")
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        logger.info("EOT 模型加载完成")

    def predict_probability(self, text: str) -> float:
        clean_text = str(text or "").strip()
        if not clean_text:
            return 0.0

        inputs = self.tokenizer(
            clean_text,
            truncation=True,
            padding="max_length",
            add_special_tokens=False,
            return_tensors="np",
            max_length=self.max_length,
        )
        ort_inputs = {
            item.name: inputs[item.name].astype(np.int64)
            for item in self.session.get_inputs()
            if item.name in inputs
        }
        outputs = self.session.run(None, ort_inputs)
        probabilities = softmax(outputs[0]).flatten()
        return float(probabilities[-1])

    def predict(self, text: str) -> EOTPrediction:
        probability = self.predict_probability(text)
        return EOTPrediction(
            text=text,
            probability=probability,
            is_end=probability >= self.threshold,
        )
