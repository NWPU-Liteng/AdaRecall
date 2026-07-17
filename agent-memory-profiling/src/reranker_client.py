from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class RerankResult:
    memory: dict
    score: float
    raw_score: float


class HeuristicRerankerClient:
    def __init__(self):
        self.mode = "heuristic"

    def score_pairs(self, pairs: Sequence[Tuple[str, dict, float]]) -> List[float]:
        return [float(vector_score) for _, _, vector_score in pairs]


class CrossEncoderRerankerClient:
    _model = None
    _tokenizer = None
    _loaded_model_dir = None
    _device = None

    def __init__(
        self,
        model_dir: str,
        max_length: int = 512,
        batch_size: int = 8,
    ):
        self.model_dir = model_dir
        self.max_length = max_length
        self.batch_size = batch_size
        self.mode = "cross_encoder"
        self._ensure_loaded()

    @classmethod
    def _ensure_loaded_common(cls, model_dir: str):
        if cls._model is not None and cls._loaded_model_dir == model_dir:
            return

        cls._tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        if torch.cuda.is_available():
            device_index = os.getenv("LOCAL_QWEN_CUDA_DEVICE", "0")
            cls._device = torch.device(f"cuda:{device_index}")
        else:
            cls._device = torch.device("cpu")

        try:
            cls._model = AutoModelForSequenceClassification.from_pretrained(
                model_dir,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if cls._device.type == "cuda" else None,
            )
        except TypeError:
            cls._model = AutoModelForSequenceClassification.from_pretrained(
                model_dir,
                trust_remote_code=True,
            )

        cls._model.to(cls._device)
        cls._model.eval()
        cls._loaded_model_dir = model_dir

    def _ensure_loaded(self):
        self._ensure_loaded_common(self.model_dir)

    @staticmethod
    def _extract_scores(logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim == 1:
            return logits.float()
        if logits.shape[-1] == 1:
            return logits[:, 0].float()
        if logits.shape[-1] == 2:
            return logits[:, 1].float()
        return logits.max(dim=-1).values.float()

    def score_pairs(self, pairs: Sequence[Tuple[str, dict, float]]) -> List[float]:
        scores: List[float] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            encoded = self._tokenizer(
                [query for query, _, _ in batch],
                [memory.get("text", "") for _, memory, _ in batch],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self._device) for k, v in encoded.items()}
            with torch.no_grad():
                outputs = self._model(**encoded)
            batch_scores = self._extract_scores(outputs.logits).cpu().tolist()
            scores.extend(float(score) for score in batch_scores)
        return scores


class ExplicitUnavailableCrossEncoderClient:
    def __init__(self, reason: str):
        self.mode = "cross_encoder_unavailable"
        self.reason = reason

    def score_pairs(self, pairs: Sequence[Tuple[str, dict, float]]) -> List[float]:
        raise RuntimeError(self.reason)


def build_cross_encoder_model_dir(config: dict, project_root: str) -> str:
    configured = str(config.get("reranker_model_dir", "")).strip()
    if configured:
        return configured
    default_dir = os.path.join(project_root, "..", "dir")
    return default_dir


def resolve_reranker_client(config: dict, project_root: str):
    backend = str(config.get("reranker_backend", "heuristic")).strip()
    if backend == "heuristic":
        return HeuristicRerankerClient()

    if backend != "cross_encoder":
        raise ValueError(f"Unsupported reranker_backend: {backend}")

    model_dir = build_cross_encoder_model_dir(config, project_root)
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        return ExplicitUnavailableCrossEncoderClient(
            f"Cross-encoder config not found at: {config_path}"
        )

    try:
        import json

        with open(config_path, "r", encoding="utf-8") as handle:
            model_config = json.load(handle)
    except Exception as exc:
        return ExplicitUnavailableCrossEncoderClient(
            f"Unable to read reranker config at {config_path}: {exc}"
        )

    architectures = model_config.get("architectures") or []
    model_type = str(model_config.get("model_type", ""))
    is_sequence_classifier = any("SequenceClassification" in str(item) for item in architectures)
    if not is_sequence_classifier:
        return ExplicitUnavailableCrossEncoderClient(
            f"Configured reranker model at {model_dir} is not a sequence-classification model. "
            f"architectures={architectures!r}, model_type={model_type!r}"
        )

    batch_size = int(config.get("reranker_batch_size", 8))
    max_length = int(config.get("reranker_max_length", 512))
    return CrossEncoderRerankerClient(
        model_dir=model_dir,
        batch_size=batch_size,
        max_length=max_length,
    )
