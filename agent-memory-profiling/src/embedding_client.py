import hashlib
import os
from typing import Iterable, List

import numpy as np
import requests
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from .runtime_compat import load_local_qwen_causal_lm, validate_local_qwen_runtime


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec.astype(np.float32, copy=False)
    return (vec / norm).astype(np.float32, copy=False)


class HashEmbeddingClient:
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.mode = "hash_embedding"

    def _token_to_index(self, token: str) -> int:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "little") % self.dim

    def _token_to_sign(self, token: str) -> float:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        return 1.0 if digest[0] % 2 == 0 else -1.0

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for raw in text.lower().split():
            token = raw.strip(".,!?;:()[]{}\"'")
            if not token:
                continue
            idx = self._token_to_index(token)
            vec[idx] += self._token_to_sign(token)
        return l2_normalize(vec)

    def embed_batch(self, texts: Iterable[str]) -> np.ndarray:
        return np.vstack([self.embed(text) for text in texts]).astype(np.float32)


class OllamaEmbeddingClient:
    def __init__(self, model: str = "", dim: int = 384, timeout: float = 15.0):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        self.model = model or os.getenv("OLLAMA_EMBED_MODEL", "")
        self.dim = dim
        self.timeout = timeout
        self.fallback = HashEmbeddingClient(dim=dim)
        self.mode = "ollama_embedding" if self.model else self.fallback.mode

    def _request_embedding(self, text: str) -> np.ndarray:
        if not self.model:
            raise RuntimeError("No Ollama embedding model configured")
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        embedding = np.asarray(data["embedding"], dtype=np.float32)
        if embedding.size == 0:
            raise RuntimeError("Empty embedding from Ollama")
        if embedding.size != self.dim:
            if embedding.size > self.dim:
                embedding = embedding[: self.dim]
            else:
                embedding = np.pad(embedding, (0, self.dim - embedding.size))
        return l2_normalize(embedding)

    def embed(self, text: str) -> np.ndarray:
        try:
            vec = self._request_embedding(text)
            self.mode = "ollama_embedding"
            return vec
        except Exception:
            self.mode = self.fallback.mode
            return self.fallback.embed(text)

    def embed_batch(self, texts: Iterable[str]) -> np.ndarray:
        vectors: List[np.ndarray] = []
        for text in texts:
            vectors.append(self.embed(text))
        return np.vstack(vectors).astype(np.float32)


class LocalQwenEmbeddingClient:
    _model = None
    _tokenizer = None
    _device = None
    _loaded_model_dir = None

    def __init__(self, model_dir: str = "", dim: int = 384, max_length: int = 512, batch_size: int = 8):
        self.model_dir = model_dir or os.getenv("LOCAL_QWEN_DIR", "../dir")
        self.dim = dim
        self.max_length = max_length
        self.batch_size = batch_size
        self.mode = "local_qwen_embedding"
        self._ensure_loaded()

    @classmethod
    def _ensure_loaded_common(cls, model_dir: str):
        if cls._model is not None and cls._loaded_model_dir == model_dir:
            return
        validate_local_qwen_runtime()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; local Qwen embedding requires GPU")
        device_index = os.getenv("LOCAL_QWEN_CUDA_DEVICE", "0")
        cls._device = torch.device(f"cuda:{device_index}")
        cls._tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        cls._model = load_local_qwen_causal_lm(model_dir).to(cls._device)
        cls._model.eval()
        cls._loaded_model_dir = model_dir

    def _ensure_loaded(self):
        self._ensure_loaded_common(self.model_dir)

    def _project_hidden(self, hidden: np.ndarray) -> np.ndarray:
        if hidden.size >= self.dim:
            return hidden[: self.dim].astype(np.float32, copy=False)
        return np.pad(hidden.astype(np.float32, copy=False), (0, self.dim - hidden.size))

    def embed(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: Iterable[str]) -> np.ndarray:
        text_list = list(texts)
        vectors: List[np.ndarray] = []
        for start in range(0, len(text_list), self.batch_size):
            batch = text_list[start : start + self.batch_size]
            encoded = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            attention_mask = encoded["attention_mask"]
            encoded = {k: v.to(self._device) for k, v in encoded.items()}
            with torch.no_grad():
                outputs = self._model(**encoded, output_hidden_states=True, use_cache=False)
                last_hidden = outputs.hidden_states[-1].float().cpu()
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            for row in pooled.numpy():
                vectors.append(l2_normalize(self._project_hidden(row)))
        return np.vstack(vectors).astype(np.float32)


class TransformerEmbeddingClient:
    _model = None
    _tokenizer = None
    _device = None
    _loaded_model_dir = None

    def __init__(self, model_dir: str, dim: int = 768, max_length: int = 512, batch_size: int = 16):
        self.model_dir = model_dir
        self.dim = dim
        self.max_length = max_length
        self.batch_size = batch_size
        self.mode = "transformer_embedding"
        self._ensure_loaded()

    @classmethod
    def _ensure_loaded_common(cls, model_dir: str):
        if cls._model is not None and cls._loaded_model_dir == model_dir:
            return
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; transformer embedding requires GPU")
        device_index = os.getenv("LOCAL_QWEN_CUDA_DEVICE", "0")
        cls._device = torch.device(f"cuda:{device_index}")
        cls._tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        cls._model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).to(cls._device)
        cls._model.eval()
        cls._loaded_model_dir = model_dir

    def _ensure_loaded(self):
        self._ensure_loaded_common(self.model_dir)

    def _project_hidden(self, hidden: np.ndarray) -> np.ndarray:
        if hidden.size >= self.dim:
            return hidden[: self.dim].astype(np.float32, copy=False)
        return np.pad(hidden.astype(np.float32, copy=False), (0, self.dim - hidden.size))

    def embed(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: Iterable[str]) -> np.ndarray:
        text_list = list(texts)
        vectors: List[np.ndarray] = []
        for start in range(0, len(text_list), self.batch_size):
            batch = text_list[start : start + self.batch_size]
            encoded = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            attention_mask = encoded["attention_mask"]
            encoded = {k: v.to(self._device) for k, v in encoded.items()}
            with torch.no_grad():
                outputs = self._model(**encoded)
                last_hidden = outputs.last_hidden_state.float().cpu()
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            for row in pooled.numpy():
                vectors.append(l2_normalize(self._project_hidden(row)))
        return np.vstack(vectors).astype(np.float32)
