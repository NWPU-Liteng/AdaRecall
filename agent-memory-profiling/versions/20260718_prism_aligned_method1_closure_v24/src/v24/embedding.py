from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


def resolve_runtime_device(requested: str = "auto") -> str:
    if requested not in {"", "auto"}:
        return requested
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def last_token_pool(last_hidden_state, attention_mask):
    """Pool the last non-padding token as required by Qwen3-Embedding."""
    import torch

    left_padded = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
    if left_padded:
        return last_hidden_state[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch = last_hidden_state.shape[0]
    return last_hidden_state[torch.arange(batch, device=last_hidden_state.device), sequence_lengths]


@dataclass
class EmbeddingRuntime:
    model_dir: str
    requested_device: str
    actual_device: str
    dtype: str
    max_length: int
    instruction: str
    dimension: int


class Qwen3Embedding:
    def __init__(
        self,
        model_dir: str | Path,
        *,
        device: str = "auto",
        dtype: str = "bfloat16",
        max_length: int = 512,
        instruction: str = "Given a question about a user's history, retrieve memory records that provide the evidence needed to answer it.",
        batch_size: int = 16,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.model_dir = Path(model_dir)
        self.requested_device = str(device)
        self.device = resolve_runtime_device(device)
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            self.device = "cpu"
        self.max_length = int(max_length)
        self.instruction = str(instruction)
        self.batch_size = int(batch_size)
        self.dtype_name = str(dtype)
        model_dtype = getattr(torch, dtype, torch.float32) if self.device.startswith("cuda") else torch.float32
        self.attention_implementation = "sdpa"
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir), local_files_only=True, trust_remote_code=True, padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        try:
            self.model = AutoModel.from_pretrained(
                str(self.model_dir), local_files_only=True, trust_remote_code=True,
                torch_dtype=model_dtype, attn_implementation="sdpa",
            )
        except ImportError as exc:
            # Transformers 4.51 rejects SDPA with torch 2.1.0 even though the
            # model itself is usable. Keep the environment fixed and record the
            # explicit eager fallback instead of upgrading torch/CUDA.
            if "SDPA" not in str(exc) and "torch>=2.1.1" not in str(exc):
                raise
            self.attention_implementation = "eager"
            self.model = AutoModel.from_pretrained(
                str(self.model_dir), local_files_only=True, trust_remote_code=True,
                torch_dtype=model_dtype, attn_implementation="eager",
            )
        except TypeError:
            self.attention_implementation = "default"
            self.model = AutoModel.from_pretrained(
                str(self.model_dir), local_files_only=True, trust_remote_code=True, torch_dtype=model_dtype
            )
        self.model.to(self.device)
        self.model.eval()
        dimension = int(getattr(self.model.config, "hidden_size", 0) or getattr(self.model.config, "d_model", 0))
        self.runtime = EmbeddingRuntime(
            str(self.model_dir), self.requested_device, self.device, str(model_dtype), self.max_length,
            self.instruction, dimension,
        )

    def _query_text(self, text: str) -> str:
        return f"Instruct: {self.instruction}\nQuery: {text}"

    def encode(self, texts: Iterable[str], *, is_query: bool = False) -> np.ndarray:
        values = [str(text) for text in texts]
        if not values:
            dimension = self.runtime.dimension or 0
            return np.empty((0, dimension), dtype=np.float32)
        output = []
        for start in range(0, len(values), self.batch_size):
            batch = values[start:start + self.batch_size]
            if is_query:
                batch = [self._query_text(text) for text in batch]
            encoded = self.tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length,
            )
            mask = encoded["attention_mask"]
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self.torch.inference_mode():
                hidden = self.model(**encoded).last_hidden_state
                pooled = last_token_pool(hidden, encoded["attention_mask"])
                pooled = self.torch.nn.functional.normalize(pooled.float(), p=2, dim=-1)
            output.append(pooled.cpu().numpy().astype(np.float32, copy=False))
            del encoded, hidden, pooled, mask
        return np.vstack(output).astype(np.float32, copy=False)

    def encode_queries(self, texts: Iterable[str]) -> np.ndarray:
        return self.encode(texts, is_query=True)

    def encode_documents(self, texts: Iterable[str]) -> np.ndarray:
        return self.encode(texts, is_query=False)

    def smoke(self) -> dict:
        query = self.encode_queries(["Which medication was prescribed for blood pressure?"])[0]
        positive = self.encode_documents(["The patient was prescribed valsartan for hypertension."])[0]
        negative = self.encode_documents(["The patient discussed a holiday itinerary and train tickets."])[0]
        values = {
            "runtime": self.runtime.__dict__,
            "attention_implementation": self.attention_implementation,
            "query_norm": float(np.linalg.norm(query)),
            "positive_similarity": float(query @ positive),
            "negative_similarity": float(query @ negative),
            "finite": bool(np.isfinite(query).all() and np.isfinite(positive).all() and np.isfinite(negative).all()),
            "dimension": int(query.shape[0]),
        }
        values["passed"] = bool(
            values["finite"] and values["dimension"] == 1024 and abs(values["query_norm"] - 1.0) < 2e-3
            and values["positive_similarity"] > values["negative_similarity"]
        )
        return values
