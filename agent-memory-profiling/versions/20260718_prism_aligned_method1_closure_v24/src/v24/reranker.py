from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .embedding import resolve_runtime_device


@dataclass
class RerankOutput:
    scores: list[float]
    input_tokens: int
    device: str


class Qwen3Reranker:
    def __init__(
        self,
        model_dir: str | Path,
        *,
        device: str = "auto",
        dtype: str = "bfloat16",
        max_length: int = 512,
        batch_size: int = 8,
        instruction: str = "Given a question about a user's history, retrieve memory records that provide the evidence needed to answer it.",
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_dir = Path(model_dir)
        self.requested_device = str(device)
        self.device = resolve_runtime_device(device)
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            self.device = "cpu"
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.instruction = str(instruction)
        self.attention_implementation = "sdpa"
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir), local_files_only=True, trust_remote_code=True, padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model_dtype = getattr(torch, dtype, torch.float32) if self.device.startswith("cuda") else torch.float32
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_dir), local_files_only=True, trust_remote_code=True,
                torch_dtype=model_dtype, attn_implementation="sdpa",
            )
        except ImportError as exc:
            if "SDPA" not in str(exc) and "torch>=2.1.1" not in str(exc):
                raise
            self.attention_implementation = "eager"
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_dir), local_files_only=True, trust_remote_code=True,
                torch_dtype=model_dtype, attn_implementation="eager",
            )
        except TypeError:
            self.attention_implementation = "default"
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_dir), local_files_only=True, trust_remote_code=True, torch_dtype=model_dtype
            )
        self.model.to(self.device)
        self.model.eval()
        self.yes_id = int(self.tokenizer.convert_tokens_to_ids("yes"))
        self.no_id = int(self.tokenizer.convert_tokens_to_ids("no"))
        if self.yes_id == self.no_id or self.yes_id < 0 or self.no_id < 0:
            raise RuntimeError(f"invalid yes/no token IDs: {self.yes_id}, {self.no_id}")

    def _format(self, query: str, document: str) -> str:
        user = f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {document}"
        messages = [
            {"role": "system", "content": "Judge whether the Document meets the requirements based on the Query and Instruct. Answer only yes or no."},
            {"role": "user", "content": user},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def score_pairs(self, query: str, documents: Sequence[str]) -> RerankOutput:
        if not documents:
            return RerankOutput([], 0, self.device)
        prompts = [self._format(str(query), str(document)) for document in documents]
        scores: list[float] = []
        token_count = 0
        for start in range(0, len(prompts), self.batch_size):
            encoded = self.tokenizer(
                prompts[start:start + self.batch_size], return_tensors="pt", padding=True,
                truncation=True, max_length=self.max_length,
            )
            token_count += int(encoded["attention_mask"].sum().item())
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self.torch.inference_mode():
                logits = self.model(**encoded).logits[:, -1, :]
                pair_logits = logits[:, [self.yes_id, self.no_id]].float()
                batch_scores = self.torch.softmax(pair_logits, dim=-1)[:, 0]
            scores.extend(float(value) for value in batch_scores.detach().cpu().tolist())
            del encoded, logits, pair_logits, batch_scores
        return RerankOutput(scores, token_count, self.device)

    def smoke(self) -> dict[str, Any]:
        output = self.score_pairs(
            "Which medication treats hypertension?",
            ["Valsartan was prescribed for hypertension.", "The patient booked a holiday train."],
        )
        return {
            "device": output.device,
            "attention_implementation": self.attention_implementation,
            "yes_token_id": self.yes_id,
            "no_token_id": self.no_id,
            "scores": output.scores,
            "input_tokens": output.input_tokens,
            "passed": bool(len(output.scores) == 2 and output.scores[0] > output.scores[1]),
        }


class DenseReferenceReranker:
    """Deterministic fallback used only for CPU contract/smoke tests."""

    device = "none"

    def score_pairs(self, query: str, documents: Sequence[str], *, scores: Sequence[float] | None = None) -> RerankOutput:
        values = [float(value) for value in (scores or [0.0] * len(documents))]
        return RerankOutput(values, 0, self.device)


def load_reranker(config: Mapping[str, Any]):
    backend = str(config.get("backend", "qwen3"))
    if backend in {"dense_reference", "dense"}:
        return DenseReferenceReranker()
    return Qwen3Reranker(
        config["model_dir"], device=str(config.get("device", "auto")), dtype=str(config.get("dtype", "bfloat16")),
        max_length=int(config.get("max_length", 512)), batch_size=int(config.get("batch_size", 8)),
        instruction=str(config.get("instruction", "")),
    )
