from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


SYSTEM = "Judge whether the Document meets the requirements based on the Query and Instruct. Answer only yes or no."
BOUNDARY_MARKER = "__METHOD1_DOCUMENT_BOUNDARY_7C81E44A__"


@dataclass(frozen=True)
class PackedPrompt:
    input_ids: list[int]
    original_tokens: int
    packed_tokens: int
    document_tokens: int
    document_tokens_kept: int
    truncated: bool
    suffix_tokens: int


@dataclass
class RerankOutput:
    scores: list[float]
    input_tokens: int
    device: str
    packing: list[PackedPrompt]


class SuffixPreservingPromptPacker:
    def __init__(
        self,
        tokenizer,
        *,
        max_length: int,
        instruction: str,
        document_head_fraction: float = 0.5,
    ) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        if not 0.0 <= document_head_fraction <= 1.0:
            raise ValueError("document_head_fraction must be in [0, 1]")
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.instruction = str(instruction)
        self.document_head_fraction = float(document_head_fraction)

    def _format(self, query: str, document: str) -> str:
        user = f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {document}"
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _encode(self, value: str) -> list[int]:
        return list(self.tokenizer(value, add_special_tokens=False)["input_ids"])

    def template_parts(self, query: str) -> tuple[list[int], list[int]]:
        marked = self._format(str(query), BOUNDARY_MARKER)
        if marked.count(BOUNDARY_MARKER) != 1:
            raise RuntimeError("chat template did not preserve a unique document boundary")
        prefix, suffix = marked.split(BOUNDARY_MARKER, maxsplit=1)
        prefix_ids, suffix_ids = self._encode(prefix), self._encode(suffix)
        if not suffix_ids:
            raise RuntimeError("assistant generation suffix is empty")
        return prefix_ids, suffix_ids

    def pack(self, query: str, document: str) -> PackedPrompt:
        prefix_ids, suffix_ids = self.template_parts(query)
        fixed = len(prefix_ids) + len(suffix_ids)
        if fixed >= self.max_length:
            raise ValueError(
                f"query/chat template consumes {fixed} tokens, leaving no document budget under {self.max_length}"
            )
        document_ids = self._encode(str(document))
        document_budget = self.max_length - fixed
        truncated = len(document_ids) > document_budget
        if truncated:
            head = int(round(document_budget * self.document_head_fraction))
            head = min(max(head, 0), document_budget)
            tail = document_budget - head
            kept = document_ids[:head] + (document_ids[-tail:] if tail else [])
        else:
            kept = document_ids
        input_ids = prefix_ids + kept + suffix_ids
        if len(input_ids) > self.max_length or input_ids[-len(suffix_ids) :] != suffix_ids:
            raise AssertionError("suffix-preserving packer violated its token budget or suffix invariant")
        return PackedPrompt(
            input_ids=input_ids,
            original_tokens=fixed + len(document_ids),
            packed_tokens=len(input_ids),
            document_tokens=len(document_ids),
            document_tokens_kept=len(kept),
            truncated=truncated,
            suffix_tokens=len(suffix_ids),
        )


class Qwen3SuffixPreservingReranker:
    def __init__(
        self,
        model_dir: str | Path,
        *,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        max_length: int = 2048,
        batch_size: int = 4,
        instruction: str,
        document_head_fraction: float = 0.5,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_dir = Path(model_dir)
        self.device = str(device)
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("formal suffix-preserving Qwen reranking requires CUDA")
        self.batch_size = int(batch_size)
        self.attention_implementation = "eager"
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir),
            local_files_only=True,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.packer = SuffixPreservingPromptPacker(
            self.tokenizer,
            max_length=max_length,
            instruction=instruction,
            document_head_fraction=document_head_fraction,
        )
        model_dtype = getattr(torch, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_dir),
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=model_dtype,
            attn_implementation="eager",
        )
        self.model.to(self.device)
        self.model.eval()
        self.yes_id = int(self.tokenizer.convert_tokens_to_ids("yes"))
        self.no_id = int(self.tokenizer.convert_tokens_to_ids("no"))
        if self.yes_id == self.no_id or min(self.yes_id, self.no_id) < 0:
            raise RuntimeError(f"invalid yes/no token IDs: {self.yes_id}, {self.no_id}")

    def score_pairs(self, query: str, documents: Sequence[str]) -> RerankOutput:
        if not documents:
            return RerankOutput([], 0, self.device, [])
        packed = [self.packer.pack(str(query), str(document)) for document in documents]
        scores: list[float] = []
        token_count = 0
        for start in range(0, len(packed), self.batch_size):
            batch = packed[start : start + self.batch_size]
            encoded = self.tokenizer.pad(
                {"input_ids": [row.input_ids for row in batch]},
                padding=True,
                return_tensors="pt",
            )
            token_count += int(encoded["attention_mask"].sum().item())
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self.torch.inference_mode():
                logits = self.model(**encoded).logits[:, -1, :]
                pair_logits = logits[:, [self.yes_id, self.no_id]].float()
                batch_scores = self.torch.softmax(pair_logits, dim=-1)[:, 0]
            scores.extend(float(value) for value in batch_scores.detach().cpu().tolist())
            del encoded, logits, pair_logits, batch_scores
        return RerankOutput(scores, token_count, self.device, packed)

    def smoke(self) -> dict[str, Any]:
        short = self.score_pairs(
            "Which medication treats hypertension?",
            ["Valsartan was prescribed for hypertension.", "The patient booked a holiday train."],
        )
        long_document = "irrelevant preface " * 3000 + "Valsartan was prescribed for hypertension."
        packed = self.packer.pack("Which medication treats hypertension?", long_document)
        _, suffix_ids = self.packer.template_parts("Which medication treats hypertension?")
        return {
            "device": short.device,
            "attention_implementation": self.attention_implementation,
            "yes_token_id": self.yes_id,
            "no_token_id": self.no_id,
            "scores": short.scores,
            "max_length": self.packer.max_length,
            "long_prompt_truncated": packed.truncated,
            "long_prompt_suffix_preserved": packed.input_ids[-len(suffix_ids) :] == suffix_ids,
            "passed": bool(
                len(short.scores) == 2
                and short.scores[0] > short.scores[1]
                and packed.truncated
                and packed.input_ids[-len(suffix_ids) :] == suffix_ids
            ),
        }
