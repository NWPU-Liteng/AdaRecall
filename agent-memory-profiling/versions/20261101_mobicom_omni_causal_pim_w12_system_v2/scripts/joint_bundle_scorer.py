"""Frozen model-native yes/no scores for fully delivered temporal source bundles."""
from __future__ import annotations

import hashlib
from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the '
          'Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
          '<|im_end|>\n<|im_start|>user\n')
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
INSTRUCTION = 'Given a web search query, retrieve relevant passages that answer the query'


def serialize_bundle(windows, ordered_indices: list[int]) -> str:
    return "\n\n".join(
        f"[EVENT {windows[index].event_id}]\n"
        f"[SOURCE_SPAN {windows[index].start_char}:{windows[index].end_char}]\n"
        f"{windows[index].text}" for index in ordered_indices)


def scorer_inputs(question: str, documents: list[str]) -> list[str]:
    return [f"{PREFIX}<Instruct>: {INSTRUCTION}\n<Query>: {question}\n<Document>: {document}{SUFFIX}"
            for document in documents]


class JointBundleScorer:
    def __init__(self, model_path: str | Path, *, device: str | torch.device, batch_size: int = 8,
                 max_length: int = 4096) -> None:
        if batch_size <= 0 or max_length <= 0:
            raise ValueError("invalid scorer resource budget")
        self.model_path = Path(model_path)
        self.device = torch.device(device)
        self.batch_size, self.max_length = batch_size, max_length
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path), local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa",
        ).eval().to(self.device)
        self.yes_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.no_id = self.tokenizer.convert_tokens_to_ids("no")
        if self.yes_id == self.no_id or self.yes_id is None or self.no_id is None:
            raise ValueError("yes/no scoring tokens unavailable")
        self.provenance = {
            "model_path": str(self.model_path), "batch_size": batch_size, "max_length": max_length,
            "dtype": "bfloat16", "attention": "sdpa", "truncation": False,
            "format_sha256": hashlib.sha256((PREFIX + INSTRUCTION + SUFFIX).encode()).hexdigest(),
            "model_config_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in self.model_path.glob("*.json")},
            "model_weight_content_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                            for path in self.model_path.glob("*.safetensors")},
        }

    def score(self, question: str, documents: list[str]) -> tuple[list[float], dict]:
        if not documents:
            raise ValueError("empty joint scoring candidates")
        started = time.perf_counter()
        inputs = scorer_inputs(question, documents)
        encoded = self.tokenizer(inputs, add_special_tokens=False, truncation=False, padding=False)["input_ids"]
        lengths = [len(tokens) for tokens in encoded]
        if max(lengths) > self.max_length:
            raise ValueError(f"joint scorer refuses silent truncation: {max(lengths)} > {self.max_length}")
        order = sorted(range(len(encoded)), key=lambda index: (lengths[index], index))
        scores = [0.0] * len(encoded)
        for batch_start in range(0, len(order), self.batch_size):
            indices = order[batch_start:batch_start + self.batch_size]
            padded = self.tokenizer.pad({"input_ids": [encoded[index] for index in indices]},
                                        padding=True, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                logits = self.model(**padded, use_cache=False, logits_to_keep=1).logits[:, -1, :]
                pairs = torch.stack([logits[:, self.no_id], logits[:, self.yes_id]], dim=-1).float()
                probabilities = pairs.softmax(dim=-1)[:, 1].tolist()
            for index, probability in zip(indices, probabilities):
                scores[index] = float(probability)
        return scores, {"seconds": time.perf_counter() - started, "input_tokens": lengths,
                        "documents_sha256": [hashlib.sha256(document.encode()).hexdigest() for document in documents],
                        "inputs_sha256": [hashlib.sha256(prompt.encode()).hexdigest() for prompt in inputs],
                        "max_input_tokens": max(lengths), "truncated_documents": 0,
                        "provenance": self.provenance}
