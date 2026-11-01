"""Independent FP32 contract for newly compiled source-closed scoring inputs."""
from __future__ import annotations

import hashlib
from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from intact_bundle_scorer import token_batches, PADDED_TOKEN_BUDGET
from joint_bundle_scorer import PREFIX, INSTRUCTION, SUFFIX, scorer_inputs


MAX_LENGTH = 32768
PROBABILITY_TOLERANCE = 1e-4


class SourceClosedScorer:
    def __init__(self, model_path, *, device, batch_size=8):
        self.model_path = Path(model_path)
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.max_length = MAX_LENGTH
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path), local_files_only=True, dtype=torch.float32, attn_implementation="sdpa",
        ).eval().to(self.device)
        if self.model.config.max_position_embeddings < MAX_LENGTH:
            raise ValueError("insufficient scorer context")
        self.yes_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.no_id = self.tokenizer.convert_tokens_to_ids("no")
        if self.yes_id is None or self.no_id is None or self.yes_id == self.no_id:
            raise ValueError("invalid scoring token identity")
        self.provenance = {
            "model_path": str(self.model_path), "batch_size": batch_size, "max_length": MAX_LENGTH,
            "dtype": "float32", "attention": "sdpa", "truncation": False,
            "tf32": False, "batching": "length_ordered_padded_token_bound_v1",
            "padded_token_budget": PADDED_TOKEN_BUDGET,
            "format_sha256": hashlib.sha256((PREFIX + INSTRUCTION + SUFFIX).encode()).hexdigest(),
            "model_config_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in self.model_path.glob("*.json")},
            "model_weight_content_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in self.model_path.glob("*.safetensors")},
        }

    def score(self, question, documents):
        if not documents:
            raise ValueError("empty source-closed scoring candidates")
        started = time.perf_counter()
        inputs = scorer_inputs(question, documents)
        encoded = self.tokenizer(inputs, add_special_tokens=False, truncation=False, padding=False)["input_ids"]
        lengths = [len(tokens) for tokens in encoded]
        if max(lengths) > self.max_length:
            raise ValueError(f"source-closed scorer refuses truncation: {max(lengths)}")
        batches = token_batches(lengths, self.batch_size)
        scores = [0.0] * len(documents)
        for indices in batches:
            padded = self.tokenizer.pad({"input_ids": [encoded[index] for index in indices]},
                                        padding=True, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                logits = self.model(**padded, use_cache=False, logits_to_keep=1).logits[:, -1, :]
                pairs = torch.stack([logits[:, self.no_id], logits[:, self.yes_id]], dim=-1)
                probabilities = pairs.softmax(dim=-1)[:, 1].tolist()
            for index, probability in zip(indices, probabilities):
                scores[index] = float(probability)
        return scores, {
            "seconds": time.perf_counter() - started, "input_tokens": lengths,
            "documents_sha256": [hashlib.sha256(document.encode()).hexdigest() for document in documents],
            "inputs_sha256": [hashlib.sha256(prompt.encode()).hexdigest() for prompt in inputs],
            "max_input_tokens": max(lengths), "truncated_documents": 0,
            "batches": [{"indices": indices, "padded_tokens": len(indices) * max(lengths[index] for index in indices)}
                        for indices in batches], "provenance": self.provenance,
        }
