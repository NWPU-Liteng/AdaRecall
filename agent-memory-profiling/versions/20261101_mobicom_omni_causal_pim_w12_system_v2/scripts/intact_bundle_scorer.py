"""Frozen neural bundle scoring with explicit padded-token resource accounting."""
from __future__ import annotations

import hashlib
import time

import torch

from joint_bundle_scorer import JointBundleScorer, scorer_inputs


PADDED_TOKEN_BUDGET = 8 * 4096
MAX_LENGTH = 32768


def token_batches(lengths, batch_size=8, token_budget=PADDED_TOKEN_BUDGET):
    if not lengths or min(lengths) <= 0 or max(lengths) > token_budget or batch_size <= 0:
        raise ValueError("invalid bounded scoring lengths")
    batches, pending = [], []
    for index in sorted(range(len(lengths)), key=lambda position: (lengths[position], position)):
        if pending and (len(pending) == batch_size or lengths[index] * (len(pending) + 1) > token_budget):
            batches.append(pending)
            pending = []
        pending.append(index)
    if pending:
        batches.append(pending)
    return batches


class IntactBundleScorer(JointBundleScorer):
    def __init__(self, model_path, *, device):
        super().__init__(model_path, device=device, max_length=MAX_LENGTH)
        if self.model.config.max_position_embeddings < MAX_LENGTH:
            raise ValueError("scorer context cannot accommodate intact source bundles")
        self.provenance.update(
            batching="length_ordered_padded_token_bound_v1",
            padded_token_budget=PADDED_TOKEN_BUDGET,
        )

    def score(self, question, documents):
        if not documents:
            raise ValueError("empty intact scoring candidates")
        started = time.perf_counter()
        inputs = scorer_inputs(question, documents)
        encoded = self.tokenizer(inputs, add_special_tokens=False, truncation=False, padding=False)["input_ids"]
        lengths = [len(tokens) for tokens in encoded]
        if max(lengths) > self.max_length:
            raise ValueError("intact scorer refuses silent truncation")
        batches = token_batches(lengths)
        scores = [0.0] * len(encoded)
        for indices in batches:
            padded = self.tokenizer.pad({"input_ids": [encoded[index] for index in indices]},
                                        padding=True, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                logits = self.model(**padded, use_cache=False, logits_to_keep=1).logits[:, -1, :]
                pairs = torch.stack([logits[:, self.no_id], logits[:, self.yes_id]], dim=-1).float()
                probabilities = pairs.softmax(dim=-1)[:, 1].tolist()
            for index, probability in zip(indices, probabilities):
                scores[index] = float(probability)
        return scores, {
            "seconds": time.perf_counter() - started,
            "input_tokens": lengths,
            "documents_sha256": [hashlib.sha256(document.encode()).hexdigest() for document in documents],
            "inputs_sha256": [hashlib.sha256(prompt.encode()).hexdigest() for prompt in inputs],
            "max_input_tokens": max(lengths), "truncated_documents": 0,
            "batches": [{"indices": indices, "padded_tokens": len(indices) * max(lengths[index] for index in indices)}
                        for indices in batches],
            "provenance": self.provenance,
        }
