from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import sys

try:
    from rouge_score import rouge_scorer
except ImportError:  # pragma: no cover
    rouge_scorer = None

try:
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
except ImportError:  # pragma: no cover
    SmoothingFunction = None
    sentence_bleu = None

from src.embedding_client import TransformerEmbeddingClient
from src.llm_client import LocalQwenLLMClient
from src.memorycd.audit import iter_jsonl
from src.memorycd.candidate_frontier import run_dense_rerank_selection
from src.memorycd.loader import open_text_auto
from src.reranker_client import CrossEncoderRerankerClient


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _hash_tenant(tenant_id: str) -> str:
    return hashlib.sha256(tenant_id.encode("utf-8")).hexdigest()[:16]


def _payload_bytes(memories: list[dict[str, Any]]) -> int:
    return sum(int(m.get("text_bytes") or len(m["memory_text"].encode("utf-8"))) for m in memories)


def _format_memory_for_prompt(memory: dict[str, Any]) -> dict[str, Any]:
    title = ""
    review = ""
    rating = None
    domain = memory.get("domain", "")
    parent_asin = memory.get("parent_asin", "")
    for line in str(memory.get("memory_text", "")).splitlines():
        if line.startswith("[Review Title] "):
            title = line[len("[Review Title] ") :]
        elif line.startswith("[Review] "):
            review = line[len("[Review] ") :]
        elif line.startswith("[Rating] "):
            raw = line[len("[Rating] ") :].split("/", 1)[0].strip()
            try:
                rating = float(raw)
            except ValueError:
                rating = None
    return {
        "domain_category": domain,
        "parent_asin": parent_asin,
        "title": title,
        "text": review,
        "rating": rating,
        "timestamp": int(memory.get("timestamp") or 0),
    }


class MemoryCDMetadataLoader:
    def __init__(self, meta_dir: Path):
        self.meta_dir = meta_dir
        self.metadata: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        for domain in ["Beauty_and_Personal_Care", "Books", "Electronics", "Home_and_Kitchen"]:
            candidates = [
                self.meta_dir / f"{domain}.jsonl.gz",
                self.meta_dir / f"{domain}.jsonl",
                self.meta_dir / f"meta_{domain}.jsonl.gz",
                self.meta_dir / f"meta_{domain}.jsonl",
            ]
            path = next((candidate for candidate in candidates if candidate.exists()), None)
            if path is None:
                continue
            with open_text_auto(path) as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    parent_asin = row.get("parent_asin")
                    if parent_asin:
                        self.metadata[str(parent_asin)] = row

    def get_metadata(self, parent_asin: str) -> dict[str, Any] | None:
        return self.metadata.get(parent_asin)


def _format_target_item(
    *,
    event: dict[str, Any],
    metadata_loader: Any,
) -> dict[str, Any]:
    query_lines = str(event["query_text"]).splitlines()
    target_asin = ""
    target_rating = None
    target_text = ""
    for line in query_lines:
        if line.startswith("Target item: "):
            target_asin = line[len("Target item: ") :].strip()
        elif line.startswith("Target rating: "):
            raw = line[len("Target rating: ") :].split("/", 1)[0].strip()
            try:
                target_rating = float(raw)
            except ValueError:
                target_rating = None
        elif line.startswith("Target review text: "):
            target_text = line[len("Target review text: ") :].strip()

    target = {
        "parent_asin": target_asin,
        "rating": target_rating,
        "text": target_text,
        "timestamp": int(event.get("target_timestamp") or 0),
    }
    meta = metadata_loader.get_metadata(target_asin) if target_asin else None
    if meta:
        meta_copy = dict(meta)
        meta_copy.pop("images", None)
        meta_copy.pop("videos", None)
        target["item_meta"] = meta_copy
        if not target.get("title"):
            target["title"] = meta_copy.get("title", "")
    return target


class LocalPromptBuilder:
    def __init__(self, include_metadata: bool = True):
        self.include_metadata = include_metadata

    def _format_interaction(
        self,
        interaction: dict[str, Any],
        include_meta: bool = True,
        include_rating: bool = True,
        include_review: bool = True,
    ) -> str:
        parts: list[str] = []
        if "domain_category" in interaction:
            parts.append(f"Domain: {interaction['domain_category']}")
        if interaction.get("title"):
            parts.append(f"Title: {interaction['title']}")
        if include_rating and interaction.get("rating") is not None:
            parts.append(f"Rating: {interaction['rating']}/5")
        if include_review and interaction.get("text"):
            text = str(interaction["text"])
            if len(text) > 200:
                text = text[:200] + "..."
            parts.append(f"Review: {text}")
        if include_meta and interaction.get("item_meta"):
            meta = interaction["item_meta"]
            if meta.get("title") and meta.get("title") != interaction.get("title"):
                parts.append(f"Product: {meta['title']}")
            if meta.get("average_rating") is not None:
                parts.append(f"Product Avg Rating: {meta['average_rating']}")
            if meta.get("main_category"):
                parts.append(f"Category: {meta['main_category']}")
        return "\n  ".join(parts)

    def build_rating_prediction_prompt(self, memory: list[dict[str, Any]], target_item: dict[str, Any]) -> str:
        parts = [
            "You are a personalized rating prediction system. Based on a user's purchase and review history, "
            "predict what rating (1-5) they would give to a new product.",
            "",
            "User's Recent Review History:",
        ]
        for i, inter in enumerate(memory, 1):
            parts.append(f"\n{i}. {self._format_interaction(inter, self.include_metadata)}")
        parts.extend(
            [
                "",
                "",
                "Target Product to Rate:",
                self._format_interaction(target_item, self.include_metadata, include_rating=False, include_review=False),
                "",
                "",
                "Based on this user's history and preferences, predict the rating (1-5) they would give to the target product.",
                "Respond with ONLY a single integer between 1 and 5.",
            ]
        )
        return "\n".join(parts)

    def build_review_summarization_prompt(self, memory: list[dict[str, Any]], target_item: dict[str, Any]) -> str:
        parts = [
            "You are a personalized review summarization system. Based on a user's review history and writing style, "
            "generate a concise review title that summarizes their full review.",
            "",
            "User's Recent Review History (showing their writing style and preferences):",
        ]
        for i, inter in enumerate(memory, 1):
            parts.append(f"\n{i}. {self._format_interaction(inter, self.include_metadata)}")
        rating = target_item.get("rating", "N/A")
        review = target_item.get("text", "")
        parts.extend(
            [
                "",
                "",
                "Target Product and Review to Summarize:",
                self._format_interaction(target_item, self.include_metadata, include_rating=False, include_review=False),
                f"User's Rating: {rating}/5",
                f"User's Full Review: {review}",
                "",
                "",
                "Generate a concise review title (3-10 words) capturing the main sentiment in the user's style.",
                "Respond with ONLY the review title, nothing else.",
            ]
        )
        return "\n".join(parts)

    def build_review_generation_prompt(self, memory: list[dict[str, Any]], target_item: dict[str, Any]) -> str:
        parts = [
            "You are a personalized review generation system. Based on a user's review history and writing style, "
            "generate a detailed review text for a product they rated.",
            "",
            "User's Recent Review History (showing their writing style and preferences):",
        ]
        for i, inter in enumerate(memory, 1):
            parts.append(f"\n{i}. {self._format_interaction(inter, self.include_metadata)}")
        rating = target_item.get("rating", "N/A")
        parts.extend(
            [
                "",
                "",
                "Target Product to Review:",
                self._format_interaction(target_item, self.include_metadata, include_rating=False, include_review=False),
                f"User's Rating: {rating}/5",
                "",
                "",
                "Generate a detailed review reflecting the user's writing style, typical length, and the rating sentiment.",
                "Respond with ONLY the review text, nothing else.",
            ]
        )
        return "\n".join(parts)

    def build_item_ranking_prompt(
        self,
        memory: list[dict[str, Any]],
        candidate_asins: list[str],
        metadata_loader: Any,
        target_item: dict[str, Any],
    ) -> str:
        parts = [
            "You are a personalized item recommendation system. Based on a user's purchase and review history, "
            "rank the candidate items in order of relevance (most likely to purchase first).",
            "",
            "User's Recent Purchase and Review History:",
        ]
        for i, inter in enumerate(memory, 1):
            parts.append(f"\n{i}. {self._format_interaction(inter, self.include_metadata)}")
        parts.extend(["", "", "Candidate Items to Rank:", ""])
        for i, asin in enumerate(candidate_asins, 1):
            meta = metadata_loader.get_metadata(asin) if metadata_loader else None
            parts.append(f"\nCandidate {i} (ASIN: {asin}):")
            if meta:
                if meta.get("title"):
                    parts.append(f"  Title: {meta['title']}")
                if meta.get("average_rating") is not None:
                    parts.append(f"  Average Rating: {meta['average_rating']}")
                if meta.get("main_category"):
                    parts.append(f"  Category: {meta['main_category']}")
                if meta.get("description"):
                    desc = str(meta["description"])[:200]
                    parts.append(f"  Description: {desc}...")
            else:
                parts.append("  (No metadata available)")
        parts.extend(
            [
                "",
                "",
                "Rank ALL candidate items from most to least relevant for this user.",
                "Respond with a comma-separated list of ASINs in ranked order, nothing else.",
                "Example: B0085BEQX8, B00123ABC9, B00456DEF0",
            ]
        )
        return "\n".join(parts)


class SimpleItemRankingEvaluator:
    @staticmethod
    def ndcg_at_k(ranked: list[str], gt: str, k: int) -> float:
        if not ranked or k <= 0:
            return 0.0
        topk = ranked[:k]
        rel = [1.0 if a == gt else 0.0 for a in topk]
        dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel))
        return dcg

    @staticmethod
    def recall_at_k(ranked: list[str], gt: str, k: int) -> float:
        if not ranked or k <= 0:
            return 0.0
        return 1.0 if gt in ranked[:k] else 0.0


class SimpleTextEvaluator:
    def __init__(self) -> None:
        self.scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True) if rouge_scorer is not None else None

    def add_prediction(self, predicted: str, reference: str) -> dict[str, float]:
        result: dict[str, float] = {"rouge1_f": 0.0, "rougeL_f": 0.0, "bleu1": 0.0, "bleu4": 0.0}
        if self.scorer is not None:
            scores = self.scorer.score(reference, predicted)
            result["rouge1_f"] = float(scores["rouge1"].fmeasure)
            result["rougeL_f"] = float(scores["rougeL"].fmeasure)
        if sentence_bleu is not None and SmoothingFunction is not None:
            try:
                ref_tokens = reference.split()
                pred_tokens = predicted.split()
                smoothing = SmoothingFunction()
                result["bleu1"] = float(sentence_bleu([ref_tokens], pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing.method1))
                result["bleu4"] = float(sentence_bleu([ref_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing.method1))
            except Exception as exc:  # pragma: no cover
                print(f"Warning: BLEU computation failed: {exc}", file=sys.stderr)
        return result


def _build_prompt_with_selected_memory(
    *,
    prompt_builder: LocalPromptBuilder,
    task: str,
    selected_memories: list[dict[str, Any]],
    target_item: dict[str, Any],
    metadata_loader: Any,
    candidate_asins: list[str] | None,
    user_id: str,
) -> tuple[str, Any]:
    _ = user_id
    if task == "rating_prediction":
        prompt = prompt_builder.build_rating_prediction_prompt(selected_memories, target_item)
        return prompt, None
    if task == "review_summarization":
        prompt = prompt_builder.build_review_summarization_prompt(selected_memories, target_item)
        return prompt, None
    if task == "review_generation":
        prompt = prompt_builder.build_review_generation_prompt(selected_memories, target_item)
        return prompt, None
    if task == "item_ranking":
        ordered_candidates = list(candidate_asins or [])
        prompt = prompt_builder.build_item_ranking_prompt(selected_memories, ordered_candidates, metadata_loader, target_item)
        return prompt, ordered_candidates
    raise ValueError(f"Unsupported task: {task}")


def _parse_prediction(
    *,
    task: str,
    response_text: str,
    ordered_candidates: list[str] | None,
) -> Any:
    if task == "rating_prediction":
        try:
            value = float(response_text)
        except ValueError:
            import re

            nums = re.findall(r"\d+\.?\d*", response_text)
            value = float(nums[0]) if nums else 3.0
        return max(1.0, min(5.0, value))
    if task in {"review_generation", "review_summarization"}:
        return response_text.strip()
    if task == "item_ranking":
        import re

        candidates = ordered_candidates or []
        ranked: list[str] = []
        seen: set[str] = set()
        for match in re.findall(r"B[0-9A-Z]{9}", response_text):
            if match in candidates and match not in seen:
                ranked.append(match)
                seen.add(match)
        for asin in candidates:
            if asin not in seen:
                ranked.append(asin)
        return ranked
    raise ValueError(f"Unsupported task: {task}")


def _per_prediction_metrics(task: str, prediction: Any, label: dict[str, Any], item_ranking_evaluator: Any, text_evaluator: Any) -> dict[str, float]:
    if task == "rating_prediction":
        gold = float(label["gold_rating"])
        return {"error": abs(float(prediction) - gold)}
    if task == "item_ranking":
        gt = str(label["positive_parent_asin"])
        return {
            "ndcg@1": item_ranking_evaluator._ndcg_at_k(prediction, gt, 1),
            "ndcg@3": item_ranking_evaluator._ndcg_at_k(prediction, gt, 3),
            "ndcg@5": item_ranking_evaluator._ndcg_at_k(prediction, gt, 5),
            "recall@1": item_ranking_evaluator._recall_at_k(prediction, gt, 1),
            "recall@3": item_ranking_evaluator._recall_at_k(prediction, gt, 3),
            "recall@5": item_ranking_evaluator._recall_at_k(prediction, gt, 5),
        }
    if task == "review_generation":
        return text_evaluator.add_prediction(str(prediction), str(label.get("gold_review_text") or ""))
    if task == "review_summarization":
        return text_evaluator.add_prediction(str(prediction), str(label.get("gold_review_title") or ""))
    raise ValueError(f"Unsupported task: {task}")


def _choose_metric_name(task: str) -> str:
    if task == "rating_prediction":
        return "rmse"
    if task == "item_ranking":
        return "ndcg@5"
    if task == "review_generation":
        return "rougeL"
    if task == "review_summarization":
        return "rougeL"
    raise ValueError(task)


def bootstrap_confidence_interval(
    values: list[float],
    *,
    num_samples: int = 1000,
    alpha: float = 0.05,
    seed: int = 20260622,
) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        return (values[0], values[0])
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(num_samples):
        sample = [values[rng.randrange(len(values))] for _ in range(len(values))]
        means.append(sum(sample) / len(sample))
    means.sort()
    low_idx = max(0, int(math.floor((alpha / 2.0) * len(means))))
    high_idx = min(len(means) - 1, int(math.ceil((1.0 - alpha / 2.0) * len(means))) - 1)
    return means[low_idx], means[high_idx]


def derive_delta(task: str, reference_metric: float) -> float:
    if task == "rating_prediction":
        return max(0.01, abs(reference_metric) * 0.01)
    return 0.01


@dataclass
class TaskRunArtifacts:
    trace_path: Path
    summary_path: Path
    rows: int


def load_protocol_runtime(
    *,
    version_dir: Path,
    protocol: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    manifest_dir = version_dir / "manifests" / f"memorycd_{protocol}"
    runtime_rows = list(iter_jsonl(manifest_dir / "runtime_manifest.jsonl"))
    label_by_id = {row["event_id"]: row for row in iter_jsonl(manifest_dir / "evaluation_labels.jsonl")}
    memory_by_id = {row["memory_id"]: row for row in iter_jsonl(manifest_dir / "memory_bank.jsonl")}
    return runtime_rows, label_by_id, memory_by_id


def load_embedding_cache(cache_dir: Path) -> tuple[np.ndarray, dict[str, int]]:
    vectors = np.load(cache_dir / "memory_embeddings.npy", mmap_mode="r")
    row_by_id = {row["memory_id"]: int(row["row"]) for row in iter_jsonl(cache_dir / "memory_embedding_ids.jsonl")}
    return vectors, row_by_id


def summarize_latency(values: Iterable[float]) -> dict[str, float]:
    vals = sorted(float(v) for v in values)
    if not vals:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
    return {
        "p50": vals[min(len(vals) - 1, int(round((len(vals) - 1) * 0.50)))],
        "p95": vals[min(len(vals) - 1, int(round((len(vals) - 1) * 0.95)))],
        "p99": vals[min(len(vals) - 1, int(round((len(vals) - 1) * 0.99)))],
        "mean": sum(vals) / len(vals),
    }


def compute_selection_for_method(
    *,
    method: str,
    event: dict[str, Any],
    memories: list[dict[str, Any]],
    doc_vecs: np.ndarray,
    embedding: TransformerEmbeddingClient,
    reranker: CrossEncoderRerankerClient | None,
    k: int,
    validate_faiss: bool,
    r: int | None = None,
    b: int | None = None,
) -> dict[str, Any]:
    if method == "recency_k8":
        selected = memories[-min(k, len(memories)) :]
        payload = _payload_bytes(selected)
        return {
            "event_id": event["event_id"],
            "protocol": event["protocol"],
            "task": event["task"],
            "setting": event["setting"],
            "tenant_id_hash": _hash_tenant(event["tenant_id"]),
            "N": len(memories),
            "R_requested": 0,
            "R_effective": 0,
            "B_requested": 0,
            "B_effective": 0,
            "K": k,
            "K_effective": len(selected),
            "dense_seed_ids": [],
            "frontier_ids": [m["memory_id"] for m in selected],
            "reranked_ids": [m["memory_id"] for m in selected],
            "final_context_ids": [m["memory_id"] for m in selected],
            "t_scope_ms": 0.0,
            "t_query_encode_ms": 0.0,
            "t_seed_ms": 0.0,
            "t_frontier_ms": 0.0,
            "t_payload_fetch_ms": 0.0,
            "t_rerank_ms": 0.0,
            "t_context_pack_ms": 0.0,
            "t_total_selection_ms": 0.0,
            "header_bytes_scanned": len(memories) * 96,
            "header_bytes_returned": len(selected) * 96,
            "payload_bytes_for_rerank": 0,
            "payload_bytes_for_qwen_context": payload,
            "unique_payload_memory_count": len({m["memory_id"] for m in selected}),
            "rerank_pair_count": 0,
            "rerank_pair_token_sum": 0,
            "reranker_pair_batch_size": 0,
            "method": method,
        }
    if method == "dense_k8":
        result = run_dense_rerank_selection(
            event=event,
            memories=memories,
            embedding_client=embedding,
            reranker_client=None,
            r=k,
            b=k,
            k=k,
            validate_faiss=validate_faiss,
            doc_vecs=doc_vecs,
        )
        result["method"] = method
        return result
    if r is None or b is None:
        raise ValueError("dense rerank method requires r and b")
    result = run_dense_rerank_selection(
        event=event,
        memories=memories,
        embedding_client=embedding,
        reranker_client=reranker,
        r=r,
        b=b,
        k=k,
        validate_faiss=validate_faiss,
        doc_vecs=doc_vecs,
    )
    result["method"] = f"dense_r{r}_b{b}_k{k}"
    return result


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
