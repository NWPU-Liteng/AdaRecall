#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import time
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import torch

from adaptive_phase_controller import PHASES, run_controller
from trace_logger import append_jsonl, ensure_output_layout
from qa_eval_utils import extract_final_answer, score_prediction

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.embedding_client import TransformerEmbeddingClient
from src.llm_client import LocalQwenLLMClient, MockLLMClient, OllamaLLMClient
from src.reranker_client import resolve_reranker_client


def _load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def _read_memory_bank_subset(path: Path, required_ids: set[str] | None = None) -> list[dict]:
    if not required_ids:
        return _read_jsonl(path)
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("memory_id") in required_ids:
                rows.append(row)
    return rows


def _filter_runtime_rows(runtime_rows: list[dict], config: dict) -> list[dict]:
    event_type_filter = str(config.get("event_type_filter", "")).strip()
    min_visible_memory_count = int(config.get("min_visible_memory_count", 0))
    filtered = []
    for row in runtime_rows:
        if event_type_filter and str(row.get("event_type", "")).strip() != event_type_filter:
            continue
        visible_count = len(row.get("memory_ids_available", []))
        if visible_count < min_visible_memory_count:
            continue
        filtered.append(row)
    event_sort_by = str(config.get("event_sort_by", "")).strip().lower()
    if event_sort_by == "visible_memory_count_desc":
        filtered.sort(
            key=lambda row: (
                -len(row.get("memory_ids_available", [])),
                str(row.get("event_time", "")),
                str(row.get("event_id", "")),
            )
        )
    elif event_sort_by == "visible_memory_count_asc":
        filtered.sort(
            key=lambda row: (
                len(row.get("memory_ids_available", [])),
                str(row.get("event_time", "")),
                str(row.get("event_id", "")),
            )
        )
    max_events = int(config.get("max_events", 0))
    if max_events > 0:
        return filtered[:max_events]
    return filtered


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_call(fn, *args, **kwargs):
    _cuda_sync()
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    _cuda_sync()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return result, elapsed_ms


def _resolve_embed_client(config: dict):
    return TransformerEmbeddingClient(
        model_dir=str(config["embedding_model_dir"]),
        dim=int(config.get("embedding_dim", 768)),
        max_length=int(config.get("embedding_max_length", 512)),
        batch_size=int(config.get("embedding_batch_size", 16)),
    )


def _resolve_llm_client(config: dict, *, max_tokens_key: str, fallback_default: int):
    backend = str(config.get("generation_backend", "mock")).strip()
    local_qwen_dir = str(PROJECT_ROOT.parent / "dir")
    max_tokens = int(config.get(max_tokens_key, config.get("llm_max_new_tokens", fallback_default)))
    if backend == "local_qwen":
        return LocalQwenLLMClient(model_dir=local_qwen_dir, max_new_tokens=max_tokens)
    if backend == "ollama":
        return OllamaLLMClient()
    return MockLLMClient()


def _embedding_cache_paths(manifest_path: Path, config: dict) -> tuple[Path, Path]:
    stem = hashlib.sha256(
        (
            str(manifest_path)
            + "::"
            + str(config.get("embedding_model_dir"))
            + "::"
            + str(config.get("embedding_dim", 768))
        ).encode("utf-8")
    ).hexdigest()[:16]
    cache_dir = manifest_path.parent / "embedding_cache"
    return cache_dir / f"{stem}.npy", cache_dir / f"{stem}.json"


def _ensure_embedding_cache(memory_bank: list[dict], manifest_path: Path, config: dict) -> tuple[np.ndarray, list[str], float]:
    vector_path, ids_path = _embedding_cache_paths(manifest_path, config)
    if vector_path.exists() and ids_path.exists():
        vectors = np.load(vector_path).astype(np.float32)
        ids = json.loads(ids_path.read_text(encoding="utf-8"))
        return vectors, ids, 0.0
    vector_path.parent.mkdir(parents=True, exist_ok=True)
    embed_client = _resolve_embed_client(config)
    texts = [row["text"] for row in memory_bank]
    vectors, elapsed_ms = _time_call(embed_client.embed_batch, texts)
    np.save(vector_path, vectors.astype(np.float32))
    ids = [row["memory_id"] for row in memory_bank]
    ids_path.write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")
    return vectors.astype(np.float32), ids, elapsed_ms


def _dense_retrieve(query_vector: np.ndarray, candidate_vectors: np.ndarray, candidate_memories: list[dict], top_r: int):
    scores = candidate_vectors @ query_vector.astype(np.float32)
    top_indices = np.argsort(scores)[-top_r:][::-1]
    return [(candidate_memories[i], float(scores[i])) for i in top_indices]


def _build_context(selected: list[tuple[dict, float]]) -> str:
    if not selected:
        return "No retrieved memory."
    parts = []
    for memory, score in selected:
        _ = score
        event_time = str(memory.get("event_time", "")).strip()
        session_id = str(memory.get("session_id", "")).strip()
        speaker = str(memory.get("speaker", "")).strip() or str(memory.get("role", "")).strip() or "unknown"
        metadata = [f"id={memory['memory_id']}"]
        if event_time:
            metadata.append(f"time={event_time}")
        if session_id:
            metadata.append(f"session={session_id}")
        parts.append(f"[{' | '.join(metadata)}] {speaker}: {memory['text']}")
    return "\n".join(parts)


def _build_context_preview(selected: list[tuple[dict, float]]) -> list[str]:
    previews: list[str] = []
    for memory, _ in selected[:2]:
        previews.append(f"{memory['speaker']}: {memory['text']}")
    return previews


def _build_answer_prompt(query_text: str, context_text: str, query_event_time: str) -> str:
    return (
        "You are evaluating a memory retrieval benchmark.\n"
        "Answer the user query using only the provided memory context.\n"
        "Treat the timestamps and session metadata as evidence when the query asks when, what time, asks about recurrence, or asks how long ago something happened.\n"
        "Use the current query timestamp together with memory timestamps when resolving relative time expressions.\n"
        "Do simple arithmetic or list-position reasoning when the evidence explicitly supports it.\n"
        "Output exactly one short final answer line.\n"
        "Do not continue the conversation.\n"
        "Do not explain your reasoning.\n"
        "Do not hedge or self-correct.\n"
        "Do not output placeholders such as [answer], [state], or [name].\n"
        "If the exact answer is not explicit in the evidence, answer exactly: insufficient information\n"
        "If the context is insufficient, answer exactly: insufficient information\n"
        "When possible, output only a short phrase, date, time, number, yes/no, or comma-separated list.\n\n"
        f"Current query timestamp: {query_event_time}\n\n"
        f"User query: {query_text}\n\n"
        f"Memory context:\n{context_text}\n"
        "\nFinal answer only:"
    )


def _run_selection(
    *,
    retrieval_query: str,
    candidate_memories: list[dict],
    candidate_vectors: np.ndarray,
    config: dict,
    reranker_client,
):
    embed_client = _resolve_embed_client(config)
    query_vector, embedding_latency_ms = _time_call(embed_client.embed, retrieval_query)
    dense_top_r, ann_latency_ms = _time_call(
        _dense_retrieve,
        query_vector,
        candidate_vectors,
        candidate_memories,
        int(config.get("retrieval_top_r", 50)),
    )
    pruned_top_b = dense_top_r[: int(config.get("pruner_top_b", 16))]
    pruner_latency_ms = 0.0
    if pruned_top_b:
        rerank_pairs = [(retrieval_query, memory, score) for memory, score in pruned_top_b]
        rerank_scores, rerank_latency_ms = _time_call(reranker_client.score_pairs, rerank_pairs)
        reranked = sorted(
            [
                (memory, float(rerank_score), float(vector_score))
                for (memory, vector_score), rerank_score in zip(pruned_top_b, rerank_scores)
            ],
            key=lambda item: item[1],
            reverse=True,
        )
    else:
        reranked = []
        rerank_latency_ms = 0.0
    final_top_k = int(config.get("final_top_k", 8))
    selected = [(memory, score) for memory, score, _ in reranked[:final_top_k]]
    context_text = _build_context(selected)
    logical_payload_bytes = sum(len(memory["text"].encode("utf-8")) for memory, _ in selected)
    header_bytes = int(config.get("header_size_bytes", 128)) * len(pruned_top_b)
    selection_latency_ms = embedding_latency_ms + ann_latency_ms + pruner_latency_ms + rerank_latency_ms
    return {
        "selected": selected,
        "dense_top_r": dense_top_r,
        "pruned_top_b": pruned_top_b,
        "context_text": context_text,
        "embedding_latency_ms": embedding_latency_ms,
        "ann_latency_ms": ann_latency_ms,
        "pruner_latency_ms": pruner_latency_ms,
        "rerank_latency_ms": rerank_latency_ms,
        "selection_latency_ms": selection_latency_ms,
        "logical_header_bytes_scanned": header_bytes,
        "logical_payload_bytes_fetched": logical_payload_bytes,
    }


def _decide_mode_action(mode: str, phase: str, llm_client, query_text: str, selected_context_preview: list[str]) -> dict:
    if mode == "shared_once":
        action = "retrieve" if phase == "planning" else "skip"
        return {
            "action": action,
            "retrieval_goal": query_text if action == "retrieve" else "",
            "needed_evidence": [],
            "reason": "shared_once baseline",
            "confidence": 1.0,
            "latency_ms": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "raw_text": "",
        }
    if mode == "fixed_three":
        return {
            "action": "retrieve",
            "retrieval_goal": query_text,
            "needed_evidence": [],
            "reason": "fixed_three baseline",
            "confidence": 1.0,
            "latency_ms": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "raw_text": "",
        }
    decision = run_controller(llm_client, phase, query_text, selected_context_preview)
    return {
        "action": decision.action,
        "retrieval_goal": decision.retrieval_goal,
        "needed_evidence": decision.needed_evidence,
        "reason": decision.reason,
        "confidence": decision.confidence,
        "latency_ms": decision.latency_ms,
        "prompt_tokens": decision.prompt_tokens,
        "completion_tokens": decision.completion_tokens,
        "raw_text": decision.raw_text,
    }


def _write_run_manifest(output_dir: Path, manifest_path: Path, config: dict, precompute_embedding_ms: float) -> Path:
    payload = {
        "manifest_path": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "seed": int(config.get("seed", 42)),
        "models": {
            "embedding_model_dir": str(config.get("embedding_model_dir")),
            "reranker_model_dir": str(config.get("reranker_model_dir")),
            "generation_backend": str(config.get("generation_backend")),
            "embedding_dim": int(config.get("embedding_dim", 768)),
            "reranker_batch_size": int(config.get("reranker_batch_size", 1)),
            "dtype": "bfloat16",
            "device": f"cuda:{os.getenv('LOCAL_QWEN_CUDA_DEVICE', '0')}" if torch.cuda.is_available() else "cpu",
        },
        "hardware": {
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "torch_version": torch.__version__,
        },
        "precompute_embedding_ms": precompute_embedding_ms,
    }
    path = output_dir / "run_manifest.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _summarize(raw_rows: list[dict], output_dir: Path) -> None:
    raw_df = pd.DataFrame(raw_rows)
    raw_df.to_json(output_dir / "raw_profile_frame.json", orient="records", force_ascii=False)
    if raw_df.empty:
        (output_dir / "summary.csv").write_text("", encoding="utf-8")
        (output_dir / "summary.md").write_text("# V4 Summary\n\nNo rows.\n", encoding="utf-8")
        return
    grouped = (
        raw_df.groupby(["dataset", "event_type", "agent_memory_mode"], dropna=False)
        .agg(
            runs=("event_id", "count"),
            avg_total_response_latency_ms=("total_response_latency_ms", "mean"),
            p50_total_response_latency_ms=("total_response_latency_ms", "median"),
            p95_total_response_latency_ms=("total_response_latency_ms", lambda s: float(s.quantile(0.95))),
            avg_retrieval_invocation_count=("retrieval_invocation_count", "mean"),
            avg_controller_latency_ms=("controller_total_latency_ms", "mean"),
            avg_selection_path_only_latency_ms=("selection_path_only_latency_ms", "mean"),
            avg_phase_selection_latency_ms=("phase_selection_total_latency_ms", "mean"),
            avg_rerank_latency_ms=("phase_rerank_total_latency_ms", "mean"),
            avg_logical_payload_bytes_fetched=("logical_payload_bytes_fetched_total", "mean"),
            avg_evidence_recall=("evidence_recall", "mean"),
            avg_answer_exact_match=("answer_exact_match", "mean"),
        )
        .reset_index()
    )
    grouped.to_csv(output_dir / "summary.csv", index=False)
    lines = ["# V4 Summary", ""]
    try:
        lines.append(grouped.to_markdown(index=False))
    except Exception:
        lines.append("```text")
        lines.append(grouped.to_string(index=False))
        lines.append("```")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile V4 real-data adaptive-phase agent runs.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    random.seed(int(config.get("seed", 42)))
    np.random.seed(int(config.get("seed", 42)))
    selection_only = bool(config.get("selection_only", False))
    active_phases = tuple(phase for phase in PHASES if not (selection_only and phase == "output"))

    manifest_path = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    paths = ensure_output_layout(output_dir)
    runtime_rows = _read_jsonl(manifest_path)
    runtime_rows = _filter_runtime_rows(runtime_rows, config)
    manifest_dir = manifest_path.parent
    referenced_ids = {memory_id for row in runtime_rows for memory_id in row.get("memory_ids_available", [])}
    memory_bank = _read_memory_bank_subset(manifest_dir / "runtime_memory_bank.jsonl", required_ids=referenced_ids)
    evaluation_by_event = {row["event_id"]: row for row in _read_jsonl(manifest_dir / "evaluation_labels.jsonl")}
    mode = str(config.get("agent_memory_mode", "adaptive_phase")).strip()
    if mode not in {"shared_once", "fixed_three", "adaptive_phase"}:
        raise ValueError(f"Unsupported agent_memory_mode: {mode}")

    memory_bank_by_id = {row["memory_id"]: row for row in memory_bank}
    memory_vectors, memory_ids, precompute_embedding_ms = _ensure_embedding_cache(memory_bank, manifest_path, config)
    memory_index = {memory_id: idx for idx, memory_id in enumerate(memory_ids)}
    controller_llm_client = _resolve_llm_client(
        config,
        max_tokens_key="controller_max_new_tokens",
        fallback_default=16,
    )
    answer_llm_client = _resolve_llm_client(
        config,
        max_tokens_key="answer_max_new_tokens",
        fallback_default=64,
    )
    reranker_client = resolve_reranker_client(config, str(PROJECT_ROOT))
    _write_run_manifest(output_dir, manifest_path, config, precompute_embedding_ms)

    raw_rows: list[dict] = []
    phase_rows: list[dict] = []
    for row in runtime_rows:
        event_id = row["event_id"]
        candidate_ids = [memory_id for memory_id in row.get("memory_ids_available", []) if memory_id in memory_bank_by_id]
        candidate_memories = [memory_bank_by_id[memory_id] for memory_id in candidate_ids]
        candidate_indices = [memory_index[memory_id] for memory_id in candidate_ids if memory_id in memory_index]
        candidate_vectors = memory_vectors[np.asarray(candidate_indices, dtype=np.int64)] if candidate_indices else np.zeros((0, memory_vectors.shape[1]), dtype=np.float32)
        _cuda_sync()
        start_total = time.perf_counter()
        accumulated_selected: list[tuple[dict, float]] = []
        accumulated_ids: list[str] = []
        phase_selection_total_latency_ms = 0.0
        phase_rerank_total_latency_ms = 0.0
        controller_total_latency_ms = 0.0
        logical_payload_bytes_total = 0
        retrieval_invocation_count = 0
        for phase in active_phases:
            selected_context_preview = _build_context_preview(accumulated_selected)
            decision = _decide_mode_action(
                mode,
                phase,
                controller_llm_client,
                row["query_text"],
                selected_context_preview,
            )
            selection_result = {
                "selected": [],
                "dense_top_r": [],
                "pruned_top_b": [],
                "context_text": "No retrieved memory.",
                "embedding_latency_ms": 0.0,
                "ann_latency_ms": 0.0,
                "pruner_latency_ms": 0.0,
                "rerank_latency_ms": 0.0,
                "selection_latency_ms": 0.0,
                "logical_header_bytes_scanned": 0,
                "logical_payload_bytes_fetched": 0,
            }
            controller_total_latency_ms += float(decision["latency_ms"])
            if decision["action"] == "retrieve" and len(candidate_memories) > 0:
                retrieval_invocation_count += 1
                selection_result = _run_selection(
                    retrieval_query=decision["retrieval_goal"] or row["query_text"],
                    candidate_memories=candidate_memories,
                    candidate_vectors=candidate_vectors,
                    config=config,
                    reranker_client=reranker_client,
                )
                for memory, score in selection_result["selected"]:
                    if memory["memory_id"] not in accumulated_ids:
                        accumulated_selected.append((memory, score))
                        accumulated_ids.append(memory["memory_id"])
            phase_selection_total_latency_ms += selection_result["selection_latency_ms"]
            phase_rerank_total_latency_ms += selection_result["rerank_latency_ms"]
            logical_payload_bytes_total += int(selection_result["logical_payload_bytes_fetched"])
            phase_rows.append(
                {
                    "run_id": output_dir.name,
                    "dataset": row["dataset"],
                    "tenant_id": row["tenant_id"],
                    "event_id": event_id,
                    "event_time": row["event_time"],
                    "phase": phase,
                    "controller_action": decision["action"],
                    "controller_goal": decision["retrieval_goal"],
                    "controller_reason": decision["reason"],
                    "controller_confidence": decision["confidence"],
                    "controller_latency_ms": decision["latency_ms"],
                    "controller_prompt_tokens": decision["prompt_tokens"],
                    "controller_completion_tokens": decision["completion_tokens"],
                    "available_memory_count": len(candidate_memories),
                    "dense_top_r_ids": [memory["memory_id"] for memory, _ in selection_result["dense_top_r"]],
                    "pruned_top_b_ids": [memory["memory_id"] for memory, _ in selection_result["pruned_top_b"]],
                    "reranked_top_k_ids": [memory["memory_id"] for memory, _ in selection_result["selected"]],
                    "answer_cited_memory_ids": [],
                    "phase_selection_latency_ms": selection_result["selection_latency_ms"],
                    "embedding_latency_ms": selection_result["embedding_latency_ms"],
                    "ann_latency_ms": selection_result["ann_latency_ms"],
                    "pruner_latency_ms": selection_result["pruner_latency_ms"],
                    "rerank_latency_ms": selection_result["rerank_latency_ms"],
                    "logical_header_bytes_scanned": selection_result["logical_header_bytes_scanned"],
                    "logical_payload_bytes_fetched": selection_result["logical_payload_bytes_fetched"],
                    "cross_tenant_candidate_count": 0,
                    "hypergraph_edge_count_before": 0,
                    "hypergraph_edge_count_after": 0,
                    "runtime_input_fields": list(row.get("runtime_allowed_fields", [])),
                    }
            )
        if selection_only:
            answer_response = {
                "text": "",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "elapsed_ms": 0.0,
            }
            answer_text = ""
        else:
            answer_prompt = _build_answer_prompt(
                row["query_text"],
                _build_context(accumulated_selected[: int(config.get("final_top_k", 8))]),
                str(row.get("event_time", "")),
            )
            answer_response = answer_llm_client.generate_with_stats(answer_prompt)
            answer_text_raw = answer_response["text"]
            answer_text = extract_final_answer(answer_text_raw, query_text=row["query_text"])
        if selection_only:
            answer_text_raw = ""
        cited_memory_ids = sorted(set(memory_id for memory_id in accumulated_ids if memory_id in answer_text))
        total_response_latency_ms = (time.perf_counter() - start_total) * 1000.0
        label = evaluation_by_event.get(event_id, {})
        gold_turn_ids = set(label.get("gold_turn_ids") or [])
        selected_ids = set(accumulated_ids)
        evidence_recall = float(len(gold_turn_ids & selected_ids) / len(gold_turn_ids)) if gold_turn_ids else 0.0
        if selection_only:
            answer_exact_match = np.nan
            answer_extracted_exact_match = np.nan
            answer_token_f1 = np.nan
            answer_extracted_token_f1 = np.nan
            answer_contains_reference = np.nan
            answer_extracted_contains_reference = np.nan
            answer_insufficient_information_rate = np.nan
        else:
            qa_scores = (
                score_prediction(
                    answer_text,
                    label.get("reference_answer"),
                    query_text=row.get("query_text"),
                )
                if label
                else {}
            )
            answer_exact_match = float(qa_scores.get("answer_exact_match", 0.0))
            answer_extracted_exact_match = float(qa_scores.get("answer_extracted_exact_match", 0.0))
            answer_token_f1 = float(qa_scores.get("answer_token_f1", 0.0))
            answer_extracted_token_f1 = float(qa_scores.get("answer_extracted_token_f1", 0.0))
            answer_contains_reference = float(qa_scores.get("answer_contains_reference", 0.0))
            answer_extracted_contains_reference = float(
                qa_scores.get("answer_extracted_contains_reference", 0.0)
            )
            answer_insufficient_information_rate = float(
                qa_scores.get("answer_insufficient_information_rate", 0.0)
            )
        raw_row = {
            "dataset": row["dataset"],
            "tenant_id": row["tenant_id"],
            "event_id": event_id,
            "event_type": row["event_type"],
            "agent_memory_mode": mode,
            "selection_only": selection_only,
            "query_text": row["query_text"],
            "answer_text": answer_text,
            "answer_text_raw": answer_text_raw,
            "answer_prompt_tokens": int(answer_response.get("prompt_tokens", 0)),
            "answer_completion_tokens": int(answer_response.get("completion_tokens", 0)),
            "answer_generation_latency_ms": float(answer_response.get("elapsed_ms", 0.0)),
            "controller_total_latency_ms": controller_total_latency_ms,
            "selection_path_only_latency_ms": phase_selection_total_latency_ms,
            "phase_selection_total_latency_ms": phase_selection_total_latency_ms,
            "phase_rerank_total_latency_ms": phase_rerank_total_latency_ms,
            "retrieval_invocation_count": retrieval_invocation_count,
            "total_response_latency_ms": total_response_latency_ms,
            "logical_payload_bytes_fetched_total": logical_payload_bytes_total,
            "evidence_recall": evidence_recall,
            "answer_exact_match": answer_exact_match,
            "answer_extracted_exact_match": answer_extracted_exact_match,
            "answer_token_f1": answer_token_f1,
            "answer_extracted_token_f1": answer_extracted_token_f1,
            "answer_contains_reference": answer_contains_reference,
            "answer_extracted_contains_reference": answer_extracted_contains_reference,
            "answer_insufficient_information_rate": answer_insufficient_information_rate,
            "selected_memory_ids": accumulated_ids,
            "cited_memory_ids": cited_memory_ids,
            "gold_turn_ids": sorted(gold_turn_ids),
            "error": "",
        }
        raw_rows.append(raw_row)
        append_jsonl(paths["raw_profile"], [raw_row])
        append_jsonl(paths["phase_trace"], phase_rows[-3:])
    _summarize(raw_rows, output_dir)
    print(f"raw_profile={paths['raw_profile']}")
    print(f"phase_trace={paths['phase_trace']}")
    print(f"summary_csv={paths['summary_csv']}")
    print(f"summary_md={paths['summary_md']}")


if __name__ == "__main__":
    main()
