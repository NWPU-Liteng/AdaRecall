from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from .faiss_store import search_flat_ip, validate_flat_ip_top_r


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_dense_rerank_selection(
    *,
    event: dict[str, Any],
    memories: list[dict[str, Any]],
    embedding_client: Any,
    reranker_client: Any | None,
    r: int,
    b: int,
    k: int,
    validate_faiss: bool = False,
    doc_vecs: np.ndarray | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    scoped = memories
    t_scope_ms = (time.perf_counter() - t0) * 1000.0

    _sync_cuda()
    t1 = time.perf_counter()
    query_vec = embedding_client.embed(event["query_text"])
    if doc_vecs is None:
        doc_vecs = embedding_client.embed_batch([m["memory_text"] for m in scoped])
    _sync_cuda()
    t_query_encode_ms = (time.perf_counter() - t1) * 1000.0

    t2 = time.perf_counter()
    scores, idx = search_flat_ip(doc_vecs, query_vec, r)
    t_seed_ms = (time.perf_counter() - t2) * 1000.0
    dense_seed = [(int(i), float(s)) for i, s in zip(idx.tolist(), scores.tolist()) if i >= 0]
    if validate_faiss:
        faiss_order = [item[0] for item in dense_seed]
        if not validate_flat_ip_top_r(doc_vecs, query_vec, faiss_order, r):
            raise RuntimeError(f"FAISS exactness mismatch for event {event['event_id']}")

    t3 = time.perf_counter()
    frontier = dense_seed[: min(b, len(dense_seed))]
    t_frontier_ms = (time.perf_counter() - t3) * 1000.0

    t4 = time.perf_counter()
    frontier_memories = [scoped[i] for i, _ in frontier]
    payload_bytes_for_rerank = sum(int(m.get("text_bytes") or len(m["memory_text"].encode("utf-8"))) for m in frontier_memories)
    t_payload_fetch_ms = (time.perf_counter() - t4) * 1000.0

    reranked: list[tuple[dict[str, Any], float]] = [(m, score) for m, (_, score) in zip(frontier_memories, frontier)]
    t_rerank_ms = 0.0
    rerank_pair_token_sum = 0
    if reranker_client is not None and frontier_memories:
        pairs = [(event["query_text"], {"text": m["memory_text"]}, score) for m, (_, score) in zip(frontier_memories, frontier)]
        _sync_cuda()
        t5 = time.perf_counter()
        rerank_scores = reranker_client.score_pairs(pairs)
        _sync_cuda()
        t_rerank_ms = (time.perf_counter() - t5) * 1000.0
        reranked = sorted(zip(frontier_memories, rerank_scores), key=lambda item: item[1], reverse=True)
        rerank_pair_token_sum = sum(len(event["query_text"].split()) + len(m["memory_text"].split()) for m in frontier_memories)

    t6 = time.perf_counter()
    final = reranked[: min(k, len(reranked))]
    selected = [item[0] for item in final]
    payload_bytes_for_qwen_context = sum(int(m.get("text_bytes") or len(m["memory_text"].encode("utf-8"))) for m in selected)
    t_context_pack_ms = (time.perf_counter() - t6) * 1000.0

    total_selection = (
        t_scope_ms
        + t_query_encode_ms
        + t_seed_ms
        + t_frontier_ms
        + t_payload_fetch_ms
        + t_rerank_ms
        + t_context_pack_ms
    )
    return {
        "event_id": event["event_id"],
        "protocol": event["protocol"],
        "task": event["task"],
        "setting": event["setting"],
        "tenant_id_hash": str(abs(hash(event["tenant_id"]))),
        "N": len(scoped),
        "R_requested": r,
        "R_effective": len(dense_seed),
        "B_requested": b,
        "B_effective": len(frontier),
        "K": k,
        "K_effective": len(selected),
        "dense_seed_ids": [scoped[i]["memory_id"] for i, _ in dense_seed],
        "frontier_ids": [m["memory_id"] for m in frontier_memories],
        "reranked_ids": [m["memory_id"] for m, _ in reranked],
        "final_context_ids": [m["memory_id"] for m in selected],
        "t_scope_ms": t_scope_ms,
        "t_query_encode_ms": t_query_encode_ms,
        "t_seed_ms": t_seed_ms,
        "t_frontier_ms": t_frontier_ms,
        "t_payload_fetch_ms": t_payload_fetch_ms,
        "t_rerank_ms": t_rerank_ms,
        "t_context_pack_ms": t_context_pack_ms,
        "t_total_selection_ms": total_selection,
        "header_bytes_scanned": len(scoped) * 96,
        "header_bytes_returned": len(dense_seed) * 96,
        "payload_bytes_for_rerank": payload_bytes_for_rerank,
        "payload_bytes_for_qwen_context": payload_bytes_for_qwen_context,
        "unique_payload_memory_count": len({m["memory_id"] for m in selected}),
        "rerank_pair_count": len(frontier) if reranker_client is not None else 0,
        "rerank_pair_token_sum": rerank_pair_token_sum,
        "reranker_pair_batch_size": getattr(reranker_client, "batch_size", 0) if reranker_client is not None else 0,
    }
