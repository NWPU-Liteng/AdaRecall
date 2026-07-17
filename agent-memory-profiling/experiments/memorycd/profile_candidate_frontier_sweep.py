#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embedding_client import TransformerEmbeddingClient
from src.memorycd.audit import iter_jsonl
from src.memorycd.faiss_store import search_flat_ip, validate_flat_ip_top_r
from src.reranker_client import CrossEncoderRerankerClient


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _hash_tenant(tenant_id: str) -> str:
    return hashlib.sha256(tenant_id.encode("utf-8")).hexdigest()[:16]


def _payload_bytes(memories: list[dict]) -> int:
    return sum(int(m.get("text_bytes") or len(m["memory_text"].encode("utf-8"))) for m in memories)


def run_event_sweep(
    *,
    event: dict,
    memories: list[dict],
    doc_vecs: np.ndarray,
    embedding: TransformerEmbeddingClient,
    reranker: CrossEncoderRerankerClient,
    r_values: list[int],
    b_values: list[int],
    k: int,
    validate_faiss: bool,
) -> list[dict]:
    t0 = time.perf_counter()
    t_scope_ms = (time.perf_counter() - t0) * 1000.0

    _sync_cuda()
    t1 = time.perf_counter()
    query_vec = embedding.embed(event["query_text"])
    _sync_cuda()
    t_query_encode_ms = (time.perf_counter() - t1) * 1000.0

    seed_by_r: dict[int, tuple[list[tuple[int, float]], float]] = {}
    for r in sorted(r_values):
        t2 = time.perf_counter()
        scores, idx = search_flat_ip(doc_vecs, query_vec, r)
        t_seed_ms = (time.perf_counter() - t2) * 1000.0
        dense_seed = [(int(i), float(s)) for i, s in zip(idx.tolist(), scores.tolist()) if i >= 0]
        if validate_faiss:
            if not validate_flat_ip_top_r(doc_vecs, query_vec, [item[0] for item in dense_seed], r):
                raise RuntimeError(f"FAISS exactness mismatch for event {event['event_id']}")
        seed_by_r[r] = (dense_seed, t_seed_ms)

    # Dense-prefix V5.1 baseline: for any R, frontier is top-B of dense seeds.
    # Since B <= R for emitted traces, top-B is identical to the top-B prefix of
    # the largest-R seed set. Rerank each B once per event and reuse across R.
    max_r = max(seed_by_r)
    max_seed, _ = seed_by_r[max_r]
    rerank_by_b: dict[int, dict] = {}
    for b in sorted(b_values):
        frontier = max_seed[: min(b, len(max_seed))]
        frontier_memories = [memories[i] for i, _ in frontier]
        t_payload0 = time.perf_counter()
        payload_bytes_for_rerank = _payload_bytes(frontier_memories)
        t_payload_fetch_ms = (time.perf_counter() - t_payload0) * 1000.0
        pairs = [(event["query_text"], {"text": m["memory_text"]}, score) for m, (_, score) in zip(frontier_memories, frontier)]
        _sync_cuda()
        t5 = time.perf_counter()
        rerank_scores = reranker.score_pairs(pairs) if pairs else []
        _sync_cuda()
        t_rerank_ms = (time.perf_counter() - t5) * 1000.0
        reranked = sorted(zip(frontier_memories, rerank_scores), key=lambda item: item[1], reverse=True)
        rerank_pair_token_sum = sum(len(event["query_text"].split()) + len(m["memory_text"].split()) for m in frontier_memories)
        rerank_by_b[b] = {
            "reranked": reranked,
            "t_rerank_ms": t_rerank_ms,
            "rerank_pair_token_sum": rerank_pair_token_sum,
            "payload_bytes_for_rerank": payload_bytes_for_rerank,
            "t_payload_fetch_ms": t_payload_fetch_ms,
        }

    traces: list[dict] = []
    for r in sorted(r_values):
        dense_seed, t_seed_ms = seed_by_r[r]
        for b in sorted(b_values):
            if b > r:
                continue
            t3 = time.perf_counter()
            frontier = dense_seed[: min(b, len(dense_seed))]
            t_frontier_ms = (time.perf_counter() - t3) * 1000.0
            frontier_memories = [memories[i] for i, _ in frontier]
            cached = rerank_by_b[b]
            reranked = cached["reranked"]

            t6 = time.perf_counter()
            final = reranked[: min(k, len(reranked))]
            selected = [item[0] for item in final]
            payload_bytes_for_qwen_context = _payload_bytes(selected)
            t_context_pack_ms = (time.perf_counter() - t6) * 1000.0
            total = (
                t_scope_ms
                + t_query_encode_ms
                + t_seed_ms
                + t_frontier_ms
                + cached["t_payload_fetch_ms"]
                + cached["t_rerank_ms"]
                + t_context_pack_ms
            )
            traces.append(
                {
                    "event_id": event["event_id"],
                    "protocol": event["protocol"],
                    "task": event["task"],
                    "setting": event["setting"],
                    "tenant_id_hash": _hash_tenant(event["tenant_id"]),
                    "N": len(memories),
                    "R_requested": r,
                    "R_effective": len(dense_seed),
                    "B_requested": b,
                    "B_effective": len(frontier),
                    "K": k,
                    "K_effective": len(selected),
                    "dense_seed_ids": [memories[i]["memory_id"] for i, _ in dense_seed],
                    "frontier_ids": [m["memory_id"] for m in frontier_memories],
                    "reranked_ids": [m["memory_id"] for m, _ in reranked],
                    "final_context_ids": [m["memory_id"] for m in selected],
                    "t_scope_ms": t_scope_ms,
                    "t_query_encode_ms": t_query_encode_ms,
                    "t_seed_ms": t_seed_ms,
                    "t_frontier_ms": t_frontier_ms,
                    "t_payload_fetch_ms": cached["t_payload_fetch_ms"],
                    "t_rerank_ms": cached["t_rerank_ms"],
                    "t_context_pack_ms": t_context_pack_ms,
                    "t_total_selection_ms": total,
                    "header_bytes_scanned": len(memories) * 96,
                    "header_bytes_returned": len(dense_seed) * 96,
                    "payload_bytes_for_rerank": cached["payload_bytes_for_rerank"],
                    "payload_bytes_for_qwen_context": payload_bytes_for_qwen_context,
                    "unique_payload_memory_count": len({m["memory_id"] for m in selected}),
                    "rerank_pair_count": len(frontier),
                    "rerank_pair_token_sum": cached["rerank_pair_token_sum"],
                    "reranker_pair_batch_size": getattr(reranker, "batch_size", 0),
                    "trace_reuse_note": "query_encoded_once_per_event; rerank_reused_per_B",
                }
            )
    return traces


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version-dir", type=Path, required=True)
    parser.add_argument("--protocol", choices=["official_single", "official_cross", "causal_cross"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-events", type=int, default=10)
    parser.add_argument("--tasks", nargs="+", default=["rating_prediction", "item_ranking"])
    parser.add_argument("--r-values", type=int, nargs="+", default=[32])
    parser.add_argument("--b-values", type=int, nargs="+", default=[8])
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--embedding-model", type=Path, default=Path("models/bge-base-en-v1.5"))
    parser.add_argument("--reranker-model", type=Path, default=Path("models/cross-encoder-ms-marco-MiniLM-L6-v2"))
    parser.add_argument("--reranker-batch-size", type=int, default=16)
    parser.add_argument("--validate-faiss", action="store_true")
    parser.add_argument("--embedding-cache-dir", type=Path, default=None)
    args = parser.parse_args()

    manifest_dir = args.version_dir / "manifests" / f"memorycd_{args.protocol}"
    runtime_path = manifest_dir / "runtime_manifest.jsonl"
    memory_path = manifest_dir / "memory_bank.jsonl"
    if not runtime_path.exists() or not memory_path.exists():
        raise SystemExit(f"Missing manifest files in {manifest_dir}")

    os.environ["LOCAL_QWEN_CUDA_DEVICE"] = str(args.gpu)

    memory_by_id = {row["memory_id"]: row for row in iter_jsonl(memory_path)}
    embedding_matrix = None
    embedding_row_by_id = {}
    if args.embedding_cache_dir is not None:
        vectors_path = args.embedding_cache_dir / "memory_embeddings.npy"
        ids_path = args.embedding_cache_dir / "memory_embedding_ids.jsonl"
        if not vectors_path.exists() or not ids_path.exists():
            raise SystemExit(f"Missing embedding cache in {args.embedding_cache_dir}")
        embedding_matrix = np.load(vectors_path, mmap_mode="r")
        embedding_row_by_id = {row["memory_id"]: int(row["row"]) for row in iter_jsonl(ids_path)}

    events = [row for row in iter_jsonl(runtime_path) if row["task"] in set(args.tasks)]
    events = events[: args.max_events] if args.max_events else events

    embedding = TransformerEmbeddingClient(str(args.embedding_model), dim=768, batch_size=16)
    reranker = CrossEncoderRerankerClient(str(args.reranker_model), batch_size=args.reranker_batch_size)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = args.output_dir / f"{args.protocol}_selection_sweep.jsonl"
    run_manifest = {
        "protocol": args.protocol,
        "max_events": args.max_events,
        "tasks": args.tasks,
        "r_values": args.r_values,
        "b_values": args.b_values,
        "k": args.k,
        "embedding_model": str(args.embedding_model),
        "reranker_model": str(args.reranker_model),
        "reranker_batch_size": args.reranker_batch_size,
        "validate_faiss": args.validate_faiss,
        "faiss_audit": "top_r_score_boundary_tolerance_1e-6" if args.validate_faiss else None,
        "embedding_cache_dir": str(args.embedding_cache_dir) if args.embedding_cache_dir else None,
        "reuse_policy": "query_encoded_once_per_event; rerank_reused_per_B",
    }
    (args.output_dir / f"{args.protocol}_run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    count = 0
    with trace_path.open("w", encoding="utf-8") as out:
        for event in events:
            if embedding_matrix is not None:
                ids = [mid for mid in event["runtime_memory_ids"] if mid in embedding_row_by_id and mid in memory_by_id]
                rows = [embedding_row_by_id[mid] for mid in ids]
                memories = [memory_by_id[mid] for mid in ids]
                doc_vecs = np.asarray(embedding_matrix[rows], dtype="float32")
            else:
                memories = [memory_by_id[mid] for mid in event["runtime_memory_ids"] if mid in memory_by_id]
                doc_vecs = embedding.embed_batch([m["memory_text"] for m in memories]) if memories else np.zeros((0, 768), dtype="float32")
            if not memories:
                continue
            traces = run_event_sweep(
                event=event,
                memories=memories,
                doc_vecs=doc_vecs,
                embedding=embedding,
                reranker=reranker,
                r_values=args.r_values,
                b_values=args.b_values,
                k=args.k,
                validate_faiss=args.validate_faiss,
            )
            for trace in traces:
                out.write(json.dumps(trace, ensure_ascii=False) + "\n")
                count += 1
            out.flush()
    print(json.dumps({"trace_path": str(trace_path), "rows": count}, indent=2))


if __name__ == "__main__":
    main()
