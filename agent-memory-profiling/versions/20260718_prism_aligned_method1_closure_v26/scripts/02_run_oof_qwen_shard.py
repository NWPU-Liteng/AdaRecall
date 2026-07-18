from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

VERSION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = VERSION_ROOT.parents[1]
V24_ROOT = PROJECT_ROOT / "versions" / "20260718_prism_aligned_method1_closure_v24"
sys.path.insert(0, str(V24_ROOT / "src"))

from v24.dataset import load_canonical  # noqa: E402
from v24.io import iter_jsonl  # noqa: E402
from v24.metrics import quality_metrics  # noqa: E402
from v24.reranker import Qwen3Reranker  # noqa: E402


METHODS = ("full_n", "dense_c", "lexical_c", "v25_fixed", "adaptive_rule", "adaptive_logistic")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.shard < 0 or args.shard >= args.num_shards:
        raise ValueError("invalid shard")

    frontier_path = VERSION_ROOT / "runs" / "v26_train_oof_frontier" / "query_results.jsonl"
    frontier_rows = list(iter_jsonl(frontier_path))
    by_query: dict[str, dict[str, dict[str, Any]]] = {}
    for row in frontier_rows:
        if row["method"] in METHODS:
            by_query.setdefault(row["query_id"], {})[row["method"]] = row
    dataset = load_canonical(V24_ROOT / "inputs", split="train")
    queries = [query for query in dataset.queries["train"] if int.from_bytes(__import__("hashlib").sha256(query.query_id.encode()).digest()[:4], "big") % args.num_shards == args.shard]
    query_by_id = {query.query_id: query for query in queries}
    memory_by_id = {memory.memory_id: memory for memory in dataset.memories}
    embeddings = np.asarray(dataset.embeddings, dtype=np.float32)
    embeddings = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    position_by_id = dataset.memory_index
    reranker = Qwen3Reranker(
        PROJECT_ROOT / "models" / "models" / "Qwen3-Reranker-0.6B",
        device=args.device,
        dtype="bfloat16",
        max_length=512,
        batch_size=args.batch_size,
    )
    if not reranker.device.startswith("cuda"):
        raise RuntimeError(f"formal Qwen OOF requires CUDA, resolved {reranker.device}")
    smoke = reranker.smoke()
    if not smoke["passed"]:
        raise RuntimeError(f"reranker smoke failed: {smoke}")

    output = VERSION_ROOT / "runs" / "v26_train_oof_qwen" / f"shard_{args.shard:02d}.jsonl"
    existing = {(row["query_id"], row["method"]): row for row in iter_jsonl(output)} if output.exists() else {}
    done_queries = {query_id for query_id, method in existing if method == METHODS[-1]}
    for query_index, query in enumerate(queries):
        if query.query_id in done_queries:
            continue
        methods = by_query[query.query_id]
        missing = set(METHODS) - set(methods)
        if missing:
            raise RuntimeError(f"missing frontier methods for {query.query_id}: {sorted(missing)}")
        union_ids: list[str] = []
        seen = set()
        for method in METHODS:
            for memory_id in methods[method]["frontier_ids"]:
                if memory_id not in seen:
                    seen.add(memory_id)
                    union_ids.append(memory_id)
        documents = [memory_by_id[memory_id].content for memory_id in union_ids]
        started = perf_counter()
        result = reranker.score_pairs(query.query, documents)
        elapsed_ms = float((perf_counter() - started) * 1000.0)
        score_by_id = dict(zip(union_ids, result.scores))
        query_vector = np.array(dataset.query_vector("train", query.query_id), dtype=np.float32, copy=True)
        query_vector /= max(float(np.linalg.norm(query_vector)), 1e-12)
        dense_by_id = {
            memory_id: float(embeddings[int(position_by_id[memory_id])] @ query_vector) for memory_id in union_ids
        }
        evaluation = dataset.evaluations["train"][query.query_id]
        for method in METHODS:
            frontier_ids = methods[method]["frontier_ids"]
            ranked = sorted(
                frontier_ids,
                key=lambda memory_id: (-float(score_by_id[memory_id]), -float(dense_by_id[memory_id]), memory_id),
            )
            final_ids = ranked[:5]
            frontier_metrics = quality_metrics(frontier_ids, evaluation)
            final_metrics = quality_metrics(final_ids, evaluation, k=5)
            row = {
                "query_id": query.query_id,
                "query_type": str(query.metadata.get("query_type", "unknown")),
                "namespace_id": query.namespace_id,
                "fold": int(methods[method]["fold"]),
                "method": method,
                "frontier_ids": frontier_ids,
                "reranked_ids": final_ids,
                "metrics": {
                    "candidate_recall_frontier": float(frontier_metrics["candidate_recall"]),
                    "evidence_group_coverage_frontier": float(frontier_metrics["evidence_group_coverage"]),
                    "candidate_recall_final": float(final_metrics["candidate_recall"]),
                    "evidence_group_coverage_final": float(final_metrics["evidence_group_coverage"]),
                    "precision_at_k": float(final_metrics["precision_at_k"]),
                    "ndcg_at_10": float(final_metrics["ndcg_at_10"]),
                },
                "cost": {
                    "candidate_count": len(frontier_ids),
                    "union_candidate_count": len(union_ids),
                    "union_rerank_ms": elapsed_ms,
                    "estimated_method_rerank_ms": elapsed_ms * len(frontier_ids) / max(len(union_ids), 1),
                    "union_input_tokens": int(result.input_tokens),
                    "estimated_method_input_tokens": int(result.input_tokens * len(frontier_ids) / max(len(union_ids), 1)),
                },
                "runtime": {
                    "device": reranker.device,
                    "attention_implementation": reranker.attention_implementation,
                    "batch_size": args.batch_size,
                    "shared_union_scoring": True,
                },
                "safety": methods[method]["safety"],
                "scientific_scope": "train_strict_5fold_oof",
            }
            append_jsonl(output, row)
            existing[(query.query_id, method)] = row
        done_queries.add(query.query_id)
        if len(done_queries) % 25 == 0:
            print(json.dumps({"shard": args.shard, "completed": len(done_queries), "total": len(queries)}), flush=True)
    print(json.dumps({"shard": args.shard, "rows": len(existing), "queries": len(done_queries), "smoke": smoke}, sort_keys=True))


if __name__ == "__main__":
    main()
