from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

VERSION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = VERSION_ROOT.parents[1]
V24_ROOT = PROJECT_ROOT / "versions" / "20260718_prism_aligned_method1_closure_v24"
sys.path.insert(0, str(V24_ROOT / "src"))

from v24.io import iter_jsonl  # noqa: E402
from v24.metrics import quality_metrics  # noqa: E402
from v24.reranker import Qwen3Reranker  # noqa: E402

METHODS = ("full_n", "dense_c", "lexical_c", "adaptive_rule")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    canonical = VERSION_ROOT / "artifacts" / "fresh" / "canonical"
    memories = list(iter_jsonl(canonical / "memory_records.jsonl"))
    queries = list(iter_jsonl(canonical / "queries.jsonl"))
    evaluations = {row["query_id"]: row for row in iter_jsonl(canonical / "evaluation.jsonl")}
    memory_by_id = {row["memory_id"]: row for row in memories}
    frontier_rows = list(iter_jsonl(VERSION_ROOT / "runs" / "v26_fresh_once" / "frontier.jsonl"))
    frontiers = {(row["query_id"], row["method"]): row for row in frontier_rows}
    queries = [query for query in queries if int.from_bytes(hashlib.sha256(query["query_id"].encode()).digest()[:4], "big") % args.num_shards == args.shard]
    model = Qwen3Reranker(PROJECT_ROOT / "models" / "models" / "Qwen3-Reranker-0.6B", device=args.device, dtype="bfloat16", max_length=512, batch_size=8)
    if not model.device.startswith("cuda") or not model.smoke()["passed"]:
        raise RuntimeError("fresh Qwen CUDA smoke failed")
    output = VERSION_ROOT / "runs" / "v26_fresh_once" / f"qwen_shard_{args.shard:02d}.jsonl"
    existing = {(row["query_id"], row["method"]) for row in iter_jsonl(output)} if output.exists() else set()
    completed = {query_id for query_id, method in existing if method == METHODS[-1]}
    for query in queries:
        if query["query_id"] in completed:
            continue
        union, seen = [], set()
        for method in METHODS:
            for memory_id in frontiers[(query["query_id"], method)]["frontier_ids"]:
                if memory_id not in seen:
                    seen.add(memory_id)
                    union.append(memory_id)
        started = perf_counter()
        result = model.score_pairs(query["query"], [memory_by_id[value]["content"] for value in union])
        elapsed = (perf_counter() - started) * 1000.0
        scores = dict(zip(union, result.scores))
        evaluation = type("Evaluation", (), {"gold_memory_ids": tuple(evaluations[query["query_id"]]["gold_memory_ids"]), "evidence_groups": tuple(tuple(x) for x in evaluations[query["query_id"]]["evidence_groups"])})()
        with output.open("a", encoding="utf-8") as handle:
            for method in METHODS:
                frontier = frontiers[(query["query_id"], method)]["frontier_ids"]
                final_ids = sorted(frontier, key=lambda value: (-float(scores[value]), value))[:5]
                metric = quality_metrics(final_ids, evaluation, k=5)
                row = {"query_id": query["query_id"], "query_type": query["metadata"]["query_type"], "namespace_id": query["namespace_id"], "method": method, "frontier_ids": frontier, "reranked_ids": final_ids, "metrics": {**frontiers[(query["query_id"], method)]["metrics"], "candidate_recall_final": metric["candidate_recall"], "evidence_group_coverage_final": metric["evidence_group_coverage"], "precision_at_k": metric["precision_at_k"], "ndcg_at_10": metric["ndcg_at_10"]}, "cost": {"candidate_count": len(frontier), "union_count": len(union), "union_ms": elapsed, "estimated_method_ms": elapsed * len(frontier) / len(union), "estimated_input_tokens": int(result.input_tokens * len(frontier) / len(union))}, "safety": frontiers[(query["query_id"], method)]["safety"], "formal_fresh_run": "v26_locomo_fresh_once_20260718"}
                handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
        completed.add(query["query_id"])
        if len(completed) % 25 == 0:
            print(json.dumps({"shard": args.shard, "completed": len(completed), "total": len(queries)}), flush=True)
    print(json.dumps({"shard": args.shard, "queries": len(completed)}, sort_keys=True))


if __name__ == "__main__":
    main()
