from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from common import PROJECT_ROOT, V24_ROOT, VERSION_ROOT, iter_jsonl, load_yaml, write_json, write_jsonl
from src.metrics import paired_bootstrap
from src.provenance import canonical_sha256, create_immutable_run, sha256_file
from src.qwen_reranker import Qwen3Reranker
from v24.dataset import load_canonical
from v24.metrics import quality_metrics

METHODS = (
    "full_hnsw_128",
    "dense_hnsw_32",
    "standard_ivf_32",
    "lexical_bm25_32",
    "semantic_graph_only_32",
    "dual_graph_only_32",
)
GRAPH_METHODS = ("semantic_graph_only_32", "dual_graph_only_32")
GATE_METRICS = ("ndcg_at_10", "precision_at_5", "recall_at_5", "group_coverage_at_5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontier-run-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--devices", default="0,2,3,4")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    return parser.parse_args()


def metrics(ranked_ids: list[str], evaluation: object) -> dict[str, float]:
    top10 = quality_metrics(ranked_ids[:10], evaluation, k=10)
    top5 = quality_metrics(ranked_ids[:5], evaluation, k=5)
    return {
        "ndcg_at_10": float(top10["ndcg_at_10"]),
        "mrr_at_10": float(top10["mrr_at_10"]),
        "precision_at_5": float(top5["precision_at_k"]),
        "recall_at_5": float(top5["recall_at_k"]),
        "group_coverage_at_5": float(top5["evidence_group_coverage"]),
    }


def worker(args: argparse.Namespace) -> None:
    frontier_dir = VERSION_ROOT / "runs" / args.frontier_run_id
    qwen_dir = VERSION_ROOT / "runs" / args.run_id
    aggregate = json.loads((frontier_dir / "aggregate_metrics.json").read_text(encoding="utf-8"))
    if set(aggregate["gate_c_promoted"]) != set(GRAPH_METHODS):
        raise RuntimeError("Qwen worker requires both registered Graph-only methods to pass Gate C")
    rows = list(iter_jsonl(frontier_dir / "per_query_frontier.jsonl"))
    by_query: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_query.setdefault(row["query_id"], {})[row["method"]] = row
    dataset = load_canonical(V24_ROOT / "inputs", split="train")
    query_order = [query.query_id for query in dataset.queries["train"]]
    memory_by_id = {memory.memory_id: memory for memory in dataset.memories}
    query_by_id = {query.query_id: query for query in dataset.queries["train"]}
    reranker = Qwen3Reranker(
        PROJECT_ROOT / "models" / "models" / "Qwen3-Reranker-0.6B",
        device="cuda:0",
        dtype="bfloat16",
        max_length=512,
        batch_size=8,
    )
    smoke = reranker.smoke()
    if not smoke["passed"] or not reranker.device.startswith("cuda"):
        raise RuntimeError(f"formal Qwen CUDA smoke failed: {smoke}")
    output_rows = []
    for query_index, query_id in enumerate(query_order):
        if query_index % args.num_shards != args.shard:
            continue
        methods = by_query[query_id]
        if set(methods) != set(METHODS):
            raise RuntimeError(f"incomplete frontier methods for {query_id}")
        union_ids = []
        seen = set()
        for method in METHODS:
            for memory_id in methods[method]["frontier_ids"]:
                if memory_id not in seen:
                    seen.add(memory_id)
                    union_ids.append(memory_id)
        documents = [memory_by_id[memory_id].content for memory_id in union_ids]
        started = time.perf_counter_ns()
        scored = reranker.score_pairs(query_by_id[query_id].query, documents)
        if reranker.device.startswith("cuda"):
            reranker.torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        score_by_id = dict(zip(union_ids, scored.scores))
        evaluation = dataset.evaluations["train"][query_id]
        for method in METHODS:
            frontier_ids = methods[method]["frontier_ids"]
            ranked = sorted(frontier_ids, key=lambda value: (-float(score_by_id[value]), value))
            output_rows.append(
                {
                    "query_id": query_id,
                    "query_type": methods[method]["query_type"],
                    "namespace_id": methods[method]["namespace_id"],
                    "fold": methods[method]["fold"],
                    "method": method,
                    "frontier_ids": frontier_ids,
                    "reranked_ids": ranked,
                    "reranker_scores": [float(score_by_id[value]) for value in ranked],
                    "reranked_ids_top10": ranked[:10],
                    "final_top5": ranked[:5],
                    "reranker_scores_top10": [float(score_by_id[value]) for value in ranked[:10]],
                    "metrics": metrics(ranked, evaluation),
                    "cost": {
                        "candidate_count": len(frontier_ids),
                        "union_candidate_count": len(union_ids),
                        "union_rerank_ms": elapsed_ms,
                        "estimated_method_rerank_ms": elapsed_ms * len(frontier_ids) / max(len(union_ids), 1),
                        "union_input_tokens": int(scored.input_tokens),
                        "estimated_method_input_tokens": int(round(scored.input_tokens * len(frontier_ids) / max(len(union_ids), 1))),
                    },
                    "device": reranker.device,
                    "attention_implementation": reranker.attention_implementation,
                    "frontier_rows_sha256": aggregate["rows_sha256"],
                    "run_id": args.run_id,
                }
            )
        if len(output_rows) % 300 == 0:
            print(f"worker {args.shard}: {len(output_rows) // len(METHODS)} queries", flush=True)
    write_jsonl(qwen_dir / f"shard_{args.shard:02d}.jsonl", output_rows)
    write_json(qwen_dir / f"shard_{args.shard:02d}_runtime.json", {"shard": args.shard, "num_shards": args.num_shards, "query_count": len(output_rows) // len(METHODS), "row_count": len(output_rows), "smoke": smoke})


def aggregate(args: argparse.Namespace, run_dir: Path, started: float) -> None:
    config = load_yaml(VERSION_ROOT / "configs" / "graph_only_main.yaml")
    frontier_dir = VERSION_ROOT / "runs" / args.frontier_run_id
    frontier_aggregate = json.loads((frontier_dir / "aggregate_metrics.json").read_text(encoding="utf-8"))
    rows = [row for path in sorted(run_dir.glob("shard_*.jsonl")) for row in iter_jsonl(path)]
    expected = 1355 * len(METHODS)
    if len(rows) != expected:
        raise RuntimeError(f"incomplete Qwen rows: {len(rows)} != {expected}")
    keys = [(row["query_id"], row["method"]) for row in rows]
    if len(set(keys)) != expected:
        raise RuntimeError("duplicate Qwen query/method rows")
    by_method = {method: {row["query_id"]: row for row in rows if row["method"] == method} for method in METHODS}
    dense = by_method["dense_hnsw_32"]
    summary = {}
    gate = config["gate"]
    gate_limits = {
        "ndcg_at_10": float(gate["ndcg_ci_lower_min"]),
        "precision_at_5": float(gate["precision_ci_lower_min"]),
        "recall_at_5": float(gate["recall5_ci_lower_min"]),
        "group_coverage_at_5": float(gate["group_coverage5_ci_lower_min"]),
    }
    for method in METHODS:
        method_rows = list(by_method[method].values())
        result = {
            "absolute": {metric: float(np.mean([row["metrics"][metric] for row in method_rows])) for metric in GATE_METRICS + ("mrr_at_10",)},
            "rerank_ms_mean_estimated": float(np.mean([row["cost"]["estimated_method_rerank_ms"] for row in method_rows])),
            "input_tokens_mean_estimated": float(np.mean([row["cost"]["estimated_method_input_tokens"] for row in method_rows])),
        }
        if method != "dense_hnsw_32":
            result["paired_vs_dense"] = {}
            for metric in GATE_METRICS:
                deltas = [row["metrics"][metric] - dense[row["query_id"]]["metrics"][metric] for row in method_rows]
                result["paired_vs_dense"][metric] = paired_bootstrap(deltas, seed=int(config["random_seed"]), samples=int(gate["bootstrap_samples"]))
        if method in GRAPH_METHODS:
            result["gate_d_passed"] = all(result["paired_vs_dense"][metric]["ci_95"][0] >= gate_limits[metric] for metric in GATE_METRICS)
        summary[method] = result
    write_jsonl(run_dir / "per_query_qwen.jsonl", sorted(rows, key=lambda row: (row["query_id"], METHODS.index(row["method"]))))
    result = {
        "schema_version": "method1-v28-qwen-oof-1",
        "status": "QWEN_OOF_COMPLETE",
        "run_id": args.run_id,
        "frontier_run_id": args.frontier_run_id,
        "frontier_rows_sha256": frontier_aggregate["rows_sha256"],
        "frontier_aggregate_sha256": sha256_file(frontier_dir / "aggregate_metrics.json"),
        "query_count": 1355,
        "row_count": len(rows),
        "runtime_seconds": time.perf_counter() - started,
        "methods": summary,
        "gate_d_promoted": [method for method in GRAPH_METHODS if summary[method]["gate_d_passed"]],
        "native_agent_access_claim": False,
        "full_32_or_128_ranking_persisted": True,
    }
    result["rows_sha256"] = sha256_file(run_dir / "per_query_qwen.jsonl")
    result["aggregate_payload_sha256"] = canonical_sha256(result)
    write_json(run_dir / "aggregate_qwen.json", result)
    print(json.dumps({"status": result["status"], "gate_d_promoted": result["gate_d_promoted"], "runtime_seconds": result["runtime_seconds"]}, indent=2))


def launcher(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    run_dir = create_immutable_run(VERSION_ROOT, args.run_id)
    processes = []
    logs = []
    for shard, device in enumerate(devices):
        log = (run_dir / f"worker_{shard:02d}.log").open("x", encoding="utf-8")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--frontier-run-id", args.frontier_run_id,
            "--run-id", args.run_id,
            "--worker",
            "--shard", str(shard),
            "--num-shards", str(len(devices)),
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = device
        processes.append(subprocess.Popen(command, cwd=VERSION_ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, text=True))
        logs.append(log)
    failures = []
    for shard, process in enumerate(processes):
        code = process.wait()
        logs[shard].close()
        if code:
            failures.append((shard, code))
    if failures:
        raise RuntimeError(f"Qwen workers failed: {failures}")
    aggregate(args, run_dir, started)


def main() -> None:
    args = parse_args()
    if args.worker:
        worker(args)
    else:
        launcher(args)


if __name__ == "__main__":
    main()
