from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from common import (
    PROJECT_ROOT,
    VERSION_ROOT,
    V24_ROOT,
    V28_ROOT,
    canonical_sha256,
    create_immutable_run,
    iter_jsonl,
    load_yaml,
    sha256_file,
    write_json,
    write_jsonl,
)
from src.qwen_reranker import Qwen3SuffixPreservingReranker
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
FRONTIER_RUN_ID = "v28_frontier_oof_20260719_attempt003_913cd3f"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--devices", default="0,1,4,5")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-device", default="cuda:0")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    return parser.parse_args()


def validate_freeze() -> tuple[dict, dict]:
    config_path = VERSION_ROOT / "configs" / "qwen_suffix_preserving.yaml"
    config = load_yaml(config_path)
    receipt_path = VERSION_ROOT / "artifacts" / "frozen" / "qwen_control_freeze_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt["status"] != "FROZEN_BEFORE_CANDIDATE_IDENTICAL_QWEN_OOF":
        raise RuntimeError("invalid corrected-Qwen freeze status")
    if receipt["config_sha256"] != sha256_file(config_path):
        raise RuntimeError("corrected-Qwen config changed after freeze")
    manifest_path = VERSION_ROOT / "artifacts" / "frozen" / "qwen_control_source_manifest.json"
    if receipt["source_manifest_sha256"] != sha256_file(manifest_path):
        raise RuntimeError("corrected-Qwen source manifest changed after freeze")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for row in manifest["files"]:
        path = VERSION_ROOT / row["path"]
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"frozen source changed: {row['path']}")
    frontier_dir = V28_ROOT / "runs" / FRONTIER_RUN_ID
    if receipt["frontier_rows_sha256"] != sha256_file(frontier_dir / "per_query_frontier.jsonl"):
        raise RuntimeError("v28 Frontier rows changed")
    if receipt["frontier_aggregate_sha256"] != sha256_file(frontier_dir / "aggregate_metrics.json"):
        raise RuntimeError("v28 Frontier aggregate changed")
    return config, receipt


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


def paired_bootstrap(values: list[float], *, seed: int, samples: int) -> dict:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 250):
        count = min(250, samples - start)
        indices = rng.integers(0, len(array), size=(count, len(array)))
        estimates[start : start + count] = array[indices].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "ci_95": [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))],
    }


def worker(args: argparse.Namespace) -> None:
    config, receipt = validate_freeze()
    run_dir = VERSION_ROOT / "runs" / args.run_id
    frontier_dir = V28_ROOT / "runs" / FRONTIER_RUN_ID
    rows = list(iter_jsonl(frontier_dir / "per_query_frontier.jsonl"))
    by_query: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_query.setdefault(row["query_id"], {})[row["method"]] = row
    dataset = load_canonical(V24_ROOT / "inputs", split="train")
    query_order = [query.query_id for query in dataset.queries["train"]]
    memory_by_id = {memory.memory_id: memory for memory in dataset.memories}
    query_by_id = {query.query_id: query for query in dataset.queries["train"]}
    model_cfg = config["model"]
    reranker = Qwen3SuffixPreservingReranker(
        PROJECT_ROOT / model_cfg["path"],
        device=args.worker_device,
        dtype=str(model_cfg["dtype"]),
        max_length=int(model_cfg["max_length"]),
        batch_size=int(model_cfg["batch_size"]),
        instruction=str(model_cfg["instruction"]),
        document_head_fraction=float(config["packing"]["document_head_fraction"]),
    )
    smoke = reranker.smoke()
    if not smoke["passed"]:
        raise RuntimeError(f"suffix-preserving CUDA Qwen smoke failed: {smoke}")
    output_rows = []
    for query_index, query_id in enumerate(query_order):
        if query_index % args.num_shards != args.shard:
            continue
        methods = by_query[query_id]
        if set(methods) != set(METHODS):
            raise RuntimeError(f"incomplete Frontier methods for {query_id}")
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
        reranker.torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        score_by_id = dict(zip(union_ids, scored.scores))
        packing_by_id = dict(zip(union_ids, scored.packing))
        evaluation = dataset.evaluations["train"][query_id]
        for method in METHODS:
            frontier_ids = methods[method]["frontier_ids"]
            ranked = sorted(frontier_ids, key=lambda value: (-float(score_by_id[value]), value))
            method_packing = [packing_by_id[value] for value in frontier_ids]
            output_rows.append({
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
                "packing": {
                    "policy": "suffix_preserving_balanced_head_tail",
                    "max_length": int(model_cfg["max_length"]),
                    "generation_suffix_preserved": True,
                    "candidate_count": len(method_packing),
                    "truncated_candidates": sum(row.truncated for row in method_packing),
                    "original_tokens_mean": float(np.mean([row.original_tokens for row in method_packing])),
                    "original_tokens_max": max(row.original_tokens for row in method_packing),
                    "packed_tokens_mean": float(np.mean([row.packed_tokens for row in method_packing])),
                    "packed_tokens_max": max(row.packed_tokens for row in method_packing),
                },
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
                "frontier_rows_sha256": receipt["frontier_rows_sha256"],
                "config_sha256": receipt["config_sha256"],
                "run_id": args.run_id,
            })
        if len(output_rows) % 300 == 0:
            print(f"worker {args.shard}: {len(output_rows) // len(METHODS)} queries", flush=True)
    write_jsonl(run_dir / f"shard_{args.shard:02d}.jsonl", output_rows, exclusive=True)
    write_json(
        run_dir / f"shard_{args.shard:02d}_runtime.json",
        {"shard": args.shard, "num_shards": args.num_shards, "query_count": len(output_rows) // len(METHODS), "smoke": smoke},
        exclusive=True,
    )


def aggregate(args: argparse.Namespace, run_dir: Path, started: float) -> None:
    config, receipt = validate_freeze()
    rows = [row for path in sorted(run_dir.glob("shard_*.jsonl")) for row in iter_jsonl(path)]
    expected = 1355 * len(METHODS)
    if len(rows) != expected:
        raise RuntimeError(f"incomplete corrected-Qwen rows: {len(rows)} != {expected}")
    keys = [(row["query_id"], row["method"]) for row in rows]
    if len(set(keys)) != expected:
        raise RuntimeError("duplicate corrected-Qwen query/method rows")
    frontier = {
        (row["query_id"], row["method"]): row
        for row in iter_jsonl(V28_ROOT / "runs" / FRONTIER_RUN_ID / "per_query_frontier.jsonl")
    }
    identity_violations = sum(
        row["frontier_ids"] != frontier[(row["query_id"], row["method"])]["frontier_ids"]
        for row in rows
    )
    suffix_violations = sum(not row["packing"]["generation_suffix_preserved"] for row in rows)
    if identity_violations or suffix_violations:
        raise RuntimeError("candidate identity or generation suffix invariant failed")
    ordered = sorted(rows, key=lambda row: (row["query_id"], METHODS.index(row["method"])))
    write_jsonl(run_dir / "per_query_qwen.jsonl", ordered, exclusive=True)
    by_method = {method: {row["query_id"]: row for row in rows if row["method"] == method} for method in METHODS}
    dense = by_method["dense_hnsw_32"]
    summary = {}
    limits = config["gate_d_ci_lower"]
    for method in METHODS:
        method_rows = list(by_method[method].values())
        result = {
            "absolute": {
                metric: float(np.mean([row["metrics"][metric] for row in method_rows]))
                for metric in GATE_METRICS + ("mrr_at_10",)
            },
            "rerank_ms_mean_estimated": float(np.mean([row["cost"]["estimated_method_rerank_ms"] for row in method_rows])),
            "input_tokens_mean_estimated": float(np.mean([row["cost"]["estimated_method_input_tokens"] for row in method_rows])),
            "truncated_candidates": int(sum(row["packing"]["truncated_candidates"] for row in method_rows)),
        }
        if method != "dense_hnsw_32":
            result["paired_vs_dense"] = {}
            for metric in GATE_METRICS:
                values = [
                    row["metrics"][metric] - dense[row["query_id"]]["metrics"][metric]
                    for row in method_rows
                ]
                result["paired_vs_dense"][metric] = paired_bootstrap(
                    values,
                    seed=int(config["random_seed"]),
                    samples=int(config["bootstrap_samples"]),
                )
        if method in GRAPH_METHODS:
            result["gate_d_passed"] = all(
                result["paired_vs_dense"][metric]["ci_95"][0] >= float(limits[metric])
                for metric in GATE_METRICS
            )
        summary[method] = result
    result = {
        "schema_version": "method1-v30-candidate-identical-corrected-qwen-oof-1",
        "status": "CORRECTED_QWEN_OOF_COMPLETE",
        "run_id": args.run_id,
        "frontier_run_id": FRONTIER_RUN_ID,
        "frontier_rows_sha256": receipt["frontier_rows_sha256"],
        "frontier_aggregate_sha256": receipt["frontier_aggregate_sha256"],
        "config_sha256": receipt["config_sha256"],
        "freeze_receipt_sha256": sha256_file(VERSION_ROOT / "artifacts" / "frozen" / "qwen_control_freeze_receipt.json"),
        "query_count": 1355,
        "row_count": len(rows),
        "runtime_seconds": time.perf_counter() - started,
        "methods": summary,
        "gate_d_promoted": [method for method in GRAPH_METHODS if summary[method]["gate_d_passed"]],
        "candidate_identity_violations": identity_violations,
        "generation_suffix_violations": suffix_violations,
        "identical_packing_for_all_methods": True,
        "qwen_feedback_to_candidates": False,
        "native_agent_access_claim": False,
        "full_ranking_persisted": True,
        "scientific_scope": "candidate-identical corrected-Qwen train OOF development evidence",
        "fresh_authorized": False,
    }
    result["rows_sha256"] = sha256_file(run_dir / "per_query_qwen.jsonl")
    result["aggregate_payload_sha256"] = canonical_sha256(result)
    write_json(run_dir / "aggregate_qwen.json", result, exclusive=True)
    print(json.dumps({"status": result["status"], "gate_d_promoted": result["gate_d_promoted"], "runtime_seconds": result["runtime_seconds"]}, indent=2))


def launcher(args: argparse.Namespace) -> None:
    validate_freeze()
    started = time.perf_counter()
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    if not devices:
        raise ValueError("at least one CUDA device is required")
    run_dir = create_immutable_run(args.run_id)
    processes = []
    logs = []
    for shard, device in enumerate(devices):
        log = (run_dir / f"worker_{shard:02d}.log").open("x", encoding="utf-8")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--run-id", args.run_id,
            "--worker",
            "--shard", str(shard),
            "--num-shards", str(len(devices)),
        ]
        processes.append(
            subprocess.Popen(
                command + ["--worker-device", f"cuda:{device}"],
                cwd=VERSION_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        )
        logs.append(log)
    failures = []
    for shard, process in enumerate(processes):
        code = process.wait()
        logs[shard].close()
        if code:
            failures.append((shard, code))
    if failures:
        raise RuntimeError(f"corrected-Qwen workers failed: {failures}")
    aggregate(args, run_dir, started)


def main() -> None:
    args = parse_args()
    if args.worker:
        worker(args)
    else:
        launcher(args)


if __name__ == "__main__":
    main()
