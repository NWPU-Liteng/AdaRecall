from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import PROJECT_ROOT, VERSION_ROOT, add_common_args, artifact_root, config_hash, load_config
from v24.dataset import load_canonical
from v24.io import write_json, write_jsonl
from v24.pipeline import Method1Runner, run_query_matrix


def _means(rows: list[dict], key: str) -> dict[str, float | None]:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (float, int))]
    if not values:
        return {"mean": None, "median": None, "count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "median": float(np.median(array)), "count": len(values)}


def _paired_gate(rows: list[dict], config: dict, backend: str) -> dict:
    grouped = defaultdict(dict)
    for row in rows:
        grouped[row["query_id"]][row["method"]] = row
    pairs = [value for value in grouped.values() if {"full_n", "dense_c", "ours_c"} <= set(value)]
    epsilon_dense = float(config["gate"]["dense_quality_epsilon"])
    epsilon_full = float(config["gate"]["full_quality_epsilon"])
    quality_keys = ["ndcg_at_10", "precision_at_k", "evidence_group_coverage_final", "candidate_recall_final"]
    deltas = {}
    for key in quality_keys:
        dense_values = [float(pair["ours_c"]["metrics"][key]) - float(pair["dense_c"]["metrics"][key]) for pair in pairs if pair["ours_c"]["metrics"].get(key) is not None and pair["dense_c"]["metrics"].get(key) is not None]
        full_values = [float(pair["ours_c"]["metrics"][key]) - float(pair["full_n"]["metrics"][key]) for pair in pairs if pair["ours_c"]["metrics"].get(key) is not None and pair["full_n"]["metrics"].get(key) is not None]
        deltas[key] = {"ours_minus_dense": _means([{"value": value} for value in dense_values], "value"), "ours_minus_full": _means([{"value": value} for value in full_values], "value")}
    ours = [pair["ours_c"] for pair in pairs]
    full = [pair["full_n"] for pair in pairs]
    dense = [pair["dense_c"] for pair in pairs]
    net_gain = [float(left["semantic_selection_ms"]) - float(right["semantic_selection_ms"]) for left, right in zip(full, ours)]
    safety_passed = not any(
        bool(row["metrics"].get("cross_namespace_violation") or row["metrics"].get("future_memory_violation") or row["metrics"].get("duplicate_frontier_ids"))
        for row in rows
    )
    candidate_gate = (
        np.mean([float(row["metrics"]["candidate_recall_frontier"]) for row in ours])
        >= np.mean([float(row["metrics"]["candidate_recall_frontier"]) for row in dense]) - epsilon_dense
        if ours and dense else False
    )
    positive_quality = any(
        (deltas[key]["ours_minus_dense"]["mean"] is not None and deltas[key]["ours_minus_dense"]["mean"] > 0.0)
        for key in quality_keys
    )
    quality_non_negative = all(
        deltas[key]["ours_minus_dense"]["mean"] is not None and deltas[key]["ours_minus_dense"]["mean"] >= -epsilon_dense
        for key in ("ndcg_at_10", "precision_at_k")
    ) if pairs else False
    full_non_inferior = all(
        deltas[key]["ours_minus_full"]["mean"] is not None and deltas[key]["ours_minus_full"]["mean"] >= -epsilon_full
        for key in ("ndcg_at_10",)
    ) if pairs else False
    cost = {
        "full_semantic_selection_ms": _means([{"value": row["semantic_selection_ms"]} for row in full], "value"),
        "ours_semantic_selection_ms": _means([{"value": row["semantic_selection_ms"]} for row in ours], "value"),
        "dense_semantic_selection_ms": _means([{"value": row["semantic_selection_ms"]} for row in dense], "value"),
        "net_gain_ms": _means([{"value": value} for value in net_gain], "value"),
        "candidate_reduction": 1.0 - float(config["budget"]["frontier_c"]) / float(config["budget"]["full_n"]),
    }
    cost_passed = bool(
        cost["net_gain_ms"]["mean"] is not None and cost["net_gain_ms"]["mean"] > 0.0
        and cost["ours_semantic_selection_ms"]["mean"] is not None
        and cost["full_semantic_selection_ms"]["mean"] is not None
        and cost["ours_semantic_selection_ms"]["mean"] < cost["full_semantic_selection_ms"]["mean"]
    )
    passed = bool(backend == "qwen3" and safety_passed and candidate_gate and quality_non_negative and positive_quality and cost_passed)
    return {
        "stage": "development", "backend": backend, "query_count": len(pairs), "config_hash": config_hash(config),
        "safety_passed": safety_passed, "candidate_gate_passed": bool(candidate_gate),
        "quality_non_negative": bool(quality_non_negative), "positive_quality_signal": bool(positive_quality),
        "full_non_inferior": bool(full_non_inferior), "cost_passed": cost_passed, "passed": passed,
        "deltas": deltas, "cost": cost, "epsilon_dense": epsilon_dense, "epsilon_full": epsilon_full,
        "test_read_status": "not_read",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--index-run-id", default="v24_index_20260718")
    parser.add_argument("--reranker-backend", choices=["qwen3", "dense_reference"], default="qwen3")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if not args.run_id:
        args.run_id = "v24_development_20260718"
    config = load_config(args.config)
    runtime = dict(config)
    runtime["reranker"] = dict(config["reranker"])
    runtime["reranker"]["model_dir"] = str(PROJECT_ROOT / config["model"]["reranker_model_dir"])
    runtime["reranker"]["device"] = args.device
    data = load_canonical(VERSION_ROOT / "inputs", split=None)
    queries = data.queries["development"]
    queries = queries[args.offset: args.offset + args.limit if args.limit else None]
    runner = Method1Runner(data, artifact_root(args.index_run_id), runtime, reranker_backend=args.reranker_backend)
    results = run_query_matrix(
        runner, queries, run_id=args.run_id, full_n=int(config["budget"]["full_n"]), seed_s=int(config["budget"]["seed_s"]),
        frontier_c=int(config["budget"]["frontier_c"]), final_k=int(config["budget"]["final_k"]),
    )
    output = VERSION_ROOT / "runs" / args.run_id
    output.mkdir(parents=True, exist_ok=True)
    rows = [result.to_mapping() for result in results]
    write_jsonl(output / "query_results.jsonl", rows)
    gate = _paired_gate(rows, config, args.reranker_backend)
    gate["run_id"] = args.run_id
    gate["index_run_id"] = args.index_run_id
    write_json(VERSION_ROOT / "aggregate" / "development_gate.json", gate)
    write_json(VERSION_ROOT / "aggregate" / "development_summary.json", {"run_id": args.run_id, "rows": len(rows), "methods": {method: _means([row["metrics"] for row in rows if row["method"] == method], "ndcg_at_10") for method in ("full_n", "dense_c", "ours_c")}, "gate": gate})
    report = [
        "# Development Gate",
        "",
        f"- run: `{args.run_id}`",
        f"- backend: `{args.reranker_backend}`",
        f"- query rows: `{len(queries)}` queries / `{len(rows)}` method rows",
        f"- passed: `{gate['passed']}`",
        "",
        "The development run is the only place where this configuration may be assessed. The test split was not read by this script.",
        "",
        "```json",
        json.dumps(gate, ensure_ascii=True, indent=2, sort_keys=True),
        "```",
    ]
    (VERSION_ROOT / "reports" / "DEVELOPMENT_GATE.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "run_id": args.run_id, "gate_passed": gate["passed"], "backend": args.reranker_backend}, sort_keys=True))


if __name__ == "__main__":
    main()
