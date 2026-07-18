from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from common import PROJECT_ROOT, VERSION_ROOT, add_common_args, artifact_root, load_config
from v24.contracts import HistoryEvent
from v24.dataset import load_canonical
from v24.io import iter_jsonl, write_json, write_jsonl
from v24.pipeline import Method1Runner


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--index-run-id", default="v24_index_20260718")
    parser.add_argument("--limit", type=int, default=128)
    args = parser.parse_args()
    if not args.run_id:
        args.run_id = "v24_agent_replay_20260718"
    frozen = VERSION_ROOT / "artifacts" / "frozen" / "development_config.json"
    if not frozen.exists():
        status = {"status": "NOT_RUN", "reason": "method_not_promoted", "task_success_available": False}
        write_json(VERSION_ROOT / "negative_results" / "agent_replay_not_run.json", status)
        print(json.dumps(status, sort_keys=True))
        return
    frozen_config = __import__("v24.io", fromlist=["read_json"]).read_json(frozen)
    config = frozen_config["resolved_config"]
    config["reranker"] = dict(config["reranker"])
    config["reranker"]["model_dir"] = str(PROJECT_ROOT / config["model"]["reranker_model_dir"])
    config["reranker"]["device"] = args.device
    data = load_canonical(VERSION_ROOT / "inputs")
    runner = Method1Runner(data, artifact_root(args.index_run_id), config, reranker_backend="qwen3")
    query_by_id = {query.query_id: query for query in data.queries["train"]}
    events = [HistoryEvent(
        event_id=str(row["event_id"]), namespace_id=str(row["namespace_id"]), query_id=str(row["query_id"]),
        query_embedding_id=str(row["query_embedding_id"]), selected_memory_ids=tuple(str(value) for value in row["selected_memory_ids"]),
        event_kind=str(row["event_kind"]), task_success=row.get("task_success"), arrival_order=int(row["arrival_order"]),
    ) for row in iter_jsonl(VERSION_ROOT / "inputs" / "history_events.jsonl")]
    rows = []
    for event in events[: int(args.limit) if args.limit else None]:
        query = query_by_id.get(event.query_id)
        if query is None:
            continue
        by_method = {}
        for method in ("full_n", "dense_c", "ours_c"):
            result = runner.run_query(
                query, method=method, run_id=args.run_id, full_n=int(config["budget"]["full_n"]), seed_s=int(config["budget"]["seed_s"]),
                frontier_c=int(config["budget"]["frontier_c"]), final_k=int(config["budget"]["final_k"]),
            )
            reference = set(event.selected_memory_ids)
            by_method[method] = {
                "final_ids": result.reranked_ids, "frontier_ids": result.frontier_ids,
                "reference_agreement_final": float(len(set(result.reranked_ids) & reference) / max(len(reference), 1)),
                "reference_agreement_frontier": float(len(set(result.frontier_ids) & reference) / max(len(reference), 1)),
                "reranker_pairs": result.reranker_pairs, "semantic_selection_ms": result.semantic_selection_ms,
            }
        rows.append({"event_id": event.event_id, "query_id": event.query_id, "namespace_id": event.namespace_id, "event_kind": event.event_kind, "reference_size": len(event.selected_memory_ids), "methods": by_method})
    output = VERSION_ROOT / "runs" / args.run_id
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "agent_replay_rows.jsonl", rows)
    summary = {"status": "COMPLETED", "run_id": args.run_id, "tasks": len(rows), "task_success_available": False, "methods": {}}
    for method in ("full_n", "dense_c", "ours_c"):
        values = [row["methods"][method] for row in rows]
        summary["methods"][method] = {
            "reference_agreement_final": float(np.mean([value["reference_agreement_final"] for value in values])) if values else None,
            "reference_agreement_frontier": float(np.mean([value["reference_agreement_frontier"] for value in values])) if values else None,
            "semantic_selection_ms_sum": float(np.sum([value["semantic_selection_ms"] for value in values])) if values else 0.0,
            "reranker_pairs_sum": int(np.sum([value["reranker_pairs"] for value in values])) if values else 0,
        }
    summary["quality_non_decreasing_vs_full"] = bool(
        summary["methods"]["ours_c"]["reference_agreement_final"] is not None
        and summary["methods"]["ours_c"]["reference_agreement_final"] >= summary["methods"]["full_n"]["reference_agreement_final"] - float(config["gate"]["full_quality_epsilon"])
    )
    write_json(VERSION_ROOT / "aggregate" / "agent_replay_summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
