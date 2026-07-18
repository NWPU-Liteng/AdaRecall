from __future__ import annotations

import argparse
import json

from common import VERSION_ROOT, add_common_args, config_hash, load_config, resolve_project_path
from v24.contracts import HistoryEvent, MemoryRecord, QueryRecord
from v24.io import iter_jsonl, sha256_file, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--max-event-size", type=int, default=32)
    args = parser.parse_args()
    config = load_config(args.config)
    memory_rows = {record.memory_id: record for record in (MemoryRecord.from_mapping(row) for row in iter_jsonl(VERSION_ROOT / "inputs" / "memory_records.jsonl"))}
    train_queries = {query.query_id: query for query in (QueryRecord.from_mapping(row) for row in iter_jsonl(VERSION_ROOT / "inputs" / "train_queries.jsonl"))}
    source = resolve_project_path(config["data"]["frozen_train_retrieval"])
    events = []
    seen_queries = set()
    dropped = 0
    for row in iter_jsonl(source):
        if str(row.get("method", "")) != "flat":
            continue
        query_id = str(row["query_id"])
        query = train_queries.get(query_id)
        if query is None or query_id in seen_queries:
            continue
        seen_queries.add(query_id)
        ids = []
        for memory_id in row.get("ann_candidate_ids", []) or []:
            memory_id = str(memory_id)
            memory = memory_rows.get(memory_id)
            if memory is None or memory.namespace_id != query.namespace_id or memory.arrival_order > query.cutoff_order:
                continue
            if memory_id not in ids:
                ids.append(memory_id)
            if len(ids) >= int(args.max_event_size):
                break
        if len(ids) < 2:
            dropped += 1
            continue
        events.append(HistoryEvent(
            event_id=f"co_candidate:{query_id}", namespace_id=query.namespace_id, query_id=query_id,
            query_embedding_id=query_id, selected_memory_ids=tuple(ids), event_kind="co_candidate",
            task_success=None, arrival_order=query.cutoff_order,
        ))
    output = VERSION_ROOT / "inputs" / "history_events.jsonl"
    write_jsonl(output, (event.to_mapping() for event in events))
    summary = {
        "schema_version": "method1-v24-history-events-1", "config_hash": config_hash(config),
        "source": str(source), "source_sha256": sha256_file(source), "event_count": len(events),
        "queries_seen": len(seen_queries), "dropped_short_or_ineligible": dropped, "event_kind": "co_candidate",
        "gold_used": False, "test_used": False, "native_agent_read_trace": False,
    }
    write_json(VERSION_ROOT / "artifacts" / "manifests" / "history_event_manifest.json", summary)
    print(json.dumps({"status": "completed", **summary}, sort_keys=True))


if __name__ == "__main__":
    main()
