from __future__ import annotations

import argparse
from collections import Counter

from common import VERSION_ROOT, add_common_args, load_config, resolve_project_path
from v24.contracts import EvaluationRecord, MemoryRecord, QueryRecord
from v24.io import iter_jsonl, sha256_file, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()
    config = load_config(args.config)
    root = VERSION_ROOT / "inputs"
    root.mkdir(parents=True, exist_ok=True)
    memory_source = resolve_project_path(config["data"]["memory_source"])
    memories = [MemoryRecord.from_source(row) for row in iter_jsonl(memory_source)]
    if len({memory.memory_id for memory in memories}) != len(memories):
        raise ValueError("duplicate memory_id in source memory file")
    write_jsonl(root / "memory_records.jsonl", (memory.to_mapping() for memory in memories))
    split_sources = {
        "train": config["data"]["train_query_source"],
        "development": config["data"]["development_query_source"],
        "test": config["data"]["test_query_source"],
    }
    split_counts = {}
    split_ids = {}
    for split, source in split_sources.items():
        rows = list(iter_jsonl(resolve_project_path(source)))
        queries = [QueryRecord.from_source(row, split) for row in rows]
        evaluations = [EvaluationRecord.from_source(row) for row in rows]
        if len({query.query_id for query in queries}) != len(queries):
            raise ValueError(f"duplicate query ID in {split}")
        # The query file is deliberately label-free. Gold fields are written separately.
        write_jsonl(root / f"{split}_queries.jsonl", (query.to_mapping() for query in queries))
        write_jsonl(root / f"{split}_evaluation.jsonl", (evaluation.to_mapping() for evaluation in evaluations))
        split_counts[split] = len(queries)
        split_ids[split] = sorted(query.query_id for query in queries)
    overlaps = {}
    names = sorted(split_ids)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            overlap = sorted(set(split_ids[left]) & set(split_ids[right]))
            if overlap:
                overlaps[f"{left}:{right}"] = overlap
    namespace_counts = Counter(memory.namespace_id for memory in memories)
    manifest = {
        "schema_version": "method1-v24-canonical-dataset-1",
        "config_hash": __import__("common").config_hash(config),
        "source_hashes": {name: sha256_file(resolve_project_path(path)) for name, path in {"memory": config["data"]["memory_source"], **{f"{key}_queries": value for key, value in split_sources.items()}}.items()},
        "memory_count": len(memories),
        "namespace_count": len(namespace_counts),
        "namespace_counts": dict(namespace_counts),
        "split_counts": split_counts,
        "query_id_overlaps": overlaps,
        "gold_location": "*_evaluation.jsonl only",
        "selector_excluded_fields": ["evidence_groups", "gold_memory_ids", "gold_answers", "query_type"],
        "causal_fields": {"memory_arrival_order": "session_id", "query_cutoff_order": "query_session_id"},
    }
    write_json(VERSION_ROOT / "artifacts" / "manifests" / "dataset_manifest.json", manifest)
    write_json(VERSION_ROOT / "artifacts" / "manifests" / "split_manifest.json", {"splits": split_ids, "overlaps": overlaps})
    print({"status": "completed", "memory_count": len(memories), "split_counts": split_counts, "overlaps": overlaps})


if __name__ == "__main__":
    main()
