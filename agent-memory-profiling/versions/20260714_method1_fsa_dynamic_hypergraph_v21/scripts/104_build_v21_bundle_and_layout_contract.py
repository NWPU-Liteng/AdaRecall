#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import sys
from itertools import product
from pathlib import Path


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.io import iter_jsonl, load_config, read_json, sha256_file, write_json_new  # noqa: E402


def main() -> None:
    data = load_config("v21_data.json")
    closure = load_config("v21_effect_closure.json")
    memory_rows = list(iter_jsonl(data["memory"]["items"]))
    embedding_ids = list(map(str, read_json(data["memory"]["embedding_ids"])))
    memory_ids = [str(row["memory_id"]) for row in memory_rows]
    if len(memory_ids) != 42034 or len(set(memory_ids)) != len(memory_ids) or set(memory_ids) != set(embedding_ids):
        raise RuntimeError("v21 neutral layout requires exact 42,034-memory ID coverage")
    if any(int(row.get("payload_bytes", -1)) < 0 for row in memory_rows):
        raise RuntimeError("every memory must have nonnegative actual payload_bytes")

    capacity = int(closure["memory_layout"]["tile_capacity_memories"])
    layout_path = VERSION_ROOT / "artifacts" / "v21_effect_closure" / "v21_neutral_memory_to_tile.csv"
    layout_path.parent.mkdir(parents=True, exist_ok=True)
    with layout_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["memory_id", "tile_id", "source_position", "payload_bytes"])
        writer.writeheader()
        for position, row in enumerate(memory_rows):
            writer.writerow({"memory_id": row["memory_id"], "tile_id": f"v21_neutral_t{position // capacity}", "source_position": position, "payload_bytes": int(row["payload_bytes"])})

    bundle_path = VERSION_ROOT / "artifacts" / "v21_evidence_bundles" / "v21_train_acceptable_evidence_bundles.jsonl"
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    query_counts, bundle_count, incidence_count = {}, 0, 0
    with bundle_path.open("x", encoding="utf-8") as handle:
        for row in iter_jsonl(data["split"]["train_query_plan"]):
            groups = [list(dict.fromkeys(map(str, group["acceptable_memory_ids"]))) for group in row.get("evidence_groups", [])]
            bundles = list(product(*groups)) if len(groups) >= 2 else []
            query_counts[str(len(groups))] = query_counts.get(str(len(groups)), 0) + 1
            for bundle_index, members in enumerate(bundles):
                record = {"query_id": str(row["query_id"]), "persona_id": str(row["persona_id"]), "query_session_id": int(row["query_session_id"]), "group_count": len(groups), "bundle_id": f"{row['query_id']}::bundle_{bundle_index}", "memory_ids": list(members), "semantics": "one_alternative_per_distinct_evidence_group", "train_supervision_only": True}
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                bundle_count += 1
                incidence_count += len(members)

    result = {
        "schema_version": "v21-bundle-layout-contract-1",
        "status": "completed",
        "memories": len(memory_ids),
        "tile_capacity": capacity,
        "tiles": (len(memory_ids) + capacity - 1) // capacity,
        "payload_bytes_total": sum(int(row["payload_bytes"]) for row in memory_rows),
        "train_query_group_counts": query_counts,
        "multi_group_queries": sum(value for key, value in query_counts.items() if int(key) >= 2),
        "acceptable_bundle_hyperedges": bundle_count,
        "bundle_memory_incidences": incidence_count,
        "bundle_semantics": "one OR-alternative representative from every distinct group",
        "runtime_gold_bundle_access": False,
        "artifacts": {
            "layout": str(layout_path.relative_to(VERSION_ROOT)),
            "bundles": str(bundle_path.relative_to(VERSION_ROOT))
        },
        "hashes": {
            "layout": sha256_file(layout_path),
            "bundles": sha256_file(bundle_path),
            "config": sha256_file(VERSION_ROOT / "configs" / "v21_effect_closure.json")
        },
        "calibration_read": False,
        "formal_test_read": False
    }
    write_json_new(VERSION_ROOT / "aggregate" / "v21_bundle_layout_contract.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
