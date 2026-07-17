#!/usr/bin/env python3
"""Stress a generic append-only text index under deterministic content growth."""
from __future__ import annotations

import csv
import hashlib
import heapq
import json
import re
from collections import defaultdict
from pathlib import Path
from time import perf_counter_ns

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
AMP_ROOT = ROOT.parents[1]
V3 = AMP_ROOT / "versions" / "20260704_method1_motivation_medmemorybench_v3"


TOKEN_RE = re.compile(r"[a-z0-9]+")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(sorted(set(TOKEN_RE.findall(text.lower()))))


def build_index(rows: list[dict], token_map: dict[str, tuple[str, ...]]) -> tuple[dict[str, set[str]], int]:
    index: dict[str, set[str]] = defaultdict(set)
    postings = 0
    for row in rows:
        memory_id = str(row["memory_id"])
        for token in token_map[memory_id]:
            index[token].add(memory_id)
            postings += 1
    return dict(index), postings


def append_to_index(index: dict[str, set[str]], rows: list[dict], token_map: dict[str, tuple[str, ...]]) -> int:
    additions = 0
    for row in rows:
        memory_id = str(row["memory_id"])
        for token in token_map[memory_id]:
            bucket = index.setdefault(token, set())
            if memory_id not in bucket:
                bucket.add(memory_id)
                additions += 1
    return additions


def lookup(index: dict[str, set[str]], terms: tuple[str, ...], rank: dict[str, int], top_k: int) -> list[str]:
    scores: dict[str, int] = defaultdict(int)
    for token in terms:
        for memory_id in index.get(token, ()):
            scores[memory_id] += 1
    return heapq.nsmallest(top_k, scores, key=lambda memory_id: (-scores[memory_id], rank[memory_id], memory_id))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("no S2 rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    protocol = json.loads((ROOT / "configs" / "v5_protocol.json").read_text(encoding="utf-8"))
    memory_path = V3 / "aggregate" / "memory_items_turn_pair.jsonl"
    query_path = V3 / "aggregate" / "queries_medmemorybench_en.jsonl"
    memories = read_jsonl(memory_path)
    queries = read_jsonl(query_path)
    if not memories or not queries:
        raise RuntimeError("S2 input data is empty")
    token_map = {str(row["memory_id"]): tokenize(str(row.get("text", ""))) for row in memories}
    ordered = sorted(memories, key=lambda row: (int(row.get("chronological_rank", 0)), str(row["memory_id"])))
    rank = {str(row["memory_id"]): idx for idx, row in enumerate(ordered)}
    base_count = int(round(len(ordered) * float(protocol["s2"]["base_fraction"])))
    if not 0 < base_count < len(ordered):
        raise RuntimeError("invalid S2 base fraction")
    growth_points = [int(value) for value in protocol["s2"]["growth_percent_of_full_corpus"]]
    probe_count = min(int(protocol["s2"]["probe_queries"]), len(queries))
    probe_positions = np.linspace(0, len(queries) - 1, num=probe_count, dtype=np.int64)
    probes = [(queries[int(position)], tokenize(str(queries[int(position)].get("query_text", "")))) for position in probe_positions]
    top_k = int(protocol["s2"]["lookup_top_k"])
    tile_capacity = int(protocol["s2"]["logical_tile_capacity"])
    tile_by_memory = {str(row["memory_id"]): rank[str(row["memory_id"])] // tile_capacity for row in ordered}

    base_rows = ordered[:base_count]
    incremental, base_postings = build_index(base_rows, token_map)
    appended_count = 0
    results: list[dict] = []
    for growth in growth_points:
        target = min(len(ordered) - base_count, int(round(len(ordered) * growth / 100.0)))
        if target < appended_count:
            raise RuntimeError("growth points must be increasing")
        append_rows = ordered[base_count + appended_count: base_count + target]
        start = perf_counter_ns()
        changed_postings = append_to_index(incremental, append_rows, token_map)
        incremental_ms = (perf_counter_ns() - start) / 1_000_000.0
        appended_count = target
        current_rows = ordered[:base_count + appended_count]
        start = perf_counter_ns()
        rebuilt, full_postings = build_index(current_rows, token_map)
        full_rebuild_ms = (perf_counter_ns() - start) / 1_000_000.0
        mismatch_count = 0
        incremental_activated = []
        rebuilt_activated = []
        for query, query_terms in probes:
            left = lookup(incremental, query_terms, rank, top_k)
            right = lookup(rebuilt, query_terms, rank, top_k)
            mismatch_count += int(left != right)
            incremental_activated.append(len({tile_by_memory[memory_id] for memory_id in left}))
            rebuilt_activated.append(len({tile_by_memory[memory_id] for memory_id in right}))
        results.append({
            "growth_percent_of_full_corpus": growth,
            "base_memory_count": base_count,
            "appended_memory_count": appended_count,
            "active_memory_count": len(current_rows),
            "base_postings": base_postings,
            "changed_postings": changed_postings,
            "full_rebuild_postings": full_postings,
            "changed_posting_fraction_of_full": changed_postings / float(max(full_postings, 1)),
            "incremental_update_ms": incremental_ms,
            "full_rebuild_ms": full_rebuild_ms,
            "full_to_incremental_time_ratio": full_rebuild_ms / max(incremental_ms, 1e-9),
            "probe_count": len(probes),
            "lookup_top_k": top_k,
            "lookup_mismatch_count": mismatch_count,
            "incremental_probe_activated_tiles_mean": float(np.mean(incremental_activated)),
            "full_rebuild_probe_activated_tiles_mean": float(np.mean(rebuilt_activated)),
            "new_payload_tile_count": (len(current_rows) + tile_capacity - 1) // tile_capacity - (base_count + tile_capacity - 1) // tile_capacity,
            "tile_capacity": tile_capacity,
        })

    output = ROOT / "aggregate" / "v5_s2_generic_content_maintenance.csv"
    write_csv(output, results)
    result = {
        "schema_version": "motivation-v5-s2-result-1",
        "status": "completed",
        "memory_count": len(ordered),
        "base_memory_count": base_count,
        "growth_points": growth_points,
        "probe_count": len(probes),
        "ordering_role": protocol["s2"]["ordering_role"],
        "index_type": "plain_token_to_memory_inverted_index",
        "lookup_exactness": all(row["lookup_mismatch_count"] == 0 for row in results),
        "method_outputs_read": False,
        "stage_inputs_read": False,
        "calibration_read": False,
        "formal_test_read": False,
        "causal_stream_claim": False,
        "access_order_claim": False,
        "co_access_claim": False,
        "pim_hardware_claim": False,
        "artifacts": {"metrics": str(output.relative_to(ROOT))},
        "hashes": {
            "protocol": sha256(ROOT / "configs" / "v5_protocol.json"),
            "metrics": sha256(output),
            "memory_items": sha256(memory_path),
            "queries": sha256(query_path),
        },
    }
    result_path = ROOT / "aggregate" / "v5_s2_generic_content_maintenance_summary.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
