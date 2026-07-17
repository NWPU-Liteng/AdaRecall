#!/usr/bin/env python3
"""Derive M3 target-k sensitivity from the existing B=64 reranked lists."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT.parents[0] / "20260704_method1_motivation_medmemorybench_v3"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _tile_index(order: list[str], capacity: int) -> dict[str, int]:
    return {memory_id: idx // capacity for idx, memory_id in enumerate(order)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m2-log", default=str(V3 / "aggregate/M2_frontier_budget_invocations.jsonl"))
    parser.add_argument("--output", default=str(ROOT / "aggregate/M3_target_k_sensitivity.csv"))
    args = parser.parse_args()
    memories = _read_jsonl(V3 / "aggregate/memory_items_turn_pair.jsonl")
    m2_rows = _read_jsonl(Path(args.m2_log))
    by_persona: dict[str, list[dict]] = defaultdict(list)
    for memory in memories:
        by_persona[str(memory["persona_id"])].append(memory)
    orders: dict[tuple[str, str, int], dict[str, int]] = {}
    for persona, rows in by_persona.items():
        rows.sort(key=lambda row: int(row.get("chronological_rank", 0)))
        chronological = [str(row["memory_id"]) for row in rows]
        random_order = list(chronological)
        random.Random(20260704 + len(persona) + len(random_order)).shuffle(random_order)
        for capacity in [8, 16, 32]:
            orders[(persona, "chronological", capacity)] = _tile_index(chronological, capacity)
            orders[(persona, "random_per_persona", capacity)] = _tile_index(random_order, capacity)
    output: list[dict] = []
    focused = [row for row in m2_rows if int(row.get("frontier_budget", -1)) == 64 and not row.get("error") and row.get("reranked_candidate_ids")]
    for row in focused:
        reranked = [str(mid) for mid in row["reranked_candidate_ids"]]
        persona = str(row["persona_id"])
        for target_k in [5, 10, 20]:
            selected = reranked[:target_k]
            for capacity in [8, 16, 32]:
                ideal_tiles = int(math.ceil(target_k / float(capacity)))
                for layout in ["chronological", "random_per_persona", "oracle_tile_packing_lower_bound"]:
                    if layout == "oracle_tile_packing_lower_bound":
                        activated = ideal_tiles
                    else:
                        index = orders[(persona, layout, capacity)]
                        activated = len({index[mid] for mid in selected})
                    output.append({
                        "query_id": row["query_id"],
                        "persona_id": persona,
                        "query_type": row.get("query_type", "unknown"),
                        "target_k": target_k,
                        "tile_capacity": capacity,
                        "layout": layout,
                        "activated_tiles": activated,
                        "ideal_tiles": ideal_tiles,
                        "tile_utilization": target_k / float(capacity * max(activated, 1)),
                        "access_amplification": (capacity * activated) / float(target_k),
                        "scatter_ratio": activated / float(max(ideal_tiles, 1)),
                        "source_frontier_budget": 64,
                        "source_m2_run_id": row.get("run_id", ""),
                    })
    if not output:
        raise RuntimeError("No B=64 reranked rows found")
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(json.dumps({"rows": len(output), "queries": len(focused), "output": str(path)}, indent=2))


if __name__ == "__main__":
    main()
