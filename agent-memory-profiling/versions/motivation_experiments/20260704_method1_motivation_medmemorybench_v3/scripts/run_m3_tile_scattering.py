#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict

from medmmb_common import load_yaml, read_jsonl, resolve_path, write_csv


def _tile_index(order: list[str], capacity: int) -> dict[str, int]:
    return {memory_id: idx // capacity for idx, memory_id in enumerate(order)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run M3 chronological tile scattering analysis from fresh M2 budget-64 runs.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--m2-log", required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)
    memories = read_jsonl(config["memory_items"])
    m2_rows = read_jsonl(args.m2_log)
    target_budget = int(config.get("target_frontier_budget", 64))
    target_top_k = int(config.get("target_final_top_k", 5))
    seed = int(config.get("seed", 20260704))
    capacities = [int(value) for value in config.get("tile_capacities", [8, 16, 32])]

    by_persona: dict[str, list[dict]] = defaultdict(list)
    for memory in memories:
        by_persona[str(memory["persona_id"])].append(memory)
    for rows in by_persona.values():
        rows.sort(key=lambda row: int(row.get("chronological_rank", 0)))

    orders: dict[tuple[str, str, int], dict[str, int]] = {}
    for persona, rows in by_persona.items():
        chronological = [str(row["memory_id"]) for row in rows]
        random_order = list(chronological)
        random.Random(seed + len(persona) + len(random_order)).shuffle(random_order)
        for capacity in capacities:
            orders[(persona, "chronological", capacity)] = _tile_index(chronological, capacity)
            orders[(persona, "random_per_persona", capacity)] = _tile_index(random_order, capacity)

    focused = [
        row for row in m2_rows
        if int(row.get("frontier_budget", -1)) == target_budget
        and int(row.get("final_top_k", -1)) == target_top_k
        and not row.get("error")
        and row.get("final_selected_ids")
    ]
    if not focused:
        raise RuntimeError("M3 requires fresh M2 rows at frontier_budget=64 and final_top_k=5.")

    output = []
    for row in focused:
        selected = [str(mid) for mid in row.get("final_selected_ids", [])]
        if len(selected) != target_top_k:
            continue
        persona = str(row["persona_id"])
        query_type = str(row.get("query_type", "unknown"))
        for capacity in capacities:
            ideal_tiles = int(math.ceil(len(selected) / float(capacity)))
            for layout in ["chronological", "random_per_persona", "oracle_tile_packing_lower_bound"]:
                if layout == "oracle_tile_packing_lower_bound":
                    activated = ideal_tiles
                else:
                    index = orders[(persona, layout, capacity)]
                    missing = [mid for mid in selected if mid not in index]
                    if missing:
                        raise RuntimeError(f"Selected memories missing from layout index for query {row['query_id']}: {missing[:3]}")
                    activated = len({index[mid] for mid in selected})
                utilization = len(selected) / float(capacity * max(activated, 1))
                access_amp = (capacity * activated) / float(len(selected))
                scatter = activated / float(max(ideal_tiles, 1))
                output.append(
                    {
                        "query_id": row["query_id"],
                        "persona_id": persona,
                        "query_type": query_type,
                        "layout": layout,
                        "tile_capacity": capacity,
                        "target_k": len(selected),
                        "activated_tiles": activated,
                        "ideal_tiles": ideal_tiles,
                        "tile_utilization": utilization,
                        "access_amplification": access_amp,
                        "scatter_ratio": scatter,
                        "source_m2_run_id": row.get("run_id", ""),
                        "source_frontier_budget": target_budget,
                    }
                )
    if not output:
        raise RuntimeError("M3 produced no scattering rows.")
    write_csv(config.get("output_csv", "aggregate/M3_tile_scattering.csv"), output)
    print(f"m3_csv={resolve_path(config.get('output_csv', 'aggregate/M3_tile_scattering.csv'))}")


if __name__ == "__main__":
    main()
