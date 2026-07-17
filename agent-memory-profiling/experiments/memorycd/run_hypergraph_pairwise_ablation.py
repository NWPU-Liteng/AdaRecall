#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FORMAL_DIR = PROJECT_ROOT / "versions" / "v0.3" / "formal_run"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "versions" / "v0.4" / "hypergraph_pairwise_ablation"


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def paired_bootstrap_ci(deltas: list[float], *, seed: int, rounds: int = 2000) -> dict[str, float]:
    if not deltas:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = random.Random(seed)
    samples = []
    n = len(deltas)
    for _ in range(rounds):
        samples.append(mean([deltas[rng.randrange(n)] for _i in range(n)]))
    samples.sort()
    return {
        "mean": mean(deltas),
        "ci_low": samples[int(0.025 * (len(samples) - 1))],
        "ci_high": samples[int(0.975 * (len(samples) - 1))],
    }


def sign_test_p_value(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    tail = min(wins, losses)
    # Two-sided exact binomial under p=0.5.
    prob = sum(math.comb(n, k) for k in range(tail + 1)) / (2**n)
    return min(1.0, 2.0 * prob)


def build_train_structure(train_events: list[dict[str, Any]]):
    pairwise_edges: dict[tuple[str, str], int] = defaultdict(int)
    canonical_counts: dict[tuple[str, ...], int] = defaultdict(int)
    for event in train_events:
        selected = sorted(set(event.get("final_selected_ids") or []))
        if len(selected) < 2:
            continue
        canonical_counts[tuple(selected)] += 1
        for i, left in enumerate(selected):
            for right in selected[i + 1 :]:
                pairwise_edges[(left, right)] += 1
    hyperedges = [(members, count) for members, count in canonical_counts.items() if len(members) >= 2]
    incident: dict[str, list[tuple[tuple[str, ...], int]]] = defaultdict(list)
    adjacency: dict[str, dict[str, int]] = defaultdict(dict)
    for (left, right), weight in pairwise_edges.items():
        adjacency[left][right] = weight
        adjacency[right][left] = weight
    for members, weight in hyperedges:
        for memory_id in members:
            incident[memory_id].append((members, weight))
    return dict(pairwise_edges), hyperedges, dict(incident), dict(adjacency)


def compose_hybrid(*, dense_anchor: list[str], ranked: list[str], budget: int) -> list[str]:
    out: list[str] = []
    seen = set()
    for memory_id in dense_anchor:
        if memory_id not in seen:
            seen.add(memory_id)
            out.append(memory_id)
        if len(out) >= budget:
            return out
    for memory_id in ranked:
        if memory_id not in seen:
            seen.add(memory_id)
            out.append(memory_id)
        if len(out) >= budget:
            return out
    return out


def rank_pairwise_candidates(
    *,
    seeds: list[str],
    visible: set[str],
    adjacency: dict[str, dict[str, int]],
    dense_rank: dict[str, int],
) -> tuple[list[str], int]:
    scores: dict[str, float] = defaultdict(float)
    for seed in seeds:
        for candidate, weight in adjacency.get(seed, {}).items():
            if candidate in visible and candidate not in seeds:
                scores[candidate] += float(weight)
    ranked = sorted(scores, key=lambda mid: (-scores[mid], dense_rank.get(mid, 10**9), mid))
    return ranked, len(scores)


def rank_hypergraph_candidates(
    *,
    seeds: list[str],
    visible: set[str],
    incident: dict[str, list[tuple[tuple[str, ...], int]]],
    dense_rank: dict[str, int],
    require_multi_seed: bool,
) -> tuple[list[str], int, dict[str, float]]:
    seed_set = set(seeds)
    activated_edges: dict[tuple[str, ...], tuple[int, int]] = {}
    for seed in seeds:
        for members, weight in incident.get(seed, []):
            overlap = len(seed_set & set(members))
            if overlap <= 0:
                continue
            if require_multi_seed and overlap < 2:
                continue
            previous = activated_edges.get(members)
            if previous is None or overlap > previous[1]:
                activated_edges[members] = (weight, overlap)

    scores: dict[str, float] = defaultdict(float)
    for members, (weight, overlap) in activated_edges.items():
        members_set = set(members)
        edge_size = max(2, len(members))
        denom = max(1, min(len(seed_set), edge_size - 1))
        high_order_gain = (overlap / denom) ** 2
        for candidate in members_set:
            if candidate in visible and candidate not in seed_set:
                scores[candidate] += float(weight) * high_order_gain * math.log2(edge_size + 1)
    ranked = sorted(scores, key=lambda mid: (-scores[mid], dense_rank.get(mid, 10**9), mid))
    diagnostics = {
        "activated_hyperedges": float(len(activated_edges)),
        "multi_seed_activated_hyperedges": float(sum(1 for _members, (_weight, overlap) in activated_edges.items() if overlap >= 2)),
        "hypergraph_candidate_count": float(len(scores)),
    }
    return ranked, len(scores), diagnostics


def recall_at_budget(selected: list[str], reference: set[str]) -> float:
    if not reference:
        return 0.0
    return len(set(selected) & reference) / len(reference)


def chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[idx : idx + size] for idx in range(0, len(items), size)]


def build_pairwise_layout(
    *,
    train_memory_ids: list[str],
    delta_memory_ids: list[str],
    capacity: int,
    adjacency: dict[str, dict[str, int]],
) -> dict[str, str]:
    degree = {memory_id: sum(adjacency.get(memory_id, {}).values()) for memory_id in train_memory_ids}
    unassigned = set(train_memory_ids)
    tiles: list[list[str]] = []
    while unassigned:
        seed = max(unassigned, key=lambda mid: (degree.get(mid, 0), mid))
        tile = [seed]
        unassigned.remove(seed)
        while unassigned and len(tile) < capacity:
            choice = max(
                unassigned,
                key=lambda mid: (
                    sum(adjacency.get(mid, {}).get(member, 0) for member in tile),
                    degree.get(mid, 0),
                    mid,
                ),
            )
            tile.append(choice)
            unassigned.remove(choice)
        tiles.append(tile)
    tiles.extend(chunked(delta_memory_ids, capacity))
    return {memory_id: f"tile_{idx:04d}" for idx, tile in enumerate(tiles) for memory_id in tile}


def load_cached_layout_mapping(formal_dir: Path, *, bank_id: str, layout: str, capacity: int) -> dict[str, str]:
    mapping_path = formal_dir / "layouts" / bank_id / layout / f"capacity_{capacity}" / "memory_to_tile.jsonl"
    if not mapping_path.exists():
        return {}
    return {row["memory_id"]: row["logical_tile_id"] for row in iter_jsonl(mapping_path)}


def refine_layout_for_hyperedges(
    *,
    base_mapping: dict[str, str],
    capacity: int,
    hyperedges: list[tuple[tuple[str, ...], int]],
    max_hyperedges: int,
) -> dict[str, str]:
    tile_items: dict[str, list[str]] = defaultdict(list)
    for memory_id, tile_id in base_mapping.items():
        tile_items[tile_id].append(memory_id)
    for tile_id in tile_items:
        tile_items[tile_id].sort()

    mapping = dict(base_mapping)
    incident_weight: Counter[str] = Counter()
    for members, weight in hyperedges:
        for memory_id in members:
            incident_weight[memory_id] += weight

    edge_order = sorted(hyperedges, key=lambda item: (-item[1], -len(item[0]), item[0]))
    for members, _weight in edge_order[:max_hyperedges]:
        placed = [member for member in members if member in mapping]
        if len(placed) < 2:
            continue
        counts = Counter(mapping[member] for member in placed)
        target_tile, target_count = counts.most_common(1)[0]
        if target_count == len(placed):
            continue
        moving_members = sorted(
            [member for member in placed if mapping[member] != target_tile],
            key=lambda mid: (-incident_weight[mid], mid),
        )
        for member in moving_members:
            if len(tile_items[target_tile]) < capacity:
                source_tile = mapping[member]
                tile_items[source_tile].remove(member)
                tile_items[target_tile].append(member)
                mapping[member] = target_tile
                continue
            outsiders = [mid for mid in tile_items[target_tile] if mid not in set(members)]
            if not outsiders:
                continue
            swap_out = min(outsiders, key=lambda mid: (incident_weight[mid], mid))
            if incident_weight[member] <= incident_weight[swap_out]:
                continue
            source_tile = mapping[member]
            tile_items[source_tile].remove(member)
            tile_items[target_tile].remove(swap_out)
            tile_items[source_tile].append(swap_out)
            tile_items[target_tile].append(member)
            mapping[member] = target_tile
            mapping[swap_out] = source_tile
    return mapping


def build_hyperedge_first_layout(
    *,
    train_memory_ids: list[str],
    delta_memory_ids: list[str],
    capacity: int,
    hyperedges: list[tuple[tuple[str, ...], int]],
    max_hyperedges: int,
) -> dict[str, str]:
    train_set = set(train_memory_ids)
    assigned: dict[str, int] = {}
    tiles: list[list[str]] = []
    edge_order = sorted(
        [
            (tuple(mid for mid in members if mid in train_set), weight)
            for members, weight in hyperedges
        ],
        key=lambda item: (-item[1], -len(item[0]), item[0]),
    )
    for members, _weight in edge_order[:max_hyperedges]:
        pending = [mid for mid in members if mid not in assigned]
        if not pending:
            continue
        candidate_tile_ids = sorted(
            {
                assigned[member]
                for member in members
                if member in assigned and len(tiles[assigned[member]]) < capacity
            }
        )
        if candidate_tile_ids:
            tile_id = max(
                candidate_tile_ids,
                key=lambda idx: (
                    len(set(tiles[idx]) & set(members)),
                    capacity - len(tiles[idx]),
                    -idx,
                ),
            )
        else:
            tile_id = len(tiles)
            tiles.append([])
        for memory_id in pending:
            if len(tiles[tile_id]) >= capacity:
                tile_id = len(tiles)
                tiles.append([])
            tiles[tile_id].append(memory_id)
            assigned[memory_id] = tile_id

    for memory_id in sorted(train_set - set(assigned)):
        if not tiles or len(tiles[-1]) >= capacity:
            tiles.append([])
        tiles[-1].append(memory_id)
        assigned[memory_id] = len(tiles) - 1
    tiles.extend(chunked(delta_memory_ids, capacity))
    return {memory_id: f"tile_{idx:04d}" for idx, tile in enumerate(tiles) for memory_id in tile}


def layout_metrics(*, event: dict[str, Any], tile_by_memory: dict[str, str]) -> dict[str, float]:
    reference = list(event.get("final_selected_ids") or [])
    visible = set(event.get("visible_memory_ids") or [])
    counts: Counter[str] = Counter()
    for memory_id in reference:
        if memory_id not in visible:
            continue
        tile_id = tile_by_memory.get(memory_id)
        if tile_id is not None:
            counts[tile_id] += 1
    mapped = sum(counts.values())
    denom = len(reference) if reference else 1
    top_counts = [count for _tile, count in counts.most_common()]
    def best_coverage(num_tiles: int) -> float:
        return sum(top_counts[:num_tiles]) / denom
    return {
        "reference_size": float(len(reference)),
        "mapped_count": float(mapped),
        "tile_span_at_k": float(len(counts)),
        "best_1_tile_coverage": best_coverage(1),
        "best_2_tile_coverage": best_coverage(2),
        "best_4_tile_coverage": best_coverage(4),
    }


def load_inputs(formal_dir: Path):
    events = list(iter_jsonl(formal_dir / "runs" / "reference_selection_trace_primary.jsonl"))
    events_by_id = {row["event_id"]: row for row in events}
    by_bank: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in events:
        by_bank[row["bank_id"]].append(row)
    for bank_id in by_bank:
        by_bank[bank_id].sort(key=lambda row: (int(row.get("timestamp") or 0), row["event_id"]))
    splits = list(iter_jsonl(formal_dir / "manifests" / "bank_causal_split.jsonl"))
    return events_by_id, by_bank, splits


def run_layout_ablation(args: argparse.Namespace) -> list[dict[str, Any]]:
    _events_by_id, by_bank, splits = load_inputs(args.formal_dir)
    rows: list[dict[str, Any]] = []
    for split in splits:
        bank_events = by_bank.get(split["bank_id"], [])
        train_ids = set(split["train_event_ids"])
        test_ids = set(split["test_event_ids"])
        train_events = [event for event in bank_events if event["event_id"] in train_ids]
        test_events = [event for event in bank_events if event["event_id"] in test_ids]
        _pairwise_edges, hyperedges, _incident, adjacency = build_train_structure(train_events)
        train_memory_ids = sorted(split.get("train_memory_ids") or [])
        delta_memory_ids = sorted(split.get("delta_memory_ids") or [])
        for capacity in args.capacities:
            cached_pairwise = load_cached_layout_mapping(
                args.formal_dir,
                bank_id=split["bank_id"],
                layout="pairwise_graph_greedy",
                capacity=capacity,
            )
            if not cached_pairwise:
                cached_pairwise = build_pairwise_layout(
                    train_memory_ids=train_memory_ids,
                    delta_memory_ids=delta_memory_ids,
                    capacity=capacity,
                    adjacency=adjacency,
                )
            layouts = {
                "pairwise_graph_greedy_cached": cached_pairwise,
                "hypergraph_pairwise_refined": refine_layout_for_hyperedges(
                    base_mapping=cached_pairwise,
                    capacity=capacity,
                    hyperedges=hyperedges,
                    max_hyperedges=args.max_layout_hyperedges,
                ),
                "hypergraph_edge_pack": build_hyperedge_first_layout(
                    train_memory_ids=train_memory_ids,
                    delta_memory_ids=delta_memory_ids,
                    capacity=capacity,
                    hyperedges=hyperedges,
                    max_hyperedges=args.max_layout_hyperedges,
                ),
            }
            for event in test_events:
                for layout_name, tile_by_memory in layouts.items():
                    rows.append(
                        {
                            "event_id": event["event_id"],
                            "bank_id": split["bank_id"],
                            "protocol": split["protocol"],
                            "task": split["task"],
                            "setting": split["setting"],
                            "layout": layout_name,
                            "capacity": capacity,
                            "train_event_count": len(train_events),
                            "train_memory_count": len(train_memory_ids),
                            "delta_memory_count": len(delta_memory_ids),
                            "hyperedge_count": len(hyperedges),
                            "used_evaluator_labels": False,
                            **layout_metrics(event=event, tile_by_memory=tile_by_memory),
                        }
                    )
    return rows


def run_ablation(args: argparse.Namespace) -> list[dict[str, Any]]:
    events_by_id, by_bank, splits = load_inputs(args.formal_dir)
    rows: list[dict[str, Any]] = []
    for split in splits:
        bank_events = by_bank.get(split["bank_id"], [])
        train_ids = set(split["train_event_ids"])
        test_ids = set(split["test_event_ids"])
        train_events = [event for event in bank_events if event["event_id"] in train_ids]
        test_events = [event for event in bank_events if event["event_id"] in test_ids]
        pairwise_edges, hyperedges, incident, adjacency = build_train_structure(train_events)
        for event in test_events:
            visible = set(event.get("visible_memory_ids") or [])
            dense_order = [mid for mid in event.get("dense_ids") or [] if mid in visible]
            reference = set(event.get("final_selected_ids") or [])
            dense_rank = {mid: idx for idx, mid in enumerate(dense_order)}
            for seed_count in args.seed_counts:
                seeds = dense_order[:seed_count]
                if not seeds:
                    continue
                pairwise_ranked, pairwise_scanned = rank_pairwise_candidates(
                    seeds=seeds,
                    visible=visible,
                    adjacency=adjacency,
                    dense_rank=dense_rank,
                )
                hyper_ranked, hyper_scanned, hyper_diag = rank_hypergraph_candidates(
                    seeds=seeds,
                    visible=visible,
                    incident=incident,
                    dense_rank=dense_rank,
                    require_multi_seed=False,
                )
                strict_hyper_ranked, strict_hyper_scanned, strict_hyper_diag = rank_hypergraph_candidates(
                    seeds=seeds,
                    visible=visible,
                    incident=incident,
                    dense_rank=dense_rank,
                    require_multi_seed=True,
                )
                for budget in args.candidate_budgets:
                    anchor_count = min(budget, max(seed_count, int(math.ceil(budget * args.anchor_multiplier))))
                    dense_selected = dense_order[:budget]
                    dense_anchor = dense_order[:anchor_count]
                    methods = {
                        "dense_prefix": (dense_selected, len(dense_selected), {}),
                        "pairwise_graph_expand": (
                            compose_hybrid(dense_anchor=dense_anchor, ranked=pairwise_ranked, budget=budget),
                            min(len(visible), anchor_count + pairwise_scanned),
                            {"pairwise_edge_count": len(pairwise_edges), "pairwise_candidate_count": pairwise_scanned},
                        ),
                        "hypergraph_high_order_expand": (
                            compose_hybrid(dense_anchor=dense_anchor, ranked=hyper_ranked, budget=budget),
                            min(len(visible), anchor_count + hyper_scanned),
                            {"hyperedge_count": len(hyperedges), **hyper_diag},
                        ),
                        "hypergraph_strict_multi_seed_expand": (
                            compose_hybrid(dense_anchor=dense_anchor, ranked=strict_hyper_ranked, budget=budget),
                            min(len(visible), anchor_count + strict_hyper_scanned),
                            {"hyperedge_count": len(hyperedges), **strict_hyper_diag},
                        ),
                    }
                    for method, (selected, scanned, diag) in methods.items():
                        rows.append(
                            {
                                "event_id": event["event_id"],
                                "bank_id": split["bank_id"],
                                "protocol": split["protocol"],
                                "task": split["task"],
                                "setting": split["setting"],
                                "method": method,
                                "seed_count": seed_count,
                                "candidate_budget": budget,
                                "anchor_multiplier": args.anchor_multiplier,
                                "reference_size": len(reference),
                                "selected_count": len(selected),
                                "reference_recall_at_b": recall_at_budget(selected, reference),
                                "logical_items_scanned": scanned,
                                "selected_ids": selected,
                                "used_evaluator_labels": False,
                                **diag,
                            }
                        )
    return rows


def summarize(rows: list[dict[str, Any]], output_dir: Path, *, seed: int) -> None:
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    key_fields = ["seed_count", "candidate_budget"]
    baseline = "dense_prefix"
    methods = sorted({row["method"] for row in rows if row["method"] != baseline})
    by_key_method_event = {
        (int(row["seed_count"]), int(row["candidate_budget"]), row["method"], row["event_id"]): row
        for row in rows
    }
    summary_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["seed_count"]), int(row["candidate_budget"]), row["method"])].append(row)
    for (seed_count, budget, method), members in sorted(grouped.items()):
        summary_rows.append(
            {
                "seed_count": seed_count,
                "candidate_budget": budget,
                "method": method,
                "events": len(members),
                "mean_reference_recall_at_b": mean([float(row["reference_recall_at_b"]) for row in members]),
                "mean_logical_items_scanned": mean([float(row["logical_items_scanned"]) for row in members]),
                "mean_selected_count": mean([float(row["selected_count"]) for row in members]),
                "mean_activated_hyperedges": mean([float(row.get("activated_hyperedges", 0.0)) for row in members]),
                "mean_multi_seed_activated_hyperedges": mean([float(row.get("multi_seed_activated_hyperedges", 0.0)) for row in members]),
            }
        )

    for seed_count in sorted({int(row["seed_count"]) for row in rows}):
        for budget in sorted({int(row["candidate_budget"]) for row in rows}):
            event_ids = sorted(
                {
                    row["event_id"]
                    for row in rows
                    if int(row["seed_count"]) == seed_count and int(row["candidate_budget"]) == budget
                }
            )
            for method in methods:
                deltas_recall: list[float] = []
                deltas_scan: list[float] = []
                wins = losses = ties = 0
                identical = 0
                paired = 0
                for event_id in event_ids:
                    base = by_key_method_event.get((seed_count, budget, baseline, event_id))
                    cur = by_key_method_event.get((seed_count, budget, method, event_id))
                    if base is None or cur is None:
                        continue
                    paired += 1
                    delta = float(cur["reference_recall_at_b"]) - float(base["reference_recall_at_b"])
                    deltas_recall.append(delta)
                    deltas_scan.append(float(cur["logical_items_scanned"]) - float(base["logical_items_scanned"]))
                    if cur["selected_ids"] == base["selected_ids"]:
                        identical += 1
                    if delta > 1e-12:
                        wins += 1
                    elif delta < -1e-12:
                        losses += 1
                    else:
                        ties += 1
                ci = paired_bootstrap_ci(deltas_recall, seed=seed + seed_count * 100 + budget * 10 + len(method))
                comparison_rows.append(
                    {
                        "left_method": method,
                        "right_method": baseline,
                        "seed_count": seed_count,
                        "candidate_budget": budget,
                        "paired_events": paired,
                        "mean_delta_recall": mean(deltas_recall),
                        "ci95_low_delta_recall": ci["ci_low"],
                        "ci95_high_delta_recall": ci["ci_high"],
                        "mean_delta_logical_items_scanned": mean(deltas_scan),
                        "wins": wins,
                        "losses": losses,
                        "ties": ties,
                        "sign_test_p_value": sign_test_p_value(wins, losses),
                        "identical_output_rate": identical / paired if paired else 0.0,
                    }
                )

            # Direct graph-vs-hypergraph comparison is the central ablation.
            pairs = [
                ("hypergraph_high_order_expand", "pairwise_graph_expand"),
                ("hypergraph_strict_multi_seed_expand", "pairwise_graph_expand"),
                ("hypergraph_high_order_expand", "hypergraph_strict_multi_seed_expand"),
            ]
            for left, right in pairs:
                deltas_recall = []
                deltas_scan = []
                wins = losses = ties = 0
                identical = 0
                paired = 0
                active = 0
                for event_id in event_ids:
                    left_row = by_key_method_event.get((seed_count, budget, left, event_id))
                    right_row = by_key_method_event.get((seed_count, budget, right, event_id))
                    if left_row is None or right_row is None:
                        continue
                    paired += 1
                    if float(left_row.get("multi_seed_activated_hyperedges", 0.0)) > 0.0:
                        active += 1
                    delta = float(left_row["reference_recall_at_b"]) - float(right_row["reference_recall_at_b"])
                    deltas_recall.append(delta)
                    deltas_scan.append(float(left_row["logical_items_scanned"]) - float(right_row["logical_items_scanned"]))
                    if left_row["selected_ids"] == right_row["selected_ids"]:
                        identical += 1
                    if delta > 1e-12:
                        wins += 1
                    elif delta < -1e-12:
                        losses += 1
                    else:
                        ties += 1
                ci = paired_bootstrap_ci(deltas_recall, seed=seed + seed_count * 1000 + budget * 100 + len(left))
                comparison_rows.append(
                    {
                        "left_method": left,
                        "right_method": right,
                        "seed_count": seed_count,
                        "candidate_budget": budget,
                        "paired_events": paired,
                        "high_order_active_events": active,
                        "mean_delta_recall": mean(deltas_recall),
                        "ci95_low_delta_recall": ci["ci_low"],
                        "ci95_high_delta_recall": ci["ci_high"],
                        "mean_delta_logical_items_scanned": mean(deltas_scan),
                        "wins": wins,
                        "losses": losses,
                        "ties": ties,
                        "sign_test_p_value": sign_test_p_value(wins, losses),
                        "identical_output_rate": identical / paired if paired else 0.0,
                    }
                )

    summary_path = reports_dir / "hypergraph_pairwise_summary.csv"
    comparison_path = reports_dir / "hypergraph_pairwise_comparisons.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    with comparison_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = sorted({key for row in comparison_rows for key in row.keys()})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison_rows)

    best_vs_dense = sorted(
        [row for row in comparison_rows if row["right_method"] == baseline],
        key=lambda row: (float(row["mean_delta_recall"]), -float(row["mean_delta_logical_items_scanned"])),
        reverse=True,
    )[:8]
    graph_vs_hyper = [
        row
        for row in comparison_rows
        if row["left_method"] == "hypergraph_high_order_expand" and row["right_method"] == "pairwise_graph_expand"
    ]
    report = [
        "# Hypergraph vs Dense and Pairwise Ablation",
        "",
        "## Scope",
        "",
        f"- per-event trace: `{output_dir / 'runs' / 'hypergraph_pairwise_ablation_per_event.jsonl'}`",
        f"- summary: `{summary_path}`",
        f"- comparisons: `{comparison_path}`",
        "- task: `item_ranking`",
        "- split source: `versions/v0.3/formal_run` causal train/test split",
        "- selection labels: uses only train-side reference-selected memory to build structures; test-side reference-selected memory is used as an offline recall target, not for selection",
        "- evaluator labels: `false`",
        "",
        "## Key Findings",
        "",
    ]
    for row in best_vs_dense[:4]:
        report.append(
            "- vs dense: "
            f"`{row['left_method']}` S={row['seed_count']} B={row['candidate_budget']} "
            f"delta_recall={float(row['mean_delta_recall']):.4f} "
            f"CI=[{float(row['ci95_low_delta_recall']):.4f},{float(row['ci95_high_delta_recall']):.4f}] "
            f"wins/losses/ties={row['wins']}/{row['losses']}/{row['ties']} "
            f"delta_scan={float(row['mean_delta_logical_items_scanned']):.2f}"
        )
    report.extend(["", "## Hypergraph vs Pairwise", ""])
    for row in graph_vs_hyper:
        report.append(
            "- "
            f"S={row['seed_count']} B={row['candidate_budget']}: "
            f"delta_recall={float(row['mean_delta_recall']):.4f} "
            f"CI=[{float(row['ci95_low_delta_recall']):.4f},{float(row['ci95_high_delta_recall']):.4f}] "
            f"wins/losses/ties={row['wins']}/{row['losses']}/{row['ties']} "
            f"identical={float(row['identical_output_rate']):.3f} "
            f"active_events={row.get('high_order_active_events', 0)}"
        )
    report.extend(
        [
            "",
            "## Interpretation",
            "",
            "This ablation is intentionally tighter than the previous Qwen S5/S5b runs. Dense retrieval is compared at the same final candidate budget B, while graph and hypergraph methods can spend extra relation-expansion scans. The right claim is therefore a quality-cost curve, not that hypergraph beats an unlimited dense traversal.",
            "",
            "The direct hypergraph-vs-pairwise rows are the important check for the layout argument. If the mean delta is near zero and identical-output rate is high, the current trace or scoring still does not separate high-order hyperedges from pairwise co-access. If hypergraph wins only on high_order_active_events, the paper should report that conditional slice rather than a global claim.",
        ]
    )
    (reports_dir / "hypergraph_pairwise_ablation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def summarize_layout(layout_rows: list[dict[str, Any]], output_dir: Path, *, seed: int) -> None:
    reports_dir = output_dir / "reports"
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in layout_rows:
        grouped[(int(row["capacity"]), row["layout"])].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (capacity, layout), members in sorted(grouped.items()):
        summary_rows.append(
            {
                "capacity": capacity,
                "layout": layout,
                "events": len(members),
                "mean_tile_span_at_k": mean([float(row["tile_span_at_k"]) for row in members]),
                "mean_best_1_tile_coverage": mean([float(row["best_1_tile_coverage"]) for row in members]),
                "mean_best_2_tile_coverage": mean([float(row["best_2_tile_coverage"]) for row in members]),
                "mean_best_4_tile_coverage": mean([float(row["best_4_tile_coverage"]) for row in members]),
            }
        )

    by_key = {(int(row["capacity"]), row["layout"], row["event_id"]): row for row in layout_rows}
    comparison_rows: list[dict[str, Any]] = []
    for capacity in sorted({int(row["capacity"]) for row in layout_rows}):
        event_ids = sorted({row["event_id"] for row in layout_rows if int(row["capacity"]) == capacity})
        for left in ["hypergraph_edge_pack", "hypergraph_pairwise_refined"]:
            right = "pairwise_graph_greedy_cached"
            for metric in ["tile_span_at_k", "best_1_tile_coverage", "best_2_tile_coverage", "best_4_tile_coverage"]:
                deltas = []
                wins = losses = ties = 0
                for event_id in event_ids:
                    left_row = by_key.get((capacity, left, event_id))
                    right_row = by_key.get((capacity, right, event_id))
                    if left_row is None or right_row is None:
                        continue
                    delta = float(left_row[metric]) - float(right_row[metric])
                    deltas.append(delta)
                    if metric == "tile_span_at_k":
                        better = delta < -1e-12
                        worse = delta > 1e-12
                    else:
                        better = delta > 1e-12
                        worse = delta < -1e-12
                    if better:
                        wins += 1
                    elif worse:
                        losses += 1
                    else:
                        ties += 1
                ci = paired_bootstrap_ci(deltas, seed=seed + capacity * 17 + len(metric) + len(left))
                comparison_rows.append(
                    {
                        "capacity": capacity,
                        "left_layout": left,
                        "right_layout": right,
                        "metric": metric,
                        "paired_events": len(deltas),
                        "mean_delta": mean(deltas),
                        "ci95_low_delta": ci["ci_low"],
                        "ci95_high_delta": ci["ci_high"],
                        "wins": wins,
                        "losses": losses,
                        "ties": ties,
                        "sign_test_p_value": sign_test_p_value(wins, losses),
                    }
                )

    summary_path = reports_dir / "layout_hypergraph_pairwise_summary.csv"
    comparison_path = reports_dir / "layout_hypergraph_pairwise_comparisons.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    with comparison_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)

    lines = [
        "",
        "## Layout Hypergraph vs Pairwise",
        "",
        f"- layout per-event trace: `{output_dir / 'runs' / 'layout_hypergraph_pairwise_per_event.jsonl'}`",
        f"- layout summary: `{summary_path}`",
        f"- layout comparisons: `{comparison_path}`",
    ]
    for row in comparison_rows:
        if row["metric"] not in {"tile_span_at_k", "best_1_tile_coverage"}:
            continue
        direction = "lower is better" if row["metric"] == "tile_span_at_k" else "higher is better"
        lines.append(
            "- "
            f"`{row['left_layout']}` vs `{row['right_layout']}` C={row['capacity']} `{row['metric']}` ({direction}): "
            f"delta={float(row['mean_delta']):.4f} "
            f"CI=[{float(row['ci95_low_delta']):.4f},{float(row['ci95_high_delta']):.4f}] "
            f"wins/losses/ties={row['wins']}/{row['losses']}/{row['ties']}"
        )
    report_path = reports_dir / "hypergraph_pairwise_ablation_report.md"
    existing = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    report_path.write_text(existing.rstrip() + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-dir", type=Path, default=DEFAULT_FORMAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed-counts", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--candidate-budgets", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--capacities", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--max-layout-hyperedges", type=int, default=512)
    parser.add_argument("--anchor-multiplier", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    rows = run_ablation(args)
    layout_rows = run_layout_ablation(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "runs" / "hypergraph_pairwise_ablation_per_event.jsonl", rows)
    write_jsonl(args.output_dir / "runs" / "layout_hypergraph_pairwise_per_event.jsonl", layout_rows)
    write_json(
        args.output_dir / "artifacts" / "run_manifest.json",
        {
            "formal_dir": str(args.formal_dir),
            "output_dir": str(args.output_dir),
            "seed_counts": args.seed_counts,
            "candidate_budgets": args.candidate_budgets,
            "anchor_multiplier": args.anchor_multiplier,
            "rows": len(rows),
            "layout_rows": len(layout_rows),
            "used_evaluator_labels_for_selection": False,
            "simulated_values_used": False,
        },
    )
    summarize(rows, args.output_dir, seed=args.seed)
    summarize_layout(layout_rows, args.output_dir, seed=args.seed)
    print(json.dumps({"rows": len(rows), "layout_rows": len(layout_rows), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
