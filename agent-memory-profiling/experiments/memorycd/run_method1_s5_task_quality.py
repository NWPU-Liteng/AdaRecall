#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.memorycd.audit import iter_jsonl


V02_SCRIPT = PROJECT_ROOT / "versions" / "v0.2" / "code" / "run_method1_v02.py"
V03_SCRIPT = PROJECT_ROOT / "versions" / "v0.3" / "code" / "run_method1_v03.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v02 = load_module("method1_s5_v02", V02_SCRIPT)
v03 = load_module("method1_s5_v03", V03_SCRIPT)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def summarize_latency(values: list[float]) -> dict[str, float]:
    vals = sorted(float(value) for value in values)
    if not vals:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
    return {
        "p50": vals[min(len(vals) - 1, int(round((len(vals) - 1) * 0.50)))],
        "p95": vals[min(len(vals) - 1, int(round((len(vals) - 1) * 0.95)))],
        "p99": vals[min(len(vals) - 1, int(round((len(vals) - 1) * 0.99)))],
        "mean": sum(vals) / len(vals),
    }


@dataclass(frozen=True)
class MethodSpec:
    method: str
    version: str
    layout: str
    capacity: int
    seed_count: int
    tile_budget: int
    candidate_budget: int
    anchor_multiplier: float | None = None
    tile_topk_factor: float | None = None

    @property
    def base_layout(self) -> str:
        if self.layout == "dense_only_candidate_baseline":
            return self.layout
        return self.layout.split("__", 1)[0]


def parse_method_spec(raw: str) -> MethodSpec:
    parts = raw.split(":")
    if len(parts) < 6:
        raise ValueError(f"Bad method spec: {raw}")
    method, version, layout = parts[:3]
    values: dict[str, str] = {}
    for item in parts[3:]:
        key, value = item.split("=", 1)
        values[key] = value
    return MethodSpec(
        method=method,
        version=version,
        layout=layout,
        capacity=int(values["cap"]),
        seed_count=int(values["seed"]),
        tile_budget=int(values["tile"]),
        candidate_budget=int(values["cand"]),
        anchor_multiplier=float(values["anchor"]) if "anchor" in values else None,
        tile_topk_factor=float(values["tiletop"]) if "tiletop" in values else None,
    )


def default_specs() -> list[MethodSpec]:
    return [
        parse_method_spec(
            "dense_b16:v0.3:dense_only_candidate_baseline:cap=16:seed=4:tile=4:cand=16"
        ),
        parse_method_spec(
            "v02_semantic_fixed_b16:v0.2:global_hypergraph_semantic_tiebreak__hybrid_anchor0.5_tiletop0.5:cap=16:seed=4:tile=4:cand=16:anchor=0.5:tiletop=0.5"
        ),
        parse_method_spec(
            "v03_semantic_adaptive_b16:v0.3:global_hypergraph_semantic_tiebreak__adaptive_hybrid:cap=16:seed=4:tile=4:cand=16"
        ),
        parse_method_spec(
            "v03_pairwise_adaptive_b16:v0.3:pairwise_graph_greedy__adaptive_hybrid:cap=16:seed=4:tile=4:cand=16"
        ),
    ]


def build_args(config: dict[str, Any], output_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        source_version_dir=Path(config["source_version_dir"]),
        output_dir=output_dir,
        protocols=config["protocols"],
        tasks=["item_ranking"],
        main_task="item_ranking",
        reference_selector_primary=config["reference_selector_primary"],
        reference_selector_sensitivity=config["reference_selector_sensitivity"],
        capacities=config["capacities"],
        seed_counts=config["seed_counts"],
        tile_budgets=config["tile_budgets"],
        candidate_budgets=config["candidate_budgets"],
        anchor_candidate_multipliers=config.get("anchor_candidate_multipliers", [0.5]),
        tile_topk_factors=config.get("tile_topk_factors", [0.5, 1.0]),
        frontier_layouts=config.get(
            "frontier_layouts",
            ["global_hypergraph_greedy", "global_hypergraph_semantic_tiebreak", "pairwise_graph_greedy"],
        ),
        seed=config["seed"],
        min_train_events_per_view=config["min_train_events_per_view"],
        min_unique_selected_memories_per_view_floor=config["min_unique_selected_memories_per_view_floor"],
        max_banks_per_slice=config.get("max_banks_per_slice"),
        reuse_existing_layouts=True,
        device="cpu",
        adaptive_fallback_threshold=config.get("adaptive_fallback_threshold", 0.12),
        adaptive_conservative_threshold=config.get("adaptive_conservative_threshold", 0.3),
        adaptive_aggressive_threshold=config.get("adaptive_aggressive_threshold", 0.55),
        adaptive_conservative_anchor=config.get("adaptive_conservative_anchor", 0.75),
        adaptive_balanced_anchor=config.get("adaptive_balanced_anchor", 0.5),
        adaptive_conservative_tile_topk_factor=config.get("adaptive_conservative_tile_topk_factor", 0.5),
        adaptive_balanced_tile_topk_factor=config.get("adaptive_balanced_tile_topk_factor", 0.5),
        adaptive_aggressive_tile_topk_factor=config.get("adaptive_aggressive_tile_topk_factor", 1.0),
        adaptive_conservative_max_extra_tiles=config.get("adaptive_conservative_max_extra_tiles", 1),
    )


def load_experiment(version: str, formal_dir: Path):
    if version == "v0.2":
        config = json.loads((formal_dir / "configs_snapshot" / "method1_v0_1_config.json").read_text(encoding="utf-8"))
        exp = v02.Method1Experiment(build_args(config, formal_dir))
    elif version == "v0.3":
        config = json.loads((formal_dir / "configs_snapshot" / "method1_v0_3_config.json").read_text(encoding="utf-8"))
        exp = v03.AdaptiveMethod1Experiment(build_args(config, formal_dir))
    else:
        raise ValueError(version)
    exp.output_dir = formal_dir.resolve()
    return exp


def load_formal_events(exp: Any, formal_dir: Path) -> None:
    rows = list(iter_jsonl(formal_dir / "runs" / "reference_selection_trace_primary.jsonl"))
    events: list[Any] = []
    by_bank: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        event = v02.EventRecord(
            event_id=row["event_id"],
            bank_id=row["bank_id"],
            tenant_id=row.get("tenant_id_hash", ""),
            protocol=row["protocol"],
            task=row["task"],
            setting=row["setting"],
            target_timestamp=int(row["timestamp"]),
            causal_index=int(row.get("causal_index", 0)),
            query_text=row["query_text"],
            context_raw=dict(row.get("context_raw") or {}),
            visible_memory_ids=list(row.get("visible_memory_ids") or []),
            visible_memory_state=str(row.get("visible_memory_state") or "snapshot"),
            dense_ids=list(row.get("dense_ids") or []),
            final_selected_ids=list(row.get("final_selected_ids") or []),
            is_warmup=bool(row.get("is_warmup")),
            source_trace_path=str(row.get("source_trace_path") or ""),
            reference_selector=str(row.get("reference_selector") or exp.primary_selector),
        )
        events.append(event)
        by_bank[event.bank_id].append(event)
    for bank_id in by_bank:
        by_bank[bank_id] = sorted(by_bank[bank_id], key=lambda item: (item.target_timestamp, item.event_id))
    exp.events_by_selector[exp.primary_selector] = events
    exp.events_by_selector_bank[exp.primary_selector] = by_bank

    exp.bank_splits = []
    exp.bank_split_by_id = {}
    for row in iter_jsonl(formal_dir / "manifests" / "bank_causal_split.jsonl"):
        split = v02.BankSplit(
            bank_id=row["bank_id"],
            protocol=row["protocol"],
            task=row["task"],
            setting=row["setting"],
            tenant_id_hash=row["tenant_id_hash"],
            train_event_ids=list(row["train_event_ids"]),
            test_event_ids=list(row["test_event_ids"]),
            train_memory_ids=list(row["train_memory_ids"]),
            delta_memory_ids=list(row["delta_memory_ids"]),
            context_hypergraph_status=row.get("context_hypergraph_status", ""),
            context_hypergraph_reason=row.get("context_hypergraph_reason", ""),
        )
        exp.bank_splits.append(split)
        exp.bank_split_by_id[split.bank_id] = split


def v02_candidate_pool(exp: Any, spec: MethodSpec, split: Any, event: Any, layout_state: dict[str, Any], train_events: list[Any], memory_by_id: dict[str, dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    visible = set(event.visible_memory_ids)
    dense_order = [mid for mid in event.dense_ids if mid in visible]
    if spec.layout == "dense_only_candidate_baseline":
        return dense_order[: spec.candidate_budget], {"policy": "dense_only", "logical_items_scanned": spec.candidate_budget, "logical_tiles_touched": 0}

    pairwise_edges, hyperedges, incident_edges = exp._build_train_structure(train_events)
    embeddings = exp._get_embeddings(split.train_memory_ids + split.delta_memory_ids)
    seeds = dense_order[: spec.seed_count]
    if not seeds:
        return [], {"policy": "empty_seed", "logical_items_scanned": 0, "logical_tiles_touched": 0}
    tile_items = layout_state["tile_items"]
    tile_by_memory = layout_state["tile_by_memory"]
    seed_tiles = exp._seed_tiles_in_rank_order(seeds, tile_by_memory)
    active_tiles = exp._select_active_tiles(
        layout_name=spec.base_layout,
        seed_tiles=seed_tiles,
        seeds=seeds,
        tile_budget=spec.tile_budget,
        tile_items=tile_items,
        pairwise_edges=pairwise_edges,
        hyperedges=hyperedges,
        incident_edges=incident_edges,
    )
    tile_topk = max(1, int(math.ceil(spec.capacity * float(spec.tile_topk_factor or 1.0))))
    raw_candidates = exp._collect_tile_candidates(
        tile_ids=active_tiles,
        seeds=seeds,
        visible=visible,
        tile_items=tile_items,
        pairwise_edges=pairwise_edges,
        incident_edges=incident_edges,
        embeddings=embeddings,
        memory_by_id=memory_by_id,
        event=event,
        tile_topk=tile_topk,
    )
    struct_scores = exp._struct_scores(
        candidates=raw_candidates,
        seeds=seeds,
        event=event,
        pairwise_edges=pairwise_edges,
        incident_edges=incident_edges,
        embeddings=embeddings,
        train_memory_set=set(split.train_memory_ids),
        memory_by_id=memory_by_id,
        dense_rank={mid: idx for idx, mid in enumerate(dense_order)},
    )
    ranked_struct = sorted(raw_candidates, key=lambda mid: (-struct_scores.get(mid, 0.0), mid))
    anchor_count = min(spec.candidate_budget, max(spec.seed_count, int(math.ceil(spec.candidate_budget * float(spec.anchor_multiplier or 0.5)))))
    pool = exp._compose_hybrid_frontier(
        dense_anchor=dense_order[:anchor_count],
        struct_ranked=ranked_struct,
        candidate_budget=spec.candidate_budget,
    )
    return pool, {
        "policy": "fixed_hybrid",
        "logical_items_scanned": len(raw_candidates),
        "logical_tiles_touched": len(active_tiles),
        "dense_anchor_count": anchor_count,
    }


def v03_candidate_pool(exp: Any, spec: MethodSpec, split: Any, event: Any, layout_state: dict[str, Any], train_events: list[Any], memory_by_id: dict[str, dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    visible = set(event.visible_memory_ids)
    dense_order = [mid for mid in event.dense_ids if mid in visible]
    if spec.layout == "dense_only_candidate_baseline":
        return dense_order[: spec.candidate_budget], {"policy": "dense_only", "logical_items_scanned": spec.candidate_budget, "logical_tiles_touched": 0}

    pairwise_edges, hyperedges, incident_edges = exp._build_train_structure(train_events)
    embeddings = exp._get_embeddings(split.train_memory_ids + split.delta_memory_ids)
    seeds = dense_order[: spec.seed_count]
    if not seeds:
        return [], {"policy": "empty_seed", "logical_items_scanned": 0, "logical_tiles_touched": 0}
    tile_items = layout_state["tile_items"]
    tile_by_memory = layout_state["tile_by_memory"]
    seed_tiles = exp._seed_tiles_in_rank_order(seeds, tile_by_memory)
    active_seed_tiles, ranked_extra_tiles = exp._rank_candidate_tiles(
        layout_name=spec.base_layout,
        seed_tiles=seed_tiles,
        seeds=seeds,
        tile_budget=spec.tile_budget,
        tile_items=tile_items,
        pairwise_edges=pairwise_edges,
        hyperedges=hyperedges,
        incident_edges=incident_edges,
    )
    stats = exp._adaptive_structure_stats(
        seeds=seeds,
        train_event_count=max(1, len(train_events)),
        ranked_extra_tiles=ranked_extra_tiles,
        pairwise_edges=pairwise_edges,
        incident_edges=incident_edges,
    )
    policy = exp._choose_adaptive_policy(
        tile_budget=spec.tile_budget,
        seed_tile_count=len(active_seed_tiles),
        stats=stats,
    )
    if policy["mode"] == "fallback_dense":
        return dense_order[: spec.candidate_budget], {
            "policy": policy["mode"],
            "logical_items_scanned": spec.candidate_budget,
            "logical_tiles_touched": 0,
            "adaptive_structure_evidence": stats["evidence"],
            "view_fallback_rate": 1.0,
        }

    chosen_extra_tiles = [tile_id for tile_id, _score in ranked_extra_tiles[: policy["extra_tiles"]]]
    active_tiles = list(active_seed_tiles) + chosen_extra_tiles
    tile_topk = max(1, int(math.ceil(spec.capacity * float(policy["tile_topk_factor"]))))
    raw_candidates = exp._collect_tile_candidates(
        tile_ids=active_tiles,
        seeds=seeds,
        visible=visible,
        tile_items=tile_items,
        pairwise_edges=pairwise_edges,
        incident_edges=incident_edges,
        embeddings=embeddings,
        memory_by_id=memory_by_id,
        event=event,
        tile_topk=tile_topk,
    )
    struct_scores = exp._struct_scores(
        candidates=raw_candidates,
        seeds=seeds,
        event=event,
        pairwise_edges=pairwise_edges,
        incident_edges=incident_edges,
        embeddings=embeddings,
        train_memory_set=set(split.train_memory_ids),
        memory_by_id=memory_by_id,
        dense_rank={mid: idx for idx, mid in enumerate(dense_order)},
    )
    ranked_struct = sorted(raw_candidates, key=lambda mid: (-struct_scores.get(mid, 0.0), mid))
    anchor_count = min(
        spec.candidate_budget,
        max(spec.seed_count, int(math.ceil(spec.candidate_budget * float(policy["anchor_multiplier"])))),
    )
    pool = exp._compose_hybrid_frontier(
        dense_anchor=dense_order[:anchor_count],
        struct_ranked=ranked_struct,
        candidate_budget=spec.candidate_budget,
    )
    return pool, {
        "policy": policy["mode"],
        "logical_items_scanned": len(raw_candidates),
        "logical_tiles_touched": len(active_tiles),
        "dense_anchor_count": anchor_count,
        "adaptive_structure_evidence": stats["evidence"],
        "view_fallback_rate": 0.0,
    }


def reconstruct_candidates(exp: Any, version: str, formal_dir: Path, specs: list[MethodSpec], context_k: int) -> list[dict[str, Any]]:
    load_formal_events(exp, formal_dir)
    specs = [spec for spec in specs if spec.version == version]
    rows: list[dict[str, Any]] = []
    by_bank = exp.events_by_selector_bank[exp.primary_selector]
    specs_by_key: dict[tuple[int, str], list[MethodSpec]] = defaultdict(list)
    for spec in specs:
        specs_by_key[(spec.capacity, spec.base_layout)].append(spec)

    for split in exp.bank_splits:
        bank_events = by_bank.get(split.bank_id, [])
        if not bank_events:
            continue
        train_ids = set(split.train_event_ids)
        test_ids = set(split.test_event_ids)
        train_events = [event for event in bank_events if event.event_id in train_ids]
        test_events = [event for event in bank_events if event.event_id in test_ids]
        memory_by_id = exp.memory_by_protocol[split.protocol]
        for (capacity, base_layout), group_specs in specs_by_key.items():
            layout_state: dict[str, Any] | None = None
            if base_layout != "dense_only_candidate_baseline":
                layout_state = exp._load_cached_layout(split=split, layout_name=base_layout, capacity=capacity)
                if layout_state is None:
                    continue
            for event in test_events:
                for spec in group_specs:
                    if version == "v0.2":
                        pool, diag = v02_candidate_pool(exp, spec, split, event, layout_state or {}, train_events, memory_by_id)
                    else:
                        pool, diag = v03_candidate_pool(exp, spec, split, event, layout_state or {}, train_events, memory_by_id)
                    if not pool:
                        continue
                    rows.append(
                        {
                            "version": version,
                            "method": spec.method,
                            "layout": spec.layout,
                            "event_id": event.event_id,
                            "bank_id": split.bank_id,
                            "protocol": split.protocol,
                            "task": event.task,
                            "setting": event.setting,
                            "capacity": spec.capacity,
                            "seed_count": spec.seed_count,
                            "tile_budget": spec.tile_budget,
                            "candidate_budget": spec.candidate_budget,
                            "candidate_pool_ids": pool,
                            "final_context_ids": pool[: min(context_k, len(pool))],
                            "context_k": context_k,
                            **diag,
                        }
                    )
    return rows


def choose_balanced_rows(rows: list[dict[str, Any]], max_events: int | None) -> list[dict[str, Any]]:
    if max_events is None or max_events <= 0:
        return rows
    event_keys = sorted({(row["protocol"], row["setting"], row["event_id"]) for row in rows})
    chosen_keys = set(event_keys[:max_events])
    return [row for row in rows if (row["protocol"], row["setting"], row["event_id"]) in chosen_keys]


def write_selection_outputs(output_dir: Path, rows: list[dict[str, Any]], specs: list[MethodSpec], source_dirs: dict[str, Path]) -> None:
    runs_dir = output_dir / "runs"
    reports_dir = output_dir / "reports"
    artifacts_dir = output_dir / "artifacts"
    runs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    selection_path = runs_dir / "method1_s5_reconstructed_candidates.jsonl"
    with selection_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(
        output_dir / "artifacts" / "run_manifest.json",
        {
            "source_formal_dirs": {key: str(value) for key, value in source_dirs.items()},
            "methods": [spec.__dict__ for spec in specs],
            "reconstructed_rows": len(rows),
            "used_evaluator_labels_for_selection": False,
        },
    )


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _slice_event_ids(
    rows: list[dict[str, Any]],
    *,
    baseline_method: str,
    hard_metric: str,
    hard_threshold: float,
) -> dict[str, set[str]]:
    baseline_rows = [row for row in rows if row["method"] == baseline_method]
    all_event_ids = {row["event_id"] for row in rows}
    hard_event_ids = {
        row["event_id"]
        for row in baseline_rows
        if float(row["metrics"].get(hard_metric, 0.0)) < hard_threshold
    }
    slices = {
        "overall": all_event_ids,
        "hard_dense_fail": hard_event_ids,
    }
    adaptive_methods = sorted(
        {
            row["method"]
            for row in rows
            if row["method"].startswith("v03_") and row.get("policy") is not None
        }
    )
    for method in adaptive_methods:
        slices[f"active_{method}"] = {
            row["event_id"]
            for row in rows
            if row["method"] == method and row.get("policy") != "fallback_dense"
        }
    return slices


def _write_slice_summary(
    *,
    rows: list[dict[str, Any]],
    output_dir: Path,
    baseline_method: str,
    hard_metric: str,
    hard_threshold: float,
) -> None:
    slices = _slice_event_ids(
        rows,
        baseline_method=baseline_method,
        hard_metric=hard_metric,
        hard_threshold=hard_threshold,
    )
    slice_summary_path = output_dir / "reports" / "s5_task_quality_slice_summary.csv"
    pairwise_path = output_dir / "reports" / "s5_task_quality_pairwise_vs_dense.csv"

    with slice_summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "slice_name",
                "protocol",
                "setting",
                "version",
                "method",
                "events",
                "mean_ndcg@5",
                "mean_recall@5",
                "mean_context_count",
                "mean_candidate_pool_count",
                "mean_logical_items_scanned",
                "mean_logical_tiles_touched",
                "mean_view_fallback_rate",
            ],
        )
        writer.writeheader()
        for slice_name, event_ids in sorted(slices.items()):
            grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
            for row in rows:
                if row["event_id"] in event_ids:
                    grouped[(row["protocol"], row["setting"], row["version"], row["method"])].append(row)
            for key, members in sorted(grouped.items()):
                protocol, setting, version, method = key
                writer.writerow(
                    {
                        "slice_name": slice_name,
                        "protocol": protocol,
                        "setting": setting,
                        "version": version,
                        "method": method,
                        "events": len(members),
                        "mean_ndcg@5": _mean([float(row["metrics"]["ndcg@5"]) for row in members]),
                        "mean_recall@5": _mean([float(row["metrics"]["recall@5"]) for row in members]),
                        "mean_context_count": _mean([len(row["final_context_ids"]) for row in members]),
                        "mean_candidate_pool_count": _mean([len(row["candidate_pool_ids"]) for row in members]),
                        "mean_logical_items_scanned": _mean([float(row.get("logical_items_scanned", 0.0)) for row in members]),
                        "mean_logical_tiles_touched": _mean([float(row.get("logical_tiles_touched", 0.0)) for row in members]),
                        "mean_view_fallback_rate": _mean([float(row.get("view_fallback_rate", 0.0)) for row in members]),
                    }
                )

    with pairwise_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "slice_name",
                "method",
                "paired_events",
                "mean_delta_ndcg@5",
                "mean_delta_recall@5",
                "mean_delta_logical_items_scanned",
                "wins",
                "losses",
                "ties",
            ],
        )
        writer.writeheader()
        rows_by_method_event = {(row["method"], row["event_id"]): row for row in rows}
        methods = sorted({row["method"] for row in rows if row["method"] != baseline_method})
        for slice_name, event_ids in sorted(slices.items()):
            for method in methods:
                deltas_ndcg: list[float] = []
                deltas_recall: list[float] = []
                deltas_scan: list[float] = []
                wins = losses = ties = 0
                for event_id in sorted(event_ids):
                    base = rows_by_method_event.get((baseline_method, event_id))
                    cur = rows_by_method_event.get((method, event_id))
                    if base is None or cur is None:
                        continue
                    delta_ndcg = float(cur["metrics"]["ndcg@5"]) - float(base["metrics"]["ndcg@5"])
                    delta_recall = float(cur["metrics"]["recall@5"]) - float(base["metrics"]["recall@5"])
                    delta_scan = float(cur.get("logical_items_scanned", 0.0)) - float(base.get("logical_items_scanned", 0.0))
                    deltas_ndcg.append(delta_ndcg)
                    deltas_recall.append(delta_recall)
                    deltas_scan.append(delta_scan)
                    if delta_ndcg > 1e-12:
                        wins += 1
                    elif delta_ndcg < -1e-12:
                        losses += 1
                    else:
                        ties += 1
                writer.writerow(
                    {
                        "slice_name": slice_name,
                        "method": method,
                        "paired_events": len(deltas_ndcg),
                        "mean_delta_ndcg@5": _mean(deltas_ndcg),
                        "mean_delta_recall@5": _mean(deltas_recall),
                        "mean_delta_logical_items_scanned": _mean(deltas_scan),
                        "wins": wins,
                        "losses": losses,
                        "ties": ties,
                    }
                )


def evaluate_rows(
    *,
    rows: list[dict[str, Any]],
    output_dir: Path,
    source_version_dir: Path,
    meta_dir: Path,
    qwen_model_dir: Path,
    gpu: int,
    max_new_tokens: int,
) -> Path:
    from src.memorycd.task_runtime import (
        _build_prompt_with_selected_memory,
        _format_memory_for_prompt,
        _format_target_item,
        _parse_prediction,
        _sync_cuda,
        LocalPromptBuilder,
        LocalQwenLLMClient,
        MemoryCDMetadataLoader,
        SimpleItemRankingEvaluator,
        load_protocol_runtime,
    )

    os.environ["LOCAL_QWEN_CUDA_DEVICE"] = str(gpu)
    protocols = sorted({row["protocol"] for row in rows})
    runtime_by_protocol: dict[str, dict[str, dict[str, Any]]] = {}
    memory_by_protocol: dict[str, dict[str, dict[str, Any]]] = {}
    label_by_protocol: dict[str, dict[str, dict[str, Any]]] = {}
    for protocol in protocols:
        runtime_rows, labels, memory_by_id = load_protocol_runtime(version_dir=source_version_dir, protocol=protocol)
        runtime_by_protocol[protocol] = {row["event_id"]: row for row in runtime_rows}
        memory_by_protocol[protocol] = memory_by_id
        label_by_protocol[protocol] = labels

    metadata_loader = MemoryCDMetadataLoader(meta_dir)
    prompt_builder = LocalPromptBuilder(include_metadata=True)
    evaluator = SimpleItemRankingEvaluator()
    llm_client = LocalQwenLLMClient(model_dir=str(qwen_model_dir), max_new_tokens=max_new_tokens)

    trace_dir = output_dir / "runs" / "task_curve_subset"
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / "method1_s5_item_ranking_task_curve_subset.jsonl"
    with trace_path.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(rows):
            protocol = row["protocol"]
            event = runtime_by_protocol[protocol][row["event_id"]]
            memory_by_id = memory_by_protocol[protocol]
            selected = [memory_by_id[mid] for mid in row["final_context_ids"] if mid in memory_by_id]
            selected_prompt_memories = [_format_memory_for_prompt(memory) for memory in selected]
            target_item = _format_target_item(event=event, metadata_loader=metadata_loader)
            prompt, ordered_candidates = _build_prompt_with_selected_memory(
                prompt_builder=prompt_builder,
                task="item_ranking",
                selected_memories=selected_prompt_memories,
                target_item=target_item,
                metadata_loader=metadata_loader,
                candidate_asins=event.get("candidate_item_metadata"),
                user_id=event["tenant_id"],
            )
            _sync_cuda()
            t0 = time.perf_counter()
            generation = llm_client.generate_with_stats(prompt)
            _sync_cuda()
            t_task_qwen_ms = (time.perf_counter() - t0) * 1000.0
            prediction = _parse_prediction(
                task="item_ranking",
                response_text=generation["text"],
                ordered_candidates=ordered_candidates,
            )
            label = label_by_protocol[protocol][row["event_id"]]
            gold = str(label["positive_parent_asin"])
            metrics = {
                "ndcg@1": evaluator.ndcg_at_k(prediction, gold, 1),
                "ndcg@3": evaluator.ndcg_at_k(prediction, gold, 3),
                "ndcg@5": evaluator.ndcg_at_k(prediction, gold, 5),
                "recall@1": evaluator.recall_at_k(prediction, gold, 1),
                "recall@3": evaluator.recall_at_k(prediction, gold, 3),
                "recall@5": evaluator.recall_at_k(prediction, gold, 5),
            }
            out = {
                **row,
                "event_index": idx,
                "prediction": prediction,
                "gold_parent_asin": gold,
                "metrics": metrics,
                "prompt_tokens": generation["prompt_tokens"],
                "completion_tokens": generation["completion_tokens"],
                "t_task_qwen_ms": t_task_qwen_ms,
                "t_total_task_ms": t_task_qwen_ms,
                "backend_mode": generation["backend_mode"],
            }
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")
            handle.flush()
    return trace_path


def summarize_task_quality(
    output_dir: Path,
    trace_path: Path,
    *,
    baseline_method: str,
    hard_metric: str,
    hard_threshold: float,
) -> None:
    rows = [json.loads(line) for line in trace_path.open("r", encoding="utf-8")]
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["protocol"], row["setting"], row["version"], row["method"])].append(row)

    summary_path = output_dir / "reports" / "s5_task_quality_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "protocol",
                "setting",
                "version",
                "method",
                "events",
                "mean_ndcg@5",
                "mean_recall@5",
                "mean_context_count",
                "mean_candidate_pool_count",
                "mean_logical_items_scanned",
                "mean_logical_tiles_touched",
                "mean_view_fallback_rate",
                "p50_task_ms",
                "p95_task_ms",
                "parse_failure_rate",
            ],
        )
        writer.writeheader()
        for key, members in sorted(grouped.items()):
            protocol, setting, version, method = key
            task_stats = summarize_latency([row["t_task_qwen_ms"] for row in members])
            writer.writerow(
                {
                    "protocol": protocol,
                    "setting": setting,
                    "version": version,
                    "method": method,
                    "events": len(members),
                    "mean_ndcg@5": sum(float(row["metrics"]["ndcg@5"]) for row in members) / len(members),
                    "mean_recall@5": sum(float(row["metrics"]["recall@5"]) for row in members) / len(members),
                    "mean_context_count": sum(len(row["final_context_ids"]) for row in members) / len(members),
                    "mean_candidate_pool_count": sum(len(row["candidate_pool_ids"]) for row in members) / len(members),
                    "mean_logical_items_scanned": sum(float(row.get("logical_items_scanned", 0.0)) for row in members) / len(members),
                    "mean_logical_tiles_touched": sum(float(row.get("logical_tiles_touched", 0.0)) for row in members) / len(members),
                    "mean_view_fallback_rate": sum(float(row.get("view_fallback_rate", 0.0)) for row in members) / len(members),
                    "p50_task_ms": task_stats["p50"],
                    "p95_task_ms": task_stats["p95"],
                    "parse_failure_rate": sum(1 for row in members if not row["prediction"]) / len(members),
                }
            )
    report = [
        "# Method1 S5 Task Quality",
        "",
        f"- trace: `{trace_path}`",
        f"- summary: `{summary_path}`",
        f"- slice summary: `{output_dir / 'reports' / 's5_task_quality_slice_summary.csv'}`",
        f"- pairwise vs dense: `{output_dir / 'reports' / 's5_task_quality_pairwise_vs_dense.csv'}`",
        "- task: `item_ranking`",
        "- selection uses reconstructed Method1 candidate pools; evaluator labels are used only for task-quality metrics.",
        "- context passed to Qwen is `final_context_ids`, capped by `--context-k`.",
        f"- hard-event rule: baseline `{baseline_method}` with `{hard_metric} < {hard_threshold}`",
    ]
    (output_dir / "reports" / "s5_task_quality_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    _write_slice_summary(
        rows=rows,
        output_dir=output_dir,
        baseline_method=baseline_method,
        hard_metric=hard_metric,
        hard_threshold=hard_threshold,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v02-formal-dir", type=Path, default=PROJECT_ROOT / "versions" / "v0.2" / "formal_run")
    parser.add_argument("--v03-formal-dir", type=Path, default=PROJECT_ROOT / "versions" / "v0.3" / "formal_run")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "versions" / "v0.4" / "s5_task_quality")
    parser.add_argument("--source-version-dir", type=Path, default=None)
    parser.add_argument("--meta-dir", type=Path, default=PROJECT_ROOT / "data" / "public_datasets" / "memorycd" / "meta")
    parser.add_argument("--qwen-model-dir", type=Path, default=PROJECT_ROOT.parent / "dir")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--context-k", type=int, default=16)
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--baseline-method", type=str, default="dense_b16")
    parser.add_argument("--hard-metric", type=str, default="ndcg@5")
    parser.add_argument("--hard-threshold", type=float, default=1.0)
    parser.add_argument("--method", action="append", default=None, help="Override method specs; repeatable.")
    parser.add_argument("--selection-only", action="store_true")
    args = parser.parse_args()

    specs = [parse_method_spec(raw) for raw in args.method] if args.method else default_specs()
    source_version_dir = args.source_version_dir
    if source_version_dir is None:
        v03_config = json.loads((args.v03_formal_dir / "configs_snapshot" / "method1_v0_3_config.json").read_text(encoding="utf-8"))
        source_version_dir = Path(v03_config["source_version_dir"])

    experiments = {
        "v0.2": load_experiment("v0.2", args.v02_formal_dir),
        "v0.3": load_experiment("v0.3", args.v03_formal_dir),
    }
    all_rows: list[dict[str, Any]] = []
    for version, exp in experiments.items():
        formal_dir = args.v02_formal_dir if version == "v0.2" else args.v03_formal_dir
        all_rows.extend(reconstruct_candidates(exp, version, formal_dir, specs, args.context_k))

    method_sets: dict[str, set[str]] = defaultdict(set)
    for row in all_rows:
        method_sets[row["method"]].add(row["event_id"])
    common_events = set.intersection(*(events for events in method_sets.values())) if method_sets else set()
    rows = [row for row in all_rows if row["event_id"] in common_events]
    rows = choose_balanced_rows(rows, args.max_events if args.max_events > 0 else None)
    rows = sorted(rows, key=lambda row: (row["protocol"], row["setting"], row["event_id"], row["method"]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_selection_outputs(
        args.output_dir,
        rows,
        specs,
        {"v0.2": args.v02_formal_dir, "v0.3": args.v03_formal_dir, "source_version_dir": source_version_dir},
    )
    if args.selection_only:
        print(json.dumps({"selection_rows": len(rows), "common_events": len({row["event_id"] for row in rows}), "output_dir": str(args.output_dir)}, indent=2))
        return

    trace_path = evaluate_rows(
        rows=rows,
        output_dir=args.output_dir,
        source_version_dir=source_version_dir,
        meta_dir=args.meta_dir,
        qwen_model_dir=args.qwen_model_dir,
        gpu=args.gpu,
        max_new_tokens=args.max_new_tokens,
    )
    summarize_task_quality(
        args.output_dir,
        trace_path,
        baseline_method=args.baseline_method,
        hard_metric=args.hard_metric,
        hard_threshold=args.hard_threshold,
    )
    print(json.dumps({"trace_path": str(trace_path), "rows": len(rows), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
