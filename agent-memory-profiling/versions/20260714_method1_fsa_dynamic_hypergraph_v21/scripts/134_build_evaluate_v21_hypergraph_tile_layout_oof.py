#!/usr/bin/env python3
"""Build and evaluate one leakage-safe hypergraph-aware logical layout unit."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.evidence_sets import fold_for  # noqa: E402
from v21.hypergraph_tile_layout import build_hypergraph_tile_layout  # noqa: E402
from v21.io import iter_jsonl, load_config, read_json, sha256_file, write_csv_new, write_json_new  # noqa: E402
from v21.uot_recipes import array_sha256  # noqa: E402


ATTEMPT_RE = re.compile(r"^attempt_[0-9]{3,}$")


def evidence_hyperedges(rows: list[dict], target_fold: int, *, heldout: bool) -> list[tuple[str, ...]]:
    output: list[tuple[str, ...]] = []
    for row in rows:
        row_fold = fold_for(str(row["persona_id"]), int(row["query_session_id"]), 5)
        if (row_fold == target_fold) != heldout:
            continue
        groups = [tuple(dict.fromkeys(map(str, group["acceptable_memory_ids"]))) for group in row["evidence_groups"]]
        if len(groups) < 2:
            continue
        for members in itertools.product(*groups):
            edge = tuple(dict.fromkeys(members))
            if len(edge) == len(groups):
                output.append(edge)
    return output


def mean_span(edges: list[tuple[str, ...]], tile: dict[str, str]) -> float:
    return float(np.mean([len({tile[memory_id] for memory_id in edge}) for edge in edges])) if edges else 0.0


def model_frontier_hyperedges(config: dict, seed: int, target_fold: int) -> tuple[list[tuple[str, ...]], list[Path]]:
    if int(config.get("model_frontier_hyperedge_weight", 0)) <= 0:
        return [], []
    point = config["evaluation_point"]
    output: list[tuple[str, ...]] = []
    paths: list[Path] = []
    for fold in range(int(config["outer_folds"])):
        if fold == target_fold:
            continue
        path = (
            VERSION_ROOT / "artifacts/v21_sparse_joint" / config["source_sparse_run_id"]
            / f"seed_{seed}" / f"fold_{fold}" / "attempts" / config["source_sparse_attempt"]
            / "v21_sparse_joint_selections.jsonl"
        )
        paths.append(path)
        for row in iter_jsonl(path):
            if (
                str(row["method"]) == "sparse_joint_semantic_bundle_hypergraph"
                and int(row["budget"]) == max(map(int, point["budgets"]))
                and int(row["activated_prototypes"]) == int(point["activated_prototypes"])
                and int(row["members_per_semantic_edge"]) == int(point["members_per_semantic_edge"])
                and int(row["semantic_bundle_beam"]) == int(point["semantic_bundle_beam"])
            ):
                field = str(config.get("model_layout_hyperedge_field", "frontier_memory_ids"))
                output.append(tuple(map(str, row[field])))
    return output, paths


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    data = load_config("v21_data.json")
    if args.run_id != config["run_id"] or args.seed not in config["fixed_seeds"] or not 0 <= args.fold < int(config["outer_folds"]):
        raise ValueError("run, seed, or fold outside layout contract")
    if not ATTEMPT_RE.fullmatch(args.attempt):
        raise ValueError("attempt must have the form attempt_NNN")
    if config.get("replication_factor") != 1 or config.get("minimum_effect_threshold") is not None:
        raise ValueError("r1 requires no memory replication and no effect threshold")
    if config.get("calibration_read") is not False or config.get("formal_test_read") is not False:
        raise ValueError("locked split access is forbidden")
    unit_dir = VERSION_ROOT / "artifacts" / "v21_hypergraph_tile_layout" / args.run_id / f"seed_{args.seed}" / f"fold_{args.fold}" / "attempts" / args.attempt
    unit_dir.mkdir(parents=True, exist_ok=False)

    ids = tuple(map(str, read_json(data["memory"]["embedding_ids"])))
    memory_rows = list(iter_jsonl(data["memory"]["items"]))
    memory_by_id = {str(row["memory_id"]): row for row in memory_rows}
    if len(ids) != 42034 or set(ids) != set(memory_by_id):
        raise ValueError("frozen memory coverage changed")
    persona = {memory_id: str(memory_by_id[memory_id]["persona_id"]) for memory_id in ids}
    payload = {memory_id: int(memory_by_id[memory_id]["payload_bytes"]) for memory_id in ids}
    chronological = {memory_id: int(memory_by_id[memory_id]["chronological_rank"]) for memory_id in ids}
    incidence_path = VERSION_ROOT / "artifacts" / "v21_phase1" / config["phase1_run_id"] / f"seed_{args.seed}" / f"fold_{args.fold}" / "v21_uot_semantic_incidence.npy"
    incidence = np.load(incidence_path, mmap_mode="r")
    semantic_vectors = None
    semantic_vectors_path = None
    if config.get("semantic_root_space") == "frozen_memory_embedding":
        semantic_vectors_path = Path(data["memory"]["embedding_vectors"])
        semantic_vectors_path = semantic_vectors_path.resolve() if semantic_vectors_path.is_absolute() else (VERSION_ROOT / semantic_vectors_path).resolve()
        semantic_vectors = np.load(semantic_vectors_path, mmap_mode="r")
    query_rows = list(iter_jsonl(data["split"]["train_query_plan"]))
    evidence_train_edges = evidence_hyperedges(query_rows, args.fold, heldout=False)
    heldout_edges = evidence_hyperedges(query_rows, args.fold, heldout=True)
    frontier_train_edges, frontier_source_paths = model_frontier_hyperedges(config, args.seed, args.fold)
    train_edges = (
        evidence_train_edges * int(config["evidence_hyperedge_structural_weight"])
        + frontier_train_edges * int(config.get("model_frontier_hyperedge_weight", 0))
    )
    layout = build_hypergraph_tile_layout(
        ids, persona, incidence, train_edges,
        capacity=int(config["tile_capacity_memories"]),
        structural_weight=1,
        structural_core_capacity=int(config.get("structural_core_capacity", config["tile_capacity_memories"])),
        semantic_vectors=semantic_vectors,
        semantic_alias_fill=bool(config.get("semantic_alias_fill", False)),
        fallback_positions=chronological if config.get("fallback_policy") == "chronological_source_order" else None,
        tile_prefix=f"v21_hyper_s{args.seed}_f{args.fold}",
    )
    hyper_tile = dict(layout.memory_to_tile)
    neutral_path = VERSION_ROOT / "artifacts/v21_effect_closure/v21_neutral_memory_to_tile.csv"
    with neutral_path.open(newline="", encoding="utf-8") as handle:
        neutral_tile = {str(row["memory_id"]): str(row["tile_id"]) for row in csv.DictReader(handle)}
    if set(neutral_tile) != set(hyper_tile):
        raise ValueError("layout comparison does not cover the same memories")

    layout_path = unit_dir / "v21_hypergraph_memory_to_tile.csv"
    with layout_path.open("x", encoding="utf-8", newline="") as handle:
        fields = ["memory_id", "tile_id", "primary_prototype", "persona_id", "payload_bytes"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for memory_id in ids:
            writer.writerow({
                "memory_id": memory_id,
                "tile_id": hyper_tile[memory_id],
                "primary_prototype": layout.primary_prototype[memory_id],
                "persona_id": persona[memory_id],
                "payload_bytes": payload[memory_id],
            })

    point = config["evaluation_point"]
    budgets = frozenset(map(int, point["budgets"]))
    methods = frozenset(map(str, config["methods"]))
    source_dir = VERSION_ROOT / "artifacts/v21_sparse_joint" / config["source_sparse_run_id"] / f"seed_{args.seed}" / f"fold_{args.fold}" / "attempts" / config["source_sparse_attempt"]
    source_manifest = read_json(source_dir / "v21_sparse_joint_unit_manifest.json")
    if source_manifest.get("calibration_read") is not False or source_manifest.get("formal_test_read") is not False:
        raise ValueError("source selection unit touched a locked split")
    selection_path = source_dir / "v21_sparse_joint_selections.jsonl"
    metrics: list[dict] = []
    for row in iter_jsonl(selection_path):
        if str(row["method"]) not in methods or int(row["budget"]) not in budgets:
            continue
        if (
            int(row["activated_prototypes"]),
            int(row["members_per_semantic_edge"]),
            int(row["semantic_bundle_beam"]),
        ) != (
            int(point["activated_prototypes"]),
            int(point["members_per_semantic_edge"]),
            int(point["semantic_bundle_beam"]),
        ):
            continue
        selected = tuple(map(str, row["selected_memory_ids"]))
        frontier = tuple(map(str, row["frontier_memory_ids"]))
        if not set(selected).issubset(frontier) or len(selected) != int(row["budget"]):
            raise ValueError("selection/frontier contract changed")
        for layout_name, tile in (("neutral_source_order", neutral_tile), ("outer_train_hypergraph_uot", hyper_tile)):
            selected_tiles = {tile[memory_id] for memory_id in selected}
            frontier_tiles = {tile[memory_id] for memory_id in frontier}
            metrics.append({
                "seed": args.seed,
                "fold": args.fold,
                "query_id": row["query_id"],
                "method": row["method"],
                "budget": row["budget"],
                "layout": layout_name,
                "selected_memory_count": len(selected),
                "frontier_memory_count": len(frontier),
                "selected_payload_tile_activations": len(selected_tiles),
                "frontier_feature_tile_activations": len(frontier_tiles),
                "selected_memories_per_active_tile": len(selected) / max(len(selected_tiles), 1),
                "frontier_memories_per_active_tile": len(frontier) / max(len(frontier_tiles), 1),
                "payload_bytes": sum(payload[memory_id] for memory_id in selected),
            })
    heldout_queries = sum(fold_for(str(row["persona_id"]), int(row["query_session_id"]), 5) == args.fold for row in query_rows)
    expected = heldout_queries * len(methods) * len(budgets) * 2
    if len(metrics) != expected:
        raise ValueError(f"expected {expected} layout metric rows, found {len(metrics)}")
    metrics_path = unit_dir / "v21_hypergraph_tile_layout_metrics.csv"
    write_csv_new(metrics_path, metrics)
    occupancy = Counter(map(len, layout.tile_members.values()))
    manifest = {
        "schema_version": "v21-hypergraph-tile-layout-oof-unit-1",
        "status": "completed",
        "run_id": args.run_id,
        "attempt": args.attempt,
        "seed": args.seed,
        "fold": args.fold,
        "memory_count": len(ids),
        "tile_count": len(layout.tile_members),
        "tile_capacity": int(config["tile_capacity_memories"]),
        "replication_factor": 1,
        "structural_core_capacity": int(config.get("structural_core_capacity", config["tile_capacity_memories"])),
        "semantic_root_space": config.get("semantic_root_space", "uot_incidence"),
        "semantic_alias_fill": bool(config.get("semantic_alias_fill", False)),
        "model_layout_hyperedge_field": config.get("model_layout_hyperedge_field", "frontier_memory_ids"),
        "fallback_policy": config.get("fallback_policy", "uot_primary_prototype"),
        "outer_train_evidence_hyperedges": len(evidence_train_edges),
        "outer_train_model_frontier_hyperedges": len(frontier_train_edges),
        "weighted_layout_hyperedges": len(train_edges),
        "outer_heldout_evidence_hyperedges_evaluation_only": len(heldout_edges),
        "outer_train_hyperedge_span_neutral": mean_span(train_edges, neutral_tile),
        "outer_train_hyperedge_span_hypergraph": mean_span(train_edges, hyper_tile),
        "outer_heldout_hyperedge_span_neutral": mean_span(heldout_edges, neutral_tile),
        "outer_heldout_hyperedge_span_hypergraph": mean_span(heldout_edges, hyper_tile),
        "tile_occupancy_histogram": {str(key): value for key, value in sorted(occupancy.items())},
        "metric_rows": len(metrics),
        "quality_selection_unchanged": True,
        "no_access_or_coaccess_trace": True,
        "claim_boundary": config["claim_boundary"],
        "calibration_read": False,
        "formal_test_read": False,
        "artifacts": {
            "layout": str(layout_path.relative_to(VERSION_ROOT)),
            "metrics": str(metrics_path.relative_to(VERSION_ROOT)),
        },
        "hashes": {
            "config": sha256_file(VERSION_ROOT / "configs" / args.config),
            "incidence": array_sha256(incidence),
            **({"semantic_vectors": sha256_file(semantic_vectors_path)} if semantic_vectors_path is not None else {}),
            "neutral_layout": sha256_file(neutral_path),
            "source_manifest": sha256_file(source_dir / "v21_sparse_joint_unit_manifest.json"),
            "source_selections": sha256_file(selection_path),
            **{
                f"outer_train_frontier_source_fold_{fold}": sha256_file(path)
                for fold, path in enumerate(frontier_source_paths)
            },
            "layout": sha256_file(layout_path),
            "metrics": sha256_file(metrics_path),
        },
    }
    write_json_new(unit_dir / "v21_hypergraph_tile_layout_manifest.json", manifest)
    print(json.dumps({
        "status": manifest["status"],
        "seed": args.seed,
        "fold": args.fold,
        "tiles": manifest["tile_count"],
        "heldout_span_delta": manifest["outer_heldout_hyperedge_span_hypergraph"] - manifest["outer_heldout_hyperedge_span_neutral"],
        "metric_rows": len(metrics),
    }, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="v21_hypergraph_tile_layout_oof_r2.json")
    parser.add_argument("--run-id", default="v21_hypergraph_tile_layout_oof_r2_20260715")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--attempt", default="attempt_001")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
