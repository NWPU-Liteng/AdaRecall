#!/usr/bin/env python3
"""Evaluate one OOF unit of persona-sharded rank and feature-Tile storage."""
from __future__ import annotations

import argparse
import csv
import json
import re
import runpy
import sys
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import torch


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.io import iter_jsonl, load_config, read_json, sha256_file, write_csv_new, write_json_new  # noqa: E402
from v21.persona_sharded_index import PersonaShardedSparseSemanticIndex, build_top_prototype_feature_tiles  # noqa: E402
from v21.sparse_semantic_index import SparseSemanticLookupRequest  # noqa: E402
from v21.uot_recipes import array_sha256  # noqa: E402


WORKER = runpy.run_path(str(VERSION_ROOT / "scripts/128_train_v21_sparse_joint_tuned_oof_r2.py"), run_name="v21_sharded_index_worker_helpers")
QESA_HELPERS = WORKER["QESA_HELPERS"]
ATTEMPT_RE = re.compile(r"^attempt_[0-9]{3,}$")


def block_ids(prefix: str, edges, entry_bytes: int, tile_bytes: int) -> set[str]:
    output: set[str] = set()
    entries_per_tile = tile_bytes // entry_bytes
    if entries_per_tile <= 0:
        raise ValueError("metadata Tile cannot be smaller than one rank entry")
    for edge in edges:
        blocks = (int(edge.rank_entries_visited) + entries_per_tile - 1) // entries_per_tile
        output.update(f"{prefix}:h{edge.prototype_id}:t{block}" for block in range(blocks))
    return output


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    data = load_config("v21_data.json")
    qesa_config = load_config("v21_qesa_r2.json")
    if args.run_id != config["run_id"] or args.seed not in config["fixed_seeds"] or not 0 <= args.fold < int(config["outer_folds"]):
        raise ValueError("run, seed, or fold outside sharded index contract")
    if not ATTEMPT_RE.fullmatch(args.attempt):
        raise ValueError("attempt must have the form attempt_NNN")
    if config.get("minimum_effect_threshold") is not None or config.get("calibration_read") is not False or config.get("formal_test_read") is not False:
        raise ValueError("thresholds and locked split reads are forbidden")
    unit_dir = VERSION_ROOT / "artifacts/v21_persona_sharded_hyper_index" / args.run_id / f"seed_{args.seed}" / f"fold_{args.fold}" / "attempts" / args.attempt
    unit_dir.mkdir(parents=True, exist_ok=False)

    ids = tuple(map(str, read_json(data["memory"]["embedding_ids"])))
    index_by_id = {memory_id: position for position, memory_id in enumerate(ids)}
    memory_rows = list(iter_jsonl(data["memory"]["items"]))
    memory_by_id = {str(row["memory_id"]): row for row in memory_rows}
    persona_by_position = tuple(str(memory_by_id[memory_id]["persona_id"]) for memory_id in ids)
    payload_bytes = {memory_id: int(memory_by_id[memory_id]["payload_bytes"]) for memory_id in ids}
    phase1_unit = VERSION_ROOT / "artifacts/v21_phase1" / config["phase1_run_id"] / f"seed_{args.seed}" / f"fold_{args.fold}"
    incidence_path = phase1_unit / "v21_uot_semantic_incidence.npy"
    incidence = np.load(incidence_path, mmap_mode="r")
    global_index, rank_path = WORKER["load_sparse_index"](config, args.seed, args.fold, incidence)
    sharded_index = PersonaShardedSparseSemanticIndex(incidence, persona_by_position)
    replication_factor = int(config["feature_replication_factor"])
    if replication_factor:
        feature_tiles, feature_tile_occupancy = build_top_prototype_feature_tiles(
            incidence,
            persona_by_position,
            replication_factor=replication_factor,
            tile_capacity=int(config["feature_tile_capacity_memories"]),
        )
    else:
        feature_tiles, feature_tile_occupancy = {}, {}

    neutral_path = VERSION_ROOT / "artifacts/v21_effect_closure/v21_neutral_memory_to_tile.csv"
    with neutral_path.open(newline="", encoding="utf-8") as handle:
        neutral_tile = {str(row["memory_id"]): str(row["tile_id"]) for row in csv.DictReader(handle)}
    query_vectors = QESA_HELPERS["load_query_vectors"](data)
    rows_by_fold = QESA_HELPERS["load_rows"](data, query_vectors, int(config["outer_folds"]))
    rows = rows_by_fold[args.fold]
    qesa_path = VERSION_ROOT / "artifacts/v21_qesa" / config["qesa_run_id"] / f"seed_{args.seed}" / f"fold_{args.fold}" / "v21_qesa_r2_models.pt"
    checkpoint = torch.load(qesa_path, map_location="cpu")
    model = WORKER["qesa_model"](checkpoint, len(rows[0]["query_embedding"]), int(qesa_config["hidden_dimension"]), args.device)
    predictions = WORKER["predict_qesa_batch"](model, np.stack([row["query_embedding"] for row in rows]), args.device)

    selection_path = VERSION_ROOT / "artifacts/v21_sparse_joint" / config["source_sparse_run_id"] / f"seed_{args.seed}" / f"fold_{args.fold}" / "attempts" / config["source_sparse_attempt"] / "v21_sparse_joint_selections.jsonl"
    selections: dict[str, tuple[str, ...]] = {}
    for row in iter_jsonl(selection_path):
        if (
            str(row["method"]) == "sparse_joint_semantic_bundle_hypergraph"
            and int(row["activated_prototypes"]) == int(config["activated_prototypes"])
            and int(row["members_per_semantic_edge"]) == int(config["members_per_semantic_edge"])
            and int(row["semantic_bundle_beam"]) == int(config["semantic_bundle_beam"])
            and int(row["budget"]) == int(config["selection_budget"])
        ):
            selections[str(row["query_id"])] = tuple(map(str, row["selected_memory_ids"]))
    if len(selections) != len(rows):
        raise ValueError("fixed B=32 selections are incomplete")

    def make_request(row: dict, prediction: np.ndarray) -> tuple[str, SparseSemanticLookupRequest]:
        persona = str(row["persona_id"])
        eligible = frozenset(index_by_id[memory_id] for memory_id in row["eligible_memory_ids"])
        active = tuple(map(int, np.argsort(-prediction, kind="stable")[: int(config["activated_prototypes"])]))
        return persona, SparseSemanticLookupRequest(active, eligible, int(config["members_per_semantic_edge"]))

    warmup = min(int(config["warmup_queries"]), len(rows))
    for row, prediction in zip(rows[:warmup], predictions[:warmup], strict=True):
        persona, request = make_request(row, prediction)
        _ = global_index.lookup(request)
        _ = sharded_index.lookup(persona, request)

    entry_bytes = int(config["rank_entry_bytes"])
    metadata_tile_bytes = int(config["rank_metadata_tile_bytes"])
    metrics: list[dict] = []
    for row, prediction in zip(rows, predictions, strict=True):
        persona, request = make_request(row, prediction)
        start = perf_counter_ns()
        global_result = global_index.lookup(request)
        global_lookup_ms = (perf_counter_ns() - start) / 1_000_000.0
        start = perf_counter_ns()
        shard_result = sharded_index.lookup(persona, request)
        shard_lookup_ms = (perf_counter_ns() - start) / 1_000_000.0
        if [edge.memory_positions for edge in global_result.edges] != [edge.memory_positions for edge in shard_result.edges]:
            raise ValueError(f"sharded lookup changed returned members for {row['query_id']}")
        if any(not np.allclose(left.incidence_scores, right.incidence_scores, atol=0.0) for left, right in zip(global_result.edges, shard_result.edges, strict=True)):
            raise ValueError(f"sharded lookup changed incidence scores for {row['query_id']}")

        selected = selections[str(row["query_id"])]
        selected_base_tiles = {f"base:{neutral_tile[memory_id]}" for memory_id in selected}
        global_feature_tiles = {
            f"base:{neutral_tile[ids[position]]}"
            for edge in global_result.edges for position in edge.memory_positions
        }
        shard_feature_tiles: set[str] = set()
        replica_hits = 0
        replica_total = 0
        for edge in shard_result.edges:
            for position in edge.memory_positions:
                replica_total += 1
                replica = feature_tiles.get((edge.prototype_id, int(position)))
                if replica is not None:
                    replica_hits += 1
                    shard_feature_tiles.add(replica)
                else:
                    shard_feature_tiles.add(f"base:{neutral_tile[ids[int(position)]]}")
        global_rank_tiles = block_ids("global-rank", global_result.edges, entry_bytes, metadata_tile_bytes)
        shard_rank_tiles = block_ids(f"persona-rank:{persona}", shard_result.edges, entry_bytes, metadata_tile_bytes)
        global_total = global_rank_tiles | global_feature_tiles | selected_base_tiles
        shard_total = shard_rank_tiles | shard_feature_tiles | selected_base_tiles
        metrics.append({
            "seed": args.seed,
            "fold": args.fold,
            "query_id": row["query_id"],
            "persona_id": persona,
            "global_rank_entries_visited": global_result.incidence_entries_visited,
            "persona_rank_entries_visited": shard_result.incidence_entries_visited,
            "global_metadata_bytes_accessed": global_result.incidence_entries_visited * entry_bytes,
            "persona_metadata_bytes_accessed": shard_result.incidence_entries_visited * entry_bytes,
            "global_metadata_tile_activations": len(global_rank_tiles),
            "persona_metadata_tile_activations": len(shard_rank_tiles),
            "global_feature_tile_activations": len(global_feature_tiles),
            "replicated_feature_tile_activations": len(shard_feature_tiles),
            "selected_payload_tile_activations": len(selected_base_tiles),
            "global_total_logical_tile_activations": len(global_total),
            "persona_sharded_total_logical_tile_activations": len(shard_total),
            "feature_replica_hit_rate": replica_hits / max(replica_total, 1),
            "global_lookup_ms": global_lookup_ms,
            "persona_sharded_lookup_ms": shard_lookup_ms,
            "payload_bytes": sum(payload_bytes[memory_id] for memory_id in selected),
            "returned_members_identical": True,
        })

    metrics_path = unit_dir / "v21_persona_sharded_hyper_index_metrics.csv"
    write_csv_new(metrics_path, metrics)
    embedding_shape = np.load((VERSION_ROOT / data["memory"]["embedding_vectors"]).resolve(), mmap_mode="r").shape
    feature_bytes_per_memory = (int(embedding_shape[1]) + int(incidence.shape[1])) * 2
    manifest = {
        "schema_version": "v21-persona-sharded-hyper-index-oof-unit-1",
        "status": "completed",
        "run_id": args.run_id,
        "attempt": args.attempt,
        "seed": args.seed,
        "fold": args.fold,
        "queries": len(metrics),
        "persona_shards": len(sharded_index.shard_sizes),
        "rank_metadata_storage_bytes_global": global_index.metadata_bytes,
        "rank_metadata_storage_bytes_sharded": sharded_index.metadata_bytes,
        "feature_replication_factor": replication_factor,
        "feature_bytes_per_memory_replica": feature_bytes_per_memory,
        "feature_replica_storage_bytes": len(feature_tiles) * feature_bytes_per_memory,
        "feature_replica_tiles": len(feature_tile_occupancy),
        "quality_selection_unchanged": True,
        "returned_members_identical_all_queries": True,
        "claim_boundary": config["claim_boundary"],
        "calibration_read": False,
        "formal_test_read": False,
        "artifacts": {"metrics": str(metrics_path.relative_to(VERSION_ROOT))},
        "hashes": {
            "config": sha256_file(VERSION_ROOT / "configs" / args.config),
            "incidence": array_sha256(incidence),
            "rank_index": sha256_file(rank_path),
            "qesa_checkpoint": sha256_file(qesa_path),
            "source_selections": sha256_file(selection_path),
            "metrics": sha256_file(metrics_path),
        },
    }
    write_json_new(unit_dir / "v21_persona_sharded_hyper_index_manifest.json", manifest)
    print(json.dumps({
        "status": manifest["status"],
        "seed": args.seed,
        "fold": args.fold,
        "queries": len(metrics),
        "global_entries_mean": float(np.mean([row["global_rank_entries_visited"] for row in metrics])),
        "sharded_entries_mean": float(np.mean([row["persona_rank_entries_visited"] for row in metrics])),
        "global_tiles_mean": float(np.mean([row["global_total_logical_tile_activations"] for row in metrics])),
        "sharded_tiles_mean": float(np.mean([row["persona_sharded_total_logical_tile_activations"] for row in metrics])),
    }, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="v21_persona_sharded_hyper_index_oof_r1.json")
    parser.add_argument("--run-id", default="v21_persona_sharded_hyper_index_oof_r1_20260715")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--attempt", default="attempt_001")
    parser.add_argument("--device", default="cuda:0")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
