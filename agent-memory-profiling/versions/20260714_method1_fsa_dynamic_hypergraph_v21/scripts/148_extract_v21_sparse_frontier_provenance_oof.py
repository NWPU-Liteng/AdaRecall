#!/usr/bin/env python3
"""Rebuild and persist exact retained semantic-bundle provenance for one OOF unit."""
from __future__ import annotations

import argparse
import json
import re
import runpy
import sys
from pathlib import Path

import numpy as np
import torch


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.io import iter_jsonl, load_config, read_json, resolve, sha256_file, write_json_new  # noqa: E402
from v21.sparse_joint import build_semantic_bundle_beam, construct_sparse_frontier  # noqa: E402
from v21.sparse_semantic_index import SparseSemanticLookupRequest  # noqa: E402
from v21.uot_recipes import array_sha256  # noqa: E402


WORKER = runpy.run_path(
    str(VERSION_ROOT / "scripts/128_train_v21_sparse_joint_tuned_oof_r2.py"),
    run_name="v21_sparse_frontier_provenance_helpers",
)
ATTEMPT_RE = re.compile(r"^attempt_[0-9]{3,}$")


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    source = load_config(config["source_config"])
    data = load_config("v21_data.json")
    qesa_config = load_config("v21_qesa_r2.json")
    if args.run_id != config["run_id"] or args.seed not in config["fixed_seeds"]:
        raise ValueError("run ID or seed outside provenance contract")
    if not 0 <= args.fold < int(config["outer_folds"]) or not ATTEMPT_RE.fullmatch(args.attempt):
        raise ValueError("fold or attempt outside provenance contract")
    if config.get("minimum_effect_threshold") is not None or config.get("calibration_read") is not False or config.get("formal_test_read") is not False:
        raise ValueError("thresholds and locked split reads are forbidden")
    unit_dir = VERSION_ROOT / "artifacts/v21_sparse_frontier_provenance" / args.run_id / f"seed_{args.seed}" / f"fold_{args.fold}" / "attempts" / args.attempt
    unit_dir.mkdir(parents=True, exist_ok=False)

    fixed_unit = VERSION_ROOT / "artifacts/v21_fixed_sparse_frontiers" / config["fixed_frontier_run_id"] / f"seed_{args.seed}" / f"fold_{args.fold}" / "attempts" / config["fixed_frontier_attempt"]
    fixed_path = fixed_unit / "v21_fixed_sparse_frontiers.jsonl"
    fixed_manifest_path = fixed_unit / "v21_fixed_sparse_frontier_manifest.json"
    fixed_manifest = read_json(fixed_manifest_path)
    if fixed_manifest.get("status") != "completed" or fixed_manifest.get("calibration_read") is not False or fixed_manifest.get("formal_test_read") is not False:
        raise ValueError("fixed frontier source violates split contract")
    if sha256_file(fixed_path) != fixed_manifest["hashes"]["frontiers"]:
        raise ValueError("fixed frontier hash mismatch")
    fixed = {str(row["query_id"]): row for row in iter_jsonl(fixed_path)}

    ids = tuple(map(str, read_json(data["memory"]["embedding_ids"])))
    index_by_id = {memory_id: position for position, memory_id in enumerate(ids)}
    raw = np.load(resolve(data["memory"]["embedding_vectors"]), mmap_mode="r").astype(np.float32)
    raw /= np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)
    query_vectors = WORKER["QESA_HELPERS"]["load_query_vectors"](data)
    rows_by_fold = WORKER["QESA_HELPERS"]["load_rows"](data, query_vectors, int(config["outer_folds"]))
    heldout_rows = rows_by_fold[args.fold]
    if len(fixed) != len(heldout_rows):
        raise ValueError("fixed frontier and heldout query counts differ")

    incidence_path = VERSION_ROOT / "artifacts/v21_phase1" / source["phase1_run_id"] / f"seed_{args.seed}" / f"fold_{args.fold}" / "v21_uot_semantic_incidence.npy"
    incidence = np.load(incidence_path, mmap_mode="r")
    qesa_path = VERSION_ROOT / "artifacts/v21_qesa" / source["qesa_run_id"] / f"seed_{args.seed}" / f"fold_{args.fold}" / "v21_qesa_r2_models.pt"
    checkpoint = torch.load(qesa_path, map_location="cpu")
    qesa = WORKER["qesa_model"](checkpoint, raw.shape[1], int(qesa_config["hidden_dimension"]), args.device)
    aspects = WORKER["predict_qesa_batch"](
        qesa, np.stack([row["query_embedding"] for row in heldout_rows]), args.device,
    )
    index, rank_path = WORKER["load_sparse_index"](source, args.seed, args.fold, incidence)
    ann_prefix = WORKER["load_ann_prefix"](data)

    output_path = unit_dir / "v21_sparse_frontier_provenance.jsonl"
    query_count = bundle_count = edge_count = 0
    with output_path.open("x", encoding="utf-8") as handle:
        for row, aspect in zip(heldout_rows, aspects, strict=True):
            query_id = str(row["query_id"])
            protected = WORKER["protected_positions"](row, ann_prefix, index_by_id)
            eligible = frozenset(index_by_id[memory_id] for memory_id in row["eligible_memory_ids"])
            active = np.argsort(-aspect, kind="stable")[: int(config["activated_prototypes"])].astype(int).tolist()
            lookup = index.lookup(SparseSemanticLookupRequest(tuple(active), eligible, int(config["members_per_semantic_edge"])))
            score_positions = set(protected)
            score_positions.update(int(position) for edge in lookup.edges for position in edge.memory_positions)
            score_array = np.asarray(sorted(score_positions), dtype=np.int64)
            member_scores = {
                int(position): float(score)
                for position, score in zip(score_array.tolist(), (raw[score_array] @ row["query_embedding"]).tolist(), strict=True)
            }
            bundles, audit = build_semantic_bundle_beam(
                lookup.edges,
                {prototype: float(aspect[prototype]) for prototype in active},
                member_scores,
                arity=int(config["semantic_bundle_arity"]),
                beam_width=int(config["semantic_bundle_beam"]),
            )
            bundles = sorted(bundles, key=lambda item: (-float(item.score), item.prototype_ids, item.memory_positions))
            frontier = construct_sparse_frontier(
                protected, bundles, lookup.edges, member_scores,
                maximum_budget=int(config["candidate_pool_cap"]),
            )
            frontier_ids = [ids[position] for position in frontier.memory_positions]
            expected_ids = list(map(str, fixed[query_id]["frontier_memory_ids"]))
            if frontier_ids != expected_ids:
                raise ValueError(f"reconstructed frontier differs from fixed source: {query_id}")

            bundle_rows = [
                {
                    "bundle_index": bundle_index,
                    "prototype_ids": list(map(int, bundle.prototype_ids)),
                    "memory_ids": [ids[position] for position in bundle.memory_positions],
                    "score": float(bundle.score),
                }
                for bundle_index, bundle in enumerate(bundles)
            ]
            edge_rows = [
                {
                    "prototype_id": int(edge.prototype_id),
                    "memory_ids": [ids[position] for position in edge.memory_positions],
                    "incidence_scores": list(map(float, edge.incidence_scores)),
                    "rank_entries_visited": int(edge.rank_entries_visited),
                }
                for edge in lookup.edges
            ]
            bundle_membership: dict[str, list[int]] = {memory_id: [] for memory_id in frontier_ids}
            prototype_membership: dict[str, set[int]] = {memory_id: set() for memory_id in frontier_ids}
            for bundle in bundle_rows:
                for memory_id in bundle["memory_ids"]:
                    if memory_id in bundle_membership:
                        bundle_membership[memory_id].append(int(bundle["bundle_index"]))
                for memory_id, prototype_id in zip(bundle["memory_ids"], bundle["prototype_ids"], strict=True):
                    if memory_id in prototype_membership:
                        prototype_membership[memory_id].add(int(prototype_id))
            for edge in edge_rows:
                for memory_id in edge["memory_ids"]:
                    if memory_id in prototype_membership:
                        prototype_membership[memory_id].add(int(edge["prototype_id"]))
            protected_ids = set(map(str, fixed[query_id]["protected_ann_memory_ids"]))
            member_rows = []
            for position, memory_id in enumerate(frontier_ids):
                bundles_for_member = bundle_membership[memory_id]
                member_rows.append({
                    "memory_id": memory_id,
                    "frontier_position": position,
                    "protected_ann": memory_id in protected_ids,
                    "retained_bundle_indices": bundles_for_member,
                    "prototype_ids": sorted(prototype_membership[memory_id]),
                    "source": "protected_ann" if memory_id in protected_ids else ("retained_bundle" if bundles_for_member else "returned_edge_fallback"),
                })
            record = {
                "query_id": query_id,
                "seed": args.seed,
                "fold": args.fold,
                "frontier_memory_ids": frontier_ids,
                "protected_ann_memory_ids": list(map(str, fixed[query_id]["protected_ann_memory_ids"])),
                "active_prototypes": [
                    {"prototype_id": prototype, "weight": float(aspect[prototype])}
                    for prototype in active
                ],
                "semantic_edges": edge_rows,
                "retained_semantic_bundles": bundle_rows,
                "frontier_member_provenance": member_rows,
                "bundle_audit": {key: int(value) for key, value in audit.items()},
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            query_count += 1
            bundle_count += len(bundle_rows)
            edge_count += len(edge_rows)

    manifest = {
        "schema_version": "v21-sparse-frontier-provenance-oof-unit-1",
        "status": "completed",
        "run_id": args.run_id,
        "attempt": args.attempt,
        "seed": args.seed,
        "fold": args.fold,
        "queries": query_count,
        "retained_semantic_bundles": bundle_count,
        "returned_semantic_edges": edge_count,
        "exact_frontier_equivalence": True,
        "provenance_fields": ["retained_semantic_bundles", "semantic_edges", "active_prototypes", "frontier_member_provenance"],
        "calibration_read": False,
        "formal_test_read": False,
        "minimum_effect_threshold": None,
        "artifacts": {"provenance": str(output_path.relative_to(VERSION_ROOT))},
        "hashes": {
            "config": sha256_file(VERSION_ROOT / "configs" / args.config),
            "fixed_frontier_manifest": sha256_file(fixed_manifest_path),
            "fixed_frontiers": sha256_file(fixed_path),
            "incidence": array_sha256(incidence),
            "qesa_checkpoint": sha256_file(qesa_path),
            "rank_index": sha256_file(rank_path),
            "provenance": sha256_file(output_path),
        },
    }
    write_json_new(unit_dir / "v21_sparse_frontier_provenance_manifest.json", manifest)
    print(json.dumps({"status": "completed", "seed": args.seed, "fold": args.fold, "queries": query_count, "bundles": bundle_count}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="v21_sparse_frontier_provenance_oof_r1.json")
    parser.add_argument("--run-id", default="v21_sparse_frontier_provenance_oof_r1_20260715")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--attempt", default="attempt_001")
    parser.add_argument("--device", default="cpu")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
