#!/usr/bin/env python3
"""Exact-vs-greedy utility headroom on a small sparse frontier, without gold labels."""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.io import iter_jsonl, load_config, read_json, resolve, sha256_file, write_csv_new, write_json_new  # noqa: E402
from v21.qesa_r2 import LatentAspectQESAR2  # noqa: E402
from v21.qhds import MonotoneSubmodularQHDS, build_features, greedy_select  # noqa: E402


RUN_ID = "v21_learned_utility_oracle_r2_20260714"
OUTPUT_REVISION = "r2_20260714"
GRID = (16, 32, 32)
CANDIDATE_CAP = 12
BUDGET = 8
PROTECTED_SEEDS = 5


def load_query_vectors(data: dict) -> dict[str, np.ndarray]:
    matrix = np.load(resolve(data["query_embeddings"]["vectors"]), mmap_mode="r")
    records = read_json(data["query_embeddings"]["index"])["records"]
    return {str(row["query_id"]): np.asarray(matrix[position], dtype=np.float32) for position, row in enumerate(records) if row.get("dataset") == data["query_embeddings"]["train_key"]}


def load_frontiers(path: Path) -> dict[str, tuple[str, ...]]:
    output: dict[str, tuple[str, ...]] = {}
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("method") != "ann_flat" or int(row.get("budget", -1)) != BUDGET:
            continue
        if (int(row["activated_prototypes"]), int(row["members_per_semantic_edge"]), int(row["semantic_bundle_beam"])) != GRID:
            continue
        query_id = str(row["query_id"])
        frontier = tuple(dict.fromkeys(map(str, row["frontier_memory_ids"])))
        if len(frontier) < CANDIDATE_CAP or len(frontier) < PROTECTED_SEEDS:
            raise ValueError(f"small oracle frontier is too short: {query_id}")
        previous = output.get(query_id)
        if previous is not None and previous != frontier:
            raise ValueError(f"inconsistent frontier for query: {query_id}")
        output[query_id] = frontier
    if not output:
        raise ValueError(f"no fixed-grid frontier rows in {path}")
    return output


def load_qesa(config: dict, qesa_config: dict, seed: int, fold: int, dimension: int) -> tuple[LatentAspectQESAR2, Path]:
    path = VERSION_ROOT / "artifacts" / "v21_qesa" / config["qesa_run_id"] / f"seed_{seed}" / f"fold_{fold}" / "v21_qesa_r2_models.pt"
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint["qesa"]
    model = LatentAspectQESAR2(
        dimension,
        int(qesa_config["hidden_dimension"]),
        int(checkpoint["selected_latent_aspects"]),
        int(state["aspect_prototype_logits"].shape[1]),
        float(checkpoint["selected_regime"]["aspect_temperature"]),
    )
    model.load_state_dict(state)
    return model.eval(), path


def load_qhds(config: dict, seed: int, fold: int) -> tuple[MonotoneSubmodularQHDS, Path]:
    path = VERSION_ROOT / "artifacts" / "v21_qhds" / config["dense_qhds_run_id"] / f"seed_{seed}" / f"fold_{fold}" / "v21_qhds_model.pt"
    checkpoint = torch.load(path, map_location="cpu")
    model = MonotoneSubmodularQHDS(3, 16)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), path


def utility(model: MonotoneSubmodularQHDS, features: torch.Tensor, incidence: torch.Tensor, aspects: torch.Tensor, selected: tuple[int, ...]) -> float:
    with torch.no_grad():
        relevance = model.relevance(features)
        selected_mask = torch.zeros(len(features), dtype=torch.bool)
        selected_mask[list(selected)] = True
        covered = (incidence * (selected_mask.to(relevance.dtype) * relevance)[:, None]).sum(dim=0)
        structural_scale, modular_scale = model.positive_scales()
        value = structural_scale * (aspects * torch.log1p(covered)).sum() + modular_scale * (selected_mask.to(relevance.dtype) * relevance).sum()
        return float(value.item())


def exact_selection(model: MonotoneSubmodularQHDS, features: torch.Tensor, incidence: torch.Tensor, aspects: torch.Tensor, protected: tuple[int, ...], budget: int) -> tuple[tuple[int, ...], float, int]:
    free = tuple(position for position in range(len(features)) if position not in protected)
    needed = budget - len(protected)
    if needed < 0 or needed > len(free):
        raise ValueError("exact oracle budget does not fit candidate pool")
    best: tuple[float, tuple[int, ...]] | None = None
    evaluated = 0
    for suffix in itertools.combinations(free, needed):
        selected = tuple(protected) + tuple(suffix)
        score = utility(model, features, incidence, aspects, selected)
        evaluated += 1
        ids_key = tuple(map(str, selected))
        if best is None or score > best[0] + 1e-12 or (abs(score - best[0]) <= 1e-12 and ids_key < tuple(map(str, best[1]))):
            best = (score, selected)
    if best is None:
        raise RuntimeError("exact oracle enumerated no candidate subsets")
    return best[1], best[0], evaluated


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the non-deployable frozen learned-utility exact oracle.")
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--sparse-attempt", default="attempt_019")
    parser.add_argument("--output-revision", default=OUTPUT_REVISION)
    args = parser.parse_args()
    if args.run_id != RUN_ID:
        raise ValueError("learned utility oracle run ID is immutable")
    if not args.output_revision or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in args.output_revision):
        raise ValueError("output revision must be a simple alphanumeric/underscore/dash suffix")
    sparse_config = load_config("v21_sparse_joint_oof_r2.json")
    qesa_config = load_config("v21_qesa_r2.json")
    data = load_config("v21_data.json")
    query_vectors = load_query_vectors(data)
    memory_ids = list(map(str, read_json(data["memory"]["embedding_ids"])))
    index_by_id = {memory_id: position for position, memory_id in enumerate(memory_ids)}
    raw = np.array(np.load(resolve(data["memory"]["embedding_vectors"]), mmap_mode="r"), dtype=np.float32, copy=True)
    raw /= np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)
    rows: list[dict] = []
    per_seed_fold: dict[tuple[int, int], list[float]] = defaultdict(list)
    source_hashes: dict[str, str] = {"config": sha256_file(VERSION_ROOT / "configs" / "v21_sparse_joint_oof_r2.json")}
    for seed in sparse_config["fixed_seeds"]:
        for fold in range(int(sparse_config["outer_folds"])):
            unit_dir = VERSION_ROOT / "artifacts" / "v21_sparse_joint" / sparse_config["run_id"] / f"seed_{seed}" / f"fold_{fold}" / "attempts" / args.sparse_attempt
            manifest_path = unit_dir / "v21_sparse_joint_unit_manifest.json"
            selection_path = unit_dir / "v21_sparse_joint_selections.jsonl"
            manifest = read_json(manifest_path)
            if manifest.get("status") != "completed" or manifest.get("attempt") != args.sparse_attempt or manifest.get("calibration_read") is not False or manifest.get("formal_test_read") is not False:
                raise ValueError(f"sparse source is not an unlocked completed OOF unit: {manifest_path}")
            if sha256_file(selection_path) != manifest["hashes"]["selections"]:
                raise ValueError(f"sparse selection hash mismatch: {selection_path}")
            frontiers = load_frontiers(selection_path)
            qesa, qesa_path = load_qesa(sparse_config, qesa_config, int(seed), fold, raw.shape[1])
            qhds, qhds_path = load_qhds(sparse_config, int(seed), fold)
            incidence_path = VERSION_ROOT / "artifacts" / "v21_phase1" / sparse_config["phase1_run_id"] / f"seed_{seed}" / f"fold_{fold}" / "v21_uot_semantic_incidence.npy"
            incidence = np.asarray(np.load(incidence_path, mmap_mode="r"), dtype=np.float32)
            source_hashes[f"qesa_{seed}_{fold}"] = sha256_file(qesa_path)
            source_hashes[f"qhds_{seed}_{fold}"] = sha256_file(qhds_path)
            source_hashes[f"incidence_{seed}_{fold}"] = sha256_file(incidence_path)
            query_ids = sorted(frontiers)
            query_batch = torch.as_tensor(np.stack([query_vectors[query_id] for query_id in query_ids]), dtype=torch.float32)
            with torch.no_grad():
                aspects_batch = qesa(query_batch)[2].detach().cpu().numpy().astype(np.float32)
            for query_id, aspects in zip(query_ids, aspects_batch, strict=True):
                frontier = frontiers[query_id][:CANDIDATE_CAP]
                protected_ids = frontier[:PROTECTED_SEEDS]
                positions = np.asarray([index_by_id[memory_id] for memory_id in frontier], dtype=np.int64)
                cosine = raw[positions] @ query_vectors[query_id]
                association = incidence[positions] @ aspects
                features = build_features(torch.as_tensor(cosine), torch.as_tensor(association))
                candidate_incidence = torch.as_tensor(incidence[positions], dtype=torch.float32)
                aspects_tensor = torch.as_tensor(aspects, dtype=torch.float32)
                protected = tuple(range(PROTECTED_SEEDS))
                exact_ids_local, exact_score, evaluated = exact_selection(qhds, features, candidate_incidence, aspects_tensor, protected, BUDGET)
                greedy_local = tuple(greedy_select(qhds, features, candidate_incidence, aspects_tensor, protected, BUDGET))
                greedy_score = utility(qhds, features, candidate_incidence, aspects_tensor, greedy_local)
                regret = max(0.0, exact_score - greedy_score)
                normalized = regret / max(abs(exact_score), 1e-12)
                record = {
                    "run_id": RUN_ID, "sparse_attempt": args.sparse_attempt, "seed": seed, "fold": fold, "query_id": query_id,
                    "grid_activated_prototypes": GRID[0], "grid_members_per_semantic_edge": GRID[1], "grid_semantic_bundle_beam": GRID[2],
                    "candidate_pool_cap": CANDIDATE_CAP, "candidate_pool_count": len(frontier), "protected_seed_count": len(protected_ids), "budget": BUDGET,
                    "exact_subsets_evaluated": evaluated, "exact_score": exact_score, "greedy_score": greedy_score,
                    "regret": regret, "normalized_regret": normalized,
                    "exact_selected_memory_ids": json.dumps([frontier[position] for position in exact_ids_local], ensure_ascii=False),
                    "greedy_selected_memory_ids": json.dumps([frontier[position] for position in greedy_local], ensure_ascii=False),
                    "gold_evidence_read": False, "calibration_read": False, "formal_test_read": False,
                }
                rows.append(record)
                per_seed_fold[(int(seed), fold)].append(normalized)

    metrics_path = VERSION_ROOT / "aggregate" / f"v21_learned_utility_oracle_metrics_{args.output_revision}.csv"
    write_csv_new(metrics_path, rows)
    normalized = np.asarray([float(row["normalized_regret"]) for row in rows], dtype=np.float64)
    regret = np.asarray([float(row["regret"]) for row in rows], dtype=np.float64)
    per_fold = {f"{seed}|{fold}": {"rows": len(values), "mean_normalized_regret": float(np.mean(values)), "p95_normalized_regret": float(np.percentile(values, 95))} for (seed, fold), values in sorted(per_seed_fold.items())}
    summary = {
        "schema_version": "v21-learned-utility-oracle-1", "status": "completed_non_deployable", "run_id": RUN_ID,
        "rows": len(rows), "candidate_pool_source": "attempt_019 sparse frontier; ann_flat row at grid 16/32/32", "candidate_pool_cap": CANDIDATE_CAP, "budget": BUDGET,
        "exact_solver": "exhaustive combinations with mandatory protected seeds", "mean_exact_score": float(np.mean([float(row["exact_score"]) for row in rows])),
        "mean_greedy_score": float(np.mean([float(row["greedy_score"]) for row in rows])), "mean_regret": float(np.mean(regret)), "p50_regret": float(np.percentile(regret, 50)), "p95_regret": float(np.percentile(regret, 95)),
        "mean_normalized_regret": float(np.mean(normalized)), "p50_normalized_regret": float(np.percentile(normalized, 50)), "p95_normalized_regret": float(np.percentile(normalized, 95)), "max_normalized_regret": float(np.max(normalized)),
        "per_seed_fold": per_fold,
        "contract": {"gold_evidence_read": False, "deployment_allowed": False, "uses_frozen_qesa": True, "uses_frozen_qhds_utility": True, "calibration_read": False, "formal_test_read": False},
        "artifacts": {"metrics": str(metrics_path.relative_to(VERSION_ROOT))}, "hashes": {"metrics": sha256_file(metrics_path), **source_hashes},
        "calibration_read": False, "formal_test_read": False,
    }
    summary_path = VERSION_ROOT / "aggregate" / f"v21_learned_utility_oracle_summary_{args.output_revision}.json"
    write_json_new(summary_path, summary)
    report_path = VERSION_ROOT / "reports" / "METHOD1_V21_LEARNED_UTILITY_ORACLE_RESULTS.md"
    report = "# Method 1 v21 Learned-Utility Exact Oracle\n\n"
    report += "This non-deployable oracle solves the frozen QHDS utility exactly on a cap-12 sparse frontier at B=8. It reads no gold evidence, calibration queries, or formal-test queries.\n\n"
    report += f"- rows: {len(rows)}\n- mean exact utility: {summary['mean_exact_score']:.6f}\n- mean greedy utility: {summary['mean_greedy_score']:.6f}\n- mean regret: {summary['mean_regret']:.6f}\n- p95 regret: {summary['p95_regret']:.6f}\n- mean normalized regret: {summary['mean_normalized_regret']:.6f}\n- p95 normalized regret: {summary['p95_normalized_regret']:.6f}\n\n"
    report += "The result is utility headroom only; the exact solver is not part of the online deployment path and does not establish a retrieval or hardware advantage.\n"
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(report)
    print(json.dumps({"status": summary["status"], "rows": len(rows), "metrics_sha256": summary["hashes"]["metrics"], "mean_normalized_regret": summary["mean_normalized_regret"]}, sort_keys=True))


if __name__ == "__main__":
    main()
