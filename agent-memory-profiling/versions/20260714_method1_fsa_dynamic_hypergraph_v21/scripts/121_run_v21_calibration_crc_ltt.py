#!/usr/bin/env python3
"""Read the 290-query calibration split exactly once for CRC/LTT mapping."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.stats import beta, binomtest


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.evidence_sets import fold_for  # noqa: E402
from v21.io import iter_jsonl, load_config, read_json, resolve, sha256_file, write_csv_new, write_json_new  # noqa: E402
from v21.qesa_r2 import LatentAspectQESAR2  # noqa: E402
from v21.qhds import MonotoneSubmodularQHDS  # noqa: E402
from v21.sparse_joint import build_semantic_bundle_beam, construct_sparse_frontier  # noqa: E402
from v21.sparse_semantic_index import SparseSemanticInvertedIndex, SparseSemanticLookupRequest  # noqa: E402


RUN_ID = "v21_calibration_crc_ltt_r1_20260714"
PRE_FREEZE = "aggregate/v21_pre_calibration_freeze_r1_20260714.json"
FIXED_SEED = 20260712
POLICY_GRID = (4, 8, 8)
PROTECTED_SEEDS = 5
BUDGETS = (8, 16, 32)
ALPHA = 0.05
QUALITY_TOLERANCE = 0.01
TOP5_TOLERANCE = 0.005


def load_query_vectors(data: dict, dataset: str) -> dict[str, np.ndarray]:
    matrix = np.load(resolve(data["query_embeddings"]["vectors"]), mmap_mode="r")
    records = read_json(data["query_embeddings"]["index"])["records"]
    return {str(row["query_id"]): np.asarray(matrix[position], dtype=np.float32) for position, row in enumerate(records) if row.get("dataset") == dataset}


def load_calibration_rows(path: Path) -> list[dict]:
    """The sole calibration-plan reader. No helper may call this function twice."""
    rows = []
    for row in iter_jsonl(path):
        groups = []
        eligible = tuple(dict.fromkeys(map(str, row["eligible_memory_ids"])))
        eligible_set = set(eligible)
        for group in row.get("evidence_groups", []):
            members = tuple(dict.fromkeys(map(str, group.get("acceptable_memory_ids", []))))
            if not members or not set(members).issubset(eligible_set):
                raise ValueError(f"invalid calibration evidence group: {row['query_id']}")
            groups.append(members)
        rows.append({"query_id": str(row["query_id"]), "persona_id": str(row["persona_id"]), "query_session_id": int(row["query_session_id"]), "eligible_memory_ids": eligible, "groups": tuple(groups)})
    if len(rows) != 290 or len({row["query_id"] for row in rows}) != 290:
        raise ValueError(f"calibration plan must contain 290 unique queries, got {len(rows)}")
    return rows


def load_ann_prefix(path: Path) -> dict[str, tuple[str, ...]]:
    output: dict[str, tuple[str, ...]] = {}
    for row in iter_jsonl(path):
        query_id = str(row["query_id"])
        candidates = tuple(dict.fromkeys(map(str, row.get("ann_candidate_ids", []))))
        previous = output.get(query_id)
        if previous is not None and previous != candidates:
            raise ValueError(f"duplicate ANN prefix mismatch: {query_id}")
        output[query_id] = candidates
    return output


def qesa_model(path: Path, qesa_config: dict, dimension: int) -> LatentAspectQESAR2:
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint["qesa"]
    model = LatentAspectQESAR2(dimension, int(qesa_config["hidden_dimension"]), int(checkpoint["selected_latent_aspects"]), int(state["aspect_prototype_logits"].shape[1]), float(checkpoint["selected_regime"]["aspect_temperature"]))
    model.load_state_dict(state)
    return model.eval()


def qhds_model(path: Path) -> MonotoneSubmodularQHDS:
    checkpoint = torch.load(path, map_location="cpu")
    model = MonotoneSubmodularQHDS(3, 16)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()


def qhds_order(model: MonotoneSubmodularQHDS, features: np.ndarray, incidence: np.ndarray, aspects: np.ndarray, protected: list[int], maximum_budget: int) -> list[int]:
    feature_tensor = torch.as_tensor(features, dtype=torch.float32)
    incidence_tensor = torch.as_tensor(incidence, dtype=torch.float32)
    aspect_tensor = torch.as_tensor(aspects, dtype=torch.float32)
    with torch.no_grad():
        relevance = model.relevance(feature_tensor)
        structural_scale, modular_scale = model.positive_scales()
        structural_scale = float(structural_scale.item()); modular_scale = float(modular_scale.item())
    selected = list(dict.fromkeys(map(int, protected)))
    selected_set = set(selected)
    covered = (incidence_tensor[selected] * relevance[selected, None]).sum(dim=0) if selected else torch.zeros(incidence_tensor.shape[1])
    while len(selected) < min(maximum_budget, len(features)):
        added = incidence_tensor * relevance[:, None]
        gains = structural_scale * ((torch.log1p(covered[None, :] + added) - torch.log1p(covered[None, :])) * aspect_tensor[None, :]).sum(dim=1) + modular_scale * relevance
        if selected_set:
            gains[list(selected_set)] = -torch.inf
        position = int(torch.argmax(gains).item())
        selected.append(position); selected_set.add(position); covered += added[position]
    return selected


def clopper_pearson_upper(failures: int, total: int, alpha: float) -> float:
    if failures >= total:
        return 1.0
    if failures <= 0:
        return float(beta.ppf(1.0 - alpha, 1, total))
    return float(beta.ppf(1.0 - alpha, failures + 1, total - failures))


def holm(pvalues: dict[int, float]) -> dict[int, float]:
    ordered = sorted(pvalues.items(), key=lambda item: (item[1], item[0]))
    output: dict[int, float] = {}
    running = 0.0
    for rank, (budget, pvalue) in enumerate(ordered):
        running = max(running, min(1.0, float(pvalue) * (len(ordered) - rank)))
        output[budget] = running
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the one permitted v21 calibration CRC/LTT pass.")
    parser.add_argument("--run-id", default=RUN_ID)
    args = parser.parse_args()
    if args.run_id != RUN_ID:
        raise ValueError("calibration run ID is immutable")
    output_path = VERSION_ROOT / "aggregate" / "v21_calibration_crc_ltt_r1_20260714.json"
    metrics_path = VERSION_ROOT / "aggregate" / "v21_calibration_crc_ltt_metrics_r1_20260714.csv"
    if output_path.exists() or metrics_path.exists():
        raise FileExistsError("calibration output already exists; calibration cannot be read again")
    freeze_path = VERSION_ROOT / PRE_FREEZE
    freeze = read_json(freeze_path)
    if freeze.get("status") != "frozen_ready_for_single_calibration_read" or freeze.get("locked_split_state", {}).get("calibration_access_count") != 0 or freeze.get("locked_split_state", {}).get("formal_test_runs") != 0:
        raise ValueError("pre-calibration freeze is not locked with zero prior access")
    data = load_config("v21_data.json")
    sparse_config = load_config("v21_sparse_joint_oof_r2.json")
    qesa_config = load_config("v21_qesa_r2.json")
    qhds_config = load_config("v21_qhds.json")
    # Exactly one call to load_calibration_rows is the controlled calibration access.
    calibration_path = resolve(data["split"]["calibration_query_plan"])
    calibration_rows = load_calibration_rows(calibration_path)
    query_vectors = load_query_vectors(data, "med_calibration")
    if set(row["query_id"] for row in calibration_rows) - set(query_vectors):
        raise ValueError("calibration query plan has no frozen query embedding")
    ids = list(map(str, read_json(data["memory"]["embedding_ids"])))
    index_by_id = {memory_id: position for position, memory_id in enumerate(ids)}
    raw = np.array(np.load(resolve(data["memory"]["embedding_vectors"]), mmap_mode="r"), dtype=np.float32, copy=True)
    raw /= np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)
    ann_prefix = load_ann_prefix(resolve(data["frozen_baseline"]["train_retrieval"]))
    model_cache: dict[int, tuple[LatentAspectQESAR2, MonotoneSubmodularQHDS, SparseSemanticInvertedIndex, np.ndarray]] = {}
    for fold in range(5):
        qesa_path = VERSION_ROOT / "artifacts" / "v21_qesa" / sparse_config["qesa_run_id"] / f"seed_{FIXED_SEED}" / f"fold_{fold}" / "v21_qesa_r2_models.pt"
        qhds_path = VERSION_ROOT / "artifacts" / "v21_qhds" / sparse_config["dense_qhds_run_id"] / f"seed_{FIXED_SEED}" / f"fold_{fold}" / "v21_qhds_model.pt"
        incidence_path = VERSION_ROOT / "artifacts" / "v21_phase1" / sparse_config["phase1_run_id"] / f"seed_{FIXED_SEED}" / f"fold_{fold}" / "v21_uot_semantic_incidence.npy"
        index_manifest_path = VERSION_ROOT / "artifacts" / "v21_sparse_semantic_index" / sparse_config["index_run_id"] / f"seed_{FIXED_SEED}" / f"fold_{fold}" / "attempts" / sparse_config["index_attempt"] / "v21_sparse_semantic_index_manifest.json"
        index_manifest = read_json(index_manifest_path)
        rank_path = VERSION_ROOT / index_manifest["artifacts"]["prototype_memory_rank_index"]
        incidence = np.array(np.load(incidence_path, mmap_mode="r"), dtype=np.float32, copy=False)
        ranks = np.load(rank_path, mmap_mode="r")
        model_cache[fold] = (qesa_model(qesa_path, qesa_config, raw.shape[1]), qhds_model(qhds_path), SparseSemanticInvertedIndex(ranks, incidence), incidence)

    records: list[dict] = []
    ann_sources = Counter()
    for row in calibration_rows:
        fold = fold_for(row["persona_id"], row["query_session_id"], 5)
        qesa, qhds, sparse_index, incidence = model_cache[fold]
        query = query_vectors[row["query_id"]]
        eligible_global = frozenset(index_by_id[memory_id] for memory_id in row["eligible_memory_ids"])
        ann_candidates = tuple(memory_id for memory_id in ann_prefix.get(row["query_id"], ()) if memory_id in set(row["eligible_memory_ids"]))
        if len(ann_candidates) >= PROTECTED_SEEDS:
            protected_ids = ann_candidates[:PROTECTED_SEEDS]; ann_source = "frozen_train_retrieval_ann_prefix"
        else:
            eligible_positions = np.asarray(sorted(eligible_global), dtype=np.int64)
            cosine = raw[eligible_positions] @ query
            order = np.argsort(-cosine, kind="stable")[:PROTECTED_SEEDS]
            protected_ids = tuple(ids[int(eligible_positions[position])] for position in order); ann_source = "frozen_memory_embedding_flat_prefix_fallback"
        ann_sources[ann_source] += 1
        with torch.no_grad():
            aspect = qesa(torch.as_tensor(query[None, :], dtype=torch.float32))[2][0].cpu().numpy().astype(np.float32)
        active = np.argsort(-aspect, kind="stable")[:POLICY_GRID[0]].astype(int).tolist()
        lookup = sparse_index.lookup(SparseSemanticLookupRequest(tuple(active), eligible_global, POLICY_GRID[1]))
        member_scores = {}
        for edge in lookup.edges:
            for position in edge.memory_positions:
                member_scores[int(position)] = max(member_scores.get(int(position), -np.inf), float(raw[int(position)] @ query))
        prototype_weights = {int(prototype): float(aspect[int(prototype)]) for prototype in active}
        bundles, bundle_audit = build_semantic_bundle_beam(lookup.edges, prototype_weights, member_scores, arity=3, beam_width=POLICY_GRID[2])
        protected_positions = [index_by_id[memory_id] for memory_id in protected_ids]
        frontier = construct_sparse_frontier(protected_positions, bundles, lookup.edges, member_scores, maximum_budget=max(BUDGETS))
        positions = np.asarray(frontier.memory_positions, dtype=np.int64)
        protected_local = [int(np.where(positions == position)[0][0]) for position in protected_positions]
        cosine = raw[positions] @ query
        association = incidence[positions] @ aspect
        features = np.stack((cosine, association, cosine * association), axis=1).astype(np.float32)
        order = qhds_order(qhds, features, incidence[positions], aspect, protected_local, max(BUDGETS))
        for budget in BUDGETS:
            selected_positions = order[: min(budget, len(order))]
            selected_ids = [ids[int(positions[position])] for position in selected_positions]
            covered = [bool(set(selected_ids).intersection(group)) for group in row["groups"]]
            top5 = set(selected_ids[:5])
            top5_covered = [bool(top5.intersection(group)) for group in row["groups"]]
            records.append({
                "run_id": RUN_ID, "seed": FIXED_SEED, "fold": fold, "query_id": row["query_id"], "persona_id": row["persona_id"], "query_session_id": row["query_session_id"], "budget": budget,
                "activated_prototypes": POLICY_GRID[0], "members_per_semantic_edge": POLICY_GRID[1], "semantic_bundle_beam": POLICY_GRID[2], "protected_seed_count": PROTECTED_SEEDS, "ann_seed_source": ann_source,
                "activated_semantic_edges": lookup.activated_semantic_edges, "incidence_entries_visited": lookup.incidence_entries_visited, "bundle_hyperedges_generated": bundle_audit["bundle_hyperedges_generated"], "bundle_hyperedges_scored": bundle_audit["bundle_hyperedges_scored"], "duplicate_bundle_expansions": bundle_audit["duplicate_bundle_expansions"], "unique_expanded_memories": len(set(frontier.memory_positions)), "semantic_bundle_unique_members": frontier.semantic_bundle_unique_members, "fallback_members_appended": frontier.fallback_members_appended, "budget_infeasible": bool(frontier.budget_infeasible or len(order) < budget), "candidate_count": len(frontier.memory_positions),
                "group_coverage": float(np.mean(covered)) if covered else 0.0, "all_groups_covered": int(all(covered)) if covered else 1, "final_top5_group_coverage": float(np.mean(top5_covered)) if top5_covered else 0.0, "final_top5_all_groups_covered": int(all(top5_covered)) if top5_covered else 1,
                "selected_memory_ids": json.dumps(selected_ids, ensure_ascii=False), "calibration_read": True, "formal_test_read": False,
            })

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv_new(metrics_path, records)
    # Dense B=32 train-OOF risk is frozen before calibration and is used only as
    # the CRC/LTT comparison target; no selector/model state is changed here.
    core_summary = read_json(VERSION_ROOT / "aggregate" / "v21_sparse_joint_core_summary_r3_attempt019.json")
    dense_key = next((key for key in core_summary["pooled_fixed_count"] if key.startswith("dense_qhds_reference|") and key.endswith("|32")), None)
    if dense_key is None:
        raise ValueError("frozen dense B=32 reference is missing")
    dense32 = core_summary["pooled_fixed_count"][dense_key]
    dense32_risk = 1.0 - float(dense32["all_groups_covered"])
    dense32_group_risk = 1.0 - float(dense32["group_coverage"])
    budget_summary = {}
    pvalues = {}
    for budget in BUDGETS:
        selected = [record for record in records if int(record["budget"]) == budget]
        failures = int(sum(1 - int(record["all_groups_covered"]) for record in selected))
        top5_failures = int(sum(1 - int(record["final_top5_all_groups_covered"]) for record in selected))
        total = len(selected)
        risk = failures / max(total, 1); top5_risk = top5_failures / max(total, 1)
        upper = clopper_pearson_upper(failures, total, ALPHA)
        top5_upper = clopper_pearson_upper(top5_failures, total, ALPHA)
        crc_target = dense32_risk + QUALITY_TOLERANCE
        ltt_target = dense32_group_risk + TOP5_TOLERANCE
        pvalues[budget] = float(binomtest(failures, total, min(max(crc_target, 1e-12), 1.0), alternative="less").pvalue)
        budget_summary[str(budget)] = {"rows": total, "all_groups_covered_failures": failures, "frontier_risk": risk, "frontier_risk_upper_95": upper, "final_top5_failures": top5_failures, "final_top5_risk": top5_risk, "final_top5_risk_upper_95": top5_upper, "crc_target": crc_target, "ltt_target": ltt_target, "crc_pass": bool(upper <= crc_target), "ltt_raw_pass": bool(top5_upper <= ltt_target)}
    adjusted = holm(pvalues)
    for budget in BUDGETS:
        budget_summary[str(budget)]["crc_ltt_holm_p_value"] = adjusted[budget]
        budget_summary[str(budget)]["ltt_pass_after_holm"] = bool(budget_summary[str(budget)]["ltt_raw_pass"] and adjusted[budget] <= ALPHA)
    selected_budget = next((budget for budget in BUDGETS if budget_summary[str(budget)]["crc_pass"]), BUDGETS[-1])
    summary = {
        "schema_version": "v21-calibration-crc-ltt-1", "status": "completed_single_calibration_read", "run_id": RUN_ID, "calibration_access_count": 1, "calibration_query_count": len(calibration_rows), "formal_test_runs": 0,
        "fixed_policy": {"grid": list(POLICY_GRID), "protected_ann_seeds": PROTECTED_SEEDS, "model_seed": FIXED_SEED, "selector": "frozen QESA r2 + sparse r3 frontier + frozen QHDS greedy", "candidate_frontier_changed": False},
        "dense32_reference": {"source_key": dense_key, "all_groups_covered_risk": dense32_risk, "group_coverage_risk": dense32_group_risk}, "budgets": budget_summary, "crc_selected_budget": selected_budget, "ltt_selected_budget_pass_after_holm": budget_summary[str(selected_budget)]["ltt_pass_after_holm"],
        "ann_seed_sources": dict(sorted(ann_sources.items())), "source_hashes": {"pre_calibration_freeze": sha256_file(freeze_path), "calibration_plan": sha256_file(calibration_path), "query_embeddings": sha256_file(resolve(data["query_embeddings"]["vectors"])), "query_index": sha256_file(resolve(data["query_embeddings"]["index"])), "memory_embeddings": sha256_file(resolve(data["memory"]["embedding_vectors"])), "ann_prefix": sha256_file(resolve(data["frozen_baseline"]["train_retrieval"])), "core_summary": sha256_file(VERSION_ROOT / "aggregate" / "v21_sparse_joint_core_summary_r3_attempt019.json")},
        "calibration_use": "CRC/LTT risk mapping only; no model, prototype, selector, hyperparameter, layout, or dynamic-policy changes",
        "artifacts": {"metrics": str(metrics_path.relative_to(VERSION_ROOT))}, "hashes": {"metrics": sha256_file(metrics_path)}, "calibration_read": True, "formal_test_read": False,
    }
    write_json_new(output_path, summary)
    report_path = VERSION_ROOT / "reports" / "METHOD1_V21_CALIBRATION_CRC_LTT_R1.md"
    report = "# Method 1 v21 Calibration CRC/LTT Mapping\n\n"
    report += "This is the single permitted calibration read: 290 calibration queries were evaluated after the pre-calibration freeze. The fixed policy was selected from OOF before this read; calibration only maps CRC/LTT risk and selects B.\n\n"
    report += f"- calibration access count: `1`\n- formal-test runs: `0`\n- fixed grid: `{POLICY_GRID}`\n- CRC-selected B: `{selected_budget}`\n- ANN seed sources: `{dict(ann_sources)}`\n\n"
    report += "| B | Frontier risk | 95% upper | CRC target | CRC pass | Final top-5 risk | 95% upper | LTT target | LTT after Holm |\n| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |\n"
    for budget in BUDGETS:
        item = budget_summary[str(budget)]
        report += f"| {budget} | {item['frontier_risk']:.6f} | {item['frontier_risk_upper_95']:.6f} | {item['crc_target']:.6f} | {item['crc_pass']} | {item['final_top5_risk']:.6f} | {item['final_top5_risk_upper_95']:.6f} | {item['ltt_target']:.6f} | {item['ltt_pass_after_holm']} |\n"
    report += "\nNo model, prototype, selector, hyperparameter, payload layout, or dynamic policy was changed. Formal test remains locked for the final one-run operation.\n"
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(report)
    print(json.dumps({"status": summary["status"], "calibration_access_count": 1, "calibration_query_count": len(calibration_rows), "crc_selected_budget": selected_budget, "ltt_selected_budget_pass_after_holm": summary["ltt_selected_budget_pass_after_holm"]}, sort_keys=True))


if __name__ == "__main__":
    main()
