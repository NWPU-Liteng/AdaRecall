#!/usr/bin/env python3
"""Measure evidence ambiguity and multi-group demand without a method run."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
AMP_ROOT = ROOT.parents[1]
V3 = AMP_ROOT / "versions" / "20260704_method1_motivation_medmemorybench_v3"
V21 = AMP_ROOT / "versions" / "20260714_method1_fsa_dynamic_hypergraph_v21"
QA = AMP_ROOT / "versions" / "20260711_method1_qa_basis_formal_v1"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else float("nan")


def main() -> None:
    protocol = json.loads((ROOT / "configs" / "v5_protocol.json").read_text(encoding="utf-8"))
    plan_path = QA / "artifacts" / "keypoint_basis_v6_schedule_layout_candidate" / "query_split" / "query_plan_train.jsonl"
    memory_path = V3 / "aggregate" / "memory_items_turn_pair.jsonl"
    query_path = V3 / "aggregate" / "queries_medmemorybench_en.jsonl"
    memory_ids_path = V21 / "configs" / "v21_data.json"
    v21_data = json.loads(memory_ids_path.read_text(encoding="utf-8"))
    memory_ids_file = V21 / v21_data["memory"]["embedding_ids"]
    memory_vectors_file = V21 / v21_data["memory"]["embedding_vectors"]
    # v21_data paths are relative to the v21 version root.
    if not memory_ids_file.exists():
        memory_ids_file = V21 / ".." / v21_data["memory"]["embedding_ids"]
    if not memory_vectors_file.exists():
        memory_vectors_file = V21 / ".." / v21_data["memory"]["embedding_vectors"]
    # The data config points to sibling versions; resolve the canonical paths.
    memory_ids_file = V3 / "aggregate" / "embedding_cache" / "f5a55a7f678df5fa.ids.json"
    memory_vectors_file = V3 / "aggregate" / "embedding_cache" / "f5a55a7f678df5fa.npy"
    query_vectors_file = QA / "artifacts" / "keypoint_basis_v15_structured_selector" / "embeddings" / "query_vectors.npy"
    query_index_file = QA / "artifacts" / "keypoint_basis_v15_structured_selector" / "embeddings" / "query_index.json"

    memories = {str(row["memory_id"]): row for row in read_jsonl(memory_path)}
    queries = {str(row["query_id"]): row for row in read_jsonl(query_path)}
    plans = read_jsonl(plan_path)
    memory_ids = list(map(str, json.loads(memory_ids_file.read_text(encoding="utf-8"))))
    memory_position = {memory_id: idx for idx, memory_id in enumerate(memory_ids)}
    memory_vectors = np.asarray(np.load(memory_vectors_file, mmap_mode="r"), dtype=np.float32)
    memory_norm = np.linalg.norm(memory_vectors, axis=1, keepdims=True)
    memory_vectors = memory_vectors / np.maximum(memory_norm, 1e-12)
    query_index = json.loads(query_index_file.read_text(encoding="utf-8"))
    records = query_index["records"]
    query_vectors = np.asarray(np.load(query_vectors_file, mmap_mode="r"), dtype=np.float32)
    query_position = {
        str(record["query_id"]): idx
        for idx, record in enumerate(records)
        if str(record.get("dataset", "")).startswith("med_train")
    }
    if len(query_position) < len(plans):
        raise RuntimeError(f"query embeddings cover {len(query_position)} train queries, plans contain {len(plans)}")
    query_norm = np.linalg.norm(query_vectors, axis=1, keepdims=True)
    query_vectors = query_vectors / np.maximum(query_norm, 1e-12)
    budgets = [int(value) for value in protocol["s1"]["budgets"]]

    rows: list[dict] = []
    missing = Counter()
    for plan in plans:
        query_id = str(plan["query_id"])
        query = queries.get(query_id, {})
        groups = [group for group in plan.get("evidence_groups", []) if group.get("acceptable_memory_ids")]
        acceptable_groups = [set(map(str, group["acceptable_memory_ids"])) for group in groups]
        eligible = [str(memory_id) for memory_id in plan.get("eligible_memory_ids", [])]
        eligible = [memory_id for memory_id in eligible if memory_id in memory_position]
        if query_id not in query_position or not eligible or not acceptable_groups:
            missing["query_or_group_or_embedding"] += 1
            continue
        query_vector = query_vectors[query_position[query_id]]
        positions = np.asarray([memory_position[memory_id] for memory_id in eligible], dtype=np.int64)
        scores = memory_vectors[positions] @ query_vector
        order = np.argsort(-scores, kind="stable")
        ranked = [eligible[int(idx)] for idx in order]
        group_members = set().union(*acceptable_groups)
        group_overlap = sum(len(left & right) for idx, left in enumerate(acceptable_groups) for right in acceptable_groups[idx + 1:])
        source_sessions = set()
        for group in groups:
            for lag in group.get("source_session_lags", []):
                source_sessions.add(int(plan.get("query_session_id", 0)) - int(lag))
        if not source_sessions:
            source_sessions = {int(plan.get("query_session_id", 0))}
        row = {
            "query_id": query_id,
            "query_type": str(plan.get("query_type", query.get("query_type", "unknown"))),
            "persona_id": str(plan.get("persona_id", query.get("persona_id", ""))),
            "query_session_id": int(plan.get("query_session_id", query.get("query_session_id", 0))),
            "evidence_group_count": len(acceptable_groups),
            "acceptable_memory_count": len(group_members),
            "alternative_count_total": sum(len(group) for group in acceptable_groups),
            "alternative_count_mean": float(np.mean([len(group) for group in acceptable_groups])),
            "cross_group_memory_overlap": group_overlap,
            "source_session_count": len(source_sessions),
            "source_session_span": max(source_sessions) - min(source_sessions) if source_sessions else 0,
        }
        for budget in budgets:
            selected = set(ranked[:budget])
            covered = sum(bool(selected & group) for group in acceptable_groups)
            row[f"flat_group_coverage_b{budget}"] = covered / float(len(acceptable_groups))
            row[f"flat_all_groups_b{budget}"] = int(covered == len(acceptable_groups))
            row[f"flat_redundancy_b{budget}"] = max(0, min(budget, len(ranked)) - covered)
        # A margin near zero means the top alternatives are not cleanly separated
        # by the frozen generic embedding, which is a workload ambiguity signal.
        group_margins = []
        score_by_id = {memory_id: float(score) for memory_id, score in zip(eligible, scores, strict=True)}
        for group in acceptable_groups:
            group_scores = sorted((score_by_id[memory_id] for memory_id in group if memory_id in score_by_id), reverse=True)
            if len(group_scores) >= 2:
                group_margins.append(group_scores[0] - group_scores[1])
        row["acceptable_top2_cosine_margin_mean"] = float(np.mean(group_margins)) if group_margins else float("nan")
        rows.append(row)

    if len(rows) < int(protocol["s1"]["minimum_query_count"]):
        raise RuntimeError(f"S1 retained only {len(rows)} queries; missing={dict(missing)}")
    output = ROOT / "aggregate" / "v5_s1_evidence_structure_per_query.csv"
    write_csv(output, rows)
    summary: list[dict] = []
    for query_type in sorted({str(row["query_type"]) for row in rows} | {"all"}):
        subset = rows if query_type == "all" else [row for row in rows if row["query_type"] == query_type]
        if not subset:
            continue
        item = {"query_type": query_type, "n": len(subset)}
        for field in (
            "evidence_group_count", "acceptable_memory_count", "alternative_count_total",
            "alternative_count_mean", "cross_group_memory_overlap", "source_session_span",
            "acceptable_top2_cosine_margin_mean",
        ):
            values = [float(row[field]) for row in subset if math.isfinite(float(row[field]))]
            item[f"{field}_mean"] = float(np.mean(values)) if values else float("nan")
            item[f"{field}_p50"] = percentile(values, 50)
        for budget in budgets:
            item[f"flat_group_coverage_b{budget}_mean"] = float(np.mean([row[f"flat_group_coverage_b{budget}"] for row in subset]))
            item[f"flat_all_groups_b{budget}_rate"] = float(np.mean([row[f"flat_all_groups_b{budget}"] for row in subset]))
            item[f"flat_redundancy_b{budget}_mean"] = float(np.mean([row[f"flat_redundancy_b{budget}"] for row in subset]))
        summary.append(item)
    summary_path = ROOT / "aggregate" / "v5_s1_evidence_structure_summary.csv"
    write_csv(summary_path, summary)
    result = {
        "schema_version": "motivation-v5-s1-result-1",
        "status": "completed",
        "rows": len(rows),
        "plans": len(plans),
        "missing": dict(missing),
        "budgets": budgets,
        "input_boundary": "raw memory/query plus train-only evidence groups and frozen embeddings",
        "method_outputs_read": False,
        "stage_inputs_read": False,
        "calibration_read": False,
        "formal_test_read": False,
        "artifacts": {
            "per_query": str(output.relative_to(ROOT)),
            "summary": str(summary_path.relative_to(ROOT)),
        },
        "hashes": {
            "protocol": sha256(ROOT / "configs" / "v5_protocol.json"),
            "per_query": sha256(output),
            "summary": sha256(summary_path),
        },
    }
    result_path = ROOT / "aggregate" / "v5_s1_evidence_structure_summary.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
