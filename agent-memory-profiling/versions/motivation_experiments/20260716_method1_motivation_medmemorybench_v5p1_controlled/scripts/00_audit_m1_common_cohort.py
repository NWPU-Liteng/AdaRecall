#!/usr/bin/env python3
"""Audit and freeze a common M1 query cohort before any model execution.

The audit is deliberately data-only.  It reads the already completed M2 query
workload, checks historical eligibility under both M1 conditions, and writes a
manifest that the controlled runner must consume.  No retrieval/model output is
used to select the cohort.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT.parents[0] / "20260704_method1_motivation_medmemorybench_v3"
sys.path.insert(0, str(V3 / "scripts"))
from medmmb_common import (  # noqa: E402
    eligible_memories_for_query,
    group_memories_by_persona,
    read_jsonl,
)


CONDITIONS = ("medical_only", "medical_plus_official_noise")
SCALE_GRID = (32, 64, 128, 256, 512, 1024, 1536)


def _sha256_ids(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m2-log", default=str(V3 / "aggregate/M2_frontier_budget_invocations.jsonl"))
    parser.add_argument("--queries", default=str(V3 / "aggregate/queries_medmemorybench_en.jsonl"))
    parser.add_argument("--memory-items", default=str(V3 / "aggregate/memory_items_turn_pair.jsonl"))
    parser.add_argument("--evidence-mapping", default=str(V3 / "aggregate/query_evidence_mapping.jsonl"))
    parser.add_argument("--output-root", default=str(ROOT))
    args = parser.parse_args()

    out_root = Path(args.output_root).resolve()
    aggregate = out_root / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)

    m2_rows = read_jsonl(args.m2_log)
    cohort_ids = sorted({str(row["query_id"]) for row in m2_rows})
    if not cohort_ids:
        raise RuntimeError("M2 log contains no query IDs")
    query_rows = {str(row["query_id"]): row for row in read_jsonl(args.queries)}
    missing = sorted(set(cohort_ids) - set(query_rows))
    if missing:
        raise RuntimeError(f"M2 cohort queries missing from query file: {missing[:5]}")
    queries = [query_rows[qid] for qid in cohort_ids]
    type_counts = Counter(str(row.get("query_type", "unknown")) for row in queries)
    if len(type_counts) != 6 or min(type_counts.values()) < 40:
        raise RuntimeError(f"Unexpected M2 cohort balance: {dict(type_counts)}")

    evidence = {
        str(row["query_id"]): {str(mid) for mid in row.get("gold_memory_ids", [])}
        for row in read_jsonl(args.evidence_mapping)
    }
    memories_by_persona = group_memories_by_persona(read_jsonl(args.memory_items))

    audit_rows: list[dict[str, Any]] = []
    eligible_counts: dict[str, dict[str, int]] = {condition: {} for condition in CONDITIONS}
    for query in queries:
        qid = str(query["query_id"])
        gold_count = len(evidence.get(qid, set()))
        for condition in CONDITIONS:
            include_noise = condition == "medical_plus_official_noise"
            eligible = eligible_memories_for_query(
                memories_by_persona,
                query,
                allow_current_session=True,
                include_noise=include_noise,
            )
            count = len(eligible)
            eligible_counts[condition][qid] = count
            audit_rows.append(
                {
                    "query_id": qid,
                    "query_type": query.get("query_type", "unknown"),
                    "persona_id": query.get("persona_id", ""),
                    "query_session_id": query.get("query_session_id", ""),
                    "condition": condition,
                    "eligible_count": count,
                    "gold_count": gold_count,
                    "gold_within_eligible": int(all(mid in {str(row["memory_id"]) for row in eligible} for mid in evidence.get(qid, set()))),
                    **{f"feasible_{scale}": int(count >= scale and gold_count <= scale) for scale in SCALE_GRID},
                }
            )

    common_feasible = {
        str(scale): [
            qid
            for qid in cohort_ids
            if all(eligible_counts[condition][qid] >= scale and len(evidence.get(qid, set())) <= scale for condition in CONDITIONS)
        ]
        for scale in SCALE_GRID
    }

    # Prefer a three-point, directly comparable trend.  The cohort is the
    # largest set for which every query is feasible at every selected scale;
    # unlike the old M1, no cell is allowed to silently drop queries.
    scale_candidates = [(128, 256, 512), (64, 128, 256), (32, 64, 128)]
    selected_scale_set: tuple[int, ...] | None = None
    selected_ids: list[str] = []
    for scales in scale_candidates:
        candidates = [
            qid
            for qid in cohort_ids
            if all(qid in common_feasible[str(scale)] for scale in scales)
        ]
        types = {str(query_rows[qid].get("query_type", "unknown")) for qid in candidates}
        if len(candidates) >= 60 and len(types) == 6:
            selected_scale_set = scales
            selected_ids = sorted(candidates)
            break
    if selected_scale_set is None:
        raise RuntimeError("No common cohort with at least 60 queries and all six query types")

    selected_query_rows = [query_rows[qid] for qid in selected_ids]
    selected_types = Counter(str(row.get("query_type", "unknown")) for row in selected_query_rows)
    query_ids_hash = _sha256_ids(selected_ids)
    _write_csv(aggregate / "v5p1_m1_feasibility_audit.csv", audit_rows)
    (aggregate / "v5p1_m1_common_query_ids.json").write_text(
        json.dumps(
            {
                "schema_version": "v5p1-m1-cohort-1",
                "source_m2_log": str(Path(args.m2_log).resolve()),
                "source_query_count": len(cohort_ids),
                "source_query_ids_hash": _sha256_ids(cohort_ids),
                "selected_query_count": len(selected_ids),
                "selected_query_ids": selected_ids,
                "selected_query_ids_hash": query_ids_hash,
                "selected_query_type_counts": dict(sorted(selected_types.items())),
                "selected_scale_set": list(selected_scale_set),
                "selected_at_max_common_scale": max(selected_scale_set),
                "common_feasible_counts": {scale: len(ids) for scale, ids in common_feasible.items()},
                "common_scales_for_selected_cohort": list(selected_scale_set),
                "conditions": list(CONDITIONS),
                "allow_current_session_context": True,
                "scale_grid": list(SCALE_GRID),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (aggregate / "v5p1_m1_common_queries.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected_query_rows),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "source_query_count": len(cohort_ids),
                "source_type_counts": dict(sorted(type_counts.items())),
                "selected_query_count": len(selected_ids),
                "selected_type_counts": dict(sorted(selected_types.items())),
                "selected_scale_set": list(selected_scale_set),
                "selected_at_max_common_scale": max(selected_scale_set),
                "common_feasible_counts": {scale: len(ids) for scale, ids in common_feasible.items()},
                "common_scales_for_selected_cohort": list(selected_scale_set),
                "query_ids_hash": query_ids_hash,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
