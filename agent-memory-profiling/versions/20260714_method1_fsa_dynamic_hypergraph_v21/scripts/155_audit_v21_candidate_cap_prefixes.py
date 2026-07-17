#!/usr/bin/env python3
"""Audit smaller prefixes of the exact cap-96 candidate frontier."""
from __future__ import annotations

import json
import runpy
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))
from v21.io import iter_jsonl, load_config, read_json, write_json_new  # noqa: E402


def main() -> None:
    config = load_config("v21_sparse_frontier_provenance_oof_r1.json")
    data = load_config("v21_data.json")
    helpers = runpy.run_path(str(VERSION_ROOT / "scripts/90_train_v21_qesa_oof.py"), run_name="cap_audit_helpers")
    query_vectors = helpers["load_query_vectors"](data)
    rows_by_fold = helpers["load_rows"](data, query_vectors, 5)
    rows = {str(row["query_id"]): row for fold in range(5) for row in rows_by_fold[fold]}
    caps = (32, 48, 64, 96)
    totals = {cap: defaultdict(float) for cap in caps}
    count = 0
    root = VERSION_ROOT / "artifacts/v21_sparse_frontier_provenance" / "v21_sparse_frontier_provenance_oof_r1_20260715"
    for path in sorted(root.glob("seed_*/fold_*/attempts/attempt_001/v21_sparse_frontier_provenance.jsonl")):
        for provenance in iter_jsonl(path):
            row = rows[str(provenance["query_id"])]
            groups = row["groups"]
            frontier = list(map(str, provenance["frontier_memory_ids"]))
            protected = set(map(str, provenance["protected_ann_memory_ids"]))
            relevant = {memory_id for group in groups for memory_id in group}
            for cap in caps:
                candidates = set(frontier[:cap])
                covered = [bool(candidates.intersection(group)) for group in groups]
                totals[cap]["group_recall_sum"] += float(np.mean(covered)) if covered else 0.0
                totals[cap]["member_recall_sum"] += len(candidates.intersection(relevant)) / max(len(relevant), 1)
                totals[cap]["all_groups_sum"] += float(all(covered)) if covered else 0.0
                totals[cap]["protected_group_sum"] += float(np.mean([bool(protected.intersection(group)) for group in groups])) if groups else 0.0
                totals[cap]["candidate_count_sum"] += len(candidates)
            count += 1
    summary = {
        "schema_version": "v21-candidate-cap-prefix-audit-1",
        "status": "completed_diagnostic_not_deployable",
        "rows": count,
        "caps": {str(cap): {"candidate_count_mean": totals[cap]["candidate_count_sum"] / count, "group_recall_upper_bound_mean": totals[cap]["group_recall_sum"] / count, "member_recall_upper_bound_mean": totals[cap]["member_recall_sum"] / count, "all_groups_covered_rate": totals[cap]["all_groups_sum"] / count, "protected_group_recall_mean": totals[cap]["protected_group_sum"] / count} for cap in caps},
        "interpretation": "prefix audit only; exact cap-specific lookup and selector OOF must be run before deployment",
        "calibration_read": False,
        "formal_test_read": False,
        "minimum_effect_threshold": None,
    }
    output = VERSION_ROOT / "aggregate/v21_candidate_cap_prefix_audit_r1_attempt001.json"
    write_json_new(output, summary)
    print(json.dumps({"status": summary["status"], "rows": count, "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
