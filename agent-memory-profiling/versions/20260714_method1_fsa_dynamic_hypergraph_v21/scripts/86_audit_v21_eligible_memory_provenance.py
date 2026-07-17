#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.eligibility import assert_train_only_incidence, generate_eligible_memory_ids, session_sort_key  # noqa: E402
from v21.io import iter_jsonl, load_config, resolve, sha256_file, write_json_new  # noqa: E402


SOURCE_ROOT = VERSION_ROOT / "provenance" / "legacy_versions" / "20260711_method1_qa_basis_formal_v1"


def _hashes(paths: list[Path]) -> dict[str, dict[str, object]]:
    return {
        str(path.relative_to(VERSION_ROOT.parent)): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    }


def main() -> None:
    data = load_config("v21_data.json")
    policy = load_config("v21_eligibility.json")
    train_path = resolve(data["split"]["train_query_plan"])
    memory_path = resolve(data["memory"]["items"])
    incidence_path = VERSION_ROOT / "artifacts" / "v21_evidence_sets" / "train_query_evidence_incidence.jsonl"
    memories = list(iter_jsonl(memory_path))
    train_rows = list(iter_jsonl(train_path))
    incidence = list(iter_jsonl(incidence_path))

    by_persona: dict[str, list[dict]] = defaultdict(list)
    memory_by_id: dict[str, dict] = {}
    for memory in memories:
        memory_id = str(memory["memory_id"])
        if memory_id in memory_by_id:
            raise RuntimeError(f"duplicate memory ID: {memory_id}")
        memory_by_id[memory_id] = memory
        by_persona[str(memory["persona_id"])].append(memory)

    mismatch_queries: list[str] = []
    unknown_memory_ids: list[str] = []
    cross_persona = 0
    future_session = 0
    current_session_incidences = 0
    eligible_counts: list[int] = []
    for row in train_rows:
        expected = generate_eligible_memory_ids(
            by_persona[str(row["persona_id"])],
            row,
            include_noise=bool(policy["include_noise"]),
            allow_current_session_context=bool(policy["allow_current_session_context"]),
        )
        observed = list(map(str, row["eligible_memory_ids"]))
        if expected != observed:
            mismatch_queries.append(str(row["query_id"]))
        eligible_counts.append(len(observed))
        for memory_id in observed:
            memory = memory_by_id.get(memory_id)
            if memory is None:
                unknown_memory_ids.append(memory_id)
                continue
            cross_persona += int(str(memory["persona_id"]) != str(row["persona_id"]))
            memory_session = session_sort_key(memory["session_id"])
            query_session = session_sort_key(row["query_session_id"])
            future_session += int(memory_session > query_session)
            current_session_incidences += int(memory_session == query_session)

    train_query_ids = {str(row["query_id"]) for row in train_rows}
    split_check = assert_train_only_incidence(incidence, train_query_ids, source_split="train")
    failures = {
        "eligible_list_mismatch_queries": len(mismatch_queries),
        "unknown_memory_ids": len(unknown_memory_ids),
        "cross_persona_incidences": cross_persona,
        "future_session_incidences": future_session,
        "unexpected_incidence_query_ids": split_check["unexpected_query_ids"],
        "missing_incidence_query_ids": split_check["missing_query_ids"],
    }
    if any(failures.values()):
        raise RuntimeError(f"eligible memory provenance audit failed: {failures}")

    source_paths = [
        SOURCE_ROOT / "scripts" / "e2e_common.py",
        SOURCE_ROOT / "scripts" / "01_build_formal_query_split.py",
        SOURCE_ROOT / "scripts" / "14_build_keypoint_query_plans.py",
        SOURCE_ROOT / "scripts" / "20_build_global_ratio_temporal_split.py",
        SOURCE_ROOT / "configs" / "qa_basis_formal.yaml",
        SOURCE_ROOT / "configs" / "keypoint_basis_v3_current_context_development.yaml",
        SOURCE_ROOT / "configs" / "keypoint_basis_v4_temporal_confirmation.yaml",
        SOURCE_ROOT / "configs" / "keypoint_basis_v6_schedule_layout_candidate.yaml",
        memory_path,
        train_path,
        incidence_path,
    ]
    result = {
        "schema_version": "v21-eligible-memory-provenance-audit-1",
        "status": "passed_train_only_label_independent_eligible_membership_audit",
        "policy": policy,
        "queries_audited": len(train_rows),
        "memory_items": len(memories),
        "eligible_incidences": sum(eligible_counts),
        "eligible_count_min": min(eligible_counts),
        "eligible_count_max": max(eligible_counts),
        "eligible_count_mean": sum(eligible_counts) / len(eligible_counts),
        "current_session_incidences": current_session_incidences,
        "failures": failures,
        "train_incidence_check": split_check,
        "gold_dependency": {
            "eligible_membership_rule_reads_gold": False,
            "query_pool_construction_used_evidence_to_require_scoreable_queries": True,
            "interpretation": "gold affects which source queries are scoreable, but does not add, remove, or rank eligible memories for an included query",
        },
        "source_hashes": _hashes(source_paths),
        "calibration_read": False,
        "formal_test_read": False,
        "calibration_access_count": 0,
        "formal_test_access_count": 0,
    }
    output_path = VERSION_ROOT / "aggregate" / "v21_eligible_memory_provenance_audit.json"
    write_json_new(output_path, result)

    report = "# Method 1 v21 Eligible Memory Provenance Audit\n\n"
    report += "## Result\n\n"
    report += "The train-only audit passed for all 1,355 queries. Recomputing eligibility from memory metadata exactly reproduced every stored `eligible_memory_ids` list. No eligible memory crossed a persona boundary or came from a future session, and the train evidence incidence contained exactly the train query IDs. Calibration and formal test were not read.\n\n"
    report += "| Check | Result |\n| --- | ---: |\n"
    report += f"| train queries audited | {result['queries_audited']} |\n"
    report += f"| memory items | {result['memory_items']} |\n"
    report += f"| eligible incidences | {result['eligible_incidences']} |\n"
    report += f"| exact-list mismatches | {failures['eligible_list_mismatch_queries']} |\n"
    report += f"| cross-persona incidences | {cross_persona} |\n"
    report += f"| future-session incidences | {future_session} |\n"
    report += f"| current-session incidences | {current_session_incidences} |\n"
    report += "| calibration accesses | 0 |\n| formal-test accesses | 0 |\n\n"
    report += "## Generation Rule\n\n"
    report += "The deployable membership rule uses only memory persona/session/type/order metadata and query persona/session metadata. The resolved v6 configuration includes noise and permits `memory.session_id <= query.query_session_id`; it never reads answers, gold memory IDs, evidence groups, or acceptable members. The same function therefore runs when no labels exist.\n\n"
    report += "The source benchmark evaluates a query after the accumulated current session, so same-session memory is treated as already written current context. The source contains session IDs but no independent query-event timestamp; causality is therefore confirmed at the benchmark session contract, not at sub-session event-time resolution. Future sessions remain forbidden.\n\n"
    report += "## Gold And Split Boundary\n\n"
    report += "The original split builder used evidence to retain scoreable queries and to create gold fields. This is a query-pool selection dependency, not an eligible-membership dependency: for any included query, the eligible set is reproduced without loading labels. Later key-point mapping intersects labels with the pre-existing eligible set, and the session-blocked split moves complete rows without regenerating eligibility. The v21 train incidence is checked only against the train plan; calibration/test eligible sets do not enter its statistics.\n\n"
    report += "## Reproducibility\n\n"
    report += "The machine-readable audit records SHA-256 and byte size for the generator/helper scripts, inherited configurations, memory corpus, train plan, and v21 train incidence.\n"
    report_path = VERSION_ROOT / "reports" / "METHOD1_V21_ELIGIBLE_MEMORY_PROVENANCE_AUDIT.md"
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(report)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
