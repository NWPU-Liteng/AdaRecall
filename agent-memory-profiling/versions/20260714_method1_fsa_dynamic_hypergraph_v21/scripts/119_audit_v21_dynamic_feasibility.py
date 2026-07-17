#!/usr/bin/env python3
"""Audit whether the frozen corpus supports a causal dynamic update claim."""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.io import iter_jsonl, load_config, resolve, sha256_file, write_json_new  # noqa: E402


RUN_ID = "v21_dynamic_feasibility_audit_r1_20260714"
TIMESTAMP_PATTERN = re.compile(r"^session=(?P<session>\d+);turn=(?P<turn>\d+)$")
REQUIRED_DYNAMIC_EVENT_TYPES = {"agent_tool_read", "agent_consumed"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit v21 dynamic-data feasibility after Gate C.")
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--gate-c", default="aggregate/v21_gate_c_decision_r1_20260714.json")
    args = parser.parse_args()
    if args.run_id != RUN_ID:
        raise ValueError("dynamic audit run ID is immutable")
    gate_path = VERSION_ROOT / args.gate_c
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("decision") != "PASS" or gate.get("calibration_read") is not False or gate.get("formal_test_read") is not False:
        raise ValueError("dynamic audit requires a completed Gate C PASS with locked splits")
    data = load_config("v21_data.json")
    memory_path = resolve(data["memory"]["items"])
    rows = list(iter_jsonl(memory_path))
    if len(rows) != 42034:
        raise ValueError(f"unexpected memory count: {len(rows)}")
    timestamp_counts = collections.Counter(str(row.get("timestamp", "")) for row in rows)
    event_ids = [row.get("event_id") for row in rows]
    event_types = collections.Counter()
    for row in rows:
        info = row.get("event_info") or {}
        event_type = info.get("access_type") or info.get("event_type") or info.get("type")
        if event_type:
            event_types[str(event_type)] += 1
    ranks = [row.get("chronological_rank") for row in rows]
    rank_unique = len(set(ranks)) == len(ranks) and all(isinstance(value, int) for value in ranks)
    timestamp_matches = [TIMESTAMP_PATTERN.match(str(row.get("timestamp", ""))) for row in rows]
    timestamp_parseable = all(match is not None for match in timestamp_matches)
    timestamp_tie_count = sum(count > 1 for count in timestamp_counts.values())
    timestamp_max_tie = max(timestamp_counts.values()) if timestamp_counts else 0
    explicit_read_count = sum(count for name, count in event_types.items() if name in REQUIRED_DYNAMIC_EVENT_TYPES)
    event_id_nonnull = sum(value is not None for value in event_ids)
    event_info_dates = sum(bool((row.get("event_info") or {}).get("date")) for row in rows)
    source_record_ids_complete = all(isinstance(row.get("source_record_ids"), list) and row["source_record_ids"] for row in rows)

    causal_order_recoverable = bool(
        explicit_read_count > 0
        and event_id_nonnull == len(rows)
        and timestamp_parseable
        and timestamp_tie_count == 0
        and rank_unique
    )
    only_session_level_time = bool(timestamp_parseable and event_info_dates == 0 and timestamp_tie_count > 0)
    decision = "dynamic_update_not_eligible_static_fsa_qehg_fallback"
    audit = {
        "schema_version": "v21-dynamic-feasibility-audit-1", "status": "completed", "run_id": RUN_ID,
        "gate_c_input": str(gate_path.relative_to(VERSION_ROOT)), "gate_c_decision": gate.get("decision"),
        "memory_items": len(rows), "timestamp": {
            "parseable": timestamp_parseable, "distinct_values": len(timestamp_counts), "tied_timestamp_values": timestamp_tie_count, "maximum_tie_size": timestamp_max_tie, "only_session_level_time": only_session_level_time,
        },
        "causal_fields": {
            "chronological_rank_present_and_unique": rank_unique, "event_id_nonnull": event_id_nonnull, "event_info_date_present": event_info_dates,
            "explicit_read_event_count": explicit_read_count, "event_types": dict(sorted(event_types.items())), "source_record_ids_complete": source_record_ids_complete,
            "causal_insertion_order_recoverable": causal_order_recoverable,
        },
        "decision": decision,
        "dynamic_update_executed": False,
        "dynamic_comparators": ["no_update", "append_only", "local_online_uot", "periodic_full_rebuild"],
        "growth_points_percent": [1, 5, 10, 20],
        "allowed_static_fallback": ["incremental_content_insertion", "frozen_projection", "online_uot_assignment", "prototype_sufficient_statistics", "sparse_semantic_incidence_overlay", "bundle_locality_overlay", "atomic_publication", "periodic_compaction"],
        "prohibited_claims": ["access_order", "co_access_trace", "agent_tool_read_demand", "PIM_hardware_update", "PIM_latency"],
        "hashes": {"memory_items": sha256_file(memory_path), "data_config": sha256_file(VERSION_ROOT / "configs" / "v21_data.json"), "gate_c": sha256_file(gate_path)},
        "calibration_read": False, "formal_test_read": False,
    }
    output_path = VERSION_ROOT / "aggregate" / "v21_dynamic_feasibility_audit_r1_20260714.json"
    write_json_new(output_path, audit)
    report_path = VERSION_ROOT / "reports" / "METHOD1_V21_DYNAMIC_FEASIBILITY_AUDIT.md"
    report = "# Method 1 v21 Dynamic Feasibility Audit\n\n"
    report += "**Decision: static FSA-QEHG fallback.** The corpus has frozen content ordering metadata, but it does not expose a causal agent memory-read stream.\n\n"
    report += f"- memory items audited: {len(rows)}\n- timestamp format parseable: `{timestamp_parseable}`; tied timestamp values: `{timestamp_tie_count}`; maximum tie size: `{timestamp_max_tie}`\n- chronological rank unique: `{rank_unique}` (ordering metadata only)\n- non-null event IDs: `{event_id_nonnull}/{len(rows)}`\n- explicit `agent_tool_read`/`agent_consumed` events: `{explicit_read_count}`\n- causal insertion order recoverable: `{causal_order_recoverable}`\n\n"
    report += "A true dynamic access-order or co-access experiment is therefore not eligible. The permitted record is a static fallback with content append stress only: frozen projection, online UOT assignment, prototype sufficient statistics, sparse incidence/bundle overlays, atomic publication, and periodic compaction. No access order, co-access trace, or PIM update/latency result is inferred. Calibration and formal-test access remain zero.\n"
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(report)
    print(json.dumps({"status": audit["status"], "decision": decision, "causal_insertion_order_recoverable": causal_order_recoverable, "explicit_read_event_count": explicit_read_count}, sort_keys=True))


if __name__ == "__main__":
    main()
