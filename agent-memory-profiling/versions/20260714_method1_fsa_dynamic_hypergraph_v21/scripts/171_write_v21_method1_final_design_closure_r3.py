#!/usr/bin/env python3
"""Add deterministic static new-memory partition publication to final closure."""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from v21.io import read_json, sha256_file, write_json_new  # noqa: E402

PREVIOUS = ROOT / "aggregate/v21_method1_final_design_closure_r2_attempt001.json"
DYNAMIC = ROOT / "aggregate/v21_persona_sharded_partition_update_stress_r1_attempt001.json"
OUT = ROOT / "aggregate/v21_method1_final_design_closure_r3_attempt001.json"
REPORT = ROOT / "reports/METHOD1_V21_FINAL_DESIGN_CLOSURE_R3_20260715.md"

def main() -> None:
    previous = read_json(PREVIOUS)
    dynamic = read_json(DYNAMIC)
    if previous.get("calibration_read") is not False or previous.get("formal_test_read") is not False:
        raise ValueError("previous closure touched locked split")
    if dynamic.get("quality_contract_satisfied") is not True or dynamic.get("calibration_read") is not False or dynamic.get("formal_test_read") is not False:
        raise ValueError("dynamic partition contract is incomplete")
    if dynamic.get("causal_stream_claim") is not False or dynamic.get("access_order_claim") is not False or dynamic.get("co_access_claim") is not False:
        raise ValueError("dynamic partition made a forbidden causal/access claim")
    manifest = {
        **previous,
        "schema_version": "v21-method1-final-design-closure-r3",
        "revision_id": "v21_method1_final_design_closure_r3_20260715",
        "status": "static_retrieval_tile_e2e_partition_incremental_publication_and_agent_oof_closed",
        "static_partition_incremental_publication": {
            "status": dynamic["status"],
            "partition_protocol": dynamic["partition_protocol"],
            "base_memory_count": dynamic["base_memory_count"],
            "new_memory_pool_count": dynamic["new_memory_pool_count"],
            "probe_count": dynamic["probe_count"],
            "quality_contract_satisfied": dynamic["quality_contract_satisfied"],
            "growth_results": dynamic["results"],
            "causal_stream_claim": False,
            "access_order_claim": False,
            "serving_adapter": "PersonaShardedSnapshotDispatch backed by AtomicPersonaShardedSnapshotPublisher",
        },
        "remaining_optimization": [
            "improve QESA/candidate discovery without active32 latency; active24 remains optional quality tier",
            "improve Agent evidence coverage/answer generation; packing and lexical handoff were negative",
            "replace full snapshot rebuild with persona-local rank cache and deferred compaction",
            "obtain an independently authorized frozen heldout protocol before reporting formal generalization",
        ],
        "inputs": {"previous_closure": str(PREVIOUS.relative_to(ROOT)), "partition_update": str(DYNAMIC.relative_to(ROOT))},
        "hashes": {"previous_closure": sha256_file(PREVIOUS), "partition_update": sha256_file(DYNAMIC)},
        "minimum_effect_threshold": None,
        "calibration_read": False,
        "formal_test_read": False,
    }
    write_json_new(OUT, manifest)
    last = dynamic["results"][-1]
    report = f'''# Method 1 v21 Final Design Closure r3

This revision closes the deterministic static new-memory publication path in addition to the retrieval, Tile, matched e2e, and Agent train-OOF results in r2.

## Dynamic-partition design

Memory IDs are assigned to a frozen base or a new-memory pool by deterministic SHA-256 partitioning. No timestamp, access order, co-access trace, or causal update interpretation is used. The base contains `{dynamic['base_memory_count']}` memories and the new-memory pool contains `{dynamic['new_memory_pool_count']}`.

At 1%, 5%, 10%, and 20% growth, the persona-sharded serving adapter, overlay lookup, compacted lookup, and B=8 selection all matched exactly for `{dynamic['probe_count']}` probes per point. At 20% growth, overlay lookup averaged `{last['overlay_lookup_ms_mean']:.3f} ms`; atomic publish was `{last['atomic_publish_ms']:.3f} ms`; full snapshot build was `{last['snapshot_build_ms']:.1f} ms`; compaction was `{last['compaction_ms']:.1f} ms`. All mismatch counters were zero.

This closes the **static incremental publication** requirement. It does not establish causal dynamic-memory behavior, which the source data cannot support.

## Overall state

The default retrieval tier remains `cap=64`, active16, B8. Active24 is an optional quality tier; active32 remains diagnostic because of its latency. Persona sharding, corrected host/logical Tile accounting, matched e2e controls, static partition publication, and Agent train-OOF execution are complete. Agent answer accuracy remains `0.259041`; balanced packing and lexical handoff did not improve it. Formal heldout performance remains unavailable.

The remaining work is optimization and independent generalization validation, not a missing static publication contract. No calibration/formal split was read and no minimum-effect threshold was used.
'''
    REPORT.write_text(report, encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "manifest": str(OUT), "report": str(REPORT)}, sort_keys=True))

if __name__ == "__main__":
    main()
