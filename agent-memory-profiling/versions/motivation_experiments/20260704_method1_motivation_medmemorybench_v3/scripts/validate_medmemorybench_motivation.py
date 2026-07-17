#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from medmmb_common import environment_manifest, read_jsonl, session_le, write_manifest


EXPECTED = [
    "reports/00_pipeline_audit.md",
    "reports/00_pipeline_delta.md",
    "aggregate/medmemorybench_schema_audit.csv",
    "aggregate/memory_items_turn_pair.jsonl",
    "aggregate/query_evidence_mapping.jsonl",
    "aggregate/query_evidence_mapping_coverage.csv",
    "aggregate/M1_history_scale_invocations.jsonl",
    "aggregate/M1_history_scale_summary.csv",
    "aggregate/M2_frontier_budget_invocations.jsonl",
    "aggregate/M2_frontier_occupancy.csv",
    "aggregate/M2_frontier_budget_summary.csv",
    "aggregate/M3_tile_scattering.csv",
    "aggregate/O1_formal_stage_trace.jsonl",
    "aggregate/O1_same_cross_stage_stats.csv",
    "aggregate/O1_stage_shuffle_null.csv",
    "aggregate/O1_pair_recurrence.csv",
]


def _load_schema(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_required(row: dict, schema: dict, label: str) -> list[str]:
    errors = []
    for key in schema.get("required", []):
        if key not in row:
            errors.append(f"{label}: missing required key {key}")
    for key, prop in schema.get("properties", {}).items():
        if "const" in prop and row.get(key) != prop["const"]:
            errors.append(f"{label}: {key} expected {prop['const']!r}, got {row.get(key)!r}")
        if "enum" in prop and key in row and row.get(key) not in prop["enum"]:
            errors.append(f"{label}: {key} not in enum: {row.get(key)!r}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate MedMemoryBench v3 motivation package outputs.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--allow-missing-formal", action="store_true", help="Allow missing formal outputs while data acquisition is blocked.")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    errors: list[str] = []

    missing = [rel for rel in EXPECTED if not (root / rel).exists()]
    if missing and not args.allow_missing_formal:
        errors.extend([f"missing expected file: {rel}" for rel in missing])

    write_manifest(root / "aggregate" / "M0_pipeline_inventory.json", environment_manifest({"validation_root": str(root)}))

    schema_dir = root / "schemas"
    memory_schema = _load_schema(schema_dir / "memory_item.schema.json")
    retrieval_schema = _load_schema(schema_dir / "retrieval_invocation.schema.json")
    stage_schema = _load_schema(schema_dir / "stage_trace.schema.json")

    memory_path = root / "aggregate" / "memory_items_turn_pair.jsonl"
    memory_by_id = {}
    if memory_path.exists():
        for idx, row in enumerate(read_jsonl(memory_path)):
            errors.extend(_validate_required(row, memory_schema, f"memory[{idx}]"))
            memory_by_id[str(row.get("memory_id"))] = row

    for rel in ["aggregate/M1_history_scale_invocations.jsonl", "aggregate/M2_frontier_budget_invocations.jsonl"]:
        path = root / rel
        if not path.exists():
            continue
        for idx, row in enumerate(read_jsonl(path)):
            errors.extend(_validate_required(row, retrieval_schema, f"{rel}[{idx}]"))
            for mid in row.get("final_selected_ids", []):
                mem = memory_by_id.get(str(mid))
                if mem:
                    if str(mem.get("persona_id")) != str(row.get("persona_id")):
                        errors.append(f"{rel}[{idx}]: cross-persona selected memory {mid}")
                    if not session_le(mem.get("session_id"), row.get("query_session_id")):
                        errors.append(f"{rel}[{idx}]: future/current-session selected memory {mid}")

    stage_path = root / "aggregate" / "O1_formal_stage_trace.jsonl"
    if stage_path.exists():
        stage_rows = read_jsonl(stage_path)
        counts = Counter()
        for idx, row in enumerate(stage_rows):
            errors.extend(_validate_required(row, stage_schema, f"O1[{idx}]"))
            counts[str(row.get("query_id"))] += 1
            if row.get("stage_label_source") != "controller_fsm_v1":
                errors.append(f"O1[{idx}]: invalid stage_label_source")
            for mid in row.get("final_selected_ids", []):
                mem = memory_by_id.get(str(mid))
                if mem:
                    if str(mem.get("persona_id")) != str(row.get("persona_id")):
                        errors.append(f"O1[{idx}]: cross-persona selected memory {mid}")
                    if not session_le(mem.get("session_id"), row.get("query_session_id")):
                        errors.append(f"O1[{idx}]: future/current-session selected memory {mid}")
        bad_counts = {qid: count for qid, count in counts.items() if count != 4}
        if bad_counts:
            errors.append(f"O1 requires exactly four stages per query; bad counts: {list(bad_counts.items())[:5]}")

    null_path = root / "aggregate" / "O1_stage_shuffle_null.csv"
    if null_path.exists():
        import pandas as pd

        df = pd.read_csv(null_path)
        if len(df) < 1000 and not args.allow_missing_formal:
            errors.append(f"O1 stage shuffle has fewer than 1000 permutations: {len(df)}")

    for fig in (root / "figures").glob("*.png"):
        text = fig.name.lower()
        for forbidden in ["pim_speedup", "hypergraph_improvement", "benchmark-provided_phase"]:
            if forbidden in text:
                errors.append(f"forbidden claim in figure filename: {fig}")

    report = root / "reports" / "VALIDATION.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    if errors:
        report.write_text("# Validation\n\nFAILED\n\n" + "\n".join(f"- {err}" for err in errors) + "\n", encoding="utf-8")
        raise SystemExit("\n".join(errors[:20]))
    report.write_text("# Validation\n\nPASSED\n", encoding="utf-8")
    print(f"validation={report}")


if __name__ == "__main__":
    main()
