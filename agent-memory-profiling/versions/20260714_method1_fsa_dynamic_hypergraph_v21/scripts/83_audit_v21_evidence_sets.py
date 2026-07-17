#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.evidence_sets import incidence_rows, load_query_evidence_sets, summarize  # noqa: E402
from v21.io import load_config, write_csv, write_json, write_jsonl  # noqa: E402


def main() -> None:
    data = load_config("v21_data.json")
    config = load_config("v21_uot_semantic.json")
    records, query_group_counts = load_query_evidence_sets(
        data["split"]["train_query_plan"], folds=int(config["adapter"]["folds"])
    )
    rows = incidence_rows(records, folds=int(config["adapter"]["folds"]))
    summary = summarize(records, query_group_counts)
    write_jsonl("artifacts/v21_evidence_sets/train_query_evidence_incidence.jsonl", rows)
    write_csv("aggregate/v21_evidence_set_incidence.csv", rows)
    result = {
        "schema_version": "v21-evidence-set-audit-1",
        "status": "completed_train_only_causal_availability_audit",
        "summary": summary,
        "group_semantics": "alternative acceptable support for one labeled evidence aspect; not an observed access event",
        "inference_policy": "evidence groups are supervision only and are unavailable at inference",
        "calibration_read": False,
        "formal_test_read": False,
    }
    write_json("aggregate/v21_evidence_set_audit.json", result)
    report = "# Method 1 v21 Evidence-Set Audit\n\n"
    report += "The train-only causal availability audit completed. Every accepted evidence member is in the corresponding query's `eligible_memory_ids`; this validates the training incidence as a past-available evidence relation. It does not create a temporal read order.\n\n"
    report += "| Measure | Value |\n| --- | ---: |\n"
    for key in ("queries", "evidence_groups", "evidence_member_incidences", "queries_with_two_or_more_groups"):
        report += f"| {key} | {summary[key]} |\n"
    report += f"| members_per_group | `{summary['members_per_group']}` |\n"
    report += f"| groups_per_query | `{summary['groups_per_query']}` |\n"
    report += f"| source_session_lags | `{summary['source_session_lags']}` |\n"
    report += "\nSame-session items are not treated as leakage: the supplied eligibility list is the causal authority and can include earlier turns in the same session. Calibration and formal test were not read.\n"
    (VERSION_ROOT / "reports/METHOD1_V21_EVIDENCE_SET_AUDIT.md").write_text(report, encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
