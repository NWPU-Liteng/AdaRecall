#!/usr/bin/env python3
"""Aggregate structured evidence-context B32 Agent OOF results."""
from __future__ import annotations
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ns = runpy.run_path(str(ROOT / "scripts/158_aggregate_v21_cap64_agent_answers_oof.py"), run_name="v21_evidence_context_b32_aggregate")
ns["SHARDS"] = [ROOT / f"artifacts/v21_cap64_agent_answers_evidence_context_b32_oof_shard{index}_attempt001.jsonl" for index in range(4)]
ns["OUTPUT"] = ROOT / "aggregate/v21_cap64_agent_answers_evidence_context_b32_oof_summary_r1_attempt001.json"
ns["COMBINED"] = ROOT / "aggregate/v21_cap64_agent_answers_evidence_context_b32_oof_rows_r1_attempt001.jsonl"
ns["REPORT"] = ROOT / "reports/METHOD1_V21_CAP64_AGENT_ANSWER_EVIDENCE_CONTEXT_B32_OOF_R1_ATTEMPT001.md"
ns["EXPECTED_ROWS"] = 1355
ns["EVAL_BUDGETS"] = (32,)
ns["QUERY_COUNT"] = 1355
for _name in ("SHARDS", "OUTPUT", "COMBINED", "REPORT", "EXPECTED_ROWS", "EVAL_BUDGETS", "QUERY_COUNT"):
    ns["main"].__globals__[_name] = ns[_name]
if __name__ == "__main__":
    ns["main"]()
