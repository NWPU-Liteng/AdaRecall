#!/usr/bin/env python3
"""Aggregate the balanced-packing Agent OOF revision."""
from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ns = runpy.run_path(str(ROOT / "scripts/158_aggregate_v21_cap64_agent_answers_oof.py"), run_name="v21_balanced_agent_aggregate")
ns["SHARDS"] = [ROOT / f"artifacts/v21_cap64_agent_answers_balanced_oof_shard{index}_attempt001.jsonl" for index in range(4)]
ns["OUTPUT"] = ROOT / "aggregate/v21_cap64_agent_answers_balanced_oof_summary_r1_attempt001.json"
ns["COMBINED"] = ROOT / "aggregate/v21_cap64_agent_answers_balanced_oof_rows_r1_attempt001.jsonl"
ns["REPORT"] = ROOT / "reports/METHOD1_V21_CAP64_AGENT_ANSWER_BALANCED_OOF_R1_ATTEMPT001.md"
for _name in ("SHARDS", "OUTPUT", "COMBINED", "REPORT"):
    ns["main"].__globals__[_name] = ns[_name]
if __name__ == "__main__":
    ns["main"]()
