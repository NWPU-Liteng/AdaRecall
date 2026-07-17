#!/usr/bin/env python3
"""Aggregate active24 matched global/sharded e2e controls."""
from __future__ import annotations
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ns = runpy.run_path(str(ROOT / "scripts/146_aggregate_v21_sparse_joint_e2e_controls.py"), run_name="v21_active24_e2e_aggregate")
main = ns["main"]
g = main.__globals__
g["GLOBAL_RUN"] = "v21_sparse_joint_cap64_active24_global_e2e_oof_r1_20260715"
g["SHARDED_RUN"] = "v21_sparse_joint_cap64_active24_sharded_e2e_oof_r1_20260715"
g["ACTIVE_PROTOTYPES"] = 24
g["MEMBERS_PER_EDGE"] = 32
g["BUNDLE_BEAM"] = 32
old_tile = "aggregate/v21_persona_sharded_hyper_index_summary_r2_attempt001.json"
new_tile = ROOT / "aggregate/v21_persona_sharded_hyper_index_summary_cap64_b8_r1_attempt001.json"
source_read = g["read_json"]
g["read_json"] = lambda path: source_read(new_tile) if str(path).endswith(old_tile) else source_read(path)
if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--attempt", "attempt_001", "--output-revision", "cap64_active24_r1_attempt001"]
    main()
