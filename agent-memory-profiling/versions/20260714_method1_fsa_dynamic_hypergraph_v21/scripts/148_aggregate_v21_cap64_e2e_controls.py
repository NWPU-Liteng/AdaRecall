#!/usr/bin/env python3
"""Aggregate cap64 matched controls without mutating the cap96 aggregate."""
from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ns = runpy.run_path(str(ROOT / "scripts/146_aggregate_v21_sparse_joint_e2e_controls.py"), run_name="v21_cap64_aggregate")
ns["GLOBAL_RUN"] = "v21_sparse_joint_cap64_global_e2e_control_oof_r1_20260715"
ns["SHARDED_RUN"] = "v21_sparse_joint_cap64_sharded_e2e_oof_r1_20260715"

original_main = ns["main"]

def main() -> None:
    import sys
    sys.argv = [sys.argv[0], "--attempt", "attempt_001", "--output-revision", "cap64_r1_attempt001"]
    original_main.__globals__["VERSION_ROOT"] = ROOT
    original_main.__globals__["Path"] = Path
    # The historical aggregator uses the old Tile summary; redirect its input for this revision.
    original_main.__globals__["read_json_original"] = original_main.__globals__["read_json"]
    original_main.__globals__["read_json"] = ns["read_json"]
    source = original_main.__globals__["read_json"]
    old_tile = ROOT / "aggregate/v21_persona_sharded_hyper_index_summary_r2_attempt001.json"
    new_tile = ROOT / "aggregate/v21_persona_sharded_hyper_index_summary_cap64_b8_r1_attempt001.json"
    original_main.__globals__["read_json"] = lambda path: source(new_tile) if str(path).endswith(str(old_tile.relative_to(ROOT))) else source(path)
    original_main()

if __name__ == "__main__":
    main()
