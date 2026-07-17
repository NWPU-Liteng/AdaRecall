#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
SCRIPTS = [
    "plot_fig_m1_history_footprint_traffic.py",
    "plot_fig_m2_frontier_cost_reference_fidelity.py",
    "plot_fig_m3_tile_access_amplification.py",
    "plot_fig_o1_same_cross_stage_locality.py",
    "plot_fig_o1_heldout_stage_transfer.py",
    "plot_fig_o1_heldout_stage_shuffle_null.py",
    "plot_fig_o1_stage_seen_node_coverage.py",
    "plot_fig_m2_budget_cost_strict_keypoint_recall.py",
]


def main() -> None:
    for script in SCRIPTS:
        subprocess.run([sys.executable, str(HERE / script)], check=True)
    print("all figure scripts complete")


if __name__ == "__main__":
    main()

