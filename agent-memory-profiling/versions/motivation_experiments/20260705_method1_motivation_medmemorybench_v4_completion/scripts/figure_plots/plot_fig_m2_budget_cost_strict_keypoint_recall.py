#!/usr/bin/env python3
from __future__ import annotations

from plot_common import save_figure


FIGURE_NAME = "Fig_M2_v4_budget_cost_strict_keypoint_recall"

# Source aggregate:
# aggregate/C_strict_evidence_budget_summary.csv
DATA = [
    {"frontier_budget": 8, "selection_ms": 94.14511450086138, "strict_keypoint_recall_at_5": 0.75},
    {"frontier_budget": 16, "selection_ms": 93.85317062515242, "strict_keypoint_recall_at_5": 0.625},
    {"frontier_budget": 32, "selection_ms": 147.13300449329836, "strict_keypoint_recall_at_5": 0.75},
    {"frontier_budget": 64, "selection_ms": 229.7859869995591, "strict_keypoint_recall_at_5": 0.75},
]


def main() -> None:
    import matplotlib.pyplot as plt

    rows = sorted(DATA, key=lambda row: row["frontier_budget"])
    budgets = [row["frontier_budget"] for row in rows]
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(budgets, [row["selection_ms"] for row in rows], marker="o", color="tab:blue", label="selection ms")
    ax1.set_xlabel("Frontier budget")
    ax1.set_ylabel("Selection latency (ms)")
    ax2 = ax1.twinx()
    ax2.plot(budgets, [row["strict_keypoint_recall_at_5"] for row in rows], marker="s", color="tab:green", label="strict recall@5")
    ax2.set_ylabel("StrictKeypointRecall@5")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines])
    ax1.set_title("M2-v4 Cost vs Strict Keypoint Recall")
    save_figure(fig, FIGURE_NAME)
    plt.close(fig)
    print(f"wrote {FIGURE_NAME}.png/.pdf")


if __name__ == "__main__":
    main()

