#!/usr/bin/env python3
from __future__ import annotations

from plot_common import save_figure


FIGURE_NAME = "Fig_O1_v4_stage_seen_node_coverage"

# Source aggregate:
# aggregate/B_heldout_stage_transfer_summary.csv
DATA = [
    {"stage": "ground", "seen_node_coverage_mean": 0.4490566037735849},
    {"stage": "plan", "seen_node_coverage_mean": 0.5150943396226415},
    {"stage": "reason", "seen_node_coverage_mean": 0.4245283018867924},
    {"stage": "reflect", "seen_node_coverage_mean": 0.4754716981132075},
    {"stage": "aggregate", "seen_node_coverage_mean": 0.4660377358490566},
]
GUARDRAIL = 0.30


def main() -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar([row["stage"] for row in DATA], [row["seen_node_coverage_mean"] for row in DATA])
    ax.axhline(GUARDRAIL, color="red", linestyle="--", label="30% guardrail")
    ax.set_ylabel("Seen-node coverage")
    ax.set_title("O1-v4 Held-Out Seen-Node Coverage")
    ax.legend()
    save_figure(fig, FIGURE_NAME)
    plt.close(fig)
    print(f"wrote {FIGURE_NAME}.png/.pdf")


if __name__ == "__main__":
    main()

