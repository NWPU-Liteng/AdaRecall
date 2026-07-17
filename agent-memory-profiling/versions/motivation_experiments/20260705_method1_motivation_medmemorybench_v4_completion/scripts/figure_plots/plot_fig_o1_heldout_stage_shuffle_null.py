#!/usr/bin/env python3
from __future__ import annotations

from plot_common import save_figure


FIGURE_NAME = "Fig_O1_v4_heldout_stage_shuffle_null"

# Source aggregate:
# aggregate/B_heldout_stage_shuffle_null.csv
# Explicit histogram data from 1,000 shuffle statistics. The raw permutation
# values remain in the aggregate CSV; this script stores the exact plotted bins.
BIN_EDGES = [
    -0.03378554, -0.03208246, -0.03037937, -0.02867629, -0.0269732,
    -0.02527012, -0.02356703, -0.02186395, -0.02016086, -0.01845778,
    -0.01675469, -0.01505161, -0.01334853, -0.01164544, -0.00994236,
    -0.00823927, -0.00653619, -0.0048331, -0.00313002, -0.00142693,
    0.00027615, 0.00197923, 0.00368232, 0.0053854, 0.00708849,
    0.00879157, 0.01049466, 0.01219774, 0.01390083, 0.01560391,
    0.01730699, 0.01901008, 0.02071316, 0.02241625, 0.02411933,
    0.02582242, 0.0275255, 0.02922859, 0.03093167, 0.03263476,
    0.03433784,
]
BIN_COUNTS = [
    2, 1, 4, 4, 5, 1, 8, 5, 9, 20, 21, 21, 24, 34, 52, 47, 69, 66, 64, 73,
    69, 65, 46, 54, 50, 35, 45, 29, 28, 13, 14, 8, 2, 6, 1, 2, 1, 1, 0, 1,
]
OBSERVED_STATISTIC = 0.1392159927035691
NULL_MEAN = -0.0004063213532043
NULL_CI95_LOW = -0.0213971580674702
NULL_CI95_HIGH = 0.018482099499402
EMPIRICAL_P_VALUE = 0.0009990009990009
NUM_PERMUTATIONS = 1000


def main() -> None:
    import matplotlib.pyplot as plt

    widths = [right - left for left, right in zip(BIN_EDGES[:-1], BIN_EDGES[1:])]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(BIN_EDGES[:-1], BIN_COUNTS, width=widths, align="edge", alpha=0.8)
    ax.axvline(OBSERVED_STATISTIC, color="red", label="observed")
    ax.axvline(NULL_MEAN, color="black", linestyle="--", linewidth=1, label="null mean")
    ax.set_xlabel("Held-out stage advantage under stage shuffle")
    ax.set_ylabel("Count")
    ax.set_title("O1-v4 Held-Out Stage-Shuffle Null")
    ax.legend()
    save_figure(fig, FIGURE_NAME)
    plt.close(fig)
    print(f"wrote {FIGURE_NAME}.png/.pdf; p={EMPIRICAL_P_VALUE:.4g}, permutations={NUM_PERMUTATIONS}")


if __name__ == "__main__":
    main()

