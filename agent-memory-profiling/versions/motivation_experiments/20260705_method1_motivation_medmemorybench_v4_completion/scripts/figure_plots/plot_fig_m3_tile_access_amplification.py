#!/usr/bin/env python3
from __future__ import annotations

from plot_common import save_figure


FIGURE_NAME = "Fig_M3_v4_tile_access_amplification"

# Source aggregate:
# aggregate/M3_tile_scattering.csv
# Resolved source:
# ../20260704_method1_motivation_medmemorybench_v3/aggregate/M3_tile_scattering.csv
# Values are means over queries for each layout and tile_capacity.
DATA = [
    {"layout": "chronological", "tile_capacity": 8, "access_amplification": 7.206666666666667},
    {"layout": "chronological", "tile_capacity": 16, "access_amplification": 13.48},
    {"layout": "chronological", "tile_capacity": 32, "access_amplification": 24.560000000000002},
    {"layout": "oracle_tile_packing_lower_bound", "tile_capacity": 8, "access_amplification": 1.6},
    {"layout": "oracle_tile_packing_lower_bound", "tile_capacity": 16, "access_amplification": 3.2},
    {"layout": "oracle_tile_packing_lower_bound", "tile_capacity": 32, "access_amplification": 6.4},
    {"layout": "random_per_persona", "tile_capacity": 8, "access_amplification": 7.9},
    {"layout": "random_per_persona", "tile_capacity": 16, "access_amplification": 15.68},
    {"layout": "random_per_persona", "tile_capacity": 32, "access_amplification": 31.040000000000003},
]


def main() -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for layout in ["chronological", "random_per_persona", "oracle_tile_packing_lower_bound"]:
        rows = sorted((row for row in DATA if row["layout"] == layout), key=lambda row: row["tile_capacity"])
        ax.plot(
            [row["tile_capacity"] for row in rows],
            [row["access_amplification"] for row in rows],
            marker="o",
            label=layout,
        )
    ax.set_xlabel("Tile capacity")
    ax.set_ylabel("Access amplification")
    ax.set_title("M3 Tile Access Amplification")
    ax.legend()
    save_figure(fig, FIGURE_NAME)
    plt.close(fig)
    print(f"wrote {FIGURE_NAME}.png/.pdf")


if __name__ == "__main__":
    main()

