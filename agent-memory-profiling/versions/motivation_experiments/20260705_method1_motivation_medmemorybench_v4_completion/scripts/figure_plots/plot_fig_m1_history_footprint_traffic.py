#!/usr/bin/env python3
from __future__ import annotations

from plot_common import save_figure


FIGURE_NAME = "Fig_M1_v4_history_footprint_traffic"

# Source aggregate:
# aggregate/M1_history_scale_summary.csv
# Resolved source:
# ../20260704_method1_motivation_medmemorybench_v3/aggregate/M1_history_scale_summary.csv
DATA = [
    {"history_condition": "medical_only", "history_scale": "128", "host_memory_bytes_mean": 538338.4207920792},
    {"history_condition": "medical_only", "history_scale": "256", "host_memory_bytes_mean": 936673.5776397516},
    {"history_condition": "medical_only", "history_scale": "512", "host_memory_bytes_mean": 1725646.9659090908},
    {"history_condition": "medical_only", "history_scale": "full", "host_memory_bytes_mean": 1435992.0166666666},
    {"history_condition": "medical_plus_official_noise", "history_scale": "128", "host_memory_bytes_mean": 525245.1777777778},
    {"history_condition": "medical_plus_official_noise", "history_scale": "256", "host_memory_bytes_mean": 921699.2264150945},
    {"history_condition": "medical_plus_official_noise", "history_scale": "512", "host_memory_bytes_mean": 1710562.391304348},
    {"history_condition": "medical_plus_official_noise", "history_scale": "1024", "host_memory_bytes_mean": 3283591.595588235},
    {"history_condition": "medical_plus_official_noise", "history_scale": "full", "host_memory_bytes_mean": 3555575.466666667},
]


def sort_key(scale: str) -> int:
    return 999999 if scale == "full" else int(scale)


def main() -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for condition in sorted({row["history_condition"] for row in DATA}):
        rows = sorted((row for row in DATA if row["history_condition"] == condition), key=lambda row: sort_key(row["history_scale"]))
        ax.plot(
            [row["history_scale"] for row in rows],
            [row["host_memory_bytes_mean"] for row in rows],
            marker="o",
            label=condition,
        )
    ax.set_xlabel("History scale")
    ax.set_ylabel("Host/index/payload bytes")
    ax.set_title("M1 History Footprint and Traffic")
    ax.legend()
    save_figure(fig, FIGURE_NAME)
    plt.close(fig)
    print(f"wrote {FIGURE_NAME}.png/.pdf")


if __name__ == "__main__":
    main()

