#!/usr/bin/env python3
"""Publication-grade figures for the method-independent motivation package."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-v5")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]

NAVY = "#2F5D8A"
TEAL = "#168C83"
ORANGE = "#D96C4B"
GOLD = "#D4A72C"
PURPLE = "#7B6DAB"
GRAY = "#69737D"
LIGHT_GRID = "#DCE1E5"
QUERY_ORDER = [
    "entity_exact_match",
    "inference_generation",
    "multi_hop_clinical_deduction",
    "multiple_choice",
    "state_update",
    "temporal_localization",
]
QUERY_LABEL = {
    "entity_exact_match": "Exact match",
    "inference_generation": "Inference",
    "multi_hop_clinical_deduction": "Multi-hop",
    "multiple_choice": "Multiple choice",
    "state_update": "State update",
    "temporal_localization": "Temporal",
}


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.3,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.7,
        "lines.markersize": 5.0,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def clean_axis(ax: plt.Axes, grid: str | None = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=LIGHT_GRID, linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.13, 1.06, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")


def save_figure(fig: plt.Figure, stem: str) -> None:
    figure_dir = ROOT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / f"{stem}.png", dpi=300, facecolor="white")
    fig.savefig(figure_dir / f"{stem}.pdf", facecolor="white")
    plt.close(fig)


def annotated_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    fmt: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    show_y: bool = True,
) -> None:
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(col_labels)), col_labels)
    ax.set_yticks(range(len(row_labels)), row_labels if show_y else [""] * len(row_labels))
    ax.set_title(title, loc="left", fontweight="bold")
    threshold = (float(np.nanmin(matrix)) + float(np.nanmax(matrix))) / 2.0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = float(matrix[row, col])
            ax.text(col, row, format(value, fmt), ha="center", va="center", fontsize=7.2,
                    color="white" if value > threshold else "#1F2933")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    plt.colorbar(image, ax=ax, fraction=0.035, pad=0.025)


def plot_m1(m1: list[dict]) -> None:
    conditions = ["medical_only", "medical_plus_official_noise"]
    labels = {"medical_only": "Medical history", "medical_plus_official_noise": "+ official noise"}
    colors = {"medical_only": NAVY, "medical_plus_official_noise": ORANGE}
    markers = {"medical_only": "o", "medical_plus_official_noise": "s"}
    scales = ["128", "256", "512", "1024", "full"]
    x = np.arange(len(scales))
    fig = plt.figure(figsize=(7.4, 4.15), layout="constrained")
    grid = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 0.85])
    axes = [fig.add_subplot(grid[0, idx]) for idx in range(3)]

    for condition in conditions:
        rows = {row["history_scale"]: row for row in m1 if row["mode"] == "formal" and row["history_condition"] == condition}
        present = [(idx, rows[scale]) for idx, scale in enumerate(scales) if scale in rows]
        xs = [item[0] for item in present]
        latency = [float(item[1]["t_total_selection_ms_mean"]) for item in present]
        latency_low = [value - float(item[1]["t_total_selection_ms_ci95_low"]) for value, item in zip(latency, present, strict=True)]
        latency_high = [float(item[1]["t_total_selection_ms_ci95_high"]) - value for value, item in zip(latency, present, strict=True)]
        axes[0].errorbar(xs, latency, yerr=[latency_low, latency_high], color=colors[condition],
                         marker=markers[condition], capsize=2.5, label=labels[condition])
        traffic = [float(item[1]["host_memory_bytes_mean"]) / 1e6 for item in present]
        traffic_low = [(float(item[1]["host_memory_bytes_mean"]) - float(item[1]["host_memory_bytes_ci95_low"])) / 1e6 for item in present]
        traffic_high = [(float(item[1]["host_memory_bytes_ci95_high"]) - float(item[1]["host_memory_bytes_mean"])) / 1e6 for item in present]
        axes[1].errorbar(xs, traffic, yerr=[traffic_low, traffic_high], color=colors[condition],
                         marker=markers[condition], capsize=2.5)

    axes[0].set_title("a  Selection cost", loc="left", fontweight="bold")
    axes[0].set_ylabel("Latency (ms)")
    axes[1].set_title("b  Host traffic", loc="left", fontweight="bold")
    axes[1].set_ylabel("Host bytes touched (MB)")
    for ax in axes[:2]:
        ax.set_xticks(x, scales)
        ax.set_xlabel("History scale")
        clean_axis(ax)
    axes[0].legend(frameon=False, loc="lower right")

    full_rows = [next(row for row in m1 if row["mode"] == "formal" and row["history_condition"] == condition and row["history_scale"] == "full") for condition in conditions]
    components = {
        "ANN": [float(row["t_ann_ms_mean"]) for row in full_rows],
        "Rerank": [float(row["t_rerank_ms_mean"]) for row in full_rows],
        "Other": [max(0.0, float(row["t_total_selection_ms_mean"]) - float(row["t_ann_ms_mean"]) - float(row["t_rerank_ms_mean"])) for row in full_rows],
    }
    bottoms = np.zeros(2)
    for name, color in zip(("ANN", "Rerank", "Other"), (TEAL, NAVY, "#B8C0C8"), strict=True):
        values = np.asarray(components[name])
        axes[2].bar([0, 1], values, bottom=bottoms, color=color, width=0.62, label=name)
        bottoms += values
    axes[2].set_xticks([0, 1], ["Medical", "+ noise"])
    axes[2].set_ylabel("Latency (ms)")
    axes[2].set_title("c  Full-history breakdown", loc="left", fontweight="bold")
    clean_axis(axes[2])
    axes[2].legend(frameon=False, loc="upper left")
    fig.suptitle("Long histories increase memory traffic; reranking dominates selection cost", fontsize=12, fontweight="bold")
    save_figure(fig, "Fig_V5_M1_history_pressure")


def plot_m2(m2: list[dict]) -> None:
    budgets = [8, 16, 32, 64]
    budget_colors = {8: "#B9D6E8", 16: "#68A4C4", 32: NAVY, 64: "#173B57"}
    fig, axes = plt.subplots(2, 3, figsize=(7.5, 5.55), sharex=True, sharey=True, layout="constrained")
    for idx, query_type in enumerate(QUERY_ORDER):
        ax = axes.flat[idx]
        rows = sorted((row for row in m2 if row["query_type"] == query_type), key=lambda row: int(row["frontier_budget"]))
        xs = np.asarray([float(row["t_total_selection_ms_mean"]) for row in rows])
        ys = np.asarray([float(row["reranker_oracle_overlap_at_5_mean"]) for row in rows])
        ax.plot(xs, ys, color=GRAY, linewidth=1.2, zorder=1)
        for row, x_value, y_value in zip(rows, xs, ys, strict=True):
            budget = int(row["frontier_budget"])
            xerr = [[x_value - float(row["t_total_selection_ms_ci95_low"])], [float(row["t_total_selection_ms_ci95_high"]) - x_value]]
            yerr = [[y_value - float(row["reranker_oracle_overlap_at_5_ci95_low"])], [float(row["reranker_oracle_overlap_at_5_ci95_high"]) - y_value]]
            ax.errorbar(x_value, y_value, xerr=xerr, yerr=yerr, fmt="o", color=budget_colors[budget],
                        ecolor=budget_colors[budget], capsize=2, markersize=5.5, zorder=2)
            ax.annotate(str(budget), (x_value, y_value), xytext=(3, -8), textcoords="offset points", fontsize=6.7)
        ax.set_title(QUERY_LABEL[query_type], loc="left", fontweight="bold")
        ax.set_xlim(42, 245)
        ax.set_ylim(0.42, 1.035)
        clean_axis(ax, grid="both")
    fig.supxlabel("Selection latency (ms)")
    fig.supylabel("Agreement with B=64 reranker reference")
    fig.suptitle("Flat frontier expansion buys reference fidelity at rapidly rising cost", fontsize=12, fontweight="bold")
    save_figure(fig, "Fig_V5_M2_flat_frontier_tradeoff")


def plot_m3(m3: list[dict]) -> None:
    capacities = [8, 16, 32]
    layouts = ["chronological", "random_per_persona", "oracle_tile_packing_lower_bound"]
    layout_labels = ["Chronological", "Random", "Oracle lower bound"]

    def matrix(field: str) -> np.ndarray:
        output = []
        for layout in layouts:
            output.append([
                float(np.mean([float(row[field]) for row in m3 if row["layout"] == layout and int(row["tile_capacity"]) == capacity]))
                for capacity in capacities
            ])
        return np.asarray(output)

    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.8), layout="constrained")
    annotated_heatmap(axes[0, 0], matrix("activated_tiles"), layout_labels, list(map(str, capacities)),
                      "a  Activated Tiles", ".2f", "Blues", vmin=1, vmax=5)
    annotated_heatmap(axes[0, 1], matrix("access_amplification"), layout_labels, list(map(str, capacities)),
                      "b  Access amplification (x)", ".1f", "Oranges", vmin=1, vmax=32, show_y=False)
    annotated_heatmap(axes[1, 0], matrix("tile_utilization") * 100.0, layout_labels, list(map(str, capacities)),
                      "c  Useful payload per activated Tile (%)", ".1f", "Greens", vmin=0, vmax=65)

    ax = axes[1, 1]
    for layout, label, color in zip(layouts, layout_labels, (NAVY, ORANGE, TEAL), strict=True):
        values = sorted(float(row["scatter_ratio"]) for row in m3 if row["layout"] == layout and int(row["tile_capacity"]) == 16)
        y = np.arange(1, len(values) + 1) / float(len(values))
        ax.step(values, y, where="post", label=label, color=color)
    ax.set_xlabel("Scatter ratio at capacity 16")
    ax.set_ylabel("Cumulative queries")
    ax.set_title("d  Query-level scattering distribution", loc="left", fontweight="bold")
    clean_axis(ax, grid="both")
    ax.legend(frameon=False, loc="lower right")
    axes[1, 0].set_xlabel("Tile capacity")
    fig.suptitle("Naive placement activates multiple Tiles and amplifies payload access", fontsize=12, fontweight="bold")
    save_figure(fig, "Fig_V5_M3_tile_activation")


def plot_s1(summary: list[dict], per_query: list[dict]) -> None:
    budgets = [4, 8, 16, 32]
    summary_by_type = {row["query_type"]: row for row in summary}
    type_labels = [QUERY_LABEL[query_type] for query_type in QUERY_ORDER]
    all_group = np.asarray([
        [float(summary_by_type[query_type][f"flat_all_groups_b{budget}_rate"]) * 100.0 for budget in budgets]
        for query_type in QUERY_ORDER
    ])
    group_coverage = np.asarray([
        [float(summary_by_type[query_type][f"flat_group_coverage_b{budget}_mean"]) * 100.0 for budget in budgets]
        for query_type in QUERY_ORDER
    ])
    fig, axes = plt.subplots(2, 2, figsize=(7.7, 6.15), layout="constrained")
    annotated_heatmap(axes[0, 0], all_group, type_labels, [f"B={budget}" for budget in budgets],
                      "a  Queries with all evidence groups covered (%)", ".0f", "Blues", vmin=0, vmax=100)
    annotated_heatmap(axes[0, 1], group_coverage, type_labels, [f"B={budget}" for budget in budgets],
                      "b  Mean evidence-group coverage (%)", ".0f", "Greens", vmin=0, vmax=100, show_y=False)

    ax = axes[1, 0]
    categories = [1, 2, 3, 4]
    colors = ["#C7DCEB", "#79ADC8", NAVY, "#173B57"]
    left = np.zeros(len(QUERY_ORDER))
    for category, color in zip(categories, colors, strict=True):
        values = []
        for query_type in QUERY_ORDER:
            subset = [int(row["evidence_group_count"]) for row in per_query if row["query_type"] == query_type]
            count = sum((value == category if category < 4 else value >= 4) for value in subset)
            values.append(100.0 * count / max(len(subset), 1))
        ax.barh(range(len(QUERY_ORDER)), values, left=left, color=color, height=0.68)
        left += np.asarray(values)
    ax.set_yticks(range(len(QUERY_ORDER)), type_labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Queries (%)")
    ax.set_title("c  Number of required evidence groups", loc="left", fontweight="bold")
    clean_axis(ax, grid="x")
    handles = [Patch(facecolor=color, label=(str(category) if category < 4 else "4+")) for category, color in zip(categories, colors, strict=True)]
    fig.legend(handles=handles, title="Required groups", frameon=False, ncol=4, loc="outside lower center")

    ax = axes[1, 1]
    group_categories = [1, 2, 3, 4]
    for budget, color, marker in zip((8, 16, 32), (ORANGE, TEAL, NAVY), ("s", "^", "o"), strict=True):
        means, errors = [], []
        for category in group_categories:
            subset = [float(row[f"flat_all_groups_b{budget}"]) for row in per_query
                      if (int(row["evidence_group_count"]) == category if category < 4 else int(row["evidence_group_count"]) >= 4)]
            mean = float(np.mean(subset)) if subset else float("nan")
            error = 1.96 * np.sqrt(mean * (1.0 - mean) / len(subset)) if subset else float("nan")
            means.append(mean * 100.0)
            errors.append(error * 100.0)
        ax.errorbar(group_categories, means, yerr=errors, color=color, marker=marker, capsize=2.5, label=f"B={budget}")
    counts = [sum((int(row["evidence_group_count"]) == category if category < 4 else int(row["evidence_group_count"]) >= 4) for row in per_query) for category in group_categories]
    ax.set_xticks(group_categories, [f"{category if category < 4 else '4+'}\nn={count}" for category, count in zip(group_categories, counts, strict=True)])
    ax.set_ylim(-2, 102)
    ax.set_xlabel("Required evidence groups")
    ax.set_ylabel("All groups covered (%)")
    ax.set_title("d  Flat retrieval degrades as composition grows", loc="left", fontweight="bold")
    clean_axis(ax)
    ax.legend(frameon=False, loc="upper right")
    fig.suptitle("Compositional evidence demand is common and poorly served by flat top-B retrieval", fontsize=12, fontweight="bold")
    save_figure(fig, "Fig_V5_S1_evidence_structure")


def plot_s2(s2: list[dict]) -> None:
    growth = np.asarray([int(row["growth_percent_of_full_corpus"]) for row in s2])
    changed = np.asarray([float(row["changed_postings"]) for row in s2])
    full = np.asarray([float(row["full_rebuild_postings"]) for row in s2])
    incremental_time = np.asarray([float(row["incremental_update_ms"]) / 1000.0 for row in s2])
    rebuild_time = np.asarray([float(row["full_rebuild_ms"]) / 1000.0 for row in s2])
    fig = plt.figure(figsize=(7.6, 5.2), layout="constrained")
    grid = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.85])
    ax_work = fig.add_subplot(grid[0, 0])
    ax_time = fig.add_subplot(grid[0, 1])
    ax_ratio = fig.add_subplot(grid[1, :2])
    ax_summary = fig.add_subplot(grid[:, 2])

    width = 0.34
    positions = np.arange(len(growth))
    ax_work.bar(positions - width / 2, changed / 1e6, width, color=TEAL, label="New postings")
    ax_work.bar(positions + width / 2, full / 1e6, width, color=NAVY, label="Full rebuild")
    ax_work.set_xticks(positions, [f"{value}%" for value in growth])
    ax_work.set_ylabel("Posting operations (M)")
    ax_work.set_title("a  Deterministic maintenance work", loc="left", fontweight="bold")
    clean_axis(ax_work)
    ax_work.legend(frameon=False)

    ax_time.plot(growth, incremental_time, color=TEAL, marker="o", label="Incremental")
    ax_time.plot(growth, rebuild_time, color=NAVY, marker="s", label="Full rebuild")
    ax_time.set_yscale("log")
    ax_time.set_xlabel("Appended content (% of full corpus)")
    ax_time.set_ylabel("Measured time (s, log scale)")
    ax_time.set_title("b  Wall-clock stress", loc="left", fontweight="bold")
    clean_axis(ax_time, grid="both")
    ax_time.legend(frameon=False)

    operation_ratio = full / np.maximum(changed, 1.0)
    time_ratio = rebuild_time / np.maximum(incremental_time, 1e-9)
    ax_ratio.plot(growth, operation_ratio, color=TEAL, marker="o", label="Operation ratio")
    ax_ratio.plot(growth, time_ratio, color=ORANGE, marker="s", label="Measured-time ratio")
    ax_ratio.axhline(1.0, color=GRAY, linestyle="--", linewidth=1)
    ax_ratio.set_xlabel("Appended content (% of full corpus)")
    ax_ratio.set_ylabel("Full rebuild / incremental")
    ax_ratio.set_title("c  Rebuild overhead remains above local maintenance", loc="left", fontweight="bold")
    clean_axis(ax_ratio)
    ax_ratio.legend(frameon=False, ncol=2)

    ax_summary.axis("off")
    total_probes = sum(int(row["probe_count"]) for row in s2)
    total_mismatches = sum(int(row["lookup_mismatch_count"]) for row in s2)
    final = s2[-1]
    summary_lines = [
        ("0", f"mismatches / {total_probes} probes"),
        (f"{100 * float(final['changed_posting_fraction_of_full']):.1f}%", "postings changed at 20% growth"),
        (f"{float(final['full_to_incremental_time_ratio']):.1f}x", "measured rebuild overhead"),
    ]
    ax_summary.text(0.0, 1.0, "d  Correctness and scope", fontsize=9.5, fontweight="bold", va="top")
    y = 0.82
    for value, label in summary_lines:
        ax_summary.text(0.02, y, value, fontsize=17, fontweight="bold", color=NAVY, va="top")
        ax_summary.text(0.02, y - 0.09, label, fontsize=7.5, color=GRAY, va="top", wrap=True)
        y -= 0.25
    ax_summary.text(0.02, 0.03, "Static content insertion only\nNo causal access-stream claim", fontsize=7.2,
                    color=ORANGE, va="bottom", fontweight="bold")
    fig.suptitle("Local append maintenance preserves exact lookups while avoiding full rebuild work", fontsize=12, fontweight="bold")
    save_figure(fig, "Fig_V5_S2_content_maintenance")


def render_all(m1: list[dict], m2: list[dict], m3: list[dict], s1_summary: list[dict], s1_per_query: list[dict], s2: list[dict]) -> None:
    configure_style()
    plot_m1(m1)
    plot_m2(m2)
    plot_m3(m3)
    plot_s1(s1_summary, s1_per_query)
    plot_s2(s2)
