#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

from plot_common import save_figure


FIGURE_NAME = "Fig_frontier_cost_reference_fidelity"

# Source aggregate:
# aggregate/M2_frontier_budget_summary.csv
# Resolved source:
# ../20260704_method1_motivation_medmemorybench_v3/aggregate/M2_frontier_budget_summary.csv
# Values are grouped means over query_type for each frontier_budget.

DATA = [
    {
        "frontier_budget": 8,
        "selection_ms": 58.54915657952612,
        "rerank_ms": 43.941024008987974,
        "oracle_overlap_at_5": 0.5541666666666667,
    },
    {
        "frontier_budget": 16,
        "selection_ms": 97.91228286964422,
        "rerank_ms": 84.51457879109512,
        "oracle_overlap_at_5": 0.7200000000000002,
    },
    {
        "frontier_budget": 32,
        "selection_ms": 143.76873724001294,
        "rerank_ms": 130.29687445832678,
        "oracle_overlap_at_5": 0.8575000000000003,
    },
    {
        "frontier_budget": 64,
        "selection_ms": 229.22917305022565,
        "rerank_ms": 215.7462576547914,
        "oracle_overlap_at_5": 1.0,
    },
]


def configure_font() -> str:
    """Register the project-local Arial font and configure Matplotlib."""

    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    script_dir = Path(__file__).resolve().parent
    font_path = script_dir / "fonts" / "Arial.ttf"

    if not font_path.exists():
        raise FileNotFoundError(
            f"Arial font file was not found: {font_path}\n"
            "Please place arial.ttf under the project's fonts/ directory."
        )

    # Register the local TTF file for this Python process.
    font_manager.fontManager.addfont(str(font_path))

    # Read the actual family name encoded in the TTF file.
    font_properties = font_manager.FontProperties(fname=str(font_path))
    font_name = font_properties.get_name()

    plt.rcParams.update(
        {
            "font.family": font_name,
            "font.size": 16,
            "pdf.fonttype": 42,   # Embed TrueType fonts in PDF.
            "ps.fonttype": 42,    # Embed TrueType fonts in PS/EPS.
            "svg.fonttype": "none",
        }
    )

    return font_name


def main() -> None:
    import matplotlib.pyplot as plt

    font_name = configure_font()
    print(f"Using font: {font_name}")

    rows = sorted(DATA, key=lambda row: row["frontier_budget"])

    budgets = [row["frontier_budget"] for row in rows]
    selection_ms = [row["selection_ms"] for row in rows]
    rerank_ms = [row["rerank_ms"] for row in rows]
    overlap_at_5 = [row["oracle_overlap_at_5"] for row in rows]

    fig, ax1 = plt.subplots(figsize=(7, 5))

    selection_line = ax1.plot(
        budgets,
        selection_ms,
        marker="o",
        markersize=8,
        linewidth=2.2,
        color="tab:blue",
        label="Selection latency",
    )

    rerank_line = ax1.plot(
        budgets,
        rerank_ms,
        marker="s",
        markersize=8,
        linewidth=2.2,
        color="tab:cyan",
        label="Reranking latency",
    )

    ax1.set_xlabel("Frontier Budget", fontsize=24)
    ax1.set_ylabel("Latency (ms)", fontsize=24)

    # Only display the four evaluated budget configurations.
    ax1.set_xticks(budgets)
    ax1.set_xticklabels([str(budget) for budget in budgets], fontsize=16)

    # Keep numerical spacing: 8→16 and 32→64 are represented proportionally.
    ax1.set_xlim(min(budgets) - 3, max(budgets) + 3)

    ax1.tick_params(
        axis="x",
        labelsize=16,
        width=1.2,
        length=5,
    )
    ax1.tick_params(
        axis="y",
        labelsize=16,
        width=1.2,
        length=5,
    )

    ax2 = ax1.twinx()

    overlap_line = ax2.plot(
        budgets,
        overlap_at_5,
        marker="^",
        markersize=8,
        linewidth=2.2,
        color="tab:red",
        label="Reference overlap@5",
    )

    ax2.set_ylabel("Reranker Oracle Overlap@5", fontsize=20)
    ax2.set_ylim(0.5, 1.05)
    ax2.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    ax2.tick_params(
        axis="y",
        labelsize=16,
        width=1.2,
        length=5,
    )

    lines = selection_line + rerank_line + overlap_line
    labels = [line.get_label() for line in lines]

    ax1.legend(
        lines,
        labels,
        loc="lower right",
        prop={
            "family": font_name,
            "size": 13,
        },
        frameon=True,
    )

    ax1.set_title(
        "Frontier Cost and Reference Fidelity",
        fontsize=20,
        pad=12,
    )

    fig.tight_layout()
    save_figure(fig, FIGURE_NAME)

    plt.close(fig)
    print(f"wrote {FIGURE_NAME}.png/.pdf")


if __name__ == "__main__":
    main()