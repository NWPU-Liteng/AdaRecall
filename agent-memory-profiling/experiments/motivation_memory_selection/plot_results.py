#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-agent-memory-motivation")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/agent-memory-motivation-cache")


def _prepare(input_path: Path) -> pd.DataFrame:
    df = pd.read_json(input_path, lines=True)
    clean = df[df["error"].fillna("") == ""].copy()
    for column in [
        "memory_size",
        "num_raw_retrieved",
        "num_after_metadata_filter",
        "num_after_relation_expansion",
        "num_after_dedup",
        "num_before_rerank",
        "query_encoding_latency_ms",
        "coarse_retrieval_latency_ms",
        "metadata_filter_latency_ms",
        "relation_expansion_latency_ms",
        "candidate_aggregation_latency_ms",
        "rerank_latency_ms",
        "context_construction_latency_ms",
        "agent_planning_latency_ms",
        "agent_tool_execution_latency_ms",
        "agent_reasoning_latency_ms",
        "agent_output_latency_ms",
        "agent_execution_total_latency_ms",
        "total_response_latency_ms",
        "estimated_host_memory_traffic_bytes",
    ]:
        if column in clean.columns:
            clean[column] = pd.to_numeric(clean[column], errors="coerce").fillna(0.0)
    return clean


def _save_or_skip(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    fig.clf()


def make_plots(input_path: Path, figures_dir: Path) -> None:
    clean = _prepare(input_path)
    if clean.empty:
        return

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    baseline = clean[clean["baseline_mode"] == "full_selection"].copy()
    if baseline.empty:
        baseline = clean.copy()

    stage_columns = [
        "coarse_retrieval_latency_ms",
        "metadata_filter_latency_ms",
        "relation_expansion_latency_ms",
        "candidate_aggregation_latency_ms",
        "rerank_latency_ms",
        "context_construction_latency_ms",
    ]
    if "agent_execution_total_latency_ms" in baseline.columns and baseline["agent_execution_total_latency_ms"].sum() > 0:
        stage_columns.append("agent_execution_total_latency_ms")

    latency = (
        baseline.groupby("memory_size", as_index=False)[
            stage_columns
        ]
        .mean()
        .sort_values("memory_size")
    )
    fig1, ax1 = plt.subplots(figsize=(9, 5))
    bottom = None
    for column in stage_columns:
        values = latency[column]
        ax1.bar(latency["memory_size"].astype(str), values, bottom=bottom, label=column.replace("_latency_ms", ""))
        bottom = values if bottom is None else bottom + values
    ax1.set_title("Response Latency Breakdown vs Memory Size")
    ax1.set_xlabel("memory size")
    ax1.set_ylabel("latency (ms)")
    ax1.legend(fontsize=8)
    _save_or_skip(fig1, figures_dir / "latency_breakdown_vs_memory_size.png")

    candidates = (
        baseline.groupby("memory_size", as_index=False)[
            [
                "num_raw_retrieved",
                "num_after_metadata_filter",
                "num_after_relation_expansion",
                "num_after_dedup",
                "num_before_rerank",
            ]
        ]
        .mean()
        .sort_values("memory_size")
    )
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    for column in [
        "num_raw_retrieved",
        "num_after_metadata_filter",
        "num_after_relation_expansion",
        "num_after_dedup",
        "num_before_rerank",
    ]:
        ax2.plot(candidates["memory_size"], candidates[column], marker="o", label=column)
    ax2.set_title("Candidate Count Breakdown vs Memory Size")
    ax2.set_xlabel("memory size")
    ax2.set_ylabel("candidate count")
    ax2.legend(fontsize=8)
    _save_or_skip(fig2, figures_dir / "candidate_count_breakdown.png")

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.scatter(clean["num_before_rerank"], clean["rerank_latency_ms"], alpha=0.6, s=18)
    ax3.set_title("Rerank Latency vs Candidates Before Rerank")
    ax3.set_xlabel("candidates before rerank")
    ax3.set_ylabel("rerank latency (ms)")
    _save_or_skip(fig3, figures_dir / "rerank_latency_vs_candidates.png")

    traffic = (
        baseline.groupby("memory_size", as_index=False)["estimated_host_memory_traffic_bytes"]
        .mean()
        .sort_values("memory_size")
    )
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    ax4.plot(
        traffic["memory_size"],
        traffic["estimated_host_memory_traffic_bytes"] / (1024.0 * 1024.0),
        marker="o",
    )
    ax4.set_title("Host-memory Traffic vs Memory Size")
    ax4.set_xlabel("memory size")
    ax4.set_ylabel("traffic (MB)")
    _save_or_skip(fig4, figures_dir / "host_memory_traffic.png")

    query_breakdown = (
        baseline.groupby("query_type", as_index=False)[
            stage_columns
        ]
        .mean()
        .sort_values("query_type")
    )
    fig5, ax5 = plt.subplots(figsize=(10, 5))
    position = range(len(query_breakdown))
    bottom = None
    for column in stage_columns:
        values = query_breakdown[column]
        ax5.bar(position, values, bottom=bottom, label=column.replace("_latency_ms", ""))
        bottom = values if bottom is None else bottom + values
    ax5.set_xticks(list(position))
    ax5.set_xticklabels(query_breakdown["query_type"], rotation=20)
    ax5.set_title("Query Type Breakdown")
    ax5.set_ylabel("latency (ms)")
    ax5.legend(fontsize=8)
    _save_or_skip(fig5, figures_dir / "query_type_breakdown.png")

    agent_stage_columns = [
        "agent_planning_latency_ms",
        "agent_tool_execution_latency_ms",
        "agent_reasoning_latency_ms",
        "agent_output_latency_ms",
    ]
    if all(column in baseline.columns for column in agent_stage_columns) and baseline[agent_stage_columns].sum().sum() > 0:
        agent_latency = (
            baseline.groupby("memory_size", as_index=False)[
                agent_stage_columns
            ]
            .mean()
            .sort_values("memory_size")
        )
        fig6, ax6 = plt.subplots(figsize=(9, 5))
        bottom = None
        for column in agent_stage_columns:
            values = agent_latency[column]
            ax6.bar(
                agent_latency["memory_size"].astype(str),
                values,
                bottom=bottom,
                label=column.replace("agent_", "").replace("_latency_ms", ""),
            )
            bottom = values if bottom is None else bottom + values
        ax6.set_title("Agent Execution Stage Breakdown vs Memory Size")
        ax6.set_xlabel("memory size")
        ax6.set_ylabel("latency (ms)")
        ax6.legend(fontsize=8)
        _save_or_skip(fig6, figures_dir / "agent_execution_stage_breakdown.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--figures-dir", required=True)
    args = parser.parse_args()
    make_plots(Path(args.input), Path(args.figures_dir))


if __name__ == "__main__":
    main()
