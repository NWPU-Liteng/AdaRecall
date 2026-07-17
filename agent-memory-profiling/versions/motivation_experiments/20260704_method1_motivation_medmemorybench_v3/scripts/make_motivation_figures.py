#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-medmmb-v3")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/medmmb-v3-cache")


def _read_csv(path: Path):
    import pandas as pd

    return pd.read_csv(path) if path.exists() else None


def _save(fig, figures: Path, name: str) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figures / f"{name}.png", dpi=180)
    fig.savefig(figures / f"{name}.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description="Make motivation figures from aggregate CSV files.")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    agg = root / "aggregate"
    figures = root / "figures"

    import matplotlib.pyplot as plt

    m1 = _read_csv(agg / "M1_history_scale_summary.csv")
    if m1 is not None and not m1.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        for condition, sub in m1.groupby("history_condition"):
            sub = sub.copy()
            sub["x"] = sub["history_scale"].apply(lambda v: 999999 if str(v) == "full" else int(v))
            sub = sub.sort_values("x")
            ax.plot(sub["history_scale"].astype(str), sub["t_total_selection_ms_mean"], marker="o", label=condition)
        ax.set_xlabel("History scale in memory items")
        ax.set_ylabel("Selection latency (ms)")
        ax.set_title("M1 History-Scale Selection Latency")
        ax.legend()
        _save(fig, figures, "Fig_M1_history_scale_latency")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        for condition, sub in m1.groupby("history_condition"):
            ax.plot(sub["history_scale"].astype(str), sub["host_memory_bytes_mean"], marker="o", label=condition)
        ax.set_xlabel("History scale in memory items")
        ax.set_ylabel("Host/index/payload bytes")
        ax.set_title("M1 History-Scale Traffic")
        ax.legend()
        _save(fig, figures, "Fig_M1_history_scale_traffic")
        plt.close(fig)

    m2 = _read_csv(agg / "M2_frontier_budget_summary.csv")
    if m2 is not None and not m2.empty:
        grouped = m2.groupby("frontier_budget", as_index=False).mean(numeric_only=True)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(grouped["t_total_selection_ms_mean"], grouped["evidence_recall_at_k_mean"], marker="o", label="Evidence Recall@K")
        ax.plot(grouped["t_total_selection_ms_mean"], grouped["evidence_completion_at_k_mean"], marker="s", label="Evidence Completion@K")
        ax.set_xlabel("Selection cost (ms)")
        ax.set_ylabel("Evidence metric")
        ax.set_title("M2 Frontier Cost vs Evidence")
        ax.legend()
        _save(fig, figures, "Fig_M2_frontier_cost_evidence")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 5))
        if "answer_generation_latency_ms_mean" in grouped:
            ax.plot(grouped["frontier_budget"], grouped.get("evidence_recall_at_k_mean", 0), marker="o")
        ax.set_xlabel("Frontier budget")
        ax.set_ylabel("Task/evidence proxy")
        ax.set_title("M2 Budget Task Quality")
        _save(fig, figures, "Fig_M2_budget_task_quality")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(grouped["frontier_budget"], grouped["reranker_oracle_overlap_at_5_mean"], marker="o")
        ax.set_xlabel("Frontier budget")
        ax.set_ylabel("RerankerOracleOverlap@5")
        ax.set_title("M2 Budget Oracle Overlap")
        _save(fig, figures, "Fig_M2_budget_oracle_overlap")
        plt.close(fig)

    m3 = _read_csv(agg / "M3_tile_scattering.csv")
    if m3 is not None and not m3.empty:
        for metric, name, ylabel in [
            ("tile_utilization", "Fig_M3_tile_utilization", "Tile utilization"),
            ("access_amplification", "Fig_M3_access_amplification", "Access amplification"),
        ]:
            fig, ax = plt.subplots(figsize=(7, 5))
            for layout, sub in m3.groupby("layout"):
                g = sub.groupby("tile_capacity", as_index=False)[metric].mean()
                ax.plot(g["tile_capacity"], g[metric], marker="o", label=layout)
            ax.set_xlabel("Tile capacity")
            ax.set_ylabel(ylabel)
            ax.set_title(name)
            ax.legend()
            _save(fig, figures, name)
            plt.close(fig)
        c16 = m3[m3["tile_capacity"] == 16]
        if not c16.empty:
            fig, ax = plt.subplots(figsize=(9, 5))
            c16.groupby(["query_type", "layout"])["scatter_ratio"].mean().unstack().plot(kind="bar", ax=ax)
            ax.set_xlabel("Native query type")
            ax.set_ylabel("Scatter ratio")
            ax.set_title("M3 Scatter Ratio by Query Type")
            _save(fig, figures, "Fig_M3_scatter_ratio_by_query_type")
            plt.close(fig)

    o1 = _read_csv(agg / "O1_same_cross_stage_stats.csv")
    if o1 is not None and not o1.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        x = range(len(o1))
        ax.bar([i - 0.2 for i in x], o1["same_stage_overlap"], width=0.4, label="same-stage")
        ax.bar([i + 0.2 for i in x], o1["cross_stage_overlap"], width=0.4, label="cross-stage")
        ax.set_xticks(list(x))
        ax.set_xticklabels(o1["stage"])
        ax.set_ylabel("Overlap")
        ax.set_title("O1 Same vs Cross Stage Overlap")
        ax.legend()
        _save(fig, figures, "Fig_O1_same_vs_cross_stage_overlap")
        plt.close(fig)

    null = _read_csv(agg / "O1_stage_shuffle_null.csv")
    if null is not None and not null.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(null["statistic"], bins=40, alpha=0.8)
        ax.axvline(null["observed_statistic"].iloc[0], color="red", label="observed")
        ax.set_xlabel("Same-minus-cross statistic under stage shuffle")
        ax.set_ylabel("Count")
        ax.set_title("O1 Stage-Shuffle Null")
        ax.legend()
        _save(fig, figures, "Fig_O1_stage_shuffle_null")
        plt.close(fig)

    pair = _read_csv(agg / "O1_pair_recurrence.csv")
    if pair is not None and not pair.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        x = range(len(pair))
        ax.bar([i - 0.2 for i in x], pair["matching_stage_pair_recall"], width=0.4, label="matching-stage")
        ax.bar([i + 0.2 for i in x], pair["global_pair_recall"], width=0.4, label="global")
        ax.set_xticks(list(x))
        ax.set_xticklabels(pair["stage"])
        ax.set_ylabel("Held-out pair recall")
        ax.set_title("O1 Stage Pair Recurrence")
        ax.legend()
        _save(fig, figures, "Fig_O1_stage_pair_recurrence")
        plt.close(fig)
    print(f"figures={figures}")


if __name__ == "__main__":
    main()
