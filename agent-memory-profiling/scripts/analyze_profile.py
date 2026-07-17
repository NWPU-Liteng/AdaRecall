#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-agent-memory")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/agent-memory-cache")

def safe_plot(df, x, y, hue, output, title):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    plt.figure(figsize=(8, 5))
    if hue and hue in df.columns:
        for key, group in df.groupby(hue):
            plt.plot(group[x], group[y], marker="o", label=str(key))
        plt.legend()
    else:
        plt.plot(df[x], df[y], marker="o")
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    Path("results/figures").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    clean = df[df["error"].fillna("") == ""].copy()
    if clean.empty:
        summary = df.groupby(["memory_size", "query_type"], dropna=False).size().reset_index(name="runs")
        summary.to_csv("results/profile_summary.csv", index=False)
        print("No clean rows available; wrote fallback summary only.")
        return

    for col in ["t_embed", "t_search", "t_filter", "t_rerank", "t_pack", "t_llm", "t_total", "memory_invocation_ratio"]:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
    if "t_memory" in clean.columns:
        clean["t_memory"] = pd.to_numeric(clean["t_memory"], errors="coerce")
    if "num_memory_calls" in clean.columns:
        clean["num_memory_calls"] = pd.to_numeric(clean["num_memory_calls"], errors="coerce")

    group_cols = [
        "memory_size",
        "query_type",
        "module",
        "candidate_k",
        "top_k",
        "backend_mode",
        "llm_mode",
        "embedding_mode",
    ]
    for optional_col in ["execution_mode", "framework_mode", "memory_rounds"]:
        if optional_col in clean.columns:
            group_cols.insert(3, optional_col)

    summary = (
        clean.groupby(group_cols, dropna=False)
        .agg(
            runs=("t_total", "count"),
            avg_t_total=("t_total", "mean"),
            p95_t_total=("t_total", lambda s: s.quantile(0.95)),
            avg_memory_invocation_ratio=("memory_invocation_ratio", "mean"),
            avg_t_embed=("t_embed", "mean"),
            avg_t_search=("t_search", "mean"),
            avg_t_filter=("t_filter", "mean"),
            avg_t_rerank=("t_rerank", "mean"),
            avg_t_pack=("t_pack", "mean"),
            avg_t_memory=("t_memory", "mean") if "t_memory" in clean.columns else ("t_total", "mean"),
            avg_t_llm=("t_llm", "mean"),
            avg_num_memory_calls=("num_memory_calls", "mean") if "num_memory_calls" in clean.columns else ("t_total", "count"),
        )
        .reset_index()
    )
    summary.to_csv("results/profile_summary.csv", index=False)

    size_df = clean.groupby("memory_size", as_index=False)["t_total"].mean().sort_values("memory_size")
    safe_plot(
        size_df,
        x="memory_size",
        y="t_total",
        hue=None,
        output="results/figures/memory_size_vs_invocation_latency.png",
        title="Memory Size vs Invocation Latency",
    )

    breakdown = clean.groupby("memory_size", as_index=False)[["t_embed", "t_search", "t_filter", "t_rerank", "t_pack", "t_llm"]].mean()
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(9, 5))
        for col in ["t_embed", "t_search", "t_filter", "t_rerank", "t_pack", "t_llm"]:
            plt.plot(breakdown["memory_size"], breakdown[col], marker="o", label=col)
        plt.legend()
        plt.title("Latency Breakdown by Memory Size")
        plt.xlabel("memory_size")
        plt.ylabel("seconds")
        plt.tight_layout()
        plt.savefig("results/figures/latency_breakdown_by_memory_size.png", dpi=160)
        plt.close()
    except Exception:
        pass

    candidate_df = clean.groupby("candidate_k", as_index=False)["t_total"].mean().sort_values("candidate_k")
    safe_plot(
        candidate_df,
        x="candidate_k",
        y="t_total",
        hue=None,
        output="results/figures/candidate_k_vs_latency.png",
        title="Candidate K vs Latency",
    )

    module_df = clean.groupby("module", as_index=False)["t_total"].mean().sort_values("t_total")
    safe_plot(
        module_df,
        x="module",
        y="t_total",
        hue=None,
        output="results/figures/module_type_vs_invocation_latency.png",
        title="Module Type vs Invocation Latency",
    )
    print("Analysis complete.")


if __name__ == "__main__":
    main()
