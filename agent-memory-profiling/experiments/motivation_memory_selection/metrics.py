#!/usr/bin/env python3
from pathlib import Path
from typing import List

import pandas as pd


def _safe_quantile(series: pd.Series, q: float) -> float:
    if series.empty:
        return 0.0
    return float(series.quantile(q))


def build_summary(raw_df: pd.DataFrame) -> pd.DataFrame:
    clean = raw_df[raw_df["error"].fillna("") == ""].copy()
    if clean.empty:
        return pd.DataFrame()

    if "agent_stage_sum_latency_ms" not in clean.columns:
        clean["agent_stage_sum_latency_ms"] = (
            pd.to_numeric(clean.get("agent_planning_latency_ms", 0.0), errors="coerce").fillna(0.0)
            + pd.to_numeric(clean.get("agent_tool_execution_latency_ms", 0.0), errors="coerce").fillna(0.0)
            + pd.to_numeric(clean.get("agent_reasoning_latency_ms", 0.0), errors="coerce").fillna(0.0)
            + pd.to_numeric(clean.get("agent_output_latency_ms", 0.0), errors="coerce").fillna(0.0)
        )
    if "agent_planning_ratio" not in clean.columns:
        total = pd.to_numeric(clean.get("agent_execution_total_latency_ms", 0.0), errors="coerce").fillna(0.0)
        planning = pd.to_numeric(clean.get("agent_planning_latency_ms", 0.0), errors="coerce").fillna(0.0)
        clean["agent_planning_ratio"] = planning / total.clip(lower=1e-9)
    if "agent_tool_execution_ratio" not in clean.columns:
        total = pd.to_numeric(clean.get("agent_execution_total_latency_ms", 0.0), errors="coerce").fillna(0.0)
        tool = pd.to_numeric(clean.get("agent_tool_execution_latency_ms", 0.0), errors="coerce").fillna(0.0)
        clean["agent_tool_execution_ratio"] = tool / total.clip(lower=1e-9)
    if "agent_reasoning_ratio" not in clean.columns:
        total = pd.to_numeric(clean.get("agent_execution_total_latency_ms", 0.0), errors="coerce").fillna(0.0)
        reasoning = pd.to_numeric(clean.get("agent_reasoning_latency_ms", 0.0), errors="coerce").fillna(0.0)
        clean["agent_reasoning_ratio"] = reasoning / total.clip(lower=1e-9)
    if "agent_output_ratio" not in clean.columns:
        total = pd.to_numeric(clean.get("agent_execution_total_latency_ms", 0.0), errors="coerce").fillna(0.0)
        output = pd.to_numeric(clean.get("agent_output_latency_ms", 0.0), errors="coerce").fillna(0.0)
        clean["agent_output_ratio"] = output / total.clip(lower=1e-9)
    if "selection_invocation_count" not in clean.columns:
        if "agent_memory_mode" in clean.columns:
            clean["selection_invocation_count"] = clean["agent_memory_mode"].astype(str).map(
                {"phase_aware": 3, "shared_once": 1}
            ).fillna(1)
        else:
            clean["selection_invocation_count"] = 1
    if "phase_selection_total_latency_ms" not in clean.columns:
        clean["phase_selection_total_latency_ms"] = pd.to_numeric(
            clean.get("selection_total_latency_ms", 0.0), errors="coerce"
        ).fillna(0.0)
    if "phase_retrieval_total_latency_ms" not in clean.columns:
        clean["phase_retrieval_total_latency_ms"] = (
            pd.to_numeric(clean.get("query_encoding_latency_ms", 0.0), errors="coerce").fillna(0.0)
            + pd.to_numeric(clean.get("coarse_retrieval_latency_ms", 0.0), errors="coerce").fillna(0.0)
            + pd.to_numeric(clean.get("metadata_filter_latency_ms", 0.0), errors="coerce").fillna(0.0)
            + pd.to_numeric(clean.get("relation_expansion_latency_ms", 0.0), errors="coerce").fillna(0.0)
            + pd.to_numeric(clean.get("candidate_aggregation_latency_ms", 0.0), errors="coerce").fillna(0.0)
            + pd.to_numeric(clean.get("context_construction_latency_ms", 0.0), errors="coerce").fillna(0.0)
        )
    if "phase_rerank_total_latency_ms" not in clean.columns:
        clean["phase_rerank_total_latency_ms"] = pd.to_numeric(
            clean.get("rerank_latency_ms", 0.0), errors="coerce"
        ).fillna(0.0)
    if "phase_selection_to_total_response_ratio" not in clean.columns:
        total_response = pd.to_numeric(clean.get("total_response_latency_ms", 0.0), errors="coerce").fillna(0.0)
        phase_selection = pd.to_numeric(clean.get("phase_selection_total_latency_ms", 0.0), errors="coerce").fillna(0.0)
        clean["phase_selection_to_total_response_ratio"] = phase_selection / total_response.clip(lower=1e-9)
    if "phase_retrieval_to_total_response_ratio" not in clean.columns:
        total_response = pd.to_numeric(clean.get("total_response_latency_ms", 0.0), errors="coerce").fillna(0.0)
        phase_retrieval = pd.to_numeric(clean.get("phase_retrieval_total_latency_ms", 0.0), errors="coerce").fillna(0.0)
        clean["phase_retrieval_to_total_response_ratio"] = phase_retrieval / total_response.clip(lower=1e-9)
    if "phase_rerank_to_total_response_ratio" not in clean.columns:
        total_response = pd.to_numeric(clean.get("total_response_latency_ms", 0.0), errors="coerce").fillna(0.0)
        phase_rerank = pd.to_numeric(clean.get("phase_rerank_total_latency_ms", 0.0), errors="coerce").fillna(0.0)
        clean["phase_rerank_to_total_response_ratio"] = phase_rerank / total_response.clip(lower=1e-9)
    if "phase_rerank_ratio" not in clean.columns:
        phase_selection = pd.to_numeric(clean.get("phase_selection_total_latency_ms", 0.0), errors="coerce").fillna(0.0)
        phase_rerank = pd.to_numeric(clean.get("phase_rerank_total_latency_ms", 0.0), errors="coerce").fillna(0.0)
        clean["phase_rerank_ratio"] = phase_rerank / phase_selection.clip(lower=1e-9)

    numeric_columns = [
        "memory_size",
        "retrieval_topk",
        "relation_expansion_depth",
        "selection_invocation_count",
        "num_raw_retrieved",
        "num_after_metadata_filter",
        "num_after_relation_expansion",
        "num_after_dedup",
        "num_before_rerank",
        "num_after_rerank",
        "query_encoding_latency_ms",
        "coarse_retrieval_latency_ms",
        "metadata_filter_latency_ms",
        "relation_expansion_latency_ms",
        "candidate_aggregation_latency_ms",
        "rerank_latency_ms",
        "context_construction_latency_ms",
        "generation_latency_ms",
        "agent_planning_latency_ms",
        "agent_tool_execution_latency_ms",
        "agent_reasoning_latency_ms",
        "agent_output_latency_ms",
        "agent_execution_latency_ms",
        "agent_execution_total_latency_ms",
        "agent_stage_sum_latency_ms",
        "selection_total_latency_ms",
        "phase_selection_total_latency_ms",
        "phase_retrieval_total_latency_ms",
        "phase_rerank_total_latency_ms",
        "total_response_latency_ms",
        "end_to_end_latency_ms",
        "selection_ratio",
        "selection_to_total_response_ratio",
        "phase_selection_to_total_response_ratio",
        "phase_retrieval_to_total_response_ratio",
        "phase_rerank_to_total_response_ratio",
        "agent_execution_ratio",
        "agent_planning_ratio",
        "agent_tool_execution_ratio",
        "agent_reasoning_ratio",
        "agent_output_ratio",
        "rerank_ratio",
        "phase_rerank_ratio",
        "estimated_host_memory_traffic_bytes",
    ]
    for column in numeric_columns:
        if column not in clean.columns:
            clean[column] = 0.0
        clean[column] = pd.to_numeric(clean[column], errors="coerce").fillna(0.0)

    clean["avg_host_memory_traffic_mb"] = clean["estimated_host_memory_traffic_bytes"] / (1024.0 * 1024.0)

    group_columns = [
        "baseline_mode",
        "memory_size",
        "query_type",
        "retrieval_topk",
        "relation_expansion_depth",
        "rerank_topk_input",
    ]
    summary = (
        clean.groupby(group_columns, dropna=False)
        .agg(
            runs=("query_id", "count"),
            avg_selection_latency_ms=("selection_total_latency_ms", "mean"),
            p50_selection_latency_ms=("selection_total_latency_ms", "median"),
            p95_selection_latency_ms=("selection_total_latency_ms", lambda s: _safe_quantile(s, 0.95)),
            p99_selection_latency_ms=("selection_total_latency_ms", lambda s: _safe_quantile(s, 0.99)),
            avg_rerank_latency_ms=("rerank_latency_ms", "mean"),
            avg_rerank_ratio=("rerank_ratio", "mean"),
            avg_selection_invocation_count=("selection_invocation_count", "mean"),
            avg_phase_selection_latency_ms=("phase_selection_total_latency_ms", "mean"),
            avg_phase_retrieval_latency_ms=("phase_retrieval_total_latency_ms", "mean"),
            avg_phase_rerank_latency_ms=("phase_rerank_total_latency_ms", "mean"),
            avg_phase_rerank_ratio=("phase_rerank_ratio", "mean"),
            avg_agent_execution_latency_ms=("agent_execution_total_latency_ms", "mean"),
            avg_agent_planning_latency_ms=("agent_planning_latency_ms", "mean"),
            avg_agent_tool_execution_latency_ms=("agent_tool_execution_latency_ms", "mean"),
            avg_agent_reasoning_latency_ms=("agent_reasoning_latency_ms", "mean"),
            avg_agent_output_latency_ms=("agent_output_latency_ms", "mean"),
            avg_agent_stage_sum_latency_ms=("agent_stage_sum_latency_ms", "mean"),
            avg_total_response_latency_ms=("total_response_latency_ms", "mean"),
            avg_selection_to_total_response_ratio=("selection_to_total_response_ratio", "mean"),
            avg_phase_selection_to_total_response_ratio=("phase_selection_to_total_response_ratio", "mean"),
            avg_phase_retrieval_to_total_response_ratio=("phase_retrieval_to_total_response_ratio", "mean"),
            avg_phase_rerank_to_total_response_ratio=("phase_rerank_to_total_response_ratio", "mean"),
            avg_agent_execution_ratio=("agent_execution_ratio", "mean"),
            avg_agent_planning_ratio=("agent_planning_ratio", "mean"),
            avg_agent_tool_execution_ratio=("agent_tool_execution_ratio", "mean"),
            avg_agent_reasoning_ratio=("agent_reasoning_ratio", "mean"),
            avg_agent_output_ratio=("agent_output_ratio", "mean"),
            avg_num_before_rerank=("num_before_rerank", "mean"),
            avg_host_memory_traffic_mb=("avg_host_memory_traffic_mb", "mean"),
        )
        .reset_index()
        .sort_values(group_columns)
    )
    return summary


def _conclusion_lines(clean: pd.DataFrame) -> List[str]:
    lines: List[str] = []
    if clean.empty:
        return ["No clean profiling rows were available."]

    baseline = clean[clean["baseline_mode"] == "full_selection"].copy()
    if baseline.empty:
        baseline = clean.copy()

    max_memory = int(baseline["memory_size"].max())
    at_max = baseline[baseline["memory_size"] == max_memory]
    rerank_share = at_max["rerank_ratio"].mean() * 100.0
    lines.append(
        f"At {max_memory:,} memories, rerank accounts for {rerank_share:.1f}% of memory selection latency on average."
    )

    fact = baseline[baseline["query_type"] == "fact"]["num_before_rerank"].mean()
    relation = baseline[baseline["query_type"] == "relation"]["num_before_rerank"].mean()
    if fact > 0:
        lines.append(
            f"Relation-heavy queries expand candidates by {relation / fact:.2f}x compared with fact queries before reranking."
        )

    traffic = (
        baseline.groupby("memory_size", dropna=False)["estimated_host_memory_traffic_bytes"]
        .mean()
        .sort_index()
    )
    if len(traffic) >= 2:
        first_size = int(traffic.index[0])
        last_size = int(traffic.index[-1])
        first_mb = traffic.iloc[0] / (1024.0 * 1024.0)
        last_mb = traffic.iloc[-1] / (1024.0 * 1024.0)
        lines.append(
            f"Host-memory traffic grows from {first_mb:.2f} MB at {first_size:,} memories to {last_mb:.2f} MB at {last_size:,} memories."
        )

    retrieval_only = clean[clean["baseline_mode"] == "retrieval_only"]["selection_total_latency_ms"].mean()
    full_selection = clean[clean["baseline_mode"] == "full_selection"]["selection_total_latency_ms"].mean()
    if retrieval_only > 0 and full_selection > 0:
        lines.append(
            "These results suggest that reducing candidates before reranking is more important than only optimizing coarse retrieval."
        )

    if bool(clean["generation_enabled"].astype(bool).any()):
        lines.append("Generation was enabled for at least one run; selection-only metrics remain the primary comparison target.")
    elif "agent_execution_enabled" in clean.columns and bool(clean["agent_execution_enabled"].astype(bool).any()):
        agent_latency = baseline["agent_execution_total_latency_ms"].mean()
        total_latency = baseline["total_response_latency_ms"].mean()
        selection_share = baseline["selection_to_total_response_ratio"].mean() * 100.0
        phase_selection_share = baseline["phase_selection_to_total_response_ratio"].mean() * 100.0
        phase_retrieval_share = baseline["phase_retrieval_to_total_response_ratio"].mean() * 100.0
        phase_rerank_share = baseline["phase_rerank_to_total_response_ratio"].mean() * 100.0
        selection_invocations = baseline["selection_invocation_count"].mean()
        memory_mode = (
            baseline["agent_memory_mode"].mode().iloc[0]
            if "agent_memory_mode" in baseline.columns and not baseline["agent_memory_mode"].mode().empty
            else "shared_once"
        )
        planning_latency = baseline["agent_planning_latency_ms"].mean()
        tool_latency = baseline["agent_tool_execution_latency_ms"].mean()
        reasoning_latency = baseline["agent_reasoning_latency_ms"].mean()
        output_latency = baseline["agent_output_latency_ms"].mean()
        lines.append(
            f"Agent execution was enabled; downstream agent latency averages {agent_latency:.2f} ms and total response latency averages {total_latency:.2f} ms."
        )
        lines.append(
            "Agent execution decomposes into "
            f"planning {planning_latency:.2f} ms, tool execution {tool_latency:.2f} ms, "
            f"reasoning {reasoning_latency:.2f} ms, and output {output_latency:.2f} ms on average."
        )
        lines.append(
            f"Memory selection accounts for {selection_share:.1f}% of total response latency on average."
        )
        lines.append(
            f"Agent memory mode is {memory_mode}; effective selection calls average {selection_invocations:.1f}x per task, and phase-level selection accounts for {phase_selection_share:.1f}% of total response latency."
        )
        lines.append(
            f"Within that phase-aware share, retrieval-like stages account for {phase_retrieval_share:.1f}% and rerank accounts for {phase_rerank_share:.1f}% of total response latency."
        )
    else:
        lines.append("End-to-end latency excludes downstream generation in this run, as intended for the motivation study.")
    return lines


def write_summary_files(raw_df: pd.DataFrame, summary_csv: Path, summary_md: Path) -> pd.DataFrame:
    summary = build_summary(raw_df)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv, index=False)

    clean = raw_df[raw_df["error"].fillna("") == ""].copy()
    lines = ["# Motivation Memory Selection Summary", ""]
    if summary.empty:
        lines.append("No clean rows were available.")
    else:
        try:
            lines.append(summary.to_markdown(index=False))
        except ImportError:
            lines.append("`tabulate` is unavailable; falling back to plain-text table output.")
            lines.append("")
            lines.append("```text")
            lines.append(summary.to_string(index=False))
            lines.append("```")
        lines.append("")
        lines.append("## Conclusions")
        lines.append("")
        for item in _conclusion_lines(clean):
            lines.append(f"- {item}")
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
