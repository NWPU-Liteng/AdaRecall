#!/usr/bin/env python3
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUITE_DIR = PROJECT_ROOT / "experiments" / "motivation_memory_selection" / "configs" / "agent_latency_suite_20260615"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "agent_latency_suite_20260615"


def _load_manifest() -> list[dict]:
    return json.loads((SUITE_DIR / "manifest.json").read_text(encoding="utf-8"))


def _read_run(path: Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def _aggregate(df: pd.DataFrame, agent_variant: str, dataset: str, mode: str) -> dict:
    clean = df[df["error"].fillna("") == ""].copy()
    if clean.empty:
        return {
            "agent_variant": agent_variant,
            "dataset": dataset,
            "mode": mode,
            "rows": 0,
        }

    if "selection_invocation_count" not in clean.columns:
        if "agent_memory_mode" in clean.columns:
            clean["selection_invocation_count"] = clean["agent_memory_mode"].astype(str).map(
                {"phase_aware": 3, "shared_once": 1}
            ).fillna(1)
        else:
            clean["selection_invocation_count"] = 1
    if "phase_selection_total_latency_ms" not in clean.columns:
        clean["phase_selection_total_latency_ms"] = clean["selection_total_latency_ms"]
    retrieval_component = clean[
        [
            "query_encoding_latency_ms",
            "coarse_retrieval_latency_ms",
            "metadata_filter_latency_ms",
            "relation_expansion_latency_ms",
            "candidate_aggregation_latency_ms",
            "context_construction_latency_ms",
        ]
    ].sum(axis=1)
    if "phase_retrieval_total_latency_ms" not in clean.columns:
        clean["phase_retrieval_total_latency_ms"] = retrieval_component
    if "phase_rerank_total_latency_ms" not in clean.columns:
        clean["phase_rerank_total_latency_ms"] = clean["rerank_latency_ms"]
    if "phase_selection_to_total_response_ratio" not in clean.columns:
        clean["phase_selection_to_total_response_ratio"] = (
            clean["phase_selection_total_latency_ms"] / clean["total_response_latency_ms"].clip(lower=1e-9)
        )
    if "phase_retrieval_to_total_response_ratio" not in clean.columns:
        clean["phase_retrieval_to_total_response_ratio"] = (
            clean["phase_retrieval_total_latency_ms"] / clean["total_response_latency_ms"].clip(lower=1e-9)
        )
    if "phase_rerank_to_total_response_ratio" not in clean.columns:
        clean["phase_rerank_to_total_response_ratio"] = (
            clean["phase_rerank_total_latency_ms"] / clean["total_response_latency_ms"].clip(lower=1e-9)
        )

    return {
        "agent_variant": agent_variant,
        "dataset": dataset,
        "mode": mode,
        "rows": int(len(clean)),
        "avg_retrieval_ms": float(
            clean[
                [
                    "query_encoding_latency_ms",
                    "coarse_retrieval_latency_ms",
                    "metadata_filter_latency_ms",
                    "relation_expansion_latency_ms",
                    "candidate_aggregation_latency_ms",
                    "context_construction_latency_ms",
                ]
            ].sum(axis=1).mean()
        ),
        "avg_rerank_ms": float(clean["rerank_latency_ms"].mean()),
        "avg_selection_ms": float(clean["selection_total_latency_ms"].mean()),
        "avg_selection_invocations": float(clean["selection_invocation_count"].mean()),
        "avg_phase_selection_ms": float(clean["phase_selection_total_latency_ms"].mean()),
        "avg_phase_retrieval_ms": float(clean["phase_retrieval_total_latency_ms"].mean()),
        "avg_phase_rerank_ms": float(clean["phase_rerank_total_latency_ms"].mean()),
        "avg_agent_planning_ms": float(clean["agent_planning_latency_ms"].mean()),
        "avg_agent_tool_ms": float(clean["agent_tool_execution_latency_ms"].mean()),
        "avg_agent_reasoning_ms": float(clean["agent_reasoning_latency_ms"].mean()),
        "avg_agent_output_ms": float(clean["agent_output_latency_ms"].mean()),
        "avg_agent_ms": float(clean["agent_execution_total_latency_ms"].mean()),
        "avg_total_ms": float(clean["total_response_latency_ms"].mean()),
        "retrieval_share_of_total": float(
            (
                clean[
                    [
                        "query_encoding_latency_ms",
                        "coarse_retrieval_latency_ms",
                        "metadata_filter_latency_ms",
                        "relation_expansion_latency_ms",
                        "candidate_aggregation_latency_ms",
                        "context_construction_latency_ms",
                    ]
                ].sum(axis=1)
                / clean["total_response_latency_ms"].clip(lower=1e-9)
            ).mean()
        ),
        "rerank_share_of_total": float(
            (clean["rerank_latency_ms"] / clean["total_response_latency_ms"].clip(lower=1e-9)).mean()
        ),
        "selection_share_of_total": float(clean["selection_to_total_response_ratio"].mean()),
        "phase_selection_share_of_total": float(clean["phase_selection_to_total_response_ratio"].mean()),
        "phase_retrieval_share_of_total": float(clean["phase_retrieval_to_total_response_ratio"].mean()),
        "phase_rerank_share_of_total": float(clean["phase_rerank_to_total_response_ratio"].mean()),
        "agent_share_of_total": float(clean["agent_execution_ratio"].mean()),
        "planning_share_of_agent": float(
            (clean["agent_planning_latency_ms"] / clean["agent_execution_total_latency_ms"].clip(lower=1e-9)).mean()
        ),
        "tool_share_of_agent": float(
            (clean["agent_tool_execution_latency_ms"] / clean["agent_execution_total_latency_ms"].clip(lower=1e-9)).mean()
        ),
        "reasoning_share_of_agent": float(
            (clean["agent_reasoning_latency_ms"] / clean["agent_execution_total_latency_ms"].clip(lower=1e-9)).mean()
        ),
        "output_share_of_agent": float(
            (clean["agent_output_latency_ms"] / clean["agent_execution_total_latency_ms"].clip(lower=1e-9)).mean()
        ),
    }


def _write_markdown(summary: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "# Agent Latency Suite Summary",
        "",
        "## Main Table",
        "",
        summary.to_markdown(index=False),
        "",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    manifest = _load_manifest()
    rows = []
    for item in manifest:
        raw_path = PROJECT_ROOT / item["output_dir"] / "raw_profile.jsonl"
        if not raw_path.exists():
            continue
        df = _read_run(raw_path)
        rows.append(_aggregate(df, item["agent_variant"], item["dataset"], item["mode"]))

    summary = pd.DataFrame(rows).sort_values(["agent_variant", "dataset", "mode"])
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_ROOT / "suite_summary.csv"
    md_path = OUTPUT_ROOT / "suite_summary.md"
    summary.to_csv(csv_path, index=False)
    _write_markdown(summary, md_path)
    print(f"suite_summary_csv={csv_path}")
    print(f"suite_summary_md={md_path}")


if __name__ == "__main__":
    main()
