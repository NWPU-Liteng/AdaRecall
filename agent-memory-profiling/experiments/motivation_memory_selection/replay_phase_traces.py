#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd

from synthetic_memory_generator import QuerySpec, read_jsonl
from trace_logger import append_jsonl, ensure_output_layout
from metrics import write_summary_files
from plot_results import make_plots
from profile_longgraph_memory_selection import (
    PROJECT_ROOT,
    _ensure_index,
    _load_config,
    _load_memories_for_setting,
    _resolve_embed_client,
    _resolve_reranker_client,
    _warmup_store,
    run_selection_profile_pass,
)


def _phase_row_to_query(row: dict) -> QuerySpec:
    return QuerySpec(
        query_id=str(row["query_id"]),
        query_type=str(row["query_type"]),
        text=str(row["phase_query_text"]),
        task_tags=list(row.get("task_tags", [])),
        entity_tags=list(row.get("entity_tags", [])),
        preferred_memory_types=list(row.get("preferred_memory_types", [])),
        temporal_window_days=int(row.get("temporal_window_days", 0)),
        gold_group_ids=list(row.get("gold_group_ids", [])),
    )


def _iter_replay_settings(config: dict) -> List[tuple[str, int, int, object]]:
    baselines = list(
        config.get(
            "replay_baseline_modes",
            ["retrieval_only", "retrieval_plus_expansion", "full_selection", "oracle_pruned_rerank"],
        )
    )
    retrieval_values = list(config.get("replay_retrieval_topk_values", config.get("retrieval_topk_values", [50])))
    depth_values = list(config.get("replay_relation_expansion_depth_values", config.get("relation_expansion_depth_values", [0, 1])))
    rerank_inputs = list(config.get("replay_rerank_topk_input_values", config.get("rerank_topk_input_values", ["all"])))

    settings: List[tuple[str, int, int, object]] = []
    for baseline_mode in baselines:
        for retrieval_topk in retrieval_values:
            if baseline_mode == "retrieval_only":
                settings.append((baseline_mode, int(retrieval_topk), 0, "all"))
                continue
            for depth in depth_values:
                if baseline_mode == "retrieval_plus_expansion":
                    settings.append((baseline_mode, int(retrieval_topk), int(depth), "all"))
                    continue
                for rerank_input in rerank_inputs:
                    settings.append((baseline_mode, int(retrieval_topk), int(depth), rerank_input))
    return settings


def run_replay(config_path: Path, phase_trace_path: Path, output_dir: Path | None = None) -> dict:
    config = _load_config(config_path)
    phase_rows = read_jsonl(phase_trace_path)
    if not phase_rows:
        raise ValueError(f"No phase trace rows found in {phase_trace_path}")

    replay_output_dir = output_dir or (PROJECT_ROOT / config.get("output_dir", "outputs/motivation_memory_selection_replay"))
    paths = ensure_output_layout(replay_output_dir)
    raw_profile_path = paths["raw_profile"]
    if raw_profile_path.exists():
        raw_profile_path.unlink()

    embed_client = _resolve_embed_client(config)
    reranker_client = _resolve_reranker_client(config)

    all_rows = []
    memory_cache = {}
    store_cache = {}

    for row in phase_rows:
        memory_size = int(row["memory_size"])
        if memory_size not in memory_cache:
            memories, _ = _load_memories_for_setting(memory_size, config, seed=int(config.get("seed", 42)))
            memory_cache[memory_size] = memories
            store, _ = _ensure_index(memories, memory_size, config)
            _warmup_store(store, embed_client)
            store_cache[memory_size] = store
        memories = memory_cache[memory_size]
        store = store_cache[memory_size]
        query = _phase_row_to_query(row)
        trace_id = str(row["trace_id"])
        phase_name = str(row["phase_name"])

        replay_rows = []
        for baseline_mode, retrieval_topk, depth, rerank_input in _iter_replay_settings(config):
            result = run_selection_profile_pass(
                query=query,
                memories=memories,
                store=store,
                embed_client=embed_client,
                reranker_client=reranker_client,
                config=config,
                baseline_mode=baseline_mode,
                retrieval_topk=retrieval_topk,
                relation_expansion_depth=depth,
                rerank_topk_input=rerank_input,
            )
            replay_rows.append(
                {
                    "trace_id": trace_id,
                    "query_id": query.query_id,
                    "query_type": query.query_type,
                    "phase_name": phase_name,
                    "memory_size": memory_size,
                    "baseline_mode": baseline_mode,
                    "retrieval_topk": retrieval_topk,
                    "relation_expansion_depth": depth,
                    "rerank_topk_input": rerank_input,
                    "num_raw_retrieved": result["counts"]["num_raw_retrieved"],
                    "num_after_metadata_filter": result["counts"]["num_after_metadata_filter"],
                    "num_after_relation_expansion": result["counts"]["num_after_relation_expansion"],
                    "num_after_dedup": result["counts"]["num_after_dedup"],
                    "num_before_rerank": result["counts"]["num_before_rerank"],
                    "num_after_rerank": result["counts"]["num_after_rerank"],
                    "query_encoding_latency_ms": result["latencies_ms"]["query_encoding_latency_ms"],
                    "coarse_retrieval_latency_ms": result["latencies_ms"]["coarse_retrieval_latency_ms"],
                    "metadata_filter_latency_ms": result["latencies_ms"]["metadata_filter_latency_ms"],
                    "relation_expansion_latency_ms": result["latencies_ms"]["relation_expansion_latency_ms"],
                    "candidate_aggregation_latency_ms": result["latencies_ms"]["candidate_aggregation_latency_ms"],
                    "rerank_latency_ms": result["latencies_ms"]["rerank_latency_ms"],
                    "context_construction_latency_ms": result["latencies_ms"]["context_construction_latency_ms"],
                    "selection_total_latency_ms": result["selection_total_latency_ms"],
                    "rerank_ratio": result["latencies_ms"]["rerank_latency_ms"] / max(result["selection_total_latency_ms"], 1e-9),
                    "selected_memory_ids": [memory["memory_id"] for memory, _ in result["selected"]],
                    "phase_prompt_chars": len(str(result["prompt"])),
                    "replay_source": "phase_trace",
                    "error": "",
                }
            )
        append_jsonl(raw_profile_path, replay_rows)
        all_rows.extend(replay_rows)

    raw_df = pd.DataFrame(all_rows)
    write_summary_files(raw_df, paths["summary_csv"], paths["summary_md"])
    make_plots(raw_profile_path, paths["figures_dir"])
    (paths["logs_dir"] / "replay_config.json").write_text(
        json.dumps(
            {
                "config_path": str(config_path),
                "phase_trace_path": str(phase_trace_path),
                "rows_total": int(len(raw_df)),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase-trace", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None
    paths = run_replay(Path(args.config), Path(args.phase_trace), output_dir=output_dir)
    print(f"raw_profile={paths['raw_profile']}")
    print(f"summary_csv={paths['summary_csv']}")
    print(f"summary_md={paths['summary_md']}")
    print(f"figures_dir={paths['figures_dir']}")


if __name__ == "__main__":
    main()
