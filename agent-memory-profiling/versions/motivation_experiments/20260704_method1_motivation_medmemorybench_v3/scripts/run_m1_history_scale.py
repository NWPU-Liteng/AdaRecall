#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

from medmmb_common import (
    VERSION_ROOT,
    build_embedding_client,
    build_nested_history,
    build_reranker_client,
    eligible_memories_for_query,
    ensure_embedding_cache,
    environment_manifest,
    evidence_metrics,
    group_memories_by_persona,
    load_yaml,
    read_jsonl,
    read_split,
    require_cuda_if,
    resolve_path,
    run_flat_selection,
    set_global_determinism,
    summarize_numeric,
    validate_no_future_or_cross_persona,
    write_csv,
    write_jsonl,
    write_manifest,
    write_progress,
)


def _load_mapping(path: str) -> dict[str, set[str]]:
    rows = read_jsonl(path)
    return {str(row["query_id"]): {str(mid) for mid in row.get("gold_memory_ids", [])} for row in rows}


def _feasible_scales(config: dict, queries: list[dict], memories_by_persona: dict, include_noise: bool) -> list[int | str]:
    counts = [
        len(eligible_memories_for_query(memories_by_persona, query, allow_current_session=bool(config.get("allow_current_session_context", False)), include_noise=include_noise))
        for query in queries
    ]
    counts = [count for count in counts if count > 0]
    if not counts:
        return []
    output: list[int | str] = []
    for scale in config.get("scale_grid", []):
        if str(scale) == "full":
            output.append("full")
        else:
            eligible_count = sum(1 for count in counts if count >= int(scale))
            if eligible_count >= int(config.get("min_formal_queries_per_scale", 80)) or len(queries) < int(config.get("min_formal_queries_per_scale", 80)):
                output.append(int(scale))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run M1 anchor-preserving nested-history profiling.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--mode", choices=["smoke", "pilot", "formal"], required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)
    retrieval_config = load_yaml(config["retrieval_config"])
    seed = int(config.get("seed", 20260704))
    set_global_determinism(seed)
    require_cuda_if({**retrieval_config, **config}, formal=args.mode in {"pilot", "formal"})

    memory_items = read_jsonl(config["memory_items"])
    all_queries = {str(row["query_id"]): row for row in read_jsonl(config["queries"])}
    query_ids = read_split(args.split, args.mode)
    queries = [all_queries[qid] for qid in query_ids if qid in all_queries]
    if not queries:
        raise RuntimeError(f"No queries selected for mode={args.mode}")
    evidence_by_query = _load_mapping(config["evidence_mapping"])
    memories_by_persona = group_memories_by_persona(memory_items)

    vectors, vector_ids, precompute_ms = ensure_embedding_cache(memory_items, retrieval_config, "m1_turn_pair")
    index_by_id = {memory_id: idx for idx, memory_id in enumerate(vector_ids)}
    embed_client = build_embedding_client(retrieval_config)
    reranker_client = build_reranker_client(retrieval_config)
    frontier_budget = int(config.get("frontier_budget", 32))
    final_top_k = int(config.get("final_top_k", 5))

    run_id = f"M1_{args.mode}_{seed}"
    write_manifest(
        f"manifests/{run_id}_manifest.json",
        environment_manifest(
            {
                "run_id": run_id,
                "mode": args.mode,
                "experiment": "M1",
                "frontier_budget": frontier_budget,
                "final_top_k": final_top_k,
                "precompute_embedding_ms": precompute_ms,
                "cuda_visible_devices_required": True,
            }
        ),
    )

    rows = []
    total_units_estimate = 0
    feasible_by_condition: dict[str, list[int | str]] = {}
    for condition in config.get("history_conditions", ["medical_only"]):
        include_noise = condition == "medical_plus_official_noise"
        feasible_by_condition[condition] = _feasible_scales(config, queries, memories_by_persona, include_noise)
        total_units_estimate += len(feasible_by_condition[condition]) * len(queries)
    write_progress(
        experiment="M1",
        mode=args.mode,
        total_queries=len(queries),
        completed_queries=0,
        total_units=total_units_estimate,
        completed_units=0,
    )
    completed_query_keys: set[str] = set()
    for condition in config.get("history_conditions", ["medical_only"]):
        include_noise = condition == "medical_plus_official_noise"
        scales = feasible_by_condition[condition]
        if args.mode == "formal" and len(scales) < 4:
            raise RuntimeError(f"M1 requires at least four nontrivial scales; got {scales}")
        for query in queries:
            allow_current = bool(config.get("allow_current_session_context", False))
            eligible = eligible_memories_for_query(memories_by_persona, query, allow_current_session=allow_current, include_noise=include_noise)
            validate_no_future_or_cross_persona(query, eligible, allow_current_session=allow_current)
            gold = evidence_by_query.get(str(query["query_id"]), set())
            for scale in scales:
                if scale != "full" and len(eligible) < int(scale):
                    continue
                if scale != "full" and len(gold) > int(scale):
                    continue
                history = build_nested_history(query=query, eligible=eligible, gold_memory_ids=gold, scale=scale, seed=seed)
                candidate_indices = [index_by_id[str(row["memory_id"])] for row in history if str(row["memory_id"]) in index_by_id]
                candidate_vectors = vectors[candidate_indices] if candidate_indices else vectors[:0]
                result = run_flat_selection(
                    query_text=str(query["query_text"]),
                    candidate_memories=history,
                    candidate_vectors=candidate_vectors,
                    retrieval_config=retrieval_config,
                    embed_client=embed_client,
                    reranker_client=reranker_client,
                    frontier_budget=frontier_budget,
                    final_top_k=final_top_k,
                )
                metrics = evidence_metrics(result["final_selected_ids"], gold)
                rows.append(
                    {
                        "run_id": run_id,
                        "mode": args.mode,
                        "history_condition": condition,
                        "query_id": str(query["query_id"]),
                        "persona_id": query["persona_id"],
                        "query_type": query.get("query_type", "unknown"),
                        "query_session_id": query["query_session_id"],
                        "history_scale": scale,
                        "n_memory_total": len(history),
                        "frontier_budget": frontier_budget,
                        "final_top_k": final_top_k,
                        "frontier_fill_ratio": result["n_reranked"] / float(frontier_budget),
                        "selection_fill_ratio": result["n_final_selected"] / float(final_top_k),
                        **{key: result[key] for key in [
                            "n_ann_raw",
                            "n_after_filter",
                            "n_after_dedup",
                            "n_materialized",
                            "n_reranked",
                            "n_final_selected",
                            "t_query_encode_ms",
                            "t_ann_ms",
                            "t_filter_ms",
                            "t_materialize_ms",
                            "t_rerank_ms",
                            "t_total_selection_ms",
                            "index_bytes_read",
                            "metadata_bytes_read",
                            "payload_bytes_read",
                            "host_memory_bytes",
                        ]},
                        "ann_candidate_ids": result["ann_candidate_ids"],
                        "reranked_candidate_ids": result["reranked_candidate_ids"],
                        "final_selected_ids": result["final_selected_ids"],
                        **metrics,
                        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                        "gpu_model": ",".join(environment_manifest().get("devices", [])),
                        "error": "",
                    }
                )
                completed_query_keys.add(str(query["query_id"]))
                if len(rows) % 25 == 0:
                    write_progress(
                        experiment="M1",
                        mode=args.mode,
                        total_queries=len(queries),
                        completed_queries=len(completed_query_keys),
                        total_units=total_units_estimate,
                        completed_units=len(rows),
                        current_query_id=str(query["query_id"]),
                        extra={"history_condition": condition, "history_scale": scale},
                    )

    if not rows:
        raise RuntimeError("M1 produced no invocation rows.")
    write_jsonl(config["output_invocations"], rows)
    write_progress(
        experiment="M1",
        mode=args.mode,
        total_queries=len(queries),
        completed_queries=len(completed_query_keys),
        total_units=total_units_estimate,
        completed_units=len(rows),
        extra={"status": "complete"},
    )
    summary = summarize_numeric(
        rows,
        ["mode", "history_condition", "history_scale"],
        ["t_total_selection_ms", "t_ann_ms", "t_rerank_ms", "host_memory_bytes", "frontier_fill_ratio"],
        seed=seed,
    )
    write_csv(config["output_summary"], summary)
    occupancy_rows = [
        {
            "mode": row["mode"],
            "history_condition": row["history_condition"],
            "history_scale": row["history_scale"],
            "query_id": row["query_id"],
            "frontier_budget": row["frontier_budget"],
            "n_reranked": row["n_reranked"],
            "frontier_fill_ratio": row["frontier_fill_ratio"],
            "n_final_selected": row["n_final_selected"],
            "selection_fill_ratio": row["selection_fill_ratio"],
        }
        for row in rows
    ]
    write_csv("tables/Table_M1_frontier_occupancy.csv", occupancy_rows)
    print(f"m1_invocations={resolve_path(config['output_invocations'])}")
    print(f"m1_summary={resolve_path(config['output_summary'])}")


if __name__ == "__main__":
    main()
