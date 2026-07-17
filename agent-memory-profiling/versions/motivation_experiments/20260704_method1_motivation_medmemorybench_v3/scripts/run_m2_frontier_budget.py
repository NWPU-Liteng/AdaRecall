#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

from medmmb_common import (
    build_answer_prompt,
    build_embedding_client,
    build_llm_client,
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
    simple_answer_normalize,
    summarize_numeric,
    timed_call,
    validate_no_future_or_cross_persona,
    write_csv,
    write_jsonl,
    write_manifest,
    write_progress,
)


def _load_mapping(path: str) -> dict[str, set[str]]:
    return {str(row["query_id"]): {str(mid) for mid in row.get("gold_memory_ids", [])} for row in read_jsonl(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run M2 flat frontier budget study.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--mode", choices=["smoke", "pilot", "formal"], required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)
    model_config = load_yaml(config["model_config"])
    retrieval_config = load_yaml(config["retrieval_config"])
    seed = int(config.get("seed", 20260704))
    set_global_determinism(seed)
    require_cuda_if({**model_config, **retrieval_config, **config}, formal=args.mode in {"pilot", "formal"})

    memory_items = read_jsonl(config["memory_items"])
    all_queries = {str(row["query_id"]): row for row in read_jsonl(config["queries"])}
    queries = [all_queries[qid] for qid in read_split(args.split, args.mode) if qid in all_queries]
    if not queries:
        raise RuntimeError(f"No queries selected for mode={args.mode}")
    evidence_by_query = _load_mapping(config["evidence_mapping"])
    memories_by_persona = group_memories_by_persona(memory_items)
    vectors, vector_ids, precompute_ms = ensure_embedding_cache(memory_items, retrieval_config, "m2_turn_pair")
    index_by_id = {memory_id: idx for idx, memory_id in enumerate(vector_ids)}
    embed_client = build_embedding_client(retrieval_config)
    reranker_client = build_reranker_client(retrieval_config)
    answer_client = None if bool(config.get("selection_only", False)) else build_llm_client(model_config, "answer_max_new_tokens")
    final_top_k = int(config.get("final_top_k", 5))
    budgets = [int(value) for value in config.get("frontier_budgets", [8, 16, 32, 64])]
    full_budget = int(config.get("full_frontier_budget", max(budgets)))
    run_id = f"M2_{args.mode}_{seed}"
    write_manifest(
        f"manifests/{run_id}_manifest.json",
        environment_manifest(
            {
                "run_id": run_id,
                "mode": args.mode,
                "experiment": "M2",
                "frontier_budgets": budgets,
                "final_top_k": final_top_k,
                "precompute_embedding_ms": precompute_ms,
            }
        ),
    )

    rows = []
    oracle_by_query: dict[str, set[str]] = {}
    deferred_rows = []
    total_units = len(queries) * len(budgets)
    write_progress(
        experiment="M2",
        mode=args.mode,
        total_queries=len(queries),
        completed_queries=0,
        total_units=total_units,
        completed_units=0,
    )
    completed_units = 0
    completed_queries: set[str] = set()
    for query in queries:
        allow_current = bool(config.get("allow_current_session_context", False))
        eligible = eligible_memories_for_query(memories_by_persona, query, allow_current_session=allow_current, include_noise=True)
        validate_no_future_or_cross_persona(query, eligible, allow_current_session=allow_current)
        if not eligible:
            continue
        candidate_indices = [index_by_id[str(row["memory_id"])] for row in eligible if str(row["memory_id"]) in index_by_id]
        candidate_vectors = vectors[candidate_indices] if candidate_indices else vectors[:0]
        gold = evidence_by_query.get(str(query["query_id"]), set())
        for budget in budgets:
            if len(eligible) < budget:
                if args.mode == "formal":
                    continue
            result = run_flat_selection(
                query_text=str(query["query_text"]),
                candidate_memories=eligible,
                candidate_vectors=candidate_vectors,
                retrieval_config=retrieval_config,
                embed_client=embed_client,
                reranker_client=reranker_client,
                frontier_budget=budget,
                final_top_k=final_top_k,
            )
            if budget == full_budget:
                oracle_by_query[str(query["query_id"])] = set(result["final_selected_ids"])
            if answer_client is not None:
                prompt = build_answer_prompt(query, result["selected_memories"])
                answer_response, answer_latency_ms = timed_call(answer_client.generate_with_stats, prompt)
                answer_raw = str(answer_response.get("text", ""))
                answer_normalized = simple_answer_normalize(answer_raw)
                prompt_tokens = int(answer_response.get("prompt_tokens", 0))
                completion_tokens = int(answer_response.get("completion_tokens", 0))
            else:
                answer_raw = ""
                answer_normalized = ""
                answer_latency_ms = 0.0
                prompt_tokens = 0
                completion_tokens = 0
            metrics = evidence_metrics(result["final_selected_ids"], gold)
            row = {
                "run_id": run_id,
                "mode": args.mode,
                "query_id": str(query["query_id"]),
                "persona_id": query["persona_id"],
                "query_type": query.get("query_type", "unknown"),
                "query_session_id": query["query_session_id"],
                "history_scale": "full",
                "n_memory_total": len(eligible),
                "frontier_budget": budget,
                "final_top_k": final_top_k,
                "frontier_fill_ratio": result["n_reranked"] / float(budget),
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
                "answer_raw": answer_raw,
                "answer_normalized": answer_normalized,
                "answer_prompt_tokens": prompt_tokens,
                "answer_completion_tokens": completion_tokens,
                "answer_generation_latency_ms": answer_latency_ms,
                "reranker_oracle_overlap_at_5": "",
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "gpu_model": ",".join(environment_manifest().get("devices", [])),
                "error": "",
            }
            deferred_rows.append(row)
            completed_units += 1
            query_budget_count = sum(1 for existing in deferred_rows if existing.get("query_id") == str(query["query_id"]))
            if query_budget_count >= len(budgets):
                completed_queries.add(str(query["query_id"]))
            if completed_units % 8 == 0:
                write_progress(
                    experiment="M2",
                    mode=args.mode,
                    total_queries=len(queries),
                    completed_queries=len(completed_queries),
                    total_units=total_units,
                    completed_units=completed_units,
                    current_query_id=str(query["query_id"]),
                    extra={"frontier_budget": budget},
                )

    for row in deferred_rows:
        oracle = oracle_by_query.get(row["query_id"], set())
        selected = set(row["final_selected_ids"])
        row["reranker_oracle_overlap_at_5"] = len(selected & oracle) / float(final_top_k) if oracle else ""
        rows.append(row)

    if not rows:
        raise RuntimeError("M2 produced no invocation rows.")
    write_jsonl(config["output_invocations"], rows)
    write_progress(
        experiment="M2",
        mode=args.mode,
        total_queries=len(queries),
        completed_queries=len({row["query_id"] for row in rows}),
        total_units=total_units,
        completed_units=len(rows),
        extra={"status": "complete"},
    )
    occupancy = [
        {
            "mode": row["mode"],
            "query_id": row["query_id"],
            "query_type": row["query_type"],
            "frontier_budget": row["frontier_budget"],
            "n_memory_total": row["n_memory_total"],
            "n_reranked": row["n_reranked"],
            "frontier_fill_ratio": row["frontier_fill_ratio"],
            "n_final_selected": row["n_final_selected"],
            "selection_fill_ratio": row["selection_fill_ratio"],
        }
        for row in rows
    ]
    write_csv(config["output_occupancy"], occupancy)
    summary = summarize_numeric(
        rows,
        ["mode", "query_type", "frontier_budget"],
        [
            "t_total_selection_ms",
            "t_rerank_ms",
            "host_memory_bytes",
            "evidence_recall_at_k",
            "evidence_completion_at_k",
            "reranker_oracle_overlap_at_5",
            "answer_generation_latency_ms",
            "frontier_fill_ratio",
        ],
        seed=seed,
    )
    write_csv(config["output_summary"], summary)
    lines = ["# M2 Frontier Occupancy Audit", ""]
    for budget in budgets:
        vals = [float(row["frontier_fill_ratio"]) for row in rows if int(row["frontier_budget"]) == budget]
        if vals:
            vals_sorted = sorted(vals)
            median = vals_sorted[len(vals_sorted) // 2]
            lines.append(f"- budget {budget}: median FrontierFillRatio={median:.4f}, n={len(vals)}")
            if args.mode == "formal" and median < float(config.get("minimum_median_frontier_fill_ratio", 0.9)):
                raise RuntimeError(f"M2 budget {budget} under-filled: median FrontierFillRatio={median:.4f}")
    resolve_path("reports/M2_frontier_occupancy_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"m2_invocations={resolve_path(config['output_invocations'])}")
    print(f"m2_summary={resolve_path(config['output_summary'])}")


if __name__ == "__main__":
    main()
