#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os

from medmmb_common import (
    build_embedding_client,
    build_llm_client,
    build_reranker_client,
    eligible_memories_for_query,
    ensure_embedding_cache,
    environment_manifest,
    group_memories_by_persona,
    load_yaml,
    read_jsonl,
    read_split,
    require_cuda_if,
    resolve_path,
    run_flat_selection,
    set_global_determinism,
    timed_call,
    validate_no_future_or_cross_persona,
    write_jsonl,
    write_manifest,
    write_progress,
    utc_now,
)


def _intent_prompt(query: dict, stage: str, instruction: str, prefix: str, prior_summaries: list[str]) -> str:
    return (
        "You are a fixed finite-state memory controller for a text-only benchmark.\n"
        "The stage label is already assigned before retrieval. Generate retrieval intent only.\n"
        "Return exactly one JSON object and nothing else.\n"
        "Do not add explanations, markdown, comments, or text after the JSON object.\n"
        "Do not answer the user query. Do not mention gold answers or source key points.\n"
        "The retrieval_goal must begin with the provided stage prefix and must be stage-specific.\n"
        "Use these JSON keys: retrieval_goal, required_slots, temporal_focus, risk_focus.\n"
        "Use a short required_slots array with at most four strings.\n"
        "temporal_focus must be one of: recent, longitudinal, update_check, unknown.\n"
        "risk_focus must be one of: none, medication, allergy, contraindication, deterioration, unknown.\n\n"
        f"Stage: {stage}\n"
        f"Stage prefix: {prefix}\n"
        f"Stage instruction: {instruction}\n"
        f"Query type: {query.get('query_type', 'unknown')}\n"
        f"Current session: {query.get('query_session_id', '')}\n"
        f"User query: {query.get('query_text', '')}\n"
        f"Prior controller summaries: {' | '.join(prior_summaries[-3:]) if prior_summaries else 'none'}\n\n"
        "Return JSON only:"
    )


def _parse_intent(raw: str, fallback_goal: str) -> dict:
    decoder = json.JSONDecoder()
    json_span = ""
    payload = {}
    for start, char in enumerate(raw):
        if char != "{":
            continue
        try:
            candidate, end = decoder.raw_decode(raw[start:])
        except Exception:
            continue
        if isinstance(candidate, dict) and candidate.get("retrieval_goal"):
            payload = candidate
            json_span = raw[start:start + end]
            break
    goal = str(payload.get("retrieval_goal") or fallback_goal).strip()
    slots = payload.get("required_slots") or []
    if not isinstance(slots, list):
        slots = [str(slots)]
    return {
        "retrieval_goal": goal[:500],
        "required_slots": [str(item)[:120] for item in slots[:4]],
        "temporal_focus": str(payload.get("temporal_focus", "unknown")),
        "risk_focus": str(payload.get("risk_focus", "unknown")),
        "raw_intent": raw,
        "parsed_intent_json": json_span if payload else "",
        "intent_parse_ok": bool(payload),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run O1 fixed controller-defined stage trace.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--mode", choices=["smoke", "pilot", "formal"], required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)
    model_config = load_yaml(config["model_config"])
    retrieval_config = load_yaml(config["retrieval_config"])
    stage_config = load_yaml(config["stage_controller_config"])
    seed = int(config.get("seed", 20260704))
    set_global_determinism(seed)
    require_cuda_if({**model_config, **retrieval_config, **config}, formal=args.mode in {"pilot", "formal"})

    memory_items = read_jsonl(config["memory_items"])
    all_queries = {str(row["query_id"]): row for row in read_jsonl(config["queries"])}
    subset_name = config.get("formal_subset_name") if args.mode == "formal" else config.get("smoke_subset_name") if args.mode == "smoke" else args.mode
    queries = [all_queries[qid] for qid in read_split(args.split, args.mode, subset_name=subset_name) if qid in all_queries]
    if not queries:
        raise RuntimeError(f"No O1 queries selected for mode={args.mode}")

    memories_by_persona = group_memories_by_persona(memory_items)
    vectors, vector_ids, precompute_ms = ensure_embedding_cache(memory_items, retrieval_config, "o1_turn_pair")
    index_by_id = {memory_id: idx for idx, memory_id in enumerate(vector_ids)}
    embed_client = build_embedding_client(retrieval_config)
    reranker_client = build_reranker_client(retrieval_config)
    controller = build_llm_client(model_config, "controller_max_new_tokens")
    stages = list(stage_config.get("stages", ["ground", "plan", "reason", "reflect"]))
    if stages != ["ground", "plan", "reason", "reflect"]:
        raise RuntimeError("O1 v1 requires exactly ground, plan, reason, reflect.")
    frontier_budget = int(config.get("frontier_budget", 32))
    final_top_k = int(config.get("final_top_k", 5))
    run_id = f"O1_{args.mode}_{seed}"
    write_manifest(
        f"manifests/{run_id}_manifest.json",
        environment_manifest(
            {
                "run_id": run_id,
                "mode": args.mode,
                "experiment": "O1",
                "frontier_budget": frontier_budget,
                "final_top_k": final_top_k,
                "stage_label_source": "controller_fsm_v1",
                "precompute_embedding_ms": precompute_ms,
            }
        ),
    )

    rows = []
    total_units = len(queries) * len(stages)
    write_progress(
        experiment="O1",
        mode=args.mode,
        total_queries=len(queries),
        completed_queries=0,
        total_units=total_units,
        completed_units=0,
    )
    completed_units = 0
    completed_queries: set[str] = set()
    audit_lines = ["# Stage Controller Audit", "", "Stage label source: `controller_fsm_v1`", ""]
    sampled = 0
    for query in queries:
        allow_current = bool(config.get("allow_current_session_context", False))
        eligible = eligible_memories_for_query(memories_by_persona, query, allow_current_session=allow_current, include_noise=True)
        validate_no_future_or_cross_persona(query, eligible, allow_current_session=allow_current)
        if len(eligible) < frontier_budget and args.mode == "formal":
            continue
        candidate_indices = [index_by_id[str(row["memory_id"])] for row in eligible if str(row["memory_id"]) in index_by_id]
        candidate_vectors = vectors[candidate_indices] if candidate_indices else vectors[:0]
        prior_summaries: list[str] = []
        for stage_index, stage in enumerate(stages):
            instruction = str(stage_config.get("stage_instructions", {}).get(stage, stage))
            prefix = str(stage_config.get("stage_query_prefixes", {}).get(stage, f"{stage} evidence:"))
            prompt = _intent_prompt(query, stage, instruction, prefix, prior_summaries)
            response, t_intent_ms = timed_call(controller.generate_with_stats, prompt)
            intent = _parse_intent(str(response.get("text", "")), fallback_goal=str(query["query_text"]))
            if not intent["retrieval_goal"].lower().startswith(prefix.lower()):
                intent["retrieval_goal"] = f"{prefix} {intent['retrieval_goal']}"
            result = run_flat_selection(
                query_text=intent["retrieval_goal"],
                candidate_memories=eligible,
                candidate_vectors=candidate_vectors,
                retrieval_config=retrieval_config,
                embed_client=embed_client,
                reranker_client=reranker_client,
                frontier_budget=frontier_budget,
                final_top_k=final_top_k,
            )
            prior_summaries.append(f"{stage}: {intent['retrieval_goal']}")
            row = {
                "run_id": run_id,
                "query_id": str(query["query_id"]),
                "persona_id": query["persona_id"],
                "query_session_id": query["query_session_id"],
                "query_type": query.get("query_type", "unknown"),
                "stage": stage,
                "stage_index": stage_index,
                "stage_label_source": "controller_fsm_v1",
                "retrieval_goal": intent["retrieval_goal"],
                "required_slots": intent["required_slots"],
                "temporal_focus": intent["temporal_focus"],
                "risk_focus": intent["risk_focus"],
                "history_scale": "full",
                "frontier_budget": frontier_budget,
                "final_top_k": final_top_k,
                "ann_candidate_ids": result["ann_candidate_ids"],
                "reranked_candidate_ids": result["reranked_candidate_ids"],
                "final_selected_ids": result["final_selected_ids"],
                "n_reranked": result["n_reranked"],
                "n_final_selected": result["n_final_selected"],
                "t_intent_generation_ms": t_intent_ms,
                "t_retrieval_ms": result["t_query_encode_ms"] + result["t_ann_ms"],
                "t_rerank_ms": result["t_rerank_ms"],
                "t_total_selection_ms": result["t_total_selection_ms"],
                "timestamp": utc_now(),
                "controller_prompt": prompt if sampled < 20 else "",
                "raw_intent": intent["raw_intent"] if sampled < 20 else "",
                "parsed_intent_json": intent["parsed_intent_json"],
                "intent_parse_ok": intent["intent_parse_ok"],
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "error": "",
            }
            rows.append(row)
            completed_units += 1
            if stage_index == len(stages) - 1:
                completed_queries.add(str(query["query_id"]))
            if completed_units % 4 == 0:
                write_progress(
                    experiment="O1",
                    mode=args.mode,
                    total_queries=len(queries),
                    completed_queries=len(completed_queries),
                    total_units=total_units,
                    completed_units=completed_units,
                    current_query_id=str(query["query_id"]),
                    extra={"stage": stage},
                )
            if sampled < 20:
                audit_lines.extend(
                    [
                        f"## Trace Sample {sampled + 1}",
                        "",
                        f"- query_id: `{row['query_id']}`",
                        f"- stage: `{stage}`",
                        f"- prompt includes retrieved memory before intent: `false`",
                        f"- retrieval_goal: {intent['retrieval_goal']}",
                        f"- selected_memory_ids: `{','.join(result['final_selected_ids'])}`",
                        "",
                    ]
                )
                sampled += 1
    if not rows:
        raise RuntimeError("O1 produced no stage rows.")
    output = config.get("output_stage_trace", "aggregate/O1_formal_stage_trace.jsonl")
    if args.mode != "formal":
        output = f"aggregate/O1_{args.mode}_stage_trace.jsonl"
    write_jsonl(output, rows)
    write_progress(
        experiment="O1",
        mode=args.mode,
        total_queries=len(queries),
        completed_queries=len(completed_queries),
        total_units=total_units,
        completed_units=len(rows),
        extra={"status": "complete"},
    )
    audit_lines.extend(
        [
            "## Validity Checks",
            "",
            "- Stage is assigned by fixed FSM before retrieval.",
            "- Every stage uses the same retrieval budget.",
            "- Stage prompts contain query text/type/session and prior controller summaries only.",
            "- Stage prompts do not include gold answers, source key points, or retrieved memory text before intent generation.",
        ]
    )
    resolve_path("reports/03_stage_controller_audit.md").write_text("\n".join(audit_lines) + "\n", encoding="utf-8")
    print(f"o1_stage_trace={resolve_path(output)}")


if __name__ == "__main__":
    main()
