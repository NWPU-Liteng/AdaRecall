#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

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


ROOT = Path(__file__).resolve().parents[1]


def _intent_prompt(query: dict, stage: str, instruction: str, prefix: str, prior_summaries: list[str]) -> str:
    return (
        "You are a fixed finite-state memory controller for a text-only benchmark.\n"
        "The stage label is assigned before retrieval. Generate retrieval intent only.\n"
        "Return exactly one JSON object and nothing else.\n"
        "Do not answer the user query. Do not mention gold answers or source key points.\n"
        "The retrieval_goal must begin with the provided stage prefix and must be stage-specific.\n"
        "Use these JSON keys: retrieval_goal, required_slots, temporal_focus, risk_focus.\n"
        "Use at most four required_slots strings.\n\n"
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
    payload = {}
    json_span = ""
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
    slots = payload.get("required_slots") or []
    if not isinstance(slots, list):
        slots = [str(slots)]
    return {
        "retrieval_goal": str(payload.get("retrieval_goal") or fallback_goal).strip()[:500],
        "required_slots": [str(item)[:120] for item in slots[:4]],
        "temporal_focus": str(payload.get("temporal_focus", "unknown")),
        "risk_focus": str(payload.get("risk_focus", "unknown")),
        "raw_intent": raw,
        "parsed_intent_json": json_span,
        "intent_parse_ok": bool(payload),
    }


def _session_key(value) -> tuple[int, str]:
    text = str(value)
    try:
        return (0, f"{int(float(text)):012d}")
    except Exception:
        return (1, text)


def _query_order(query: dict) -> tuple:
    return (_session_key(query.get("query_session_id")), str(query.get("query_id")))


def _select_population(queries: list[dict], count: int) -> list[dict]:
    valid = [q for q in queries if q.get("query_session_id") is not None]
    by_qid = {str(q["query_id"]): q for q in valid}
    reused_ids = []
    v3_trace = ROOT.parent / "20260704_method1_motivation_medmemorybench_v3" / "aggregate" / "O1_formal_stage_trace.jsonl"
    if v3_trace.exists():
        for row in read_jsonl(v3_trace):
            qid = str(row.get("query_id", ""))
            if qid in by_qid and qid not in reused_ids:
                reused_ids.append(qid)
    selected_ids = []
    for qid in reused_ids:
        if qid not in selected_ids:
            selected_ids.append(qid)
    by_persona = defaultdict(list)
    for q in valid:
        by_persona[str(q.get("persona_id"))].append(q)
    for rows in by_persona.values():
        rows.sort(key=_query_order)
    # Round-robin through personas to add chronologically diverse queries not already in v3 O1.
    while len(selected_ids) < count:
        progressed = False
        for persona in sorted(by_persona):
            for q in by_persona[persona]:
                qid = str(q["query_id"])
                if qid not in selected_ids:
                    selected_ids.append(qid)
                    progressed = True
                    break
            if len(selected_ids) >= count:
                break
        if not progressed:
            break
    selected = [by_qid[qid] for qid in selected_ids[:count]]
    selected.sort(key=lambda q: (str(q.get("persona_id")), _query_order(q)))
    return selected


def _extract_turn(memory: dict) -> int | None:
    mid = str(memory.get("memory_id", ""))
    if "_x" in mid:
        try:
            return int(mid.rsplit("_x", 1)[1])
        except Exception:
            return None
    value = memory.get("exchange_id")
    try:
        return int(value)
    except Exception:
        return None


def _write_runtime_manifest(run_id: str, extra: dict) -> None:
    payload = environment_manifest({"run_id": run_id, **extra})
    try:
        payload["nvidia_smi"] = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT, text=True)
    except Exception as exc:
        payload["nvidia_smi"] = f"unavailable: {exc}"
    write_manifest("manifests/runtime_manifest.json", payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run B O1-v4 stage trace population.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["smoke", "formal"], required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)
    model_config = load_yaml(config["model_config"])
    retrieval_config = load_yaml(config["retrieval_config"])
    stage_config = load_yaml(config["stage_controller_config"])
    seed = int(config.get("seed", 20260705))
    set_global_determinism(seed)
    require_cuda_if({**model_config, **retrieval_config, **config}, formal=True)

    memory_items = read_jsonl(config["memory_items"])
    queries_all = read_jsonl(config["queries"])
    count = int(config.get("smoke_query_count", 1)) if args.mode == "smoke" else int(config.get("formal_query_count", 240))
    queries = _select_population(queries_all, count)
    if not queries:
        raise RuntimeError("No chronologically orderable B queries selected.")

    memories_by_id = {str(row["memory_id"]): row for row in memory_items}
    memories_by_persona = group_memories_by_persona(memory_items)
    # Reuse the v3 O1 cache for the unchanged turn-pair memory corpus and embedding model.
    vectors, vector_ids, precompute_ms = ensure_embedding_cache(memory_items, retrieval_config, "o1_turn_pair")
    index_by_id = {memory_id: idx for idx, memory_id in enumerate(vector_ids)}
    embed_client = build_embedding_client(retrieval_config)
    reranker_client = build_reranker_client(retrieval_config)
    controller = build_llm_client(model_config, "controller_max_new_tokens")
    stages = list(stage_config.get("stages", ["ground", "plan", "reason", "reflect"]))
    stages = [stage for stage in stages if stage in {"ground", "plan", "reason", "reflect"}]
    frontier_budget = int(config.get("frontier_budget", 64))
    final_top_k = int(config.get("final_top_k", 5))
    run_id = f"B_stage_trace_v4_{args.mode}_{seed}"
    _write_runtime_manifest(
        run_id,
        {
            "mode": args.mode,
            "experiment": "B_stage_trace_v4",
            "frontier_budget": frontier_budget,
            "final_top_k": final_top_k,
            "precompute_embedding_ms": precompute_ms,
            "stage_label_source": "controller_fsm_v1",
        },
    )
    write_manifest(
        f"manifests/{run_id}_manifest.json",
        environment_manifest(
            {
                "run_id": run_id,
                "mode": args.mode,
                "experiment": "B_stage_trace_v4",
                "frontier_budget": frontier_budget,
                "final_top_k": final_top_k,
                "stage_label_source": "controller_fsm_v1",
                "selected_queries": len(queries),
                "precompute_embedding_ms": precompute_ms,
            }
        ),
    )

    query_time_order = {str(q["query_id"]): idx for idx, q in enumerate(sorted(queries, key=lambda q: (str(q.get("persona_id")), _query_order(q))))}
    rows = []
    total_units = len(queries) * len(stages)
    write_progress(experiment="B", mode=args.mode, total_queries=len(queries), completed_queries=0, total_units=total_units, completed_units=0)
    completed_units = 0
    completed_queries = set()
    population_rows = []
    for q in queries:
        population_rows.append(
            {
                "query_id": str(q["query_id"]),
                "persona_id": q.get("persona_id"),
                "query_type": q.get("query_type", ""),
                "query_session_id": q.get("query_session_id"),
                "query_time_order": query_time_order[str(q["query_id"])],
                "source": "selected_for_v4",
            }
        )
        allow_current = bool(config.get("allow_current_session_context", True))
        eligible = eligible_memories_for_query(memories_by_persona, q, allow_current_session=allow_current, include_noise=True)
        validate_no_future_or_cross_persona(q, eligible, allow_current_session=allow_current)
        candidate_indices = [index_by_id[str(row["memory_id"])] for row in eligible if str(row["memory_id"]) in index_by_id]
        candidate_vectors = vectors[candidate_indices] if candidate_indices else vectors[:0]
        prior_summaries: list[str] = []
        for stage_order, stage in enumerate(stages):
            instruction = str(stage_config.get("stage_instructions", {}).get(stage, stage))
            prefix = str(stage_config.get("stage_query_prefixes", {}).get(stage, f"{stage} evidence:"))
            prompt = _intent_prompt(q, stage, instruction, prefix, prior_summaries)
            response, t_intent_ms = timed_call(controller.generate_with_stats, prompt)
            intent = _parse_intent(str(response.get("text", "")), fallback_goal=str(q["query_text"]))
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
            selected = [str(mid) for mid in result["final_selected_ids"]]
            selected_memories = [memories_by_id[mid] for mid in selected if mid in memories_by_id]
            prior_summaries.append(f"{stage}: {intent['retrieval_goal']}")
            rows.append(
                {
                    "run_id": run_id,
                    "query_id": str(q["query_id"]),
                    "persona_id": q.get("persona_id"),
                    "query_type": q.get("query_type", ""),
                    "query_session_id": q.get("query_session_id"),
                    "query_time_order": query_time_order[str(q["query_id"])],
                    "stage": stage,
                    "stage_order": stage_order,
                    "controller_version": "controller_fsm_v1",
                    "stage_label_source": "controller_fsm_v1",
                    "retrieval_goal": intent["retrieval_goal"],
                    "required_slots": intent["required_slots"],
                    "selected_memory_ids": selected,
                    "final_selected_ids": selected,
                    "selected_memory_session_ids": [m.get("session_id") for m in selected_memories],
                    "selected_memory_turn_indices": [_extract_turn(m) for m in selected_memories],
                    "selection_latency_ms": result["t_total_selection_ms"],
                    "controller_intent_latency_ms": t_intent_ms,
                    "t_total_selection_ms": result["t_total_selection_ms"],
                    "t_intent_generation_ms": t_intent_ms,
                    "n_reranked": result["n_reranked"],
                    "n_final_selected": result["n_final_selected"],
                    "frontier_budget": frontier_budget,
                    "final_top_k": final_top_k,
                    "intent_parse_ok": intent["intent_parse_ok"],
                    "parsed_intent_json": intent["parsed_intent_json"],
                    "timestamp": utc_now(),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                    "local_qwen_cuda_device": os.environ.get("LOCAL_QWEN_CUDA_DEVICE", ""),
                    "error": "",
                }
            )
            completed_units += 1
        completed_queries.add(str(q["query_id"]))
        if len(completed_queries) % 2 == 0 or len(completed_queries) == len(queries):
            write_progress(
                experiment="B",
                mode=args.mode,
                total_queries=len(queries),
                completed_queries=len(completed_queries),
                total_units=total_units,
                completed_units=completed_units,
                current_query_id=str(q["query_id"]),
                extra={"stage": "reflect"},
            )
    output = "aggregate/B_stage_trace_v4_smoke.jsonl" if args.mode == "smoke" else config.get("output_stage_trace", "aggregate/B_stage_trace_v4.jsonl")
    write_jsonl(output, rows)
    if args.mode == "formal":
        write_jsonl("aggregate/B_stage_trace_v4_population.jsonl", population_rows)
    write_progress(experiment="B", mode=args.mode, total_queries=len(queries), completed_queries=len(completed_queries), total_units=total_units, completed_units=completed_units, extra={"status": "complete"})
    print(f"b_stage_trace={resolve_path(output)}")


if __name__ == "__main__":
    main()
