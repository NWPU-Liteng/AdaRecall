#!/usr/bin/env python3
import json
import os
import random
import statistics
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from synthetic_memory_generator import QuerySpec, build_query_workload, load_or_generate_memories, read_jsonl, write_jsonl
from trace_logger import append_jsonl, ensure_output_layout

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-agent-memory-motivation")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/agent-memory-motivation-cache")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.embedding_client import HashEmbeddingClient, LocalQwenEmbeddingClient, OllamaEmbeddingClient, TransformerEmbeddingClient
from src.agent_executor import execute_agent_response
from src.llm_client import LocalQwenLLMClient, MockLLMClient, OllamaLLMClient
from src.reranker_client import resolve_reranker_client
from src.vector_store import AnnNumpyVectorStore, NumpyVectorStore

from metrics import write_summary_files
from plot_results import make_plots


def _load_config(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Config must be JSON-compatible YAML: {path}") from exc


def _resolve_embed_client(config: dict):
    backend = config.get("embedding_backend", "hash")
    dim = int(config.get("embedding_dim", 384))
    local_qwen_dir = str(PROJECT_ROOT.parent / "dir")
    transformer_model_dir = str(config.get("embedding_model_dir", "")).strip()
    if backend == "local_qwen":
        return LocalQwenEmbeddingClient(model_dir=local_qwen_dir, dim=dim)
    if backend == "transformer":
        if not transformer_model_dir:
            raise ValueError("embedding_model_dir is required when embedding_backend='transformer'")
        max_length = int(config.get("embedding_max_length", 512))
        batch_size = int(config.get("embedding_batch_size", 16))
        return TransformerEmbeddingClient(
            model_dir=transformer_model_dir,
            dim=dim,
            max_length=max_length,
            batch_size=batch_size,
        )
    if backend == "ollama_hash":
        return OllamaEmbeddingClient(dim=dim)
    return HashEmbeddingClient(dim=dim)


def _resolve_llm_client(config: dict):
    backend = config.get("generation_backend", "mock")
    local_qwen_dir = str(PROJECT_ROOT.parent / "dir")
    max_tokens = int(config.get("llm_max_new_tokens", 24))
    if backend == "local_qwen":
        return LocalQwenLLMClient(model_dir=local_qwen_dir, max_new_tokens=max_tokens)
    if backend == "ollama":
        return OllamaLLMClient()
    return MockLLMClient()


def _resolve_reranker_client(config: dict):
    return resolve_reranker_client(config, str(PROJECT_ROOT))


def _data_dir() -> Path:
    return PROJECT_ROOT / "data" / "motivation_memory_selection"


def _normalize_memory_type(memory_type: str) -> str:
    aliases = {
        "health_constraint": "constraint",
        "constraint": "constraint",
        "planning": "procedural",
        "procedural": "procedural",
        "feedback": "episodic",
        "episodic": "episodic",
        "profile": "profile",
        "semantic": "semantic",
    }
    return aliases.get(str(memory_type or "").strip(), str(memory_type or "episodic").strip() or "episodic")


def _normalize_memory_row(raw: dict, fallback_index: int) -> dict:
    memory_id = raw.get("memory_id") or raw.get("id") or f"memory_{fallback_index:07d}"
    text = raw.get("text") or raw.get("content") or raw.get("summary") or ""
    summary = raw.get("summary") or (text[:120] if text else "")
    timestamp = raw.get("timestamp") or raw.get("time") or "2026-06-01"
    task_tags = raw.get("task_tags") or raw.get("module_tags") or ["general"]
    entity_tags = raw.get("entity_tags") or []
    relations = raw.get("relations") or raw.get("linked_memory_ids") or []
    gold_group_id = raw.get("gold_group_id") or raw.get("session_id") or raw.get("conversation_id") or "group_default"
    return {
        "memory_id": str(memory_id),
        "text": str(text),
        "summary": str(summary),
        "timestamp": str(timestamp),
        "memory_type": _normalize_memory_type(raw.get("memory_type")),
        "task_tags": list(task_tags) if isinstance(task_tags, list) else [str(task_tags)],
        "entity_tags": list(entity_tags) if isinstance(entity_tags, list) else [str(entity_tags)] if entity_tags else [],
        "importance": float(raw.get("importance", 0.5)),
        "recency": float(raw.get("recency", 0.5)),
        "relations": list(relations) if isinstance(relations, list) else [str(relations)] if relations else [],
        "gold_group_id": str(gold_group_id),
    }


def _resolve_external_memory_path(size: int, config: dict) -> Path | None:
    mapping = config.get("input_memory_paths", {})
    if not isinstance(mapping, dict):
        return None
    candidate = mapping.get(str(size)) or mapping.get(size)
    if not candidate:
        return None
    path = Path(candidate)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def _load_memories_for_setting(size: int, config: dict, seed: int) -> tuple[list[dict], str]:
    external_path = _resolve_external_memory_path(size, config)
    if external_path is not None:
        rows = read_jsonl(external_path)
        normalized = [_normalize_memory_row(row, idx + 1) for idx, row in enumerate(rows)]
        if len(normalized) > size:
            sample_mode = str(config.get("external_memory_sampling", "head"))
            if sample_mode == "random":
                rng = random.Random(seed + size + len(normalized))
                normalized = rng.sample(normalized, size)
            else:
                normalized = normalized[:size]
        return normalized, f"{external_path}#sampled={len(normalized)}"
    generated = load_or_generate_memories(_memory_path(size), size, seed=seed)
    return generated, "synthetic_generator"


def _memory_path(size: int) -> Path:
    return _data_dir() / f"motivation_memories_{size}.jsonl"


def _index_stem(size: int, config: dict) -> str:
    namespace = str(config.get("index_namespace", "")).strip()
    if namespace:
        return f"{namespace}_{size}"
    return f"{size}"


def _vector_path(size: int, config: dict) -> Path:
    return _data_dir() / f"motivation_index_vectors_{_index_stem(size, config)}.npy"


def _metadata_path(size: int, config: dict) -> Path:
    return _data_dir() / f"motivation_index_metadata_{_index_stem(size, config)}.jsonl"


def _ann_path(size: int, config: dict) -> Path:
    return _data_dir() / f"motivation_index_ann_{_index_stem(size, config)}.npz"


def _ensure_index(memories: List[dict], size: int, config: dict):
    vector_path = _vector_path(size, config)
    metadata_path = _metadata_path(size, config)
    ann_path = _ann_path(size, config)
    retrieval_backend = config.get("retrieval_backend", "ann_numpy")
    dim = int(config.get("embedding_dim", 384))
    embed_client = _resolve_embed_client(config)

    if vector_path.exists() and metadata_path.exists():
        if retrieval_backend != "ann_numpy" or ann_path.exists():
            if retrieval_backend == "ann_numpy":
                store = AnnNumpyVectorStore(str(vector_path), str(metadata_path), str(ann_path))
            else:
                store = NumpyVectorStore(str(vector_path), str(metadata_path))
            store.load()
            return store, getattr(embed_client, "mode", "hash_embedding")

    vectors = embed_client.embed_batch(memory["text"] for memory in memories)
    if retrieval_backend == "ann_numpy":
        store = AnnNumpyVectorStore(str(vector_path), str(metadata_path), str(ann_path))
    else:
        store = NumpyVectorStore(str(vector_path), str(metadata_path))
    store.build(vectors, memories)
    return store, getattr(embed_client, "mode", "hash_embedding")


def _memory_indices(memories: List[dict]) -> Tuple[Dict[str, dict], Dict[str, List[str]]]:
    by_id = {memory["memory_id"]: memory for memory in memories}
    graph: Dict[str, List[str]] = {}
    for memory in memories:
        graph[memory["memory_id"]] = list(memory.get("relations", []))
    return by_id, graph


def _query_matches_metadata(query: QuerySpec, memory: dict, now: str) -> bool:
    _ = now
    task_overlap = bool(set(query.task_tags) & set(memory.get("task_tags", [])))
    entity_overlap = bool(set(query.entity_tags) & set(memory.get("entity_tags", [])))
    type_match = memory.get("memory_type") in query.preferred_memory_types
    if query.query_type == "fact":
        return type_match and (task_overlap or entity_overlap)
    if query.query_type == "temporal":
        return task_overlap or entity_overlap or type_match
    if query.query_type == "relation":
        return task_overlap or entity_overlap or memory.get("memory_type") in {"constraint", "semantic"}
    if query.query_type == "plan_continuity":
        return task_overlap or memory.get("memory_type") in {"procedural", "profile", "episodic"}
    if query.query_type == "constraint":
        return entity_overlap or memory.get("memory_type") in {"constraint", "profile"}
    return task_overlap or entity_overlap or type_match


def _metadata_filter(query: QuerySpec, candidates: List[Tuple[dict, float]]) -> List[Tuple[dict, float]]:
    filtered = []
    for memory, score in candidates:
        if _query_matches_metadata(query, memory, "2026-06-01"):
            filtered.append((memory, score))
    return filtered or candidates


def _expand_relations(
    filtered: List[Tuple[dict, float]],
    graph: Dict[str, List[str]],
    by_id: Dict[str, dict],
    depth: int,
) -> Tuple[List[Tuple[dict, float]], int]:
    if depth <= 0:
        return list(filtered), 0

    visited = {memory["memory_id"] for memory, _ in filtered}
    queue = deque((memory["memory_id"], 0, score) for memory, score in filtered)
    expanded: List[Tuple[dict, float]] = []
    traversed_edges = 0
    while queue:
        current_id, current_depth, inherited_score = queue.popleft()
        if current_depth >= depth:
            continue
        for neighbor_id in graph.get(current_id, []):
            traversed_edges += 1
            if neighbor_id in visited:
                continue
            neighbor = by_id.get(neighbor_id)
            if neighbor is None:
                continue
            visited.add(neighbor_id)
            # Relation traversal expands candidates but slightly decays the
            # inherited semantic score to reflect weaker direct relevance.
            next_score = inherited_score * (0.92 ** (current_depth + 1))
            expanded.append((neighbor, next_score))
            queue.append((neighbor_id, current_depth + 1, next_score))
    return expanded, traversed_edges


def _aggregate_candidates(filtered: List[Tuple[dict, float]], expanded: List[Tuple[dict, float]]) -> List[Tuple[dict, float]]:
    merged: Dict[str, Tuple[dict, float]] = {}
    for memory, score in filtered + expanded:
        current = merged.get(memory["memory_id"])
        if current is None or score > current[1]:
            merged[memory["memory_id"]] = (memory, score)
    return sorted(merged.values(), key=lambda item: item[1], reverse=True)


def _rerank_score(query: QuerySpec, memory: dict, vector_score: float) -> float:
    task_overlap = len(set(query.task_tags) & set(memory.get("task_tags", [])))
    entity_overlap = len(set(query.entity_tags) & set(memory.get("entity_tags", [])))
    importance = float(memory.get("importance", 0.0))
    recency = float(memory.get("recency", 0.0))
    relation_bonus = 0.18 if memory.get("gold_group_id") in query.gold_group_ids else 0.0
    if query.query_type == "relation":
        relation_bonus += 0.12
    if query.query_type == "plan_continuity":
        relation_bonus += 0.08
    if query.query_type == "constraint" and memory.get("memory_type") == "constraint":
        relation_bonus += 0.12
    return float(vector_score) + 0.22 * importance + 0.16 * recency + 0.1 * task_overlap + 0.08 * entity_overlap + relation_bonus


def _heuristic_rerank(query: QuerySpec, rerank_input: List[Tuple[dict, float]]) -> List[Tuple[dict, float]]:
    rescored = [
        (memory, _rerank_score(query, memory, vector_score))
        for memory, vector_score in rerank_input
    ]
    rescored.sort(key=lambda item: item[1], reverse=True)
    return rescored


def _cross_encoder_rerank(query: QuerySpec, rerank_input: List[Tuple[dict, float]], reranker_client) -> List[Tuple[dict, float]]:
    pairs = [(query.text, memory, vector_score) for memory, vector_score in rerank_input]
    model_scores = reranker_client.score_pairs(pairs)
    rescored = []
    for (memory, vector_score), model_score in zip(rerank_input, model_scores):
        combined_score = float(model_score) + 0.05 * float(vector_score)
        rescored.append((memory, combined_score))
    rescored.sort(key=lambda item: item[1], reverse=True)
    return rescored


def _package_context(query: QuerySpec, selected: List[Tuple[dict, float]], top_k: int) -> str:
    lines = [f"Query: {query.text}", "Candidate memories:"]
    for memory, score in selected[:top_k]:
        lines.append(
            f"- [{memory['memory_id']}] type={memory['memory_type']} score={score:.4f} "
            f"time={memory['timestamp']} group={memory['gold_group_id']} text={memory['text']}"
        )
    lines.append("Respond briefly.")
    return "\n".join(lines)


def _estimate_traffic(
    scanned_memory_count: int,
    relation_edge_count: int,
    num_before_rerank: int,
    config: dict,
) -> Dict[str, int]:
    header_size = int(config.get("header_size_bytes", 128))
    edge_entry_size = int(config.get("edge_entry_size_bytes", 8))
    candidate_payload_size = int(config.get("candidate_payload_size_bytes", 512))
    embedding_size = int(config.get("embedding_size_bytes", 1536))

    embedding_traffic = scanned_memory_count * embedding_size
    structural_traffic = scanned_memory_count * header_size + relation_edge_count * edge_entry_size
    host_traffic = structural_traffic + num_before_rerank * candidate_payload_size
    return {
        "estimated_host_memory_traffic_bytes": int(host_traffic),
        "estimated_embedding_traffic_bytes": int(embedding_traffic),
        "estimated_structural_traffic_bytes": int(structural_traffic),
        "pim_optimization_potential_bytes": int(num_before_rerank * candidate_payload_size),
    }


def _search_scanned_count(store, memories: List[dict]) -> int:
    stats = getattr(store, "last_search_stats", {}) or {}
    if "candidate_vectors" in stats:
        return int(stats["candidate_vectors"])
    return len(memories)


def _selection_latency_keys() -> List[str]:
    return [
        "query_encoding_latency_ms",
        "coarse_retrieval_latency_ms",
        "metadata_filter_latency_ms",
        "relation_expansion_latency_ms",
        "candidate_aggregation_latency_ms",
        "rerank_latency_ms",
        "context_construction_latency_ms",
    ]


def _selection_retrieval_latency_keys() -> List[str]:
    return [
        "query_encoding_latency_ms",
        "coarse_retrieval_latency_ms",
        "metadata_filter_latency_ms",
        "relation_expansion_latency_ms",
        "candidate_aggregation_latency_ms",
        "context_construction_latency_ms",
    ]


def _agent_latency_keys() -> List[str]:
    return [
        "agent_planning_latency_ms",
        "agent_tool_execution_latency_ms",
        "agent_reasoning_latency_ms",
        "agent_output_latency_ms",
    ]


def _resolve_agent_memory_mode(config: dict) -> str:
    mode = str(config.get("agent_memory_mode", "shared_once")).strip() or "shared_once"
    if mode not in {"shared_once", "phase_aware"}:
        return "shared_once"
    return mode


def _phase_plan(config: dict) -> List[Tuple[str, str]]:
    phases: List[Tuple[str, str]] = []
    if int(config.get("agent_planning_llm_rounds", 0)) > 0 or bool(config.get("agent_enable_intermediate_llm_steps", False)):
        phases.append(("planning", "Planning phase memory search."))
    if int(config.get("agent_reasoning_llm_rounds", 0)) > 0 or bool(config.get("agent_enable_intermediate_llm_steps", False)):
        phases.append(("reasoning", "Reasoning phase memory search."))
    if int(config.get("agent_output_llm_rounds", 1)) > 0:
        phases.append(("output", "Output phase memory search."))
    if not phases:
        phases.append(("output", "Output phase memory search."))
    return phases


def _phase_query_text(query: QuerySpec, phase_name: str) -> str:
    suffix_map = {
        "planning": "Focus on plan setup, constraints, and long-term continuity.",
        "reasoning": "Focus on linked evidence, recent outcomes, and causal support.",
        "output": "Focus on concise answer support and user-facing grounding.",
    }
    suffix = suffix_map.get(phase_name, "Focus on the next agent phase.")
    return f"{query.text} {suffix}"


def _run_selection_pass(
    query: QuerySpec,
    memories: List[dict],
    store,
    embed_client,
    reranker_client,
    config: dict,
    baseline_mode: str,
    retrieval_topk: int,
    relation_expansion_depth: int,
    rerank_topk_input,
    by_id: Dict[str, dict],
    graph: Dict[str, List[str]],
) -> dict:
    final_topk = int(config.get("final_topk", 8))
    latencies_ms = {key: 0.0 for key in _selection_latency_keys()}
    counts = {
        "num_raw_retrieved": 0,
        "num_after_metadata_filter": 0,
        "num_after_relation_expansion": 0,
        "num_after_dedup": 0,
        "num_before_rerank": 0,
        "num_after_rerank": 0,
    }

    start = time.perf_counter()
    query_vector = embed_client.embed(query.text)
    latencies_ms["query_encoding_latency_ms"] = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    raw_candidates = store.search(query_vector, retrieval_topk)
    latencies_ms["coarse_retrieval_latency_ms"] = (time.perf_counter() - start) * 1000.0
    counts["num_raw_retrieved"] = len(raw_candidates)

    if baseline_mode == "retrieval_only":
        filtered = list(raw_candidates)
        counts["num_after_metadata_filter"] = len(filtered)
    else:
        start = time.perf_counter()
        filtered = _metadata_filter(query, raw_candidates)
        latencies_ms["metadata_filter_latency_ms"] = (time.perf_counter() - start) * 1000.0
        counts["num_after_metadata_filter"] = len(filtered)

    relation_depth = 0 if baseline_mode == "retrieval_only" else relation_expansion_depth
    start = time.perf_counter()
    expanded, relation_edge_count = _expand_relations(filtered, graph, by_id, relation_depth)
    latencies_ms["relation_expansion_latency_ms"] = (time.perf_counter() - start) * 1000.0
    counts["num_after_relation_expansion"] = len(expanded)

    if baseline_mode == "retrieval_only":
        aggregated = list(raw_candidates)
    else:
        start = time.perf_counter()
        aggregated = _aggregate_candidates(filtered, expanded)
        latencies_ms["candidate_aggregation_latency_ms"] = (time.perf_counter() - start) * 1000.0
    counts["num_after_dedup"] = len(aggregated)

    rerank_input = list(aggregated)
    if baseline_mode == "retrieval_only":
        rerank_input = list(raw_candidates)
    if rerank_topk_input != "all":
        rerank_input = rerank_input[: int(rerank_topk_input)]
    if baseline_mode == "oracle_pruned_rerank":
        prune_limit = int(config.get("oracle_prune_limit", 50))
        oracle = [item for item in aggregated if item[0].get("gold_group_id") in query.gold_group_ids]
        spill = [item for item in aggregated if item[0].get("gold_group_id") not in query.gold_group_ids]
        rerank_input = (oracle + spill)[: min(prune_limit, len(oracle) + len(spill))]
        if rerank_topk_input != "all":
            rerank_input = rerank_input[: min(int(rerank_topk_input), len(rerank_input))]
    counts["num_before_rerank"] = len(rerank_input)

    start = time.perf_counter()
    if baseline_mode in {"retrieval_only", "retrieval_plus_expansion"}:
        selected = rerank_input[:final_topk]
    else:
        reranker_backend = str(config.get("reranker_backend", "heuristic"))
        if reranker_backend == "cross_encoder":
            rescored = _cross_encoder_rerank(query, rerank_input, reranker_client)
        else:
            rescored = _heuristic_rerank(query, rerank_input)
        selected = rescored[:final_topk]
    latencies_ms["rerank_latency_ms"] = (time.perf_counter() - start) * 1000.0
    counts["num_after_rerank"] = len(selected)

    start = time.perf_counter()
    prompt = _package_context(query, selected, final_topk)
    latencies_ms["context_construction_latency_ms"] = (time.perf_counter() - start) * 1000.0

    return {
        "latencies_ms": latencies_ms,
        "counts": counts,
        "selected": selected,
        "prompt": prompt,
        "selection_total_latency_ms": sum(latencies_ms.values()),
        "relation_edge_count": relation_edge_count,
        "scanned_memory_count": _search_scanned_count(store, memories),
    }


def run_selection_profile_pass(
    query: QuerySpec,
    memories: List[dict],
    store,
    embed_client,
    reranker_client,
    config: dict,
    baseline_mode: str,
    retrieval_topk: int,
    relation_expansion_depth: int,
    rerank_topk_input,
) -> dict:
    by_id, graph = _memory_indices(memories)
    return _run_selection_pass(
        query=query,
        memories=memories,
        store=store,
        embed_client=embed_client,
        reranker_client=reranker_client,
        config=config,
        baseline_mode=baseline_mode,
        retrieval_topk=retrieval_topk,
        relation_expansion_depth=relation_expansion_depth,
        rerank_topk_input=rerank_topk_input,
        by_id=by_id,
        graph=graph,
    )


def _run_single_query(
    query: QuerySpec,
    memories: List[dict],
    store,
    embed_client,
    llm_client,
    reranker_client,
    config: dict,
    baseline_mode: str,
    memory_size: int,
    retrieval_topk: int,
    relation_expansion_depth: int,
    rerank_topk_input,
) -> dict:
    by_id, graph = _memory_indices(memories)
    generation_enabled = bool(config.get("enable_generation", False))
    agent_execution_enabled = bool(config.get("enable_agent_execution", False))
    agent_memory_mode = _resolve_agent_memory_mode(config)
    start_total = time.perf_counter()

    latencies_ms = {
        "query_encoding_latency_ms": 0.0,
        "coarse_retrieval_latency_ms": 0.0,
        "metadata_filter_latency_ms": 0.0,
        "relation_expansion_latency_ms": 0.0,
        "candidate_aggregation_latency_ms": 0.0,
        "rerank_latency_ms": 0.0,
        "context_construction_latency_ms": 0.0,
        "generation_latency_ms": 0.0,
        "agent_planning_latency_ms": 0.0,
        "agent_tool_execution_latency_ms": 0.0,
        "agent_reasoning_latency_ms": 0.0,
        "agent_output_latency_ms": 0.0,
        "agent_execution_latency_ms": 0.0,
    }
    counts = {
        "num_raw_retrieved": 0,
        "num_after_metadata_filter": 0,
        "num_after_relation_expansion": 0,
        "num_after_dedup": 0,
        "num_before_rerank": 0,
        "num_after_rerank": 0,
    }
    record = {
        "query_id": query.query_id,
        "query_type": query.query_type,
        "memory_size": int(memory_size),
        "baseline_mode": baseline_mode,
        "retrieval_topk": int(retrieval_topk),
        "relation_expansion_depth": int(relation_expansion_depth),
        "rerank_topk_input": rerank_topk_input,
        "generation_enabled": generation_enabled,
        "agent_execution_enabled": agent_execution_enabled,
        "end_to_end_includes_generation": generation_enabled or agent_execution_enabled,
        "agent_memory_mode": agent_memory_mode,
        "selection_invocation_count": 1,
        "error": "",
    }

    try:
        phase_trace_rows: List[dict] = []
        phase_selection_total = 0.0
        phase_rerank_total = 0.0
        phase_retrieval_total = 0.0
        phase_selected_map: Dict[str, List[Tuple[dict, float]]] = {}
        phase_context_map: Dict[str, str] = {}
        phase_latency_map: Dict[str, float] = {}
        phase_rerank_map: Dict[str, float] = {}
        phase_candidate_map: Dict[str, int] = {}

        first_pass = _run_selection_pass(
            query=query,
            memories=memories,
            store=store,
            embed_client=embed_client,
            reranker_client=reranker_client,
            config=config,
            baseline_mode=baseline_mode,
            retrieval_topk=retrieval_topk,
            relation_expansion_depth=relation_expansion_depth,
            rerank_topk_input=rerank_topk_input,
            by_id=by_id,
            graph=graph,
        )
        for key, value in first_pass["latencies_ms"].items():
            latencies_ms[key] = value
        counts.update(first_pass["counts"])
        selected = first_pass["selected"]
        prompt = first_pass["prompt"]
        relation_edge_count = int(first_pass["relation_edge_count"])
        scanned_memory_count = int(first_pass["scanned_memory_count"])

        if agent_execution_enabled and agent_memory_mode == "phase_aware":
            phase_rows: List[dict] = []
            phase_plan = _phase_plan(config)
            phase_selection_total = 0.0
            phase_rerank_total = 0.0
            phase_retrieval_total = 0.0
            for phase_name, _ in phase_plan:
                phase_query = QuerySpec(
                    query_id=f"{query.query_id}:{phase_name}",
                    query_type=query.query_type,
                    text=_phase_query_text(query, phase_name),
                    task_tags=list(query.task_tags),
                    entity_tags=list(query.entity_tags),
                    preferred_memory_types=list(query.preferred_memory_types),
                    temporal_window_days=query.temporal_window_days,
                    gold_group_ids=list(query.gold_group_ids),
                )
                phase_pass = _run_selection_pass(
                    query=phase_query,
                    memories=memories,
                    store=store,
                    embed_client=embed_client,
                    reranker_client=reranker_client,
                    config=config,
                    baseline_mode=baseline_mode,
                    retrieval_topk=retrieval_topk,
                    relation_expansion_depth=relation_expansion_depth,
                    rerank_topk_input=rerank_topk_input,
                    by_id=by_id,
                    graph=graph,
                )
                phase_latency = float(phase_pass["selection_total_latency_ms"])
                phase_rerank = float(phase_pass["latencies_ms"]["rerank_latency_ms"])
                phase_retrieval = sum(
                    float(phase_pass["latencies_ms"][key]) for key in _selection_retrieval_latency_keys()
                )
                phase_selection_total += phase_latency
                phase_rerank_total += phase_rerank
                phase_retrieval_total += phase_retrieval
                phase_selected_map[phase_name] = phase_pass["selected"]
                phase_context_map[phase_name] = str(phase_pass["prompt"])
                phase_latency_map[phase_name] = phase_latency
                phase_rerank_map[phase_name] = phase_rerank
                phase_candidate_map[phase_name] = int(phase_pass["counts"]["num_before_rerank"])
                phase_rows.append(
                    {
                        "trace_id": f"{query.query_id}:{baseline_mode}:{memory_size}:{retrieval_topk}:{relation_expansion_depth}:{rerank_topk_input}",
                        "query_id": query.query_id,
                        "query_type": query.query_type,
                        "base_query_text": query.text,
                        "phase_name": phase_name,
                        "phase_query_text": phase_query.text,
                        "task_tags": list(query.task_tags),
                        "entity_tags": list(query.entity_tags),
                        "preferred_memory_types": list(query.preferred_memory_types),
                        "gold_group_ids": list(query.gold_group_ids),
                        "temporal_window_days": int(query.temporal_window_days),
                        "memory_size": int(memory_size),
                        "baseline_mode": baseline_mode,
                        "retrieval_topk": int(retrieval_topk),
                        "relation_expansion_depth": int(relation_expansion_depth),
                        "rerank_topk_input": rerank_topk_input,
                        "phase_selection_latency_ms": phase_latency,
                        "phase_rerank_latency_ms": phase_rerank,
                        "phase_retrieval_latency_ms": phase_retrieval,
                        "phase_num_before_rerank": int(phase_pass["counts"]["num_before_rerank"]),
                        "phase_num_after_rerank": int(phase_pass["counts"]["num_after_rerank"]),
                        "phase_selected_memory_ids": [memory["memory_id"] for memory, _ in phase_pass["selected"]],
                        "phase_prompt_chars": len(str(phase_pass["prompt"])),
                        "agent_memory_mode": agent_memory_mode,
                    }
                )
            phase_trace_rows = phase_rows
            if phase_plan:
                last_phase = phase_plan[-1][0]
                selected = phase_selected_map[last_phase]
                prompt = phase_context_map[last_phase]
                counts["num_before_rerank"] = max(phase_candidate_map.values())
                counts["num_after_rerank"] = len(selected)
            record["selection_invocation_count"] = len(phase_plan)
        else:
            phase_selection_total = float(first_pass["selection_total_latency_ms"])
            phase_rerank_total = float(first_pass["latencies_ms"]["rerank_latency_ms"])
            phase_retrieval_total = sum(float(first_pass["latencies_ms"][key]) for key in _selection_retrieval_latency_keys())
            phase_trace_rows = [
                {
                    "trace_id": f"{query.query_id}:{baseline_mode}:{memory_size}:{retrieval_topk}:{relation_expansion_depth}:{rerank_topk_input}",
                    "query_id": query.query_id,
                    "query_type": query.query_type,
                    "base_query_text": query.text,
                    "phase_name": "shared",
                    "phase_query_text": query.text,
                    "task_tags": list(query.task_tags),
                    "entity_tags": list(query.entity_tags),
                    "preferred_memory_types": list(query.preferred_memory_types),
                    "gold_group_ids": list(query.gold_group_ids),
                    "temporal_window_days": int(query.temporal_window_days),
                    "memory_size": int(memory_size),
                    "baseline_mode": baseline_mode,
                    "retrieval_topk": int(retrieval_topk),
                    "relation_expansion_depth": int(relation_expansion_depth),
                    "rerank_topk_input": rerank_topk_input,
                    "phase_selection_latency_ms": phase_selection_total,
                    "phase_rerank_latency_ms": phase_rerank_total,
                    "phase_retrieval_latency_ms": phase_retrieval_total,
                    "phase_num_before_rerank": int(first_pass["counts"]["num_before_rerank"]),
                    "phase_num_after_rerank": int(first_pass["counts"]["num_after_rerank"]),
                    "phase_selected_memory_ids": [memory["memory_id"] for memory, _ in selected],
                    "phase_prompt_chars": len(prompt),
                    "agent_memory_mode": agent_memory_mode,
                }
            ]
            record["selection_invocation_count"] = 1

        if agent_execution_enabled:
            start = time.perf_counter()
            agent_result = execute_agent_response(
                query_text=query.text,
                context_prompt=prompt,
                selected=selected,
                llm_client=llm_client,
                config=config,
                phase_contexts=phase_context_map if phase_context_map else None,
                phase_selected=phase_selected_map if phase_selected_map else None,
            )
            latencies_ms["agent_planning_latency_ms"] = float(agent_result.get("agent_planning_latency_ms", 0.0))
            latencies_ms["agent_tool_execution_latency_ms"] = float(agent_result.get("agent_tool_execution_latency_ms", 0.0))
            latencies_ms["agent_reasoning_latency_ms"] = float(agent_result.get("agent_reasoning_latency_ms", 0.0))
            latencies_ms["agent_output_latency_ms"] = float(agent_result.get("agent_output_latency_ms", 0.0))
            latencies_ms["agent_execution_latency_ms"] = (time.perf_counter() - start) * 1000.0
            latencies_ms["generation_latency_ms"] = latencies_ms["agent_output_latency_ms"]
            answer = str(agent_result.get("answer", ""))
            agent_steps = int(agent_result.get("agent_steps", 0))
            agent_tool_calls = int(agent_result.get("agent_tool_calls", 0))
            agent_backend_mode = str(agent_result.get("agent_backend_mode", "unknown"))
        elif generation_enabled:
            start = time.perf_counter()
            answer = llm_client.generate(prompt)
            latencies_ms["generation_latency_ms"] = (time.perf_counter() - start) * 1000.0
            agent_steps = 0
            agent_tool_calls = 0
            agent_backend_mode = "disabled"
        else:
            answer = ""
            agent_steps = 0
            agent_tool_calls = 0
            agent_backend_mode = "disabled"

        end_total = time.perf_counter()
        selection_total = sum(
            latencies_ms[key] for key in _selection_latency_keys()
        )
        end_to_end = (end_total - start_total) * 1000.0
        agent_execution_total = latencies_ms["agent_execution_latency_ms"]
        total_response_latency = phase_selection_total + agent_execution_total
        agent_stage_sum = sum(
            latencies_ms[key] for key in _agent_latency_keys()
        )
        traffic = _estimate_traffic(
            scanned_memory_count=scanned_memory_count,
            relation_edge_count=relation_edge_count,
            num_before_rerank=counts["num_before_rerank"],
            config=config,
        )
        record.update(counts)
        record.update(latencies_ms)
        record.update(traffic)
        record.update(
            {
                "selection_total_latency_ms": selection_total,
                "phase_selection_total_latency_ms": phase_selection_total,
                "phase_retrieval_total_latency_ms": phase_retrieval_total,
                "phase_rerank_total_latency_ms": phase_rerank_total,
                "selection_refresh_overhead_latency_ms": max(phase_selection_total - selection_total, 0.0),
                "agent_execution_total_latency_ms": agent_execution_total,
                "agent_stage_sum_latency_ms": agent_stage_sum,
                "total_response_latency_ms": total_response_latency,
                "end_to_end_latency_ms": end_to_end,
                "selection_ratio": selection_total / max(end_to_end, 1e-9),
                "selection_to_total_response_ratio": selection_total / max(total_response_latency, 1e-9),
                "phase_selection_to_total_response_ratio": phase_selection_total / max(total_response_latency, 1e-9),
                "phase_retrieval_to_total_response_ratio": phase_retrieval_total / max(total_response_latency, 1e-9),
                "phase_rerank_to_total_response_ratio": phase_rerank_total / max(total_response_latency, 1e-9),
                "agent_execution_ratio": agent_execution_total / max(total_response_latency, 1e-9),
                "agent_planning_ratio": latencies_ms["agent_planning_latency_ms"] / max(agent_execution_total, 1e-9),
                "agent_tool_execution_ratio": latencies_ms["agent_tool_execution_latency_ms"] / max(agent_execution_total, 1e-9),
                "agent_reasoning_ratio": latencies_ms["agent_reasoning_latency_ms"] / max(agent_execution_total, 1e-9),
                "agent_output_ratio": latencies_ms["agent_output_latency_ms"] / max(agent_execution_total, 1e-9),
                "rerank_ratio": latencies_ms["rerank_latency_ms"] / max(selection_total, 1e-9),
                "phase_rerank_ratio": phase_rerank_total / max(phase_selection_total, 1e-9),
                "prompt_chars": len(prompt),
                "answer_chars": len(answer),
                "agent_steps": agent_steps,
                "agent_tool_calls": agent_tool_calls,
                "agent_backend_mode": agent_backend_mode,
                "selected_memory_ids": [memory["memory_id"] for memory, _ in selected],
                "phase_trace_rows": phase_trace_rows,
                "embedding_mode": getattr(embed_client, "mode", "hash_embedding"),
                "llm_mode": getattr(llm_client, "mode", "mock"),
                "reranker_mode": getattr(reranker_client, "mode", "heuristic"),
                "retrieval_backend_mode": getattr(store, "mode", "numpy_fallback"),
            }
        )
        return record
    except Exception as exc:
        end_total = time.perf_counter()
        selection_total = sum(
            latencies_ms[key] for key in _selection_latency_keys()
        )
        agent_execution_total = latencies_ms.get("agent_execution_latency_ms", 0.0)
        agent_stage_sum = sum(
            latencies_ms[key] for key in _agent_latency_keys()
        )
        record.update(counts)
        record.update(latencies_ms)
        record.update(
            {
                "selection_total_latency_ms": selection_total,
                "phase_selection_total_latency_ms": 0.0,
                "phase_retrieval_total_latency_ms": 0.0,
                "phase_rerank_total_latency_ms": 0.0,
                "selection_refresh_overhead_latency_ms": 0.0,
                "agent_execution_total_latency_ms": agent_execution_total,
                "agent_stage_sum_latency_ms": agent_stage_sum,
                "total_response_latency_ms": selection_total + agent_execution_total,
                "end_to_end_latency_ms": (end_total - start_total) * 1000.0,
                "selection_ratio": 0.0,
                "selection_to_total_response_ratio": 0.0,
                "phase_selection_to_total_response_ratio": 0.0,
                "phase_retrieval_to_total_response_ratio": 0.0,
                "phase_rerank_to_total_response_ratio": 0.0,
                "agent_execution_ratio": 0.0,
                "agent_planning_ratio": 0.0,
                "agent_tool_execution_ratio": 0.0,
                "agent_reasoning_ratio": 0.0,
                "agent_output_ratio": 0.0,
                "rerank_ratio": 0.0,
                "phase_rerank_ratio": 0.0,
                "estimated_host_memory_traffic_bytes": 0,
                "estimated_embedding_traffic_bytes": 0,
                "estimated_structural_traffic_bytes": 0,
                "pim_optimization_potential_bytes": 0,
                "prompt_chars": 0,
                "answer_chars": 0,
                "agent_steps": 0,
                "agent_tool_calls": 0,
                "agent_backend_mode": "error",
                "selected_memory_ids": [],
                "phase_trace_rows": [],
                "embedding_mode": getattr(embed_client, "mode", "hash_embedding"),
                "llm_mode": getattr(llm_client, "mode", "mock"),
                "reranker_mode": getattr(reranker_client, "mode", "heuristic"),
                "retrieval_backend_mode": getattr(store, "mode", "numpy_fallback"),
                "error": str(exc),
            }
        )
        return record


def _iter_settings(config: dict):
    baselines = list(config.get("baseline_modes", ["retrieval_only", "retrieval_plus_expansion", "full_selection"]))
    retrieval_values = list(config.get("retrieval_topk_values", [50]))
    depth_values = list(config.get("relation_expansion_depth_values", [0, 1, 2]))
    rerank_inputs = list(config.get("rerank_topk_input_values", ["all", 100]))

    for baseline_mode in baselines:
        for retrieval_topk in retrieval_values:
            if baseline_mode == "retrieval_only":
                yield baseline_mode, retrieval_topk, 0, "all"
                continue
            for depth in depth_values:
                if baseline_mode == "retrieval_plus_expansion":
                    yield baseline_mode, retrieval_topk, depth, "all"
                    continue
                for rerank_input in rerank_inputs:
                    yield baseline_mode, retrieval_topk, depth, rerank_input


def apply_quick_overrides(config: dict) -> dict:
    quick = dict(config)
    if not quick.get("memory_sizes"):
        quick["memory_sizes"] = [1000, 5000]
    quick["query_types"] = ["fact", "relation"]
    quick["num_queries_per_setting"] = 5
    quick["retrieval_topk_values"] = [50]
    quick["relation_expansion_depth_values"] = [0, 1]
    quick["rerank_topk_input_values"] = ["all", 50]
    return quick


def _warmup_store(store, embed_client) -> None:
    try:
        query_vector = embed_client.embed("Warmup query for motivation memory selection.")
        store.search(query_vector, 8)
    except Exception:
        pass


def run_profile(config_path: Path, quick: bool = False) -> Dict[str, Path]:
    config = _load_config(config_path)
    if quick:
        config = apply_quick_overrides(config)

    output_dir = PROJECT_ROOT / config.get("output_dir", "outputs/motivation_memory_selection")
    paths = ensure_output_layout(output_dir)
    raw_profile_path = paths["raw_profile"]
    phase_trace_path = paths["phase_trace"]
    if raw_profile_path.exists():
        raw_profile_path.unlink()
    if phase_trace_path.exists():
        phase_trace_path.unlink()

    embed_client = _resolve_embed_client(config)
    llm_client = _resolve_llm_client(config)
    reranker_client = _resolve_reranker_client(config)
    seed = int(config.get("seed", 42))

    run_log = {
        "config_path": str(config_path),
        "quick_mode": quick,
        "embedding_mode": getattr(embed_client, "mode", "hash_embedding"),
        "llm_mode": getattr(llm_client, "mode", "mock"),
        "reranker_mode": getattr(reranker_client, "mode", "heuristic"),
        "agent_execution_enabled": bool(config.get("enable_agent_execution", False)),
        "agent_execution_backend": str(config.get("agent_execution_backend", "disabled")),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (paths["logs_dir"] / "run_config.json").write_text(json.dumps(run_log, indent=2, ensure_ascii=False), encoding="utf-8")

    all_rows: List[dict] = []
    for memory_size in config.get("memory_sizes", [1000, 5000]):
        memories, memory_source = _load_memories_for_setting(int(memory_size), config, seed=seed)
        # Keep a metadata mirror next to the vectors so the vector store can load it
        write_jsonl(_metadata_path(int(memory_size), config), memories)
        store, index_embedding_mode = _ensure_index(memories, int(memory_size), config)
        _warmup_store(store, embed_client)
        run_log["index_embedding_mode"] = index_embedding_mode
        run_log["memory_source"] = memory_source

        for query_type in config.get("query_types", ["fact", "relation"]):
            workload = build_query_workload(
                memories=memories,
                query_type=query_type,
                num_queries=int(config.get("num_queries_per_setting", 100)),
                seed=seed,
            )
            for baseline_mode, retrieval_topk, depth, rerank_input in _iter_settings(config):
                rows = []
                for query in workload:
                    rows.append(
                        _run_single_query(
                            query=query,
                            memories=memories,
                            store=store,
                            embed_client=embed_client,
                            llm_client=llm_client,
                            reranker_client=reranker_client,
                            config=config,
                            baseline_mode=baseline_mode,
                            memory_size=int(memory_size),
                            retrieval_topk=int(retrieval_topk),
                            relation_expansion_depth=int(depth),
                            rerank_topk_input=rerank_input,
                        )
                    )
                phase_rows = []
                for row in rows:
                    phase_rows.extend(row.pop("phase_trace_rows", []))
                append_jsonl(raw_profile_path, rows)
                if phase_rows:
                    append_jsonl(phase_trace_path, phase_rows)
                all_rows.extend(rows)

    raw_df = pd.DataFrame(all_rows)
    write_summary_files(raw_df, paths["summary_csv"], paths["summary_md"])
    make_plots(raw_profile_path, paths["figures_dir"])

    clean = raw_df[raw_df["error"].fillna("") == ""]
    final_log = {
        "rows_total": int(len(raw_df)),
        "rows_clean": int(len(clean)),
        "rows_error": int(len(raw_df) - len(clean)),
        "avg_selection_latency_ms": float(clean["selection_total_latency_ms"].mean()) if not clean.empty else 0.0,
        "avg_agent_execution_latency_ms": float(clean["agent_execution_total_latency_ms"].mean()) if "agent_execution_total_latency_ms" in clean and not clean.empty else 0.0,
        "avg_total_response_latency_ms": float(clean["total_response_latency_ms"].mean()) if "total_response_latency_ms" in clean and not clean.empty else 0.0,
        "p95_selection_latency_ms": float(clean["selection_total_latency_ms"].quantile(0.95)) if not clean.empty else 0.0,
        "p95_total_response_latency_ms": float(clean["total_response_latency_ms"].quantile(0.95)) if "total_response_latency_ms" in clean and not clean.empty else 0.0,
    }
    (paths["logs_dir"] / "run_result.json").write_text(json.dumps(final_log, indent=2, ensure_ascii=False), encoding="utf-8")
    return paths
