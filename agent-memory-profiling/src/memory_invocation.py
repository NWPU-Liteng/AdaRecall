from datetime import date
from typing import Dict, List, TypedDict, Tuple

from langgraph.graph import END, START, StateGraph
from .timer import timer


MODULE_MEMORY_TYPES = {
    "planning": {"health_constraint", "profile", "planning", "feedback"},
    "reflection": {"feedback", "episodic", "planning"},
    "risk_control": {"health_constraint", "feedback", "profile"},
    "profile_update": {"profile", "feedback", "episodic"},
}

ENTITY_KEYWORDS = {
    "knee",
    "sleep",
    "weight",
    "running",
    "diet",
    "fatigue",
    "hydration",
    "recovery",
}
AGENT_QUERY_TEMPLATES = {
    "planning": [
        "Find long-term planning memories about recovery, sleep, and health constraints.",
        "Find planning notes that mention routine adjustment, fatigue, or training load.",
        "Find high-importance profile details that should constrain the next plan.",
    ],
    "reflection": [
        "Find episodic and feedback memories that summarize recent training outcomes.",
        "Find repeated patterns involving fatigue, sleep, or recovery that should be reflected on.",
        "Find notable planning changes that explain the current trend.",
    ],
    "risk_control": [
        "Find risk signals involving pain, fatigue, or abrupt volume increases.",
        "Find high-importance health constraints that should block unsafe advice.",
        "Find recent feedback that suggests overtraining or poor recovery.",
    ],
    "profile_update": [
        "Find long-term profile memories that may need to be updated or overwritten.",
        "Find recent feedback and episodic observations that changed the user baseline.",
        "Find recency-sensitive notes about diet, weight, sleep, or fatigue.",
    ],
}


def _recency_bonus(iso_date: str) -> float:
    try:
        memory_date = date.fromisoformat(iso_date)
        delta_days = max((date(2026, 6, 1) - memory_date).days, 0)
    except Exception:
        return 0.0
    return max(0.0, 0.1 - min(delta_days, 365) / 3650.0)


def filter_candidates(module: str, candidates: List[Tuple[dict, float]]) -> List[Tuple[dict, float]]:
    preferred = MODULE_MEMORY_TYPES.get(module)
    if not preferred:
        return candidates
    filtered = []
    for payload, score in candidates:
        if payload.get("memory_type") in preferred:
            filtered.append((payload, score))
    return filtered or candidates


def _infer_entities(query: str) -> List[str]:
    lowered = query.lower()
    return [entity for entity in sorted(ENTITY_KEYWORDS) if entity in lowered]


def build_memory_queries(query: str, module: str, memory_rounds: int) -> List[str]:
    queries = [query]
    entities = _infer_entities(query)
    if entities:
        joined_entities = ", ".join(entities)
        queries.append(f"{query} Focus on entity-specific memories about {joined_entities}.")
    for template in AGENT_QUERY_TEMPLATES.get(module, []):
        queries.append(f"{query} {template}")

    expanded: List[str] = []
    while len(expanded) < max(memory_rounds, 1):
        expanded.extend(queries)
    return expanded[: max(memory_rounds, 1)]


def rerank_candidates_for_query(
    module: str,
    query: str,
    candidates: List[Tuple[dict, float]],
) -> List[Tuple[dict, float, float]]:
    query_entities = set(_infer_entities(query))
    reranked = []
    for payload, vector_score in candidates:
        module_bonus = 0.15 if module in payload.get("module_tags", []) else 0.0
        importance = float(payload.get("importance", 0.0))
        entity_overlap = len(set(payload.get("entity_tags", [])) & query_entities)
        score = (
            float(vector_score)
            + 0.1 * importance
            + module_bonus
            + _recency_bonus(payload.get("time", ""))
            + 0.08 * entity_overlap
        )
        reranked.append((payload, vector_score, score))
    reranked.sort(key=lambda item: item[2], reverse=True)
    return reranked


def merge_selected_candidates(
    selected_groups: List[List[Tuple[dict, float, float]]],
) -> List[Tuple[dict, float, float]]:
    merged: Dict[str, Tuple[dict, float, float]] = {}
    for group in selected_groups:
        for payload, vector_score, score in group:
            memory_id = payload["memory_id"]
            current = merged.get(memory_id)
            if current is None or score > current[2]:
                merged[memory_id] = (payload, vector_score, score)
    return sorted(merged.values(), key=lambda item: item[2], reverse=True)


def package_context(query: str, top_memories: List[Tuple[dict, float, float]]) -> str:
    lines = [f"User query: {query}", "Relevant memories:"]
    for payload, _, score in top_memories:
        lines.append(
            f"- [{payload['memory_id']}] type={payload['memory_type']} score={score:.4f} "
            f"time={payload['time']} content={payload['content']}"
        )
    lines.append("Answer briefly with a helpful coaching response.")
    return "\n".join(lines)


class LangGraphMemoryState(TypedDict, total=False):
    query: str
    module: str
    candidate_k: int
    top_k: int
    memory_queries: List[str]
    current_round: int
    total_rounds: int
    round_selected: List[List[Tuple[dict, float, float]]]
    last_candidates: List[Tuple[dict, float]]
    last_filtered: List[Tuple[dict, float]]
    selected: List[Tuple[dict, float, float]]
    prompt: str
    answer: str
    prompt_chars: int
    stats: Dict[str, float]


def run_memory_pipeline(
    query: str,
    module: str,
    candidate_k: int,
    top_k: int,
    embed_client,
    llm_client,
    vector_store,
    execution_mode: str = "single",
    memory_rounds: int = 1,
) -> Dict[str, object]:
    stats: Dict[str, object] = {
        "t_embed": 0.0,
        "t_search": 0.0,
        "t_filter": 0.0,
        "t_rerank": 0.0,
        "t_pack": 0.0,
        "t_llm": 0.0,
    }
    query_count = max(memory_rounds, 1) if execution_mode == "agent" else 1
    query_plan = build_memory_queries(query, module, query_count)
    round_selected: List[List[Tuple[dict, float, float]]] = []
    last_candidates: List[Tuple[dict, float]] = []
    last_filtered: List[Tuple[dict, float]] = []

    for memory_query in query_plan:
        round_stats: Dict[str, float] = {}
        with timer(round_stats, "t_embed"):
            query_vector = embed_client.embed(memory_query)

        with timer(round_stats, "t_search"):
            candidates = vector_store.search(query_vector, candidate_k)

        with timer(round_stats, "t_filter"):
            filtered = filter_candidates(module, candidates)

        with timer(round_stats, "t_rerank"):
            reranked = rerank_candidates_for_query(module, memory_query, filtered)
            selected = reranked[:top_k]

        stats["t_embed"] += round_stats["t_embed"]
        stats["t_search"] += round_stats["t_search"]
        stats["t_filter"] += round_stats["t_filter"]
        stats["t_rerank"] += round_stats["t_rerank"]
        round_selected.append(selected)
        last_candidates = candidates
        last_filtered = filtered

    merged_selected = merge_selected_candidates(round_selected)
    selected = merged_selected[:top_k]

    with timer(stats, "t_pack"):
        prompt = package_context(query, selected)
        stats["prompt_chars"] = len(prompt)

    with timer(stats, "t_llm"):
        answer = llm_client.generate(prompt)

    stats["answer"] = answer
    stats["num_candidates"] = len(last_candidates)
    stats["num_after_filter"] = len(last_filtered)
    stats["num_memory_calls"] = len(query_plan)
    memory_time = stats["t_embed"] + stats["t_search"] + stats["t_filter"] + stats["t_rerank"] + stats["t_pack"]
    stats["t_memory"] = memory_time
    stats["memory_invocation_ratio"] = memory_time / max(memory_time + stats["t_llm"], 1e-9)
    stats["selected_memories"] = [payload["memory_id"] for payload, _, _ in selected]
    return stats


def run_langgraph_memory_pipeline(
    query: str,
    module: str,
    candidate_k: int,
    top_k: int,
    embed_client,
    llm_client,
    vector_store,
    memory_rounds: int = 4,
) -> Dict[str, object]:
    total_rounds = max(memory_rounds, 1)

    def plan_queries(state: LangGraphMemoryState) -> LangGraphMemoryState:
        query_plan = build_memory_queries(state["query"], state["module"], state["total_rounds"])
        return {
            "memory_queries": query_plan,
            "current_round": 0,
            "round_selected": [],
            "stats": {
                "t_embed": 0.0,
                "t_search": 0.0,
                "t_filter": 0.0,
                "t_rerank": 0.0,
                "t_pack": 0.0,
                "t_llm": 0.0,
            },
        }

    def retrieve_round(state: LangGraphMemoryState) -> LangGraphMemoryState:
        stats = dict(state["stats"])
        round_index = state["current_round"]
        memory_query = state["memory_queries"][round_index]
        round_stats: Dict[str, float] = {}

        with timer(round_stats, "t_embed"):
            query_vector = embed_client.embed(memory_query)
        with timer(round_stats, "t_search"):
            candidates = vector_store.search(query_vector, state["candidate_k"])
        with timer(round_stats, "t_filter"):
            filtered = filter_candidates(state["module"], candidates)
        with timer(round_stats, "t_rerank"):
            reranked = rerank_candidates_for_query(state["module"], memory_query, filtered)
            selected = reranked[: state["top_k"]]

        stats["t_embed"] += round_stats["t_embed"]
        stats["t_search"] += round_stats["t_search"]
        stats["t_filter"] += round_stats["t_filter"]
        stats["t_rerank"] += round_stats["t_rerank"]

        round_selected = list(state["round_selected"])
        round_selected.append(selected)
        return {
            "stats": stats,
            "round_selected": round_selected,
            "last_candidates": candidates,
            "last_filtered": filtered,
            "current_round": round_index + 1,
        }

    def route_after_retrieve(state: LangGraphMemoryState):
        if state["current_round"] < state["total_rounds"]:
            return "retrieve_round"
        return "package_context"

    def package_context_node(state: LangGraphMemoryState) -> LangGraphMemoryState:
        stats = dict(state["stats"])
        selected = merge_selected_candidates(state["round_selected"])[: state["top_k"]]
        pack_stats: Dict[str, float] = {}
        with timer(pack_stats, "t_pack"):
            prompt = package_context(state["query"], selected)
        stats["t_pack"] += pack_stats["t_pack"]
        return {
            "stats": stats,
            "selected": selected,
            "prompt": prompt,
            "prompt_chars": len(prompt),
        }

    def generate_answer(state: LangGraphMemoryState) -> LangGraphMemoryState:
        stats = dict(state["stats"])
        llm_stats: Dict[str, float] = {}
        with timer(llm_stats, "t_llm"):
            answer = llm_client.generate(state["prompt"])
        stats["t_llm"] += llm_stats["t_llm"]
        return {
            "stats": stats,
            "answer": answer,
        }

    graph = StateGraph(LangGraphMemoryState)
    graph.add_node("plan_queries", plan_queries)
    graph.add_node("retrieve_round", retrieve_round)
    graph.add_node("package_context", package_context_node)
    graph.add_node("generate_answer", generate_answer)
    graph.add_edge(START, "plan_queries")
    graph.add_edge("plan_queries", "retrieve_round")
    graph.add_conditional_edges("retrieve_round", route_after_retrieve)
    graph.add_edge("package_context", "generate_answer")
    graph.add_edge("generate_answer", END)

    app = graph.compile()
    final_state = app.invoke(
        {
            "query": query,
            "module": module,
            "candidate_k": candidate_k,
            "top_k": top_k,
            "total_rounds": total_rounds,
        }
    )
    stats = dict(final_state["stats"])
    memory_time = stats["t_embed"] + stats["t_search"] + stats["t_filter"] + stats["t_rerank"] + stats["t_pack"]
    stats["answer"] = final_state["answer"]
    stats["prompt_chars"] = final_state["prompt_chars"]
    stats["num_candidates"] = len(final_state.get("last_candidates", []))
    stats["num_after_filter"] = len(final_state.get("last_filtered", []))
    stats["num_memory_calls"] = final_state["current_round"]
    stats["t_memory"] = memory_time
    stats["memory_invocation_ratio"] = memory_time / max(memory_time + stats["t_llm"], 1e-9)
    stats["selected_memories"] = [payload["memory_id"] for payload, _, _ in final_state["selected"]]
    return stats
