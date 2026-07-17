from __future__ import annotations

import time
from typing import Dict, List, Tuple


def _sleep_ms(duration_ms: float) -> None:
    if duration_ms > 0:
        time.sleep(duration_ms / 1000.0)


def _build_agent_prompt(query_text: str, context_prompt: str, step_name: str) -> str:
    return (
        f"Agent step: {step_name}\n"
        f"User query: {query_text}\n\n"
        f"Available memory context:\n{context_prompt}\n\n"
        "Produce a concise intermediate result for this step."
    )


def _run_calibrated_mock_agent(
    query_text: str,
    selected: List[Tuple[dict, float]],
    config: dict,
) -> dict:
    step_count = int(config.get("agent_execution_steps", 3))
    tool_calls_per_step = int(config.get("agent_tool_calls_per_step", 1))
    planning_ms = float(config.get("agent_mock_planning_ms", 4.0))
    reasoning_ms = float(config.get("agent_mock_reasoning_ms_per_step", 8.0)) * max(step_count, 1)
    tool_ms = (
        float(config.get("agent_mock_tool_ms_per_call", 3.0))
        * max(step_count, 1)
        * max(tool_calls_per_step, 0)
    )
    output_ms = float(config.get("agent_mock_output_ms", 12.0))

    _sleep_ms(planning_ms)
    _sleep_ms(tool_ms)
    _sleep_ms(reasoning_ms)
    _sleep_ms(output_ms)

    answer = f"mock agent answer using {len(selected)} selected memories for: {query_text[:80]}"
    return {
        "answer": answer,
        "agent_planning_latency_ms": planning_ms,
        "agent_tool_execution_latency_ms": tool_ms,
        "agent_reasoning_latency_ms": reasoning_ms,
        "agent_output_latency_ms": output_ms,
        "agent_steps": step_count,
        "agent_tool_calls": step_count * tool_calls_per_step,
        "agent_backend_mode": "calibrated_mock",
    }


def _run_llm_agent(
    query_text: str,
    context_prompt: str,
    selected: List[Tuple[dict, float]],
    llm_client,
    config: dict,
    phase_contexts: Dict[str, str] | None = None,
    phase_selected: Dict[str, List[Tuple[dict, float]]] | None = None,
) -> dict:
    step_count = int(config.get("agent_execution_steps", 3))
    enable_intermediate_llm = bool(config.get("agent_enable_intermediate_llm_steps", False))
    tool_calls_per_step = int(config.get("agent_tool_calls_per_step", 1))
    planning_rounds = int(
        config.get(
            "agent_planning_llm_rounds",
            1 if enable_intermediate_llm else 0,
        )
    )
    reasoning_rounds = int(
        config.get(
            "agent_reasoning_llm_rounds",
            max(step_count - 1, 0) if enable_intermediate_llm else 0,
        )
    )
    output_rounds = int(config.get("agent_output_llm_rounds", 1))

    phase_contexts = dict(phase_contexts or {})
    phase_selected = dict(phase_selected or {})
    planning_context = phase_contexts.get("planning", context_prompt)
    reasoning_context = phase_contexts.get("reasoning", context_prompt)
    output_context = phase_contexts.get("output", context_prompt)

    start = time.perf_counter()
    planning_prompt = _build_agent_prompt(query_text, planning_context, "planning")
    planning_trace = ""
    for round_index in range(max(planning_rounds, 0)):
        suffix = f"\nPlanning round {round_index + 1}."
        planning_trace = llm_client.generate(planning_prompt + suffix)
    planning_latency = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    # Tool execution is represented by deterministic memory inspections. This
    # keeps the profiler focused on downstream agent flow without external IO.
    inspected = []
    tool_memory_pool = phase_selected.get("reasoning") or phase_selected.get("planning") or selected
    for step_index in range(max(step_count, 1)):
        for memory, score in tool_memory_pool[: max(tool_calls_per_step, 0)]:
            inspected.append((step_index, memory.get("memory_id", ""), score))
    tool_latency = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    reasoning_trace = ""
    for round_index in range(max(reasoning_rounds, 0)):
        reasoning_prompt = (
            _build_agent_prompt(query_text, reasoning_context, f"reasoning_{round_index + 1}")
            + f"\nObserved tool results: {len(inspected)} memory/tool observations."
        )
        if planning_trace:
            reasoning_prompt += f"\nPlanning trace:\n{planning_trace}"
        reasoning_trace = llm_client.generate(reasoning_prompt)
    reasoning_latency = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    final_prompt = (
        f"{output_context}\n\n"
        f"Agent inspected {len(inspected)} memory/tool observations. "
        "Now produce the final user-facing answer."
    )
    if planning_trace:
        final_prompt += f"\n\nPlanning trace:\n{planning_trace}"
    if reasoning_trace:
        final_prompt += f"\n\nReasoning trace:\n{reasoning_trace}"
    answer = ""
    for round_index in range(max(output_rounds, 0)):
        suffix = f"\nOutput round {round_index + 1}."
        answer = llm_client.generate(final_prompt + suffix)
    output_latency = (time.perf_counter() - start) * 1000.0

    return {
        "answer": answer,
        "agent_planning_latency_ms": planning_latency,
        "agent_tool_execution_latency_ms": tool_latency,
        "agent_reasoning_latency_ms": reasoning_latency,
        "agent_output_latency_ms": output_latency,
        "agent_steps": step_count,
        "agent_tool_calls": len(inspected),
        "agent_backend_mode": getattr(llm_client, "mode", "llm"),
    }


def execute_agent_response(
    query_text: str,
    context_prompt: str,
    selected: List[Tuple[dict, float]],
    llm_client,
    config: dict,
    phase_contexts: Dict[str, str] | None = None,
    phase_selected: Dict[str, List[Tuple[dict, float]]] | None = None,
) -> dict:
    backend = str(config.get("agent_execution_backend", "llm")).strip()
    if backend == "calibrated_mock":
        return _run_calibrated_mock_agent(query_text, selected, config)
    if backend == "mock":
        return {
            "answer": f"mock agent answer using {len(selected)} selected memories",
            "agent_planning_latency_ms": 0.0,
            "agent_tool_execution_latency_ms": 0.0,
            "agent_reasoning_latency_ms": 0.0,
            "agent_output_latency_ms": 0.0,
            "agent_steps": int(config.get("agent_execution_steps", 1)),
            "agent_tool_calls": 0,
            "agent_backend_mode": "mock",
        }
    return _run_llm_agent(
        query_text,
        context_prompt,
        selected,
        llm_client,
        config,
        phase_contexts=phase_contexts,
        phase_selected=phase_selected,
    )
