#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from dataclasses import dataclass


PHASES = ("planning", "reasoning", "output")


@dataclass
class ControllerDecision:
    action: str
    retrieval_goal: str
    needed_evidence: list[str]
    reason: str
    confidence: float
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    raw_text: str


def _preview_retrieved_context(selected_context_preview: list[str]) -> str:
    if not selected_context_preview:
        return "none"
    return " || ".join(text[:120] for text in selected_context_preview[:2])


def _controller_prompt(phase: str, query_text: str, selected_context_preview: list[str]) -> str:
    preview = _preview_retrieved_context(selected_context_preview)
    retrieved_count = len(selected_context_preview)
    return (
        "Decide whether this agent phase needs another memory retrieval.\n"
        f"phase={phase}\n"
        f"query={query_text}\n"
        f"retrieved_count={retrieved_count}\n"
        f"retrieved_preview={preview}\n\n"
        "Output exactly one minified JSON object and nothing else.\n"
        'If retrieval is needed: {"action":"retrieve","retrieval_goal":"<short evidence need>"}\n'
        'If retrieval is not needed: {"action":"skip"}\n'
        "Rules:\n"
        "- Keep retrieval_goal under 12 words.\n"
        "- Use skip by default in output phase unless current evidence is clearly insufficient.\n"
        "- Do not answer the user query.\n"
        "- Do not explain your reasoning.\n"
    )


def _extract_json(text: str) -> dict | None:
    stripped = text.strip()
    for candidate in (stripped, *re.findall(r"\{.*\}", stripped, flags=re.DOTALL)):
        if not candidate:
            continue
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def run_controller(llm_client, phase: str, query_text: str, selected_context_preview: list[str]) -> ControllerDecision:
    response = llm_client.generate_with_stats(_controller_prompt(phase, query_text, selected_context_preview))
    payload = _extract_json(response["text"]) or {}
    action = str(payload.get("action") or "").strip().lower()
    normalized_text = str(response["text"]).upper()
    if action not in {"retrieve", "skip"}:
        if "SKIP" in normalized_text and "RETRIEVE" not in normalized_text:
            action = "skip"
        elif "RETRIEVE" in normalized_text:
            action = "retrieve"
    if action not in {"retrieve", "skip"}:
        action = "skip" if phase == "output" else "retrieve"
    retrieval_goal = str(payload.get("retrieval_goal") or "").strip()
    if action == "retrieve" and not retrieval_goal:
        retrieval_goal = query_text
    retrieval_goal = " ".join(retrieval_goal.split()[:12]).strip()
    needed_evidence = payload.get("needed_evidence") or []
    if not isinstance(needed_evidence, list):
        needed_evidence = []
    reason = str(payload.get("reason") or "").strip()
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return ControllerDecision(
        action=action,
        retrieval_goal=retrieval_goal,
        needed_evidence=[str(item) for item in needed_evidence[:8]],
        reason=reason,
        confidence=confidence,
        latency_ms=float(response.get("elapsed_ms", 0.0)),
        prompt_tokens=int(response.get("prompt_tokens", 0)),
        completion_tokens=int(response.get("completion_tokens", 0)),
        raw_text=str(response.get("text", "")),
    )
