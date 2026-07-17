#!/usr/bin/env python3
"""Evaluate structured, query-aware evidence packing on the B8 handoff."""
from __future__ import annotations
import re
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ns = runpy.run_path(str(ROOT / "scripts/157_generate_v21_cap64_agent_answers_oof.py"), run_name="v21_evidence_context_helpers")
STOP = {"the", "a", "an", "is", "are", "was", "were", "what", "which", "who", "when", "where", "how", "does", "did", "and", "or", "of", "to", "in", "on", "for", "with", "from", "my", "this", "that", "patient"}

def tokens(text: str) -> set[str]:
    return {value for value in re.findall(r"[a-z0-9]+", str(text).lower()) if value not in STOP and len(value) > 2}

def evidence_context(memory_by_id: dict[str, dict], ids: list[str], token_cap: int = 1800, query: dict | None = None) -> str:
    query_tokens = tokens(str((query or {}).get("query_text", "")))
    per_memory = max(48, token_cap // max(len(ids), 1))
    parts: list[str] = []
    used = 0
    for memory_id in ids:
        memory = memory_by_id[memory_id]
        header = f"[memory_id={memory_id} | session={memory.get('session_id')} | type={memory.get('session_type', 'unknown')}]"
        segments = []
        for point in memory.get("knowledge_points", []):
            if not isinstance(point, dict) or not point.get("content"):
                continue
            text = f"{point.get('name', '')}: {point.get('content', '')}"
            segments.append(text)
        raw_sentences = re.split(r"(?<=[.!?])\s+", str(memory.get("text", "")))
        segments.extend(sentence.strip() for sentence in raw_sentences if sentence.strip())
        ranked = sorted(enumerate(segments), key=lambda item: (-len(query_tokens.intersection(tokens(item[1]))), item[0]))
        words: list[str] = []
        for _position, segment in ranked:
            candidate = segment.split()
            if not candidate:
                continue
            remaining = max(0, per_memory - len(words))
            if remaining <= 0:
                break
            words.extend(candidate[:remaining])
        remaining_total = max(0, token_cap - used - len(header.split()))
        if remaining_total <= 0:
            break
        words = words[:remaining_total]
        parts.append(" ".join([header, *words]))
        used += len(header.split()) + len(words)
    return "\n\n".join(parts) if parts else "No retrieved memory."

def concise_prompt(row: dict, context: str) -> str:
    return ("You are answering a text-only healthcare memory benchmark. This is research data, not clinical advice.\n"
            "Use only the memory context. Do not use outside knowledge.\n"
            "Return the shortest answer phrase that directly answers the question. Preserve the exact number, unit, medication name, category, or option label found in memory. Do not add explanation, markdown, or a repeated question. If the context is insufficient, return exactly: insufficient information\n\n"
            f"Query type: {row.get('query_type', 'unknown')}\nUser query: {row.get('query_text', '')}\n\nMemory context:\n{context}\n\nAnswer phrase:")

ns["context"] = evidence_context
ns["prompt"] = concise_prompt
if __name__ == "__main__":
    ns["main"]()
