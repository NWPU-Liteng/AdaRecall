#!/usr/bin/env python3
"""Evaluate query-aware lexical handoff ordering on the recommended B8 path."""
from __future__ import annotations
import re
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ns = runpy.run_path(str(ROOT / "scripts/157_generate_v21_cap64_agent_answers_oof.py"), run_name="v21_lexical_handoff_helpers")
STOP = {"the", "a", "an", "is", "are", "was", "were", "what", "which", "who", "when", "where", "how", "does", "did", "and", "or", "of", "to", "in", "on", "for", "with", "from", "my", "this", "that"}

def lexical_order(query: dict, ids: list[str], memory_by_id: dict[str, dict]) -> list[str]:
    query_tokens = {token for token in re.findall(r"[a-z0-9]+", str(query.get("query_text", "")).lower()) if token not in STOP and len(token) > 2}
    scored = []
    for position, memory_id in enumerate(ids):
        memory_tokens = set(re.findall(r"[a-z0-9]+", str(memory_by_id[memory_id].get("text", "")).lower()))
        overlap = len(query_tokens.intersection(memory_tokens))
        scored.append((-overlap, position, memory_id))
    return [memory_id for _score, _position, memory_id in sorted(scored)]

ns["order_context_ids"] = lexical_order
if __name__ == "__main__":
    ns["main"]()
