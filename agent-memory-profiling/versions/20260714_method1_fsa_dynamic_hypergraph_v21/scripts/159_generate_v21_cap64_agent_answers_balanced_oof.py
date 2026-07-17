#!/usr/bin/env python3
"""Run the existing Agent OOF protocol with balanced context packing."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ns = runpy.run_path(str(ROOT / "scripts/157_generate_v21_cap64_agent_answers_oof.py"), run_name="v21_balanced_agent_helpers")

def balanced_context(memory_by_id: dict[str, dict], ids: list[str], token_cap: int = 1800, query: dict | None = None) -> str:
    if not ids:
        return "No retrieved memory."
    headers = []
    bodies = []
    for memory_id in ids:
        memory = memory_by_id[memory_id]
        headers.append(f"[memory_id={memory_id} | session={memory.get('session_id')} | type={memory.get('session_type', 'unknown')}]")
        bodies.append(str(memory.get("text", "")).split())
    # Give every selected item a bounded foothold, then distribute the remainder.
    base = min(40, max(8, token_cap // max(len(ids), 1)))
    quotas = [base] * len(ids)
    remaining = max(0, token_cap - sum(len(h.split()) + q for h, q in zip(headers, quotas, strict=True)))
    per_item = min(48, remaining // max(len(ids), 1))
    quotas = [quota + per_item for quota in quotas]
    remaining -= per_item * len(ids)
    for index in range(min(len(ids), remaining)):
        quotas[index] += 1
    parts = []
    for header, words, quota in zip(headers, bodies, quotas, strict=True):
        if len(words) <= quota:
            snippet = words
        else:
            left = max(1, quota // 2)
            snippet = words[:left] + ["..."] + words[-max(1, quota - left - 1):]
        parts.append(" ".join([header, *snippet]))
    return "\n\n".join(parts)

ns["context"] = balanced_context

if __name__ == "__main__":
    ns["main"]()
