from __future__ import annotations

import json
from pathlib import Path


FORBIDDEN_RUNTIME_KEYS = {
    "gold_rating",
    "positive_parent_asin",
    "gold_review_title",
    "gold_review_text",
    "answer",
    "oracle",
}


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def audit_manifest(manifest_dir: Path) -> dict:
    runtime_path = manifest_dir / "runtime_manifest.jsonl"
    memory_path = manifest_dir / "memory_bank.jsonl"
    labels_path = manifest_dir / "evaluation_labels.jsonl"
    memories = {row["memory_id"]: row for row in iter_jsonl(memory_path)}
    tenant_errors = 0
    causal_errors = 0
    leakage_errors = 0
    event_count = 0
    for event in iter_jsonl(runtime_path):
        event_count += 1
        if FORBIDDEN_RUNTIME_KEYS.intersection(event):
            leakage_errors += 1
        serialized = json.dumps(event, ensure_ascii=False).lower()
        for forbidden in ("gold_", "oracle", "positive_parent_asin"):
            if forbidden in serialized:
                leakage_errors += 1
                break
        for memory_id in event.get("runtime_memory_ids", []):
            memory = memories.get(memory_id)
            if not memory:
                tenant_errors += 1
                continue
            if memory.get("tenant_id") != event.get("tenant_id"):
                tenant_errors += 1
            if event.get("protocol") == "causal_cross" and int(memory.get("timestamp") or 0) >= int(event.get("target_timestamp") or 0):
                causal_errors += 1
    label_count = sum(1 for _ in iter_jsonl(labels_path)) if labels_path.exists() else 0
    return {
        "manifest_dir": str(manifest_dir),
        "event_count": event_count,
        "label_count": label_count,
        "tenant_errors": tenant_errors,
        "causal_errors": causal_errors,
        "leakage_errors": leakage_errors,
        "passed": tenant_errors == 0 and causal_errors == 0 and leakage_errors == 0 and event_count == label_count,
    }
