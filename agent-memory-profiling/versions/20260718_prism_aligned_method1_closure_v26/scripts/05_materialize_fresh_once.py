from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VERSION_ROOT = Path(__file__).resolve().parents[1]
SOURCE = VERSION_ROOT / "artifacts" / "fresh" / "source" / "locomo10.json"
LOCK = VERSION_ROOT / "artifacts" / "fresh" / "fresh_read_lock.json"
RECEIPT = VERSION_ROOT / "artifacts" / "fresh" / "fresh_read_receipt.json"
OUTPUT = VERSION_ROOT / "artifacts" / "fresh" / "canonical"
SESSION = re.compile(r"^session_(\d+)$")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def main() -> None:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if lock.get("status") != "LOCKED_FOR_SINGLE_FRESH_READ" or lock.get("formal_runs_allowed") != 1:
        raise RuntimeError("fresh read lock is absent or invalid")
    if RECEIPT.exists() or OUTPUT.exists():
        raise RuntimeError("fresh source has already been consumed; formal rerun forbidden")
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if digest != "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4":
        raise RuntimeError("fresh source hash changed")
    started = datetime.now(timezone.utc).isoformat()
    with RECEIPT.open("x", encoding="utf-8") as handle:
        json.dump(
            {
                "run_id": "v26_locomo_fresh_once_20260718",
                "status": "STARTED_GOLD_READ_CONSUMED",
                "started_at_utc": started,
                "source_sha256": digest,
                "formal_read_number": 1,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    with SOURCE.open("r", encoding="utf-8") as handle:
        samples = json.load(handle)
    if not isinstance(samples, list) or not samples:
        raise RuntimeError("unexpected LoCoMo root schema")
    memories: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    seen_memory = set()
    excluded_no_evidence = 0
    category_counts: dict[str, int] = {}
    for sample in samples:
        sample_id = str(sample["sample_id"])
        conversation = sample["conversation"]
        sessions = sorted(
            ((int(match.group(1)), key) for key in conversation for match in [SESSION.match(key)] if match),
            key=lambda value: value[0],
        )
        if not sessions:
            raise RuntimeError(f"no sessions in {sample_id}")
        arrival = 0
        local_ids = set()
        for session_number, key in sessions:
            turns = conversation[key]
            if not isinstance(turns, list):
                raise RuntimeError(f"invalid session {sample_id}/{key}")
            for turn_index, turn in enumerate(turns):
                arrival += 1
                dia_id = str(turn["dia_id"])
                memory_id = f"{sample_id}:{dia_id}"
                if memory_id in seen_memory:
                    raise RuntimeError(f"duplicate memory ID {memory_id}")
                seen_memory.add(memory_id)
                local_ids.add(dia_id)
                text = f"{turn.get('speaker', '')}: {turn.get('text', '')}".strip()
                caption = str(turn.get("blip_caption", "") or "").strip()
                if caption:
                    text += f"\nImage caption: {caption}"
                memories.append(
                    {
                        "memory_id": memory_id,
                        "namespace_id": sample_id,
                        "content": text,
                        "arrival_order": arrival,
                        "eligible_from_order": 0,
                        "metadata": {"session": session_number, "turn": turn_index, "dia_id": dia_id},
                    }
                )
        for qa_index, qa in enumerate(sample.get("qa", [])):
            evidence = qa.get("evidence", []) or []
            if isinstance(evidence, str):
                evidence = [evidence]
            evidence = [str(value) for value in evidence]
            if not evidence:
                excluded_no_evidence += 1
                continue
            missing = sorted(set(evidence) - local_ids)
            if missing:
                raise RuntimeError(f"gold evidence IDs missing from {sample_id}: {missing[:5]}")
            query_id = f"{sample_id}:qa:{qa_index}"
            category = str(qa.get("category", "unknown"))
            category_counts[category] = category_counts.get(category, 0) + 1
            gold = [f"{sample_id}:{value}" for value in evidence]
            queries.append(
                {
                    "query_id": query_id,
                    "namespace_id": sample_id,
                    "query": str(qa["question"]),
                    "cutoff_order": arrival,
                    "split": "fresh_holdout",
                    "metadata": {"query_type": category, "source_qa_index": qa_index},
                }
            )
            evaluations.append(
                {
                    "query_id": query_id,
                    "evidence_groups": [[value] for value in gold],
                    "gold_memory_ids": gold,
                    "gold_answers": [str(qa.get("answer", ""))],
                }
            )
    if len(queries) != len(evaluations) or not queries:
        raise RuntimeError("fresh canonical query/evaluation mismatch")
    write_jsonl(OUTPUT / "memory_records.jsonl", memories)
    write_jsonl(OUTPUT / "queries.jsonl", queries)
    write_jsonl(OUTPUT / "evaluation.jsonl", evaluations)
    summary = {
        "sample_count": len(samples),
        "memory_count": len(memories),
        "included_query_count": len(queries),
        "excluded_no_evidence_query_count": excluded_no_evidence,
        "query_type_counts": category_counts,
        "cross_namespace_gold_violations": 0,
    }
    write_json(OUTPUT / "manifest.json", summary)
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    receipt.update({"status": "COMPLETED_GOLD_READ_ONCE", "completed_at_utc": datetime.now(timezone.utc).isoformat(), "canonical": summary})
    write_json(RECEIPT, receipt)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
