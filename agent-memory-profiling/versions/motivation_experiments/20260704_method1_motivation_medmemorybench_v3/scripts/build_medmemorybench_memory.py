#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from medmmb_common import (
    VERSION_ROOT,
    count_tokens,
    first_present,
    infer_persona_id,
    infer_role,
    infer_session_id,
    infer_text,
    infer_timestamp,
    infer_turn_id,
    load_yaml,
    parse_maybe_json,
    resolve_path,
    session_sort_key,
    write_jsonl,
)
from probe_medmemorybench_schema import _candidate_files, _classify_file, _read_table


def _session_type(path: Path, row: dict[str, Any]) -> str:
    explicit = str(first_present(row, ["session_type"], "") or "").strip()
    if explicit in {"medical", "noise_family", "noise_health"}:
        return explicit
    source = " ".join([str(path).lower(), str(first_present(row, ["session_type", "type", "category"], "")).lower()])
    if "noise" in source and "family" in source:
        return "noise_family"
    if "noise" in source and "health" in source:
        return "noise_health"
    if "noise" in source:
        return "noise_health"
    if "medical" in source or "dialog" in source:
        return "medical"
    return "unknown"


def _dialogue_rows(data_root: Path, language: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidate_paths = _candidate_files(data_root, language)
    dialogue_paths: list[Path] = []
    for path in candidate_paths:
        try:
            df = _read_table(path)
        except Exception:
            continue
        kind = _classify_file(path, list(df.columns))
        if kind in {"dialogues", "dialogues_with_noise"}:
            dialogue_paths.append(path)
    if any("dialogues_with_noise" in path.name.lower() for path in dialogue_paths):
        selected_paths = [path for path in dialogue_paths if "dialogues_with_noise" in path.name.lower()]
    else:
        selected_paths = dialogue_paths
    for path in selected_paths:
        try:
            df = _read_table(path)
        except Exception:
            continue
        kind = _classify_file(path, list(df.columns))
        if kind not in {"dialogues", "dialogues_with_noise"}:
            continue
        for idx, record in enumerate(df.to_dict(orient="records")):
            text = infer_text(record)
            if not text:
                continue
            persona = infer_persona_id(record)
            session = infer_session_id(record)
            turn = infer_turn_id(record, idx)
            rows.append(
                {
                    "persona_id": persona,
                    "session_id": session,
                    "turn_id": turn,
                    "role": infer_role(record),
                    "text": text,
                    "timestamp": infer_timestamp(record, f"session={session};turn={turn}"),
                    "event_id": first_present(record, ["event_id", "event", "event_idx"], None),
                    "session_type": _session_type(path, record),
                    "knowledge_points": parse_maybe_json(first_present(record, ["knowledge_points", "key_points"], [])) or [],
                    "event_info": parse_maybe_json(first_present(record, ["event_info", "event_metadata", "metadata"], {})) or {},
                    "source_record_ids": [f"{path.name}:{idx}"],
                    "source_file": str(path),
                    "source_row_index": idx,
                }
            )
    rows.sort(key=lambda row: (str(row["persona_id"]), session_sort_key(row["session_id"]), session_sort_key(row["turn_id"]), row["source_file"], row["source_row_index"]))
    return rows


def _make_memory_id(persona: Any, session: Any, exchange: int, suffix: str = "") -> str:
    base = f"medmmb_en_p{persona}_s{session}_x{exchange}"
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in base)
    return safe + suffix


def _turn_pair_items(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["persona_id"]), str(row["session_id"]))].append(row)
    output: list[dict[str, Any]] = []
    rank = 0
    for (persona, session), items in sorted(grouped.items(), key=lambda item: (str(item[0][0]), session_sort_key(item[0][1]))):
        items.sort(key=lambda row: session_sort_key(row["turn_id"]))
        idx = 0
        exchange = 0
        while idx < len(items):
            current = items[idx]
            nxt = items[idx + 1] if idx + 1 < len(items) else None
            paired = False
            pair_rows = [current]
            if current["role"] == "patient" and nxt and nxt["role"] == "doctor":
                pair_rows.append(nxt)
                paired = True
                idx += 2
            else:
                idx += 1
            if paired:
                text = f"[Patient | session={session} | turn={current['turn_id']}]: {current['text']}\n[Doctor  | session={session} | turn={nxt['turn_id']}]: {nxt['text']}"
                memory_kind = "turn_pair"
                pairing_status = "paired"
            else:
                role_label = "Patient" if current["role"] == "patient" else "Doctor" if current["role"] == "doctor" else "Speaker"
                text = f"[{role_label} | session={session} | turn={current['turn_id']}]: {current['text']}"
                memory_kind = "singleton_turn"
                pairing_status = "singleton"
            kp: list[Any] = []
            event_info: dict[str, Any] = {}
            source_ids: list[str] = []
            for item in pair_rows:
                value = item.get("knowledge_points")
                if isinstance(value, list):
                    kp.extend(value)
                elif value:
                    kp.append(value)
                if isinstance(item.get("event_info"), dict):
                    event_info.update(item["event_info"])
                source_ids.extend(item.get("source_record_ids", []))
            memory = {
                "memory_id": _make_memory_id(persona, session, exchange),
                "dataset": "medmemorybench_en",
                "persona_id": current["persona_id"],
                "session_id": current["session_id"],
                "exchange_id": exchange,
                "timestamp": current["timestamp"],
                "event_id": current.get("event_id"),
                "session_type": current.get("session_type", "unknown"),
                "memory_kind": memory_kind,
                "text": text,
                "payload_bytes": len(text.encode("utf-8")),
                "token_count": count_tokens(text),
                "knowledge_points": kp,
                "event_info": event_info,
                "source_record_ids": source_ids,
                "pairing_status": pairing_status,
                "chronological_rank": rank,
            }
            output.append(memory)
            rank += 1
            exchange += 1
    return output


def _session_summary_items(rows: list[dict[str, Any]], token_cap: int) -> list[dict[str, Any]]:
    turn_pairs = _turn_pair_items(rows)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in turn_pairs:
        grouped[(str(row["persona_id"]), str(row["session_id"]))].append(row)
    output: list[dict[str, Any]] = []
    rank = 0
    for (persona, session), items in sorted(grouped.items(), key=lambda item: (str(item[0][0]), session_sort_key(item[0][1]))):
        text_parts: list[str] = []
        tokens = 0
        for item in items:
            item_tokens = str(item["text"]).split()
            remaining = max(0, token_cap - tokens)
            if remaining <= 0:
                break
            text_parts.append(" ".join(item_tokens[:remaining]))
            tokens += min(len(item_tokens), remaining)
        text = "\n".join(text_parts).strip()
        if not text:
            continue
        output.append(
            {
                "memory_id": _make_memory_id(persona, session, 0, suffix="_session_summary_proxy"),
                "dataset": "medmemorybench_en",
                "persona_id": items[0]["persona_id"],
                "session_id": items[0]["session_id"],
                "exchange_id": 0,
                "timestamp": items[0]["timestamp"],
                "event_id": items[0].get("event_id"),
                "session_type": items[0].get("session_type", "unknown"),
                "memory_kind": "session_summary_proxy",
                "text": text,
                "payload_bytes": len(text.encode("utf-8")),
                "token_count": count_tokens(text),
                "knowledge_points": [],
                "event_info": {},
                "source_record_ids": [source for item in items for source in item.get("source_record_ids", [])],
                "pairing_status": "singleton",
                "chronological_rank": rank,
            }
        )
        rank += 1
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MedMemoryBench memory items.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--granularity", choices=["turn_pair", "session_summary_proxy"], default="turn_pair")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)
    data_root = resolve_path(config["data_root"], VERSION_ROOT)
    if not data_root.exists():
        raise FileNotFoundError(f"Official MedMemoryBench data root not found: {data_root}")
    rows = _dialogue_rows(data_root, str(config.get("language", "en")))
    if not rows:
        raise RuntimeError("No dialogue rows with text could be recovered from official data.")
    if args.granularity == "turn_pair":
        memories = _turn_pair_items(rows)
    else:
        memories = _session_summary_items(rows, int(config.get("session_summary_token_cap", 768)))
    if not memories:
        raise RuntimeError("No memory items were produced.")
    write_jsonl(args.output, memories)

    persona_counts = Counter(str(row["persona_id"]) for row in memories)
    kind_counts = Counter(str(row["memory_kind"]) for row in memories)
    lines = [
        "# Memory Item and Evidence Audit",
        "",
        "This report is initialized by memory construction and completed by `build_evidence_mapping.py`.",
        "",
        f"Granularity: `{args.granularity}`",
        f"Memory items: {len(memories)}",
        f"Personas: {len(persona_counts)}",
        "",
        "## Memory Kinds",
        "",
    ]
    for key, count in sorted(kind_counts.items()):
        lines.append(f"- `{key}`: {count}")
    report = resolve_path("reports/02_memory_item_and_evidence_audit.md")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"memory_items={resolve_path(args.output)}")


if __name__ == "__main__":
    from collections import Counter

    main()
