#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from medmmb_common import flatten_strings, normalize_text, parse_maybe_json, read_jsonl, resolve_path, write_csv, write_jsonl


def _event_ids(value: Any) -> set[str]:
    ids: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if "event" in str(key).lower() or "session" in str(key).lower():
                if isinstance(item, (str, int, float)):
                    ids.add(str(item))
            ids |= _event_ids(item)
    elif isinstance(value, list):
        for item in value:
            ids |= _event_ids(item)
    elif isinstance(value, (str, int, float)):
        text = str(value)
        if text.strip():
            ids.add(text)
    return ids


def _source_key_points(value: Any) -> list[Any]:
    parsed = parse_maybe_json(value)
    if parsed is None:
        return []
    if isinstance(parsed, list):
        return parsed
    return [parsed]


def _source_content(value: Any) -> str:
    if isinstance(value, dict):
        for key in ["content", "text", "event", "name"]:
            if value.get(key):
                return str(value[key])
    strings = flatten_strings(value)
    return " ".join(strings)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic query-to-memory evidence mapping.")
    parser.add_argument("--memory-items", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--output-root", default=".")
    args = parser.parse_args()

    output_root = resolve_path(args.output_root)
    memories = read_jsonl(args.memory_items)
    queries = read_jsonl(args.queries)
    if not memories:
        raise RuntimeError("No memory items found.")
    if not queries:
        raise RuntimeError("No queries found.")

    by_session: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_session_medical: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_event: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for memory in memories:
        persona = str(memory["persona_id"])
        session_id = str(memory.get("session_id", "")).strip()
        if session_id:
            by_session[(persona, session_id)].append(memory)
            if str(memory.get("session_type", "")) == "medical":
                by_session_medical[(persona, session_id)].append(memory)
        event_id = str(memory.get("event_id", "")).strip()
        if event_id and event_id.lower() not in {"none", "nan"}:
            by_event[(persona, event_id)].append(memory)

    mapping_rows: list[dict[str, Any]] = []
    method_counts: Counter[str] = Counter()
    for query in queries:
        persona = str(query["persona_id"])
        source_kps = _source_key_points(query.get("source_key_points"))
        gold_ids: list[str] = []
        methods: list[str] = []
        for source in source_kps:
            source_norm = normalize_text(_source_content(source))
            matched: list[dict[str, Any]] = []
            source_obj = source if isinstance(source, dict) else None
            if source_obj and source_obj.get("session_id") is not None:
                key = (persona, str(source_obj.get("session_id")))
                matched = by_session_medical.get(key) or by_session.get(key, [])
                if matched:
                    methods.append("session_id_exact")
            if not matched:
                for event_id in _event_ids(source):
                    matched = by_event.get((persona, str(event_id)), [])
                    if matched:
                        methods.append("event_id_exact")
                        break
            if not matched:
                methods.append("unresolved")
            for memory in matched:
                mid = str(memory["memory_id"])
                if mid not in gold_ids:
                    gold_ids.append(mid)
        mapped_count = sum(1 for method in methods if method != "unresolved")
        for method in methods:
            method_counts[method] += 1
        mapping_rows.append(
            {
                "query_id": str(query["query_id"]),
                "persona_id": query["persona_id"],
                "query_session_id": query["query_session_id"],
                "source_key_point_count": len(source_kps),
                "mapped_key_point_count": mapped_count,
                "gold_memory_ids": gold_ids,
                "mapping_methods": methods,
                "is_eligible_for_evidence_metrics": bool(gold_ids and mapped_count > 0),
            }
        )

    write_jsonl(output_root / "aggregate" / "query_evidence_mapping.jsonl", mapping_rows)
    eligible = sum(1 for row in mapping_rows if row["is_eligible_for_evidence_metrics"])
    coverage_rows = [
        {
            "n_queries": len(mapping_rows),
            "eligible_queries": eligible,
            "evidence_mapping_coverage": eligible / float(len(mapping_rows)) if mapping_rows else 0.0,
            "method": method,
            "count": count,
        }
        for method, count in sorted(method_counts.items())
    ]
    if not coverage_rows:
        coverage_rows = [{"n_queries": len(mapping_rows), "eligible_queries": eligible, "evidence_mapping_coverage": 0.0, "method": "none", "count": 0}]
    write_csv(output_root / "aggregate" / "query_evidence_mapping_coverage.csv", coverage_rows)

    report_path = output_root / "reports" / "02_memory_item_and_evidence_audit.md"
    previous = report_path.read_text(encoding="utf-8") if report_path.exists() else "# Memory Item and Evidence Audit\n"
    lines = [
        previous.rstrip(),
        "",
        "## Evidence Mapping",
        "",
        f"Queries: {len(mapping_rows)}",
        f"Eligible evidence-metric queries: {eligible}",
        f"Evidence Mapping Coverage: {eligible / float(len(mapping_rows)) if mapping_rows else 0.0:.4f}",
        "",
        "### Mapping Methods",
        "",
    ]
    for method, count in sorted(method_counts.items()):
        lines.append(f"- `{method}`: {count}")
    if mapping_rows and eligible / float(len(mapping_rows)) < 0.60:
        lines.extend(
            [
                "",
                "Coverage is below 60%; evidence metrics must not be the sole primary result.",
            ]
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"evidence_mapping={output_root / 'aggregate' / 'query_evidence_mapping.jsonl'}")


if __name__ == "__main__":
    main()
