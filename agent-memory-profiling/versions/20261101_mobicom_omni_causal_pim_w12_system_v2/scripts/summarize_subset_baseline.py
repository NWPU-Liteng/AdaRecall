#!/usr/bin/env python3
"""Summarize a frozen subset against existing Reader and Judge artifacts."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def atomic_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_reader_rows(run_root: Path, users: list[str]) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for user_id in users:
        for suffix in (".json", ".partial.json"):
            path = run_root / f"user_{user_id}_shard_0{suffix}"
            if not path.exists():
                continue
            document = json.loads(path.read_text())
            for row in document.get("rows") or []:
                result[row["question_id"]] = row
    return result


def load_judge_rows(root: Path) -> dict[str, dict]:
    return {path.stem: json.loads(path.read_text())
            for path in root.glob("*/*.json") if path.is_file()}


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def timing(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
    }


def is_correct(row: dict | None) -> bool:
    return bool(row and row.get("parse_valid") and row.get("judge_label") == "CORRECT")


def summarize(cohort_path: Path, prepared: Path, judge_root: Path,
              reader_root: Path) -> dict[str, Any]:
    cohort = json.loads(cohort_path.read_text())
    users = sorted(cohort["question_ids_by_user"], key=int)
    question_ids = [identity for user_id in users
                    for identity in cohort["question_ids_by_user"][user_id]]
    manifests = {row["question_id"]: row for row in load_jsonl(prepared / "manifests.jsonl")}
    labels = {row["question_id"]: row for row in load_jsonl(prepared / "labels.jsonl")}
    readers = load_reader_rows(reader_root, users)
    judges = load_judge_rows(judge_root)
    selected = set(question_ids)
    judged_ids = selected & set(judges)

    def block(ids: set[str]) -> dict[str, Any]:
        judged = ids & set(judges)
        correct = sum(is_correct(judges[identity]) for identity in judged)
        return {
            "questions": len(ids),
            "judged": len(judged),
            "correct": correct,
            "success_rate_judged": correct / len(judged) if judged else None,
        }

    by_user = {user_id: block(set(cohort["question_ids_by_user"][user_id]))
               for user_id in users}
    by_split: dict[str, dict] = {}
    for split in sorted({manifests[identity]["split"] for identity in selected}):
        by_split[split] = block({identity for identity in selected
                                 if manifests[identity]["split"] == split})
    by_type: dict[str, dict] = {}
    for question_type in sorted({labels[identity]["question_type"] for identity in selected}):
        ids = {identity for identity in selected
               if labels[identity]["question_type"] == question_type}
        by_type[question_type] = block(ids)

    user_set = set(users)
    population_ids = {
        identity for identity, manifest in manifests.items()
        if str(manifest["person_id"]) in user_set
        and manifest["split"] in {"development", "validation"}
    }
    population_by_type = Counter(labels[identity]["question_type"]
                                 for identity in population_ids)
    population_total = len(population_ids)
    weighted_numerator = 0.0
    for question_type, stats in by_type.items():
        if stats["success_rate_judged"] is None:
            continue
        weighted_numerator += (population_by_type[question_type] / population_total
                               * stats["success_rate_judged"])

    latency_names = ("service_seconds", "ttft_seconds", "prefill_seconds",
                     "decode_seconds", "input_tokens")
    latency = {}
    for name in latency_names:
        values = [float(readers[identity][name]) for identity in question_ids
                  if identity in readers and isinstance(readers[identity].get(name), (int, float))]
        latency[name] = timing(values)

    overall = block(selected)
    return {
        "schema": "mobilemem-omni-subset-baseline-v1",
        "subset": {
            "name": cohort.get("subset", {}).get("name"),
            "cohort": str(cohort_path),
            "questions": len(question_ids),
            "users": users,
            "reader_rows": len(selected & set(readers)),
            "judge_rows": len(judged_ids),
        },
        "overall": overall,
        "by_user": by_user,
        "by_split": by_split,
        "by_type": by_type,
        "population_reference": {
            "slice": "same users, development+validation",
            "questions": population_total,
            "by_type": dict(sorted(population_by_type.items())),
            "natural_weighted_success": weighted_numerator,
        },
        "latency": latency,
        "boundary": "descriptive baseline from existing host BF16 artifacts; not a PIM measurement",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--judge-rows", type=Path, required=True)
    parser.add_argument("--reader-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.cohort.resolve(), args.prepared.resolve(),
                       args.judge_rows.resolve(), args.reader_root.resolve())
    atomic_write(args.output.resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
