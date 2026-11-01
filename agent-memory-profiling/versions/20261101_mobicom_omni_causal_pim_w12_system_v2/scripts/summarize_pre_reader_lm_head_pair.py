"""Summarize a fresh Host versus functional lm_head replacement pair."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def geometric_mean(values):
    if not values or any(value <= 0 for value in values):
        raise ValueError("positive speedup values required")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def summarize(host, replacement):
    if not host.get("complete") or not replacement.get("complete"):
        raise ValueError("complete Host and replacement runs required")
    if host.get("evaluation_label_files_read") != [] or replacement.get("evaluation_label_files_read") != []:
        raise ValueError("paired runs must not read evaluation labels")
    dispatch = replacement.get("lm_head_functional_dispatch", {})
    if dispatch.get("mode") != "replace" or dispatch.get("physical_pim_executed"):
        raise ValueError("replacement run must be functional GPU emulation only")
    if dispatch.get("calls", 0) != dispatch.get("replaced_calls", 0):
        raise ValueError("all admitted lm_head calls must be replaced")
    host_rows = {row["question_id"]: row for row in host["rows"]}
    replacement_rows = {row["question_id"]: row for row in replacement["rows"]}
    if set(host_rows) != set(replacement_rows) or not host_rows:
        raise ValueError("paired question IDs differ")
    rows = []
    for question_id in host_rows:
        base = host_rows[question_id]
        candidate = replacement_rows[question_id]
        rows.append({
            "question_id": question_id,
            "host_service_seconds": base["service_seconds"],
            "replacement_service_seconds": candidate["service_seconds"],
            "service_speedup": base["service_seconds"] / candidate["service_seconds"],
            "host_ttft_seconds": base["ttft_seconds"],
            "replacement_ttft_seconds": candidate["ttft_seconds"],
            "ttft_speedup": base["ttft_seconds"] / candidate["ttft_seconds"],
            "answer_tokens_equal": base["answer_token_ids"] == candidate["answer_token_ids"],
            "answer_text_equal": base["answer"] == candidate["answer"],
        })
    service_speedup = geometric_mean([row["service_speedup"] for row in rows])
    ttft_speedup = geometric_mean([row["ttft_speedup"] for row in rows])
    tokens_equal = all(row["answer_tokens_equal"] for row in rows)
    return {
        "schema_version": "pre-reader-lm-head-paired-gate-v1",
        "question_count": len(rows),
        "resident_bytes": dispatch.get("resident_bytes"),
        "functional_dispatch_calls": dispatch.get("calls"),
        "rows": rows,
        "aggregate": {
            "geometric_mean_service_speedup": service_speedup,
            "geometric_mean_ttft_speedup": ttft_speedup,
            "all_answer_tokens_equal": tokens_equal,
            "all_answer_text_equal": all(row["answer_text_equal"] for row in rows),
        },
        "gate": {
            "required_speedup": 1.05,
            "passed": (service_speedup >= 1.05 and ttft_speedup >= 1.05 and tokens_equal),
            "physical_pim_executed": False,
            "boundary": "same-GPU functional bank-owned emulation pair; not physical PIM latency",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=Path, required=True)
    parser.add_argument("--replacement", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh pair summary required")
    result = summarize(json.loads(args.host.read_text()), json.loads(args.replacement.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["gate"], ensure_ascii=False))


if __name__ == "__main__":
    main()
