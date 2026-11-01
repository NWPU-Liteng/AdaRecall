"""Summarize paired persistent-replay Host runs without reading evaluation labels."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * fraction
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return values[lower]
    weight = index - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def load(path):
    payload = json.loads(Path(path).read_text())
    if not payload.get("complete"):
        raise ValueError(f"incomplete run: {path}")
    if payload.get("evaluation_label_files_read") != []:
        raise ValueError(f"evaluation labels accessed: {path}")
    return payload


def row_metrics(row):
    timing = row["proof_plan"]["retrieval"]["timing"]
    replay = row["persistent_replay"]
    memory_path = (replay["visible_index_filter_seconds"]
                   + timing.get("dense_search_seconds", 0.0)
                   + timing.get("reranker_seconds", 0.0)
                   + timing.get("frontier_compile_seconds", 0.0))
    return {
        "question_id": row["question_id"],
        "service_seconds": row["service_seconds"],
        "ttft_seconds": row["ttft_seconds"],
        "prefill_seconds": row["prefill_seconds"],
        "decode_seconds": row["decode_seconds"],
        "query_parser_seconds": timing.get("query_parser_total_seconds"),
        "query_vector_seconds": timing.get("query_vector_encode_seconds"),
        "visible_index_filter_seconds": replay["visible_index_filter_seconds"],
        "dense_search_seconds": timing.get("dense_search_seconds"),
        "reranker_seconds": timing.get("reranker_seconds"),
        "frontier_compile_seconds": timing.get("frontier_compile_seconds"),
        "memory_path_seconds": memory_path,
        "visible_record_count": replay["visible_record_count"],
        "full_record_count": replay["full_record_count"],
        "visible_vector_bytes": replay["visible_vector_bytes"],
        "full_vector_bytes": replay["full_vector_bytes"],
        "answer_sha256": row["answer_sha256"],
        "answer_token_ids": row["answer_token_ids"],
        "stop_reason": row["stop_reason"],
        "materialized_object_ids": row["materialized_object_ids"],
    }


def summarize_run(rows):
    fields = ["service_seconds", "ttft_seconds", "prefill_seconds", "decode_seconds",
              "query_parser_seconds", "query_vector_seconds", "visible_index_filter_seconds",
              "dense_search_seconds", "reranker_seconds", "frontier_compile_seconds",
              "memory_path_seconds"]
    summary = {"queries": len(rows)}
    for field in fields:
        values = [row[field] for row in rows if row[field] is not None]
        summary[field] = {
            "mean": sum(values) / len(values) if values else None,
            "p50": percentile(values, 0.50),
            "p95": percentile(values, 0.95),
            "sum": sum(values) if values else None,
        }
    summary["mean_visible_record_fraction"] = (
        sum(row["visible_record_count"] / row["full_record_count"] for row in rows) / len(rows))
    summary["mean_visible_vector_fraction"] = summary["mean_visible_record_fraction"]
    summary["answer_stop_reason_counts"] = {
        reason: sum(row["stop_reason"] == reason for row in rows)
        for reason in sorted({row["stop_reason"] for row in rows})
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    filter_run, mask_run = load(args.filter), load(args.mask)
    if filter_run["question_ids"] != mask_run["question_ids"]:
        raise ValueError("paired runs have different question streams")
    filter_rows = [row_metrics(row) for row in filter_run["rows"]]
    mask_rows = [row_metrics(row) for row in mask_run["rows"]]
    filter_by_id = {row["question_id"]: row for row in filter_rows}
    mask_by_id = {row["question_id"]: row for row in mask_rows}
    if set(filter_by_id) != set(mask_by_id):
        raise ValueError("paired runs have different completed question identities")
    paired = []
    for question_id in filter_run["question_ids"]:
        left, right = filter_by_id[question_id], mask_by_id[question_id]
        paired.append({
            "question_id": question_id,
            "filter_service_seconds": left["service_seconds"],
            "mask_service_seconds": right["service_seconds"],
            "filter_ttft_seconds": left["ttft_seconds"],
            "mask_ttft_seconds": right["ttft_seconds"],
            "service_speedup_filter_over_mask": right["service_seconds"] / left["service_seconds"],
            "ttft_speedup_filter_over_mask": right["ttft_seconds"] / left["ttft_seconds"],
            "answer_token_equal": left["answer_token_ids"] == right["answer_token_ids"],
            "answer_equal": left["answer_sha256"] == right["answer_sha256"],
            "stop_reason_equal": left["stop_reason"] == right["stop_reason"],
            "materialized_objects_equal": left["materialized_object_ids"] == right["materialized_object_ids"],
            "visible_record_count": left["visible_record_count"],
            "full_record_count": left["full_record_count"],
            "filter_memory_path_seconds": left["memory_path_seconds"],
            "mask_memory_path_seconds": right["memory_path_seconds"],
        })
    output = {
        "schema": "persistent-host-pair-summary-v1",
        "measurement_boundary": "server Host wall-clock; model/index startup excluded from per-query service",
        "runtime_labels_read": False,
        "filter_run": {"path": str(args.filter), "summary": summarize_run(filter_rows)},
        "mask_run": {"path": str(args.mask), "summary": summarize_run(mask_rows)},
        "pair": {
            "queries": len(paired),
            "answer_token_equal_count": sum(row["answer_token_equal"] for row in paired),
            "answer_equal_count": sum(row["answer_equal"] for row in paired),
            "stop_reason_equal_count": sum(row["stop_reason_equal"] for row in paired),
            "materialized_objects_equal_count": sum(row["materialized_objects_equal"] for row in paired),
            "filter_over_mask_service_speedup_mean": sum(row["service_speedup_filter_over_mask"] for row in paired) / len(paired),
            "filter_over_mask_ttft_speedup_mean": sum(row["ttft_speedup_filter_over_mask"] for row in paired) / len(paired),
            "paired_rows": paired,
        },
        "status": "pass",
    }
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "status": output["status"], "queries": len(paired),
        "answer_token_equal_count": output["pair"]["answer_token_equal_count"],
        "filter_over_mask_service_speedup_mean": output["pair"]["filter_over_mask_service_speedup_mean"],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
