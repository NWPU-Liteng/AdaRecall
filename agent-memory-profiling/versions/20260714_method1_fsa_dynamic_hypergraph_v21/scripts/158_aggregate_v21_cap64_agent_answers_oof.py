#!/usr/bin/env python3
"""Aggregate train-OOF cap64 Agent accuracy, tokens, and total latency."""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))
from v21.io import iter_jsonl, read_json, sha256_file, write_json_new  # noqa: E402

SHARDS = [VERSION_ROOT / f"artifacts/v21_cap64_agent_answers_oof_shard{index}_attempt003.jsonl" for index in range(4)]
OUTPUT = VERSION_ROOT / "aggregate/v21_cap64_agent_answers_oof_summary_r1_attempt001.json"
COMBINED = VERSION_ROOT / "aggregate/v21_cap64_agent_answers_oof_rows_r1_attempt001.jsonl"
REPORT = VERSION_ROOT / "reports/METHOD1_V21_CAP64_AGENT_ANSWER_OOF_R1_ATTEMPT001.md"
EXPECTED_ROWS = 4065
EVAL_BUDGETS = (8, 16, 32)
QUERY_COUNT = 1355


def norm(value: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"^(final answer|answer)\s*[:\-]\s*", "", text, flags=re.IGNORECASE)
    return " ".join(text.split())


def score(row: dict) -> tuple[int, str]:
    prediction = norm(row["answer_raw"])
    gold = [norm(value) for value in row.get("gold_answers", [])]
    if not prediction or prediction == "insufficient information":
        return 0, "empty_or_insufficient"
    if str(row["query_type"]) == "multiple_choice":
        match = re.search(r"(?:^|answer\s*[:\-]?\s*)([a-e])(?:\b|[.)])", prediction, flags=re.IGNORECASE)
        if match:
            label = match.group(1).lower()
            labels = {candidate.group(1).lower() for value in row.get("gold_answers", []) if (candidate := re.match(r"\s*([A-E])[.)]\s+", str(value), flags=re.IGNORECASE))}
            if labels:
                return int(label in labels), "multiple_choice_option_label"
    return int(any(value and (value == prediction or value in prediction) for value in gold)), "normalized_contains_gold"


def summarize(items: list[dict]) -> dict:
    def values(key: str) -> np.ndarray:
        return np.asarray([float(row[key]) for row in items], dtype=np.float64)
    generation = values("answer_generation_latency_ms")
    retrieval = values("retrieval_query_e2e_ms")
    total = generation + retrieval
    return {"rows": len(items), "answer_accuracy": float(values("answer_correct").mean()), "prompt_tokens_mean": float(values("answer_prompt_tokens").mean()), "prompt_tokens_p95": float(np.percentile(values("answer_prompt_tokens"), 95)), "completion_tokens_mean": float(values("answer_completion_tokens").mean()), "generation_latency_ms_mean": float(generation.mean()), "generation_latency_ms_p95": float(np.percentile(generation, 95)), "retrieval_latency_ms_mean": float(retrieval.mean()), "total_agent_latency_ms_mean": float(total.mean()), "total_agent_latency_ms_p95": float(np.percentile(total, 95))}


def main() -> None:
    rows, hashes = [], {}
    for index, path in enumerate(SHARDS):
        manifest = read_json(path.with_suffix(".manifest.json"))
        if manifest.get("status") != "completed" or manifest.get("calibration_read") is not False or manifest.get("formal_test_read") is not False or manifest.get("minimum_effect_threshold") is not None:
            raise ValueError(f"invalid Agent shard {path}")
        shard = list(iter_jsonl(path))
        if len(shard) != int(manifest["rows"]):
            raise ValueError(f"Agent shard row mismatch {path}")
        rows.extend(shard)
        hashes[f"shard_{index}"] = sha256_file(path)
    if len(rows) != EXPECTED_ROWS or len({(row["query_id"], int(row["budget"])) for row in rows}) != EXPECTED_ROWS:
        raise ValueError("Agent answer rows are incomplete or duplicated")
    rescored = []
    for row in rows:
        correct, rule = score(row)
        if row.get("backend_mode") != "local_qwen" or row.get("generation_error"):
            raise ValueError("Agent generation did not use the required local Qwen backend")
        rescored.append({**row, "answer_correct": correct, "scoring_rule": rule, "total_agent_latency_ms": float(row["answer_generation_latency_ms"]) + float(row["retrieval_query_e2e_ms"])})
    rescored.sort(key=lambda row: (str(row["query_id"]), int(row["budget"])))
    with COMBINED.open("x", encoding="utf-8") as handle:
        for row in rescored:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    by_budget = {str(budget): summarize([row for row in rescored if int(row["budget"]) == budget]) for budget in EVAL_BUDGETS}
    grouped = defaultdict(list)
    for row in rescored:
        grouped[(int(row["budget"]), str(row["query_type"]))].append(row)
    by_type = {f"B={budget}|{query_type}": summarize(items) for (budget, query_type), items in sorted(grouped.items())}
    summary = {"schema_version": "v21-cap64-agent-answer-oof-summary-1", "status": f"completed_{QUERY_COUNT}_train_oof_queries_x_{len(EVAL_BUDGETS)}_budgets", "seed": 20260712, "queries": QUERY_COUNT, "rows": len(rescored), "method": "cap64_qhds", "context_word_cap": 1800, "answer_max_new_tokens": 48, "scoring_boundary": "rule-based normalized contains-gold; multiple-choice label aware; train OOF only", "by_budget": by_budget, "by_query_type": by_type, "minimum_effect_threshold": None, "calibration_read": False, "formal_test_read": False, "artifacts": {"combined_rows": str(COMBINED.relative_to(VERSION_ROOT))}, "hashes": {**hashes, "combined_rows": sha256_file(COMBINED)}}
    write_json_new(OUTPUT, summary)
    report = "# Method 1 v21 cap64 Agent Answer OOF\n\nTrain-only OOF evaluation with local Qwen, a fixed 1,800-word context cap, and at most 48 generated tokens. Calibration and formal test were not read.\n\n| budget | answer accuracy | prompt tokens | completion tokens | retrieval ms | generation ms | total Agent ms |\n|---:|---:|---:|---:|---:|---:|---:|\n"
    for budget in EVAL_BUDGETS:
        value = by_budget[str(budget)]
        report += f"| {budget} | {value['answer_accuracy']:.6f} | {value['prompt_tokens_mean']:.1f} | {value['completion_tokens_mean']:.1f} | {value['retrieval_latency_ms_mean']:.3f} | {value['generation_latency_ms_mean']:.3f} | {value['total_agent_latency_ms_mean']:.3f} |\n"
    report += "\nThis is final-answer execution on the authorized train-OOF split, not formal heldout performance. Token cost is reported as local-model token counts, not API currency cost.\n"
    with REPORT.open("x", encoding="utf-8") as handle:
        handle.write(report)
    print(json.dumps({"status": summary["status"], "summary": str(OUTPUT), "report": str(REPORT)}, sort_keys=True))


if __name__ == "__main__":
    main()
