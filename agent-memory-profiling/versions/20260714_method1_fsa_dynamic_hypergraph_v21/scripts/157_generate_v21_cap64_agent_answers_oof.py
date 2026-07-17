#!/usr/bin/env python3
"""Generate train-OOF Agent answers for cap64 QHDS B=8/16/32 handoffs."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from time import perf_counter

VERSION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = VERSION_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.io import iter_jsonl, load_config, read_json, resolve, write_json_new  # noqa: E402


def normalize(text: str) -> str:
    lines = [line.strip() for line in str(text or "").strip().splitlines() if line.strip()]
    answer = lines[0] if lines else ""
    return re.sub(r"^(final answer|answer)\s*[:\-]\s*", "", answer, flags=re.IGNORECASE).strip()


def score(query_type: str, prediction: str, gold_answers: list[str]) -> int:
    pred = normalize(prediction).lower()
    gold = [str(item).strip().lower() for item in gold_answers]
    if not pred or pred == "insufficient information":
        return 0
    return int(any(value and (value == pred or value in pred) for value in gold))


def prompt(row: dict, context: str) -> str:
    return ("You are evaluating a text-only healthcare memory benchmark. This is research data, not clinical advice.\n"
            "Answer the user query using only the provided memory context.\n"
            "Do not use external knowledge or tools.\n"
            "If the memory context is insufficient, answer exactly: insufficient information\n"
            "Output one concise final answer line.\n\n"
            f"Query type: {row.get('query_type', 'unknown')}\nCurrent session: {row.get('query_session_id', '')}\n"
            f"User query: {row.get('query_text', '')}\n\nMemory context:\n{context}\n\nFinal answer only:")


def context(memory_by_id: dict[str, dict], ids: list[str], token_cap: int = 1800, query: dict | None = None) -> str:
    parts, used = [], 0
    for memory_id in ids:
        memory = memory_by_id[memory_id]
        header = f"[memory_id={memory_id} | session={memory.get('session_id')} | type={memory.get('session_type', 'unknown')}]"
        words = (header + "\n" + str(memory.get("text", ""))).split()
        remaining = max(0, token_cap - used)
        if remaining <= 0:
            break
        parts.append(" ".join(words[:remaining]))
        used += min(len(words), remaining)
    return "\n\n".join(parts) if parts else "No retrieved memory."


def order_context_ids(query: dict, ids: list[str], memory_by_id: dict[str, dict]) -> list[str]:
    """Default handoff order preserves the selector's order."""
    return list(ids)


def original_query_id(value: str) -> str:
    match = re.fullmatch(r"queries_(.+)_\d+", str(value))
    if not match:
        raise ValueError(f"unexpected OOF query ID: {value}")
    return match.group(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-dir", default=str(PROJECT_ROOT.parent / "dir"))
    parser.add_argument("--budgets", default="8,16,32")
    args = parser.parse_args()
    data = load_config("v21_data.json")
    memory_by_id = {str(row["memory_id"]): row for row in iter_jsonl(data["memory"]["items"])}
    plan = {str(row["query_id"]): row for row in iter_jsonl(data["split"]["train_query_plan"])}
    source_root = VERSION_ROOT / "artifacts/v21_sparse_joint" / "v21_sparse_joint_cap64_oof_r1_20260715"
    selections, metrics = [], {}
    for fold in range(5):
        unit = source_root / "seed_20260712" / f"fold_{fold}" / "attempts/attempt_001"
        for row in iter_jsonl(unit / "v21_sparse_joint_selections.jsonl"):
            if str(row["method"]) == "sparse_joint_semantic_bundle_hypergraph" and int(row["activated_prototypes"]) == 16 and int(row["members_per_semantic_edge"]) == 32 and int(row["semantic_bundle_beam"]) == 32:
                selections.append(row)
        import csv
        with (unit / "v21_sparse_joint_query_metrics.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if str(row["method"]) == "sparse_joint_semantic_bundle_hypergraph" and int(row["activated_prototypes"]) == 16 and int(row["members_per_semantic_edge"]) == 32 and int(row["semantic_bundle_beam"]) == 32:
                    metrics[(str(row["query_id"]), int(row["budget"]))] = float(row["query_e2e_ms"])
    tasks = []
    for selection in selections:
        if int(selection["budget"]) != 32:
            continue
        source_id = str(selection["query_id"])
        original = source_id if source_id in plan else original_query_id(source_id)
        if original not in plan:
            raise ValueError(f"query plan missing {source_id}")
        for budget in (8, 16, 32):
            tasks.append((selection, plan[original], budget))
    requested_budgets = {int(value) for value in str(args.budgets).split(",") if value.strip()}
    if not requested_budgets or not requested_budgets.issubset({8, 16, 32}):
        raise ValueError("budgets must be a nonempty subset of 8,16,32")
    tasks = [item for item in tasks if item[2] in requested_budgets]
    tasks.sort(key=lambda item: (str(item[1]["query_id"]), item[2]))
    tasks = tasks[args.offset: args.offset + args.limit if args.limit else None]
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ["LOCAL_QWEN_CUDA_DEVICE"] = str(args.device)
    from src.llm_client import LocalQwenLLMClient
    client = LocalQwenLLMClient(model_dir=str(Path(args.model_dir).resolve()), max_new_tokens=48, use_chat_template=False)
    if getattr(client, "mode", "") != "local_qwen":
        raise RuntimeError(f"local Qwen required, got {getattr(client, 'mode', '')}")
    output = []
    for index, (selection, query, budget) in enumerate(tasks, start=1):
        selected = order_context_ids(query, list(map(str, selection["selected_memory_ids"])), memory_by_id)[:budget]
        text_prompt = prompt(query, context(memory_by_id, selected, query=query))
        start = perf_counter()
        response = client.generate_with_stats(text_prompt)
        elapsed = float(response.get("elapsed_ms", (perf_counter() - start) * 1000.0))
        raw = str(response.get("text", ""))
        output.append({"query_id": str(query["query_id"]), "source_query_id": str(selection["query_id"]), "seed": 20260712, "fold": int(selection["fold"]), "budget": budget, "method": "cap64_qhds", "query_type": str(query.get("query_type", "")), "query_text": str(query.get("query_text", "")), "gold_answers": list(query.get("gold_answers", [])), "selected_memory_ids": selected, "answer_raw": raw, "answer_normalized": normalize(raw), "answer_correct": score(str(query.get("query_type", "")), raw, list(query.get("gold_answers", []))), "answer_prompt_tokens": int(response.get("prompt_tokens", 0)), "answer_completion_tokens": int(response.get("completion_tokens", 0)), "answer_generation_latency_ms": elapsed, "retrieval_query_e2e_ms": metrics[(str(selection["query_id"]), budget)], "backend_mode": response.get("backend_mode", getattr(client, "mode", "")), "generation_error": ""})
        if index % 25 == 0 or index == len(tasks):
            print(json.dumps({"device": args.device, "progress": index, "total": len(tasks)}, sort_keys=True), flush=True)
    output_path = VERSION_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    manifest_path = output_path.with_suffix(".manifest.json")
    write_json_new(manifest_path, {"schema_version": "v21-cap64-agent-answer-oof-shard-1", "status": "completed", "rows": len(output), "offset": args.offset, "device": args.device, "calibration_read": False, "formal_test_read": False, "minimum_effect_threshold": None})
    print(json.dumps({"status": "completed", "rows": len(output), "output": str(output_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
