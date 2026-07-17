#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embedding_client import TransformerEmbeddingClient
from src.llm_client import LocalQwenLLMClient
from src.memorycd.task_runtime import (
    _build_prompt_with_selected_memory,
    _format_memory_for_prompt,
    _format_target_item,
    _parse_prediction,
    LocalPromptBuilder,
    MemoryCDMetadataLoader,
    _sync_cuda,
    compute_selection_for_method,
    load_embedding_cache,
    load_protocol_runtime,
    write_json,
)
from src.reranker_client import CrossEncoderRerankerClient


def _load_subset_event_ids(subset_path: Path) -> set[str]:
    return {json.loads(line)["event_id"] for line in subset_path.open("r", encoding="utf-8")}


def _iter_configs() -> list[tuple[str, int | None, int | None]]:
    return [
        ("recency_k8", None, None),
        ("dense_k8", None, None),
        ("dense_r32_b8_k8", 32, 8),
        ("dense_r32_b16_k8", 32, 16),
        ("dense_r32_b32_k8", 32, 32),
        ("dense_r64_b16_k8", 64, 16),
        ("dense_r64_b32_k8", 64, 32),
        ("dense_r64_b64_k8", 64, 64),
        ("dense_r128_b16_k8", 128, 16),
        ("dense_r128_b32_k8", 128, 32),
        ("dense_r128_b64_k8", 128, 64),
    ]


def _parse_method_specs(methods: list[str] | None) -> list[tuple[str, int | None, int | None]]:
    if not methods:
        return _iter_configs()
    parsed: list[tuple[str, int | None, int | None]] = []
    for method in methods:
        if method in {"recency_k8", "dense_k8"}:
            parsed.append((method, None, None))
            continue
        parts = method.split("_")
        if len(parts) != 4 or parts[0] != "dense" or not parts[1].startswith("r") or not parts[2].startswith("b") or not parts[3].startswith("k"):
            raise ValueError(f"Unsupported method spec: {method}")
        parsed.append((method, int(parts[1][1:]), int(parts[2][1:])))
    return parsed


def _task_max_new_tokens(task: str) -> int:
    if task == "rating_prediction":
        return 8
    if task == "item_ranking":
        return 64
    if task == "review_summarization":
        return 32
    if task == "review_generation":
        return 256
    raise ValueError(task)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version-dir", type=Path, required=True)
    parser.add_argument("--protocol", choices=["official_single", "official_cross", "causal_cross"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--embedding-model", type=Path, default=Path("models/bge-base-en-v1.5"))
    parser.add_argument("--reranker-model", type=Path, default=Path("models/cross-encoder-ms-marco-MiniLM-L6-v2"))
    parser.add_argument("--qwen-model-dir", type=Path, default=Path("../dir"))
    parser.add_argument("--embedding-cache-dir", type=Path, required=True)
    parser.add_argument("--meta-dir", type=Path, default=Path("data/public_datasets/memorycd/meta"))
    parser.add_argument("--subset-path", type=Path, default=None)
    parser.add_argument("--tasks", nargs="+", default=["rating_prediction", "item_ranking"])
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--max-events-per-stratum", type=int, default=200)
    parser.add_argument("--reranker-batch-size", type=int, default=16)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--validate-faiss", action="store_true")
    parser.add_argument("--warmup-events", type=int, default=20)
    args = parser.parse_args()

    os.environ["LOCAL_QWEN_CUDA_DEVICE"] = str(args.gpu)

    runtime_rows, _, memory_by_id = load_protocol_runtime(version_dir=args.version_dir, protocol=args.protocol)
    subset_path = args.subset_path or (args.version_dir / "manifests" / f"memorycd_{args.protocol}" / "task_curve_subset_event_ids.jsonl")
    subset_event_ids = _load_subset_event_ids(subset_path)
    metadata_loader = MemoryCDMetadataLoader(args.meta_dir)
    embedding_matrix, embedding_row_by_id = load_embedding_cache(args.embedding_cache_dir)

    events = [row for row in runtime_rows if row["event_id"] in subset_event_ids and row["task"] in set(args.tasks)]
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for event in events:
        grouped[(event["task"], event["setting"])].append(event)
    selected_events: list[dict] = []
    for key, rows in sorted(grouped.items()):
        rows_sorted = sorted(rows, key=lambda row: row["event_id"])
        selected_events.extend(rows_sorted[: args.max_events_per_stratum])

    embedding = TransformerEmbeddingClient(str(args.embedding_model), dim=768, batch_size=16)
    reranker = CrossEncoderRerankerClient(str(args.reranker_model), batch_size=args.reranker_batch_size)
    prompt_builder = LocalPromptBuilder(include_metadata=True)
    method_specs = _parse_method_specs(args.methods)
    task_suffix = "-".join(sorted(args.tasks))
    llm_clients = {task: LocalQwenLLMClient(model_dir=str(args.qwen_model_dir), max_new_tokens=_task_max_new_tokens(task)) for task in set(args.tasks)}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = args.output_dir / f"{args.protocol}_{task_suffix}_task_curve_subset.jsonl"
    summary_path = args.output_dir / f"{args.protocol}_{task_suffix}_task_curve_run_manifest.json"

    run_manifest = {
        "protocol": args.protocol,
        "tasks": args.tasks,
        "subset_path": str(subset_path),
        "subset_event_count": len(selected_events),
        "max_events_per_stratum": args.max_events_per_stratum,
        "methods": [name for name, _, _ in method_specs],
        "k": args.k,
        "warmup_events": args.warmup_events,
        "embedding_model": str(args.embedding_model),
        "reranker_model": str(args.reranker_model),
        "qwen_model_dir": str(args.qwen_model_dir),
        "embedding_cache_dir": str(args.embedding_cache_dir),
        "validate_faiss": args.validate_faiss,
        "cold_warm_policy": "first_20_events_warmup_excluded_from_latency_aggregates",
    }
    write_json(summary_path, run_manifest)

    row_count = 0
    with trace_path.open("w", encoding="utf-8") as out:
        for event_idx, event in enumerate(selected_events):
            ids = [mid for mid in event["runtime_memory_ids"] if mid in embedding_row_by_id and mid in memory_by_id]
            rows = [embedding_row_by_id[mid] for mid in ids]
            memories = [memory_by_id[mid] for mid in ids]
            if not memories:
                continue
            doc_vecs = np.asarray(embedding_matrix[rows], dtype="float32")
            for method_name, r, b in method_specs:
                selection = compute_selection_for_method(
                    method=method_name,
                    event=event,
                    memories=memories,
                    doc_vecs=doc_vecs,
                    embedding=embedding,
                    reranker=reranker,
                    k=args.k,
                    validate_faiss=args.validate_faiss,
                    r=r,
                    b=b,
                )
                selected = [memory_by_id[mid] for mid in selection["final_context_ids"] if mid in memory_by_id]
                selected_prompt_memories = [_format_memory_for_prompt(memory) for memory in selected]
                target_item = _format_target_item(event=event, metadata_loader=metadata_loader)
                prompt, ordered_candidates = _build_prompt_with_selected_memory(
                    prompt_builder=prompt_builder,
                    task=event["task"],
                    selected_memories=selected_prompt_memories,
                    target_item=target_item,
                    metadata_loader=metadata_loader,
                    candidate_asins=event.get("candidate_item_metadata"),
                    user_id=event["tenant_id"],
                )
                llm_client = llm_clients[event["task"]]
                _sync_cuda()
                t0 = time.perf_counter()
                generation = llm_client.generate_with_stats(prompt)
                _sync_cuda()
                t_task_qwen_ms = (time.perf_counter() - t0) * 1000.0
                prediction = _parse_prediction(
                    task=event["task"],
                    response_text=generation["text"],
                    ordered_candidates=ordered_candidates,
                )
                trace = {
                    **selection,
                    "event_index": event_idx,
                    "method": method_name,
                    "event_id": event["event_id"],
                    "prediction": prediction,
                    "prompt_tokens": generation["prompt_tokens"],
                    "completion_tokens": generation["completion_tokens"],
                    "t_task_qwen_ms": t_task_qwen_ms,
                    "t_total_task_ms": selection["t_total_selection_ms"] + t_task_qwen_ms,
                    "backend_mode": generation["backend_mode"],
                    "warmup_event": event_idx < args.warmup_events,
                }
                out.write(json.dumps(trace, ensure_ascii=False) + "\n")
                row_count += 1
            out.flush()
    print(json.dumps({"trace_path": str(trace_path), "rows": row_count}, indent=2))


if __name__ == "__main__":
    main()
