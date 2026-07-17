#!/usr/bin/env python3
"""Run the v5.1 controlled M1 history-scale experiment.

Every condition x scale cell consumes the exact frozen query cohort produced by
``00_audit_m1_common_cohort.py``.  The runner intentionally records invocation
rows before aggregation so that cohort equality and byte accounting can be
audited independently of the summary statistics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT.parents[0] / "20260704_method1_motivation_medmemorybench_v3"
sys.path.insert(0, str(V3 / "scripts"))
from medmmb_common import (  # noqa: E402
    build_nested_history,
    build_reranker_client,
    eligible_memories_for_query,
    environment_manifest,
    evidence_metrics,
    group_memories_by_persona,
    load_yaml,
    read_jsonl,
    run_flat_selection,
    set_global_determinism,
    summarize_numeric,
    validate_no_future_or_cross_persona,
)


def _sha256_ids(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_progress(payload: dict[str, Any]) -> None:
    path = ROOT / "runs" / "progress" / "M1_controlled_progress.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class CPUTransformerEmbeddingClient:
    """CPU mirror of the BGE mean-pooling path for environments without CUDA."""

    def __init__(self, model_dir: str, dim: int = 768, max_length: int = 512, batch_size: int = 16):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.dim = dim
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).to(torch.device("cpu"))
        self.model.eval()
        self.mode = "transformer_embedding_cpu"

    def embed(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoded = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            with self.torch.no_grad():
                output = self.model(**encoded)
                hidden = output.last_hidden_state.float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            for row in pooled.numpy():
                if row.size >= self.dim:
                    row = row[: self.dim]
                else:
                    row = np.pad(row, (0, self.dim - row.size))
                norm = np.linalg.norm(row)
                vectors.append((row / norm if norm else row).astype(np.float32))
        return np.vstack(vectors).astype(np.float32)


def _load_cached_vectors() -> tuple[np.ndarray, list[str], str]:
    cache_dir = V3 / "aggregate" / "embedding_cache"
    npy_files = sorted(cache_dir.glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No reusable embedding cache under {cache_dir}")
    vector_path = npy_files[0]
    ids_path = vector_path.with_suffix(".ids.json")
    if not ids_path.exists():
        raise FileNotFoundError(ids_path)
    vectors = np.load(vector_path).astype(np.float32)
    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    if len(ids) != len(vectors):
        raise RuntimeError(f"Embedding cache length mismatch: {vector_path}")
    return vectors, [str(mid) for mid in ids], str(vector_path)


def _load_config(path: str) -> dict[str, Any]:
    import yaml

    config_path = Path(path).resolve()
    payload = dict(yaml.safe_load(config_path.read_text(encoding="utf-8")) or {})
    for key in ("memory_items", "queries", "evidence_mapping", "retrieval_config"):
        value = payload.get(key)
        if value and not Path(str(value)).is_absolute():
            payload[key] = str((config_path.parent / str(value)).resolve())
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["smoke", "formal"], default="formal")
    parser.add_argument("--allow-cpu", action="store_true", help="run the same protocol on CPU when CUDA is unavailable")
    args = parser.parse_args()

    config = _load_config(args.config)
    seed = int(config.get("seed", 20260716))
    set_global_determinism(seed)
    import torch

    cuda_available = bool(torch.cuda.is_available())
    if args.mode == "formal" and not cuda_available and not args.allow_cpu:
        raise RuntimeError("CUDA unavailable; rerun explicitly with --allow-cpu for an annotated CPU fallback.")

    cohort_path = ROOT / "aggregate" / "v5p1_m1_common_query_ids.json"
    if not cohort_path.exists():
        raise FileNotFoundError(f"Run 00_audit_m1_common_cohort.py first: {cohort_path}")
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    query_ids = [str(qid) for qid in cohort["selected_query_ids"]]
    scales = [int(scale) for scale in cohort["selected_scale_set"]]
    if not query_ids or len(scales) < 3:
        raise RuntimeError("Controlled M1 requires a non-empty cohort and at least three scales")

    memory_items = read_jsonl(config["memory_items"])
    all_queries = {str(row["query_id"]): row for row in read_jsonl(config["queries"])}
    queries = [all_queries[qid] for qid in query_ids if qid in all_queries]
    if len(queries) != len(query_ids):
        raise RuntimeError("Frozen cohort contains query IDs absent from the configured query file")
    if args.mode == "smoke":
        # One query per type keeps the smoke path representative while making
        # model/device failures cheap to diagnose.  Formal mode always uses
        # the complete frozen cohort.
        by_type: dict[str, list[dict[str, Any]]] = {}
        for query in queries:
            by_type.setdefault(str(query.get("query_type", "unknown")), []).append(query)
        queries = [rows[0] for _, rows in sorted(by_type.items())]
    evidence_by_query = {
        str(row["query_id"]): {str(mid) for mid in row.get("gold_memory_ids", [])}
        for row in read_jsonl(config["evidence_mapping"])
    }
    memories_by_persona = group_memories_by_persona(memory_items)
    vectors, vector_ids, cache_path = _load_cached_vectors()
    index_by_id = {memory_id: idx for idx, memory_id in enumerate(vector_ids)}

    retrieval_config = load_yaml(config["retrieval_config"])
    model_dir = str((V3 / "../../models" / "bge-base-en-v1.5").resolve())
    if cuda_available:
        from src.embedding_client import TransformerEmbeddingClient

        embed_client = TransformerEmbeddingClient(
            model_dir=model_dir,
            dim=int(retrieval_config.get("embedding_dim", 768)),
            max_length=int(retrieval_config.get("embedding_max_length", 512)),
            batch_size=int(retrieval_config.get("embedding_batch_size", 16)),
        )
        execution_device = "cuda"
    else:
        embed_client = CPUTransformerEmbeddingClient(
            model_dir=model_dir,
            dim=int(retrieval_config.get("embedding_dim", 768)),
            max_length=int(retrieval_config.get("embedding_max_length", 512)),
            batch_size=int(retrieval_config.get("embedding_batch_size", 16)),
        )
        execution_device = "cpu_fallback"
    reranker_client = build_reranker_client(retrieval_config)
    if getattr(reranker_client, "mode", "") != "cross_encoder":
        raise RuntimeError(f"Controlled M1 requires the true cross encoder, got {getattr(reranker_client, 'mode', '')}")

    conditions = [str(value) for value in config.get("history_conditions", ["medical_only", "medical_plus_official_noise"])]
    frontier_budget = int(config.get("frontier_budget", 32))
    final_top_k = int(config.get("final_top_k", 5))
    run_id = f"M1_controlled_{args.mode}_{seed}"
    total_units = len(queries) * len(conditions) * len(scales)
    _write_progress({"run_id": run_id, "status": "running", "total_units": total_units, "completed_units": 0})

    rows: list[dict[str, Any]] = []
    completed = 0
    for condition in conditions:
        include_noise = condition == "medical_plus_official_noise"
        for query in queries:
            qid = str(query["query_id"])
            eligible = eligible_memories_for_query(
                memories_by_persona,
                query,
                allow_current_session=bool(config.get("allow_current_session_context", True)),
                include_noise=include_noise,
            )
            validate_no_future_or_cross_persona(
                query,
                eligible,
                allow_current_session=bool(config.get("allow_current_session_context", True)),
            )
            gold = evidence_by_query.get(qid, set())
            if not eligible:
                raise RuntimeError(f"No eligible memories for frozen query {qid} ({condition})")
            for scale in scales:
                if len(eligible) < scale or len(gold) > scale:
                    raise RuntimeError(f"Frozen feasibility violated for query={qid} condition={condition} scale={scale}")
                history = build_nested_history(query=query, eligible=eligible, gold_memory_ids=gold, scale=scale, seed=seed)
                candidate_indices = [index_by_id[str(row["memory_id"])] for row in history]
                candidate_vectors = vectors[candidate_indices]
                result = run_flat_selection(
                    query_text=str(query["query_text"]),
                    candidate_memories=history,
                    candidate_vectors=candidate_vectors,
                    retrieval_config=retrieval_config,
                    embed_client=embed_client,
                    reranker_client=reranker_client,
                    frontier_budget=frontier_budget,
                    final_top_k=final_top_k,
                )
                rows.append(
                    {
                        "run_id": run_id,
                        "mode": args.mode,
                        "execution_device": execution_device,
                        "history_condition": condition,
                        "include_noise": int(include_noise),
                        "query_id": qid,
                        "persona_id": query["persona_id"],
                        "query_type": query.get("query_type", "unknown"),
                        "query_session_id": query["query_session_id"],
                        "history_scale": scale,
                        "n_memory_total": len(history),
                        "frontier_budget": frontier_budget,
                        "final_top_k": final_top_k,
                        "frontier_fill_ratio": result["n_reranked"] / float(frontier_budget),
                        "selection_fill_ratio": result["n_final_selected"] / float(final_top_k),
                        **{key: result[key] for key in [
                            "n_ann_raw", "n_after_filter", "n_after_dedup", "n_materialized",
                            "n_reranked", "n_final_selected", "t_query_encode_ms", "t_ann_ms",
                            "t_filter_ms", "t_materialize_ms", "t_rerank_ms", "t_total_selection_ms",
                            "index_bytes_read", "metadata_bytes_read", "payload_bytes_read", "host_memory_bytes",
                        ]},
                        **evidence_metrics(result["final_selected_ids"], gold),
                        "ann_candidate_ids": result["ann_candidate_ids"],
                        "reranked_candidate_ids": result["reranked_candidate_ids"],
                        "final_selected_ids": result["final_selected_ids"],
                        "common_query_count": len(query_ids),
                        "query_ids_hash": cohort["selected_query_ids_hash"],
                        "seed": seed,
                        "allow_current_session_context": int(config.get("allow_current_session_context", True)),
                        "cuda_available": int(cuda_available),
                        "gpu_model": ",".join(environment_manifest().get("devices", [])),
                        "embedding_cache": cache_path,
                        "error": "",
                    }
                )
                completed += 1
                if completed % 10 == 0 or completed == total_units:
                    _write_progress(
                        {
                            "run_id": run_id,
                            "status": "running",
                            "total_units": total_units,
                            "completed_units": completed,
                            "current_query_id": qid,
                            "condition": condition,
                            "history_scale": scale,
                        }
                    )

    invocation_path = ROOT / "aggregate" / "M1_controlled_invocations.jsonl"
    summary_path = ROOT / "aggregate" / "M1_controlled_summary.csv"
    _write_jsonl(invocation_path, rows)
    summary = summarize_numeric(
        rows,
        ["mode", "execution_device", "history_condition", "history_scale"],
        ["t_total_selection_ms", "t_query_encode_ms", "t_ann_ms", "t_rerank_ms", "host_memory_bytes", "metadata_bytes_read", "payload_bytes_read", "index_bytes_read", "frontier_fill_ratio"],
        seed=seed,
    )
    for row in summary:
        row.update({
            "common_query_count": len(query_ids),
            "query_ids_hash": cohort["selected_query_ids_hash"],
            "frontier_budget": frontier_budget,
            "final_top_k": final_top_k,
            "seed": seed,
            "allow_current_session_context": int(config.get("allow_current_session_context", True)),
            "include_noise": int(row["history_condition"] == "medical_plus_official_noise"),
        })
    _write_csv(summary_path, summary)
    manifest = environment_manifest(
        {
            "run_id": run_id,
            "experiment": "M1_controlled_v5p1",
            "mode": args.mode,
            "execution_device": execution_device,
            "common_query_count": len(query_ids),
            "query_ids_hash": cohort["selected_query_ids_hash"],
            "query_type_counts": cohort["selected_query_type_counts"],
            "history_conditions": conditions,
            "history_scales": scales,
            "frontier_budget": frontier_budget,
            "final_top_k": final_top_k,
            "allow_current_session_context": bool(config.get("allow_current_session_context", True)),
            "seed": seed,
            "invocation_rows": len(rows),
            "embedding_cache": cache_path,
            "cpu_fallback_requested": bool(args.allow_cpu),
        }
    )
    (ROOT / "manifests" / f"{run_id}_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_progress({"run_id": run_id, "status": "complete", "total_units": total_units, "completed_units": completed})
    print(json.dumps({"run_id": run_id, "execution_device": execution_device, "invocations": str(invocation_path), "summary": str(summary_path), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
